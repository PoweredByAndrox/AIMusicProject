from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .melody_alignment import DEFAULT_SEPARATOR_MODEL, NOTE_NAMES, load_or_create_note_events


logger = logging.getLogger(__name__)

NOTE_INDEX = {note_name: index for index, note_name in enumerate(NOTE_NAMES)}
EDGE_TYPES = {"intro", "outro"}
ACTIVE_TYPES = {"drop_high", "drop_bass", "melodic", "verse", "buildup"}
LOW_TYPES = {"break", "ambient"}


def _segment_family(segment_type: str | None) -> str:
    normalized = (segment_type or "").strip().lower()
    if normalized in EDGE_TYPES:
        return "edge"
    if normalized in ACTIVE_TYPES:
        return "active"
    if normalized in LOW_TYPES:
        return "low"
    return "other"


def _copy_segment(segment: dict[str, Any]) -> dict[str, Any]:
    copied = dict(segment)
    copied["_merged_children"] = [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "type": str(segment["type"]),
            "importance": float(segment.get("importance", 1.0)),
            "label_hint": segment.get("label_hint"),
        }
    ]
    return copied


def _event_window_stats(
    note_events: Sequence[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    hist = np.zeros(12, dtype=np.float32)
    weighted_duration = 0.0
    onset_count = 0
    event_count = 0
    confidence_total = 0.0

    if end_sec <= start_sec:
        return {
            "hist": hist,
            "weighted_duration": 0.0,
            "event_count": 0,
            "onset_count": 0,
            "mean_confidence": 0.0,
        }

    for event in note_events:
        event_start = float(event["start"])
        event_end = float(event["end"])
        overlap = min(end_sec, event_end) - max(start_sec, event_start)
        if overlap <= 0.0:
            continue

        note_index = NOTE_INDEX.get(str(event.get("note_class", "")))
        confidence = float(event.get("confidence", 0.0) or 0.0)
        if note_index is not None:
            hist[note_index] += float(overlap) * max(confidence, 0.15)
        weighted_duration += float(overlap) * max(confidence, 0.15)
        event_count += 1
        confidence_total += confidence
        if start_sec <= event_start < end_sec:
            onset_count += 1

    return {
        "hist": hist,
        "weighted_duration": round(float(weighted_duration), 4),
        "event_count": int(event_count),
        "onset_count": int(onset_count),
        "mean_confidence": round(float(confidence_total / event_count), 4) if event_count else 0.0,
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 and right_norm <= 1e-8:
        return 1.0
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _compute_boundary_continuity(
    left_segment: dict[str, Any],
    right_segment: dict[str, Any],
    *,
    other_events: Sequence[dict[str, Any]],
    bass_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    boundary_sec = float(left_segment["end"])
    left_duration = max(0.0, float(left_segment["end"]) - float(left_segment["start"]))
    right_duration = max(0.0, float(right_segment["end"]) - float(right_segment["start"]))
    left_window = min(3.5, max(1.25, left_duration))
    right_window = min(3.5, max(1.25, right_duration))

    left_other = _event_window_stats(other_events, max(float(left_segment["start"]), boundary_sec - left_window), boundary_sec)
    right_other = _event_window_stats(other_events, boundary_sec, min(float(right_segment["end"]), boundary_sec + right_window))
    left_bass = _event_window_stats(bass_events, max(float(left_segment["start"]), boundary_sec - left_window), boundary_sec)
    right_bass = _event_window_stats(bass_events, boundary_sec, min(float(right_segment["end"]), boundary_sec + right_window))

    pitch_similarity = (
        0.72 * _cosine_similarity(left_other["hist"], right_other["hist"])
        + 0.28 * _cosine_similarity(left_bass["hist"], right_bass["hist"])
    )

    left_weighted = float(left_other["weighted_duration"]) + 0.65 * float(left_bass["weighted_duration"])
    right_weighted = float(right_other["weighted_duration"]) + 0.65 * float(right_bass["weighted_duration"])
    density_ratio = min(left_weighted, right_weighted) / max(left_weighted, right_weighted) if max(left_weighted, right_weighted) > 1e-6 else 1.0

    left_onsets = int(left_other["onset_count"]) + int(left_bass["onset_count"])
    right_onsets = int(right_other["onset_count"]) + int(right_bass["onset_count"])
    onset_ratio = min(left_onsets, right_onsets) / max(left_onsets, right_onsets) if max(left_onsets, right_onsets) > 0 else 1.0

    confidence_ratio = min(
        max(float(left_other["mean_confidence"]), float(left_bass["mean_confidence"])),
        max(float(right_other["mean_confidence"]), float(right_bass["mean_confidence"])),
    )

    continuity_score = (
        0.68 * pitch_similarity
        + 0.18 * density_ratio
        + 0.08 * onset_ratio
        + 0.06 * confidence_ratio
    )

    return {
        "pitch_similarity": round(float(pitch_similarity), 4),
        "density_ratio": round(float(density_ratio), 4),
        "onset_ratio": round(float(onset_ratio), 4),
        "confidence_ratio": round(float(confidence_ratio), 4),
        "continuity_score": round(float(continuity_score), 4),
        "event_count_total": int(left_other["event_count"] + right_other["event_count"] + left_bass["event_count"] + right_bass["event_count"]),
    }


def _should_merge_adjacent_segments(
    left_segment: dict[str, Any],
    right_segment: dict[str, Any],
    continuity: dict[str, Any],
) -> bool:
    left_type = str(left_segment.get("type", ""))
    right_type = str(right_segment.get("type", ""))
    if left_type in EDGE_TYPES or right_type in EDGE_TYPES:
        return False

    gap_sec = max(0.0, float(right_segment["start"]) - float(left_segment["end"]))
    if gap_sec > 1.25:
        return False

    left_duration = max(0.0, float(left_segment["end"]) - float(left_segment["start"]))
    right_duration = max(0.0, float(right_segment["end"]) - float(right_segment["start"]))
    combined_duration = left_duration + right_duration + gap_sec
    same_family = _segment_family(left_type) == _segment_family(right_type)
    same_hint = bool(left_segment.get("label_hint")) and left_segment.get("label_hint") == right_segment.get("label_hint")
    short_pair = left_duration <= 7.5 and right_duration <= 7.5

    strong_continuity = (
        float(continuity["continuity_score"]) >= 0.9
        and float(continuity["pitch_similarity"]) >= 0.82
    )
    moderate_continuity = (
        float(continuity["continuity_score"]) >= 0.84
        and float(continuity["pitch_similarity"]) >= 0.76
        and float(continuity["density_ratio"]) >= 0.42
    )

    if strong_continuity and combined_duration <= 40.0 and (same_family or same_hint or short_pair):
        return True
    if moderate_continuity and same_family and combined_duration <= 22.0:
        return True
    if moderate_continuity and short_pair and combined_duration <= 26.0 and (
        same_family or same_hint or int(continuity["event_count_total"]) >= 6
    ):
        return True
    return False


def _choose_merged_segment_type(
    left_segment: dict[str, Any],
    right_segment: dict[str, Any],
) -> str:
    left_weight = max(0.0, float(left_segment["end"]) - float(left_segment["start"])) * float(left_segment.get("importance", 1.0))
    right_weight = max(0.0, float(right_segment["end"]) - float(right_segment["start"])) * float(right_segment.get("importance", 1.0))
    return str(right_segment["type"]) if right_weight > left_weight else str(left_segment["type"])


def _choose_run_segment_type(run_segments: Sequence[dict[str, Any]]) -> str:
    weights: dict[str, float] = {}
    for segment in run_segments:
        duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        weight = duration * float(segment.get("importance", 1.0))
        weights[str(segment["type"])] = weights.get(str(segment["type"]), 0.0) + weight
    return max(weights.items(), key=lambda item: item[1])[0]


def _suppress_short_oscillation_runs(
    segments: Sequence[dict[str, Any]],
    *,
    other_events: Sequence[dict[str, Any]],
    bass_events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(segments) < 3:
        return list(segments), {"runs_merged": 0, "merge_debug": []}

    result: list[dict[str, Any]] = []
    merge_debug: list[dict[str, Any]] = []
    index = 0

    while index < len(segments):
        run = [dict(segments[index])]
        run_boundaries: list[dict[str, Any]] = []
        cursor = index
        while cursor + 1 < len(segments):
            left_segment = run[-1]
            right_segment = dict(segments[cursor + 1])
            left_duration = max(0.0, float(left_segment["end"]) - float(left_segment["start"]))
            right_duration = max(0.0, float(right_segment["end"]) - float(right_segment["start"]))
            run_duration = max(0.0, float(right_segment["end"]) - float(run[0]["start"]))

            if left_duration > 8.5 or right_duration > 8.5 or run_duration > 34.0:
                break
            if _segment_family(str(left_segment.get("type"))) == "edge" or _segment_family(str(right_segment.get("type"))) == "edge":
                break

            continuity = _compute_boundary_continuity(
                left_segment,
                right_segment,
                other_events=other_events,
                bass_events=bass_events,
            )
            if (
                float(continuity["density_ratio"]) < 0.55
                or float(continuity["onset_ratio"]) < 0.68
                or float(continuity["confidence_ratio"]) < 0.42
                or int(continuity["event_count_total"]) < 18
            ):
                break

            run.append(right_segment)
            run_boundaries.append(continuity)
            cursor += 1

        distinct_types = {str(segment["type"]) for segment in run}
        if len(run) >= 3 and len(distinct_types) >= 2:
            merged = dict(run[0])
            merged["end"] = float(run[-1]["end"])
            merged["type"] = _choose_run_segment_type(run)
            merged["importance"] = max(float(segment.get("importance", 1.0)) for segment in run)
            if not merged.get("label_hint"):
                merged["label_hint"] = next((segment.get("label_hint") for segment in run if segment.get("label_hint")), None)
            result.append(merged)
            merge_debug.append(
                {
                    "run_start": round(float(run[0]["start"]), 4),
                    "run_end": round(float(run[-1]["end"]), 4),
                    "segments_merged": len(run),
                    "types": [str(segment["type"]) for segment in run],
                    "avg_pitch_similarity": round(float(np.mean([item["pitch_similarity"] for item in run_boundaries])), 4) if run_boundaries else None,
                    "avg_density_ratio": round(float(np.mean([item["density_ratio"] for item in run_boundaries])), 4) if run_boundaries else None,
                    "avg_onset_ratio": round(float(np.mean([item["onset_ratio"] for item in run_boundaries])), 4) if run_boundaries else None,
                }
            )
            index = cursor + 1
            continue

        result.append(dict(segments[index]))
        index += 1

    return result, {"runs_merged": len(merge_debug), "merge_debug": merge_debug[:12]}


def _split_undersegmented_long_segments(
    segments: Sequence[dict[str, Any]],
    *,
    other_events: Sequence[dict[str, Any]],
    bass_events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_segments: list[dict[str, Any]] = []
    split_debug: list[dict[str, Any]] = []

    for segment in segments:
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        duration_sec = max(0.0, end_sec - start_sec)
        if duration_sec < 38.0 or _segment_family(str(segment.get("type"))) == "edge":
            split_segments.append(dict(segment))
            continue

        candidates: list[tuple[float, float, dict[str, Any]]] = []
        probe_times = np.arange(start_sec + 8.0, end_sec - 8.0, 2.0, dtype=np.float32)
        for boundary_sec in probe_times:
            left_probe = {"start": start_sec, "end": float(boundary_sec), "type": segment["type"], "importance": segment.get("importance", 1.0)}
            right_probe = {"start": float(boundary_sec), "end": end_sec, "type": segment["type"], "importance": segment.get("importance", 1.0)}
            continuity = _compute_boundary_continuity(
                left_probe,
                right_probe,
                other_events=other_events,
                bass_events=bass_events,
            )
            split_score = (
                0.72 * (1.0 - float(continuity["pitch_similarity"]))
                + 0.12 * (1.0 - float(continuity["density_ratio"]))
                + 0.16 * min(1.0, float(continuity["event_count_total"]) / 72.0)
            )
            if (
                float(continuity["pitch_similarity"]) <= 0.42
                and int(continuity["event_count_total"]) >= 24
                and split_score >= 0.62
            ):
                candidates.append((split_score, float(boundary_sec), continuity))

        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen_boundaries: list[tuple[float, dict[str, Any]]] = []
        for split_score, boundary_sec, continuity in candidates:
            if all(abs(boundary_sec - existing_boundary) >= 16.0 for existing_boundary, _ in chosen_boundaries):
                chosen_boundaries.append((boundary_sec, continuity))
            if len(chosen_boundaries) >= 2:
                break

        if not chosen_boundaries:
            split_segments.append(dict(segment))
            continue

        boundaries = [start_sec] + [boundary for boundary, _ in sorted(chosen_boundaries, key=lambda item: item[0])] + [end_sec]
        for part_start, part_end in zip(boundaries[:-1], boundaries[1:]):
            child = dict(segment)
            child["start"] = float(part_start)
            child["end"] = float(part_end)
            split_segments.append(child)

        split_debug.append(
            {
                "segment_start": round(start_sec, 4),
                "segment_end": round(end_sec, 4),
                "segment_type": str(segment["type"]),
                "split_points": [
                    {
                        "boundary_sec": round(boundary_sec, 4),
                        "pitch_similarity": continuity["pitch_similarity"],
                        "density_ratio": continuity["density_ratio"],
                        "event_count_total": continuity["event_count_total"],
                    }
                    for boundary_sec, continuity in chosen_boundaries
                ],
            }
        )

    return split_segments, {"segments_split": len(split_debug), "split_debug": split_debug[:12]}


def refine_segments_with_note_events(
    segments: Sequence[dict[str, Any]],
    *,
    other_events: Sequence[dict[str, Any]],
    bass_events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sorted_segments = sorted((dict(segment) for segment in segments), key=lambda item: float(item["start"]))
    if len(sorted_segments) < 2:
        return list(sorted_segments), {
            "status": "skipped",
            "reason": "not_enough_segments",
            "original_segments": len(sorted_segments),
            "refined_segments": len(sorted_segments),
            "merges_applied": 0,
        }

    if not other_events and not bass_events:
        return list(sorted_segments), {
            "status": "skipped",
            "reason": "no_note_events",
            "original_segments": len(sorted_segments),
            "refined_segments": len(sorted_segments),
            "merges_applied": 0,
        }

    refined: list[dict[str, Any]] = []
    merge_debug: list[dict[str, Any]] = []
    current = _copy_segment(sorted_segments[0])
    merges_applied = 0

    for raw_next in sorted_segments[1:]:
        next_segment = _copy_segment(raw_next)
        continuity = _compute_boundary_continuity(
            current,
            next_segment,
            other_events=other_events,
            bass_events=bass_events,
        )
        if _should_merge_adjacent_segments(current, next_segment, continuity):
            merges_applied += 1
            merge_debug.append(
                {
                    "left_type": current["type"],
                    "right_type": next_segment["type"],
                    "boundary_sec": round(float(current["end"]), 4),
                    **continuity,
                }
            )
            current["end"] = float(next_segment["end"])
            current["type"] = _choose_merged_segment_type(current, next_segment)
            current["importance"] = max(float(current.get("importance", 1.0)), float(next_segment.get("importance", 1.0)))
            if not current.get("label_hint"):
                current["label_hint"] = next_segment.get("label_hint")
            current["_merged_children"].extend(next_segment["_merged_children"])
            continue

        current.pop("_merged_children", None)
        refined.append(current)
        current = next_segment

    current.pop("_merged_children", None)
    refined.append(current)

    oscillation_refined, oscillation_summary = _suppress_short_oscillation_runs(
        refined,
        other_events=other_events,
        bass_events=bass_events,
    )
    split_refined, split_summary = _split_undersegmented_long_segments(
        oscillation_refined,
        other_events=other_events,
        bass_events=bass_events,
    )

    return split_refined, {
        "status": "ok",
        "original_segments": len(sorted_segments),
        "refined_segments": len(split_refined),
        "merges_applied": merges_applied,
        "merge_debug": merge_debug[:12],
        "oscillation_runs_merged": oscillation_summary["runs_merged"],
        "oscillation_merge_debug": oscillation_summary["merge_debug"],
        "segments_split": split_summary["segments_split"],
        "split_debug": split_summary["split_debug"],
    }


def refine_segments_with_melodic_continuity(
    track_id: int,
    audio_path: str | Path,
    segments: Sequence[dict[str, Any]],
    *,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(segments) < 2:
        return list(segments), {
            "status": "skipped",
            "reason": "not_enough_segments",
            "track_id": int(track_id),
            "original_segments": len(segments),
            "refined_segments": len(segments),
            "merges_applied": 0,
        }

    try:
        payload = load_or_create_note_events(
            audio_path,
            stems=("other", "bass"),
            separator_model=separator_model,
            device=device,
        )
    except Exception as exc:
        logger.warning("Track %s: failed to load melodic note events for segment refinement: %s", track_id, exc)
        return list(segments), {
            "status": "failed",
            "reason": "load_or_create_note_events_failed",
            "track_id": int(track_id),
            "error": str(exc),
            "original_segments": len(segments),
            "refined_segments": len(segments),
            "merges_applied": 0,
        }

    other_stem = payload.get("stems", {}).get("other", {})
    bass_stem = payload.get("stems", {}).get("bass", {})
    other_events = other_stem.get("orchestral_note_events") or other_stem.get("note_events") or []
    bass_events = bass_stem.get("note_events") or []

    refined_segments, summary = refine_segments_with_note_events(
        segments,
        other_events=other_events,
        bass_events=bass_events,
    )
    summary.update(
        {
            "track_id": int(track_id),
            "separator_model": separator_model,
            "other_event_count": len(other_events),
            "bass_event_count": len(bass_events),
        }
    )
    return refined_segments, summary
