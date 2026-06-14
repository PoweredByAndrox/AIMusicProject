from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.no_reference_score_matrix import pair_compatibility_score, score_candidate  # noqa: E402
from tools.run_no_reference_result_judge import (  # noqa: E402
    align_audio,
    as_list,
    judge_audio,
    read_audio,
    resample_audio,
    resolve_path,
    stable_unique,
)


DEFAULT_MIN_DELTA = 0.01
DEFAULT_PHASE_TRIGGER = 0.55
DEFAULT_MAX_OPPOSITE_REGRESSION = 0.05
DEFAULT_MIN_WINDOW_LOCAL_WHOLE_TRACK_DELTA = 0.001
DEFAULT_WINDOW_CROSSFADE_MS = 50.0
PHASE_EVIDENCE_REASONS = {
    "phase_or_stereo_inconsistency",
    "severe_phase_comb_filtering",
    "severe_phase_collapse",
}

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_edit_model_registry",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_best_of_bad",
]

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported_as_result": False,
    "can_publish_as_result": False,
    "can_mark_release_ready": False,
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def short_slug(value: str, max_length: int = 72) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    clean = "_".join(part for part in clean.split("_") if part)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    if len(clean) > max_length:
        clean = clean[:max_length].rstrip("_-.")
    return f"{clean or 'candidate'}_{digest}"


def metric(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def mode_fitness(metrics: dict[str, Any]) -> float:
    return round(
        0.42 * metric(metrics, "vocal_score")
        + 0.30 * metric(metrics, "minus_score")
        + 0.18 * metric(metrics, "pair_compatibility_score")
        + 0.10 * metric(metrics, "overall_no_reference_score"),
        6,
    )


def evaluate_pair(mix: np.ndarray, vocal: np.ndarray, minus: np.ndarray, sample_rate: int) -> dict[str, Any]:
    metrics = judge_audio(mix, vocal, minus, sample_rate)
    metrics.update(score_candidate(metrics, "vocal"))
    metrics.update(score_candidate(metrics, "minus"))
    metrics.update(pair_compatibility_score(metrics, metrics))
    metrics["phase_alignment_fitness"] = mode_fitness(metrics)
    return metrics


def interval_phase_fitness(
    mix: np.ndarray,
    vocal: np.ndarray,
    minus: np.ndarray,
    sample_rate: int,
    intervals: Sequence[tuple[int, int]],
) -> float | None:
    weighted_sum = 0.0
    total_frames = 0
    for start, end in intervals:
        if end <= start:
            continue
        metrics = evaluate_pair(mix[start:end], vocal[start:end], minus[start:end], sample_rate)
        frames = end - start
        weighted_sum += metric(metrics, "phase_alignment_fitness") * frames
        total_frames += frames
    if total_frames <= 0:
        return None
    return round(weighted_sum / total_frames, 6)


def load_aligned_audio(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    missing = [
        key
        for key in ("mix_path", "vocal_path")
        if not row.get(key)
    ]
    if not row.get("instrumental_path") and not row.get("minus_path"):
        missing.append("instrumental_path")
    if missing:
        raise ValueError(f"missing_required_audio_path:{','.join(missing)}")
    mix_path = resolve_path(row["mix_path"])
    vocal_path = resolve_path(row["vocal_path"])
    minus_path = resolve_path(row.get("instrumental_path") or row.get("minus_path"))
    for label, path in (("mix_path", mix_path), ("vocal_path", vocal_path), ("instrumental_path", minus_path)):
        if not path.exists():
            raise ValueError(f"missing_required_audio_path:{label}:{path}")
    mix, sample_rate = read_audio(mix_path)
    vocal, vocal_rate = read_audio(vocal_path)
    minus, minus_rate = read_audio(minus_path)
    vocal = resample_audio(vocal, vocal_rate, sample_rate)
    minus = resample_audio(minus, minus_rate, sample_rate)
    mix, vocal, minus = align_audio([mix, vocal, minus])
    return mix.astype(np.float32), vocal.astype(np.float32), minus.astype(np.float32), sample_rate


def has_phase_issue(metrics: dict[str, Any], phase_trigger: float, windowed_evidence_rows: Sequence[dict[str, Any]] | None = None) -> bool:
    reasons = set(as_list(metrics.get("reasons")) + as_list(metrics.get("vocal_blockers")) + as_list(metrics.get("minus_blockers")))
    return (
        metric(metrics, "vocal_phase_consistency_score", metric(metrics, "phase_consistency_score", 1.0)) < phase_trigger
        or metric(metrics, "instrumental_phase_consistency_score", metric(metrics, "phase_consistency_score", 1.0)) < phase_trigger
        or bool(reasons.intersection(PHASE_EVIDENCE_REASONS))
        or windowed_phase_evidence_count(windowed_evidence_rows or []) > 0
    )


def windowed_phase_evidence_count(rows: Sequence[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        evidence = (
            as_list(row.get("reasons"))
            + as_list(row.get("vocal_blockers"))
            + as_list(row.get("minus_blockers"))
            + as_list(row.get("vocal_dominant_defect"))
            + as_list(row.get("minus_dominant_defect"))
            + as_list(row.get("dominant_defects"))
        )
        if PHASE_EVIDENCE_REASONS.intersection(str(item) for item in evidence):
            count += 1
    return count


def parse_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def has_phase_evidence(row: dict[str, Any]) -> bool:
    evidence = (
        as_list(row.get("reasons"))
        + as_list(row.get("vocal_blockers"))
        + as_list(row.get("minus_blockers"))
        + as_list(row.get("vocal_dominant_defect"))
        + as_list(row.get("minus_dominant_defect"))
        + as_list(row.get("dominant_defects"))
    )
    return bool(PHASE_EVIDENCE_REASONS.intersection(str(item) for item in evidence))


def iter_window_bound_rows(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield row
    for key in ("window_metrics", "windows"):
        nested = row.get(key)
        if not isinstance(nested, list):
            continue
        for item in nested:
            if isinstance(item, dict):
                yield item


def windowed_phase_intervals(
    rows: Sequence[dict[str, Any]],
    *,
    sample_rate: int,
    total_frames: int,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for row in rows:
        for bound_row in iter_window_bound_rows(row):
            if not has_phase_evidence(bound_row):
                continue
            start_sec = parse_float(first_present(bound_row, "window_start_sec", "start_sec"))
            duration_sec = parse_float(first_present(bound_row, "window_duration_sec", "duration_sec"))
            end_sec = parse_float(first_present(bound_row, "window_end_sec", "end_sec"))
            if start_sec is None:
                continue
            if end_sec is None and duration_sec is not None:
                end_sec = start_sec + duration_sec
            if end_sec is None or end_sec <= start_sec:
                continue
            start = max(0, min(total_frames, int(round(start_sec * sample_rate))))
            end = max(0, min(total_frames, int(round(end_sec * sample_rate))))
            if end > start:
                intervals.append((start, end))
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def window_mask(total_frames: int, intervals: Sequence[tuple[int, int]], *, sample_rate: int, crossfade_ms: float = DEFAULT_WINDOW_CROSSFADE_MS) -> np.ndarray:
    mask = np.zeros(total_frames, dtype=np.float32)
    fade_samples = max(0, int(round(sample_rate * crossfade_ms / 1000.0)))
    for start, end in intervals:
        length = end - start
        if length <= 0:
            continue
        segment = np.ones(length, dtype=np.float32)
        edge = min(fade_samples, length // 2)
        if edge > 0:
            segment[:edge] = np.linspace(0.0, 1.0, edge, endpoint=False, dtype=np.float32)
            segment[-edge:] = np.linspace(1.0, 0.0, edge, endpoint=False, dtype=np.float32)
        mask[start:end] = np.maximum(mask[start:end], segment)
    return mask


def apply_window_mask(original: np.ndarray, corrected: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shaped_mask = mask[:, None] if original.ndim == 2 else mask
    return (original * (1.0 - shaped_mask) + corrected * shaped_mask).astype(np.float32)


def windowed_evidence_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("track_id") or ""), str(row.get("candidate_id") or "")


def build_windowed_evidence_map(rows: Sequence[dict[str, Any]] | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows or []:
        key = windowed_evidence_key(row)
        if not all(key):
            continue
        result.setdefault(key, []).append(row)
    return result


def compensated_variants(vocal: np.ndarray, minus: np.ndarray) -> list[dict[str, Any]]:
    if vocal.ndim < 2 or vocal.shape[1] < 2:
        return []
    variants: list[dict[str, Any]] = []
    channels = vocal.shape[1]

    mid = np.mean(vocal, axis=1, keepdims=True)
    corrected = np.repeat(mid, channels, axis=1).astype(np.float32)
    variants.append(
        {
            "variant_id": "vocal_mid_mono_compensated",
            "corrected_vocal": corrected,
            "corrected_minus": (minus + (vocal - corrected)).astype(np.float32),
            "action": "bounded_phase_alignment_candidate",
        }
    )

    for channel_index, name in ((0, "vocal_flip_left_compensated"), (1, "vocal_flip_right_compensated")):
        corrected = vocal.copy()
        corrected[:, channel_index] *= -1.0
        variants.append(
            {
                "variant_id": name,
                "corrected_vocal": corrected.astype(np.float32),
                "corrected_minus": (minus + (vocal - corrected)).astype(np.float32),
                "action": "bounded_phase_alignment_candidate",
            }
        )
    return variants


def window_local_compensated_variants(
    vocal: np.ndarray,
    minus: np.ndarray,
    *,
    intervals: Sequence[tuple[int, int]],
    sample_rate: int,
) -> list[dict[str, Any]]:
    if not intervals:
        return []
    mask = window_mask(vocal.shape[0], intervals, sample_rate=sample_rate)
    variants: list[dict[str, Any]] = []
    for variant in compensated_variants(vocal, minus):
        variants.append(
            {
                "variant_id": f"{variant['variant_id']}__window_local",
                "corrected_vocal": apply_window_mask(vocal, variant["corrected_vocal"], mask),
                "corrected_minus": apply_window_mask(minus, variant["corrected_minus"], mask),
                "action": "bounded_window_local_phase_alignment_candidate",
                "window_count": len(intervals),
                "source_variant_id": variant["variant_id"],
            }
        )
    return variants


def reject_variant(baseline: dict[str, Any], candidate: dict[str, Any], *, max_opposite_regression: float) -> str | None:
    if as_list(candidate.get("vocal_blockers")):
        return "new_or_remaining_vocal_blocker"
    if as_list(candidate.get("minus_blockers")):
        return "new_or_remaining_minus_blocker"
    if as_list(candidate.get("pair_blockers")):
        return "pair_compatibility_blocker"
    if metric(candidate, "minus_score") < metric(baseline, "minus_score") - max_opposite_regression:
        return "minus_regression_severe"
    if metric(candidate, "vocal_score") < metric(baseline, "vocal_score"):
        return "vocal_regression"
    if metric(candidate, "overall_no_reference_score") < metric(baseline, "overall_no_reference_score") - 0.05:
        return "overall_score_regression"
    return None


def write_candidate_audio(output_dir: Path, row: dict[str, Any], variant_id: str, vocal: np.ndarray, minus: np.ndarray, sample_rate: int) -> tuple[str, str]:
    candidate_dir = output_dir / "phase_alignment_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    stem = short_slug(f"{row.get('track_id')}_{row.get('candidate_id')}_{variant_id}")
    vocal_path = candidate_dir / f"{stem}__vocal.wav"
    minus_path = candidate_dir / f"{stem}__minus.wav"
    sf.write(vocal_path, vocal.astype(np.float32), sample_rate, subtype="FLOAT")
    sf.write(minus_path, minus.astype(np.float32), sample_rate, subtype="FLOAT")
    return str(vocal_path), str(minus_path)


def blocked_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "track_id": str(row.get("track_id") or ""),
        "input_candidate_id": str(row.get("candidate_id") or ""),
        "selected_candidate_id": None,
        "optimization_iterations": 0,
        "dominant_defects_before": [],
        "actions_tried": [],
        "actions_rejected": [],
        "fitness_before": 0.0,
        "fitness_after": 0.0,
        "improvement_delta": 0.0,
        "final_decision": "blocked_hard_case",
        "selected_vocal_path": "",
        "selected_minus_path": "",
        "reasons": stable_unique([reason, "no_best_of_bad"]),
        "rollback_reason": reason,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_correction_judge": False,
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + as_list(row.get("forbidden_actions"))),
    }


def optimize_candidate(
    row: dict[str, Any],
    *,
    output_dir: Path,
    min_delta: float,
    phase_trigger: float,
    max_opposite_regression: float,
    min_window_local_whole_track_delta: float,
    windowed_evidence_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        mix, vocal, minus, sample_rate = load_aligned_audio(row)
    except ValueError as exc:
        return blocked_row(row, str(exc).split(":", maxsplit=1)[0])

    baseline = evaluate_pair(mix, vocal, minus, sample_rate)
    evidence_count = windowed_phase_evidence_count(windowed_evidence_rows or [])
    evidence_intervals = windowed_phase_intervals(
        windowed_evidence_rows or [],
        sample_rate=sample_rate,
        total_frames=vocal.shape[0],
    )
    fitness_before = metric(baseline, "phase_alignment_fitness")
    local_window_fitness_before = interval_phase_fitness(mix, vocal, minus, sample_rate, evidence_intervals)
    dominant_defects = stable_unique(
        as_list(baseline.get("vocal_dominant_defect"))
        + as_list(baseline.get("minus_dominant_defect"))
    )

    base = {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "mix_path": str(row.get("mix_path") or ""),
        "vocal_path": str(row.get("vocal_path") or ""),
        "instrumental_path": str(row.get("instrumental_path") or row.get("minus_path") or ""),
        "minus_path": str(row.get("minus_path") or row.get("instrumental_path") or ""),
        "source_candidate_id": str(row.get("candidate_id") or ""),
        "input_candidate_id": str(row.get("candidate_id") or ""),
        "selected_candidate_id": str(row.get("candidate_id") or ""),
        "optimization_iterations": 0,
        "dominant_defects_before": dominant_defects,
        "actions_tried": [],
        "actions_rejected": [],
        "fitness_before": fitness_before,
        "fitness_after": fitness_before,
        "vocal_score_before": metric(baseline, "vocal_score"),
        "vocal_score_after": metric(baseline, "vocal_score"),
        "minus_score_before": metric(baseline, "minus_score"),
        "minus_score_after": metric(baseline, "minus_score"),
        "mixture_consistency_before": metric(baseline, "mixture_consistency_score"),
        "mixture_consistency_after": metric(baseline, "mixture_consistency_score"),
        "vocal_phase_before": metric(baseline, "vocal_phase_consistency_score", metric(baseline, "phase_consistency_score")),
        "vocal_phase_after": metric(baseline, "vocal_phase_consistency_score", metric(baseline, "phase_consistency_score")),
        "minus_phase_before": metric(baseline, "instrumental_phase_consistency_score", metric(baseline, "phase_consistency_score")),
        "minus_phase_after": metric(baseline, "instrumental_phase_consistency_score", metric(baseline, "phase_consistency_score")),
        "improvement_delta": 0.0,
        "whole_track_improvement_delta": 0.0,
        "local_window_fitness_before": local_window_fitness_before,
        "local_window_fitness_after": local_window_fitness_before,
        "local_window_improvement_delta": 0.0,
        "final_decision": "original_candidate_kept",
        "selected_vocal_path": str(row.get("vocal_path") or ""),
        "selected_minus_path": str(row.get("instrumental_path") or row.get("minus_path") or ""),
        "reasons": stable_unique(as_list(baseline.get("reasons"))),
        "rollback_reason": "no_phase_candidate_variant_needed",
        "windowed_phase_evidence_count": evidence_count,
        "windowed_phase_windows_used": len(evidence_intervals),
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_correction_judge": False,
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + as_list(row.get("forbidden_actions"))),
    }

    if evidence_count > 0:
        base["reasons"] = stable_unique(as_list(base.get("reasons")) + ["windowed_phase_evidence_triggered"])

    if not has_phase_issue(baseline, phase_trigger, windowed_evidence_rows):
        return base

    variants = window_local_compensated_variants(
        vocal,
        minus,
        intervals=evidence_intervals,
        sample_rate=sample_rate,
    ) + compensated_variants(vocal, minus)
    base["optimization_iterations"] = len(variants)
    base["actions_tried"] = stable_unique(str(variant["action"]) for variant in variants)
    if not variants:
        base["rollback_reason"] = "no_stereo_phase_candidate_available"
        return base

    scored: list[tuple[float, float, float | None, str, str, dict[str, Any], np.ndarray, np.ndarray]] = []
    for variant in variants:
        variant_metrics = evaluate_pair(mix, variant["corrected_vocal"], variant["corrected_minus"], sample_rate)
        rejection = reject_variant(baseline, variant_metrics, max_opposite_regression=max_opposite_regression)
        whole_track_delta = round(metric(variant_metrics, "phase_alignment_fitness") - fitness_before, 6)
        local_window_delta = None
        selection_delta = whole_track_delta
        if variant["action"] == "bounded_window_local_phase_alignment_candidate" and local_window_fitness_before is not None:
            local_window_fitness_after = interval_phase_fitness(
                mix,
                variant["corrected_vocal"],
                variant["corrected_minus"],
                sample_rate,
                evidence_intervals,
            )
            if local_window_fitness_after is not None:
                local_window_delta = round(local_window_fitness_after - local_window_fitness_before, 6)
                selection_delta = local_window_delta
        if (
            rejection is None
            and variant["action"] == "bounded_window_local_phase_alignment_candidate"
            and whole_track_delta < min_window_local_whole_track_delta
        ):
            rejection = "window_local_whole_track_delta_below_floor"
        if rejection is not None:
            rejected_action = {
                "variant_id": variant["variant_id"],
                "action": variant["action"],
                "reason": rejection,
                "fitness_delta": selection_delta,
                "whole_track_fitness_delta": whole_track_delta,
                "local_window_fitness_delta": local_window_delta,
            }
            if rejection == "window_local_whole_track_delta_below_floor":
                rejected_action["min_window_local_whole_track_delta"] = min_window_local_whole_track_delta
            base["actions_rejected"].append(rejected_action)
            continue
        scored.append(
            (
                selection_delta,
                whole_track_delta,
                local_window_delta,
                str(variant["variant_id"]),
                str(variant["action"]),
                variant_metrics,
                variant["corrected_vocal"],
                variant["corrected_minus"],
            )
        )

    if not scored:
        if any(action.get("reason") == "window_local_whole_track_delta_below_floor" for action in base["actions_rejected"]):
            base["rollback_reason"] = "window_local_whole_track_delta_below_floor"
        else:
            base["rollback_reason"] = "all_phase_candidate_variants_rejected"
        return base
    scored.sort(key=lambda item: (-item[0], -item[1], item[3]))
    best_delta, best_whole_track_delta, best_local_window_delta, best_variant_id, best_action, best_metrics, best_vocal, best_minus = scored[0]
    if best_delta < min_delta:
        base["rollback_reason"] = "fitness_improvement_below_min_delta"
        base["actions_rejected"].append(
            {
                "variant_id": best_variant_id,
                "action": best_action,
                "reason": "fitness_improvement_below_min_delta",
                "fitness_delta": best_delta,
                "whole_track_fitness_delta": best_whole_track_delta,
                "local_window_fitness_delta": best_local_window_delta,
            }
        )
        return base

    selected_vocal_path, selected_minus_path = write_candidate_audio(output_dir, row, best_variant_id, best_vocal, best_minus, sample_rate)
    selected_candidate_id = f"{row.get('candidate_id')}__{best_variant_id}"
    local_window_fitness_after = None
    if local_window_fitness_before is not None:
        local_window_fitness_after = interval_phase_fitness(mix, best_vocal, best_minus, sample_rate, evidence_intervals)
    base.update(
        {
            "candidate_id": selected_candidate_id,
            "selected_candidate_id": selected_candidate_id,
            "fitness_after": metric(best_metrics, "phase_alignment_fitness"),
            "vocal_score_after": metric(best_metrics, "vocal_score"),
            "minus_score_after": metric(best_metrics, "minus_score"),
            "mixture_consistency_after": metric(best_metrics, "mixture_consistency_score"),
            "vocal_phase_after": metric(best_metrics, "vocal_phase_consistency_score", metric(best_metrics, "phase_consistency_score")),
            "minus_phase_after": metric(best_metrics, "instrumental_phase_consistency_score", metric(best_metrics, "phase_consistency_score")),
            "improvement_delta": best_delta,
            "whole_track_improvement_delta": best_whole_track_delta,
            "local_window_fitness_after": local_window_fitness_after,
            "local_window_improvement_delta": best_local_window_delta or 0.0,
            "final_decision": "phase_corrected_candidate_selected",
            "selected_vocal_path": selected_vocal_path,
            "selected_minus_path": selected_minus_path,
            "reasons": stable_unique([f"dominant_defect:{defect}" for defect in dominant_defects] + ["phase_alignment_candidate_improved"]),
            "rollback_reason": "",
            "requires_post_correction_judge": True,
            "candidate_manifest": "",
            "mix_path": str(row.get("mix_path") or ""),
            "instrumental_path": selected_minus_path,
            "minus_path": selected_minus_path,
            "vocal_path": selected_vocal_path,
        }
    )
    return base


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(row.get("final_decision") or "") for row in rows)
    rejected = Counter(
        str(action.get("reason") or "")
        for row in rows
        for action in row.get("actions_rejected", [])
        if isinstance(action, dict)
    )
    selected = decisions.get("phase_corrected_candidate_selected", 0)
    safety = dict(SAFETY_FLAGS)
    safety["preview_candidate_audio_rendered"] = selected > 0
    return {
        "total_candidates": len(rows),
        "phase_corrected_candidate_selected": selected,
        "original_candidate_kept": decisions.get("original_candidate_kept", 0),
        "blocked_hard_case": decisions.get("blocked_hard_case", 0),
        "average_fitness_before": round(sum(metric(row, "fitness_before") for row in rows) / len(rows), 6) if rows else 0.0,
        "average_fitness_after": round(sum(metric(row, "fitness_after") for row in rows) / len(rows), 6) if rows else 0.0,
        "average_improvement_delta": round(sum(metric(row, "improvement_delta") for row in rows) / len(rows), 6) if rows else 0.0,
        "top_rejected_actions": dict(rejected.most_common(10)),
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "requires_post_correction_judge_count": sum(1 for row in rows if row.get("requires_post_correction_judge") is True),
        "safety": safety,
    }


def build_phase_alignment_result(
    candidate_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    min_delta: float = DEFAULT_MIN_DELTA,
    phase_trigger: float = DEFAULT_PHASE_TRIGGER,
    max_opposite_regression: float = DEFAULT_MAX_OPPOSITE_REGRESSION,
    min_window_local_whole_track_delta: float = DEFAULT_MIN_WINDOW_LOCAL_WHOLE_TRACK_DELTA,
    max_candidates: int = 0,
    windowed_judge_rows: Sequence[dict[str, Any]] | None = None,
    emit_progress: bool = False,
) -> dict[str, Any]:
    ordered = sorted(
        candidate_rows,
        key=lambda row: (
            str(row.get("track_id") or ""),
            str(row.get("candidate_id") or ""),
            str(row.get("vocal_path") or ""),
            str(row.get("instrumental_path") or row.get("minus_path") or ""),
        ),
    )
    if max_candidates > 0:
        ordered = ordered[:max_candidates]
    windowed_evidence = build_windowed_evidence_map(windowed_judge_rows)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        optimized = optimize_candidate(
            row,
            output_dir=output_dir,
            min_delta=min_delta,
            phase_trigger=phase_trigger,
            max_opposite_regression=max_opposite_regression,
            min_window_local_whole_track_delta=min_window_local_whole_track_delta,
            windowed_evidence_rows=windowed_evidence.get(windowed_evidence_key(row), []),
        )
        rows.append(optimized)
        if emit_progress:
            print(
                json.dumps(
                    {
                        "event": "no_reference_phase_alignment_optimizer_progress",
                        "index": index,
                        "total": len(ordered),
                        "track_id": str(row.get("track_id") or ""),
                        "candidate_id": str(row.get("candidate_id") or ""),
                        "final_decision": optimized["final_decision"],
                        "elapsed_sec": round(time.monotonic() - started, 3),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    return {"rows": rows, "summary": build_summary(rows)}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Phase Alignment Optimizer Summary",
        "",
        f"- total_candidates: {summary['total_candidates']}",
        f"- phase_corrected_candidate_selected: {summary['phase_corrected_candidate_selected']}",
        f"- original_candidate_kept: {summary['original_candidate_kept']}",
        f"- blocked_hard_case: {summary['blocked_hard_case']}",
        f"- average_fitness_before: {summary['average_fitness_before']}",
        f"- average_fitness_after: {summary['average_fitness_after']}",
        f"- average_improvement_delta: {summary['average_improvement_delta']}",
        f"- can_publish_as_result_count: {summary['can_publish_as_result_count']}",
        f"- can_mark_release_ready_count: {summary['can_mark_release_ready_count']}",
        f"- requires_post_correction_judge_count: {summary['requires_post_correction_judge_count']}",
        "",
        "## Top Rejected Actions",
        "",
    ]
    for key, value in summary["top_rejected_actions"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "no_reference_phase_alignment_optimizer_result.jsonl", result["rows"])
    write_summary_md(output_dir / "no_reference_phase_alignment_optimizer_summary.md", result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render bounded no-reference phase/stereo correction candidates.")
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--phase-trigger", type=float, default=DEFAULT_PHASE_TRIGGER)
    parser.add_argument("--max-opposite-regression", type=float, default=DEFAULT_MAX_OPPOSITE_REGRESSION)
    parser.add_argument("--min-window-local-whole-track-delta", type=float, default=DEFAULT_MIN_WINDOW_LOCAL_WHOLE_TRACK_DELTA)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--windowed-judge-jsonl", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    rows = list(iter_jsonl(args.candidates_jsonl))
    if not rows:
        raise SystemExit("No candidate rows supplied")
    print(
        json.dumps(
            {
                "event": "no_reference_phase_alignment_optimizer_start",
                "route": "bounded_no_reference_phase_alignment_candidate",
                "candidate_rows": len(rows),
                "output_dir": str(args.output_dir),
                "acceleration_mode": "external_only_cpu_dsp_candidate_render",
                "gpu_inference_skipped": True,
                "gpu_inference_skip_reason": "existing_audio_no_model_inference",
                "reference_required": False,
                "training_started": False,
                "models_changed": False,
                "cleanup_routes_added": False,
                "thresholds_relaxed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    result = build_phase_alignment_result(
        rows,
        output_dir=args.output_dir,
        min_delta=args.min_delta,
        phase_trigger=args.phase_trigger,
        max_opposite_regression=args.max_opposite_regression,
        min_window_local_whole_track_delta=args.min_window_local_whole_track_delta,
        max_candidates=args.max_candidates,
        windowed_judge_rows=[row for path in args.windowed_judge_jsonl for row in iter_jsonl(path)],
        emit_progress=True,
    )
    write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "event": "no_reference_phase_alignment_optimizer_end",
                "route": "bounded_no_reference_phase_alignment_candidate",
                "output_jsonl": str(args.output_dir / "no_reference_phase_alignment_optimizer_result.jsonl"),
                "summary_md": str(args.output_dir / "no_reference_phase_alignment_optimizer_summary.md"),
                **result["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
