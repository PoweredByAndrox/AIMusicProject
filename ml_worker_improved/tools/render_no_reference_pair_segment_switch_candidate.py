from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.render_no_reference_segment_switch_candidate import (
    SAFETY_FLAGS,
    as_list,
    as_list_of_dicts,
    candidate_minus_path,
    candidate_vocal_path,
    fit_channels,
    has_severe_reason,
    iter_jsonl,
    metric,
    read_audio,
    resample_audio,
    window_key,
    write_json,
)


def render_stem(
    selected_windows: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
    *,
    output_path: Path,
    path_getter: Callable[[dict[str, Any]], str],
    target_sample_rate: int | None = None,
    crossfade_ms: int = 80,
) -> tuple[int, int]:
    first_candidate = candidates_by_id[str(selected_windows[0]["selected_candidate_id"])]
    first_audio, sample_rate = read_audio(path_getter(first_candidate))
    target_rate = int(target_sample_rate or sample_rate)
    first_audio = resample_audio(first_audio, sample_rate, target_rate)
    channels = first_audio.shape[1]
    total_duration = max(float(row["window_start_sec"]) + float(row["window_duration_sec"]) for row in selected_windows)
    total_frames = max(1, int(round(total_duration * target_rate)))
    rendered = np.zeros((total_frames, channels), dtype=np.float32)
    weight = np.zeros((total_frames, 1), dtype=np.float32)
    fade_frames = max(0, int(round((crossfade_ms / 1000.0) * target_rate)))

    audio_cache: dict[str, np.ndarray] = {str(selected_windows[0]["selected_candidate_id"]): first_audio}
    for window in selected_windows:
        candidate_id = str(window["selected_candidate_id"])
        if candidate_id not in audio_cache:
            audio, source_rate = read_audio(path_getter(candidates_by_id[candidate_id]))
            audio_cache[candidate_id] = fit_channels(resample_audio(audio, source_rate, target_rate), channels)
        source = audio_cache[candidate_id]
        start = max(0, int(round(float(window["window_start_sec"]) * target_rate)))
        duration = max(1, int(round(float(window["window_duration_sec"]) * target_rate)))
        end = min(total_frames, start + duration)
        if end <= start:
            continue
        segment = source[start:end]
        if segment.shape[0] < end - start:
            padded = np.zeros((end - start, channels), dtype=np.float32)
            padded[: segment.shape[0]] = segment
            segment = padded
        curve = np.ones((end - start, 1), dtype=np.float32)
        if fade_frames > 0 and curve.shape[0] > 2:
            fade = min(fade_frames, curve.shape[0] // 2)
            curve[:fade] *= np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float32).reshape(-1, 1)
            curve[-fade:] *= np.linspace(1.0, 0.0, fade, endpoint=True, dtype=np.float32).reshape(-1, 1)
        rendered[start:end] += segment.astype(np.float32) * curve
        weight[start:end] += curve

    rendered = rendered / np.maximum(weight, 1.0e-6)
    rendered = np.clip(rendered, -1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), rendered, target_rate, subtype="PCM_24")
    return target_rate, int(rendered.shape[0])


def is_low_activity_proxy_gap(window: dict[str, Any]) -> bool:
    reasons = set(as_list(window.get("reasons")))
    vocal_blockers = set(as_list(window.get("vocal_blockers")))
    return reasons <= {"no_reference_proxy_passed"} and vocal_blockers == {"vocal_missing_on_active_region"}


def best_safe_windows_for_score(
    judge_rows: list[dict[str, Any]],
    *,
    min_window_score: float,
    score_field: str,
    decision_field: str | None = None,
    allow_select_alternative: bool = False,
    block_severe_reasons: bool = True,
    allow_low_activity_proxy_gap: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    choices_by_window: dict[tuple[float, float], list[dict[str, Any]]] = {}
    blocked: list[dict[str, Any]] = []
    allowed_decisions = {"accept_candidate"}
    if allow_select_alternative:
        allowed_decisions.add("select_alternative_candidate")

    for row in sorted(judge_rows, key=lambda item: str(item.get("candidate_id") or "")):
        candidate_id = str(row.get("candidate_id") or "")
        for window in as_list_of_dicts(row.get("window_metrics")):
            key = window_key(window)
            if key[1] <= 0:
                continue
            score = metric(window, score_field, metric(window, "window_fitness"))
            decision = str(window.get(decision_field) or "") if decision_field else ""
            enriched = {
                "candidate_id": candidate_id,
                "window_start_sec": key[0],
                "window_duration_sec": key[1],
                "selection_score_field": score_field,
                "selection_score": round(score, 6),
                "window_fitness": round(metric(window, "window_fitness", score), 6),
                "reasons": as_list(window.get("reasons")),
            }
            if score < min_window_score:
                blocked.append({**enriched, "blocked_reason": "selection_score_below_threshold"})
                continue
            low_activity_proxy_gap = allow_low_activity_proxy_gap and is_low_activity_proxy_gap(window)
            if decision_field and decision not in allowed_decisions:
                if not low_activity_proxy_gap:
                    blocked.append({**enriched, "blocked_reason": "selection_decision_not_allowed", "selection_decision": decision})
                    continue
                enriched["selection_decision"] = decision
                enriched["selection_note"] = "low_activity_proxy_gap"
            if block_severe_reasons and has_severe_reason(window):
                blocked.append({**enriched, "blocked_reason": "severe_window_reason"})
                continue
            choices_by_window.setdefault(key, []).append(enriched)

    selected: list[dict[str, Any]] = []
    for key, choices in sorted(choices_by_window.items()):
        selected.append(
            max(
                choices,
                key=lambda item: (
                    float(item["selection_score"]),
                    float(item["window_fitness"]),
                    str(item["candidate_id"]),
                ),
            )
        )
    expected_keys = sorted({window_key(window) for row in judge_rows for window in as_list_of_dicts(row.get("window_metrics")) if window_key(window)[1] > 0})
    selected_keys = {(row["window_start_sec"], row["window_duration_sec"]) for row in selected}
    missing = [
        {
            "window_start_sec": key[0],
            "window_duration_sec": key[1],
            "reason": f"no_safe_{score_field}_candidate_for_window",
        }
        for key in expected_keys
        if key not in selected_keys
    ]
    return selected, missing


def build_pair_segment_switch_candidate(
    candidate_rows: list[dict[str, Any]],
    windowed_judge_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    min_window_score: float = 0.72,
    crossfade_ms: int = 80,
    selection_score_field: str = "minus_score",
    selection_decision_field: str | None = None,
    allow_select_alternative: bool = False,
    block_severe_reasons: bool = True,
    allow_low_activity_proxy_gap: bool = False,
) -> dict[str, Any]:
    candidates_by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows if row.get("candidate_id")}
    selected, missing = best_safe_windows_for_score(
        windowed_judge_rows,
        min_window_score=min_window_score,
        score_field=selection_score_field,
        decision_field=selection_decision_field,
        allow_select_alternative=allow_select_alternative,
        block_severe_reasons=block_severe_reasons,
        allow_low_activity_proxy_gap=allow_low_activity_proxy_gap,
    )
    selected = [
        {
            **row,
            "selected_candidate_id": row["candidate_id"],
            "selected_vocal_path": candidate_vocal_path(candidates_by_id.get(str(row["candidate_id"]), {})),
            "selected_minus_path": candidate_minus_path(candidates_by_id.get(str(row["candidate_id"]), {})),
        }
        for row in selected
        if str(row["candidate_id"]) in candidates_by_id
        and candidate_vocal_path(candidates_by_id[str(row["candidate_id"])])
        and candidate_minus_path(candidates_by_id[str(row["candidate_id"])])
    ]
    base_manifest = {
        "manifest_type": "no_reference_pair_segment_switch_candidate",
        "track_id": str(candidate_rows[0].get("track_id") if candidate_rows else ""),
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_judge": True,
        "selected_window_count": len(selected),
        "blocked_window_count": len(missing),
        "selection_score_field": selection_score_field,
        "selection_decision_field": selection_decision_field or "",
        "selection_allows_select_alternative": allow_select_alternative,
        "selection_blocks_severe_reasons": block_severe_reasons,
        "selection_allows_low_activity_proxy_gap": allow_low_activity_proxy_gap,
        "selected_windows": selected,
        "blocked_windows": missing,
        "safety_flags": dict(SAFETY_FLAGS),
        "audio_copied": False,
        "training_started": False,
        "models_changed": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "rendered_vocal_preview_path": "",
        "rendered_minus_preview_path": "",
        "candidate_row_jsonl": "",
    }
    if not selected or missing:
        return {
            **base_manifest,
            "render_status": "blocked_incomplete_window_coverage",
            "audio_exported": False,
            "dominant_failure_reason": "incomplete_safe_pair_window_coverage",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = output_dir / "segment_switch_vocal_preview.flac"
    minus_path = output_dir / "segment_switch_minus_preview.flac"
    vocal_rate, vocal_frames = render_stem(
        selected,
        candidates_by_id,
        output_path=vocal_path,
        path_getter=candidate_vocal_path,
        crossfade_ms=crossfade_ms,
    )
    minus_rate, minus_frames = render_stem(
        selected,
        candidates_by_id,
        output_path=minus_path,
        path_getter=candidate_minus_path,
        target_sample_rate=vocal_rate,
        crossfade_ms=crossfade_ms,
    )
    frames = min(vocal_frames, minus_frames)
    candidate_row = {
        "track_id": base_manifest["track_id"],
        "candidate_id": "pair_segment_switch_preview",
        "mix_path": str(candidate_rows[0].get("mix_path") if candidate_rows else ""),
        "vocal_path": str(vocal_path),
        "instrumental_path": str(minus_path),
        "minus_path": str(minus_path),
        "candidate_family": "pair_segment_switch_preview",
        "candidate_origin": "existing_candidates_window_pair_switch",
        "is_complete_pair": True,
    }
    candidate_row_jsonl = output_dir / "pair_segment_switch_candidate_row.jsonl"
    with candidate_row_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(candidate_row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        **base_manifest,
        "render_status": "rendered_preview_candidate",
        "audio_exported": True,
        "dominant_failure_reason": "",
        "rendered_vocal_preview_path": str(vocal_path),
        "rendered_minus_preview_path": str(minus_path),
        "sample_rate": vocal_rate,
        "minus_sample_rate": minus_rate,
        "frames": frames,
        "duration_sec": round(float(frames) / float(vocal_rate), 6) if vocal_rate else 0.0,
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Pair Segment Switch Preview",
        "",
        f"- render_status: `{manifest.get('render_status')}`",
        f"- selected_window_count: `{manifest.get('selected_window_count')}`",
        f"- blocked_window_count: `{manifest.get('blocked_window_count')}`",
        f"- rendered_vocal_preview_path: `{manifest.get('rendered_vocal_preview_path')}`",
        f"- rendered_minus_preview_path: `{manifest.get('rendered_minus_preview_path')}`",
        "- delivery_allowed: `false`",
        "- can_publish_as_result: `false`",
        "- can_mark_release_ready: `false`",
        "- requires_post_render_judge: `true`",
        "",
        "This artifact is a preview-only candidate built from existing generated candidates. It is not a listen-pack or release result.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render preview-only vocal+minus segment-switch candidate from existing candidates.")
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--min-window-score", type=float, default=0.72)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    parser.add_argument("--selection-score-field", default="minus_score")
    parser.add_argument("--selection-decision-field", default="")
    parser.add_argument("--allow-select-alternative", action="store_true")
    parser.add_argument("--allow-low-activity-proxy-gap", action="store_true")
    parser.add_argument("--ignore-severe-reasons-for-selection", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_pair_segment_switch_candidate(
        list(iter_jsonl(args.candidate_pool_jsonl)),
        list(iter_jsonl(args.windowed_judge_jsonl)),
        output_dir=args.output_dir,
        min_window_score=args.min_window_score,
        crossfade_ms=args.crossfade_ms,
        selection_score_field=args.selection_score_field,
        selection_decision_field=args.selection_decision_field or None,
        allow_select_alternative=args.allow_select_alternative,
        block_severe_reasons=not args.ignore_severe_reasons_for_selection,
        allow_low_activity_proxy_gap=args.allow_low_activity_proxy_gap,
    )
    write_json(args.manifest_json, manifest)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary(manifest), encoding="utf-8")
    print(json.dumps({key: manifest.get(key) for key in ("render_status", "selected_window_count", "blocked_window_count", "delivery_allowed", "can_publish_as_result")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
