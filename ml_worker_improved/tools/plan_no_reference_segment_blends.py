from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.run_no_reference_optimizer import as_list, clip01, stable_unique  # noqa: E402


SEVERE_BLOCKERS = {
    "bad_mixture_reconstruction",
    "vocal_leakage_in_minus",
    "instrumental_leakage_in_vocal",
    "silence_hallucination",
    "robotic_or_noise_artifact",
    "thin_or_overcleaned_vocal",
    "transient_damage",
    "phase_or_stereo_inconsistency",
    "low_no_reference_score",
}

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_execute_segment_blend_without_approval",
    "do_not_export_or_copy_audio",
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
    "audio_exported": False,
    "audio_copied": False,
    "segment_blend_executed": False,
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


def metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def candidate_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("track_id") or ""), str(row.get("candidate_id") or ""))


def candidate_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("candidate_id") or ""),
        str(row.get("vocal_path") or ""),
        str(row.get("instrumental_path") or row.get("minus_path") or ""),
    )


def full_candidate_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    candidate_id, vocal_path, instrumental_path = candidate_identity(row)
    return (str(row.get("track_id") or ""), candidate_id, vocal_path, instrumental_path)


def has_path_identity(row: dict[str, Any]) -> bool:
    return bool(row.get("vocal_path") and (row.get("instrumental_path") or row.get("minus_path")))


def optimizer_base_candidate(row: dict[str, Any]) -> str:
    selected = str(row.get("selected_candidate_id") or "")
    if selected:
        return selected
    return str(row.get("input_candidate_id") or "")


def rejected_reason_by_candidate(row: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, list[str]] = {}
    for action in row.get("actions_rejected", []):
        if isinstance(action, dict) and action.get("candidate_id"):
            candidate_id = str(action["candidate_id"])
            reasons.setdefault(candidate_id, [])
            reason = str(action.get("reason") or "rejected_globally")
            if reason not in reasons[candidate_id]:
                reasons[candidate_id].append(reason)
    return {candidate_id: ";".join(values) for candidate_id, values in reasons.items()}


def window_key(window: dict[str, Any]) -> tuple[str, float, float]:
    return (
        str(window.get("kind") or ""),
        round(metric(window, "start_sec"), 6),
        round(metric(window, "duration_sec"), 6),
    )


def window_has_severe_blocker(window: dict[str, Any]) -> bool:
    reasons = set(as_list(window.get("reasons")))
    if reasons.intersection(SEVERE_BLOCKERS):
        return True
    return (
        metric(window, "vocal_leakage_in_minus_score") > 0.60
        or metric(window, "instrumental_leakage_in_vocal_score") > 0.72
        or metric(window, "silence_hallucination_score") > 0.75
        or metric(window, "robotic_artifact_score") > 0.82
        or metric(window, "thin_vocal_score") > 0.78
        or metric(window, "transient_damage_score") > 0.82
        or metric(window, "phase_consistency_score", 1.0) < 0.30
        or metric(window, "mixture_consistency_score", 1.0) < 0.55
    )


def blend_action(base_window: dict[str, Any], alt_window: dict[str, Any], *, min_segment_delta: float) -> tuple[str, str, str] | None:
    vocal_delta = metric(alt_window, "vocal_score") - metric(base_window, "vocal_score")
    minus_delta = metric(alt_window, "minus_score") - metric(base_window, "minus_score")
    vocal_improves = vocal_delta > min_segment_delta
    minus_improves = minus_delta > min_segment_delta
    vocal_regresses = vocal_delta < -min_segment_delta
    minus_regresses = minus_delta < -min_segment_delta
    pair_decision = str(alt_window.get("pair_decision") or "pair_compatible")
    pair_score = metric(alt_window, "pair_compatibility_score", 1.0)
    if vocal_improves and minus_improves:
        if pair_decision == "pair_compatible" and pair_score >= 0.72:
            return ("both", "use_alt_both_window", "vocal_and_minus_window_scores_improve")
        return None
    if vocal_improves and minus_regresses:
        return ("vocal", "use_alt_vocal_window", "alternate_improves_vocal_but_damages_minus")
    if minus_improves and vocal_regresses:
        return ("minus", "use_alt_minus_window", "alternate_improves_minus_but_damages_vocal")
    if vocal_improves and not minus_regresses:
        return ("vocal", "use_alt_vocal_window", "alternate_improves_vocal_window")
    if minus_improves and not vocal_regresses:
        return ("minus", "use_alt_minus_window", "alternate_improves_minus_window")
    return None


def source_stem_improvement_delta(base_window: dict[str, Any], alt_window: dict[str, Any], source_stem: str) -> float:
    vocal_delta = metric(alt_window, "vocal_score") - metric(base_window, "vocal_score")
    minus_delta = metric(alt_window, "minus_score") - metric(base_window, "minus_score")
    if source_stem == "vocal":
        return round(vocal_delta, 6)
    if source_stem == "minus":
        return round(minus_delta, 6)
    return round(min(vocal_delta, minus_delta), 6)


def plan_for_track(
    track_id: str,
    judge_rows: list[dict[str, Any]],
    optimizer_row: dict[str, Any],
    *,
    min_segment_delta: float,
) -> tuple[list[dict[str, Any]], int]:
    base_candidate_id = optimizer_base_candidate(optimizer_row)
    if not base_candidate_id:
        return [], 0
    sorted_judge_rows = sorted(judge_rows, key=lambda row: candidate_identity(row))
    selected_vocal_path = str(optimizer_row.get("selected_vocal_path") or "")
    selected_instrumental_path = str(optimizer_row.get("selected_minus_path") or "")
    exact_base_rows = [
        row
        for row in sorted_judge_rows
        if str(row.get("candidate_id") or "") == base_candidate_id
        and (not selected_vocal_path or str(row.get("vocal_path") or "") == selected_vocal_path)
        and (not selected_instrumental_path or str(row.get("instrumental_path") or row.get("minus_path") or "") == selected_instrumental_path)
    ]
    base_row = exact_base_rows[0] if exact_base_rows else next((row for row in sorted_judge_rows if str(row.get("candidate_id") or "") == base_candidate_id), None)
    if base_row is None:
        return [], 0
    base_windows = {window_key(window): window for window in dict_list(base_row.get("window_metrics"))}
    rejected = rejected_reason_by_candidate(optimizer_row)
    rows: list[dict[str, Any]] = []
    blocked_segments = 0
    base_identity = candidate_identity(base_row)
    for alt_row in sorted_judge_rows:
        alt_candidate_id = str(alt_row.get("candidate_id") or "")
        if not alt_candidate_id or candidate_identity(alt_row) == base_identity:
            continue
        for alt_window in sorted(
            dict_list(alt_row.get("window_metrics")),
            key=window_key,
        ):
            key = window_key(alt_window)
            base_window = base_windows.get(key)
            if base_window is None:
                continue
            if window_has_severe_blocker(alt_window):
                blocked_segments += 1
                continue
            action = blend_action(base_window, alt_window, min_segment_delta=min_segment_delta)
            if action is None:
                continue
            source_stem, segment_action, reason = action
            improvement_delta = source_stem_improvement_delta(base_window, alt_window, source_stem)
            rows.append(
                {
                    "track_id": track_id,
                    "base_candidate_id": base_candidate_id,
                    "alternate_candidate_id": alt_candidate_id,
                    "base_vocal_path": str(base_row.get("vocal_path") or ""),
                    "base_instrumental_path": str(base_row.get("instrumental_path") or base_row.get("minus_path") or ""),
                    "alternate_vocal_path": str(alt_row.get("vocal_path") or ""),
                    "alternate_instrumental_path": str(alt_row.get("instrumental_path") or alt_row.get("minus_path") or ""),
                    "window_kind": key[0],
                    "window_start_sec": key[1],
                    "window_duration_sec": key[2],
                    "source_stem": source_stem,
                    "reason_for_blend": reason,
                    "base_window_fitness": round(metric(base_window, "window_fitness"), 6),
                    "alternate_window_fitness": round(metric(alt_window, "window_fitness"), 6),
                    "improvement_delta": improvement_delta,
                    "blocker_check_passed": True,
                    "rejected_if_applied_globally_reason": rejected.get(alt_candidate_id, "not_selected_globally"),
                    "segment_blend_action": segment_action,
                    "can_execute_blend_now": False,
                    "requires_crossfade": True,
                    "forbidden_actions": stable_unique(
                        BASE_FORBIDDEN_ACTIONS
                        + as_list(base_row.get("forbidden_actions"))
                        + as_list(alt_row.get("forbidden_actions"))
                    ),
                }
            )
    rows.sort(key=lambda row: (row["track_id"], row["window_start_sec"], row["alternate_candidate_id"], row["segment_blend_action"]))
    return rows, blocked_segments


def build_summary(rows: list[dict[str, Any]], *, tracks_analyzed: int, blocked_segments: int) -> dict[str, Any]:
    actions = Counter(str(row.get("segment_blend_action") or "") for row in rows)
    reasons = Counter(str(row.get("reason_for_blend") or "") for row in rows)
    tracks_with_plan = len({str(row.get("track_id") or "") for row in rows})
    return {
        "total_tracks_analyzed": tracks_analyzed,
        "tracks_with_segment_blend_plan": tracks_with_plan,
        "total_segment_blend_windows": len(rows),
        "use_alt_vocal_window": actions.get("use_alt_vocal_window", 0),
        "use_alt_minus_window": actions.get("use_alt_minus_window", 0),
        "use_alt_both_window": actions.get("use_alt_both_window", 0),
        "average_segment_improvement_delta": round(sum(metric(row, "improvement_delta") for row in rows) / len(rows), 6) if rows else 0.0,
        "blocked_segment_candidates": blocked_segments,
        "top_reasons_for_blend": dict(reasons.most_common(10)),
        "safety": dict(SAFETY_FLAGS),
    }


def build_segment_blend_plan(
    candidate_pool_rows: list[dict[str, Any]],
    windowed_judge_rows: list[dict[str, Any]],
    optimizer_result_rows: list[dict[str, Any]],
    *,
    min_segment_delta: float = 0.03,
) -> dict[str, Any]:
    complete_full_candidates = {
        full_candidate_key(row)
        for row in candidate_pool_rows
        if row.get("is_complete_pair") is not False and row.get("track_id") and row.get("candidate_id") and has_path_identity(row)
    }
    complete_simple_candidates = {
        candidate_key(row)
        for row in candidate_pool_rows
        if row.get("is_complete_pair") is not False and row.get("track_id") and row.get("candidate_id") and not has_path_identity(row)
    }
    judge_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windowed_judge_rows:
        if complete_full_candidates and full_candidate_key(row) not in complete_full_candidates and candidate_key(row) not in complete_simple_candidates:
            continue
        if not complete_full_candidates and complete_simple_candidates and candidate_key(row) not in complete_simple_candidates:
            continue
        if not row.get("window_metrics"):
            raise ValueError(f"Windowed judge row is missing window_metrics: {candidate_key(row)}")
        judge_by_track[str(row.get("track_id") or "")].append(row)
    optimizer_by_track = {str(row.get("track_id") or ""): row for row in optimizer_result_rows if row.get("track_id")}
    all_rows: list[dict[str, Any]] = []
    blocked_segments = 0
    for track_id in sorted(set(judge_by_track).intersection(optimizer_by_track)):
        track_rows, blocked = plan_for_track(
            track_id,
            judge_by_track[track_id],
            optimizer_by_track[track_id],
            min_segment_delta=min_segment_delta,
        )
        all_rows.extend(track_rows)
        blocked_segments += blocked
    all_rows.sort(key=lambda row: (row["track_id"], row["window_start_sec"], row["base_candidate_id"], row["alternate_candidate_id"], row["segment_blend_action"]))
    return {
        "rows": all_rows,
        "summary": build_summary(all_rows, tracks_analyzed=len(set(judge_by_track).intersection(optimizer_by_track)), blocked_segments=blocked_segments),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Segment Blend Plan Summary",
        "",
        f"- total_tracks_analyzed: {summary['total_tracks_analyzed']}",
        f"- tracks_with_segment_blend_plan: {summary['tracks_with_segment_blend_plan']}",
        f"- total_segment_blend_windows: {summary['total_segment_blend_windows']}",
        f"- use_alt_vocal_window: {summary['use_alt_vocal_window']}",
        f"- use_alt_minus_window: {summary['use_alt_minus_window']}",
        f"- use_alt_both_window: {summary['use_alt_both_window']}",
        f"- average_segment_improvement_delta: {summary['average_segment_improvement_delta']}",
        f"- blocked_segment_candidates: {summary['blocked_segment_candidates']}",
        "",
        "## Top Reasons For Blend",
        "",
    ]
    for key, value in summary["top_reasons_for_blend"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], *, output_jsonl: Path, summary_md: Path) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_summary_md(summary_md, result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan metadata-only no-reference segment blends from windowed judge evidence.")
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--optimizer-result-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--min-segment-delta", type=float, default=0.03)
    args = parser.parse_args(argv)
    result = build_segment_blend_plan(
        list(iter_jsonl(args.candidate_pool_jsonl)),
        list(iter_jsonl(args.windowed_judge_jsonl)),
        list(iter_jsonl(args.optimizer_result_jsonl)),
        min_segment_delta=float(args.min_segment_delta),
    )
    write_outputs(result, output_jsonl=args.output_jsonl, summary_md=args.summary_md)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
