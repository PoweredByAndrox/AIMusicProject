from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.run_no_reference_optimizer import as_list, stable_unique  # noqa: E402


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
    "do_not_render_audio",
    "do_not_publish_blended_audio",
    "do_not_mark_release_ready",
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
    "audio_rendered": False,
    "audio_exported": False,
    "audio_copied": False,
}

DURATION_TOLERANCE_SEC = 0.25


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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_source_stem(value: Any) -> str:
    stem = str(value or "").strip().lower()
    if stem in {"vocal", "vocals"}:
        return "vocal"
    if stem in {"minus", "instrumental"}:
        return "minus"
    if stem == "both":
        return "both"
    return "unknown"


def target_stems(source_stem: str) -> set[str]:
    if source_stem == "both":
        return {"vocal", "minus"}
    if source_stem in {"vocal", "minus"}:
        return {source_stem}
    return set()


def path_same(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except (OSError, RuntimeError):
        return left == right


def base_minus_path(row: dict[str, Any]) -> str:
    return str(row.get("base_minus_path") or row.get("base_instrumental_path") or "")


def alt_minus_path(row: dict[str, Any]) -> str:
    return str(row.get("alt_minus_path") or row.get("alternate_instrumental_path") or "")


def selected_source_path(row: dict[str, Any], source_stem: str) -> str | dict[str, str]:
    alt_vocal = str(row.get("alternate_vocal_path") or row.get("alt_vocal_path") or "")
    alt_minus = alt_minus_path(row)
    if source_stem == "vocal":
        return alt_vocal
    if source_stem == "minus":
        return alt_minus
    if source_stem == "both":
        return {"vocal": alt_vocal, "minus": alt_minus}
    return ""


def paths_for_source_stem(row: dict[str, Any], source_stem: str) -> list[str]:
    base_vocal = str(row.get("base_vocal_path") or "")
    alt_vocal = str(row.get("alternate_vocal_path") or row.get("alt_vocal_path") or "")
    base_minus = base_minus_path(row)
    alternate_minus = alt_minus_path(row)
    if source_stem == "vocal":
        return [base_vocal, alt_vocal]
    if source_stem == "minus":
        return [base_minus, alternate_minus]
    if source_stem == "both":
        return [base_vocal, base_minus, alt_vocal, alternate_minus]
    return []


def active_source_path_distinct(row: dict[str, Any], source_stem: str) -> bool:
    base_vocal = str(row.get("base_vocal_path") or "")
    alt_vocal = str(row.get("alternate_vocal_path") or row.get("alt_vocal_path") or "")
    base_minus = base_minus_path(row)
    alternate_minus = alt_minus_path(row)
    distinct_by_stem = {
        "vocal": bool(base_vocal and alt_vocal and not path_same(base_vocal, alt_vocal)),
        "minus": bool(base_minus and alternate_minus and not path_same(base_minus, alternate_minus)),
    }
    stems = target_stems(source_stem)
    return any(distinct_by_stem[stem] for stem in stems)


def audio_header(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {"path": path_text, "exists": False, "error": "missing_path"}
    path = Path(path_text)
    if not path.exists():
        return {"path": path_text, "exists": False, "error": "path_not_found"}
    try:
        info = sf.info(str(path))
    except Exception as exc:  # pragma: no cover - exercised only for corrupt files
        return {"path": path_text, "exists": True, "error": f"header_read_failed:{type(exc).__name__}"}
    return {
        "path": path_text,
        "exists": True,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.duration), 6),
        "error": "",
    }


def compatibility(headers: list[dict[str, Any]], *, allow_sample_rate_resample: bool = False) -> dict[str, bool]:
    if not headers or any(not header.get("exists") or header.get("error") for header in headers):
        return {
            "sample_rate_compatible": False,
            "sample_rate_resample_allowed": bool(allow_sample_rate_resample),
            "sample_rate_requires_resample": False,
            "channels_compatible": False,
            "duration_compatible": False,
        }
    sample_rates = {int(header.get("sample_rate") or 0) for header in headers}
    channels = {int(header.get("channels") or 0) for header in headers}
    durations = [float(header.get("duration_sec") or 0.0) for header in headers]
    duration_span = max(durations) - min(durations) if durations else 0.0
    sample_rate_compatible = len(sample_rates) == 1
    return {
        "sample_rate_compatible": sample_rate_compatible,
        "sample_rate_resample_allowed": bool(allow_sample_rate_resample),
        "sample_rate_requires_resample": bool(allow_sample_rate_resample and not sample_rate_compatible),
        "channels_compatible": len(channels) == 1,
        "duration_compatible": duration_span <= DURATION_TOLERANCE_SEC,
    }


def boundary_is_safe(
    *,
    start_sec: float,
    duration_sec: float,
    headers: list[dict[str, Any]],
    crossfade_ms: int,
    pre_roll_ms: int,
    post_roll_ms: int,
) -> bool:
    if start_sec < 0 or duration_sec <= 0:
        return False
    if crossfade_ms < 0 or pre_roll_ms < 0 or post_roll_ms < 0:
        return False
    crossfade_sec = crossfade_ms / 1000.0
    if crossfade_sec >= duration_sec:
        return False
    valid_durations = [
        float(header.get("duration_sec") or 0.0)
        for header in headers
        if header.get("exists") and not header.get("error")
    ]
    if not valid_durations:
        return False
    expanded_start = start_sec - (pre_roll_ms / 1000.0)
    expanded_end = start_sec + duration_sec + (post_roll_ms / 1000.0)
    return expanded_start >= 0 and expanded_end <= min(valid_durations) + 0.001


def expanded_interval(row: dict[str, Any], pre_roll_ms: int, post_roll_ms: int) -> tuple[float, float]:
    start_sec = as_float(row.get("window_start_sec"))
    duration_sec = as_float(row.get("window_duration_sec"))
    return (start_sec - pre_roll_ms / 1000.0, start_sec + duration_sec + post_roll_ms / 1000.0)


def overlap_indexes(rows: list[dict[str, Any]], *, pre_roll_ms: int, post_roll_ms: int) -> set[int]:
    grouped: dict[str, list[tuple[int, set[str], tuple[float, float]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        stems = target_stems(str(row["source_stem"]))
        grouped[str(row["track_id"])].append((index, stems, expanded_interval(row, pre_roll_ms, post_roll_ms)))
    overlapping: set[int] = set()
    for group in grouped.values():
        for left_pos, (left_index, left_stems, left_interval) in enumerate(group):
            for right_index, right_stems, right_interval in group[left_pos + 1 :]:
                if not left_stems.intersection(right_stems):
                    continue
                if left_interval[0] < right_interval[1] and right_interval[0] < left_interval[1]:
                    overlapping.add(left_index)
                    overlapping.add(right_index)
    return overlapping


def rows_overlap(left: dict[str, Any], right: dict[str, Any], *, pre_roll_ms: int, post_roll_ms: int) -> bool:
    if str(left.get("track_id") or "") != str(right.get("track_id") or ""):
        return False
    left_stems = target_stems(str(left.get("source_stem") or ""))
    right_stems = target_stems(str(right.get("source_stem") or ""))
    if not left_stems.intersection(right_stems):
        return False
    left_interval = expanded_interval(left, pre_roll_ms, post_roll_ms)
    right_interval = expanded_interval(right, pre_roll_ms, post_roll_ms)
    return left_interval[0] < right_interval[1] and right_interval[0] < left_interval[1]


def select_nonoverlap_indexes(
    rows: list[dict[str, Any]],
    *,
    pre_roll_ms: int,
    post_roll_ms: int,
    eligible_indexes: set[int] | None = None,
) -> set[int]:
    eligible = set(range(len(rows))) if eligible_indexes is None else set(eligible_indexes)
    selected: list[int] = []
    for index in sorted(
        eligible,
        key=lambda item: (
            -as_float(rows[item].get("improvement_delta")),
            str(rows[item].get("track_id") or ""),
            as_float(rows[item].get("window_start_sec")),
            as_float(rows[item].get("window_duration_sec")),
            str(rows[item].get("source_stem") or ""),
            str(rows[item].get("alternate_candidate_id") or ""),
        ),
    ):
        if any(rows_overlap(rows[index], rows[other], pre_roll_ms=pre_roll_ms, post_roll_ms=post_roll_ms) for other in selected):
            continue
        selected.append(index)
    return set(selected)


def render_prerequisites_safe(row: dict[str, Any]) -> bool:
    if row["missing_source_paths"]:
        return False
    if not row["active_source_path_distinct"]:
        return False
    if not row["channels_compatible"] or not row["duration_compatible"]:
        return False
    if not row["sample_rate_compatible"] and not row["sample_rate_resample_allowed"]:
        return False
    return bool(row["boundary_safe"])


def render_status(row: dict[str, Any]) -> str:
    if row["missing_source_paths"]:
        return "blocked_missing_source_path"
    if not row["active_source_path_distinct"]:
        return "blocked_noop_self_blend"
    if not row["channels_compatible"] or not row["duration_compatible"]:
        return "blocked_incompatible_audio_shape"
    if not row["sample_rate_compatible"] and not row["sample_rate_resample_allowed"]:
        return "blocked_incompatible_audio_shape"
    if row["overlaps_with_other_planned_window"]:
        return "blocked_overlap_conflict"
    if not row["boundary_safe"]:
        return "blocked_unsafe_boundary"
    return "render_ready"


def make_dry_run_row(
    row: dict[str, Any],
    *,
    crossfade_ms: int,
    pre_roll_ms: int,
    post_roll_ms: int,
    overlaps: bool,
    overlap_conflict_resolution: str = "not_needed",
    allow_sample_rate_resample: bool = False,
) -> dict[str, Any]:
    source_stem = normalize_source_stem(row.get("source_stem"))
    paths = paths_for_source_stem(row, source_stem)
    headers = [audio_header(path) for path in paths]
    compat = compatibility(headers, allow_sample_rate_resample=allow_sample_rate_resample)
    start_sec = as_float(row.get("window_start_sec"))
    duration_sec = as_float(row.get("window_duration_sec"))
    missing_paths = [str(header["path"]) for header in headers if not header.get("exists")]
    header_errors = {
        str(header["path"]): str(header.get("error"))
        for header in headers
        if header.get("exists") and header.get("error")
    }
    dry_row = {
        "track_id": str(row.get("track_id") or ""),
        "base_candidate_id": str(row.get("base_candidate_id") or ""),
        "alternate_candidate_id": str(row.get("alternate_candidate_id") or ""),
        "source_stem": source_stem,
        "window_start_sec": round(start_sec, 6),
        "window_duration_sec": round(duration_sec, 6),
        "crossfade_ms": int(crossfade_ms),
        "pre_roll_ms": int(pre_roll_ms),
        "post_roll_ms": int(post_roll_ms),
        "base_vocal_path": str(row.get("base_vocal_path") or ""),
        "base_minus_path": base_minus_path(row),
        "alt_vocal_path": str(row.get("alternate_vocal_path") or row.get("alt_vocal_path") or ""),
        "alt_minus_path": alt_minus_path(row),
        "selected_source_path_for_window": selected_source_path(row, source_stem),
        "selected_target_stem": source_stem,
        "active_source_path_distinct": active_source_path_distinct(row, source_stem),
        **compat,
        "boundary_safe": boundary_is_safe(
            start_sec=start_sec,
            duration_sec=duration_sec,
            headers=headers,
            crossfade_ms=int(crossfade_ms),
            pre_roll_ms=int(pre_roll_ms),
            post_roll_ms=int(post_roll_ms),
        ),
        "overlaps_with_other_planned_window": bool(overlaps),
        "overlap_conflict_resolution": overlap_conflict_resolution,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "missing_source_paths": missing_paths,
        "audio_header_errors": header_errors,
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + as_list(row.get("forbidden_actions"))),
    }
    status = render_status(dry_row)
    dry_row["render_status"] = status
    dry_row["can_render_later"] = status == "render_ready"
    return dry_row


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("render_status") or "") for row in rows)
    ready_rows = [row for row in rows if row.get("render_status") == "render_ready"]
    ready_by_stem = Counter(str(row.get("source_stem") or "") for row in ready_rows)
    return {
        "total_planned_windows": len(rows),
        "render_ready": statuses.get("render_ready", 0),
        "blocked_incompatible_audio_shape": statuses.get("blocked_incompatible_audio_shape", 0),
        "blocked_overlap_conflict": statuses.get("blocked_overlap_conflict", 0),
        "blocked_missing_source_path": statuses.get("blocked_missing_source_path", 0),
        "blocked_noop_self_blend": statuses.get("blocked_noop_self_blend", 0),
        "blocked_unsafe_boundary": statuses.get("blocked_unsafe_boundary", 0),
        "tracks_render_ready": len({str(row.get("track_id") or "") for row in ready_rows}),
        "vocal_windows_ready": ready_by_stem.get("vocal", 0),
        "minus_windows_ready": ready_by_stem.get("minus", 0),
        "both_windows_ready": ready_by_stem.get("both", 0),
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "safety": dict(SAFETY_FLAGS),
    }


def build_dry_run_contract(
    segment_blend_rows: list[dict[str, Any]],
    *,
    windowed_judge_rows: list[dict[str, Any]],
    candidate_pool_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    crossfade_ms: int = 80,
    pre_roll_ms: int = 120,
    post_roll_ms: int = 120,
    allow_sample_rate_resample: bool = False,
    resolve_overlap_conflicts: bool = False,
) -> dict[str, Any]:
    del windowed_judge_rows, candidate_pool_rows, optimizer_rows
    normalized = sorted(
        (
            {
                **row,
                "source_stem": normalize_source_stem(row.get("source_stem")),
                "track_id": str(row.get("track_id") or ""),
                "window_start_sec": round(as_float(row.get("window_start_sec")), 6),
                "window_duration_sec": round(as_float(row.get("window_duration_sec")), 6),
            }
            for row in segment_blend_rows
        ),
        key=lambda item: (
            str(item.get("track_id") or ""),
            as_float(item.get("window_start_sec")),
            as_float(item.get("window_duration_sec")),
            str(item.get("source_stem") or ""),
            str(item.get("base_candidate_id") or ""),
            str(item.get("alternate_candidate_id") or ""),
        ),
    )
    overlapping = overlap_indexes(normalized, pre_roll_ms=int(pre_roll_ms), post_roll_ms=int(post_roll_ms))
    provisional_rows = [
        make_dry_run_row(
            row,
            crossfade_ms=int(crossfade_ms),
            pre_roll_ms=int(pre_roll_ms),
            post_roll_ms=int(post_roll_ms),
            overlaps=False,
            allow_sample_rate_resample=bool(allow_sample_rate_resample),
        )
        for row in normalized
    ]
    eligible_for_overlap_selection = {
        index for index, row in enumerate(provisional_rows) if render_prerequisites_safe(row)
    }
    selected_nonoverlap = (
        select_nonoverlap_indexes(
            normalized,
            pre_roll_ms=int(pre_roll_ms),
            post_roll_ms=int(post_roll_ms),
            eligible_indexes=eligible_for_overlap_selection,
        )
        if resolve_overlap_conflicts
        else set()
    )
    rows = [
        make_dry_run_row(
            row,
            crossfade_ms=int(crossfade_ms),
            pre_roll_ms=int(pre_roll_ms),
            post_roll_ms=int(post_roll_ms),
            overlaps=index in overlapping and index not in selected_nonoverlap,
            overlap_conflict_resolution=(
                "selected_highest_improvement_nonoverlap"
                if index in overlapping and index in selected_nonoverlap
                else ("blocked_lower_score_overlap" if index in overlapping else "not_needed")
            ),
            allow_sample_rate_resample=bool(allow_sample_rate_resample),
        )
        for index, row in enumerate(normalized)
    ]
    rows.sort(
        key=lambda item: (
            item["track_id"],
            item["window_start_sec"],
            item["window_duration_sec"],
            item["source_stem"],
            item["base_candidate_id"],
            item["alternate_candidate_id"],
        )
    )
    return {"rows": rows, "render_contract_rows": rows, "summary": build_summary(rows)}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Segment Blend Dry-Run Summary",
        "",
        f"- total_planned_windows: {summary['total_planned_windows']}",
        f"- render_ready: {summary['render_ready']}",
        f"- blocked_incompatible_audio_shape: {summary['blocked_incompatible_audio_shape']}",
        f"- blocked_overlap_conflict: {summary['blocked_overlap_conflict']}",
        f"- blocked_missing_source_path: {summary['blocked_missing_source_path']}",
        f"- blocked_noop_self_blend: {summary['blocked_noop_self_blend']}",
        f"- blocked_unsafe_boundary: {summary['blocked_unsafe_boundary']}",
        f"- tracks_render_ready: {summary['tracks_render_ready']}",
        f"- vocal_windows_ready: {summary['vocal_windows_ready']}",
        f"- minus_windows_ready: {summary['minus_windows_ready']}",
        f"- both_windows_ready: {summary['both_windows_ready']}",
        f"- can_publish_as_result_count: {summary['can_publish_as_result_count']}",
        f"- can_mark_release_ready_count: {summary['can_mark_release_ready_count']}",
        "",
        "## Safety",
        "",
    ]
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    result: dict[str, Any],
    *,
    output_jsonl: Path,
    render_contract_jsonl: Path,
    summary_md: Path,
) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_jsonl(render_contract_jsonl, result.get("render_contract_rows", result["rows"]))
    write_summary_md(summary_md, result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a metadata-only dry-run contract for no-reference segment blends.")
    parser.add_argument("--segment-blend-plan-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--optimizer-result-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--render-contract-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    parser.add_argument("--pre-roll-ms", type=int, default=120)
    parser.add_argument("--post-roll-ms", type=int, default=120)
    parser.add_argument("--allow-sample-rate-resample", action="store_true")
    parser.add_argument("--resolve-overlap-conflicts", action="store_true")
    args = parser.parse_args(argv)
    result = build_dry_run_contract(
        list(iter_jsonl(args.segment_blend_plan_jsonl)),
        windowed_judge_rows=list(iter_jsonl(args.windowed_judge_jsonl)),
        candidate_pool_rows=list(iter_jsonl(args.candidate_pool_jsonl)),
        optimizer_rows=list(iter_jsonl(args.optimizer_result_jsonl)),
        crossfade_ms=args.crossfade_ms,
        pre_roll_ms=args.pre_roll_ms,
        post_roll_ms=args.post_roll_ms,
        allow_sample_rate_resample=args.allow_sample_rate_resample,
        resolve_overlap_conflicts=args.resolve_overlap_conflicts,
    )
    write_outputs(
        result,
        output_jsonl=args.output_jsonl,
        render_contract_jsonl=args.render_contract_jsonl,
        summary_md=args.summary_md,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
