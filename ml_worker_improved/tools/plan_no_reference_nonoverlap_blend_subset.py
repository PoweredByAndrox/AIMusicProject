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

from tools.no_reference_score_matrix import detect_blockers  # noqa: E402

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_render_audio",
    "do_not_export_audio",
    "do_not_copy_audio",
    "do_not_publish_as_final_result",
    "do_not_mark_release_ready",
]

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_rendered": False,
    "audio_exported": False,
    "audio_copied": False,
}

RENDERABLE_OR_OVERLAP_ONLY_STATUSES = {"render_ready", "blocked_overlap_conflict"}
EXPAND_PREVIEW_DECISION = "preview_improved_window"
EXPAND_PREVIEW_NEXT_STEP = "expand_preview_policy"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_source_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"vocal", "vocals"}:
        return "vocal"
    if text in {"minus", "instrumental"}:
        return "minus"
    if text == "both":
        return "both"
    return "vocal"


def normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def base_vocal_path(row: dict[str, Any]) -> str:
    return normalized_path(row.get("base_vocal_path"))


def alt_vocal_path(row: dict[str, Any]) -> str:
    return normalized_path(row.get("alternate_vocal_path") or row.get("alt_vocal_path"))


def base_minus_path(row: dict[str, Any]) -> str:
    return normalized_path(row.get("base_minus_path") or row.get("base_instrumental_path"))


def alt_minus_path(row: dict[str, Any]) -> str:
    return normalized_path(row.get("alternate_minus_path") or row.get("alt_minus_path") or row.get("alternate_instrumental_path"))


def path_signature(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        base_vocal_path(row),
        alt_vocal_path(row),
        base_minus_path(row),
        alt_minus_path(row),
    )


def target_stems(source_stem: str) -> set[str]:
    if source_stem == "both":
        return {"vocal", "minus"}
    return {source_stem}


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, float, float, str, str, str, str]:
    return (
        str(row.get("track_id") or ""),
        str(row.get("base_candidate_id") or ""),
        str(row.get("alternate_candidate_id") or ""),
        normalize_source_stem(row.get("source_stem")),
        round(as_float(row.get("window_start_sec")), 6),
        round(as_float(row.get("window_duration_sec")), 6),
        *path_signature(row),
    )


def row_identity_key(row: dict[str, Any]) -> tuple[str, str, str, str, float, float]:
    return (
        str(row.get("track_id") or ""),
        str(row.get("base_candidate_id") or ""),
        str(row.get("alternate_candidate_id") or ""),
        normalize_source_stem(row.get("source_stem")),
        round(as_float(row.get("window_start_sec")), 6),
        round(as_float(row.get("window_duration_sec")), 6),
    )


def output_sort_key(row: dict[str, Any]) -> tuple[str, str, float, float, str, str, str, str, str, str, str]:
    return (
        str(row.get("track_id") or ""),
        str(row.get("overlap_group_id") or ""),
        as_float(row.get("window_start_sec")),
        as_float(row.get("window_duration_sec")),
        normalize_source_stem(row.get("source_stem")),
        str(row.get("base_candidate_id") or ""),
        str(row.get("alternate_candidate_id") or ""),
        *path_signature(row),
    )


def expected_improvement_delta(row: dict[str, Any]) -> float:
    if row.get("improvement_delta") is not None:
        return round(as_float(row.get("improvement_delta")), 6)
    return round(as_float(row.get("alternate_window_fitness")) - as_float(row.get("base_window_fitness")), 6)


def expanded_interval(row: dict[str, Any]) -> tuple[float, float]:
    start = as_float(row.get("window_start_sec"))
    duration = as_float(row.get("window_duration_sec"))
    pre_roll_ms = int(as_float(row.get("pre_roll_ms"), 120.0))
    post_roll_ms = int(as_float(row.get("post_roll_ms"), 120.0))
    return (start - pre_roll_ms / 1000.0, start + duration + post_roll_ms / 1000.0)


def intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def rows_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("track_id") or "") != str(right.get("track_id") or ""):
        return False
    if not target_stems(normalize_source_stem(left.get("source_stem"))).intersection(target_stems(normalize_source_stem(right.get("source_stem")))):
        return False
    return intervals_overlap(expanded_interval(left), expanded_interval(right))


def preview_row_allows_expansion(row: dict[str, Any]) -> bool:
    return (
        str(row.get("preview_decision") or "") == EXPAND_PREVIEW_DECISION
        and str(row.get("allowed_next_step") or "") == EXPAND_PREVIEW_NEXT_STEP
        and row.get("can_publish_as_result") is not True
        and row.get("can_mark_release_ready") is not True
    )


def preview_policy_allows_expansion(preview_rows: list[dict[str, Any]]) -> bool:
    return any(preview_row_allows_expansion(row) for row in preview_rows)


def preview_feedback_by_key(preview_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, float, float, str, str, str, str], list[dict[str, Any]]]:
    feedback: dict[tuple[str, str, str, str, float, float, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in preview_rows:
        feedback[row_key(row)].append(row)
    return feedback


def preview_feedback_by_identity_key(preview_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]]:
    feedback: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in preview_rows:
        feedback[row_identity_key(row)].append(row)
    return feedback


def pair_feedback_is_rejection(row: dict[str, Any]) -> bool:
    if str(row.get("score_mode") or "") != "both":
        return False
    if str(row.get("pair_decision") or "") == "pair_compatible" and not as_list(row.get("pair_blockers")):
        return False
    return True


def pair_feedback_by_identity_key(windowed_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]]:
    feedback: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in windowed_rows:
        if pair_feedback_is_rejection(row):
            feedback[row_identity_key(row)].append(row)
    return feedback


def matrix_feedback_by_candidate_id(windowed_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    feedback: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windowed_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            feedback[candidate_id].append(row)
    return feedback


def source_stem_matches_improved_metric(row: dict[str, Any]) -> bool:
    source_stem = normalize_source_stem(row.get("source_stem"))
    reason = str(row.get("reason_for_blend") or "")
    if source_stem == "both":
        return reason == "vocal_and_minus_window_scores_improve"
    if source_stem == "vocal":
        return "vocal" in reason and reason != "vocal_and_minus_window_scores_improve"
    if source_stem == "minus":
        return "minus" in reason and reason != "vocal_and_minus_window_scores_improve"
    return False


def target_path_identity_blockers(row: dict[str, Any]) -> list[str]:
    source_stem = normalize_source_stem(row.get("source_stem"))
    blockers: list[str] = []
    if source_stem in {"vocal", "both"} and base_vocal_path(row) == alt_vocal_path(row):
        blockers.append("target_vocal_path_identity_noop_risk")
    if source_stem in {"minus", "both"} and base_minus_path(row) == alt_minus_path(row):
        blockers.append("target_minus_path_identity_noop_risk")
    return blockers


def dry_run_by_key(dry_run_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, float, float], dict[str, Any]]:
    indexed: dict[tuple[str, str, str, str, float, float], dict[str, Any]] = {}
    for row in dry_run_rows:
        indexed.setdefault(row_key(row), row)
    return indexed


def normalize_plan_rows(plan_rows: list[dict[str, Any]], dry_run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dry_index = dry_run_by_key(dry_run_rows)
    normalized: list[dict[str, Any]] = []
    for row in plan_rows:
        dry_row = dry_index.get(row_key(row), {})
        merged = {
            **row,
            "track_id": str(row.get("track_id") or ""),
            "base_candidate_id": str(row.get("base_candidate_id") or ""),
            "alternate_candidate_id": str(row.get("alternate_candidate_id") or ""),
            "source_stem": normalize_source_stem(row.get("source_stem")),
            "window_start_sec": round(as_float(row.get("window_start_sec")), 6),
            "window_duration_sec": round(as_float(row.get("window_duration_sec")), 6),
            "expected_improvement_delta": expected_improvement_delta(row),
            "dry_run_contract_found": bool(dry_row),
            "render_status": str(dry_row.get("render_status") or ""),
            "crossfade_ms": int(as_float(dry_row.get("crossfade_ms") or row.get("crossfade_ms"), 80.0)),
            "pre_roll_ms": int(as_float(dry_row.get("pre_roll_ms"), 120.0)),
            "post_roll_ms": int(as_float(dry_row.get("post_roll_ms"), 120.0)),
            "forbidden_actions": stable_unique(
                BASE_FORBIDDEN_ACTIONS
                + as_list(row.get("forbidden_actions"))
                + as_list(dry_row.get("forbidden_actions"))
            ),
        }
        normalized.append(merged)
    return sorted(normalized, key=output_sort_key)


def safety_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if row.get("dry_run_contract_found") is not True:
        blockers.append("missing_dry_run_contract")
    render_status = str(row.get("render_status") or "")
    if render_status not in RENDERABLE_OR_OVERLAP_ONLY_STATUSES:
        blockers.append(f"render_status={render_status}")
    if row.get("blocker_check_passed") is False:
        blockers.append("blocker_check_passed=false")
    if row.get("can_publish_as_result") is True or row.get("can_mark_release_ready") is True:
        blockers.append("unexpected_publish_or_release_flag")
    if expected_improvement_delta(row) <= 0.0:
        blockers.append("non_positive_expected_improvement_delta")
    if not source_stem_matches_improved_metric(row):
        blockers.append("source_stem_does_not_match_improved_metric")
    blockers.extend(target_path_identity_blockers(row))
    return blockers


def preview_feedback_blockers(
    row: dict[str, Any],
    feedback_index: dict[tuple[str, str, str, str, float, float, str, str, str, str], list[dict[str, Any]]],
    identity_feedback_index: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]] | None = None,
) -> list[str]:
    blockers: list[str] = []
    exact_feedback = feedback_index.get(row_key(row), [])
    for feedback in exact_feedback:
        decision = str(feedback.get("preview_decision") or "missing_preview_decision")
        if preview_row_allows_expansion(feedback):
            continue
        blockers.append(f"preview_feedback={decision}")
    if exact_feedback:
        return stable_unique(blockers)
    if blockers:
        return stable_unique(blockers)
    for feedback in (identity_feedback_index or {}).get(row_identity_key(row), []):
        if row_key(feedback) == row_key(row) or preview_row_allows_expansion(feedback):
            continue
        decision = str(feedback.get("preview_decision") or "missing_preview_decision")
        blockers.append(f"preview_feedback_identity={decision}")
    return stable_unique(blockers)


def pair_feedback_blockers(
    row: dict[str, Any],
    feedback_index: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]],
) -> list[str]:
    blockers: list[str] = []
    for feedback in feedback_index.get(row_identity_key(row), []):
        decision = str(feedback.get("pair_decision") or feedback.get("decision") or "missing_pair_decision")
        blockers.append(f"pair_feedback={decision}")
    return stable_unique(blockers)


def matrix_guard_blockers(
    row: dict[str, Any],
    matrix_index: dict[str, list[dict[str, Any]]],
) -> list[str]:
    blockers: list[str] = []
    alternate_id = str(row.get("alternate_candidate_id") or "")
    for matrix_row in matrix_index.get(alternate_id, []):
        minus_blockers = stable_unique(as_list(matrix_row.get("minus_blockers")) + detect_blockers(matrix_row, "minus"))
        if "severe_vocal_bleed_in_minus" in minus_blockers:
            blockers.append("matrix_minus_guard=severe_vocal_bleed_in_minus")
        pair_decision = str(matrix_row.get("pair_decision") or "")
        if pair_decision and pair_decision != "pair_compatible":
            blockers.append(f"matrix_pair_guard={pair_decision}")
    return stable_unique(blockers)


def assign_overlap_groups(rows: list[dict[str, Any]]) -> dict[int, str]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            if rows_conflict(left, rows[right_index]):
                union(left_index, right_index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)

    group_ids: dict[int, str] = {}
    for number, indexes in enumerate(sorted(grouped.values(), key=lambda values: output_sort_key(rows[min(values)])), start=1):
        group_id = f"overlap_group_{number:04d}"
        for index in indexes:
            group_ids[index] = group_id
    return group_ids


def rank_by_group(rows: list[dict[str, Any]], group_ids: dict[int, str]) -> dict[int, int]:
    members_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group_id in group_ids.items():
        members_by_group[group_id].append(index)
    ranks: dict[int, int] = {}
    for indexes in members_by_group.values():
        for rank, index in enumerate(
            sorted(
                indexes,
                key=lambda item: (
                    -as_float(rows[item].get("expected_improvement_delta")),
                    output_sort_key(rows[item]),
                ),
            ),
            start=1,
        ):
            ranks[index] = rank
    return ranks


def select_nonoverlap(rows: list[dict[str, Any]], group_ids: dict[int, str]) -> set[int]:
    selected: list[int] = []
    for index in sorted(
        range(len(rows)),
        key=lambda item: (
            -as_float(rows[item].get("expected_improvement_delta")),
            output_sort_key(rows[item]),
        ),
    ):
        if any(rows_conflict(rows[index], rows[chosen]) for chosen in selected):
            continue
        selected.append(index)
    return set(selected)


def build_output_rows(
    plan_rows: list[dict[str, Any]],
    *,
    preview_policy_met: bool,
    preview_feedback_index: dict[tuple[str, str, str, str, float, float, str, str, str, str], list[dict[str, Any]]],
    identity_feedback_index: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]] | None = None,
    pair_feedback_index: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]] | None = None,
    matrix_feedback_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    group_ids = assign_overlap_groups(plan_rows)
    ranks = rank_by_group(plan_rows, group_ids)
    blocker_map = {index: safety_blockers(row) for index, row in enumerate(plan_rows)}
    preview_blocker_map = {
        index: preview_feedback_blockers(row, preview_feedback_index, identity_feedback_index)
        for index, row in enumerate(plan_rows)
    }
    pair_blocker_map = {
        index: pair_feedback_blockers(row, pair_feedback_index or {})
        for index, row in enumerate(plan_rows)
    }
    matrix_blocker_map = {
        index: matrix_guard_blockers(row, matrix_feedback_index or {})
        for index, row in enumerate(plan_rows)
    }
    eligible_indexes = [
        index
        for index, blockers in blocker_map.items()
        if preview_policy_met
        and not blockers
        and not preview_blocker_map[index]
        and not pair_blocker_map[index]
        and not matrix_blocker_map[index]
    ]
    selected_indexes = select_nonoverlap([plan_rows[index] for index in eligible_indexes], {}) if eligible_indexes else set()
    selected_original_indexes = {eligible_indexes[index] for index in selected_indexes}
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(plan_rows):
        blockers = blocker_map[index]
        if not preview_policy_met:
            status = "rejected_preview_policy_not_met"
            reason = "preview_policy_not_met"
        elif preview_blocker_map[index]:
            status = "rejected_preview_policy_not_met"
            reason = ";".join(preview_blocker_map[index])
        elif pair_blocker_map[index]:
            status = "rejected_preview_policy_not_met"
            reason = ";".join(pair_blocker_map[index])
        elif matrix_blocker_map[index]:
            status = "rejected_preview_policy_not_met"
            reason = ";".join(matrix_blocker_map[index])
        elif blockers:
            status = "rejected_safety_blocker"
            reason = ";".join(blockers)
        elif index in selected_original_indexes:
            status = "selected_nonoverlap"
            reason = "highest_safe_expected_improvement_in_overlap_group"
        else:
            status = "rejected_lower_score_overlap"
            reason = "overlaps_selected_higher_expected_improvement_window"
        render_status = {
            "selected_nonoverlap": "render_ready",
            "rejected_lower_score_overlap": "blocked_overlap_conflict",
            "rejected_safety_blocker": "blocked_safety_blocker",
            "rejected_preview_policy_not_met": "blocked_preview_policy",
        }[status]
        rows.append(
            {
                "track_id": str(row.get("track_id") or ""),
                "base_candidate_id": str(row.get("base_candidate_id") or ""),
                "alternate_candidate_id": str(row.get("alternate_candidate_id") or ""),
                "source_stem": normalize_source_stem(row.get("source_stem")),
                "window_start_sec": round(as_float(row.get("window_start_sec")), 6),
                "window_duration_sec": round(as_float(row.get("window_duration_sec")), 6),
                "expected_improvement_delta": round(as_float(row.get("expected_improvement_delta")), 6),
                "overlap_group_id": group_ids[index],
                "selected_rank_in_group": ranks[index],
                "selection_reason": reason,
                "conflict_resolution_status": status,
                "can_render_preview_later": status == "selected_nonoverlap",
                "render_status": render_status,
                "can_render_later": status == "selected_nonoverlap",
                "base_vocal_path": base_vocal_path(row),
                "base_minus_path": base_minus_path(row),
                "alt_vocal_path": alt_vocal_path(row),
                "alt_minus_path": alt_minus_path(row),
                "crossfade_ms": int(as_float(row.get("crossfade_ms"), 80.0)),
                "pre_roll_ms": int(as_float(row.get("pre_roll_ms"), 120.0)),
                "post_roll_ms": int(as_float(row.get("post_roll_ms"), 120.0)),
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "forbidden_actions": stable_unique(row.get("forbidden_actions")),
            }
        )
    return sorted(rows, key=output_sort_key)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("conflict_resolution_status") or "") for row in rows)
    selected = [row for row in rows if row.get("conflict_resolution_status") == "selected_nonoverlap"]
    selected_by_stem = Counter(str(row.get("source_stem") or "") for row in selected)
    expected_deltas = [as_float(row.get("expected_improvement_delta")) for row in selected]
    return {
        "total_planned_windows": len(rows),
        "overlap_groups": len({str(row.get("overlap_group_id") or "") for row in rows}),
        "selected_nonoverlap": statuses.get("selected_nonoverlap", 0),
        "rejected_lower_score_overlap": statuses.get("rejected_lower_score_overlap", 0),
        "rejected_safety_blocker": statuses.get("rejected_safety_blocker", 0),
        "rejected_preview_policy_not_met": statuses.get("rejected_preview_policy_not_met", 0),
        "rejected_preview_feedback": sum(
            1
            for row in rows
            if row.get("conflict_resolution_status") == "rejected_preview_policy_not_met"
            and str(row.get("selection_reason") or "").startswith("preview_feedback=")
        ),
        "rejected_pair_feedback": sum(
            1
            for row in rows
            if row.get("conflict_resolution_status") == "rejected_preview_policy_not_met"
            and str(row.get("selection_reason") or "").startswith("pair_feedback=")
        ),
        "rejected_matrix_guard": sum(
            1
            for row in rows
            if row.get("conflict_resolution_status") == "rejected_preview_policy_not_met"
            and str(row.get("selection_reason") or "").startswith("matrix_")
        ),
        "tracks_with_selected_windows": len({str(row.get("track_id") or "") for row in selected}),
        "vocal_windows_selected": selected_by_stem.get("vocal", 0),
        "minus_windows_selected": selected_by_stem.get("minus", 0),
        "both_windows_selected": selected_by_stem.get("both", 0),
        "average_expected_improvement_delta": round(sum(expected_deltas) / len(expected_deltas), 6) if expected_deltas else 0.0,
        "can_render_preview_later_count": sum(1 for row in rows if row.get("can_render_preview_later") is True),
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "safety": dict(SAFETY_FLAGS),
    }


def build_nonoverlap_subset(
    segment_blend_rows: list[dict[str, Any]],
    *,
    dry_run_rows: list[dict[str, Any]],
    preview_window_judge_rows: list[dict[str, Any]],
    windowed_judge_rows: list[dict[str, Any]],
    candidate_pool_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    del candidate_pool_rows
    plan_rows = normalize_plan_rows(segment_blend_rows, dry_run_rows)
    preview_policy_met = preview_policy_allows_expansion(preview_window_judge_rows)
    rows = build_output_rows(
        plan_rows,
        preview_policy_met=preview_policy_met,
        preview_feedback_index=preview_feedback_by_key(preview_window_judge_rows),
        identity_feedback_index=preview_feedback_by_identity_key(preview_window_judge_rows),
        pair_feedback_index=pair_feedback_by_identity_key(windowed_judge_rows),
        matrix_feedback_index=matrix_feedback_by_candidate_id(windowed_judge_rows),
    )
    return {"rows": rows, "summary": build_summary(rows)}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Non-Overlap Blend Subset Summary",
        "",
        f"- total_planned_windows: {summary['total_planned_windows']}",
        f"- overlap_groups: {summary['overlap_groups']}",
        f"- selected_nonoverlap: {summary['selected_nonoverlap']}",
        f"- rejected_lower_score_overlap: {summary['rejected_lower_score_overlap']}",
        f"- rejected_safety_blocker: {summary['rejected_safety_blocker']}",
        f"- rejected_preview_policy_not_met: {summary['rejected_preview_policy_not_met']}",
        f"- rejected_preview_feedback: {summary['rejected_preview_feedback']}",
        f"- rejected_pair_feedback: {summary['rejected_pair_feedback']}",
        f"- rejected_matrix_guard: {summary['rejected_matrix_guard']}",
        f"- tracks_with_selected_windows: {summary['tracks_with_selected_windows']}",
        f"- vocal_windows_selected: {summary['vocal_windows_selected']}",
        f"- minus_windows_selected: {summary['minus_windows_selected']}",
        f"- both_windows_selected: {summary['both_windows_selected']}",
        f"- average_expected_improvement_delta: {summary['average_expected_improvement_delta']}",
        f"- can_render_preview_later count: {summary['can_render_preview_later_count']}",
        f"- can_publish_as_result count: {summary['can_publish_as_result_count']}",
        f"- can_mark_release_ready count: {summary['can_mark_release_ready_count']}",
        "",
        "## Safety",
    ]
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], *, output_jsonl: Path, summary_md: Path) -> None:
    write_jsonl(output_jsonl, list(result["rows"]))
    write_summary_md(summary_md, dict(result["summary"]))


def collect_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(iter_jsonl(path))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a deterministic non-overlapping subset for no-reference segment blend previews.")
    parser.add_argument("--segment-blend-plan-jsonl", type=Path, required=True)
    parser.add_argument("--dry-run-jsonl", type=Path, required=True)
    parser.add_argument("--preview-window-judge-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_nonoverlap_subset(
        list(iter_jsonl(args.segment_blend_plan_jsonl)),
        dry_run_rows=list(iter_jsonl(args.dry_run_jsonl)),
        preview_window_judge_rows=collect_jsonl(args.preview_window_judge_jsonl),
        windowed_judge_rows=collect_jsonl(args.windowed_judge_jsonl),
        candidate_pool_rows=list(iter_jsonl(args.candidate_pool_jsonl)),
    )
    write_outputs(result, output_jsonl=args.output_jsonl, summary_md=args.summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
