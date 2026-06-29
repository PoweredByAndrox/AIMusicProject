from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ACTION_BLOCK_UNKNOWN = "block_unknown"
ACTION_QUARANTINE_ROUTE = "quarantine_route"
ACTION_PRIORITIZE_ROUTE = "prioritize_route"
ACTION_SWITCH_VOCAL_LANE = "switch_vocal_lane"
ACTION_SWITCH_MINUS_LANE = "switch_minus_lane"
ACTION_SWITCH_STEM_LANE = "switch_stem_lane"
ACTION_ENABLE_REPAIR_OPTION = "enable_repair_option"
ACTION_RETUNE_TEMPLATE_PARAM = "retune_template_param"

ACTIONS = (
    ACTION_BLOCK_UNKNOWN,
    ACTION_QUARANTINE_ROUTE,
    ACTION_PRIORITIZE_ROUTE,
    ACTION_SWITCH_VOCAL_LANE,
    ACTION_SWITCH_MINUS_LANE,
    ACTION_SWITCH_STEM_LANE,
    ACTION_ENABLE_REPAIR_OPTION,
    ACTION_RETUNE_TEMPLATE_PARAM,
)

LANE_VOCAL = "vocal_lane"
LANE_MINUS = "minus_lane"
LANE_STEM = "stem_lane"
LANE_BASS_SUPPORT = "bass_support_lane"
LANE_NON_VOCAL_SUPPORT = "non_vocal_support_lane"

LANES = (
    LANE_VOCAL,
    LANE_MINUS,
    LANE_STEM,
    LANE_BASS_SUPPORT,
    LANE_NON_VOCAL_SUPPORT,
)

ISSUE_BAD_RECONSTRUCTION = "bad_reconstruction"
ISSUE_STEREO_COLLAPSE = "stereo_collapse"
ISSUE_MISSING_DETAIL = "missing_detail"
ISSUE_VOCAL_LEAK = "vocal_leak"
ISSUE_INSTRUMENT_LEAK = "instrument_leak"
ISSUE_INFRA_FAILURE = "infra_failure"
ISSUE_UNKNOWN = "unknown"

ISSUES = (
    ISSUE_BAD_RECONSTRUCTION,
    ISSUE_STEREO_COLLAPSE,
    ISSUE_MISSING_DETAIL,
    ISSUE_VOCAL_LEAK,
    ISSUE_INSTRUMENT_LEAK,
    ISSUE_INFRA_FAILURE,
    ISSUE_UNKNOWN,
)

COUNTERFACTUAL_REQUIRED_ACTIONS = {
    ACTION_PRIORITIZE_ROUTE,
    ACTION_SWITCH_VOCAL_LANE,
    ACTION_SWITCH_MINUS_LANE,
    ACTION_SWITCH_STEM_LANE,
    ACTION_ENABLE_REPAIR_OPTION,
    ACTION_RETUNE_TEMPLATE_PARAM,
}

COUNTERFACTUAL_LABEL_SOURCES = {
    "counterfactual_replay",
    "locked_counterfactual_replay",
}

GENERATED_GROUND_TRUTH_LABEL_TYPES = {
    "generated_separator",
    "separator_output",
    "blocked_candidate_mining",
    "pseudo_teacher",
}


def normalize_policy_action(action: str | None) -> str:
    value = str(action or "").strip().lower()
    return value if value in ACTIONS else ACTION_BLOCK_UNKNOWN


def normalize_issue_type(issue_type: str | None) -> str:
    value = str(issue_type or "").strip().lower()
    return value if value in ISSUES else ISSUE_UNKNOWN


def policy_action_can_accept_delivery(_action: str | None) -> bool:
    return False


def route_policy_artifact_path_is_safe(path: str | Path) -> bool:
    parts = Path(str(path)).parts
    return all(len(part) <= 64 for part in parts if part not in {Path(str(path)).anchor, ""})


def safe_policy_row_id(index: int) -> str:
    return f"p{int(index):06d}"


def short_safe_segment(value: str, *, fallback: str = "item") -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    if not segment:
        segment = fallback
    return segment[:32]


def extract_lane_routes(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    lane_routes: dict[str, list[str]] = {lane: [] for lane in LANES}
    routes = payload.get("target_specific_composition", {}).get("routes") if isinstance(payload.get("target_specific_composition"), Mapping) else None
    if not isinstance(routes, list):
        routes = payload.get("target_specific_commands")
    if not isinstance(routes, list):
        routes = []
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        lane = str(route.get("lane") or route.get("role") or "").strip()
        if lane not in lane_routes:
            continue
        route_id = str(route.get("route_id") or route.get("route_alias") or route.get("registered_model_id") or "").strip()
        if route_id and route_id not in lane_routes[lane]:
            lane_routes[lane].append(route_id)
    return {lane: values for lane, values in lane_routes.items() if values}


def extract_route_signature(payload: Mapping[str, Any]) -> str:
    signature = str(payload.get("route_signature") or payload.get("route_family") or "").strip()
    if signature:
        return signature
    lane_routes = extract_lane_routes(payload)
    parts: list[str] = []
    for lane in LANES:
        for route_id in lane_routes.get(lane, []):
            parts.append(f"{lane}:{route_id}")
    return "|".join(parts)


def policy_row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize_issue_type(str(row.get("issue_type") or "")),
        str(row.get("root_cause") or "").strip().lower(),
        str(row.get("affected_stem") or "").strip().lower(),
        extract_route_signature(row),
    )


def validate_policy_dataset_row(row: Mapping[str, Any]) -> None:
    required = [
        "row_id",
        "source_run",
        "window_id",
        "candidate_id",
        "route_signature",
        "lane_routes",
        "issue_type",
        "root_cause",
        "gate_verdicts",
        "forensic_features",
        "damage_features",
        "reconstruction_features",
        "current_action",
        "oracle_action",
        "oracle_label_source",
        "usable_for_training",
        "usable_for_promotion",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"policy_dataset_row_missing_fields:{','.join(missing)}")
    action = normalize_policy_action(str(row.get("oracle_action") or ""))
    if action != row.get("oracle_action"):
        raise ValueError(f"unknown_policy_action_blocks:{row.get('oracle_action')}")
    if policy_action_can_accept_delivery(action):
        raise ValueError(f"policy_action_cannot_accept_delivery:{action}")
    if bool(row.get("usable_for_promotion")):
        raise ValueError("policy_dataset_rows_are_not_promotion_rows")
    label_type = str(row.get("label_type") or "").strip().lower()
    if bool(row.get("is_ground_truth")) and label_type in GENERATED_GROUND_TRUTH_LABEL_TYPES:
        raise ValueError(f"generated_separator_output_cannot_be_ground_truth:{label_type}")
    label_source = str(row.get("oracle_label_source") or "").strip()
    if action in COUNTERFACTUAL_REQUIRED_ACTIONS and label_source not in COUNTERFACTUAL_LABEL_SOURCES:
        raise ValueError(f"counterfactual_evidence_required_for_oracle_action:{action}")
    for path_key in ("source_run",):
        value = str(row.get(path_key) or "")
        if value and not route_policy_artifact_path_is_safe(value):
            raise ValueError(f"policy_artifact_path_segment_too_long:{path_key}")
