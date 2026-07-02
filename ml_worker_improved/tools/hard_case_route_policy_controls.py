from __future__ import annotations

from typing import Any


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def hard_case_policy_applies(policy: dict[str, Any]) -> bool:
    return bool(policy) and bool(
        policy.get("training_objective_required") is True
        or policy.get("artifact_safe_route_required") is True
        or str(policy.get("route_block_reason") or "").strip()
        or str(policy.get("route_reason") or "").strip()
    )


def hard_case_training_objective_required(policy: dict[str, Any]) -> bool:
    return policy.get("training_objective_required") is True


def hard_case_artifact_safe_route_required(policy: dict[str, Any]) -> bool:
    return policy.get("artifact_safe_route_required") is True


def hard_case_allowed_route_ids(policy: dict[str, Any]) -> list[str]:
    if not hard_case_artifact_safe_route_required(policy):
        return []
    raw = policy.get("artifact_safe_candidate_route_ids")
    if not isinstance(raw, list):
        raw = []
    values = [str(route_id).strip() for route_id in raw if str(route_id or "").strip()]
    return stable_unique(values)


def hard_case_policy_fields(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "hard_case_route_policy_applied": hard_case_policy_applies(policy),
        "hard_case_training_objective_required": hard_case_training_objective_required(policy),
        "hard_case_artifact_safe_route_required": hard_case_artifact_safe_route_required(policy),
        "hard_case_allowed_route_ids": hard_case_allowed_route_ids(policy),
        "hard_case_policy_route_reason": str(policy.get("route_reason") or ""),
        "hard_case_policy_block_reason": str(policy.get("route_block_reason") or ""),
    }
