from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .stem_route_policy_common import (
        ACTION_BLOCK_UNKNOWN,
        ACTION_PRIORITIZE_ROUTE,
        ACTION_QUARANTINE_ROUTE,
        ACTIONS,
        extract_route_signature,
        normalize_issue_type,
        normalize_policy_action,
        policy_row_key,
        validate_policy_dataset_row,
    )
except ImportError:  # pragma: no cover - direct script execution
    from stem_route_policy_common import (  # type: ignore
        ACTION_BLOCK_UNKNOWN,
        ACTION_PRIORITIZE_ROUTE,
        ACTION_QUARANTINE_ROUTE,
        ACTIONS,
        extract_route_signature,
        normalize_issue_type,
        normalize_policy_action,
        policy_row_key,
        validate_policy_dataset_row,
    )


def _counter_to_probs(counter: Mapping[str, int]) -> dict[str, float]:
    total = float(sum(max(0, int(value)) for value in counter.values()))
    if total <= 0.0:
        return {action: 1.0 if action == ACTION_BLOCK_UNKNOWN else 0.0 for action in ACTIONS}
    return {action: float(max(0, int(counter.get(action, 0)))) / total for action in ACTIONS}


def _best_action(counter: Mapping[str, int]) -> str:
    probs = _counter_to_probs(counter)
    return max(ACTIONS, key=lambda action: (probs[action], -ACTIONS.index(action)))


def _route_signature_key(row: Mapping[str, Any]) -> str:
    return extract_route_signature(row)


def _issue_key(row: Mapping[str, Any]) -> str:
    return normalize_issue_type(str(row.get("issue_type") or ""))


def train_stem_route_policy_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_counts: dict[str, Counter[str]] = defaultdict(Counter)
    route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: Counter[str] = Counter()
    usable_row_count = 0
    for row in rows:
        if not bool(row.get("usable_for_training", True)):
            continue
        validate_policy_dataset_row(row)
        action = normalize_policy_action(str(row.get("oracle_action") or ""))
        if action == ACTION_BLOCK_UNKNOWN and str(row.get("oracle_action") or "") != ACTION_BLOCK_UNKNOWN:
            continue
        exact_counts["|".join(policy_row_key(row))][action] += 1
        route_signature = _route_signature_key(row)
        if route_signature:
            route_counts[route_signature][action] += 1
        issue = _issue_key(row)
        if issue:
            issue_counts[issue][action] += 1
        action_counts[action] += 1
        usable_row_count += 1
    return {
        "schema": "stem_route_policy_checkpoint.v1",
        "model_class": "route_policy_table_v1",
        "shadow_only": True,
        "delivery_allowed": False,
        "actions": list(ACTIONS),
        "exact_counts": {key: dict(counter) for key, counter in sorted(exact_counts.items())},
        "route_counts": {key: dict(counter) for key, counter in sorted(route_counts.items())},
        "issue_counts": {key: dict(counter) for key, counter in sorted(issue_counts.items())},
        "action_counts": dict(sorted(action_counts.items())),
        "usable_training_row_count": usable_row_count,
        "fallback_action": ACTION_BLOCK_UNKNOWN,
    }


def score_policy_row(model: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    exact_key = "|".join(policy_row_key(row))
    route_key = _route_signature_key(row)
    issue_key = _issue_key(row)
    counts: Mapping[str, int] | None = None
    match_level = "fallback"
    for level, table_name, key in (
        ("exact", "exact_counts", exact_key),
        ("route_signature", "route_counts", route_key),
        ("issue_type", "issue_counts", issue_key),
    ):
        table = model.get(table_name)
        if isinstance(table, Mapping) and key and isinstance(table.get(key), Mapping):
            counts = table[key]  # type: ignore[index]
            match_level = level
            break
    if counts is None:
        counts = {ACTION_BLOCK_UNKNOWN: 1}
        blockers.append("policy_no_matching_training_key")
    probs = _counter_to_probs(counts)
    action = _best_action(counts)
    if action not in ACTIONS:
        action = ACTION_BLOCK_UNKNOWN
        blockers.append("policy_unknown_action")
    if action == ACTION_QUARANTINE_ROUTE:
        score = probs.get(ACTION_QUARANTINE_ROUTE, 0.0)
    elif action == ACTION_PRIORITIZE_ROUTE:
        score = probs.get(ACTION_PRIORITIZE_ROUTE, 0.0)
    elif action == ACTION_BLOCK_UNKNOWN:
        score = probs.get(ACTION_BLOCK_UNKNOWN, 0.0)
    else:
        score = probs.get(action, 0.0)
    return {
        "schema": "stem_route_policy_score.v1",
        "action": action,
        "action_probs": probs,
        "score": float(score),
        "match_level": match_level,
        "shadow_only": True,
        "delivery_allowed": False,
        "blockers": blockers,
    }


def save_stem_route_policy_checkpoint(model: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_stem_route_policy_checkpoint(path: Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if model.get("schema") != "stem_route_policy_checkpoint.v1":
        raise ValueError(f"unexpected_route_policy_checkpoint_schema:{model.get('schema')}")
    if bool(model.get("delivery_allowed")):
        raise ValueError("route_policy_checkpoint_cannot_allow_delivery")
    return model
