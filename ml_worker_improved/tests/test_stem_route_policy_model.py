from __future__ import annotations

import json
from pathlib import Path

from tools.stem_route_policy_common import (
    ACTION_BLOCK_UNKNOWN,
    ACTION_PRIORITIZE_ROUTE,
    ACTION_QUARANTINE_ROUTE,
    ACTIONS,
    normalize_policy_action,
    policy_action_can_accept_delivery,
)
from tools.stem_route_policy_model import (
    load_stem_route_policy_checkpoint,
    save_stem_route_policy_checkpoint,
    score_policy_row,
    train_stem_route_policy_from_rows,
)
from tools.validate_stem_route_policy_checkpoint import validate_stem_route_policy_checkpoint


def _policy_row(
    row_id: str,
    *,
    action: str,
    issue_type: str = "infra_failure",
    route_signature: str = "minus_lane:broken",
    oracle_label_source: str = "root_cause_queue",
) -> dict:
    return {
        "schema": "stem_route_policy_row.v1",
        "row_id": row_id,
        "source_run": "run.json",
        "window_id": f"w{row_id[-3:]}",
        "candidate_id": f"c{row_id[-3:]}",
        "route_signature": route_signature,
        "lane_routes": {"minus_lane": ["broken"]},
        "issue_type": issue_type,
        "root_cause": "infra_gap" if issue_type == "infra_failure" else "routing_gap",
        "affected_stem": "",
        "gate_verdicts": {"candidate_verdict": "blocked_unknown"},
        "forensic_features": {},
        "damage_features": {},
        "reconstruction_features": {},
        "current_action": ACTION_BLOCK_UNKNOWN,
        "oracle_action": action,
        "oracle_label_source": oracle_label_source,
        "usable_for_training": True,
        "usable_for_promotion": False,
        "is_ground_truth": False,
        "label_type": "blocked_candidate_mining",
    }


def test_policy_action_cannot_accept_delivery() -> None:
    assert all(policy_action_can_accept_delivery(action) is False for action in ACTIONS)


def test_unknown_policy_action_blocks() -> None:
    assert normalize_policy_action("accept_delivery") == ACTION_BLOCK_UNKNOWN
    assert normalize_policy_action("") == ACTION_BLOCK_UNKNOWN


def test_train_tiny_policy_model_scores_action_probs(tmp_path: Path) -> None:
    rows = [
        _policy_row("p000001", action=ACTION_QUARANTINE_ROUTE),
        _policy_row("p000002", action=ACTION_PRIORITIZE_ROUTE, issue_type="bad_reconstruction", route_signature="stem_lane:better", oracle_label_source="counterfactual_replay"),
    ]
    model = train_stem_route_policy_from_rows(rows)
    checkpoint = save_stem_route_policy_checkpoint(model, tmp_path / "model.pt")
    loaded = load_stem_route_policy_checkpoint(checkpoint)

    scored = score_policy_row(loaded, rows[0])

    assert scored["action"] == ACTION_QUARANTINE_ROUTE
    assert scored["action_probs"][ACTION_QUARANTINE_ROUTE] > 0.0
    assert scored["shadow_only"] is True
    assert scored["delivery_allowed"] is False


def test_validator_fails_unchanged_replay(tmp_path: Path) -> None:
    rows = [_policy_row("p000001", action=ACTION_QUARANTINE_ROUTE)]
    model = train_stem_route_policy_from_rows(rows)
    checkpoint = save_stem_route_policy_checkpoint(model, tmp_path / "model.pt")
    rows_jsonl = tmp_path / "rows.jsonl"
    rows_jsonl.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    before_after_jsonl = tmp_path / "before_after.jsonl"
    before_after_jsonl.write_text(
        json.dumps(
            {
                "row_id": "p000001",
                "before_verdict": "blocked_unknown",
                "after_verdict": "blocked_unknown",
                "action": ACTION_QUARANTINE_ROUTE,
                "improved": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = validate_stem_route_policy_checkpoint(
        checkpoint_path=checkpoint,
        rows_jsonl_path=rows_jsonl,
        output_dir=tmp_path / "val",
        before_after_jsonl_path=before_after_jsonl,
    )

    assert manifest["promotion_allowed"] is False
    assert "no_counterfactual_improvement" in manifest["failed_checks"]


def test_validator_passes_only_with_improved_counterfactual_rows(tmp_path: Path) -> None:
    rows = [_policy_row("p000001", action=ACTION_PRIORITIZE_ROUTE, issue_type="bad_reconstruction", route_signature="stem_lane:better", oracle_label_source="counterfactual_replay")]
    model = train_stem_route_policy_from_rows(rows)
    checkpoint = save_stem_route_policy_checkpoint(model, tmp_path / "model.pt")
    rows_jsonl = tmp_path / "rows.jsonl"
    rows_jsonl.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    before_after_jsonl = tmp_path / "before_after.jsonl"
    before_after_jsonl.write_text(
        json.dumps(
            {
                "row_id": "p000001",
                "before_verdict": "blocked_bad_reconstruction",
                "after_verdict": "blocked_leak",
                "action": ACTION_PRIORITIZE_ROUTE,
                "improved": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = validate_stem_route_policy_checkpoint(
        checkpoint_path=checkpoint,
        rows_jsonl_path=rows_jsonl,
        output_dir=tmp_path / "val",
        before_after_jsonl_path=before_after_jsonl,
    )

    assert manifest["promotion_allowed"] is True
    assert manifest["false_accept_count"] == 0
