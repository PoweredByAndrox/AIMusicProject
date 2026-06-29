from __future__ import annotations

import json
from pathlib import Path

from tools.run_route_action_counterfactuals import run_route_action_counterfactuals
from tools.stem_route_policy_common import ACTION_SWITCH_MINUS_LANE


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_observed_counterfactual_marks_improved_only_from_actual_candidate_evidence(tmp_path: Path) -> None:
    rows_jsonl = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "schema": "stem_route_policy_row.v1",
                "row_id": "p000001",
                "source_root_cause_row_id": "rc000001",
                "window_id": "w000",
                "candidate_id": "c001",
                "route_signature": "stem_lane:a|vocal_lane:v|minus_lane:bad_minus",
                "issue_type": "bad_reconstruction",
                "gate_verdicts": {"candidate_verdict": "blocked_bad_reconstruction"},
                "oracle_action": "block_unknown",
            }
        ],
    )
    run_json = _write_json(
        tmp_path / "run.json",
        {
            "schema": "fully_automatic_stem_search_run.v1",
            "candidates": [
                {
                    "window_id": "w000",
                    "candidate_id": "c001",
                    "verdict": "blocked_bad_reconstruction",
                    "route_signature": "stem_lane:a|vocal_lane:v|minus_lane:bad_minus",
                    "delivery_allowed": False,
                },
                {
                    "window_id": "w000",
                    "candidate_id": "c002",
                    "verdict": "blocked_damage",
                    "route_signature": "stem_lane:a|vocal_lane:v|minus_lane:better_minus",
                    "delivery_allowed": False,
                },
            ],
        },
    )

    manifest = run_route_action_counterfactuals(
        rows_jsonl_path=rows_jsonl,
        output_dir=tmp_path / "cf",
        run_json_path=run_json,
        max_actions=8,
    )
    before_after = [json.loads(line) for line in Path(manifest["before_after_jsonl"]).read_text(encoding="utf-8").splitlines()]

    assert manifest["observed_counterfactual_improvement_count"] == 1
    assert before_after[0]["improved"] is True
    assert before_after[0]["action"] == ACTION_SWITCH_MINUS_LANE
    assert before_after[0]["root_cause_row_id"] == "rc000001"
    assert before_after[0]["evidence_source"] == "observed_run_candidate"
    assert before_after[0]["after_candidate_id"] == "c002"


def test_observed_counterfactual_does_not_improve_on_same_or_worse_gate(tmp_path: Path) -> None:
    rows_jsonl = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "schema": "stem_route_policy_row.v1",
                "row_id": "p000001",
                "source_root_cause_row_id": "rc000001",
                "window_id": "w000",
                "candidate_id": "c001",
                "route_signature": "stem_lane:a|vocal_lane:v|minus_lane:bad_minus",
                "issue_type": "bad_reconstruction",
                "gate_verdicts": {"candidate_verdict": "blocked_damage"},
                "oracle_action": "block_unknown",
            }
        ],
    )
    run_json = _write_json(
        tmp_path / "run.json",
        {
            "schema": "fully_automatic_stem_search_run.v1",
            "candidates": [
                {
                    "window_id": "w000",
                    "candidate_id": "c001",
                    "verdict": "blocked_damage",
                    "route_signature": "stem_lane:a|vocal_lane:v|minus_lane:bad_minus",
                    "delivery_allowed": False,
                },
                {
                    "window_id": "w000",
                    "candidate_id": "c002",
                    "verdict": "blocked_bad_reconstruction",
                    "route_signature": "stem_lane:a|vocal_lane:v|minus_lane:better_minus",
                    "delivery_allowed": False,
                },
            ],
        },
    )

    manifest = run_route_action_counterfactuals(
        rows_jsonl_path=rows_jsonl,
        output_dir=tmp_path / "cf",
        run_json_path=run_json,
        max_actions=8,
    )
    before_after = [json.loads(line) for line in Path(manifest["before_after_jsonl"]).read_text(encoding="utf-8").splitlines()]

    assert manifest["observed_counterfactual_improvement_count"] == 0
    assert before_after[0]["improved"] is False
    assert before_after[0]["action"] == "block_unknown"
