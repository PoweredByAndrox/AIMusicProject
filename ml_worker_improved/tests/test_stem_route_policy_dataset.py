from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_stem_route_policy_dataset import build_stem_route_policy_dataset
from tools.stem_route_policy_common import (
    ACTION_BLOCK_UNKNOWN,
    ACTION_QUARANTINE_ROUTE,
    ACTION_SWITCH_MINUS_LANE,
    ISSUE_BAD_RECONSTRUCTION,
    ISSUE_INFRA_FAILURE,
    validate_policy_dataset_row,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _minimal_run(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "fully_automatic_stem_search_run.v1",
            "mix_path": "D:/music/raw-track-title-must-not-be-used.mp3",
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "candidates": [
                {
                    "candidate_id": "c001",
                    "window_id": "w000",
                    "verdict": "blocked_unknown",
                    "route_signature": "minus_lane:bs_mamba2_vocals",
                    "target_specific_composition": {"routes": [{"lane": "minus_lane", "route_id": "bs_mamba2_vocals"}]},
                    "score_path": str(path.parent / "w000" / "c001" / "score.json"),
                },
                {
                    "candidate_id": "c002",
                    "window_id": "w000",
                    "verdict": "blocked_bad_reconstruction",
                    "route_signature": "stem_lane:htdemucs_6s|vocal_lane:mdx23c_vocals|minus_lane:becruily",
                    "target_specific_composition": {
                        "routes": [
                            {"lane": "stem_lane", "route_id": "htdemucs_6s"},
                            {"lane": "vocal_lane", "route_id": "mdx23c_vocals"},
                            {"lane": "minus_lane", "route_id": "becruily"},
                        ]
                    },
                    "score_path": str(path.parent / "w000" / "c002" / "score.json"),
                },
            ],
        },
    )


def test_infra_failure_maps_to_quarantine_route(tmp_path: Path) -> None:
    run_json = _minimal_run(tmp_path / "run.json")
    queue_jsonl = _write_jsonl(
        tmp_path / "queue.jsonl",
        [
            {
                "schema": "training_first_root_cause_sample.v1",
                "row_id": "rc000001",
                "source_run_json": str(run_json),
                "candidate_ids": ["c001"],
                "window_ids": ["w000"],
                "route_signature": "minus_lane:bs_mamba2_vocals",
                "issue_type": ISSUE_INFRA_FAILURE,
                "root_cause": "infra_gap",
                "affected_stem": "",
                "quarantined_for_next_run": True,
                "is_ground_truth": False,
                "label_type": "blocked_candidate_mining",
            }
        ],
    )

    manifest = build_stem_route_policy_dataset(run_json_path=run_json, queue_jsonl_path=queue_jsonl, output_dir=tmp_path / "ds")
    rows = [json.loads(line) for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()]

    assert rows[0]["oracle_action"] == ACTION_QUARANTINE_ROUTE
    assert rows[0]["oracle_label_source"] == "root_cause_queue"
    assert rows[0]["usable_for_training"] is True
    assert rows[0]["usable_for_promotion"] is False


def test_bad_reconstruction_without_evidence_maps_to_block_unknown(tmp_path: Path) -> None:
    run_json = _minimal_run(tmp_path / "run.json")
    queue_jsonl = _write_jsonl(
        tmp_path / "queue.jsonl",
        [
            {
                "schema": "training_first_root_cause_sample.v1",
                "row_id": "rc000002",
                "source_run_json": str(run_json),
                "candidate_ids": ["c002"],
                "window_ids": ["w000"],
                "route_signature": "stem_lane:htdemucs_6s|vocal_lane:mdx23c_vocals|minus_lane:becruily",
                "issue_type": ISSUE_BAD_RECONSTRUCTION,
                "root_cause": "routing_gap",
                "affected_stem": "drums",
                "is_ground_truth": False,
                "label_type": "blocked_candidate_mining",
            }
        ],
    )

    manifest = build_stem_route_policy_dataset(run_json_path=run_json, queue_jsonl_path=queue_jsonl, output_dir=tmp_path / "ds")
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert row["oracle_action"] == ACTION_BLOCK_UNKNOWN
    assert row["oracle_label_source"] == "no_counterfactual_evidence"
    assert row["usable_for_training"] is True


def test_counterfactual_evidence_can_set_route_switch_oracle_action(tmp_path: Path) -> None:
    run_json = _minimal_run(tmp_path / "run.json")
    queue_jsonl = _write_jsonl(
        tmp_path / "queue.jsonl",
        [
            {
                "schema": "training_first_root_cause_sample.v1",
                "row_id": "rc000002",
                "source_run_json": str(run_json),
                "candidate_ids": ["c002"],
                "window_ids": ["w000"],
                "route_signature": "stem_lane:htdemucs_6s|vocal_lane:mdx23c_vocals|minus_lane:becruily",
                "issue_type": ISSUE_BAD_RECONSTRUCTION,
                "root_cause": "routing_gap",
                "affected_stem": "drums",
                "is_ground_truth": False,
                "label_type": "blocked_candidate_mining",
            }
        ],
    )
    before_after_jsonl = _write_jsonl(
        tmp_path / "before_after.jsonl",
        [
            {
                "schema": "stem_route_policy_counterfactual_before_after.v1",
                "root_cause_row_id": "rc000002",
                "action": ACTION_SWITCH_MINUS_LANE,
                "improved": True,
                "evidence_source": "observed_run_candidate",
                "delivery_allowed": False,
                "listen_pack_created": False,
            }
        ],
    )

    manifest = build_stem_route_policy_dataset(
        run_json_path=run_json,
        queue_jsonl_path=queue_jsonl,
        output_dir=tmp_path / "ds",
        counterfactual_evidence_jsonl_path=before_after_jsonl,
    )
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert row["oracle_action"] == ACTION_SWITCH_MINUS_LANE
    assert row["oracle_label_source"] == "counterfactual_replay"
    assert row["usable_for_promotion"] is False


def test_policy_dataset_requires_counterfactual_evidence_for_oracle_action(tmp_path: Path) -> None:
    row = {
        "schema": "stem_route_policy_row.v1",
        "row_id": "p000001",
        "source_run": "run.json",
        "window_id": "w000",
        "candidate_id": "c002",
        "route_signature": "stem_lane:a|vocal_lane:b|minus_lane:c",
        "lane_routes": {},
        "issue_type": ISSUE_BAD_RECONSTRUCTION,
        "root_cause": "routing_gap",
        "affected_stem": "drums",
        "gate_verdicts": {},
        "forensic_features": {},
        "damage_features": {},
        "reconstruction_features": {},
        "current_action": ACTION_BLOCK_UNKNOWN,
        "oracle_action": "switch_minus_lane",
        "oracle_label_source": "no_counterfactual_evidence",
        "usable_for_training": True,
        "usable_for_promotion": False,
        "is_ground_truth": False,
        "label_type": "blocked_candidate_mining",
    }

    with pytest.raises(ValueError, match="counterfactual"):
        validate_policy_dataset_row(row)


def test_policy_dataset_rejects_generated_output_as_gt(tmp_path: Path) -> None:
    row = {
        "schema": "stem_route_policy_row.v1",
        "row_id": "p000002",
        "source_run": "run.json",
        "window_id": "w000",
        "candidate_id": "c001",
        "route_signature": "minus_lane:bs_mamba2_vocals",
        "lane_routes": {},
        "issue_type": ISSUE_INFRA_FAILURE,
        "root_cause": "infra_gap",
        "affected_stem": "",
        "gate_verdicts": {},
        "forensic_features": {},
        "damage_features": {},
        "reconstruction_features": {},
        "current_action": ACTION_BLOCK_UNKNOWN,
        "oracle_action": ACTION_QUARANTINE_ROUTE,
        "oracle_label_source": "root_cause_queue",
        "usable_for_training": True,
        "usable_for_promotion": False,
        "is_ground_truth": True,
        "label_type": "generated_separator",
    }

    with pytest.raises(ValueError, match="generated"):
        validate_policy_dataset_row(row)


def test_raw_track_id_not_in_policy_artifact_paths(tmp_path: Path) -> None:
    raw_track = "ytmp3free.cc_karak-yani-rainbow-youtubemp3free.org"
    run_json = _minimal_run(tmp_path / "run.json")
    queue_jsonl = _write_jsonl(
        tmp_path / "queue.jsonl",
        [
            {
                "schema": "training_first_root_cause_sample.v1",
                "row_id": "rc000001",
                "source_run_json": str(run_json),
                "candidate_ids": ["c001"],
                "window_ids": ["w000"],
                "route_signature": "minus_lane:bs_mamba2_vocals",
                "issue_type": ISSUE_INFRA_FAILURE,
                "root_cause": "infra_gap",
                "affected_stem": "",
                "quarantined_for_next_run": True,
                "is_ground_truth": False,
                "label_type": "blocked_candidate_mining",
            }
        ],
    )

    manifest = build_stem_route_policy_dataset(run_json_path=run_json, queue_jsonl_path=queue_jsonl, output_dir=tmp_path / "policy_ds")

    assert raw_track not in manifest["rows_jsonl"]
    assert raw_track not in manifest["manifest_path"]
    assert raw_track not in manifest["splits_json"]
