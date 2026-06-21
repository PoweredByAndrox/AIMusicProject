from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_aris_cycle_status import build_full_pack_aris_cycle_status


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact_set(tmp_path: Path, *, unsafe_intake: bool = False) -> dict[str, Path]:
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "blocked",
            "final_result_allowed": False,
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": ["required_stem_train_ready_missing:guitar"],
        },
    )
    queue = _write_json(
        tmp_path / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_full_pack_loop",
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )
    plan = _write_json(
        tmp_path / "plan.json",
        {
            "schema": "full_pack_loop_iteration_plan.v1",
            "plan_status": "blocked_waiting_for_verified_reference_corpus_and_minus_repair",
            "ready_action_count": 0,
            "blocked_action_count": 6,
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )
    gap = _write_json(
        tmp_path / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "gap_count": 6,
            "missing_reference_stem_count": 13,
            "missing_training_job_count": 1,
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )
    intake = _write_json(
        tmp_path / "intake.json",
        {
            "schema": "full_pack_reference_corpus_intake_manifest.v1",
            "intake_status": "no_accepted_references",
            "accepted_reference_rows": 0,
            "remaining_reference_stem_count": 13,
            "delivery_allowed": unsafe_intake,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )
    corpus = _write_json(
        tmp_path / "corpus.json",
        {
            "schema": "verified_audio_training_corpus_manifest.v1",
            "corpus_validation_status": "failed",
            "failed_checks": ["insufficient_unique_tracks"],
            "unique_track_count": 1,
            "min_unique_tracks": 3,
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )
    training = _write_json(
        tmp_path / "training.json",
        {
            "schema": "generic_audio_finetune_job_plan.v1",
            "summary": {
                "trainable_audio_examples": 52,
                "job_count": 3,
                "ready_job_count": 0,
                "blocked_job_count": 3,
                "min_unique_tracks": 3,
            },
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )
    return {
        "machine_gate_json": gate,
        "repair_queue_json": queue,
        "full_pack_loop_plan_json": plan,
        "reference_gap_plan_json": gap,
        "reference_intake_json": intake,
        "verified_corpus_manifest_json": corpus,
        "training_plan_json": training,
    }


def test_cycle_status_routes_blocked_loop_to_reference_collection(tmp_path: Path) -> None:
    manifest = build_full_pack_aris_cycle_status(output_dir=tmp_path / "out", **_artifact_set(tmp_path))

    assert manifest["schema"] == "full_pack_aris_cycle_status.v1"
    assert manifest["cycle_status"] == "blocked_waiting_for_verified_reference_corpus"
    assert manifest["next_action"] == "collect_verified_reference_corpus"
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["not_claimed"] == ["accepted_stems", "listen_result", "training_started", "model_promoted"]
    assert manifest["blocking_summary"]["remaining_reference_stem_count"] == 13
    assert manifest["blocking_summary"]["missing_training_job_count"] == 1
    assert manifest["blocking_summary"]["ready_training_job_count"] == 0
    assert (tmp_path / "out" / "status.json").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_cycle_status_blocks_unsafe_input_contract(tmp_path: Path) -> None:
    manifest = build_full_pack_aris_cycle_status(output_dir=tmp_path / "out", **_artifact_set(tmp_path, unsafe_intake=True))

    assert manifest["cycle_status"] == "blocked_invalid_cycle_inputs"
    assert manifest["next_action"] == "fix_cycle_input_contracts"
    assert "reference_intake_delivery_allowed_true" in manifest["failed_checks"]
    assert manifest["delivery_allowed"] is False
