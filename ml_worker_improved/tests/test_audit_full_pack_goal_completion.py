from __future__ import annotations

import json
from pathlib import Path

from tools.audit_full_pack_goal_completion import audit_full_pack_goal_completion


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _machine_gate(tmp_path: Path, *, accepted: bool = False) -> Path:
    return _write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "accepted" if accepted else "blocked",
            "final_result_allowed": accepted,
            "delivery_allowed": accepted,
            "presentation_status": "accepted_for_listen_pack" if accepted else "machine_gate_blocked_not_for_listen_pack",
            "failed_checks": [] if accepted else ["required_stem_train_ready_missing:minus"],
            "missing_required_stems": [],
            "required_stems": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
            "stem_types": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"] if accepted else ["vocal"],
            "required_machine_gates": [
                "aggregate_pack_validation_passed",
                "full_required_stem_coverage",
                "required_stem_machine_quality_evidence",
                "source_vocal_minus_master_pair_compatible",
                "minus_quality_audit_passed_without_warnings",
                "required_stem_train_ready_model_evidence",
            ],
            "require_training_coverage": True,
            "train_ready_stems": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"] if accepted else ["vocal"],
            "contract": {
                "training_coverage_required": True,
                "train_ready_model_evidence_required": True,
            },
            "release_ready": False,
            "training_started": False,
        },
    )


def _cycle(tmp_path: Path, *, complete: bool = False) -> Path:
    return _write_json(
        tmp_path / "cycle" / "status.json",
        {
            "schema": "full_pack_aris_cycle_status.v1",
            "cycle_status": "accepted_full_pack_machine_ready" if complete else "blocked_waiting_for_verified_reference_corpus",
            "next_action": "final_delivery_audit" if complete else "collect_verified_reference_corpus",
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
            "blocking_summary": {
                "remaining_reference_stem_count": 0 if complete else 13,
                "ready_training_job_count": 3 if complete else 0,
                "blocked_training_job_count": 0 if complete else 3,
                "verified_corpus_failed_checks": [] if complete else ["insufficient_unique_tracks"],
                "machine_gate_failed_checks": [] if complete else ["required_stem_train_ready_missing:minus"],
            },
        },
    )


def _collection_queue(tmp_path: Path, *, complete: bool = False) -> Path:
    return _write_json(
        tmp_path / "collect" / "queue.json",
        {
            "schema": "full_pack_reference_collection_queue_manifest.v1",
            "queue_status": "no_reference_collection_needed" if complete else "ready_for_reference_collection",
            "collection_item_count": 0 if complete else 13,
            "remaining_by_stem": {} if complete else {"minus": 2, "drums": 2},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )


def _corpus(tmp_path: Path, *, complete: bool = False) -> Path:
    return _write_json(
        tmp_path / "corpus" / "val.json",
        {
            "schema": "verified_audio_training_corpus_manifest.v1",
            "corpus_validation_status": "passed" if complete else "failed",
            "failed_checks": [] if complete else ["insufficient_unique_tracks"],
            "verified_trainable_examples": 64 if complete else 52,
            "unique_track_count": 4 if complete else 1,
            "min_unique_tracks": 3,
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _training_plan(tmp_path: Path, *, complete: bool = False) -> Path:
    return _write_json(
        tmp_path / "train" / "plan.json",
        {
            "schema": "generic_audio_finetune_job_plan.v1",
            "summary": {
                "job_count": 3,
                "ready_job_count": 3 if complete else 0,
                "blocked_job_count": 0 if complete else 3,
                "trainable_audio_examples": 64 if complete else 52,
            },
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )


def test_goal_completion_audit_blocks_current_reference_collection_state(tmp_path: Path) -> None:
    audit = audit_full_pack_goal_completion(
        machine_gate_json=_machine_gate(tmp_path, accepted=False),
        cycle_status_json=_cycle(tmp_path, complete=False),
        collection_queue_json=_collection_queue(tmp_path, complete=False),
        verified_corpus_manifest_json=_corpus(tmp_path, complete=False),
        training_plan_json=_training_plan(tmp_path, complete=False),
        output_dir=tmp_path / "audit",
    )

    assert audit["goal_complete"] is False
    assert audit["completion_status"] == "machine_evidence_incomplete"
    assert audit["delivery_allowed"] is False
    assert audit["release_ready"] is False
    assert audit["training_started"] is False
    assert "machine_gate_not_accepted" in audit["remaining_blockers"]
    assert "cycle_next_action:collect_verified_reference_corpus" in audit["remaining_blockers"]
    assert "reference_collection_pending" in audit["remaining_blockers"]
    assert "verified_corpus_failed:insufficient_unique_tracks" in audit["remaining_blockers"]
    assert "training_jobs_blocked" in audit["remaining_blockers"]


def test_goal_completion_audit_accepts_only_machine_complete_state(tmp_path: Path) -> None:
    audit = audit_full_pack_goal_completion(
        machine_gate_json=_machine_gate(tmp_path, accepted=True),
        cycle_status_json=_cycle(tmp_path, complete=True),
        collection_queue_json=_collection_queue(tmp_path, complete=True),
        verified_corpus_manifest_json=_corpus(tmp_path, complete=True),
        training_plan_json=_training_plan(tmp_path, complete=True),
        output_dir=tmp_path / "audit",
    )

    assert audit["goal_complete"] is True
    assert audit["completion_status"] == "machine_accepted"
    assert audit["result_delivery_allowed"] is True
    assert audit["delivery_allowed"] is False
    assert audit["presentation_status"] == "completion_audit_not_for_listen_pack"
    assert audit["remaining_blockers"] == []


def test_goal_completion_audit_blocks_gate_without_actual_full_pack_stems(tmp_path: Path) -> None:
    gate = json.loads(_machine_gate(tmp_path, accepted=True).read_text(encoding="utf-8"))
    gate["required_stems"] = ["vocal", "minus"]
    gate["stem_types"] = ["vocal", "minus"]
    gate_path = _write_json(tmp_path / "gate" / "gate.json", gate)

    audit = audit_full_pack_goal_completion(
        machine_gate_json=gate_path,
        cycle_status_json=_cycle(tmp_path, complete=True),
        collection_queue_json=_collection_queue(tmp_path, complete=True),
        verified_corpus_manifest_json=_corpus(tmp_path, complete=True),
        training_plan_json=_training_plan(tmp_path, complete=True),
        output_dir=tmp_path / "audit",
    )

    assert audit["goal_complete"] is False
    assert "required_stem_missing_from_gate_contract:bass" in audit["remaining_blockers"]
    assert "required_stem_missing_from_gate_pack:bass" in audit["remaining_blockers"]
    assert "required_stem_missing_from_gate_contract:piano" in audit["remaining_blockers"]
    assert "required_stem_missing_from_gate_pack:piano" in audit["remaining_blockers"]
