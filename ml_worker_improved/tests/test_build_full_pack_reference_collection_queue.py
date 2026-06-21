from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_reference_collection_queue import build_full_pack_reference_collection_queue


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _cycle(tmp_path: Path, *, delivery_allowed: bool = False, next_action: str = "collect_verified_reference_corpus") -> Path:
    return _write_json(
        tmp_path / "cycle" / "status.json",
        {
            "schema": "full_pack_aris_cycle_status.v1",
            "cycle_status": "blocked_waiting_for_verified_reference_corpus",
            "next_action": next_action,
            "blocking_summary": {
                "remaining_reference_stem_count": 5,
                "missing_training_job_count": 1,
                "ready_training_job_count": 0,
            },
            "delivery_allowed": delivery_allowed,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )


def _gap(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q002",
                "stem_type": "drums",
                "required_reference_stem_count": 2,
                "missing_unique_track_count": 2,
                "missing_training_job": False,
                "priority": "P2",
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g002",
                "queue_item_id": "q005",
                "stem_type": "guitar",
                "required_reference_stem_count": 3,
                "missing_unique_track_count": 3,
                "missing_training_job": True,
                "priority": "P1",
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "gap_count": 2,
            "missing_reference_stem_count": 5,
            "outputs": {"rows_jsonl": str(rows)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )


def _intake(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "intake" / "intake.json",
        {
            "schema": "full_pack_reference_corpus_intake_manifest.v1",
            "intake_status": "no_accepted_references",
            "accepted_reference_rows": 0,
            "remaining_reference_stem_count": 5,
            "remaining_by_stem": {"drums": 2, "guitar": 3},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )


def _raw_worklist(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "rwork" / "items.jsonl",
        [
            {
                "schema": "full_pack_raw_reference_bundle_work_item.v1",
                "bundle_alias": "r001",
                "source_collection_id": "r001",
                "gap_id": "g001",
                "queue_item_id": "q002",
                "stem_type": "drums",
                "slot_index": 1,
                "work_item_status": "needs_verified_ground_truth_manifest",
                "raw_reference_candidate_matched": True,
                "candidate_directory": "/app/lab_artifacts/lp_v39_001/audio/t001",
                "source_mix_candidate_path": "/host_testing_stuff_music/track.mp3",
                "target_stem_candidate_path": "/app/lab_artifacts/lp_v39_001/audio/t001/drums.flac",
                "source_mix_resolution": "pack_manifest_source_audio_path",
                "source_mix_manifest_json": "/app/lab_artifacts/lp_v39_001/pack.json",
                "track_id_hint": "t001",
                "required_actions": [
                    "verify_source_provenance",
                    "write_verified_ground_truth_manifest",
                ],
                "blockers": [],
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "rwork" / "work.json",
        {
            "schema": "full_pack_raw_reference_bundle_worklist.v1",
            "worklist_status": "raw_reference_bundle_worklist_ready",
            "failed_checks": [],
            "work_item_count": 1,
            "outputs": {"items_jsonl": str(rows)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_collection_queue_expands_remaining_reference_slots(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_collection_queue(
        cycle_status_json=_cycle(tmp_path),
        reference_gap_plan_json=_gap(tmp_path),
        reference_intake_json=_intake(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "full_pack_reference_collection_queue_manifest.v1"
    assert manifest["queue_status"] == "ready_for_reference_collection"
    assert manifest["collection_item_count"] == 5
    assert manifest["remaining_by_stem"] == {"drums": 2, "guitar": 3}
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False

    items = _read_jsonl(tmp_path / "out" / "items.jsonl")
    assert [row["collection_id"] for row in items] == ["r001", "r002", "r003", "r004", "r005"]
    assert [row["stem_type"] for row in items] == ["guitar", "guitar", "guitar", "drums", "drums"]
    assert all(row["required_manifest_name"] == "ground_truth_manifest.json" for row in items)
    assert all(row["candidate_intake_command_args"] == ["--candidate-root", "<candidate_bundle_root>"] for row in items)
    assert all("new_unique_track_id" in row["acceptance_criteria"] for row in items)

    template = _read_json(tmp_path / "out" / "tpl.json")
    assert template["schema"] == "full_pack_reference_collection_template.v1"
    assert template["required_manifest_fields"]["ground_truth_contract.verified"] is True
    assert "FL Studio" in template["accepted_source_provenance"]
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_collection_queue_attaches_raw_reference_evidence_without_accepting_candidates(tmp_path: Path) -> None:
    raw_worklist = _raw_worklist(tmp_path)

    manifest = build_full_pack_reference_collection_queue(
        cycle_status_json=_cycle(tmp_path),
        reference_gap_plan_json=_gap(tmp_path),
        reference_intake_json=_intake(tmp_path),
        raw_reference_worklist_json=raw_worklist,
        output_dir=tmp_path / "out",
    )

    assert manifest["queue_status"] == "ready_for_reference_collection"
    assert manifest["raw_reference_worklist_json"] == str(raw_worklist)
    assert manifest["raw_reference_evidence_item_count"] == 1
    assert manifest["delivery_allowed"] is False

    items = _read_jsonl(tmp_path / "out" / "items.jsonl")
    drums = next(row for row in items if row["stem_type"] == "drums" and row["slot_index"] == 1)
    evidence = drums["raw_reference_evidence"]
    assert evidence["schema"] == "full_pack_raw_reference_evidence.v1"
    assert evidence["raw_reference_candidate_matched"] is True
    assert evidence["source_mix_resolution"] == "pack_manifest_source_audio_path"
    assert evidence["source_mix_manifest_json"].endswith("/pack.json")
    assert evidence["target_stem_candidate_path"].endswith("/drums.flac")
    assert evidence["requires_clean_reference_bundle"] is True
    assert evidence["accepted_as_ground_truth"] is False
    assert evidence["delivery_allowed"] is False


def test_collection_queue_blocks_unsafe_cycle_status(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_collection_queue(
        cycle_status_json=_cycle(tmp_path, delivery_allowed=True),
        reference_gap_plan_json=_gap(tmp_path),
        reference_intake_json=_intake(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert manifest["queue_status"] == "blocked"
    assert manifest["collection_item_count"] == 0
    assert "cycle_status_delivery_allowed_true" in manifest["failed_checks"]


def test_collection_queue_blocks_missing_gap_rows(tmp_path: Path) -> None:
    gap = _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )

    manifest = build_full_pack_reference_collection_queue(
        cycle_status_json=_cycle(tmp_path),
        reference_gap_plan_json=gap,
        reference_intake_json=_intake(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert manifest["queue_status"] == "blocked"
    assert manifest["collection_item_count"] == 0
    assert "reference_gap_rows_jsonl_missing" in manifest["failed_checks"]
