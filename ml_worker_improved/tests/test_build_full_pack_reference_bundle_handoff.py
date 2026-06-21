from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_reference_bundle_handoff import build_full_pack_reference_bundle_handoff


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collection_queue(
    tmp_path: Path,
    *,
    delivery_allowed: bool = False,
    release_ready: bool = False,
    collection_item_count: int = 2,
    missing_required_files: bool = False,
) -> Path:
    rows = [
        {
            "schema": "full_pack_reference_collection_item.v1",
            "collection_id": "r001",
            "stem_type": "minus",
            "slot_index": 1,
            "gap_id": "g001",
            "queue_item_id": "q001",
            "priority": "P0",
            "missing_training_job": False,
            "required_manifest_name": "ground_truth_manifest.json",
            "required_files": ["source_mix.flac", "stems/minus.flac", "ground_truth_manifest.json"],
            "acceptance_criteria": [
                "new_unique_track_id",
                "new_track_stem_pair",
                "ground_truth_contract_verified_true",
                "user_supplied_flstudio_music_source",
                "trainable_permission_status",
                "source_and_target_audio_paths_exist",
                "passes_full_pack_reference_corpus_intake",
            ],
            "candidate_intake_command_args": ["--candidate-root", "<candidate_bundle_root>"],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
        {
            "schema": "full_pack_reference_collection_item.v1",
            "collection_id": "r002",
            "stem_type": "guitar",
            "slot_index": 1,
            "gap_id": "g005",
            "queue_item_id": "q005",
            "priority": "P1",
            "missing_training_job": True,
            "required_manifest_name": "ground_truth_manifest.json",
            "required_files": ["source_mix.flac", "stems/guitar.flac", "ground_truth_manifest.json"],
            "acceptance_criteria": [
                "new_unique_track_id",
                "new_track_stem_pair",
                "ground_truth_contract_verified_true",
                "user_supplied_flstudio_music_source",
                "trainable_permission_status",
                "source_and_target_audio_paths_exist",
                "passes_full_pack_reference_corpus_intake",
            ],
            "candidate_intake_command_args": ["--candidate-root", "<candidate_bundle_root>"],
            "raw_reference_evidence": {
                "schema": "full_pack_raw_reference_evidence.v1",
                "source_collection_id": "r002",
                "source_worklist_bundle_alias": "r002",
                "work_item_status": "needs_verified_ground_truth_manifest",
                "raw_reference_candidate_matched": True,
                "candidate_directory": "/app/lab_artifacts/lp_v39_001/audio/t001",
                "source_mix_candidate_path": "/host_testing_stuff_music/track.mp3",
                "target_stem_candidate_path": "/app/lab_artifacts/lp_v39_001/audio/t001/guitar.flac",
                "source_mix_resolution": "pack_manifest_source_audio_path",
                "source_mix_manifest_json": "/app/lab_artifacts/lp_v39_001/pack.json",
                "track_id_hint": "t001",
                "required_actions": ["write_verified_ground_truth_manifest"],
                "blockers": [],
                "requires_clean_reference_bundle": True,
                "accepted_as_ground_truth": False,
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    ]
    if missing_required_files:
        rows[0]["required_files"] = ["source_mix.flac", "ground_truth_manifest.json"]
    items = _write_jsonl(
        tmp_path / "collect" / "items.jsonl",
        rows,
    )
    return _write_json(
        tmp_path / "collect" / "queue.json",
        {
            "schema": "full_pack_reference_collection_queue_manifest.v1",
            "manifest_type": "full_pack_reference_collection_queue",
            "queue_status": "ready_for_reference_collection",
            "failed_checks": [],
            "collection_item_count": collection_item_count,
            "remaining_by_stem": {"minus": 1, "guitar": 1},
            "outputs": {"items_jsonl": str(items), "manifest_json": str(tmp_path / "collect" / "queue.json")},
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "delivery_allowed": delivery_allowed,
            "presentation_status": "reference_collection_queue_not_for_listen_pack",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "safety": {
                "metadata_only": True,
                "audio_read": False,
                "audio_copied": False,
                "dataset_built": False,
                "training_started": False,
                "not_release_artifact": True,
            },
        },
    )


def test_reference_bundle_handoff_expands_full_pack_slots_into_short_templates(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_bundle_handoff(
        collection_queue_json=_collection_queue(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "full_pack_reference_bundle_handoff_manifest.v1"
    assert manifest["handoff_status"] == "ready_for_reference_bundle_collection"
    assert manifest["handoff_item_count"] == 2
    assert manifest["remaining_by_stem"] == {"minus": 1, "guitar": 1}
    assert manifest["required_bundle_root_layout"] == "refs/<alias>/"
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["raw_reference_evidence_item_count"] == 1
    assert manifest["raw_reference_evidence_by_status"] == {"needs_verified_ground_truth_manifest": 1}

    rows = _read_jsonl(tmp_path / "out" / "items.jsonl")
    assert [row["bundle_alias"] for row in rows] == ["r001", "r002"]
    assert [row["stem_type"] for row in rows] == ["minus", "guitar"]
    assert rows[0]["required_directory"] == "refs/r001"
    assert rows[1]["required_files"] == ["source_mix.flac", "stems/guitar.flac", "ground_truth_manifest.json"]
    assert rows[1]["ground_truth_manifest_template"]["stems"]["guitar"]["path"] == "stems/guitar.flac"
    assert rows[1]["raw_reference_evidence"]["source_mix_resolution"] == "pack_manifest_source_audio_path"
    assert rows[1]["raw_reference_evidence"]["accepted_as_ground_truth"] is False
    assert rows[1]["raw_reference_evidence"]["requires_clean_reference_bundle"] is True
    assert "passes_full_pack_reference_corpus_intake" in rows[0]["acceptance_criteria"]
    assert rows[0]["next_intake_command_args"] == ["--candidate-root", "<handoff_root>/refs/r001"]

    template = _read_json(tmp_path / "out" / "tpl.json")
    assert template["schema"] == "full_pack_reference_bundle_template.v1"
    assert template["required_files"] == ["source_mix.flac", "stems/<stem>.flac", "ground_truth_manifest.json"]
    assert template["ground_truth_manifest_template"]["ground_truth_contract"]["verified"] is True
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_reference_bundle_handoff_blocks_unsafe_collection_queue(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_bundle_handoff(
        collection_queue_json=_collection_queue(tmp_path, delivery_allowed=True),
        output_dir=tmp_path / "out",
    )

    assert manifest["handoff_status"] == "blocked"
    assert manifest["handoff_item_count"] == 0
    assert "collection_queue_delivery_allowed_true" in manifest["failed_checks"]
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False
    assert _read_jsonl(tmp_path / "out" / "items.jsonl") == []


def test_reference_bundle_handoff_blocks_release_ready_collection_queue(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_bundle_handoff(
        collection_queue_json=_collection_queue(tmp_path, release_ready=True),
        output_dir=tmp_path / "out",
    )

    assert manifest["handoff_status"] == "blocked"
    assert "collection_queue_release_ready_true" in manifest["failed_checks"]


def test_reference_bundle_handoff_blocks_collection_item_count_drift(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_bundle_handoff(
        collection_queue_json=_collection_queue(tmp_path, collection_item_count=3),
        output_dir=tmp_path / "out",
    )

    assert manifest["handoff_status"] == "blocked"
    assert "collection_queue_item_count_mismatch" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "out" / "items.jsonl") == []


def test_reference_bundle_handoff_blocks_missing_required_stem_file(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_bundle_handoff(
        collection_queue_json=_collection_queue(tmp_path, missing_required_files=True),
        output_dir=tmp_path / "out",
    )

    assert manifest["handoff_status"] == "blocked"
    assert "collection_item_required_files_missing" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "out" / "items.jsonl") == []
