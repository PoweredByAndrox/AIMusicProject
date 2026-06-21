from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_reference_bundle_preflight import build_full_pack_reference_bundle_preflight
from tools.build_full_pack_reference_bundle_scaffold import build_full_pack_reference_bundle_scaffold


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


def _handoff(root: Path, *, delivery_allowed: bool = False, item_count: int = 2) -> Path:
    rows = [
        {
            "schema": "full_pack_reference_bundle_handoff_item.v1",
            "bundle_alias": "r001",
            "source_collection_id": "r001",
            "stem_type": "minus",
            "slot_index": 1,
            "gap_id": "g001",
            "queue_item_id": "q001",
            "priority": "P0",
            "missing_training_job": False,
            "required_directory": "refs/r001",
            "required_manifest_name": "ground_truth_manifest.json",
            "required_files": ["source_mix.flac", "stems/minus.flac", "ground_truth_manifest.json"],
            "acceptance_criteria": ["passes_full_pack_reference_corpus_intake"],
            "ground_truth_manifest_template": {
                "track_name": "<unique_track_id>",
                "ground_truth_contract": {
                    "verified": True,
                    "source": "FL Studio split mixer tracks plus full mix supplied by user",
                },
                "mix": {"path": "source_mix.flac"},
                "stems": {"minus": {"path": "stems/minus.flac"}},
                "report_alignment_contract": {"rights_status": "user_owned"},
            },
            "next_intake_command_args": ["--candidate-root", "<handoff_root>/refs/r001"],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
        {
            "schema": "full_pack_reference_bundle_handoff_item.v1",
            "bundle_alias": "r002",
            "source_collection_id": "r002",
            "stem_type": "guitar",
            "slot_index": 1,
            "gap_id": "g005",
            "queue_item_id": "q005",
            "priority": "P1",
            "missing_training_job": True,
            "required_directory": "refs/r002",
            "required_manifest_name": "ground_truth_manifest.json",
            "required_files": ["source_mix.flac", "stems/guitar.flac", "ground_truth_manifest.json"],
            "acceptance_criteria": ["passes_full_pack_reference_corpus_intake"],
            "ground_truth_manifest_template": {
                "track_name": "<unique_track_id>",
                "ground_truth_contract": {
                    "verified": True,
                    "source": "FL Studio split mixer tracks plus full mix supplied by user",
                },
                "mix": {"path": "source_mix.flac"},
                "stems": {"guitar": {"path": "stems/guitar.flac"}},
                "report_alignment_contract": {"rights_status": "user_owned"},
            },
            "next_intake_command_args": ["--candidate-root", "<handoff_root>/refs/r002"],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    ]
    items = _write_jsonl(root / "items.jsonl", rows)
    return _write_json(
        root / "handoff.json",
        {
            "schema": "full_pack_reference_bundle_handoff_manifest.v1",
            "manifest_type": "full_pack_reference_bundle_handoff",
            "handoff_status": "ready_for_reference_bundle_collection",
            "failed_checks": [],
            "handoff_item_count": item_count,
            "remaining_by_stem": {"minus": 1, "guitar": 1},
            "required_bundle_root_layout": "refs/<alias>/",
            "outputs": {"items_jsonl": str(items), "manifest_json": str(root / "handoff.json")},
            "delivery_allowed": delivery_allowed,
            "release_ready": False,
            "training_started": False,
            "can_publish_as_result": False,
            "safety": {
                "metadata_only": True,
                "audio_read": False,
                "audio_copied": False,
                "audio_decoded": False,
                "dataset_built": False,
                "training_started": False,
                "not_release_artifact": True,
            },
        },
    )


def test_reference_bundle_scaffold_creates_dirs_and_templates_without_audio_or_manifest(tmp_path: Path) -> None:
    handoff_root = tmp_path / "hand"
    handoff_json = _handoff(handoff_root)

    manifest = build_full_pack_reference_bundle_scaffold(
        handoff_json=handoff_json,
        handoff_root=handoff_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "full_pack_reference_bundle_scaffold_manifest.v1"
    assert manifest["scaffold_status"] == "created_reference_bundle_scaffold"
    assert manifest["bundle_directory_count"] == 2
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    rows = _read_jsonl(tmp_path / "out" / "items.jsonl")
    assert [row["bundle_alias"] for row in rows] == ["r001", "r002"]
    assert all(row["scaffold_status"] == "created" for row in rows)
    assert all(Path(row["bundle_dir"]).is_dir() for row in rows)
    assert (handoff_root / "refs" / "r001" / "stems").is_dir()
    assert (handoff_root / "refs" / "r001" / "tpl.json").is_file()
    assert not (handoff_root / "refs" / "r001" / "source_mix.flac").exists()
    assert not (handoff_root / "refs" / "r001" / "ground_truth_manifest.json").exists()
    assert _read_json(handoff_root / "refs" / "r001" / "tpl.json")["stems"]["minus"]["path"] == "stems/minus.flac"
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())

    preflight = build_full_pack_reference_bundle_preflight(
        handoff_json=handoff_json,
        handoff_root=handoff_root,
        output_dir=tmp_path / "pre",
    )
    assert preflight["preflight_status"] == "blocked_waiting_for_reference_bundles"
    blocked = _read_jsonl(tmp_path / "pre" / "blocked.jsonl")
    assert "bundle_directory_missing" not in blocked[0]["blockers"]
    assert "required_file_missing:source_mix.flac" in blocked[0]["blockers"]
    assert "required_file_missing:stems/minus.flac" in blocked[0]["blockers"]
    assert "ground_truth_manifest_missing" in blocked[0]["blockers"]


def test_reference_bundle_scaffold_blocks_unsafe_handoff(tmp_path: Path) -> None:
    handoff_root = tmp_path / "hand"
    handoff_json = _handoff(handoff_root, delivery_allowed=True)

    manifest = build_full_pack_reference_bundle_scaffold(
        handoff_json=handoff_json,
        handoff_root=handoff_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["scaffold_status"] == "blocked"
    assert manifest["bundle_directory_count"] == 0
    assert "handoff_delivery_allowed_true" in manifest["failed_checks"]
    assert not (handoff_root / "refs").exists()
    assert _read_jsonl(tmp_path / "out" / "items.jsonl") == []
