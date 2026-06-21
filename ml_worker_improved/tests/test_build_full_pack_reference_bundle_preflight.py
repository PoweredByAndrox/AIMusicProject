from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_reference_bundle_preflight import build_full_pack_reference_bundle_preflight


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


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


def _bundle(root: Path, alias: str, stem: str, *, verified: bool = True, rights_status: str = "user_owned") -> None:
    bundle = root / "refs" / alias
    (bundle / "stems").mkdir(parents=True)
    (bundle / "source_mix.flac").write_bytes(b"mix")
    (bundle / "stems" / f"{stem}.flac").write_bytes(stem.encode("ascii"))
    _write_json(
        bundle / "ground_truth_manifest.json",
        {
            "track_name": f"track-{alias}",
            "ground_truth_contract": {
                "verified": verified,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {stem: {"path": f"stems/{stem}.flac"}},
            "report_alignment_contract": {"rights_status": rights_status},
        },
    )


def test_reference_bundle_preflight_accepts_ready_full_pack_handoff(tmp_path: Path) -> None:
    handoff_root = tmp_path / "hand"
    handoff_json = _handoff(handoff_root)
    _bundle(handoff_root, "r001", "minus")
    _bundle(handoff_root, "r002", "guitar")

    manifest = build_full_pack_reference_bundle_preflight(
        handoff_json=handoff_json,
        handoff_root=handoff_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "full_pack_reference_bundle_preflight_manifest.v1"
    assert manifest["preflight_status"] == "ready_for_reference_intake"
    assert manifest["ready_bundle_count"] == 2
    assert manifest["blocked_bundle_count"] == 0
    assert manifest["candidate_roots"] == [str(handoff_root / "refs" / "r001"), str(handoff_root / "refs" / "r002")]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False

    ready = _read_jsonl(tmp_path / "out" / "ready.jsonl")
    assert [row["bundle_alias"] for row in ready] == ["r001", "r002"]
    assert all(row["preflight_status"] == "ready_for_reference_intake" for row in ready)
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_reference_bundle_preflight_blocks_missing_bundle_directories(tmp_path: Path) -> None:
    handoff_root = tmp_path / "hand"
    handoff_json = _handoff(handoff_root)

    manifest = build_full_pack_reference_bundle_preflight(
        handoff_json=handoff_json,
        handoff_root=handoff_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["preflight_status"] == "blocked_waiting_for_reference_bundles"
    assert manifest["ready_bundle_count"] == 0
    assert manifest["blocked_bundle_count"] == 2
    assert manifest["candidate_roots"] == []

    blocked = _read_jsonl(tmp_path / "out" / "blocked.jsonl")
    assert [row["bundle_alias"] for row in blocked] == ["r001", "r002"]
    assert all("bundle_directory_missing" in row["blockers"] for row in blocked)


def test_reference_bundle_preflight_blocks_unsafe_handoff(tmp_path: Path) -> None:
    handoff_root = tmp_path / "hand"
    handoff_json = _handoff(handoff_root, delivery_allowed=True)

    manifest = build_full_pack_reference_bundle_preflight(
        handoff_json=handoff_json,
        handoff_root=handoff_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["preflight_status"] == "blocked"
    assert manifest["ready_bundle_count"] == 0
    assert "handoff_delivery_allowed_true" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "out" / "ready.jsonl") == []


def test_reference_bundle_preflight_rejects_unverified_bundle(tmp_path: Path) -> None:
    handoff_root = tmp_path / "hand"
    handoff_json = _handoff(handoff_root)
    _bundle(handoff_root, "r001", "minus", verified=False)
    _bundle(handoff_root, "r002", "guitar")

    manifest = build_full_pack_reference_bundle_preflight(
        handoff_json=handoff_json,
        handoff_root=handoff_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["preflight_status"] == "partial_ready_waiting_for_reference_bundles"
    assert manifest["ready_bundle_count"] == 1
    assert manifest["blocked_bundle_count"] == 1
    assert manifest["candidate_roots"] == [str(handoff_root / "refs" / "r002")]

    blocked = _read_jsonl(tmp_path / "out" / "blocked.jsonl")
    assert blocked[0]["bundle_alias"] == "r001"
    assert "ground_truth_not_verified" in blocked[0]["blockers"]
