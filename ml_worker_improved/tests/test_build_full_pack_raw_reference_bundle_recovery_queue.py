from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_raw_reference_bundle_recovery_queue import (
    build_full_pack_raw_reference_bundle_recovery_queue,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _blocked_row(
    alias: str,
    *,
    stems: list[str],
    blockers: list[str],
    source_mix_resolution: str = "same_directory_audio",
    bundle_build_attempted: bool = False,
) -> dict:
    return {
        "schema": "full_pack_raw_reference_verified_bundle_row.v1",
        "bundle_alias": alias,
        "verified_bundle_status": "blocked",
        "requested_stems": stems,
        "verified_stems": [],
        "blockers": blockers,
        "source_work_item_aliases": [alias.replace("b", "r", 1)],
        "source_work_item_count": 1,
        "candidate_directory": f"/app/lab_artifacts/src/{alias}",
        "source_mix_candidate_path": f"/app/lab_artifacts/src/{alias}/source.flac",
        "source_mix_resolution": source_mix_resolution,
        "source_mix_manifest_json": "/app/lab_artifacts/lp_v39_001/pack.json"
        if source_mix_resolution == "pack_manifest_source_audio_path"
        else "",
        "diagnostic_bundle_root": f"/app/lab_artifacts/diag/{alias}",
        "diagnostic_manifest_json": f"/app/lab_artifacts/diag/{alias}/gt_rej.json",
        "ground_truth_manifest_json": "",
        "candidate_root": "",
        "full_mix_reconstruction_status": "rejected" if bundle_build_attempted else "",
        "failure_detail": "",
        "rights_status": "project_owned",
        "bundle_build_attempted": bundle_build_attempted,
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
        "can_publish_as_result": False,
    }


def _raw_verified_bundles(tmp_path: Path, *, delivery_allowed: bool = False) -> Path:
    blocked = _write_jsonl(
        tmp_path / "rvb" / "blocked.jsonl",
        [
            _blocked_row(
                "b001",
                stems=["guitar"],
                blockers=[
                    "source_mix_from_listen_pack_manifest_requires_clean_reference_bundle",
                    "source_not_under_allowed_root",
                ],
                source_mix_resolution="pack_manifest_source_audio_path",
            ),
            _blocked_row(
                "b002",
                stems=["minus"],
                blockers=["input_scope_not_clean", "full_mix_reconstruction_not_accepted"],
                bundle_build_attempted=True,
            ),
            _blocked_row(
                "b003",
                stems=["drums"],
                blockers=["verified_bundle_missing_requested_stem:drums", "full_mix_reconstruction_not_accepted"],
                bundle_build_attempted=True,
            ),
        ],
    )
    return _write_json(
        tmp_path / "rvb" / "bundles.json",
        {
            "schema": "full_pack_raw_reference_verified_bundles_manifest.v1",
            "bundle_status": "blocked_waiting_for_verified_raw_reference_bundles",
            "failed_checks": [],
            "verified_bundle_count": 0,
            "blocked_bundle_count": 3,
            "ready_by_stem": {},
            "blocked_by_stem": {"drums": 1, "guitar": 1, "minus": 1},
            "outputs": {"blocked_jsonl": str(blocked), "manifest_json": str(tmp_path / "rvb" / "bundles.json")},
            "delivery_allowed": delivery_allowed,
            "release_ready": False,
            "training_started": False,
            "can_publish_as_result": False,
            "safety": {
                "metadata_only": False,
                "audio_read": True,
                "audio_copied": False,
                "audio_decoded": True,
                "ground_truth_manifest_written": False,
                "not_release_artifact": True,
            },
        },
    )


def test_raw_reference_bundle_recovery_queue_routes_blocked_bundle_classes(tmp_path: Path) -> None:
    manifest = build_full_pack_raw_reference_bundle_recovery_queue(
        raw_verified_bundles_json=_raw_verified_bundles(tmp_path),
        output_dir=tmp_path / "queue",
    )

    assert manifest["schema"] == "full_pack_raw_reference_bundle_recovery_queue_manifest.v1"
    assert manifest["queue_status"] == "ready_for_raw_reference_recovery"
    assert manifest["queue_item_count"] == 3
    assert manifest["route_intent_counts"] == {
        "collect_clean_reference_bundle": 1,
        "repair_or_replace_reconstruction_failed_bundle": 1,
        "replace_missing_requested_stem_candidate": 1,
    }
    assert manifest["blocked_by_stem"] == {"drums": 1, "guitar": 1, "minus": 1}
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    rows = _read_jsonl(tmp_path / "queue" / "rows.jsonl")
    by_alias = {row["source_bundle_alias"]: row for row in rows}
    assert by_alias["b001"]["route_intent"] == "collect_clean_reference_bundle"
    assert by_alias["b001"]["source_mix_resolution"] == "pack_manifest_source_audio_path"
    assert "collect_clean_project_reference_export" in by_alias["b001"]["required_actions"]
    assert by_alias["b001"]["delivery_allowed"] is False
    assert by_alias["b002"]["route_intent"] == "repair_or_replace_reconstruction_failed_bundle"
    assert "provide_clean_scoped_export" in by_alias["b002"]["required_actions"]
    assert by_alias["b003"]["route_intent"] == "replace_missing_requested_stem_candidate"
    assert by_alias["b003"]["target_stem_type"] == "drums"
    assert "collect_requested_stem_candidate" in by_alias["b003"]["required_actions"]
    assert all(row["accepted_as_ground_truth"] is False for row in rows)
    assert all(len(path.name) <= 32 for path in (tmp_path / "queue").iterdir() if path.is_file())


def test_raw_reference_bundle_recovery_queue_blocks_unsafe_source_manifest(tmp_path: Path) -> None:
    manifest = build_full_pack_raw_reference_bundle_recovery_queue(
        raw_verified_bundles_json=_raw_verified_bundles(tmp_path, delivery_allowed=True),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert manifest["queue_item_count"] == 0
    assert "raw_verified_bundles_delivery_allowed_true" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "queue" / "rows.jsonl") == []


def test_raw_reference_bundle_recovery_queue_blocks_unsafe_blocked_rows(tmp_path: Path) -> None:
    blocked_row = _blocked_row(
        "b001",
        stems=["guitar"],
        blockers=["full_mix_reconstruction_not_accepted"],
        bundle_build_attempted=True,
    )
    blocked_row["ground_truth_manifest_json"] = "/app/lab_artifacts/unsafe/gt.json"
    blocked_row["candidate_root"] = "/app/lab_artifacts/unsafe/candidate"
    blocked = _write_jsonl(tmp_path / "rvb" / "blocked.jsonl", [blocked_row])
    source = _write_json(
        tmp_path / "rvb" / "bundles.json",
        {
            "schema": "full_pack_raw_reference_verified_bundles_manifest.v1",
            "bundle_status": "blocked_waiting_for_verified_raw_reference_bundles",
            "failed_checks": [],
            "verified_bundle_count": 0,
            "blocked_bundle_count": 1,
            "outputs": {"blocked_jsonl": str(blocked), "manifest_json": str(tmp_path / "rvb" / "bundles.json")},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "can_publish_as_result": False,
        },
    )

    manifest = build_full_pack_raw_reference_bundle_recovery_queue(
        raw_verified_bundles_json=source,
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert manifest["queue_item_count"] == 0
    assert "raw_verified_bundle_row_001_ground_truth_manifest_json_present" in manifest["failed_checks"]
    assert "raw_verified_bundle_row_001_candidate_root_present" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "queue" / "rows.jsonl") == []


def test_raw_reference_bundle_recovery_queue_blocks_unsafe_blocked_row_flags(tmp_path: Path) -> None:
    blocked_row = _blocked_row(
        "b001",
        stems=["guitar"],
        blockers=["full_mix_reconstruction_not_accepted"],
        bundle_build_attempted=True,
    )
    blocked_row["release_ready"] = True
    blocked_row["training_started"] = True
    blocked_row["can_publish_as_result"] = True
    blocked = _write_jsonl(tmp_path / "rvb" / "blocked.jsonl", [blocked_row])
    source = _write_json(
        tmp_path / "rvb" / "bundles.json",
        {
            "schema": "full_pack_raw_reference_verified_bundles_manifest.v1",
            "bundle_status": "blocked_waiting_for_verified_raw_reference_bundles",
            "failed_checks": [],
            "verified_bundle_count": 0,
            "blocked_bundle_count": 1,
            "outputs": {"blocked_jsonl": str(blocked), "manifest_json": str(tmp_path / "rvb" / "bundles.json")},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "can_publish_as_result": False,
        },
    )

    manifest = build_full_pack_raw_reference_bundle_recovery_queue(
        raw_verified_bundles_json=source,
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert manifest["queue_item_count"] == 0
    assert "raw_verified_bundle_row_001_release_ready_true" in manifest["failed_checks"]
    assert "raw_verified_bundle_row_001_training_started_true" in manifest["failed_checks"]
    assert "raw_verified_bundle_row_001_can_publish_as_result_true" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "queue" / "rows.jsonl") == []


def test_raw_reference_bundle_recovery_queue_handles_no_blocked_rows(tmp_path: Path) -> None:
    blocked = _write_jsonl(tmp_path / "rvb" / "blocked.jsonl", [])
    source = _write_json(
        tmp_path / "rvb" / "bundles.json",
        {
            "schema": "full_pack_raw_reference_verified_bundles_manifest.v1",
            "bundle_status": "verified_reference_bundles_ready",
            "failed_checks": [],
            "verified_bundle_count": 1,
            "blocked_bundle_count": 0,
            "outputs": {"blocked_jsonl": str(blocked), "manifest_json": str(tmp_path / "rvb" / "bundles.json")},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "can_publish_as_result": False,
        },
    )

    manifest = build_full_pack_raw_reference_bundle_recovery_queue(
        raw_verified_bundles_json=source,
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "no_raw_reference_recovery_needed"
    assert manifest["queue_item_count"] == 0
    assert _read_json(tmp_path / "queue" / "queue.json")["route_intent_counts"] == {}
