import json
from pathlib import Path

from tools.run_short_listen_review_packet import run_short_listen_review_packet


REQUIRED_FULL_PACK_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")


def _pack(root: Path, *, release_ready: bool = False) -> Path:
    pack = root / "pack.json"
    _touch(root / "audio/vrep/sub/v001.flac")
    _touch(root / "audio/mpr/min/m001.flac")
    _write_json(
        pack,
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "failed_checks": [],
            "coverage_json": str(root / "coverage.json"),
            "items": [
                {
                    "audio_path": "audio/vrep/sub/v001.flac",
                    "source_alias": "vrep",
                    "stem_type": "vocal",
                    "track_alias": "vrep",
                    "duration_sec": 100.0,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "audio_path": "audio/mpr/min/m001.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                    "track_alias": "mpr",
                    "duration_sec": 100.0,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
            "artifact_name_policy": {
                "schema": "short_windows_safe_artifact_names.v1",
                "raw_track_or_stem_ids_in_path": False,
                "provenance_in_manifest": True,
            },
        },
    )
    _write_json(
        root / "coverage.json",
        {
            "schema": "aggregate_short_listen_pack_coverage.v1",
            "pack_id": "lp_test",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "stem_groups": [
                {
                    "group_alias": "six",
                    "item_count": 7,
                    "stem_types": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
                    "semantic_quality_verified": False,
                }
            ],
        },
    )
    _write_json(
        root / "validation" / "validation.json",
        {
            "schema": "aggregate_short_listen_pack_validation.v1",
            "validation_status": "passed",
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    return pack


def _subset_pack(root: Path) -> Path:
    pack = root / "pack.json"
    _touch(root / "audio/sub/g001.flac")
    _touch(root / "audio/sub/g002.flac")
    _write_json(
        pack,
        {
            "schema": "short_listen_subset_pack_manifest.v1",
            "pack_id": "lp_subset",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "selected_count": 2,
            "listen_audio_files": [
                {
                    "artifact_relative_path": "audio/sub/g001.flac",
                    "stem_type": "vocal",
                    "duration_sec": 100.0,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "artifact_relative_path": "audio/sub/g002.flac",
                    "stem_type": "minus",
                    "duration_sec": 100.0,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )
    return pack


def _validation(root: Path) -> Path:
    path = root / "validation.json"
    _write_json(
        path,
        {
            "schema": "aggregate_short_listen_pack_validation.v1",
            "validation_status": "passed",
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    return path


def test_run_short_listen_review_packet_builds_all_review_artifacts(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    out = tmp_path / "packet"

    manifest = run_short_listen_review_packet(pack_manifest_json=pack, output_dir=out)

    assert manifest["schema"] == "short_listen_review_packet_manifest.v1"
    assert manifest["packet_status"] == "ready_for_listen_review"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["failed_checks"] == []
    assert manifest["stages"]["feedback_template"]["template_status"] == "ready_for_human_review"
    assert manifest["stages"]["review_handoff"]["handoff_status"] == "ready_for_listen_review"
    assert manifest["stages"]["review_handoff_validation"]["validation_status"] == "passed"
    assert manifest["stages"]["goal_audit"]["goal_status"] == "machine_evidence_incomplete"
    assert (out / "packet.json").exists()
    assert (out / "feedback_template" / "items.jsonl").exists()
    assert (out / "review_handoff" / "playlist.m3u").exists()
    assert (out / "review_handoff_validation" / "validation.json").exists()
    assert (out / "goal_audit" / "goal_status_audit.json").exists()
    for path in out.rglob("*"):
        if path.is_file() or path.is_dir():
            assert len(path.name) <= 32


def test_run_short_listen_review_packet_skips_goal_audit_for_subset_pack(tmp_path: Path) -> None:
    pack = _subset_pack(tmp_path / "pack")
    validation = _validation(tmp_path / "validation")
    out = tmp_path / "packet"

    manifest = run_short_listen_review_packet(
        pack_manifest_json=pack,
        aggregate_validation_json=validation,
        output_dir=out,
    )

    assert manifest["packet_status"] == "ready_for_listen_review"
    assert manifest["failed_checks"] == []
    assert manifest["stages"]["review_handoff_validation"]["validation_status"] == "passed"
    assert manifest["stages"]["goal_audit"]["goal_status"] == "not_run"
    assert manifest["stages"]["goal_audit"]["remaining_blockers"] == ["aggregate_goal_audit_not_applicable_to_subset_pack"]
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert not (out / "goal_audit" / "goal_status_audit.json").exists()
    assert manifest["stages"]["feedback_template"]["template_status"] == "ready_for_human_review"
    assert manifest["stages"]["review_handoff"]["handoff_status"] == "ready_for_listen_review"
    assert (out / "packet.json").exists()
    assert (out / "feedback_template" / "items.jsonl").exists()
    assert (out / "review_handoff" / "playlist.m3u").exists()
    assert (out / "review_handoff_validation" / "validation.json").exists()
    for path in out.rglob("*"):
        if path.is_file() or path.is_dir():
            assert len(path.name) <= 32


def test_run_short_listen_review_packet_full_pack_contract_blocks_missing_stems(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    validation = _validation(tmp_path / "validation")

    manifest = run_short_listen_review_packet(
        pack_manifest_json=pack,
        aggregate_validation_json=validation,
        output_dir=tmp_path / "packet",
        required_stems=REQUIRED_FULL_PACK_STEMS,
    )

    assert manifest["packet_status"] == "failed_safety_contract"
    assert manifest["required_stems"] == REQUIRED_FULL_PACK_STEMS
    assert manifest["present_stems"] == ["vocal", "minus"]
    assert manifest["missing_required_stems"] == ["drums", "bass", "other", "guitar", "piano"]
    assert "full_pack_required_stem_missing:drums" in manifest["failed_checks"]
    assert "feedback_template" not in manifest["stages"]
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_run_short_listen_review_packet_blocks_subset_schema_with_full_pack_contract(tmp_path: Path) -> None:
    pack = _subset_pack(tmp_path / "pack")
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["required_stems"] = REQUIRED_FULL_PACK_STEMS
    payload["present_stems"] = ["vocal", "minus"]
    payload["missing_required_stems"] = ["drums", "bass", "other", "guitar", "piano"]
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = run_short_listen_review_packet(
        pack_manifest_json=pack,
        aggregate_validation_json=_validation(tmp_path / "validation"),
        output_dir=tmp_path / "packet",
    )

    assert manifest["packet_status"] == "failed_safety_contract"
    assert manifest["required_stems"] == REQUIRED_FULL_PACK_STEMS
    assert manifest["present_stems"] == ["vocal", "minus"]
    assert manifest["missing_required_stems"] == ["drums", "bass", "other", "guitar", "piano"]
    assert "full_pack_required_stem_missing:drums" in manifest["failed_checks"]
    assert "feedback_template" not in manifest["stages"]
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_run_short_listen_review_packet_fails_closed_on_release_ready_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack", release_ready=True)

    manifest = run_short_listen_review_packet(pack_manifest_json=pack, output_dir=tmp_path / "packet")

    assert manifest["packet_status"] == "failed_safety_contract"
    assert "feedback_template:pack_release_ready_true" in manifest["failed_checks"]
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
