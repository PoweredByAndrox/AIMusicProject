import json
from pathlib import Path

from tools.build_short_listen_feedback_template import build_short_listen_feedback_template


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pack(path: Path, *, release_ready: bool = False) -> Path:
    payload = {
        "schema": "aggregate_short_listen_pack_manifest.v1",
        "pack_id": "lp_test",
        "delivery_status": "aggregate_validation_pending",
        "delivery_allowed": False,
        "presentation_status": "aggregate_validation_pending_not_for_listen_pack",
        "release_status": "aggregate_validation_pending",
        "release_ready": release_ready,
        "can_publish_as_result": False,
        "listenable": False,
        "deliverable": False,
        "aggregate_validation_ready": True,
        "failed_checks": [],
        "items": [
            {
                "audio_path": "audio/vrep/sub/v001.flac",
                "source_alias": "vrep",
                "stem_type": "vocal",
                "track_alias": "vrep",
                "release_ready": False,
                "can_publish_as_result": False,
                "source_row": {"source_candidate_id": "very_long_candidate_id_kept_only_in_manifest"},
            },
            {
                "audio_path": "audio/mpr/min/m001.flac",
                "source_alias": "mpr",
                "stem_type": "minus",
                "track_alias": "mpr",
                "release_ready": False,
                "can_publish_as_result": False,
                "source_row": {"source_candidate_id": "minus_candidate_id_kept_only_in_manifest"},
            },
        ],
    }
    _write_json(path, payload)
    return path


def _subset_pack(path: Path) -> Path:
    payload = {
        "schema": "short_listen_subset_pack_manifest.v1",
        "pack_id": "lp_subset",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "failed_checks": [],
        "listen_audio_files": [
            {
                "artifact_relative_path": "audio/sub/g001.flac",
                "source_alias": "vrep",
                "stem_type": "vocal",
                "track_alias": "vrep",
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "artifact_relative_path": "audio/sub/g002.flac",
                "source_alias": "mfs",
                "stem_type": "minus",
                "track_alias": "mfs",
                "minus_audit_gate": {"audit_decision": "passed"},
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
    }
    _write_json(path, payload)
    return path


def _repair_selection_pack(path: Path) -> Path:
    payload = {
        "schema": "short_listen_repair_selection_pack_manifest.v1",
        "pack_id": "lp_repair",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "failed_checks": [],
        "listen_audio_files": [
            {
                "artifact_relative_path": "audio/min/m001.flac",
                "source_alias": "mws",
                "stem_type": "minus",
                "track_alias": "trk",
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    }
    _write_json(path, payload)
    return path


def _candidate_wave_pack(path: Path) -> Path:
    payload = {
        "schema": "short_candidate_wave_listen_pack_manifest.v1",
        "pack_id": "cw_listen",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "failed_checks": [],
        "listen_audio_files": [
            {
                "artifact_relative_path": "audio/min/m001.flac",
                "source_alias": "cw",
                "stem_type": "minus",
                "track_alias": "t001",
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    }
    _write_json(path, payload)
    return path


def _machine_gate(path: Path, *, pack: Path, accepted: bool = True) -> Path:
    _write_json(
        path.parent / "validation.json",
        {
            "schema": "aggregate_short_listen_pack_validation.v1",
            "validation_status": "passed" if accepted else "failed",
            "failed_checks": [] if accepted else ["blocked_fixture"],
            "pack_manifest_json": str(pack),
        },
    )
    _write_json(
        path.parent / "semantic_review.json",
        {
            "schema": "short_listen_semantic_review.v1",
            "review_status": "passed" if accepted else "blocked",
            "failed_checks": [] if accepted else ["blocked_fixture"],
            "pack_manifest_json": str(pack),
        },
    )
    _write_json(
        path.parent / "minus_audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed" if accepted else "blocked",
            "failed_checks": [] if accepted else ["blocked_fixture"],
            "pack_manifest_json": str(pack),
        },
    )
    _write_json(
        path,
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "accepted" if accepted else "blocked",
            "final_result_allowed": accepted,
            "delivery_status": "accepted" if accepted else "blocked",
            "delivery_allowed": accepted,
            "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "pack_manifest_json": str(pack),
            "validation_json": str(path.parent / "validation.json"),
            "minus_audit_json": str(path.parent / "minus_audit.json"),
            "minus_audit_status": "completed" if accepted else "blocked",
            "semantic_review_json": str(path.parent / "semantic_review.json"),
            "semantic_review_evidence_count": 2 if accepted else 0,
            "failed_checks": [] if accepted else ["minus_audit_row_warnings_present"],
        },
    )
    return path


def _missing_evidence_machine_gate(path: Path, *, pack: Path) -> Path:
    _write_json(
        path,
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "accepted",
            "final_result_allowed": True,
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "pack_manifest_json": str(pack),
            "validation_json": str(path.parent / "validation.json"),
            "minus_audit_json": str(path.parent / "minus_audit.json"),
            "minus_audit_status": "completed",
            "semantic_review_json": str(path.parent / "semantic_review.json"),
            "semantic_review_evidence_count": 2,
            "failed_checks": [],
        },
    )
    return path


def test_build_short_listen_feedback_template_writes_all_short_paths(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack)
    out = tmp_path / "feedback_template"

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=out)

    assert manifest["schema"] == "short_listen_feedback_template_manifest.v1"
    assert manifest["template_status"] == "ready_for_human_review"
    assert manifest["ready_for_feedback_manifest"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["item_count"] == 2
    assert manifest["failed_checks"] == []

    assert (out / "tpl.json").exists()
    assert (out / "items.jsonl").exists()
    assert (out / "summary.md").exists()
    assert all(len(path.name) <= 32 for path in out.iterdir() if path.is_file())
    rows = [json.loads(line) for line in (out / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["audio_path"] for row in rows] == ["audio/vrep/sub/v001.flac", "audio/mpr/min/m001.flac"]
    assert {row["verdict"] for row in rows} == {""}
    assert rows[0]["allowed_verdicts"] == ["accepted", "rejected"]
    assert "very_long_candidate_id" not in rows[0]["audio_path"]
    assert rows[1]["suggested_issue_tags"] == [
        "vocal_bleed_in_minus",
        "minus_muffled",
        "phase_or_stereo_issue",
        "missing_instrument_detail",
        "artifact_noise",
    ]


def test_build_short_listen_feedback_template_blocks_subset_pack(tmp_path: Path) -> None:
    pack = _subset_pack(tmp_path / "subset" / "pack.json")
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert manifest["pack_id"] == "lp_subset"
    assert "pack_schema_unexpected" in manifest["failed_checks"]


def test_build_short_listen_feedback_template_blocks_repair_selection_pack(tmp_path: Path) -> None:
    pack = _repair_selection_pack(tmp_path / "repair" / "pack.json")
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert manifest["pack_id"] == "lp_repair"
    assert "pack_schema_unexpected" in manifest["failed_checks"]


def test_build_short_listen_feedback_template_blocks_candidate_wave_pack(tmp_path: Path) -> None:
    pack = _candidate_wave_pack(tmp_path / "cw" / "pack.json")
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert manifest["pack_id"] == "cw_listen"
    assert "pack_schema_unexpected" in manifest["failed_checks"]


def test_build_short_listen_feedback_template_blocks_release_ready_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json", release_ready=True)
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert "pack_release_ready_true" in manifest["failed_checks"]
    assert not (tmp_path / "out" / "items.jsonl").read_text(encoding="utf-8").strip()


def test_build_short_listen_feedback_template_blocks_stale_accepted_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["delivery_status"] = "accepted"
    payload["delivery_allowed"] = True
    payload["presentation_status"] = "accepted_for_listen_pack"
    payload["release_status"] = "listen_preview_only"
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert "pack_delivery_status_not_aggregate_validation_pending" in manifest["failed_checks"]
    assert "pack_delivery_allowed_not_false" in manifest["failed_checks"]


def test_build_short_listen_feedback_template_blocks_missing_machine_gate(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert "machine_gate_json_missing" in manifest["failed_checks"]


def test_build_short_listen_feedback_template_blocks_unaccepted_machine_gate(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack, accepted=False)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert "machine_gate_not_accepted" in manifest["failed_checks"]


def test_build_short_listen_feedback_template_blocks_missing_gate_evidence_files(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")
    gate = _missing_evidence_machine_gate(tmp_path / "gate" / "gate.json", pack=pack)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert "machine_gate_validation_json_missing" in manifest["failed_checks"]
    assert "machine_gate_semantic_review_json_missing" in manifest["failed_checks"]
    assert "machine_gate_minus_audit_json_missing" in manifest["failed_checks"]


def test_build_short_listen_feedback_template_blocks_incomplete_semantic_review(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")
    gate = _machine_gate(tmp_path / "gate" / "gate.json", pack=pack)
    semantic = json.loads((tmp_path / "gate" / "semantic_review.json").read_text(encoding="utf-8"))
    semantic["review_status"] = "pending"
    (tmp_path / "gate" / "semantic_review.json").write_text(json.dumps(semantic, sort_keys=True), encoding="utf-8")

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, machine_gate_json=gate, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert "machine_gate_semantic_review_not_passed" in manifest["failed_checks"]
