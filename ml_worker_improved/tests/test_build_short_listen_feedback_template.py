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
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_ready": release_ready,
        "can_publish_as_result": False,
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


def test_build_short_listen_feedback_template_writes_all_short_paths(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")
    out = tmp_path / "feedback_template"

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, output_dir=out)

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


def test_build_short_listen_feedback_template_accepts_subset_pack(tmp_path: Path) -> None:
    pack = _subset_pack(tmp_path / "subset" / "pack.json")

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "ready_for_human_review"
    assert manifest["pack_id"] == "lp_subset"
    rows = [json.loads(line) for line in (tmp_path / "out" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["audio_path"] for row in rows] == ["audio/sub/g001.flac", "audio/sub/g002.flac"]
    assert rows[1]["stem_type"] == "minus"


def test_build_short_listen_feedback_template_accepts_repair_selection_pack(tmp_path: Path) -> None:
    pack = _repair_selection_pack(tmp_path / "repair" / "pack.json")

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "ready_for_human_review"
    assert manifest["pack_id"] == "lp_repair"
    rows = [json.loads(line) for line in (tmp_path / "out" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["audio_path"] for row in rows] == ["audio/min/m001.flac"]
    assert rows[0]["stem_type"] == "minus"


def test_build_short_listen_feedback_template_accepts_candidate_wave_pack(tmp_path: Path) -> None:
    pack = _candidate_wave_pack(tmp_path / "cw" / "pack.json")

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "ready_for_human_review"
    assert manifest["pack_id"] == "cw_listen"
    rows = [json.loads(line) for line in (tmp_path / "out" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["audio_path"] for row in rows] == ["audio/min/m001.flac"]
    assert rows[0]["stem_type"] == "minus"


def test_build_short_listen_feedback_template_blocks_release_ready_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json", release_ready=True)

    manifest = build_short_listen_feedback_template(pack_manifest_json=pack, output_dir=tmp_path / "out")

    assert manifest["template_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert "pack_release_ready_true" in manifest["failed_checks"]
    assert not (tmp_path / "out" / "items.jsonl").read_text(encoding="utf-8").strip()
