import json
from pathlib import Path

from tools.build_short_listen_review_handoff import build_short_listen_review_handoff


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _pack(path: Path, *, release_ready: bool = False) -> Path:
    _write_json(
        path,
        {
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
                    "duration_sec": 241.25,
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_row": {"source_candidate_id": "long_vocal_candidate_id_kept_out_of_review_files"},
                },
                {
                    "audio_path": "audio/mpr/min/m001.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                    "track_alias": "mpr",
                    "duration_sec": 241.25,
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_row": {"source_candidate_id": "long_minus_candidate_id_kept_out_of_review_files"},
                },
            ],
        },
    )
    return path


def _subset_pack(path: Path) -> Path:
    _write_json(
        path,
        {
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
                    "duration_sec": 241.25,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "artifact_relative_path": "audio/sub/g002.flac",
                    "source_alias": "mfs",
                    "stem_type": "minus",
                    "track_alias": "mfs",
                    "duration_sec": 241.25,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )
    return path


def _repair_selection_pack(path: Path) -> Path:
    _write_json(
        path,
        {
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
                    "duration_sec": 205.5,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )
    return path


def _candidate_wave_pack(path: Path) -> Path:
    _write_json(
        path,
        {
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
                    "duration_sec": 205.5,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )
    return path


def _template(path: Path, *, release_ready: bool = False) -> Path:
    root = path.parent
    template_jsonl = root / "feedback_items_template.jsonl"
    rows = [
        {
            "schema": "short_listen_feedback_template_item.v1",
            "review_rank": 1,
            "audio_path": "audio/vrep/sub/v001.flac",
            "verdict": "",
            "allowed_verdicts": ["accepted", "rejected"],
            "issue_tags": [],
            "suggested_issue_tags": ["vocal_artifact", "vocal_missing_detail"],
            "note": "",
            "source_alias": "vrep",
            "stem_type": "vocal",
            "template_only": True,
        },
        {
            "schema": "short_listen_feedback_template_item.v1",
            "review_rank": 2,
            "audio_path": "audio/mpr/min/m001.flac",
            "verdict": "",
            "allowed_verdicts": ["accepted", "rejected"],
            "issue_tags": [],
            "suggested_issue_tags": ["vocal_bleed_in_minus", "minus_muffled"],
            "note": "",
            "source_alias": "mpr",
            "stem_type": "minus",
            "template_only": True,
        },
    ]
    _write_jsonl(template_jsonl, rows)
    _write_json(
        path,
        {
            "schema": "short_listen_feedback_template_manifest.v1",
            "pack_id": "lp_test",
            "template_status": "ready_for_human_review",
            "ready_for_feedback_manifest": False,
            "template_jsonl": str(template_jsonl),
            "item_count": len(rows),
            "failed_checks": [],
            "release_ready": release_ready,
            "can_publish_as_result": False,
        },
    )
    return path


def _subset_template(path: Path) -> Path:
    root = path.parent
    template_jsonl = root / "items.jsonl"
    rows = [
        {
            "schema": "short_listen_feedback_template_item.v1",
            "review_rank": 1,
            "audio_path": "audio/sub/g001.flac",
            "verdict": "",
            "allowed_verdicts": ["accepted", "rejected"],
            "issue_tags": [],
            "suggested_issue_tags": ["vocal_artifact"],
            "note": "",
            "source_alias": "vrep",
            "stem_type": "vocal",
            "template_only": True,
        },
        {
            "schema": "short_listen_feedback_template_item.v1",
            "review_rank": 2,
            "audio_path": "audio/sub/g002.flac",
            "verdict": "",
            "allowed_verdicts": ["accepted", "rejected"],
            "issue_tags": [],
            "suggested_issue_tags": ["minus_muffled"],
            "note": "",
            "source_alias": "mfs",
            "stem_type": "minus",
            "template_only": True,
        },
    ]
    _write_jsonl(template_jsonl, rows)
    _write_json(
        path,
        {
            "schema": "short_listen_feedback_template_manifest.v1",
            "pack_id": "lp_subset",
            "template_status": "ready_for_human_review",
            "ready_for_feedback_manifest": False,
            "template_jsonl": str(template_jsonl),
            "item_count": len(rows),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    return path


def _repair_template(path: Path) -> Path:
    root = path.parent
    template_jsonl = root / "items.jsonl"
    rows = [
        {
            "schema": "short_listen_feedback_template_item.v1",
            "review_rank": 1,
            "audio_path": "audio/min/m001.flac",
            "verdict": "",
            "allowed_verdicts": ["accepted", "rejected"],
            "issue_tags": [],
            "suggested_issue_tags": ["minus_muffled", "vocal_bleed_in_minus"],
            "note": "",
            "source_alias": "mws",
            "stem_type": "minus",
            "template_only": True,
        }
    ]
    _write_jsonl(template_jsonl, rows)
    _write_json(
        path,
        {
            "schema": "short_listen_feedback_template_manifest.v1",
            "pack_id": "lp_repair",
            "template_status": "ready_for_human_review",
            "ready_for_feedback_manifest": False,
            "template_jsonl": str(template_jsonl),
            "item_count": len(rows),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    return path


def _candidate_wave_template(path: Path) -> Path:
    root = path.parent
    template_jsonl = root / "items.jsonl"
    rows = [
        {
            "schema": "short_listen_feedback_template_item.v1",
            "review_rank": 1,
            "audio_path": "audio/min/m001.flac",
            "verdict": "",
            "allowed_verdicts": ["accepted", "rejected"],
            "issue_tags": [],
            "suggested_issue_tags": ["minus_muffled", "vocal_bleed_in_minus"],
            "note": "",
            "source_alias": "cw",
            "stem_type": "minus",
            "template_only": True,
        }
    ]
    _write_jsonl(template_jsonl, rows)
    _write_json(
        path,
        {
            "schema": "short_listen_feedback_template_manifest.v1",
            "pack_id": "cw_listen",
            "template_status": "ready_for_human_review",
            "ready_for_feedback_manifest": False,
            "template_jsonl": str(template_jsonl),
            "item_count": len(rows),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    return path


def _set_template_pack_id(path: Path, pack_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pack_id"] = pack_id
    _write_json(path, payload)


def test_build_short_listen_review_handoff_writes_playlist_and_checklist(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "agg" / "pack.json")
    template = _template(tmp_path / "feedback_template" / "feedback_template_manifest.json")

    manifest = build_short_listen_review_handoff(
        pack_manifest_json=pack,
        feedback_template_manifest_json=template,
        output_dir=tmp_path / "handoff",
    )

    assert manifest["schema"] == "short_listen_review_handoff_manifest.v1"
    assert manifest["handoff_status"] == "ready_for_listen_review"
    assert manifest["item_count"] == 2
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["failed_checks"] == []
    assert [item["audio_path"] for item in manifest["items"]] == [
        "audio/vrep/sub/v001.flac",
        "audio/mpr/min/m001.flac",
    ]

    assert (tmp_path / "handoff" / "handoff.json").exists()
    assert (tmp_path / "handoff" / "playlist.m3u").exists()
    assert (tmp_path / "handoff" / "checklist.md").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "handoff").iterdir() if path.is_file())
    playlist = (tmp_path / "handoff" / "playlist.m3u").read_text(encoding="utf-8")
    checklist = (tmp_path / "handoff" / "checklist.md").read_text(encoding="utf-8")
    assert "../agg/audio/vrep/sub/v001.flac" in playlist
    assert "../agg/audio/mpr/min/m001.flac" in playlist
    assert manifest["items"][0]["playlist_entry"] == "../agg/audio/vrep/sub/v001.flac"
    assert "long_vocal_candidate_id" not in playlist
    assert "long_minus_candidate_id" not in checklist
    assert "`audio/mpr/min/m001.flac`" in checklist
    assert "vocal_bleed_in_minus" in checklist


def test_build_short_listen_review_handoff_accepts_subset_pack(tmp_path: Path) -> None:
    pack = _subset_pack(tmp_path / "subset" / "pack.json")
    template = _subset_template(tmp_path / "feedback_template" / "tpl.json")

    manifest = build_short_listen_review_handoff(
        pack_manifest_json=pack,
        feedback_template_manifest_json=template,
        output_dir=tmp_path / "handoff",
    )

    assert manifest["handoff_status"] == "ready_for_listen_review"
    assert manifest["pack_id"] == "lp_subset"
    assert [item["audio_path"] for item in manifest["items"]] == ["audio/sub/g001.flac", "audio/sub/g002.flac"]
    playlist = (tmp_path / "handoff" / "playlist.m3u").read_text(encoding="utf-8")
    assert "../subset/audio/sub/g001.flac" in playlist
    assert "../subset/audio/sub/g002.flac" in playlist


def test_build_short_listen_review_handoff_accepts_repair_selection_pack(tmp_path: Path) -> None:
    pack = _repair_selection_pack(tmp_path / "repair" / "pack.json")
    template = _repair_template(tmp_path / "feedback_template" / "tpl.json")

    manifest = build_short_listen_review_handoff(
        pack_manifest_json=pack,
        feedback_template_manifest_json=template,
        output_dir=tmp_path / "handoff",
    )

    assert manifest["handoff_status"] == "ready_for_listen_review"
    assert manifest["pack_id"] == "lp_repair"
    assert [item["audio_path"] for item in manifest["items"]] == ["audio/min/m001.flac"]
    playlist = (tmp_path / "handoff" / "playlist.m3u").read_text(encoding="utf-8")
    assert "../repair/audio/min/m001.flac" in playlist


def test_build_short_listen_review_handoff_accepts_candidate_wave_pack(tmp_path: Path) -> None:
    pack = _candidate_wave_pack(tmp_path / "cw" / "pack.json")
    template = _candidate_wave_template(tmp_path / "feedback_template" / "tpl.json")

    manifest = build_short_listen_review_handoff(
        pack_manifest_json=pack,
        feedback_template_manifest_json=template,
        output_dir=tmp_path / "handoff",
    )

    assert manifest["handoff_status"] == "ready_for_listen_review"
    assert manifest["pack_id"] == "cw_listen"
    assert [item["audio_path"] for item in manifest["items"]] == ["audio/min/m001.flac"]
    playlist = (tmp_path / "handoff" / "playlist.m3u").read_text(encoding="utf-8")
    assert "../cw/audio/min/m001.flac" in playlist


def test_build_short_listen_review_handoff_blocks_release_ready_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json", release_ready=True)
    template = _template(tmp_path / "feedback_template" / "feedback_template_manifest.json")

    manifest = build_short_listen_review_handoff(
        pack_manifest_json=pack,
        feedback_template_manifest_json=template,
        output_dir=tmp_path / "handoff",
    )

    assert manifest["handoff_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert "pack_release_ready_true" in manifest["failed_checks"]
    assert (tmp_path / "handoff" / "playlist.m3u").read_text(encoding="utf-8").strip() == "#EXTM3U"


def test_build_short_listen_review_handoff_blocks_pack_id_mismatch(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json")
    template = _template(tmp_path / "feedback_template" / "feedback_template_manifest.json")
    _set_template_pack_id(template, "different_pack")

    manifest = build_short_listen_review_handoff(
        pack_manifest_json=pack,
        feedback_template_manifest_json=template,
        output_dir=tmp_path / "handoff",
    )

    assert manifest["handoff_status"] == "blocked"
    assert "feedback_template_pack_id_mismatch" in manifest["failed_checks"]
