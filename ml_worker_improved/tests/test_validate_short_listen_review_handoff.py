import json
from pathlib import Path

from tools.validate_short_listen_review_handoff import validate_short_listen_review_handoff


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_handoff(
    root: Path,
    *,
    release_ready: bool = False,
    playlist_extra: str = "",
    subset_pack: bool = False,
    repair_selection_pack: bool = False,
    candidate_wave_pack: bool = False,
) -> Path:
    pack = root / "pack.json"
    template = root / "feedback_template" / "feedback_template_manifest.json"
    template_jsonl = root / "feedback_template" / "feedback_items_template.jsonl"
    handoff = root / "review_handoff"
    listen_audio_schema = subset_pack or repair_selection_pack or candidate_wave_pack
    pack_schema = "aggregate_short_listen_pack_manifest.v1"
    pack_id = "lp_test"
    if subset_pack:
        pack_schema = "short_listen_subset_pack_manifest.v1"
        pack_id = "lp_subset"
    if repair_selection_pack:
        pack_schema = "short_listen_repair_selection_pack_manifest.v1"
        pack_id = "lp_repair"
    if candidate_wave_pack:
        pack_schema = "short_candidate_wave_listen_pack_manifest.v1"
        pack_id = "cw_listen"
    _write_json(
        pack,
        {
            "schema": pack_schema,
            "pack_id": pack_id,
            "delivery_status": "accepted" if listen_audio_schema else "aggregate_validation_pending",
            "delivery_allowed": True if listen_audio_schema else False,
            "presentation_status": "accepted_for_listen_pack" if listen_audio_schema else "aggregate_validation_pending_not_for_listen_pack",
            "release_status": "listen_preview_only" if listen_audio_schema else "aggregate_validation_pending",
            "release_ready": False,
            "can_publish_as_result": False,
            "listenable": True if listen_audio_schema else False,
            "deliverable": False,
            "aggregate_validation_ready": False if listen_audio_schema else True,
            "failed_checks": [],
            ("listen_audio_files" if listen_audio_schema else "items"): [
                {
                    ("artifact_relative_path" if listen_audio_schema else "audio_path"): "audio/vrep/sub/v001.flac",
                    "source_alias": "vrep",
                    "stem_type": "vocal",
                },
                {
                    ("artifact_relative_path" if listen_audio_schema else "audio_path"): "audio/mpr/min/m001.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                },
            ],
        },
    )
    for relative_audio in ("audio/vrep/sub/v001.flac", "audio/mpr/min/m001.flac"):
        audio_path = root / relative_audio
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"audio-placeholder")
    template_jsonl.parent.mkdir(parents=True, exist_ok=True)
    template_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"audio_path": "audio/vrep/sub/v001.flac", "review_rank": 1, "suggested_issue_tags": ["vocal_artifact"]}),
                json.dumps({"audio_path": "audio/mpr/min/m001.flac", "review_rank": 2, "suggested_issue_tags": ["minus_muffled"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "validation" / "validation.json",
        {
            "schema": "aggregate_short_listen_pack_validation.v1",
            "validation_status": "passed",
            "failed_checks": [],
            "pack_manifest_json": str(pack),
        },
    )
    _write_json(
        root / "semantic" / "review.json",
        {
            "schema": "short_listen_semantic_review.v1",
            "review_status": "passed",
            "failed_checks": [],
            "pack_manifest_json": str(pack),
        },
    )
    _write_json(
        root / "minus" / "audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed",
            "failed_checks": [],
            "pack_manifest_json": str(pack),
        },
    )
    _write_json(
        root / "gate" / "gate.json",
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
            "validation_json": str(root / "validation" / "validation.json"),
            "minus_audit_json": str(root / "minus" / "audit.json"),
            "minus_audit_status": "completed",
            "semantic_review_json": str(root / "semantic" / "review.json"),
            "semantic_review_evidence_count": 2,
            "failed_checks": [],
        },
    )
    _write_json(
        template,
        {
            "schema": "short_listen_feedback_template_manifest.v1",
            "pack_id": pack_id,
            "template_status": "ready_for_human_review",
            "ready_for_feedback_manifest": False,
            "machine_gate_json": str(root / "gate" / "gate.json"),
            "template_jsonl": str(template_jsonl),
            "item_count": 2,
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    _write_json(
        handoff / "handoff.json",
        {
            "schema": "short_listen_review_handoff_manifest.v1",
            "pack_id": pack_id,
            "pack_manifest_json": str(pack),
            "feedback_template_manifest_json": str(template),
            "machine_gate_json": str(root / "gate" / "gate.json"),
            "handoff_status": "ready_for_listen_review",
            "item_count": 2,
            "items": [
                {
                    "review_rank": 1,
                    "audio_path": "audio/vrep/sub/v001.flac",
                    "playlist_entry": "../audio/vrep/sub/v001.flac",
                    "source_alias": "vrep",
                    "stem_type": "vocal",
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "review_rank": 2,
                    "audio_path": "audio/mpr/min/m001.flac",
                    "playlist_entry": "../audio/mpr/min/m001.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
            "playlist_m3u": str(handoff / "playlist.m3u"),
            "listen_checklist_md": str(handoff / "checklist.md"),
            "failed_checks": [],
            "release_ready": release_ready,
            "can_publish_as_result": False,
        },
    )
    (handoff / "playlist.m3u").write_text(
        "#EXTM3U\n#EXTINF:-1,001 vocal vrep\n../audio/vrep/sub/v001.flac\n#EXTINF:-1,002 minus mpr\n../audio/mpr/min/m001.flac\n"
        + playlist_extra,
        encoding="utf-8",
    )
    (handoff / "checklist.md").write_text(
        "# checklist\n- `audio/vrep/sub/v001.flac`\n- `audio/mpr/min/m001.flac`\n",
        encoding="utf-8",
    )
    return handoff / "handoff.json"


def test_validate_short_listen_review_handoff_accepts_valid_handoff(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path)

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "passed"
    assert result["failed_checks"] == []
    assert result["item_count"] == 2
    assert result["release_ready"] is False
    assert result["can_publish_as_result"] is False
    assert (tmp_path / "validation" / "validation.json").exists()
    assert (tmp_path / "validation" / "summary.md").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "validation").iterdir() if path.is_file())


def test_validate_short_listen_review_handoff_blocks_subset_pack(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path, subset_pack=True)

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "pack_schema_unexpected" in result["failed_checks"]
    assert result["item_count"] == 0


def test_validate_short_listen_review_handoff_blocks_repair_selection_pack(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path, repair_selection_pack=True)

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "pack_schema_unexpected" in result["failed_checks"]
    assert result["item_count"] == 0


def test_validate_short_listen_review_handoff_blocks_candidate_wave_pack(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path, candidate_wave_pack=True)

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "pack_schema_unexpected" in result["failed_checks"]
    assert result["item_count"] == 0


def test_validate_short_listen_review_handoff_blocks_release_ready(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path, release_ready=True)

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "handoff_release_ready_true" in result["failed_checks"]


def test_validate_short_listen_review_handoff_blocks_release_ready_item(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path)
    handoff = json.loads(manifest.read_text(encoding="utf-8"))
    handoff["items"][0]["release_ready"] = True
    _write_json(manifest, handoff)

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "handoff_item_1_release_ready_true" in result["failed_checks"]


def test_validate_short_listen_review_handoff_blocks_playlist_drift(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path, playlist_extra="../audio/not/in/manifest.flac\n")

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "playlist_entries_do_not_match_manifest_items" in result["failed_checks"]


def test_validate_short_listen_review_handoff_blocks_playlist_label_leakage(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path)
    playlist = tmp_path / "review_handoff" / "playlist.m3u"
    playlist.write_text(
        "#EXTM3U\n"
        "#EXTINF:-1,candidate_id both_c6e689414df7c622\n"
        "../audio/vrep/sub/v001.flac\n"
        "#EXTINF:-1,002 minus mpr\n"
        "../audio/mpr/min/m001.flac\n",
        encoding="utf-8",
    )

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "playlist_contains_raw_or_long_labels" in result["failed_checks"]


def test_validate_short_listen_review_handoff_blocks_missing_playlist_targets(tmp_path: Path) -> None:
    manifest = _write_handoff(tmp_path)
    (tmp_path / "audio" / "mpr" / "min" / "m001.flac").unlink()

    result = validate_short_listen_review_handoff(handoff_manifest_json=manifest, output_dir=tmp_path / "validation")

    assert result["validation_status"] == "failed"
    assert "playlist_target_missing:../audio/mpr/min/m001.flac" in result["failed_checks"]
