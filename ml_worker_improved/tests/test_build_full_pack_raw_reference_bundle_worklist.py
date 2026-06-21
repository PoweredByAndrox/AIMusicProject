from __future__ import annotations

import json
from pathlib import Path

from tools.audit_full_pack_raw_reference_candidates import audit_full_pack_raw_reference_candidates
from tools.build_full_pack_raw_reference_bundle_worklist import build_full_pack_raw_reference_bundle_worklist


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return path


def _gap_plan(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q001",
                "stem_type": "minus",
                "missing_unique_track_count": 1,
                "required_reference_stem_count": 1,
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g002",
                "queue_item_id": "q002",
                "stem_type": "bass",
                "missing_unique_track_count": 1,
                "required_reference_stem_count": 1,
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
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "failed_checks": [],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _raw_candidates(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "pack-a"
    _audio(raw / "00_source_mix.flac")
    _audio(raw / "01_bass_reference_teacher.flac")
    _audio(raw / "02_companion_instrumental.flac")
    manifest = audit_full_pack_raw_reference_candidates(
        gap_plan_json=_gap_plan(tmp_path),
        candidate_roots=[tmp_path / "raw"],
        output_dir=tmp_path / "raw_out",
    )
    return Path(manifest["outputs"]["manifest_json"])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _handoff(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "hand" / "items.jsonl",
        [
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
                "required_files": ["source_mix.flac", "stems/minus.flac", "ground_truth_manifest.json"],
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
                "required_files": ["source_mix.flac", "stems/guitar.flac", "ground_truth_manifest.json"],
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "hand" / "handoff.json",
        {
            "schema": "full_pack_reference_bundle_handoff_manifest.v1",
            "manifest_type": "full_pack_reference_bundle_handoff",
            "handoff_status": "ready_for_reference_bundle_collection",
            "failed_checks": [],
            "handoff_item_count": 2,
            "outputs": {
                "items_jsonl": str(rows),
                "manifest_json": str(tmp_path / "hand" / "handoff.json"),
            },
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def test_raw_reference_bundle_worklist_writes_short_templates_without_accepting_rows(tmp_path: Path) -> None:
    raw_json = _raw_candidates(tmp_path)

    manifest = build_full_pack_raw_reference_bundle_worklist(
        raw_candidates_json=raw_json,
        output_dir=tmp_path / "work",
    )

    assert manifest["schema"] == "full_pack_raw_reference_bundle_worklist.v1"
    assert manifest["worklist_status"] == "raw_reference_bundle_worklist_ready"
    assert manifest["work_item_count"] == 2
    assert manifest["work_items_by_stem"] == {"bass": 1, "minus": 1}
    assert manifest["accepted_reference_rows"] == 0
    assert manifest["eligible_reference_material_rows"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["metadata_only"] is True
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    rows = _read_jsonl(tmp_path / "work" / "items.jsonl")
    assert [row["bundle_alias"] for row in rows] == ["r001", "r002"]
    assert all(row["work_item_status"] == "needs_verified_ground_truth_manifest" for row in rows)
    assert all(Path(row["template_json"]).name == "tpl.json" for row in rows)
    assert all(Path(row["template_json"]).is_file() for row in rows)
    assert all("write_verified_ground_truth_manifest" in row["required_actions"] for row in rows)
    assert all(row["delivery_allowed"] is False for row in rows)
    assert all(row["source_mix_resolution"] == "same_directory_audio" for row in rows)
    assert all(row["source_mix_manifest_json"] == "" for row in rows)

    first_template = _read_json(Path(rows[0]["template_json"]))
    assert first_template["ground_truth_contract"]["verified"] == "<must_be_true_after_verification>"
    assert first_template["mix"]["path"] == "source_mix.flac"
    assert "candidate_source_mix_path" in first_template["source_candidates"]
    assert first_template["source_candidates"]["candidate_source_mix_resolution"] == "same_directory_audio"
    assert first_template["source_candidates"]["candidate_source_mix_manifest_json"] == ""
    assert not (Path(rows[0]["bundle_dir"]) / "ground_truth_manifest.json").exists()
    assert not (Path(rows[0]["bundle_dir"]) / "source_mix.flac").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "work").iterdir() if path.is_file())


def test_raw_reference_bundle_worklist_blocks_unsafe_raw_manifest(tmp_path: Path) -> None:
    raw_json = _raw_candidates(tmp_path)
    raw_payload = _read_json(raw_json)
    raw_payload["delivery_allowed"] = True
    unsafe = _write_json(tmp_path / "unsafe_raw.json", raw_payload)

    manifest = build_full_pack_raw_reference_bundle_worklist(
        raw_candidates_json=unsafe,
        output_dir=tmp_path / "work",
    )

    assert manifest["worklist_status"] == "blocked"
    assert manifest["work_item_count"] == 0
    assert "raw_candidates_delivery_allowed_true" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "work" / "items.jsonl") == []
    assert not (tmp_path / "work" / "refs").exists()


def test_raw_reference_worklist_blocks_manifest_resolved_listen_pack_sources(tmp_path: Path) -> None:
    raw_json = _raw_candidates(tmp_path)
    raw_payload = _read_json(raw_json)
    rows_path = Path(raw_payload["outputs"]["rows_jsonl"])
    raw_rows = _read_jsonl(rows_path)
    raw_rows[0]["source_mix_resolution"] = "pack_manifest_source_audio_path"
    raw_rows[0]["source_mix_manifest_json"] = str(tmp_path / "pack.json")
    _write_jsonl(rows_path, raw_rows)

    manifest = build_full_pack_raw_reference_bundle_worklist(
        raw_candidates_json=raw_json,
        output_dir=tmp_path / "work",
    )

    assert manifest["worklist_status"] == "raw_reference_bundle_worklist_ready"
    assert manifest["work_items_by_status"] == {
        "blocked_raw_reference_provenance": 1,
        "needs_verified_ground_truth_manifest": 1,
    }
    rows = _read_jsonl(tmp_path / "work" / "items.jsonl")
    blocked = [row for row in rows if row["work_item_status"] == "blocked_raw_reference_provenance"]
    assert len(blocked) == 1
    assert blocked[0]["raw_reference_candidate_matched"] is True
    assert blocked[0]["source_mix_resolution"] == "pack_manifest_source_audio_path"
    assert "source_mix_from_listen_pack_manifest_requires_clean_reference_bundle" in blocked[0]["blockers"]
    assert blocked[0]["delivery_allowed"] is False
    assert blocked[0]["release_ready"] is False


def test_slot_aware_worklist_prefers_unblocked_raw_candidate(tmp_path: Path) -> None:
    raw_json = _raw_candidates(tmp_path)
    raw_payload = _read_json(raw_json)
    rows_path = Path(raw_payload["outputs"]["rows_jsonl"])
    raw_rows = _read_jsonl(rows_path)
    minus_row = next(row for row in raw_rows if row["stem_type"] == "minus")
    blocked_minus = {
        **minus_row,
        "id": f"{minus_row['id']}_blocked",
        "source_mix_resolution": "pack_manifest_source_audio_path",
        "source_mix_manifest_json": str(tmp_path / "pack.json"),
    }
    raw_rows = [blocked_minus, *raw_rows]
    _write_jsonl(rows_path, raw_rows)

    manifest = build_full_pack_raw_reference_bundle_worklist(
        raw_candidates_json=raw_json,
        handoff_json=_handoff(tmp_path),
        output_dir=tmp_path / "work",
    )

    assert manifest["matched_raw_candidate_count"] == 1
    assert manifest["missing_raw_candidate_slot_count"] == 1
    assert manifest["unmatched_raw_candidate_count"] == 2
    rows = _read_jsonl(tmp_path / "work" / "items.jsonl")
    assert rows[0]["stem_type"] == "minus"
    assert rows[0]["work_item_status"] == "needs_verified_ground_truth_manifest"
    assert rows[0]["source_mix_resolution"] == "same_directory_audio"
    assert rows[0]["blockers"] == []


def test_raw_reference_worklist_aligns_to_handoff_slots_and_marks_missing_required_stems(tmp_path: Path) -> None:
    raw_json = _raw_candidates(tmp_path)
    handoff_json = _handoff(tmp_path)

    manifest = build_full_pack_raw_reference_bundle_worklist(
        raw_candidates_json=raw_json,
        handoff_json=handoff_json,
        output_dir=tmp_path / "work",
    )

    assert manifest["worklist_status"] == "raw_reference_bundle_worklist_ready"
    assert manifest["slot_aware"] is True
    assert manifest["formal_slot_count"] == 2
    assert manifest["matched_raw_candidate_count"] == 1
    assert manifest["missing_raw_candidate_slot_count"] == 1
    assert manifest["work_items_by_stem"] == {"guitar": 1, "minus": 1}
    assert manifest["work_items_by_status"] == {
        "blocked_missing_raw_reference_candidate": 1,
        "needs_verified_ground_truth_manifest": 1,
    }
    assert manifest["accepted_reference_rows"] == 0
    assert manifest["eligible_reference_material_rows"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    rows = _read_jsonl(tmp_path / "work" / "items.jsonl")
    assert [row["bundle_alias"] for row in rows] == ["r001", "r002"]
    assert [row["source_collection_id"] for row in rows] == ["r001", "r002"]
    assert [row["gap_id"] for row in rows] == ["g001", "g005"]
    assert [row["queue_item_id"] for row in rows] == ["q001", "q005"]
    assert rows[0]["stem_type"] == "minus"
    assert rows[0]["raw_reference_candidate_matched"] is True
    assert rows[0]["work_item_status"] == "needs_verified_ground_truth_manifest"
    assert rows[0]["bundle_dir"].endswith("refs/r001")
    assert rows[1]["stem_type"] == "guitar"
    assert rows[1]["raw_reference_candidate_matched"] is False
    assert rows[1]["work_item_status"] == "blocked_missing_raw_reference_candidate"
    assert rows[1]["blockers"] == ["missing_raw_reference_candidate"]
    assert rows[1]["candidate_directory"] == ""
    assert rows[1]["required_files"] == ["source_mix.flac", "stems/guitar.flac", "ground_truth_manifest.json"]
    assert all(Path(row["template_json"]).name == "tpl.json" for row in rows)
