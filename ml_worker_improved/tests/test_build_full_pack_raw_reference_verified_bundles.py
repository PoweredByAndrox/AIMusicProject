from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_full_pack_raw_reference_verified_bundles import (
    build_full_pack_raw_reference_verified_bundles,
)
from tools.build_full_pack_reference_corpus_intake import build_full_pack_reference_corpus_intake


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


def _flstudio_export(root: Path, *, track_name: str = "track-a") -> dict[str, Path]:
    export_dir = root / "golden_audio" / track_name
    export_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    frame_count = sample_rate // 2
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    bass = 0.18 * np.sin(2 * np.pi * 90 * time)
    piano = 0.07 * np.sin(2 * np.pi * 440 * time)
    source_mix = bass + piano
    source_mix_path = export_dir / "00_source_mix.flac"
    bass_path = export_dir / "01_bass_reference_teacher.flac"
    piano_path = export_dir / "02_piano_reference_teacher.flac"
    sf.write(source_mix_path, np.column_stack([source_mix, source_mix]), sample_rate, subtype="PCM_24")
    sf.write(bass_path, np.column_stack([bass, bass]), sample_rate, subtype="PCM_24")
    sf.write(piano_path, np.column_stack([piano, piano]), sample_rate, subtype="PCM_24")
    return {
        "export_dir": export_dir,
        "source_mix": source_mix_path,
        "bass": bass_path,
        "piano": piano_path,
    }


def _flstudio_export_with_mismatched_stem(root: Path, *, track_name: str = "track-a") -> dict[str, Path]:
    export_dir = root / "golden_audio" / track_name
    export_dir.mkdir(parents=True, exist_ok=True)
    time = np.arange(4000, dtype=np.float32) / 8000.0
    bass = 0.18 * np.sin(2 * np.pi * 90 * time)
    source_mix_path = export_dir / "00_source_mix.flac"
    bass_path = export_dir / "01_bass_reference_teacher.flac"
    sf.write(source_mix_path, np.column_stack([bass, bass]), 8000, subtype="PCM_24")
    sf.write(bass_path, np.column_stack([bass, bass]), 16000, subtype="PCM_24")
    return {
        "export_dir": export_dir,
        "source_mix": source_mix_path,
        "bass": bass_path,
        "piano": bass_path,
    }


def _flstudio_export_with_rejected_reconstruction(root: Path, *, track_name: str = "track-a") -> dict[str, Path]:
    export_dir = root / "golden_audio" / track_name
    export_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    frame_count = sample_rate // 2
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    bass = 0.18 * np.sin(2 * np.pi * 90 * time)
    source_mix_path = export_dir / "00_source_mix.flac"
    bass_path = export_dir / "01_bass_reference_teacher.flac"
    sf.write(source_mix_path, np.column_stack([bass * 0.5, bass * 0.5]), sample_rate, subtype="PCM_24")
    sf.write(bass_path, np.column_stack([bass, bass]), sample_rate, subtype="PCM_24")
    return {
        "export_dir": export_dir,
        "source_mix": source_mix_path,
        "bass": bass_path,
        "piano": bass_path,
    }


def _flstudio_export_with_mixed_scope_warning(root: Path, *, track_name: str = "track-a") -> dict[str, Path]:
    export_dir = root / "golden_audio" / track_name
    export_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    frame_count = sample_rate // 2
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    bass = 0.18 * np.sin(2 * np.pi * 90 * time)
    unrelated = 0.03 * np.sin(2 * np.pi * 330 * time)
    source_mix_path = export_dir / "00_source_mix.flac"
    bass_path = export_dir / "01_bass_reference_teacher.flac"
    unrelated_path = export_dir / "02_unrelated_render.flac"
    sf.write(source_mix_path, np.column_stack([bass, bass]), sample_rate, subtype="PCM_24")
    sf.write(bass_path, np.column_stack([bass, bass]), sample_rate, subtype="PCM_24")
    sf.write(unrelated_path, np.column_stack([unrelated, unrelated]), sample_rate, subtype="PCM_24")
    return {
        "export_dir": export_dir,
        "source_mix": source_mix_path,
        "bass": bass_path,
        "piano": bass_path,
    }


def _raw_worklist(tmp_path: Path, export: dict[str, Path]) -> Path:
    items_path = tmp_path / "raw_work" / "items.jsonl"
    rows = [
        {
            "schema": "full_pack_raw_reference_bundle_work_item.v1",
            "work_item_status": "needs_verified_ground_truth_manifest",
            "bundle_alias": "r001",
            "bundle_dir": str(tmp_path / "raw_work" / "refs" / "r001"),
            "template_json": str(tmp_path / "raw_work" / "refs" / "r001" / "tpl.json"),
            "stem_type": "bass",
            "track_id_hint": "track-a",
            "candidate_directory": str(export["export_dir"]),
            "source_mix_candidate_path": str(export["source_mix"]),
            "target_stem_candidate_path": str(export["bass"]),
            "required_actions": [
                "verify_source_provenance",
                "verify_trainable_rights",
                "run_reference_waveform_audit",
                "copy_or_render_candidate_audio_after_verification",
                "write_verified_ground_truth_manifest",
                "rerun_full_pack_reference_bundle_preflight",
            ],
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "training_started": False,
        }
    ]
    _write_jsonl(items_path, rows)
    return _write_json(
        tmp_path / "raw_work" / "work.json",
        {
            "schema": "full_pack_raw_reference_bundle_worklist.v1",
            "manifest_type": "full_pack_raw_reference_bundle_worklist",
            "worklist_status": "raw_reference_bundle_worklist_ready",
            "failed_checks": [],
            "work_item_count": len(rows),
            "outputs": {"items_jsonl": str(items_path)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "safety": {
                "metadata_only": True,
                "audio_read": False,
                "audio_copied": False,
                "ground_truth_manifest_written": False,
            },
        },
    )


def _slot_aware_raw_worklist_with_missing_stem(tmp_path: Path, export: dict[str, Path]) -> Path:
    items_path = tmp_path / "raw_work" / "items.jsonl"
    rows = [
        {
            "schema": "full_pack_raw_reference_bundle_work_item.v1",
            "work_item_status": "needs_verified_ground_truth_manifest",
            "bundle_alias": "r001",
            "bundle_dir": str(tmp_path / "raw_work" / "refs" / "r001"),
            "template_json": str(tmp_path / "raw_work" / "refs" / "r001" / "tpl.json"),
            "stem_type": "bass",
            "source_collection_id": "r001",
            "gap_id": "g001",
            "queue_item_id": "q001",
            "raw_reference_candidate_matched": True,
            "blockers": [],
            "track_id_hint": "track-a",
            "candidate_directory": str(export["export_dir"]),
            "source_mix_candidate_path": str(export["source_mix"]),
            "target_stem_candidate_path": str(export["bass"]),
            "required_actions": ["write_verified_ground_truth_manifest"],
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "training_started": False,
        },
        {
            "schema": "full_pack_raw_reference_bundle_work_item.v1",
            "work_item_status": "blocked_missing_raw_reference_candidate",
            "bundle_alias": "r002",
            "bundle_dir": str(tmp_path / "raw_work" / "refs" / "r002"),
            "template_json": str(tmp_path / "raw_work" / "refs" / "r002" / "tpl.json"),
            "stem_type": "guitar",
            "source_collection_id": "r002",
            "gap_id": "g005",
            "queue_item_id": "q005",
            "raw_reference_candidate_matched": False,
            "blockers": ["missing_raw_reference_candidate"],
            "track_id_hint": "",
            "candidate_directory": "",
            "source_mix_candidate_path": "",
            "target_stem_candidate_path": "",
            "required_actions": ["collect_matching_raw_reference_candidate"],
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "training_started": False,
        },
    ]
    _write_jsonl(items_path, rows)
    return _write_json(
        tmp_path / "raw_work" / "work.json",
        {
            "schema": "full_pack_raw_reference_bundle_worklist.v1",
            "manifest_type": "full_pack_raw_reference_bundle_worklist",
            "worklist_status": "raw_reference_bundle_worklist_ready",
            "failed_checks": [],
            "slot_aware": True,
            "formal_slot_count": 2,
            "matched_raw_candidate_count": 1,
            "missing_raw_candidate_slot_count": 1,
            "work_item_count": len(rows),
            "outputs": {"items_jsonl": str(items_path)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "safety": {
                "metadata_only": True,
                "audio_read": False,
                "audio_copied": False,
                "ground_truth_manifest_written": False,
            },
        },
    )


def _gap_plan(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q001",
                "stem_type": "bass",
                "missing_unique_track_count": 1,
                "required_reference_stem_count": 1,
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "failed_checks": [],
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def test_raw_reference_verified_bundles_feed_existing_intake(tmp_path: Path) -> None:
    export = _flstudio_export(tmp_path)
    raw_worklist = _raw_worklist(tmp_path, export)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "golden_audio"],
        rights_status="project_owned",
        require_full_mix_reconstruction=True,
    )

    assert manifest["schema"] == "full_pack_raw_reference_verified_bundles_manifest.v1"
    assert manifest["bundle_status"] == "verified_reference_bundles_ready"
    assert manifest["verified_bundle_count"] == 1
    assert manifest["blocked_bundle_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is True
    assert manifest["safety"]["audio_copied"] is True
    assert manifest["safety"]["audio_decoded"] is True
    assert manifest["safety"]["ground_truth_manifest_written"] is True

    ready = _read_jsonl(tmp_path / "verified" / "ready.jsonl")
    assert [row["bundle_alias"] for row in ready] == ["b001"]
    bundle_root = Path(ready[0]["candidate_root"])
    assert bundle_root.name == "b001"
    assert (bundle_root / "ground_truth_manifest.json").is_file()
    assert (bundle_root / "source_mix.flac").is_file()
    assert (bundle_root / "stems" / "bass.flac").is_file()
    assert all(len(path.name) <= 32 for path in (tmp_path / "verified").iterdir() if path.is_file())

    intake = build_full_pack_reference_corpus_intake(
        gap_plan_json=_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_write_jsonl(tmp_path / "existing.jsonl", []),
        candidate_roots=[bundle_root],
        output_dir=tmp_path / "intake",
    )

    assert intake["intake_status"] == "accepted_reference_corpus_complete"
    assert intake["accepted_reference_rows"] == 1
    audio_rows = _read_jsonl(Path(intake["outputs"]["accepted_audio_training_examples_jsonl"]))
    assert audio_rows[0]["stem_type"] == "bass"
    assert audio_rows[0]["permission_status"] == "project_owned"
    assert audio_rows[0]["reference_verified"] is True


def test_raw_reference_verified_bundles_reject_source_outside_allowed_root(tmp_path: Path) -> None:
    export = _flstudio_export(tmp_path / "outside")
    raw_worklist = _raw_worklist(tmp_path, export)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "allowed"],
        rights_status="project_owned",
    )

    assert manifest["bundle_status"] == "blocked_waiting_for_verified_raw_reference_bundles"
    assert manifest["verified_bundle_count"] == 0
    assert manifest["blocked_bundle_count"] == 1
    assert manifest["safety"]["ground_truth_manifest_written"] is False
    blocked = _read_jsonl(tmp_path / "verified" / "blocked.jsonl")
    assert "source_not_under_allowed_root" in blocked[0]["blockers"]
    assert not (tmp_path / "verified" / "refs" / "b001" / "ground_truth_manifest.json").exists()


def test_raw_reference_verified_bundles_blocks_manifest_resolved_listen_pack_sources(tmp_path: Path) -> None:
    export = _flstudio_export(tmp_path)
    raw_worklist = _raw_worklist(tmp_path, export)
    items_path = tmp_path / "raw_work" / "items.jsonl"
    rows = _read_jsonl(items_path)
    rows[0]["source_mix_resolution"] = "pack_manifest_source_audio_path"
    rows[0]["source_mix_manifest_json"] = str(tmp_path / "listen_pack" / "pack.json")
    _write_jsonl(items_path, rows)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "golden_audio"],
        rights_status="project_owned",
        require_full_mix_reconstruction=True,
    )

    assert manifest["bundle_status"] == "blocked_waiting_for_verified_raw_reference_bundles"
    assert manifest["verified_bundle_count"] == 0
    assert manifest["blocked_bundle_count"] == 1
    assert manifest["candidate_roots"] == []
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_decoded"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False
    blocked = _read_jsonl(tmp_path / "verified" / "blocked.jsonl")
    assert blocked[0]["source_mix_resolution"] == "pack_manifest_source_audio_path"
    assert Path(blocked[0]["source_mix_manifest_json"]).name == "pack.json"
    assert "source_mix_from_listen_pack_manifest_requires_clean_reference_bundle" in blocked[0]["blockers"]
    assert not (tmp_path / "verified" / "refs" / "b001" / "ground_truth_manifest.json").exists()


def test_raw_reference_verified_bundles_preserves_worklist_provenance_blockers(tmp_path: Path) -> None:
    export = _flstudio_export(tmp_path)
    raw_worklist = _raw_worklist(tmp_path, export)
    items_path = tmp_path / "raw_work" / "items.jsonl"
    rows = _read_jsonl(items_path)
    rows[0]["work_item_status"] = "blocked_raw_reference_provenance"
    rows[0]["source_mix_resolution"] = "pack_manifest_source_audio_path"
    rows[0]["source_mix_manifest_json"] = str(tmp_path / "listen_pack" / "pack.json")
    rows[0]["blockers"] = ["source_mix_from_listen_pack_manifest_requires_clean_reference_bundle"]
    _write_jsonl(items_path, rows)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "golden_audio"],
        rights_status="project_owned",
        require_full_mix_reconstruction=True,
    )

    assert manifest["bundle_status"] == "blocked_waiting_for_verified_raw_reference_bundles"
    assert manifest["verified_bundle_count"] == 0
    assert manifest["blocked_bundle_count"] == 1
    assert manifest["source_work_item_count"] == 1
    assert manifest["buildable_work_item_count"] == 0
    assert manifest["blocked_source_work_item_count"] == 1
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_decoded"] is False
    blocked = _read_jsonl(tmp_path / "verified" / "blocked.jsonl")
    assert blocked[0]["source_work_item_aliases"] == ["r001"]
    assert blocked[0]["bundle_build_attempted"] is False
    assert blocked[0]["source_mix_resolution"] == "pack_manifest_source_audio_path"
    assert blocked[0]["source_mix_manifest_json"].endswith("pack.json")
    assert blocked[0]["blockers"] == ["source_mix_from_listen_pack_manifest_requires_clean_reference_bundle"]


def test_raw_reference_verified_bundles_records_build_failure_detail(tmp_path: Path) -> None:
    export = _flstudio_export_with_mismatched_stem(tmp_path)
    raw_worklist = _raw_worklist(tmp_path, export)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "golden_audio"],
        rights_status="project_owned",
    )

    assert manifest["verified_bundle_count"] == 0
    blocked = _read_jsonl(tmp_path / "verified" / "blocked.jsonl")
    assert "bundle_build_failed:ValueError" in blocked[0]["blockers"]
    assert "Sample-rate mismatch" in blocked[0]["failure_detail"]
    assert len(blocked[0]["failure_detail"]) <= 240


def test_raw_reference_verified_bundles_hides_rejected_ground_truth_manifest(tmp_path: Path) -> None:
    export = _flstudio_export_with_rejected_reconstruction(tmp_path)
    raw_worklist = _raw_worklist(tmp_path, export)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "golden_audio"],
        rights_status="project_owned",
        require_full_mix_reconstruction=True,
    )

    assert manifest["verified_bundle_count"] == 0
    blocked = _read_jsonl(tmp_path / "verified" / "blocked.jsonl")
    assert "full_mix_reconstruction_not_accepted" in blocked[0]["blockers"]
    diag_root = Path(blocked[0]["diagnostic_bundle_root"])
    assert not (diag_root / "ground_truth_manifest.json").exists()
    assert (diag_root / "gt_rej.json").is_file()
    assert blocked[0]["ground_truth_manifest_json"] == ""
    assert Path(blocked[0]["diagnostic_manifest_json"]).name == "gt_rej.json"


def test_raw_reference_verified_bundles_diagnoses_mixed_scope_without_accepting(tmp_path: Path) -> None:
    export = _flstudio_export_with_mixed_scope_warning(tmp_path)
    raw_worklist = _raw_worklist(tmp_path, export)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "golden_audio"],
        rights_status="project_owned",
        require_full_mix_reconstruction=True,
    )

    assert manifest["verified_bundle_count"] == 0
    assert manifest["blocked_bundle_count"] == 1
    assert manifest["candidate_roots"] == []
    assert manifest["safety"]["ground_truth_manifest_written"] is False
    blocked = _read_jsonl(tmp_path / "verified" / "blocked.jsonl")
    assert "input_scope_not_clean" in blocked[0]["blockers"]
    assert "full_mix_reconstruction_not_accepted" in blocked[0]["blockers"]
    assert blocked[0]["ground_truth_manifest_json"] == ""
    assert Path(blocked[0]["diagnostic_manifest_json"]).name == "gt_rej.json"
    diag = _read_json(Path(blocked[0]["diagnostic_manifest_json"]))
    assert diag["input_scope_contract"]["status"] == "accepted_with_warnings"
    assert diag["input_scope_contract"]["allow_mixed_stem_dir"] is True


def test_raw_reference_verified_bundles_preserves_missing_raw_handoff_slot(tmp_path: Path) -> None:
    export = _flstudio_export(tmp_path)
    raw_worklist = _slot_aware_raw_worklist_with_missing_stem(tmp_path, export)

    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=raw_worklist,
        output_dir=tmp_path / "verified",
        allowed_source_roots=[tmp_path / "golden_audio"],
        rights_status="project_owned",
    )

    assert manifest["verified_bundle_count"] == 1
    assert manifest["blocked_bundle_count"] == 1
    assert manifest["blocked_by_stem"] == {"guitar": 1}
    assert manifest["ready_by_stem"] == {"bass": 1}
    blocked = _read_jsonl(tmp_path / "verified" / "blocked.jsonl")
    assert blocked[0]["bundle_alias"] == "r002"
    assert blocked[0]["source_work_item_aliases"] == ["r002"]
    assert blocked[0]["requested_stems"] == ["guitar"]
    assert blocked[0]["blockers"] == ["missing_raw_reference_candidate"]
    assert blocked[0]["candidate_root"] == ""
    assert blocked[0]["ground_truth_manifest_json"] == ""
