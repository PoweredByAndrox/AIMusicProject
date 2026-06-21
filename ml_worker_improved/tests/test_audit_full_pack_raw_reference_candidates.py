from __future__ import annotations

import json
from pathlib import Path

from tools.audit_full_pack_raw_reference_candidates import audit_full_pack_raw_reference_candidates


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


def _gap_plan(tmp_path: Path, stems: tuple[str, ...] = ("minus", "vocal", "bass")) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": f"g{index:03d}",
                "queue_item_id": f"q{index:03d}",
                "stem_type": stem,
                "missing_unique_track_count": 1,
                "required_reference_stem_count": 1,
                "release_ready": False,
                "can_publish_as_result": False,
            }
            for index, stem in enumerate(stems, start=1)
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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stem_pack_manifest(root: Path, *, source_mix: Path, stems: tuple[str, ...]) -> Path:
    track_dir = root / "audio" / "t001"
    stem_rows = []
    for stem in stems:
        stem_path = _audio(track_dir / f"{stem}.flac")
        stem_rows.append(
            {
                "schema": "stem_manifest_audio_pack_stem.v1",
                "artifact_relative_path": f"audio/t001/{stem}.flac",
                "listen_audio_path": str(stem_path),
                "stem_type": stem,
                "track_alias": "t001",
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    return _write_json(
        root / "pack.json",
        {
            "schema": "stem_manifest_audio_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "source_audio_path": str(source_mix),
            "stems": stem_rows,
            "failed_checks": [],
        },
    )


def _aggregate_6s_pack_manifest(root: Path, *, source_manifest: Path, stems: tuple[str, ...]) -> Path:
    track_dir = root / "audio" / "stm" / "t001"
    items = []
    for stem in stems:
        _audio(track_dir / f"{stem}.flac")
        items.append(
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "audio_path": f"audio/stm/t001/{stem}.flac",
                "source_alias": "stm",
                "source_manifest": str(source_manifest),
                "stem_type": stem,
                "track_alias": "t001",
                "semantic_quality_verified": False,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    return _write_json(
        root / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "item_count": len(items),
            "items": items,
            "source_manifests": [{"source_alias": "stm", "manifest_path": str(source_manifest)}],
            "failed_checks": [],
        },
    )


def test_raw_reference_candidate_audit_reports_needs_manifest_without_accepting_audio(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "pack-a"
    _audio(raw / "01_INPUT__full_mix.wav")
    _audio(raw / "My Track Acapella.wav")
    _audio(raw / "My Track Instrumental.wav")
    _audio(raw / "My Track Bass.wav")

    manifest = audit_full_pack_raw_reference_candidates(
        gap_plan_json=_gap_plan(tmp_path),
        candidate_roots=[tmp_path / "raw"],
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "full_pack_raw_reference_candidates_audit.v1"
    assert manifest["raw_candidate_status"] == "raw_reference_candidates_need_verified_manifest"
    assert manifest["raw_candidate_rows"] == 3
    assert manifest["accepted_reference_rows"] == 0
    assert manifest["eligible_reference_material_rows"] == 0
    assert manifest["raw_candidates_by_stem"] == {"bass": 1, "minus": 1, "vocal": 1}
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["metadata_only"] is True
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["audio_decoded"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False
    assert not (raw / "ground_truth_manifest.json").exists()

    rows = _read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert {row["stem_type"] for row in rows} == {"bass", "minus", "vocal"}
    assert all(row["triage_status"] == "needs_verified_ground_truth_manifest" for row in rows)
    assert all("ground_truth_manifest.json" in row["missing_acceptance_evidence"] for row in rows)
    assert all(row["delivery_allowed"] is False for row in rows)
    assert all(Path(row["source_mix_candidate_path"]).name == "01_INPUT__full_mix.wav" for row in rows)
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_raw_reference_candidate_audit_keeps_stem_only_audio_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "pack-b"
    _audio(raw / "only_acapella.wav")

    manifest = audit_full_pack_raw_reference_candidates(
        gap_plan_json=_gap_plan(tmp_path),
        candidate_roots=[tmp_path / "raw"],
        output_dir=tmp_path / "out",
    )

    assert manifest["raw_candidate_status"] == "blocked_no_raw_reference_candidates"
    assert manifest["raw_candidate_rows"] == 0
    assert manifest["rejected_directory_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False

    rejected = _read_jsonl(tmp_path / "out" / "reject.jsonl")
    assert rejected[0]["rejection_reasons"] == ["missing_source_mix_candidate"]
    assert not (raw / "ground_truth_manifest.json").exists()


def test_raw_reference_candidate_audit_resolves_6s_pack_source_mix_from_manifest(tmp_path: Path) -> None:
    source_mix = _audio(tmp_path / "source" / "source_mix.flac")
    source_manifest = _stem_pack_manifest(tmp_path / "lp" / "s001", source_mix=source_mix, stems=("guitar", "other"))
    aggregate = tmp_path / "full001"
    _aggregate_6s_pack_manifest(aggregate, source_manifest=source_manifest, stems=("guitar", "other"))

    manifest = audit_full_pack_raw_reference_candidates(
        gap_plan_json=_gap_plan(tmp_path, stems=("guitar", "other")),
        candidate_roots=[aggregate],
        output_dir=tmp_path / "out",
    )

    assert manifest["raw_candidate_status"] == "raw_reference_candidates_need_verified_manifest"
    assert manifest["raw_candidate_rows"] == 2
    assert manifest["rejected_directory_count"] == 0
    assert manifest["raw_candidates_by_stem"] == {"guitar": 1, "other": 1}
    assert manifest["delivery_allowed"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    rows = _read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert {row["stem_type"] for row in rows} == {"guitar", "other"}
    assert {Path(row["source_mix_candidate_path"]) for row in rows} == {source_mix}
    assert all(row["source_mix_resolution"] == "pack_manifest_source_audio_path" for row in rows)
    assert all(Path(row["source_mix_manifest_json"]) == aggregate / "pack.json" for row in rows)
    assert all(row["triage_status"] == "needs_verified_ground_truth_manifest" for row in rows)
    assert all(row["delivery_allowed"] is False for row in rows)


def test_raw_reference_candidate_audit_rejects_6s_pack_when_manifest_source_mix_missing(tmp_path: Path) -> None:
    missing_mix = tmp_path / "source" / "missing_source_mix.flac"
    source_manifest = _stem_pack_manifest(tmp_path / "lp" / "s001", source_mix=missing_mix, stems=("guitar",))
    aggregate = tmp_path / "full001"
    _aggregate_6s_pack_manifest(aggregate, source_manifest=source_manifest, stems=("guitar",))

    manifest = audit_full_pack_raw_reference_candidates(
        gap_plan_json=_gap_plan(tmp_path, stems=("guitar",)),
        candidate_roots=[aggregate],
        output_dir=tmp_path / "out",
    )

    assert manifest["raw_candidate_status"] == "blocked_no_raw_reference_candidates"
    assert manifest["raw_candidate_rows"] == 0
    assert manifest["rejected_directory_count"] == 1
    rejected = _read_jsonl(tmp_path / "out" / "reject.jsonl")
    assert rejected[0]["rejection_reasons"] == ["missing_source_mix_candidate"]


def test_raw_reference_candidate_audit_writes_periodic_scan_progress(tmp_path: Path) -> None:
    for index in range(1, 4):
        raw = tmp_path / "raw" / f"pack-{index}"
        _audio(raw / "01_INPUT__full_mix.wav")
        _audio(raw / "My Track Instrumental.wav")

    audit_full_pack_raw_reference_candidates(
        gap_plan_json=_gap_plan(tmp_path),
        candidate_roots=[tmp_path / "raw"],
        output_dir=tmp_path / "out",
        progress_every=1,
    )

    progress_rows = _read_jsonl(tmp_path / "out" / "progress.jsonl")
    scan_rows = [row for row in progress_rows if row["event"] == "scan_directory_progress"]
    assert [row["directories_seen"] for row in scan_rows] == [1, 2, 3]
    assert all(row["audio_directory_count"] == 3 for row in scan_rows)
    assert all(row["gpu_inference_skipped"] is True for row in scan_rows)
