from __future__ import annotations

import json
from pathlib import Path

from tools.audit_full_pack_reference_material_discovery import audit_full_pack_reference_material_discovery


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _audio(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return str(path)


def _gap_plan(tmp_path: Path, *, stem: str = "minus", delivery_allowed: bool = False) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q001",
                "stem_type": stem,
                "missing_unique_track_count": 1,
                "required_reference_stem_count": 1,
                "missing_training_job": False,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "gap_count": 1,
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "failed_checks": [],
            "delivery_allowed": delivery_allowed,
            "release_ready": False,
            "training_started": False,
        },
    )


def _existing_examples(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    return _write_jsonl(tmp_path / "existing.jsonl", rows or [])


def _existing_row(tmp_path: Path, *, track_id: str, stem: str) -> dict:
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": f"{track_id}-{stem}",
        "track_id": track_id,
        "stem_type": stem,
        "source_audio_path": _audio(tmp_path / "existing" / track_id / "mix.flac"),
        "target_audio_path": _audio(tmp_path / "existing" / track_id / f"{stem}.flac"),
        "permission_status": "project_owned",
        "trainable": True,
        "reference_verified": True,
    }


def _ground_truth_bundle(
    root: Path,
    name: str,
    *,
    track_id: str,
    stems: list[str],
    source: str = "FL Studio split mixer tracks plus full mix supplied by user",
    verified: bool = True,
    rights_status: str = "user_owned",
    write_audio: bool = True,
    extra_payload: dict | None = None,
) -> Path:
    bundle = root / name
    if write_audio:
        _audio(bundle / "source_mix.flac")
    stem_payload = {}
    for stem in stems:
        stem_payload[stem] = {"path": f"stems/{stem}.flac", "sha256": f"sha-{track_id}-{stem}"}
        if write_audio:
            _audio(bundle / "stems" / f"{stem}.flac")
    payload = {
        "track_name": track_id,
        "ground_truth_contract": {"verified": verified, "source": source},
        "mix": {"path": "source_mix.flac"},
        "stems": stem_payload,
        "report_alignment_contract": {"rights_status": rights_status},
    }
    if extra_payload:
        payload.update(extra_payload)
    return _write_json(bundle / "ground_truth_manifest.json", payload)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_reference_material_discovery_reports_eligible_verified_bundle(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _ground_truth_bundle(candidates, "r001", track_id="track-new", stems=["minus"])

    manifest = audit_full_pack_reference_material_discovery(
        gap_plan_json=_gap_plan(tmp_path, stem="minus"),
        existing_audio_training_examples_jsonl=_existing_examples(tmp_path),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "full_pack_reference_material_discovery_audit.v1"
    assert manifest["discovery_status"] == "eligible_reference_material_found"
    assert manifest["candidate_manifest_count"] == 1
    assert manifest["eligible_candidate_rows"] == 1
    assert manifest["rejected_candidate_rows"] == 0
    assert manifest["eligible_by_stem"] == {"minus": 1}
    assert manifest["remaining_after_eligible_by_stem"] == {"minus": 0}
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["metadata_only"] is True
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False

    eligible = _read_jsonl(tmp_path / "out" / "eligible.jsonl")
    assert [(row["track_id"], row["stem_type"]) for row in eligible] == [("track-new", "minus")]
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_reference_material_discovery_rejects_duplicate_existing_track_stem(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _ground_truth_bundle(candidates, "r001", track_id="track-a", stems=["minus"])

    manifest = audit_full_pack_reference_material_discovery(
        gap_plan_json=_gap_plan(tmp_path, stem="minus"),
        existing_audio_training_examples_jsonl=_existing_examples(
            tmp_path,
            [_existing_row(tmp_path, track_id="track-a", stem="minus")],
        ),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["discovery_status"] == "blocked_no_eligible_reference_material"
    assert manifest["eligible_candidate_rows"] == 0
    assert manifest["rejected_candidate_rows"] == 1
    assert manifest["rejection_reason_counts"] == {"duplicate_existing_track_stem": 1}


def test_reference_material_discovery_rejects_fixture_and_missing_rights_sources(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _ground_truth_bundle(
        candidates,
        "fixture",
        track_id="track-fixture",
        stems=["drums"],
        source="ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
        rights_status="",
        write_audio=False,
    )

    manifest = audit_full_pack_reference_material_discovery(
        gap_plan_json=_gap_plan(tmp_path, stem="drums"),
        existing_audio_training_examples_jsonl=_existing_examples(tmp_path),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["discovery_status"] == "blocked_no_eligible_reference_material"
    assert manifest["eligible_candidate_rows"] == 0
    assert manifest["rejected_candidate_rows"] == 1
    assert manifest["rejection_reason_counts"]["source_domain_not_allowed"] == 1
    assert manifest["rejection_reason_counts"]["synthetic_fixture_not_allowed"] == 1
    assert manifest["rejection_reason_counts"]["missing_permission_status"] == 1
    assert manifest["rejection_reason_counts"]["source_audio_not_found"] == 1
    assert manifest["rejection_reason_counts"]["target_audio_not_found"] == 1


def test_reference_material_discovery_rejects_unsafe_manifest_flags(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _ground_truth_bundle(
        candidates,
        "unsafe",
        track_id="track-unsafe",
        stems=["minus"],
        extra_payload={
            "ground_truth_contract": {
                "verified": True,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
                "delivery_allowed": True,
            },
        },
    )

    manifest = audit_full_pack_reference_material_discovery(
        gap_plan_json=_gap_plan(tmp_path, stem="minus"),
        existing_audio_training_examples_jsonl=_existing_examples(tmp_path),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["discovery_status"] == "blocked_no_eligible_reference_material"
    assert manifest["eligible_candidate_rows"] == 0
    assert manifest["rejected_candidate_rows"] == 1
    assert manifest["rejection_reason_counts"] == {"ground_truth_manifest_ground_truth_contract_delivery_allowed_true": 1}
