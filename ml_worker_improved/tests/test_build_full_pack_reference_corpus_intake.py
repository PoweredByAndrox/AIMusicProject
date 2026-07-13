from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_full_pack_reference_corpus_intake import build_full_pack_reference_corpus_intake


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


def _gap_plan(tmp_path: Path, *, delivery_allowed: bool = False) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q002",
                "stem_type": "drums",
                "missing_unique_track_count": 2,
                "required_reference_stem_count": 2,
                "missing_training_job": False,
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g002",
                "queue_item_id": "q005",
                "stem_type": "guitar",
                "missing_unique_track_count": 3,
                "required_reference_stem_count": 3,
                "missing_training_job": True,
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
            "gap_count": 2,
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "failed_checks": [],
            "delivery_allowed": delivery_allowed,
            "release_ready": False,
            "training_started": False,
        },
    )


def _minus_gap_plan(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "minus_gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "gm01",
                "queue_item_id": "qm01",
                "stem_type": "minus",
                "missing_unique_track_count": 1,
                "required_reference_stem_count": 1,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "minus_gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "gap_count": 1,
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "failed_checks": [],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _existing_examples(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "existing.jsonl",
        [
            {
                "schema": "generic_finetuning_audio_training_example.v1",
                "id": "existing-drums",
                "track_id": "track-a",
                "stem_type": "drums",
                "source_audio_path": _audio(tmp_path / "existing" / "mix.flac"),
                "target_audio_path": _audio(tmp_path / "existing" / "drums.flac"),
                "permission_status": "project_owned",
                "trainable": True,
                "reference_verified": True,
            }
        ],
    )


def _ground_truth_bundle(
    root: Path,
    name: str,
    *,
    track_id: str,
    stems: list[str],
    source: str = "FL Studio split mixer tracks plus full mix supplied by user",
    verified: bool = True,
    rights_status: str = "user_owned",
    source_mix_1_to_1: bool = True,
) -> Path:
    bundle = root / name
    mix = _audio(bundle / "source_mix.flac")
    stem_payload = {}
    for stem in stems:
        stem_payload[stem] = {"path": f"stems/{stem}.flac", "sha256": f"sha-{track_id}-{stem}"}
        _audio(bundle / "stems" / f"{stem}.flac")
    return _write_json(
        bundle / "ground_truth_manifest.json",
        {
            "track_name": track_id,
            "ground_truth_contract": {"verified": verified, "source": source},
            "full_mix_reconstruction_contract": {
                "status": "accepted" if source_mix_1_to_1 else "rejected",
                "can_claim_source_mix_1_to_1": source_mix_1_to_1,
            },
            "mix": {"path": Path(mix).name},
            "stems": stem_payload,
            "report_alignment_contract": {"rights_status": rights_status},
        },
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minus_semantic_audit(
    manifest_path: Path,
    *,
    decision: str = "accepted",
    training_allowed: bool = True,
    source_sha256: str | None = None,
    target_sha256: str | None = None,
) -> Path:
    payload = _read_json(manifest_path)
    source_path = manifest_path.parent / payload["mix"]["path"]
    target_path = manifest_path.parent / payload["stems"]["minus"]["path"]
    return _write_json(
        manifest_path.parent / "sem.json",
        {
            "schema": "minus_reference_semantic_audit.v1",
            "decision": decision,
            "training_allowed": training_allowed,
            "delivery_allowed": False,
            "detector_count": 2,
            "detectors": [
                {"detector_id": "detector_a"},
                {"detector_id": "detector_b"},
            ],
            "evidence": {},
            "thresholds": {
                "min_gap_db": 6.0,
                "max_reference_vocal_share_db": -20.0,
            },
            "source_mix": {
                "path": str(source_path),
                "sha256": source_sha256 or _sha256(source_path),
            },
            "minus_reference": {
                "path": str(target_path),
                "sha256": target_sha256 or _sha256(target_path),
            },
        },
    )


def test_intake_accepts_only_new_verified_required_reference_stems(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _ground_truth_bundle(candidates, "accepted", track_id="track-b", stems=["drums", "guitar"])
    _ground_truth_bundle(candidates, "duplicate", track_id="track-a", stems=["drums"])
    _ground_truth_bundle(
        candidates,
        "synthetic",
        track_id="track-c",
        stems=["drums"],
        source="ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
    )

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_existing_examples(tmp_path),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "full_pack_reference_corpus_intake_manifest.v1"
    assert manifest["intake_status"] == "accepted_with_remaining_gaps"
    assert manifest["accepted_reference_rows"] == 2
    assert manifest["accepted_audio_training_examples"] == 2
    assert manifest["rejected_reference_rows"] == 2
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["remaining_reference_stem_count"] == 3

    rows = _read_jsonl(tmp_path / "out" / "rows.jsonl")
    accepted = [row for row in rows if row["intake_status"] == "accepted_for_corpus_import"]
    rejected = {row["track_id"]: row for row in rows if row["intake_status"] == "rejected"}
    assert {(row["track_id"], row["stem_type"]) for row in accepted} == {("track-b", "drums"), ("track-b", "guitar")}
    assert "duplicate_existing_track_stem" in rejected["track-a"]["rejection_reasons"]
    assert "synthetic_fixture_not_allowed" in rejected["track-c"]["rejection_reasons"]

    examples = _read_jsonl(tmp_path / "out" / "audio.jsonl")
    assert {row["schema"] for row in examples} == {"generic_finetuning_audio_training_example.v1"}
    assert {row["permission_status"] for row in examples} == {"project_owned"}
    assert all(row["trainable"] is True for row in examples)
    assert all(row["reference_verified"] is True for row in examples)
    assert all(row["source_gate"]["trainable_ground_truth"] is True for row in examples)
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_intake_rejects_rejected_full_mix_reconstruction(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _ground_truth_bundle(
        candidates,
        "rejected_reconstruction",
        track_id="track-b",
        stems=["drums"],
        source_mix_1_to_1=False,
    )

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_existing_examples(tmp_path),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["intake_status"] == "no_accepted_references"
    assert manifest["accepted_reference_rows"] == 0
    assert manifest["accepted_audio_training_examples"] == 0
    assert manifest["rejected_reference_rows"] == 1
    rows = _read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert rows[0]["intake_status"] == "rejected"
    assert "ground_truth_reconstruction_not_trainable" in rows[0]["rejection_reasons"]
    assert rows[0]["source_gate"]["trainable_ground_truth"] is False


def test_intake_accepts_minus_only_with_matching_semantic_audit(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    manifest_path = _ground_truth_bundle(candidates, "minus_ok", track_id="track-minus", stems=["minus"])
    _minus_semantic_audit(manifest_path)

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_minus_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_write_jsonl(tmp_path / "empty.jsonl", []),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["intake_status"] == "accepted_reference_corpus_complete"
    assert manifest["accepted_reference_rows"] == 1
    row = _read_jsonl(tmp_path / "out" / "rows.jsonl")[0]
    assert row["semantic_audit"]["status"] == "accepted"
    assert row["source_gate"]["trainable_ground_truth"] is True
    example = _read_jsonl(tmp_path / "out" / "audio.jsonl")[0]
    assert example["semantic_audit"]["decision"] == "accepted"


def test_intake_rejects_minus_when_semantic_audit_is_missing(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _ground_truth_bundle(candidates, "minus_missing", track_id="track-minus", stems=["minus"])

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_minus_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_write_jsonl(tmp_path / "empty.jsonl", []),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["accepted_reference_rows"] == 0
    assert manifest["rejection_reason_counts"]["minus_reference_semantic_audit_missing"] == 1


def test_intake_rejects_minus_when_semantic_roles_are_reversed(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    manifest_path = _ground_truth_bundle(candidates, "minus_rev", track_id="track-minus", stems=["minus"])
    _minus_semantic_audit(manifest_path, decision="roles_reversed", training_allowed=False)

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_minus_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_write_jsonl(tmp_path / "empty.jsonl", []),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["accepted_reference_rows"] == 0
    assert manifest["rejection_reason_counts"]["minus_reference_semantic_audit_roles_reversed"] == 1
    row = _read_jsonl(tmp_path / "out" / "rows.jsonl")[0]
    assert row["source_gate"]["trainable_ground_truth"] is False


def test_intake_rejects_minus_when_semantic_audit_hashes_do_not_match(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    manifest_path = _ground_truth_bundle(candidates, "minus_hash", track_id="track-minus", stems=["minus"])
    _minus_semantic_audit(manifest_path, target_sha256="not-the-target-hash")

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_minus_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_write_jsonl(tmp_path / "empty.jsonl", []),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["accepted_reference_rows"] == 0
    assert manifest["rejection_reason_counts"]["minus_reference_semantic_audit_target_sha256_mismatch"] == 1


def test_intake_rejects_minus_when_detector_count_is_not_backed_by_unique_rows(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    manifest_path = _ground_truth_bundle(candidates, "minus_det", track_id="track-minus", stems=["minus"])
    audit_path = _minus_semantic_audit(manifest_path)
    audit = _read_json(audit_path)
    audit["detectors"] = [
        {"detector_id": "detector_a"},
        {"detector_id": "detector_a"},
    ]
    _write_json(audit_path, audit)

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_minus_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_write_jsonl(tmp_path / "empty.jsonl", []),
        candidate_roots=[candidates],
        output_dir=tmp_path / "out",
    )

    assert manifest["accepted_reference_rows"] == 0
    reasons = manifest["rejection_reason_counts"]
    assert reasons["minus_reference_semantic_audit_detector_count_mismatch"] == 1
    assert reasons["minus_reference_semantic_audit_insufficient_detectors"] == 1


def test_intake_rejects_debug_test_artifact_ground_truth_bundles(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    _ground_truth_bundle(
        lab_root / "dbg_loop" / "test_reference_bundle_0" / "clean",
        "g001",
        track_id="track-debug",
        stems=["guitar"],
    )

    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_existing_examples(tmp_path),
        candidate_roots=[lab_root],
        output_dir=tmp_path / "out",
    )

    assert manifest["intake_status"] == "no_accepted_references"
    assert manifest["accepted_reference_rows"] == 0
    assert manifest["rejected_reference_rows"] == 1
    assert manifest["rejection_reason_counts"]["candidate_root_non_production_artifact"] == 1
    rows = _read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert rows[0]["track_id"] == "track-debug"
    assert rows[0]["stem_type"] == "guitar"
    assert "candidate_root_non_production_artifact" in rows[0]["rejection_reasons"]
    assert _read_jsonl(tmp_path / "out" / "audio.jsonl") == []


def test_intake_blocks_unsafe_gap_plan(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=_gap_plan(tmp_path, delivery_allowed=True),
        existing_audio_training_examples_jsonl=_existing_examples(tmp_path),
        candidate_roots=[tmp_path / "missing"],
        output_dir=tmp_path / "out",
    )

    assert manifest["intake_status"] == "blocked"
    assert manifest["accepted_reference_rows"] == 0
    assert "gap_plan_delivery_allowed_true" in manifest["failed_checks"]
