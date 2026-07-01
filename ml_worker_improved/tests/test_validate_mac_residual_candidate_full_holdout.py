from __future__ import annotations

import json
from pathlib import Path

from tools import validate_mac_residual_candidate_full_holdout as validator


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(tmp_path: Path, *, source_guard: Path | None = None) -> Path:
    guard_manifest = (source_guard or (tmp_path / "train_guard")) / "manifest.json"
    return _write_json(
        tmp_path / "it7dl" / "r001" / "manifest.json",
        {
            "schema": "mac_residual_damage_limited_repair_route.v1",
            "status": "damage_limited_repair_passed",
            "source_training_status": "trained_repair_passed_pitch_guard",
            "source_guard_manifest": str(guard_manifest),
            "after_score": {"verdict": "pass", "regions": []},
            "damage_audit": {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "pass",
                "failed_region_count": 0,
                "delivery_allowed": False,
                "listen_pack_created": False,
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_limited_repair_diagnostic_only",
            "full_holdout_resolution_status": "not_evaluated",
        },
    )


def _linked_holdout(path: Path, *, candidate: Path, guard_dir: Path, status: str = "validation_passed") -> Path:
    return _write_json(
        path,
        {
            "schema": "mac_mel_refiner_top1_holdout_validation.v1",
            "status": status,
            "source_candidate_manifest": str(candidate),
            "holdout_results": [
                {
                    "holdout_id": "h001",
                    "guard_dir": str(guard_dir),
                    "passed": status == "validation_passed",
                    "remaining_failure_count": 0 if status == "validation_passed" else 3,
                    "delivery_allowed": False,
                    "listen_pack_created": False,
                }
            ],
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "top1_holdout_validation_diagnostic_only",
        },
    )


def test_residual_candidate_full_holdout_blocks_missing_evidence(tmp_path: Path) -> None:
    manifest = validator.run_residual_candidate_full_holdout_validation(
        candidate_manifest_path=_candidate(tmp_path),
        holdout_manifest_paths=[],
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "validation_blocked_no_holdout_evidence"
    assert manifest["full_holdout_resolution_status"] == "failed"
    assert manifest["failed_checks"] == ["no_holdout_evidence"]
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    progress = (tmp_path / "out" / "progress.jsonl").read_text(encoding="utf-8")
    assert '"gpu_inference_skipped": true' in progress


def test_residual_candidate_full_holdout_blocks_unlinked_evidence(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    holdout = _write_json(
        tmp_path / "holdout" / "manifest.json",
        {
            "schema": "mac_mel_refiner_top1_holdout_validation.v1",
            "status": "validation_passed",
            "holdout_results": [{"holdout_id": "h001", "guard_dir": str(tmp_path / "holdout_guard"), "passed": True}],
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )

    manifest = validator.run_residual_candidate_full_holdout_validation(
        candidate_manifest_path=candidate,
        holdout_manifest_paths=[holdout],
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "validation_blocked_unlinked_holdout_evidence"
    assert manifest["failed_checks"] == ["holdout_evidence_unlinked_to_candidate"]
    assert manifest["holdout_results"][0]["linked_to_candidate"] is False


def test_residual_candidate_full_holdout_blocks_training_overlap(tmp_path: Path) -> None:
    train_guard = tmp_path / "train_guard"
    candidate = _candidate(tmp_path, source_guard=train_guard)
    holdout = _linked_holdout(tmp_path / "holdout" / "manifest.json", candidate=candidate, guard_dir=train_guard)

    manifest = validator.run_residual_candidate_full_holdout_validation(
        candidate_manifest_path=candidate,
        holdout_manifest_paths=[holdout],
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "validation_blocked_training_overlap"
    assert manifest["failed_checks"] == ["training_overlap"]
    assert manifest["training_overlap_evidence"][0]["candidate_source_guard_manifest"] == str(train_guard / "manifest.json")
    assert manifest["delivery_allowed"] is False


def test_residual_candidate_full_holdout_passes_linked_independent_evidence_without_delivery(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, source_guard=tmp_path / "train_guard")
    holdout = _linked_holdout(tmp_path / "holdout" / "manifest.json", candidate=candidate, guard_dir=tmp_path / "holdout_guard")

    manifest = validator.run_residual_candidate_full_holdout_validation(
        candidate_manifest_path=candidate,
        holdout_manifest_paths=[holdout],
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "validation_passed"
    assert manifest["full_holdout_resolution_status"] == "passed"
    assert manifest["failed_checks"] == []
    assert manifest["passed_holdout_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
