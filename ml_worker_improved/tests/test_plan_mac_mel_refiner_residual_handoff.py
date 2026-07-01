from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path


def _planner():
    module = importlib.import_module("tools.plan_mac_mel_refiner_residual_handoff")
    assert module is not None
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stage(root: Path, *, remaining: int = 46, delivery_allowed: bool = False) -> Path:
    _write_json(
        root / "manifest.json",
        {
            "schema": "mac_mel_refiner_fulltrack_stage.v1",
            "status": "diagnostic_blocked_pitch_guard",
            "after_pitch_verdict": "fail",
            "delivery_allowed": delivery_allowed,
            "listen_pack_created": False,
            "presentation_status": "fulltrack_diagnostic_only",
            "replay_manifest": str(root / "replay" / "manifest.json"),
            "failure_taxonomy_summary": {
                "schema": "mac_mel_refiner_fulltrack_taxonomy_summary.v1",
                "remaining_failure_count": remaining,
                "before_region_count": 69,
                "after_region_count": remaining,
                "coverage_status": "fulltrack_remaining_failures",
                "after_failure_taxonomy": {
                    "by_fail_mode": {"standard_score": 14, "strong_projection_stft": 32},
                    "by_probe": {"0.0": 39, "-9.0": 4, "-5.0": 2, "-12.0": 1},
                    "regions_capped": True,
                    "selected_region_count": 16,
                    "total_fail_regions": remaining,
                },
            },
        },
    )
    return root / "manifest.json"


def _validation(
    path: Path,
    *,
    stage_manifest: Path | None,
    status: str = "validation_blocked_holdout_failure",
    remaining: int = 46,
    delivery_allowed: bool = False,
) -> Path:
    holdout_results: list[dict[str, object]] = []
    if stage_manifest is not None:
        holdout_results.append(
            {
                "holdout_id": "h001",
                "guard_dir": str(path.parent / "guard"),
                "stage_manifest": str(stage_manifest),
                "stage_status": "diagnostic_blocked_pitch_guard",
                "after_pitch_verdict": "fail",
                "remaining_failure_count": remaining,
                "coverage_status": "fulltrack_remaining_failures",
                "delivery_allowed": False,
                "listen_pack_created": False,
                "passed": False,
            }
        )
    _write_json(
        path,
        {
            "schema": "mac_mel_refiner_top1_holdout_validation.v1",
            "status": status,
            "failed_checks": ["holdout_pitch_guard_failed"] if status == "validation_blocked_holdout_failure" else [],
            "source_benchmark_manifest": str(path.parent / "bench" / "manifest.json"),
            "checkpoint_root": str(path.parent / "ckpt"),
            "selected_checkpoint_id": "g1.5",
            "checkpoint": str(path.parent / "ckpt" / "g1.5" / "model.pt"),
            "delivery_allowed": delivery_allowed,
            "listen_pack_created": False,
            "presentation_status": "top1_holdout_validation_diagnostic_only",
            "holdout_count": len(holdout_results),
            "passed_holdout_count": 0,
            "remaining_failure_count": remaining if holdout_results else 0,
            "holdout_results": holdout_results,
        },
    )
    return path


def test_residual_handoff_plan_builds_ready_item_from_failed_holdout() -> None:
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage_manifest = _stage(root / "val" / "h001")
        validation = _validation(root / "val" / "manifest.json", stage_manifest=stage_manifest)
        residual_root = root / "resid"

        manifest = planner.plan_residual_handoff(
            validation_manifest_path=validation,
            output_dir=root / "plan",
            residual_output_root=residual_root,
            target_gain=2.5,
            epochs=260,
            max_regions=16,
            require_mps=True,
        )

        assert manifest["status"] == "residual_handoff_ready"
        assert manifest["delivery_allowed"] is False
        assert manifest["listen_pack_created"] is False
        assert manifest["presentation_status"] == "residual_handoff_diagnostic_only"
        assert manifest["item_count"] == 6
        assert manifest["region_batch_count"] == 6
        assert manifest["remaining_failure_count"] == 46
        assert manifest["source_remaining_failure_count"] == 46
        assert manifest["taxonomy_coverage"]["source_total_fail_regions"] == 46
        assert manifest["taxonomy_coverage"]["source_selected_region_count"] == 16
        assert manifest["taxonomy_coverage"]["regions_capped"] is True
        item = manifest["items"][0]
        assert item["work_id"] == "r001"
        assert item["holdout_id"] == "h001"
        assert item["source_failed_stage_dir"] == str(stage_manifest.parent)
        assert item["residual_output_dir"] == str(residual_root / "r001")
        assert item["remaining_failure_count"] == 39
        assert item["source_remaining_failure_count"] == 46
        assert item["taxonomy_batch"] == {
            "axis": "probe",
            "value": "0.0",
            "expected_region_count": 39,
            "batch_id": "b001",
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_limit": 16,
            "chunk_count": 3,
            "chunk_expected_region_count": 16,
        }
        assert item["training_region_count"] == 16
        assert item["ready"] is True
        assert item["command"]["tool"] == "tools/train_mac_refiner_from_fulltrack_residuals.py"
        assert "--failed-stage-dir" in item["command"]["args"]
        assert str(stage_manifest.parent) in item["command"]["args"]
        assert "--region-batch-json" in item["command"]["args"]
        assert "--require-mps" in item["command"]["args"]

        second = manifest["items"][1]
        assert second["work_id"] == "r002"
        assert second["taxonomy_batch"] == {
            "axis": "probe",
            "value": "0.0",
            "expected_region_count": 39,
            "batch_id": "b001",
            "chunk_index": 1,
            "chunk_start": 16,
            "chunk_limit": 16,
            "chunk_count": 3,
            "chunk_expected_region_count": 16,
        }
        assert second["training_region_count"] == 16

        third = manifest["items"][2]
        assert third["work_id"] == "r003"
        assert third["taxonomy_batch"] == {
            "axis": "probe",
            "value": "0.0",
            "expected_region_count": 39,
            "batch_id": "b001",
            "chunk_index": 2,
            "chunk_start": 32,
            "chunk_limit": 16,
            "chunk_count": 3,
            "chunk_expected_region_count": 7,
        }
        assert third["training_region_count"] == 7

        fourth = manifest["items"][3]
        assert fourth["work_id"] == "r004"
        assert fourth["taxonomy_batch"] == {
            "axis": "probe",
            "value": "-9.0",
            "expected_region_count": 4,
            "batch_id": "b002",
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_limit": 16,
            "chunk_count": 1,
            "chunk_expected_region_count": 4,
        }


def test_residual_handoff_blocks_when_validation_has_no_failed_holdouts() -> None:
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        validation = _validation(
            root / "val" / "manifest.json",
            stage_manifest=None,
            status="validation_blocked_holdout_failure",
            remaining=0,
        )

        manifest = planner.plan_residual_handoff(
            validation_manifest_path=validation,
            output_dir=root / "plan",
            residual_output_root=root / "resid",
        )

        assert manifest["status"] == "residual_handoff_blocked_no_failed_holdouts"
        assert manifest["failed_checks"] == ["no_failed_holdouts"]
        assert manifest["items"] == []
        assert manifest["delivery_allowed"] is False


def test_residual_handoff_blocks_non_holdout_failure_status_even_with_failed_rows() -> None:
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage_manifest = _stage(root / "val" / "h001")
        validation = _validation(
            root / "val" / "manifest.json",
            stage_manifest=stage_manifest,
            status="validation_blocked_benchmark_not_promoted",
            remaining=46,
        )

        manifest = planner.plan_residual_handoff(
            validation_manifest_path=validation,
            output_dir=root / "plan",
            residual_output_root=root / "resid",
        )

        assert manifest["status"] == "residual_handoff_blocked_validation_status"
        assert manifest["failed_checks"] == ["validation_status_not_holdout_failure"]
        assert manifest["items"] == []


def test_residual_handoff_blocks_missing_stage_manifest() -> None:
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        missing_stage = root / "val" / "h001" / "manifest.json"
        validation = _validation(root / "val" / "manifest.json", stage_manifest=missing_stage)

        manifest = planner.plan_residual_handoff(
            validation_manifest_path=validation,
            output_dir=root / "plan",
            residual_output_root=root / "resid",
        )

        assert manifest["status"] == "residual_handoff_blocked_invalid_failed_stage"
        assert manifest["failed_checks"] == ["failed_stage_manifest_missing"]
        assert manifest["invalid_items"][0]["holdout_id"] == "h001"
        assert manifest["item_count"] == 0


def test_residual_handoff_resolves_copied_stage_from_holdout_alias() -> None:
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        local_stage = _stage(root / "copy" / "val" / "h001")
        producer_stage = Path("/Users/andrejviluha/aimusic_local_artifacts/mac_api/it6h001/val/h001/manifest.json")
        validation = _validation(root / "copy" / "val" / "manifest.json", stage_manifest=producer_stage)

        manifest = planner.plan_residual_handoff(
            validation_manifest_path=validation,
            output_dir=root / "plan",
            residual_output_root=root / "resid",
        )

        assert manifest["status"] == "residual_handoff_ready"
        assert manifest["items"][0]["stage_manifest"] == str(local_stage)
        assert manifest["items"][0]["source_failed_stage_dir"] == str(local_stage.parent)


def test_residual_handoff_falls_back_to_single_item_without_taxonomy_batches() -> None:
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage_manifest = _stage(root / "val" / "h001")
        payload = json.loads(stage_manifest.read_text(encoding="utf-8"))
        payload["failure_taxonomy_summary"]["after_failure_taxonomy"]["by_probe"] = {}
        payload["failure_taxonomy_summary"]["after_failure_taxonomy"]["by_fail_mode"] = {}
        stage_manifest.write_text(json.dumps(payload), encoding="utf-8")
        validation = _validation(root / "val" / "manifest.json", stage_manifest=stage_manifest)

        manifest = planner.plan_residual_handoff(
            validation_manifest_path=validation,
            output_dir=root / "plan",
            residual_output_root=root / "resid",
        )

        assert manifest["status"] == "residual_handoff_ready"
        assert manifest["item_count"] == 3
        assert manifest["region_batch_count"] == 3
        assert manifest["items"][0]["taxonomy_batch"]["axis"] == "all"
        assert manifest["items"][0]["taxonomy_batch"]["chunk_expected_region_count"] == 16
        assert manifest["items"][1]["taxonomy_batch"]["chunk_expected_region_count"] == 16
        assert manifest["items"][2]["taxonomy_batch"]["chunk_expected_region_count"] == 14
        assert "--region-batch-json" not in manifest["items"][0]["command"]["args"]


def test_residual_handoff_blocks_delivery_contract_violations() -> None:
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage_manifest = _stage(root / "val" / "h001", delivery_allowed=True)
        validation = _validation(root / "val" / "manifest.json", stage_manifest=stage_manifest)

        manifest = planner.plan_residual_handoff(
            validation_manifest_path=validation,
            output_dir=root / "plan",
            residual_output_root=root / "resid",
        )

        assert manifest["status"] == "residual_handoff_blocked_invalid_failed_stage"
        assert manifest["failed_checks"] == ["failed_stage_delivery_contract_invalid"]
        assert manifest["delivery_allowed"] is False
