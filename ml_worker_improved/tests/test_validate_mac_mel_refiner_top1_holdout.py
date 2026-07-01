from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path


def _validator():
    module = importlib.import_module("tools.validate_mac_mel_refiner_top1_holdout")
    assert module is not None
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pt")
    return path


def _guard(path: Path) -> Path:
    _write_json(path / "manifest.json", {"delivery_allowed": False, "listen_pack_created": False})
    return path


def _benchmark(path: Path, *, outcome: str = "promote_to_next_diagnostic_stage", selected: str | None = "g1.5", source_guard: str = "") -> Path:
    _write_json(
        path,
        {
            "schema": "mac_mel_refiner_benchmark.v1",
            "outcome": outcome,
            "selected_checkpoint_id": selected,
            "source_guard_dir": source_guard,
            "delivery_allowed": False,
            "listen_pack_created": False,
            "failure_taxonomy_summary": {
                "schema": "mac_mel_refiner_benchmark_taxonomy_summary.v1",
                "delivery_allowed": False,
                "total_before_regions": 4,
                "total_after_regions": 0,
            },
        },
    )
    return path


def _stage_manifest(*, guard: Path, output_dir: Path, status: str, verdict: str, remaining: int) -> dict[str, object]:
    return {
        "schema": "mac_mel_refiner_fulltrack_stage.v1",
        "status": status,
        "source_guard_dir": str(guard),
        "after_pitch_verdict": verdict,
        "failed_checks": [] if status == "diagnostic_passed" else ["pitch_guard_failed"],
        "failure_taxonomy_summary": {
            "schema": "mac_mel_refiner_fulltrack_taxonomy_summary.v1",
            "remaining_failure_count": remaining,
            "before_region_count": 4,
            "after_region_count": remaining,
            "coverage_status": "fulltrack_clear" if remaining == 0 else "fulltrack_remaining_failures",
        },
        "manifest_path": str(output_dir / "manifest.json"),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "fulltrack_diagnostic_only",
    }


def test_top1_holdout_blocks_benchmark_not_promoted() -> None:
    validator = _validator()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark = _benchmark(root / "bench.json", outcome="reject_as_overfit", selected=None)
        checkpoint_root = root / "ckpt"

        manifest = validator.run_top1_holdout_validation(
            benchmark_manifest_path=benchmark,
            checkpoint_root=checkpoint_root,
            holdout_guard_dirs=[_guard(root / "holdout")],
            output_dir=root / "out",
            require_mps=False,
        )

        assert manifest["status"] == "validation_blocked_benchmark_not_promoted"
        assert manifest["failed_checks"] == ["benchmark_not_promoted"]
        assert manifest["holdout_count"] == 0
        assert manifest["delivery_allowed"] is False


def test_top1_holdout_blocks_failed_holdout() -> None:
    validator = _validator()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark = _benchmark(root / "bench.json", source_guard=str(root / "training_guard"))
        checkpoint_root = root / "ckpt"
        checkpoint = _checkpoint(checkpoint_root / "g1.5" / "model.pt")
        holdout = _guard(root / "holdout")

        def fake_stage(**kwargs):
            return _stage_manifest(
                guard=Path(kwargs["mel_guard_dir"]),
                output_dir=Path(kwargs["output_dir"]),
                status="diagnostic_blocked_pitch_guard",
                verdict="fail",
                remaining=46,
            )

        original = validator.run_fulltrack_diagnostic_stage
        try:
            validator.run_fulltrack_diagnostic_stage = fake_stage
            manifest = validator.run_top1_holdout_validation(
                benchmark_manifest_path=benchmark,
                checkpoint_root=checkpoint_root,
                holdout_guard_dirs=[holdout],
                output_dir=root / "out",
                require_mps=True,
                max_regions=16,
            )
        finally:
            validator.run_fulltrack_diagnostic_stage = original

        assert manifest["selected_checkpoint_id"] == "g1.5"
        assert manifest["checkpoint"] == str(checkpoint)
        assert manifest["status"] == "validation_blocked_holdout_failure"
        assert manifest["failed_checks"] == ["holdout_pitch_guard_failed"]
        assert manifest["holdout_results"][0]["remaining_failure_count"] == 46
        assert manifest["delivery_allowed"] is False


def test_top1_holdout_passes_all_holdouts_without_delivery() -> None:
    validator = _validator()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark = _benchmark(root / "bench.json", source_guard=str(root / "training_guard"))
        checkpoint_root = root / "ckpt"
        _checkpoint(checkpoint_root / "g1.5" / "model.pt")
        holdouts = [_guard(root / "h001"), _guard(root / "h002")]
        calls: list[Path] = []

        def fake_stage(**kwargs):
            guard = Path(kwargs["mel_guard_dir"])
            calls.append(guard)
            return _stage_manifest(
                guard=guard,
                output_dir=Path(kwargs["output_dir"]),
                status="diagnostic_passed",
                verdict="pass",
                remaining=0,
            )

        original = validator.run_fulltrack_diagnostic_stage
        try:
            validator.run_fulltrack_diagnostic_stage = fake_stage
            manifest = validator.run_top1_holdout_validation(
                benchmark_manifest_path=benchmark,
                checkpoint_root=checkpoint_root,
                holdout_guard_dirs=holdouts,
                output_dir=root / "out",
                require_mps=True,
            )
        finally:
            validator.run_fulltrack_diagnostic_stage = original

        assert calls == holdouts
        assert manifest["status"] == "validation_passed"
        assert manifest["holdout_count"] == 2
        assert manifest["failed_checks"] == []
        assert manifest["delivery_allowed"] is False
        assert manifest["listen_pack_created"] is False


def test_top1_holdout_blocks_training_overlap_with_benchmark_guard() -> None:
    validator = _validator()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        training_guard = _guard(root / "training_guard")
        benchmark = _benchmark(root / "bench.json", source_guard=str(training_guard))
        checkpoint_root = root / "ckpt"
        _checkpoint(checkpoint_root / "g1.5" / "model.pt")

        manifest = validator.run_top1_holdout_validation(
            benchmark_manifest_path=benchmark,
            checkpoint_root=checkpoint_root,
            holdout_guard_dirs=[training_guard],
            output_dir=root / "out",
            require_mps=False,
        )

        assert manifest["status"] == "validation_blocked_training_overlap"
        assert manifest["failed_checks"] == ["training_overlap"]
        assert manifest["holdout_count"] == 1
        assert manifest["delivery_allowed"] is False
