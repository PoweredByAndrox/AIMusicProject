#!/usr/bin/env python3
"""Validate TOP-1 Mac MelBand refiner promotion on full-track holdouts.

This is a diagnostic gate between a bounded benchmark promotion and any claim
that a checkpoint generalizes. It never creates listen/result delivery.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def run_fulltrack_diagnostic_stage(**kwargs: Any) -> dict[str, Any]:
    from tools.run_mac_mel_refiner_fulltrack_stage import run_fulltrack_diagnostic_stage as _run

    return _run(**kwargs)


def _safe_checkpoint_alias(value: Any) -> str:
    alias = str(value or "").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not alias or len(alias) > 16 or any(char not in allowed for char in alias):
        raise ValueError(f"unsafe selected checkpoint id: {value!r}")
    return alias


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _base_manifest(*, benchmark_manifest_path: Path, checkpoint_root: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "mac_mel_refiner_top1_holdout_validation.v1",
        "source_benchmark_manifest": str(benchmark_manifest_path),
        "checkpoint_root": str(checkpoint_root),
        "output_dir": str(output_dir),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "top1_holdout_validation_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "no_threshold_relaxation": True,
            "requires_bounded_benchmark_promotion": True,
            "requires_fulltrack_holdout_clear": True,
            "blocks_training_overlap_with_benchmark_guard": True,
        },
    }


def _write_manifest(
    *,
    output_dir: Path,
    progress_path: Path,
    benchmark_manifest_path: Path,
    checkpoint_root: Path,
    status: str,
    failed_checks: list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        **_base_manifest(benchmark_manifest_path=benchmark_manifest_path, checkpoint_root=checkpoint_root, output_dir=output_dir),
        "status": status,
        "failed_checks": list(failed_checks),
        **dict(extra or {}),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", run_status=status, failed_checks=failed_checks)
    return manifest


def _has_benchmark_failure_taxonomy(benchmark_manifest: Mapping[str, Any]) -> bool:
    taxonomy = benchmark_manifest.get("failure_taxonomy_summary")
    return isinstance(taxonomy, Mapping) and str(taxonomy.get("schema") or "") == "mac_mel_refiner_benchmark_taxonomy_summary.v1"


def _holdout_result(stage_manifest: Mapping[str, Any], *, guard_dir: Path, index: int) -> dict[str, Any]:
    taxonomy = dict(stage_manifest.get("failure_taxonomy_summary") or {})
    remaining = int(taxonomy.get("remaining_failure_count") or taxonomy.get("after_region_count") or 0)
    status = str(stage_manifest.get("status") or "")
    verdict = str(stage_manifest.get("after_pitch_verdict") or "")
    delivery_allowed = bool(stage_manifest.get("delivery_allowed"))
    listen_pack_created = bool(stage_manifest.get("listen_pack_created"))
    passed = status == "diagnostic_passed" and verdict == "pass" and remaining == 0 and not delivery_allowed and not listen_pack_created
    return {
        "holdout_id": f"h{index + 1:03d}",
        "guard_dir": str(guard_dir),
        "stage_manifest": str(stage_manifest.get("manifest_path") or ""),
        "stage_status": status,
        "after_pitch_verdict": verdict,
        "remaining_failure_count": remaining,
        "coverage_status": taxonomy.get("coverage_status"),
        "delivery_allowed": delivery_allowed,
        "listen_pack_created": listen_pack_created,
        "passed": bool(passed),
    }


def _status_from_results(results: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if any(bool(row.get("delivery_allowed")) or bool(row.get("listen_pack_created")) for row in results):
        return "validation_blocked_delivery_contract", ["holdout_delivery_contract_invalid"]
    if any(not bool(row.get("passed")) for row in results):
        return "validation_blocked_holdout_failure", ["holdout_pitch_guard_failed"]
    return "validation_passed", []


def run_top1_holdout_validation(
    *,
    benchmark_manifest_path: Path,
    checkpoint_root: Path,
    holdout_guard_dirs: Sequence[Path],
    output_dir: Path,
    require_mps: bool = True,
    max_regions: int = 16,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start", require_mps=bool(require_mps), holdout_count=len(holdout_guard_dirs))
    benchmark_manifest = _load_json(Path(benchmark_manifest_path))
    if benchmark_manifest.get("outcome") != "promote_to_next_diagnostic_stage":
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            checkpoint_root=checkpoint_root,
            status="validation_blocked_benchmark_not_promoted",
            failed_checks=["benchmark_not_promoted"],
            extra={"benchmark_outcome": benchmark_manifest.get("outcome"), "holdout_count": 0, "holdout_results": []},
        )
    if bool(benchmark_manifest.get("delivery_allowed")) or bool(benchmark_manifest.get("listen_pack_created")):
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            checkpoint_root=checkpoint_root,
            status="validation_blocked_benchmark_contract",
            failed_checks=["benchmark_delivery_contract_invalid"],
            extra={"holdout_count": 0, "holdout_results": []},
        )
    if not _has_benchmark_failure_taxonomy(benchmark_manifest):
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            checkpoint_root=checkpoint_root,
            status="validation_blocked_benchmark_missing_failure_taxonomy",
            failed_checks=["benchmark_missing_failure_taxonomy_summary"],
            extra={"holdout_count": 0, "holdout_results": []},
        )
    holdout_dirs = [Path(path) for path in holdout_guard_dirs]
    if not holdout_dirs:
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            checkpoint_root=checkpoint_root,
            status="validation_blocked_no_holdouts",
            failed_checks=["no_holdout_guards"],
            extra={"holdout_count": 0, "holdout_results": []},
        )

    selected_checkpoint_id = _safe_checkpoint_alias(benchmark_manifest.get("selected_checkpoint_id"))
    checkpoint_path = Path(checkpoint_root) / selected_checkpoint_id / "model.pt"
    if not checkpoint_path.is_file():
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            checkpoint_root=checkpoint_root,
            status="validation_blocked_missing_checkpoint",
            failed_checks=["missing_selected_checkpoint"],
            extra={"selected_checkpoint_id": selected_checkpoint_id, "checkpoint": str(checkpoint_path), "holdout_count": len(holdout_dirs), "holdout_results": []},
        )

    benchmark_guard = benchmark_manifest.get("source_guard_dir")
    overlap_evidence: list[dict[str, Any]] = []
    if benchmark_guard:
        benchmark_guard_path = Path(str(benchmark_guard))
        for index, guard_dir in enumerate(holdout_dirs):
            if _same_path(guard_dir, benchmark_guard_path):
                overlap_evidence.append({"holdout_id": f"h{index + 1:03d}", "guard_dir": str(guard_dir), "benchmark_source_guard_dir": str(benchmark_guard_path)})
    if overlap_evidence:
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            checkpoint_root=checkpoint_root,
            status="validation_blocked_training_overlap",
            failed_checks=["training_overlap"],
            extra={
                "selected_checkpoint_id": selected_checkpoint_id,
                "checkpoint": str(checkpoint_path),
                "holdout_count": len(holdout_dirs),
                "holdout_results": [],
                "training_overlap_evidence": overlap_evidence,
            },
        )

    holdout_results: list[dict[str, Any]] = []
    for index, guard_dir in enumerate(holdout_dirs):
        holdout_id = f"h{index + 1:03d}"
        stage_output_dir = output_dir / holdout_id
        _append_progress(progress_path, "holdout", "start", holdout_id=holdout_id, guard_dir=str(guard_dir))
        stage_manifest = run_fulltrack_diagnostic_stage(
            benchmark_manifest_path=Path(benchmark_manifest_path),
            mel_guard_dir=guard_dir,
            checkpoint_root=Path(checkpoint_root),
            output_dir=stage_output_dir,
            require_mps=bool(require_mps),
            max_regions=int(max_regions),
        )
        stage_manifest.setdefault("manifest_path", str(stage_output_dir / "manifest.json"))
        result = _holdout_result(stage_manifest, guard_dir=guard_dir, index=index)
        holdout_results.append(result)
        _append_progress(
            progress_path,
            "holdout",
            "end",
            holdout_id=holdout_id,
            stage_status=result["stage_status"],
            remaining_failure_count=result["remaining_failure_count"],
            passed=result["passed"],
        )

    status, failed_checks = _status_from_results(holdout_results)
    return _write_manifest(
        output_dir=output_dir,
        progress_path=progress_path,
        benchmark_manifest_path=benchmark_manifest_path,
        checkpoint_root=checkpoint_root,
        status=status,
        failed_checks=failed_checks,
        extra={
            "selected_checkpoint_id": selected_checkpoint_id,
            "checkpoint": str(checkpoint_path),
            "benchmark_outcome": benchmark_manifest.get("outcome"),
            "holdout_count": len(holdout_results),
            "holdout_results": holdout_results,
            "passed_holdout_count": sum(1 for row in holdout_results if bool(row.get("passed"))),
            "remaining_failure_count": sum(int(row.get("remaining_failure_count") or 0) for row in holdout_results),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--holdout-guard-dir", type=Path, action="append", dest="holdout_guard_dirs", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_top1_holdout_validation(
        benchmark_manifest_path=args.benchmark_manifest,
        checkpoint_root=args.checkpoint_root,
        holdout_guard_dirs=list(args.holdout_guard_dirs or []),
        output_dir=args.out,
        require_mps=bool(args.require_mps),
        max_regions=int(args.max_regions),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
