#!/usr/bin/env python3
"""Plan residual Mac/MPS refiner training from failed TOP-1 holdouts.

The planner does not train, repair, or publish audio. It converts a
diagnostic-only TOP-1 holdout validation failure into short, repeatable
work items for ``train_mac_refiner_from_fulltrack_residuals.py``.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _base_manifest(*, validation_manifest_path: Path, output_dir: Path, residual_output_root: Path) -> dict[str, Any]:
    return {
        "schema": "mac_mel_refiner_residual_handoff_plan.v1",
        "source_validation_manifest": str(validation_manifest_path),
        "output_dir": str(output_dir),
        "residual_output_root": str(residual_output_root),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "residual_handoff_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "no_threshold_relaxation": True,
            "source_is_top1_holdout_validation": True,
            "output_items_are_training_commands_only": True,
        },
    }


def _write_manifest(
    *,
    output_dir: Path,
    progress_path: Path,
    validation_manifest_path: Path,
    residual_output_root: Path,
    status: str,
    failed_checks: Sequence[str],
    items: Sequence[Mapping[str, Any]] | None = None,
    invalid_items: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ready_items = [dict(item) for item in items or []]
    bad_items = [dict(item) for item in invalid_items or []]
    manifest = {
        **_base_manifest(validation_manifest_path=validation_manifest_path, output_dir=output_dir, residual_output_root=residual_output_root),
        "status": status,
        "failed_checks": list(failed_checks),
        "item_count": len(ready_items),
        "invalid_item_count": len(bad_items),
        "items": ready_items,
        "invalid_items": bad_items,
        **dict(extra or {}),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_items_jsonl(output_dir / "items.jsonl", ready_items)
    _append_progress(progress_path, "manifest", "end", run_status=status, item_count=len(ready_items), failed_checks=list(failed_checks))
    return manifest


def _write_items_jsonl(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(dict(item), sort_keys=True) + "\n")
        handle.flush()


def _remaining_from_stage(stage_manifest: Mapping[str, Any]) -> int:
    taxonomy = stage_manifest.get("failure_taxonomy_summary")
    if isinstance(taxonomy, Mapping):
        return int(taxonomy.get("remaining_failure_count") or taxonomy.get("after_region_count") or 0)
    return 0


def _after_failure_taxonomy(stage_manifest: Mapping[str, Any]) -> dict[str, Any]:
    taxonomy = stage_manifest.get("failure_taxonomy_summary")
    if not isinstance(taxonomy, Mapping):
        return {}
    after = taxonomy.get("after_failure_taxonomy")
    return dict(after) if isinstance(after, Mapping) else {}


def _taxonomy_coverage(stage_manifest: Mapping[str, Any]) -> dict[str, Any]:
    after = _after_failure_taxonomy(stage_manifest)
    total = int(after.get("total_fail_regions") or _remaining_from_stage(stage_manifest))
    selected = int(after.get("selected_region_count") or 0)
    return {
        "source_total_fail_regions": total,
        "source_selected_region_count": selected,
        "regions_capped": bool(after.get("regions_capped") or (selected > 0 and total > selected)),
    }


def _sorted_count_items(value: Any) -> list[tuple[str, int]]:
    if not isinstance(value, Mapping):
        return []
    rows: list[tuple[str, int]] = []
    for key, count in value.items():
        if isinstance(count, (int, float)) and int(count) > 0:
            rows.append((str(key), int(count)))
    return sorted(rows, key=lambda item: (-item[1], item[0]))


def _taxonomy_batches(stage_manifest: Mapping[str, Any], *, max_regions: int) -> list[dict[str, Any]]:
    after = _after_failure_taxonomy(stage_manifest)
    batches = [
        {"axis": "probe", "value": value, "expected_region_count": count}
        for value, count in _sorted_count_items(after.get("by_probe"))
    ]
    if not batches:
        batches = [
            {"axis": "fail_mode", "value": value, "expected_region_count": count}
            for value, count in _sorted_count_items(after.get("by_fail_mode"))
        ]
    if not batches:
        batches = [{"axis": "all", "value": "all", "expected_region_count": _remaining_from_stage(stage_manifest)}]
    max_chunk_regions = max(1, int(max_regions))
    chunked: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        expected = max(0, int(batch.get("expected_region_count") or 0))
        chunk_count = max(1, math.ceil(expected / max_chunk_regions)) if expected > 0 else 1
        for chunk_index in range(chunk_count):
            chunk_start = chunk_index * max_chunk_regions
            chunk_expected = max(0, min(max_chunk_regions, expected - chunk_start))
            chunked.append(
                {
                    **batch,
                    "batch_id": f"b{index + 1:03d}",
                    "chunk_index": int(chunk_index),
                    "chunk_start": int(chunk_start),
                    "chunk_limit": int(max_chunk_regions),
                    "chunk_count": int(chunk_count),
                    "chunk_expected_region_count": int(chunk_expected),
                }
            )
    return chunked


def _candidate_failed_holdouts(validation_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in list(validation_manifest.get("holdout_results") or []):
        if not isinstance(row, Mapping):
            continue
        remaining = int(row.get("remaining_failure_count") or 0)
        if bool(row.get("passed")) or remaining <= 0:
            continue
        rows.append(dict(row))
    return rows


def _resolve_stage_manifest_path(row: Mapping[str, Any], *, validation_manifest_path: Path) -> Path:
    raw_path = Path(str(row.get("stage_manifest") or ""))
    if raw_path.is_file():
        return raw_path
    holdout_id = str(row.get("holdout_id") or "").strip()
    if holdout_id:
        candidate = validation_manifest_path.parent / holdout_id / "manifest.json"
        if candidate.is_file():
            return candidate
    if raw_path.name == "manifest.json" and raw_path.parent.name:
        candidate = validation_manifest_path.parent / raw_path.parent.name / "manifest.json"
        if candidate.is_file():
            return candidate
    return raw_path


def _invalid_stage_reason(stage_manifest: Mapping[str, Any], *, stage_manifest_path: Path) -> str | None:
    if bool(stage_manifest.get("delivery_allowed")) or bool(stage_manifest.get("listen_pack_created")):
        return "failed_stage_delivery_contract_invalid"
    if str(stage_manifest.get("status") or "") != "diagnostic_blocked_pitch_guard":
        return "failed_stage_not_blocked_pitch_guard"
    if str(stage_manifest.get("after_pitch_verdict") or "") != "fail":
        return "failed_stage_pitch_verdict_not_fail"
    taxonomy = stage_manifest.get("failure_taxonomy_summary")
    if not isinstance(taxonomy, Mapping) or str(taxonomy.get("schema") or "") != "mac_mel_refiner_fulltrack_taxonomy_summary.v1":
        return "failed_stage_missing_failure_taxonomy"
    if _remaining_from_stage(stage_manifest) <= 0:
        return "failed_stage_has_no_remaining_regions"
    if stage_manifest_path.name != "manifest.json":
        return "failed_stage_manifest_path_not_manifest_json"
    return None


def _command_args(
    *,
    failed_stage_dir: Path,
    residual_output_dir: Path,
    target_gain: float,
    epochs: int,
    max_regions: int,
    require_mps: bool,
    region_batch: Mapping[str, Any] | None = None,
) -> list[str]:
    args = [
        "--failed-stage-dir",
        str(failed_stage_dir),
        "--out",
        str(residual_output_dir),
        "--target-gain",
        str(float(target_gain)),
        "--epochs",
        str(int(epochs)),
        "--max-regions",
        str(int(max_regions)),
    ]
    batch = dict(region_batch or {})
    if batch and str(batch.get("axis") or "") != "all":
        args.extend(["--region-batch-json", json.dumps(batch, sort_keys=True)])
    args.append("--require-mps" if require_mps else "--no-require-mps")
    return args


def _planned_item(
    *,
    row: Mapping[str, Any],
    index: int,
    stage_manifest_path: Path,
    stage_manifest: Mapping[str, Any],
    residual_output_root: Path,
    target_gain: float,
    epochs: int,
    max_regions: int,
    require_mps: bool,
    region_batch: Mapping[str, Any],
) -> dict[str, Any]:
    work_id = f"r{index + 1:03d}"
    failed_stage_dir = stage_manifest_path.parent
    residual_output_dir = residual_output_root / work_id
    taxonomy = dict(stage_manifest.get("failure_taxonomy_summary") or {})
    batch = dict(region_batch)
    return {
        "work_id": work_id,
        "holdout_id": str(row.get("holdout_id") or f"h{index + 1:03d}"),
        "ready": True,
        "source_holdout_guard_dir": str(row.get("guard_dir") or ""),
        "stage_manifest": str(stage_manifest_path),
        "source_failed_stage_dir": str(failed_stage_dir),
        "residual_output_dir": str(residual_output_dir),
        "remaining_failure_count": int(batch.get("expected_region_count") or 0),
        "training_region_count": int(batch.get("chunk_expected_region_count") or batch.get("expected_region_count") or 0),
        "source_remaining_failure_count": _remaining_from_stage(stage_manifest),
        "coverage_status": taxonomy.get("coverage_status"),
        "failure_taxonomy_summary": taxonomy,
        "taxonomy_batch": batch,
        "taxonomy_coverage": _taxonomy_coverage(stage_manifest),
        "expected_execution_context": "macos_mps_ssh",
        "expected_acceleration_mode": "gpu_inference_mps",
        "command": {
            "tool": "tools/train_mac_refiner_from_fulltrack_residuals.py",
            "args": _command_args(
                failed_stage_dir=failed_stage_dir,
                residual_output_dir=residual_output_dir,
                target_gain=target_gain,
                epochs=epochs,
                max_regions=max_regions,
                require_mps=require_mps,
                region_batch=batch,
            ),
        },
    }


def _invalid_item(row: Mapping[str, Any], *, reason: str, stage_manifest_path: Path | None = None) -> dict[str, Any]:
    return {
        "holdout_id": str(row.get("holdout_id") or ""),
        "stage_manifest": str(stage_manifest_path or row.get("stage_manifest") or ""),
        "remaining_failure_count": int(row.get("remaining_failure_count") or 0),
        "reason": reason,
        "ready": False,
    }


def plan_residual_handoff(
    *,
    validation_manifest_path: Path,
    output_dir: Path,
    residual_output_root: Path,
    target_gain: float = 2.5,
    epochs: int = 260,
    max_regions: int = 16,
    require_mps: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start", require_mps=bool(require_mps), max_regions=int(max_regions))
    validation_manifest = _load_json(Path(validation_manifest_path))

    if bool(validation_manifest.get("delivery_allowed")) or bool(validation_manifest.get("listen_pack_created")):
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            validation_manifest_path=validation_manifest_path,
            residual_output_root=residual_output_root,
            status="residual_handoff_blocked_validation_contract",
            failed_checks=["validation_delivery_contract_invalid"],
            items=[],
            invalid_items=[],
        )
    if str(validation_manifest.get("schema") or "") != "mac_mel_refiner_top1_holdout_validation.v1":
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            validation_manifest_path=validation_manifest_path,
            residual_output_root=residual_output_root,
            status="residual_handoff_blocked_validation_schema",
            failed_checks=["validation_schema_invalid"],
            items=[],
            invalid_items=[],
            extra={"validation_schema": validation_manifest.get("schema")},
        )
    if str(validation_manifest.get("status") or "") != "validation_blocked_holdout_failure":
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            validation_manifest_path=validation_manifest_path,
            residual_output_root=residual_output_root,
            status="residual_handoff_blocked_validation_status",
            failed_checks=["validation_status_not_holdout_failure"],
            items=[],
            invalid_items=[],
            extra={"validation_status": validation_manifest.get("status")},
        )

    failed_rows = _candidate_failed_holdouts(validation_manifest)
    if not failed_rows:
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            validation_manifest_path=validation_manifest_path,
            residual_output_root=residual_output_root,
            status="residual_handoff_blocked_no_failed_holdouts",
            failed_checks=["no_failed_holdouts"],
            items=[],
            invalid_items=[],
            extra={"validation_status": validation_manifest.get("status")},
        )

    ready_items: list[dict[str, Any]] = []
    invalid_items: list[dict[str, Any]] = []
    failed_checks: list[str] = []
    taxonomy_coverages: list[dict[str, Any]] = []
    for row in failed_rows:
        stage_manifest_path = _resolve_stage_manifest_path(row, validation_manifest_path=Path(validation_manifest_path))
        if not str(row.get("stage_manifest") or "") or not stage_manifest_path.is_file():
            reason = "failed_stage_manifest_missing"
            failed_checks.append(reason)
            invalid_items.append(_invalid_item(row, reason=reason, stage_manifest_path=stage_manifest_path))
            continue
        stage_manifest = _load_json(stage_manifest_path)
        reason = _invalid_stage_reason(stage_manifest, stage_manifest_path=stage_manifest_path)
        if reason is not None:
            failed_checks.append(reason)
            invalid_items.append(_invalid_item(row, reason=reason, stage_manifest_path=stage_manifest_path))
            continue
        taxonomy_coverages.append({"holdout_id": str(row.get("holdout_id") or ""), **_taxonomy_coverage(stage_manifest)})
        for batch in _taxonomy_batches(stage_manifest, max_regions=int(max_regions)):
            ready_items.append(
                _planned_item(
                    row=row,
                    index=len(ready_items),
                    stage_manifest_path=stage_manifest_path,
                    stage_manifest=stage_manifest,
                    residual_output_root=residual_output_root,
                    target_gain=float(target_gain),
                    epochs=int(epochs),
                    max_regions=int(max_regions),
                    require_mps=bool(require_mps),
                    region_batch=batch,
                )
            )

    if invalid_items:
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            validation_manifest_path=validation_manifest_path,
            residual_output_root=residual_output_root,
            status="residual_handoff_blocked_invalid_failed_stage",
            failed_checks=sorted(set(failed_checks)),
            items=[],
            invalid_items=invalid_items,
            extra={"validation_status": validation_manifest.get("status")},
        )

    return _write_manifest(
        output_dir=output_dir,
        progress_path=progress_path,
        validation_manifest_path=validation_manifest_path,
        residual_output_root=residual_output_root,
        status="residual_handoff_ready",
        failed_checks=[],
        items=ready_items,
        invalid_items=[],
        extra={
            "validation_status": validation_manifest.get("status"),
            "source_benchmark_manifest": validation_manifest.get("source_benchmark_manifest"),
            "selected_checkpoint_id": validation_manifest.get("selected_checkpoint_id"),
            "checkpoint": validation_manifest.get("checkpoint"),
            "remaining_failure_count": sum(int(item.get("training_region_count") or 0) for item in ready_items),
            "source_remaining_failure_count": sum(
                int(coverage.get("source_total_fail_regions") or 0)
                for coverage in taxonomy_coverages
            ),
            "region_batch_count": len(ready_items),
            "taxonomy_coverage": taxonomy_coverages[0] if len(taxonomy_coverages) == 1 else {},
            "taxonomy_coverages": taxonomy_coverages,
            "target_gain": float(target_gain),
            "epochs": int(epochs),
            "max_regions": int(max_regions),
            "require_mps": bool(require_mps),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--residual-output-root", type=Path, required=True)
    parser.add_argument("--target-gain", type=float, default=2.5)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = plan_residual_handoff(
        validation_manifest_path=args.validation_manifest,
        output_dir=args.out,
        residual_output_root=args.residual_output_root,
        target_gain=float(args.target_gain),
        epochs=int(args.epochs),
        max_regions=int(args.max_regions),
        require_mps=bool(args.require_mps),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
