#!/usr/bin/env python3
"""Resolve independent full-holdout evidence for a Mac residual candidate.

This is a metadata-only diagnostic gate. It does not run model inference, does
not read audio bytes, and never creates listen/result delivery.
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


PASSABLE_CANDIDATE_SCHEMAS = {
    "mac_residual_damage_limited_repair_route.v1": "damage_limited_repair_passed",
    "mac_residual_damage_audit_route.v1": "residual_damage_audit_passed",
}


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


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _guard_dir(path: Any) -> Path | None:
    if not path:
        return None
    guard_path = Path(str(path))
    if guard_path.name == "manifest.json":
        return guard_path.parent
    return guard_path


def _mapping_value(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_id_from_path(path: Path) -> str:
    if path.parent.name and path.parent.parent.name:
        return f"{path.parent.parent.name}_{path.parent.name}"
    return path.parent.name or path.stem


def _candidate_preflight(candidate: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    schema = str(candidate.get("schema") or "")
    expected_status = PASSABLE_CANDIDATE_SCHEMAS.get(schema)
    if expected_status is None:
        blockers.append("unsupported_candidate_schema")
    elif str(candidate.get("status") or "") != expected_status:
        blockers.append("candidate_status_not_passed")
    after_score = _mapping_value(candidate, "after_score")
    damage_audit = _mapping_value(candidate, "damage_audit")
    if str(after_score.get("verdict") or "") != "pass":
        blockers.append("candidate_pitch_not_passed")
    if str(damage_audit.get("verdict") or "") != "pass":
        blockers.append("candidate_damage_not_passed")
    if bool(candidate.get("delivery_allowed")) or bool(candidate.get("listen_pack_created")):
        blockers.append("candidate_delivery_contract_invalid")
    if bool(damage_audit.get("delivery_allowed")) or bool(damage_audit.get("listen_pack_created")):
        blockers.append("candidate_damage_contract_invalid")
    return blockers


def _linked_to_candidate(holdout: Mapping[str, Any], candidate_manifest_path: Path) -> bool:
    values = [
        holdout.get("source_candidate_manifest"),
        holdout.get("candidate_manifest"),
        holdout.get("selected_candidate_manifest"),
    ]
    for value in values:
        if value and _same_path(Path(str(value)), candidate_manifest_path):
            return True
    return False


def _holdout_guard_dirs(holdout: Mapping[str, Any]) -> list[Path]:
    guard_dirs: list[Path] = []
    for key in ("source_guard_dir", "mel_guard_dir", "guard_dir", "source_guard_manifest"):
        guard_dir = _guard_dir(holdout.get(key))
        if guard_dir is not None:
            guard_dirs.append(guard_dir)
    for row in holdout.get("holdout_results") or []:
        if isinstance(row, Mapping):
            guard_dir = _guard_dir(row.get("guard_dir") or row.get("source_guard_dir") or row.get("source_guard_manifest"))
            if guard_dir is not None:
                guard_dirs.append(guard_dir)
    return guard_dirs


def _remaining_failures(holdout: Mapping[str, Any]) -> int:
    taxonomy = _mapping_value(holdout, "failure_taxonomy_summary")
    return int(taxonomy.get("remaining_failure_count") or taxonomy.get("after_region_count") or 0)


def _holdout_passed(holdout: Mapping[str, Any]) -> bool:
    if bool(holdout.get("delivery_allowed")) or bool(holdout.get("listen_pack_created")):
        return False
    schema = str(holdout.get("schema") or "")
    status = str(holdout.get("status") or "")
    if schema == "mac_mel_refiner_top1_holdout_validation.v1":
        rows = [row for row in holdout.get("holdout_results") or [] if isinstance(row, Mapping)]
        return status == "validation_passed" and bool(rows) and all(bool(row.get("passed")) for row in rows)
    if schema == "mac_mel_refiner_chain_validation.v1":
        return status == "validation_passed" and not bool(holdout.get("training_overlap"))
    if schema == "mac_mel_refiner_fulltrack_stage.v1":
        return status == "diagnostic_passed" and str(holdout.get("after_pitch_verdict") or "") == "pass" and _remaining_failures(holdout) == 0
    return status == "validation_passed"


def _holdout_result(path: Path, holdout: Mapping[str, Any], *, candidate_manifest_path: Path) -> dict[str, Any]:
    return {
        "manifest": str(path),
        "schema": str(holdout.get("schema") or ""),
        "status": str(holdout.get("status") or ""),
        "linked_to_candidate": _linked_to_candidate(holdout, candidate_manifest_path),
        "guard_dirs": [str(path) for path in _holdout_guard_dirs(holdout)],
        "delivery_allowed": bool(holdout.get("delivery_allowed")),
        "listen_pack_created": bool(holdout.get("listen_pack_created")),
        "passed": _holdout_passed(holdout),
        "failed_checks": [str(value) for value in holdout.get("failed_checks") or []],
    }


def _base_manifest(*, candidate_manifest_path: Path, output_dir: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "mac_residual_candidate_full_holdout_validation.v1",
        "source_candidate_manifest": str(candidate_manifest_path),
        "candidate_id": _candidate_id_from_path(candidate_manifest_path),
        "candidate_schema": str(candidate.get("schema") or ""),
        "candidate_status": str(candidate.get("status") or ""),
        "candidate_source_guard_manifest": str(candidate.get("source_guard_manifest") or candidate.get("guard_manifest") or ""),
        "output_dir": str(output_dir),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "residual_candidate_full_holdout_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "metadata_only": True,
            "gpu_inference_skipped": True,
            "requires_linked_holdout_evidence": True,
            "requires_independent_holdout_evidence": True,
            "blocks_training_overlap_with_candidate_source_guard": True,
        },
    }


def _write_manifest(
    *,
    output_dir: Path,
    progress_path: Path,
    candidate_manifest_path: Path,
    candidate: Mapping[str, Any],
    status: str,
    failed_checks: list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        **_base_manifest(candidate_manifest_path=candidate_manifest_path, output_dir=output_dir, candidate=candidate),
        "status": status,
        "full_holdout_resolution_status": "passed" if status == "validation_passed" else "failed",
        "failed_checks": list(failed_checks),
        **dict(extra or {}),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", run_status=status, failed_checks=failed_checks)
    return manifest


def run_residual_candidate_full_holdout_validation(
    *,
    candidate_manifest_path: Path,
    holdout_manifest_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(
        progress_path,
        "preflight",
        "start",
        acceleration_mode="cpu_metadata_validation",
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="metadata_full_holdout_resolver_uses_existing_manifests",
        holdout_count=len(holdout_manifest_paths),
    )
    candidate_manifest_path = Path(candidate_manifest_path)
    candidate = _load_json(candidate_manifest_path)
    candidate_blockers = _candidate_preflight(candidate)
    if candidate_blockers:
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            candidate_manifest_path=candidate_manifest_path,
            candidate=candidate,
            status="validation_blocked_candidate_not_ready",
            failed_checks=candidate_blockers,
            extra={"holdout_count": 0, "holdout_results": []},
        )
    if not holdout_manifest_paths:
        return _write_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            candidate_manifest_path=candidate_manifest_path,
            candidate=candidate,
            status="validation_blocked_no_holdout_evidence",
            failed_checks=["no_holdout_evidence"],
            extra={"holdout_count": 0, "holdout_results": []},
        )

    holdout_results: list[dict[str, Any]] = []
    for path in holdout_manifest_paths:
        holdout_path = Path(path)
        holdout = _load_json(holdout_path)
        result = _holdout_result(holdout_path, holdout, candidate_manifest_path=candidate_manifest_path)
        holdout_results.append(result)
        _append_progress(progress_path, "holdout", "checked", manifest=str(holdout_path), holdout_status=result["status"], passed=result["passed"])

    candidate_guard = _guard_dir(candidate.get("source_guard_manifest") or candidate.get("guard_manifest"))
    overlap_evidence: list[dict[str, Any]] = []
    if candidate_guard is not None:
        for result in holdout_results:
            for guard_dir in result["guard_dirs"]:
                if _same_path(Path(guard_dir), candidate_guard):
                    overlap_evidence.append(
                        {
                            "holdout_manifest": result["manifest"],
                            "holdout_guard_dir": guard_dir,
                            "candidate_source_guard_manifest": str(candidate.get("source_guard_manifest") or candidate.get("guard_manifest") or ""),
                        }
                    )

    if any(bool(row["delivery_allowed"]) or bool(row["listen_pack_created"]) for row in holdout_results):
        status, failed = "validation_blocked_delivery_contract", ["holdout_delivery_contract_invalid"]
    elif any(not bool(row["linked_to_candidate"]) for row in holdout_results):
        status, failed = "validation_blocked_unlinked_holdout_evidence", ["holdout_evidence_unlinked_to_candidate"]
    elif overlap_evidence:
        status, failed = "validation_blocked_training_overlap", ["training_overlap"]
    elif any(not bool(row["passed"]) for row in holdout_results):
        status, failed = "validation_blocked_holdout_failure", ["holdout_validation_failed"]
    else:
        status, failed = "validation_passed", []

    return _write_manifest(
        output_dir=output_dir,
        progress_path=progress_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate=candidate,
        status=status,
        failed_checks=failed,
        extra={
            "holdout_count": len(holdout_results),
            "holdout_results": holdout_results,
            "passed_holdout_count": sum(1 for row in holdout_results if bool(row.get("passed"))),
            "training_overlap_evidence": overlap_evidence,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", dest="holdout_manifest_paths", default=[])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_residual_candidate_full_holdout_validation(
        candidate_manifest_path=args.candidate_manifest,
        holdout_manifest_paths=list(args.holdout_manifest_paths or []),
        output_dir=args.out,
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
