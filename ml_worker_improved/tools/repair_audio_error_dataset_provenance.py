#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_audio_error_dataset_provenance import (  # noqa: E402
    AUDIT_JSONL,
    DATASET_MANIFEST,
    DATASET_ROWS,
    SAFETY_FLAGS,
    is_real_value,
    is_synthetic_candidate_id,
    iter_jsonl,
    read_json,
    stable_unique,
    write_json,
    write_jsonl,
)
from tools.extract_audio_error_maps import extract_error_maps_for_window, write_error_maps  # noqa: E402


ORIGINAL_MAPS_INDEX = "audio_error_maps_index.jsonl"
REPAIRED_MANIFEST = "repaired_audio_error_grounding_manifest.json"
REPAIRED_ROWS = "repaired_audio_error_window_dataset.jsonl"
REPAIRED_MAPS_INDEX = "repaired_audio_error_maps_index.jsonl"
REPAIR_REPORT = "provenance_repair_report.json"
REPAIR_SUMMARY = "provenance_repair_summary.md"

METRIC_FIELDS = (
    "residual_energy",
    "mixture_consistency_error",
    "phase_collapse_score",
    "stereo_width_score",
    "vocal_bleed_proxy_score",
    "instrumental_bleed_proxy_score",
    "missing_detail_score",
    "transient_damage_score",
    "artifact_noise_score",
)


def has_existing_map(row: dict[str, Any]) -> bool:
    paths = row.get("error_map_paths")
    if not isinstance(paths, dict):
        return False
    npz = str(paths.get("npz") or "")
    return bool(npz and Path(npz).exists())


def audio_paths_exist(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in ("mix_path", "vocal_candidate_path", "minus_candidate_path"):
        value = str(row.get(field) or "")
        if not value:
            missing.append(f"missing_{field}")
        elif not Path(value).exists():
            missing.append(f"missing_audio_file:{field}")
    return missing


def base_safety_reasons(row: dict[str, Any], *, require_map: bool = True) -> list[str]:
    reasons: list[str] = []
    if not is_real_value(row.get("candidate_id")):
        reasons.append("missing_candidate_id")
    elif is_synthetic_candidate_id(row.get("candidate_id")):
        reasons.append("synthetic_candidate_id")
    if not is_real_value(row.get("route_id")):
        reasons.append("missing_route_id")
    if not is_real_value(row.get("pair_compatibility_status")):
        reasons.append("missing_pair_compatibility_status")
    if not is_real_value(row.get("judge_decision")):
        reasons.append("missing_window_judge_decision")
    try:
        if float(row.get("window_end_sec") or 0.0) <= float(row.get("window_start_sec") or 0.0):
            reasons.append("invalid_window_boundaries")
    except (TypeError, ValueError):
        reasons.append("invalid_window_boundaries")
    reasons.extend(audio_paths_exist(row))
    if row.get("release_ready") is True:
        reasons.append("release_ready_forbidden")
    if row.get("training_started") is True or row.get("separator_training_started") is True:
        reasons.append("training_started_forbidden")
    if row.get("provenance_repaired") is True and not row.get("repair_source_files"):
        reasons.append("missing_repair_source_evidence")
    if require_map and not has_existing_map(row):
        reasons.append("missing_error_map")
    return stable_unique(reasons)


def row_id_for(index: int, row: dict[str, Any]) -> str:
    return str(row.get("row_id") or index)


def audit_rows_by_id(audit_dir: Path) -> dict[str, dict[str, Any]]:
    rows = list(iter_jsonl(audit_dir / AUDIT_JSONL))
    return {str(row.get("row_id") or row.get("row_index") or ""): row for row in rows}


def maybe_extract_map(
    row: dict[str, Any],
    output_dir: Path,
    *,
    row_index: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    if has_existing_map(row):
        return row, None, ""
    prelim_reasons = base_safety_reasons(row, require_map=False)
    if prelim_reasons:
        return row, None, ";".join(prelim_reasons + ["missing_error_map"])
    try:
        result = extract_error_maps_for_window(
            mix_path=Path(str(row["mix_path"])),
            vocal_path=Path(str(row["vocal_candidate_path"])),
            minus_path=Path(str(row["minus_candidate_path"])),
            window_start_sec=float(row["window_start_sec"]),
            window_end_sec=float(row["window_end_sec"]),
        )
        map_row_id = f"{row_index}__{row.get('window_id') or row_id_for(row_index, row)}"
        npz_path = write_error_maps(
            output_dir=output_dir / "error_maps",
            dataset_id=str(row.get("dataset_id") or "repaired_audio_error_grounding"),
            row_id=map_row_id,
            maps=result["maps"],
            metadata=result["metadata"],
        )
        updated = {**row, "error_map_paths": {"npz": str(npz_path)}}
        for field in METRIC_FIELDS:
            updated[field] = result["metrics"].get(field, updated.get(field, 0.0))
        maps_index_row = {
            "dataset_id": updated.get("dataset_id", ""),
            "track_id": updated.get("track_id", ""),
            "run_id": updated.get("run_id", ""),
            "candidate_id": updated.get("candidate_id", ""),
            "stem_type": updated.get("stem_type", ""),
            "window_id": updated.get("window_id", ""),
            "npz_path": str(npz_path),
            "release_ready": False,
        }
        return updated, maps_index_row, ""
    except Exception as exc:  # pragma: no cover - real corrupt audio path.
        return row, None, f"error_map_extraction_failed:{type(exc).__name__}:{exc}"


def apply_repair(row: dict[str, Any], audit_row: dict[str, Any] | None) -> dict[str, Any]:
    updated = {
        **row,
        "release_ready": False,
        "training_started": False,
        "separator_training_started": False,
        "provenance_repaired": False,
        "repaired_fields": [],
        "repair_source_files": [],
        "repair_confidence": 0.0,
    }
    if not audit_row or audit_row.get("repairable") is not True:
        return updated
    repair_values = audit_row.get("repair_values") if isinstance(audit_row.get("repair_values"), dict) else {}
    repaired_fields: list[str] = []
    for field in ("candidate_id", "route_id", "pair_compatibility_status"):
        if is_real_value(updated.get(field)):
            continue
        value = repair_values.get(field)
        if is_real_value(value):
            updated[field] = str(value)
            repaired_fields.append(field)
    if repaired_fields:
        updated["provenance_repaired"] = True
        updated["repaired_fields"] = repaired_fields
        updated["repair_source_files"] = stable_unique(str(value) for value in audit_row.get("repair_source_files") or [] if str(value))
        updated["repair_confidence"] = float(audit_row.get("repair_confidence") or 1.0)
    return updated


def summarize_rows(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows_total": len(after_rows),
        "safe_for_training_count_before": sum(1 for row in before_rows if row.get("safe_for_training") is True),
        "safe_for_training_count_after": sum(1 for row in after_rows if row.get("safe_for_training") is True),
        "unsafe_count_before": sum(1 for row in before_rows if row.get("safe_for_training") is not True),
        "unsafe_count_after": sum(1 for row in after_rows if row.get("safe_for_training") is not True),
        "rows_repaired": sum(1 for row in after_rows if row.get("provenance_repaired") is True),
        "rows_unrepairable": sum(1 for row in after_rows if row.get("safe_for_training") is not True),
        "missing_route_id_before": sum(1 for row in before_rows if not is_real_value(row.get("route_id"))),
        "missing_route_id_after": sum(1 for row in after_rows if not is_real_value(row.get("route_id"))),
        "missing_pair_compatibility_status_before": sum(1 for row in before_rows if not is_real_value(row.get("pair_compatibility_status"))),
        "missing_pair_compatibility_status_after": sum(1 for row in after_rows if not is_real_value(row.get("pair_compatibility_status"))),
        "repair_source_files_used": sorted({source for row in after_rows for source in row.get("repair_source_files") or []}),
    }


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Audio Error Dataset Provenance Repair",
        "",
        f"- rows_total: {report['rows_total']}",
        f"- safe_for_training_count_before: {report['safe_for_training_count_before']}",
        f"- safe_for_training_count_after: {report['safe_for_training_count_after']}",
        f"- unsafe_count_before: {report['unsafe_count_before']}",
        f"- unsafe_count_after: {report['unsafe_count_after']}",
        f"- rows_repaired: {report['rows_repaired']}",
        f"- rows_unrepairable: {report['rows_unrepairable']}",
        f"- missing_route_id_before: {report['missing_route_id_before']}",
        f"- missing_route_id_after: {report['missing_route_id_after']}",
        f"- missing_pair_compatibility_status_before: {report['missing_pair_compatibility_status_before']}",
        f"- missing_pair_compatibility_status_after: {report['missing_pair_compatibility_status_after']}",
        f"- can_start_model_2_training: {str(report['can_start_model_2_training']).lower()}",
        f"- reason_if_training_not_ready: {report['reason_if_training_not_ready']}",
        "- release_ready: false",
        "- training_started: false",
        "- separator_training_started: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def repair_audio_error_dataset_provenance(
    *,
    dataset_dir: Path,
    audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_manifest = read_json(dataset_dir / DATASET_MANIFEST)
    original_rows = list(iter_jsonl(dataset_dir / DATASET_ROWS))
    original_maps_index = list(iter_jsonl(dataset_dir / ORIGINAL_MAPS_INDEX))
    audit_map = audit_rows_by_id(audit_dir)
    repaired_rows: list[dict[str, Any]] = []
    repaired_maps_index: list[dict[str, Any]] = list(original_maps_index)
    for index, row in enumerate(original_rows):
        updated = apply_repair(row, audit_map.get(row_id_for(index, row)))
        updated, map_index_row, map_error = maybe_extract_map(updated, output_dir, row_index=index)
        reasons = base_safety_reasons(updated, require_map=True)
        if map_error:
            reasons.extend(map_error.split(";"))
        reasons = stable_unique(reasons)
        updated["safe_for_training"] = not reasons
        updated["unsafe_reason"] = ";".join(reasons)
        updated.update(SAFETY_FLAGS)
        repaired_rows.append(updated)
        if map_index_row is not None:
            repaired_maps_index.append(map_index_row)
    report = summarize_rows(original_rows, repaired_rows)
    report.update(
        {
            "dataset_id": original_manifest.get("dataset_id", ""),
            "can_start_model_2_training": report["safe_for_training_count_after"] > report["safe_for_training_count_before"],
            "reason_if_training_not_ready": "",
            **SAFETY_FLAGS,
        }
    )
    if not report["can_start_model_2_training"]:
        report["reason_if_training_not_ready"] = "safe_for_training_count_did_not_increase"
    manifest = {**original_manifest, **report}
    manifest.update(SAFETY_FLAGS)
    write_json(output_dir / REPAIRED_MANIFEST, manifest)
    write_jsonl(output_dir / REPAIRED_ROWS, repaired_rows)
    write_jsonl(output_dir / REPAIRED_MAPS_INDEX, repaired_maps_index)
    write_json(output_dir / REPAIR_REPORT, report)
    write_summary(output_dir / REPAIR_SUMMARY, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair audio error dataset provenance from an audit report.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = repair_audio_error_dataset_provenance(
        dataset_dir=args.dataset_dir,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
