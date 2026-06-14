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
    SAFETY_FLAGS,
    is_real_value,
    is_synthetic_candidate_id,
    iter_jsonl,
    read_json,
)


REPAIRED_MANIFEST = "repaired_audio_error_grounding_manifest.json"
REPAIRED_ROWS = "repaired_audio_error_window_dataset.jsonl"
REPAIRED_MAPS_INDEX = "repaired_audio_error_maps_index.jsonl"
REPAIR_REPORT = "provenance_repair_report.json"


def any_true_flag(payload: Any, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and value is True:
                return True
            if any_true_flag(value, keys):
                return True
    if isinstance(payload, list):
        return any(any_true_flag(item, keys) for item in payload)
    return False


def path_exists(value: Any) -> bool:
    text = str(value or "")
    return bool(text and Path(text).exists())


def validate_safe_row(row: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    for field in ("candidate_id", "route_id", "pair_compatibility_status"):
        if not is_real_value(row.get(field)):
            errors.append(f"safe_row_missing_{field}:{index}")
    if str(row.get("route_id") or "").strip().lower() == "unknown":
        errors.append(f"safe_row_forbidden_unknown_route_id:{index}")
    if is_synthetic_candidate_id(row.get("candidate_id")):
        errors.append(f"safe_row_synthetic_candidate_id:{index}")
    if row.get("provenance_repaired") is True:
        if not row.get("repair_source_files") or not row.get("repaired_fields"):
            errors.append(f"safe_repaired_row_missing_source_evidence:{index}")
        if float(row.get("repair_confidence") or 0.0) <= 0.0:
            errors.append(f"safe_repaired_row_missing_repair_confidence:{index}")
    for field in ("mix_path", "vocal_candidate_path", "minus_candidate_path"):
        if not path_exists(row.get(field)):
            errors.append(f"safe_row_missing_audio_file:{index}:{field}")
    maps = row.get("error_map_paths")
    if not isinstance(maps, dict) or not path_exists(maps.get("npz")):
        errors.append(f"safe_row_missing_error_map:{index}")
    try:
        if float(row.get("window_end_sec") or 0.0) <= float(row.get("window_start_sec") or 0.0):
            errors.append(f"safe_row_invalid_window_boundaries:{index}")
    except (TypeError, ValueError):
        errors.append(f"safe_row_invalid_window_boundaries:{index}")
    return errors


def validate_audio_error_dataset_provenance_repair(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / REPAIRED_MANIFEST
    rows_path = dataset_dir / REPAIRED_ROWS
    maps_index_path = dataset_dir / REPAIRED_MAPS_INDEX
    report_path = dataset_dir / REPAIR_REPORT
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    rows = list(iter_jsonl(rows_path))
    maps_index = list(iter_jsonl(maps_index_path))
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("missing_repaired_manifest")
    if not rows_path.exists():
        errors.append("missing_repaired_window_dataset")
    if not maps_index_path.exists():
        errors.append("missing_repaired_maps_index")
    if not report_path.exists():
        errors.append("missing_provenance_repair_report")
    forbidden_flags = {"release_ready", "training_started", "separator_training_started"}
    if any_true_flag(manifest, forbidden_flags):
        errors.append("forbidden_true_flag:manifest")
    if any_true_flag(report, forbidden_flags):
        errors.append("forbidden_true_flag:repair_report")
    if any_true_flag(rows, forbidden_flags):
        errors.append("forbidden_true_flag:rows")
    safe_after = 0
    unsafe_after = 0
    rows_repaired = 0
    missing_route_after = 0
    missing_pair_after = 0
    for index, row in enumerate(rows):
        if not is_real_value(row.get("route_id")):
            missing_route_after += 1
        if not is_real_value(row.get("pair_compatibility_status")):
            missing_pair_after += 1
        if row.get("provenance_repaired") is True:
            rows_repaired += 1
        if row.get("safe_for_training") is True:
            safe_after += 1
            errors.extend(validate_safe_row(row, index))
        else:
            unsafe_after += 1
    safe_before = int(report.get("safe_for_training_count_before") or manifest.get("safe_for_training_count_before") or 0)
    unsafe_before = int(report.get("unsafe_count_before") or manifest.get("unsafe_count_before") or 0)
    result = {
        "ok": not errors,
        "errors": errors,
        "rows_total": len(rows),
        "safe_for_training_count_before": safe_before,
        "safe_for_training_count_after": safe_after,
        "unsafe_count_before": unsafe_before,
        "unsafe_count_after": unsafe_after,
        "rows_repaired": rows_repaired,
        "rows_unrepairable": unsafe_after,
        "missing_route_id_after": missing_route_after,
        "missing_pair_compatibility_status_after": missing_pair_after,
        "maps_index_rows": len(maps_index),
        **SAFETY_FLAGS,
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate repaired audio error dataset provenance.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_audio_error_dataset_provenance_repair(args.dataset_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
