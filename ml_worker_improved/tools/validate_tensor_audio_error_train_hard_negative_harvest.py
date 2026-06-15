#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


HARVEST_MANIFEST_FILE = "tensor_audio_error_train_hard_negative_harvest_manifest.json"
HARVEST_ROWS_FILE = "tensor_audio_error_train_hard_negative_candidates.jsonl"
HARVEST_VALIDATION_FILE = "tensor_audio_error_train_hard_negative_harvest_validation.json"
HARVEST_SCHEMA = "tensor_audio_error_train_hard_negative_harvest.v1"
HARVEST_ROW_SCHEMA = "tensor_audio_error_train_hard_negative_candidate.v1"
ALLOWED_STATUSES = {
    "harvest_ready",
    "blocked_no_train_negative_source",
    "blocked_unsafe_generation_plan",
    "blocked_missing_tensor_maps",
    "blocked_leakage_risk",
}
FORBIDDEN_TRUE_FLAGS = {
    "release_ready",
    "can_publish_as_result",
    "runtime_gate_decision_applied",
    "runtime_gates_changed",
    "training_started",
    "critic_training_started",
    "separator_training_started",
    "repair_adapter_training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected_json_object:{path}:{line_number}")
            yield payload


def has_true_flag(payload: Any, flag: str) -> bool:
    if isinstance(payload, dict):
        return payload.get(flag) is True or any(has_true_flag(value, flag) for value in payload.values())
    if isinstance(payload, list):
        return any(has_true_flag(item, flag) for item in payload)
    return False


def validate_tensor_audio_error_train_hard_negative_harvest(harvest_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = harvest_dir / HARVEST_MANIFEST_FILE
    rows_path = harvest_dir / HARVEST_ROWS_FILE
    if not manifest_path.exists():
        errors.append(f"missing_required_file:{HARVEST_MANIFEST_FILE}")
        manifest: dict[str, Any] = {}
    else:
        manifest = read_json(manifest_path)
    rows = list(iter_jsonl(rows_path)) if rows_path.exists() else []
    if not rows_path.exists():
        errors.append(f"missing_required_file:{HARVEST_ROWS_FILE}")

    status = str(manifest.get("harvest_status") or "")
    if manifest.get("schema") != HARVEST_SCHEMA:
        errors.append("invalid_harvest_schema")
    if status not in ALLOWED_STATUSES:
        errors.append("invalid_harvest_status")
    if int(manifest.get("candidate_rows_total") or 0) != len(rows):
        errors.append("candidate_rows_total_mismatch")
    if status == "harvest_ready" and int(manifest.get("negative_rows") or 0) == 0:
        errors.append("harvest_ready_without_negative_rows")
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag(manifest, flag):
            errors.append(f"{flag}_true")
    if manifest.get("dev_test_rows_used") is True:
        errors.append("dev_test_rows_used_true")
    if manifest.get("locked_eval_rows_used") is True:
        errors.append("locked_eval_rows_used_true")
    if manifest.get("hardcase_rows_used") is True:
        errors.append("hardcase_rows_used_true")

    negative_rows = 0
    hard_negative_rows = 0
    for index, row in enumerate(rows):
        if row.get("row_schema") != HARVEST_ROW_SCHEMA:
            errors.append(f"invalid_row_schema:{index}")
        if row.get("split_overlap_checked") is not True:
            errors.append(f"row_split_overlap_not_checked:{index}")
        if row.get("blocked_overlap_keys"):
            errors.append(f"row_blocked_overlap_keys_present:{index}")
        if str(row.get("split") or "") != "train":
            errors.append(f"row_split_not_train:{index}")
        if row.get("train_allowed") is not True:
            errors.append(f"row_train_allowed_not_true:{index}")
        if row.get("hardcase_holdout") is True or row.get("hardcase") is True:
            errors.append(f"row_hardcase_holdout_true:{index}")
        if row.get("locked_eval_row") is True:
            errors.append(f"row_locked_eval_row_true:{index}")
        if int(row.get("true_label") or 0) != 0:
            errors.append(f"row_true_label_not_zero:{index}")
        else:
            negative_rows += 1
        if str(row.get("source_mining_label") or "") != "hard_negative_false_accept":
            errors.append(f"row_source_mining_label_not_hard_negative:{index}")
        else:
            hard_negative_rows += 1
        tensor_map_path = Path(str(row.get("tensor_map_path") or ""))
        if not tensor_map_path.exists():
            errors.append(f"row_missing_tensor_map:{index}")
        if not row.get("tensor_map_names"):
            errors.append(f"row_tensor_map_names_empty:{index}")
        if row.get("shadow_only") is not True:
            errors.append(f"row_shadow_only_not_true:{index}")
        for flag in sorted(FORBIDDEN_TRUE_FLAGS):
            if has_true_flag(row, flag):
                errors.append(f"row_forbidden_true_flag:{index}:{flag}")

    result = {
        "schema": "tensor_audio_error_train_hard_negative_harvest_validation.v1",
        "harvest_dir": str(harvest_dir),
        "ok": not errors,
        "errors": sorted(set(errors)),
        "harvest_status": status,
        "candidate_rows_total": len(rows),
        "negative_rows": negative_rows,
        "hard_negative_rows": hard_negative_rows,
        "dev_test_rows_used": False,
        "locked_eval_rows_used": False,
        "hardcase_rows_used": False,
        "training_started": False,
        "release_ready": False,
        "runtime_gates_changed": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
    }
    write_json(harvest_dir / HARVEST_VALIDATION_FILE, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate train-only tensor hard-negative harvest artifacts.")
    parser.add_argument("--harvest-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tensor_audio_error_train_hard_negative_harvest(args.harvest_dir)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
