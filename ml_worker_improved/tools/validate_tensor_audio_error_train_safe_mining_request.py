#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUEST_MANIFEST_FILE = "tensor_audio_error_train_safe_mining_request_manifest.json"
REQUEST_ROWS_FILE = "tensor_audio_error_train_safe_mining_requirements.jsonl"
REQUEST_VALIDATION_FILE = "tensor_audio_error_train_safe_mining_request_validation.json"

REQUEST_SCHEMA = "tensor_audio_error_train_safe_mining_request.v1"
REQUEST_ROW_SCHEMA = "tensor_audio_error_train_safe_mining_requirement.v1"
FORBIDDEN_ROW_SPLITS = {"dev", "test", "hardcase", "holdout"}
ALLOWED_TARGETS = {
    "hard_negative_false_accept": 0,
    "false_reject_positive": 1,
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


def validate_tensor_audio_error_train_safe_mining_request(request_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = request_dir / REQUEST_MANIFEST_FILE
    rows_path = request_dir / REQUEST_ROWS_FILE
    if not manifest_path.exists():
        errors.append(f"missing_required_file:{REQUEST_MANIFEST_FILE}")
        manifest: dict[str, Any] = {}
    else:
        manifest = read_json(manifest_path)
    if not rows_path.exists():
        errors.append(f"missing_required_file:{REQUEST_ROWS_FILE}")
        rows: list[dict[str, Any]] = []
    else:
        rows = list(iter_jsonl(rows_path))

    if manifest.get("schema") != REQUEST_SCHEMA:
        errors.append("invalid_request_schema")
    if int(manifest.get("requirements_total") or 0) != len(rows):
        errors.append("requirements_total_mismatch")
    if manifest.get("locked_eval_only_source") is not True:
        errors.append("locked_eval_only_source_not_true")
    if manifest.get("review_only_source") is not True:
        errors.append("review_only_source_not_true")
    if manifest.get("training_rows_created") is True:
        errors.append("training_rows_created_true")
    if manifest.get("training_allowed") is True:
        errors.append("training_allowed_true")
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag(manifest, flag):
            errors.append(f"{flag}_true")

    for index, row in enumerate(rows):
        if row.get("requirement_schema") != REQUEST_ROW_SCHEMA:
            errors.append(f"invalid_row_schema:{index}")
        if row.get("training_allowed") is True:
            errors.append(f"row_training_allowed_true:{index}")
        if row.get("release_ready") is True:
            errors.append(f"row_release_ready_true:{index}")
        if row.get("shadow_only") is not True:
            errors.append(f"row_shadow_only_not_true:{index}")
        if str(row.get("must_use_split") or "") != "train":
            errors.append(f"row_must_use_split_not_train:{index}")
        forbidden = {str(item) for item in row.get("forbidden_splits", []) or []}
        if not FORBIDDEN_ROW_SPLITS.issubset(forbidden):
            errors.append(f"row_forbidden_splits_incomplete:{index}")
        target_label = str(row.get("target_mining_label") or "")
        if target_label not in ALLOWED_TARGETS:
            errors.append(f"row_unknown_target_mining_label:{index}")
        elif int(row.get("target_training_label") or 0) != ALLOWED_TARGETS[target_label]:
            errors.append(f"row_target_label_mismatch:{index}")
        query = row.get("recommended_train_query")
        if not isinstance(query, dict) or str(query.get("split") or "") != "train":
            errors.append(f"row_query_split_not_train:{index}")
        if any(has_true_flag(row, flag) for flag in FORBIDDEN_TRUE_FLAGS):
            errors.append(f"row_forbidden_true_flag:{index}")

    result = {
        "schema": "tensor_audio_error_train_safe_mining_request_validation.v1",
        "request_dir": str(request_dir),
        "ok": not errors,
        "errors": sorted(set(errors)),
        "requirements_total": len(rows),
        "training_allowed": False,
        "training_rows_created": False,
        "release_ready": False,
        "runtime_gates_changed": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
    }
    write_json(request_dir / REQUEST_VALIDATION_FILE, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a train-safe tensor critic mining request.")
    parser.add_argument("--request-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tensor_audio_error_train_safe_mining_request(args.request_dir)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
