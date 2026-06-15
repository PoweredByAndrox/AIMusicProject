#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


SLICE_FILE = "tensor_audio_error_locked_mining_slice.jsonl"
SLICE_MANIFEST = "tensor_audio_error_locked_mining_slice_manifest.json"
SLICE_VALIDATION = "tensor_audio_error_locked_mining_slice_validation.json"

LOCKED_EVAL_SPLITS = {"dev", "test"}
ALLOWED_SLICE_LABELS = {
    "candidate_regressed_false_reject",
    "persistent_false_reject",
    "candidate_regressed_hard_negative",
    "persistent_hard_negative",
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


def validate_tensor_audio_error_locked_mining_slice(slice_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = slice_dir / SLICE_MANIFEST
    rows_path = slice_dir / SLICE_FILE
    if not manifest_path.exists():
        errors.append(f"missing_required_file:{SLICE_MANIFEST}")
        manifest: dict[str, Any] = {}
    else:
        manifest = read_json(manifest_path)
    if not rows_path.exists():
        errors.append(f"missing_required_file:{SLICE_FILE}")
        rows: list[dict[str, Any]] = []
    else:
        rows = list(iter_jsonl(rows_path))
    if manifest.get("schema") != "tensor_audio_error_locked_mining_slice.v1":
        errors.append("invalid_manifest_schema")
    if manifest.get("slice_status") != "candidate_only_locked_eval_review":
        errors.append("invalid_slice_status")
    if manifest.get("locked_eval_only") is not True:
        errors.append("locked_eval_only_not_true")
    if int(manifest.get("candidate_rows_total") or 0) != len(rows):
        errors.append("candidate_rows_total_mismatch")
    if manifest.get("training_allowed") is True:
        errors.append("manifest_training_allowed_true")
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag(manifest, flag):
            errors.append(f"{flag}_true")
    for index, row in enumerate(rows):
        if row.get("slice_row_schema") != "tensor_audio_error_locked_mining_slice_row.v1":
            errors.append(f"invalid_row_schema:{index}")
        if str(row.get("slice_label") or "") not in ALLOWED_SLICE_LABELS:
            errors.append(f"invalid_slice_label:{index}")
        if str(row.get("split") or "") not in LOCKED_EVAL_SPLITS:
            errors.append(f"row_train_split_not_allowed:{index}")
        if row.get("training_allowed") is True:
            errors.append(f"row_training_allowed_true:{index}")
        if row.get("shadow_only") is not True:
            errors.append(f"row_shadow_only_not_true:{index}")
        for flag in sorted(FORBIDDEN_TRUE_FLAGS):
            if has_true_flag(row, flag):
                errors.append(f"row_{flag}_true:{index}")
    result = {
        "schema": "tensor_audio_error_locked_mining_slice_validation.v1",
        "slice_dir": str(slice_dir),
        "ok": not errors,
        "errors": sorted(set(errors)),
        "candidate_rows_total": len(rows),
        "release_ready": False,
        "runtime_gates_changed": False,
        "training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
    }
    write_json(slice_dir / SLICE_VALIDATION, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a locked-eval-only tensor critic mining slice.")
    parser.add_argument("--slice-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tensor_audio_error_locked_mining_slice(args.slice_dir)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
