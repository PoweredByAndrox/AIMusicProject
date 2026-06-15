#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


BUNDLE_MANIFEST_FILE = "tensor_audio_error_train_only_mining_bundle_manifest.json"
BUNDLE_ROWS_FILE = "tensor_audio_error_train_only_training_candidates.jsonl"
BUNDLE_VALIDATION_FILE = "tensor_audio_error_train_only_mining_bundle_validation.json"

BUNDLE_SCHEMA = "tensor_audio_error_train_only_mining_bundle.v1"
BUNDLE_ROW_SCHEMA = "tensor_audio_error_train_only_training_candidate.v1"
USABLE_STATUS = "train_only_candidate_only"
BLOCKED_STATUS = "blocked_insufficient_train_safe_rows"
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


def validate_tensor_audio_error_train_only_mining_bundle(bundle_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = bundle_dir / BUNDLE_MANIFEST_FILE
    rows_path = bundle_dir / BUNDLE_ROWS_FILE
    if not manifest_path.exists():
        errors.append(f"missing_required_file:{BUNDLE_MANIFEST_FILE}")
        manifest: dict[str, Any] = {}
    else:
        manifest = read_json(manifest_path)
    if not rows_path.exists():
        errors.append(f"missing_required_file:{BUNDLE_ROWS_FILE}")
        rows: list[dict[str, Any]] = []
    else:
        rows = list(iter_jsonl(rows_path))

    if manifest.get("schema") != BUNDLE_SCHEMA:
        errors.append("invalid_bundle_schema")
    status = str(manifest.get("bundle_status") or "")
    if status not in {USABLE_STATUS, BLOCKED_STATUS}:
        errors.append("invalid_bundle_status")
    if int(manifest.get("candidate_rows_total") or 0) != len(rows):
        errors.append("candidate_rows_total_mismatch")
    if manifest.get("tensor_maps_required") is not True:
        errors.append("tensor_maps_required_not_true")
    if manifest.get("scalar_features_used") is True:
        errors.append("scalar_features_used_true")
    if manifest.get("dev_test_mining_rows_used_for_training") is True:
        errors.append("dev_test_mining_rows_used_for_training_true")
    if manifest.get("locked_eval_rows_used_for_training") is True:
        errors.append("locked_eval_rows_used_for_training_true")
    if manifest.get("hardcase_holdout_rows_used_for_training") is True:
        errors.append("hardcase_holdout_rows_used_for_training_true")
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag(manifest, flag):
            errors.append(f"{flag}_true")

    positives = 0
    negatives = 0
    for index, row in enumerate(rows):
        if row.get("training_candidate_schema") != BUNDLE_ROW_SCHEMA or row.get("bundle_row_schema") != BUNDLE_ROW_SCHEMA:
            errors.append(f"invalid_row_schema:{index}")
        if row.get("training_candidate_status") != "candidate_only":
            errors.append(f"row_not_candidate_only:{index}")
        if row.get("eligible_for_future_training_review") is not True:
            errors.append(f"row_not_eligible_for_future_training_review:{index}")
        if str(row.get("split") or "") != "train":
            errors.append(f"row_split_not_train:{index}")
        if row.get("train_allowed") is not True:
            errors.append(f"row_train_allowed_not_true:{index}")
        if row.get("hardcase_holdout") is True:
            errors.append(f"row_hardcase_holdout_true:{index}")
        if row.get("locked_eval_row") is True:
            errors.append(f"row_locked_eval_row_true:{index}")
        if row.get("scalar_features_used") is True:
            errors.append(f"row_scalar_features_used_true:{index}")
        if row.get("shadow_only") is not True:
            errors.append(f"row_shadow_only_not_true:{index}")
        tensor_map_path = Path(str(row.get("tensor_map_path") or ""))
        if not tensor_map_path.exists():
            errors.append(f"row_missing_tensor_map:{index}")
        label = int(row.get("recommended_training_label") or 0)
        mining_label = str(row.get("source_mining_label") or "")
        if label == 1:
            positives += 1
        else:
            negatives += 1
        if mining_label == "hard_negative_false_accept" and label != 0:
            errors.append(f"row_hard_negative_label_mismatch:{index}")
        elif mining_label == "false_reject_positive" and label != 1:
            errors.append(f"row_false_reject_label_mismatch:{index}")
        elif mining_label not in {"hard_negative_false_accept", "false_reject_positive"}:
            errors.append(f"row_unknown_mining_label:{index}")
        for flag in sorted(FORBIDDEN_TRUE_FLAGS):
            if has_true_flag(row, flag):
                errors.append(f"row_forbidden_true_flag:{index}:{flag}")

    if status == USABLE_STATUS:
        if manifest.get("training_allowed") is not True:
            errors.append("usable_bundle_training_allowed_not_true")
        if positives == 0 or negatives == 0:
            errors.append("usable_bundle_lacks_positive_or_negative_class")
        if positives < int(manifest.get("min_train_positive_rows") or 0):
            errors.append("usable_bundle_positive_rows_below_minimum")
        if negatives < int(manifest.get("min_train_negative_rows") or 0):
            errors.append("usable_bundle_negative_rows_below_minimum")
        if len(rows) < int(manifest.get("min_train_rows_total") or 0):
            errors.append("usable_bundle_rows_below_minimum")
    elif manifest.get("training_allowed") is True:
        errors.append("blocked_bundle_training_allowed_true")

    result = {
        "schema": "tensor_audio_error_train_only_mining_bundle_validation.v1",
        "bundle_dir": str(bundle_dir),
        "ok": not errors,
        "errors": sorted(set(errors)),
        "bundle_status": status,
        "candidate_rows_total": len(rows),
        "positive_rows": positives,
        "negative_rows": negatives,
        "dev_test_mining_rows_used_for_training": False,
        "locked_eval_rows_used_for_training": False,
        "hardcase_rows_used_for_training": False,
        "hardcase_train_rows_used_for_training": any(row.get("hardcase") is True for row in rows),
        "hardcase_holdout_rows_used_for_training": manifest.get("hardcase_holdout_rows_used_for_training") is True
        or any(row.get("hardcase_holdout") is True for row in rows),
        "training_started": False,
        "release_ready": False,
        "runtime_gates_changed": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
    }
    write_json(bundle_dir / BUNDLE_VALIDATION_FILE, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a train-only tensor critic mining bundle.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tensor_audio_error_train_only_mining_bundle(args.bundle_dir)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
