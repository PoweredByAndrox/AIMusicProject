#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


COMPARISON_JSON = "tensor_audio_error_critic_variant_comparison.json"
DECISION_MANIFEST = "tensor_audio_error_critic_decision_manifest.json"
VALIDATION_JSON = "tensor_audio_error_critic_decision_validation.json"

ALLOWED_DECISIONS = {
    "keep_as_specialist_shadow_model",
    "mine_next_dataset_slice",
    "candidate_rejected_runtime",
}
REQUIRED_HARDCASE_METRIC_KEYS = {"rows", "hard_negative_count", "false_reject_count", "accuracy"}
REQUIRED_HARDCASE_DELTA_KEYS = {"hard_negative_count", "false_reject_count", "accuracy"}

FORBIDDEN_TRUE_FLAGS = {
    "release_ready",
    "can_publish_as_result",
    "runtime_gate_update_allowed",
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


def has_true_flag(payload: Any, flag: str) -> bool:
    if isinstance(payload, dict):
        return payload.get(flag) is True or any(has_true_flag(value, flag) for value in payload.values())
    if isinstance(payload, list):
        return any(has_true_flag(item, flag) for item in payload)
    return False


def non_empty_list(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    return isinstance(value, list) and len(value) > 0


def has_hardcase_metrics(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    def metric_block_ok(block: Any) -> bool:
        return isinstance(block, dict) and REQUIRED_HARDCASE_METRIC_KEYS.issubset(block.keys())

    def delta_block_ok(block: Any) -> bool:
        return isinstance(block, dict) and REQUIRED_HARDCASE_DELTA_KEYS.issubset(block.keys())

    baseline = payload.get("baseline")
    candidate = payload.get("candidate")
    delta = payload.get("delta")
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        if "hardcase_holdout" in baseline and "hardcase_holdout" in candidate:
            delta_block = delta.get("hardcase_holdout") if isinstance(delta, dict) else {}
            return (
                metric_block_ok(baseline["hardcase_holdout"])
                and metric_block_ok(candidate["hardcase_holdout"])
                and delta_block_ok(delta_block)
            )
        return metric_block_ok(baseline) and metric_block_ok(candidate) and delta_block_ok(delta)
    return False


def validate_tensor_audio_error_critic_decision(decision_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    decision_path = decision_dir / DECISION_MANIFEST
    comparison_path = decision_dir / COMPARISON_JSON
    if not decision_path.exists():
        errors.append(f"missing_required_file:{DECISION_MANIFEST}")
        decision: dict[str, Any] = {}
    else:
        decision = read_json(decision_path)
    if not comparison_path.exists():
        errors.append(f"missing_required_file:{COMPARISON_JSON}")
        comparison: dict[str, Any] = {}
    else:
        comparison = read_json(comparison_path)

    if decision.get("schema") != "tensor_audio_error_critic_decision.v1":
        errors.append("invalid_decision_schema")
    if comparison.get("schema") != "tensor_audio_error_critic_variant_comparison.v1":
        errors.append("invalid_comparison_schema")

    decision_value = str(decision.get("decision") or "")
    if decision_value not in ALLOWED_DECISIONS or "promote" in decision_value:
        errors.append("promotion_decision_not_allowed")
    if decision.get("promotion_allowed") is True:
        errors.append("promotion_allowed_true")
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag({"decision": decision, "comparison": comparison}, flag):
            errors.append(f"{flag}_true")

    contamination = decision.get("contamination_status") if isinstance(decision.get("contamination_status"), dict) else {}
    comparison_contamination = (
        comparison.get("contamination_status") if isinstance(comparison.get("contamination_status"), dict) else {}
    )
    locked_eval = comparison.get("locked_eval") if isinstance(comparison.get("locked_eval"), dict) else {}
    if non_empty_list(locked_eval, "errors"):
        errors.append("locked_eval_errors_present")
    for variant_name in ("baseline", "candidate"):
        variant = comparison.get(variant_name) if isinstance(comparison.get(variant_name), dict) else {}
        if non_empty_list(variant, "safety_errors"):
            errors.append(f"{variant_name}_safety_errors_present")
    if (
        contamination.get("non_locked_split_used") is True
        or comparison_contamination.get("non_locked_split_used") is True
        or decision.get("non_locked_split_used") is True
    ):
        errors.append("non_locked_split_used")
    if (
        contamination.get("dev_test_mining_rows_used_for_training") is True
        or comparison_contamination.get("dev_test_mining_rows_used_for_training") is True
        or decision.get("dev_test_mining_rows_used_for_training") is True
    ):
        errors.append("dev_test_mining_rows_used_for_training")
    if int(
        contamination.get("critical_leakage_count")
        or comparison_contamination.get("critical_leakage_count")
        or locked_eval.get("critical_leakage_count")
        or 0
    ) != 0:
        errors.append("critical_leakage_present")
    if locked_eval and locked_eval.get("future_r1_r2_r3_r4_must_use_this_split") is not True:
        errors.append("locked_eval_future_split_contract_missing")

    if not has_hardcase_metrics(decision.get("hardcase_holdout")):
        errors.append("missing_decision_hardcase_holdout_metrics")
    if not has_hardcase_metrics(comparison):
        errors.append("missing_comparison_hardcase_holdout_metrics")

    result = {
        "schema": "tensor_audio_error_critic_decision_validation.v1",
        "decision_dir": str(decision_dir),
        "ok": not errors,
        "errors": sorted(set(errors)),
        "release_ready": False,
        "runtime_gates_changed": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
    }
    write_json(decision_dir / VALIDATION_JSON, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a tensor audio error critic variant decision artifact.")
    parser.add_argument("--decision-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tensor_audio_error_critic_decision(args.decision_dir)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
