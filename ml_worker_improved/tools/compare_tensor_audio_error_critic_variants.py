#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


COMPARISON_JSON = "tensor_audio_error_critic_variant_comparison.json"
COMPARISON_JSONL = "tensor_audio_error_critic_variant_comparison.jsonl"
DECISION_MANIFEST = "tensor_audio_error_critic_decision_manifest.json"
DECISION_SUMMARY = "tensor_audio_error_critic_decision_summary.md"

SCORES_FILE = "tensor_audio_error_shadow_scores.jsonl"
SCORES_SUMMARY_FILE = "tensor_audio_error_shadow_score_summary.json"
ANALYSIS_SUMMARY_FILE = "tensor_audio_error_shadow_analysis_summary.json"

ALLOWED_DECISIONS = [
    "keep_as_specialist_shadow_model",
    "mine_next_dataset_slice",
    "candidate_rejected_runtime",
]

SAFETY_FLAGS = {
    "training_started": False,
    "critic_training_started": False,
    "separator_training_started": False,
    "repair_adapter_training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "runtime_gates_changed": False,
    "release_ready": False,
    "can_publish_as_result": False,
    "shadow_only": True,
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def has_forbidden_true(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(payload.get(key) is True for key in FORBIDDEN_TRUE_FLAGS) or any(
            has_forbidden_true(value) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(has_forbidden_true(item) for item in payload)
    return False


def label_for(row: dict[str, Any]) -> int:
    return int(row.get("true_label", row.get("label", 0)) or 0)


def accepts(row: dict[str, Any]) -> bool:
    if "would_accept_at_calibrated_threshold" in row:
        return row.get("would_accept_at_calibrated_threshold") is True
    return str(row.get("shadow_decision") or "") == "shadow_accept_candidate"


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def metrics_for_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if label_for(row) == 1]
    negatives = [row for row in rows if label_for(row) == 0]
    accepted = [row for row in rows if accepts(row)]
    false_accept = [row for row in negatives if accepts(row)]
    false_reject = [row for row in positives if not accepts(row)]
    true_accept = [row for row in positives if accepts(row)]
    correct = sum(1 for row in rows if (label_for(row) == 1 and accepts(row)) or (label_for(row) == 0 and not accepts(row)))
    return {
        "rows": len(rows),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "accepted_rows": len(accepted),
        "true_accept": len(true_accept),
        "false_accept": len(false_accept),
        "false_reject": len(false_reject),
        "hard_negative_count": len(false_accept),
        "false_reject_count": len(false_reject),
        "accuracy": safe_ratio(correct, len(rows)),
        "false_accept_rate": safe_ratio(len(false_accept), len(negatives)),
        "false_reject_rate": safe_ratio(len(false_reject), len(positives)),
        "accept_coverage": safe_ratio(len(true_accept), len(positives)) or 0.0,
    }


def group_metrics(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {value: metrics_for_rows(items) for value, items in sorted(grouped.items())}


def normalize_path_text(path_text: str) -> str:
    return str(Path(path_text)).replace("\\", "/").rstrip("/")


def validate_locked_eval(locked_eval_dir: Path) -> dict[str, Any]:
    manifest = read_json(locked_eval_dir / "leak_safe_split_manifest.json")
    leakage = read_json(locked_eval_dir / "leakage_report.json")
    holdout = read_json(locked_eval_dir / "hardcase_holdout_manifest.json")
    hardcase_rows = holdout.get("hardcase_rows") if isinstance(holdout.get("hardcase_rows"), list) else []
    errors: list[str] = []
    if manifest.get("schema") != "leak_safe_audio_eval_harness.v1":
        errors.append("invalid_locked_eval_manifest_schema")
    if int(manifest.get("critical_leakage_count") or 0) != 0:
        errors.append("locked_eval_critical_leakage_present")
    if int(leakage.get("critical_leakage_count") or 0) != 0:
        errors.append("leakage_report_critical_leakage_present")
    if holdout.get("locked") is not True:
        errors.append("hardcase_holdout_not_locked")
    if manifest.get("future_r1_r2_r3_r4_must_use_this_split") is not True:
        errors.append("future_r1_r2_r3_r4_not_locked_to_split")
    if has_forbidden_true({"manifest": manifest, "leakage": leakage, "holdout": holdout}):
        errors.append("locked_eval_forbidden_true_flag")
    return {
        "manifest": manifest,
        "leakage": leakage,
        "holdout": holdout,
        "hardcase_row_uids": {str(row.get("row_uid") or "") for row in hardcase_rows if row.get("row_uid")},
        "errors": errors,
    }


def load_tensor_dataset_contract(model_manifest: dict[str, Any], locked_eval_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(str(model_manifest.get("source_dataset_dir") or ""))
    manifest_path = dataset_dir / "locked_tensor_audio_error_dataset_manifest.json"
    if not manifest_path.exists():
        return {
            "available": False,
            "non_locked_split_used": True,
            "errors": [f"missing_tensor_dataset_manifest:{manifest_path}"],
        }
    manifest = read_json(manifest_path)
    expected_split_manifest = locked_eval_dir / "leak_safe_split_manifest.json"
    non_locked = False
    errors: list[str] = []
    if manifest.get("schema") != "locked_tensor_audio_error_dataset.v1":
        non_locked = True
        errors.append("invalid_tensor_dataset_schema")
    if int(manifest.get("critical_leakage_count") or 0) != 0:
        non_locked = True
        errors.append("tensor_dataset_critical_leakage_present")
    if manifest.get("release_ready") is True:
        errors.append("tensor_dataset_release_ready_true")
    if normalize_path_text(str(manifest.get("source_split_manifest") or "")) != normalize_path_text(str(expected_split_manifest)):
        non_locked = True
        errors.append("tensor_dataset_source_split_manifest_mismatch")
    return {
        "available": True,
        "manifest_path": str(manifest_path),
        "source_split_manifest": str(manifest.get("source_split_manifest") or ""),
        "source_split_dataset_id": str(manifest.get("source_split_dataset_id") or ""),
        "non_locked_split_used": non_locked,
        "errors": errors,
    }


def load_variant(
    *,
    name: str,
    locked_eval_dir: Path,
    hardcase_row_uids: set[str],
    model_dir: Path,
    scores_dir: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    model_manifest = read_json(model_dir / "tensor_audio_error_critic_manifest.json")
    scores_summary = read_json(scores_dir / SCORES_SUMMARY_FILE)
    score_rows = list(iter_jsonl(scores_dir / SCORES_FILE))
    analysis = read_json(analysis_dir / ANALYSIS_SUMMARY_FILE)
    safety_errors: list[str] = []
    if model_manifest.get("release_ready") is True:
        safety_errors.append("model_release_ready_true")
    if model_manifest.get("separator_training_started") is True:
        safety_errors.append("model_separator_training_started_true")
    if model_manifest.get("repair_adapter_training_started") is True:
        safety_errors.append("model_repair_adapter_training_started_true")
    if has_forbidden_true({"scores_summary": scores_summary, "score_rows": score_rows, "analysis": analysis}):
        safety_errors.append("score_or_analysis_forbidden_true_flag")

    dataset_contract = load_tensor_dataset_contract(model_manifest, locked_eval_dir)
    locked_eval_rows = [row for row in score_rows if str(row.get("split") or "") in {"dev", "test"}]
    if not locked_eval_rows:
        safety_errors.append("missing_locked_eval_score_rows")
    hardcase_rows = [row for row in locked_eval_rows if str(row.get("row_uid") or "") in hardcase_row_uids]
    locked_eval_metrics = metrics_for_rows(locked_eval_rows)
    all_score_metrics = metrics_for_rows(score_rows)
    analysis_hard_negative_count = int(analysis.get("hard_negative_count", all_score_metrics["hard_negative_count"]) or 0)
    analysis_false_reject_count = int(analysis.get("false_reject_count", all_score_metrics["false_reject_count"]) or 0)
    analysis_negative_rows = int(analysis.get("negative_rows", all_score_metrics["negative_rows"]) or 0)
    analysis_positive_rows = int(analysis.get("positive_rows", all_score_metrics["positive_rows"]) or 0)
    threshold_metrics = model_manifest.get("threshold_metrics") if isinstance(model_manifest.get("threshold_metrics"), dict) else {}
    eval_metrics = model_manifest.get("eval_metrics") if isinstance(model_manifest.get("eval_metrics"), dict) else {}
    mining = {
        "mining_bundle_used": model_manifest.get("mining_bundle_used") is True,
        "mining_candidate_rows_total": int(model_manifest.get("mining_candidate_rows_total") or 0),
        "mining_train_rows_available": int(model_manifest.get("mining_train_rows_available") or 0),
        "mining_train_rows_matched": int(model_manifest.get("mining_train_rows_matched") or 0),
        "mining_train_rows_unmatched": int(model_manifest.get("mining_train_rows_unmatched") or 0),
        "mining_non_train_rows_ignored": int(model_manifest.get("mining_non_train_rows_ignored") or 0),
        "dev_test_mining_rows_used_for_training": model_manifest.get("dev_test_mining_rows_used_for_training") is True,
        "mining_non_train_rows_used_for_training": int(model_manifest.get("mining_non_train_rows_used_for_training") or 0),
        "mining_dev_test_rows_used_for_training": int(model_manifest.get("mining_dev_test_rows_used_for_training") or 0),
        "mining_sample_weight_applied": model_manifest.get("mining_sample_weight_applied") is True,
        "mining_weight_multiplier": model_manifest.get("mining_weight_multiplier"),
    }
    return {
        "variant": name,
        "model_dir": str(model_dir),
        "scores_dir": str(scores_dir),
        "analysis_dir": str(analysis_dir),
        "model_kind": str(model_manifest.get("model_kind") or analysis.get("model_kind") or scores_summary.get("model_kind") or ""),
        "calibration_status": str(analysis.get("calibration_status") or scores_summary.get("calibration_status") or ""),
        "auroc": eval_metrics.get("auroc"),
        "accuracy": eval_metrics.get("accuracy"),
        "hard_negative_count": locked_eval_metrics["hard_negative_count"],
        "false_reject_count": locked_eval_metrics["false_reject_count"],
        "false_accept_rate": locked_eval_metrics["false_accept_rate"],
        "false_reject_rate": locked_eval_metrics["false_reject_rate"],
        "accept_coverage": locked_eval_metrics["accept_coverage"],
        "model_eval_metrics": eval_metrics,
        "locked_eval_metrics": locked_eval_metrics,
        "all_shadow_score_metrics": all_score_metrics,
        "train_rows_ignored_for_locked_eval": len(score_rows) - len(locked_eval_rows),
        "shadow_analysis_summary_metrics": {
            "rows_analyzed": int(analysis.get("rows_analyzed") or 0),
            "hard_negative_count": analysis_hard_negative_count,
            "false_reject_count": analysis_false_reject_count,
            "false_accept_rate": float(analysis_hard_negative_count / analysis_negative_rows) if analysis_negative_rows else None,
            "false_reject_rate": float(analysis_false_reject_count / analysis_positive_rows) if analysis_positive_rows else None,
            "used_for_decision": False,
        },
        "threshold_metrics": threshold_metrics,
        "hardcase_holdout": metrics_for_rows(hardcase_rows),
        "per_stem_metrics": group_metrics(locked_eval_rows, "stem_type"),
        "per_route_metrics": group_metrics(locked_eval_rows, "route_id"),
        "per_track_metrics": group_metrics(locked_eval_rows, "track_id"),
        "mining": mining,
        "tensor_dataset_contract": dataset_contract,
        "safety_errors": [*safety_errors, *dataset_contract.get("errors", [])],
        "release_ready": False,
        "runtime_gates_changed": False,
        "shadow_only": True,
    }


def numeric_delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    try:
        return float(candidate) - float(baseline)
    except (TypeError, ValueError):
        return None


def metric_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "auroc",
        "accuracy",
        "hard_negative_count",
        "false_reject_count",
        "false_accept_rate",
        "false_reject_rate",
        "accept_coverage",
    ]
    return {key: numeric_delta(candidate.get(key), baseline.get(key)) for key in keys}


def hardcase_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        key: numeric_delta(candidate.get("hardcase_holdout", {}).get(key), baseline.get("hardcase_holdout", {}).get(key))
        for key in [
            "rows",
            "hard_negative_count",
            "false_reject_count",
            "false_accept_rate",
            "false_reject_rate",
            "accept_coverage",
            "accuracy",
        ]
    }


def contamination_status(
    *,
    locked_eval: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    def mining_uses_dev_test_rows(variant: dict[str, Any]) -> bool:
        mining = variant.get("mining", {})
        return bool(
            mining.get("dev_test_mining_rows_used_for_training")
            or int(mining.get("mining_non_train_rows_used_for_training") or 0) > 0
            or int(mining.get("mining_dev_test_rows_used_for_training") or 0) > 0
        )

    candidate_mining = candidate.get("mining", {})
    baseline_mining = baseline.get("mining", {})
    baseline_dev_test_mining_rows_used_for_training = mining_uses_dev_test_rows(baseline)
    candidate_dev_test_mining_rows_used_for_training = mining_uses_dev_test_rows(candidate)
    dev_test_mining_rows_used_for_training = (
        baseline_dev_test_mining_rows_used_for_training or candidate_dev_test_mining_rows_used_for_training
    )
    non_locked_split_used = bool(
        baseline.get("tensor_dataset_contract", {}).get("non_locked_split_used")
        or candidate.get("tensor_dataset_contract", {}).get("non_locked_split_used")
    )
    return {
        "locked_eval_dir": str(locked_eval["manifest"].get("output_dir") or ""),
        "critical_leakage_count": int(locked_eval["manifest"].get("critical_leakage_count") or 0),
        "leakage_report_critical_leakage_count": int(locked_eval["leakage"].get("critical_leakage_count") or 0),
        "hardcase_holdout_locked": locked_eval["holdout"].get("locked") is True,
        "future_r1_r2_r3_r4_must_use_this_split": locked_eval["manifest"].get("future_r1_r2_r3_r4_must_use_this_split") is True,
        "non_locked_split_used": non_locked_split_used,
        "dev_test_mining_rows_used_for_training": dev_test_mining_rows_used_for_training,
        "baseline_dev_test_mining_rows_used_for_training": baseline_dev_test_mining_rows_used_for_training,
        "candidate_dev_test_mining_rows_used_for_training": candidate_dev_test_mining_rows_used_for_training,
        "baseline_mining_rows_used": baseline_mining,
        "candidate_mining_rows_used": candidate_mining,
        "baseline_tensor_dataset_contract": baseline.get("tensor_dataset_contract", {}),
        "candidate_tensor_dataset_contract": candidate.get("tensor_dataset_contract", {}),
    }


def choose_decision(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    deltas: dict[str, Any],
    hardcase_deltas: dict[str, Any],
    contamination: dict[str, Any],
    locked_eval_errors: Sequence[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    safety_failed = bool(locked_eval_errors or baseline.get("safety_errors") or candidate.get("safety_errors"))
    safety_failed = safety_failed or contamination.get("non_locked_split_used") is True
    safety_failed = safety_failed or contamination.get("dev_test_mining_rows_used_for_training") is True
    safety_failed = safety_failed or int(contamination.get("critical_leakage_count") or 0) != 0
    if safety_failed:
        reasons.extend([*locked_eval_errors, *baseline.get("safety_errors", []), *candidate.get("safety_errors", [])])
        reasons.append("safety_flags_failed")
        return "candidate_rejected_runtime", sorted(set(reasons))

    hardcase_improved = (
        (hardcase_deltas.get("hard_negative_count") is not None and float(hardcase_deltas["hard_negative_count"]) < 0)
        or (hardcase_deltas.get("false_reject_count") is not None and float(hardcase_deltas["false_reject_count"]) < 0)
        or (hardcase_deltas.get("accuracy") is not None and float(hardcase_deltas["accuracy"]) > 0)
    )
    false_accept_not_regressed = (
        (candidate.get("false_accept_rate") is None or baseline.get("false_accept_rate") is None)
        or float(candidate["false_accept_rate"]) <= float(baseline["false_accept_rate"])
    )
    auroc_dropped = deltas.get("auroc") is not None and float(deltas["auroc"]) < 0
    global_hard_negatives_improved = deltas.get("hard_negative_count") is not None and float(deltas["hard_negative_count"]) < 0
    global_false_rejects_improved = deltas.get("false_reject_count") is not None and float(deltas["false_reject_count"]) < 0

    if hardcase_improved and false_accept_not_regressed:
        reasons.append("hardcase_holdout_improved_without_false_accept_regression")
        if auroc_dropped:
            reasons.append("auroc_dropped_keep_shadow_no_promotion")
        return "keep_as_specialist_shadow_model", reasons
    if global_false_rejects_improved and not hardcase_improved:
        return "mine_next_dataset_slice", ["false_rejects_improved_but_hardcase_holdout_did_not_improve"]
    if global_hard_negatives_improved and auroc_dropped:
        return "mine_next_dataset_slice", ["hard_negatives_improved_but_auroc_dropped"]
    return "mine_next_dataset_slice", ["mixed_or_tiny_improvements_require_more_locked_mining"]


def write_decision_summary(path: Path, decision_manifest: dict[str, Any], comparison: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Critic Variant Decision",
        "",
        f"- decision: `{decision_manifest['decision']}`",
        f"- decision_reasons: `{json.dumps(decision_manifest['decision_reasons'], sort_keys=True)}`",
        f"- baseline_auroc: {comparison['baseline'].get('auroc')}",
        f"- candidate_auroc: {comparison['candidate'].get('auroc')}",
        f"- hard_negative_delta: {comparison['delta'].get('hard_negative_count')}",
        f"- false_reject_delta: {comparison['delta'].get('false_reject_count')}",
        f"- hardcase_holdout_delta: `{json.dumps(comparison['delta'].get('hardcase_holdout', {}), sort_keys=True)}`",
        "- promotion_allowed: false",
        "- runtime_gate_update_allowed: false",
        "- runtime_gates_changed: false",
        "- release_ready: false",
        "- separator_training_started: false",
        "- repair_adapter_training_started: false",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_tensor_audio_error_critic_variants(
    *,
    locked_eval_dir: Path,
    baseline_model_dir: Path,
    baseline_scores_dir: Path,
    baseline_analysis_dir: Path,
    candidate_model_dir: Path,
    candidate_scores_dir: Path,
    candidate_analysis_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    locked_eval = validate_locked_eval(locked_eval_dir)
    baseline = load_variant(
        name="baseline",
        locked_eval_dir=locked_eval_dir,
        hardcase_row_uids=locked_eval["hardcase_row_uids"],
        model_dir=baseline_model_dir,
        scores_dir=baseline_scores_dir,
        analysis_dir=baseline_analysis_dir,
    )
    candidate = load_variant(
        name="candidate",
        locked_eval_dir=locked_eval_dir,
        hardcase_row_uids=locked_eval["hardcase_row_uids"],
        model_dir=candidate_model_dir,
        scores_dir=candidate_scores_dir,
        analysis_dir=candidate_analysis_dir,
    )
    deltas = metric_deltas(candidate, baseline)
    deltas["hardcase_holdout"] = hardcase_delta(candidate, baseline)
    contamination = contamination_status(locked_eval=locked_eval, baseline=baseline, candidate=candidate)
    decision, reasons = choose_decision(
        baseline=baseline,
        candidate=candidate,
        deltas=deltas,
        hardcase_deltas=deltas["hardcase_holdout"],
        contamination=contamination,
        locked_eval_errors=locked_eval["errors"],
    )
    comparison = {
        "schema": "tensor_audio_error_critic_variant_comparison.v1",
        "locked_eval": {
            "locked_eval_dir": str(locked_eval_dir),
            "schema": locked_eval["manifest"].get("schema"),
            "source_dataset_id": locked_eval["manifest"].get("source_dataset_id"),
            "critical_leakage_count": int(locked_eval["manifest"].get("critical_leakage_count") or 0),
            "warning_leakage_count": int(locked_eval["manifest"].get("warning_leakage_count") or 0),
            "hardcase_rows_count": int(locked_eval["holdout"].get("hardcase_rows_count") or 0),
            "future_r1_r2_r3_r4_must_use_this_split": locked_eval["manifest"].get("future_r1_r2_r3_r4_must_use_this_split") is True,
            "errors": locked_eval["errors"],
        },
        "baseline": baseline,
        "candidate": candidate,
        "delta": deltas,
        "contamination_status": contamination,
        "runtime_gates_changed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "shadow_only": True,
    }
    decision_manifest = {
        "schema": "tensor_audio_error_critic_decision.v1",
        "comparison_path": str(output_dir / COMPARISON_JSON),
        "decision": decision,
        "decision_allowed_values": ALLOWED_DECISIONS,
        "decision_reasons": reasons,
        "promotion_allowed": False,
        "runtime_gate_update_allowed": False,
        "runtime_gates_changed": False,
        "contamination_status": contamination,
        "hardcase_holdout": {
            "baseline": baseline["hardcase_holdout"],
            "candidate": candidate["hardcase_holdout"],
            "delta": deltas["hardcase_holdout"],
        },
        "mining_rows_used": candidate.get("mining", {}),
        "non_locked_split_used": contamination["non_locked_split_used"],
        "dev_test_mining_rows_used_for_training": contamination["dev_test_mining_rows_used_for_training"],
        **SAFETY_FLAGS,
    }
    write_json(output_dir / COMPARISON_JSON, comparison)
    write_jsonl(
        output_dir / COMPARISON_JSONL,
        [
            {"variant": "baseline", **{key: baseline.get(key) for key in ["auroc", "accuracy", "hard_negative_count", "false_reject_count", "false_accept_rate", "false_reject_rate", "accept_coverage"]}, "release_ready": False},
            {"variant": "candidate", **{key: candidate.get(key) for key in ["auroc", "accuracy", "hard_negative_count", "false_reject_count", "false_accept_rate", "false_reject_rate", "accept_coverage"]}, "release_ready": False},
            {"variant": "delta", **deltas, "release_ready": False},
        ],
    )
    write_json(output_dir / DECISION_MANIFEST, decision_manifest)
    write_decision_summary(output_dir / DECISION_SUMMARY, decision_manifest, comparison)
    print(json.dumps(decision_manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return decision_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tensor audio error critic variants on the locked eval split.")
    parser.add_argument("--locked-eval-dir", type=Path, required=True)
    parser.add_argument("--baseline-model-dir", type=Path, required=True)
    parser.add_argument("--baseline-scores-dir", type=Path, required=True)
    parser.add_argument("--baseline-analysis-dir", type=Path, required=True)
    parser.add_argument("--candidate-model-dir", type=Path, required=True)
    parser.add_argument("--candidate-scores-dir", type=Path, required=True)
    parser.add_argument("--candidate-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compare_tensor_audio_error_critic_variants(
        locked_eval_dir=args.locked_eval_dir,
        baseline_model_dir=args.baseline_model_dir,
        baseline_scores_dir=args.baseline_scores_dir,
        baseline_analysis_dir=args.baseline_analysis_dir,
        candidate_model_dir=args.candidate_model_dir,
        candidate_scores_dir=args.candidate_scores_dir,
        candidate_analysis_dir=args.candidate_analysis_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
