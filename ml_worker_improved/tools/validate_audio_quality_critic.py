from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    DATASET_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    MODEL_MANIFEST_SCHEMA,
    PROMOTION_LABEL_TYPES,
    VALIDATION_SCHEMA,
    iter_jsonl,
    read_json,
    write_json,
)
from tools.docker_runtime import require_docker_runtime

REQUIRED_EXPERIMENT_AUDIT_CHECKS = (
    "gt_provenance",
    "score_normalization",
    "result_existence",
    "dead_code",
    "scope",
    "eval_type",
)
EXPERIMENT_AUDIT_PACKET_SCHEMA = "audio_quality_experiment_audit_packet.v1"


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_identity_values(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("track_id", "candidate_id", "row_id"):
        value = str(row.get(key) or "").strip()
        if value:
            values.add(f"{key}:{value}")
    for ref in _as_list(row.get("source_manifest_refs")):
        value = str(ref or "").strip()
        if value:
            values.add(f"source_manifest_ref:{value}")
    return values


def _critic_accepts(row: dict[str, Any], errors: list[str]) -> bool:
    if "critic_accepts" in row:
        return bool(row.get("critic_accepts"))
    if "accept_probability" in row:
        return float(row.get("accept_probability") or 0.0) >= 0.5
    errors.append("holdout_prediction_missing")
    return False


def _load_training_identities(manifest: dict[str, Any], errors: list[str]) -> set[str]:
    dataset_manifest_value = str(manifest.get("dataset_manifest_json") or "").strip()
    if not dataset_manifest_value:
        errors.append("dataset_manifest_json_missing")
        return set()

    dataset_manifest_path = Path(dataset_manifest_value)
    if not dataset_manifest_path.exists():
        errors.append("dataset_manifest_missing")
        return set()

    dataset_manifest = read_json(dataset_manifest_path)
    dataset_jsonl_value = ""
    outputs = dataset_manifest.get("outputs")
    if isinstance(outputs, dict):
        dataset_jsonl_value = str(outputs.get("dataset_jsonl") or outputs.get("dataset") or "").strip()
    if not dataset_jsonl_value:
        dataset_jsonl_value = str(dataset_manifest.get("dataset_jsonl") or "").strip()
    dataset_jsonl_path = Path(dataset_jsonl_value) if dataset_jsonl_value else dataset_manifest_path.parent / DATASET_FILE
    if not dataset_jsonl_path.exists():
        errors.append("dataset_jsonl_missing")
        return set()

    identities: set[str] = set()
    for row in iter_jsonl(dataset_jsonl_path):
        if str(row.get("split") or "").strip().lower() == "train":
            identities.update(_row_identity_values(row))
    return identities


def _validate_holdout_scores(
    manifest: dict[str, Any],
    holdout_scores_jsonl: Path,
    *,
    max_false_accept_rate: float,
    min_holdout_rows: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    metrics: dict[str, Any] = {
        "rows": 0,
        "promotion_label_rows": 0,
        "non_promotion_label_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "false_accept": 0,
        "false_accept_rate": None,
        "split_leakage_count": 0,
        "hard_case_holdout_count": 0,
        "missing_rag_ref_count": 0,
        "missing_source_manifest_ref_count": 0,
        "source_lock_mismatch_count": 0,
    }
    if not holdout_scores_jsonl.exists():
        return metrics, ["missing_holdout_scores_jsonl"]

    source_lock_pack_json = str(manifest.get("source_lock_pack_json") or "").strip()
    if not source_lock_pack_json:
        errors.append("source_lock_pack_json_missing")
    training_identities = _load_training_identities(manifest, errors)

    for row in iter_jsonl(holdout_scores_jsonl):
        metrics["rows"] += 1
        label_type = str(row.get("label_type") or "").strip()
        label = int(row.get("label") or 0)
        split = str(row.get("split") or "").strip().lower()
        if label_type not in PROMOTION_LABEL_TYPES:
            metrics["non_promotion_label_rows"] += 1
            errors.append("non_promotion_label_in_holdout")
        else:
            metrics["promotion_label_rows"] += 1
        if split == "train":
            errors.append("holdout_split_marked_train")
        if not _as_list(row.get("rag_refs")):
            metrics["missing_rag_ref_count"] += 1
            errors.append("holdout_rag_refs_missing")
        if not _as_list(row.get("source_manifest_refs")):
            metrics["missing_source_manifest_ref_count"] += 1
            errors.append("holdout_source_manifest_refs_missing")
        row_source_lock = str(row.get("source_lock_pack_json") or "").strip()
        if not row_source_lock:
            errors.append("holdout_source_lock_missing")
        elif source_lock_pack_json and row_source_lock != source_lock_pack_json:
            metrics["source_lock_mismatch_count"] += 1
            errors.append("source_lock_mismatch")
        if _row_identity_values(row) & training_identities:
            metrics["split_leakage_count"] += 1
            errors.append("split_leakage_detected")

        accepted = _critic_accepts(row, errors)
        if label > 0:
            metrics["positive_rows"] += 1
        else:
            metrics["negative_rows"] += 1
            if _as_list(row.get("blocker_reasons")) or _as_list(row.get("issue_tags")):
                metrics["hard_case_holdout_count"] += 1
            if accepted:
                metrics["false_accept"] += 1

    if metrics["rows"] < int(min_holdout_rows):
        errors.append("locked_holdout_rows_below_minimum")
    if metrics["positive_rows"] <= 0:
        errors.append("positive_holdout_missing")
    if metrics["negative_rows"] <= 0:
        errors.append("negative_holdout_missing")
    if metrics["hard_case_holdout_count"] <= 0:
        errors.append("hard_case_holdout_missing")
    if metrics["negative_rows"] > 0:
        metrics["false_accept_rate"] = float(metrics["false_accept"]) / float(metrics["negative_rows"])
        if float(metrics["false_accept_rate"]) > float(max_false_accept_rate):
            errors.append("false_accept_rate_above_budget")
    return metrics, errors


def _validate_experiment_audit(experiment_audit_json: Path | None) -> tuple[str, list[str]]:
    if experiment_audit_json is None:
        return "missing", ["missing_experiment_audit_json"]
    if not experiment_audit_json.exists():
        return "missing", ["experiment_audit_json_missing"]

    try:
        audit = read_json(experiment_audit_json)
    except Exception:
        return "invalid", ["experiment_audit_json_invalid"]

    errors: list[str] = []
    verdict = _normalize_status(audit.get("overall_verdict"))
    integrity_status = _normalize_status(audit.get("integrity_status"))
    if audit.get("delivery_allowed") is True:
        errors.append("experiment_audit_delivery_allowed_true")
    if audit.get("release_ready") is True:
        errors.append("experiment_audit_release_ready_true")
    if audit.get("can_publish_as_result") is True:
        errors.append("experiment_audit_can_publish_as_result_true")
    if not str(audit.get("auditor") or "").strip():
        errors.append("experiment_auditor_missing")
    if not str(audit.get("reviewer_trace_ref") or "").strip():
        errors.append("reviewer_trace_ref_missing")
    source_packet_manifest_json = str(audit.get("source_packet_manifest_json") or "").strip()
    source_audit_json = str(audit.get("source_audit_json") or "").strip()
    try:
        source_packet_path_count = int(audit.get("source_packet_path_count") or 0)
    except (TypeError, ValueError):
        source_packet_path_count = 0
        errors.append("experiment_audit_packet_path_count_invalid")
    if not source_packet_manifest_json or not source_audit_json or source_packet_path_count <= 0:
        errors.append("experiment_audit_ingest_provenance_missing")
    else:
        packet_manifest_path = Path(source_packet_manifest_json)
        source_audit_path = Path(source_audit_json)
        if not packet_manifest_path.exists():
            errors.append("experiment_audit_packet_manifest_missing")
        else:
            try:
                packet_manifest = read_json(packet_manifest_path)
            except Exception:
                errors.append("experiment_audit_packet_manifest_invalid")
            else:
                if packet_manifest.get("schema") != EXPERIMENT_AUDIT_PACKET_SCHEMA:
                    errors.append("experiment_audit_packet_schema_unexpected")
                if packet_manifest.get("packet_status") != "ready_for_independent_review":
                    errors.append("experiment_audit_packet_not_ready")
                if packet_manifest.get("experiment_audit_pass") is True:
                    errors.append("experiment_audit_packet_claims_pass")
                if packet_manifest.get("delivery_allowed") is True:
                    errors.append("experiment_audit_packet_delivery_allowed_true")
                if packet_manifest.get("release_ready") is True:
                    errors.append("experiment_audit_packet_release_ready_true")
                if int(packet_manifest.get("path_count") or 0) != source_packet_path_count:
                    errors.append("experiment_audit_packet_path_count_mismatch")
        if not source_audit_path.exists():
            errors.append("experiment_audit_source_audit_json_missing")
    checks = audit.get("checks")
    if not isinstance(checks, dict) or not checks:
        errors.append("experiment_audit_checks_missing")
    else:
        for check_name in REQUIRED_EXPERIMENT_AUDIT_CHECKS:
            check = checks.get(check_name)
            if not isinstance(check, dict):
                errors.append("required_audit_check_missing")
                continue
            if _normalize_status(check.get("status")) != "pass":
                errors.append("audit_check_not_pass")
            if not _as_list(check.get("evidence")):
                errors.append("audit_check_evidence_missing")
    if verdict != "pass" or integrity_status != "pass":
        errors.append("experiment_audit_not_pass")
    return integrity_status or verdict or "unknown", errors


def validate_audio_quality_critic(
    model_dir: Path,
    *,
    require_promotion_ready: bool = False,
    max_false_accept_rate: float = 0.0,
    holdout_scores_jsonl: Path | None = None,
    min_holdout_rows: int = 2,
    experiment_audit_json: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = model_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return {"schema": VALIDATION_SCHEMA, "ok": False, "errors": ["missing_model_manifest"], "model_dir": str(model_dir)}
    manifest = read_json(manifest_path)
    if manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        errors.append("invalid_model_manifest_schema")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_true")
    if manifest.get("delivery_allowed") is True:
        errors.append("delivery_allowed_true")
    if manifest.get("model_registry_edited") is True:
        errors.append("model_registry_edited_true")
    if manifest.get("runtime_gates_changed") is True:
        errors.append("runtime_gates_changed_true")
    if manifest.get("thresholds_relaxed") is True:
        errors.append("thresholds_relaxed_true")
    if manifest.get("sklearn_used") is True:
        errors.append("sklearn_used_true")
    if manifest.get("proxy_only_model") is True:
        errors.append("proxy_only_model_true")
    if manifest.get("input_tensor_required") is not True:
        errors.append("input_tensor_required_not_true")
    if manifest.get("training_completed") is True and not (model_dir / MODEL_FILE).exists():
        errors.append("missing_model_file")
    if manifest.get("critical_leakage_count", 0) not in {0, None}:
        errors.append("critical_leakage_detected")
    if manifest.get("rag_refs_complete") is not True:
        errors.append("rag_refs_missing")
    false_accept = manifest.get("false_accept_rate")
    if false_accept is not None and float(false_accept) > float(max_false_accept_rate):
        errors.append("false_accept_rate_above_budget")
    label_counts = manifest.get("label_type_counts") if isinstance(manifest.get("label_type_counts"), dict) else {}
    verified_rows = int(manifest.get("verified_training_row_count") or 0)
    holdout_metrics: dict[str, Any] = {}
    experiment_audit_status = ""
    has_holdout_scores = holdout_scores_jsonl is not None
    if require_promotion_ready:
        if verified_rows <= 0:
            errors.append("verified_training_rows_missing")
        if not any(int(label_counts.get(label_type, 0) or 0) > 0 for label_type in PROMOTION_LABEL_TYPES):
            errors.append("promotion_label_type_missing")
        if str(manifest.get("device") or "").lower() != "cuda":
            errors.append("gpu_training_not_verified")
        if has_holdout_scores:
            holdout_metrics, holdout_errors = _validate_holdout_scores(
                manifest,
                holdout_scores_jsonl,
                max_false_accept_rate=float(max_false_accept_rate),
                min_holdout_rows=int(min_holdout_rows),
            )
            errors.extend(holdout_errors)
        else:
            errors.append("missing_holdout_scores_jsonl")
            if manifest.get("shadow_only") is not False:
                errors.append("model_is_shadow_only")
            if manifest.get("promotion_allowed") is not True:
                errors.append("promotion_allowed_not_true")
        experiment_audit_status, experiment_audit_errors = _validate_experiment_audit(experiment_audit_json)
        errors.extend(experiment_audit_errors)
    return {
        "schema": VALIDATION_SCHEMA,
        "ok": not errors,
        "errors": sorted(set(errors)),
        "model_dir": str(model_dir),
        "require_promotion_ready": bool(require_promotion_ready),
        "max_false_accept_rate": float(max_false_accept_rate),
        "holdout_scores_jsonl": str(holdout_scores_jsonl) if holdout_scores_jsonl else "",
        "holdout_metrics": holdout_metrics,
        "experiment_audit_json": str(experiment_audit_json) if experiment_audit_json else "",
        "experiment_audit_status": experiment_audit_status,
        "promotion_allowed": bool(require_promotion_ready and has_holdout_scores and not errors),
        "release_ready": False,
        "delivery_allowed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate audio quality critic safety and promotion contract.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-promotion-ready", action="store_true")
    parser.add_argument("--max-false-accept-rate", type=float, default=0.0)
    parser.add_argument("--holdout-scores-jsonl", type=Path)
    parser.add_argument("--min-holdout-rows", type=int, default=2)
    parser.add_argument("--experiment-audit-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="validate_audio_quality_critic.py", entrypoint_path="/app/tools/validate_audio_quality_critic.py")
    args = parse_args(argv)
    result = validate_audio_quality_critic(
        args.model_dir,
        require_promotion_ready=bool(args.require_promotion_ready),
        max_false_accept_rate=float(args.max_false_accept_rate),
        holdout_scores_jsonl=args.holdout_scores_jsonl,
        min_holdout_rows=int(args.min_holdout_rows),
        experiment_audit_json=args.experiment_audit_json,
    )
    if args.output_json:
        write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
