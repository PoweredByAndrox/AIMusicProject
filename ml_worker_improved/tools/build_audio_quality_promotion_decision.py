from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (  # noqa: E402
    BLOCKED_FILE,
    MANIFEST_FILE,
    MODEL_MANIFEST_SCHEMA,
    PROGRESS_FILE,
    SAFETY_FLAGS,
    VALIDATION_SCHEMA,
    append_progress,
    iter_jsonl,
    read_json,
    stable_hash,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


PROMOTION_DECISION_SCHEMA = "audio_quality_promotion_decision.v1"
SUMMARY_FILE = "summary.md"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _blocked_row(reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "audio_quality_promotion_decision_blocked_row.v1",
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        **SAFETY_FLAGS,
    }


def _read_json_or_error(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"{label}_missing")
        return {}
    try:
        return read_json(path)
    except Exception:
        errors.append(f"{label}_invalid")
        return {}


def _holdout_metrics(holdout_scores_jsonl: Path, errors: list[str]) -> dict[str, int]:
    metrics = {"row_count": 0, "positive_rows": 0, "negative_rows": 0, "false_accept": 0}
    if not holdout_scores_jsonl.exists():
        errors.append("holdout_scores_jsonl_missing")
        return metrics
    try:
        rows = list(iter_jsonl(holdout_scores_jsonl))
    except Exception:
        errors.append("holdout_scores_jsonl_invalid")
        return metrics
    metrics["row_count"] = len(rows)
    for row in rows:
        label = int(row.get("label") or 0)
        accepts = bool(row.get("critic_accepts")) or float(row.get("accept_probability") or 0.0) >= 0.5
        if label > 0:
            metrics["positive_rows"] += 1
        else:
            metrics["negative_rows"] += 1
            if accepts:
                metrics["false_accept"] += 1
    if metrics["row_count"] <= 0:
        errors.append("holdout_scores_empty")
    return metrics


def _model_reasons(model_manifest: dict[str, Any], model_manifest_json: Path) -> list[str]:
    reasons: list[str] = []
    if model_manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        reasons.append("model_manifest_schema_unexpected")
    if model_manifest.get("delivery_allowed") is True:
        reasons.append("model_delivery_allowed_true")
    if model_manifest.get("release_ready") is True:
        reasons.append("model_release_ready_true")
    if model_manifest.get("can_publish_as_result") is True:
        reasons.append("model_can_publish_as_result_true")
    if model_manifest.get("input_tensor_required") is not True:
        reasons.append("model_input_tensor_required_not_true")
    model_file = str(model_manifest.get("model_file") or "").strip()
    if model_manifest.get("training_completed") is True and model_file and not Path(model_file).exists():
        reasons.append("model_file_missing")
    if not str(model_manifest_json).strip():
        reasons.append("model_manifest_json_missing")
    return reasons


def _validation_reasons(validation: dict[str, Any], *, validation_json: Path, model_manifest_json: Path, holdout_scores_jsonl: Path, experiment_audit_json: Path) -> list[str]:
    reasons: list[str] = []
    if validation.get("schema") != VALIDATION_SCHEMA:
        reasons.append("validation_schema_unexpected")
    if validation.get("ok") is not True:
        reasons.append("validation_not_ok")
    if validation.get("promotion_allowed") is not True:
        reasons.append("validation_promotion_allowed_not_true")
    if validation.get("delivery_allowed") is True:
        reasons.append("validation_delivery_allowed_true")
    if validation.get("release_ready") is True:
        reasons.append("validation_release_ready_true")
    if validation.get("require_promotion_ready") is not True:
        reasons.append("validation_require_promotion_ready_not_true")
    if Path(str(validation.get("model_dir") or "")) != model_manifest_json.parent:
        reasons.append("validation_model_dir_mismatch")
    if Path(str(validation.get("holdout_scores_jsonl") or "")) != holdout_scores_jsonl:
        reasons.append("validation_holdout_scores_mismatch")
    if Path(str(validation.get("experiment_audit_json") or "")) != experiment_audit_json:
        reasons.append("validation_experiment_audit_mismatch")
    if not validation_json.exists():
        reasons.append("validation_json_missing")
    return reasons


def _audit_reasons(audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(audit.get("overall_verdict") or "").strip().lower() != "pass":
        reasons.append("experiment_audit_not_pass")
    if str(audit.get("integrity_status") or "").strip().lower() != "pass":
        reasons.append("experiment_audit_integrity_not_pass")
    if not str(audit.get("source_packet_manifest_json") or "").strip():
        reasons.append("experiment_audit_ingest_provenance_missing")
    if audit.get("delivery_allowed") is True:
        reasons.append("experiment_audit_delivery_allowed_true")
    if audit.get("release_ready") is True:
        reasons.append("experiment_audit_release_ready_true")
    if audit.get("can_publish_as_result") is True:
        reasons.append("experiment_audit_can_publish_as_result_true")
    return reasons


def _summary_text(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Audio Quality Promotion Decision",
            "",
            f"Decision: {manifest['promotion_decision']}",
            f"Status: {manifest['decision_status']}",
            f"Runtime authority enabled: {str(manifest['runtime_authority_enabled']).lower()}",
            f"Delivery allowed: {str(manifest['delivery_allowed']).lower()}",
            f"Release ready: {str(manifest['release_ready']).lower()}",
            "",
        ]
    )


def build_audio_quality_promotion_decision(
    *,
    model_manifest_json: Path,
    validation_json: Path,
    holdout_scores_jsonl: Path,
    experiment_audit_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        model_manifest_json=str(model_manifest_json),
        validation_json=str(validation_json),
        holdout_scores_jsonl=str(holdout_scores_jsonl),
        experiment_audit_json=str(experiment_audit_json),
        expected_acceleration_mode="metadata_only_promotion_decision",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="promotion_decision_metadata_only",
    )

    errors: list[str] = []
    model_manifest = _read_json_or_error(model_manifest_json, "model_manifest", errors)
    validation = _read_json_or_error(validation_json, "validation_json", errors)
    audit = _read_json_or_error(experiment_audit_json, "experiment_audit_json", errors)
    holdout_metrics = _holdout_metrics(holdout_scores_jsonl, errors)
    if model_manifest:
        errors.extend(_model_reasons(model_manifest, model_manifest_json))
    if validation:
        errors.extend(
            _validation_reasons(
                validation,
                validation_json=validation_json,
                model_manifest_json=model_manifest_json,
                holdout_scores_jsonl=holdout_scores_jsonl,
                experiment_audit_json=experiment_audit_json,
            )
        )
    if audit:
        errors.extend(_audit_reasons(audit))

    validation_errors = sorted(str(item) for item in _as_list(validation.get("errors")) if item) if validation else []
    errors.extend(validation_errors)
    errors = sorted(set(errors))
    accepted = not errors
    blocked_rows = [] if accepted else [_blocked_row(errors)]
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)

    manifest = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "manifest_type": "audio_quality_promotion_decision",
        "decision_id": "aqp_" + stable_hash([str(model_manifest_json), str(validation_json), str(holdout_scores_jsonl), str(experiment_audit_json)])[:12],
        "decision_status": "accepted" if accepted else "blocked",
        "promotion_decision": "accepted" if accepted else "blocked",
        "promotion_allowed": bool(accepted),
        "runtime_authority_enabled": False,
        "model_manifest_json": str(model_manifest_json),
        "validation_json": str(validation_json),
        "holdout_scores_jsonl": str(holdout_scores_jsonl),
        "experiment_audit_json": str(experiment_audit_json),
        "validation_errors": validation_errors,
        "errors": errors,
        "holdout_metrics": holdout_metrics,
        "model_id": str(model_manifest.get("model_id") or ""),
        "presentation_status": "audio_quality_promotion_decision_not_for_listen_pack",
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "summary_md": str(output_dir / SUMMARY_FILE),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "decision_is_not_delivery": True,
            "decision_is_not_release": True,
            "runtime_authority_requires_separate_gate": True,
            "normalized_experiment_audit_required": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    (output_dir / SUMMARY_FILE).write_text(_summary_text(manifest), encoding="utf-8")
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        decision_status=manifest["decision_status"],
        promotion_allowed=manifest["promotion_allowed"],
        runtime_authority_enabled=False,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an evidence-only audio quality critic promotion decision artifact.")
    parser.add_argument("--model-manifest-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--holdout-scores-jsonl", type=Path, required=True)
    parser.add_argument("--experiment-audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_promotion_decision.py",
        entrypoint_path="/app/tools/build_audio_quality_promotion_decision.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_promotion_decision(
        model_manifest_json=args.model_manifest_json,
        validation_json=args.validation_json,
        holdout_scores_jsonl=args.holdout_scores_jsonl,
        experiment_audit_json=args.experiment_audit_json,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "decision_status": manifest["decision_status"],
                "promotion_allowed": manifest["promotion_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["decision_status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
