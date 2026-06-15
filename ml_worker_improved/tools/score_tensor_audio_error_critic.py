#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from tools.train_tensor_audio_error_critic import (  # noqa: E402
    REQUIRED_MAP_NAMES,
    load_rows,
    read_json,
    write_json,
    write_jsonl,
)
from tools.validate_tensor_audio_error_critic import validate_tensor_audio_error_critic  # noqa: E402


SCORES_FILE = "tensor_audio_error_shadow_scores.jsonl"
SUMMARY_FILE = "tensor_audio_error_shadow_score_summary.json"

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


def validate_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest = read_json(dataset_dir / "locked_tensor_audio_error_dataset_manifest.json")
    errors: list[str] = []
    if manifest.get("schema") != "locked_tensor_audio_error_dataset.v1":
        errors.append("invalid_tensor_dataset_schema")
    if manifest.get("tensor_maps_required") is not True:
        errors.append("tensor_maps_not_required")
    if manifest.get("scalar_fallback_used") is True:
        errors.append("scalar_fallback_used")
    if int(manifest.get("critical_leakage_count") or 0) != 0:
        errors.append("critical_leakage_present")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_dataset_not_allowed")
    if errors:
        raise ValueError(f"unsafe_tensor_dataset_manifest:{','.join(errors)}")
    return manifest


def load_model_bundle(model_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    model_file = str(manifest.get("model_file") or "tensor_audio_error_critic.joblib")
    bundle = joblib.load(model_dir / model_file)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("invalid_tensor_critic_model_bundle")
    required = [str(item) for item in bundle.get("required_map_names", []) or []]
    if sorted(required) != sorted(REQUIRED_MAP_NAMES):
        raise ValueError("required_tensor_map_names_mismatch")
    return bundle


def select_threshold(calibration: dict[str, Any], stem_type: str) -> tuple[str, dict[str, Any]]:
    by_stem = calibration.get("by_stem") if isinstance(calibration.get("by_stem"), dict) else {}
    stem_payload = by_stem.get(stem_type)
    if isinstance(stem_payload, dict) and stem_payload.get("status") == "usable" and stem_payload.get("threshold") is not None:
        return f"stem:{stem_type}", stem_payload
    global_payload = calibration.get("global")
    if not isinstance(global_payload, dict) or global_payload.get("threshold") is None:
        raise ValueError("missing_calibrated_global_threshold")
    return "global", global_payload


def score_row_payload(
    *,
    row: dict[str, Any],
    probability: float,
    threshold_scope: str,
    threshold_payload: dict[str, Any],
    calibration: dict[str, Any],
    model_kind: str,
) -> dict[str, Any]:
    threshold = float(threshold_payload["threshold"])
    decision = "shadow_accept_candidate" if probability >= threshold else "shadow_block_candidate"
    return {
        "row_uid": str(row.get("row_uid") or ""),
        "canonical_row_key": str(row.get("canonical_row_key") or ""),
        "track_id": str(row.get("track_id") or ""),
        "run_id": str(row.get("run_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "route_id": str(row.get("route_id") or ""),
        "candidate_family": str(row.get("candidate_family") or ""),
        "source_model_family": str(row.get("source_model_family") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "split": str(row.get("split") or ""),
        "true_label": int(row.get("label") or 0),
        "window_start_sec": float(row.get("window_start_sec") or 0.0),
        "window_end_sec": float(row.get("window_end_sec") or 0.0),
        "predicted_accept_probability": round(float(probability), 6),
        "calibrated_threshold": threshold,
        "threshold_scope": threshold_scope,
        "calibration_status": str(calibration.get("calibration_status") or ""),
        "shadow_decision": decision,
        "would_accept_at_calibrated_threshold": decision == "shadow_accept_candidate",
        "runtime_gate_decision_applied": False,
        "model_kind": model_kind,
        **SAFETY_FLAGS,
    }


def score_tensor_audio_error_critic(
    *,
    dataset_dir: Path,
    model_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = validate_dataset_manifest(dataset_dir)
    validation = validate_tensor_audio_error_critic(
        model_dir,
        calibration_dir=calibration_dir,
        require_trained=True,
        require_usable_calibration=True,
    )
    if validation.get("ok") is not True:
        raise ValueError(f"tensor_critic_validation_failed:{','.join(validation.get('errors', []))}")
    model_manifest = read_json(model_dir / "tensor_audio_error_critic_manifest.json")
    calibration = read_json(calibration_dir / "tensor_audio_error_thresholds.json")
    bundle = load_model_bundle(model_dir, model_manifest)
    rows, excluded = load_rows(dataset_dir)
    scored_rows: list[dict[str, Any]] = []
    if rows:
        features = np.vstack([row["features"] for row in rows])
        probabilities = bundle["model"].predict_proba(features)[:, 1].astype(float)
        for row, probability in zip(rows, probabilities):
            stem_type = str(row.get("stem_type") or "unknown")
            threshold_scope, threshold_payload = select_threshold(calibration, stem_type)
            scored_rows.append(
                score_row_payload(
                    row=row,
                    probability=float(probability),
                    threshold_scope=threshold_scope,
                    threshold_payload=threshold_payload,
                    calibration=calibration,
                    model_kind=str(model_manifest.get("model_kind") or ""),
                )
            )
    write_jsonl(output_dir / SCORES_FILE, scored_rows)
    decision_counts = dict(sorted(Counter(str(row["shadow_decision"]) for row in scored_rows).items()))
    stem_counts = dict(sorted(Counter(str(row["stem_type"]) for row in scored_rows).items()))
    summary = {
        "schema": "tensor_audio_error_shadow_scores.v1",
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "calibration_dir": str(calibration_dir),
        "output_dir": str(output_dir),
        "source_dataset_id": str(dataset_manifest.get("source_split_dataset_id") or ""),
        "model_kind": str(model_manifest.get("model_kind") or ""),
        "calibration_status": str(calibration.get("calibration_status") or ""),
        "score_file": str(output_dir / SCORES_FILE),
        "rows_loaded": len(rows),
        "rows_scored": len(scored_rows),
        "excluded_reason_counts": dict(sorted(excluded.items())),
        "shadow_decision_counts": decision_counts,
        "stem_counts": stem_counts,
        "global_threshold": (calibration.get("global") or {}).get("threshold"),
        "use_calibrated_thresholds": True,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / SUMMARY_FILE, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score tensor audio error dataset rows with a calibrated shadow critic.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_tensor_audio_error_critic(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
