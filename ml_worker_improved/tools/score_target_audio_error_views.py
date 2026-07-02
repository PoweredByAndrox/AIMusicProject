#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import joblib  # noqa: E402

from tools.train_target_audio_error_critics import (  # noqa: E402
    MANIFEST_FILE,
    SAFETY_FLAGS,
    TARGETS,
    THRESHOLDS_FILE,
    load_target_rows,
    parse_bool,
    predict_bundle,
    read_json,
    thresholded_metrics,
    write_json,
    write_jsonl,
)


SCORED_FILES = {
    "minus": "scored_minus_critic_windows.jsonl",
    "vocal": "scored_vocal_critic_windows.jsonl",
}


def calibrated_model_path(target: str) -> str:
    return f"calibrated_{target}_audio_error_critic.joblib"


def load_bundle(*, model_dir: Path, calibration_dir: Path | None, target: str, model_file: str) -> tuple[dict[str, Any], str]:
    if calibration_dir is not None:
        path = calibration_dir / calibrated_model_path(target)
        if path.exists():
            return joblib.load(path), "calibrated_model"
    return joblib.load(model_dir / model_file), "base_model"


def apply_threshold_decisions(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    threshold_payload: dict[str, Any] | None,
    allow_abstain: bool,
) -> None:
    if not threshold_payload:
        return
    accept_threshold = threshold_payload.get("accept_threshold")
    block_threshold = threshold_payload.get("block_threshold")
    if accept_threshold is None or block_threshold is None:
        return
    metrics = thresholded_metrics(rows, predictions, accept_threshold=float(accept_threshold), block_threshold=float(block_threshold))
    for prediction, decision in zip(predictions, metrics.get("decisions", [])):
        if decision == "abstain" and not allow_abstain:
            decision = "blocked_hard_case"
        prediction["critic_decision"] = decision
        prediction["abstained"] = decision == "abstain"


def scored_payload(row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_name": row["view_name"],
        "target_name": row["target_name"],
        "row_id": row["row_id"],
        "track_id": row["track_id"],
        "candidate_id": row["candidate_id"],
        "route_id": row["route_id"],
        "window_start_sec": row["window_start_sec"],
        "window_end_sec": row["window_end_sec"],
        "true_label": int(row["label"]),
        "predicted_accept_probability": prediction["predicted_accept_probability"],
        "predicted_block_probability": prediction["predicted_block_probability"],
        "predicted_blockers": prediction["predicted_blockers"],
        "critic_decision": prediction["critic_decision"],
        "abstained": bool(prediction.get("abstained", False)),
        "calibration_source": prediction.get("calibration_source", "base_model"),
        "confidence": prediction["confidence"],
        "shadow_only": True,
        "release_ready": False,
        "separator_training_started": False,
    }


def score_target_audio_error_views(
    *,
    model_dir: Path,
    views_dir: Path,
    output_dir: Path,
    calibration_dir: Path | None = None,
    use_thresholds: bool = False,
    allow_abstain: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(model_dir / MANIFEST_FILE)
    threshold_manifest = read_json(model_dir / THRESHOLDS_FILE) if use_thresholds else {}
    summary: dict[str, Any] = {
        "schema": "target_audio_error_critic_scores.v1",
        "model_dir": str(model_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "views_dir": str(views_dir),
        "output_dir": str(output_dir),
        "use_thresholds": bool(use_thresholds),
        "allow_abstain": bool(allow_abstain),
        "minus_rows_scored": 0,
        "vocal_rows_scored": 0,
        **SAFETY_FLAGS,
    }
    for target, cfg in TARGETS.items():
        rows_out: list[dict[str, Any]] = []
        trained = manifest.get(f"{target}_critic_training_completed") is True
        model_path = model_dir / cfg["model_file"]
        if trained and model_path.exists():
            bundle, calibration_source = load_bundle(model_dir=model_dir, calibration_dir=calibration_dir, target=target, model_file=cfg["model_file"])
            rows = load_target_rows(views_dir, target, include_score_only_rows=True)["usable_rows"]
            predictions = predict_bundle(bundle, rows)
            apply_threshold_decisions(
                rows,
                predictions,
                threshold_payload=(threshold_manifest.get("targets") or {}).get(target) if use_thresholds else None,
                allow_abstain=allow_abstain,
            )
            rows_out = [scored_payload(row, {**prediction, "calibration_source": calibration_source}) for row, prediction in zip(rows, predictions)]
        write_jsonl(output_dir / SCORED_FILES[target], rows_out)
        summary[f"{target}_rows_scored"] = len(rows_out)
    write_json(output_dir / "target_audio_error_critic_score_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score target-specific audio error views with shadow critics.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-thresholds", type=parse_bool, default=False)
    parser.add_argument("--allow-abstain", type=parse_bool, default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_target_audio_error_views(
        model_dir=args.model_dir,
        calibration_dir=args.calibration_dir,
        views_dir=args.views_dir,
        output_dir=args.output_dir,
        use_thresholds=args.use_thresholds,
        allow_abstain=args.allow_abstain,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
