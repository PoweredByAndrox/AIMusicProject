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
    EVAL_FILE,
    MANIFEST_FILE,
    SAFETY_FLAGS,
    TARGETS,
    THRESHOLDS_FILE,
    compute_target_metrics,
    evaluation_payload,
    load_target_rows,
    parse_bool,
    predict_bundle,
    read_json,
    thresholded_metrics,
    write_json,
    write_jsonl,
)


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
) -> dict[str, Any]:
    if not threshold_payload:
        return {}
    accept_threshold = threshold_payload.get("accept_threshold")
    block_threshold = threshold_payload.get("block_threshold")
    if accept_threshold is None or block_threshold is None:
        return {}
    metrics = thresholded_metrics(
        rows,
        predictions,
        accept_threshold=float(accept_threshold),
        block_threshold=float(block_threshold),
    )
    for prediction, decision in zip(predictions, metrics.get("decisions", [])):
        prediction["critic_decision"] = decision
        prediction["abstained"] = decision == "abstain"
    return {key: value for key, value in metrics.items() if key != "decisions"}


def write_eval_summary(path: Path, result: dict[str, Any]) -> None:
    lines = ["# Target Audio Error Critic Evaluation", ""]
    for target, metrics in result["targets"].items():
        lines.extend(
            [
                f"## {target}",
                "",
                f"- rows_evaluated: {metrics.get('rows')}",
                f"- accuracy: {metrics.get('accuracy')}",
                f"- false_accept_rate: {metrics.get('false_accept_rate')}",
                f"- false_reject_rate: {metrics.get('false_reject_rate')}",
                f"- critic_usable_for_shadow_scoring: {str(metrics.get('critic_usable_for_shadow_scoring')).lower()}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_target_audio_error_critics(
    *,
    model_dir: Path,
    views_dir: Path,
    output_dir: Path,
    calibration_dir: Path | None = None,
    use_thresholds: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(model_dir / MANIFEST_FILE)
    threshold_manifest = read_json(model_dir / THRESHOLDS_FILE) if use_thresholds else {}
    target_results: dict[str, Any] = {}
    eval_rows: list[dict[str, Any]] = []
    for target, cfg in TARGETS.items():
        trained = manifest.get(f"{target}_critic_training_completed") is True
        model_path = model_dir / cfg["model_file"]
        if not trained or not model_path.exists():
            metrics = {
                "target": target,
                "target_name": cfg["view_name"],
                "rows": 0,
                "false_accept_rate": None,
                "false_reject_rate": None,
                "critic_usable_for_shadow_scoring": False,
                "reason_if_not_usable": manifest.get(f"reason_if_{target}_not_usable") or "target_not_trained",
                **SAFETY_FLAGS,
            }
            target_results[target] = metrics
            write_json(output_dir / cfg["metrics_file"], metrics)
            continue
        bundle, calibration_source = load_bundle(model_dir=model_dir, calibration_dir=calibration_dir, target=target, model_file=cfg["model_file"])
        rows = load_target_rows(views_dir, target)["usable_rows"]
        predictions = predict_bundle(bundle, rows)
        threshold_metrics = apply_threshold_decisions(
            rows,
            predictions,
            threshold_payload=(threshold_manifest.get("targets") or {}).get(target) if use_thresholds else None,
        )
        metrics = compute_target_metrics(target, rows, predictions)
        if threshold_metrics:
            metrics.update(
                {
                    "false_accept_rate": threshold_metrics["false_accept_rate"],
                    "false_reject_rate": threshold_metrics["false_reject_rate"],
                    "true_positive": threshold_metrics["true_positive"],
                    "true_negative": threshold_metrics["true_negative"],
                    "false_positive": threshold_metrics["false_positive"],
                    "false_negative": threshold_metrics["false_negative"],
                    "precision": threshold_metrics["precision"],
                    "recall": threshold_metrics["recall"],
                    "auroc": threshold_metrics["auroc"],
                    "brier_score": threshold_metrics["brier_score"],
                    "abstain_rate": threshold_metrics["abstain_rate"],
                    "abstain_rows": threshold_metrics["abstain_rows"],
                    "chosen_accept_threshold": threshold_metrics["accept_threshold"],
                    "chosen_block_threshold": threshold_metrics["block_threshold"],
                }
            )
        metrics.update(
            {
                "model_id": bundle.get("model_id", ""),
                "rows_used": len(rows),
                "calibration_source": calibration_source,
                "thresholds_used": bool(threshold_metrics),
                "critic_usable_for_shadow_scoring": manifest.get(f"{target}_critic_usable_for_shadow_scoring") is True,
                "reason_if_not_usable": manifest.get(f"reason_if_{target}_not_usable") or "",
            }
        )
        target_results[target] = metrics
        write_json(output_dir / cfg["metrics_file"], metrics)
        for row, prediction in zip(rows, predictions):
            eval_rows.append(evaluation_payload(row, prediction, target=target, model_id=str(bundle.get("model_id") or "")))
    result = {
        "schema": "target_audio_error_critic_evaluation.v1",
        "model_dir": str(model_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "views_dir": str(views_dir),
        "output_dir": str(output_dir),
        "use_thresholds": bool(use_thresholds),
        "targets": target_results,
        "rows_evaluated": len(eval_rows),
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "target_audio_error_critic_eval_summary.json", result)
    write_jsonl(output_dir / EVAL_FILE, eval_rows)
    write_eval_summary(output_dir / "target_audio_error_critic_summary.md", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shadow-only target audio error critics.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-thresholds", type=parse_bool, default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_target_audio_error_critics(
        model_dir=args.model_dir,
        calibration_dir=args.calibration_dir,
        views_dir=args.views_dir,
        output_dir=args.output_dir,
        use_thresholds=args.use_thresholds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
