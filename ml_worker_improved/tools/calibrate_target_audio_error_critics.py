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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from tools.train_target_audio_error_critics import (  # noqa: E402
    MANIFEST_FILE,
    SAFETY_FLAGS,
    TARGETS,
    apply_probability_calibration,
    calibration_summary,
    clip01,
    confusion_metrics,
    load_target_rows,
    predict_bundle,
    read_json,
    write_json,
)


CALIBRATION_MANIFEST_FILE = "critic_calibration_manifest.json"
CALIBRATION_METRICS_FILE = "critic_calibration_metrics.json"
CALIBRATION_SUMMARY_FILE = "critic_calibration_summary.md"


def calibration_model_path(target: str) -> str:
    return f"calibrated_{target}_audio_error_critic.joblib"


def base_probabilities(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    raw_calibration = bundle.pop("calibration", None)
    try:
        return [clip01(item["predicted_accept_probability"]) for item in predict_bundle(bundle, rows)]
    finally:
        if raw_calibration is not None:
            bundle["calibration"] = raw_calibration


def expected_calibration_error(labels: list[int], scores: list[float], bins: int = 10) -> float | None:
    if not labels:
        return None
    total = len(labels)
    error = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        pairs = [(label, score) for label, score in zip(labels, scores) if lower <= score < upper or (bucket == bins - 1 and score <= upper)]
        if not pairs:
            continue
        confidence = sum(score for _, score in pairs) / len(pairs)
        accuracy = sum(label for label, _ in pairs) / len(pairs)
        error += (len(pairs) / total) * abs(confidence - accuracy)
    return round(error, 6)


def choose_calibration_method(method: str, row_count: int) -> str:
    if method in {"sigmoid", "isotonic"}:
        return method
    if method != "auto":
        raise ValueError(f"unsupported_calibration_method:{method}")
    return "isotonic" if row_count >= 100 else "sigmoid"


def fit_calibrator(method: str, scores: list[float], labels: list[int]) -> Any:
    matrix = [[score] for score in scores]
    if method == "sigmoid":
        calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2718)
        calibrator.fit(matrix, labels)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(scores, labels)
        return calibrator
    raise ValueError(f"unsupported_calibration_method:{method}")


def calibrated_scores(method: str, calibrator: Any, scores: list[float]) -> list[float]:
    bundle = {"calibration": {"status": "calibrated", "method": method, "model": calibrator}}
    return apply_probability_calibration(bundle, scores)


def metrics_for_scores(labels: list[int], scores: list[float]) -> dict[str, Any]:
    metrics = confusion_metrics(labels, scores)
    metrics["brier_score"] = round(float(brier_score_loss(labels, scores)), 6) if labels else None
    metrics["expected_calibration_error"] = expected_calibration_error(labels, scores)
    metrics["reliability_bins"] = calibration_summary(labels, scores, bins=10)
    return metrics


def calibrate_one_target(
    *,
    target: str,
    model_dir: Path,
    views_dir: Path,
    output_dir: Path,
    method: str,
    min_calibration_rows: int,
    min_negative_test_rows: int,
    min_negative_train_rows: int,
) -> dict[str, Any]:
    cfg = TARGETS[target]
    model_path = model_dir / cfg["model_file"]
    if not model_path.exists():
        return {
            "target": target,
            "target_name": cfg["view_name"],
            "calibration_status": "skipped",
            "calibration_method": "none",
            "calibration_skipped_reason": "missing_model_artifact",
            "rows": 0,
        }
    bundle = joblib.load(model_path)
    rows = [
        row
        for row in load_target_rows(
            views_dir,
            target,
            group_split="candidate_id",
            min_negative_test_rows=min_negative_test_rows,
            min_negative_dev_rows=min_negative_test_rows,
            min_negative_train_rows=min_negative_train_rows,
        )["usable_rows"]
        if row.get("split") == "dev"
    ]
    labels = [int(row["label"]) for row in rows]
    if len(rows) < min_calibration_rows:
        reason = "insufficient_dev_data"
    elif len(set(labels)) < 2:
        reason = "single_class_dev_data"
    else:
        reason = ""
    base_scores = base_probabilities(bundle, rows) if rows else []
    before = metrics_for_scores(labels, base_scores) if rows and len(set(labels)) > 1 else {}
    if reason:
        return {
            "target": target,
            "target_name": cfg["view_name"],
            "calibration_status": "skipped",
            "calibration_method": "none",
            "calibration_skipped_reason": reason,
            "rows": len(rows),
            "before": before,
            **SAFETY_FLAGS,
        }
    selected_method = choose_calibration_method(method, len(rows))
    calibrator = fit_calibrator(selected_method, base_scores, labels)
    after_scores = calibrated_scores(selected_method, calibrator, base_scores)
    after = metrics_for_scores(labels, after_scores)
    if after.get("false_accept_rate") is not None and before.get("false_accept_rate") is not None and float(after["false_accept_rate"]) > float(before["false_accept_rate"]):
        return {
            "target": target,
            "target_name": cfg["view_name"],
            "calibration_status": "skipped",
            "calibration_method": selected_method,
            "calibration_skipped_reason": "calibration_would_increase_false_accept_rate",
            "rows": len(rows),
            "before": before,
            "after": after,
            **SAFETY_FLAGS,
        }
    bundle["calibration"] = {
        "status": "calibrated",
        "method": selected_method,
        "model": calibrator,
        "source_model_dir": str(model_dir),
    }
    output_path = output_dir / calibration_model_path(target)
    joblib.dump(bundle, output_path)
    return {
        "target": target,
        "target_name": cfg["view_name"],
        "calibration_status": "calibrated",
        "calibration_method": selected_method,
        "calibration_skipped_reason": "",
        "rows": len(rows),
        "calibrated_model_file": output_path.name,
        "before": before,
        "after": after,
        **SAFETY_FLAGS,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = ["# Target Audio Error Critic Calibration", ""]
    for target, payload in manifest["targets"].items():
        lines.extend(
            [
                f"## {target}",
                "",
                f"- calibration_status: {payload.get('calibration_status')}",
                f"- calibration_method: {payload.get('calibration_method')}",
                f"- calibration_skipped_reason: {payload.get('calibration_skipped_reason')}",
                f"- rows: {payload.get('rows')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def calibrate_target_audio_error_critics(
    *,
    model_dir: Path,
    views_dir: Path,
    output_dir: Path,
    method: str = "auto",
    min_calibration_rows: int = 20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_manifest = read_json(model_dir / MANIFEST_FILE)
    targets = model_manifest.get("targets_requested") or list(TARGETS)
    min_negative_test_rows = int((model_manifest.get("minimum_reliability") or {}).get("test_negative_rows") or 0)
    min_negative_train_rows = int((model_manifest.get("minimum_readiness") or {}).get("negative_labels") or 0)
    target_results = {
        target: calibrate_one_target(
            target=target,
            model_dir=model_dir,
            views_dir=views_dir,
            output_dir=output_dir,
            method=method,
            min_calibration_rows=min_calibration_rows,
            min_negative_test_rows=min_negative_test_rows,
            min_negative_train_rows=min_negative_train_rows,
        )
        for target in targets
        if target in TARGETS
    }
    manifest = {
        "schema": "target_audio_error_critic_calibration.v1",
        "model_dir": str(model_dir),
        "views_dir": str(views_dir),
        "output_dir": str(output_dir),
        "method": method,
        "targets": target_results,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / CALIBRATION_MANIFEST_FILE, manifest)
    write_json(output_dir / CALIBRATION_METRICS_FILE, manifest)
    write_summary(output_dir / CALIBRATION_SUMMARY_FILE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate shadow-only target audio error critic probabilities.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("auto", "sigmoid", "isotonic"), default="auto")
    parser.add_argument("--min-calibration-rows", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calibrate_target_audio_error_critics(
        model_dir=args.model_dir,
        views_dir=args.views_dir,
        output_dir=args.output_dir,
        method=args.method,
        min_calibration_rows=args.min_calibration_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
