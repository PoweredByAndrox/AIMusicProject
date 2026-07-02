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


def top_counts(values: list[str], *, limit: int = 12) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def is_score_only_row(row: dict[str, Any], target: str) -> bool:
    label_source = str(row.get("label_source") or "")
    return label_source in set(TARGETS[target].get("score_only_label_sources", ()))


def score_only_audit(target: str, rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_only_rows = 0
    score_only_negative_rows = 0
    score_only_positive_rows = 0
    false_accept_rows = 0
    false_reject_rows = 0
    false_accept_sources: list[str] = []
    false_accept_families: list[str] = []
    false_accept_routes: list[str] = []
    score_only_sources: list[str] = []
    score_only_families: list[str] = []

    for source_row, scored_row in zip(rows, scored_rows):
        if not is_score_only_row(source_row, target):
            continue
        score_only_rows += 1
        label = int(source_row.get("label"))
        decision = str(scored_row.get("critic_decision") or "")
        label_source = str(source_row.get("label_source") or "")
        features = source_row.get("features") if isinstance(source_row.get("features"), dict) else {}
        candidate_family = str(features.get("candidate_family") or source_row.get("candidate_family") or "")
        route_id = str(source_row.get("route_id") or "")
        score_only_sources.append(label_source)
        score_only_families.append(candidate_family)
        if label == 0:
            score_only_negative_rows += 1
            if decision == "accept_candidate":
                false_accept_rows += 1
                false_accept_sources.append(label_source)
                false_accept_families.append(candidate_family)
                false_accept_routes.append(route_id)
        elif label == 1:
            score_only_positive_rows += 1
            if decision == "blocked_hard_case":
                false_reject_rows += 1

    false_accept_rate = 0.0
    if score_only_negative_rows:
        false_accept_rate = false_accept_rows / score_only_negative_rows
    false_reject_rate = 0.0
    if score_only_positive_rows:
        false_reject_rate = false_reject_rows / score_only_positive_rows
    if false_accept_rows:
        integrity_status = "blocked_model_blind_spot"
    elif score_only_negative_rows:
        integrity_status = "passed_score_only_negative_audit"
    elif score_only_rows:
        integrity_status = "no_score_only_negative_rows"
    else:
        integrity_status = "no_score_only_rows"

    return {
        "target": target,
        "score_only_rows_scored": score_only_rows,
        "score_only_negative_rows": score_only_negative_rows,
        "score_only_positive_rows": score_only_positive_rows,
        "score_only_false_accept_rows": false_accept_rows,
        "score_only_false_reject_rows": false_reject_rows,
        "score_only_false_accept_rate": round(false_accept_rate, 6),
        "score_only_false_reject_rate": round(false_reject_rate, 6),
        "score_only_sources": top_counts(score_only_sources),
        "score_only_candidate_families": top_counts(score_only_families),
        "score_only_false_accept_sources": top_counts(false_accept_sources),
        "score_only_false_accept_candidate_families": top_counts(false_accept_families),
        "score_only_false_accept_routes": top_counts(false_accept_routes),
        "score_only_integrity_status": integrity_status,
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


def scored_payload(row: dict[str, Any], prediction: dict[str, Any], *, target: str) -> dict[str, Any]:
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
        "label_source": str(row.get("label_source") or ""),
        "score_only": is_score_only_row(row, target),
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
        "score_only_blind_spot_detected": False,
        "score_only_audit": {},
        **SAFETY_FLAGS,
    }
    for target, cfg in TARGETS.items():
        rows_out: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
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
            rows_out = [
                scored_payload(row, {**prediction, "calibration_source": calibration_source}, target=target)
                for row, prediction in zip(rows, predictions)
            ]
        write_jsonl(output_dir / SCORED_FILES[target], rows_out)
        summary[f"{target}_rows_scored"] = len(rows_out)
        audit = score_only_audit(target, rows, rows_out)
        summary["score_only_audit"][target] = audit
        summary[f"{target}_score_only_rows_scored"] = audit["score_only_rows_scored"]
        summary[f"{target}_score_only_negative_rows"] = audit["score_only_negative_rows"]
        summary[f"{target}_score_only_false_accept_rows"] = audit["score_only_false_accept_rows"]
        summary[f"{target}_score_only_integrity_status"] = audit["score_only_integrity_status"]
        if audit["score_only_integrity_status"] == "blocked_model_blind_spot":
            summary["score_only_blind_spot_detected"] = True
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
