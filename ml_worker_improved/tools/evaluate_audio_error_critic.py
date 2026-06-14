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

from tools.train_audio_error_critic import (
    EVAL_FILE,
    MANIFEST_FILE,
    METRICS_FILE,
    MODEL_FILE,
    SAFETY_FLAGS,
    SUMMARY_FILE,
    compute_metrics,
    evaluation_row,
    grouped_metrics,
    load_bundle,
    load_strict_training_rows,
    null_metrics,
    predict_with_bundle,
    read_json,
    write_json,
    write_jsonl,
)


def write_eval_summary(path: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        "# Audio Error Critic Evaluation",
        "",
        f"- model_id: `{manifest.get('model_id', '')}`",
        f"- decision: {metrics.get('decision') or manifest.get('decision')}",
        f"- rows_evaluated: {metrics.get('rows')}",
        f"- accept_block_accuracy: {metrics.get('accept_block_accuracy')}",
        f"- precision: {metrics.get('precision')}",
        f"- recall: {metrics.get('recall')}",
        f"- false_accept_rate: {metrics.get('false_accept_rate')}",
        f"- false_reject_rate: {metrics.get('false_reject_rate')}",
        f"- phase_collapse_mae: {metrics.get('phase_collapse_regression_mae')}",
        f"- vocal_bleed_mae: {metrics.get('vocal_bleed_regression_mae')}",
        f"- pair_compatibility_accuracy: {metrics.get('pair_compatibility_accuracy')}",
        f"- critic_usable_for_shadow_scoring: {str(metrics.get('critic_usable_for_shadow_scoring')).lower()}",
        f"- critic_not_usable_reason: {metrics.get('critic_not_usable_reason')}",
        "- unsafe_rows_used: false",
        "- separator_training_started: false",
        "- release_ready: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_audio_error_critic(*, model_dir: Path, dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(model_dir / MANIFEST_FILE)
    data = load_strict_training_rows(dataset_dir)
    reason = str(manifest.get("critic_not_usable_reason") or "model_not_usable")
    if manifest.get("critic_usable_for_shadow_scoring") is not True or not (model_dir / MODEL_FILE).exists():
        metrics = {
            **null_metrics(reason),
            "model_id": manifest.get("model_id", ""),
            "rows_evaluated": 0,
            "strict_rows_available": len(data["usable_rows"]),
        }
        write_json(output_dir / METRICS_FILE, metrics)
        write_jsonl(output_dir / EVAL_FILE, [])
        write_eval_summary(output_dir / SUMMARY_FILE, manifest, metrics)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        return metrics

    bundle = load_bundle(model_dir / MODEL_FILE)
    rows = data["usable_rows"]
    predictions = [predict_with_bundle(bundle, features=row["features"]) for row in rows]
    metrics = {
        **compute_metrics(rows, predictions),
        "per_track_metrics": grouped_metrics(rows, predictions, "track_id"),
        "per_route_metrics": grouped_metrics(rows, predictions, "route_id"),
        "model_id": manifest.get("model_id", bundle.get("model_id", "")),
        "rows_evaluated": len(rows),
        "strict_rows_available": len(rows),
        "decision": "critic_evaluated",
        "critic_usable_for_shadow_scoring": True,
        "critic_not_usable_reason": "",
        "unsafe_rows_used": False,
        **SAFETY_FLAGS,
    }
    eval_rows = []
    for row, prediction in zip(rows, predictions):
        payload = evaluation_row(row, prediction)
        payload["model_id"] = str(metrics["model_id"])
        eval_rows.append(payload)
    write_json(output_dir / METRICS_FILE, metrics)
    write_jsonl(output_dir / EVAL_FILE, eval_rows)
    write_eval_summary(output_dir / SUMMARY_FILE, manifest, metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a shadow-only audio error critic.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_audio_error_critic(model_dir=args.model_dir, dataset_dir=args.dataset_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
