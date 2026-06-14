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
    MANIFEST_FILE,
    MODEL_FILE,
    SAFETY_FLAGS,
    SCORED_FILE,
    load_bundle,
    load_strict_training_rows,
    predict_with_bundle,
    read_json,
    strict_exclusion_reasons,
    write_json,
    write_jsonl,
)


SCORE_SUMMARY_FILE = "audio_error_critic_score_summary.json"


def base_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": str(row.get("dataset_id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "run_id": str(row.get("run_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "route_id": str(row.get("route_id") or ""),
        "stem_type": str(row.get("stem_type") or ""),
        "window_id": str(row.get("window_id") or ""),
        "window_start_sec": row.get("window_start_sec"),
        "window_end_sec": row.get("window_end_sec"),
        "judge_decision": str(row.get("judge_decision") or ""),
        "pair_compatibility_status": str(row.get("pair_compatibility_status") or ""),
        "safe_for_training": row.get("safe_for_training") is True,
    }


def diagnostic_row(row: dict[str, Any], *, reason: str, model_id: str = "") -> dict[str, Any]:
    return {
        **base_identity(row),
        "model_id": model_id,
        "critic_score_available": False,
        "critic_not_usable_reason": reason,
        "shadow_only": True,
        "unsafe_rows_used": False,
        **SAFETY_FLAGS,
    }


def scored_row(row: dict[str, Any], prediction: dict[str, float], *, model_id: str) -> dict[str, Any]:
    return {
        **base_identity(row["source_row"] if "source_row" in row else row),
        "model_id": model_id,
        **prediction,
        "critic_score_available": True,
        "critic_not_usable_reason": "",
        "shadow_only": True,
        "unsafe_rows_used": False,
        **SAFETY_FLAGS,
    }


def score_audio_error_windows_with_critic(*, model_dir: Path, dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(model_dir / MANIFEST_FILE)
    data = load_strict_training_rows(dataset_dir)
    raw_rows = data["raw_rows"]
    reason = str(manifest.get("critic_not_usable_reason") or "model_not_usable")
    model_id = str(manifest.get("model_id") or "")
    if manifest.get("critic_usable_for_shadow_scoring") is not True or not (model_dir / MODEL_FILE).exists():
        rows = [diagnostic_row(row, reason=reason, model_id=model_id) for row in raw_rows]
        summary = {
            "model_id": model_id,
            "rows_total": len(raw_rows),
            "rows_scored": 0,
            "rows_not_scored": len(rows),
            "critic_usable_for_shadow_scoring": False,
            "critic_not_usable_reason": reason,
            "unsafe_rows_used": False,
            **SAFETY_FLAGS,
        }
        write_jsonl(output_dir / SCORED_FILE, rows)
        write_json(output_dir / SCORE_SUMMARY_FILE, summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return summary

    bundle = load_bundle(model_dir / MODEL_FILE)
    scored: list[dict[str, Any]] = []
    for row in data["usable_rows"]:
        prediction = predict_with_bundle(bundle, features=row["features"])
        scored.append(scored_row(row, prediction, model_id=model_id))
    skipped = [
        diagnostic_row(row, reason=";".join(strict_exclusion_reasons(row)) or "not_strict_safe", model_id=model_id)
        for row in raw_rows
        if strict_exclusion_reasons(row)
    ]
    rows = scored + skipped
    summary = {
        "model_id": model_id,
        "rows_total": len(raw_rows),
        "rows_scored": len(scored),
        "rows_not_scored": len(skipped),
        "critic_usable_for_shadow_scoring": True,
        "critic_not_usable_reason": "",
        "unsafe_rows_used": False,
        **SAFETY_FLAGS,
    }
    write_jsonl(output_dir / SCORED_FILE, rows)
    write_json(output_dir / SCORE_SUMMARY_FILE, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score strict-safe audio error windows with a shadow critic.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_audio_error_windows_with_critic(model_dir=args.model_dir, dataset_dir=args.dataset_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
