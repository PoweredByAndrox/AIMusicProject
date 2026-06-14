#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.train_audio_error_critic import MANIFEST_FILE, METRICS_FILE, MODEL_FILE, SCORED_FILE, read_json


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def any_true_flag(payload: Any, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and value is True:
                return True
            if any_true_flag(value, keys):
                return True
    if isinstance(payload, list):
        return any(any_true_flag(item, keys) for item in payload)
    return False


def validate_manifest(manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("unsafe_rows_used") is True:
        errors.append("unsafe_rows_used_must_be_false:manifest")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_must_be_false:manifest")
    if manifest.get("training_started") is True:
        errors.append("training_started_must_be_false:manifest")
    if manifest.get("separator_training_started") is True:
        errors.append("separator_training_started_must_be_false:manifest")
    if manifest.get("melody_transcription_ran") is True:
        errors.append("melody_transcription_must_not_run:manifest")
    if manifest.get("recommendation_index_updated") is True:
        errors.append("recommendation_index_must_not_update:manifest")
    if not manifest.get("feature_columns"):
        errors.append("missing_feature_schema:manifest")
    if not manifest.get("target_columns"):
        errors.append("missing_target_schema:manifest")
    if manifest.get("critic_usable_for_shadow_scoring") is True and manifest.get("shadow_only") is not True:
        errors.append("critic_must_be_shadow_only:manifest")


def validate_audio_error_critic_model(*, model_dir: Path, eval_dir: Path, scores_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / MANIFEST_FILE
    metrics_path = eval_dir / METRICS_FILE
    scores_path = scores_dir / SCORED_FILE
    manifest = read_json(manifest_path)
    metrics = read_json(metrics_path)
    score_rows = list(iter_jsonl(scores_path))
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("missing_model_manifest")
    validate_manifest(manifest, errors)
    forbidden = {"release_ready", "training_started", "separator_training_started", "melody_transcription_ran", "recommendation_index_updated"}
    if any_true_flag(metrics, forbidden):
        errors.append("forbidden_true_flag:metrics")
    if any_true_flag(score_rows, forbidden):
        errors.append("forbidden_true_flag:scores")
    if any_true_flag(score_rows, {"unsafe_rows_used"}):
        errors.append("unsafe_rows_used_must_be_false:scores")
    usable = manifest.get("critic_usable_for_shadow_scoring") is True
    if usable:
        if not (model_dir / MODEL_FILE).exists():
            errors.append("missing_model_artifact")
        if not metrics_path.exists():
            errors.append("missing_eval_metrics")
        elif "false_accept_rate" not in metrics:
            errors.append("missing_false_accept_rate:metrics")
        if not scores_path.exists():
            errors.append("missing_scored_windows")
        elif not score_rows:
            errors.append("missing_scored_window_rows")
        if manifest.get("can_promote_to_production_gate") is True:
            errors.append("critic_must_not_promote_to_production_gate")
    else:
        if not manifest.get("critic_not_usable_reason"):
            errors.append("missing_critic_not_usable_reason:manifest")
        if (model_dir / MODEL_FILE).exists():
            errors.append("skipped_model_must_not_write_model_artifact")
    result = {
        "ok": not errors,
        "errors": errors,
        "model_dir": str(model_dir),
        "eval_dir": str(eval_dir),
        "scores_dir": str(scores_dir),
        "model_id": str(manifest.get("model_id") or ""),
        "critic_usable_for_shadow_scoring": usable,
        "critic_not_usable_reason": str(manifest.get("critic_not_usable_reason") or ""),
        "unsafe_rows_used": False,
        "release_ready": False,
        "training_started": False,
        "separator_training_started": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate shadow-only audio error critic artifacts.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_audio_error_critic_model(model_dir=args.model_dir, eval_dir=args.eval_dir, scores_dir=args.scores_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
