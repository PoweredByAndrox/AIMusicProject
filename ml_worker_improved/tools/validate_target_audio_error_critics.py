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

from tools.audit_audio_error_dataset_provenance import is_synthetic_candidate_id  # noqa: E402
from tools.score_target_audio_error_views import SCORED_FILES  # noqa: E402
from tools.train_target_audio_error_critics import MANIFEST_FILE, RELIABILITY_REPORT_FILE, TARGETS, parse_bool, read_json  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    yield payload


def any_true_flag(payload: Any, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        return any(payload.get(key) is True for key in keys)
    if isinstance(payload, list):
        return any(any_true_flag(item, keys) for item in payload)
    return False


def validate_target_audio_error_critics(
    *,
    model_dir: Path,
    eval_dir: Path,
    scores_dir: Path,
    calibration_dir: Path | None = None,
    max_false_accept_rate: float | None = None,
    min_negative_test_rows: int = 0,
) -> dict[str, Any]:
    manifest_path = model_dir / MANIFEST_FILE
    manifest = read_json(manifest_path)
    reliability_report = read_json(model_dir / RELIABILITY_REPORT_FILE)
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("missing_model_manifest")
    if manifest.get("unsafe_rows_used") is True:
        errors.append("unsafe_rows_used_must_be_false:manifest")
    if manifest.get("pair_rows_used") is True:
        errors.append("pair_rows_used_must_be_false:manifest")
    if manifest.get("overall_rows_used") is True:
        errors.append("overall_rows_used_must_be_false:manifest")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_must_be_false:manifest")
    if manifest.get("separator_training_started") is True:
        errors.append("separator_training_started_must_be_false:manifest")
    if manifest.get("shadow_only") is not True:
        errors.append("shadow_only_must_be_true:manifest")
    if manifest.get("runtime_gates_changed") is True:
        errors.append("runtime_gates_changed_must_be_false:manifest")
    forbidden = {"release_ready", "separator_training_started", "melody_transcription_ran", "recommendation_index_updated"}
    score_rows_all: list[dict[str, Any]] = []
    target_usability: dict[str, Any] = {}
    for target, cfg in TARGETS.items():
        trained = manifest.get(f"{target}_critic_training_completed") is True
        model_path = model_dir / cfg["model_file"]
        if trained and not model_path.exists():
            errors.append(f"missing_model_artifact:{target}")
        if not trained and model_path.exists():
            errors.append(f"model_artifact_exists_but_training_skipped:{target}")
        metrics_path = eval_dir / cfg["metrics_file"]
        metrics = read_json(metrics_path)
        if trained:
            if not metrics_path.exists():
                errors.append(f"missing_eval_metrics:{target}")
            elif "false_accept_rate" not in metrics:
                errors.append(f"missing_false_accept_rate:{target}")
        if any_true_flag(metrics, forbidden | {"unsafe_rows_used", "pair_rows_used", "overall_rows_used"}):
            errors.append(f"forbidden_true_flag:metrics:{target}")
        scored_path = scores_dir / SCORED_FILES[target]
        scored_rows = list(iter_jsonl(scored_path))
        score_rows_all.extend(scored_rows)
        if trained and not scored_path.exists():
            errors.append(f"missing_scored_windows:{target}")
        if any_true_flag(scored_rows, forbidden):
            errors.append(f"forbidden_true_flag:scores:{target}")
        for index, row in enumerate(scored_rows):
            if row.get("shadow_only") is not True:
                errors.append(f"shadow_only_must_be_true:score:{target}:{index}")
            route_id = str(row.get("route_id") or "").strip().lower()
            if route_id == "unknown":
                errors.append(f"unknown_route_id_used_in_score:{target}:{index}")
            if is_synthetic_candidate_id(row.get("candidate_id")):
                errors.append(f"synthetic_candidate_id_used_in_score:{target}:{index}")
            if row.get("abstained") is True and row.get("critic_decision") == "accept_candidate":
                errors.append(f"abstain_counted_as_accept:{target}:{index}")
        report_target = (reliability_report.get("targets") or {}).get(target, {})
        strict_requested = max_false_accept_rate is not None or min_negative_test_rows > 0
        target_errors: list[str] = []
        if trained and not report_target:
            target_errors.append(f"missing_reliability_report:{target}")
        if report_target:
            if report_target.get("candidate_id_overlap_train_test") is True:
                target_errors.append(f"candidate_id_overlap_train_test:{target}")
            false_accept_rate = report_target.get("false_accept_rate")
            if false_accept_rate is None:
                target_errors.append(f"missing_false_accept_rate:{target}")
            elif max_false_accept_rate is not None and float(false_accept_rate) > float(max_false_accept_rate):
                target_errors.append(f"false_accept_rate_above_max:{target}")
            if int(report_target.get("test_negative_rows") or 0) < int(min_negative_test_rows):
                target_errors.append(f"test_negative_rows_below_minimum:{target}")
            if str(report_target.get("calibration_status") or "") != "calibrated" and not str(report_target.get("calibration_skipped_reason") or ""):
                target_errors.append(f"missing_calibration_skipped_reason:{target}")
            if strict_requested and not report_target.get("per_route_false_accept_audit"):
                target_errors.append(f"missing_per_route_false_accept_audit:{target}")
            if strict_requested and not report_target.get("per_track_false_accept_audit"):
                target_errors.append(f"missing_per_track_false_accept_audit:{target}")
        errors.extend(target_errors)
        target_usability[target] = {
            "critic_usable_for_shadow_scoring": trained and not target_errors,
            "reason_if_not_usable": ";".join(target_errors),
            "max_false_accept_rate": max_false_accept_rate,
            "min_negative_test_rows": min_negative_test_rows,
        }
    result = {
        "ok": not errors,
        "errors": errors,
        "model_dir": str(model_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "eval_dir": str(eval_dir),
        "scores_dir": str(scores_dir),
        "targets": target_usability,
        "unsafe_rows_used": False,
        "pair_rows_used": False,
        "overall_rows_used": False,
        "shadow_only": manifest.get("shadow_only") is True,
        "release_ready": False,
        "separator_training_started": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "rows_scored": len(score_rows_all),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate shadow-only target audio error critic artifacts.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.10)
    parser.add_argument("--min-negative-test-rows", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_target_audio_error_critics(
        model_dir=args.model_dir,
        calibration_dir=args.calibration_dir,
        eval_dir=args.eval_dir,
        scores_dir=args.scores_dir,
        max_false_accept_rate=args.max_false_accept_rate,
        min_negative_test_rows=args.min_negative_test_rows,
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
