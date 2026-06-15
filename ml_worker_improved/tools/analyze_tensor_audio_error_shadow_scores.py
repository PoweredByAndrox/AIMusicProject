#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.score_tensor_audio_error_critic import SCORES_FILE, SUMMARY_FILE  # noqa: E402
from tools.train_tensor_audio_error_critic import read_json, write_json, write_jsonl  # noqa: E402


ANALYSIS_SUMMARY_FILE = "tensor_audio_error_shadow_analysis_summary.json"
HARD_NEGATIVES_FILE = "tensor_audio_error_hard_negative_candidates.jsonl"
FALSE_REJECTS_FILE = "tensor_audio_error_false_reject_candidates.jsonl"
IMPROVEMENT_REPORT_FILE = "tensor_audio_error_model_improvement_report.md"

FORBIDDEN_TRUE_FLAGS = {
    "release_ready",
    "can_publish_as_result",
    "runtime_gate_decision_applied",
    "runtime_gates_changed",
    "training_started",
    "separator_training_started",
    "repair_adapter_training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
}

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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected_json_object:{path}:{line_number}")
            yield payload


def has_forbidden_true(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(payload.get(key) is True for key in FORBIDDEN_TRUE_FLAGS)
    if isinstance(payload, list):
        return any(has_forbidden_true(item) for item in payload)
    return False


def validate_shadow_scores(summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    errors: list[str] = []
    if summary.get("schema") != "tensor_audio_error_shadow_scores.v1":
        errors.append("invalid_shadow_score_summary_schema")
    if summary.get("shadow_only") is not True:
        errors.append("summary_shadow_only_not_true")
    if has_forbidden_true(summary):
        errors.append("summary_forbidden_true_flag")
    if int(summary.get("rows_scored") or 0) != len(rows):
        errors.append("summary_rows_scored_mismatch")
    summary_decisions = summary.get("shadow_decision_counts")
    if isinstance(summary_decisions, dict):
        actual_decisions = dict(sorted(Counter(str(row.get("shadow_decision") or "unknown") for row in rows).items()))
        normalized_summary_decisions = {str(key): int(value or 0) for key, value in summary_decisions.items()}
        if normalized_summary_decisions != actual_decisions:
            errors.append("summary_shadow_decision_counts_mismatch")
    for index, row in enumerate(rows):
        if row.get("shadow_only") is not True:
            errors.append(f"row_shadow_only_not_true:{index}")
        if str(row.get("shadow_decision") or "") not in {"shadow_accept_candidate", "shadow_block_candidate"}:
            errors.append(f"row_invalid_shadow_decision:{index}")
        if has_forbidden_true(row):
            errors.append(f"row_forbidden_true_flag:{index}")
    if errors:
        raise ValueError(f"unsafe_shadow_scores:{','.join(errors)}")


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def group_counts(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    payload: dict[str, dict[str, Any]] = {}
    for value, items in sorted(grouped.items()):
        negatives = [row for row in items if int(row.get("true_label") or 0) == 0]
        positives = [row for row in items if int(row.get("true_label") or 0) == 1]
        hard_negatives = [row for row in negatives if row.get("shadow_decision") == "shadow_accept_candidate"]
        false_rejects = [row for row in positives if row.get("shadow_decision") == "shadow_block_candidate"]
        payload[value] = {
            "rows": len(items),
            "negative_rows": len(negatives),
            "positive_rows": len(positives),
            "hard_negative_count": len(hard_negatives),
            "false_reject_count": len(false_rejects),
            "hard_negative_rate": safe_ratio(len(hard_negatives), len(negatives)),
            "false_reject_rate": safe_ratio(len(false_rejects), len(positives)),
            "mean_predicted_accept_probability": float(mean(float(row.get("predicted_accept_probability") or 0.0) for row in items)),
        }
    return payload


def mining_payload(row: dict[str, Any], *, mining_label: str) -> dict[str, Any]:
    return {
        "mining_label": mining_label,
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
        "true_label": int(row.get("true_label") or 0),
        "window_start_sec": float(row.get("window_start_sec") or 0.0),
        "window_end_sec": float(row.get("window_end_sec") or 0.0),
        "predicted_accept_probability": float(row.get("predicted_accept_probability") or 0.0),
        "calibrated_threshold": float(row.get("calibrated_threshold") or 0.0),
        "threshold_scope": str(row.get("threshold_scope") or ""),
        "shadow_decision": str(row.get("shadow_decision") or ""),
        "runtime_gate_decision_applied": False,
        "release_ready": False,
        "shadow_only": True,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Shadow Analysis",
        "",
        f"Rows analyzed: {summary['rows_analyzed']}",
        f"Hard negative candidates: {summary['hard_negative_count']}",
        f"False reject candidates: {summary['false_reject_count']}",
        f"Model kind: {summary.get('model_kind', '')}",
        f"Calibration status: {summary.get('calibration_status', '')}",
        "",
        "## Safety",
        "",
        "- release_ready=false",
        "- runtime_gates_changed=false",
        "- training_started=false",
        "- separator_training_started=false",
        "- repair_adapter_training_started=false",
        "",
        "## Top Groups",
        "",
    ]
    for group_name, groups in (("stem", summary.get("by_stem", {})), ("route", summary.get("by_route", {}))):
        lines.append(f"### By {group_name}")
        for key, payload in list(groups.items())[:10]:
            lines.append(
                f"- {key}: rows={payload['rows']}, hard_negatives={payload['hard_negative_count']}, "
                f"false_rejects={payload['false_reject_count']}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def analyze_tensor_audio_error_shadow_scores(
    *,
    scores_dir: Path,
    output_dir: Path,
    max_hard_negative_rows: int = 200,
) -> dict[str, Any]:
    if int(max_hard_negative_rows) < 0:
        raise ValueError("max_hard_negative_rows_must_be_non_negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(scores_dir / SUMMARY_FILE)
    rows = list(iter_jsonl(scores_dir / SCORES_FILE))
    validate_shadow_scores(summary, rows)
    negatives = [row for row in rows if int(row.get("true_label") or 0) == 0]
    positives = [row for row in rows if int(row.get("true_label") or 0) == 1]
    hard_negatives = [row for row in negatives if row.get("shadow_decision") == "shadow_accept_candidate"]
    false_rejects = [row for row in positives if row.get("shadow_decision") == "shadow_block_candidate"]
    hard_negatives = sorted(hard_negatives, key=lambda row: float(row.get("predicted_accept_probability") or 0.0), reverse=True)
    false_rejects = sorted(false_rejects, key=lambda row: float(row.get("predicted_accept_probability") or 0.0))
    capped_hard_negatives = hard_negatives[: int(max_hard_negative_rows)]
    write_jsonl(output_dir / HARD_NEGATIVES_FILE, [mining_payload(row, mining_label="hard_negative_false_accept") for row in capped_hard_negatives])
    write_jsonl(output_dir / FALSE_REJECTS_FILE, [mining_payload(row, mining_label="false_reject_positive") for row in false_rejects])
    analysis = {
        "schema": "tensor_audio_error_shadow_analysis.v1",
        "scores_dir": str(scores_dir),
        "output_dir": str(output_dir),
        "model_kind": str(summary.get("model_kind") or ""),
        "calibration_status": str(summary.get("calibration_status") or ""),
        "rows_analyzed": len(rows),
        "negative_rows": len(negatives),
        "positive_rows": len(positives),
        "hard_negative_count": len(hard_negatives),
        "hard_negative_rows_written": len(capped_hard_negatives),
        "false_reject_count": len(false_rejects),
        "hard_negative_rate": safe_ratio(len(hard_negatives), len(negatives)),
        "false_reject_rate": safe_ratio(len(false_rejects), len(positives)),
        "shadow_decision_counts": dict(sorted(Counter(str(row.get("shadow_decision") or "unknown") for row in rows).items())),
        "by_stem": group_counts(rows, "stem_type"),
        "by_route": group_counts(rows, "route_id"),
        "by_candidate_family": group_counts(rows, "candidate_family"),
        "by_source_model_family": group_counts(rows, "source_model_family"),
        "hard_negative_candidates_path": str(output_dir / HARD_NEGATIVES_FILE),
        "false_reject_candidates_path": str(output_dir / FALSE_REJECTS_FILE),
        "model_improvement_report_path": str(output_dir / IMPROVEMENT_REPORT_FILE),
        **SAFETY_FLAGS,
    }
    write_json(output_dir / ANALYSIS_SUMMARY_FILE, analysis)
    write_report(output_dir / IMPROVEMENT_REPORT_FILE, analysis)
    print(json.dumps(analysis, ensure_ascii=False, sort_keys=True), flush=True)
    return analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze calibrated tensor critic shadow scores for hard-negative mining.")
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hard-negative-rows", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analyze_tensor_audio_error_shadow_scores(
        scores_dir=args.scores_dir,
        output_dir=args.output_dir,
        max_hard_negative_rows=args.max_hard_negative_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
