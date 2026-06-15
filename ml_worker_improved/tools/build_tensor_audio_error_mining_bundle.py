#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.analyze_tensor_audio_error_shadow_scores import (  # noqa: E402
    ANALYSIS_SUMMARY_FILE,
    FALSE_REJECTS_FILE,
    HARD_NEGATIVES_FILE,
    has_forbidden_true,
)
from tools.train_tensor_audio_error_critic import read_json, write_json, write_jsonl  # noqa: E402


TRAINING_CANDIDATES_FILE = "tensor_audio_error_training_candidates.jsonl"
MINING_BUNDLE_MANIFEST_FILE = "tensor_audio_error_mining_bundle_manifest.json"
MINING_BUNDLE_SUMMARY_FILE = "tensor_audio_error_mining_bundle_summary.md"

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


def stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def validate_mining_row(
    row: dict[str, Any],
    *,
    row_type: str,
    index: int,
    expected_mining_label: str,
    expected_true_label: int,
    expected_shadow_decision: str,
) -> list[str]:
    errors: list[str] = []
    if str(row.get("mining_label") or "") != expected_mining_label:
        errors.append(f"{row_type}_row_invalid_mining_label:{index}")
    if int(row.get("true_label") or 0) != expected_true_label:
        errors.append(f"{row_type}_row_invalid_true_label:{index}")
    if str(row.get("shadow_decision") or "") != expected_shadow_decision:
        errors.append(f"{row_type}_row_invalid_shadow_decision:{index}")
    return errors


def validate_analysis(summary: dict[str, Any], hard_negative_rows: Sequence[dict[str, Any]], false_reject_rows: Sequence[dict[str, Any]]) -> None:
    errors: list[str] = []
    if summary.get("schema") != "tensor_audio_error_shadow_analysis.v1":
        errors.append("invalid_analysis_schema")
    if summary.get("shadow_only") is not True:
        errors.append("analysis_shadow_only_not_true")
    if has_forbidden_true(summary):
        errors.append("analysis_forbidden_true_flag")
    if int(summary.get("hard_negative_rows_written") or 0) != len(hard_negative_rows):
        errors.append("hard_negative_rows_written_mismatch")
    if int(summary.get("false_reject_count") or 0) != len(false_reject_rows):
        errors.append("false_reject_rows_mismatch")
    for label, rows in (("hard_negative", hard_negative_rows), ("false_reject", false_reject_rows)):
        for index, row in enumerate(rows):
            if row.get("shadow_only") is not True:
                errors.append(f"{label}_row_shadow_only_not_true:{index}")
            if has_forbidden_true(row):
                errors.append(f"{label}_row_forbidden_true_flag:{index}")
            if label == "hard_negative":
                errors.extend(
                    validate_mining_row(
                        row,
                        row_type=label,
                        index=index,
                        expected_mining_label="hard_negative_false_accept",
                        expected_true_label=0,
                        expected_shadow_decision="shadow_accept_candidate",
                    )
                )
            else:
                errors.extend(
                    validate_mining_row(
                        row,
                        row_type=label,
                        index=index,
                        expected_mining_label="false_reject_positive",
                        expected_true_label=1,
                        expected_shadow_decision="shadow_block_candidate",
                    )
                )
    if errors:
        raise ValueError(f"unsafe_shadow_analysis:{','.join(errors)}")


def recommended_label(mining_label: str) -> int:
    if mining_label == "hard_negative_false_accept":
        return 0
    if mining_label == "false_reject_positive":
        return 1
    raise ValueError(f"unknown_mining_label:{mining_label}")


def bundle_row(row: dict[str, Any], *, source_analysis_dir: Path) -> dict[str, Any]:
    mining_label = str(row.get("mining_label") or "")
    key_payload = {
        "mining_label": mining_label,
        "row_uid": str(row.get("row_uid") or ""),
        "canonical_row_key": str(row.get("canonical_row_key") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "stem_type": str(row.get("stem_type") or ""),
        "window_start_sec": float(row.get("window_start_sec") or 0.0),
        "window_end_sec": float(row.get("window_end_sec") or 0.0),
    }
    label = recommended_label(mining_label)
    return {
        "training_candidate_id": stable_id(key_payload),
        "bundle_row_schema": "tensor_audio_error_training_candidate.v1",
        "source_analysis_dir": str(source_analysis_dir),
        "source_mining_label": mining_label,
        "recommended_training_label": label,
        "source_true_label": int(row.get("true_label") or 0),
        "label_source": "shadow_error_mining",
        "eligible_for_future_training_review": True,
        "training_candidate_status": "candidate_only",
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
        "window_start_sec": float(row.get("window_start_sec") or 0.0),
        "window_end_sec": float(row.get("window_end_sec") or 0.0),
        "predicted_accept_probability": float(row.get("predicted_accept_probability") or 0.0),
        "calibrated_threshold": float(row.get("calibrated_threshold") or 0.0),
        "threshold_scope": str(row.get("threshold_scope") or ""),
        "shadow_decision": str(row.get("shadow_decision") or ""),
        "runtime_gate_decision_applied": False,
        **SAFETY_FLAGS,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Mining Bundle",
        "",
        f"Candidate rows: {manifest['candidate_rows_total']}",
        f"Hard negative rows: {manifest['hard_negative_rows']}",
        f"False reject rows: {manifest['false_reject_rows']}",
        f"Bundle status: {manifest['bundle_status']}",
        "",
        "## Safety",
        "",
        "- training_started=false",
        "- runtime_gates_changed=false",
        "- release_ready=false",
        "- separator_training_started=false",
        "- repair_adapter_training_started=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_tensor_audio_error_mining_bundle(
    *,
    analysis_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(analysis_dir / ANALYSIS_SUMMARY_FILE)
    hard_negative_rows = list(iter_jsonl(analysis_dir / HARD_NEGATIVES_FILE))
    false_reject_rows = list(iter_jsonl(analysis_dir / FALSE_REJECTS_FILE))
    validate_analysis(summary, hard_negative_rows, false_reject_rows)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for source_row in [*hard_negative_rows, *false_reject_rows]:
        candidate = bundle_row(source_row, source_analysis_dir=analysis_dir)
        rows_by_id[candidate["training_candidate_id"]] = candidate
    candidate_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    write_jsonl(output_dir / TRAINING_CANDIDATES_FILE, candidate_rows)
    label_counts = dict(sorted(Counter(str(row["recommended_training_label"]) for row in candidate_rows).items()))
    mining_counts = dict(sorted(Counter(str(row["source_mining_label"]) for row in candidate_rows).items()))
    manifest = {
        "schema": "tensor_audio_error_mining_bundle.v1",
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "source_scores_dir": str(summary.get("scores_dir") or ""),
        "source_model_kind": str(summary.get("model_kind") or ""),
        "source_calibration_status": str(summary.get("calibration_status") or ""),
        "bundle_status": "candidate_only",
        "candidate_rows_total": len(candidate_rows),
        "hard_negative_rows": int(mining_counts.get("hard_negative_false_accept") or 0),
        "false_reject_rows": int(mining_counts.get("false_reject_positive") or 0),
        "recommended_label_counts": label_counts,
        "mining_label_counts": mining_counts,
        "training_candidates_path": str(output_dir / TRAINING_CANDIDATES_FILE),
        "summary_path": str(output_dir / MINING_BUNDLE_SUMMARY_FILE),
        "eligible_for_future_training_review": True,
        "training_started": False,
        "training_completed": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "shadow_only": True,
    }
    write_json(output_dir / MINING_BUNDLE_MANIFEST_FILE, manifest)
    write_summary(output_dir / MINING_BUNDLE_SUMMARY_FILE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a candidate-only tensor audio error mining bundle from shadow analysis.")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_tensor_audio_error_mining_bundle(analysis_dir=args.analysis_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
