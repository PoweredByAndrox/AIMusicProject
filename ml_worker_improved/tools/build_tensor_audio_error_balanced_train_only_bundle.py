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

from tools.validate_tensor_audio_error_balanced_train_only_bundle import (
    BUNDLE_MANIFEST_FILE,
    BUNDLE_ROWS_FILE,
    USABLE_STATUS,
    validate_tensor_audio_error_balanced_train_only_bundle,
)


REQUEST_MANIFEST_FILE = "tensor_audio_error_train_safe_mining_request_manifest.json"
POSITIVE_BUNDLE_MANIFEST_FILE = "tensor_audio_error_train_only_mining_bundle_manifest.json"
POSITIVE_BUNDLE_ROWS_FILE = "tensor_audio_error_train_only_training_candidates.jsonl"
HARVEST_MANIFEST_FILE = "tensor_audio_error_train_hard_negative_harvest_manifest.json"
HARVEST_ROWS_FILE = "tensor_audio_error_train_hard_negative_candidates.jsonl"
DATASET_MANIFEST_FILE = "locked_tensor_audio_error_dataset_manifest.json"
SPLIT_MANIFEST_FILE = "leak_safe_split_manifest.json"
LEAKAGE_REPORT_FILE = "leakage_report.json"
BUNDLE_SUMMARY_FILE = "tensor_audio_error_train_only_mining_bundle_summary.md"
ROOT_BLOCKED_SUMMARY_FILE = "blocked_summary.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
                raise ValueError(f"expected_json_object:{path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def row_key(row: dict[str, Any]) -> str:
    for key in ("row_uid", "canonical_row_key"):
        value = str(row.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return "candidate_window:" + "|".join(
        [
            str(row.get("candidate_id") or ""),
            str(row.get("stem_type") or ""),
            str(float(row.get("window_start_sec") or 0.0)),
            str(float(row.get("window_end_sec") or 0.0)),
        ]
    )


def normalize_positive_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "training_candidate_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "bundle_row_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "recommended_training_label": int(row.get("recommended_training_label") or row.get("source_true_label") or 1),
        "source_true_label": int(row.get("source_true_label") or row.get("recommended_training_label") or 1),
        "source_mining_label": "false_reject_positive",
        "training_candidate_status": "candidate_only",
        "eligible_for_future_training_review": True,
        "split": "train",
        "train_allowed": True,
        "hardcase_holdout": False,
        "locked_eval_row": False,
        "scalar_features_used": False,
        "runtime_gate_decision_applied": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "release_ready": False,
        "shadow_only": True,
    }


def row_from_harvest(row: dict[str, Any], *, request_dir: Path) -> dict[str, Any]:
    payload = {
        "harvest_candidate_id": str(row.get("harvest_candidate_id") or ""),
        "row_uid": str(row.get("row_uid") or ""),
        "canonical_row_key": str(row.get("canonical_row_key") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "window_start_sec": float(row.get("window_start_sec") or 0.0),
        "window_end_sec": float(row.get("window_end_sec") or 0.0),
    }
    return {
        "training_candidate_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "bundle_row_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "training_candidate_id": stable_id(payload),
        "source_requirement_id": "r1m_hard_negative_harvest",
        "source_request_dir": str(request_dir),
        "row_uid": payload["row_uid"],
        "canonical_row_key": payload["canonical_row_key"],
        "track_id": str(row.get("track_id") or ""),
        "run_id": str(row.get("run_id") or ""),
        "candidate_id": payload["candidate_id"],
        "route_id": str(row.get("route_id") or ""),
        "candidate_family": str(row.get("candidate_family") or ""),
        "source_model_family": str(row.get("source_model_family") or ""),
        "stem_type": payload["stem_type"],
        "split": "train",
        "train_allowed": True,
        "hardcase_holdout": False,
        "locked_eval_row": False,
        "tensor_map_path": str(row.get("tensor_map_path") or ""),
        "tensor_map_names": [str(item) for item in row.get("tensor_map_names", []) or []],
        "recommended_training_label": 0,
        "source_true_label": 0,
        "source_mining_label": "hard_negative_false_accept",
        "label_source": "train_split_existing_judge_or_train_only_generation",
        "training_candidate_status": "candidate_only",
        "eligible_for_future_training_review": True,
        "source_match_score": 5,
        "route_family_or_unknown": str(row.get("route_id") or "").split("__", 1)[0] or "unknown",
        "scalar_features_used": False,
        "runtime_gate_decision_applied": False,
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


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Balanced Train-Only Mining Bundle",
        "",
        f"Bundle status: {manifest['bundle_status']}",
        f"Candidate rows: {manifest['candidate_rows_total']}",
        f"Positive rows: {manifest['positive_rows']}",
        f"Negative rows: {manifest['negative_rows']}",
        f"Hard-negative rows: {manifest['hard_negative_rows']}",
        f"Blocked reason: {manifest['blocked_reason']}",
        "",
        "## Safety",
        "",
        "- split=train only",
        "- dev_test_mining_rows_used_for_training=false",
        "- locked_eval_rows_used_for_training=false",
        "- hardcase_rows_used_for_training=false",
        "- runtime_gates_changed=false",
        "- release_ready=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_tensor_audio_error_balanced_train_only_bundle(
    *,
    request_dir: Path,
    positive_bundle_dir: Path,
    hard_negative_harvest_dir: Path,
    tensor_dataset_dir: Path,
    split_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_manifest = read_json(request_dir / REQUEST_MANIFEST_FILE)
    positive_manifest = read_json(positive_bundle_dir / POSITIVE_BUNDLE_MANIFEST_FILE)
    harvest_manifest = read_json(hard_negative_harvest_dir / HARVEST_MANIFEST_FILE)
    dataset_manifest = read_json(tensor_dataset_dir / DATASET_MANIFEST_FILE)
    split_manifest = read_json(split_dir / SPLIT_MANIFEST_FILE)
    leakage_report = read_json(split_dir / LEAKAGE_REPORT_FILE)

    errors: list[str] = []
    if dataset_manifest.get("tensor_maps_required") is not True:
        errors.append("tensor_dataset_not_tensor_maps_required")
    if dataset_manifest.get("scalar_fallback_used") is True:
        errors.append("tensor_dataset_scalar_fallback_used")
    if int(dataset_manifest.get("critical_leakage_count") or 0) != 0:
        errors.append("tensor_dataset_critical_leakage_present")
    if int(split_manifest.get("critical_leakage_count") or 0) != 0:
        errors.append("split_critical_leakage_present")
    if int(leakage_report.get("critical_leakage_count") or 0) != 0:
        errors.append("leakage_report_critical_leakage_present")

    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(positive_bundle_dir / POSITIVE_BUNDLE_ROWS_FILE):
        normalized = normalize_positive_row(row)
        if int(normalized.get("recommended_training_label") or 0) == 1:
            rows_by_key[row_key(normalized)] = normalized
    if harvest_manifest.get("harvest_status") == "harvest_ready":
        for row in iter_jsonl(hard_negative_harvest_dir / HARVEST_ROWS_FILE):
            normalized = row_from_harvest(row, request_dir=request_dir)
            rows_by_key[row_key(normalized)] = normalized

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    label_counts = Counter(int(row.get("recommended_training_label") or 0) for row in rows)
    mining_counts = Counter(str(row.get("source_mining_label") or "") for row in rows)
    stem_counts = Counter(str(row.get("stem_type") or "unknown") for row in rows)
    min_total = int(request_manifest.get("required_train_rows_min") or 40)
    min_positive = int(request_manifest.get("required_train_positive_rows_min") or 10)
    min_negative = int(request_manifest.get("required_train_negative_rows_min") or 10)
    positive_rows = int(label_counts.get(1) or 0)
    negative_rows = int(label_counts.get(0) or 0)
    hard_negative_rows = int(mining_counts.get("hard_negative_false_accept") or 0)

    blocked_reasons: list[str] = list(errors)
    if positive_rows < min_positive:
        blocked_reasons.append("positive_rows_below_minimum")
    if negative_rows < min_negative:
        blocked_reasons.append("negative_rows_below_minimum")
    if hard_negative_rows < min_negative:
        blocked_reasons.append("hard_negative_rows_below_minimum")
    if len(rows) < min_total:
        blocked_reasons.append("candidate_rows_below_minimum")
    if positive_rows == 0:
        blocked_reasons.append("missing_positive_class")
    if negative_rows == 0:
        blocked_reasons.append("missing_negative_class")

    if errors:
        status = "blocked_leakage_risk"
    elif negative_rows < min_negative or hard_negative_rows < min_negative:
        status = "blocked_missing_negative_class"
    elif positive_rows < min_positive:
        status = "blocked_missing_positive_class"
    elif len(rows) < min_total:
        status = "blocked_insufficient_train_safe_rows"
    else:
        status = USABLE_STATUS
    training_allowed = status == USABLE_STATUS

    manifest = {
        "schema": "tensor_audio_error_train_only_mining_bundle.v1",
        "source_request_dir": str(request_dir),
        "source_positive_bundle_dir": str(positive_bundle_dir),
        "source_hard_negative_harvest_dir": str(hard_negative_harvest_dir),
        "source_tensor_dataset_dir": str(tensor_dataset_dir),
        "source_split_dir": str(split_dir),
        "bundle_status": status,
        "candidate_rows_total": len(rows),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "hard_negative_rows": hard_negative_rows,
        "false_reject_rows": int(mining_counts.get("false_reject_positive") or 0),
        "stem_counts": dict(sorted(stem_counts.items())),
        "min_train_rows_total": min_total,
        "min_train_positive_rows": min_positive,
        "min_train_negative_rows": min_negative,
        "source_positive_rows": int(positive_manifest.get("positive_rows") or 0),
        "source_harvest_negative_rows": int(harvest_manifest.get("negative_rows") or 0),
        "blocked_reason": ",".join(sorted(set(blocked_reasons))),
        "training_allowed": training_allowed,
        "tensor_maps_required": True,
        "scalar_features_used": False,
        "dev_test_mining_rows_used_for_training": False,
        "locked_eval_rows_used_for_training": False,
        "hardcase_rows_used_for_training": False,
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
    write_jsonl(output_dir / BUNDLE_ROWS_FILE, rows)
    write_json(output_dir / BUNDLE_MANIFEST_FILE, manifest)
    write_summary(output_dir / BUNDLE_SUMMARY_FILE, manifest)
    validation = validate_tensor_audio_error_balanced_train_only_bundle(output_dir)
    if validation.get("ok") is not True:
        raise ValueError("balanced_train_only_bundle_validation_failed:" + ",".join(validation.get("errors") or []))
    if status != USABLE_STATUS:
        write_summary(output_dir.parent / ROOT_BLOCKED_SUMMARY_FILE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced train-only tensor critic mining bundle.")
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--positive-bundle-dir", type=Path, required=True)
    parser.add_argument("--hard-negative-harvest-dir", type=Path, required=True)
    parser.add_argument("--tensor-dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_tensor_audio_error_balanced_train_only_bundle(
        request_dir=args.request_dir,
        positive_bundle_dir=args.positive_bundle_dir,
        hard_negative_harvest_dir=args.hard_negative_harvest_dir,
        tensor_dataset_dir=args.tensor_dataset_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
