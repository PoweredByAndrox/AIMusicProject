#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASET_ROWS_FILE = "tensor_audio_error_dataset.jsonl"
DATASET_MANIFEST_FILE = "locked_tensor_audio_error_dataset_manifest.json"
SPLIT_ROWS_FILE = "leak_safe_split_rows.jsonl"
HARDCASE_MANIFEST_FILE = "hardcase_holdout_manifest.json"
REQUEST_ROWS_FILE = "tensor_audio_error_train_safe_mining_requirements.jsonl"
REQUEST_MANIFEST_FILE = "tensor_audio_error_train_safe_mining_request_manifest.json"
BUNDLE_ROWS_FILE = "tensor_audio_error_train_only_training_candidates.jsonl"
BUNDLE_MANIFEST_FILE = "tensor_audio_error_train_only_mining_bundle_manifest.json"

AUDIT_MANIFEST_FILE = "tensor_audio_error_train_negative_gap_audit.json"
AUDIT_ROWS_FILE = "tensor_audio_error_train_negative_gap_rows.jsonl"
AUDIT_SUMMARY_FILE = "tensor_audio_error_train_negative_gap_summary.md"

FORBIDDEN_TRUE_FLAGS = {
    "release_ready",
    "can_publish_as_result",
    "runtime_gate_decision_applied",
    "runtime_gates_changed",
    "training_started",
    "critic_training_started",
    "separator_training_started",
    "repair_adapter_training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
}


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


def has_true_flag(payload: Any, flag: str) -> bool:
    if isinstance(payload, dict):
        return payload.get(flag) is True or any(has_true_flag(value, flag) for value in payload.values())
    if isinstance(payload, list):
        return any(has_true_flag(item, flag) for item in payload)
    return False


def row_key_set(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in (
        "row_uid",
        "canonical_row_key",
        "candidate_id",
        "track_id",
        "exact_audio_sha256",
        "near_duplicate_audio_hash",
        "audio_fingerprint",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            keys.add(f"{key}:{value}")
    return keys


def hardcase_keys(split_dir: Path) -> set[str]:
    keys: set[str] = set()
    for row in iter_jsonl(split_dir / SPLIT_ROWS_FILE):
        if str(row.get("split") or "") in {"dev", "test", "hardcase", "holdout"} or row.get("hardcase_holdout") is True:
            keys.update(row_key_set(row))
    manifest_path = split_dir / HARDCASE_MANIFEST_FILE
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for row in manifest.get("hardcase_rows", []) or []:
            if isinstance(row, dict):
                keys.update(row_key_set(row))
    return keys


def tensor_map_exists(row: dict[str, Any]) -> bool:
    path = Path(str(row.get("tensor_map_path") or ""))
    return path.exists()


def is_negative(row: dict[str, Any]) -> bool:
    return int(row.get("label") or 0) == 0


def safe_train_negative_reason(row: dict[str, Any], blocked_keys: set[str]) -> str:
    if str(row.get("split") or "") != "train":
        return "split_not_train"
    if row.get("train_allowed") is not True:
        return "train_allowed_not_true"
    if row.get("hardcase_holdout") is True:
        return "hardcase_holdout_true"
    if row_key_set(row) & blocked_keys:
        return "locked_eval_or_hardcase_key_overlap"
    if row.get("scalar_features_used") is True:
        return "scalar_fallback_used"
    if row.get("release_ready") is True:
        return "release_ready_true"
    if not tensor_map_exists(row):
        return "missing_tensor_map"
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag(row, flag):
            return f"forbidden_true_flag:{flag}"
    return ""


def diagnostic_decision(
    *,
    train_negative_rows_total: int,
    train_safe_negative_rows_total: int,
    train_positive_rows_total: int,
    missing_tensor_maps: int,
    leakage_exclusions: int,
    r1i_negative_rows: int,
) -> str:
    if train_negative_rows_total == 0:
        return "train_negatives_absent_in_tensor_dataset"
    if train_positive_rows_total == 0:
        return "train_labels_one_class_only"
    if train_safe_negative_rows_total == 0 and missing_tensor_maps >= train_negative_rows_total:
        return "train_negatives_exist_but_missing_tensor_maps"
    if train_safe_negative_rows_total == 0 and leakage_exclusions >= train_negative_rows_total:
        return "train_negatives_exist_but_leakage_excluded"
    if r1i_negative_rows == 0:
        return "train_negatives_exist_but_r1i_filter_too_strict"
    return "train_negatives_exist_but_r1i_filter_too_strict"


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Train Negative Gap Audit",
        "",
        f"Diagnostic decision: {manifest['diagnostic_decision']}",
        f"Train positives: {manifest['train_positive_rows_total']}",
        f"Train negatives: {manifest['train_negative_rows_total']}",
        f"Train-safe negatives: {manifest['train_safe_negative_rows_total']}",
        f"R1I positives: {manifest['r1i_positive_rows']}",
        f"R1I negatives: {manifest['r1i_negative_rows']}",
        "",
        "## Safety",
        "",
        "- training_started=false",
        "- critic_training_started=false",
        "- separator_training_started=false",
        "- repair_adapter_training_started=false",
        "- runtime_gates_changed=false",
        "- release_ready=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def audit_tensor_audio_error_train_negative_gap(
    *,
    tensor_dataset_dir: Path,
    split_dir: Path,
    request_dir: Path,
    bundle_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = read_json(tensor_dataset_dir / DATASET_MANIFEST_FILE)
    request_manifest = read_json(request_dir / REQUEST_MANIFEST_FILE)
    bundle_manifest = read_json(bundle_dir / BUNDLE_MANIFEST_FILE)
    dataset_rows = list(iter_jsonl(tensor_dataset_dir / DATASET_ROWS_FILE))
    request_rows = list(iter_jsonl(request_dir / REQUEST_ROWS_FILE))
    bundle_rows = list(iter_jsonl(bundle_dir / BUNDLE_ROWS_FILE))
    blocked_keys = hardcase_keys(split_dir)

    split_counts = Counter(str(row.get("split") or "unknown") for row in dataset_rows)
    label_counts_by_split = Counter(f"{row.get('split') or 'unknown'}:{int(row.get('label') or 0)}" for row in dataset_rows)
    stem_counts_by_split = Counter(f"{row.get('split') or 'unknown'}:{row.get('stem_type') or 'unknown'}" for row in dataset_rows)
    label_counts_by_stem = Counter(f"{row.get('stem_type') or 'unknown'}:{int(row.get('label') or 0)}" for row in dataset_rows)
    request_target_counts = Counter(str(row.get("target_mining_label") or "unknown") for row in request_rows)

    train_rows = [row for row in dataset_rows if str(row.get("split") or "") == "train"]
    train_negative_rows = [row for row in train_rows if is_negative(row)]
    train_positive_rows = [row for row in train_rows if not is_negative(row)]
    dev_test_hardcase_negative_rows = [
        row
        for row in dataset_rows
        if str(row.get("split") or "") in {"dev", "test", "hardcase", "holdout"} and is_negative(row)
    ]

    audit_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    safe_train_negative_rows = 0
    for index, row in enumerate(dataset_rows):
        reason = safe_train_negative_reason(row, blocked_keys) if is_negative(row) else ""
        if reason:
            exclusion_counts[reason] += 1
        safe = is_negative(row) and reason == ""
        if safe:
            safe_train_negative_rows += 1
        audit_rows.append(
            {
                "audit_row_schema": "tensor_audio_error_train_negative_gap_row.v1",
                "dataset_index": index,
                "row_uid": str(row.get("row_uid") or ""),
                "canonical_row_key": str(row.get("canonical_row_key") or ""),
                "track_id": str(row.get("track_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "stem_type": str(row.get("stem_type") or "unknown"),
                "split": str(row.get("split") or "unknown"),
                "label": int(row.get("label") or 0),
                "train_allowed": row.get("train_allowed") is True,
                "hardcase_holdout": row.get("hardcase_holdout") is True or row.get("hardcase") is True,
                "tensor_map_exists": tensor_map_exists(row),
                "scalar_fallback_used": row.get("scalar_features_used") is True,
                "exclusion_reason": reason,
                "safe_for_hard_negative_harvest": safe,
                "release_ready": False,
            }
        )

    r1i_positive_rows = int(bundle_manifest.get("positive_rows") or 0)
    r1i_negative_rows = int(bundle_manifest.get("negative_rows") or 0)
    missing_tensor_maps = sum(1 for row in train_negative_rows if not tensor_map_exists(row))
    leakage_exclusions = int(exclusion_counts.get("locked_eval_or_hardcase_key_overlap") or 0)
    decision = diagnostic_decision(
        train_negative_rows_total=len(train_negative_rows),
        train_safe_negative_rows_total=safe_train_negative_rows,
        train_positive_rows_total=len(train_positive_rows),
        missing_tensor_maps=missing_tensor_maps,
        leakage_exclusions=leakage_exclusions,
        r1i_negative_rows=r1i_negative_rows,
    )
    manifest = {
        "schema": "tensor_audio_error_train_negative_gap_audit.v1",
        "source_tensor_dataset_dir": str(tensor_dataset_dir),
        "source_split_dir": str(split_dir),
        "source_request_dir": str(request_dir),
        "source_bundle_dir": str(bundle_dir),
        "diagnostic_decision": decision,
        "dataset_rows_total": len(dataset_rows),
        "total_rows_by_split": dict(sorted(split_counts.items())),
        "label_counts_by_split": dict(sorted(label_counts_by_split.items())),
        "stem_type_counts_by_split": dict(sorted(stem_counts_by_split.items())),
        "label_counts_by_stem_type": dict(sorted(label_counts_by_stem.items())),
        "request_target_mining_label_counts": dict(sorted(request_target_counts.items())),
        "request_negative_targets": int(request_target_counts.get("hard_negative_false_accept") or 0),
        "train_negative_rows_total": len(train_negative_rows),
        "train_positive_rows_total": len(train_positive_rows),
        "train_safe_negative_rows_total": safe_train_negative_rows,
        "train_hard_negative_analog_candidates": safe_train_negative_rows,
        "dev_test_hardcase_negative_rows_total": len(dev_test_hardcase_negative_rows),
        "dev_test_hardcase_hard_negative_rows_total": len(dev_test_hardcase_negative_rows),
        "r1i_candidate_rows_total": int(bundle_manifest.get("candidate_rows_total") or len(bundle_rows)),
        "r1i_positive_rows": r1i_positive_rows,
        "r1i_negative_rows": r1i_negative_rows,
        "r1i_hard_negative_rows": int(bundle_manifest.get("hard_negative_rows") or 0),
        "rows_excluded_from_r1i_and_why": dict(sorted(exclusion_counts.items())),
        "candidate_leakage_exclusions": leakage_exclusions,
        "track_leakage_exclusions": leakage_exclusions,
        "canonical_row_key_leakage_exclusions": leakage_exclusions,
        "missing_tensor_maps": missing_tensor_maps,
        "scalar_fallback_exclusions": int(exclusion_counts.get("scalar_fallback_used") or 0),
        "hardcase_holdout_exclusions": int(exclusion_counts.get("hardcase_holdout_true") or 0),
        "train_allowed_false_exclusions": int(exclusion_counts.get("train_allowed_not_true") or 0),
        "tensor_maps_required": dataset_manifest.get("tensor_maps_required") is True,
        "scalar_fallback_used": dataset_manifest.get("scalar_fallback_used") is True,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "shadow_only": True,
    }
    write_jsonl(output_dir / AUDIT_ROWS_FILE, audit_rows)
    write_json(output_dir / AUDIT_MANIFEST_FILE, manifest)
    write_summary(output_dir / AUDIT_SUMMARY_FILE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit why train-safe tensor hard negatives are missing.")
    parser.add_argument("--tensor-dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_tensor_audio_error_train_negative_gap(
        tensor_dataset_dir=args.tensor_dataset_dir,
        split_dir=args.split_dir,
        request_dir=args.request_dir,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
