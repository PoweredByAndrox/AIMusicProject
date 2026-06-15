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

from tools.validate_tensor_audio_error_train_only_mining_bundle import (  # noqa: E402
    BLOCKED_STATUS,
    BUNDLE_MANIFEST_FILE,
    BUNDLE_ROWS_FILE,
    USABLE_STATUS,
    validate_tensor_audio_error_train_only_mining_bundle,
)
from tools.validate_tensor_audio_error_train_safe_mining_request import (  # noqa: E402
    REQUEST_MANIFEST_FILE,
    REQUEST_ROWS_FILE,
    validate_tensor_audio_error_train_safe_mining_request,
)


DATASET_MANIFEST_FILE = "locked_tensor_audio_error_dataset_manifest.json"
DATASET_ROWS_FILE = "tensor_audio_error_dataset.jsonl"
SPLIT_MANIFEST_FILE = "leak_safe_split_manifest.json"
LEAKAGE_REPORT_FILE = "leakage_report.json"
SPLIT_ROWS_FILE = "leak_safe_split_rows.jsonl"
HARDCASE_MANIFEST_FILE = "hardcase_holdout_manifest.json"
SLICE_ROWS_FILE = "tensor_audio_error_locked_mining_slice.jsonl"
BUNDLE_SUMMARY_FILE = "tensor_audio_error_train_only_mining_bundle_summary.md"
BLOCKED_SUMMARY_FILE = "blocked_summary.md"

SAFETY_FLAGS = {
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
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected_json_object:{path}:{line_number}")
            yield payload


def maybe_iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.exists() else []


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def has_true_flag(payload: Any, flag: str) -> bool:
    if isinstance(payload, dict):
        return payload.get(flag) is True or any(has_true_flag(value, flag) for value in payload.values())
    if isinstance(payload, list):
        return any(has_true_flag(item, flag) for item in payload)
    return False


def label_to_mining_label(label: int) -> str:
    return "hard_negative_false_accept" if int(label) == 0 else "false_reject_positive"


def route_family(route_id: str) -> str:
    value = str(route_id or "").strip()
    if not value:
        return "unknown"
    return value.split("__", 1)[0] or value


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


def blocked_keys_from_request(request_manifest: dict[str, Any], split_dir: Path) -> set[str]:
    keys: set[str] = set()
    source_slice_dir = Path(str(request_manifest.get("source_locked_slice_dir") or ""))
    if source_slice_dir.exists():
        for row in maybe_iter_jsonl(source_slice_dir / SLICE_ROWS_FILE):
            keys.update(row_key_set(row))
    for row in maybe_iter_jsonl(split_dir / SPLIT_ROWS_FILE):
        split = str(row.get("split") or "")
        if split in {"dev", "test", "hardcase", "holdout"} or row.get("hardcase_holdout") is True:
            keys.update(row_key_set(row))
    hardcase_manifest = split_dir / HARDCASE_MANIFEST_FILE
    if hardcase_manifest.exists():
        manifest = read_json(hardcase_manifest)
        for row in manifest.get("hardcase_rows", []) or []:
            if isinstance(row, dict):
                keys.update(row_key_set(row))
    return keys


def validate_sources(request_dir: Path, tensor_dataset_dir: Path, split_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    request_validation = validate_tensor_audio_error_train_safe_mining_request(request_dir)
    if request_validation.get("ok") is not True:
        raise ValueError(f"unsafe_train_safe_request:{','.join(request_validation.get('errors') or [])}")
    request_manifest = read_json(request_dir / REQUEST_MANIFEST_FILE)
    request_rows = list(iter_jsonl(request_dir / REQUEST_ROWS_FILE))
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
        errors.append("split_manifest_critical_leakage_present")
    if int(leakage_report.get("critical_leakage_count") or 0) != 0:
        errors.append("leakage_report_critical_leakage_present")
    for source_name, payload in (
        ("request_manifest", request_manifest),
        ("dataset_manifest", dataset_manifest),
        ("split_manifest", split_manifest),
        ("leakage_report", leakage_report),
    ):
        for flag in sorted(FORBIDDEN_TRUE_FLAGS):
            if has_true_flag(payload, flag):
                errors.append(f"{source_name}_forbidden_true_flag:{flag}")
    if errors:
        raise ValueError(f"unsafe_train_only_bundle_source:{','.join(sorted(set(errors)))}")
    dataset_rows = list(iter_jsonl(tensor_dataset_dir / DATASET_ROWS_FILE))
    return request_manifest, request_rows, dataset_rows


def match_score(row: dict[str, Any], requirement: dict[str, Any]) -> int:
    if str(row.get("stem_type") or "unknown") != str(requirement.get("stem_type") or "unknown"):
        return 0
    if int(row.get("label") or 0) != int(requirement.get("target_training_label") or 0):
        return 0
    score = 1
    row_top = str(row.get("top_error_channel") or row.get("top_error_channel_or_unknown") or "")
    req_top = str(requirement.get("top_error_channel_or_unknown") or "")
    if row_top and req_top and row_top == req_top:
        score = max(score, 4)
    blockers = {str(item) for item in row.get("blockers", []) or []}
    req_error = str(requirement.get("blocker_or_error_family_or_unknown") or "")
    if req_error and req_error != "unknown" and req_error in blockers:
        score = max(score, 3)
    if (
        str(row.get("source_model_family") or "") == str(requirement.get("source_model_family_or_unknown") or "")
        or str(row.get("candidate_family") or "") == str(requirement.get("candidate_family_or_unknown") or "")
    ):
        score = max(score, 2)
    return score


def eligible_train_row(row: dict[str, Any], blocked_keys: set[str]) -> tuple[bool, str]:
    if str(row.get("split") or "") != "train":
        return False, "split_not_train"
    if row.get("train_allowed") is not True:
        return False, "train_allowed_not_true"
    if row.get("hardcase_holdout") is True or row.get("hardcase") is True:
        return False, "hardcase_holdout_true"
    if row.get("release_ready") is True:
        return False, "release_ready_true"
    if row.get("scalar_features_used") is True:
        return False, "scalar_features_used_true"
    if row_key_set(row) & blocked_keys:
        return False, "locked_eval_or_hardcase_key_overlap"
    tensor_map_path = Path(str(row.get("tensor_map_path") or ""))
    if not tensor_map_path.exists():
        return False, "missing_tensor_map"
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag(row, flag):
            return False, f"forbidden_true_flag:{flag}"
    return True, ""


def candidate_row(source_row: dict[str, Any], requirement: dict[str, Any], *, request_dir: Path) -> dict[str, Any]:
    label = int(source_row.get("label") or 0)
    payload = {
        "source_requirement_id": str(requirement.get("requirement_id") or ""),
        "row_uid": str(source_row.get("row_uid") or ""),
        "canonical_row_key": str(source_row.get("canonical_row_key") or ""),
        "candidate_id": str(source_row.get("candidate_id") or ""),
        "stem_type": str(source_row.get("stem_type") or "unknown"),
        "window_start_sec": float(source_row.get("window_start_sec") or 0.0),
        "window_end_sec": float(source_row.get("window_end_sec") or 0.0),
    }
    return {
        "training_candidate_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "bundle_row_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "training_candidate_id": stable_id(payload),
        "source_requirement_id": str(requirement.get("requirement_id") or ""),
        "source_request_dir": str(request_dir),
        "row_uid": str(source_row.get("row_uid") or ""),
        "canonical_row_key": str(source_row.get("canonical_row_key") or ""),
        "track_id": str(source_row.get("track_id") or ""),
        "run_id": str(source_row.get("run_id") or ""),
        "candidate_id": str(source_row.get("candidate_id") or ""),
        "route_id": str(source_row.get("route_id") or ""),
        "candidate_family": str(source_row.get("candidate_family") or ""),
        "source_model_family": str(source_row.get("source_model_family") or ""),
        "stem_type": str(source_row.get("stem_type") or "unknown"),
        "split": "train",
        "train_allowed": True,
        "hardcase_holdout": False,
        "locked_eval_row": False,
        "tensor_map_path": str(source_row.get("tensor_map_path") or ""),
        "tensor_map_names": [str(item) for item in source_row.get("tensor_map_names", []) or []],
        "recommended_training_label": label,
        "source_true_label": label,
        "source_mining_label": label_to_mining_label(label),
        "label_source": "train_safe_analog_from_locked_eval_request",
        "training_candidate_status": "candidate_only",
        "eligible_for_future_training_review": True,
        "source_match_score": int(match_score(source_row, requirement)),
        "route_family_or_unknown": route_family(str(source_row.get("route_id") or "")),
        "scalar_features_used": False,
        **SAFETY_FLAGS,
    }


def build_candidates(
    *,
    request_dir: Path,
    requirements: Sequence[dict[str, Any]],
    dataset_rows: Sequence[dict[str, Any]],
    blocked_keys: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    reject_reasons: Counter[str] = Counter()
    rows_by_id: dict[str, dict[str, Any]] = {}
    for requirement in requirements:
        scored: list[tuple[int, dict[str, Any]]] = []
        for source_row in dataset_rows:
            ok, reason = eligible_train_row(source_row, blocked_keys)
            if not ok:
                reject_reasons[reason] += 1
                continue
            score = match_score(source_row, requirement)
            if score <= 0:
                continue
            scored.append((score, source_row))
        for _, source_row in sorted(scored, key=lambda item: (-item[0], str(item[1].get("row_uid") or ""))):
            row = candidate_row(source_row, requirement, request_dir=request_dir)
            rows_by_id[row["training_candidate_id"]] = row
    return [rows_by_id[key] for key in sorted(rows_by_id)], reject_reasons


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Train-Only Mining Bundle",
        "",
        f"Bundle status: {manifest['bundle_status']}",
        f"Candidate rows: {manifest['candidate_rows_total']}",
        f"Positive rows: {manifest['positive_rows']}",
        f"Negative rows: {manifest['negative_rows']}",
        f"Hard-negative rows: {manifest['hard_negative_rows']}",
        f"False-reject rows: {manifest['false_reject_rows']}",
        "",
        "## Safety",
        "",
        "- split=train only",
        "- dev_test_mining_rows_used_for_training=false",
        "- locked_eval_rows_used_for_training=false",
        "- hardcase_rows_used_for_training=false",
        "- runtime_gates_changed=false",
        "- release_ready=false",
        "- separator_training_started=false",
        "- repair_adapter_training_started=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_blocked_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Tensor R1 Train-Safe Loop Blocked",
        "",
        f"Blocked reason: {manifest['blocked_reason']}",
        f"Candidate rows: {manifest['candidate_rows_total']}",
        f"Positive rows: {manifest['positive_rows']}",
        f"Negative rows: {manifest['negative_rows']}",
        "",
        "R1J training was not run because the train-only mining bundle did not satisfy the minimum two-class safety contract.",
        "",
        "- release_ready=false",
        "- runtime_gates_changed=false",
        "- separator_training_started=false",
        "- repair_adapter_training_started=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_tensor_audio_error_train_only_mining_bundle(
    *,
    request_dir: Path,
    tensor_dataset_dir: Path,
    split_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_manifest, requirements, dataset_rows = validate_sources(request_dir, tensor_dataset_dir, split_dir)
    blocked_keys = blocked_keys_from_request(request_manifest, split_dir)
    candidates, reject_reasons = build_candidates(
        request_dir=request_dir,
        requirements=requirements,
        dataset_rows=dataset_rows,
        blocked_keys=blocked_keys,
    )
    label_counts = Counter(int(row.get("recommended_training_label") or 0) for row in candidates)
    mining_counts = Counter(str(row.get("source_mining_label") or "") for row in candidates)
    stem_counts = Counter(str(row.get("stem_type") or "unknown") for row in candidates)
    min_total = int(request_manifest.get("required_train_rows_min") or 40)
    min_positive = int(request_manifest.get("required_train_positive_rows_min") or 10)
    min_negative = int(request_manifest.get("required_train_negative_rows_min") or 10)
    positive_rows = int(label_counts.get(1) or 0)
    negative_rows = int(label_counts.get(0) or 0)
    enough = len(candidates) >= min_total and positive_rows >= min_positive and negative_rows >= min_negative
    status = USABLE_STATUS if enough else BLOCKED_STATUS
    training_allowed = status == USABLE_STATUS
    blocked_reasons: list[str] = []
    if len(candidates) < min_total:
        blocked_reasons.append("candidate_rows_below_minimum")
    if positive_rows < min_positive:
        blocked_reasons.append("positive_rows_below_minimum")
    if negative_rows < min_negative:
        blocked_reasons.append("negative_rows_below_minimum")
    if positive_rows == 0 or negative_rows == 0:
        blocked_reasons.append("missing_positive_or_negative_class")
    write_jsonl(output_dir / BUNDLE_ROWS_FILE, candidates)
    manifest = {
        "schema": "tensor_audio_error_train_only_mining_bundle.v1",
        "source_request_dir": str(request_dir),
        "source_tensor_dataset_dir": str(tensor_dataset_dir),
        "source_split_dir": str(split_dir),
        "bundle_status": status,
        "candidate_rows_total": len(candidates),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "hard_negative_rows": int(mining_counts.get("hard_negative_false_accept") or 0),
        "false_reject_rows": int(mining_counts.get("false_reject_positive") or 0),
        "stem_counts": dict(sorted(stem_counts.items())),
        "reject_reason_counts": dict(sorted(reject_reasons.items())),
        "min_train_rows_total": min_total,
        "min_train_positive_rows": min_positive,
        "min_train_negative_rows": min_negative,
        "min_per_stem_rows": int(request_manifest.get("min_per_stem_rows") or 10),
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
    write_json(output_dir / BUNDLE_MANIFEST_FILE, manifest)
    write_summary(output_dir / BUNDLE_SUMMARY_FILE, manifest)
    validation = validate_tensor_audio_error_train_only_mining_bundle(output_dir)
    if validation.get("ok") is not True:
        raise ValueError(
            "train_only_mining_bundle_validation_failed:"
            + ",".join(str(error) for error in validation.get("errors", []) or [])
        )
    if status == BLOCKED_STATUS:
        write_blocked_summary(output_dir.parent / BLOCKED_SUMMARY_FILE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-only mining bundle from a train-safe tensor critic request.")
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--tensor-dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_tensor_audio_error_train_only_mining_bundle(
        request_dir=args.request_dir,
        tensor_dataset_dir=args.tensor_dataset_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
