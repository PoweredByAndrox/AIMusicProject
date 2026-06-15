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

from tools.validate_tensor_audio_error_train_safe_mining_request import (  # noqa: E402
    REQUEST_MANIFEST_FILE,
    REQUEST_ROWS_FILE,
    REQUEST_VALIDATION_FILE,
    validate_tensor_audio_error_train_safe_mining_request,
)


SLICE_MANIFEST_FILE = "tensor_audio_error_locked_mining_slice_manifest.json"
SLICE_ROWS_FILE = "tensor_audio_error_locked_mining_slice.jsonl"
SLICE_VALIDATION_FILE = "tensor_audio_error_locked_mining_slice_validation.json"
SPLIT_MANIFEST_FILE = "leak_safe_split_manifest.json"
LEAKAGE_REPORT_FILE = "leakage_report.json"
DATASET_MANIFEST_FILE = "locked_tensor_audio_error_dataset_manifest.json"
REQUEST_SUMMARY_FILE = "tensor_audio_error_train_safe_mining_request_summary.md"

DEFAULT_THRESHOLDS = {
    "required_train_rows_min": 40,
    "required_train_positive_rows_min": 10,
    "required_train_negative_rows_min": 10,
    "min_per_label_rows": 10,
    "min_per_stem_rows": 10,
    "min_hard_negative_rows": 10,
    "min_false_reject_rows": 10,
}
SAFETY_FLAGS = {
    "training_allowed": False,
    "training_rows_created": False,
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
    "training_allowed",
    "training_started",
    "critic_training_started",
    "separator_training_started",
    "repair_adapter_training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
}
SLICE_LABEL_TO_TARGET = {
    "persistent_hard_negative": ("hard_negative_false_accept", 0),
    "candidate_regressed_hard_negative": ("hard_negative_false_accept", 0),
    "persistent_false_reject": ("false_reject_positive", 1),
    "candidate_regressed_false_reject": ("false_reject_positive", 1),
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


def first_nonempty(row: dict[str, Any], keys: Sequence[str], default: str = "unknown") -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return default


def route_family(route_id: str) -> str:
    value = str(route_id or "").strip()
    if not value:
        return "unknown"
    return value.split("__", 1)[0] or value


def validate_inputs(
    *,
    locked_slice_dir: Path,
    split_dir: Path,
    tensor_dataset_dir: Path,
    slice_manifest: dict[str, Any],
    slice_validation: dict[str, Any],
    slice_rows: Sequence[dict[str, Any]],
) -> None:
    errors: list[str] = []
    if not locked_slice_dir.exists():
        errors.append("source_locked_slice_missing")
    if slice_manifest.get("schema") != "tensor_audio_error_locked_mining_slice.v1":
        errors.append("invalid_source_slice_schema")
    if slice_manifest.get("locked_eval_only") is not True:
        errors.append("source_slice_not_locked_eval_only")
    if slice_manifest.get("training_allowed") is True:
        errors.append("source_locked_slice_training_allowed_true")
    if slice_validation.get("ok") is not True:
        errors.append("source_locked_slice_validation_not_ok")
    if int(slice_manifest.get("candidate_rows_total") or 0) != len(slice_rows):
        errors.append("source_locked_slice_row_count_mismatch")
    for index, row in enumerate(slice_rows):
        if str(row.get("split") or "") == "train":
            errors.append(f"source_locked_slice_has_train_row:{index}")
        if row.get("training_allowed") is True:
            errors.append("source_locked_slice_has_training_allowed_row")
        for flag in sorted(FORBIDDEN_TRUE_FLAGS):
            if has_true_flag(row, flag):
                errors.append(f"source_locked_slice_row_forbidden_true_flag:{index}:{flag}")
    split_manifest = read_json(split_dir / SPLIT_MANIFEST_FILE)
    leakage_report = read_json(split_dir / LEAKAGE_REPORT_FILE)
    dataset_manifest = read_json(tensor_dataset_dir / DATASET_MANIFEST_FILE)
    if int(split_manifest.get("critical_leakage_count") or 0) != 0:
        errors.append("split_manifest_critical_leakage_present")
    if int(leakage_report.get("critical_leakage_count") or 0) != 0:
        errors.append("leakage_report_critical_leakage_present")
    if dataset_manifest.get("tensor_maps_required") is not True:
        errors.append("tensor_dataset_not_tensor_maps_required")
    if dataset_manifest.get("scalar_fallback_used") is True:
        errors.append("tensor_dataset_scalar_fallback_used")
    for source_name, payload in (
        ("source_slice", slice_manifest),
        ("split_manifest", split_manifest),
        ("leakage_report", leakage_report),
        ("tensor_dataset", dataset_manifest),
    ):
        for flag in sorted(FORBIDDEN_TRUE_FLAGS - {"training_allowed"}):
            if has_true_flag(payload, flag):
                errors.append(f"{source_name}_forbidden_true_flag:{flag}")
    if errors:
        raise ValueError(f"unsafe_train_safe_mining_request_source:{','.join(sorted(set(errors)))}")


def requirement_row(group_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = group_rows[0]
    source_slice_label = str(first.get("slice_label") or "")
    target_mining_label, target_training_label = SLICE_LABEL_TO_TARGET[source_slice_label]
    stem_type = str(first.get("stem_type") or "unknown")
    top_error_channel = first_nonempty(first, ("top_error_channel", "top_error_channel_or_unknown"))
    candidate_family = first_nonempty(first, ("candidate_family",))
    source_model_family = first_nonempty(first, ("source_model_family",))
    route = route_family(str(first.get("route_id") or ""))
    split_counts = dict(sorted(Counter(str(row.get("split") or "unknown") for row in group_rows).items()))
    payload = {
        "source_slice_label": source_slice_label,
        "target_mining_label": target_mining_label,
        "target_training_label": target_training_label,
        "stem_type": stem_type,
        "top_error_channel_or_unknown": top_error_channel,
        "candidate_family_or_unknown": candidate_family,
        "source_model_family_or_unknown": source_model_family,
        "route_family_or_unknown": route,
    }
    return {
        "requirement_schema": "tensor_audio_error_train_safe_mining_requirement.v1",
        "requirement_id": stable_id(payload),
        **payload,
        "locked_eval_evidence_count": len(group_rows),
        "locked_eval_split_counts": split_counts,
        "recommended_train_query": {
            "split": "train",
            "stem_type": stem_type,
            "target_training_label": target_training_label,
            "target_mining_label": target_mining_label,
            "top_error_channel_or_unknown": top_error_channel,
            "candidate_family_or_unknown": candidate_family,
            "source_model_family_or_unknown": source_model_family,
            "route_family_or_unknown": route,
        },
        "must_use_split": "train",
        "forbidden_splits": ["dev", "test", "hardcase", "holdout"],
        **SAFETY_FLAGS,
    }


def build_requirements(slice_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in slice_rows:
        slice_label = str(row.get("slice_label") or "")
        if slice_label not in SLICE_LABEL_TO_TARGET:
            continue
        key = (
            str(row.get("stem_type") or "unknown"),
            slice_label,
            first_nonempty(row, ("top_error_channel", "top_error_channel_or_unknown")),
            first_nonempty(row, ("candidate_family",)),
            first_nonempty(row, ("source_model_family",)),
            route_family(str(row.get("route_id") or "")),
        )
        groups.setdefault(key, []).append(row)
    return [requirement_row(groups[key]) for key in sorted(groups)]


def write_summary(path: Path, manifest: dict[str, Any], requirements: Sequence[dict[str, Any]]) -> None:
    target_counts = Counter(str(row.get("target_mining_label") or "") for row in requirements)
    lines = [
        "# Tensor Audio Error Train-Safe Mining Request",
        "",
        f"Requirements total: {manifest['requirements_total']}",
        f"Hard-negative requirements: {target_counts.get('hard_negative_false_accept', 0)}",
        f"False-reject requirements: {target_counts.get('false_reject_positive', 0)}",
        f"Required train rows minimum: {manifest['required_train_rows_min']}",
        "",
        "## Safety",
        "",
        "- locked_eval_only_source=true",
        "- review_only_source=true",
        "- training_rows_created=false",
        "- training_allowed=false",
        "- runtime_gates_changed=false",
        "- release_ready=false",
        "- separator_training_started=false",
        "- repair_adapter_training_started=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_tensor_audio_error_train_safe_mining_request(
    *,
    locked_slice_dir: Path,
    split_dir: Path,
    tensor_dataset_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_manifest = read_json(locked_slice_dir / SLICE_MANIFEST_FILE)
    slice_validation = read_json(locked_slice_dir / SLICE_VALIDATION_FILE)
    slice_rows = list(iter_jsonl(locked_slice_dir / SLICE_ROWS_FILE))
    validate_inputs(
        locked_slice_dir=locked_slice_dir,
        split_dir=split_dir,
        tensor_dataset_dir=tensor_dataset_dir,
        slice_manifest=slice_manifest,
        slice_validation=slice_validation,
        slice_rows=slice_rows,
    )
    requirements = build_requirements(slice_rows)
    write_jsonl(output_dir / REQUEST_ROWS_FILE, requirements)
    manifest = {
        "schema": "tensor_audio_error_train_safe_mining_request.v1",
        "source_locked_slice_dir": str(locked_slice_dir),
        "source_locked_slice_manifest_path": str(locked_slice_dir / SLICE_MANIFEST_FILE),
        "source_split_dir": str(split_dir),
        "source_tensor_dataset_dir": str(tensor_dataset_dir),
        "locked_eval_only_source": True,
        "review_only_source": True,
        "requirements_total": len(requirements),
        "source_locked_slice_rows": len(slice_rows),
        **DEFAULT_THRESHOLDS,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / REQUEST_MANIFEST_FILE, manifest)
    write_summary(output_dir / REQUEST_SUMMARY_FILE, manifest, requirements)
    validation = validate_tensor_audio_error_train_safe_mining_request(output_dir)
    if validation.get("ok") is not True:
        raise ValueError(
            "train_safe_mining_request_validation_failed:"
            + ",".join(str(error) for error in validation.get("errors", []) or [])
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-safe mining request from locked tensor critic evidence.")
    parser.add_argument("--locked-slice-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--tensor-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_tensor_audio_error_train_safe_mining_request(
        locked_slice_dir=args.locked_slice_dir,
        split_dir=args.split_dir,
        tensor_dataset_dir=args.tensor_dataset_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
