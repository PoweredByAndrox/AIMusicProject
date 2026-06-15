#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.validate_tensor_audio_error_train_hard_negative_harvest import (
    HARVEST_MANIFEST_FILE,
    HARVEST_ROWS_FILE,
    HARVEST_ROW_SCHEMA,
    validate_tensor_audio_error_train_hard_negative_harvest,
)


DATASET_ROWS_FILE = "tensor_audio_error_dataset.jsonl"
DATASET_MANIFEST_FILE = "locked_tensor_audio_error_dataset_manifest.json"
AUDIT_MANIFEST_FILE = "tensor_audio_error_train_negative_gap_audit.json"
SPLIT_ROWS_FILE = "leak_safe_split_rows.jsonl"
HARDCASE_MANIFEST_FILE = "hardcase_holdout_manifest.json"
HARVEST_SKIPPED_FILE = "tensor_audio_error_train_hard_negative_skipped.jsonl"
HARVEST_SUMMARY_FILE = "tensor_audio_error_train_hard_negative_harvest_summary.md"
ROOT_BLOCKED_SUMMARY_FILE = "blocked_summary.md"

ERROR_MAP_NAMES = (
    "stft_residual_magnitude",
    "phase_difference_map",
    "stereo_width_map",
    "vocal_bleed_proxy_map",
    "instrumental_bleed_proxy_map",
    "missing_detail_proxy_map",
    "transient_loss_proxy_map",
    "artifact_noise_proxy_map",
    "reconstruction_error_map",
)
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


def stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


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


def blocked_split_keys(split_dir: Path) -> set[str]:
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


def top_error_channel(tensor_map_path: Path, tensor_map_names: Sequence[str]) -> tuple[str, dict[str, float], bool]:
    if not tensor_map_path.exists():
        return "", {}, False
    scores: dict[str, float] = {}
    with np.load(tensor_map_path) as payload:
        names = [name for name in tensor_map_names if name in payload.files] or [
            name for name in ERROR_MAP_NAMES if name in payload.files
        ]
        for name in names:
            arr = np.asarray(payload[name], dtype=np.float32)
            scores[name] = float(np.mean(np.abs(np.nan_to_num(arr)))) if arr.size else 0.0
    return (max(scores, key=scores.get) if scores else "", scores, bool(scores))


def safe_existing_train_negative(row: dict[str, Any], blocked_keys: set[str]) -> tuple[bool, str]:
    if str(row.get("split") or "") != "train":
        return False, "split_not_train"
    if row.get("train_allowed") is not True:
        return False, "train_allowed_not_true"
    if int(row.get("label") or 0) != 0:
        return False, "label_not_zero"
    if row.get("hardcase_holdout") is True:
        return False, "hardcase_holdout_true"
    if row.get("locked_eval_row") is True:
        return False, "locked_eval_row_true"
    if row_key_set(row) & blocked_keys:
        return False, "locked_eval_or_hardcase_key_overlap"
    if row.get("scalar_features_used") is True:
        return False, "scalar_features_used_true"
    if row.get("release_ready") is True:
        return False, "release_ready_true"
    if not Path(str(row.get("tensor_map_path") or "")).exists():
        return False, "missing_tensor_map"
    if not row.get("tensor_map_names"):
        return False, "missing_tensor_map_names"
    for flag in sorted(FORBIDDEN_TRUE_FLAGS):
        if has_true_flag(row, flag):
            return False, f"forbidden_true_flag:{flag}"
    return True, ""


def candidate_row(source_row: dict[str, Any]) -> dict[str, Any]:
    tensor_map_path = Path(str(source_row.get("tensor_map_path") or ""))
    tensor_map_names = [str(item) for item in source_row.get("tensor_map_names", []) or []]
    top_channel, channel_scores, _ = top_error_channel(tensor_map_path, tensor_map_names)
    if not top_channel:
        blockers = [str(item) for item in source_row.get("blockers", []) or []]
        top_channel = blockers[0] if blockers else "hard_negative_false_accept"
    payload = {
        "row_uid": str(source_row.get("row_uid") or ""),
        "canonical_row_key": str(source_row.get("canonical_row_key") or ""),
        "candidate_id": str(source_row.get("candidate_id") or ""),
        "stem_type": str(source_row.get("stem_type") or "unknown"),
        "window_start_sec": float(source_row.get("window_start_sec") or 0.0),
        "window_end_sec": float(source_row.get("window_end_sec") or 0.0),
    }
    blockers = [str(item) for item in source_row.get("blockers", []) or []]
    return {
        "row_schema": HARVEST_ROW_SCHEMA,
        "harvest_candidate_id": stable_id(payload),
        "row_uid": payload["row_uid"],
        "canonical_row_key": payload["canonical_row_key"],
        "track_id": str(source_row.get("track_id") or ""),
        "run_id": str(source_row.get("run_id") or ""),
        "candidate_id": payload["candidate_id"],
        "route_id": str(source_row.get("route_id") or ""),
        "candidate_family": str(source_row.get("candidate_family") or ""),
        "source_model_family": str(source_row.get("source_model_family") or ""),
        "stem_type": payload["stem_type"],
        "split": "train",
        "train_allowed": True,
        "hardcase": source_row.get("hardcase") is True,
        "hardcase_holdout": False,
        "locked_eval_row": False,
        "split_overlap_checked": True,
        "blocked_overlap_keys": [],
        "tensor_map_path": str(tensor_map_path),
        "tensor_map_names": tensor_map_names,
        "true_label": 0,
        "source_mining_label": "hard_negative_false_accept",
        "label_source": "train_split_existing_judge_or_train_only_generation",
        "error_reason": top_channel,
        "blockers": blockers,
        "top_error_channel": top_channel,
        "channel_scores": channel_scores,
        "window_start_sec": payload["window_start_sec"],
        "window_end_sec": payload["window_end_sec"],
        "runtime_gate_decision_applied": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "release_ready": False,
        "shadow_only": True,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Train Hard-Negative Harvest",
        "",
        f"Harvest status: {manifest['harvest_status']}",
        f"Source mode: {manifest['source_mode']}",
        f"Candidate rows: {manifest['candidate_rows_total']}",
        f"Negative rows: {manifest['negative_rows']}",
        f"Hard-negative rows: {manifest['hard_negative_rows']}",
        "",
        "## Safety",
        "",
        "- split=train only",
        "- dev_test_rows_used=false",
        "- locked_eval_rows_used=false",
        f"- hardcase_train_rows_used={str(manifest.get('hardcase_train_rows_used') is True).lower()}",
        f"- hardcase_holdout_rows_used={str(manifest.get('hardcase_holdout_rows_used') is True).lower()}",
        "- training_started=false",
        "- runtime_gates_changed=false",
        "- release_ready=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def harvest_tensor_audio_error_train_hard_negatives(
    *,
    negative_gap_audit_dir: Path,
    tensor_dataset_dir: Path,
    split_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_manifest = read_json(negative_gap_audit_dir / AUDIT_MANIFEST_FILE)
    dataset_manifest = read_json(tensor_dataset_dir / DATASET_MANIFEST_FILE)
    dataset_rows = list(iter_jsonl(tensor_dataset_dir / DATASET_ROWS_FILE))
    blocked_keys = blocked_split_keys(split_dir)
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    for row in dataset_rows:
        ok, reason = safe_existing_train_negative(row, blocked_keys)
        if ok:
            candidates.append(candidate_row(row))
        elif int(row.get("label") or 0) == 0:
            skip_counts[reason] += 1
            skipped.append(
                {
                    "row_uid": str(row.get("row_uid") or ""),
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "split": str(row.get("split") or ""),
                    "label": int(row.get("label") or 0),
                    "skip_reason": reason,
                    "release_ready": False,
                }
            )

    status = "harvest_ready" if candidates else "blocked_no_train_negative_source"
    if candidates and dataset_manifest.get("tensor_maps_required") is not True:
        status = "blocked_missing_tensor_maps"
    manifest = {
        "schema": "tensor_audio_error_train_hard_negative_harvest.v1",
        "source_negative_gap_audit_dir": str(negative_gap_audit_dir),
        "source_tensor_dataset_dir": str(tensor_dataset_dir),
        "source_diagnostic_decision": str(audit_manifest.get("diagnostic_decision") or ""),
        "harvest_status": status,
        "source_mode": "existing_train_rows" if candidates else "none",
        "source_a_existing_train_rows_used": bool(candidates),
        "source_b_candidate_generation_used": False,
        "candidate_rows_total": len(candidates),
        "negative_rows": len(candidates),
        "hard_negative_rows": len(candidates),
        "positive_rows": 0,
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "split": "train",
        "train_only": True,
        "dev_test_rows_used": False,
        "locked_eval_rows_used": False,
        "hardcase_rows_used": False,
        "hardcase_train_rows_used": any(row.get("hardcase") is True for row in candidates),
        "hardcase_holdout_rows_used": False,
        "release_ready": False,
        "runtime_gates_changed": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "shadow_only": True,
    }
    write_jsonl(output_dir / HARVEST_ROWS_FILE, candidates)
    write_jsonl(output_dir / HARVEST_SKIPPED_FILE, skipped)
    write_json(output_dir / HARVEST_MANIFEST_FILE, manifest)
    write_summary(output_dir / HARVEST_SUMMARY_FILE, manifest)
    if status != "harvest_ready":
        write_summary(output_dir.parent / ROOT_BLOCKED_SUMMARY_FILE, manifest)
    validation = validate_tensor_audio_error_train_hard_negative_harvest(output_dir)
    if validation.get("ok") is not True:
        raise ValueError("hard_negative_harvest_validation_failed:" + ",".join(validation.get("errors") or []))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest train-only hard-negative tensor critic rows.")
    parser.add_argument("--negative-gap-audit-dir", type=Path, required=True)
    parser.add_argument("--tensor-dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=args.negative_gap_audit_dir,
        tensor_dataset_dir=args.tensor_dataset_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
