#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DECISION_MANIFEST = "tensor_audio_error_critic_decision_manifest.json"
COMPARISON_JSON = "tensor_audio_error_critic_variant_comparison.json"
SCORES_FILE = "tensor_audio_error_shadow_scores.jsonl"
SLICE_FILE = "tensor_audio_error_locked_mining_slice.jsonl"
SLICE_MANIFEST = "tensor_audio_error_locked_mining_slice_manifest.json"
SLICE_SUMMARY = "tensor_audio_error_locked_mining_slice_summary.md"

ALLOWED_DECISIONS = {"mine_next_dataset_slice", "keep_as_specialist_shadow_model"}
LOCKED_EVAL_SPLITS = {"dev", "test"}

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

FORBIDDEN_TRUE_FLAGS = {
    "release_ready",
    "can_publish_as_result",
    "promotion_allowed",
    "runtime_gate_update_allowed",
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def has_forbidden_true(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(payload.get(key) is True for key in FORBIDDEN_TRUE_FLAGS) or any(
            has_forbidden_true(value) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(has_forbidden_true(item) for item in payload)
    return False


def non_empty_list(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    return isinstance(value, list) and len(value) > 0


def stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def label_for(row: dict[str, Any]) -> int:
    return int(row.get("true_label", row.get("label", 0)) or 0)


def accepts(row: dict[str, Any]) -> bool:
    if "would_accept_at_calibrated_threshold" in row:
        return row.get("would_accept_at_calibrated_threshold") is True
    return str(row.get("shadow_decision") or "") == "shadow_accept_candidate"


def validate_decision(decision: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    decision_value = str(decision.get("decision") or "")
    if decision.get("schema") != "tensor_audio_error_critic_decision.v1":
        errors.append("invalid_decision_schema")
    if comparison.get("schema") != "tensor_audio_error_critic_variant_comparison.v1":
        errors.append("invalid_comparison_schema")
    if decision_value == "candidate_rejected_runtime":
        errors.append("source_decision_rejected_runtime_not_sliceable")
    elif decision_value not in ALLOWED_DECISIONS or "promote" in decision_value:
        errors.append("promotion_decision_not_allowed")
    if decision.get("promotion_allowed") is True:
        errors.append("promotion_allowed_true")
    locked_eval = comparison.get("locked_eval") if isinstance(comparison.get("locked_eval"), dict) else {}
    if non_empty_list(locked_eval, "errors"):
        errors.append("locked_eval_errors_present")
    for variant_name in ("baseline", "candidate"):
        variant = comparison.get(variant_name) if isinstance(comparison.get(variant_name), dict) else {}
        if non_empty_list(variant, "safety_errors"):
            errors.append(f"{variant_name}_safety_errors_present")
    if int(locked_eval.get("critical_leakage_count") or 0) != 0:
        errors.append("locked_eval_critical_leakage_present")
    if locked_eval and locked_eval.get("future_r1_r2_r3_r4_must_use_this_split") is not True:
        errors.append("locked_eval_future_split_contract_missing")
    contamination = comparison.get("contamination_status") if isinstance(comparison.get("contamination_status"), dict) else {}
    if decision.get("non_locked_split_used") is True or contamination.get("non_locked_split_used") is True:
        errors.append("non_locked_split_used")
    if (
        decision.get("dev_test_mining_rows_used_for_training") is True
        or contamination.get("dev_test_mining_rows_used_for_training") is True
    ):
        errors.append("dev_test_mining_rows_used_for_training")
    if has_forbidden_true({"decision": decision, "comparison": comparison}):
        errors.append("forbidden_true_flag")
    return errors


def row_match_key(row: dict[str, Any]) -> str:
    return str(row.get("row_uid") or row.get("canonical_row_key") or "")


def slice_label(*, label: int, baseline_accept: bool, candidate_accept: bool) -> str:
    if label == 1 and baseline_accept and not candidate_accept:
        return "candidate_regressed_false_reject"
    if label == 1 and not baseline_accept and not candidate_accept:
        return "persistent_false_reject"
    if label == 0 and not baseline_accept and candidate_accept:
        return "candidate_regressed_hard_negative"
    if label == 0 and baseline_accept and candidate_accept:
        return "persistent_hard_negative"
    return ""


def slice_row(*, baseline_row: dict[str, Any], candidate_row: dict[str, Any], source_decision_dir: Path) -> dict[str, Any] | None:
    split = str(candidate_row.get("split") or baseline_row.get("split") or "")
    if split not in LOCKED_EVAL_SPLITS:
        return None
    label = label_for(candidate_row)
    baseline_accept = accepts(baseline_row)
    candidate_accept = accepts(candidate_row)
    label_name = slice_label(label=label, baseline_accept=baseline_accept, candidate_accept=candidate_accept)
    if not label_name:
        return None
    key_payload = {
        "slice_label": label_name,
        "row_uid": row_match_key(candidate_row),
        "candidate_id": str(candidate_row.get("candidate_id") or ""),
        "stem_type": str(candidate_row.get("stem_type") or "unknown"),
        "split": split,
        "window_start_sec": float(candidate_row.get("window_start_sec") or 0.0),
        "window_end_sec": float(candidate_row.get("window_end_sec") or 0.0),
    }
    return {
        "slice_row_id": stable_id(key_payload),
        "slice_row_schema": "tensor_audio_error_locked_mining_slice_row.v1",
        "source_decision_dir": str(source_decision_dir),
        "slice_label": label_name,
        "recommended_training_label": label,
        "training_allowed": False,
        "training_candidate_status": "locked_eval_review_only",
        "row_uid": row_match_key(candidate_row),
        "canonical_row_key": str(candidate_row.get("canonical_row_key") or baseline_row.get("canonical_row_key") or ""),
        "track_id": str(candidate_row.get("track_id") or baseline_row.get("track_id") or ""),
        "run_id": str(candidate_row.get("run_id") or baseline_row.get("run_id") or ""),
        "candidate_id": str(candidate_row.get("candidate_id") or baseline_row.get("candidate_id") or ""),
        "route_id": str(candidate_row.get("route_id") or baseline_row.get("route_id") or ""),
        "candidate_family": str(candidate_row.get("candidate_family") or baseline_row.get("candidate_family") or ""),
        "source_model_family": str(candidate_row.get("source_model_family") or baseline_row.get("source_model_family") or ""),
        "stem_type": str(candidate_row.get("stem_type") or baseline_row.get("stem_type") or "unknown"),
        "split": split,
        "true_label": label,
        "window_start_sec": float(candidate_row.get("window_start_sec") or baseline_row.get("window_start_sec") or 0.0),
        "window_end_sec": float(candidate_row.get("window_end_sec") or baseline_row.get("window_end_sec") or 0.0),
        "baseline_predicted_accept_probability": float(baseline_row.get("predicted_accept_probability") or 0.0),
        "candidate_predicted_accept_probability": float(candidate_row.get("predicted_accept_probability") or 0.0),
        "baseline_shadow_decision": str(baseline_row.get("shadow_decision") or ""),
        "candidate_shadow_decision": str(candidate_row.get("shadow_decision") or ""),
        "baseline_would_accept": baseline_accept,
        "candidate_would_accept": candidate_accept,
        **SAFETY_FLAGS,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Tensor Audio Error Locked Mining Slice",
        "",
        f"Candidate rows: {manifest['candidate_rows_total']}",
        f"Candidate regressed false rejects: {manifest['candidate_regressed_false_reject_rows']}",
        f"Persistent false rejects: {manifest['persistent_false_reject_rows']}",
        f"Candidate regressed hard negatives: {manifest['candidate_regressed_hard_negative_rows']}",
        f"Persistent hard negatives: {manifest['persistent_hard_negative_rows']}",
        f"Train rows ignored: {manifest['train_rows_ignored']}",
        "",
        "## Safety",
        "",
        "- locked_eval_only=true",
        "- training_started=false",
        "- runtime_gates_changed=false",
        "- release_ready=false",
        "- separator_training_started=false",
        "- repair_adapter_training_started=false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_tensor_audio_error_locked_mining_slice(*, decision_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = read_json(decision_dir / DECISION_MANIFEST)
    comparison = read_json(decision_dir / COMPARISON_JSON)
    errors = validate_decision(decision, comparison)
    if errors:
        raise ValueError(f"unsafe_tensor_critic_decision:{','.join(errors)}")
    baseline_scores_dir = Path(str((comparison.get("baseline") or {}).get("scores_dir") or ""))
    candidate_scores_dir = Path(str((comparison.get("candidate") or {}).get("scores_dir") or ""))
    baseline_rows = {row_match_key(row): row for row in iter_jsonl(baseline_scores_dir / SCORES_FILE) if row_match_key(row)}
    candidate_rows = {row_match_key(row): row for row in iter_jsonl(candidate_scores_dir / SCORES_FILE) if row_match_key(row)}
    rows_by_id: dict[str, dict[str, Any]] = {}
    train_rows_ignored = 0
    missing_baseline_rows = 0
    for key, candidate_row in candidate_rows.items():
        split = str(candidate_row.get("split") or "")
        if split == "train":
            train_rows_ignored += 1
            continue
        baseline_row = baseline_rows.get(key)
        if baseline_row is None:
            missing_baseline_rows += 1
            continue
        row = slice_row(baseline_row=baseline_row, candidate_row=candidate_row, source_decision_dir=decision_dir)
        if row is not None:
            rows_by_id[row["slice_row_id"]] = row
    slice_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    label_counts = dict(sorted(Counter(row["slice_label"] for row in slice_rows).items()))
    split_counts = dict(sorted(Counter(row["split"] for row in slice_rows).items()))
    write_jsonl(output_dir / SLICE_FILE, slice_rows)
    manifest = {
        "schema": "tensor_audio_error_locked_mining_slice.v1",
        "decision_dir": str(decision_dir),
        "output_dir": str(output_dir),
        "source_decision": str(decision.get("decision") or ""),
        "source_decision_reasons": [str(item) for item in decision.get("decision_reasons", []) or []],
        "baseline_scores_dir": str(baseline_scores_dir),
        "candidate_scores_dir": str(candidate_scores_dir),
        "slice_status": "candidate_only_locked_eval_review",
        "locked_eval_only": True,
        "candidate_rows_total": len(slice_rows),
        "candidate_regressed_false_reject_rows": int(label_counts.get("candidate_regressed_false_reject") or 0),
        "persistent_false_reject_rows": int(label_counts.get("persistent_false_reject") or 0),
        "candidate_regressed_hard_negative_rows": int(label_counts.get("candidate_regressed_hard_negative") or 0),
        "persistent_hard_negative_rows": int(label_counts.get("persistent_hard_negative") or 0),
        "slice_label_counts": label_counts,
        "split_counts": split_counts,
        "train_rows_ignored": train_rows_ignored,
        "missing_baseline_rows": missing_baseline_rows,
        "slice_rows_path": str(output_dir / SLICE_FILE),
        "summary_path": str(output_dir / SLICE_SUMMARY),
        "eligible_for_future_training_review": True,
        "training_allowed": False,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / SLICE_MANIFEST, manifest)
    write_summary(output_dir / SLICE_SUMMARY, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a locked-eval-only tensor critic mining slice from a variant decision.")
    parser.add_argument("--decision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_tensor_audio_error_locked_mining_slice(decision_dir=args.decision_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
