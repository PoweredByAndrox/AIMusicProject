from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    BLOCKED_FILE,
    DATASET_FILE,
    MANIFEST_FILE,
    PROGRESS_FILE,
    PROMOTION_LABEL_TYPES,
    SAFETY_FLAGS,
    append_progress,
    as_list,
    iter_jsonl,
    read_json,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime


LOCKED_HOLDOUT_SCHEMA = "audio_quality_locked_holdout_pack.v1"
HOLDOUT_SCORES_FILE = "holdout_scores.jsonl"


def _identity_values(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("track_id", "candidate_id", "row_id"):
        value = str(row.get(key) or "").strip()
        if value:
            values.add(f"{key}:{value}")
    for ref in as_list(row.get("source_manifest_refs")):
        value = str(ref or "").strip()
        if value:
            values.add(f"source_manifest_ref:{value}")
    return values


def _dataset_jsonl_from_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> Path:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    dataset_jsonl = str(outputs.get("dataset_jsonl") or manifest.get("dataset_jsonl") or "").strip()
    return Path(dataset_jsonl) if dataset_jsonl else dataset_dir / DATASET_FILE


def _blocked_row(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "row_id": str(row.get("row_id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "label_type": str(row.get("label_type") or ""),
        "split": str(row.get("split") or ""),
        "blocked_reasons": sorted(set(reasons)),
        "delivery_allowed": False,
        "release_ready": False,
    }


def _score_rows_by_id(scores_jsonl: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(scores_jsonl):
        row_id = str(row.get("row_id") or "").strip()
        if row_id:
            rows[row_id] = row
    return rows


def _candidate_reasons(
    row: dict[str, Any],
    *,
    score_row: dict[str, Any] | None,
    source_lock_pack_json: str,
    training_identities: set[str],
) -> list[str]:
    reasons: list[str] = []
    if row.get("safe_for_training") is not True:
        reasons.append("unsafe_dataset_row")
    if row.get("delivery_allowed") is True:
        reasons.append("row_delivery_allowed_true")
    if row.get("release_ready") is True:
        reasons.append("row_release_ready_true")
    if row.get("model_registry_edited") is True:
        reasons.append("row_model_registry_edited_true")
    if row.get("runtime_gates_changed") is True:
        reasons.append("row_runtime_gates_changed_true")
    if row.get("thresholds_relaxed") is True:
        reasons.append("row_thresholds_relaxed_true")
    if row.get("pseudo_teacher_targets_are_ground_truth") is True:
        reasons.append("pseudo_teacher_claims_ground_truth")
    if row.get("generated_stem_claims_ground_truth") is True:
        reasons.append("generated_stem_claims_ground_truth")
    if row.get("label_type") not in PROMOTION_LABEL_TYPES:
        reasons.append("non_promotion_label_type")
    if str(row.get("split") or "").strip() != "locked_holdout":
        reasons.append("not_locked_holdout_split")
    if _identity_values(row) & training_identities:
        reasons.append("split_leakage_detected")
    if not as_list(row.get("source_manifest_refs")):
        reasons.append("missing_source_manifest_refs")
    if not as_list(row.get("rag_refs")):
        reasons.append("missing_rag_refs")
    if source_lock_pack_json and str(row.get("source_lock_pack_json") or "") != source_lock_pack_json:
        reasons.append("source_lock_mismatch")
    if score_row is None:
        reasons.append("missing_score_row")
    else:
        if score_row.get("delivery_allowed") is True:
            reasons.append("score_delivery_allowed_true")
        if score_row.get("release_ready") is True:
            reasons.append("score_release_ready_true")
        if "critic_accepts" not in score_row and "accept_probability" not in score_row:
            reasons.append("missing_critic_prediction")
        if source_lock_pack_json and str(score_row.get("source_lock_pack_json") or "") != source_lock_pack_json:
            reasons.append("score_source_lock_mismatch")
    return reasons


def _holdout_row(row: dict[str, Any], score_row: dict[str, Any]) -> dict[str, Any]:
    merged = {**row, **score_row}
    merged["label_type"] = str(row.get("label_type") or "")
    merged["label"] = int(row.get("label") or 0)
    merged["split"] = "locked_holdout"
    merged["source_manifest_refs"] = [str(item) for item in as_list(row.get("source_manifest_refs"))]
    merged["rag_refs"] = as_list(row.get("rag_refs"))
    merged["issue_tags"] = [str(item) for item in as_list(row.get("issue_tags"))]
    merged["blocker_reasons"] = [str(item) for item in as_list(row.get("blocker_reasons"))]
    merged["delivery_allowed"] = False
    merged["release_ready"] = False
    return merged


def build_audio_quality_locked_holdout_pack(
    *,
    dataset_dir: Path,
    score_manifest_json: Path,
    output_dir: Path,
    min_holdout_rows: int = 2,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        dataset_dir=str(dataset_dir),
        score_manifest_json=str(score_manifest_json),
    )

    dataset_manifest = read_json(dataset_dir / MANIFEST_FILE)
    score_manifest = read_json(score_manifest_json)
    source_lock_pack_json = str(dataset_manifest.get("source_lock_pack_json") or "").strip()
    manifest_reasons: list[str] = []
    if not source_lock_pack_json:
        manifest_reasons.append("source_lock_pack_json_missing")
    if dataset_manifest.get("delivery_allowed") is True:
        manifest_reasons.append("dataset_delivery_allowed_true")
    if dataset_manifest.get("release_ready") is True:
        manifest_reasons.append("dataset_release_ready_true")
    if dataset_manifest.get("model_registry_edited") is True:
        manifest_reasons.append("dataset_model_registry_edited_true")
    if dataset_manifest.get("runtime_gates_changed") is True:
        manifest_reasons.append("dataset_runtime_gates_changed_true")
    if dataset_manifest.get("thresholds_relaxed") is True:
        manifest_reasons.append("dataset_thresholds_relaxed_true")
    if score_manifest.get("delivery_allowed") is True:
        manifest_reasons.append("score_delivery_allowed_true")
    if score_manifest.get("release_ready") is True:
        manifest_reasons.append("score_release_ready_true")
    if score_manifest.get("model_registry_edited") is True:
        manifest_reasons.append("score_model_registry_edited_true")
    if score_manifest.get("runtime_gates_changed") is True:
        manifest_reasons.append("score_runtime_gates_changed_true")
    if score_manifest.get("thresholds_relaxed") is True:
        manifest_reasons.append("score_thresholds_relaxed_true")
    score_source_lock = str(score_manifest.get("source_lock_pack_json") or "").strip()
    if not score_source_lock:
        manifest_reasons.append("score_manifest_source_lock_missing")
    elif source_lock_pack_json and score_source_lock != source_lock_pack_json:
        manifest_reasons.append("score_manifest_source_lock_mismatch")
    dataset_jsonl = _dataset_jsonl_from_manifest(dataset_dir, dataset_manifest)
    scores_jsonl = Path(str(score_manifest.get("scores_jsonl") or ""))
    rows = list(iter_jsonl(dataset_jsonl))
    scores_by_id = _score_rows_by_id(scores_jsonl)
    training_identities: set[str] = set()
    for row in rows:
        if str(row.get("split") or "").strip() == "train" and row.get("safe_for_training") is True:
            training_identities.update(_identity_values(row))

    holdout_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("split") or "").strip() != "locked_holdout":
            continue
        row_id = str(row.get("row_id") or "").strip()
        score_row = scores_by_id.get(row_id)
        reasons = [
            *manifest_reasons,
            *_candidate_reasons(
                row,
                score_row=score_row,
                source_lock_pack_json=source_lock_pack_json,
                training_identities=training_identities,
            ),
        ]
        if reasons:
            blocked_rows.append(_blocked_row(row, reasons))
            continue
        holdout_rows.append(_holdout_row(row, score_row or {}))

    positive_rows = sum(1 for row in holdout_rows if int(row.get("label") or 0) > 0)
    negative_rows = len(holdout_rows) - positive_rows
    hard_case_rows = sum(1 for row in holdout_rows if int(row.get("label") or 0) <= 0 and (as_list(row.get("blocker_reasons")) or as_list(row.get("issue_tags"))))
    errors: list[str] = list(manifest_reasons)
    if len(holdout_rows) < int(min_holdout_rows):
        errors.append("locked_holdout_rows_below_minimum")
    if positive_rows <= 0:
        errors.append("positive_holdout_missing")
    if negative_rows <= 0:
        errors.append("negative_holdout_missing")
    if hard_case_rows <= 0:
        errors.append("hard_case_holdout_missing")
    blocked_reason_set = {reason for row in blocked_rows for reason in row["blocked_reasons"]}
    if "split_leakage_detected" in blocked_reason_set:
        errors.append("split_leakage_detected")

    write_jsonl(output_dir / HOLDOUT_SCORES_FILE, holdout_rows)
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)
    manifest = {
        "schema": LOCKED_HOLDOUT_SCHEMA,
        "manifest_type": "audio_quality_locked_holdout_pack",
        "dataset_manifest_json": str(dataset_dir / MANIFEST_FILE),
        "dataset_jsonl": str(dataset_jsonl),
        "score_manifest_json": str(score_manifest_json),
        "scores_jsonl": str(scores_jsonl),
        "source_lock_pack_json": source_lock_pack_json,
        "row_count": len(holdout_rows),
        "blocked_row_count": len(blocked_rows),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "hard_case_holdout_count": hard_case_rows,
        "min_holdout_rows": int(min_holdout_rows),
        "errors": sorted(set(errors)),
        "promotion_evidence_ready": not errors,
        "outputs": {
            "holdout_scores_jsonl": str(output_dir / HOLDOUT_SCORES_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(output_dir / PROGRESS_FILE),
        },
        "presentation_status": "audio_quality_locked_holdout_pack_not_for_listen",
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        row_count=len(holdout_rows),
        blocked_row_count=len(blocked_rows),
        promotion_evidence_ready=manifest["promotion_evidence_ready"],
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build locked holdout score evidence for audio quality critic promotion.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--score-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-holdout-rows", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_locked_holdout_pack.py",
        entrypoint_path="/app/tools/build_audio_quality_locked_holdout_pack.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_locked_holdout_pack(
        dataset_dir=args.dataset_dir,
        score_manifest_json=args.score_manifest_json,
        output_dir=args.output_dir,
        min_holdout_rows=int(args.min_holdout_rows),
    )
    print(json.dumps({"manifest_json": str(args.output_dir / MANIFEST_FILE), "promotion_evidence_ready": manifest["promotion_evidence_ready"]}, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["promotion_evidence_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
