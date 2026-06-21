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
    MANIFEST_FILE,
    PROGRESS_FILE,
    SAFETY_FLAGS,
    append_progress,
    as_float,
    as_list,
    iter_jsonl,
    read_json,
    stable_hash,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime


MANIFEST_SCHEMA = "audio_quality_review_decision_ingest_manifest.v1"
REVIEW_QUEUE_SCHEMA = "audio_quality_review_queue_manifest.v1"
REVIEW_ROW_SCHEMA = "audio_quality_review_queue_row.v1"
DECISION_SCHEMA = "audio_quality_review_decision.v1"
OUTCOME_FILE = "audio_quality_learning_outcome.jsonl"
ALLOWED_DECISIONS = {"accept_quality", "reject_quality", "needs_more_evidence", "exclude"}
VERIFIED_OUTCOME_TYPES = {"human_verified", "reference_verified"}
BLOCKED_TRUE_FLAGS = (
    "delivery_allowed",
    "release_ready",
    "can_publish_as_result",
    "thresholds_relaxed",
    "model_registry_edited",
    "runtime_gates_changed",
)


def _unique(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _manifest_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != REVIEW_QUEUE_SCHEMA:
        errors.append("review_queue_manifest_schema_unexpected")
    for key in BLOCKED_TRUE_FLAGS:
        if manifest.get(key) is True:
            errors.append(f"review_queue_manifest_{key}_true")
    if not str(manifest.get("source_lock_pack_json") or "").strip():
        errors.append("review_queue_manifest_source_lock_missing")
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    review_jsonl = Path(str(outputs.get("review_jsonl") or ""))
    if not str(review_jsonl) or not review_jsonl.exists():
        errors.append("review_jsonl_missing")
    return sorted(set(errors))


def _review_jsonl_from_manifest(manifest: dict[str, Any]) -> Path:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    return Path(str(outputs.get("review_jsonl") or ""))


def _load_decisions(decisions_jsonl: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if decisions_jsonl is None or not str(decisions_jsonl):
        return {}, []
    decisions: dict[str, dict[str, Any]] = {}
    blocked: list[dict[str, Any]] = []
    for row in iter_jsonl(decisions_jsonl):
        queue_id = str(row.get("queue_id") or "").strip()
        reasons: list[str] = []
        if row.get("schema") != DECISION_SCHEMA:
            reasons.append("decision_schema_unexpected")
        decision = str(row.get("decision") or "").strip()
        if decision not in ALLOWED_DECISIONS:
            reasons.append("invalid_decision")
        if not queue_id:
            reasons.append("missing_queue_id")
        if queue_id in decisions:
            reasons.append("duplicate_queue_id")
        if reasons:
            blocked.append(_blocked_row(row, reasons, source="decision"))
            continue
        decisions[queue_id] = row
    return decisions, blocked


def _decision_reasons(decision: dict[str, Any] | None) -> list[str]:
    if decision is None:
        return ["missing_review_decision"]
    decision_value = str(decision.get("decision") or "").strip()
    if decision_value in {"needs_more_evidence", "exclude"}:
        return [decision_value]
    outcome_type = str(decision.get("outcome_type") or "").strip()
    reasons: list[str] = []
    if outcome_type not in VERIFIED_OUTCOME_TYPES:
        reasons.append("invalid_or_missing_outcome_type")
    if outcome_type == "human_verified" and not str(decision.get("reviewer_id") or "").strip():
        reasons.append("missing_reviewer_id")
    if outcome_type == "reference_verified" and not str(decision.get("reference_evidence_ref") or "").strip():
        reasons.append("missing_reference_evidence_ref")
    return reasons


def _blocked_row(row: dict[str, Any], reasons: Sequence[str], *, source: str) -> dict[str, Any]:
    return {
        "schema": "audio_quality_review_decision_blocked_row.v1",
        "source": source,
        "queue_id": str(row.get("queue_id") or ""),
        "row_ref": str(row.get("row_ref") or ""),
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def _label_from_decision(decision: dict[str, Any]) -> int:
    return 1 if str(decision.get("decision") or "") == "accept_quality" else 0


def _outcome_row(
    *,
    review_row: dict[str, Any],
    decision: dict[str, Any],
    review_queue_manifest_json: Path,
    decisions_jsonl: Path | None,
    source_lock_pack_json: str,
) -> dict[str, Any]:
    label_type = str(decision.get("outcome_type") or "").strip()
    label = _label_from_decision(decision)
    return {
        **review_row,
        "schema": "audio_quality_learning_outcome.v1",
        "outcome_id": "aqo_" + stable_hash([review_queue_manifest_json.as_posix(), decisions_jsonl.as_posix() if decisions_jsonl else "", review_row.get("queue_id")])[:12],
        "label_type": label_type,
        "label": label,
        "split": str(decision.get("split") or review_row.get("split") or ""),
        "verified_outcome": True,
        "raw_prediction_only": False,
        "safe_for_training": True,
        "review_decision": str(decision.get("decision") or ""),
        "reviewer_id": str(decision.get("reviewer_id") or ""),
        "reference_evidence_ref": str(decision.get("reference_evidence_ref") or ""),
        "review_notes": str(decision.get("review_notes") or ""),
        "source_review_queue_manifest_json": str(review_queue_manifest_json),
        "source_review_decisions_jsonl": str(decisions_jsonl or ""),
        "source_lock_pack_json": source_lock_pack_json,
        "generated_stem_claims_ground_truth": False,
        "pseudo_teacher_targets_are_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "window_start_sec": as_float(review_row.get("window_start_sec"), 0.0),
        "window_end_sec": as_float(review_row.get("window_end_sec"), 0.0),
        "source_manifest_refs": [str(item) for item in as_list(review_row.get("source_manifest_refs"))],
        "rag_refs": as_list(review_row.get("rag_refs")),
    }


def ingest_audio_quality_review_decisions(
    *,
    review_queue_manifest_json: Path,
    decisions_jsonl: Path | None,
    output_dir: Path,
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
        review_queue_manifest_json=str(review_queue_manifest_json),
        decisions_jsonl=str(decisions_jsonl or ""),
        expected_acceleration_mode="external_only_cpu_review_decision_ingest",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="review_decision_ingest_metadata_only",
    )

    review_manifest = read_json(review_queue_manifest_json)
    manifest_errors = _manifest_errors(review_manifest)
    review_jsonl = _review_jsonl_from_manifest(review_manifest)
    decisions, blocked_rows = _load_decisions(decisions_jsonl)
    source_lock_pack_json = str(review_manifest.get("source_lock_pack_json") or "").strip()
    outcomes: list[dict[str, Any]] = []
    seen_queue_ids: set[str] = set()

    if review_jsonl.exists():
        review_rows = list(iter_jsonl(review_jsonl))
        append_progress(progress_path, "scan_review_rows", start_time=start_time, review_queue_count=len(review_rows), decision_count=len(decisions))
        for review_row in review_rows:
            queue_id = str(review_row.get("queue_id") or "").strip()
            seen_queue_ids.add(queue_id)
            row_errors: list[str] = list(manifest_errors)
            if review_row.get("schema") != REVIEW_ROW_SCHEMA:
                row_errors.append("review_row_schema_unexpected")
            decision = decisions.get(queue_id)
            row_errors.extend(_decision_reasons(decision))
            row_errors = _unique(row_errors)
            if row_errors:
                blocked_rows.append(_blocked_row(review_row, row_errors, source="review_row"))
                continue
            outcomes.append(
                _outcome_row(
                    review_row=review_row,
                    decision=decision or {},
                    review_queue_manifest_json=review_queue_manifest_json,
                    decisions_jsonl=decisions_jsonl,
                    source_lock_pack_json=source_lock_pack_json,
                )
            )

    for queue_id in sorted(set(decisions) - seen_queue_ids):
        blocked_rows.append(_blocked_row(decisions[queue_id], ["unknown_queue_id"], source="decision"))

    errors = sorted(set(manifest_errors))
    ingest_status = "blocked" if manifest_errors else "passed"
    write_jsonl(output_dir / OUTCOME_FILE, outcomes)
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_review_decision_ingest",
        "ingest_status": ingest_status,
        "review_queue_manifest_json": str(review_queue_manifest_json),
        "review_jsonl": str(review_jsonl),
        "decisions_jsonl": str(decisions_jsonl or ""),
        "source_lock_pack_json": source_lock_pack_json,
        "verified_outcome_count": len(outcomes),
        "blocked_decision_count": len(blocked_rows),
        "errors": errors,
        "outputs": {
            "audio_quality_learning_outcome_jsonl": str(output_dir / OUTCOME_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
        },
        "presentation_status": "audio_quality_review_decision_ingest_not_for_listen_pack",
        "training_started": False,
        "promotion_allowed": False,
        "safety": {
            "requires_explicit_human_or_reference_decision": True,
            "raw_predictions_excluded_from_training": True,
            "no_self_label_as_ground_truth": True,
            "not_release_artifact": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        ingest_status=ingest_status,
        verified_outcome_count=len(outcomes),
        blocked_decision_count=len(blocked_rows),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest AQC review decisions into verified learning outcomes.")
    parser.add_argument("--review-queue-manifest-json", type=Path, required=True)
    parser.add_argument("--decisions-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="ingest_audio_quality_review_decisions.py",
        entrypoint_path="/app/tools/ingest_audio_quality_review_decisions.py",
    )
    args = parse_args(argv)
    manifest = ingest_audio_quality_review_decisions(
        review_queue_manifest_json=args.review_queue_manifest_json,
        decisions_jsonl=args.decisions_jsonl,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "ingest_status": manifest["ingest_status"],
                "verified_outcome_count": manifest["verified_outcome_count"],
                "blocked_decision_count": manifest["blocked_decision_count"],
                "errors": manifest["errors"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["ingest_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
