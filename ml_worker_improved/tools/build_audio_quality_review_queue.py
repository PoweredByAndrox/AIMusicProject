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
    REVIEW_FILE,
    SAFETY_FLAGS,
    SCORE_MANIFEST_SCHEMA,
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


MANIFEST_SCHEMA = "audio_quality_review_queue_manifest.v1"
BLOCKED_TRUE_FLAGS = (
    "delivery_allowed",
    "release_ready",
    "can_publish_as_result",
    "thresholds_relaxed",
    "model_registry_edited",
    "runtime_gates_changed",
)
REQUESTED_OUTCOME_TYPES = ["human_verified", "reference_verified"]


def _unique(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _score_manifest_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != SCORE_MANIFEST_SCHEMA:
        errors.append("score_manifest_schema_unexpected")
    for key in BLOCKED_TRUE_FLAGS:
        if manifest.get(key) is True:
            errors.append(f"score_manifest_{key}_true")
    if not str(manifest.get("source_lock_pack_json") or "").strip():
        errors.append("score_manifest_source_lock_missing")
    scores_jsonl = Path(str(manifest.get("scores_jsonl") or ""))
    if not str(scores_jsonl) or not scores_jsonl.exists():
        errors.append("scores_jsonl_missing")
    return sorted(set(errors))


def _review_reasons(row: dict[str, Any], *, uncertainty_threshold: float) -> list[str]:
    reasons: list[str] = []
    if row.get("proxy_critic_disagreement") is True:
        reasons.append("proxy_critic_disagreement")
    if as_float(row.get("uncertainty"), 0.0) >= float(uncertainty_threshold):
        reasons.append("critic_uncertainty_high")
    return _unique(reasons)


def _priority(reasons: Sequence[str], row: dict[str, Any]) -> int:
    priority = 50
    if "proxy_critic_disagreement" in reasons:
        priority += 30
    if "critic_uncertainty_high" in reasons:
        priority += 20
    priority += min(20, int(as_float(row.get("uncertainty"), 0.0) * 20))
    return min(priority, 100)


def _review_row(
    *,
    row: dict[str, Any],
    reasons: Sequence[str],
    queue_index: int,
    score_manifest_json: Path,
    source_lock_pack_json: str,
) -> dict[str, Any]:
    return {
        "schema": "audio_quality_review_queue_row.v1",
        "queue_id": f"q{queue_index:03d}",
        "row_ref": "aqrq_" + stable_hash([score_manifest_json.as_posix(), row.get("row_id"), row.get("candidate_id"), queue_index])[:12],
        "review_reasons": list(reasons),
        "review_priority": _priority(reasons, row),
        "requested_outcome_types": REQUESTED_OUTCOME_TYPES,
        "source_score_manifest_json": str(score_manifest_json),
        "source_lock_pack_json": source_lock_pack_json,
        "row_id": str(row.get("row_id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "stem_type": str(row.get("stem_type") or ""),
        "mix_path": str(row.get("mix_path") or ""),
        "vocal_path": str(row.get("vocal_path") or ""),
        "minus_path": str(row.get("minus_path") or ""),
        "tensor_path": str(row.get("tensor_path") or ""),
        "features": row.get("features") if isinstance(row.get("features"), (dict, list)) else None,
        "targets": row.get("targets") if isinstance(row.get("targets"), dict) else {},
        "route_id": str(row.get("route_id") or ""),
        "window_start_sec": as_float(row.get("window_start_sec"), 0.0),
        "window_end_sec": as_float(row.get("window_end_sec"), 0.0),
        "label_type": str(row.get("label_type") or ""),
        "proxy_decision": str(row.get("proxy_decision") or row.get("decision") or row.get("pair_decision") or ""),
        "selected_action": str(row.get("selected_action") or ""),
        "accept_probability": as_float(row.get("accept_probability"), 0.0),
        "uncertainty": as_float(row.get("uncertainty"), 0.0),
        "proxy_critic_disagreement": row.get("proxy_critic_disagreement") is True,
        "issue_tags": [str(item) for item in as_list(row.get("issue_tags"))],
        "blocker_reasons": [str(item) for item in as_list(row.get("blocker_reasons"))],
        "source_manifest_refs": [str(item) for item in as_list(row.get("source_manifest_refs"))],
        "rag_refs": as_list(row.get("rag_refs")),
        "raw_prediction_only": True,
        "verified_outcome": False,
        "safe_for_training": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def _blocked_row(reasons: Sequence[str], *, score_manifest_json: Path) -> dict[str, Any]:
    return {
        "schema": "audio_quality_review_queue_blocked_row.v1",
        "score_manifest_json": str(score_manifest_json),
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        "ready_for_review": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def build_audio_quality_review_queue(
    *,
    score_manifest_json: Path,
    output_dir: Path,
    uncertainty_threshold: float = 0.8,
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
        score_manifest_json=str(score_manifest_json),
        uncertainty_threshold=float(uncertainty_threshold),
        expected_acceleration_mode="external_only_cpu_score_review_queue",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="review_queue_metadata_only",
    )

    score_manifest = read_json(score_manifest_json)
    errors = _score_manifest_errors(score_manifest)
    source_lock_pack_json = str(score_manifest.get("source_lock_pack_json") or "").strip()
    review_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    scores_jsonl = Path(str(score_manifest.get("scores_jsonl") or ""))

    if errors:
        blocked_rows.append(_blocked_row(errors, score_manifest_json=score_manifest_json))
    else:
        scored_rows = list(iter_jsonl(scores_jsonl))
        append_progress(progress_path, "scan_scores", start_time=start_time, scored_row_count=len(scored_rows))
        for row in scored_rows:
            reasons = _review_reasons(row, uncertainty_threshold=float(uncertainty_threshold))
            if not reasons:
                continue
            review_rows.append(
                _review_row(
                    row=row,
                    reasons=reasons,
                    queue_index=len(review_rows) + 1,
                    score_manifest_json=score_manifest_json,
                    source_lock_pack_json=source_lock_pack_json,
                )
            )

    queue_status = "blocked" if errors else ("review_required" if review_rows else "no_review_required")
    write_jsonl(output_dir / REVIEW_FILE, review_rows)
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_review_queue",
        "queue_status": queue_status,
        "score_manifest_json": str(score_manifest_json),
        "scores_jsonl": str(scores_jsonl),
        "source_lock_pack_json": source_lock_pack_json,
        "uncertainty_threshold": float(uncertainty_threshold),
        "review_queue_count": len(review_rows),
        "blocked_row_count": len(blocked_rows),
        "errors": errors,
        "outputs": {
            "review_jsonl": str(output_dir / REVIEW_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
        },
        "presentation_status": "audio_quality_review_queue_not_for_listen_pack",
        "training_started": False,
        "promotion_allowed": False,
        "safety": {
            "raw_predictions_excluded_from_training": True,
            "requires_human_or_reference_verification": True,
            "no_self_label_as_ground_truth": True,
            "not_release_artifact": True,
            "short_output_names": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        queue_status=queue_status,
        review_queue_count=len(review_rows),
        blocked_row_count=len(blocked_rows),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build human/reference review queue from AQC shadow scores.")
    parser.add_argument("--score-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_review_queue.py",
        entrypoint_path="/app/tools/build_audio_quality_review_queue.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_review_queue(
        score_manifest_json=args.score_manifest_json,
        output_dir=args.output_dir,
        uncertainty_threshold=float(args.uncertainty_threshold),
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "queue_status": manifest["queue_status"],
                "review_queue_count": manifest["review_queue_count"],
                "errors": manifest["errors"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["queue_status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
