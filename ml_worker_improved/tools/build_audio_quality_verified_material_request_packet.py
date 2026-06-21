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
    iter_jsonl,
    read_json,
    stable_hash,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime

MANIFEST_SCHEMA = "audio_quality_verified_material_request_packet.v1"
RECOVERY_SCHEMA = "audio_quality_verified_corpus_recovery_plan.v1"
REQUEST_SCHEMA = "audio_quality_verified_material_request_row.v1"
DECISION_TEMPLATE_SCHEMA = "audio_quality_verified_material_decision_template.v1"
REQUESTS_FILE = "requests.jsonl"
DECISIONS_FILE = "decisions.jsonl"
BLOCKED_TRUE_FLAGS = (
    "delivery_allowed",
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "promotion_allowed",
    "thresholds_relaxed",
)
ALLOWED_OUTCOME_TYPES = ["human_verified", "reference_verified", "reject_quality", "exclude"]


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
    if manifest.get("schema") != RECOVERY_SCHEMA:
        errors.append("recovery_manifest_schema_unexpected")
    for key in BLOCKED_TRUE_FLAGS:
        if manifest.get(key) is True:
            errors.append(f"recovery_manifest_{key}_true")
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    rows_jsonl_text = str(outputs.get("rows_jsonl") or "").strip()
    if not rows_jsonl_text:
        errors.append("recovery_rows_jsonl_missing")
    else:
        rows_jsonl = Path(rows_jsonl_text)
        if not rows_jsonl.exists():
            errors.append("recovery_rows_jsonl_missing")
        elif not rows_jsonl.is_file():
            errors.append("recovery_rows_jsonl_not_file")
    return sorted(set(errors))


def _rows_jsonl(manifest: dict[str, Any]) -> Path:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    return Path(str(outputs.get("rows_jsonl") or ""))


def _source_ref(row: dict[str, Any]) -> dict[str, str]:
    return {
        "root": str(row.get("root") or ""),
        "root_alias": str(row.get("root_alias") or ""),
        "relative_path": str(row.get("relative_path") or ""),
        "chunk_id": str(row.get("chunk_id") or ""),
    }


def _request_reasons(row: dict[str, Any]) -> list[str]:
    reasons = list(row.get("blockers") if isinstance(row.get("blockers"), list) else [])
    if row.get("audio_exists") is not True:
        reasons.append("raw_reference_audio_not_available_for_review")
    if row.get("raw_reference_verified") is True:
        reasons.append("raw_reference_already_verified")
    return _unique(reasons)


def _is_requestable(row: dict[str, Any]) -> bool:
    return (
        row.get("schema") == "audio_quality_verified_corpus_recovery_row.v1"
        and row.get("audio_exists") is True
        and row.get("raw_reference_verified") is not True
    )


def _request_row(
    *,
    row: dict[str, Any],
    recovery_manifest_json: Path,
    request_index: int,
) -> dict[str, Any]:
    request_id = "aqvr_" + stable_hash(
        [
            recovery_manifest_json.as_posix(),
            row.get("root_alias"),
            row.get("relative_path"),
            row.get("chunk_id"),
            row.get("raw_reference_path"),
        ]
    )[:12]
    return {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "request_index": request_index,
        "source_recovery_manifest_json": str(recovery_manifest_json),
        "source_recovery_row_ref": _source_ref(row),
        "source_blockers": _request_reasons(row),
        "raw_reference_path": str(row.get("raw_reference_path") or ""),
        "resolved_audio_path": str(row.get("resolved_audio_path") or ""),
        "audio_exists": bool(row.get("audio_exists")),
        "track_id_hint": str(row.get("track_id_hint") or ""),
        "stem_type": str(row.get("stem_type") or ""),
        "model_id": str(row.get("model_id") or ""),
        "presentation_status": str(row.get("presentation_status") or ""),
        "verification_required": True,
        "allowed_outcome_types": ALLOWED_OUTCOME_TYPES,
        "raw_prediction_only": True,
        "verified_outcome": False,
        "safe_for_training": False,
        "generated_stem_claims_ground_truth": False,
        "pseudo_teacher_targets_are_ground_truth": False,
        "must_not_mark_generated_stem_ground_truth": True,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def _decision_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": DECISION_TEMPLATE_SCHEMA,
        "request_id": str(row.get("request_id") or ""),
        "source_recovery_manifest_json": str(row.get("source_recovery_manifest_json") or ""),
        "allowed_outcome_types": ALLOWED_OUTCOME_TYPES,
        "decision": "",
        "outcome_type": "",
        "reviewer_id": "",
        "reference_evidence_ref": "",
        "review_notes": "",
        "delivery_allowed": False,
        "release_ready": False,
    }


def _blocked_row(row: dict[str, Any], reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "audio_quality_verified_material_request_blocked_row.v1",
        "source_recovery_row_ref": _source_ref(row),
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def build_audio_quality_verified_material_request_packet(
    *,
    recovery_manifest_json: Path,
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
        recovery_manifest_json=str(recovery_manifest_json),
        expected_acceleration_mode="external_only_cpu_verified_material_request_planning",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="verified_material_request_packet_metadata_only",
    )

    recovery_manifest = read_json(recovery_manifest_json)
    errors = _manifest_errors(recovery_manifest)
    requests: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    recovery_rows_jsonl = _rows_jsonl(recovery_manifest)

    if errors:
        blocked_rows.append(_blocked_row({}, errors))
    else:
        recovery_rows = list(iter_jsonl(recovery_rows_jsonl))
        append_progress(progress_path, "scan_recovery_rows", start_time=start_time, recovery_row_count=len(recovery_rows))
        for row in recovery_rows:
            reasons = _request_reasons(row)
            if _is_requestable(row):
                requests.append(_request_row(row=row, recovery_manifest_json=recovery_manifest_json, request_index=len(requests) + 1))
            else:
                blocked_rows.append(_blocked_row(row, reasons or ["recovery_row_not_requestable"]))

    decision_templates = [_decision_template(row) for row in requests]
    request_status = "blocked" if errors else ("verification_required" if requests else "no_requestable_material")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_verified_material_request_packet",
        "request_status": request_status,
        "recovery_manifest_json": str(recovery_manifest_json),
        "recovery_rows_jsonl": str(recovery_rows_jsonl),
        "request_count": len(requests),
        "blocked_row_count": len(blocked_rows),
        "errors": errors,
        "allowed_outcome_types": ALLOWED_OUTCOME_TYPES,
        "outputs": {
            "requests_jsonl": str(output_dir / REQUESTS_FILE),
            "decisions_jsonl": str(output_dir / DECISIONS_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
        },
        "presentation_status": "audio_quality_verified_material_request_packet_not_for_listen_pack",
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "promotion_allowed": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "ground_truth_manifest_written": False,
        "safety": {
            "metadata_only": True,
            "audio_copied": False,
            "audio_decoded": False,
            "dataset_built": False,
            "requires_human_or_reference_verification": True,
            "no_self_label_as_ground_truth": True,
            "not_release_artifact": True,
            "short_output_names": True,
        },
        **SAFETY_FLAGS,
    }
    write_jsonl(output_dir / REQUESTS_FILE, requests)
    write_jsonl(output_dir / DECISIONS_FILE, decision_templates)
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        request_status=request_status,
        request_count=len(requests),
        blocked_row_count=len(blocked_rows),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verified-material request packet from AQC recovery evidence.")
    parser.add_argument("--recovery-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_verified_material_request_packet.py",
        entrypoint_path="/app/tools/build_audio_quality_verified_material_request_packet.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_verified_material_request_packet(
        recovery_manifest_json=args.recovery_manifest_json,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "request_status": manifest["request_status"],
                "request_count": manifest["request_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
