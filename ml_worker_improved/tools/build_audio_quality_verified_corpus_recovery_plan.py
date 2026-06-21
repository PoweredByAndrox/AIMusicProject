from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (  # noqa: E402
    PROGRESS_FILE,
    SAFETY_FLAGS,
    append_progress,
    iter_jsonl,
    load_rag_current,
    read_json,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


MANIFEST_SCHEMA = "audio_quality_verified_corpus_recovery_plan.v1"
ROWS_FILE = "rows.jsonl"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.md"
VERIFIED_CORPUS_SCHEMA = "verified_audio_training_corpus_manifest.v1"
REQUIRED_ARTIFACTS = [
    "verified_audio_training_corpus_manifest.v1",
    "generic_finetuning_audio_training_example.v1",
    "full_pack_reference_corpus_gap_plan.v1",
    "generic_audio_finetune_job_plan.v1",
    "full_pack_raw_reference_candidates_audit.v1",
    "full_pack_raw_reference_bundle_worklist.v1",
    "full_pack_raw_reference_verified_bundles_manifest.v1",
]
GENERATED_AQC_EVIDENCE_SCHEMAS = {
    "audio_quality_verified_corpus_recovery_plan.v1",
    "audio_quality_verified_corpus_recovery_row.v1",
    "audio_quality_verified_material_request_packet.v1",
    "audio_quality_verified_material_request_row.v1",
    "audio_quality_verified_material_request_blocked_row.v1",
    "audio_quality_verified_material_decision_template.v1",
    "audio_quality_verified_corpus_discovery_manifest.v1",
    "audio_quality_verified_corpus_discovery_blocked_row.v1",
    "audio_quality_learning_cycle_manifest.v1",
    "audio_quality_learning_handoff.v1",
    "audio_quality_rag_errata_manifest.v1",
}


def _unique(items: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _rag_tables(rag_current_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rag = load_rag_current(rag_current_json)
    if not rag.get("idx_exists"):
        return rag, [], [], []
    idx_path = rag_current_json.parent / str(rag.get("idx") or "")
    idx = read_json(idx_path)
    idx_dir = idx_path.parent
    outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
    manifest_refs = list(iter_jsonl(idx_dir / str(outputs.get("manifest_refs") or "manifest_refs.jsonl")))
    chunks = list(iter_jsonl(idx_dir / str(outputs.get("chunks") or "chunks.jsonl")))
    files = list(iter_jsonl(idx_dir / str(outputs.get("files") or "files.jsonl")))
    return rag, manifest_refs, chunks, files


def _status_fields(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status_fields")
    return status if isinstance(status, dict) else {}


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in ("relative_path", "excerpt", "text", "original_path"))


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or row.get("excerpt") or "")
    try:
        payload = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_verified_corpus_ref(row: dict[str, Any]) -> bool:
    status = _status_fields(row)
    if status.get("schema") == VERIFIED_CORPUS_SCHEMA:
        return True
    return _row_payload(row).get("schema") == VERIFIED_CORPUS_SCHEMA


def _is_full_pack_source_ref(row: dict[str, Any]) -> bool:
    status = _status_fields(row)
    return (
        status.get("manifest_type") == "aris_accepted_full_pack_source_manifest"
        or "aris_accepted_full_pack_source_manifest" in _row_text(row)
    )


def _extract_raw_reference_path(payload: dict[str, Any], text: str) -> str:
    raw_ref = payload.get("raw_reference") if isinstance(payload.get("raw_reference"), dict) else {}
    if raw_ref.get("path"):
        return str(raw_ref.get("path") or "")
    match = re.search(r'"path"\s*:\s*"([^"]*00_raw_ref\.flac)"', text)
    return match.group(1) if match else ""


def _resolve_audio_path(path_text: str, *, project_root: Path | None) -> Path:
    path_text = str(path_text or "").strip()
    if not path_text:
        return Path("")
    if project_root is not None and path_text.startswith("/app/"):
        return project_root / path_text[len("/app/") :]
    return Path(path_text)


def _raw_reference_payload(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    text = str(row.get("text") or row.get("excerpt") or "")
    try:
        payload = json.loads(text)
    except Exception:  # noqa: BLE001
        payload = {}
    return payload if isinstance(payload, dict) else {}, text


def _generated_aqc_schema(row: dict[str, Any], payload: dict[str, Any]) -> str:
    status = _status_fields(row)
    status_schema = str(status.get("schema") or "")
    if status_schema:
        return status_schema
    sample_rows = status.get("sample_rows") if isinstance(status.get("sample_rows"), list) else []
    for sample in sample_rows:
        if isinstance(sample, dict) and sample.get("schema"):
            return str(sample.get("schema") or "")
    return str(payload.get("schema") or "")


def _raw_reference_row(row: dict[str, Any], *, project_root: Path | None) -> dict[str, Any] | None:
    payload, text = _raw_reference_payload(row)
    if _generated_aqc_schema(row, payload) in GENERATED_AQC_EVIDENCE_SCHEMAS:
        return None
    if "raw_reference" not in text and "00_raw_ref.flac" not in text:
        return None
    raw_path = _extract_raw_reference_path(payload, text)
    if not raw_path:
        return None
    resolved = _resolve_audio_path(raw_path, project_root=project_root)
    status = _status_fields(row)
    contract = payload.get("ground_truth_contract") if isinstance(payload.get("ground_truth_contract"), dict) else {}
    verified = contract.get("verified") is True
    blockers: list[str] = []
    if not verified:
        blockers.append("raw_reference_not_verified_ground_truth")
    if payload.get("model_id") or isinstance(payload.get("model"), dict):
        blockers.append("raw_reference_from_separator_model_output")
    if status.get("presentation_status") == "diagnostic_candidate_not_for_listen_pack" or payload.get("presentation_status") == "diagnostic_candidate_not_for_listen_pack":
        blockers.append("raw_reference_from_diagnostic_candidate_manifest")
    if not resolved.exists():
        blockers.append("raw_reference_audio_missing")
    relative_parts = Path(str(row.get("relative_path") or "")).parts
    target_stem = str(payload.get("target_stem") or "")
    stem_type = target_stem or (str(relative_parts[-2]) if len(relative_parts) >= 2 else "")
    return {
        "schema": "audio_quality_verified_corpus_recovery_row.v1",
        "root": str(row.get("root") or ""),
        "root_alias": str(row.get("root_alias") or ""),
        "relative_path": str(row.get("relative_path") or ""),
        "chunk_id": str(row.get("chunk_id") or ""),
        "raw_reference_path": raw_path,
        "resolved_audio_path": str(resolved) if raw_path else "",
        "audio_exists": bool(raw_path and resolved.exists()),
        "track_id_hint": str(relative_parts[1] if len(relative_parts) > 1 else ""),
        "stem_type": stem_type,
        "model_id": str(payload.get("model_id") or ((payload.get("model") or {}).get("model_id") if isinstance(payload.get("model"), dict) else "")),
        "presentation_status": str(status.get("presentation_status") or payload.get("presentation_status") or ""),
        "raw_reference_verified": verified,
        "blockers": _unique(blockers),
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def _summary_md(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Audio Quality Verified Corpus Recovery Plan",
            "",
            f"- recovery_status: `{manifest['recovery_status']}`",
            f"- verified_corpus_ref_count: `{manifest['verified_corpus_ref_count']}`",
            f"- raw_reference_manifest_count: `{manifest['raw_reference_manifest_count']}`",
            f"- raw_reference_unverified_count: `{manifest['raw_reference_unverified_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_audio_quality_verified_corpus_recovery_plan(
    *,
    rag_current_json: Path,
    output_dir: Path,
    project_root: Path | None = None,
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
        rag_current_json=str(rag_current_json),
        expected_acceleration_mode="external_only_cpu_rag_metadata_recovery_planning",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="verified_corpus_recovery_plan_no_model_inference",
    )

    rag, manifest_refs, chunks, files = _rag_tables(rag_current_json)
    verified_refs = [row for row in [*manifest_refs, *chunks] if _is_verified_corpus_ref(row)]
    full_pack_refs = [row for row in manifest_refs if _is_full_pack_source_ref(row)]
    raw_rows = [
        recovery_row
        for row in chunks
        for recovery_row in [_raw_reference_row(row, project_root=project_root)]
        if recovery_row is not None
    ]
    blocker_counts = Counter(reason for row in raw_rows for reason in row["blockers"])
    existing_audio_count = sum(1 for row in raw_rows if row["audio_exists"])
    verified_raw_count = sum(1 for row in raw_rows if row["raw_reference_verified"])
    unverified_raw_count = len(raw_rows) - verified_raw_count
    if verified_refs:
        status = "verified_corpus_ref_present"
        next_action = "run_verified_corpus_discovery"
    elif raw_rows:
        status = "blocked_waiting_for_verified_corpus_material"
        next_action = "verify_raw_reference_material_then_build_verified_corpus"
    else:
        status = "blocked_no_verified_or_raw_reference_material"
        next_action = "collect_human_or_reference_verified_corpus"

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_verified_corpus_recovery_plan",
        "recovery_status": status,
        "next_action": next_action,
        "current_rag_id": str(rag.get("rag_id") or ""),
        "rag_current_json": str(rag_current_json),
        "rag_idx": str(rag.get("idx") or ""),
        "rag_idx_exists": bool(rag.get("idx_exists")),
        "scanned_manifest_ref_count": len(manifest_refs),
        "scanned_chunk_count": len(chunks),
        "scanned_file_count": len(files),
        "verified_corpus_ref_count": len(verified_refs),
        "full_pack_source_manifest_count": len(full_pack_refs),
        "raw_reference_manifest_count": len(raw_rows),
        "raw_reference_existing_audio_count": existing_audio_count,
        "raw_reference_verified_count": verified_raw_count,
        "raw_reference_unverified_count": unverified_raw_count,
        "raw_reference_blocker_counts": dict(sorted(blocker_counts.items())),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "rows_jsonl": str(output_dir / ROWS_FILE),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_FILE),
        },
        "presentation_status": "audio_quality_verified_corpus_recovery_plan_not_for_listen_pack",
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "promotion_allowed": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "dataset_built": False,
            "training_started": False,
            "ground_truth_manifest_written": False,
            "raw_reference_not_auto_ground_truth": True,
            "not_release_artifact": True,
        },
        **SAFETY_FLAGS,
    }
    write_jsonl(output_dir / ROWS_FILE, raw_rows)
    write_json(output_dir / MANIFEST_FILE, manifest)
    (output_dir / SUMMARY_FILE).write_text(_summary_md(manifest), encoding="utf-8")
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        recovery_status=status,
        raw_reference_manifest_count=len(raw_rows),
        verified_corpus_ref_count=len(verified_refs),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan verified-corpus recovery from current RAG without creating labels.")
    parser.add_argument("--rag-current-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_verified_corpus_recovery_plan.py",
        entrypoint_path="/app/tools/build_audio_quality_verified_corpus_recovery_plan.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_verified_corpus_recovery_plan(
        rag_current_json=args.rag_current_json,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "recovery_status": manifest["recovery_status"],
                "raw_reference_manifest_count": manifest["raw_reference_manifest_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
