from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (  # noqa: E402
    BLOCKED_FILE,
    MANIFEST_FILE,
    MODEL_MANIFEST_SCHEMA,
    PROGRESS_FILE,
    SAFETY_FLAGS,
    VALIDATION_SCHEMA,
    append_progress,
    iter_jsonl,
    load_rag_current,
    read_json,
    stable_hash,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


MANIFEST_SCHEMA = "audio_quality_promotion_candidate_discovery.v1"
CANDIDATES_FILE = "candidates.jsonl"
SUMMARY_FILE = "summary.md"
SELECTED_ACTION = "build_audio_quality_promotion_decision"
RAG_SEARCH_ORDER = ["current", "idx", "manifest_refs", "chunks", "files"]


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("\\", "/").rstrip("/").lower()


def _location_key(row: dict[str, Any]) -> str:
    root_alias = str(row.get("root_alias") or "").strip().lower()
    relative_path = str(row.get("relative_path") or "").strip().replace("\\", "/").lower()
    if not root_alias or not relative_path:
        return ""
    return f"{root_alias}:{relative_path}"


def _identity_key(row: dict[str, Any]) -> str:
    path_key = _path_key(row.get("original_path") or row.get("source_manifest_json"))
    if path_key:
        return f"path:{path_key}"
    location_key = _location_key(row)
    if location_key:
        return f"loc:{location_key}"
    return "chunk:" + str(row.get("chunk_id") or stable_hash(row)[:16])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _load_rag_tables(
    rag_current_json: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rag = load_rag_current(rag_current_json)
    if not rag.get("idx_exists"):
        return rag, [], [], []
    idx_path = rag_current_json.parent / str(rag.get("idx") or "")
    idx = read_json(idx_path)
    idx_dir = idx_path.parent
    outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
    manifest_refs = _load_jsonl(idx_dir / str(outputs.get("manifest_refs") or "manifest_refs.jsonl"))
    chunks = _load_jsonl(idx_dir / str(outputs.get("chunks") or "chunks.jsonl"))
    files = _load_jsonl(idx_dir / str(outputs.get("files") or "files.jsonl"))
    return rag, manifest_refs, chunks, files


def _files_by_location(files: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in files:
        key = _location_key(row)
        if key and key not in result:
            result[key] = row
    return result


def _merge_ref(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key == "source_tables":
            merged[key] = _unique([*(merged.get(key) or []), *(value or [])])
            continue
        if value not in ("", None, [], {}):
            merged[key] = value
    return merged


def _table_ref(row: dict[str, Any], *, source_table: str, file_by_location: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref = dict(row)
    ref["source_tables"] = _unique([source_table, *(row.get("source_tables") or [])])
    location = _location_key(row)
    if location and location in file_by_location:
        ref = _merge_ref(file_by_location[location], ref)
        ref["source_tables"] = _unique([source_table, "files", *(row.get("source_tables") or [])])
    return ref


def _is_validation_ref(row: dict[str, Any]) -> bool:
    status = row.get("status_fields") if isinstance(row.get("status_fields"), dict) else {}
    if status.get("schema") == VALIDATION_SCHEMA:
        return True
    text = " ".join(str(row.get(key) or "") for key in ("relative_path", "excerpt", "text"))
    return VALIDATION_SCHEMA in text


def _validation_refs_from_rag_tables(
    *,
    manifest_refs: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    files: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    file_by_location = _files_by_location(files)
    collected: dict[str, dict[str, Any]] = {}
    for source_table, rows in (("manifest_refs", manifest_refs), ("chunks", chunks)):
        for row in rows:
            if not _is_validation_ref(row):
                continue
            ref = _table_ref(row, source_table=source_table, file_by_location=file_by_location)
            key = _identity_key(ref)
            collected[key] = _merge_ref(collected[key], ref) if key in collected else ref
    return list(collected.values())


def _all_evidence_refs(
    *,
    manifest_refs: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    files: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    file_by_location = _files_by_location(files)
    collected: dict[str, dict[str, Any]] = {}
    for source_table, rows in (("manifest_refs", manifest_refs), ("chunks", chunks), ("files", files)):
        for row in rows:
            ref = _table_ref(row, source_table=source_table, file_by_location=file_by_location)
            key = _identity_key(ref)
            collected[key] = _merge_ref(collected[key], ref) if key in collected else ref
    return list(collected.values())


def _existing_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text or text == ".":
        return None
    path = Path(text)
    if path.exists():
        return path
    normalized = text.replace("\\", "/")
    marker = "/AIMusicProject/"
    if marker in normalized:
        candidate = APP_ROOT.parent / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return None


def _ref_path_values(ref: dict[str, Any]) -> list[str]:
    return [
        str(ref.get("original_path") or ""),
        str(ref.get("source_manifest_json") or ""),
    ]


def _ref_matches_path(ref: dict[str, Any], path: Path) -> bool:
    target = _path_key(path)
    if not target:
        return False
    for value in _ref_path_values(ref):
        if _path_key(value) == target:
            return True
    existing = _existing_path(path)
    if existing is not None:
        existing_key = _path_key(existing)
        return any(_path_key(_existing_path(value) or value) == existing_key for value in _ref_path_values(ref))
    return False


def _evidence_for_path(path: Path, evidence_refs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for ref in evidence_refs:
        if _ref_matches_path(ref, path):
            return ref
    return None


def _rag_ref(ref: dict[str, Any], rag: dict[str, Any], *, ref_type: str) -> dict[str, Any]:
    return {
        "rag_id": str(rag.get("rag_id") or ""),
        "idx": str(rag.get("idx") or ""),
        "idx_exists": bool(rag.get("idx_exists")),
        "ref_type": ref_type,
        "source_tables": _unique(ref.get("source_tables") or [ref.get("source_table")]),
        "root": str(ref.get("root") or ""),
        "root_alias": str(ref.get("root_alias") or ""),
        "relative_path": str(ref.get("relative_path") or ""),
        "original_path": str(ref.get("original_path") or ref.get("source_manifest_json") or ""),
        "chunk_id": str(ref.get("chunk_id") or ""),
        "sha256": str(ref.get("sha256") or ref.get("text_sha256") or ""),
        "zip_member": str(ref.get("zip_member") or ref.get("text_zip_member") or ""),
    }


def _rag_ref_reasons(ref: dict[str, Any] | None, prefix: str) -> list[str]:
    if ref is None:
        return [f"{prefix}_rag_ref_missing"]
    reasons: list[str] = []
    if not str(ref.get("chunk_id") or "").strip():
        reasons.append(f"{prefix}_rag_ref_chunk_id_missing")
    if not str(ref.get("sha256") or ref.get("text_sha256") or "").strip():
        reasons.append(f"{prefix}_rag_ref_sha256_missing")
    return reasons


def _read_json_path(path: Path, label: str, reasons: list[str]) -> dict[str, Any]:
    existing = _existing_path(path)
    if existing is None:
        reasons.append(f"{label}_missing")
        return {}
    if existing.is_dir():
        reasons.append(f"{label}_is_directory")
        return {}
    try:
        return read_json(existing)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"{label}_read_failed:{type(exc).__name__}")
        return {}


def _holdout_score_reasons(path: Path) -> list[str]:
    existing = _existing_path(path)
    if existing is None:
        return ["holdout_scores_jsonl_missing"]
    if existing.is_dir():
        return ["holdout_scores_jsonl_is_directory"]
    try:
        rows = list(iter_jsonl(existing))
    except Exception as exc:  # noqa: BLE001
        return [f"holdout_scores_jsonl_read_failed:{type(exc).__name__}"]
    return [] if rows else ["holdout_scores_empty"]


def _validation_reasons(validation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if validation.get("schema") != VALIDATION_SCHEMA:
        reasons.append("validation_schema_unexpected")
    if validation.get("ok") is not True:
        reasons.append("validation_not_ok")
    if validation.get("promotion_allowed") is not True:
        reasons.append("validation_promotion_allowed_not_true")
    if validation.get("delivery_allowed") is True:
        reasons.append("validation_delivery_allowed_true")
    if validation.get("release_ready") is True:
        reasons.append("validation_release_ready_true")
    if validation.get("require_promotion_ready") is not True:
        reasons.append("validation_require_promotion_ready_not_true")
    return reasons


def _model_reasons(model_manifest: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if model_manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        reasons.append("model_manifest_schema_unexpected")
    if model_manifest.get("input_tensor_required") is not True:
        reasons.append("model_input_tensor_required_not_true")
    if model_manifest.get("training_completed") is not True:
        reasons.append("model_training_completed_not_true")
    if model_manifest.get("delivery_allowed") is True:
        reasons.append("model_delivery_allowed_true")
    if model_manifest.get("release_ready") is True:
        reasons.append("model_release_ready_true")
    if model_manifest.get("can_publish_as_result") is True:
        reasons.append("model_can_publish_as_result_true")
    model_file = str(model_manifest.get("model_file") or "").strip()
    if model_file and _existing_path(model_file) is None:
        reasons.append("model_file_missing")
    return reasons


def _audit_reasons(audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(audit.get("overall_verdict") or "").strip().lower() != "pass":
        reasons.append("experiment_audit_not_pass")
    if str(audit.get("integrity_status") or "").strip().lower() != "pass":
        reasons.append("experiment_audit_integrity_not_pass")
    if not str(audit.get("source_packet_manifest_json") or "").strip():
        reasons.append("experiment_audit_ingest_provenance_missing")
    if audit.get("delivery_allowed") is True:
        reasons.append("experiment_audit_delivery_allowed_true")
    if audit.get("release_ready") is True:
        reasons.append("experiment_audit_release_ready_true")
    if audit.get("can_publish_as_result") is True:
        reasons.append("experiment_audit_can_publish_as_result_true")
    return reasons


def _validation_path(ref: dict[str, Any]) -> Path | None:
    for value in _ref_path_values(ref):
        existing = _existing_path(value)
        if existing is not None:
            return existing
    return None


def _blocked_row(ref: dict[str, Any], reasons: Sequence[str], *, validation_json: Path | None = None) -> dict[str, Any]:
    return {
        "schema": "audio_quality_promotion_candidate_blocked_row.v1",
        "validation_json": str(validation_json or ""),
        "root_alias": str(ref.get("root_alias") or ""),
        "relative_path": str(ref.get("relative_path") or ""),
        "chunk_id": str(ref.get("chunk_id") or ""),
        "sha256": str(ref.get("sha256") or ref.get("text_sha256") or ""),
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        "ready_for_promotion_decision": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def _candidate_row(
    *,
    rag: dict[str, Any],
    row_index: int,
    validation_ref: dict[str, Any],
    model_ref: dict[str, Any],
    holdout_ref: dict[str, Any],
    audit_ref: dict[str, Any],
    model_manifest_json: Path,
    validation_json: Path,
    holdout_scores_jsonl: Path,
    experiment_audit_json: Path,
) -> dict[str, Any]:
    row_id = "aqp_" + stable_hash([str(validation_json), str(model_manifest_json), str(holdout_scores_jsonl), str(experiment_audit_json), row_index])[:12]
    return {
        "schema": "audio_quality_promotion_candidate.v1",
        "row_id": row_id,
        "ready_for_promotion_decision": True,
        "selected_action": SELECTED_ACTION,
        "model_manifest_json": str(model_manifest_json),
        "validation_json": str(validation_json),
        "holdout_scores_jsonl": str(holdout_scores_jsonl),
        "experiment_audit_json": str(experiment_audit_json),
        "rag_refs": [
            _rag_ref(validation_ref, rag, ref_type="audio_quality_validation_ref"),
            _rag_ref(model_ref, rag, ref_type="audio_quality_model_manifest_ref"),
            _rag_ref(holdout_ref, rag, ref_type="audio_quality_holdout_scores_ref"),
            _rag_ref(audit_ref, rag, ref_type="audio_quality_experiment_audit_ref"),
        ],
        "source_manifest_refs": [str(model_manifest_json), str(validation_json), str(experiment_audit_json)],
        "blocked_reasons": [],
        "promotion_allowed": False,
        "runtime_authority_enabled": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def _summary_text(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Audio Quality Promotion Candidate Discovery",
            "",
            f"Status: {manifest['discovery_status']}",
            f"Current RAG: `{manifest['current_rag_id']}`",
            f"Candidate count: `{manifest['candidate_count']}`",
            f"Blocked refs: `{manifest['blocked_ref_count']}`",
            "- delivery_allowed: `false`",
            "- release_ready: `false`",
            "",
        ]
    )


def discover_audio_quality_promotion_candidates(*, rag_current_json: Path, output_dir: Path) -> dict[str, Any]:
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
        expected_acceleration_mode="external_only_cpu_rag_metadata_discovery",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="promotion_candidate_discovery_no_model_inference",
    )

    rag, manifest_refs, chunks, files = _load_rag_tables(rag_current_json)
    validation_refs = _validation_refs_from_rag_tables(manifest_refs=manifest_refs, chunks=chunks, files=files)
    evidence_refs = _all_evidence_refs(manifest_refs=manifest_refs, chunks=chunks, files=files)
    append_progress(
        progress_path,
        "scan_rag_tables",
        start_time=start_time,
        current_rag_id=str(rag.get("rag_id") or ""),
        rag_idx=str(rag.get("idx") or ""),
        rag_idx_exists=bool(rag.get("idx_exists")),
        scanned_manifest_ref_count=len(manifest_refs),
        scanned_chunk_count=len(chunks),
        scanned_file_count=len(files),
        validation_ref_count=len(validation_refs),
        rag_search_order=RAG_SEARCH_ORDER,
    )

    candidates: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    if not rag.get("idx_exists"):
        blocked_rows.append(_blocked_row({}, ["rag_idx_missing"]))

    for validation_ref in validation_refs:
        reasons: list[str] = []
        reasons.extend(_rag_ref_reasons(validation_ref, "validation"))
        validation_json = _validation_path(validation_ref)
        if validation_json is None:
            reasons.append("validation_json_missing")
            blocked_rows.append(_blocked_row(validation_ref, _unique(reasons)))
            continue

        validation = _read_json_path(validation_json, "validation_json", reasons)
        reasons.extend(_validation_reasons(validation) if validation else [])
        model_dir_text = str(validation.get("model_dir") or "").strip()
        holdout_scores_text = str(validation.get("holdout_scores_jsonl") or "").strip()
        experiment_audit_text = str(validation.get("experiment_audit_json") or "").strip()
        model_manifest_json = Path(model_dir_text) / MANIFEST_FILE if model_dir_text else Path("")
        holdout_scores_jsonl = Path(holdout_scores_text) if holdout_scores_text else Path("")
        experiment_audit_json = Path(experiment_audit_text) if experiment_audit_text else Path("")

        if not model_dir_text:
            reasons.append("model_manifest_json_missing")
        if not holdout_scores_text:
            reasons.append("holdout_scores_jsonl_missing")
        if not experiment_audit_text:
            reasons.append("experiment_audit_json_missing")

        model_ref = _evidence_for_path(model_manifest_json, evidence_refs) if model_dir_text else None
        holdout_ref = _evidence_for_path(holdout_scores_jsonl, evidence_refs) if holdout_scores_text else None
        audit_ref = _evidence_for_path(experiment_audit_json, evidence_refs) if experiment_audit_text else None
        if model_dir_text:
            reasons.extend(_rag_ref_reasons(model_ref, "model_manifest"))
        if holdout_scores_text:
            reasons.extend(_rag_ref_reasons(holdout_ref, "holdout_scores"))
        if experiment_audit_text:
            reasons.extend(_rag_ref_reasons(audit_ref, "experiment_audit"))

        model_manifest = _read_json_path(model_manifest_json, "model_manifest", reasons) if model_dir_text else {}
        audit = _read_json_path(experiment_audit_json, "experiment_audit_json", reasons) if experiment_audit_text else {}
        reasons.extend(_model_reasons(model_manifest) if model_manifest else [])
        reasons.extend(_audit_reasons(audit) if audit else [])
        reasons.extend(_holdout_score_reasons(holdout_scores_jsonl) if holdout_scores_text else [])
        if validation:
            reasons.extend(str(item) for item in validation.get("errors", []) if item)
        reasons = _unique(reasons)
        if reasons or model_ref is None or holdout_ref is None or audit_ref is None:
            blocked_rows.append(_blocked_row(validation_ref, reasons, validation_json=validation_json))
            continue
        candidates.append(
            _candidate_row(
                rag=rag,
                row_index=len(candidates) + 1,
                validation_ref=validation_ref,
                model_ref=model_ref,
                holdout_ref=holdout_ref,
                audit_ref=audit_ref,
                model_manifest_json=model_manifest_json,
                validation_json=validation_json,
                holdout_scores_jsonl=holdout_scores_jsonl,
                experiment_audit_json=experiment_audit_json,
            )
        )

    errors: list[str] = []
    if not validation_refs:
        errors.append("promotion_validation_ref_not_found_in_current_rag")
    if not candidates:
        errors.append("promotion_candidate_not_found")
    if not rag.get("idx_exists"):
        errors.append("rag_idx_missing")
    write_jsonl(output_dir / CANDIDATES_FILE, candidates)
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_promotion_candidate_discovery",
        "discovery_status": "ready" if candidates and not errors else "blocked",
        "errors": sorted(set(errors)),
        "current_rag_id": str(rag.get("rag_id") or ""),
        "rag_current_json": str(rag_current_json),
        "rag_idx": str(rag.get("idx") or ""),
        "rag_idx_exists": bool(rag.get("idx_exists")),
        "scanned_manifest_ref_count": len(manifest_refs),
        "scanned_chunk_count": len(chunks),
        "scanned_file_count": len(files),
        "validation_ref_count": len(validation_refs),
        "candidate_count": len(candidates),
        "blocked_ref_count": len(blocked_rows),
        "rag_search_order": RAG_SEARCH_ORDER,
        "outputs": {
            "candidates_jsonl": str(output_dir / CANDIDATES_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "summary_md": str(output_dir / SUMMARY_FILE),
            "progress_jsonl": str(progress_path),
        },
        "presentation_status": "audio_quality_promotion_candidate_discovery_not_for_listen_pack",
        "promotion_allowed": False,
        "runtime_authority_enabled": False,
        "safety": {
            "current_rag_only": True,
            "promotion_decision_not_run": True,
            "chunk_sha_backed_refs_required": True,
            "not_delivery_artifact": True,
            "not_release_artifact": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    (output_dir / SUMMARY_FILE).write_text(_summary_text(manifest), encoding="utf-8")
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        discovery_status=manifest["discovery_status"],
        candidate_count=len(candidates),
        blocked_ref_count=len(blocked_rows),
        promotion_allowed=False,
        runtime_authority_enabled=False,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover current-RAG audio quality critic promotion decision candidates.")
    parser.add_argument("--rag-current-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="discover_audio_quality_promotion_candidates.py", entrypoint_path="/app/tools/discover_audio_quality_promotion_candidates.py")
    args = parse_args(argv)
    manifest = discover_audio_quality_promotion_candidates(rag_current_json=args.rag_current_json, output_dir=args.output_dir)
    print(json.dumps({"manifest_json": str(args.output_dir / MANIFEST_FILE), "discovery_status": manifest["discovery_status"], "candidate_count": manifest["candidate_count"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
