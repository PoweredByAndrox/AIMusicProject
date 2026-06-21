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

from tools.audio_quality_critic_common import (
    BLOCKED_FILE,
    MANIFEST_FILE,
    PROGRESS_FILE,
    SAFETY_FLAGS,
    append_progress,
    iter_jsonl,
    load_rag_current,
    read_json,
    stable_hash,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime


ROWS_FILE = "rows.jsonl"
MANIFEST_SCHEMA = "audio_quality_verified_corpus_discovery_manifest.v1"
VERIFIED_CORPUS_SCHEMA = "verified_audio_training_corpus_manifest.v1"
SOURCE_LOCK_SCHEMA = "audio_quality_source_lock.v1"
SELECTED_ACTION = "run_build_audio_quality_verified_rows"
BLOCKED_TRUE_FLAGS = (
    "delivery_allowed",
    "release_ready",
    "can_publish_as_result",
    "thresholds_relaxed",
    "model_registry_edited",
    "runtime_gates_changed",
)


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    try:
        if path.exists():
            text = str(path.resolve())
    except OSError:
        pass
    return text.replace("\\", "/").lower()


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("")
    path = Path(text)
    return path if path.is_absolute() else base_dir / path


def _manifest_flag_reasons(prefix: str, payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            reasons.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                reasons.append(f"{prefix}_safety_{key}_true")
    return reasons


def _source_lock_reasons(source_lock_pack_json: Path, *, rag_current_json: Path) -> list[str]:
    if not str(source_lock_pack_json) or not source_lock_pack_json.exists():
        return ["source_lock_pack_json_missing"]
    try:
        source_lock = read_json(source_lock_pack_json)
    except Exception as exc:  # noqa: BLE001
        return ["source_lock_pack_json_read_failed:" + type(exc).__name__]
    reasons: list[str] = []
    if source_lock.get("schema") != SOURCE_LOCK_SCHEMA:
        reasons.append("source_lock_schema_unexpected")
    reasons.extend(_manifest_flag_reasons("source_lock", source_lock))
    if source_lock.get("rag_idx_exists") is not True:
        reasons.append("source_lock_rag_idx_exists_not_true")
    rag = load_rag_current(rag_current_json)
    lock_rag_id = str(source_lock.get("current_rag_id") or "").strip()
    if not lock_rag_id:
        reasons.append("source_lock_current_rag_missing")
    elif lock_rag_id != str(rag.get("rag_id") or ""):
        reasons.append("source_lock_current_rag_mismatch")
    return _unique(reasons)


def _rag_tables(
    rag_current_json: Path,
) -> tuple[dict[str, Any], Path, dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rag = load_rag_current(rag_current_json)
    if not rag.get("idx_exists"):
        return rag, Path(""), {}, [], [], []
    idx_path = rag_current_json.parent / str(rag.get("idx") or "")
    idx = read_json(idx_path)
    idx_dir = idx_path.parent
    outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
    roots = _load_jsonl(idx_dir / str(outputs.get("roots") or "roots.jsonl"))
    refs = _load_jsonl(idx_dir / str(outputs.get("manifest_refs") or "manifest_refs.jsonl"))
    chunks = _load_jsonl(idx_dir / str(outputs.get("chunks") or "chunks.jsonl"))
    files = _load_jsonl(idx_dir / str(outputs.get("files") or "files.jsonl"))
    roots_by_alias = {str(row.get("root_alias") or ""): row for row in roots if row.get("root_alias")}
    return rag, idx_dir, roots_by_alias, refs, chunks, files


def _is_verified_corpus_ref(ref: dict[str, Any]) -> bool:
    status = ref.get("status_fields") if isinstance(ref.get("status_fields"), dict) else {}
    if status.get("schema") == VERIFIED_CORPUS_SCHEMA:
        return True
    text = str(ref.get("text") or ref.get("excerpt") or "")
    try:
        payload = json.loads(text)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(payload, dict) and payload.get("schema") == VERIFIED_CORPUS_SCHEMA


def _rag_ref(ref: dict[str, Any], rag: dict[str, Any]) -> dict[str, Any]:
    return {
        "rag_id": str(rag.get("rag_id") or ""),
        "idx": str(rag.get("idx") or ""),
        "idx_exists": bool(rag.get("idx_exists")),
        "ref_type": "verified_corpus_manifest_chunk_ref",
        "source_tables": _unique(ref.get("source_tables") or [ref.get("source_table")]),
        "root": str(ref.get("root") or ""),
        "root_alias": str(ref.get("root_alias") or ""),
        "relative_path": str(ref.get("relative_path") or ""),
        "chunk_id": str(ref.get("chunk_id") or ""),
        "sha256": str(ref.get("sha256") or ref.get("text_sha256") or ""),
        "zip_member": str(ref.get("zip_member") or ref.get("text_zip_member") or ""),
    }


def _location_key(row: dict[str, Any]) -> str:
    root_alias = str(row.get("root_alias") or "").strip().lower()
    relative_path = str(row.get("relative_path") or "").strip().replace("\\", "/").lower()
    if not root_alias or not relative_path:
        return ""
    return f"{root_alias}\0{relative_path}"


def _identity_key(row: dict[str, Any]) -> str:
    path_key = _path_key(row.get("original_path") or row.get("source_manifest_json"))
    if path_key:
        return f"path:{path_key}"
    location_key = _location_key(row)
    if location_key:
        return f"loc:{location_key}"
    return "chunk:" + str(row.get("chunk_id") or stable_hash(row)[:16])


def _files_by_location(files: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
        if key == "status_fields" and isinstance(value, dict):
            status = merged.get("status_fields") if isinstance(merged.get("status_fields"), dict) else {}
            merged[key] = {**status, **value}
            continue
        if merged.get(key) in ("", None, [], {}) and value not in ("", None, [], {}):
            merged[key] = value
    return merged


def _discovery_ref(row: dict[str, Any], *, source_table: str, file_by_location: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref = dict(row)
    ref["source_tables"] = _unique([*(ref.get("source_tables") or []), source_table])
    file_row = file_by_location.get(_location_key(ref))
    if file_row:
        ref["source_tables"] = _unique([*ref["source_tables"], "files"])
        for target_key, source_key in (
            ("original_path", "original_path"),
            ("sha256", "text_sha256"),
            ("zip_member", "text_zip_member"),
        ):
            if not ref.get(target_key) and file_row.get(source_key):
                ref[target_key] = file_row[source_key]
    return ref


def _verified_refs_from_rag_tables(
    *,
    manifest_refs: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    files: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    file_by_location = _files_by_location(files)
    collected: dict[str, dict[str, Any]] = {}
    for source_table, rows in (("manifest_refs", manifest_refs), ("chunks", chunks)):
        for row in rows:
            if not _is_verified_corpus_ref(row):
                continue
            ref = _discovery_ref(row, source_table=source_table, file_by_location=file_by_location)
            key = _identity_key(ref)
            if key in collected:
                collected[key] = _merge_ref(collected[key], ref)
            else:
                collected[key] = ref
    return list(collected.values())


def _candidate_manifest_paths(ref: dict[str, Any], roots_by_alias: dict[str, dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for key in ("original_path", "source_manifest_json"):
        value = str(ref.get(key) or "").strip()
        if value:
            paths.append(Path(value))
    root_alias = str(ref.get("root_alias") or "").strip()
    relative_path = str(ref.get("relative_path") or "").strip()
    root = roots_by_alias.get(root_alias)
    if root and relative_path:
        original_root = str(root.get("original_path") or "").strip()
        if original_root:
            paths.append(Path(original_root) / relative_path)
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _path_key(path)
        if key and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _existing_manifest_path(ref: dict[str, Any], roots_by_alias: dict[str, dict[str, Any]]) -> Path | None:
    for path in _candidate_manifest_paths(ref, roots_by_alias):
        if path.exists():
            return path
    return None


def _corpus_reasons(manifest_path: Path, ref: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    rag_ref = _rag_ref(ref, {"rag_id": "", "idx": "", "idx_exists": True})
    if not rag_ref["chunk_id"]:
        reasons.append("rag_ref_chunk_id_missing")
    if not rag_ref["sha256"]:
        reasons.append("rag_ref_sha256_missing")
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:  # noqa: BLE001
        return _unique([*reasons, "verified_manifest_read_failed:" + type(exc).__name__])
    if manifest.get("schema") != VERIFIED_CORPUS_SCHEMA:
        reasons.append("verified_manifest_schema_unexpected")
    if manifest.get("corpus_validation_status") != "passed":
        reasons.append("corpus_validation_not_passed")
    if manifest.get("failed_checks") not in ([], None):
        reasons.append("corpus_failed_checks_not_empty")
    reasons.extend(_manifest_flag_reasons("verified_corpus_manifest", manifest))
    base_dir = manifest_path.parent
    examples_jsonl = _resolve_path(manifest.get("audio_training_examples_jsonl"), base_dir=base_dir)
    report_jsonl = _resolve_path(manifest.get("row_report_jsonl"), base_dir=base_dir)
    if not str(examples_jsonl) or not examples_jsonl.exists():
        reasons.append("audio_training_examples_jsonl_missing")
    if not str(report_jsonl) or not report_jsonl.exists():
        reasons.append("row_report_jsonl_missing")
    return _unique(reasons)


def _ready_row(
    *,
    ref: dict[str, Any],
    rag: dict[str, Any],
    manifest_path: Path,
    source_lock_pack_json: Path,
    row_index: int,
) -> dict[str, Any]:
    return {
        "schema": "audio_quality_verified_corpus_discovery_row.v1",
        "row_id": "vcd_" + stable_hash([str(manifest_path), ref.get("chunk_id"), row_index])[:12],
        "ready_for_verified_row_intake": True,
        "selected_action": SELECTED_ACTION,
        "verified_corpus_manifest_json": str(manifest_path),
        "source_lock_pack_json": str(source_lock_pack_json),
        "rag_refs": [_rag_ref(ref, rag)],
        "source_manifest_refs": [str(manifest_path)],
        "blocked_reasons": [],
        "delivery_allowed": False,
        "release_ready": False,
    }


def _blocked_row(ref: dict[str, Any], reasons: Sequence[str], *, manifest_path: Path | None = None) -> dict[str, Any]:
    return {
        "schema": "audio_quality_verified_corpus_discovery_blocked_row.v1",
        "verified_corpus_manifest_json": str(manifest_path or ""),
        "root_alias": str(ref.get("root_alias") or ""),
        "relative_path": str(ref.get("relative_path") or ""),
        "chunk_id": str(ref.get("chunk_id") or ""),
        "sha256": str(ref.get("sha256") or ""),
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        "ready_for_verified_row_intake": False,
        "delivery_allowed": False,
        "release_ready": False,
    }


def discover_audio_quality_verified_corpora(
    *,
    rag_current_json: Path,
    source_lock_pack_json: Path,
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
        rag_current_json=str(rag_current_json),
        source_lock_pack_json=str(source_lock_pack_json),
        expected_acceleration_mode="external_only_cpu_rag_metadata_discovery",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="verified_corpus_discovery_no_model_inference",
    )

    rag, _idx_dir, roots_by_alias, manifest_refs, chunks, files = _rag_tables(rag_current_json)
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
        rag_search_order=["current", "idx", "manifest_refs", "chunks", "files"],
    )
    source_lock_errors = _source_lock_reasons(source_lock_pack_json, rag_current_json=rag_current_json)
    verified_refs = _verified_refs_from_rag_tables(manifest_refs=manifest_refs, chunks=chunks, files=files)
    append_progress(
        progress_path,
        "classify_verified_refs",
        start_time=start_time,
        verified_corpus_ref_count=len(verified_refs),
        source_lock_error_count=len(source_lock_errors),
    )
    ready_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    if not verified_refs:
        blocked_rows.append(_blocked_row({}, ["verified_corpus_manifest_not_found_in_current_rag"]))

    for ref in verified_refs:
        manifest_path = _existing_manifest_path(ref, roots_by_alias)
        reasons = list(source_lock_errors)
        if manifest_path is None:
            reasons.append("verified_manifest_file_not_available")
        else:
            reasons.extend(_corpus_reasons(manifest_path, ref))
        reasons = _unique(reasons)
        if reasons:
            blocked_rows.append(_blocked_row(ref, reasons, manifest_path=manifest_path))
            continue
        ready_rows.append(
            _ready_row(
                ref=ref,
                rag=rag,
                manifest_path=manifest_path,
                source_lock_pack_json=source_lock_pack_json,
                row_index=len(ready_rows) + 1,
            )
        )

    errors: list[str] = []
    if not verified_refs:
        errors.append("verified_corpus_manifest_not_found_in_current_rag")
    if verified_refs and not ready_rows:
        errors.append("no_ready_verified_corpus_refs")
    if source_lock_errors:
        errors.extend(source_lock_errors)

    write_jsonl(output_dir / ROWS_FILE, ready_rows)
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_verified_corpus_discovery",
        "build_status": "passed" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "current_rag_id": str(rag.get("rag_id") or ""),
        "rag_current_json": str(rag_current_json),
        "rag_idx": str(rag.get("idx") or ""),
        "rag_idx_exists": bool(rag.get("idx_exists")),
        "source_lock_pack_json": str(source_lock_pack_json),
        "verified_corpus_ref_count": len(verified_refs),
        "scanned_manifest_ref_count": len(manifest_refs),
        "scanned_chunk_count": len(chunks),
        "scanned_file_count": len(files),
        "rag_search_order": ["current", "idx", "manifest_refs", "chunks", "files"],
        "ready_count": len(ready_rows),
        "blocked_ref_count": len(blocked_rows),
        "outputs": {
            "rows_jsonl": str(output_dir / ROWS_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
        },
        "presentation_status": "audio_quality_verified_corpus_discovery_not_for_listen_pack",
        "training_started": False,
        "promotion_allowed": False,
        "safety": {
            "current_rag_only": True,
            "chunk_sha_backed_refs_required": True,
            "source_lock_required": True,
            "not_release_artifact": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        build_status=manifest["build_status"],
        verified_corpus_ref_count=len(verified_refs),
        ready_count=len(ready_rows),
        blocked_ref_count=len(blocked_rows),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover current-RAG verified corpus manifests for AQC verified-row intake.")
    parser.add_argument("--rag-current-json", type=Path, required=True)
    parser.add_argument("--source-lock-pack-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="discover_audio_quality_verified_corpora.py",
        entrypoint_path="/app/tools/discover_audio_quality_verified_corpora.py",
    )
    args = parse_args(argv)
    manifest = discover_audio_quality_verified_corpora(
        rag_current_json=args.rag_current_json,
        source_lock_pack_json=args.source_lock_pack_json,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "build_status": manifest["build_status"],
                "ready_count": manifest["ready_count"],
                "errors": manifest["errors"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["build_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
