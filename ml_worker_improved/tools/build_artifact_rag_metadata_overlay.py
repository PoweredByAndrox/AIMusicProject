#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for import_root in (APP_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from docker_runtime import require_docker_runtime  # noqa: E402


SCHEMA = "artifact_rag_pack.v1"
OVERLAY_SCHEMA = "artifact_rag_metadata_overlay.v1"
INDEX_SCHEMA = "artifact_rag_index.v2"
IDX_NAME = "idx.json"
SUMMARY_NAME = "sum.md"
ROOTS_NAME = "roots.jsonl"
FILES_NAME = "files.jsonl"
MANIFEST_REFS_NAME = "manifest_refs.jsonl"
CHUNKS_NAME = "chunks.jsonl"
TEXT_REFS_NAME = "text_refs.zip"
CLEANUP_NAME = "cleanup.json"
PATH_MAP_NAME = "path_map.jsonl"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()

TEXT_EXTENSIONS = {".csv", ".json", ".jsonl", ".log", ".m3u", ".md", ".ps1", ".sh", ".txt", ".yaml", ".yml"}
STATUS_KEYS = {
    "schema",
    "manifest_type",
    "delivery_status",
    "delivery_allowed",
    "presentation_status",
    "release_ready",
    "can_publish_as_result",
    "promotion_allowed",
    "runtime_authority_enabled",
    "discovery_status",
    "decision_status",
    "promotion_decision",
    "build_status",
    "errors",
    "failed_checks",
    "candidate_count",
    "blocked_ref_count",
    "ready_for_promotion_decision",
    "blocked_reasons",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "artifact_rag_metadata_overlay_no_model_inference",
        "expected_acceleration_mode": "cpu_filesystem_metadata_only",
        "delivery_allowed": False,
        "release_ready": False,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _relative_to(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        return resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label}_must_be_inside_artifact_rag_root") from exc


def _validate_short_relative_path(relative: Path, *, label: str) -> None:
    for part in relative.parts:
        if part in ("", ".", ".."):
            raise ValueError(f"{label}_has_invalid_segment")
        if len(part) > 64:
            raise ValueError(f"{label}_segment_too_long")


def _load_base_pack(rag_current_json: Path, *, base_rag_id: str | None = None) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    current = read_json(rag_current_json)
    if base_rag_id:
        base_id = str(base_rag_id).strip()
        _validate_short_relative_path(Path(base_id), label="base_rag_id")
        if len(Path(base_id).parts) != 1:
            raise ValueError("base_rag_id_must_be_single_segment")
        idx_rel = f"{base_id}/{IDX_NAME}"
        current = {**current, "rag_id": base_id, "idx": idx_rel}
    else:
        idx_rel = str(current.get("idx") or "")
    if not idx_rel:
        raise ValueError("current_idx_missing")
    idx_path = rag_current_json.parent / idx_rel
    idx = read_json(idx_path)
    pack_dir = idx_path.parent
    outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
    tables = {
        "roots": list(iter_jsonl(pack_dir / str(outputs.get("roots") or ROOTS_NAME))),
        "files": list(iter_jsonl(pack_dir / str(outputs.get("files") or FILES_NAME))),
        "manifest_refs": list(iter_jsonl(pack_dir / str(outputs.get("manifest_refs") or MANIFEST_REFS_NAME))),
        "chunks": list(iter_jsonl(pack_dir / str(outputs.get("chunks") or CHUNKS_NAME))),
        "path_map": list(iter_jsonl(pack_dir / str(outputs.get("path_map") or PATH_MAP_NAME))),
    }
    return pack_dir, current, idx, tables


def _summary_int(idx: dict[str, Any], key: str) -> int | None:
    summary = idx.get("summary") if isinstance(idx.get("summary"), dict) else {}
    value = summary.get(key)
    if isinstance(value, int):
        return value
    return None


def _validate_base_counts(idx: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    checks = {
        "root_count": len(tables["roots"]),
        "file_count": len(tables["files"]),
        "manifest_ref_count": len(tables["manifest_refs"]),
        "chunk_count": len(tables["chunks"]),
    }
    for key, actual in checks.items():
        declared = _summary_int(idx, key)
        if declared is not None and declared != actual:
            raise ValueError(f"base_{key}_mismatch: declared={declared} actual={actual}")


def _next_root_alias(roots: Sequence[dict[str, Any]]) -> int:
    max_alias = 0
    for row in roots:
        alias = str(row.get("root_alias") or "")
        if alias.startswith("r") and alias[1:].isdigit():
            max_alias = max(max_alias, int(alias[1:]))
    return max_alias + 1


def _read_text(path: Path, max_text_bytes: int) -> tuple[bytes, str] | None:
    data = path.read_bytes()
    if len(data) > max_text_bytes:
        return None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return data, text


def _extract_json_status(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    for key in STATUS_KEYS:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            result[key] = value
    return result


def _extract_status_fields(path: Path, text: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return _extract_json_status(json.loads(text))
        except json.JSONDecodeError:
            return {}
    if suffix == ".jsonl":
        sample_rows: list[dict[str, Any]] = []
        for raw_line in text.splitlines()[:20]:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = _extract_json_status(row)
            if status:
                sample_rows.append(status)
        return {"sample_rows": sample_rows} if sample_rows else {}
    return {}


def _chunks_for_text(*, root: str, root_alias: str, relative_path: str, status_fields: dict[str, Any], text: str, chunk_chars: int, start_index: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for offset, start in enumerate(range(0, max(len(text), 1), chunk_chars)):
        index = start_index + offset
        chunk_text = text[start : start + chunk_chars]
        chunks.append(
            {
                "root": root,
                "root_alias": root_alias,
                "relative_path": relative_path,
                "chunk_id": f"{root_alias}_{index:06d}",
                "chunk_index": index,
                "status_fields": status_fields,
                "text": chunk_text,
            }
        )
    return chunks


def _scan_metadata_root(
    *,
    artifact_rag_root: Path,
    metadata_root: Path,
    root_alias: str,
    max_text_bytes: int,
    chunk_chars: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    relative_root = _relative_to(metadata_root, artifact_rag_root, label="include_root")
    _validate_short_relative_path(relative_root, label="include_root")
    root_name = relative_root.as_posix()
    files: list[dict[str, Any]] = []
    manifest_refs: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    path_map: list[dict[str, Any]] = []
    zip_entries: dict[str, bytes] = {}
    total_bytes = 0
    text_snapshot_files = 0
    next_chunk_index = 1
    for file_path in sorted(path for path in metadata_root.rglob("*") if path.is_file()):
        rel_file = file_path.relative_to(metadata_root).as_posix()
        _validate_short_relative_path(Path(rel_file), label="metadata_file")
        data = file_path.read_bytes()
        total_bytes += len(data)
        text_snapshot = file_path.suffix.lower() in TEXT_EXTENSIONS
        text_sha = ""
        zip_member = ""
        status_fields: dict[str, Any] = {}
        text = ""
        if text_snapshot:
            text_payload = _read_text(file_path, max_text_bytes=max_text_bytes)
            if text_payload is not None:
                snapshot_bytes, text = text_payload
                text_sha = sha256_bytes(snapshot_bytes)
                zip_member = f"{root_alias}/{rel_file}"
                zip_entries[zip_member] = snapshot_bytes
                text_snapshot_files += 1
                status_fields = _extract_status_fields(file_path, text)
                manifest_refs.append(
                    {
                        "root": root_name,
                        "root_alias": root_alias,
                        "relative_path": rel_file,
                        "sha256": text_sha,
                        "zip_member": zip_member,
                        "status_fields": status_fields,
                        "excerpt": text[:1600],
                    }
                )
                file_chunks = _chunks_for_text(
                    root=root_name,
                    root_alias=root_alias,
                    relative_path=rel_file,
                    status_fields=status_fields,
                    text=text,
                    chunk_chars=chunk_chars,
                    start_index=next_chunk_index,
                )
                next_chunk_index += len(file_chunks)
                chunks.extend(file_chunks)
        files.append(
            {
                "root": root_name,
                "root_alias": root_alias,
                "relative_path": rel_file,
                "original_path": str(file_path),
                "bytes": len(data),
                "extension": file_path.suffix.lower(),
                "text_snapshot": bool(zip_member),
                "text_sha256": text_sha,
                "text_zip_member": zip_member,
            }
        )
        path_map.append(
            {
                "root": root_name,
                "root_alias": root_alias,
                "relative_path": rel_file,
                "original_path": str(file_path),
                "retention_mode": "metadata_overlay",
                "binary_bytes_available": True,
            }
        )
    root_row = {
        "root": root_name,
        "root_alias": root_alias,
        "requested": root_name,
        "original_path": str(metadata_root),
        "bytes": total_bytes,
        "gib": round(total_bytes / 1024**3, 3),
        "files": len(files),
        "text_snapshot_files": text_snapshot_files,
    }
    return root_row, files, manifest_refs, chunks, path_map, zip_entries


def _copy_base_text_refs(base_pack_dir: Path, output_dir: Path, zip_entries: dict[str, bytes]) -> None:
    output_zip = output_dir / TEXT_REFS_NAME
    base_zip = base_pack_dir / TEXT_REFS_NAME
    if base_zip.exists():
        shutil.copyfile(base_zip, output_zip)
        mode = "a"
    else:
        mode = "w"
    with zipfile.ZipFile(output_zip, mode, compression=zipfile.ZIP_DEFLATED) as archive:
        existing = set(archive.namelist())
        for name, data in zip_entries.items():
            if name not in existing:
                archive.writestr(name, data)


def _write_summary(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        f"# Artifact RAG {manifest['rag_id']}",
        "",
        f"- schema: `{manifest['schema']}`",
        f"- overlay_schema: `{manifest['overlay_schema']}`",
        f"- base_rag_id: `{manifest['base_rag_id']}`",
        f"- roots: `{summary['root_count']}`",
        f"- base_roots: `{summary['base_root_count']}`",
        f"- overlay_roots: `{summary['overlay_root_count']}`",
        "- delivery_allowed: `false`",
        "- release_ready: `false`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _update_current_pointer(rag_current_json: Path, output_dir: Path, manifest: dict[str, Any]) -> None:
    payload = {
        "schema": "artifact_rag_current.v1",
        "rag_id": manifest["rag_id"],
        "status": manifest.get("status", ""),
        "current_retention_mode": manifest.get("current_retention_mode", ""),
        "binary_audio_model_bytes_available": manifest.get("binary_audio_model_bytes_available", True),
        "idx": f"{output_dir.name}/{IDX_NAME}",
        "summary": f"{output_dir.name}/{SUMMARY_NAME}",
        "updated_at_epoch": int(time.time()),
        "query_hint": manifest["query_hint"],
    }
    write_json(rag_current_json, payload)


def build_artifact_rag_metadata_overlay(
    *,
    rag_current_json: Path,
    output_dir: Path,
    include_roots: Sequence[Path],
    base_rag_id: str | None = None,
    update_current: bool = True,
    max_text_bytes: int = 2_000_000,
    chunk_chars: int = 2000,
) -> dict[str, Any]:
    artifact_rag_root = rag_current_json.parent.resolve()
    output_relative = _relative_to(output_dir, artifact_rag_root, label="output_dir")
    if len(output_relative.parts) != 1:
        raise ValueError("output_dir_must_be_direct_child_of_artifact_rag_root")
    _validate_short_relative_path(output_relative, label="output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    append_progress(progress_path, "start", rag_current_json=str(rag_current_json), output_dir=str(output_dir), include_root_count=len(include_roots))

    base_pack_dir, current, base_idx, base_tables = _load_base_pack(rag_current_json, base_rag_id=base_rag_id)
    if output_dir.resolve() == base_pack_dir.resolve():
        raise ValueError("output_dir_must_not_equal_base_pack_dir")
    _validate_base_counts(base_idx, base_tables)

    roots = list(base_tables["roots"])
    files = list(base_tables["files"])
    manifest_refs = list(base_tables["manifest_refs"])
    chunks = list(base_tables["chunks"])
    path_map = list(base_tables["path_map"])
    zip_entries: dict[str, bytes] = {}
    alias_index = _next_root_alias(roots)
    overlay_roots: list[dict[str, Any]] = []
    metadata_delivery_allowed = False
    for include_root in include_roots:
        root_path = include_root.resolve()
        _relative_to(root_path, artifact_rag_root, label="include_root")
        if root_path == output_dir.resolve() or output_dir.resolve() in root_path.parents:
            raise ValueError("include_root_must_not_be_output_dir")
        if root_path == base_pack_dir.resolve() or base_pack_dir.resolve() in root_path.parents:
            raise ValueError("include_root_must_not_be_base_pack_dir")
        root_alias = f"r{alias_index:03d}"
        alias_index += 1
        root_row, file_rows, ref_rows, chunk_rows, path_rows, root_zip_entries = _scan_metadata_root(
            artifact_rag_root=artifact_rag_root,
            metadata_root=root_path,
            root_alias=root_alias,
            max_text_bytes=max_text_bytes,
            chunk_chars=chunk_chars,
        )
        for row in ref_rows:
            status = row.get("status_fields") if isinstance(row.get("status_fields"), dict) else {}
            if status.get("delivery_allowed") is True or status.get("release_ready") is True or status.get("can_publish_as_result") is True:
                metadata_delivery_allowed = True
        roots.append(root_row)
        files.extend(file_rows)
        manifest_refs.extend(ref_rows)
        chunks.extend(chunk_rows)
        path_map.extend(path_rows)
        zip_entries.update(root_zip_entries)
        overlay_roots.append(root_row)

    base_summary = base_idx.get("summary") if isinstance(base_idx.get("summary"), dict) else {}
    source_bytes = sum(int(row.get("bytes") or 0) for row in roots)
    text_snapshot_source_bytes = sum(int(row.get("bytes") or 0) for row in files if row.get("text_snapshot"))
    manifest = {
        "schema": SCHEMA,
        "overlay_schema": OVERLAY_SCHEMA,
        "index_schema": INDEX_SCHEMA,
        "rag_id": output_dir.name,
        "base_rag_id": str(base_idx.get("rag_id") or current.get("rag_id") or ""),
        "base_idx": str(current.get("idx") or ""),
        "fingerprint": stable_hash({"base": current.get("idx"), "include_roots": [row["root"] for row in overlay_roots], "counts": [len(roots), len(files), len(manifest_refs), len(chunks)]}),
        "artifact_rag_root": str(artifact_rag_root),
        "root": str(output_dir),
        "summary": {
            "base_root_count": len(base_tables["roots"]),
            "base_file_count": len(base_tables["files"]),
            "base_manifest_ref_count": len(base_tables["manifest_refs"]),
            "base_chunk_count": len(base_tables["chunks"]),
            "overlay_root_count": len(overlay_roots),
            "root_count": len(roots),
            "file_count": len(files),
            "manifest_ref_count": len(manifest_refs),
            "chunk_count": len(chunks),
            "source_bytes": source_bytes,
            "source_gib": round(source_bytes / 1024**3, 3),
            "text_snapshot_files": sum(1 for row in files if row.get("text_snapshot")),
            "text_snapshot_source_bytes": text_snapshot_source_bytes,
            "cleanup_mode": "none",
            "cleanup_executed": False,
            "planned_roots": len(overlay_roots),
            "deleted_roots": 0,
            "thin_deleted_roots": 0,
            "moved_roots": 0,
            "dry_run_roots": 0,
            "cleanup_status": "metadata_overlay_no_cleanup",
        },
        "roots": roots,
        "overlay_roots": overlay_roots,
        "outputs": {
            "idx": IDX_NAME,
            "summary": SUMMARY_NAME,
            "roots": ROOTS_NAME,
            "files": FILES_NAME,
            "manifest_refs": MANIFEST_REFS_NAME,
            "chunks": CHUNKS_NAME,
            "text_refs_zip": TEXT_REFS_NAME,
            "cleanup": CLEANUP_NAME,
            "path_map": PATH_MAP_NAME,
            "progress": PROGRESS_NAME,
        },
        "retention_contract": {
            **(base_idx.get("retention_contract") if isinstance(base_idx.get("retention_contract"), dict) else {}),
            "llm_retrieval_complete_for_textual_evidence": True,
            "metadata_overlay_preserves_base_pack": True,
            "listen_delivery_artifact": False,
            "delivery_allowed": False,
            "release_ready": False,
        },
        "overlay_safety": {
            "base_coverage_preserved": (
                len(roots) >= len(base_tables["roots"])
                and len(files) >= len(base_tables["files"])
                and len(manifest_refs) >= len(base_tables["manifest_refs"])
                and len(chunks) >= len(base_tables["chunks"])
            ),
            "metadata_roots_delivery_allowed": metadata_delivery_allowed,
            "promotion_not_fabricated": True,
            "base_current_replaced_by_overlay_only_after_preservation_check": True,
        },
        "query_hint": (
            "Use artifact_rag/current.json -> idx.json, then chunks.jsonl/manifest_refs.jsonl. "
            "This pack is a metadata overlay that preserves the base RAG tables and adds small artifact_rag evidence roots."
        ),
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "model_registry_edited": False,
        "runtime_gates_changed": False,
        "thresholds_relaxed": False,
    }
    if not manifest["overlay_safety"]["base_coverage_preserved"]:
        raise ValueError("base_coverage_not_preserved")
    if metadata_delivery_allowed:
        raise ValueError("metadata_root_delivery_or_release_flag_true")

    write_jsonl(output_dir / ROOTS_NAME, roots)
    write_jsonl(output_dir / FILES_NAME, files)
    write_jsonl(output_dir / MANIFEST_REFS_NAME, manifest_refs)
    write_jsonl(output_dir / CHUNKS_NAME, chunks)
    write_jsonl(output_dir / PATH_MAP_NAME, path_map)
    _copy_base_text_refs(base_pack_dir, output_dir, zip_entries)
    write_json(
        output_dir / CLEANUP_NAME,
        {
            "schema": "artifact_rag_cleanup.v1",
            "rag_id": output_dir.name,
            "cleanup_summary": manifest["summary"],
            "roots": overlay_roots,
            "path_map_rows": [row for row in path_map if row.get("retention_mode") == "metadata_overlay"],
        },
    )
    write_json(output_dir / IDX_NAME, manifest)
    _write_summary(output_dir / SUMMARY_NAME, manifest)
    if update_current:
        _update_current_pointer(rag_current_json, output_dir, manifest)
    append_progress(
        progress_path,
        "finish",
        base_rag_id=manifest["base_rag_id"],
        rag_id=manifest["rag_id"],
        root_count=manifest["summary"]["root_count"],
        file_count=manifest["summary"]["file_count"],
        manifest_ref_count=manifest["summary"]["manifest_ref_count"],
        chunk_count=manifest["summary"]["chunk_count"],
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a safe artifact_rag metadata overlay pack without dropping base RAG coverage.")
    parser.add_argument("--rag-current-json", type=Path, default=Path("/app/artifact_rag/current.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-root", action="append", type=Path, required=True)
    parser.add_argument("--base-rag-id")
    parser.add_argument("--no-current", action="store_true")
    parser.add_argument("--max-text-bytes", type=int, default=2_000_000)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_artifact_rag_metadata_overlay.py", entrypoint_path="/app/tools/build_artifact_rag_metadata_overlay.py")
    args = parse_args(argv)
    manifest = build_artifact_rag_metadata_overlay(
        rag_current_json=args.rag_current_json,
        output_dir=args.output_dir,
        include_roots=args.include_root,
        base_rag_id=args.base_rag_id,
        update_current=not bool(args.no_current),
        max_text_bytes=int(args.max_text_bytes),
        chunk_chars=int(args.chunk_chars),
    )
    print(json.dumps({"rag_id": manifest["rag_id"], "idx": str(args.output_dir / IDX_NAME), "summary": manifest["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
