#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for import_root in (APP_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from docker_runtime import require_docker_runtime  # noqa: E402
from tools.artifact_name_policy import indexed_alias, validate_artifact_segment  # noqa: E402

SCHEMA = "artifact_rag_pack.v1"
INDEX_SCHEMA = "artifact_rag_index.v2"
START_TIME = time.monotonic()

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
PRUNE_NAME = "prune.json"
DEFAULT_ARTIFACT_RAG_MOUNT_ROOT = Path("/app/artifact_rag")
MOVE_PROGRESS_BYTES = 256 * 1024 * 1024
MOVE_PROGRESS_FILES = 250
SCAN_PROGRESS_BYTES = 5 * 1024 * 1024 * 1024
SCAN_PROGRESS_FILES = 1000

TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".m3u",
    ".md",
    ".ps1",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
MANIFEST_NAME_MARKERS = (
    "manifest",
    "judge",
    "matrix",
    "gate",
    "plan",
    "queue",
    "summary",
    "progress",
    "report",
    "validation",
)
STATUS_KEYS = (
    "schema",
    "manifest_type",
    "track_id",
    "stem_type",
    "candidate_id",
    "route_id",
    "route_allowed",
    "route_block_reason",
    "delivery_status",
    "delivery_allowed",
    "presentation_status",
    "release_status",
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "dominant_failure_reason",
    "failed_checks",
    "blockers",
    "summary",
)


@dataclass(frozen=True)
class RootSpec:
    requested: str
    name: str
    alias: str
    path: Path


@dataclass(frozen=True)
class TextSnapshot:
    zip_member: str
    sha256: str
    data: bytes
    text: str


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def mark(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "elapsed_sec": round(time.monotonic() - START_TIME, 3),
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "artifact_rag_storage_management_only",
            "expected_acceleration_mode": "cpu_filesystem_metadata_only",
        }
        payload.update(fields)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


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
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_output_mount_contract(
    *,
    output_dir: Path,
    cleanup_mode: str,
    execute_cleanup: bool,
    artifact_rag_mount_root: Path = DEFAULT_ARTIFACT_RAG_MOUNT_ROOT,
    mount_checker: Any | None = None,
) -> None:
    if not execute_cleanup or cleanup_mode == "none":
        return
    resolved_output = output_dir.resolve()
    resolved_mount_root = artifact_rag_mount_root.resolve()
    if not is_under(resolved_output, resolved_mount_root):
        return
    checker = mount_checker or (lambda path: path.exists() and path.is_mount())
    if not checker(resolved_mount_root):
        raise RuntimeError(
            f"{resolved_mount_root} is not an active Docker mount. "
            "Refusing cleanup because RAG output could be written into the /app source mount. "
            "docker-compose.yml must mount ./artifact_rag:/app/artifact_rag."
        )


def resolve_requested_root(lab_root: Path, requested: str, alias: str) -> RootSpec:
    normalized = str(requested or "").replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("include-root must be non-empty")
    raw_path = Path(normalized)
    if raw_path.is_absolute() or ":" in normalized:
        raise ValueError(f"include-root must be relative to lab_artifacts: {requested}")
    parts = [part for part in normalized.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"include-root cannot contain traversal: {requested}")
    path = (lab_root / Path(*parts)).resolve()
    if not is_under(path, lab_root):
        raise ValueError(f"resolved root is outside lab_artifacts: {requested}")
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"include-root does not exist: {path}")
    validate_artifact_segment(alias)
    return RootSpec(requested=normalized, name=parts[-1], alias=alias, path=path)


def build_root_specs(lab_root: Path, requested_roots: Sequence[str]) -> list[RootSpec]:
    if not requested_roots:
        raise ValueError("at least one --include-root is required")
    seen: set[str] = set()
    specs: list[RootSpec] = []
    for raw in requested_roots:
        normalized = str(raw or "").replace("\\", "/").strip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        specs.append(resolve_requested_root(lab_root, normalized, indexed_alias("r", len(specs) + 1)))
    return specs


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def move_root_to_vault(source: Path, target: Path, *, progress: ProgressWriter, root_name: str, root_alias: str) -> dict[str, Any]:
    try:
        source.rename(target)
        return {"move_strategy": "rename", "copied_files": 0, "copied_bytes": 0}
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EACCES, errno.EPERM}:
            raise

    copied_files = 0
    copied_bytes = 0
    last_report_files = 0
    last_report_bytes = 0
    progress.mark("cleanup_copy_start", root=root_name, root_alias=root_alias, source=str(source), target=str(target))
    target.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        relative_path = path.relative_to(source)
        target_path = target / relative_path
        if path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        size = int(path.stat().st_size)
        shutil.copy2(path, target_path)
        copied_files += 1
        copied_bytes += size
        if copied_files - last_report_files >= MOVE_PROGRESS_FILES or copied_bytes - last_report_bytes >= MOVE_PROGRESS_BYTES:
            progress.mark(
                "cleanup_copy_progress",
                root=root_name,
                root_alias=root_alias,
                copied_files=copied_files,
                copied_bytes=copied_bytes,
            )
            last_report_files = copied_files
            last_report_bytes = copied_bytes
    progress.mark("cleanup_copy_verify", root=root_name, root_alias=root_alias, copied_files=copied_files, copied_bytes=copied_bytes)
    shutil.rmtree(source)
    return {"move_strategy": "copy_then_delete", "copied_files": copied_files, "copied_bytes": copied_bytes}


def directory_stats(path: Path) -> dict[str, int]:
    files = 0
    bytes_total = 0
    if not path.exists():
        return {"files": 0, "bytes": 0}
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            bytes_total += int(file_path.stat().st_size)
        except OSError:
            continue
        files += 1
    return {"files": files, "bytes": bytes_total}


def prune_existing_vault_pack(*, pack_dir: Path, allow_prune_vault: bool = False) -> dict[str, Any]:
    if not allow_prune_vault:
        raise ValueError("vault pruning requires --allow-prune-vault")
    pack_dir = pack_dir.resolve()
    idx_path = pack_dir / IDX_NAME
    cleanup_path = pack_dir / CLEANUP_NAME
    if not idx_path.exists() or not cleanup_path.exists():
        raise FileNotFoundError(f"RAG pack must contain idx.json and cleanup.json: {pack_dir}")
    if pack_dir.name in {"", ".", ".."} or pack_dir.parent.name != "artifact_rag":
        raise ValueError(f"refusing to prune non-artifact_rag pack: {pack_dir}")

    progress = ProgressWriter(pack_dir / PROGRESS_NAME)
    vault_dir = pack_dir / "vault"
    progress.mark("vault_prune_start", pack=str(pack_dir), vault=str(vault_dir))
    existing_prune_path = pack_dir / PRUNE_NAME
    existing_prune = read_json(existing_prune_path) if existing_prune_path.exists() else {}
    stats = directory_stats(vault_dir)
    vault_roots = 0
    if vault_dir.exists():
        vault_roots = sum(1 for item in vault_dir.iterdir() if item.is_dir())
        progress.mark("vault_prune_delete_start", vault=str(vault_dir), vault_roots=vault_roots, **stats)
        shutil.rmtree(vault_dir)
    elif existing_prune:
        vault_roots = int(existing_prune.get("pruned_vault_roots") or 0)
        stats = {
            "files": int(existing_prune.get("pruned_vault_files") or 0),
            "bytes": int(existing_prune.get("pruned_vault_bytes") or 0),
        }
        progress.mark("vault_prune_already_pruned", vault=str(vault_dir), vault_roots=vault_roots, **stats)
    progress.mark("vault_prune_delete_finish", vault=str(vault_dir), vault_roots=vault_roots, **stats)

    path_map_path = pack_dir / PATH_MAP_NAME
    path_map_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path_map_path):
        updated = dict(row)
        updated["historical_vault_path"] = updated.get("vault_path", "")
        updated["vault_path_available"] = False
        updated["binary_bytes_available"] = False
        updated["retention_mode"] = "thin_rag_only_after_vault_prune"
        path_map_rows.append(updated)
    write_jsonl(path_map_path, path_map_rows)

    prune_payload = {
        "schema": "artifact_rag_vault_prune.v1",
        "rag_id": pack_dir.name,
        "status": "vault_pruned",
        "current_retention_mode": "thin_rag_only",
        "binary_audio_model_bytes_available": False,
        "llm_retrieval_complete_for_textual_evidence": True,
        "pruned_vault_roots": vault_roots,
        "pruned_vault_files": stats["files"],
        "pruned_vault_bytes": stats["bytes"],
        "pruned_vault_gib": round(stats["bytes"] / 1024**3, 3),
        "path_map_status": "historical_only_vault_pruned",
        "updated_at_epoch": int(time.time()),
    }
    write_json(pack_dir / PRUNE_NAME, prune_payload)

    idx = read_json(idx_path)
    prior_status = str(idx.get("status") or "")
    idx["prior_status_before_vault_prune"] = prior_status
    idx["status"] = "thin_rag_binary_pruned" if prior_status != "partial_interrupted_recovered" else "partial_interrupted_recovered_thin_pruned"
    idx["current_retention_mode"] = "thin_rag_only"
    idx["path_map_status"] = "historical_only_vault_pruned"
    idx["vault_paths_available"] = False
    idx["binary_audio_model_bytes_available"] = False
    idx["llm_retrieval_complete_for_textual_evidence"] = True
    idx["vault_prune"] = prune_payload
    idx["query_hint"] = (
        "Use current.json -> idx.json, then chunks.jsonl/manifest_refs.jsonl/text_refs.zip. "
        "This pack is thin_rag_only: bulky binary audio/model bytes were pruned; path_map vault paths are historical only."
    )
    outputs = idx.setdefault("outputs", {})
    if isinstance(outputs, dict):
        outputs["prune_json"] = PRUNE_NAME
    summary = idx.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["current_retention_mode"] = "thin_rag_only"
        summary["binary_audio_model_bytes_available"] = False
        summary["vault_paths_available"] = False
        summary["pruned_vault_roots"] = vault_roots
        summary["pruned_vault_files"] = stats["files"]
        summary["pruned_vault_gib"] = round(stats["bytes"] / 1024**3, 3)
    contract = idx.setdefault("retention_contract", {})
    if isinstance(contract, dict):
        contract["binary_audio_model_bytes_available"] = False
        contract["exact_byte_vault_available"] = False
        contract["move_to_vault_preserves_binary_bytes"] = False
        contract["llm_retrieval_complete_for_textual_evidence"] = True
        contract["current_retention_mode"] = "thin_rag_only"
    write_json(idx_path, idx)
    write_summary_md(pack_dir / SUMMARY_NAME, idx)

    cleanup = read_json(cleanup_path)
    cleanup["current_retention_mode"] = "thin_rag_only"
    cleanup["binary_audio_model_bytes_available"] = False
    cleanup["vault_paths_available"] = False
    cleanup["path_map_status"] = "historical_only_vault_pruned"
    cleanup["vault_prune"] = prune_payload
    cleanup_summary = cleanup.setdefault("cleanup_summary", {})
    if isinstance(cleanup_summary, dict):
        cleanup_summary["current_retention_mode"] = "thin_rag_only"
        cleanup_summary["binary_audio_model_bytes_available"] = False
        cleanup_summary["vault_paths_available"] = False
        cleanup_summary["pruned_vault_roots"] = vault_roots
        cleanup_summary["pruned_vault_files"] = stats["files"]
        cleanup_summary["pruned_vault_gib"] = round(stats["bytes"] / 1024**3, 3)
    write_json(cleanup_path, cleanup)
    current_path = pack_dir.parent / "current.json"
    if current_path.exists():
        try:
            current = read_json(current_path)
        except (json.JSONDecodeError, ValueError):
            current = {}
        if current.get("idx") == f"{pack_dir.name}/{IDX_NAME}":
            current.update(
                {
                    "status": idx["status"],
                    "current_retention_mode": "thin_rag_only",
                    "binary_audio_model_bytes_available": False,
                    "summary": f"{pack_dir.name}/{SUMMARY_NAME}",
                    "updated_at_epoch": int(time.time()),
                    "query_hint": idx["query_hint"],
                }
            )
            write_json(current_path, current)
    progress.mark("vault_prune_finish", **prune_payload)
    return prune_payload


def read_text_snapshot(path: Path, *, max_text_bytes: int) -> TextSnapshot | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_text_bytes:
        return None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return TextSnapshot(zip_member="", sha256=sha256_bytes(data), data=data, text=text)


def extract_json_status(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    for key in STATUS_KEYS:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            result[key] = value
    return result


def extract_status_fields(path: Path, text: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return extract_json_status(json.loads(text))
        except json.JSONDecodeError:
            return {}
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for raw_line in text.splitlines()[:20]:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = extract_json_status(payload)
            if status:
                rows.append(status)
        return {"sample_rows": rows[:5]} if rows else {}
    return {}


def chunk_text(text: str, *, chunk_chars: int) -> list[str]:
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    if chunk_chars <= 0:
        return [clean]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(clean):
        chunks.append(clean[cursor : cursor + chunk_chars])
        cursor += chunk_chars
    return chunks or [""]


def should_snapshot_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def should_extract_manifest_ref(path: Path) -> bool:
    lower_name = path.name.lower()
    return path.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".log", ".yaml", ".yml"} and any(
        marker in lower_name for marker in MANIFEST_NAME_MARKERS
    )


def scan_roots(
    *,
    root_specs: Sequence[RootSpec],
    output_dir: Path,
    max_text_bytes: int,
    chunk_chars: int,
    progress: ProgressWriter,
) -> dict[str, Any]:
    file_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    manifest_ref_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    text_snapshot_count = 0
    text_snapshot_source_bytes = 0

    text_zip_path = output_dir / TEXT_REFS_NAME
    with zipfile.ZipFile(text_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as text_zip:
        for root_spec in root_specs:
            root_file_count = 0
            root_bytes = 0
            root_text_refs = 0
            last_report_files = 0
            last_report_bytes = 0
            progress.mark("root_scan_start", root=root_spec.name, root_alias=root_spec.alias, path=str(root_spec.path))
            for file_path in iter_files(root_spec.path):
                rel_path = file_path.relative_to(root_spec.path).as_posix()
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                root_file_count += 1
                root_bytes += int(stat.st_size)
                if (
                    root_file_count - last_report_files >= SCAN_PROGRESS_FILES
                    or root_bytes - last_report_bytes >= SCAN_PROGRESS_BYTES
                ):
                    progress.mark(
                        "root_scan_progress",
                        root=root_spec.name,
                        root_alias=root_spec.alias,
                        files=root_file_count,
                        bytes=root_bytes,
                    )
                    last_report_files = root_file_count
                    last_report_bytes = root_bytes
                zip_member = ""
                text_sha256 = ""
                text_snapshot = None
                status_fields: dict[str, Any] = {}
                if should_snapshot_text(file_path):
                    text_snapshot = read_text_snapshot(file_path, max_text_bytes=max_text_bytes)
                    if text_snapshot is not None:
                        zip_member = f"{root_spec.alias}/{rel_path}"
                        text_zip.writestr(zip_member, text_snapshot.data)
                        text_sha256 = text_snapshot.sha256
                        text_snapshot_count += 1
                        root_text_refs += 1
                        text_snapshot_source_bytes += int(stat.st_size)
                        if should_extract_manifest_ref(file_path):
                            status_fields = extract_status_fields(file_path, text_snapshot.text)
                            manifest_ref_rows.append(
                                {
                                    "root": root_spec.name,
                                    "root_alias": root_spec.alias,
                                    "relative_path": rel_path,
                                    "zip_member": zip_member,
                                    "sha256": text_sha256,
                                    "bytes": int(stat.st_size),
                                    "status_fields": status_fields,
                                    "excerpt": text_snapshot.text[:2000],
                                }
                            )
                            for index, chunk in enumerate(chunk_text(text_snapshot.text, chunk_chars=chunk_chars), start=1):
                                chunk_rows.append(
                                    {
                                        "chunk_id": f"{root_spec.alias}_{len(chunk_rows) + 1:06d}",
                                        "root": root_spec.name,
                                        "root_alias": root_spec.alias,
                                        "relative_path": rel_path,
                                        "chunk_index": index,
                                        "status_fields": status_fields,
                                        "text": chunk,
                                    }
                                )

                file_rows.append(
                    {
                        "root": root_spec.name,
                        "root_alias": root_spec.alias,
                        "relative_path": rel_path,
                        "original_path": str(file_path),
                        "bytes": int(stat.st_size),
                        "mtime": int(stat.st_mtime),
                        "extension": file_path.suffix.lower(),
                        "text_snapshot": bool(zip_member),
                        "text_zip_member": zip_member,
                        "text_sha256": text_sha256,
                    }
                )
            root_rows.append(
                {
                    "root": root_spec.name,
                    "requested": root_spec.requested,
                    "root_alias": root_spec.alias,
                    "original_path": str(root_spec.path),
                    "bytes": root_bytes,
                    "gib": round(root_bytes / 1024**3, 3),
                    "files": root_file_count,
                    "text_snapshot_files": root_text_refs,
                }
            )
            progress.mark(
                "root_scan_finish",
                root=root_spec.name,
                root_alias=root_spec.alias,
                bytes=root_bytes,
                files=root_file_count,
                text_snapshot_files=root_text_refs,
            )

    write_jsonl(output_dir / ROOTS_NAME, root_rows)
    write_jsonl(output_dir / FILES_NAME, file_rows)
    write_jsonl(output_dir / MANIFEST_REFS_NAME, manifest_ref_rows)
    write_jsonl(output_dir / CHUNKS_NAME, chunk_rows)
    return {
        "root_rows": root_rows,
        "file_rows": file_rows,
        "manifest_ref_rows": manifest_ref_rows,
        "chunk_rows": chunk_rows,
        "text_snapshot_count": text_snapshot_count,
        "text_snapshot_source_bytes": text_snapshot_source_bytes,
        "text_zip_path": text_zip_path,
    }


def cleanup_roots(
    *,
    lab_root: Path,
    root_specs: Sequence[RootSpec],
    output_dir: Path,
    cleanup_mode: str,
    execute_cleanup: bool,
    allow_metadata_only_delete: bool,
    progress: ProgressWriter,
    allow_thin_rag_delete: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cleanup_rows: list[dict[str, Any]] = []
    path_map_rows: list[dict[str, Any]] = []
    if cleanup_mode not in {"none", "metadata_only_delete", "thin_rag_delete", "move_to_vault"}:
        raise ValueError(f"unsupported cleanup mode: {cleanup_mode}")
    if cleanup_mode == "metadata_only_delete" and not allow_metadata_only_delete:
        raise ValueError("metadata_only_delete requires --allow-metadata-only-delete")
    if cleanup_mode == "thin_rag_delete" and not allow_thin_rag_delete:
        raise ValueError("thin_rag_delete requires --allow-thin-rag-delete")

    vault_dir = output_dir / "vault"
    path_map_path = output_dir / PATH_MAP_NAME
    write_jsonl(path_map_path, [])
    if cleanup_mode == "move_to_vault":
        vault_dir.mkdir(parents=True, exist_ok=True)

    for root_spec in root_specs:
        row = {
            "root": root_spec.name,
            "root_alias": root_spec.alias,
            "original_path": str(root_spec.path),
            "cleanup_mode": cleanup_mode,
            "execute_cleanup": execute_cleanup,
            "cleanup_status": "planned",
        }
        if cleanup_mode == "none":
            row["cleanup_status"] = "not_requested"
            cleanup_rows.append(row)
            continue
        if not is_under(root_spec.path, lab_root):
            raise ValueError(f"refusing unsafe root path: {root_spec.path}")
        if not execute_cleanup:
            row["cleanup_status"] = "dry_run"
            cleanup_rows.append(row)
            continue
        progress.mark("cleanup_start", root=root_spec.name, root_alias=root_spec.alias, mode=cleanup_mode)
        if cleanup_mode == "metadata_only_delete":
            shutil.rmtree(root_spec.path)
            row["cleanup_status"] = "deleted_metadata_only"
        elif cleanup_mode == "thin_rag_delete":
            shutil.rmtree(root_spec.path)
            row["cleanup_status"] = "deleted_thin_rag_only"
            row["binary_bytes_available"] = False
            row["retention_mode"] = "metadata_text_rag_only"
        elif cleanup_mode == "move_to_vault":
            target = vault_dir / root_spec.alias
            if target.exists():
                raise FileExistsError(f"vault target already exists: {target}")
            move_result = move_root_to_vault(
                root_spec.path,
                target,
                progress=progress,
                root_name=root_spec.name,
                root_alias=root_spec.alias,
            )
            row["cleanup_status"] = "moved_to_vault"
            row["vault_path"] = str(target)
            row.update(move_result)
            path_map_row = {
                "root": root_spec.name,
                "root_alias": root_spec.alias,
                "original_path": str(root_spec.path),
                "vault_path": str(target),
                "retention_mode": "exact_byte_vault",
                **move_result,
            }
            path_map_rows.append(path_map_row)
            append_jsonl(path_map_path, path_map_row)
        progress.mark("cleanup_finish", **row)
        cleanup_rows.append(row)
    summary = {
        "cleanup_mode": cleanup_mode,
        "cleanup_executed": execute_cleanup,
        "planned_roots": len(root_specs),
        "deleted_roots": sum(1 for row in cleanup_rows if row["cleanup_status"] == "deleted_metadata_only"),
        "thin_deleted_roots": sum(1 for row in cleanup_rows if row["cleanup_status"] == "deleted_thin_rag_only"),
        "moved_roots": sum(1 for row in cleanup_rows if row["cleanup_status"] == "moved_to_vault"),
        "dry_run_roots": sum(1 for row in cleanup_rows if row["cleanup_status"] == "dry_run"),
    }
    return summary, path_map_rows


def write_summary_md(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        f"# Artifact RAG {manifest['rag_id']}",
        "",
        f"- schema: `{manifest['schema']}`",
        f"- lab_artifacts_root: `{manifest.get('lab_artifacts_root', '')}`",
        f"- roots: `{summary.get('root_count', '')}`",
        f"- files: `{summary.get('file_count', '')}`",
        f"- source_gib: `{summary.get('source_gib', '')}`",
        f"- text_snapshot_files: `{summary.get('text_snapshot_files', '')}`",
        f"- cleanup_mode: `{summary.get('cleanup_mode', '')}`",
        f"- cleanup_executed: `{str(summary.get('cleanup_executed', '')).lower()}`",
        f"- current_retention_mode: `{manifest.get('current_retention_mode') or summary.get('current_retention_mode') or 'covered'}`",
        f"- binary_audio_model_bytes_available: `{str(manifest.get('binary_audio_model_bytes_available', True)).lower()}`",
        "",
        "## Retrieval",
        "",
        "- Start with `idx.json` or `../current.json`.",
        "- Search `chunks.jsonl` and `manifest_refs.jsonl` before browsing lab artifacts.",
        "- Use `path_map.jsonl` only as historical provenance when `vault_paths_available=false`.",
        "- `metadata_only_delete` keeps exact text refs and metadata, not binary audio/model bytes.",
        "- `thin_rag_only` keeps LLM-readable evidence and intentionally prunes bulky binary bytes.",
        "",
        "## Roots",
        "",
    ]
    for row in manifest.get("roots", []):
        lines.append(f"- `{row['root_alias']}` `{row['root']}`: {row['gib']} GiB, {row['files']} files")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_current_pointer(output_dir: Path, manifest: dict[str, Any]) -> None:
    current_path = output_dir.parent / "current.json"
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
    write_json(current_path, payload)


def build_artifact_rag_pack(
    *,
    lab_root: Path,
    output_dir: Path,
    include_roots: Sequence[str],
    cleanup_mode: str = "none",
    execute_cleanup: bool = False,
    allow_metadata_only_delete: bool = False,
    allow_thin_rag_delete: bool = False,
    max_text_bytes: int = 2_000_000,
    chunk_chars: int = 2000,
    update_current: bool = True,
) -> dict[str, Any]:
    lab_root = lab_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressWriter(output_dir / PROGRESS_NAME)
    progress.mark(
        "start",
        lab_root=str(lab_root),
        output_dir=str(output_dir),
        cleanup_mode=cleanup_mode,
        execute_cleanup=execute_cleanup,
    )
    root_specs = build_root_specs(lab_root, include_roots)
    scan = scan_roots(
        root_specs=root_specs,
        output_dir=output_dir,
        max_text_bytes=max_text_bytes,
        chunk_chars=chunk_chars,
        progress=progress,
    )
    total_bytes = sum(int(row["bytes"]) for row in scan["root_rows"])
    rag_id = output_dir.name
    fingerprint = stable_hash(
        {
            "roots": [(row["root"], row["bytes"], row["files"]) for row in scan["root_rows"]],
            "cleanup_mode": cleanup_mode,
            "text_snapshot_files": scan["text_snapshot_count"],
        }
    )
    provisional_cleanup_summary = {
        "cleanup_mode": cleanup_mode,
        "cleanup_executed": execute_cleanup,
        "planned_roots": len(root_specs),
        "deleted_roots": 0,
        "thin_deleted_roots": 0,
        "moved_roots": 0,
        "dry_run_roots": 0,
        "cleanup_status": "pending",
    }
    provisional_manifest = {
        "schema": SCHEMA,
        "index_schema": INDEX_SCHEMA,
        "rag_id": rag_id,
        "fingerprint": fingerprint,
        "lab_artifacts_root": str(lab_root),
        "root": str(output_dir),
        "summary": {
            "root_count": len(scan["root_rows"]),
            "file_count": len(scan["file_rows"]),
            "source_bytes": total_bytes,
            "source_gib": round(total_bytes / 1024**3, 3),
            "text_snapshot_files": scan["text_snapshot_count"],
            "text_snapshot_source_bytes": scan["text_snapshot_source_bytes"],
            "manifest_ref_count": len(scan["manifest_ref_rows"]),
            "chunk_count": len(scan["chunk_rows"]),
            **provisional_cleanup_summary,
        },
        "roots": scan["root_rows"],
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
            "metadata_only_delete_preserves_binary_bytes": False,
            "thin_rag_delete_preserves_binary_bytes": False,
            "move_to_vault_preserves_binary_bytes": cleanup_mode == "move_to_vault",
            "llm_retrieval_complete_for_textual_evidence": True,
            "binary_audio_model_bytes_available": cleanup_mode == "move_to_vault",
            "listen_delivery_artifact": False,
            "release_ready": False,
            "delivery_allowed": False,
            "thin_rag_delete_note": (
                "thin_rag_delete keeps structured metadata, manifests, logs, text refs, chunks, "
                "status fields, and provenance for LLM retrieval; it intentionally deletes bulky "
                "binary audio/model bytes from lab_artifacts to free disk space."
            ),
        },
        "query_hint": (
            "Use artifact_rag/current.json -> idx.json, then chunks.jsonl/manifest_refs.jsonl. "
            "Use path_map.jsonl for exact-byte vault roots. Do not browse lab_artifacts first for covered roots."
        ),
        "path_map_rows": [],
    }
    write_jsonl(output_dir / PATH_MAP_NAME, [])
    write_json(output_dir / IDX_NAME, provisional_manifest)
    write_json(
        output_dir / CLEANUP_NAME,
        {
            "schema": "artifact_rag_cleanup.v1",
            "rag_id": rag_id,
            "cleanup_summary": provisional_cleanup_summary,
            "roots": scan["root_rows"],
            "path_map_rows": [],
            "allow_metadata_only_delete": allow_metadata_only_delete,
            "allow_thin_rag_delete": allow_thin_rag_delete,
        },
    )
    write_summary_md(output_dir / SUMMARY_NAME, provisional_manifest)
    if update_current:
        update_current_pointer(output_dir, provisional_manifest)
    progress.mark("index_checkpoint", idx=str(output_dir / IDX_NAME), **provisional_manifest["summary"])

    cleanup_summary, path_map_rows = cleanup_roots(
        lab_root=lab_root,
        root_specs=root_specs,
        output_dir=output_dir,
        cleanup_mode=cleanup_mode,
        execute_cleanup=execute_cleanup,
        allow_metadata_only_delete=allow_metadata_only_delete,
        allow_thin_rag_delete=allow_thin_rag_delete,
        progress=progress,
    )
    manifest = {
        "schema": SCHEMA,
        "index_schema": INDEX_SCHEMA,
        "rag_id": rag_id,
        "fingerprint": fingerprint,
        "lab_artifacts_root": str(lab_root),
        "root": str(output_dir),
        "summary": {
            "root_count": len(scan["root_rows"]),
            "file_count": len(scan["file_rows"]),
            "source_bytes": total_bytes,
            "source_gib": round(total_bytes / 1024**3, 3),
            "text_snapshot_files": scan["text_snapshot_count"],
            "text_snapshot_source_bytes": scan["text_snapshot_source_bytes"],
            "manifest_ref_count": len(scan["manifest_ref_rows"]),
            "chunk_count": len(scan["chunk_rows"]),
            **cleanup_summary,
        },
        "roots": scan["root_rows"],
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
            "metadata_only_delete_preserves_binary_bytes": False,
            "thin_rag_delete_preserves_binary_bytes": False,
            "move_to_vault_preserves_binary_bytes": cleanup_mode == "move_to_vault",
            "llm_retrieval_complete_for_textual_evidence": True,
            "binary_audio_model_bytes_available": cleanup_mode == "move_to_vault",
            "listen_delivery_artifact": False,
            "release_ready": False,
            "delivery_allowed": False,
            "thin_rag_delete_note": (
                "thin_rag_delete keeps structured metadata, manifests, logs, text refs, chunks, "
                "status fields, and provenance for LLM retrieval; it intentionally deletes bulky "
                "binary audio/model bytes from lab_artifacts to free disk space."
            ),
        },
        "query_hint": (
            "Use artifact_rag/current.json -> idx.json, then chunks.jsonl/manifest_refs.jsonl. "
            "Use path_map.jsonl for exact-byte vault roots. Do not browse lab_artifacts first for covered roots."
        ),
        "path_map_rows": path_map_rows,
    }
    write_json(output_dir / IDX_NAME, manifest)
    write_json(
        output_dir / CLEANUP_NAME,
        {
            "schema": "artifact_rag_cleanup.v1",
            "rag_id": rag_id,
            "cleanup_summary": cleanup_summary,
            "roots": scan["root_rows"],
            "path_map_rows": path_map_rows,
            "allow_metadata_only_delete": allow_metadata_only_delete,
            "allow_thin_rag_delete": allow_thin_rag_delete,
        },
    )
    write_summary_md(output_dir / SUMMARY_NAME, manifest)
    if update_current:
        update_current_pointer(output_dir, manifest)
    progress.mark("finish", idx=str(output_dir / IDX_NAME), **manifest["summary"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG retention packs for lab_artifacts roots.")
    parser.add_argument("--lab-root", type=Path, default=Path("/app/lab_artifacts"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-root", action="append", default=[])
    parser.add_argument("--prune-existing-vault", type=Path)
    parser.add_argument("--allow-prune-vault", action="store_true")
    parser.add_argument("--cleanup-mode", choices=("none", "metadata_only_delete", "thin_rag_delete", "move_to_vault"), default="none")
    parser.add_argument("--execute-cleanup", action="store_true")
    parser.add_argument("--allow-metadata-only-delete", action="store_true")
    parser.add_argument("--allow-thin-rag-delete", action="store_true")
    parser.add_argument("--max-text-bytes", type=int, default=2_000_000)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--artifact-rag-mount-root", type=Path, default=DEFAULT_ARTIFACT_RAG_MOUNT_ROOT)
    parser.add_argument("--no-current", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="manage_artifact_rag.py", entrypoint_path="/app/tools/manage_artifact_rag.py")
    args = parse_args(argv)
    if args.prune_existing_vault is not None:
        manifest = prune_existing_vault_pack(
            pack_dir=args.prune_existing_vault,
            allow_prune_vault=bool(args.allow_prune_vault),
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --prune-existing-vault is used")
    validate_output_mount_contract(
        output_dir=args.output_dir,
        cleanup_mode=str(args.cleanup_mode),
        execute_cleanup=bool(args.execute_cleanup),
        artifact_rag_mount_root=args.artifact_rag_mount_root,
    )
    manifest = build_artifact_rag_pack(
        lab_root=args.lab_root,
        output_dir=args.output_dir,
        include_roots=args.include_root,
        cleanup_mode=str(args.cleanup_mode),
        execute_cleanup=bool(args.execute_cleanup),
        allow_metadata_only_delete=bool(args.allow_metadata_only_delete),
        allow_thin_rag_delete=bool(args.allow_thin_rag_delete),
        max_text_bytes=int(args.max_text_bytes),
        chunk_chars=int(args.chunk_chars),
        update_current=not bool(args.no_current),
    )
    print(
        json.dumps(
            {
                "idx": str(args.output_dir / IDX_NAME),
                "rag_id": manifest["rag_id"],
                "summary": manifest["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
