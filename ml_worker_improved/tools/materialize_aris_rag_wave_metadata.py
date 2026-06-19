#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from artifact_name_policy import validate_artifact_relative_path  # noqa: E402
from docker_runtime import require_docker_runtime  # noqa: E402

DEFAULT_RAG_ROOT = Path("/app/artifact_rag")
START_TIME = time.monotonic()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_output_inside_rag(*, artifact_rag_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = _resolved(artifact_rag_root)
    output = _resolved(output_dir)
    if output != root and root not in output.parents:
        raise ValueError("output_dir must be inside artifact_rag_root")
    try:
        validate_artifact_relative_path(output.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError("output_dir must use short Windows-safe relative path inside artifact_rag_root") from exc
    return root, output


def _resolve_inside(root: Path, relative_path: str | Path, *, field_name: str) -> Path:
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise ValueError(f"{field_name} must be relative to artifact rag root")
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes artifact rag root: {relative_path}") from exc
    return resolved


def _resolve_pack(rag_root: Path, rag_id: str | None) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    current = read_json(rag_root / "current.json")
    idx_rel = Path(rag_id) / "idx.json" if rag_id else Path(str(current.get("idx") or ""))
    if not str(idx_rel):
        raise ValueError("current.json does not contain idx")
    idx_path = _resolve_inside(rag_root, idx_rel, field_name="rag_id" if rag_id else "current.idx")
    idx = read_json(idx_path)
    return current, idx_path.parent, idx


def _progress_row(event: str, **fields: Any) -> dict[str, Any]:
    row = {
        "event": event,
        "phase": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "rag_metadata_only_cpu_materialization",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "rag_text_metadata_materialization_only",
        "binary_audio_available": False,
        "audio_exported": False,
        "models_changed": False,
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    return row


def append_progress(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_progress_row(event, **fields), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _matches_include(row: dict[str, Any], include_substrings: Sequence[str]) -> bool:
    relative_path = str(row.get("relative_path") or "")
    status_fields = json.dumps(row.get("status_fields") or {}, ensure_ascii=False, sort_keys=True)
    haystack = f"{relative_path}\n{status_fields}".lower()
    return all(str(item).lower() in haystack for item in include_substrings)


def _manifest_ref_rows(pack_dir: Path, include_substrings: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(pack_dir / "manifest_refs.jsonl"):
        relative_path = str(row.get("relative_path") or "")
        status_fields = row.get("status_fields") if isinstance(row.get("status_fields"), dict) else {}
        if not relative_path.endswith("registered_candidate_family_wave_manifest.json"):
            continue
        if status_fields.get("manifest_type") != "registered_candidate_family_wave":
            continue
        if include_substrings and not _matches_include(row, include_substrings):
            continue
        rows.append(row)
    return rows


def _zip_json(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    payload = json.loads(archive.read(member).decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"zip member is not a JSON object: {member}")
    return payload


def _zip_jsonl(archive: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(archive.read(member).decode("utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"zip member JSONL row is not an object: {member}:{line_number}")
        rows.append(payload)
    return rows


def _same_directory_jobs_member(manifest_member: str, zip_names: set[str]) -> str | None:
    candidate = str(Path(manifest_member).with_name("registered_candidate_family_wave_jobs.jsonl")).replace("\\", "/")
    return candidate if candidate in zip_names else None


def _historical_path_to_zip_member(path: str, *, root_alias: str, zip_names: set[str]) -> str | None:
    normalized = str(path or "").replace("\\", "/")
    if normalized in zip_names:
        return normalized
    for marker in ("/gt90_cycle_001/",):
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            candidate = f"{root_alias}/{normalized[marker_index + 1:]}"
            if candidate in zip_names:
                return candidate
    return None


def _metadata_blocked(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = dict(payload)
    blocked["binary_audio_available"] = False
    blocked["audio_bytes_available"] = False
    blocked["binary_audio_model_bytes_available"] = False
    blocked["metadata_only"] = True
    blocked["delivery_allowed"] = False
    blocked["release_ready"] = False
    blocked["can_publish_as_result"] = False
    blocked["presentation_status"] = "blocked_from_listen_pack"
    return blocked


def materialize_aris_rag_wave_metadata(
    *,
    artifact_rag_root: Path,
    output_dir: Path,
    rag_id: str | None = None,
    include_substrings: Sequence[str] = (),
    limit: int = 32,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    current, pack_dir, idx = _resolve_pack(root, rag_id)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "run_start",
        pack_dir=str(pack_dir),
        output_dir=str(output),
        include_substrings=list(include_substrings),
    )

    text_zip = pack_dir / "text_refs.zip"
    if not text_zip.exists():
        raise FileNotFoundError(f"RAG text_refs.zip missing: {text_zip}")

    recovered_refs: list[dict[str, Any]] = []
    blocked_refs: list[dict[str, Any]] = []
    wave_manifest_jsons: list[str] = []
    materialized_job_count = 0
    materialized_exec_count = 0
    with zipfile.ZipFile(text_zip) as archive:
        zip_names = set(archive.namelist())
        manifest_rows = _manifest_ref_rows(pack_dir, include_substrings)[:limit]
        for manifest_index, row in enumerate(manifest_rows, start=1):
            manifest_member = str(row.get("zip_member") or "")
            root_alias = str(row.get("root_alias") or "")
            if manifest_member not in zip_names:
                blocked_refs.append({"zip_member": manifest_member, "reason": "manifest_zip_member_missing"})
                continue
            jobs_member = _same_directory_jobs_member(manifest_member, zip_names)
            if not jobs_member:
                blocked_refs.append({"zip_member": manifest_member, "reason": "jobs_zip_member_missing"})
                continue
            ref_dir = output / "refs" / f"r{manifest_index:03d}"
            exec_dir = ref_dir / "exec"
            wave_manifest_path = ref_dir / "wave.json"
            jobs_path = ref_dir / "jobs.jsonl"
            wave = _metadata_blocked(_zip_json(archive, manifest_member))
            wave["source_zip_member"] = manifest_member
            wave["source_manifest_sha256"] = row.get("sha256")
            wave["source_relative_path"] = row.get("relative_path")
            wave["historical_jobs_jsonl"] = wave.get("jobs_jsonl")
            wave["jobs_jsonl"] = str(jobs_path)
            jobs: list[dict[str, Any]] = []
            for job_index, job in enumerate(_zip_jsonl(archive, jobs_member), start=1):
                historical_exec = str(job.get("execution_manifest_path") or "")
                exec_member = _historical_path_to_zip_member(historical_exec, root_alias=root_alias, zip_names=zip_names)
                if not exec_member:
                    blocked_refs.append(
                        {
                            "zip_member": jobs_member,
                            "job_index": job_index,
                            "reason": "execution_manifest_zip_member_missing",
                            "historical_execution_manifest_path": historical_exec,
                        }
                    )
                    continue
                exec_payload = _metadata_blocked(_zip_json(archive, exec_member))
                candidate_manifest = exec_payload.get("candidate_manifest")
                if isinstance(candidate_manifest, dict):
                    exec_payload["candidate_manifest"] = _metadata_blocked(candidate_manifest)
                exec_payload["source_zip_member"] = exec_member
                exec_payload["historical_execution_manifest_path"] = historical_exec
                exec_path = exec_dir / f"e{job_index:03d}.json"
                write_json(exec_path, exec_payload)
                rewritten_job = _metadata_blocked(job)
                rewritten_job["execution_manifest_path"] = str(exec_path)
                rewritten_job["historical_execution_manifest_path"] = historical_exec
                rewritten_job["source_zip_member"] = jobs_member
                jobs.append(rewritten_job)
                materialized_exec_count += 1
            if not jobs:
                blocked_refs.append({"zip_member": jobs_member, "reason": "no_execution_manifests_materialized"})
                continue
            write_json(wave_manifest_path, wave)
            write_jsonl(jobs_path, jobs)
            wave_manifest_jsons.append(str(wave_manifest_path))
            materialized_job_count += len(jobs)
            recovered_refs.append(
                {
                    "schema": "aris_rag_wave_metadata_ref.v1",
                    "wave_manifest_json": str(wave_manifest_path),
                    "jobs_jsonl": str(jobs_path),
                    "source_manifest_zip_member": manifest_member,
                    "source_jobs_zip_member": jobs_member,
                    "job_count": len(jobs),
                    "binary_audio_available": False,
                    "delivery_allowed": False,
                }
            )
            append_progress(
                progress_path,
                "manifest_materialized",
                wave_manifest_json=str(wave_manifest_path),
                job_count=len(jobs),
            )

    recovered_refs_path = output / "refs.jsonl"
    blocked_refs_path = output / "blocked.jsonl"
    summary_path = output / "summary.md"
    manifest_path = output / "manifest.json"
    write_jsonl(recovered_refs_path, recovered_refs)
    write_jsonl(blocked_refs_path, blocked_refs)
    manifest = {
        "schema": "aris_rag_wave_metadata_materialization.v1",
        "manifest_type": "aris_rag_wave_metadata_materialization",
        "status": "metadata_only_blocked" if recovered_refs else "no_metadata_materialized",
        "artifact_rag_root": str(root),
        "pack_dir": str(pack_dir),
        "rag_id": idx.get("rag_id") or pack_dir.name,
        "current_status": current.get("status") or idx.get("status"),
        "current_retention_mode": current.get("current_retention_mode") or idx.get("current_retention_mode"),
        "binary_audio_available": False,
        "binary_audio_model_bytes_available": False,
        "audio_bytes_available": False,
        "metadata_only": True,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "presentation_status": "blocked_from_listen_pack",
        "output_dir": str(output),
        "progress_jsonl": str(progress_path),
        "recovered_refs_jsonl": str(recovered_refs_path),
        "blocked_refs_jsonl": str(blocked_refs_path),
        "summary_md": str(summary_path),
        "wave_manifest_jsons": wave_manifest_jsons,
        "materialized_manifest_count": len(wave_manifest_jsons),
        "materialized_job_count": materialized_job_count,
        "materialized_execution_manifest_count": materialized_exec_count,
        "blocked_ref_count": len(blocked_refs),
        "include_substrings": list(include_substrings),
        "command_args": list(command_args or []),
        "safety": {
            "audio_read": False,
            "audio_exported": False,
            "models_changed": False,
            "thresholds_relaxed": False,
            "objective_gate_evidence_invented": False,
            "metadata_only_rows_must_be_blocked_by_runner": True,
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(
        "\n".join(
            [
                "# ARIS RAG Wave Metadata Materialization",
                "",
                f"Status: {manifest['status']}",
                f"Wave manifests: {manifest['materialized_manifest_count']}",
                f"Jobs: {manifest['materialized_job_count']}",
                "Binary audio bytes are not available; output is metadata evidence only.",
                "Materialized rows are blocked from listen packs.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    append_progress(
        progress_path,
        "run_finish",
        status=manifest["status"],
        manifest_path=str(manifest_path),
        materialized_job_count=materialized_job_count,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize registered wave metadata from a thin artifact RAG pack.")
    parser.add_argument("--artifact-rag-root", type=Path, default=DEFAULT_RAG_ROOT)
    parser.add_argument("--rag-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-substring", action="append", default=[])
    parser.add_argument("--limit", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="materialize_aris_rag_wave_metadata.py",
        entrypoint_path="/app/tools/materialize_aris_rag_wave_metadata.py",
    )
    args = parse_args(argv)
    manifest = materialize_aris_rag_wave_metadata(
        artifact_rag_root=args.artifact_rag_root,
        rag_id=args.rag_id,
        output_dir=args.output_dir,
        include_substrings=args.include_substring,
        limit=int(args.limit),
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "metadata_only_blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
