from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import track_alias, validate_artifact_relative_path
from docker_runtime import require_docker_runtime


AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}
FAMILY_MODEL_IDS = {
    "scnet": "scnet_xl_ihf_musdb18",
    "bs_roformer": "hybrid_htdemucs_ft_bs_roformer_6s",
    "mdx23c": "mdx23c_vocals",
    "melband": "melband_roformer_vocals",
    "bs_mamba2": "bs_mamba2_vocals",
    "demucs": "htdemucs_ft",
    "htdemucs": "htdemucs_ft",
}
MANIFEST_NAME = "manifest.json"
JOBS_NAME = "jobs.jsonl"
PROGRESS_NAME = "progress.jsonl"
SUMMARY_NAME = "summary.md"
EXECUTION_MANIFEST_NAME = "registry_candidate_execution_manifest.json"
START_TIME = time.monotonic()

CommandRunner = Callable[..., int]


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


def _ensure_path_inside_rag(*, artifact_rag_root: Path, path: Path, label: str) -> Path:
    root = _resolved(artifact_rag_root)
    target = _resolved(path)
    if target != root and root not in target.parents:
        raise ValueError(f"{label} must be inside artifact_rag_root")
    return target


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_progress(path: Path, event: str, **fields: Any) -> None:
    row = {
        "event": event,
        "phase": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "gpu_candidate_generation",
        "gpu_inference_required": True,
        "gpu_inference_skipped": False,
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_audio_files(corpus_dir: Path, *, limit: int) -> list[Path]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus_dir is missing: {corpus_dir}")
    rows = sorted(path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if limit > 0:
        return rows[:limit]
    return rows


def _default_command_runner(command: list[str], *, output_dir: Path, job: dict[str, Any]) -> int:
    del output_dir, job
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _relative_output_dir(path: Path, *, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    validate_artifact_relative_path(relative)
    return relative


def _job_output_dir(
    output: Path,
    *,
    track_alias_value: str,
    work_item_id: str,
    family_index: int,
    target_stem: str,
) -> Path:
    safe = output / "jobs" / track_alias_value / work_item_id / f"f{family_index:03d}" / target_stem
    validate_artifact_relative_path(safe.relative_to(output).as_posix())
    return safe


def _candidate_jobs(work_order: dict[str, Any], *, source_files: Sequence[Path], output: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    work_items = list(work_order.get("work_items") or [])
    for track_index, source_audio_path in enumerate(source_files, start=1):
        track_alias_value = track_alias(track_index)
        track_id = _sha256_file(source_audio_path)[:16]
        for work_item_index, item in enumerate(work_items, start=1):
            work_item_id = str(item.get("id") or f"w{work_item_index:03d}")
            route_id = str(item.get("route_id") or "")
            matrix = item.get("candidate_matrix") if isinstance(item.get("candidate_matrix"), dict) else {}
            families = [str(family) for family in matrix.get("families") or []]
            stems = [str(stem) for stem in matrix.get("stems") or ["vocal", "minus"]]
            for family_index, family in enumerate(families, start=1):
                model_id = FAMILY_MODEL_IDS.get(family)
                for target_stem in stems:
                    job_output_dir = _job_output_dir(
                        output,
                        track_alias_value=track_alias_value,
                        work_item_id=work_item_id,
                        family_index=family_index,
                        target_stem=target_stem,
                    )
                    command = [
                        sys.executable,
                        "/app/tools/run_registered_model_candidate.py",
                        "--model-id",
                        str(model_id or family),
                        "--device",
                        "cuda",
                        "--target-stem",
                        target_stem,
                        str(source_audio_path),
                        str(job_output_dir),
                    ]
                    jobs.append(
                        {
                            "schema": "aris_top_route_candidate_generation_job.v1",
                            "id": f"j{len(jobs) + 1:04d}",
                            "track_alias": track_alias_value,
                            "track_id": track_id,
                            "source_audio_path": str(source_audio_path),
                            "work_item_id": work_item_id,
                            "route_id": route_id,
                            "family": family,
                            "family_index": family_index,
                            "model_id": str(model_id or ""),
                            "target_stem": target_stem,
                            "output_dir": str(job_output_dir),
                            "relative_output_dir": _relative_output_dir(job_output_dir, root=output),
                            "execution_manifest_path": str(job_output_dir / EXECUTION_MANIFEST_NAME),
                            "command": command,
                            "status": "queued" if model_id else "failed",
                            "candidate_status": "pending" if model_id else "failed",
                            "failed_reason": "" if model_id else "unsupported_candidate_family",
                            "delivery_allowed": False,
                            "presentation_status": "blocked_from_listen_pack",
                            "release_ready": False,
                        }
                    )
    return jobs


def _finish_job_from_execution_manifest(job: dict[str, Any], *, returncode: int, error: str = "") -> dict[str, Any]:
    output_dir = Path(job["output_dir"])
    execution_manifest_path = Path(job["execution_manifest_path"])
    if returncode == 0 and execution_manifest_path.exists():
        execution = _read_json(execution_manifest_path)
        completed = execution.get("status") == "completed"
        candidate_manifest = execution.get("candidate_manifest") if isinstance(execution.get("candidate_manifest"), dict) else {}
        target_audio_path_value = str(candidate_manifest.get("target_audio_path") or "")
        target_audio_path = Path(target_audio_path_value) if target_audio_path_value else None
        if target_audio_path is not None and not target_audio_path.is_absolute():
            target_audio_path = output_dir / target_audio_path
        target_audio_exists = target_audio_path is not None and target_audio_path.exists() and target_audio_path.stat().st_size > 0
        binary_audio_available = candidate_manifest.get("binary_audio_available") is True or target_audio_exists
        job.update(
            {
                "status": "completed" if completed else "failed",
                "candidate_status": "generated_pending_gate" if completed else "failed",
                "failed_reason": "" if completed else str(execution.get("reason") or "execution_manifest_not_completed"),
                "returncode": returncode,
                "target_audio_path": str(target_audio_path) if target_audio_path is not None else "",
                "target_audio_bytes": target_audio_path.stat().st_size if target_audio_exists else 0,
                "binary_audio_available": binary_audio_available,
                "audio_bytes_available": candidate_manifest.get("audio_bytes_available") is True or binary_audio_available,
                "metadata_only": False if binary_audio_available else candidate_manifest.get("metadata_only") is True,
            }
        )
    else:
        job.update(
            {
                "status": "failed",
                "candidate_status": "failed",
                "failed_reason": error or ("execution_manifest_missing" if returncode == 0 else "registered_candidate_command_failed"),
                "returncode": returncode,
                "binary_audio_available": False,
                "audio_bytes_available": False,
                "metadata_only": True,
            }
        )
    job["output_artifact_count"] = sum(1 for path in output_dir.rglob("*") if path.is_file()) if output_dir.exists() else 0
    job["delivery_allowed"] = False
    job["presentation_status"] = "blocked_from_listen_pack"
    job["release_ready"] = False
    return job


def _summary_text(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ARIS Top Route Candidate Generation",
            "",
            f"- status: `{manifest['status']}`",
            f"- job_count: `{manifest['job_count']}`",
            f"- completed_count: `{manifest['completed_count']}`",
            f"- failed_count: `{manifest['failed_count']}`",
            "- delivery_allowed: `false`",
            "- matrix_gate_required_before_delivery: `true`",
            "",
        ]
    )


def run_aris_top_route_work_order(
    *,
    artifact_rag_root: Path,
    work_order_json: Path,
    corpus_dir: Path,
    output_dir: Path,
    limit: int = 0,
    max_jobs: int = 0,
    reuse_existing: bool = False,
    command_runner: CommandRunner | None = None,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    work_order_path = _ensure_path_inside_rag(
        artifact_rag_root=root,
        path=work_order_json,
        label="work_order_json",
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    _append_progress(
        progress_path,
        "run_start",
        work_order_json=str(work_order_path),
        corpus_dir=str(corpus_dir),
        output_dir=str(output),
        max_jobs=int(max_jobs),
        reuse_existing=bool(reuse_existing),
    )
    source_files = _discover_audio_files(corpus_dir, limit=int(limit))
    work_order = _read_json(work_order_path)
    jobs = _candidate_jobs(work_order, source_files=source_files, output=output)
    if max_jobs > 0:
        jobs = jobs[: int(max_jobs)]
    runner = command_runner or _default_command_runner
    executed_jobs: list[dict[str, Any]] = []
    for job_index, job in enumerate(jobs, start=1):
        _append_progress(
            progress_path,
            "job_start",
            job_index=job_index,
            job_count=len(jobs),
            job_id=job["id"],
            track_alias=job["track_alias"],
            route_id=job["route_id"],
            family=job["family"],
            model_id=job["model_id"],
            target_stem=job["target_stem"],
            output_dir=job["output_dir"],
        )
        if job["status"] == "failed":
            _finish_job_from_execution_manifest(job, returncode=127, error=str(job["failed_reason"]))
        elif reuse_existing and Path(job["execution_manifest_path"]).exists():
            _append_progress(
                progress_path,
                "job_reuse",
                job_index=job_index,
                job_count=len(jobs),
                job_id=job["id"],
                execution_manifest_path=job["execution_manifest_path"],
            )
            _finish_job_from_execution_manifest(job, returncode=0)
        else:
            try:
                returncode = int(runner(list(job["command"]), output_dir=Path(job["output_dir"]), job=dict(job)))
                _finish_job_from_execution_manifest(job, returncode=returncode)
            except Exception as exc:
                _finish_job_from_execution_manifest(
                    job,
                    returncode=1,
                    error=f"runner_exception:{exc.__class__.__name__}",
                )
                job["error"] = str(exc)
        executed_jobs.append(job)
        _append_progress(
            progress_path,
            "job_finish",
            job_index=job_index,
            job_count=len(jobs),
            job_id=job["id"],
            status=job["status"],
            candidate_status=job["candidate_status"],
            failed_reason=job["failed_reason"],
            output_artifact_count=job.get("output_artifact_count", 0),
        )
    jobs_path = output / JOBS_NAME
    _write_jsonl(jobs_path, executed_jobs)
    completed_count = sum(1 for job in executed_jobs if job["status"] == "completed")
    failed_count = sum(1 for job in executed_jobs if job["status"] != "completed")
    status = "completed" if executed_jobs and failed_count == 0 else "blocked"
    manifest = {
        "schema": "aris_top_route_candidate_generation_manifest.v1",
        "manifest_type": "aris_top_route_candidate_generation",
        "status": status,
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "work_order_json": str(work_order_path),
        "corpus_dir": str(corpus_dir),
        "jobs_jsonl": str(jobs_path),
        "progress_jsonl": str(progress_path),
        "summary_md": str(output / SUMMARY_NAME),
        "job_count": len(executed_jobs),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "matrix_gate_required_before_delivery": True,
        "reference_or_no_reference_gate_required": True,
        "rejected_artifacts_not_deliverable": True,
        "fallback_as_success": False,
        "manual_listening_required": False,
        "command_args": list(command_args or []),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    (output / SUMMARY_NAME).write_text(_summary_text(manifest), encoding="utf-8")
    _append_progress(
        progress_path,
        "run_finish",
        status=status,
        manifest_json=str(output / MANIFEST_NAME),
        completed_count=completed_count,
        failed_count=failed_count,
        job_count=len(executed_jobs),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict ARIS top-route candidate generation work order.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--work-order-json", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_aris_top_route_work_order.py",
        entrypoint_path="/app/tools/run_aris_top_route_work_order.py",
    )
    args = parse_args(argv)
    manifest = run_aris_top_route_work_order(
        artifact_rag_root=args.artifact_rag_root,
        work_order_json=args.work_order_json,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        max_jobs=args.max_jobs,
        reuse_existing=args.reuse_existing,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
