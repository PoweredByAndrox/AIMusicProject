#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from run_testing_stuff_music_corpus_job_matrix import execute_job_matrix  # noqa: E402
from tools.train_generic_finetuning_controller_model import stable_hash  # noqa: E402
from worker.model_registry import ModelRegistry  # noqa: E402


CommandRunner = Callable[..., Any]

CANDIDATE_WAVE_EXECUTION_JOB_MATRIX_NAME = "jobs.jsonl"
CANDIDATE_WAVE_EXECUTION_MANIFEST_NAME = "exec.json"
CANDIDATE_WAVE_EXECUTION_PROGRESS_NAME = "progress.jsonl"

START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "gpu_inference_candidate_generation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def candidate_jobs_path(candidate_wave_manifest: dict[str, Any]) -> Path:
    outputs = candidate_wave_manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get("jobs_jsonl"):
        raise ValueError("candidate wave manifest is missing outputs.jobs_jsonl")
    return Path(str(outputs["jobs_jsonl"]))


def normalize_candidate_wave_job(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    candidate_wave_job_status = str(row.get("job_status") or "")
    normalized["candidate_wave_job_status"] = candidate_wave_job_status
    normalized["candidate_wave_execution_ready"] = bool(row.get("execution_ready"))
    if not normalized.get("execution_manifest_path") and normalized.get("output_dir"):
        normalized["execution_manifest_path"] = str(Path(str(normalized["output_dir"])) / "registry_candidate_execution_manifest.json")
    if candidate_wave_job_status == "planned" and bool(row.get("execution_ready")):
        normalized["job_status"] = "queued"
        normalized["gpu_inference_required"] = True
        normalized["gpu_inference_skipped"] = False
        normalized.pop("gpu_inference_skip_reason", None)
    else:
        normalized["job_status"] = candidate_wave_job_status
    normalized.setdefault("target_stem", row.get("stem_type") or "unknown")
    normalized.setdefault("release_ready", False)
    normalized.setdefault("delivery_allowed", False)
    normalized.setdefault("presentation_status", "candidate_wave_execution_not_for_listen_pack")
    normalized.setdefault("release_status", "not_release_ready")
    return normalized


def normalize_candidate_wave_jobs(jobs_jsonl: Path) -> list[dict[str, Any]]:
    return [normalize_candidate_wave_job(row) for row in iter_jsonl(jobs_jsonl)]


def build_wrapper_summary(
    *,
    candidate_wave_manifest_json: Path,
    normalized_rows: Sequence[dict[str, Any]],
    execution_summary: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    summary = dict(execution_summary)
    summary.update(
        {
            "candidate_wave_manifest_json": str(candidate_wave_manifest_json),
            "normalized_job_count": len(normalized_rows),
            "queued_from_planned_job_count": sum(
                1
                for row in normalized_rows
                if row.get("candidate_wave_job_status") == "planned" and row.get("job_status") == "queued"
            ),
            "audio_inference_started": bool(summary.get("execution_started")) and not bool(dry_run),
            "training_started": False,
            "fine_tune_started": False,
            "model_registry_edited": False,
            "release_ready": False,
        }
    )
    return summary


def run_candidate_wave_jobs(
    *,
    candidate_wave_manifest_json: Path,
    output_root: Path,
    max_jobs: int,
    stems: Sequence[str] | None = None,
    track_ids: Sequence[str] | None = None,
    model_ids: Sequence[str] | None = None,
    dry_run: bool = False,
    fail_fast: bool = False,
    command_runner: CommandRunner | None = None,
    registry: ModelRegistry | None = None,
    runtime_root: Path = Path("/app"),
) -> dict[str, Any]:
    if max_jobs <= 0:
        raise ValueError("max_jobs must be > 0")
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / CANDIDATE_WAVE_EXECUTION_PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    progress(
        progress_path,
        "start",
        candidate_wave_manifest_json=str(candidate_wave_manifest_json),
        max_jobs=max_jobs,
        dry_run=bool(dry_run),
    )

    candidate_wave_manifest = read_json(candidate_wave_manifest_json)
    jobs_jsonl = candidate_jobs_path(candidate_wave_manifest)
    normalized_rows = normalize_candidate_wave_jobs(jobs_jsonl)
    normalized_jobs_path = output_root / CANDIDATE_WAVE_EXECUTION_JOB_MATRIX_NAME
    write_jsonl(normalized_jobs_path, normalized_rows)
    progress(
        progress_path,
        "normalize_jobs",
        source_jobs_jsonl=str(jobs_jsonl),
        normalized_jobs_jsonl=str(normalized_jobs_path),
        normalized_job_count=len(normalized_rows),
    )

    execution_output_root = output_root / "job_execution"
    progress(progress_path, "execute_jobs", execution_output_root=str(execution_output_root))
    execution_result = execute_job_matrix(
        job_matrix_jsonl=normalized_jobs_path,
        output_root=execution_output_root,
        max_jobs=max_jobs,
        stems=stems,
        track_ids=track_ids,
        model_ids=model_ids,
        dry_run=bool(dry_run),
        command_runner=command_runner,
        fail_fast=bool(fail_fast),
        registry=registry,
        runtime_root=runtime_root,
    )

    summary = build_wrapper_summary(
        candidate_wave_manifest_json=candidate_wave_manifest_json,
        normalized_rows=normalized_rows,
        execution_summary=execution_result["summary"],
        dry_run=bool(dry_run),
    )
    wrapper_manifest = {
        "schema": "generic_controller_selector_candidate_wave_execution_manifest.v1",
        "manifest_type": "generic_controller_selector_candidate_wave_execution",
        "candidate_wave_manifest_json": str(candidate_wave_manifest_json),
        "candidate_wave_fingerprint": str(candidate_wave_manifest.get("fingerprint") or ""),
        "output_root": str(output_root),
        "fingerprint": stable_hash(
            {
                "candidate_wave_manifest_json": str(candidate_wave_manifest_json),
                "candidate_wave_fingerprint": str(candidate_wave_manifest.get("fingerprint") or ""),
                "max_jobs": int(max_jobs),
                "dry_run": bool(dry_run),
                "normalized_rows": normalized_rows,
            }
        ),
        "summary": summary,
        "manifest": execution_result["manifest"],
        "execution_outputs": execution_result["manifest"].get("output_files", {}),
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "candidate_wave_execution_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "outputs": {
            "normalized_job_matrix_jsonl": str(normalized_jobs_path),
            "manifest_json": str(output_root / CANDIDATE_WAVE_EXECUTION_MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
            "execution_output_root": str(execution_output_root),
        },
    }
    result = {
        "summary": summary,
        "manifest": execution_result["manifest"],
        "rows": execution_result["rows"],
        "wrapper_manifest": wrapper_manifest,
    }
    progress(progress_path, "write_outputs", execution_status=execution_result["manifest"].get("execution_status"))
    write_json(output_root / CANDIDATE_WAVE_EXECUTION_MANIFEST_NAME, result)
    progress(progress_path, "finish", manifest_path=str(output_root / CANDIDATE_WAVE_EXECUTION_MANIFEST_NAME))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded controller/selector candidate-wave job batch.")
    parser.add_argument("--candidate-wave-manifest-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, required=True)
    parser.add_argument("--stem", action="append", default=[])
    parser.add_argument("--track-id", action="append", default=[])
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_generic_controller_selector_candidate_wave_jobs.py",
        entrypoint_path="/app/tools/run_generic_controller_selector_candidate_wave_jobs.py",
    )
    args = parse_args(argv)
    result = run_candidate_wave_jobs(
        candidate_wave_manifest_json=args.candidate_wave_manifest_json,
        output_root=args.output_root,
        max_jobs=int(args.max_jobs),
        stems=args.stem or None,
        track_ids=args.track_id or None,
        model_ids=args.model_id or None,
        dry_run=bool(args.dry_run),
        fail_fast=bool(args.fail_fast),
        runtime_root=args.runtime_root,
    )
    print(json.dumps({"summary": result["summary"], "manifest": result["manifest"]}, ensure_ascii=False, sort_keys=True))
    return 1 if result["summary"].get("failed_jobs") else 0


if __name__ == "__main__":
    raise SystemExit(main())
