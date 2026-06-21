#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_candidate_conditioned_audio_training_view import (  # noqa: E402
    iter_jsonl,
    row_is_candidate_view_eligible,
    row_stem,
    write_json,
    write_jsonl,
)
from docker_runtime import require_docker_runtime  # noqa: E402
from train_teacher_mask_separator import validate_stem_name  # noqa: E402


MANIFEST_NAME = "registered_candidate_family_wave_manifest.json"
JOBS_NAME = "registered_candidate_family_wave_jobs.jsonl"
PROGRESS_NAME = "registered_candidate_family_wave_progress.jsonl"
START_TIME = time.monotonic()


@dataclass(frozen=True)
class CandidateFamily:
    label: str
    model_id: str
    output_root: Path


Heartbeat = Callable[[float, int | None], None]
CommandRunner = Callable[[list[str], Path, Heartbeat], int]


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "gpu_registered_model_inference_wave",
        "gpu_inference_required": True,
        "gpu_inference_skipped": False,
        "fallback_used": False,
        "release_ready": False,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def parse_family(value: str) -> CandidateFamily:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("--family must use LABEL=MODEL_ID=OUTPUT_ROOT")
    label, model_id, output_root = (part.strip() for part in parts)
    if not label:
        raise ValueError("family label is empty")
    if not model_id:
        raise ValueError("family model_id is empty")
    if not output_root:
        raise ValueError("family output_root is empty")
    return CandidateFamily(label=label, model_id=model_id, output_root=Path(output_root))


def _selected_rows(
    rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    include_example_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if include_example_ids and row_id not in include_example_ids:
            continue
        if not row_is_candidate_view_eligible(row, stem=stem):
            continue
        selected.append(row)
        seen_ids.add(row_id)
    missing = sorted(include_example_ids - seen_ids)
    return selected, missing


def _job_row(
    row: dict[str, Any],
    *,
    family: CandidateFamily,
    output_dir: Path,
    command: list[str],
    status: str,
    returncode: int | None = None,
    failed_reason: str = "",
) -> dict[str, Any]:
    return {
        "schema": "registered_candidate_family_wave_job.v1",
        "id": str(row.get("id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "stem_type": row_stem(row),
        "source_audio_path": str(row.get("source_audio_path") or ""),
        "target_audio_path": str(row.get("target_audio_path") or ""),
        "source_candidate_family": family.label,
        "source_candidate_model_id": family.model_id,
        "output_dir": str(output_dir),
        "execution_manifest_path": str(output_dir / "registry_candidate_execution_manifest.json"),
        "command": command,
        "status": status,
        "returncode": returncode,
        "failed_reason": failed_reason,
        "fallback_used": False,
        "delivery_allowed": False,
        "release_ready": False,
    }


def _default_run_command(command: list[str], output_dir: Path, heartbeat: Heartbeat) -> int:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.monotonic()
    last_heartbeat = -1.0
    while True:
        returncode = proc.poll()
        now = time.monotonic()
        if last_heartbeat < 0.0 or now - last_heartbeat >= 15.0:
            heartbeat(round(now - started, 3), returncode)
            last_heartbeat = now
        if returncode is not None:
            break
        time.sleep(5.0)
    stdout, stderr = proc.communicate()
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    return int(proc.returncode or 0)


def _build_command(
    *,
    family: CandidateFamily,
    source_audio_path: str,
    output_dir: Path,
    stem: str,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "/app/tools/run_registered_model_candidate.py",
        "--model-id",
        family.model_id,
        "--device",
        device,
        "--target-stem",
        stem,
        source_audio_path,
        str(output_dir),
    ]


def run_registered_candidate_family_wave(
    *,
    audio_training_examples_jsonl: Path,
    families: Sequence[CandidateFamily],
    output_dir: Path,
    stem: str = "vocal",
    include_example_ids: set[str] | None = None,
    device: str = "cuda",
    continue_on_failure: bool = False,
    run_command: CommandRunner | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    if str(device) != "cuda":
        raise ValueError("registered candidate family wave requires --device cuda")
    if not families:
        raise ValueError("at least one --family is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    run_command = run_command or _default_run_command
    include_example_ids = set(include_example_ids or set())
    rows = list(iter_jsonl(audio_training_examples_jsonl))
    selected_rows, missing_ids = _selected_rows(rows, stem=stem, include_example_ids=include_example_ids)
    progress(
        progress_path,
        "start",
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        stem=stem,
        row_count=len(selected_rows),
        missing_ids=missing_ids,
        family_count=len(families),
        device=device,
    )
    if missing_ids:
        raise ValueError(f"included example ids missing or unsafe for candidate wave: {', '.join(missing_ids)}")

    job_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families, start=1):
        family.output_root.mkdir(parents=True, exist_ok=True)
        progress(
            progress_path,
            "family_start",
            family_index=family_index,
            family_count=len(families),
            source_candidate_family=family.label,
            model_id=family.model_id,
            output_root=str(family.output_root),
            job_count=len(selected_rows),
            device=device,
        )
        for job_index, row in enumerate(selected_rows, start=1):
            row_id = str(row.get("id") or "")
            source_audio_path = str(row.get("source_audio_path") or "")
            output_root = family.output_root / row_id
            command = _build_command(
                family=family,
                source_audio_path=source_audio_path,
                output_dir=output_root,
                stem=stem,
                device=device,
            )
            progress(
                progress_path,
                "job_start",
                family_index=family_index,
                source_candidate_family=family.label,
                model_id=family.model_id,
                job_index=job_index,
                job_count=len(selected_rows),
                example_id=row_id,
                track_id=str(row.get("track_id") or ""),
                source_audio_path=source_audio_path,
                output_dir=str(output_root),
                command=command,
                device=device,
            )

            def heartbeat(job_elapsed_sec: float, subprocess_returncode: int | None) -> None:
                progress(
                    progress_path,
                    "job_heartbeat",
                    source_candidate_family=family.label,
                    model_id=family.model_id,
                    job_index=job_index,
                    job_count=len(selected_rows),
                    example_id=row_id,
                    job_elapsed_sec=job_elapsed_sec,
                    subprocess_returncode=subprocess_returncode,
                    device=device,
                )

            returncode = int(run_command(command, output_dir=output_root, heartbeat=heartbeat))
            status = "completed" if returncode == 0 else "failed"
            failed_reason = "" if returncode == 0 else "registered_candidate_command_failed"
            job_rows.append(
                _job_row(
                    row,
                    family=family,
                    output_dir=output_root,
                    command=command,
                    status=status,
                    returncode=returncode,
                    failed_reason=failed_reason,
                )
            )
            progress(
                progress_path,
                "job_finish",
                source_candidate_family=family.label,
                model_id=family.model_id,
                job_index=job_index,
                job_count=len(selected_rows),
                example_id=row_id,
                status=status,
                returncode=returncode,
                failed_reason=failed_reason,
                manifest_path=str(output_root / "registry_candidate_execution_manifest.json"),
                device=device,
            )
            if status != "completed" and not continue_on_failure:
                progress(progress_path, "failed_fast", example_id=row_id, source_candidate_family=family.label)
                break
        family_jobs = [row for row in job_rows if row["source_candidate_family"] == family.label]
        progress(
            progress_path,
            "family_finish",
            source_candidate_family=family.label,
            model_id=family.model_id,
            completed_count=sum(1 for row in family_jobs if row["status"] == "completed"),
            failed_count=sum(1 for row in family_jobs if row["status"] == "failed"),
            job_count=len(family_jobs),
            device=device,
        )
        if any(row["status"] == "failed" for row in family_jobs) and not continue_on_failure:
            break

    jobs_path = output_dir / JOBS_NAME
    write_jsonl(jobs_path, job_rows)
    completed_count = sum(1 for row in job_rows if row["status"] == "completed")
    failed_count = sum(1 for row in job_rows if row["status"] == "failed")
    manifest = {
        "schema": "registered_candidate_family_wave_manifest.v1",
        "manifest_type": "registered_candidate_family_wave",
        "status": "passed" if job_rows and failed_count == 0 else "failed",
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "output_dir": str(output_dir),
        "stem": stem,
        "device": device,
        "family_count": len(families),
        "families": [
            {"label": family.label, "model_id": family.model_id, "output_root": str(family.output_root)}
            for family in families
        ],
        "job_count": len(job_rows),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "jobs_jsonl": str(jobs_path),
        "progress_jsonl": str(progress_path),
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "registered_candidate_family_wave_not_for_listen_pack",
        "release_status": "candidate_generation_only_requires_training_eval",
        "release_ready": False,
        "safety": {
            "fallback_used": False,
            "registered_model_required": True,
            "candidate_sources_are_not_ground_truth": True,
            "dirty_stems_never_ground_truth": True,
            "release_ready_forced_false": True,
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        status=manifest["status"],
        completed_count=completed_count,
        failed_count=failed_count,
        job_count=len(job_rows),
        manifest_path=str(output_dir / MANIFEST_NAME),
        device=device,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered candidate source families for training examples.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--family", action="append", default=[], help="LABEL=MODEL_ID=OUTPUT_ROOT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    parser.add_argument("--include-example-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_registered_candidate_family_wave.py",
        entrypoint_path="/app/tools/run_registered_candidate_family_wave.py",
    )
    args = parse_args(argv)
    manifest = run_registered_candidate_family_wave(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        families=[parse_family(str(item)) for item in args.family],
        output_dir=args.output_dir,
        stem=str(args.stem),
        include_example_ids={str(item) for item in args.include_example_id},
        device=str(args.device),
        continue_on_failure=bool(args.continue_on_failure),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "status": manifest["status"],
                "job_count": manifest["job_count"],
                "completed_count": manifest["completed_count"],
                "failed_count": manifest["failed_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
