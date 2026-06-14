#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402

RUN_ROWS_NAME = "generic_audio_finetune_run_rows.jsonl"
RUN_MANIFEST_NAME = "generic_audio_finetune_run_manifest.json"
PROGRESS_NAME = "generic_audio_finetune_run_progress.jsonl"
START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, *, require_gpu: bool = True, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "gpu_training" if require_gpu else "cpu_diagnostic_training",
        "gpu_training_required": require_gpu,
        "gpu_training_skipped": not require_gpu,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def command_output_dir(command: Sequence[str]) -> Path:
    try:
        index = list(command).index("--output-dir")
    except ValueError as exc:
        raise ValueError(f"training command missing --output-dir: {command}") from exc
    try:
        return Path(command[index + 1])
    except IndexError as exc:
        raise ValueError(f"training command has --output-dir without value: {command}") from exc


def validate_training_output(output_dir: Path, *, require_gpu: bool) -> dict[str, Any]:
    manifest_path = output_dir / "stem_manifest.json"
    progress_path = output_dir / "teacher_mask_separator_progress.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not progress_path.exists():
        raise FileNotFoundError(progress_path)
    manifest = read_json(manifest_path)
    progress_rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    first_event = str(progress_rows[0].get("event") or "") if progress_rows else ""
    last_event = str(progress_rows[-1].get("event") or "") if progress_rows else ""
    if first_event != "start" or last_event != "finish":
        raise ValueError(f"incomplete progress trail: {progress_path}")
    diagnostics = manifest.get("training", {}).get("diagnostics", {})
    device_used = float(diagnostics.get("device_used", 0.0))
    if require_gpu and device_used != 1.0:
        raise ValueError(f"GPU was required but trainer did not report CUDA use: {manifest_path}")
    checkpoint_path = Path(str(manifest.get("training", {}).get("checkpoint_path") or ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    stem_paths: list[Path] = []
    stems = manifest.get("stems")
    if isinstance(stems, dict):
        for stem_payload in stems.values():
            if isinstance(stem_payload, dict) and str(stem_payload.get("path") or "").strip():
                stem_paths.append(Path(str(stem_payload.get("path"))))
    if not stem_paths:
        raise ValueError(f"manifest has no emitted stem path: {manifest_path}")
    for stem_path in stem_paths:
        if not stem_path.exists():
            raise FileNotFoundError(stem_path)
    return {
        "manifest_path": str(manifest_path),
        "progress_path": str(progress_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(manifest.get("training", {}).get("checkpoint_sha256") or ""),
        "output_stem_paths": [str(path) for path in stem_paths],
        "training_loss": diagnostics.get("training_loss"),
        "device_used": device_used,
        "sample_rate_alignment": manifest.get("training", {}).get("sample_rate_alignment", {}),
        "delivery_allowed": manifest.get("presentation", {}).get("result_delivery_allowed") is True,
        "progress_first_event": first_event,
        "progress_last_event": last_event,
        "progress_event_count": len(progress_rows),
    }


def select_example_jobs(plan: dict[str, Any], *, scope: str | None) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    selected: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for job in plan.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        if job.get("status") != "ready_for_gpu_training":
            continue
        if scope and str(job.get("scope") or "") != scope:
            continue
        examples = [example for example in job.get("example_jobs") or [] if isinstance(example, dict)]
        for index, example in enumerate(examples, start=1):
            selected.append((job, example, index))
    return selected


def run_generic_audio_finetune_jobs(
    *,
    job_plan_json: Path,
    output_dir: Path,
    scope: str | None = None,
    skip_existing_complete: bool = False,
    require_gpu: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    plan = read_json(job_plan_json)
    selected_jobs = select_example_jobs(plan, scope=scope)
    progress(
        progress_path,
        "start",
        require_gpu=require_gpu,
        job_plan_json=str(job_plan_json),
        scope=str(scope or ""),
        selected_example_count=len(selected_jobs),
    )
    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    executed_count = 0
    validated_existing_count = 0

    for job, example, index in selected_jobs:
        command = [str(item) for item in example.get("command") or []]
        if not command:
            raise ValueError(f"example has empty command: {example}")
        output_path = command_output_dir(command)
        base_row = {
            "scope": str(job.get("scope") or ""),
            "index": index,
            "example_count": len(job.get("example_jobs") or []),
            "example_id": str(example.get("example_id") or ""),
            "track_id": str(example.get("track_id") or ""),
            "source_audio_path": str(example.get("source_audio_path") or ""),
            "target_audio_path": str(example.get("target_audio_path") or ""),
            "output_dir": str(output_path),
            "command": command,
        }
        progress(
            progress_path,
            "example_start",
            require_gpu=require_gpu,
            **{key: base_row[key] for key in ("scope", "index", "example_id", "track_id", "output_dir")},
        )
        start_time = time.monotonic()
        try:
            if skip_existing_complete and (output_path / "stem_manifest.json").exists() and (output_path / "teacher_mask_separator_progress.jsonl").exists():
                validation = validate_training_output(output_path, require_gpu=require_gpu)
                row = {**base_row, **validation, "status": "skipped_existing_complete", "elapsed_sec": 0.0, "returncode": 0}
                validated_existing_count += 1
            else:
                proc = subprocess.run(command, cwd="/app", text=True, capture_output=True)
                elapsed = round(time.monotonic() - start_time, 3)
                if proc.returncode != 0:
                    row = {
                        **base_row,
                        "status": "failed",
                        "elapsed_sec": elapsed,
                        "returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-1200:],
                        "stderr_tail": proc.stderr[-1200:],
                    }
                    failed_rows.append(row)
                    rows.append(row)
                    progress(progress_path, "example_failed", require_gpu=require_gpu, **row)
                    continue
                validation = validate_training_output(output_path, require_gpu=require_gpu)
                row = {
                    **base_row,
                    **validation,
                    "status": "executed",
                    "elapsed_sec": elapsed,
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-1200:],
                    "stderr_tail": proc.stderr[-1200:],
                }
                executed_count += 1
        except Exception as exc:
            row = {**base_row, "status": "failed", "elapsed_sec": round(time.monotonic() - start_time, 3), "error": str(exc)}
            failed_rows.append(row)
            rows.append(row)
            progress(progress_path, "example_failed", require_gpu=require_gpu, **row)
            continue
        rows.append(row)
        progress(
            progress_path,
            "example_finish",
            require_gpu=require_gpu,
            **{key: row.get(key) for key in ("scope", "index", "example_id", "track_id", "status", "elapsed_sec", "device_used")},
        )

    rows_path = output_dir / RUN_ROWS_NAME
    write_jsonl(rows_path, rows)
    manifest = {
        "schema": "generic_audio_finetune_run_manifest.v1",
        "manifest_type": "generic_audio_finetune_run",
        "job_plan_json": str(job_plan_json),
        "output_dir": str(output_dir),
        "scope": str(scope or ""),
        "selected_example_count": len(selected_jobs),
        "run_example_count": len(rows),
        "executed_example_count": executed_count,
        "validated_existing_example_count": validated_existing_count,
        "failed_example_count": len(failed_rows),
        "all_examples_used_gpu": bool(rows) and all(float(row.get("device_used", 0.0)) == 1.0 for row in rows if row.get("status") != "failed"),
        "all_progress_complete": bool(rows)
        and all(row.get("progress_first_event") == "start" and row.get("progress_last_event") == "finish" for row in rows if row.get("status") != "failed"),
        "resampled_example_count": sum(1 for row in rows if (row.get("sample_rate_alignment") or {}).get("teacher_target_resampled") is True),
        "training_started": executed_count > 0,
        "fine_tune_started": executed_count > 0,
        "models_changed": executed_count > 0,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "training_artifact_not_for_listen_pack",
        "release_status": "training_artifact_only_requires_holdout_validation",
        "release_ready": False,
        "run_rows_jsonl": str(rows_path),
        "progress_jsonl": str(progress_path),
    }
    write_json(output_dir / RUN_MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        require_gpu=require_gpu,
        **{key: manifest[key] for key in ("run_example_count", "failed_example_count", "executed_example_count", "validated_existing_example_count")},
    )
    if failed_rows:
        raise RuntimeError(f"generic_audio_finetune_failed_examples:{len(failed_rows)}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ready generic audio fine-tune jobs from a job plan.")
    parser.add_argument("--job-plan-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope")
    parser.add_argument("--skip-existing-complete", action="store_true")
    parser.add_argument("--allow-non-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_generic_audio_finetune_jobs.py",
        entrypoint_path="/app/tools/run_generic_audio_finetune_jobs.py",
    )
    args = parse_args(argv)
    manifest = run_generic_audio_finetune_jobs(
        job_plan_json=args.job_plan_json,
        output_dir=args.output_dir,
        scope=args.scope,
        skip_existing_complete=bool(args.skip_existing_complete),
        require_gpu=not bool(args.allow_non_gpu),
    )
    print(json.dumps({"summary": {key: manifest[key] for key in ("run_example_count", "failed_example_count", "executed_example_count", "validated_existing_example_count")}, "manifest": str(args.output_dir / RUN_MANIFEST_NAME)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
