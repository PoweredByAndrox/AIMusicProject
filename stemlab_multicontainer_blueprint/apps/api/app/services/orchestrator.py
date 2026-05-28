from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx

from app.config import DATA_ROOT, DRY_RUN, WORKER_URLS
from app.schemas import JobCreateRequest, JobStatus, WorkerInvocation
from app.services.job_store import save_job, job_dir
from app.services.preset_loader import load_preset


def plan_invocations(preset: Dict[str, Any]) -> List[WorkerInvocation]:
    invocations: List[WorkerInvocation] = []
    for phase in preset.get("phases", []):
        phase_id = phase["id"]
        for step in phase.get("steps", []):
            invocations.append(
                WorkerInvocation(
                    phase_id=phase_id,
                    stem=step.get("stem"),
                    service=step["service"],
                    task=step["task"],
                    model=step.get("model"),
                    preset=step.get("preset"),
                    prompt=step.get("prompt"),
                    input_bus=step.get("input_bus"),
                    build_from=step.get("build_from"),
                    raw_step=step,
                )
            )
    return invocations


def create_job(request: JobCreateRequest) -> JobStatus:
    preset = load_preset(request.preset_name)
    job_id = uuid.uuid4().hex[:12]
    output_root = str((DATA_ROOT / "outputs" / job_id).resolve())
    invocations = plan_invocations(preset)
    job = JobStatus(
        job_id=job_id,
        status="queued",
        request=request,
        preset=preset,
        invocations=invocations,
        output_root=output_root,
        run_log=[],
    )
    save_job(job)
    return job


async def run_job(job: JobStatus) -> JobStatus:
    work_dir = Path(job.output_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    job.status = "running"
    save_job(job)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, invocation in enumerate(job.invocations, start=1):
            service_url = WORKER_URLS[invocation.service]
            payload = {
                "job_id": job.job_id,
                "phase_id": invocation.phase_id,
                "step_index": idx,
                "track_name": job.request.track_name,
                "input_path": job.request.input_path,
                "output_root": job.output_root,
                "target_stem": invocation.stem,
                "task": invocation.task,
                "model": invocation.model,
                "preset": invocation.preset,
                "prompt": invocation.prompt,
                "input_bus": invocation.input_bus,
                "build_from": invocation.build_from,
                "raw_step": invocation.raw_step,
                "dry_run": DRY_RUN or job.preset.get("settings", {}).get("dry_run", True),
            }
            try:
                response = await client.post(f"{service_url}/run", json=payload)
                response.raise_for_status()
                result = response.json()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "status": "error",
                    "service": invocation.service,
                    "task": invocation.task,
                    "error": str(exc),
                }
                job.run_log.append(result)
                job.status = "failed"
                save_job(job)
                return job

            job.run_log.append(result)
            save_job(job)

    job.status = "completed"
    manifest_path = work_dir / "job_manifest.json"
    manifest_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    save_job(job)
    return job
