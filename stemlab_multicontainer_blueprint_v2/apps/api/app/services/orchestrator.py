from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx

from app.config import DATA_ROOT, DRY_RUN, WORKER_URLS
from app.schemas import JobCreateRequest, JobStatus, WorkerInvocation
from app.services.job_store import save_job
from app.services.preset_loader import load_preset

SUCCESS_STATUSES = {"completed", "dry_run_completed", "skipped_missing_reference"}
BSROFORMER_SW_MODELS = {
    "roformer-model-bs-roformer-sw-by-jarredou",
}
BSROFORMER_DEREVERB_MODELS = {
    "roformer-model-bs-roformer-de-reverb",
}


def resolve_service_url(invocation: WorkerInvocation) -> tuple[str, str]:
    service_key = invocation.service

    if service_key == "bsroformer-worker":
        if invocation.model in BSROFORMER_SW_MODELS or invocation.task in {
            "coarse_6stem",
            "guitar_refine_from_6stem",
            "piano_refine_from_6stem",
        }:
            service_key = "bsroformer-sw-worker"
        elif invocation.model in BSROFORMER_DEREVERB_MODELS or invocation.task == "dereverb_cleanup":
            service_key = "bsroformer-dereverb-worker"

    return service_key, WORKER_URLS[service_key]


def plan_invocations(preset: Dict[str, Any], request: JobCreateRequest) -> List[WorkerInvocation]:
    invocations: List[WorkerInvocation] = []
    for phase in preset.get("phases", []):
        phase_id = phase["id"]
        for step in phase.get("steps", []):
            if step.get("service") == "lyrics-worker" and not request.enable_lyrics:
                continue
            if step.get("only_if_reference_lyrics") and not request.lyrics_reference_path:
                continue
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
    preset = request.preset_override or load_preset(request.preset_name)
    stored_request = request.model_copy(update={"preset_override": None})
    job_id = uuid.uuid4().hex[:12]
    output_root = str((DATA_ROOT / "outputs" / job_id).resolve())
    invocations = plan_invocations(preset, request)
    job = JobStatus(
        job_id=job_id,
        status="queued",
        request=stored_request,
        preset=preset,
        invocations=invocations,
        output_root=output_root,
        run_log=[],
        artifacts={},
    )
    save_job(job)
    return job


def resolve_input_path(job: JobStatus, invocation: WorkerInvocation) -> str:
    if invocation.input_bus and invocation.input_bus in job.artifacts:
        return job.artifacts[invocation.input_bus]
    if invocation.stem and invocation.stem in job.artifacts:
        return job.artifacts[invocation.stem]
    return job.request.input_path


async def run_job(job: JobStatus) -> JobStatus:
    work_dir = Path(job.output_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    job.status = "running"
    save_job(job)

    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=15.0)) as client:
        for idx, invocation in enumerate(job.invocations, start=1):
            resolved_service_key, service_url = resolve_service_url(invocation)
            resolved_input_path = resolve_input_path(job, invocation)
            include = (invocation.build_from or {}).get("include", []) if invocation.build_from else []
            exclude = set((invocation.build_from or {}).get("exclude", []) if invocation.build_from else [])
            build_from_paths = {
                key: path
                for key, path in job.artifacts.items()
                if (not include or key in include) and key not in exclude
            }
            payload = {
                "job_id": job.job_id,
                "phase_id": invocation.phase_id,
                "step_index": idx,
                "track_name": job.request.track_name,
                "input_path": resolved_input_path,
                "output_root": job.output_root,
                "target_stem": invocation.stem,
                "task": invocation.task,
                "model": invocation.model,
                "preset": invocation.preset,
                "prompt": invocation.prompt,
                "input_bus": invocation.input_bus,
                "build_from": invocation.build_from,
                "build_from_paths": build_from_paths,
                "raw_step": invocation.raw_step,
                "dry_run": DRY_RUN or job.preset.get("settings", {}).get("dry_run", True),
                "lyrics_language": job.request.lyrics_language,
                "lyrics_reference_path": job.request.lyrics_reference_path,
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
                    "resolved_input_path": resolved_input_path,
                    "error": str(exc),
                }
                job.run_log.append(result)
                job.status = "failed"
                save_job(job)
                return job

            result["resolved_input_path"] = resolved_input_path
            result["build_from_paths"] = build_from_paths
            result["resolved_service"] = resolved_service_key
            result["resolved_service_url"] = service_url
            artifact_key_map = invocation.raw_step.get("artifact_keys") or {}
            result["artifact_key_map"] = artifact_key_map
            job.run_log.append(result)
            if result.get("status") in SUCCESS_STATUSES:
                for key, path in (result.get("named_outputs") or {}).items():
                    if isinstance(path, str):
                        resolved_key = artifact_key_map.get(key, key)
                        job.artifacts[resolved_key] = path
            else:
                job.status = "failed"
                save_job(job)
                return job
            save_job(job)

    job.status = "completed"
    manifest_path = work_dir / "job_manifest.json"
    manifest_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    save_job(job)
    return job
