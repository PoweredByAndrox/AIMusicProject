import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas import JobCreateRequest
from app.services.job_store import load_job, list_jobs
from app.services.orchestrator import create_job, run_job

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("")
def get_jobs() -> dict:
    jobs = list_jobs()
    return {"jobs": [job.model_dump() for job in jobs.values()]}


@router.post("")
def post_job(request: JobCreateRequest, background_tasks: BackgroundTasks) -> dict:
    job = create_job(request)
    background_tasks.add_task(asyncio.run, run_job(job))
    return job.model_dump()


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return load_job(job_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
