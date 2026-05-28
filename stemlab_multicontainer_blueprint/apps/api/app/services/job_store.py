from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from app.config import DATA_ROOT
from app.schemas import JobStatus

JOBS_ROOT = DATA_ROOT / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    path = JOBS_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_job(job: JobStatus) -> None:
    path = job_dir(job.job_id) / "job.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_job(job_id: str) -> JobStatus:
    path = job_dir(job_id) / "job.json"
    return JobStatus.model_validate_json(path.read_text(encoding="utf-8"))


def list_jobs() -> Dict[str, JobStatus]:
    jobs: Dict[str, JobStatus] = {}
    for path in JOBS_ROOT.glob("*/job.json"):
        try:
            payload = path.read_text(encoding="utf-8")
            if not payload.strip():
                continue
            job = JobStatus.model_validate_json(payload)
        except Exception:
            continue
        jobs[job.job_id] = job
    return jobs
