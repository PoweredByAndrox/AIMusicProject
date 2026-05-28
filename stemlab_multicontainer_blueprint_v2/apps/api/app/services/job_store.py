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
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"job file is empty: {path}")
    return JobStatus.model_validate_json(raw)


def list_jobs() -> Dict[str, JobStatus]:
    jobs: Dict[str, JobStatus] = {}
    job_files = sorted(JOBS_ROOT.glob("*/job.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in job_files:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        try:
            job = JobStatus.model_validate_json(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        jobs[job.job_id] = job
    return jobs
