from fastapi import APIRouter, HTTPException, Query

from app.services.job_store import load_job
from app.services.virtualdj_compare import scan_virtualdj_baseline

router = APIRouter(prefix="/api/v1/compare", tags=["compare"])


@router.get("/virtualdj")
def get_virtualdj_compare(
    baseline_root: str = Query(..., description="Container-visible folder containing VirtualDJ exported stems"),
    job_id: str | None = Query(None, description="Optional job id from our pipeline for auto-matching"),
) -> dict:
    job = None
    if job_id:
        try:
            job = load_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    try:
        return scan_virtualdj_baseline(baseline_root=baseline_root, job=job)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
