from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    track_name: str = Field(..., description="User-facing track label")
    input_path: str = Field(..., description="Path visible inside containers")
    preset_name: str = Field(..., description="Preset file name without extension")
    notes: Optional[str] = None


class WorkerInvocation(BaseModel):
    phase_id: str
    stem: Optional[str] = None
    service: str
    task: str
    model: Optional[str] = None
    preset: Optional[str] = None
    prompt: Optional[str] = None
    input_bus: Optional[str] = None
    build_from: Optional[Dict[str, Any]] = None
    raw_step: Dict[str, Any]


class JobStatus(BaseModel):
    job_id: str
    status: str
    request: JobCreateRequest
    preset: Dict[str, Any]
    invocations: List[WorkerInvocation] = []
    run_log: List[Dict[str, Any]] = []
    output_root: str
