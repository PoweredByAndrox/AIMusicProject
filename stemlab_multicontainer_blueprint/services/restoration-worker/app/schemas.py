from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RunRequest(BaseModel):
    job_id: str
    phase_id: str
    step_index: int
    track_name: str
    input_path: str
    output_root: str
    target_stem: Optional[str] = None
    task: str
    model: Optional[str] = None
    preset: Optional[str] = None
    prompt: Optional[str] = None
    input_bus: Optional[str] = None
    build_from: Optional[Dict[str, Any]] = None
    raw_step: Dict[str, Any]
    dry_run: bool = True


class RunResult(BaseModel):
    service: str
    status: str
    task: str
    target_stem: Optional[str] = None
    command: List[str]
    output_files: List[str]
    notes: str
