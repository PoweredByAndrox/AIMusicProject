from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
    build_from_paths: Dict[str, str] = Field(default_factory=dict)
    raw_step: Dict[str, Any]
    dry_run: bool = True
    lyrics_language: Optional[str] = None
    lyrics_reference_path: Optional[str] = None


class RunResult(BaseModel):
    service: str
    status: str
    task: str
    target_stem: Optional[str] = None
    command: List[str] = Field(default_factory=list)
    output_files: List[str] = Field(default_factory=list)
    named_outputs: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    notes: str
