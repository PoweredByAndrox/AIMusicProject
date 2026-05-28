from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    track_name: str = Field(..., description="User-facing track label")
    input_path: str = Field(..., description="Path visible inside containers")
    preset_name: str = Field(..., description="Preset file name without extension")
    preset_override: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional inline preset payload used instead of loading preset_name from disk",
    )
    notes: Optional[str] = None
    enable_lyrics: bool = Field(False, description="Run lyrics-worker phases for this job")
    lyrics_language: Optional[str] = Field(None, description="Language code like en, ru, de")
    lyrics_reference_path: Optional[str] = Field(
        None,
        description="Optional container-visible .txt/.lrc path for exact reference lyrics alignment",
    )


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


class ArtifactInfo(BaseModel):
    key: str
    path: str
    kind: str
    exists: bool
    size_bytes: int = 0
    url: Optional[str] = None
    source: str = "scan"
    relative_path: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    request: JobCreateRequest
    preset: Dict[str, Any]
    invocations: List[WorkerInvocation] = []
    run_log: List[Dict[str, Any]] = []
    output_root: str
    artifacts: Dict[str, str] = {}


class TandemSearchRequest(BaseModel):
    base_job_id: str = Field(..., description="Completed job id used as the search baseline")
    max_candidates: int = Field(5, ge=1, le=6, description="How many built-in tandem candidates to test")
    lyrics_validate_top_k: int = Field(2, ge=0, le=3, description="How many top coarse candidates to rerun with lyrics validation")
    reference_instrumental_path: Optional[str] = Field(
        None,
        description="Optional container-visible reference instrumental path used as an explicit target for the search",
    )
    force_reference_target: bool = Field(
        False,
        description="If true, apply reference penalties even when analyzer thinks the reference may be a different master or arrangement",
    )


class TandemSearchStatus(BaseModel):
    search_id: str
    status: str
    request: TandemSearchRequest
    output_root: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    error: Optional[str] = None
