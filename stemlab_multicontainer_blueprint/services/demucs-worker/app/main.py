from pathlib import Path
from fastapi import FastAPI
from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(title="demucs-worker", version="0.1.0")


def build_command(request: RunRequest) -> list[str]:
    base = ["python", "-m", "demucs"]
    model = request.model or "htdemucs_ft"
    if request.task == "coarse_6stem":
        return base + ["-n", "htdemucs_6s", "--shifts", "2", "--overlap", "0.25", request.input_path]
    if request.task == "coarse_vdbo_validation":
        return base + ["-n", model, "--shifts", "2", "--overlap", "0.25", request.input_path]
    if request.task == "drums_focus":
        return base + ["-n", model, "--two-stems", "drums", "--shifts", "2", request.input_path]
    if request.task == "bass_focus":
        return base + ["-n", model, "--two-stems", "bass", "--shifts", "2", request.input_path]
    return base + ["-n", model, request.input_path]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "demucs-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "demucs-worker",
        "tasks": [
            "coarse_6stem",
            "coarse_vdbo_validation",
            "drums_focus",
            "bass_focus",
        ],
        "models": ["htdemucs_ft", "htdemucs_6s", "htdemucs"],
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    command = build_command(request)
    target = request.target_stem or request.task
    output_root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    output_root.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        service="demucs-worker",
        status="dry_run_completed" if request.dry_run else "todo_real_run",
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=[str(output_root / f"{target}.flac")],
        notes="Dry-run wrapper around Demucs. Replace with subprocess.run for real execution.",
    )
    return result.model_dump()
