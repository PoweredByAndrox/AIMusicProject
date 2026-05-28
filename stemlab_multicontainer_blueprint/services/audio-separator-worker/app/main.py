from pathlib import Path
from fastapi import FastAPI
from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(title="audio-separator-worker", version="0.1.0")


def build_command(request: RunRequest) -> list[str]:
    preset = request.preset or "instrumental_balanced"
    if request.task == "harmonic_bus_cleanup":
        return ["audio-separator", request.input_path, "--ensemble_preset", preset]
    return ["audio-separator", request.input_path, "--ensemble_preset", preset]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "audio-separator-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "audio-separator-worker",
        "tasks": ["harmonic_bus_cleanup", "ensemble_cleanup"],
        "presets": [
            "instrumental_balanced",
            "instrumental_clean",
            "instrumental_full",
            "vocal_balanced",
        ],
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    command = build_command(request)
    target = request.target_stem or request.task
    output_root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    output_root.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        service="audio-separator-worker",
        status="dry_run_completed" if request.dry_run else "todo_real_run",
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=[str(output_root / f"{target}.flac")],
        notes="Dry-run wrapper around Audio Separator. Good place for ensemble presets and UVR-style models.",
    )
    return result.model_dump()
