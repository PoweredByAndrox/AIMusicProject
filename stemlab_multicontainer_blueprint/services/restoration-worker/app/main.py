from pathlib import Path
from fastapi import FastAPI
from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(title="restoration-worker", version="0.1.0")


def build_command(request: RunRequest) -> list[str]:
    recipe = request.raw_step.get("recipe", "generic_cleanup")
    return [
        "python",
        "-m",
        "restoration_stub",
        "--input",
        request.input_path,
        "--output",
        request.output_root,
        "--recipe",
        str(recipe),
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "restoration-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "restoration-worker",
        "tasks": [
            "dereverb_cleanup",
            "harmonic_smoothing",
            "artifact_repair",
        ],
        "recipes": [
            "vocals_clean",
            "strings_clean",
            "synth_clean",
            "generic_cleanup",
        ],
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    command = build_command(request)
    target = request.target_stem or request.task
    output_root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    output_root.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        service="restoration-worker",
        status="dry_run_completed" if request.dry_run else "needs_custom_integration",
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=[str(output_root / f"{target}.flac")],
        notes="Stub service for stem-specific restoration experts.",
    )
    return result.model_dump()
