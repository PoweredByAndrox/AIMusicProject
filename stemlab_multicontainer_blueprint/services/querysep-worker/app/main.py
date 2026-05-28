from pathlib import Path
from fastapi import FastAPI
from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(title="querysep-worker", version="0.1.0")


def build_command(request: RunRequest) -> list[str]:
    prompt = request.prompt or "generic_target"
    # Example adapter target for Banquet-style BYOQ inference.
    return [
        "python",
        "train.py",
        "inference_byoq",
        f"--input_path={request.input_path}",
        f"--output_path={request.output_root}",
        f"--query_path=/app/queries/{prompt}.wav",
        "--use_cuda=true",
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "querysep-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "querysep-worker",
        "tasks": ["prompt_extract", "residual_export"],
        "supported_prompts": [
            "strings_orchestra",
            "synth_lead",
            "synth_pad",
            "arps_plucks",
            "brass_woodwinds",
            "organ_reeds",
            "fx_risers_impacts",
            "ambience_residual",
        ],
        "state": "stub_by_default",
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    command = build_command(request)
    target = request.target_stem or request.prompt or request.task
    output_root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    output_root.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        service="querysep-worker",
        status="dry_run_completed" if request.dry_run else "needs_custom_integration",
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=[str(output_root / f"{target}.flac")],
        notes="Stub service. Replace with Banquet/QSCNet/TUSS adapter and real checkpoints.",
    )
    return result.model_dump()
