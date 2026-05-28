import shutil
from pathlib import Path
from fastapi import FastAPI
from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN, MODELS_ROOT

app = FastAPI(title="bsroformer-worker", version="0.1.0")


def prepare_input_folder(request: RunRequest) -> str:
    source_path = Path(request.input_path)
    if not source_path.is_file():
        return str(source_path.parent)

    isolated_root = Path(request.output_root) / "_inputs"
    isolated_root.mkdir(parents=True, exist_ok=True)
    staged_path = isolated_root / source_path.name
    if not staged_path.exists():
        shutil.copy2(source_path, staged_path)
    return str(isolated_root)


def build_command(request: RunRequest, input_folder: str) -> list[str]:
    model_slug = request.model or "roformer-model-bs-roformer-sw-by-jarredou"
    model_dir = MODELS_ROOT / model_slug
    config_path = model_dir / "model.yaml"
    checkpoint_path = model_dir / "model.ckpt"

    if request.task == "coarse_6stem":
        return [
            "bs-roformer-infer",
            "--config_path", str(config_path),
            "--model_path", str(checkpoint_path),
            "--input_folder", input_folder,
            "--store_dir", request.output_root,
        ]

    if request.task in {"vocals_resurrection", "instrumental_resurrection", "dereverb_cleanup", "guitar_refine_from_6stem", "piano_refine_from_6stem"}:
        return [
            "bs-roformer-infer",
            "--config_path", str(config_path),
            "--model_path", str(checkpoint_path),
            "--input_folder", input_folder,
            "--store_dir", request.output_root,
        ]

    return [
        "bs-roformer-infer",
        "--config_path", str(config_path),
        "--model_path", str(checkpoint_path),
        "--input_folder", input_folder,
        "--store_dir", request.output_root,
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "bsroformer-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "bsroformer-worker",
        "tasks": [
            "coarse_6stem",
            "vocals_resurrection",
            "instrumental_resurrection",
            "dereverb_cleanup",
            "guitar_refine_from_6stem",
            "piano_refine_from_6stem",
        ],
        "models": [
            "roformer-model-bs-roformer-sw-by-jarredou",
            "roformer-model-bs-roformer-vocals-resurrection-by-unwa",
            "roformer-model-bs-roformer-instrumental-resurrection-by-unwa",
            "roformer-model-bs-roformer-de-reverb",
        ],
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    input_folder = prepare_input_folder(request)
    command = build_command(request, input_folder)
    target = request.target_stem or request.task
    output_root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    output_root.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        service="bsroformer-worker",
        status="dry_run_completed" if request.dry_run else "todo_real_run",
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=[str(output_root / f"{target}.flac")],
        notes="Dry-run wrapper around BS-RoFormer-Infer. Map config/model paths to real downloaded checkpoints.",
    )
    return result.model_dump()
