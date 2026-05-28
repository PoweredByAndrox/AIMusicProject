from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import FastAPI

from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(
    title="audio-separator-worker",
    version="0.4.0",
    description=(
        "Heuristic bus-mixer (NOT the audio-separator PyPI package). "
        "Performs ffmpeg-based stem aggregation and per-bus EQ. "
        "The honest functional name is bus-mixer-worker; the legacy slug "
        "is kept for back-compat. See R12 in stem-tandem-improvement-roadmap."
    ),
)

PRESET_FILTERS = {
    "instrumental_balanced": "highpass=f=110,lowpass=f=14000,equalizer=f=2600:t=o:w=1.2:g=1.2,alimiter=limit=0.97",
    "instrumental_clean": "highpass=f=130,lowpass=f=12500,equalizer=f=3000:t=o:w=1.2:g=1.6,alimiter=limit=0.97",
    "instrumental_full": "highpass=f=90,lowpass=f=15000,equalizer=f=2400:t=o:w=1.3:g=1.0,alimiter=limit=0.97",
    "vocal_balanced": "highpass=f=140,lowpass=f=12000,equalizer=f=3200:t=o:w=1.2:g=1.8,alimiter=limit=0.97",
}

INPUT_WEIGHTS = {
    "other": 1.0,
    "guitar": 0.95,
    "piano": 0.95,
}


def output_root_for(request: RunRequest) -> Path:
    target = request.target_stem or request.task
    root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def output_path_for(request: RunRequest, output_root: Path) -> Path:
    target = request.target_stem or request.task
    return output_root / f"{target}.wav"


def resolve_recipe(request: RunRequest) -> str:
    candidate = request.preset or "instrumental_balanced"
    return candidate if candidate in PRESET_FILTERS else "instrumental_balanced"


def collect_source_inputs(request: RunRequest) -> list[tuple[str, str]]:
    if request.build_from_paths:
        pairs = [(key, path) for key, path in request.build_from_paths.items() if Path(path).exists()]
        if pairs:
            return pairs

    input_path = Path(request.input_path)
    if input_path.exists():
        return [("input", request.input_path)]
    return []


def build_command(request: RunRequest, output_root: Path) -> tuple[list[str], Path, str, list[tuple[str, str]]]:
    output_path = output_path_for(request, output_root)
    recipe = resolve_recipe(request)
    source_inputs = collect_source_inputs(request)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    for _, source_path in source_inputs:
        command.extend(["-i", source_path])

    filter_chain = PRESET_FILTERS[recipe]
    if len(source_inputs) <= 1:
        command.extend(
            [
                "-vn",
                "-af",
                filter_chain,
                # 32-bit float preserves the headroom we need for downstream
                # ML refinement steps (R-PCM in the tandem-improvement audit).
                "-c:a",
                "pcm_f32le",
                str(output_path),
            ]
        )
        return command, output_path, recipe, source_inputs

    mix_labels = "".join(f"[{index}:a]" for index in range(len(source_inputs)))
    weights = " ".join(f"{INPUT_WEIGHTS.get(key, 1.0):.2f}" for key, _ in source_inputs)
    filter_complex = f"{mix_labels}amix=inputs={len(source_inputs)}:weights='{weights}':normalize=0,{filter_chain}[out]"
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_f32le",
            str(output_path),
        ]
    )
    return command, output_path, recipe, source_inputs


def planned_outputs(request: RunRequest, output_root: Path) -> dict[str, str]:
    target = request.target_stem or request.task
    return {target: str(output_path_for(request, output_root))}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "audio-separator-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "audio-separator-worker",
        "honest_name": "bus-mixer-worker",
        "package_status": (
            "audio-separator PyPI package is NOT installed in this image; the "
            "tasks below are pure ffmpeg bus assembly + EQ, not learned "
            "separation. R12 of the tandem-improvement roadmap tracks the "
            "two follow-ups: either pip install the real package or finish "
            "the rename to bus-mixer-worker."
        ),
        "tasks": ["harmonic_bus_cleanup", "ensemble_cleanup"],
        "presets": sorted(PRESET_FILTERS.keys()),
        "state": (
            "deterministic ffmpeg-based bus assembly and EQ - heuristic, "
            "not ML separation. For per-stem ensemble use restoration-worker "
            "stem_ensemble (R9); for soft-mask post-filtering use "
            "restoration-worker wiener_postfilter (R8)."
        ),
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    output_root = output_root_for(request)
    command, output_path, recipe, source_inputs = build_command(request, output_root)
    named_outputs = planned_outputs(request, output_root)

    if not source_inputs:
        result = RunResult(
            service="audio-separator-worker",
            status="failed",
            task=request.task,
            target_stem=request.target_stem,
            command=command,
            output_files=list(named_outputs.values()),
            named_outputs={},
            metadata={"recipe": recipe, "output_root": str(output_root)},
            notes="No valid source inputs were available for harmonic bus assembly.",
        )
        return result.model_dump()

    metadata = {
        "recipe": recipe,
        "filter_chain": PRESET_FILTERS[recipe],
        "source_inputs": [{"stem": key, "path": path} for key, path in source_inputs],
        "output_root": str(output_root),
    }
    status = "dry_run_completed"
    notes = "Dry-run audio-separator command assembled."

    if not request.dry_run:
        completed = subprocess.run(command, capture_output=True, text=True)
        metadata["stdout"] = completed.stdout[-4000:]
        metadata["stderr"] = completed.stderr[-4000:]
        status = "completed" if completed.returncode == 0 and output_path.exists() else "failed"
        notes = (
            "ffmpeg-based bus assembly and EQ completed. "
            "NOTE: this worker is NOT the audio-separator PyPI package; "
            "it is a deterministic heuristic mixer (honest name: bus-mixer-worker). "
            "For learned per-stem ensemble use restoration-worker stem_ensemble (R9)."
        )

    result = RunResult(
        service="audio-separator-worker",
        status=status,
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=list(named_outputs.values()),
        named_outputs=named_outputs if status == "completed" else {},
        metadata=metadata,
        notes=notes,
    )
    return result.model_dump()
