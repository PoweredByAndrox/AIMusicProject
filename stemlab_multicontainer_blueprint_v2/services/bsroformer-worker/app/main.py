from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

from fastapi import FastAPI

from app.runtime import cache_snapshot, resolve_model_paths, run_inprocess_cached
from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(title="bsroformer-worker", version="0.3.0")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
MULTI_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 2


def output_root_for(request: RunRequest) -> Path:
    target = request.target_stem or request.task
    root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _needs_refresh(source: Path, target: Path) -> bool:
    if not target.exists():
        return True
    if not source.exists():
        return False
    try:
        source_stat = source.stat()
        target_stat = target.stat()
    except OSError:
        return True
    return (
        source_stat.st_mtime_ns > target_stat.st_mtime_ns
        or source_stat.st_size != target_stat.st_size
    )


def normalize_input_to_wav(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        "pcm_f32le",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _wav_matches_target(path: Path) -> bool:
    """Return True iff a .wav file is already at TARGET_SAMPLE_RATE/TARGET_CHANNELS.

    All four BS/Mel-Band Roformer checkpoints in this repo are trained on
    44.1 kHz stereo audio (see model yamls: ``audio.sample_rate: 44100``).
    Feeding 48 kHz or mono input directly into ``demix_track`` causes pitch
    shift and band-mismatch artefacts. Use ffmpeg path even for .wav inputs
    when the header does not match.
    """

    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getframerate() == TARGET_SAMPLE_RATE
                and handle.getnchannels() == TARGET_CHANNELS
            )
    except (wave.Error, EOFError, OSError):
        return False


def stage_single_input(request: RunRequest, output_root: Path) -> tuple[Path, Path]:
    input_path = Path(request.input_path)
    staged_dir = output_root / "_isolated_input"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_input = staged_dir / "input.wav"
    if input_path.exists() and input_path.is_file():
        needs_refresh = _needs_refresh(input_path, staged_input)
        if input_path.suffix.lower() == ".wav" and _wav_matches_target(input_path):
            if needs_refresh:
                shutil.copy2(input_path, staged_input)
        elif needs_refresh:
            normalize_input_to_wav(input_path, staged_input)
    return staged_dir, staged_input


def build_command(request: RunRequest, output_root: Path) -> tuple[list[str], Path]:
    model_slug = request.model or "roformer-model-bs-roformer-sw-by-jarredou"
    config_path, checkpoint_path = resolve_model_paths(model_slug)
    staged_dir, staged_input = stage_single_input(request, output_root)
    command = [
        "bsroformer-inprocess",
        "--config_path",
        str(config_path),
        "--model_path",
        str(checkpoint_path),
        "--input_folder",
        str(staged_dir),
        "--store_dir",
        str(output_root),
    ]
    return command, staged_input


def planned_outputs(request: RunRequest, output_root: Path) -> dict[str, str]:
    if request.task == "coarse_6stem":
        named = {stem: str(output_root / f"{stem}.wav") for stem in MULTI_STEMS}
        named["instrumental_mix"] = str(output_root / "instrumental_mix.wav")
        return named
    if request.task == "instrumental_resurrection":
        return {"instrumental_mix": str(output_root / "instrumental_mix.wav")}
    if request.task == "vocals_resurrection":
        return {"vocals": str(output_root / "vocals.wav")}
    if request.task == "dereverb_cleanup":
        target = request.target_stem or "vocals"
        return {target: str(output_root / f"{target}.wav")}
    target = request.target_stem or request.task
    return {target: str(output_root / f"{target}.wav")}


def find_generated_audio_files(output_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        if path.parent.name == "_isolated_input":
            continue
        candidates.append(path)
    return sorted(candidates)


def match_generated_file(files: list[Path], stem: str, request: RunRequest | None = None) -> Path | None:
    stem_l = stem.lower()
    aliases = [stem_l]
    if stem_l == "instrumental_mix":
        aliases.extend(["instrumental", "minus", "no_vocals", "karaoke"])
    if request and request.task == "dereverb_cleanup" and stem_l == "vocals":
        aliases = ["vocals_noreverb", "noreverb", "vocals"] + aliases

    # Respect alias priority instead of returning the first generic filename
    # that happens to contain "vocals". This is critical for dereverb paths
    # where we need `vocals_noreverb` before `vocals_instrumental`.
    for alias in aliases:
        for path in files:
            if path.stem.lower() == alias:
                return path
        for path in files:
            if alias in path.name.lower():
                return path
    return None


def normalize_audio_to_pcm(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        # Promote intermediate model outputs to 32-bit float WAV so the rest of
        # the tandem pipeline preserves the full numerical range produced by
        # the separator. R-PCM in the stem-tandem-improvement-roadmap.
        "-c:a",
        "pcm_f32le",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def materialize_outputs(request: RunRequest, output_root: Path, named_outputs: dict[str, str]) -> dict[str, str]:
    generated = find_generated_audio_files(output_root)
    materialized: dict[str, str] = {}

    for stem, target_str in named_outputs.items():
        source = match_generated_file(generated, stem, request=request)
        if source is None:
            continue
        target = Path(target_str)
        if source.resolve() != target.resolve():
            normalize_audio_to_pcm(source, target)
        materialized[stem] = str(target)

    if request.task == "coarse_6stem" and "instrumental_mix" not in materialized:
        source = match_generated_file(generated, "instrumental_mix", request=request)
        if source is not None:
            target = Path(named_outputs["instrumental_mix"])
            if source.resolve() != target.resolve():
                normalize_audio_to_pcm(source, target)
            materialized["instrumental_mix"] = str(target)

    return materialized


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "bsroformer-worker",
        "dry_run": DRY_RUN,
        "cache": cache_snapshot(),
    }


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
        "fixes": [
            "isolated single-file input folder instead of scanning /app/data/inputs",
            "deterministic named_outputs for gateway routing",
            "in-process cached model runtime instead of bs-roformer-infer subprocess",
            "mel-band-roformer checkpoints are detected and loaded with the correct model family",
        ],
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    output_root = output_root_for(request)
    command, staged_input = build_command(request, output_root)
    named_outputs = planned_outputs(request, output_root)
    metadata = {
        "input_mode": "isolated_single_file_folder",
        "staged_input": str(staged_input),
        "output_root": str(output_root),
        "runtime_mode": "inprocess_cached",
    }

    status = "dry_run_completed"
    notes = "BS-RoFormer request assembled with isolated single-file wav staging. The real runtime now uses an in-process cached model instead of a fresh subprocess load."
    if not request.dry_run:
        success, stdout, stderr, runtime_metadata = run_inprocess_cached(
            request.model or "roformer-model-bs-roformer-sw-by-jarredou",
            input_folder=staged_input.parent,
            store_dir=output_root,
        )
        metadata.update(runtime_metadata)
        metadata["stdout"] = stdout[-4000:]
        metadata["stderr"] = stderr[-4000:]
        status = "completed" if success else "failed"
        notes = "Real BS-RoFormer inference invoked in-process with a cached model runtime. Stable artifact keys are still materialized for the gateway/UI."
        if success:
            materialized = materialize_outputs(request, output_root, named_outputs)
            metadata["generated_audio_files"] = [str(path) for path in find_generated_audio_files(output_root)]
            metadata["materialized_outputs"] = materialized
            if materialized:
                named_outputs = materialized
                output_files = list(materialized.values())
            else:
                status = "failed"
                notes = "BS-RoFormer inference completed but no recognizable audio outputs were materialized into stable artifact keys."
                output_files = list(named_outputs.values())
        else:
            output_files = list(named_outputs.values())
    else:
        output_files = list(named_outputs.values())

    result = RunResult(
        service="bsroformer-worker",
        status=status,
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=output_files,
        named_outputs=named_outputs,
        metadata=metadata,
        notes=notes,
    )
    return result.model_dump()
