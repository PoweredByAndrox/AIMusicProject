#!/usr/bin/env python3
"""Reference-assisted DRUM teacher reconstruction from a full mix.

This is not a no-reference production separator. It uses the verified target
DRUM reference at application time to create a teacher/oracle artifact and to
measure how far the current model stack is from the requested stem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


SAFE_STEM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
EPSILON = 1.0e-12
MODEL_ID = "reference_guided_drum_reconstructor"
MODEL_NAME = "Reference-guided DRUM reconstructor"


@dataclass(frozen=True)
class ProjectionGainSolution:
    gains: dict[str, float]
    solve_method: str
    fallback_used: bool


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError(f"invalid audio shape for {path}: {audio.shape}")
    return audio.astype(np.float64, copy=False), int(sample_rate)


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate, format="FLAC", subtype="PCM_24")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stem_name(name: str) -> str:
    if not SAFE_STEM_NAME_PATTERN.match(name):
        raise ValueError(f"unsafe stem name: {name!r}")
    return name


def canonical_stem_name(stem_name: str) -> str:
    canonical = "".join(character for character in stem_name.lower() if character.isalnum())
    if canonical.endswith("s"):
        canonical = canonical[:-1]
    return canonical


def parse_named_path(raw_value: str) -> tuple[str, Path]:
    if "=" not in raw_value:
        raise ValueError(f"expected NAME=PATH, got {raw_value!r}")
    raw_name, raw_path = raw_value.split("=", 1)
    return validate_stem_name(raw_name.strip()), Path(raw_path.strip())


def validate_non_target_reference_paths(non_target_reference_paths: dict[str, Path], *, stem_name: str) -> None:
    target_name = canonical_stem_name(stem_name)
    for source_name in non_target_reference_paths:
        if canonical_stem_name(source_name) == target_name:
            raise ValueError(f"non-target references must not include target stem: {source_name}")


def fit_audio_arrays(audio_arrays: list[np.ndarray]) -> list[np.ndarray]:
    if not audio_arrays:
        raise ValueError("audio_arrays cannot be empty")
    frame_count = min(audio.shape[0] for audio in audio_arrays)
    channel_count = min(audio.shape[1] for audio in audio_arrays)
    if frame_count <= 0 or channel_count <= 0:
        raise ValueError("audio arrays must have positive frame and channel counts")
    return [audio[:frame_count, :channel_count].astype(np.float64, copy=False) for audio in audio_arrays]


def validate_sample_rates(named_sample_rates: dict[str, int]) -> int:
    sample_rates = set(named_sample_rates.values())
    if len(sample_rates) != 1:
        raise ValueError(f"sample-rate mismatch: {named_sample_rates}")
    return int(next(iter(sample_rates)))


def solve_projection_gain_solution(
    *,
    source_mix_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    non_target_reference_stems: dict[str, np.ndarray],
    stem_name: str,
    ridge: float,
) -> ProjectionGainSolution:
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    validate_non_target_reference_paths({name: Path(name) for name in non_target_reference_stems}, stem_name=stem_name)
    source_names = [stem_name, *sorted(non_target_reference_stems)]
    fitted = fit_audio_arrays([source_mix_audio, target_reference_audio, *[non_target_reference_stems[name] for name in sorted(non_target_reference_stems)]])
    source_mix = fitted[0]
    source_vectors = [fitted[1], *fitted[2:]]
    matrix_size = len(source_vectors)
    gram = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    target = np.zeros(matrix_size, dtype=np.float64)
    mix_flat = source_mix.reshape(-1)
    flat_sources = [audio.reshape(-1) for audio in source_vectors]
    for row_index, row_source in enumerate(flat_sources):
        target[row_index] = float(np.dot(row_source, mix_flat))
        for column_index, column_source in enumerate(flat_sources):
            gram[row_index, column_index] = float(np.dot(row_source, column_source))
    if ridge > 0.0:
        gram += np.eye(matrix_size, dtype=np.float64) * ridge
    try:
        gains = np.linalg.solve(gram, target)
        solve_method = "linear_solve"
        fallback_used = False
    except np.linalg.LinAlgError:
        gains, *_ = np.linalg.lstsq(gram, target, rcond=None)
        solve_method = "least_squares_after_singular_matrix"
        fallback_used = True
    return ProjectionGainSolution(
        gains={source_name: float(gain) for source_name, gain in zip(source_names, gains)},
        solve_method=solve_method,
        fallback_used=fallback_used,
    )


def solve_projection_gains(
    *,
    source_mix_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    non_target_reference_stems: dict[str, np.ndarray],
    stem_name: str,
    ridge: float,
) -> dict[str, float]:
    return solve_projection_gain_solution(
        source_mix_audio=source_mix_audio,
        target_reference_audio=target_reference_audio,
        non_target_reference_stems=non_target_reference_stems,
        stem_name=stem_name,
        ridge=ridge,
    ).gains


def build_reference_locked_teacher(
    *,
    target_reference_audio: np.ndarray,
    target_projection_gain: float,
    reference_lock_strength: float,
) -> np.ndarray:
    if not 0.0 <= reference_lock_strength <= 1.0:
        raise ValueError("reference_lock_strength must be in [0, 1]")
    projected = target_projection_gain * target_reference_audio
    locked = target_reference_audio
    return ((1.0 - reference_lock_strength) * projected) + (reference_lock_strength * locked)


def peak_limit(audio: np.ndarray, max_peak: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 < max_peak <= 1.0:
        raise ValueError("max_peak must be in (0, 1]")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    limited = False
    output = audio.astype(np.float64, copy=True)
    if peak > max_peak:
        output *= max_peak / peak
        limited = True
    return output, {"peak_before_limit": round(peak, 9), "peak_limited": limited, "max_peak": float(max_peak)}


def audio_summary(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def build_reference_guided_drum_reconstruction(
    *,
    source_mix_path: Path,
    target_reference_path: Path,
    non_target_reference_paths: dict[str, Path],
    output_dir: Path,
    listen_dir: Path | None,
    stem_name: str,
    reference_lock_strength: float,
    ridge: float,
    max_peak: float = 0.999,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem_name = validate_stem_name(stem_name)
    validate_non_target_reference_paths(non_target_reference_paths, stem_name=stem_name)
    loaded: dict[str, tuple[np.ndarray, int]] = {
        "source_mix": read_audio(source_mix_path),
        "target_reference": read_audio(target_reference_path),
    }
    for source_name, source_path in non_target_reference_paths.items():
        loaded[source_name] = read_audio(source_path)
    sample_rate = validate_sample_rates({name: sample_rate for name, (_, sample_rate) in loaded.items()})
    target_reference_full_audio = loaded["target_reference"][0]
    source_mix_audio, target_reference_analysis_audio, *non_target_audio = fit_audio_arrays(
        [loaded["source_mix"][0], loaded["target_reference"][0], *[loaded[name][0] for name in sorted(non_target_reference_paths)]]
    )
    fitted_non_targets = {
        name: audio
        for name, audio in zip(sorted(non_target_reference_paths), non_target_audio)
    }
    projection_solution = solve_projection_gain_solution(
        source_mix_audio=source_mix_audio,
        target_reference_audio=target_reference_analysis_audio,
        non_target_reference_stems=fitted_non_targets,
        stem_name=stem_name,
        ridge=ridge,
    )
    projection_gains = projection_solution.gains
    teacher_audio = build_reference_locked_teacher(
        target_reference_audio=target_reference_full_audio,
        target_projection_gain=projection_gains[stem_name],
        reference_lock_strength=reference_lock_strength,
    )
    exact_reference_copy = reference_lock_strength >= 1.0
    if exact_reference_copy:
        limiting = {
            "peak_before_limit": round(float(np.max(np.abs(target_reference_full_audio))) if target_reference_full_audio.size else 0.0, 9),
            "peak_limited": False,
            "max_peak": float(max_peak),
            "exact_reference_copy": True,
        }
    else:
        teacher_audio, limiting = peak_limit(teacher_audio, max_peak)
        limiting["exact_reference_copy"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"02_reference_locked_teacher_{stem_name}.flac"
    projection_path = output_dir / f"01_projection_scaled_teacher_{stem_name}.flac"
    write_flac(projection_path, projection_gains[stem_name] * target_reference_full_audio, sample_rate)
    if exact_reference_copy:
        shutil.copy2(target_reference_path, output_path)
    else:
        write_flac(output_path, teacher_audio, sample_rate)
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        write_flac(listen_dir / f"01_projection_scaled_teacher_{stem_name}.flac", projection_gains[stem_name] * target_reference_full_audio, sample_rate)
        if exact_reference_copy:
            shutil.copy2(target_reference_path, listen_dir / f"02_reference_locked_teacher_{stem_name}.flac")
        else:
            write_flac(listen_dir / f"02_reference_locked_teacher_{stem_name}.flac", teacher_audio, sample_rate)
        write_json(
            listen_dir / "candidate_map.json",
            {
                "01_projection_scaled_teacher": str(listen_dir / f"01_projection_scaled_teacher_{stem_name}.flac"),
                "02_reference_locked_teacher": str(listen_dir / f"02_reference_locked_teacher_{stem_name}.flac"),
            },
        )
    output_summary = audio_summary(output_path)
    projection_summary = audio_summary(projection_path)
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "target_reference_assisted": True,
            "target_reference_used_at_application": True,
            "application_reference_free": False,
            "delivery_allowed_without_reference": False,
            "no_reference_production_claim": False,
            "delivery_allowed_without_quality_gate": False,
            "fallback_used": projection_solution.fallback_used,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "route": {
            "type": "reference_guided_drum_reconstruction",
            "primary_output": "reference_locked_teacher",
            "interpretation": "teacher_oracle_for_training_and_debugging_not_no_reference_production",
        },
        "inputs": {
            "source_mix": audio_summary(source_mix_path),
            "target_reference": audio_summary(target_reference_path),
            "non_target_references": {
                name: audio_summary(path)
                for name, path in sorted(non_target_reference_paths.items())
            },
        },
        "config": {
            "stem_name": stem_name,
            "reference_lock_strength": float(reference_lock_strength),
            "ridge": float(ridge),
            "max_peak": float(max_peak),
        },
        "projection_gains": {name: round(float(gain), 12) for name, gain in projection_gains.items()},
        "diagnostics": {
            "sample_rate": sample_rate,
            "analysis_frames": int(target_reference_analysis_audio.shape[0]),
            "analysis_channels": int(target_reference_analysis_audio.shape[1]),
            "frames": int(output_summary["frames"]),
            "channels": int(output_summary["channels"]),
            "projection_solve_method": projection_solution.solve_method,
            **limiting,
        },
        "diagnostic_outputs": {
            "projection_scaled_teacher": projection_summary,
        },
        "stems": {
            stem_name: output_summary,
        },
    }
    write_json(output_dir / "stem_manifest.json", manifest)
    write_json(output_dir / "reference_guided_drum_reconstructor_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reference-assisted DRUM teacher reconstruction.")
    parser.add_argument("--source-mix", required=True, type=Path)
    parser.add_argument("--target-reference", required=True, type=Path)
    parser.add_argument("--non-target-reference", action="append", default=[], help="NAME=PATH. Repeatable.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--stem-name", default="drums")
    parser.add_argument("--reference-lock-strength", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--max-peak", type=float, default=0.999)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="reference_guided_drum_reconstructor.py",
        entrypoint_path="/app/tools/reference_guided_drum_reconstructor.py",
    )
    args = parse_args(argv)
    non_target_reference_paths = dict(parse_named_path(raw) for raw in args.non_target_reference)
    manifest = build_reference_guided_drum_reconstruction(
        source_mix_path=args.source_mix,
        target_reference_path=args.target_reference,
        non_target_reference_paths=non_target_reference_paths,
        output_dir=args.output_dir,
        listen_dir=args.listen_dir,
        stem_name=str(args.stem_name),
        reference_lock_strength=float(args.reference_lock_strength),
        ridge=float(args.ridge),
        max_peak=float(args.max_peak),
        command_args=sys.argv[1:] if argv is None else argv,
    )
    print(json.dumps({"output": manifest["stems"][str(args.stem_name)], "projection_gains": manifest["projection_gains"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
