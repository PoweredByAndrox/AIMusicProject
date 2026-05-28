#!/usr/bin/env python3
"""Reference-trained, application-reference-free DRUM dictionary separator.

Training consumes verified isolated stems to build reusable spectral templates.
Application consumes only the full mix and the saved profile bank. This makes the
candidate suitable as a teacher-derived diagnostic route without hiding target
reference use at application time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
MODEL_ID = "teacher_dictionary_drum_separator_v1"
APPLICATION_MODEL_ID = "teacher_dictionary_drum_separator_application_v1"
EPSILON = 1.0e-9


@dataclass(frozen=True)
class DictionaryBuildConfig:
    n_fft: int = 2048
    hop_length: int = 512
    max_templates_per_stem: int = 32
    active_percentile: float = 75.0
    template_floor: float = 1.0e-7


@dataclass(frozen=True)
class DictionaryApplyConfig:
    mask_floor: float = 0.0
    mask_ceiling: float = 0.98
    temporal_smoothing_frames: int = 3
    max_peak: float = 0.999
    allow_missing_holdout: bool = False


def validate_stem_name(name: str) -> str:
    if not SAFE_STEM_NAME_PATTERN.match(name):
        raise ValueError(f"unsafe stem name: {name!r}")
    return name


def parse_named_path(raw_value: str) -> tuple[str, Path]:
    if "=" not in raw_value:
        raise ValueError(f"expected NAME=PATH, got {raw_value!r}")
    raw_name, raw_path = raw_value.split("=", 1)
    return validate_stem_name(raw_name.strip()), Path(raw_path.strip())


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError(f"invalid audio shape for {path}: {audio.shape}")
    return audio.astype(np.float64, copy=False), int(sample_rate)


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate, format="FLAC", subtype="PCM_24")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_build_config(config: DictionaryBuildConfig) -> None:
    if config.n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if config.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if config.max_templates_per_stem <= 0:
        raise ValueError("max_templates_per_stem must be positive")
    if not 0.0 <= config.active_percentile <= 100.0:
        raise ValueError("active_percentile must be in [0, 100]")
    if config.template_floor <= 0.0:
        raise ValueError("template_floor must be positive")


def validate_apply_config(config: DictionaryApplyConfig) -> None:
    if not 0.0 <= config.mask_floor <= 1.0:
        raise ValueError("mask_floor must be in [0, 1]")
    if not 0.0 <= config.mask_ceiling <= 1.0:
        raise ValueError("mask_ceiling must be in [0, 1]")
    if config.mask_floor > config.mask_ceiling:
        raise ValueError("mask_floor must be <= mask_ceiling")
    if config.temporal_smoothing_frames < 1:
        raise ValueError("temporal_smoothing_frames must be positive")
    if not 0.0 < config.max_peak <= 1.0:
        raise ValueError("max_peak must be in (0, 1]")


def mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1).astype(np.float64, copy=False)


def fit_audio_to_shortest(audio_by_name: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    frame_count = min(audio.shape[0] for audio in audio_by_name.values())
    channel_count = min(audio.shape[1] for audio in audio_by_name.values())
    if frame_count <= 0 or channel_count <= 0:
        raise ValueError("audio inputs must have positive shape")
    return {
        name: audio[:frame_count, :channel_count].astype(np.float64, copy=False)
        for name, audio in audio_by_name.items()
    }


def stft_magnitude(audio: np.ndarray, *, config: DictionaryBuildConfig) -> np.ndarray:
    spectrogram = compute_stft(mono(audio), config=config)
    return np.abs(spectrogram).astype(np.float64, copy=False)


def compute_stft(signal_audio: np.ndarray, *, config: DictionaryBuildConfig) -> np.ndarray:
    n_fft = int(config.n_fft)
    hop_length = int(config.hop_length)
    window = np.hanning(n_fft).astype(np.float64)
    centered = np.pad(signal_audio.astype(np.float64, copy=False), (n_fft // 2, n_fft // 2))
    if centered.shape[0] < n_fft:
        centered = np.pad(centered, (0, n_fft - centered.shape[0]))
    frame_count = 1 + int(np.ceil(max(0, centered.shape[0] - n_fft) / float(hop_length)))
    total_length = (frame_count - 1) * hop_length + n_fft
    if centered.shape[0] < total_length:
        centered = np.pad(centered, (0, total_length - centered.shape[0]))
    spectrogram = np.empty((n_fft // 2 + 1, frame_count), dtype=np.complex128)
    for frame_index in range(frame_count):
        start = frame_index * hop_length
        frame = centered[start:start + n_fft] * window
        spectrogram[:, frame_index] = np.fft.rfft(frame)
    return spectrogram


def inverse_stft(spectrogram: np.ndarray, *, config: DictionaryBuildConfig, length: int) -> np.ndarray:
    n_fft = int(config.n_fft)
    hop_length = int(config.hop_length)
    window = np.hanning(n_fft).astype(np.float64)
    frame_count = int(spectrogram.shape[1])
    total_length = (frame_count - 1) * hop_length + n_fft
    reconstructed = np.zeros(total_length, dtype=np.float64)
    window_power = np.zeros(total_length, dtype=np.float64)
    for frame_index in range(frame_count):
        start = frame_index * hop_length
        frame = np.fft.irfft(spectrogram[:, frame_index], n=n_fft)
        reconstructed[start:start + n_fft] += frame * window
        window_power[start:start + n_fft] += window * window
    nonzero = window_power > 1.0e-12
    reconstructed[nonzero] /= window_power[nonzero]
    trim_start = n_fft // 2
    trimmed = reconstructed[trim_start:trim_start + length]
    if trimmed.shape[0] < length:
        trimmed = np.pad(trimmed, (0, length - trimmed.shape[0]))
    return trimmed[:length].astype(np.float64, copy=False)


def select_template_frame_indices(magnitude: np.ndarray, *, config: DictionaryBuildConfig) -> np.ndarray:
    frame_energy = np.mean(magnitude, axis=0)
    if frame_energy.size == 0:
        raise ValueError("cannot build templates from empty magnitude")
    threshold = float(np.percentile(frame_energy, float(config.active_percentile)))
    active_indices = np.flatnonzero(frame_energy >= max(threshold, float(config.template_floor)))
    if active_indices.size == 0:
        active_indices = np.array([int(np.argmax(frame_energy))], dtype=np.int64)
    if active_indices.size <= int(config.max_templates_per_stem):
        return active_indices.astype(np.int64, copy=False)
    positions = np.linspace(0, active_indices.size - 1, int(config.max_templates_per_stem))
    return active_indices[np.round(positions).astype(np.int64)]


def build_templates_for_stem(
    stem_name: str,
    audio: np.ndarray,
    *,
    config: DictionaryBuildConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    magnitude = stft_magnitude(audio, config=config)
    selected_indices = select_template_frame_indices(magnitude, config=config)
    templates: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for template_index, frame_index in enumerate(selected_indices):
        template = magnitude[:, int(frame_index)].astype(np.float64, copy=True)
        norm = float(np.linalg.norm(template))
        if norm <= float(config.template_floor):
            continue
        templates.append(template / norm)
        metadata.append(
            {
                "template_index": int(template_index),
                "stem_name": stem_name,
                "source_frame_index": int(frame_index),
                "l2_norm_before_normalize": round(norm, 12),
            }
        )
    if not templates:
        raise ValueError(f"no usable templates for stem: {stem_name}")
    return np.stack(templates, axis=1), metadata


def build_teacher_dictionary_profile(
    *,
    target_stem_name: str,
    target_reference_path: Path,
    non_target_reference_paths: dict[str, Path],
    output_profile_bank_path: Path,
    config: DictionaryBuildConfig,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    validate_build_config(config)
    target_stem_name = validate_stem_name(target_stem_name)
    if target_stem_name in non_target_reference_paths:
        raise ValueError("non-target references must not include target stem")
    loaded: dict[str, tuple[np.ndarray, int, Path]] = {
        target_stem_name: (*read_audio(target_reference_path), target_reference_path),
    }
    for stem_name, stem_path in sorted(non_target_reference_paths.items()):
        loaded[validate_stem_name(stem_name)] = (*read_audio(stem_path), stem_path)
    sample_rates = {sample_rate for _, sample_rate, _ in loaded.values()}
    if len(sample_rates) != 1:
        raise ValueError(f"sample-rate mismatch: {sorted(sample_rates)}")
    sample_rate = int(next(iter(sample_rates)))
    aligned_audio = fit_audio_to_shortest({stem_name: audio for stem_name, (audio, _, _) in loaded.items()})
    dictionary_blocks: list[np.ndarray] = []
    labels: list[str] = []
    template_metadata: list[dict[str, Any]] = []
    for stem_name in [target_stem_name, *sorted(non_target_reference_paths)]:
        templates, metadata = build_templates_for_stem(stem_name, aligned_audio[stem_name], config=config)
        dictionary_blocks.append(templates)
        labels.extend([stem_name] * templates.shape[1])
        template_metadata.extend(metadata)
    dictionary_matrix = np.concatenate(dictionary_blocks, axis=1).astype(np.float32, copy=False)
    profile_bank_path = output_profile_bank_path
    npz_path = profile_bank_path.with_suffix(".npz")
    profile_bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        dictionary_matrix=dictionary_matrix,
        template_labels=np.array(labels),
    )
    profile_bank = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_name": "Teacher dictionary DRUM separator profile bank",
        "contract": {
            "profile_source_reference_assisted": True,
            "application_reference_free": True,
            "target_reference_used_at_application": False,
            "requires_holdout_validation": True,
            "holdout_validation_status": "missing",
            "delivery_allowed_without_reference": False,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "stores_timestamp_rules": False,
        },
        "target_stem_name": target_stem_name,
        "sample_rate": sample_rate,
        "dictionary": {
            "npz_path": str(npz_path),
            "frequency_bins": int(dictionary_matrix.shape[0]),
            "template_count": int(dictionary_matrix.shape[1]),
            "target_template_count": int(sum(1 for label in labels if label == target_stem_name)),
            "non_target_template_count": int(sum(1 for label in labels if label != target_stem_name)),
            "template_metadata": template_metadata,
        },
        "config": {
            "n_fft": int(config.n_fft),
            "hop_length": int(config.hop_length),
            "max_templates_per_stem": int(config.max_templates_per_stem),
            "active_percentile": float(config.active_percentile),
            "template_floor": float(config.template_floor),
        },
        "inputs": {
            "target_reference": audio_summary(target_reference_path),
            "non_target_references": {
                stem_name: audio_summary(stem_path)
                for stem_name, stem_path in sorted(non_target_reference_paths.items())
            },
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    write_json(profile_bank_path, profile_bank)
    return profile_bank


def load_dictionary(profile_bank: dict[str, Any], profile_bank_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_npz_path = Path(profile_bank["dictionary"]["npz_path"])
    npz_path = raw_npz_path if raw_npz_path.is_absolute() else profile_bank_path.parent / raw_npz_path
    with np.load(str(npz_path), allow_pickle=False) as payload:
        dictionary_matrix = payload["dictionary_matrix"].astype(np.float64, copy=False)
        template_labels = payload["template_labels"].astype(str)
    if dictionary_matrix.ndim != 2 or dictionary_matrix.shape[1] != template_labels.shape[0]:
        raise ValueError("invalid dictionary profile bank layout")
    return dictionary_matrix, template_labels


def smooth_mask(mask: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return mask
    kernel = np.ones(int(frames), dtype=np.float64) / float(frames)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=mask)


def build_dictionary_mask(
    mix_magnitude: np.ndarray,
    dictionary_matrix: np.ndarray,
    template_labels: np.ndarray,
    *,
    target_stem_name: str,
    config: DictionaryApplyConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    validate_apply_config(config)
    if mix_magnitude.shape[0] != dictionary_matrix.shape[0]:
        raise ValueError(
            f"dictionary frequency bins do not match mix: {dictionary_matrix.shape[0]} != {mix_magnitude.shape[0]}"
        )
    pseudo_inverse = np.linalg.pinv(dictionary_matrix, rcond=1.0e-4)
    activations = np.maximum(0.0, pseudo_inverse @ mix_magnitude)
    target_indices = np.flatnonzero(template_labels == target_stem_name)
    if target_indices.size == 0:
        raise ValueError(f"profile bank has no templates for target stem: {target_stem_name}")
    estimated_total = np.maximum(0.0, dictionary_matrix @ activations)
    target_dictionary = dictionary_matrix[:, target_indices]
    target_activations = activations[target_indices, :]
    estimated_target = np.maximum(0.0, target_dictionary @ target_activations)
    mask = estimated_target / (estimated_total + EPSILON)
    mask = smooth_mask(mask, int(config.temporal_smoothing_frames))
    mask = np.clip(mask, float(config.mask_floor), float(config.mask_ceiling))
    diagnostics = {
        "mean_mask": round(float(np.mean(mask)), 9),
        "p95_mask": round(float(np.percentile(mask, 95)), 9),
        "target_template_count": int(target_indices.size),
        "non_target_template_count": int(template_labels.size - target_indices.size),
        "activation_mean": round(float(np.mean(activations)), 12),
    }
    return mask.astype(np.float64, copy=False), diagnostics


def apply_dictionary_to_mix(
    source_mix_audio: np.ndarray,
    *,
    sample_rate: int,
    dictionary_matrix: np.ndarray,
    template_labels: np.ndarray,
    target_stem_name: str,
    build_config: DictionaryBuildConfig,
    apply_config: DictionaryApplyConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    mix_mono_stft = compute_stft(mono(source_mix_audio), config=build_config)
    mask, mask_diagnostics = build_dictionary_mask(
        np.abs(mix_mono_stft).astype(np.float64, copy=False),
        dictionary_matrix,
        template_labels,
        target_stem_name=target_stem_name,
        config=apply_config,
    )
    reconstructed_channels: list[np.ndarray] = []
    for channel_index in range(source_mix_audio.shape[1]):
        channel_stft = compute_stft(source_mix_audio[:, channel_index], config=build_config)
        reconstructed = inverse_stft(
            channel_stft * mask,
            config=build_config,
            length=int(source_mix_audio.shape[0]),
        )
        reconstructed_channels.append(reconstructed.astype(np.float64, copy=False))
    output = np.stack(reconstructed_channels, axis=1)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    peak_limited = False
    if peak > float(apply_config.max_peak):
        output *= float(apply_config.max_peak) / peak
        peak_limited = True
    diagnostics = {
        **mask_diagnostics,
        "peak_before_limit": round(peak, 9),
        "peak_limited": peak_limited,
        "max_peak": float(apply_config.max_peak),
    }
    return output.astype(np.float32, copy=False), diagnostics


def apply_teacher_dictionary_profile(
    *,
    profile_bank_path: Path,
    source_mix_path: Path,
    output_dir: Path,
    listen_dir: Path | None,
    apply_config: DictionaryApplyConfig,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    validate_apply_config(apply_config)
    profile_bank = read_json(profile_bank_path)
    profile_contract = profile_bank.get("contract", {})
    holdout_status = str(profile_contract.get("holdout_validation_status", "missing"))
    if holdout_status != "passed" and not bool(apply_config.allow_missing_holdout):
        raise ValueError(
            "teacher dictionary profile bank requires passed holdout validation before application; "
            "use --allow-missing-holdout only for diagnostic artifact generation"
        )
    target_stem_name = validate_stem_name(str(profile_bank["target_stem_name"]))
    build_config_payload = profile_bank["config"]
    build_config = DictionaryBuildConfig(
        n_fft=int(build_config_payload["n_fft"]),
        hop_length=int(build_config_payload["hop_length"]),
        max_templates_per_stem=int(build_config_payload["max_templates_per_stem"]),
        active_percentile=float(build_config_payload["active_percentile"]),
        template_floor=float(build_config_payload["template_floor"]),
    )
    source_mix_audio, sample_rate = read_audio(source_mix_path)
    if int(profile_bank["sample_rate"]) != int(sample_rate):
        raise ValueError("profile bank sample rate does not match source mix")
    dictionary_matrix, template_labels = load_dictionary(profile_bank, profile_bank_path)
    output_audio, diagnostics = apply_dictionary_to_mix(
        source_mix_audio,
        sample_rate=sample_rate,
        dictionary_matrix=dictionary_matrix,
        template_labels=template_labels,
        target_stem_name=target_stem_name,
        build_config=build_config,
        apply_config=apply_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"01_teacher_dictionary_{target_stem_name}.flac"
    write_flac(output_path, output_audio, sample_rate)
    manifest = {
        "schema_version": 1,
        "model_id": APPLICATION_MODEL_ID,
        "model_name": "Teacher dictionary DRUM separator application",
        "model": {"model_id": APPLICATION_MODEL_ID, "family": "teacher_dictionary_separator"},
        "contract": {
            "benchmark_compatible": True,
            "full_track": True,
            "preview": False,
            "fallback_used": False,
            "target_reference_assisted": False,
            "profile_source_reference_assisted": bool(
                profile_bank.get("contract", {}).get("profile_source_reference_assisted")
            ),
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "quality_gate_required": True,
            "requires_holdout_validation": True,
            "holdout_validation_status": profile_bank.get("contract", {}).get("holdout_validation_status", "missing"),
            "delivery_allowed_without_reference": False,
            "not_a_no_reference_production_route_until_holdout_passes": True,
            "per_track_hardcode_allowed": False,
            "stores_timestamp_rules": False,
        },
        "profile_application": {
            "profile_bank_path": str(profile_bank_path),
            "uses_reference_during_application": False,
            "rule_scope": "teacher_dictionary_spectral_templates",
        },
        "inputs": {
            "source_mix": audio_summary(source_mix_path),
            "profile_bank_path": str(profile_bank_path),
        },
        "config": {
            "mask_floor": float(apply_config.mask_floor),
            "mask_ceiling": float(apply_config.mask_ceiling),
            "temporal_smoothing_frames": int(apply_config.temporal_smoothing_frames),
            "max_peak": float(apply_config.max_peak),
            "allow_missing_holdout": bool(apply_config.allow_missing_holdout),
        },
        "diagnostics": diagnostics,
        "stems": {
            target_stem_name: {
                **audio_summary(output_path),
                "role": "teacher_dictionary_application_stem",
                "profile_source_reference_assisted": True,
                "target_reference_used_at_application": False,
            }
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    write_json(output_dir / "stem_manifest.json", manifest)
    write_json(output_dir / "teacher_dictionary_drum_separator_manifest.json", manifest)
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, listen_dir / f"01_teacher_dictionary_{target_stem_name}.flac")
        write_json(
            listen_dir / "candidate_map.json",
            {"01_teacher_dictionary": str(listen_dir / f"01_teacher_dictionary_{target_stem_name}.flac")},
        )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/apply a reference-trained dictionary DRUM separator without target reference during application."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-bank")
    build.add_argument("--target-stem-name", default="drums")
    build.add_argument("--target-reference", required=True, type=Path)
    build.add_argument("--non-target-reference", action="append", default=[], help="NAME=PATH. Repeatable.")
    build.add_argument("--output-profile-bank", required=True, type=Path)
    build.add_argument("--n-fft", type=int, default=DictionaryBuildConfig.n_fft)
    build.add_argument("--hop-length", type=int, default=DictionaryBuildConfig.hop_length)
    build.add_argument("--max-templates-per-stem", type=int, default=DictionaryBuildConfig.max_templates_per_stem)
    build.add_argument("--active-percentile", type=float, default=DictionaryBuildConfig.active_percentile)
    build.add_argument("--template-floor", type=float, default=DictionaryBuildConfig.template_floor)

    apply = subparsers.add_parser("apply-bank")
    apply.add_argument("--profile-bank", required=True, type=Path)
    apply.add_argument("--source-mix", required=True, type=Path)
    apply.add_argument("--output-dir", required=True, type=Path)
    apply.add_argument("--listen-dir", type=Path)
    apply.add_argument("--mask-floor", type=float, default=DictionaryApplyConfig.mask_floor)
    apply.add_argument("--mask-ceiling", type=float, default=DictionaryApplyConfig.mask_ceiling)
    apply.add_argument("--temporal-smoothing-frames", type=int, default=DictionaryApplyConfig.temporal_smoothing_frames)
    apply.add_argument("--max-peak", type=float, default=DictionaryApplyConfig.max_peak)
    apply.add_argument("--allow-missing-holdout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="teacher_dictionary_drum_separator.py",
        entrypoint_path="/app/tools/teacher_dictionary_drum_separator.py",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raw_args = sys.argv[1:] if argv is None else argv
    if args.command == "build-bank":
        profile_bank = build_teacher_dictionary_profile(
            target_stem_name=str(args.target_stem_name),
            target_reference_path=args.target_reference,
            non_target_reference_paths=dict(parse_named_path(raw) for raw in args.non_target_reference),
            output_profile_bank_path=args.output_profile_bank,
            config=DictionaryBuildConfig(
                n_fft=int(args.n_fft),
                hop_length=int(args.hop_length),
                max_templates_per_stem=int(args.max_templates_per_stem),
                active_percentile=float(args.active_percentile),
                template_floor=float(args.template_floor),
            ),
            command_args=raw_args,
        )
        print(json.dumps({"profile_bank_path": str(args.output_profile_bank), "contract": profile_bank["contract"]}, indent=2))
        return 0
    manifest = apply_teacher_dictionary_profile(
        profile_bank_path=args.profile_bank,
        source_mix_path=args.source_mix,
        output_dir=args.output_dir,
        listen_dir=args.listen_dir,
        apply_config=DictionaryApplyConfig(
            mask_floor=float(args.mask_floor),
            mask_ceiling=float(args.mask_ceiling),
            temporal_smoothing_frames=int(args.temporal_smoothing_frames),
            max_peak=float(args.max_peak),
            allow_missing_holdout=bool(args.allow_missing_holdout),
        ),
        command_args=raw_args,
    )
    print(json.dumps({"manifest_path": str(args.output_dir / "stem_manifest.json"), "contract": manifest["contract"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
