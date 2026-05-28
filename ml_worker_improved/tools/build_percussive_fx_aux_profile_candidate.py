#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_percussive_fx_aux_candidate import (  # noqa: E402
    _fit_audio,
    _load_json,
    _peak_dbfs,
    _read_audio,
    _rms_dbfs,
    _safe_listen_path,
    _sha256_file,
    _stem_path_from_manifest,
    _write_flac,
    _write_json,
)
from docker_runtime import require_docker_runtime  # noqa: E402


TARGET_STEM = "drums_plus_percussive_fx_aux"
AUX_STEM = "percussive_fx_aux"
DEFAULT_ROUTE_NAME = "percussive_fx_aux_profile_detector_v1"
EPSILON = 1.0e-12


@dataclass(frozen=True)
class ProfilePercussiveAuxConfig:
    frame_size: int = 2048
    hop_size: int = 512
    min_profile_similarity: float = 0.22
    min_flatness: float = 0.16
    flux_percentile: float = 70.0
    activity_floor_ratio: float = 0.02
    pre_roll_frames: int = 1
    tail_frames: int = 4
    aux_gain: float = 0.85
    max_aux_rms_ratio: float = 0.65
    limiter_peak: float = 0.98
    spectral_mask_floor: float = 0.35
    spectral_mask_power: float = 1.35
    spectral_mask_low_percentile: float = 35.0
    spectral_mask_high_percentile: float = 98.0
    tonal_peak_ratio: float = 4.0
    tonal_suppression_floor: float = 0.08


def _safe_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio * audio) + EPSILON))


def _manifest_model_id(manifest: dict[str, Any]) -> str:
    model_id = manifest.get("model_id") or manifest.get("model_name")
    if not model_id and isinstance(manifest.get("model"), dict):
        model_id = manifest["model"].get("model_id") or manifest["model"].get("base_model")
    return str(model_id or "unknown_model")


def _read_base_audio(base_manifest_path: Path, base_manifest: dict[str, Any]) -> tuple[np.ndarray, int, dict[str, Any]]:
    stem_path = _stem_path_from_manifest(base_manifest, base_manifest_path, TARGET_STEM)
    if stem_path is None:
        stem_path = _stem_path_from_manifest(base_manifest, base_manifest_path, "drums")
    if stem_path is None:
        raise ValueError(f"base manifest does not contain {TARGET_STEM} or drums")
    audio, sample_rate = _read_audio(stem_path)
    return audio, sample_rate, {
        "model_id": _manifest_model_id(base_manifest),
        "stem_path": str(stem_path),
        "sha256": _sha256_file(stem_path),
        "contract": base_manifest.get("contract", {}),
    }


def _frame_magnitudes(mono: np.ndarray, config: ProfilePercussiveAuxConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_size = int(config.frame_size)
    hop_size = int(config.hop_size)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if mono.shape[0] < frame_size:
        mono = np.pad(mono, (0, frame_size - mono.shape[0]))
    frame_count = 1 + int(np.ceil(max(0, mono.shape[0] - frame_size) / float(hop_size)))
    window = np.hanning(frame_size).astype(np.float32)
    magnitudes: list[np.ndarray] = []
    rms_values: list[float] = []
    flatness_values: list[float] = []
    flux_values: list[float] = []
    previous_magnitude: np.ndarray | None = None
    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = mono[start : start + frame_size]
        if frame.shape[0] < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.shape[0]))
        magnitude = np.abs(np.fft.rfft(frame * window)).astype(np.float64) + EPSILON
        magnitudes.append(magnitude.astype(np.float32))
        rms_values.append(float(np.sqrt(np.mean(frame * frame) + EPSILON)))
        flatness_values.append(float(np.exp(np.mean(np.log(magnitude))) / (np.mean(magnitude) + EPSILON)))
        if previous_magnitude is None:
            flux_values.append(0.0)
        else:
            positive_delta = np.maximum(magnitude - previous_magnitude, 0.0)
            flux_values.append(float(np.sum(positive_delta) / (np.sum(magnitude) + EPSILON)))
        previous_magnitude = magnitude
    return (
        np.stack(magnitudes, axis=0).astype(np.float32),
        np.asarray(rms_values, dtype=np.float32),
        np.asarray(flatness_values, dtype=np.float32),
        np.asarray(flux_values, dtype=np.float32),
    )


def _normalize_magnitudes(magnitudes: np.ndarray) -> np.ndarray:
    return magnitudes / (np.linalg.norm(magnitudes, axis=1, keepdims=True) + EPSILON)


def _teacher_profile(teacher_aux: np.ndarray, config: ProfilePercussiveAuxConfig) -> tuple[np.ndarray, dict[str, Any]]:
    teacher_mono = np.mean(teacher_aux, axis=1).astype(np.float32)
    magnitudes, rms_values, flatness_values, flux_values = _frame_magnitudes(teacher_mono, config)
    active_floor = max(float(np.max(rms_values)) * float(config.activity_floor_ratio), 1.0e-7)
    active_mask = rms_values >= active_floor
    if not np.any(active_mask):
        raise ValueError("teacher aux profile has no active frames")
    normalized = _normalize_magnitudes(magnitudes[active_mask])
    profile = np.mean(normalized, axis=0)
    profile = profile / float(np.linalg.norm(profile) + EPSILON)
    return profile.astype(np.float32), {
        "active_frame_count": int(np.sum(active_mask)),
        "total_frame_count": int(rms_values.size),
        "mean_active_flatness": round(float(np.mean(flatness_values[active_mask])), 6),
        "mean_active_flux": round(float(np.mean(flux_values[active_mask])), 6),
        "active_floor": round(active_floor, 9),
    }


def _expanded_gate(frame_gate: np.ndarray, *, pre_roll_frames: int, tail_frames: int) -> np.ndarray:
    expanded = np.zeros_like(frame_gate, dtype=bool)
    active_indices = np.flatnonzero(frame_gate)
    for frame_index in active_indices:
        start = max(0, int(frame_index) - int(pre_roll_frames))
        end = min(frame_gate.size, int(frame_index) + int(tail_frames) + 1)
        expanded[start:end] = True
    return expanded


def _sample_gate(frame_gate: np.ndarray, sample_count: int, config: ProfilePercussiveAuxConfig) -> np.ndarray:
    sample_gate = np.zeros(sample_count, dtype=np.float32)
    for frame_index, enabled in enumerate(frame_gate):
        if not enabled:
            continue
        start = frame_index * int(config.hop_size)
        end = min(start + int(config.frame_size), sample_count)
        if start >= sample_count:
            break
        sample_gate[start:end] = 1.0
    if sample_gate.size > 8:
        kernel = np.hanning(9).astype(np.float32)
        kernel /= float(np.sum(kernel))
        sample_gate = np.convolve(sample_gate, kernel, mode="same").astype(np.float32)
    return sample_gate


def _profile_frequency_mask(profile: np.ndarray, config: ProfilePercussiveAuxConfig) -> np.ndarray:
    smoothing_bins = max(3, int(profile.size // 64) | 1)
    smoothing_kernel = np.ones(smoothing_bins, dtype=np.float32) / float(smoothing_bins)
    smoothed_profile = np.convolve(profile.astype(np.float32), smoothing_kernel, mode="same")
    low = float(np.percentile(smoothed_profile, float(config.spectral_mask_low_percentile)))
    high = float(np.percentile(smoothed_profile, float(config.spectral_mask_high_percentile)))
    if high <= low:
        normalized = smoothed_profile / (float(np.max(smoothed_profile)) + EPSILON)
    else:
        normalized = np.clip((smoothed_profile - low) / (high - low + EPSILON), 0.0, 1.0)
    shaped = np.power(normalized, float(config.spectral_mask_power))
    floor = float(np.clip(config.spectral_mask_floor, 0.0, 1.0))
    return (floor + (1.0 - floor) * shaped).astype(np.float32)


def _render_profile_masked_residual(
    residual_audio: np.ndarray,
    frame_gate: np.ndarray,
    profile: np.ndarray,
    background_magnitude: np.ndarray,
    config: ProfilePercussiveAuxConfig,
) -> np.ndarray:
    frame_size = int(config.frame_size)
    hop_size = int(config.hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    frequency_mask = _profile_frequency_mask(profile, config)
    rendered = np.zeros_like(residual_audio, dtype=np.float32)
    normalization = np.zeros(residual_audio.shape[0], dtype=np.float32)
    for frame_index, enabled in enumerate(frame_gate):
        if not enabled:
            continue
        start = frame_index * hop_size
        if start >= residual_audio.shape[0]:
            break
        end = min(start + frame_size, residual_audio.shape[0])
        valid_count = end - start
        if valid_count <= 0:
            continue
        for channel_index in range(residual_audio.shape[1]):
            frame = np.zeros(frame_size, dtype=np.float32)
            frame[:valid_count] = residual_audio[start:end, channel_index]
            spectrum = np.fft.rfft(frame * window)
            magnitude = np.abs(spectrum).astype(np.float32) + EPSILON
            cleaned_magnitude = np.maximum(magnitude - background_magnitude, 0.0)
            background_removed_spectrum = spectrum * (cleaned_magnitude / magnitude)
            tonal_mask = _tonal_peak_suppression_mask(cleaned_magnitude + EPSILON, config)
            masked_frame = np.fft.irfft(
                background_removed_spectrum * frequency_mask * tonal_mask,
                n=frame_size,
            ).astype(np.float32)
            rendered[start:end, channel_index] += masked_frame[:valid_count] * window[:valid_count]
        normalization[start:end] += window[:valid_count] * window[:valid_count]
    normalization_floor = max(float(np.max(normalization)) * 1.0e-3, 1.0e-6)
    active = normalization > normalization_floor
    rendered[active, :] /= normalization[active, None]
    rendered[~active, :] = 0.0
    return rendered


def _tonal_peak_suppression_mask(magnitude: np.ndarray, config: ProfilePercussiveAuxConfig) -> np.ndarray:
    smoothing_bins = max(5, int(magnitude.size // 96) | 1)
    smoothing_kernel = np.ones(smoothing_bins, dtype=np.float32) / float(smoothing_bins)
    local_average = np.convolve(magnitude.astype(np.float32), smoothing_kernel, mode="same") + EPSILON
    peak_ratio = magnitude / local_average
    allowed_ratio = max(float(config.tonal_peak_ratio), 1.0)
    suppression = np.minimum(1.0, allowed_ratio / (peak_ratio + EPSILON))
    floor = float(np.clip(config.tonal_suppression_floor, 0.0, 1.0))
    return np.maximum(suppression, floor).astype(np.float32)


def detect_profile_matched_aux(
    residual_audio: np.ndarray,
    teacher_aux: np.ndarray,
    *,
    base_audio: np.ndarray,
    config: ProfilePercussiveAuxConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    profile, profile_metrics = _teacher_profile(teacher_aux, config)
    residual_mono = np.mean(residual_audio, axis=1).astype(np.float32)
    magnitudes, rms_values, flatness_values, flux_values = _frame_magnitudes(residual_mono, config)
    normalized = _normalize_magnitudes(magnitudes)
    profile_similarity = normalized @ profile
    activity_floor = max(float(np.max(rms_values)) * float(config.activity_floor_ratio), 1.0e-7)
    flux_threshold = float(np.percentile(flux_values, float(config.flux_percentile)))
    frame_gate = (
        (rms_values >= activity_floor)
        & (profile_similarity >= float(config.min_profile_similarity))
        & ((flatness_values >= float(config.min_flatness)) | (flux_values >= flux_threshold))
    )
    expanded = _expanded_gate(
        frame_gate,
        pre_roll_frames=int(config.pre_roll_frames),
        tail_frames=int(config.tail_frames),
    )
    background_frames = magnitudes[~expanded] if np.any(~expanded) else magnitudes
    background_magnitude = np.percentile(background_frames, 60.0, axis=0).astype(np.float32)
    gate = _sample_gate(expanded, residual_audio.shape[0], config)
    aux_audio = _render_profile_masked_residual(residual_audio, expanded, profile, background_magnitude, config) * gate[
        :, None
    ] * float(config.aux_gain)
    base_anchor_rms = max(float(np.percentile(np.abs(base_audio), 95.0)), _safe_rms(base_audio))
    max_aux_rms = base_anchor_rms * float(config.max_aux_rms_ratio)
    aux_rms = _safe_rms(aux_audio)
    rms_gain = 1.0
    if aux_rms > max_aux_rms > 0.0:
        rms_gain = max_aux_rms / aux_rms
        aux_audio = aux_audio * float(rms_gain)
        aux_rms = _safe_rms(aux_audio)
    return aux_audio.astype(np.float32, copy=False), {
        "teacher_profile": profile_metrics,
        "candidate_frame_count": int(rms_values.size),
        "detected_frame_count": int(np.sum(frame_gate)),
        "expanded_frame_count": int(np.sum(expanded)),
        "mean_profile_similarity": round(float(np.mean(profile_similarity)), 6),
        "max_profile_similarity": round(float(np.max(profile_similarity)), 6),
        "activity_floor": round(activity_floor, 9),
        "flux_threshold": round(flux_threshold, 6),
        "render_method": "spectral_profile_masked_residual",
        "background_magnitude_subtraction": {
            "source": "non_detected_residual_frames",
            "percentile": 60.0,
        },
        "spectral_mask": {
            "floor": round(float(config.spectral_mask_floor), 6),
            "power": round(float(config.spectral_mask_power), 6),
            "low_percentile": round(float(config.spectral_mask_low_percentile), 6),
            "high_percentile": round(float(config.spectral_mask_high_percentile), 6),
            "tonal_peak_ratio": round(float(config.tonal_peak_ratio), 6),
            "tonal_suppression_floor": round(float(config.tonal_suppression_floor), 6),
        },
        "aux_rms_dbfs": _rms_dbfs(aux_audio),
        "rms_limiter_gain": round(float(rms_gain), 9),
        "max_aux_rms": round(float(max_aux_rms), 9),
        "audio_copied_to_output": False,
        "timing_used_for_detection": False,
    }


def _attenuate_aux_to_peak(base_audio: np.ndarray, aux_audio: np.ndarray, max_peak: float) -> tuple[np.ndarray, float]:
    rendered = base_audio + aux_audio
    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    if peak <= max_peak or peak <= 0.0:
        return aux_audio, 1.0
    gain = float(max_peak / peak)
    return (aux_audio * gain).astype(np.float32), gain


def build_percussive_fx_aux_profile_candidate(
    *,
    base_manifest_path: Path,
    source_audio_path: Path,
    output_dir: Path,
    teacher_aux_path: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
    config: ProfilePercussiveAuxConfig = ProfilePercussiveAuxConfig(),
) -> dict[str, Any]:
    started_at = time.perf_counter()
    base_manifest_path = base_manifest_path.resolve()
    source_audio_path = source_audio_path.resolve()
    teacher_aux_path = teacher_aux_path.resolve()
    output_dir = output_dir.resolve()
    if not teacher_aux_path.exists():
        raise FileNotFoundError(f"teacher aux file is missing: {teacher_aux_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = _load_json(base_manifest_path)
    base_audio, sample_rate, base_info = _read_base_audio(base_manifest_path, base_manifest)
    source_audio, source_sample_rate = _read_audio(source_audio_path)
    teacher_aux, teacher_sample_rate = _read_audio(teacher_aux_path)
    if source_sample_rate != sample_rate:
        raise ValueError(f"sample-rate mismatch: source={source_sample_rate}, base={sample_rate}")
    if teacher_sample_rate != sample_rate:
        raise ValueError(f"sample-rate mismatch: teacher={teacher_sample_rate}, base={sample_rate}")
    source_audio = _fit_audio(source_audio, base_audio.shape[0], base_audio.shape[1])
    teacher_aux = _fit_audio(teacher_aux, base_audio.shape[0], base_audio.shape[1])
    residual_audio = source_audio - base_audio
    aux_audio, detection_metrics = detect_profile_matched_aux(
        residual_audio,
        teacher_aux,
        base_audio=base_audio,
        config=config,
    )
    aux_audio, peak_gain = _attenuate_aux_to_peak(base_audio, aux_audio, float(config.limiter_peak))
    rendered = (base_audio + aux_audio).astype(np.float32)

    target_path = output_dir / f"{TARGET_STEM}.flac"
    aux_path = output_dir / f"{AUX_STEM}.flac"
    residual_path = output_dir / "fullmix_minus_base_residual.flac"
    _write_flac(target_path, rendered, sample_rate)
    _write_flac(aux_path, aux_audio, sample_rate)
    _write_flac(residual_path, residual_audio, sample_rate)
    manifest_path = output_dir / "stem_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": route_name,
        "model_name": route_name,
        "contract": {
            "benchmark_compatible": True,
            "target_reference_assisted": False,
            "profile_source_reference_assisted": True,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": False,
            "reference_teacher_pack": False,
            "registered_model_candidate": False,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "no_reference_production_claim": False,
            "quality_gate_required": True,
        },
        "pipeline": {
            "route": route_name,
            "family": "teacher_profile_calibrated_percussive_fx_aux_detector",
            "base_manifest": str(base_manifest_path),
            "base_candidate": base_info,
            "source_audio_path": str(source_audio_path),
            "source_audio_role": "full_mix",
            "target_definition": TARGET_STEM,
            "base_preserving_addition": True,
            "base_audio_unmodified": True,
            "teacher_profile": {
                "teacher_aux_path": str(teacher_aux_path),
                "teacher_aux_sha256": _sha256_file(teacher_aux_path),
                "teacher_role": "calibration_profile_only",
                "audio_copied_to_output": False,
                "timing_used_for_detection": False,
                "profile_used_at_application": True,
                "profile_metrics": detection_metrics["teacher_profile"],
            },
            "detector": {
                "config": asdict(config),
                "metrics": detection_metrics,
                "peak_limiter_gain": round(float(peak_gain), 9),
                "residual_path": str(residual_path),
            },
        },
        "stems": {
            AUX_STEM: {
                "path": str(aux_path),
                "target_definition": AUX_STEM,
                "source_audio_role": "full_mix_minus_base_residual",
                "derived": False,
                "sha256": _sha256_file(aux_path),
                "rms_dbfs": _rms_dbfs(aux_audio),
                "peak_dbfs": _peak_dbfs(aux_audio),
            },
            TARGET_STEM: {
                "path": str(target_path),
                "target_definition": TARGET_STEM,
                "estimated_aux_path": str(aux_path),
                "source_model_stem": TARGET_STEM,
                "source_candidate_model_id": base_info["model_id"],
                "derived": False,
                "sha256": _sha256_file(target_path),
                "rms_dbfs": _rms_dbfs(rendered),
                "peak_dbfs": _peak_dbfs(rendered),
            },
        },
        "execution": {
            "tool": "ml_worker_improved/tools/build_percussive_fx_aux_profile_candidate.py",
            "mode": "teacher_profile_calibration_fullmix_application",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        },
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _copy_listen_files(manifest: dict[str, Any], listen_dir: Path, listen_label: str) -> dict[str, Any]:
    listen_dir = listen_dir.resolve()
    target_listen_path = _safe_listen_path(listen_dir, listen_label)
    aux_listen_path = _safe_listen_path(listen_dir, "02_percussive_fx_aux.flac")
    target_listen_path.parent.mkdir(parents=True, exist_ok=True)
    aux_listen_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(manifest["stems"][TARGET_STEM]["path"], target_listen_path)
    shutil.copy2(manifest["stems"][AUX_STEM]["path"], aux_listen_path)
    return {
        "listen_dir": str(listen_dir),
        "files": {
            TARGET_STEM: str(target_listen_path),
            AUX_STEM: str(aux_listen_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clap/riser-class percussive-FX aux candidate from full-mix residual using a teacher profile."
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--teacher-aux", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--frame-size", type=int, default=ProfilePercussiveAuxConfig.frame_size)
    parser.add_argument("--hop-size", type=int, default=ProfilePercussiveAuxConfig.hop_size)
    parser.add_argument("--min-profile-similarity", type=float, default=ProfilePercussiveAuxConfig.min_profile_similarity)
    parser.add_argument("--min-flatness", type=float, default=ProfilePercussiveAuxConfig.min_flatness)
    parser.add_argument("--flux-percentile", type=float, default=ProfilePercussiveAuxConfig.flux_percentile)
    parser.add_argument("--activity-floor-ratio", type=float, default=ProfilePercussiveAuxConfig.activity_floor_ratio)
    parser.add_argument("--pre-roll-frames", type=int, default=ProfilePercussiveAuxConfig.pre_roll_frames)
    parser.add_argument("--tail-frames", type=int, default=ProfilePercussiveAuxConfig.tail_frames)
    parser.add_argument("--aux-gain", type=float, default=ProfilePercussiveAuxConfig.aux_gain)
    parser.add_argument("--max-aux-rms-ratio", type=float, default=ProfilePercussiveAuxConfig.max_aux_rms_ratio)
    parser.add_argument("--limiter-peak", type=float, default=ProfilePercussiveAuxConfig.limiter_peak)
    parser.add_argument("--spectral-mask-floor", type=float, default=ProfilePercussiveAuxConfig.spectral_mask_floor)
    parser.add_argument("--spectral-mask-power", type=float, default=ProfilePercussiveAuxConfig.spectral_mask_power)
    parser.add_argument(
        "--spectral-mask-low-percentile",
        type=float,
        default=ProfilePercussiveAuxConfig.spectral_mask_low_percentile,
    )
    parser.add_argument(
        "--spectral-mask-high-percentile",
        type=float,
        default=ProfilePercussiveAuxConfig.spectral_mask_high_percentile,
    )
    parser.add_argument("--tonal-peak-ratio", type=float, default=ProfilePercussiveAuxConfig.tonal_peak_ratio)
    parser.add_argument(
        "--tonal-suppression-floor",
        type=float,
        default=ProfilePercussiveAuxConfig.tonal_suppression_floor,
    )
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-label", default="01_drums_plus_percussive_fx_aux.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_percussive_fx_aux_profile_candidate.py",
        entrypoint_path="/app/tools/build_percussive_fx_aux_profile_candidate.py",
    )
    args = parse_args()
    config = ProfilePercussiveAuxConfig(
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        min_profile_similarity=float(args.min_profile_similarity),
        min_flatness=float(args.min_flatness),
        flux_percentile=float(args.flux_percentile),
        activity_floor_ratio=float(args.activity_floor_ratio),
        pre_roll_frames=int(args.pre_roll_frames),
        tail_frames=int(args.tail_frames),
        aux_gain=float(args.aux_gain),
        max_aux_rms_ratio=float(args.max_aux_rms_ratio),
        limiter_peak=float(args.limiter_peak),
        spectral_mask_floor=float(args.spectral_mask_floor),
        spectral_mask_power=float(args.spectral_mask_power),
        spectral_mask_low_percentile=float(args.spectral_mask_low_percentile),
        spectral_mask_high_percentile=float(args.spectral_mask_high_percentile),
        tonal_peak_ratio=float(args.tonal_peak_ratio),
        tonal_suppression_floor=float(args.tonal_suppression_floor),
    )
    manifest = build_percussive_fx_aux_profile_candidate(
        base_manifest_path=args.base_manifest,
        source_audio_path=args.source_audio,
        output_dir=args.output_dir,
        teacher_aux_path=args.teacher_aux,
        route_name=str(args.route_name),
        config=config,
    )
    if args.listen_dir is not None:
        manifest["listen"] = _copy_listen_files(manifest, args.listen_dir, str(args.listen_label))
        _write_json(Path(manifest["manifest_path"]), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
