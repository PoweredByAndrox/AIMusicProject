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

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import librosa
import numpy as np
import soundfile as sf


LEAKAGE_EVIDENCE_MODES = ("model_proxy", "source_residual_proxy", "verified_reference_stems")
SAFE_STEM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class MicroDebleedConfig:
    n_fft: int = 4096
    hop_length: int = 1024
    min_freq_hz: float = 170.0
    max_freq_hz: float = 7600.0
    min_freq_ramp_hz: float = 700.0
    max_freq_ramp_hz: float = 2400.0
    projection_strength: float = 0.72
    dynamic_strength: float = 0.18
    frame_floor: float = 0.04
    frame_range: float = 0.16
    max_alpha: float = 0.95
    max_reduction: float = 0.55
    transient_protection: float = 0.30
    nonpercussive_gate_strength: float = 0.0
    nonpercussive_gate_min_gain: float = 0.12
    nonpercussive_gate_percussive_ceiling: float = 0.30
    nonpercussive_gate_harmonic_floor: float = 0.50
    nonpercussive_gate_harmonic_range: float = 0.20
    low_activity_gate_strength: float = 0.0
    low_activity_gate_min_gain: float = 0.10
    low_activity_gate_threshold_dbfs: float = -24.0
    low_activity_gate_range_db: float = 8.0
    low_band_projection_strength: float = 0.0
    low_band_projection_max_reduction: float = 0.35
    low_band_projection_frame_floor: float = 0.02
    low_band_projection_frame_range: float = 0.10
    post_transient_tail_gate_strength: float = 0.0
    post_transient_tail_gate_min_gain: float = 0.20
    post_transient_tail_start_ms: float = 35.0
    post_transient_tail_end_ms: float = 220.0
    post_transient_tail_harmonic_floor: float = 0.55
    post_transient_tail_harmonic_range: float = 0.25
    post_transient_tail_leakage_floor: float = 0.015
    post_transient_tail_leakage_range: float = 0.08
    post_transient_tail_projection_floor: float = 0.04
    post_transient_tail_projection_range: float = 0.20
    post_transient_tail_onset_percentile: float = 85.0
    max_peak: float = 0.999


def _emit(message: str) -> None:
    print(f"[drum_micro_leakage_debleed] {message}", flush=True)


def _validate_stem_name(stem_name: str) -> str:
    normalized = stem_name.strip()
    if not normalized:
        raise ValueError("stem_name must not be empty")
    if not SAFE_STEM_NAME_PATTERN.match(normalized):
        raise ValueError(f"unsafe stem_name: {stem_name!r}")
    return normalized


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise ValueError(f"Invalid audio shape for {path}: {audio.shape}")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Non-finite audio samples in {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _copy_as_flac(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".flac":
        shutil.copyfile(source_path, target_path)
        return
    audio, sample_rate = sf.read(str(source_path), always_2d=True, dtype="float32")
    sf.write(str(target_path), audio, sample_rate, subtype="PCM_24")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return math.sqrt(float(np.mean(np.square(mono, dtype=np.float32))))


def _dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(max(float(amplitude), 1e-12))


def _fit_length(audio: np.ndarray, frame_count: int) -> np.ndarray:
    return np.asarray(audio[:frame_count], dtype=np.float32)


def _require_finite(name: str, value: float) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _require_range(name: str, value: float, minimum: float, maximum: float) -> float:
    numeric_value = _require_finite(name, value)
    if not minimum <= numeric_value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}: {numeric_value}")
    return numeric_value


def _require_positive(name: str, value: float) -> float:
    numeric_value = _require_finite(name, value)
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be positive: {numeric_value}")
    return numeric_value


def _validate_config(config: MicroDebleedConfig) -> None:
    if int(config.n_fft) <= 0:
        raise ValueError(f"n_fft must be positive: {config.n_fft}")
    if int(config.hop_length) <= 0:
        raise ValueError(f"hop_length must be positive: {config.hop_length}")
    _require_range("min_freq_hz", config.min_freq_hz, 0.0, 192000.0)
    _require_range("max_freq_hz", config.max_freq_hz, 0.0, 192000.0)
    if float(config.max_freq_hz) <= float(config.min_freq_hz):
        raise ValueError("max_freq_hz must be greater than min_freq_hz")
    _require_positive("min_freq_ramp_hz", config.min_freq_ramp_hz)
    _require_positive("max_freq_ramp_hz", config.max_freq_ramp_hz)
    for name in (
        "projection_strength",
        "dynamic_strength",
        "frame_floor",
        "low_band_projection_strength",
        "low_band_projection_frame_floor",
        "post_transient_tail_start_ms",
        "post_transient_tail_end_ms",
        "post_transient_tail_harmonic_floor",
        "post_transient_tail_leakage_floor",
        "post_transient_tail_projection_floor",
    ):
        if _require_finite(name, getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative: {getattr(config, name)}")
    if float(config.post_transient_tail_end_ms) <= float(config.post_transient_tail_start_ms):
        raise ValueError("post_transient_tail_end_ms must be greater than post_transient_tail_start_ms")
    for name in (
        "frame_range",
        "nonpercussive_gate_harmonic_range",
        "low_activity_gate_range_db",
        "low_band_projection_frame_range",
        "post_transient_tail_harmonic_range",
        "post_transient_tail_leakage_range",
        "post_transient_tail_projection_range",
    ):
        _require_positive(name, getattr(config, name))
    for name in (
        "max_reduction",
        "transient_protection",
        "nonpercussive_gate_strength",
        "nonpercussive_gate_min_gain",
        "nonpercussive_gate_percussive_ceiling",
        "nonpercussive_gate_harmonic_floor",
        "low_activity_gate_strength",
        "low_activity_gate_min_gain",
        "low_band_projection_max_reduction",
        "max_alpha",
        "post_transient_tail_gate_strength",
        "post_transient_tail_gate_min_gain",
    ):
        _require_range(name, getattr(config, name), 0.0, 1.0)
    _require_range("post_transient_tail_onset_percentile", config.post_transient_tail_onset_percentile, 0.0, 100.0)
    _require_range("max_peak", config.max_peak, 1e-6, 1.0)


def _build_length_audit(named_audio: dict[str, tuple[np.ndarray, int]], aligned_frames: int) -> dict[str, Any]:
    frame_counts = {name: int(audio.shape[0]) for name, (audio, _sample_rate) in named_audio.items()}
    longest_frames = max(frame_counts.values(), default=0)
    shortest_frames = min(frame_counts.values(), default=0)
    per_input = {
        name: {
            "frames": frames,
            "trimmed_frames": max(0, frames - int(aligned_frames)),
            "trimmed_fraction": round(max(0, frames - int(aligned_frames)) / max(frames, 1), 9),
        }
        for name, frames in sorted(frame_counts.items())
    }
    return {
        "policy": "trim_to_shortest_input",
        "aligned_frames": int(aligned_frames),
        "shortest_input_frames": int(shortest_frames),
        "longest_input_frames": int(longest_frames),
        "max_trimmed_frames": max((item["trimmed_frames"] for item in per_input.values()), default=0),
        "inputs": per_input,
    }


def _validate_same_layout(named_audio: dict[str, tuple[np.ndarray, int]]) -> tuple[int, int]:
    sample_rates = {sample_rate for _, sample_rate in named_audio.values()}
    if len(sample_rates) != 1:
        raise ValueError(f"Sample rate mismatch: {sorted(sample_rates)}")
    channel_counts = {audio.shape[1] for audio, _ in named_audio.values()}
    if len(channel_counts) != 1:
        raise ValueError(f"Channel count mismatch: {sorted(channel_counts)}")
    min_frames = min(audio.shape[0] for audio, _ in named_audio.values())
    if min_frames <= 0:
        raise ValueError("Input audio is empty")
    return next(iter(sample_rates)), int(min_frames)


def _clip_peak(audio: np.ndarray, max_peak: float) -> tuple[np.ndarray, float, bool]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= float(max_peak):
        return audio.astype(np.float32, copy=False), 1.0, False
    gain = float(max_peak) / peak
    return (audio * gain).astype(np.float32), gain, True


def _frequency_weight(sample_rate: int, config: MicroDebleedConfig) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=int(config.n_fft))
    low_gate = np.clip((freqs - float(config.min_freq_hz)) / max(float(config.min_freq_ramp_hz), 1e-6), 0.0, 1.0)
    high_gate = np.clip((float(config.max_freq_hz) - freqs) / max(float(config.max_freq_ramp_hz), 1e-6), 0.0, 1.0)
    return (low_gate * high_gate)[:, np.newaxis].astype(np.float32)


def _widen_frame_gate(frame_gate: np.ndarray) -> np.ndarray:
    gate_values = frame_gate.reshape(-1).astype(np.float32)
    if gate_values.shape[0] <= 2:
        return gate_values[np.newaxis, :]
    widened_gate = gate_values.copy()
    widened_gate[1:] = np.maximum(widened_gate[1:], gate_values[:-1])
    widened_gate[:-1] = np.maximum(widened_gate[:-1], gate_values[1:])
    return widened_gate[np.newaxis, :].astype(np.float32)


def _frame_gate_to_sample_gain(
    frame_gate: np.ndarray,
    *,
    sample_count: int,
    hop_length: int,
    strength: float,
    min_gain: float,
) -> np.ndarray:
    frame_values = frame_gate.reshape(-1).astype(np.float32)
    if frame_values.size == 0:
        return np.ones(sample_count, dtype=np.float32)
    frame_positions = np.arange(frame_values.shape[0], dtype=np.float32) * float(hop_length)
    sample_positions = np.arange(sample_count, dtype=np.float32)
    sample_gate = np.interp(sample_positions, frame_positions, frame_values).astype(np.float32)
    return np.clip(1.0 - float(strength) * sample_gate, float(min_gain), 1.0).astype(np.float32)


def _micro_debleed_channel(
    drum_channel: np.ndarray,
    leakage_channel: np.ndarray,
    *,
    sample_rate: int,
    config: MicroDebleedConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    drum_stft = librosa.stft(
        drum_channel,
        n_fft=int(config.n_fft),
        hop_length=int(config.hop_length),
        window="hann",
    )
    leakage_stft = librosa.stft(
        leakage_channel,
        n_fft=int(config.n_fft),
        hop_length=int(config.hop_length),
        window="hann",
    )
    drum_mag = np.abs(drum_stft)
    leakage_mag = np.abs(leakage_stft)
    harmonic_mag, percussive_mag = librosa.decompose.hpss(drum_mag, margin=(1.0, 2.6))
    harmonic_ratio = harmonic_mag / (harmonic_mag + percussive_mag + 1e-9)
    percussive_ratio = percussive_mag / (harmonic_mag + percussive_mag + 1e-9)
    leakage_support = leakage_mag / (leakage_mag + drum_mag + 1e-9)
    positive_projection = np.real(drum_stft * np.conj(leakage_stft)) / (leakage_mag**2 + 1e-9)
    positive_projection = np.clip(positive_projection, 0.0, float(config.max_alpha))

    freq_weight = _frequency_weight(sample_rate, config)
    frame_pressure = np.mean(leakage_support * harmonic_ratio * freq_weight, axis=0)
    frame_gate = np.clip(
        (frame_pressure - float(config.frame_floor)) / max(float(config.frame_range), 1e-6),
        0.0,
        1.0,
    )[np.newaxis, :]
    frame_percussive = np.mean(percussive_ratio, axis=0)[np.newaxis, :]
    transient_guard = np.clip(1.0 - float(config.transient_protection) * frame_percussive, 0.42, 1.0)
    harmonic_mask = leakage_support * harmonic_ratio * freq_weight * frame_gate * transient_guard
    proxy_mask = leakage_support * freq_weight * frame_gate * transient_guard * 0.72
    mask = np.clip(np.maximum(harmonic_mask, proxy_mask) * (1.0 + 4.0 * frame_gate), 0.0, 1.0)
    strength = float(config.projection_strength) + float(config.dynamic_strength) * frame_gate
    cleaned_stft = drum_stft - strength * mask * positive_projection * leakage_stft
    low_band_projection_gate = np.zeros_like(frame_gate, dtype=np.float32)
    low_band_projection_mask = np.zeros_like(mask, dtype=np.float32)
    if float(config.low_band_projection_strength) > 0.0:
        low_band_pressure = np.mean(leakage_support * freq_weight, axis=0)[np.newaxis, :]
        low_band_projection_gate = np.clip(
            (low_band_pressure - float(config.low_band_projection_frame_floor))
            / max(float(config.low_band_projection_frame_range), 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)
        low_band_transient_guard = np.clip(1.0 - float(config.transient_protection) * frame_percussive, 0.30, 1.0)
        low_band_projection_mask = np.clip(
            leakage_support * freq_weight * low_band_projection_gate * low_band_transient_guard,
            0.0,
            1.0,
        )
        cleaned_stft = (
            cleaned_stft
            - float(config.low_band_projection_strength)
            * low_band_projection_mask
            * positive_projection
            * leakage_stft
        )

    max_reduction = np.maximum(
        float(config.max_reduction) * np.sqrt(mask),
        float(config.low_band_projection_max_reduction) * np.sqrt(low_band_projection_mask),
    )
    magnitude_floor = drum_mag * (1.0 - max_reduction)
    cleaned_mag = np.abs(cleaned_stft)
    cleaned_stft = np.where(
        cleaned_mag < magnitude_floor,
        cleaned_stft * (magnitude_floor / (cleaned_mag + 1e-9)),
        cleaned_stft,
    )
    nonpercussive_gate = np.zeros_like(frame_gate, dtype=np.float32)
    if float(config.nonpercussive_gate_strength) > 0.0:
        weighted_drum_mag = drum_mag * np.maximum(freq_weight, 0.08)
        weighted_total = np.sum(weighted_drum_mag, axis=0, keepdims=True) + 1e-9
        frame_harmonic = np.sum(harmonic_mag * np.maximum(freq_weight, 0.08), axis=0, keepdims=True) / weighted_total
        frame_percussive_for_gate = (
            np.sum(percussive_mag * np.maximum(freq_weight, 0.08), axis=0, keepdims=True) / weighted_total
        )
        percussive_clearance = (
            float(config.nonpercussive_gate_percussive_ceiling) - frame_percussive_for_gate
        ) / max(float(config.nonpercussive_gate_percussive_ceiling), 1e-6)
        harmonic_pressure = (
            frame_harmonic - float(config.nonpercussive_gate_harmonic_floor)
        ) / max(float(config.nonpercussive_gate_harmonic_range), 1e-6)
        nonpercussive_gate = (
            np.clip(percussive_clearance, 0.0, 1.0) * np.clip(harmonic_pressure, 0.0, 1.0)
        ).astype(np.float32)
        nonpercussive_gate = _widen_frame_gate(nonpercussive_gate)
        nonpercussive_mask = np.clip(nonpercussive_gate * np.maximum(freq_weight, 0.08), 0.0, 1.0)
        attenuation = np.clip(
            1.0 - float(config.nonpercussive_gate_strength) * nonpercussive_mask,
            float(config.nonpercussive_gate_min_gain),
            1.0,
        )
        cleaned_stft = cleaned_stft * attenuation
    cleaned = librosa.istft(
        cleaned_stft,
        hop_length=int(config.hop_length),
        window="hann",
        length=drum_channel.shape[0],
    ).astype(np.float32)
    if float(config.nonpercussive_gate_strength) > 0.0 and nonpercussive_gate.size:
        cleaned = (
            cleaned
            * _frame_gate_to_sample_gain(
                nonpercussive_gate,
                sample_count=cleaned.shape[0],
                hop_length=int(config.hop_length),
                strength=float(config.nonpercussive_gate_strength),
                min_gain=float(config.nonpercussive_gate_min_gain),
            )
        ).astype(np.float32)
    post_transient_tail_gate = np.zeros_like(frame_gate, dtype=np.float32)
    if float(config.post_transient_tail_gate_strength) > 0.0:
        onset_envelope = librosa.onset.onset_strength(
            y=drum_channel.astype(np.float32),
            sr=int(sample_rate),
            hop_length=int(config.hop_length),
        ).astype(np.float32)
        frame_count = int(drum_mag.shape[1])
        if onset_envelope.shape[0] < frame_count:
            onset_envelope = np.pad(onset_envelope, (0, frame_count - onset_envelope.shape[0]))
        onset_envelope = onset_envelope[:frame_count]
        active_onsets = onset_envelope[onset_envelope > 0.0]
        if active_onsets.size:
            onset_threshold = float(
                np.percentile(active_onsets, float(config.post_transient_tail_onset_percentile))
            )
            onset_strength = np.clip(onset_envelope / max(onset_threshold, 1e-9), 0.0, 1.0)
            onset_frames = np.flatnonzero(onset_strength >= 1.0)
            tail_frames = np.zeros(frame_count, dtype=np.float32)
            start_frames = max(
                1,
                int(round(float(config.post_transient_tail_start_ms) * 0.001 * sample_rate / int(config.hop_length))),
            )
            end_frames = max(
                start_frames + 1,
                int(round(float(config.post_transient_tail_end_ms) * 0.001 * sample_rate / int(config.hop_length))),
            )
            for onset_frame in onset_frames:
                tail_start = int(onset_frame) + start_frames
                tail_end = min(frame_count, int(onset_frame) + end_frames)
                if tail_start < tail_end:
                    tail_frames[tail_start:tail_end] = np.maximum(
                        tail_frames[tail_start:tail_end],
                        float(onset_strength[onset_frame]),
                    )
            weighted_freq = np.maximum(freq_weight, 0.08)
            weighted_total = np.sum(drum_mag * weighted_freq, axis=0, keepdims=True) + 1e-9
            harmonic_pressure = np.sum(harmonic_mag * weighted_freq, axis=0, keepdims=True) / weighted_total
            harmonic_gate = np.clip(
                (harmonic_pressure - float(config.post_transient_tail_harmonic_floor))
                / max(float(config.post_transient_tail_harmonic_range), 1e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            leakage_pressure = np.mean(leakage_support * weighted_freq, axis=0, keepdims=True)
            leakage_gate = np.clip(
                (leakage_pressure - float(config.post_transient_tail_leakage_floor))
                / max(float(config.post_transient_tail_leakage_range), 1e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            projection_pressure = np.mean(positive_projection * weighted_freq, axis=0, keepdims=True)
            projection_gate = np.clip(
                (projection_pressure - float(config.post_transient_tail_projection_floor))
                / max(float(config.post_transient_tail_projection_range), 1e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            proxy_evidence_gate = leakage_gate * projection_gate
            post_transient_tail_gate = (tail_frames[np.newaxis, :] * harmonic_gate * proxy_evidence_gate).astype(
                np.float32
            )
            post_transient_tail_gate = _widen_frame_gate(post_transient_tail_gate)
            cleaned = (
                cleaned
                * _frame_gate_to_sample_gain(
                    post_transient_tail_gate,
                    sample_count=cleaned.shape[0],
                    hop_length=int(config.hop_length),
                    strength=float(config.post_transient_tail_gate_strength),
                    min_gain=float(config.post_transient_tail_gate_min_gain),
                )
            ).astype(np.float32)
    low_activity_gate = np.zeros_like(frame_gate, dtype=np.float32)
    if float(config.low_activity_gate_strength) > 0.0:
        frame_rms = librosa.feature.rms(
            y=drum_channel.astype(np.float32),
            frame_length=int(config.n_fft),
            hop_length=int(config.hop_length),
            center=True,
        ).astype(np.float32)
        frame_rms_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
        low_activity_gate = np.clip(
            (float(config.low_activity_gate_threshold_dbfs) - frame_rms_dbfs)
            / max(float(config.low_activity_gate_range_db), 1e-6),
            0.0,
            1.0,
        )
        low_activity_gate = _widen_frame_gate(low_activity_gate)
        cleaned = (
            cleaned
            * _frame_gate_to_sample_gain(
                low_activity_gate,
                sample_count=cleaned.shape[0],
                hop_length=int(config.hop_length),
                strength=float(config.low_activity_gate_strength),
                min_gain=float(config.low_activity_gate_min_gain),
            )
        ).astype(np.float32)
    positive_values = positive_projection[positive_projection > 0.0]
    return cleaned, {
        "mean_frame_gate": round(float(np.mean(frame_gate)), 9),
        "p95_frame_gate": round(float(np.percentile(frame_gate, 95)), 9),
        "mean_mask": round(float(np.mean(mask)), 9),
        "p95_mask": round(float(np.percentile(mask, 95)), 9),
        "mean_nonpercussive_gate": round(float(np.mean(nonpercussive_gate)), 9),
        "p95_nonpercussive_gate": round(float(np.percentile(nonpercussive_gate, 95)), 9),
        "mean_low_activity_gate": round(float(np.mean(low_activity_gate)), 9),
        "p95_low_activity_gate": round(float(np.percentile(low_activity_gate, 95)), 9),
        "mean_low_band_projection_gate": round(float(np.mean(low_band_projection_gate)), 9),
        "p95_low_band_projection_gate": round(float(np.percentile(low_band_projection_gate, 95)), 9),
        "mean_low_band_projection_mask": round(float(np.mean(low_band_projection_mask)), 9),
        "p95_low_band_projection_mask": round(float(np.percentile(low_band_projection_mask, 95)), 9),
        "mean_post_transient_tail_gate": round(float(np.mean(post_transient_tail_gate)), 9),
        "p95_post_transient_tail_gate": round(float(np.percentile(post_transient_tail_gate, 95)), 9),
        "mean_positive_projection": round(float(np.mean(positive_values)) if positive_values.size else 0.0, 9),
    }


def _micro_debleed_audio(
    drum_audio: np.ndarray,
    leakage_audio: np.ndarray,
    *,
    sample_rate: int,
    config: MicroDebleedConfig,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    output = np.zeros_like(drum_audio, dtype=np.float32)
    reports: list[dict[str, float | int]] = []
    for channel_index in range(drum_audio.shape[1]):
        channel_output, channel_report = _micro_debleed_channel(
            drum_audio[:, channel_index],
            leakage_audio[:, channel_index],
            sample_rate=sample_rate,
            config=config,
        )
        output[:, channel_index] = channel_output
        reports.append({"channel": int(channel_index), **channel_report})
    return output.astype(np.float32), reports


def build_drum_micro_leakage_debleed(
    *,
    drum_stem_path: Path,
    leakage_stem_paths: list[Path],
    source_mix_path: Path | None = None,
    output_dir: Path,
    listen_dir: Path | None,
    raw_ref_path: Path | None,
    output_label: str,
    input_label: str,
    config: MicroDebleedConfig,
    stem_name: str = "drums",
    leakage_evidence_mode: str = "model_proxy",
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem_name = _validate_stem_name(stem_name)
    if not leakage_stem_paths and source_mix_path is None:
        raise ValueError("At least one --leakage-stem-path or --source-mix-path is required")
    if leakage_evidence_mode not in LEAKAGE_EVIDENCE_MODES:
        raise ValueError(
            f"leakage_evidence_mode must be one of {', '.join(LEAKAGE_EVIDENCE_MODES)}: {leakage_evidence_mode}"
        )
    if leakage_evidence_mode == "model_proxy":
        if source_mix_path is not None:
            raise ValueError("model_proxy mode must not use --source-mix-path; use source_residual_proxy instead")
        if not leakage_stem_paths:
            raise ValueError("model_proxy mode requires at least one --leakage-stem-path")
    if leakage_evidence_mode == "source_residual_proxy" and source_mix_path is None:
        raise ValueError("source_residual_proxy mode requires --source-mix-path")
    if leakage_evidence_mode == "verified_reference_stems" and not leakage_stem_paths:
        raise ValueError("verified_reference_stems mode requires at least one --leakage-stem-path")
    if leakage_evidence_mode == "verified_reference_stems" and source_mix_path is not None:
        raise ValueError("verified_reference_stems mode must not use --source-mix-path")
    reference_assisted = leakage_evidence_mode == "verified_reference_stems"
    _validate_config(config)
    for path in [drum_stem_path, *leakage_stem_paths, *([source_mix_path] if source_mix_path is not None else [])]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input path: {path}")
    if raw_ref_path is not None and not raw_ref_path.exists():
        raise FileNotFoundError(f"Missing raw reference path: {raw_ref_path}")

    loaded: dict[str, tuple[np.ndarray, int]] = {"drums": _read_audio(drum_stem_path)}
    for index, leakage_path in enumerate(leakage_stem_paths):
        loaded[f"leakage_{index}"] = _read_audio(leakage_path)
    if source_mix_path is not None:
        loaded["source_mix"] = _read_audio(source_mix_path)
    sample_rate, min_frames = _validate_same_layout(loaded)
    length_audit = _build_length_audit(loaded, min_frames)
    drums = _fit_length(loaded["drums"][0], min_frames)
    leakage_refs = [_fit_length(loaded[f"leakage_{index}"][0], min_frames) for index in range(len(leakage_stem_paths))]
    if source_mix_path is not None:
        source_mix = _fit_length(loaded["source_mix"][0], min_frames)
        leakage_refs.append((source_mix - drums).astype(np.float32))
    leakage_sum = np.sum(np.stack(leakage_refs, axis=0), axis=0).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    _emit("building_micro_leakage_debleed")
    repaired, channel_reports = _micro_debleed_audio(drums, leakage_sum, sample_rate=sample_rate, config=config)
    repaired, headroom_gain, headroom_applied = _clip_peak(repaired, float(config.max_peak))
    output_path = output_dir / output_label
    _write_flac(output_path, repaired, sample_rate)

    diagnostic_manifest_path = output_dir / "drum_micro_leakage_debleed_manifest.json"
    diagnostic_manifest = {
        "status": "completed",
        "contract": {
            "accepted_by_listening": False,
            "fallback_used": False,
            "not_timestamp_hardcoded": True,
            "purpose": (
                "Apply reference-assisted micro leakage debleed from verified non-drum stems while preserving drum transients."
                if reference_assisted
                else "Apply no-reference micro leakage debleed from non-drum proxy buses while preserving drum transients."
            ),
            "stem_name": stem_name,
            "target_reference_assisted": reference_assisted,
            "target_reference_used_at_application": reference_assisted,
            "application_reference_free": not reference_assisted,
            "delivery_allowed_without_reference": not reference_assisted,
            "no_reference_production_claim": not reference_assisted,
        },
        "pipeline": {
            "mode": "drum_micro_leakage_debleed_v2",
            "strategy": "adaptive_projection_with_optional_nonpercussive_low_activity_low_band_and_tail_gates",
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
        "inputs": {
            "stem_name": stem_name,
            "drum_stem_path": str(drum_stem_path),
            "leakage_stem_paths": [str(path) for path in leakage_stem_paths],
            "leakage_evidence_mode": leakage_evidence_mode,
            "source_mix_path": str(source_mix_path) if source_mix_path is not None else None,
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "sample_rate": sample_rate,
            "frames": int(repaired.shape[0]),
            "rms_dbfs": round(_dbfs(_safe_rms(repaired)), 6),
        },
        "headroom": {
            "gain": round(float(headroom_gain), 9),
            "applied": bool(headroom_applied),
            "max_peak": float(config.max_peak),
        },
        "config": {
            "n_fft": int(config.n_fft),
            "hop_length": int(config.hop_length),
            "min_freq_hz": float(config.min_freq_hz),
            "max_freq_hz": float(config.max_freq_hz),
            "min_freq_ramp_hz": float(config.min_freq_ramp_hz),
            "max_freq_ramp_hz": float(config.max_freq_ramp_hz),
            "projection_strength": float(config.projection_strength),
            "dynamic_strength": float(config.dynamic_strength),
            "frame_floor": float(config.frame_floor),
            "frame_range": float(config.frame_range),
            "max_alpha": float(config.max_alpha),
            "max_reduction": float(config.max_reduction),
            "transient_protection": float(config.transient_protection),
            "nonpercussive_gate_strength": float(config.nonpercussive_gate_strength),
            "nonpercussive_gate_min_gain": float(config.nonpercussive_gate_min_gain),
            "nonpercussive_gate_percussive_ceiling": float(config.nonpercussive_gate_percussive_ceiling),
            "nonpercussive_gate_harmonic_floor": float(config.nonpercussive_gate_harmonic_floor),
            "nonpercussive_gate_harmonic_range": float(config.nonpercussive_gate_harmonic_range),
            "low_activity_gate_strength": float(config.low_activity_gate_strength),
            "low_activity_gate_min_gain": float(config.low_activity_gate_min_gain),
            "low_activity_gate_threshold_dbfs": float(config.low_activity_gate_threshold_dbfs),
            "low_activity_gate_range_db": float(config.low_activity_gate_range_db),
            "low_band_projection_strength": float(config.low_band_projection_strength),
            "low_band_projection_max_reduction": float(config.low_band_projection_max_reduction),
            "low_band_projection_frame_floor": float(config.low_band_projection_frame_floor),
            "low_band_projection_frame_range": float(config.low_band_projection_frame_range),
            "post_transient_tail_gate_strength": float(config.post_transient_tail_gate_strength),
            "post_transient_tail_gate_min_gain": float(config.post_transient_tail_gate_min_gain),
            "post_transient_tail_start_ms": float(config.post_transient_tail_start_ms),
            "post_transient_tail_end_ms": float(config.post_transient_tail_end_ms),
            "post_transient_tail_harmonic_floor": float(config.post_transient_tail_harmonic_floor),
            "post_transient_tail_harmonic_range": float(config.post_transient_tail_harmonic_range),
            "post_transient_tail_leakage_floor": float(config.post_transient_tail_leakage_floor),
            "post_transient_tail_leakage_range": float(config.post_transient_tail_leakage_range),
            "post_transient_tail_projection_floor": float(config.post_transient_tail_projection_floor),
            "post_transient_tail_projection_range": float(config.post_transient_tail_projection_range),
            "post_transient_tail_onset_percentile": float(config.post_transient_tail_onset_percentile),
            "max_peak": float(config.max_peak),
        },
        "diagnostics": {
            "channels": channel_reports,
            "input_rms_dbfs": round(_dbfs(_safe_rms(drums)), 6),
            "output_rms_dbfs": round(_dbfs(_safe_rms(repaired)), 6),
            "leakage_proxy_rms_dbfs": round(_dbfs(_safe_rms(leakage_sum)), 6),
            "length_audit": length_audit,
        },
    }
    diagnostic_manifest_path.write_text(json.dumps(diagnostic_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    candidate_manifest = {
        "schema_version": 1,
        "model_id": "drum_micro_leakage_debleed",
        "model_name": "drum_micro_leakage_debleed",
        "model": {
            "model_id": "drum_micro_leakage_debleed",
            "family": "reference_assisted_drum_micro_debleed" if reference_assisted else "no_reference_drum_micro_debleed",
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "stem_name": stem_name,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": reference_assisted,
            "target_reference_used_at_application": reference_assisted,
            "application_reference_free": not reference_assisted,
            "delivery_allowed_without_reference": not reference_assisted,
            "no_reference_production_claim": not reference_assisted,
            "delivery_allowed_without_quality_gate": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "route": {
            "mode": "drum_micro_leakage_debleed_v2",
            "diagnostic_manifest_path": str(diagnostic_manifest_path),
        },
        "inputs": {
            "stem_name": stem_name,
            "drum_stem_path": str(drum_stem_path),
            "leakage_stem_paths": [str(path) for path in leakage_stem_paths],
            "leakage_evidence_mode": leakage_evidence_mode,
            "source_mix_path": str(source_mix_path) if source_mix_path is not None else None,
        },
        "stems": {
            stem_name: {
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "role": "drum_micro_leakage_debleed_candidate",
            }
        },
    }
    (output_dir / "stem_manifest.json").write_text(json.dumps(candidate_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        if raw_ref_path is not None:
            _copy_as_flac(raw_ref_path, listen_dir / "00_raw_ref.flac")
        _copy_as_flac(drum_stem_path, listen_dir / input_label)
        _copy_as_flac(output_path, listen_dir / output_label)
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    input_label: str(drum_stem_path),
                    output_label: str(output_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return diagnostic_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply micro debleed to a DRUM stem using declared non-drum leakage evidence "
            "(model proxy, source residual proxy, or verified reference stems)."
        )
    )
    parser.add_argument("drum_stem_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--leakage-stem-path", action="append", type=Path, default=[])
    parser.add_argument("--source-mix-path", type=Path)
    parser.add_argument(
        "--leakage-evidence-mode",
        choices=LEAKAGE_EVIDENCE_MODES,
        default="model_proxy",
        help="Declare whether leakage evidence is model proxy, source residual proxy, or verified reference stems.",
    )
    parser.add_argument("--raw-ref-path", type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--output-label", default="02_drum_micro_leakage_debleed.flac")
    parser.add_argument("--input-label", default="01_input_drums.flac")
    parser.add_argument("--stem-name", default="drums")
    parser.add_argument("--n-fft", type=int, default=MicroDebleedConfig.n_fft)
    parser.add_argument("--hop-length", type=int, default=MicroDebleedConfig.hop_length)
    parser.add_argument("--min-freq-hz", type=float, default=MicroDebleedConfig.min_freq_hz)
    parser.add_argument("--max-freq-hz", type=float, default=MicroDebleedConfig.max_freq_hz)
    parser.add_argument("--min-freq-ramp-hz", type=float, default=MicroDebleedConfig.min_freq_ramp_hz)
    parser.add_argument("--max-freq-ramp-hz", type=float, default=MicroDebleedConfig.max_freq_ramp_hz)
    parser.add_argument("--projection-strength", type=float, default=MicroDebleedConfig.projection_strength)
    parser.add_argument("--dynamic-strength", type=float, default=MicroDebleedConfig.dynamic_strength)
    parser.add_argument("--frame-floor", type=float, default=MicroDebleedConfig.frame_floor)
    parser.add_argument("--frame-range", type=float, default=MicroDebleedConfig.frame_range)
    parser.add_argument("--max-alpha", type=float, default=MicroDebleedConfig.max_alpha)
    parser.add_argument("--max-reduction", type=float, default=MicroDebleedConfig.max_reduction)
    parser.add_argument("--transient-protection", type=float, default=MicroDebleedConfig.transient_protection)
    parser.add_argument("--nonpercussive-gate-strength", type=float, default=MicroDebleedConfig.nonpercussive_gate_strength)
    parser.add_argument("--nonpercussive-gate-min-gain", type=float, default=MicroDebleedConfig.nonpercussive_gate_min_gain)
    parser.add_argument(
        "--nonpercussive-gate-percussive-ceiling",
        type=float,
        default=MicroDebleedConfig.nonpercussive_gate_percussive_ceiling,
    )
    parser.add_argument(
        "--nonpercussive-gate-harmonic-floor",
        type=float,
        default=MicroDebleedConfig.nonpercussive_gate_harmonic_floor,
    )
    parser.add_argument(
        "--nonpercussive-gate-harmonic-range",
        type=float,
        default=MicroDebleedConfig.nonpercussive_gate_harmonic_range,
    )
    parser.add_argument("--low-activity-gate-strength", type=float, default=MicroDebleedConfig.low_activity_gate_strength)
    parser.add_argument("--low-activity-gate-min-gain", type=float, default=MicroDebleedConfig.low_activity_gate_min_gain)
    parser.add_argument(
        "--low-activity-gate-threshold-dbfs",
        type=float,
        default=MicroDebleedConfig.low_activity_gate_threshold_dbfs,
    )
    parser.add_argument(
        "--low-activity-gate-range-db",
        type=float,
        default=MicroDebleedConfig.low_activity_gate_range_db,
    )
    parser.add_argument("--low-band-projection-strength", type=float, default=MicroDebleedConfig.low_band_projection_strength)
    parser.add_argument(
        "--low-band-projection-max-reduction",
        type=float,
        default=MicroDebleedConfig.low_band_projection_max_reduction,
    )
    parser.add_argument(
        "--low-band-projection-frame-floor",
        type=float,
        default=MicroDebleedConfig.low_band_projection_frame_floor,
    )
    parser.add_argument(
        "--low-band-projection-frame-range",
        type=float,
        default=MicroDebleedConfig.low_band_projection_frame_range,
    )
    parser.add_argument(
        "--post-transient-tail-gate-strength",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_gate_strength,
    )
    parser.add_argument(
        "--post-transient-tail-gate-min-gain",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_gate_min_gain,
    )
    parser.add_argument(
        "--post-transient-tail-start-ms",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_start_ms,
    )
    parser.add_argument(
        "--post-transient-tail-end-ms",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_end_ms,
    )
    parser.add_argument(
        "--post-transient-tail-harmonic-floor",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_harmonic_floor,
    )
    parser.add_argument(
        "--post-transient-tail-harmonic-range",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_harmonic_range,
    )
    parser.add_argument(
        "--post-transient-tail-leakage-floor",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_leakage_floor,
    )
    parser.add_argument(
        "--post-transient-tail-leakage-range",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_leakage_range,
    )
    parser.add_argument(
        "--post-transient-tail-projection-floor",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_projection_floor,
    )
    parser.add_argument(
        "--post-transient-tail-projection-range",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_projection_range,
    )
    parser.add_argument(
        "--post-transient-tail-onset-percentile",
        type=float,
        default=MicroDebleedConfig.post_transient_tail_onset_percentile,
    )
    return parser.parse_args()


def main() -> None:
    require_docker_runtime(
        tool_name="drum_micro_leakage_debleed.py",
        entrypoint_path="/app/tools/drum_micro_leakage_debleed.py",
    )
    args = parse_args()
    manifest = build_drum_micro_leakage_debleed(
        drum_stem_path=args.drum_stem_path,
        leakage_stem_paths=list(args.leakage_stem_path),
        source_mix_path=args.source_mix_path,
        output_dir=args.output_dir,
        listen_dir=args.listen_dir,
        raw_ref_path=args.raw_ref_path,
        output_label=str(args.output_label),
        input_label=str(args.input_label),
        stem_name=str(args.stem_name),
        leakage_evidence_mode=str(args.leakage_evidence_mode),
        config=MicroDebleedConfig(
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            min_freq_hz=float(args.min_freq_hz),
            max_freq_hz=float(args.max_freq_hz),
            min_freq_ramp_hz=float(args.min_freq_ramp_hz),
            max_freq_ramp_hz=float(args.max_freq_ramp_hz),
            projection_strength=float(args.projection_strength),
            dynamic_strength=float(args.dynamic_strength),
            frame_floor=float(args.frame_floor),
            frame_range=float(args.frame_range),
            max_alpha=float(args.max_alpha),
            max_reduction=float(args.max_reduction),
            transient_protection=float(args.transient_protection),
            nonpercussive_gate_strength=float(args.nonpercussive_gate_strength),
            nonpercussive_gate_min_gain=float(args.nonpercussive_gate_min_gain),
            nonpercussive_gate_percussive_ceiling=float(args.nonpercussive_gate_percussive_ceiling),
            nonpercussive_gate_harmonic_floor=float(args.nonpercussive_gate_harmonic_floor),
            nonpercussive_gate_harmonic_range=float(args.nonpercussive_gate_harmonic_range),
            low_activity_gate_strength=float(args.low_activity_gate_strength),
            low_activity_gate_min_gain=float(args.low_activity_gate_min_gain),
            low_activity_gate_threshold_dbfs=float(args.low_activity_gate_threshold_dbfs),
            low_activity_gate_range_db=float(args.low_activity_gate_range_db),
            low_band_projection_strength=float(args.low_band_projection_strength),
            low_band_projection_max_reduction=float(args.low_band_projection_max_reduction),
            low_band_projection_frame_floor=float(args.low_band_projection_frame_floor),
            low_band_projection_frame_range=float(args.low_band_projection_frame_range),
            post_transient_tail_gate_strength=float(args.post_transient_tail_gate_strength),
            post_transient_tail_gate_min_gain=float(args.post_transient_tail_gate_min_gain),
            post_transient_tail_start_ms=float(args.post_transient_tail_start_ms),
            post_transient_tail_end_ms=float(args.post_transient_tail_end_ms),
            post_transient_tail_harmonic_floor=float(args.post_transient_tail_harmonic_floor),
            post_transient_tail_harmonic_range=float(args.post_transient_tail_harmonic_range),
            post_transient_tail_leakage_floor=float(args.post_transient_tail_leakage_floor),
            post_transient_tail_leakage_range=float(args.post_transient_tail_leakage_range),
            post_transient_tail_projection_floor=float(args.post_transient_tail_projection_floor),
            post_transient_tail_projection_range=float(args.post_transient_tail_projection_range),
            post_transient_tail_onset_percentile=float(args.post_transient_tail_onset_percentile),
        ),
        command_args=list(sys.argv[1:]),
    )
    print(json.dumps({"output": manifest["output"], "diagnostics": manifest["diagnostics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
