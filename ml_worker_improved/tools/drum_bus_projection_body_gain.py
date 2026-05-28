from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

@dataclass(frozen=True)
class ValidationWindow:
    label: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class DrumRouteConfig:
    n_fft: int = 4096
    hop_length: int = 1024
    projection_strength: float = 0.34
    projection_dynamic_strength: float = 0.24
    projection_frame_floor: float = 0.055
    projection_frame_range: float = 0.18
    projection_max_alpha: float = 0.85
    projection_max_reduction: float = 0.28
    gain_frame_sec: float = 0.15
    gain_hop_sec: float = 0.03
    gain_smoothing_sec: float = 0.22
    max_body_gain: float = 1.12
    low_noise_flatness_limit: float = 0.0348
    chroma_limit: float = 0.106
    alt_chroma_tolerance: float = 0.006
    max_peak: float = 0.999


def _emit(message: str) -> None:
    print(f"[drum_bus_projection_body_gain] {message}", flush=True)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise ValueError(f"Invalid audio shape for {path}: {audio.shape}")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Non-finite audio samples in {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _copy_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1).astype(np.float32)


def _dbfs(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), 1e-12)))


def _safe_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _clip_peak(audio: np.ndarray, *, max_peak: float) -> tuple[np.ndarray, float, bool]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0 or peak <= float(max_peak):
        return audio.astype(np.float32, copy=False), 1.0, False
    gain = float(max_peak) / peak
    return (audio * gain).astype(np.float32), gain, True


def _validate_same_layout(named_audio: dict[str, tuple[np.ndarray, int]]) -> tuple[int, int]:
    sample_rates = {name: sample_rate for name, (_, sample_rate) in named_audio.items()}
    unique_sample_rates = set(sample_rates.values())
    if len(unique_sample_rates) != 1:
        raise ValueError(f"Sample rate mismatch: {sample_rates}")
    channel_counts = {name: audio.shape[1] for name, (audio, _) in named_audio.items()}
    unique_channel_counts = set(channel_counts.values())
    if len(unique_channel_counts) != 1:
        raise ValueError(f"Channel mismatch: {channel_counts}")
    min_frames = min(audio.shape[0] for audio, _ in named_audio.values())
    if min_frames < 2048:
        raise ValueError(f"Audio is too short for drum route: {min_frames} frames")
    return int(next(iter(unique_sample_rates))), int(min_frames)


def _fit_length(audio: np.ndarray, frame_count: int) -> np.ndarray:
    return np.asarray(audio[:frame_count], dtype=np.float32)


def _hpss_mono(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if audio.size < 2048:
        return audio * 0.0, audio * 0.0
    return librosa.effects.hpss(audio)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    sample_count = min(len(left), len(right))
    if sample_count < 1024:
        return 0.0
    left_centered = left[:sample_count] - float(np.mean(left[:sample_count]))
    right_centered = right[:sample_count] - float(np.mean(right[:sample_count]))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _spectral_flatness(audio: np.ndarray) -> float:
    if len(audio) < 2048:
        return 0.0
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)) + 1e-10
    return float(np.mean(librosa.feature.spectral_flatness(S=spectrum)))


def _chroma_peak_share(audio: np.ndarray, sample_rate: int) -> float:
    if len(audio) < 4096:
        return 0.0
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=4096, hop_length=1024)
    energy = np.mean(chroma, axis=1) + 1e-12
    return float(np.max(energy) / np.sum(energy))


def _band_rms_db(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if len(audio) < 1024:
        return -240.0
    spectrum = np.fft.rfft(audio * np.hanning(len(audio)))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / float(sample_rate))
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return -240.0
    return _dbfs(float(np.sqrt(np.mean(np.square(np.abs(spectrum[mask]), dtype=np.float64)))))


def _audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = _safe_rms(audio)
    mono = _mono(audio)
    harmonic, percussive = _hpss_mono(mono)
    harmonic_rms = _safe_rms(harmonic)
    percussive_rms = _safe_rms(percussive)
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_sec": round(float(audio.shape[0] / sample_rate), 6),
        "rms_dbfs": round(_dbfs(rms), 6),
        "peak_dbfs": round(_dbfs(peak), 6),
        "percussive_dbfs": round(_dbfs(percussive_rms), 6),
        "harmonic_dbfs": round(_dbfs(harmonic_rms), 6),
        "percussive_minus_harmonic_db": round(_dbfs(percussive_rms) - _dbfs(harmonic_rms), 6),
        "melody_minus_low_db": round(
            _band_rms_db(mono, sample_rate, 1000.0, 6000.0)
            - _band_rms_db(mono, sample_rate, 20.0, 250.0),
            6,
        ),
        "chroma_peak_share": round(_chroma_peak_share(mono, sample_rate), 9),
        "spectral_flatness": round(_spectral_flatness(mono), 9),
    }


def _metrics_with_leakage(
    audio: np.ndarray,
    sample_rate: int,
    leakage_refs: list[np.ndarray],
) -> dict[str, float | int]:
    metrics = _audio_metrics(audio, sample_rate)
    mono = _mono(audio)
    metrics["max_ref_corr_to_non_drum_buses"] = round(
        max([abs(_correlation(mono, _mono(ref))) for ref in leakage_refs] or [0.0]),
        9,
    )
    return metrics


def _slice_audio(audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_frame = max(0, int(round(float(start_sec) * sample_rate)))
    end_frame = min(audio.shape[0], int(round(float(end_sec) * sample_rate)))
    if end_frame <= start_frame:
        return audio[:0]
    return audio[start_frame:end_frame]


def _frequency_weight(sample_rate: int, n_fft: int) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    low_weight = np.clip((freqs - 180.0) / 900.0, 0.0, 1.0)
    high_weight = np.clip((15000.0 - freqs) / 4500.0, 0.25, 1.0)
    return (low_weight * high_weight)[:, np.newaxis]


def _project_non_drum_leakage(
    base_drums: np.ndarray,
    leakage_sum: np.ndarray,
    *,
    sample_rate: int,
    config: DrumRouteConfig,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    output = np.zeros_like(base_drums, dtype=np.float32)
    reports: list[dict[str, float | int]] = []
    freq_weight = _frequency_weight(sample_rate, int(config.n_fft))

    for channel_index in range(base_drums.shape[1]):
        drums_channel = base_drums[:, channel_index]
        leakage_channel = leakage_sum[:, channel_index]
        drum_stft = librosa.stft(
            drums_channel,
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
        harmonic_mag, percussive_mag = librosa.decompose.hpss(drum_mag, margin=(1.0, 2.4))
        harmonic_ratio = harmonic_mag / (harmonic_mag + percussive_mag + 1e-9)
        leakage_support = leakage_mag / (leakage_mag + drum_mag + 1e-9)
        positive_projection = np.real(drum_stft * np.conj(leakage_stft)) / (leakage_mag**2 + 1e-9)
        positive_projection = np.clip(positive_projection, 0.0, float(config.projection_max_alpha))

        frame_pressure = np.mean(leakage_support * harmonic_ratio * freq_weight, axis=0)
        frame_gate = np.clip(
            (frame_pressure - float(config.projection_frame_floor)) / float(config.projection_frame_range),
            0.0,
            1.0,
        )[np.newaxis, :]
        mask = leakage_support * harmonic_ratio * freq_weight * frame_gate
        strength = float(config.projection_strength) + float(config.projection_dynamic_strength) * frame_gate
        cleaned_stft = drum_stft - strength * mask * positive_projection * leakage_stft

        magnitude_floor = drum_mag * (1.0 - float(config.projection_max_reduction) * mask)
        cleaned_mag = np.abs(cleaned_stft)
        cleaned_stft = np.where(
            cleaned_mag < magnitude_floor,
            cleaned_stft * (magnitude_floor / (cleaned_mag + 1e-9)),
            cleaned_stft,
        )
        output[:, channel_index] = librosa.istft(
            cleaned_stft,
            hop_length=int(config.hop_length),
            window="hann",
            length=base_drums.shape[0],
        ).astype(np.float32)
        reports.append(
            {
                "channel": int(channel_index),
                "mean_mask": round(float(np.mean(mask)), 9),
                "p95_mask": round(float(np.percentile(mask, 95)), 9),
                "mean_positive_projection": round(
                    float(np.mean(positive_projection[positive_projection > 0.0]))
                    if np.any(positive_projection > 0.0)
                    else 0.0,
                    9,
                ),
            }
        )

    return output.astype(np.float32), reports


def _frame_rms_envelope(signal: np.ndarray, frame_count: int, hop_count: int) -> tuple[np.ndarray, np.ndarray]:
    positions: list[int] = []
    values: list[float] = []
    for start_frame in range(0, len(signal), hop_count):
        end_frame = min(len(signal), start_frame + frame_count)
        positions.append(start_frame)
        values.append(float(np.sqrt(np.mean(signal[start_frame:end_frame] ** 2))) if end_frame > start_frame else 0.0)
    return np.asarray(positions), np.asarray(values)


def _smooth_envelope(envelope: np.ndarray, sample_rate: int, smoothing_sec: float) -> np.ndarray:
    window_length = max(1, int(round(float(smoothing_sec) * sample_rate)))
    if window_length <= 1:
        return envelope.astype(np.float32)
    window = np.hanning(window_length).astype(np.float32)
    window_sum = float(np.sum(window))
    if window_sum <= 0.0:
        return envelope.astype(np.float32)
    window /= window_sum
    return np.convolve(envelope, window, mode="same").astype(np.float32)


def _build_dynamic_body_gain(
    debleed_drums: np.ndarray,
    base_drums: np.ndarray,
    body_reference_drums: np.ndarray,
    leakage_sum: np.ndarray,
    *,
    sample_rate: int,
    config: DrumRouteConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    debleed_mono = _mono(debleed_drums)
    base_mono = _mono(base_drums)
    body_ref_mono = _mono(body_reference_drums)
    _, debleed_percussive = _hpss_mono(debleed_mono)
    _, base_percussive = _hpss_mono(base_mono)
    _, body_ref_percussive = _hpss_mono(body_ref_mono)

    frame_count = max(2048, int(round(float(config.gain_frame_sec) * sample_rate)))
    hop_count = max(512, int(round(float(config.gain_hop_sec) * sample_rate)))
    positions, debleed_env = _frame_rms_envelope(debleed_percussive, frame_count, hop_count)
    _, base_env = _frame_rms_envelope(base_percussive, frame_count, hop_count)
    _, body_ref_env = _frame_rms_envelope(body_ref_percussive, frame_count, hop_count)

    gain_frames: list[float] = []
    active_segments: list[dict[str, float]] = []
    leakage_mono = _mono(leakage_sum)
    for index, start_frame in enumerate(positions):
        end_frame = min(len(debleed_mono), start_frame + frame_count)
        debleed_slice = debleed_mono[start_frame:end_frame]
        leakage_slice = leakage_mono[start_frame:end_frame]
        body_ref_slice = body_ref_mono[start_frame:end_frame]
        flatness = _spectral_flatness(debleed_slice)
        correlation = abs(_correlation(debleed_slice, leakage_slice))
        chroma = _chroma_peak_share(debleed_slice, sample_rate)
        body_ref_chroma = _chroma_peak_share(body_ref_slice, sample_rate)

        target_env = max(base_env[index] * 1.03, min(body_ref_env[index] * 0.98, base_env[index] * 1.14))
        needed_gain = max(0.0, target_env - debleed_env[index]) / (debleed_env[index] + 1e-9)
        low_noise_segment = (
            flatness < float(config.low_noise_flatness_limit)
            and chroma < float(config.chroma_limit)
            and body_ref_chroma <= chroma + float(config.alt_chroma_tolerance)
        )
        if needed_gain > 0.015 and low_noise_segment:
            correlation_guard = 1.0 - min(0.45, max(0.0, correlation - 0.155) * 2.4)
            gain = 1.0 + min(float(config.max_body_gain) - 1.0, needed_gain * 0.75 * correlation_guard)
        else:
            gain = 1.0
        gain_frames.append(gain)
        if gain > 1.005:
            active_segments.append(
                {
                    "start_sec": round(float(start_frame / sample_rate), 6),
                    "end_sec": round(float(end_frame / sample_rate), 6),
                    "gain": round(float(gain), 9),
                    "needed_gain": round(float(needed_gain), 9),
                    "flatness": round(float(flatness), 9),
                    "corr_to_leakage_sum": round(float(correlation), 9),
                    "chroma_peak_share": round(float(chroma), 9),
                    "body_ref_chroma_peak_share": round(float(body_ref_chroma), 9),
                }
            )

    gain_envelope = (
        np.interp(np.arange(debleed_drums.shape[0]), positions, np.asarray(gain_frames)).astype(np.float32)
        if len(positions)
        else np.ones(debleed_drums.shape[0], dtype=np.float32)
    )
    gain_envelope = _smooth_envelope(gain_envelope, sample_rate, float(config.gain_smoothing_sec))
    gain_envelope = np.clip(gain_envelope, 1.0, float(config.max_body_gain))
    routed = (debleed_drums * gain_envelope[:, np.newaxis]).astype(np.float32)
    diagnostics = {
        "mean_gain": round(float(np.mean(gain_envelope)), 9),
        "p95_gain": round(float(np.percentile(gain_envelope, 95)), 9),
        "max_gain": round(float(np.max(gain_envelope)), 9),
        "active_frame_count": len(active_segments),
        "active_segments": active_segments,
    }
    return routed, diagnostics


def _window_metrics(
    *,
    base_drums: np.ndarray,
    debleed_drums: np.ndarray,
    body_reference_drums: np.ndarray,
    output_drums: np.ndarray,
    sample_rate: int,
    leakage_refs: list[np.ndarray],
    validation_windows: list[ValidationWindow],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for window in validation_windows:
        base_slice = _slice_audio(base_drums, sample_rate, window.start_sec, window.end_sec)
        debleed_slice = _slice_audio(debleed_drums, sample_rate, window.start_sec, window.end_sec)
        body_ref_slice = _slice_audio(body_reference_drums, sample_rate, window.start_sec, window.end_sec)
        output_slice = _slice_audio(output_drums, sample_rate, window.start_sec, window.end_sec)
        leakage_slices = [
            _slice_audio(leakage_ref, sample_rate, window.start_sec, window.end_sec) for leakage_ref in leakage_refs
        ]
        metrics[window.label] = {
            "start_sec": float(window.start_sec),
            "end_sec": float(window.end_sec),
            "base_drums": _metrics_with_leakage(base_slice, sample_rate, leakage_slices),
            "debleed_intermediate": _metrics_with_leakage(debleed_slice, sample_rate, leakage_slices),
            "body_reference_drums": _metrics_with_leakage(body_ref_slice, sample_rate, leakage_slices),
            "final_output": _metrics_with_leakage(output_slice, sample_rate, leakage_slices),
        }
    return metrics


def parse_validation_window(raw_window: str) -> ValidationWindow:
    parts = raw_window.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Validation window must be label:start_sec:end_sec")
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("Validation window label is empty")
    start_sec = float(parts[1])
    end_sec = float(parts[2])
    if start_sec < 0.0 or end_sec <= start_sec:
        raise argparse.ArgumentTypeError(f"Invalid validation window bounds: {raw_window}")
    return ValidationWindow(label=label, start_sec=start_sec, end_sec=end_sec)


def build_drum_bus_projection_body_gain(
    *,
    base_drums_path: Path,
    body_reference_drums_path: Path,
    leakage_stem_paths: list[Path],
    output_dir: Path,
    raw_ref_path: Path | None,
    listen_dir: Path | None,
    validation_windows: list[ValidationWindow],
    config: DrumRouteConfig,
    source_label: str,
    debleed_label: str,
    output_label: str,
    body_reference_label: str,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    if not leakage_stem_paths:
        raise ValueError("At least one --leakage-stem-path is required")
    paths_to_check = [base_drums_path, body_reference_drums_path, *leakage_stem_paths]
    if raw_ref_path is not None:
        paths_to_check.append(raw_ref_path)
    for path in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(f"Missing input path: {path}")

    loaded: dict[str, tuple[np.ndarray, int]] = {
        "base_drums": _read_audio(base_drums_path),
        "body_reference_drums": _read_audio(body_reference_drums_path),
    }
    for index, leakage_path in enumerate(leakage_stem_paths):
        loaded[f"leakage_{index}"] = _read_audio(leakage_path)
    sample_rate, min_frames = _validate_same_layout(loaded)
    base_drums = _fit_length(loaded["base_drums"][0], min_frames)
    body_reference_drums = _fit_length(loaded["body_reference_drums"][0], min_frames)
    leakage_refs = [_fit_length(loaded[f"leakage_{index}"][0], min_frames) for index in range(len(leakage_stem_paths))]
    leakage_sum = np.sum(np.stack(leakage_refs, axis=0), axis=0).astype(np.float32)

    _emit("projecting_non_drum_bus_leakage")
    debleed_drums, projection_reports = _project_non_drum_leakage(
        base_drums,
        leakage_sum,
        sample_rate=sample_rate,
        config=config,
    )
    _emit("building_dynamic_body_gain")
    body_gain_candidate, body_gain_diagnostics = _build_dynamic_body_gain(
        debleed_drums,
        base_drums,
        body_reference_drums,
        leakage_sum,
        sample_rate=sample_rate,
        config=config,
    )
    clipped_body_gain_candidate, headroom_gain, headroom_applied = _clip_peak(
        body_gain_candidate,
        max_peak=float(config.max_peak),
    )
    if headroom_applied and headroom_gain < 0.985:
        final_drums = debleed_drums.astype(np.float32, copy=False)
        final_selection = {
            "selected": "debleed_intermediate",
            "rejected": "dynamic_body_gain",
            "reason": "dynamic_body_gain_requires_global_headroom_reduction",
            "candidate_headroom_gain": round(float(headroom_gain), 9),
        }
        final_headroom_gain = 1.0
        final_headroom_applied = False
    else:
        final_drums = clipped_body_gain_candidate.astype(np.float32, copy=False)
        final_selection = {
            "selected": "dynamic_body_gain",
            "rejected": None,
            "reason": None,
            "candidate_headroom_gain": round(float(headroom_gain), 9),
        }
        final_headroom_gain = headroom_gain
        final_headroom_applied = headroom_applied

    output_dir.mkdir(parents=True, exist_ok=True)
    debleed_output_path = output_dir / "drums_bus_projection_debleed_intermediate.flac"
    _write_flac(debleed_output_path, debleed_drums, sample_rate)
    output_filename = (
        "drums_bus_projection_selected_debleed.flac"
        if final_selection["selected"] == "debleed_intermediate"
        else "drums_bus_projection_dynamic_body_gain.flac"
    )
    output_path = output_dir / output_filename
    _write_flac(output_path, final_drums, sample_rate)

    final_output_audio, final_sample_rate = _read_audio(output_path)
    debleed_output_audio, debleed_sample_rate = _read_audio(debleed_output_path)
    if final_sample_rate != sample_rate or debleed_sample_rate != sample_rate:
        raise ValueError("Unexpected sample-rate drift after writing output files")

    manifest_path = output_dir / "drum_bus_projection_body_gain_manifest.json"
    candidate_manifest_path = output_dir / "stem_manifest.json"
    manifest: dict[str, Any] = {
        "status": "completed",
        "contract": {
            "accepted_by_listening": False,
            "not_timestamp_hardcoded": True,
            "validation_windows_from_user_feedback_only": True,
            "fallback_used": False,
        },
        "pipeline": {
            "mode": "drum_bus_projection_dynamic_body_gain_v1",
            "strategy": "complex_projection_debleed_against_non_drum_buses_plus_dynamic_body_gain",
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": command_args or [],
        },
        "inputs": {
            "base_drums_path": str(base_drums_path),
            "body_reference_drums_path": str(body_reference_drums_path),
            "leakage_stem_paths": [str(path) for path in leakage_stem_paths],
            "raw_ref_path": str(raw_ref_path) if raw_ref_path else None,
        },
        "outputs": {
            "debleed_intermediate_path": str(debleed_output_path),
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "candidate_manifest_path": str(candidate_manifest_path),
        },
        "config": config.__dict__,
        "headroom": {
            "gain": round(float(final_headroom_gain), 9),
            "applied": bool(final_headroom_applied),
            "max_peak": float(config.max_peak),
        },
        "selection": final_selection,
        "diagnostics": {
            "projection": projection_reports,
            "dynamic_body_gain": body_gain_diagnostics,
        },
        "metrics": {
            "base_drums": _metrics_with_leakage(base_drums, sample_rate, leakage_refs),
            "debleed_intermediate": _metrics_with_leakage(debleed_output_audio, sample_rate, leakage_refs),
            "body_reference_drums": _metrics_with_leakage(body_reference_drums, sample_rate, leakage_refs),
            "final_output": {
                **_metrics_with_leakage(final_output_audio, sample_rate, leakage_refs),
                "sha256": _sha256(output_path),
            },
            "validation_windows": _window_metrics(
                base_drums=base_drums,
                debleed_drums=debleed_output_audio,
                body_reference_drums=body_reference_drums,
                output_drums=final_output_audio,
                sample_rate=sample_rate,
                leakage_refs=leakage_refs,
                validation_windows=validation_windows,
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    candidate_manifest = {
        "schema_version": 1,
        "model_id": "drum_bus_projection_body_gain",
        "model_name": "drum_bus_projection_body_gain",
        "model": {
            "model_id": "drum_bus_projection_body_gain",
            "family": "no_reference_drum_debleed_body_gain",
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "delivery_allowed_without_quality_gate": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "route": {
            "mode": "drum_bus_projection_dynamic_body_gain_v1",
            "selected_output": final_selection["selected"],
            "selection_reason": final_selection["reason"],
            "diagnostic_manifest_path": str(manifest_path),
        },
        "inputs": {
            "base_drums_path": str(base_drums_path),
            "body_reference_drums_path": str(body_reference_drums_path),
            "leakage_stem_paths": [str(path) for path in leakage_stem_paths],
        },
        "stems": {
            "drums": {
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "role": "drum_bus_projection_body_gain_candidate",
            }
        },
    }
    candidate_manifest_path.write_text(json.dumps(candidate_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        candidate_map: dict[str, str] = {}
        if raw_ref_path is not None:
            raw_target = listen_dir / "00_raw_ref.flac"
            _copy_file(raw_ref_path, raw_target)
            candidate_map[raw_target.name] = str(raw_target)
        source_target = listen_dir / source_label
        debleed_target = listen_dir / debleed_label
        output_target = listen_dir / output_label
        body_reference_target = listen_dir / body_reference_label
        _copy_file(base_drums_path, source_target)
        _copy_file(debleed_output_path, debleed_target)
        _copy_file(output_path, output_target)
        _copy_file(body_reference_drums_path, body_reference_target)
        candidate_map[source_target.name] = str(source_target)
        candidate_map[debleed_target.name] = str(debleed_target)
        candidate_map[output_target.name] = str(output_target)
        candidate_map[body_reference_target.name] = str(body_reference_target)
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(candidate_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (listen_dir / "00_README.txt").write_text(
            "\n".join(
                [
                    "DRUM bus-projection dynamic body-gain listen pack.",
                    "This pack is not accepted until listening confirms the target windows.",
                    "00_raw_ref.flac: optional raw/source reference when provided.",
                    f"{source_label}: model base drums.",
                    f"{debleed_label}: non-drum-bus projection debleed intermediate.",
                    f"{output_label}: final selected output ({final_selection['selected']}).",
                    f"{body_reference_label}: body reference model, not selected output.",
                    f"Selection reason: {final_selection['reason'] or 'dynamic body gain accepted'}.",
                    "Validation windows are report-only and do not drive routing.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["listen_pack"] = {"listen_dir": str(listen_dir), "candidate_map": candidate_map}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible DRUM debleed/body-gain candidate from model stems."
    )
    parser.add_argument("base_drums_path")
    parser.add_argument("body_reference_drums_path")
    parser.add_argument("output_dir")
    parser.add_argument("--leakage-stem-path", action="append", default=[], required=True)
    parser.add_argument("--raw-ref-path")
    parser.add_argument("--listen-dir")
    parser.add_argument("--validation-window", action="append", type=parse_validation_window, default=[])
    parser.add_argument("--source-label", default="01_base_drums.flac")
    parser.add_argument("--debleed-label", default="02_debleed_intermediate.flac")
    parser.add_argument("--output-label", default="03_selected_drums.flac")
    parser.add_argument("--body-reference-label", default="04_body_reference_drums.flac")
    parser.add_argument("--max-body-gain", type=float, default=DrumRouteConfig.max_body_gain)
    parser.add_argument("--max-peak", type=float, default=DrumRouteConfig.max_peak)
    parser.add_argument("--low-noise-flatness-limit", type=float, default=DrumRouteConfig.low_noise_flatness_limit)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="drum_bus_projection_body_gain.py",
        entrypoint_path="/app/tools/drum_bus_projection_body_gain.py",
    )
    args = parse_args()
    config = DrumRouteConfig(
        max_body_gain=float(args.max_body_gain),
        max_peak=float(args.max_peak),
        low_noise_flatness_limit=float(args.low_noise_flatness_limit),
    )
    manifest = build_drum_bus_projection_body_gain(
        base_drums_path=Path(args.base_drums_path).resolve(),
        body_reference_drums_path=Path(args.body_reference_drums_path).resolve(),
        leakage_stem_paths=[Path(path).resolve() for path in args.leakage_stem_path],
        output_dir=Path(args.output_dir).resolve(),
        raw_ref_path=Path(args.raw_ref_path).resolve() if args.raw_ref_path else None,
        listen_dir=Path(args.listen_dir).resolve() if args.listen_dir else None,
        validation_windows=list(args.validation_window),
        config=config,
        source_label=str(args.source_label),
        debleed_label=str(args.debleed_label),
        output_label=str(args.output_label),
        body_reference_label=str(args.body_reference_label),
        command_args=sys.argv[1:],
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
