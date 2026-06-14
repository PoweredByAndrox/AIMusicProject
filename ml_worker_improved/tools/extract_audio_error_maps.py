#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


EPS = 1.0e-8
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 256
ERROR_MAP_NAMES = (
    "waveform_residual",
    "stft_residual_magnitude",
    "phase_difference_map",
    "stereo_width_map",
    "vocal_bleed_proxy_map",
    "instrumental_bleed_proxy_map",
    "missing_detail_proxy_map",
    "transient_loss_proxy_map",
    "artifact_noise_proxy_map",
    "reconstruction_error_map",
)


def slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._-")[:128] or "empty"


def clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def as_float32_2d(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected_1d_or_2d_audio:shape={array.shape}")
    return array


def align_audio_triplet(
    mix: np.ndarray,
    vocal: np.ndarray,
    minus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [as_float32_2d(mix), as_float32_2d(vocal), as_float32_2d(minus)]
    min_frames = min(array.shape[0] for array in arrays)
    if min_frames <= 0:
        raise ValueError("empty_audio_window")
    arrays = [array[:min_frames] for array in arrays]
    channel_counts = [array.shape[1] for array in arrays]
    max_channels = max(channel_counts)
    if all(count in {1, max_channels} for count in channel_counts):
        target_channels = max_channels
    else:
        target_channels = min(channel_counts)
    aligned: list[np.ndarray] = []
    for array in arrays:
        if array.shape[1] == target_channels:
            aligned.append(array)
        elif array.shape[1] == 1:
            aligned.append(np.repeat(array, target_channels, axis=1))
        else:
            aligned.append(array[:, :target_channels])
    return aligned[0], aligned[1], aligned[2]


def read_audio_window(path: Path, start_sec: float, duration_sec: float) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    info = sf.info(str(path))
    start_frame = max(0, int(round(float(start_sec) * int(info.samplerate))))
    frames = max(1, int(round(float(duration_sec) * int(info.samplerate))))
    audio, sample_rate = sf.read(str(path), start=start_frame, frames=frames, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"empty_audio_window:{path}")
    return as_float32_2d(audio), int(sample_rate)


def _stft_mono(audio: np.ndarray, *, n_fft: int, hop_length: int) -> np.ndarray:
    mono = np.mean(as_float32_2d(audio), axis=1).astype(np.float32)
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.size))
    starts = list(range(0, max(1, mono.size - n_fft + 1), hop_length))
    tail_start = max(0, mono.size - n_fft)
    if starts[-1] != tail_start:
        starts.append(tail_start)
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.stack([mono[start : start + n_fft] * window for start in starts], axis=0)
    return np.fft.rfft(frames, axis=1).T


def _downsample_axis(array: np.ndarray, axis: int, max_items: int) -> np.ndarray:
    if max_items <= 0 or array.shape[axis] <= max_items:
        return array
    indices = np.linspace(0, array.shape[axis] - 1, num=max_items).round().astype(int)
    return np.take(array, indices, axis=axis)


def downsample_map(array: np.ndarray, *, max_bins: int, max_frames: int) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.ndim == 1:
        result = result[None, :]
    result = _downsample_axis(result, 0, max_bins)
    result = _downsample_axis(result, 1, max_frames)
    return np.asarray(result, dtype=np.float32)


def downsample_waveform(array: np.ndarray, max_samples: int) -> np.ndarray:
    result = as_float32_2d(array)
    if max_samples > 0 and result.shape[0] > max_samples:
        indices = np.linspace(0, result.shape[0] - 1, num=max_samples).round().astype(int)
        result = result[indices]
    return np.asarray(result, dtype=np.float32)


def stereo_width_curve(audio: np.ndarray, *, n_fft: int, hop_length: int) -> np.ndarray:
    data = as_float32_2d(audio)
    if data.shape[1] < 2:
        return np.zeros(max(1, int(np.ceil(data.shape[0] / max(1, hop_length)))), dtype=np.float32)
    left = data[:, 0]
    right = data[:, 1]
    if left.size < n_fft:
        left = np.pad(left, (0, n_fft - left.size))
        right = np.pad(right, (0, n_fft - right.size))
    starts = list(range(0, max(1, left.size - n_fft + 1), hop_length))
    tail_start = max(0, left.size - n_fft)
    if starts[-1] != tail_start:
        starts.append(tail_start)
    values = []
    for start in starts:
        l_frame = left[start : start + n_fft]
        r_frame = right[start : start + n_fft]
        l_centered = l_frame - np.mean(l_frame)
        r_centered = r_frame - np.mean(r_frame)
        denom = float(np.sqrt(np.sum(l_centered * l_centered) * np.sum(r_centered * r_centered)) + EPS)
        corr = float(np.sum(l_centered * r_centered) / denom)
        values.append((1.0 - np.clip(corr, -1.0, 1.0)) * 0.5)
    return np.asarray(values, dtype=np.float32)


def _tile_curve(curve: np.ndarray, freq_bins: int) -> np.ndarray:
    if curve.ndim != 1:
        curve = np.ravel(curve)
    return np.tile(curve[None, :], (freq_bins, 1)).astype(np.float32)


def _safe_mean(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    return float(np.mean(array))


def compute_audio_error_maps(
    mix: np.ndarray,
    vocal: np.ndarray,
    minus: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    map_max_bins: int = 128,
    map_max_frames: int = 256,
    max_waveform_samples: int = 4096,
) -> dict[str, Any]:
    mix, vocal, minus = align_audio_triplet(mix, vocal, minus)
    pair_sum = vocal + minus
    residual = mix - pair_sum

    mix_stft = _stft_mono(mix, n_fft=n_fft, hop_length=hop_length)
    vocal_stft = _stft_mono(vocal, n_fft=n_fft, hop_length=hop_length)
    minus_stft = _stft_mono(minus, n_fft=n_fft, hop_length=hop_length)
    sum_stft = _stft_mono(pair_sum, n_fft=n_fft, hop_length=hop_length)
    residual_stft = _stft_mono(residual, n_fft=n_fft, hop_length=hop_length)

    mix_mag = np.abs(mix_stft).astype(np.float32)
    vocal_mag = np.abs(vocal_stft).astype(np.float32)
    minus_mag = np.abs(minus_stft).astype(np.float32)
    sum_mag = np.abs(sum_stft).astype(np.float32)
    residual_mag = np.abs(residual_stft).astype(np.float32)

    phase_delta = np.angle(np.exp(1j * (np.angle(mix_stft) - np.angle(sum_stft)))).astype(np.float32)
    phase_difference = np.abs(phase_delta) / np.pi
    vocal_minus_overlap = np.minimum(vocal_mag, minus_mag)
    vocal_bleed_proxy = vocal_minus_overlap / (minus_mag + EPS)
    instrumental_bleed_proxy = vocal_minus_overlap / (vocal_mag + EPS)
    missing_detail = np.maximum(mix_mag - sum_mag, 0.0) / (mix_mag + EPS)
    reconstruction_error = np.abs(mix_stft - sum_stft).astype(np.float32)
    artifact_noise = residual_mag / (mix_mag + EPS)

    mix_energy = np.sqrt(np.mean(np.square(mix), axis=1) + EPS)
    sum_energy = np.sqrt(np.mean(np.square(pair_sum), axis=1) + EPS)
    frame_count = residual_mag.shape[1]
    if mix_energy.size < n_fft:
        mix_energy = np.pad(mix_energy, (0, n_fft - mix_energy.size))
        sum_energy = np.pad(sum_energy, (0, n_fft - sum_energy.size))
    energy_diff = np.abs(np.diff(mix_energy, prepend=mix_energy[0]) - np.diff(sum_energy, prepend=sum_energy[0]))
    if energy_diff.size == 0:
        transient_curve = np.zeros(frame_count, dtype=np.float32)
    else:
        sample_idx = np.linspace(0, energy_diff.size - 1, num=frame_count).round().astype(int)
        transient_curve = energy_diff[sample_idx].astype(np.float32)
        transient_curve = transient_curve / (float(np.max(transient_curve)) + EPS)
    transient_loss = _tile_curve(transient_curve, residual_mag.shape[0])

    width_curve = stereo_width_curve(pair_sum, n_fft=n_fft, hop_length=hop_length)
    if width_curve.size != frame_count:
        sample_idx = np.linspace(0, max(0, width_curve.size - 1), num=frame_count).round().astype(int)
        width_curve = width_curve[sample_idx] if width_curve.size else np.zeros(frame_count, dtype=np.float32)
    stereo_width = _tile_curve(width_curve, residual_mag.shape[0])

    maps = {
        "waveform_residual": downsample_waveform(residual, max_waveform_samples),
        "stft_residual_magnitude": downsample_map(residual_mag, max_bins=map_max_bins, max_frames=map_max_frames),
        "phase_difference_map": downsample_map(phase_difference, max_bins=map_max_bins, max_frames=map_max_frames),
        "stereo_width_map": downsample_map(stereo_width, max_bins=map_max_bins, max_frames=map_max_frames),
        "vocal_bleed_proxy_map": downsample_map(vocal_bleed_proxy, max_bins=map_max_bins, max_frames=map_max_frames),
        "instrumental_bleed_proxy_map": downsample_map(
            instrumental_bleed_proxy,
            max_bins=map_max_bins,
            max_frames=map_max_frames,
        ),
        "missing_detail_proxy_map": downsample_map(missing_detail, max_bins=map_max_bins, max_frames=map_max_frames),
        "transient_loss_proxy_map": downsample_map(transient_loss, max_bins=map_max_bins, max_frames=map_max_frames),
        "artifact_noise_proxy_map": downsample_map(artifact_noise, max_bins=map_max_bins, max_frames=map_max_frames),
        "reconstruction_error_map": downsample_map(
            reconstruction_error,
            max_bins=map_max_bins,
            max_frames=map_max_frames,
        ),
    }

    mix_rms = float(np.sqrt(np.mean(np.square(mix))) + EPS)
    residual_rms = float(np.sqrt(np.mean(np.square(residual))) + EPS)
    high_band = artifact_noise[int(artifact_noise.shape[0] * 0.75) :, :] if artifact_noise.shape[0] > 4 else artifact_noise
    metrics = {
        "residual_energy": round(float(np.mean(np.square(residual))), 10),
        "mixture_consistency_error": round(residual_rms / mix_rms, 10),
        "phase_collapse_score": round(clip01(_safe_mean(phase_difference)), 10),
        "stereo_width_score": round(clip01(_safe_mean(stereo_width)), 10),
        "vocal_bleed_proxy_score": round(clip01(_safe_mean(vocal_bleed_proxy)), 10),
        "instrumental_bleed_proxy_score": round(clip01(_safe_mean(instrumental_bleed_proxy)), 10),
        "missing_detail_score": round(clip01(_safe_mean(missing_detail)), 10),
        "transient_damage_score": round(clip01(_safe_mean(transient_loss)), 10),
        "artifact_noise_score": round(clip01(_safe_mean(high_band)), 10),
    }
    metadata = {
        "sample_rate": int(sample_rate),
        "channels": int(mix.shape[1]),
        "duration_sec": round(float(mix.shape[0]) / float(sample_rate), 6) if sample_rate else 0.0,
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "map_max_bins": int(map_max_bins),
        "map_max_frames": int(map_max_frames),
        "max_waveform_samples": int(max_waveform_samples),
        "raw_frame_count": int(frame_count),
        "raw_frequency_bins": int(residual_mag.shape[0]),
    }
    return {
        "maps": maps,
        "metrics": metrics,
        "metadata": metadata,
        "release_ready": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }


def write_error_maps(
    *,
    output_dir: Path,
    dataset_id: str,
    row_id: str,
    maps: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slug(dataset_id)}__{slug(row_id)}.npz"
    payload = {name: np.asarray(maps[name], dtype=np.float32) for name in ERROR_MAP_NAMES if name in maps}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    np.savez_compressed(path, **payload)
    return path


def extract_error_maps_for_window(
    *,
    mix_path: Path,
    vocal_path: Path,
    minus_path: Path,
    window_start_sec: float,
    window_end_sec: float,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    map_max_bins: int = 128,
    map_max_frames: int = 256,
    max_waveform_samples: int = 4096,
) -> dict[str, Any]:
    duration_sec = max(0.0, float(window_end_sec) - float(window_start_sec))
    if duration_sec <= 0:
        raise ValueError("invalid_window_boundaries")
    mix, mix_sr = read_audio_window(mix_path, window_start_sec, duration_sec)
    vocal, vocal_sr = read_audio_window(vocal_path, window_start_sec, duration_sec)
    minus, minus_sr = read_audio_window(minus_path, window_start_sec, duration_sec)
    if len({mix_sr, vocal_sr, minus_sr}) != 1:
        raise ValueError(f"sample_rate_mismatch:mix={mix_sr}:vocal={vocal_sr}:minus={minus_sr}")
    result = compute_audio_error_maps(
        mix,
        vocal,
        minus,
        sample_rate=mix_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        map_max_bins=map_max_bins,
        map_max_frames=map_max_frames,
        max_waveform_samples=max_waveform_samples,
    )
    return {
        **result,
        "sample_rate": int(result["metadata"]["sample_rate"]),
        "channels": int(result["metadata"]["channels"]),
        "duration_sec": float(result["metadata"]["duration_sec"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract no-reference audio error maps for one vocal/minus window.")
    parser.add_argument("--mix-path", type=Path, required=True)
    parser.add_argument("--vocal-path", type=Path, required=True)
    parser.add_argument("--minus-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="manual")
    parser.add_argument("--row-id", default="window")
    parser.add_argument("--window-start-sec", type=float, default=0.0)
    parser.add_argument("--window-end-sec", type=float, required=True)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--map-max-bins", type=int, default=128)
    parser.add_argument("--map-max-frames", type=int, default=256)
    parser.add_argument("--max-waveform-samples", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = extract_error_maps_for_window(
        mix_path=args.mix_path,
        vocal_path=args.vocal_path,
        minus_path=args.minus_path,
        window_start_sec=args.window_start_sec,
        window_end_sec=args.window_end_sec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        map_max_bins=args.map_max_bins,
        map_max_frames=args.map_max_frames,
        max_waveform_samples=args.max_waveform_samples,
    )
    npz_path = write_error_maps(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        row_id=args.row_id,
        maps=result["maps"],
        metadata=result["metadata"],
    )
    print(
        json.dumps(
            {
                "ok": True,
                "npz_path": str(npz_path),
                "metrics": result["metrics"],
                "release_ready": False,
                "melody_transcription_ran": False,
                "recommendation_index_updated": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
