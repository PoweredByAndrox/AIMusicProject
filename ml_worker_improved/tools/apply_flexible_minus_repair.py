from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import soundfile as sf
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import istft, resample_poly, stft


EPSILON = 1e-12


@dataclass(frozen=True)
class RepairTuning:
    pure_base: bool
    ambience_strength: float
    ambience_max_gain_db: float
    ambience_percentile_window_sec: float
    texture_strength: float
    texture_percentile_window_sec: float
    texture_max_floor_ratio: float
    enable_raw_kick_repair: bool
    kick_strength: float
    kick_max_blend: float
    kick_activity_threshold: float
    kick_window_sec: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"Audio is empty: {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def resample_to_sample_rate(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(int(source_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // divisor
    down = int(source_sample_rate) // divisor
    resampled = resample_poly(audio, up, down, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def match_length(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(int(left.shape[0]), int(right.shape[0]))
    if length <= 0:
        raise RuntimeError("Audio inputs have no overlapping frames")
    return left[:length], right[:length]


def rms_dbfs(audio: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + EPSILON))


def peak_dbfs(audio: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.max(np.abs(audio)) + EPSILON))


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    width = max(float(edge1) - float(edge0), 1e-8)
    t = np.clip((values - float(edge0)) / width, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_ambience_frequency_weight(freqs: np.ndarray) -> np.ndarray:
    presence = smoothstep(4600.0, 6200.0, freqs) * (1.0 - smoothstep(8500.0, 10500.0, freqs)) * 0.45
    air = smoothstep(7500.0, 9500.0, freqs) * (1.0 - smoothstep(13500.0, 16000.0, freqs))
    return np.clip(presence + air, 0.0, 1.0).astype(np.float32)


def build_texture_frequency_weight(freqs: np.ndarray) -> np.ndarray:
    low_presence = smoothstep(850.0, 1300.0, freqs) * (1.0 - smoothstep(1500.0, 2100.0, freqs)) * 0.28
    body_air = smoothstep(1450.0, 2200.0, freqs) * (1.0 - smoothstep(5600.0, 7600.0, freqs)) * 0.72
    air_noise = smoothstep(5600.0, 7200.0, freqs) * (1.0 - smoothstep(14000.0, 16000.0, freqs))
    return np.clip(low_presence + body_air + air_noise, 0.0, 1.0).astype(np.float32)


def build_kick_frequency_weight(freqs: np.ndarray) -> np.ndarray:
    sub = smoothstep(28.0, 42.0, freqs) * (1.0 - smoothstep(145.0, 190.0, freqs))
    upper_thump = smoothstep(135.0, 170.0, freqs) * (1.0 - smoothstep(210.0, 260.0, freqs)) * 0.35
    return np.clip(sub + upper_thump, 0.0, 1.0).astype(np.float32)


def build_kick_repair_frequency_weight(freqs: np.ndarray) -> np.ndarray:
    low_body = build_kick_frequency_weight(freqs)
    punch = smoothstep(240.0, 360.0, freqs) * (1.0 - smoothstep(760.0, 1050.0, freqs)) * 0.58
    attack = smoothstep(1300.0, 1800.0, freqs) * (1.0 - smoothstep(4300.0, 5600.0, freqs)) * 0.42
    return np.clip(low_body + punch + attack, 0.0, 1.0).astype(np.float32)


def band_energy_ratio_db(audio: np.ndarray, sample_rate: int, *, low_hz: float, high_hz: float) -> float:
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    band_power = float(np.sum(np.abs(spectrum[band]) ** 2) + EPSILON)
    total_power = float(np.sum(np.abs(spectrum) ** 2) + EPSILON)
    return 10.0 * math.log10(band_power / total_power)


def audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    return {
        "rms_dbfs": round(rms_dbfs(audio), 6),
        "peak_dbfs": round(peak_dbfs(audio), 6),
        "kick_band_ratio_db_35_180": round(band_energy_ratio_db(audio, sample_rate, low_hz=35.0, high_hz=180.0), 6),
        "kick_punch_band_ratio_db_250_900": round(band_energy_ratio_db(audio, sample_rate, low_hz=250.0, high_hz=900.0), 6),
        "kick_attack_band_ratio_db_1600_4500": round(band_energy_ratio_db(audio, sample_rate, low_hz=1600.0, high_hz=4500.0), 6),
        "presence_band_ratio_db_4k_8k": round(band_energy_ratio_db(audio, sample_rate, low_hz=4000.0, high_hz=8000.0), 6),
        "air_band_ratio_db_8k_14k": round(band_energy_ratio_db(audio, sample_rate, low_hz=8000.0, high_hz=14000.0), 6),
    }


def vocal_residue_proxy_metrics(raw_audio: np.ndarray, minus_audio: np.ndarray) -> dict[str, float]:
    """Estimate vocal-like content retained in a minus stem from source-minus residual projection."""
    raw_audio, minus_audio = match_length(raw_audio, minus_audio)
    residual_audio = raw_audio - minus_audio
    residual_mono = residual_audio.mean(axis=1).astype(np.float64, copy=False)
    minus_mono = minus_audio.mean(axis=1).astype(np.float64, copy=False)
    residual_mono = residual_mono - float(np.mean(residual_mono))
    minus_mono = minus_mono - float(np.mean(minus_mono))

    residual_energy = float(np.dot(residual_mono, residual_mono))
    minus_energy = float(np.dot(minus_mono, minus_mono))
    if residual_energy <= EPSILON or minus_energy <= EPSILON:
        return {
            "vocal_residue_corr": 0.0,
            "vocal_to_instrumental_rms_ratio_db": -120.0,
        }

    shared_energy = float(np.dot(minus_mono, residual_mono))
    vocal_residue_corr = abs(shared_energy) / math.sqrt(max(residual_energy * minus_energy, EPSILON))
    projection_gain = shared_energy / max(residual_energy, EPSILON)
    projected_residue = residual_mono * projection_gain
    projected_rms = math.sqrt(float(np.mean(np.square(projected_residue))) + EPSILON)
    minus_rms = math.sqrt(float(np.mean(np.square(minus_mono))) + EPSILON)
    vocal_to_instrumental_rms_ratio_db = 20.0 * math.log10(projected_rms / max(minus_rms, EPSILON))
    return {
        "vocal_residue_corr": round(float(vocal_residue_corr), 8),
        "vocal_to_instrumental_rms_ratio_db": round(float(vocal_to_instrumental_rms_ratio_db), 6),
    }


def repair_metric_summary(base_minus: np.ndarray, repaired: np.ndarray, sample_rate: int) -> dict[str, object]:
    added_layer = repaired - base_minus
    base_metrics = audio_metrics(base_minus, sample_rate)
    repaired_metrics = audio_metrics(repaired, sample_rate)
    added_metrics = audio_metrics(added_layer, sample_rate)
    return {
        "base": base_metrics,
        "repaired": repaired_metrics,
        "added_layer": added_metrics,
        "drift_vs_base": {
            "rms_db": round(repaired_metrics["rms_dbfs"] - base_metrics["rms_dbfs"], 6),
            "peak_db": round(repaired_metrics["peak_dbfs"] - base_metrics["peak_dbfs"], 6),
            "kick_band_ratio_db": round(
                repaired_metrics["kick_band_ratio_db_35_180"] - base_metrics["kick_band_ratio_db_35_180"],
                6,
            ),
            "kick_punch_band_ratio_db": round(
                repaired_metrics["kick_punch_band_ratio_db_250_900"] - base_metrics["kick_punch_band_ratio_db_250_900"],
                6,
            ),
            "kick_attack_band_ratio_db": round(
                repaired_metrics["kick_attack_band_ratio_db_1600_4500"] - base_metrics["kick_attack_band_ratio_db_1600_4500"],
                6,
            ),
            "presence_band_ratio_db": round(
                repaired_metrics["presence_band_ratio_db_4k_8k"] - base_metrics["presence_band_ratio_db_4k_8k"],
                6,
            ),
            "air_band_ratio_db": round(
                repaired_metrics["air_band_ratio_db_8k_14k"] - base_metrics["air_band_ratio_db_8k_14k"],
                6,
            ),
            "added_layer_vs_base_rms_db": round(added_metrics["rms_dbfs"] - base_metrics["rms_dbfs"], 6),
        },
    }


def build_added_layer_monitor(added_layer: np.ndarray, sample_rate: int) -> np.ndarray:
    monitored = np.zeros_like(added_layer, dtype=np.float32)
    n_fft = 2048
    hop_length = 512
    win_length = 2048

    for channel_index in range(added_layer.shape[1]):
        freqs, _, spectrum = stft(
            added_layer[:, channel_index],
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        monitor_weight = smoothstep(650.0, 1050.0, freqs) * (1.0 - smoothstep(14500.0, 17000.0, freqs))
        monitor_weight *= 0.82 + 0.18 * smoothstep(1800.0, 3200.0, freqs)
        monitor_spectrum = spectrum * monitor_weight[:, None]
        _, monitored_channel = istft(
            monitor_spectrum,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        monitored[: min(len(monitored_channel), len(added_layer)), channel_index] = monitored_channel[: len(added_layer)]
    return monitored.astype(np.float32)


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, int(sample_rate), subtype="PCM_24")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        **audio_metrics(audio, sample_rate),
    }


def apply_ambience_floor_repair(
    base_minus: np.ndarray,
    sample_rate: int,
    *,
    strength: float,
    max_gain_db: float,
    percentile_window_sec: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = np.zeros_like(base_minus, dtype=np.float32)
    debug: list[dict[str, float]] = []
    n_fft = 4096
    hop_length = 1024
    win_length = 4096

    for channel_index in range(base_minus.shape[1]):
        freqs, times, spectrum = stft(
            base_minus[:, channel_index],
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        magnitude = np.abs(spectrum).astype(np.float32)
        phase = np.exp(1j * np.angle(spectrum))
        freq_weight = build_ambience_frequency_weight(freqs)
        weighted_band = freq_weight > 0.05
        band_energy = np.mean(magnitude[weighted_band], axis=0) if np.any(weighted_band) else np.mean(magnitude, axis=0)

        window_frames = max(9, int(round(float(percentile_window_sec) * sample_rate / hop_length)))
        if window_frames % 2 == 0:
            window_frames += 1
        target_floor = percentile_filter(band_energy, percentile=62, size=window_frames, mode="nearest")
        target_floor = uniform_filter1d(target_floor, size=9, mode="nearest")
        deficit = np.clip((target_floor - band_energy) / np.maximum(target_floor, 1e-8), 0.0, 1.0)
        deficit = np.clip((deficit - 0.055) / 0.945, 0.0, 1.0)
        deficit = uniform_filter1d(deficit.astype(np.float32), size=7, mode="nearest")

        max_gain = 10.0 ** (float(max_gain_db) / 20.0)
        gain = 1.0 + float(strength) * deficit[None, :] * freq_weight[:, None]
        gain = np.clip(gain, 1.0, max_gain)
        repaired_spectrum = magnitude * gain * phase
        _, repaired_channel = istft(
            repaired_spectrum,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        repaired[: min(len(repaired_channel), len(base_minus)), channel_index] = repaired_channel[: len(base_minus)]
        debug.append(
            {
                "channel": float(channel_index),
                "deficit_mean": float(np.mean(deficit)),
                "deficit_max": float(np.max(deficit)),
                "active_frame_ratio": float(np.mean(deficit > 0.02)),
                "mean_gain_db_on_weighted_band": float(20.0 * np.log10(np.mean(gain[weighted_band]) + EPSILON))
                if np.any(weighted_band)
                else 0.0,
                "max_gain_db": float(20.0 * np.log10(np.max(gain) + EPSILON)),
                "frames": float(len(times)),
            }
        )
    return repaired.astype(np.float32), debug


def apply_texture_floor_inpainting(
    base_minus: np.ndarray,
    sample_rate: int,
    *,
    strength: float,
    percentile_window_sec: float,
    max_floor_ratio: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = np.zeros_like(base_minus, dtype=np.float32)
    debug: list[dict[str, float]] = []
    n_fft = 4096
    hop_length = 1024
    win_length = 4096

    for channel_index in range(base_minus.shape[1]):
        freqs, times, spectrum = stft(
            base_minus[:, channel_index],
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        magnitude = np.abs(spectrum).astype(np.float32)
        phase = np.exp(1j * np.angle(spectrum))
        texture_weight = build_texture_frequency_weight(freqs)
        texture_band = texture_weight > 0.05

        window_frames = max(9, int(round(float(percentile_window_sec) * sample_rate / hop_length)))
        if window_frames % 2 == 0:
            window_frames += 1
        target_floor = percentile_filter(magnitude, percentile=72, size=(1, window_frames), mode="nearest")
        target_floor = uniform_filter1d(target_floor, size=7, axis=1, mode="nearest")
        deficit = np.clip((target_floor - magnitude) / np.maximum(target_floor, 1e-8), 0.0, 1.0)
        weighted_deficit = deficit * texture_weight[:, None]
        floor_addition = target_floor * weighted_deficit * float(strength)
        max_addition = target_floor * texture_weight[:, None] * float(max_floor_ratio)
        floor_addition = np.minimum(floor_addition, max_addition)

        repaired_spectrum = (magnitude + floor_addition) * phase
        _, repaired_channel = istft(
            repaired_spectrum,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        repaired[: min(len(repaired_channel), len(base_minus)), channel_index] = repaired_channel[: len(base_minus)]
        debug.append(
            {
                "channel": float(channel_index),
                "deficit_mean_on_texture_band": float(np.mean(deficit[texture_band])) if np.any(texture_band) else 0.0,
                "deficit_max_on_texture_band": float(np.max(deficit[texture_band])) if np.any(texture_band) else 0.0,
                "active_frame_ratio": float(np.mean(np.mean(weighted_deficit[texture_band], axis=0) > 0.02))
                if np.any(texture_band)
                else 0.0,
                "mean_added_floor_ratio": float(
                    np.mean(floor_addition[texture_band] / np.maximum(target_floor[texture_band], 1e-8))
                )
                if np.any(texture_band)
                else 0.0,
                "max_added_floor_ratio": float(np.max(floor_addition / np.maximum(target_floor, 1e-8))),
                "frames": float(len(times)),
            }
        )
    return repaired.astype(np.float32), debug


def apply_kick_transient_restore(
    raw_audio: np.ndarray,
    base_minus: np.ndarray,
    sample_rate: int,
    *,
    strength: float,
    max_blend: float,
    activity_threshold: float,
    window_sec: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = np.zeros_like(base_minus, dtype=np.float32)
    debug: list[dict[str, float]] = []
    n_fft = 4096
    hop_length = 512
    win_length = 4096

    for channel_index in range(base_minus.shape[1]):
        freqs, times, raw_spectrum = stft(
            raw_audio[:, channel_index],
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        _, _, base_spectrum = stft(
            base_minus[:, channel_index],
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        raw_magnitude = np.abs(raw_spectrum).astype(np.float32)
        base_magnitude = np.abs(base_spectrum).astype(np.float32)
        kick_weight = build_kick_frequency_weight(freqs)
        repair_weight = build_kick_repair_frequency_weight(freqs)
        kick_band = kick_weight > 0.2
        repair_band = repair_weight > 0.05
        raw_kick_energy = np.mean(raw_magnitude[kick_band], axis=0)
        base_kick_energy = np.mean(base_magnitude[kick_band], axis=0)

        window_frames = max(7, int(round(float(window_sec) * sample_rate / hop_length)))
        if window_frames % 2 == 0:
            window_frames += 1
        local_floor = percentile_filter(raw_kick_energy, percentile=35, size=window_frames, mode="nearest")
        transient_ratio = raw_kick_energy / np.maximum(local_floor, 1e-8)
        transient_activity = np.clip(
            (transient_ratio - float(activity_threshold)) / max(2.5 - float(activity_threshold), 1e-8),
            0.0,
            1.0,
        )
        transient_activity = uniform_filter1d(transient_activity.astype(np.float32), size=3, mode="nearest")
        sustain_floor = percentile_filter(raw_kick_energy, percentile=20, size=window_frames * 3, mode="nearest")
        sustain_ratio = raw_kick_energy / np.maximum(sustain_floor, 1e-8)
        sustain_activity = np.clip(
            (sustain_ratio - 1.02) / max(float(activity_threshold) - 1.02, 1e-8),
            0.0,
            1.0,
        )
        sustain_activity = uniform_filter1d(sustain_activity.astype(np.float32), size=9, mode="nearest")
        kick_activity = np.maximum(transient_activity, sustain_activity * 0.55)
        kick_deficit = np.clip((raw_kick_energy - base_kick_energy) / np.maximum(raw_kick_energy, 1e-8), 0.0, 1.0)
        spectral_deficit = np.clip((raw_magnitude - base_magnitude) / np.maximum(raw_magnitude, 1e-8), 0.0, 1.0)
        restore_gate = kick_activity[None, :] * np.maximum(spectral_deficit, kick_deficit[None, :] * kick_weight[:, None])
        blend = np.clip(float(strength) * restore_gate * repair_weight[:, None], 0.0, float(max_blend))

        restored_spectrum = base_spectrum * (1.0 - blend) + raw_spectrum * blend
        _, repaired_channel = istft(
            restored_spectrum,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        repaired[: min(len(repaired_channel), len(base_minus)), channel_index] = repaired_channel[: len(base_minus)]
        debug.append(
            {
                "channel": float(channel_index),
                "active_frame_ratio": float(np.mean(restore_gate > 0.02)),
                "transient_activity_mean": float(np.mean(transient_activity)),
                "sustain_activity_mean": float(np.mean(sustain_activity)),
                "restore_gate_mean": float(np.mean(restore_gate)),
                "restore_gate_max": float(np.max(restore_gate)),
                "mean_blend_on_kick_band": float(np.mean(blend[kick_band])),
                "mean_blend_on_repair_band": float(np.mean(blend[repair_band])),
                "max_blend": float(np.max(blend)),
                "frames": float(len(times)),
            }
        )
    return repaired.astype(np.float32), debug


def clip_to_peak(audio: np.ndarray, max_peak: float = 0.999) -> np.ndarray:
    return np.clip(audio, -float(max_peak), float(max_peak)).astype(np.float32)


def parse_focus_window(value: str) -> tuple[str, float, float]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("focus window must use label:start_sec:end_sec")
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("focus window label must not be empty")
    start_sec = float(parts[1])
    end_sec = float(parts[2])
    if start_sec < 0.0 or end_sec <= start_sec:
        raise argparse.ArgumentTypeError("focus window must satisfy 0 <= start < end")
    return label, start_sec, end_sec


def write_focus_clip(path: Path, audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> dict[str, object]:
    start_frame = max(0, int(round(start_sec * sample_rate)))
    end_frame = min(audio.shape[0], int(round(end_sec * sample_rate)))
    if end_frame <= start_frame:
        raise RuntimeError(f"Focus window has no audio: {start_sec}:{end_sec}")
    return write_flac(path, audio[start_frame:end_frame], sample_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply conservative minus repair for vocal-window ambience dropouts and kick punch/attack attenuation."
    )
    parser.add_argument("raw_audio_path")
    parser.add_argument("base_minus_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_flexible_minus_repaired.flac")
    parser.add_argument(
        "--pure-base",
        action="store_true",
        help="Bypass all repair layers and publish the base minus unchanged as 02.",
    )
    parser.add_argument("--ambience-strength", type=float, default=0.22)
    parser.add_argument("--ambience-max-gain-db", type=float, default=0.65)
    parser.add_argument("--ambience-percentile-window-sec", type=float, default=3.0)
    parser.add_argument("--texture-strength", type=float, default=0.34)
    parser.add_argument("--texture-percentile-window-sec", type=float, default=4.5)
    parser.add_argument("--texture-max-floor-ratio", type=float, default=0.16)
    parser.add_argument("--enable-raw-kick-repair", action="store_true")
    parser.add_argument("--kick-strength", type=float, default=0.28)
    parser.add_argument("--kick-max-blend", type=float, default=0.08)
    parser.add_argument("--kick-activity-threshold", type=float, default=1.18)
    parser.add_argument("--kick-window-sec", type=float, default=1.2)
    parser.add_argument("--focus-window", action="append", type=parse_focus_window, default=[])
    return parser


def main() -> int:
    require_docker_runtime(
        tool_name="apply_flexible_minus_repair.py",
        entrypoint_path="/app/tools/apply_flexible_minus_repair.py",
    )
    args = build_parser().parse_args()
    raw_audio_path = Path(args.raw_audio_path)
    base_minus_path = Path(args.base_minus_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    raw_audio, raw_sample_rate = read_audio(raw_audio_path)
    base_minus, base_sample_rate = read_audio(base_minus_path)
    raw_audio = resample_to_sample_rate(raw_audio, raw_sample_rate, base_sample_rate)
    raw_audio, base_minus = match_length(raw_audio, base_minus)
    sample_rate = int(base_sample_rate)

    tuning = RepairTuning(
        pure_base=bool(args.pure_base),
        ambience_strength=float(args.ambience_strength),
        ambience_max_gain_db=float(args.ambience_max_gain_db),
        ambience_percentile_window_sec=float(args.ambience_percentile_window_sec),
        texture_strength=float(args.texture_strength),
        texture_percentile_window_sec=float(args.texture_percentile_window_sec),
        texture_max_floor_ratio=float(args.texture_max_floor_ratio),
        enable_raw_kick_repair=bool(args.enable_raw_kick_repair) and not bool(args.pure_base),
        kick_strength=float(args.kick_strength),
        kick_max_blend=float(args.kick_max_blend),
        kick_activity_threshold=float(args.kick_activity_threshold),
        kick_window_sec=float(args.kick_window_sec),
    )
    if tuning.pure_base:
        repaired = base_minus.astype(np.float32, copy=True)
        ambience_debug = [{"enabled": False, "reason": "pure_base_bypass"}]
        texture_debug = [{"enabled": False, "reason": "pure_base_bypass"}]
        kick_debug = [{"enabled": False, "reason": "pure_base_bypass"}]
    else:
        ambience_repaired, ambience_debug = apply_ambience_floor_repair(
            base_minus,
            sample_rate,
            strength=tuning.ambience_strength,
            max_gain_db=tuning.ambience_max_gain_db,
            percentile_window_sec=tuning.ambience_percentile_window_sec,
        )
        texture_repaired, texture_debug = apply_texture_floor_inpainting(
            ambience_repaired,
            sample_rate,
            strength=tuning.texture_strength,
            percentile_window_sec=tuning.texture_percentile_window_sec,
            max_floor_ratio=tuning.texture_max_floor_ratio,
        )
        if tuning.enable_raw_kick_repair:
            repaired, kick_debug = apply_kick_transient_restore(
                raw_audio,
                texture_repaired,
                sample_rate,
                strength=tuning.kick_strength,
                max_blend=tuning.kick_max_blend,
                activity_threshold=tuning.kick_activity_threshold,
                window_sec=tuning.kick_window_sec,
            )
        else:
            repaired = texture_repaired
            kick_debug = [
                {
                    "enabled": False,
                    "reason": "raw_mix_kick_repair_disabled_by_default",
                }
            ]
        repaired = clip_to_peak(repaired)
    added_layer_delta_raw = repaired - base_minus
    added_layer_monitor = build_added_layer_monitor(added_layer_delta_raw, sample_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "raw_ref_flac": write_flac(output_dir / "00_raw_ref_resampled.flac", raw_audio, sample_rate),
        "base_minus_flac": write_flac(output_dir / "01_base_minus.flac", base_minus, sample_rate),
        "repaired_minus_flac": write_flac(output_dir / "02_flexible_minus_repaired.flac", repaired, sample_rate),
        "added_layer_flac": write_flac(output_dir / "03_added_layer_only.flac", added_layer_monitor, sample_rate),
        "added_layer_delta_raw_flac": write_flac(output_dir / "04_added_layer_delta_raw.flac", added_layer_delta_raw, sample_rate),
    }

    focus_artifacts: dict[str, dict[str, object]] = {}
    focus_metrics: dict[str, dict[str, object]] = {}
    for label, start_sec, end_sec in args.focus_window:
        start_frame = max(0, int(round(start_sec * sample_rate)))
        end_frame = min(repaired.shape[0], int(round(end_sec * sample_rate)))
        if end_frame <= start_frame:
            raise RuntimeError(f"Focus window has no audio: {start_sec}:{end_sec}")
        focus_artifacts[label] = {
            "raw_ref": write_focus_clip(output_dir / f"focus_{label}_00_raw_ref.flac", raw_audio, sample_rate, start_sec, end_sec),
            "base_minus": write_focus_clip(output_dir / f"focus_{label}_01_base_minus.flac", base_minus, sample_rate, start_sec, end_sec),
            "repaired_minus": write_focus_clip(
                output_dir / f"focus_{label}_02_flexible_minus_repaired.flac",
                repaired,
                sample_rate,
                start_sec,
                end_sec,
            ),
            "added_layer": write_focus_clip(output_dir / f"focus_{label}_03_added_layer_only.flac", added_layer_monitor, sample_rate, start_sec, end_sec),
            "added_layer_delta_raw": write_focus_clip(
                output_dir / f"focus_{label}_04_added_layer_delta_raw.flac",
                added_layer_delta_raw,
                sample_rate,
                start_sec,
                end_sec,
            ),
        }
        focus_metrics[label] = {
            "window_sec": {
                "start": float(start_sec),
                "end": float(end_sec),
                "duration": round(float(end_frame - start_frame) / float(sample_rate), 6),
            },
            **repair_metric_summary(base_minus[start_frame:end_frame], repaired[start_frame:end_frame], sample_rate),
        }

    whole_track_metrics = repair_metric_summary(base_minus, repaired, sample_rate)
    whole_track_vocal_residue_metrics = vocal_residue_proxy_metrics(raw_audio, repaired)

    manifest = {
        "schema_version": 1,
        "raw_audio_path": str(raw_audio_path),
        "base_minus_path": str(base_minus_path),
        "repair": {
            "name": "flexible_minus_repair_v2",
            "safety_contract": [
                "High-band ambience repair is derived only from the base minus.",
                "Default repair is vocal-safe self-only and does not mix raw full audio into the repaired minus.",
                "Pure base mode bypasses ambience, texture, and kick repair when --pure-base is set.",
                "Raw kick repair is experimental and only runs when --enable-raw-kick-repair is set.",
                "No reference-minus or verified ground truth is required or used.",
                "This is an artifact repair stage, not an objective quality benchmark.",
            ],
            "tuning": {
                "pure_base": tuning.pure_base,
                "ambience_strength": tuning.ambience_strength,
                "ambience_max_gain_db": tuning.ambience_max_gain_db,
                "ambience_percentile_window_sec": tuning.ambience_percentile_window_sec,
                "texture_strength": tuning.texture_strength,
                "texture_percentile_window_sec": tuning.texture_percentile_window_sec,
                "texture_max_floor_ratio": tuning.texture_max_floor_ratio,
                "enable_raw_kick_repair": tuning.enable_raw_kick_repair,
                "kick_strength": tuning.kick_strength,
                "kick_max_blend": tuning.kick_max_blend,
                "kick_activity_threshold": tuning.kick_activity_threshold,
                "kick_window_sec": tuning.kick_window_sec,
            },
        },
        "audio": {
            "source_sample_rate": int(raw_sample_rate),
            "output_sample_rate": sample_rate,
            "frames": int(repaired.shape[0]),
            "duration_sec": round(float(repaired.shape[0]) / float(sample_rate), 6),
        },
        "metrics": {
            "base_rms_dbfs": whole_track_metrics["base"]["rms_dbfs"],
            "repaired_rms_dbfs": whole_track_metrics["repaired"]["rms_dbfs"],
            "rms_drift_db_vs_base": whole_track_metrics["drift_vs_base"]["rms_db"],
            "added_layer_rms_dbfs": whole_track_metrics["added_layer"]["rms_dbfs"],
            "added_layer_vs_base_rms_db": whole_track_metrics["drift_vs_base"]["added_layer_vs_base_rms_db"],
            **whole_track_vocal_residue_metrics,
            "kick_band_ratio_drift_db": whole_track_metrics["drift_vs_base"]["kick_band_ratio_db"],
            "kick_punch_band_ratio_drift_db": whole_track_metrics["drift_vs_base"]["kick_punch_band_ratio_db"],
            "kick_attack_band_ratio_drift_db": whole_track_metrics["drift_vs_base"]["kick_attack_band_ratio_db"],
            "presence_band_ratio_drift_db": whole_track_metrics["drift_vs_base"]["presence_band_ratio_db"],
            "air_band_ratio_drift_db": whole_track_metrics["drift_vs_base"]["air_band_ratio_db"],
            "detail": whole_track_metrics,
        },
        "debug": {
            "ambience": ambience_debug,
            "texture": texture_debug,
            "kick": kick_debug,
        },
        "artifact_contract": {
            "added_layer_flac": "Band-limited monitor artifact for listening. It is not used to build the repaired minus.",
            "added_layer_delta_raw_flac": "Exact repaired-minus minus base-minus delta for audit and metrics.",
        },
        "artifacts": artifacts,
        "focus_artifacts": focus_artifacts,
        "focus_metrics": focus_metrics,
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        listen_map: dict[str, str] = {}
        for source_key, target_name in (
            ("raw_ref_flac", "00_raw_ref.flac"),
            ("base_minus_flac", "01_base_minus.flac"),
            ("repaired_minus_flac", args.listen_label),
            ("added_layer_flac", "03_added_layer_only.flac"),
            ("added_layer_delta_raw_flac", "04_added_layer_delta_raw.flac"),
        ):
            source_path = Path(str(artifacts[source_key]["path"]))
            target_path = listen_dir / target_name
            shutil.copyfile(source_path, target_path)
            listen_map[target_name] = str(source_path)
        for label, focus_group in focus_artifacts.items():
            for focus_key, focus_info in focus_group.items():
                target_name = f"focus_{label}_{focus_key}.flac"
                source_path = Path(str(focus_info["path"]))
                shutil.copyfile(source_path, listen_dir / target_name)
                listen_map[target_name] = str(source_path)
        repaired_description = "pure base minus; no repair layer." if tuning.pure_base else "flexible repaired minus."
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n"
            "00 = raw full mix resampled to output rate.\n"
            "01 = base minus.\n"
            f"02 = {repaired_description}\n"
            "03 = monitor-friendly added layer only; not used to build 02.\n"
            "04 = exact raw delta for audit; usually sounds phasey/noisy.\n"
            "focus_* = short A/B windows.\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(json.dumps(listen_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["listen_dir"] = str(listen_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
