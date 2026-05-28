from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    return audio.astype(np.float32), int(sample_rate)


def _match_length(*arrays: np.ndarray) -> list[np.ndarray]:
    length = min(arr.shape[0] for arr in arrays)
    return [arr[:length] for arr in arrays]


def _freq_weight(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.full_like(freqs, 0.15, dtype=np.float32)
    min_gain = np.full_like(freqs, 0.7, dtype=np.float32)

    band_low = (freqs >= 120.0) & (freqs < 250.0)
    band_voice = (freqs >= 250.0) & (freqs < 4500.0)
    band_air = (freqs >= 4500.0) & (freqs < 8000.0)

    weight[band_low] = 0.45
    weight[band_voice] = 1.0
    weight[band_air] = 0.42

    min_gain[freqs < 120.0] = 0.9
    min_gain[band_low] = 0.55
    min_gain[band_voice] = 0.22
    min_gain[band_air] = 0.5

    return weight, min_gain


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values, kernel, mode="same")


def _band_overlap(minus_mag: np.ndarray, vocal_mag: np.ndarray, freqs: np.ndarray, *, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    a = minus_mag[mask]
    b = vocal_mag[mask]
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.minimum(a, b).sum() / max(b.sum(), 1e-9))


def main() -> None:
    require_docker_runtime(
        tool_name="apply_minus_vocal_residue_cleanup.py",
        entrypoint_path="/app/tools/apply_minus_vocal_residue_cleanup.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_audio_path")
    parser.add_argument("minus_audio_path")
    parser.add_argument("primary_vocals_path")
    parser.add_argument("secondary_vocals_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_minus_refined.flac")
    parser.add_argument("--strength", type=float, default=0.82)
    parser.add_argument("--secondary-weight", type=float, default=0.75)
    parser.add_argument("--mask-power", type=float, default=1.6)
    parser.add_argument("--activity-threshold", type=float, default=0.08)
    args = parser.parse_args()

    raw_audio_path = Path(args.raw_audio_path)
    minus_audio_path = Path(args.minus_audio_path)
    primary_vocals_path = Path(args.primary_vocals_path)
    secondary_vocals_path = Path(args.secondary_vocals_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    raw, raw_sr = _load_audio(raw_audio_path)
    minus, minus_sr = _load_audio(minus_audio_path)
    primary_vocals, primary_sr = _load_audio(primary_vocals_path)
    secondary_vocals, secondary_sr = _load_audio(secondary_vocals_path)

    if len({raw_sr, minus_sr, primary_sr, secondary_sr}) != 1:
        raise ValueError("All inputs must have the same sample rate")

    raw, minus, primary_vocals, secondary_vocals = _match_length(raw, minus, primary_vocals, secondary_vocals)
    sample_rate = raw_sr

    n_fft = 2048
    hop_length = 512
    win_length = 2048
    refined = minus.copy()
    metrics: dict[str, float] = {}

    for channel_index in range(minus.shape[1]):
        raw_ch = raw[:, channel_index]
        minus_ch = minus[:, channel_index]
        primary_ch = primary_vocals[:, channel_index]
        secondary_ch = secondary_vocals[:, channel_index]

        freqs, _, raw_z = stft(raw_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")
        _, _, minus_z = stft(minus_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")
        _, _, primary_z = stft(primary_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")
        _, _, secondary_z = stft(secondary_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")

        raw_mag = np.abs(raw_z)
        minus_mag = np.abs(minus_z)
        vocal_mag = np.maximum(np.abs(primary_z), float(args.secondary_weight) * np.abs(secondary_z))

        voice_band = (freqs >= 250.0) & (freqs < 4500.0)
        time_activity = vocal_mag[voice_band].mean(axis=0) / np.maximum(raw_mag[voice_band].mean(axis=0), 1e-8)
        time_activity = np.clip((time_activity - float(args.activity_threshold)) / max(1.0 - float(args.activity_threshold), 1e-8), 0.0, 1.0)
        time_activity = _smooth(time_activity.astype(np.float32), width=9)

        raw_mask = np.clip(vocal_mag / np.maximum(raw_mag, 1e-8), 0.0, 1.0)
        minus_mask = np.clip(vocal_mag / np.maximum(vocal_mag + minus_mag, 1e-8), 0.0, 1.0)
        mask = np.maximum(raw_mask, minus_mask)

        freq_weight, min_gain = _freq_weight(freqs)
        attenuation = float(args.strength) * np.power(mask, float(args.mask_power)) * freq_weight[:, None] * time_activity[None, :]
        gain = np.clip(1.0 - attenuation, min_gain[:, None], 1.0)

        refined_z = minus_z * gain
        _, refined_ch = istft(
            refined_z,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        copy_len = min(refined.shape[0], refined_ch.shape[0])
        refined[:copy_len, channel_index] = refined_ch[:copy_len]

        if channel_index == 0:
            metrics = {
                "dense_overlap_primary_before": _band_overlap(minus_mag, np.abs(primary_z), freqs, lo=250.0, hi=4500.0),
                "dense_overlap_primary_after": _band_overlap(np.abs(refined_z), np.abs(primary_z), freqs, lo=250.0, hi=4500.0),
                "dense_overlap_secondary_before": _band_overlap(minus_mag, np.abs(secondary_z), freqs, lo=250.0, hi=4500.0),
                "dense_overlap_secondary_after": _band_overlap(np.abs(refined_z), np.abs(secondary_z), freqs, lo=250.0, hi=4500.0),
                "time_activity_mean": float(time_activity.mean()),
                "time_activity_max": float(time_activity.max()),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = output_dir / "minus_refined.flac"
    raw_ref_path = output_dir / "raw_ref.flac"
    minus_base_path = output_dir / "minus_base.flac"
    sf.write(str(refined_path), refined, sample_rate, subtype="PCM_24")
    sf.write(str(raw_ref_path), raw, sample_rate, subtype="PCM_24")
    sf.write(str(minus_base_path), minus, sample_rate, subtype="PCM_24")

    manifest = {
        "raw_audio_path": str(raw_audio_path),
        "minus_audio_path": str(minus_audio_path),
        "primary_vocals_path": str(primary_vocals_path),
        "secondary_vocals_path": str(secondary_vocals_path),
        "artifacts": {
            "raw_ref_flac": str(raw_ref_path),
            "minus_base_flac": str(minus_base_path),
            "minus_refined_flac": str(refined_path),
        },
        "cleanup": {
            "name": "minus_vocal_residue_cleanup_v1",
            "strength": float(args.strength),
            "secondary_weight": float(args.secondary_weight),
            "mask_power": float(args.mask_power),
            "activity_threshold": float(args.activity_threshold),
        },
        "audio": {
            "sample_rate": int(sample_rate),
            "frames": int(refined.shape[0]),
            "duration_sec": round(float(refined.shape[0]) / float(sample_rate), 6),
        },
        "metrics": metrics,
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(listen_dir / "00_raw_ref.flac"), raw, sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / "01_minus_base.flac"), minus, sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / args.listen_label), refined, sample_rate, subtype="PCM_24")
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw reference\n01 = BS Mamba2 minus base\n02 = residue-cleaned minus\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    "00_raw_ref.flac": str(raw_ref_path),
                    "01_minus_base.flac": str(minus_base_path),
                    args.listen_label: str(refined_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
