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
from scipy.signal import istft, stft


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    return audio.astype(np.float32), int(sample_rate)


def _match_length(*arrays: np.ndarray) -> list[np.ndarray]:
    length = min(arr.shape[0] for arr in arrays)
    return [arr[:length] for arr in arrays]


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


def _band_mean(magnitude: np.ndarray, freqs: np.ndarray, *, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    band = magnitude[mask]
    if band.size == 0:
        return 0.0
    return float(np.mean(band))


def _freq_weight(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.zeros_like(freqs, dtype=np.float32)
    min_gain = np.ones_like(freqs, dtype=np.float32)

    band_low = (freqs >= 120.0) & (freqs < 250.0)
    band_voice = (freqs >= 250.0) & (freqs < 4500.0)
    band_presence = (freqs >= 4500.0) & (freqs < 8000.0)

    weight[band_low] = 0.22
    weight[band_voice] = 1.0
    weight[band_presence] = 0.18

    min_gain[freqs < 120.0] = 0.99
    min_gain[band_low] = 0.86
    min_gain[band_voice] = 0.62
    min_gain[band_presence] = 0.89

    return weight, min_gain


def _extract_windows(gate: np.ndarray, times: np.ndarray, *, threshold: float) -> list[dict[str, float]]:
    active = gate >= threshold
    windows: list[dict[str, float]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            end = index
            windows.append(
                {
                    "start_sec": float(times[start]),
                    "end_sec": float(times[end - 1]),
                    "mean_gate": float(gate[start:end].mean()),
                    "max_gate": float(gate[start:end].max()),
                }
            )
            start = None
    if start is not None:
        windows.append(
            {
                "start_sec": float(times[start]),
                "end_sec": float(times[len(active) - 1]),
                "mean_gate": float(gate[start:].mean()),
                "max_gate": float(gate[start:].max()),
            }
        )
    return windows


def _frame_score_gate(
    frame_score: np.ndarray,
    *,
    score_percentile: float,
    smooth_width: int,
    apply_threshold: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    positive_scores = frame_score[frame_score > 0.0]
    if positive_scores.size == 0:
        return np.zeros_like(frame_score, dtype=np.float32), {
            "frame_score_threshold": 1.0,
            "frame_score_max": 0.0,
            "positive_frame_ratio": 0.0,
            "gate_mode": "empty_score",
        }

    score_max = float(np.max(positive_scores))
    score_threshold = float(np.percentile(positive_scores, float(score_percentile)))
    gate_mode = "continuous_percentile"
    if score_threshold >= score_max:
        score_threshold = max(0.0, score_max * 0.5)
        gate_mode = "continuous_max_guard"

    soft_gate = np.clip(
        (frame_score - score_threshold) / max(score_max - score_threshold, 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    smoothed_gate = _smooth(soft_gate, width=int(smooth_width))
    gate = np.clip(
        (smoothed_gate - float(apply_threshold)) / max(1.0 - float(apply_threshold), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)

    if not np.any(gate > 0.0):
        recovered_gate = _smooth(
            np.clip(frame_score / max(score_max, 1e-8), 0.0, 1.0).astype(np.float32),
            width=int(smooth_width),
        )
        gate = np.clip(
            (recovered_gate - float(apply_threshold)) / max(1.0 - float(apply_threshold), 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        gate_mode = "continuous_sparse_recovery"

    return gate, {
        "frame_score_threshold": score_threshold,
        "frame_score_max": score_max,
        "positive_frame_ratio": float(np.mean(frame_score > 0.0)),
        "gate_mode": gate_mode,
    }


def main() -> None:
    require_docker_runtime(
        tool_name="apply_windowed_hybrid_minus_cleanup.py",
        entrypoint_path="/app/tools/apply_windowed_hybrid_minus_cleanup.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_audio_path")
    parser.add_argument("base_minus_path")
    parser.add_argument("guide_minus_path")
    parser.add_argument("primary_vocals_path")
    parser.add_argument("secondary_vocals_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_minus_windowed_hybrid.flac")
    parser.add_argument("--strength", type=float, default=0.84)
    parser.add_argument("--secondary-weight", type=float, default=0.82)
    parser.add_argument("--mask-power", type=float, default=1.55)
    parser.add_argument("--guide-power", type=float, default=1.25)
    parser.add_argument("--activity-threshold", type=float, default=0.06)
    parser.add_argument("--score-percentile", type=float, default=76.0)
    parser.add_argument("--gate-smooth-width", type=int, default=41)
    parser.add_argument("--gate-apply-threshold", type=float, default=0.18)
    args = parser.parse_args()

    raw_audio_path = Path(args.raw_audio_path)
    base_minus_path = Path(args.base_minus_path)
    guide_minus_path = Path(args.guide_minus_path)
    primary_vocals_path = Path(args.primary_vocals_path)
    secondary_vocals_path = Path(args.secondary_vocals_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    raw, raw_sr = _load_audio(raw_audio_path)
    base_minus, base_sr = _load_audio(base_minus_path)
    guide_minus, guide_sr = _load_audio(guide_minus_path)
    primary_vocals, primary_sr = _load_audio(primary_vocals_path)
    secondary_vocals, secondary_sr = _load_audio(secondary_vocals_path)

    if len({raw_sr, base_sr, guide_sr, primary_sr, secondary_sr}) != 1:
        raise ValueError("All inputs must have the same sample rate")

    raw, base_minus, guide_minus, primary_vocals, secondary_vocals = _match_length(
        raw, base_minus, guide_minus, primary_vocals, secondary_vocals
    )
    sample_rate = raw_sr

    n_fft = 2048
    hop_length = 512
    win_length = 2048
    refined = base_minus.copy()
    metrics: dict[str, float | list[dict[str, float]]] = {}

    for channel_index in range(base_minus.shape[1]):
        raw_ch = raw[:, channel_index]
        base_ch = base_minus[:, channel_index]
        guide_ch = guide_minus[:, channel_index]
        primary_ch = primary_vocals[:, channel_index]
        secondary_ch = secondary_vocals[:, channel_index]

        freqs, times, raw_z = stft(raw_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")
        _, _, base_z = stft(base_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")
        _, _, guide_z = stft(guide_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")
        _, _, primary_z = stft(primary_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")
        _, _, secondary_z = stft(secondary_ch, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, boundary="zeros")

        raw_mag = np.abs(raw_z)
        base_mag = np.abs(base_z)
        guide_mag = np.abs(guide_z)
        vocal_mag = np.maximum(np.abs(primary_z), float(args.secondary_weight) * np.abs(secondary_z))

        voice_band = (freqs >= 250.0) & (freqs < 4500.0)
        raw_voice_mean = np.maximum(raw_mag[voice_band].mean(axis=0), 1e-8)
        time_activity = vocal_mag[voice_band].mean(axis=0) / raw_voice_mean
        time_activity = np.clip(
            (time_activity - float(args.activity_threshold)) / max(1.0 - float(args.activity_threshold), 1e-8),
            0.0,
            1.0,
        )
        time_activity = _smooth(time_activity.astype(np.float32), width=9)

        raw_mask = np.clip(vocal_mag / np.maximum(raw_mag, 1e-8), 0.0, 1.0)
        base_mask = np.clip(vocal_mag / np.maximum(vocal_mag + base_mag, 1e-8), 0.0, 1.0)
        cleanup_mask = np.maximum(raw_mask, base_mask)

        guide_ratio = np.clip(guide_mag / np.maximum(base_mag, 1e-8), 0.0, 1.0)
        guide_improvement = np.power(np.clip(1.0 - guide_ratio, 0.0, 1.0), float(args.guide_power))

        frame_score = time_activity * guide_improvement[voice_band].mean(axis=0)
        gate, gate_metrics = _frame_score_gate(
            frame_score,
            score_percentile=float(args.score_percentile),
            smooth_width=int(args.gate_smooth_width),
            apply_threshold=float(args.gate_apply_threshold),
        )

        freq_weight, min_gain = _freq_weight(freqs)
        attenuation = (
            float(args.strength)
            * np.power(cleanup_mask, float(args.mask_power))
            * guide_improvement
            * freq_weight[:, None]
            * time_activity[None, :]
            * gate[None, :]
        )
        gain = np.clip(1.0 - attenuation, min_gain[:, None], 1.0)

        refined_z = base_z * gain
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
            windows = _extract_windows(gate, times, threshold=0.2)
            metrics = {
                "dense_overlap_primary_before": _band_overlap(base_mag, np.abs(primary_z), freqs, lo=250.0, hi=4500.0),
                "dense_overlap_primary_after": _band_overlap(np.abs(refined_z), np.abs(primary_z), freqs, lo=250.0, hi=4500.0),
                "dense_overlap_secondary_before": _band_overlap(base_mag, np.abs(secondary_z), freqs, lo=250.0, hi=4500.0),
                "dense_overlap_secondary_after": _band_overlap(np.abs(refined_z), np.abs(secondary_z), freqs, lo=250.0, hi=4500.0),
                "voice_band_mean_base": _band_mean(base_mag, freqs, lo=250.0, hi=4500.0),
                "voice_band_mean_after": _band_mean(np.abs(refined_z), freqs, lo=250.0, hi=4500.0),
                "air_band_mean_base": _band_mean(base_mag, freqs, lo=4500.0, hi=8000.0),
                "air_band_mean_after": _band_mean(np.abs(refined_z), freqs, lo=4500.0, hi=8000.0),
                **gate_metrics,
                "gate_mean": float(gate.mean()),
                "gate_max": float(gate.max()),
                "selected_windows": windows[:12],
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = output_dir / "minus_windowed_hybrid.flac"
    raw_ref_path = output_dir / "raw_ref.flac"
    minus_base_path = output_dir / "minus_base.flac"
    minus_guide_path = output_dir / "minus_guide.flac"
    sf.write(str(refined_path), refined, sample_rate, subtype="PCM_24")
    sf.write(str(raw_ref_path), raw, sample_rate, subtype="PCM_24")
    sf.write(str(minus_base_path), base_minus, sample_rate, subtype="PCM_24")
    sf.write(str(minus_guide_path), guide_minus, sample_rate, subtype="PCM_24")

    manifest = {
        "raw_audio_path": str(raw_audio_path),
        "base_minus_path": str(base_minus_path),
        "guide_minus_path": str(guide_minus_path),
        "primary_vocals_path": str(primary_vocals_path),
        "secondary_vocals_path": str(secondary_vocals_path),
        "artifacts": {
            "raw_ref_flac": str(raw_ref_path),
            "minus_base_flac": str(minus_base_path),
            "minus_guide_flac": str(minus_guide_path),
            "minus_windowed_hybrid_flac": str(refined_path),
        },
        "cleanup": {
            "name": "windowed_hybrid_minus_cleanup_v1",
            "strength": float(args.strength),
            "secondary_weight": float(args.secondary_weight),
            "mask_power": float(args.mask_power),
            "guide_power": float(args.guide_power),
            "activity_threshold": float(args.activity_threshold),
            "score_percentile": float(args.score_percentile),
            "gate_smooth_width": int(args.gate_smooth_width),
            "gate_apply_threshold": float(args.gate_apply_threshold),
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
        sf.write(str(listen_dir / "01_bs_minus_refined.flac"), base_minus, sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / args.listen_label), refined, sample_rate, subtype="PCM_24")
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw reference\n01 = BS Mamba2 refined minus\n02 = window-specific hybrid cleanup\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    "00_raw_ref.flac": str(raw_ref_path),
                    "01_bs_minus_refined.flac": str(minus_base_path),
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
