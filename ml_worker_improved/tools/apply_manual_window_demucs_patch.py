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


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    width = max(edge1 - edge0, 1e-8)
    t = np.clip((x - edge0) / width, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _time_gate(times: np.ndarray, *, start_sec: float, end_sec: float, edge_sec: float) -> np.ndarray:
    rise = _smoothstep(start_sec - edge_sec, start_sec + edge_sec, times)
    fall = 1.0 - _smoothstep(end_sec - edge_sec, end_sec + edge_sec, times)
    return np.clip(rise * fall, 0.0, 1.0).astype(np.float32)


def _freq_gate(freqs: np.ndarray, *, lo_soft: float, lo_full: float, hi_full: float, hi_soft: float) -> np.ndarray:
    rise = _smoothstep(lo_soft, lo_full, freqs)
    fall = 1.0 - _smoothstep(hi_full, hi_soft, freqs)
    return np.clip(rise * fall, 0.0, 1.0).astype(np.float32)


def _band_overlap(minus_mag: np.ndarray, vocal_mag: np.ndarray, freqs: np.ndarray, *, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    a = minus_mag[mask]
    b = vocal_mag[mask]
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.minimum(a, b).sum() / max(b.sum(), 1e-9))


def _window_mask(times: np.ndarray, *, start_sec: float, end_sec: float) -> np.ndarray:
    return (times >= start_sec) & (times <= end_sec)


def main() -> None:
    require_docker_runtime(
        tool_name="apply_manual_window_demucs_patch.py",
        entrypoint_path="/app/tools/apply_manual_window_demucs_patch.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_audio_path")
    parser.add_argument("base_minus_path")
    parser.add_argument("guide_minus_path")
    parser.add_argument("primary_vocals_path")
    parser.add_argument("secondary_vocals_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_manual_window_patch.flac")
    parser.add_argument("--start-sec", type=float, default=50.0)
    parser.add_argument("--end-sec", type=float, default=70.0)
    parser.add_argument("--edge-sec", type=float, default=0.75)
    parser.add_argument("--strength", type=float, default=0.88)
    parser.add_argument("--secondary-weight", type=float, default=0.82)
    parser.add_argument("--mask-power", type=float, default=1.45)
    parser.add_argument("--guide-power", type=float, default=1.2)
    parser.add_argument("--lo-soft", type=float, default=180.0)
    parser.add_argument("--lo-full", type=float, default=280.0)
    parser.add_argument("--hi-full", type=float, default=4300.0)
    parser.add_argument("--hi-soft", type=float, default=6200.0)
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
    metrics: dict[str, float] = {}

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
        primary_mag = np.abs(primary_z)
        secondary_mag = np.abs(secondary_z)
        vocal_mag = np.maximum(primary_mag, float(args.secondary_weight) * secondary_mag)

        raw_mask = np.clip(vocal_mag / np.maximum(raw_mag, 1e-8), 0.0, 1.0)
        base_mask = np.clip(vocal_mag / np.maximum(vocal_mag + base_mag, 1e-8), 0.0, 1.0)
        cleanup_mask = np.maximum(raw_mask, base_mask)
        guide_improvement = np.power(
            np.clip((base_mag - guide_mag) / np.maximum(base_mag, 1e-8), 0.0, 1.0),
            float(args.guide_power),
        )

        time_gate = _time_gate(
            times,
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec),
            edge_sec=float(args.edge_sec),
        )
        freq_gate = _freq_gate(
            freqs,
            lo_soft=float(args.lo_soft),
            lo_full=float(args.lo_full),
            hi_full=float(args.hi_full),
            hi_soft=float(args.hi_soft),
        )

        blend = (
            float(args.strength)
            * np.power(cleanup_mask, float(args.mask_power))
            * guide_improvement
            * freq_gate[:, None]
            * time_gate[None, :]
        )
        blend = np.clip(blend, 0.0, 1.0)

        refined_mag = base_mag * (1.0 - blend) + guide_mag * blend
        refined_z = refined_mag * np.exp(1j * np.angle(base_z))
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
            focus_mask = _window_mask(times, start_sec=float(args.start_sec), end_sec=float(args.end_sec))
            metrics = {
                "focus_start_sec": float(args.start_sec),
                "focus_end_sec": float(args.end_sec),
                "dense_overlap_primary_before": _band_overlap(base_mag[:, focus_mask], primary_mag[:, focus_mask], freqs, lo=250.0, hi=4500.0),
                "dense_overlap_primary_after": _band_overlap(np.abs(refined_z)[:, focus_mask], primary_mag[:, focus_mask], freqs, lo=250.0, hi=4500.0),
                "dense_overlap_secondary_before": _band_overlap(base_mag[:, focus_mask], secondary_mag[:, focus_mask], freqs, lo=250.0, hi=4500.0),
                "dense_overlap_secondary_after": _band_overlap(np.abs(refined_z)[:, focus_mask], secondary_mag[:, focus_mask], freqs, lo=250.0, hi=4500.0),
                "focus_voice_band_mean_base": float(np.mean(base_mag[(freqs >= 250.0) & (freqs < 4500.0)][:, focus_mask])),
                "focus_voice_band_mean_after": float(np.mean(np.abs(refined_z)[(freqs >= 250.0) & (freqs < 4500.0)][:, focus_mask])),
                "focus_air_band_mean_base": float(np.mean(base_mag[(freqs >= 4500.0) & (freqs < 8000.0)][:, focus_mask])),
                "focus_air_band_mean_after": float(np.mean(np.abs(refined_z)[(freqs >= 4500.0) & (freqs < 8000.0)][:, focus_mask])),
                "time_gate_mean": float(time_gate.mean()),
                "time_gate_max": float(time_gate.max()),
                "blend_mean_focus": float(np.mean(blend[:, focus_mask])),
                "blend_max_focus": float(np.max(blend[:, focus_mask])),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = output_dir / "minus_manual_window_patch.flac"
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
            "minus_manual_window_patch_flac": str(refined_path),
        },
        "cleanup": {
            "name": "manual_window_demucs_patch_v1",
            "start_sec": float(args.start_sec),
            "end_sec": float(args.end_sec),
            "edge_sec": float(args.edge_sec),
            "strength": float(args.strength),
            "secondary_weight": float(args.secondary_weight),
            "mask_power": float(args.mask_power),
            "guide_power": float(args.guide_power),
            "lo_soft": float(args.lo_soft),
            "lo_full": float(args.lo_full),
            "hi_full": float(args.hi_full),
            "hi_soft": float(args.hi_soft),
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
            "Listen only here.\n00 = raw reference\n01 = BS Mamba2 refined minus\n02 = manual 50-70s Demucs-guided patch\n",
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
