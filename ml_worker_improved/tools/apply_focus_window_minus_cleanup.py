from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import istft, stft

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from run_ralph_minus_improvement_loop import load_audio, load_focus_windows, match_length  # noqa: E402


EPSILON = 1e-12


def build_focus_window_gate(
    *,
    focus_windows: Sequence[dict[str, Any]],
    frame_count: int,
    sample_rate: int,
    smoothing_sec: float,
) -> np.ndarray:
    gate = np.zeros(frame_count, dtype=np.float32)
    for window in focus_windows:
        start = max(0, min(int(round(float(window["start_sec"]) * sample_rate)), frame_count))
        end = max(start, min(int(round(float(window["end_sec"]) * sample_rate)), frame_count))
        if end > start:
            gate[start:end] = 1.0
    smoothing_frames = max(int(round(float(smoothing_sec) * sample_rate)), 1)
    if smoothing_frames > 1:
        kernel = np.hanning(smoothing_frames * 2 + 1).astype(np.float32)
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0.0:
            kernel /= kernel_sum
            gate = np.convolve(gate, kernel, mode="same").astype(np.float32)
    return np.clip(gate, 0.0, 1.0)


def frequency_weight(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.zeros_like(freqs, dtype=np.float32)
    min_gain = np.ones_like(freqs, dtype=np.float32)

    low_voice = (freqs >= 160.0) & (freqs < 260.0)
    core_voice = (freqs >= 260.0) & (freqs < 5200.0)
    upper_presence = (freqs >= 5200.0) & (freqs < 7800.0)

    weight[low_voice] = 0.18
    weight[core_voice] = 1.0
    weight[upper_presence] = 0.12

    min_gain[freqs < 120.0] = 0.995
    min_gain[low_voice] = 0.92
    min_gain[core_voice] = 0.84
    min_gain[upper_presence] = 0.96
    return weight, min_gain


def apply_focus_window_cleanup(
    *,
    base_minus: np.ndarray,
    primary_vocals: np.ndarray,
    secondary_vocals: np.ndarray,
    focus_windows: Sequence[dict[str, Any]],
    sample_rate: int,
    strength: float,
    secondary_weight: float,
    mask_power: float,
    smoothing_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_minus, primary_vocals, secondary_vocals = match_length(
        [base_minus, primary_vocals, secondary_vocals]
    )
    if not focus_windows:
        return base_minus.copy(), {
            "cleanup_status": "skipped_no_focus_windows",
            "focus_window_count": 0,
        }

    sample_gate = build_focus_window_gate(
        focus_windows=focus_windows,
        frame_count=base_minus.shape[0],
        sample_rate=sample_rate,
        smoothing_sec=smoothing_sec,
    )
    if not np.any(sample_gate > 0.0):
        return base_minus.copy(), {
            "cleanup_status": "skipped_empty_focus_gate",
            "focus_window_count": len(focus_windows),
        }

    n_fft = 2048
    hop_length = 512
    win_length = 2048
    refined = base_minus.copy()
    first_channel_metrics: dict[str, Any] = {}

    for channel_index in range(base_minus.shape[1]):
        base_ch = base_minus[:, channel_index]
        primary_ch = primary_vocals[:, min(channel_index, primary_vocals.shape[1] - 1)]
        secondary_ch = secondary_vocals[:, min(channel_index, secondary_vocals.shape[1] - 1)]

        freqs, times, base_z = stft(
            base_ch,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        _, _, primary_z = stft(
            primary_ch,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        _, _, secondary_z = stft(
            secondary_ch,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )

        base_mag = np.abs(base_z)
        vocal_mag = np.maximum(np.abs(primary_z), float(secondary_weight) * np.abs(secondary_z))
        cleanup_mask = np.clip(vocal_mag / np.maximum(vocal_mag + base_mag, EPSILON), 0.0, 1.0)
        time_gate = np.interp(
            times,
            np.arange(sample_gate.shape[0], dtype=np.float64) / float(sample_rate),
            sample_gate,
            left=0.0,
            right=0.0,
        ).astype(np.float32)
        weight, min_gain = frequency_weight(freqs)
        attenuation = (
            float(strength)
            * np.power(cleanup_mask, float(mask_power))
            * weight[:, None]
            * time_gate[None, :]
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
        refined[:copy_len, channel_index] = refined_ch[:copy_len].astype(np.float32)

        if channel_index == 0:
            first_channel_metrics = {
                "focus_gate_mean": float(time_gate.mean()),
                "focus_gate_max": float(time_gate.max()),
                "focus_gate_active_ratio": float(np.mean(time_gate > 0.05)),
                "mean_gain": float(gain.mean()),
                "min_gain": float(gain.min()),
            }

    return refined.astype(np.float32), {
        "cleanup_status": "applied_focus_window_minus_cleanup",
        "focus_window_count": len(focus_windows),
        "focus_windows": list(focus_windows),
        "strength": float(strength),
        "secondary_weight": float(secondary_weight),
        "mask_power": float(mask_power),
        "smoothing_sec": float(smoothing_sec),
        "metrics": first_channel_metrics,
    }


def calculate_basic_metrics(base_minus: np.ndarray, refined_minus: np.ndarray) -> dict[str, float]:
    base_minus, refined_minus = match_length([base_minus, refined_minus])
    delta = refined_minus - base_minus
    base_rms = float(np.sqrt(np.mean(base_minus * base_minus)) + EPSILON)
    delta_rms = float(np.sqrt(np.mean(delta * delta)) + EPSILON)
    return {
        "refined_rms_dbfs": float(
            20.0 * np.log10(float(np.sqrt(np.mean(refined_minus * refined_minus))) + EPSILON)
        ),
        "delta_vs_base_rms_db": float(20.0 * np.log10(delta_rms / base_rms)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_audio_path")
    parser.add_argument("base_minus_path")
    parser.add_argument("guide_minus_path")
    parser.add_argument("primary_vocals_path")
    parser.add_argument("secondary_vocals_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_candidate_product_minus.flac")
    parser.add_argument("--focus-windows-jsonl", required=True)
    parser.add_argument("--focus-track-id", default="")
    parser.add_argument("--focus-candidate-id", default="")
    parser.add_argument("--max-focus-windows", type=int, default=16)
    parser.add_argument("--focus-include-low-confidence-accepted", action="store_true")
    parser.add_argument("--focus-max-accept-probability", type=float, default=0.0)
    parser.add_argument("--focus-min-block-probability", type=float, default=0.0)
    parser.add_argument("--strength", type=float, default=0.025)
    parser.add_argument("--secondary-weight", type=float, default=0.82)
    parser.add_argument("--mask-power", type=float, default=1.35)
    parser.add_argument("--smoothing-sec", type=float, default=0.08)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="apply_focus_window_minus_cleanup.py",
        entrypoint_path="/app/tools/apply_focus_window_minus_cleanup.py",
    )
    args = parse_args(argv)
    raw_audio_path = Path(args.raw_audio_path)
    base_minus_path = Path(args.base_minus_path)
    guide_minus_path = Path(args.guide_minus_path)
    primary_vocals_path = Path(args.primary_vocals_path)
    secondary_vocals_path = Path(args.secondary_vocals_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    raw_audio, sample_rate = load_audio(raw_audio_path)
    base_minus, _ = load_audio(base_minus_path, sample_rate)
    guide_minus, _ = load_audio(guide_minus_path, sample_rate)
    primary_vocals, _ = load_audio(primary_vocals_path, sample_rate)
    secondary_vocals, _ = load_audio(secondary_vocals_path, sample_rate)
    raw_audio, base_minus, guide_minus, primary_vocals, secondary_vocals = match_length(
        [raw_audio, base_minus, guide_minus, primary_vocals, secondary_vocals]
    )
    focus_windows = load_focus_windows(
        Path(args.focus_windows_jsonl),
        track_id=str(args.focus_track_id or ""),
        candidate_id=str(args.focus_candidate_id or ""),
        max_windows=int(args.max_focus_windows),
        include_low_confidence_accepts=bool(args.focus_include_low_confidence_accepted),
        max_accept_probability=float(args.focus_max_accept_probability),
        min_block_probability=float(args.focus_min_block_probability),
    )
    refined_minus, cleanup_report = apply_focus_window_cleanup(
        base_minus=base_minus,
        primary_vocals=primary_vocals,
        secondary_vocals=secondary_vocals,
        focus_windows=focus_windows,
        sample_rate=sample_rate,
        strength=float(args.strength),
        secondary_weight=float(args.secondary_weight),
        mask_power=float(args.mask_power),
        smoothing_sec=float(args.smoothing_sec),
    )
    metrics = calculate_basic_metrics(base_minus, refined_minus)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_mix_path = output_dir / "raw_mix.flac"
    base_product_minus_path = output_dir / "base_product_minus.flac"
    guide_product_minus_path = output_dir / "guide_product_minus.flac"
    refined_path = output_dir / "focus_product_minus.flac"
    sf.write(str(raw_mix_path), raw_audio, sample_rate, subtype="PCM_24")
    sf.write(str(base_product_minus_path), base_minus, sample_rate, subtype="PCM_24")
    sf.write(str(guide_product_minus_path), guide_minus, sample_rate, subtype="PCM_24")
    sf.write(str(refined_path), refined_minus, sample_rate, subtype="PCM_24")

    manifest = {
        "schema_version": 1,
        "stem_type": "minus",
        "artifact_role": "candidate_product_minus",
        "candidate_product_artifact_key": "focus_product_minus_flac",
        "raw_audio_path": str(raw_audio_path),
        "base_minus_path": str(base_minus_path),
        "guide_minus_path": str(guide_minus_path),
        "primary_vocals_path": str(primary_vocals_path),
        "secondary_vocals_path": str(secondary_vocals_path),
        "focus_windows_jsonl": str(args.focus_windows_jsonl),
        "focus_track_id": str(args.focus_track_id or ""),
        "focus_candidate_id": str(args.focus_candidate_id or ""),
        "artifacts": {
            "raw_mix_flac": str(raw_mix_path),
            "base_product_minus_flac": str(base_product_minus_path),
            "guide_product_minus_flac": str(guide_product_minus_path),
            "focus_product_minus_flac": str(refined_path),
            "raw_ref_flac": str(raw_mix_path),
            "minus_base_flac": str(base_product_minus_path),
            "minus_guide_flac": str(guide_product_minus_path),
            "minus_focus_window_cleanup_flac": str(refined_path),
        },
        "cleanup": {
            "name": "focus_window_minus_cleanup_v1",
            "source": "target_audio_error_blocked_windows",
            "no_per_track_timestamps": True,
            "strength": float(args.strength),
            "secondary_weight": float(args.secondary_weight),
            "mask_power": float(args.mask_power),
            "smoothing_sec": float(args.smoothing_sec),
            "max_focus_windows": int(args.max_focus_windows),
        },
        "audio": {
            "sample_rate": int(sample_rate),
            "frames": int(refined_minus.shape[0]),
            "duration_sec": round(float(refined_minus.shape[0]) / float(sample_rate), 6),
        },
        "metrics": metrics,
        "cleanup_report": cleanup_report,
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(listen_dir / "00_raw_mix.flac"), raw_audio, sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / "01_base_product_minus.flac"), base_minus, sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / args.listen_label), refined_minus, sample_rate, subtype="PCM_24")
        (listen_dir / "00_README.txt").write_text(
            "Diagnostic focus-window candidate. 00 = raw mix, 01 = base product minus, 02 = candidate product minus. RALPH manifest decides whether it is deliverable.\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    "00_raw_mix.flac": str(raw_mix_path),
                    "01_base_product_minus.flac": str(base_product_minus_path),
                    args.listen_label: str(refined_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
