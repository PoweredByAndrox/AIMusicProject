from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_vocal_residue_forensic_pack import (  # noqa: E402
    build_masked_residue_proxy,
    calculate_window_metrics,
    load_audio,
    match_length,
)
from docker_runtime import require_docker_runtime  # noqa: E402


EPSILON = 1e-12


def build_review_window_gate(
    *,
    windows: list[dict[str, Any]],
    frame_count: int,
    sample_rate: int,
    smoothing_sec: float,
) -> np.ndarray:
    gate = np.zeros(frame_count, dtype=np.float32)
    for window in windows:
        if window.get("vocal_evidence_status") != "needs_human_review":
            continue
        start = max(int(float(window["start_sec"]) * sample_rate), 0)
        end = min(int(float(window["end_sec"]) * sample_rate), frame_count)
        if end > start:
            gate[start:end] = 1.0
    smoothing_frames = max(int(smoothing_sec * sample_rate), 1)
    if smoothing_frames > 1:
        kernel = np.hanning(smoothing_frames * 2 + 1).astype(np.float32)
        kernel /= float(kernel.sum()) + EPSILON
        gate = np.convolve(gate, kernel, mode="same").astype(np.float32)
    return np.clip(gate, 0.0, 1.0)


def build_actual_residue_gain_mask(
    *,
    base_minus: np.ndarray,
    primary_vocals: np.ndarray,
    review_gate: np.ndarray,
    sample_rate: int,
    strength: float,
    mask_power: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_fft = 4096
    hop_length = 1024
    frequencies, times, base_stft = signal.stft(
        base_minus,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
    )
    _, _, vocal_stft = signal.stft(
        primary_vocals,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
    )
    vocal_magnitude = np.abs(vocal_stft)
    voice_band = (frequencies >= 160.0) & (frequencies <= 5200.0)
    presence_band = (frequencies >= 1800.0) & (frequencies <= 4500.0)
    low_band = frequencies < 140.0
    time_gate = np.interp(
        times,
        np.arange(review_gate.shape[0], dtype=np.float64) / float(sample_rate),
        review_gate,
        left=0.0,
        right=0.0,
    ).astype(np.float32)
    voice_magnitude = vocal_magnitude[voice_band, :]
    if voice_magnitude.size == 0:
        raise ValueError("Voice band is empty for the selected sample rate.")
    mask_floor = float(np.percentile(voice_magnitude, 72.0)) + EPSILON
    residue_mask = np.clip((vocal_magnitude / mask_floor) ** mask_power, 0.0, 1.0)
    band_weight = np.zeros((frequencies.shape[0], 1), dtype=np.float32)
    band_weight[voice_band, 0] = 0.58
    band_weight[presence_band, 0] = 0.30
    band_weight[low_band, 0] = 0.0
    gain = 1.0 - float(strength) * residue_mask * band_weight * time_gate[None, :]
    minimum_gain = np.ones((frequencies.shape[0], 1), dtype=np.float32)
    minimum_gain[voice_band, 0] = 0.78
    minimum_gain[presence_band, 0] = 0.86
    minimum_gain[low_band, 0] = 0.995
    gain = np.maximum(gain, minimum_gain)
    return gain.astype(np.float32), {
        "mean_review_gate": float(time_gate.mean()),
        "max_review_gate": float(time_gate.max()),
        "active_frame_ratio": float(np.mean(time_gate > 0.05)),
    }


def apply_actual_masked_residue_cleanup(
    *,
    base_minus: np.ndarray,
    primary_vocals: np.ndarray,
    sample_rate: int,
    strength: float,
    mask_power: float,
    smoothing_sec: float,
    window_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_minus, primary_vocals = match_length([base_minus, primary_vocals])
    base_mono = base_minus.mean(axis=1)
    vocal_mono = primary_vocals.mean(axis=1)
    masked_residue = build_masked_residue_proxy(
        minus_mono=base_mono,
        vocal_mono=vocal_mono,
        sample_rate=sample_rate,
    )
    windows = calculate_window_metrics(
        minus_mono=base_mono,
        vocal_mono=vocal_mono,
        masked_residue=masked_residue,
        sample_rate=sample_rate,
        window_sec=window_sec,
    )
    review_gate = build_review_window_gate(
        windows=windows,
        frame_count=base_minus.shape[0],
        sample_rate=sample_rate,
        smoothing_sec=smoothing_sec,
    )
    if not np.any(review_gate > 0.0):
        return base_minus.copy(), {
            "cleanup_status": "skipped_no_needs_residue_review_windows",
            "windows": windows,
            "review_window_count": 0,
        }

    refined_channels: list[np.ndarray] = []
    gain_metrics: dict[str, float] | None = None
    for channel_index in range(base_minus.shape[1]):
        gain, gain_metrics = build_actual_residue_gain_mask(
            base_minus=base_minus[:, channel_index],
            primary_vocals=primary_vocals[:, min(channel_index, primary_vocals.shape[1] - 1)],
            review_gate=review_gate,
            sample_rate=sample_rate,
            strength=strength,
            mask_power=mask_power,
        )
        n_fft = 4096
        hop_length = 1024
        _, _, channel_stft = signal.stft(
            base_minus[:, channel_index],
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary="zeros",
        )
        _, refined_channel = signal.istft(
            channel_stft * gain,
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            input_onesided=True,
        )
        refined_channels.append(refined_channel[: base_minus.shape[0]].astype(np.float32))
    refined = np.column_stack(refined_channels).astype(np.float32)
    return refined, {
        "cleanup_status": "applied_actual_masked_residue_windows",
        "windows": windows,
        "review_window_count": sum(
            1 for window in windows if window.get("vocal_evidence_status") == "needs_human_review"
        ),
        "projection_only_rejected_count": sum(
            1
            for window in windows
            if window.get("vocal_evidence_status") == "rejected_as_projection_only"
        ),
        "gain_metrics": gain_metrics or {},
    }


def calculate_basic_metrics(base_minus: np.ndarray, refined_minus: np.ndarray) -> dict[str, float]:
    base_minus, refined_minus = match_length([base_minus, refined_minus])
    delta = refined_minus - base_minus
    base_rms = float(np.sqrt(np.mean(base_minus * base_minus)) + EPSILON)
    delta_rms = float(np.sqrt(np.mean(delta * delta)) + EPSILON)
    return {
        "refined_rms_dbfs": float(20.0 * np.log10(float(np.sqrt(np.mean(refined_minus * refined_minus))) + EPSILON)),
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
    parser.add_argument("--listen-label", default="02_actual_masked_residue_minus.flac")
    parser.add_argument("--strength", type=float, default=0.10)
    parser.add_argument("--mask-power", type=float, default=1.35)
    parser.add_argument("--smoothing-sec", type=float, default=0.10)
    parser.add_argument("--window-sec", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="apply_actual_masked_residue_minus_cleanup.py",
        entrypoint_path="/app/tools/apply_actual_masked_residue_minus_cleanup.py",
    )
    args = parse_args(argv)
    raw_audio_path = Path(args.raw_audio_path)
    base_minus_path = Path(args.base_minus_path)
    guide_minus_path = Path(args.guide_minus_path)
    primary_vocals_path = Path(args.primary_vocals_path)
    secondary_vocals_path = Path(args.secondary_vocals_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_audio, raw_sample_rate = load_audio(raw_audio_path)
    base_minus, base_sample_rate = load_audio(base_minus_path, raw_sample_rate)
    primary_vocals, _ = load_audio(primary_vocals_path, raw_sample_rate)
    raw_audio, base_minus, primary_vocals = match_length([raw_audio, base_minus, primary_vocals])
    refined_minus, cleanup_report = apply_actual_masked_residue_cleanup(
        base_minus=base_minus,
        primary_vocals=primary_vocals,
        sample_rate=raw_sample_rate,
        strength=float(args.strength),
        mask_power=float(args.mask_power),
        smoothing_sec=float(args.smoothing_sec),
        window_sec=float(args.window_sec),
    )
    metrics = calculate_basic_metrics(base_minus, refined_minus)

    raw_ref_path = output_dir / "raw_ref.flac"
    minus_base_path = output_dir / "minus_base.flac"
    refined_path = output_dir / "minus_actual_masked_residue.flac"
    sf.write(str(raw_ref_path), raw_audio, raw_sample_rate, subtype="PCM_24")
    sf.write(str(minus_base_path), base_minus, raw_sample_rate, subtype="PCM_24")
    sf.write(str(refined_path), refined_minus, raw_sample_rate, subtype="PCM_24")

    manifest = {
        "schema_version": 1,
        "raw_audio_path": str(raw_audio_path),
        "base_minus_path": str(base_minus_path),
        "guide_minus_path": str(guide_minus_path),
        "primary_vocals_path": str(primary_vocals_path),
        "secondary_vocals_path": str(secondary_vocals_path),
        "artifacts": {
            "raw_ref_flac": str(raw_ref_path),
            "minus_base_flac": str(minus_base_path),
            "minus_actual_masked_residue_flac": str(refined_path),
        },
        "cleanup": {
            "name": "actual_masked_residue_minus_cleanup_v1",
            "strength": float(args.strength),
            "mask_power": float(args.mask_power),
            "smoothing_sec": float(args.smoothing_sec),
            "window_sec": float(args.window_sec),
            "projection_only_is_not_vocal_evidence": True,
        },
        "audio": {
            "sample_rate": int(raw_sample_rate),
            "frames": int(refined_minus.shape[0]),
            "duration_sec": round(float(refined_minus.shape[0]) / float(raw_sample_rate), 6),
        },
        "metrics": metrics,
        "cleanup_report": cleanup_report,
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(listen_dir / "00_raw_ref.flac"), raw_audio, raw_sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / "01_minus_base.flac"), base_minus, raw_sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / args.listen_label), refined_minus, raw_sample_rate, subtype="PCM_24")
        (listen_dir / "00_README.txt").write_text(
            "Diagnostic candidate output. Projection-only melody contamination windows are excluded from cleanup.\n",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
