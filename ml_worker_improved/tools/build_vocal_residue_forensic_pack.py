from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


EPSILON = 1e-12


def load_audio(path: Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        audio = signal.resample_poly(audio, target_sample_rate, sample_rate, axis=0).astype(
            np.float32
        )
        sample_rate = target_sample_rate
    return audio, sample_rate


def match_length(audio_items: list[np.ndarray]) -> list[np.ndarray]:
    frame_count = min(item.shape[0] for item in audio_items)
    return [item[:frame_count] for item in audio_items]


def stereo_from_mono(audio: np.ndarray) -> np.ndarray:
    return np.column_stack([audio, audio]).astype("float32")


def normalize_diagnostic_audio(audio: np.ndarray, *, target_peak: float = 0.74) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + EPSILON)
    return (audio * min(target_peak / peak, 20.0)).astype("float32")


def build_masked_residue_proxy(
    *, minus_mono: np.ndarray, vocal_mono: np.ndarray, sample_rate: int
) -> np.ndarray:
    n_fft = 4096
    hop_length = 1024
    frequencies, _, minus_stft = signal.stft(
        minus_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
    )
    _, _, vocal_stft = signal.stft(
        vocal_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
    )
    voice_band = ((frequencies >= 160.0) & (frequencies <= 5200.0)).astype(np.float32)[:, None]
    vocal_magnitude = np.abs(vocal_stft)
    voice_band_magnitude = vocal_magnitude[(frequencies >= 160.0) & (frequencies <= 5200.0), :]
    frame_activity = voice_band_magnitude.mean(axis=0)
    activity_floor = max(float(np.percentile(frame_activity, 65.0)), EPSILON)
    activity_ceiling = float(np.percentile(frame_activity, 95.0))
    activity = np.clip(
        (frame_activity - activity_floor) / (activity_ceiling - activity_floor + EPSILON),
        0.0,
        1.0,
    )
    mask = (
        np.clip(
            (vocal_magnitude / (np.percentile(voice_band_magnitude, 90.0) + EPSILON)) ** 0.65,
            0.0,
            1.0,
        )
        * voice_band
        * activity[None, :]
    )
    _, residue = signal.istft(
        minus_stft * mask,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
    )
    residue = residue[: minus_mono.shape[0]].astype("float32")
    voice_band_filter = signal.butter(
        4,
        [160.0 / (sample_rate / 2.0), 5200.0 / (sample_rate / 2.0)],
        btype="bandpass",
        output="sos",
    )
    return signal.sosfiltfilt(voice_band_filter, residue).astype("float32")


def build_projection_proxy(
    *, minus_mono: np.ndarray, vocal_mono: np.ndarray, sample_rate: int
) -> np.ndarray:
    projection = np.zeros_like(minus_mono, dtype="float32")
    frame_size = max(int(sample_rate), 1)
    for start in range(0, minus_mono.shape[0], frame_size):
        end = min(minus_mono.shape[0], start + frame_size)
        minus_window = minus_mono[start:end]
        vocal_window = vocal_mono[start:end]
        scale = float(
            np.dot(minus_window, vocal_window) / (np.dot(vocal_window, vocal_window) + EPSILON)
        )
        projection[start:end] = (scale * vocal_window).astype("float32")
    voice_band_filter = signal.butter(
        4,
        [160.0 / (sample_rate / 2.0), 5200.0 / (sample_rate / 2.0)],
        btype="bandpass",
        output="sos",
    )
    return signal.sosfiltfilt(voice_band_filter, projection).astype("float32")


def classify_vocal_residue_window(window: dict[str, float]) -> dict[str, Any]:
    projection_strong = (
        window["vocal_projection_ratio_db"] >= -18.0
        or window["vocal_residue_corr"] >= 0.14
    )
    masked_residue_strong = window["masked_residue_ratio_db"] >= -20.0

    if projection_strong and not masked_residue_strong:
        return {
            "label": "projection_only_melody_contamination",
            "vocal_evidence_status": "rejected_as_projection_only",
            "reason": "projection_proxy_is_strong_but_actual_masked_residue_is_weak",
        }
    if projection_strong and masked_residue_strong:
        return {
            "label": "needs_residue_review",
            "vocal_evidence_status": "needs_human_review",
            "reason": "projection_and_actual_masked_residue_are_both_active",
        }
    return {
        "label": "low_priority_residue",
        "vocal_evidence_status": "low_priority",
        "reason": "projection_proxy_is_not_strong_enough",
    }


def calculate_window_metrics(
    *,
    minus_mono: np.ndarray,
    vocal_mono: np.ndarray,
    masked_residue: np.ndarray,
    sample_rate: int,
    window_sec: float,
) -> list[dict[str, Any]]:
    window_size = max(int(sample_rate * window_sec), 1)
    windows: list[dict[str, Any]] = []
    for start in range(0, minus_mono.shape[0], window_size):
        end = min(minus_mono.shape[0], start + window_size)
        if end - start < sample_rate:
            continue
        minus_window = minus_mono[start:end]
        vocal_window = vocal_mono[start:end]
        residue_window = masked_residue[start:end]
        corr = abs(
            float(
                np.dot(minus_window, vocal_window)
                / (np.linalg.norm(minus_window) * np.linalg.norm(vocal_window) + EPSILON)
            )
        )
        projection_scale = float(
            np.dot(minus_window, vocal_window) / (np.dot(vocal_window, vocal_window) + EPSILON)
        )
        projected_vocal = projection_scale * vocal_window
        projection_ratio_db = float(
            20.0
            * math.log10(
                (float(np.sqrt(np.mean(projected_vocal * projected_vocal))) + EPSILON)
                / (float(np.sqrt(np.mean(minus_window * minus_window))) + EPSILON)
            )
        )
        masked_residue_ratio_db = float(
            20.0
            * math.log10(
                (float(np.sqrt(np.mean(residue_window * residue_window))) + EPSILON)
                / (float(np.sqrt(np.mean(minus_window * minus_window))) + EPSILON)
            )
        )
        window = {
            "start_sec": round(float(start) / float(sample_rate), 6),
            "end_sec": round(float(end) / float(sample_rate), 6),
            "vocal_residue_corr": corr,
            "vocal_projection_ratio_db": projection_ratio_db,
            "masked_residue_ratio_db": masked_residue_ratio_db,
            "suspicion_score": corr
            + max((projection_ratio_db + 30.0) / 30.0, 0.0) * 0.12
            + max((masked_residue_ratio_db + 45.0) / 45.0, 0.0) * 0.08,
        }
        window.update(classify_vocal_residue_window(window))
        windows.append(window)
    return sorted(windows, key=lambda item: item["suspicion_score"], reverse=True)


def write_window_clips(
    *,
    windows: list[dict[str, Any]],
    minus_audio: np.ndarray,
    masked_residue_boosted: np.ndarray,
    projection_boosted: np.ndarray,
    output_dir: Path,
    sample_rate: int,
    clip_count: int,
) -> list[dict[str, Any]]:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    for index, window in enumerate(windows[:clip_count], start=1):
        start_frame = max(int(window["start_sec"] * sample_rate) - int(0.5 * sample_rate), 0)
        end_frame = min(
            int(window["end_sec"] * sample_rate) + int(0.5 * sample_rate),
            minus_audio.shape[0],
        )
        prefix = "top%02d_%03d_%03d_%s" % (
            index,
            int(window["start_sec"]),
            int(window["end_sec"]),
            window["label"],
        )
        minus_clip = clips_dir / f"{prefix}_minus.flac"
        residue_clip = clips_dir / f"{prefix}_actual_masked_residue_boosted.flac"
        projection_clip = clips_dir / f"{prefix}_projection_proxy_reference_only.flac"
        sf.write(str(minus_clip), minus_audio[start_frame:end_frame], sample_rate, subtype="PCM_24")
        sf.write(
            str(residue_clip),
            stereo_from_mono(masked_residue_boosted[start_frame:end_frame]),
            sample_rate,
            subtype="PCM_24",
        )
        sf.write(
            str(projection_clip),
            stereo_from_mono(projection_boosted[start_frame:end_frame]),
            sample_rate,
            subtype="PCM_24",
        )
        clip_record = dict(window)
        clip_record.update(
            {
                "minus_clip": str(minus_clip),
                "actual_masked_residue_clip": str(residue_clip),
                "projection_proxy_reference_clip": str(projection_clip),
            }
        )
        clips.append(clip_record)
    return clips


def build_forensic_pack(args: argparse.Namespace) -> dict[str, Any]:
    minus_path = Path(args.minus_audio_path)
    vocal_guide_path = Path(args.vocal_guide_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minus_audio, sample_rate = load_audio(minus_path)
    vocal_audio, _ = load_audio(vocal_guide_path, sample_rate)
    minus_audio, vocal_audio = match_length([minus_audio, vocal_audio])
    minus_mono = minus_audio.mean(axis=1)
    vocal_mono = vocal_audio.mean(axis=1)

    masked_residue = build_masked_residue_proxy(
        minus_mono=minus_mono,
        vocal_mono=vocal_mono,
        sample_rate=sample_rate,
    )
    projection_proxy = build_projection_proxy(
        minus_mono=minus_mono,
        vocal_mono=vocal_mono,
        sample_rate=sample_rate,
    )
    masked_residue_boosted = normalize_diagnostic_audio(masked_residue)
    projection_boosted = normalize_diagnostic_audio(projection_proxy)

    current_minus_path = output_dir / "00_current_release_minus.flac"
    masked_residue_path = output_dir / "01_actual_masked_residue_boosted.flac"
    projection_path = output_dir / "02_projection_proxy_reference_only_rejected_as_vocal_evidence.flac"
    shutil.copyfile(minus_path, current_minus_path)
    sf.write(
        str(masked_residue_path),
        stereo_from_mono(masked_residue_boosted),
        sample_rate,
        subtype="PCM_24",
    )
    sf.write(
        str(projection_path),
        stereo_from_mono(projection_boosted),
        sample_rate,
        subtype="PCM_24",
    )

    windows = calculate_window_metrics(
        minus_mono=minus_mono,
        vocal_mono=vocal_mono,
        masked_residue=masked_residue,
        sample_rate=sample_rate,
        window_sec=float(args.window_sec),
    )
    clips = write_window_clips(
        windows=windows,
        minus_audio=minus_audio,
        masked_residue_boosted=masked_residue_boosted,
        projection_boosted=projection_boosted,
        output_dir=output_dir,
        sample_rate=sample_rate,
        clip_count=int(args.clip_count),
    )
    status_counts: dict[str, int] = {}
    for window in windows:
        status = str(window["vocal_evidence_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    manifest = {
        "schema_version": 2,
        "mode": "diagnostic_vocal_residue_forensic_listen_pack",
        "production_delivery": False,
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_forensic_evidence_only",
        "projection_proxy_policy": {
            "projection_only_is_not_vocal_evidence": True,
            "projection_proxy_file_is_reference_only": str(projection_path),
        },
        "source_minus_path": str(minus_path),
        "vocal_guide_path": str(vocal_guide_path),
        "source_release_manifest": str(Path(args.source_release_manifest))
        if args.source_release_manifest
        else None,
        "artifacts": {
            "current_release_minus": str(current_minus_path),
            "actual_masked_residue_boosted": str(masked_residue_path),
            "projection_proxy_reference_only": str(projection_path),
            "clips_dir": str(output_dir / "clips"),
        },
        "window_status_counts": status_counts,
        "top_suspicious_windows": clips,
        "note": (
            "Diagnostic-only. Projection-only clips are reference evidence and must not be "
            "treated as vocal residue because harmonic melody can correlate with the vocal guide."
        ),
    }
    (output_dir / "vocal_residue_forensic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "00_README.txt").write_text(
        "Diagnostic-only forensic pack. Listen to 01_actual_masked_residue_boosted first. "
        "02_projection_proxy_reference_only is intentionally rejected as standalone vocal evidence "
        "because it can contain melodic minus harmonics.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("minus_audio_path")
    parser.add_argument("vocal_guide_path")
    parser.add_argument("output_dir")
    parser.add_argument("--source-release-manifest")
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--clip-count", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_vocal_residue_forensic_pack.py",
        entrypoint_path="/app/tools/build_vocal_residue_forensic_pack.py",
    )
    args = parse_args(argv)
    manifest = build_forensic_pack(args)
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "delivery_status": manifest["delivery_status"],
                "window_status_counts": manifest["window_status_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
