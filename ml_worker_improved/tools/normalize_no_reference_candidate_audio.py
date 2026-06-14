from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audio_info(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def duration_tolerance(input_duration_sec: float, *, max_duration_drift_sec: float, max_duration_drift_ratio: float) -> float:
    return max(float(max_duration_drift_sec), float(input_duration_sec) * float(max_duration_drift_ratio))


def shape_mismatch_reasons(input_info: dict[str, Any], raw_info: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(input_info["sample_rate"]) != int(raw_info["sample_rate"]):
        reasons.append("sample_rate_mismatch")
    if int(input_info["channels"]) != int(raw_info["channels"]):
        reasons.append("channel_mismatch")
    if int(input_info["frames"]) != int(raw_info["frames"]):
        reasons.append("duration_drift")
    return reasons


def _match_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    current_channels = int(audio.shape[1])
    if current_channels == target_channels:
        return audio
    if target_channels <= 0:
        raise ValueError("target channel count must be positive")
    if target_channels == 1:
        return audio.mean(axis=1, keepdims=True).astype(np.float32)
    if current_channels == 1:
        return np.repeat(audio, target_channels, axis=1).astype(np.float32)
    if current_channels > target_channels:
        return audio[:, :target_channels].astype(np.float32)
    extra = np.repeat(audio[:, -1:], target_channels - current_channels, axis=1)
    return np.concatenate([audio, extra], axis=1).astype(np.float32)


def _resample_linear(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if int(source_sample_rate) == int(target_sample_rate):
        return audio.astype(np.float32)
    source_frames = int(audio.shape[0])
    if source_frames <= 0:
        raise ValueError("raw output contains no frames")
    target_frames = max(1, int(round(source_frames * float(target_sample_rate) / float(source_sample_rate))))
    if source_frames == 1:
        return np.repeat(audio, target_frames, axis=0).astype(np.float32)
    source_positions = np.linspace(0.0, float(source_frames - 1), num=source_frames, dtype=np.float64)
    target_positions = np.linspace(0.0, float(source_frames - 1), num=target_frames, dtype=np.float64)
    resampled = np.empty((target_frames, int(audio.shape[1])), dtype=np.float32)
    for channel in range(int(audio.shape[1])):
        resampled[:, channel] = np.interp(target_positions, source_positions, audio[:, channel]).astype(np.float32)
    return resampled


def _match_frames(audio: np.ndarray, target_frames: int) -> np.ndarray:
    current_frames = int(audio.shape[0])
    if current_frames == target_frames:
        return audio.astype(np.float32)
    if current_frames > target_frames:
        return audio[:target_frames].astype(np.float32)
    pad = np.zeros((target_frames - current_frames, int(audio.shape[1])), dtype=np.float32)
    return np.concatenate([audio, pad], axis=0).astype(np.float32)


def normalize_candidate_audio_to_input_shape(
    *,
    input_audio_path: Path,
    raw_output_path: Path,
    normalized_output_path: Path,
    max_duration_drift_sec: float = 0.25,
    max_duration_drift_ratio: float = 0.02,
) -> dict[str, Any]:
    input_audio_path = Path(input_audio_path)
    raw_output_path = Path(raw_output_path)
    normalized_output_path = Path(normalized_output_path)
    if raw_output_path.resolve() == normalized_output_path.resolve():
        raise ValueError("normalized_output_path_must_not_overwrite_raw_output")

    input_info = audio_info(input_audio_path)
    raw_info = audio_info(raw_output_path)
    tolerance = duration_tolerance(
        float(input_info["duration_sec"]),
        max_duration_drift_sec=float(max_duration_drift_sec),
        max_duration_drift_ratio=float(max_duration_drift_ratio),
    )
    duration_delta = abs(float(input_info["duration_sec"]) - float(raw_info["duration_sec"]))
    if duration_delta > tolerance:
        raise ValueError(
            "duration_drift_exceeds_tolerance:"
            f"delta={duration_delta:.6f}:tolerance={tolerance:.6f}"
        )

    reasons = shape_mismatch_reasons(input_info, raw_info)
    if not reasons:
        return {
            "input_audio_path": str(input_audio_path),
            "raw_output_path": str(raw_output_path),
            "normalized_output_path": str(raw_output_path),
            "input_sample_rate": int(input_info["sample_rate"]),
            "input_channels": int(input_info["channels"]),
            "input_duration_sec": float(input_info["duration_sec"]),
            "raw_output_sample_rate": int(raw_info["sample_rate"]),
            "raw_output_channels": int(raw_info["channels"]),
            "raw_output_duration_sec": float(raw_info["duration_sec"]),
            "normalized_output_sample_rate": int(raw_info["sample_rate"]),
            "normalized_output_channels": int(raw_info["channels"]),
            "normalized_output_duration_sec": float(raw_info["duration_sec"]),
            "shape_normalization_applied": False,
            "shape_normalization_reason": "shape_already_valid",
            "original_files_overwritten": False,
            "release_ready": False,
        }

    audio, sample_rate = sf.read(str(raw_output_path), always_2d=True, dtype="float32")
    normalized = _match_channels(audio.astype(np.float32), int(input_info["channels"]))
    normalized = _resample_linear(normalized, int(sample_rate), int(input_info["sample_rate"]))
    normalized = _match_frames(normalized, int(input_info["frames"]))

    normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(normalized_output_path), normalized, int(input_info["sample_rate"]), subtype="PCM_24")
    normalized_info = audio_info(normalized_output_path)
    if (
        int(normalized_info["sample_rate"]) != int(input_info["sample_rate"])
        or int(normalized_info["channels"]) != int(input_info["channels"])
        or int(normalized_info["frames"]) != int(input_info["frames"])
    ):
        raise RuntimeError("shape_normalization_failed_to_match_input_shape")

    return {
        "input_audio_path": str(input_audio_path),
        "raw_output_path": str(raw_output_path),
        "normalized_output_path": str(normalized_output_path),
        "input_sample_rate": int(input_info["sample_rate"]),
        "input_channels": int(input_info["channels"]),
        "input_duration_sec": float(input_info["duration_sec"]),
        "raw_output_sample_rate": int(raw_info["sample_rate"]),
        "raw_output_channels": int(raw_info["channels"]),
        "raw_output_duration_sec": float(raw_info["duration_sec"]),
        "normalized_output_sample_rate": int(normalized_info["sample_rate"]),
        "normalized_output_channels": int(normalized_info["channels"]),
        "normalized_output_duration_sec": float(normalized_info["duration_sec"]),
        "shape_normalization_applied": True,
        "shape_normalization_reason": ",".join(reasons),
        "original_files_overwritten": False,
        "release_ready": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize a no-reference candidate audio file to the input audio shape.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--normalized-output", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    args = parser.parse_args(argv)
    report = normalize_candidate_audio_to_input_shape(
        input_audio_path=args.input_audio,
        raw_output_path=args.raw_output,
        normalized_output_path=args.normalized_output,
    )
    if args.manifest_json:
        write_json(args.manifest_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
