#!/usr/bin/env python3
"""Diagnostic oracle upper-bound harness for verified reference stems.

This tool is intentionally reference-assisted. It must not be used as a
production no-reference candidate route.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


SAFE_STEM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
EPSILON = 1.0e-12


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return np.asarray(audio, dtype=np.float64), int(sample_rate)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_24")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_stem_name(name: str) -> str:
    if not SAFE_STEM_NAME_PATTERN.match(name):
        raise ValueError(f"unsafe stem name: {name!r}")
    return name


def fit_length(audio: np.ndarray, frame_count: int) -> np.ndarray:
    if audio.shape[0] == frame_count:
        return audio
    if audio.shape[0] > frame_count:
        return audio[:frame_count]
    pad = np.zeros((frame_count - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=0)


def validate_audio_compatible(
    *,
    source_mix_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    reference_stems: dict[str, np.ndarray],
) -> None:
    if source_mix_audio.ndim != 2 or target_reference_audio.ndim != 2:
        raise ValueError("source mix and target reference must be two-dimensional audio arrays")
    if source_mix_audio.shape[1] != target_reference_audio.shape[1]:
        raise ValueError(
            "channel count mismatch: "
            f"source_mix={source_mix_audio.shape[1]}, target_reference={target_reference_audio.shape[1]}"
        )
    for stem_name, stem_audio in reference_stems.items():
        if stem_audio.ndim != 2:
            raise ValueError(f"reference stem {stem_name!r} must be a two-dimensional audio array")
        if stem_audio.shape[1] != source_mix_audio.shape[1]:
            raise ValueError(
                "channel count mismatch: "
                f"source_mix={source_mix_audio.shape[1]}, reference_stem[{stem_name}]={stem_audio.shape[1]}"
            )


def validate_non_target_reference_stems(stem_name: str, reference_stems: dict[str, np.ndarray]) -> None:
    if stem_name in reference_stems:
        raise ValueError("target stem must not be listed as a non-target reference")


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _dbfs_from_rms(rms: float) -> float:
    if rms <= EPSILON:
        return -240.0
    return float(20.0 * math.log10(rms))


def _cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.ravel(reference).astype(np.float64)
    cand = np.ravel(candidate).astype(np.float64)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    if denom <= EPSILON:
        return 1.0 if np.linalg.norm(ref - cand) <= EPSILON else 0.0
    return float(np.dot(ref, cand) / denom)


def signal_to_distortion_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_energy = float(np.sum(np.square(reference, dtype=np.float64)))
    error_energy = float(np.sum(np.square(reference - candidate, dtype=np.float64)))
    if error_energy <= EPSILON:
        return 120.0
    return float(10.0 * math.log10((reference_energy + EPSILON) / error_energy))


def calculate_reference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    frame_count = min(reference.shape[0], candidate.shape[0])
    reference = reference[:frame_count]
    candidate = candidate[:frame_count]
    cosine = _cosine_similarity(reference, candidate)
    sdr_db = signal_to_distortion_db(reference, candidate)
    residual = candidate - reference
    residual_rms_dbfs = _dbfs_from_rms(_rms(residual))
    likeness_from_cosine = max(0.0, min(100.0, 100.0 * cosine))
    likeness_from_sdr = max(0.0, min(100.0, 60.0 + 2.0 * sdr_db))
    return {
        "reference_likeness_percent": float((likeness_from_cosine + likeness_from_sdr) / 2.0),
        "cosine_similarity": float(cosine),
        "sdr_db": float(sdr_db),
        "residual_rms_dbfs": float(residual_rms_dbfs),
        "rms_drift_db": float(_dbfs_from_rms(_rms(candidate)) - _dbfs_from_rms(_rms(reference))),
    }


def _frame_starts(frame_count: int, n_fft: int, hop_size: int) -> list[int]:
    if n_fft <= 0 or hop_size <= 0:
        raise ValueError("n_fft and hop_size must be positive")
    if frame_count <= n_fft:
        return [0]
    starts = list(range(0, frame_count - n_fft + 1, hop_size))
    final_start = frame_count - n_fft
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _stft_channel(audio: np.ndarray, n_fft: int, hop_size: int) -> tuple[np.ndarray, list[int]]:
    starts = _frame_starts(audio.shape[0], n_fft, hop_size)
    window = np.hanning(n_fft).astype(np.float64)
    spectra = []
    for start in starts:
        frame = np.zeros(n_fft, dtype=np.float64)
        available = audio[start : min(start + n_fft, audio.shape[0])]
        frame[: available.shape[0]] = available
        spectra.append(np.fft.rfft(frame * window))
    return np.stack(spectra, axis=0), starts


def _istft_channel(spectra: np.ndarray, starts: list[int], frame_count: int, n_fft: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float64)
    output = np.zeros(frame_count, dtype=np.float64)
    weight = np.zeros(frame_count, dtype=np.float64)
    for spectrum, start in zip(spectra, starts):
        frame = np.fft.irfft(spectrum, n=n_fft).real
        end = min(start + n_fft, frame_count)
        valid = end - start
        output[start:end] += frame[:valid] * window[:valid]
        weight[start:end] += np.square(window[:valid])
    active = weight > EPSILON
    output[active] /= weight[active]
    return output


def build_ideal_ratio_mask_oracle(
    *,
    source_mix_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    reference_stems: dict[str, np.ndarray],
    n_fft: int,
    hop_size: int,
    mask_power: float = 2.0,
) -> np.ndarray:
    validate_audio_compatible(
        source_mix_audio=source_mix_audio,
        target_reference_audio=target_reference_audio,
        reference_stems=reference_stems,
    )
    frame_count = min(source_mix_audio.shape[0], target_reference_audio.shape[0])
    source_mix_audio = source_mix_audio[:frame_count]
    target_reference_audio = target_reference_audio[:frame_count]
    fitted_stems = {
        name: fit_length(stem, frame_count)
        for name, stem in reference_stems.items()
    }
    output_channels: list[np.ndarray] = []
    for channel_index in range(source_mix_audio.shape[1]):
        source_spectra, starts = _stft_channel(source_mix_audio[:, channel_index], n_fft, hop_size)
        target_spectra, _ = _stft_channel(target_reference_audio[:, channel_index], n_fft, hop_size)
        target_magnitude = np.abs(target_spectra) ** mask_power
        non_target_magnitude = np.zeros_like(target_magnitude)
        for stem_audio in fitted_stems.values():
            stem_spectra, _ = _stft_channel(stem_audio[:, channel_index], n_fft, hop_size)
            non_target_magnitude += np.abs(stem_spectra) ** mask_power
        if not fitted_stems:
            residual = source_mix_audio[:, channel_index] - target_reference_audio[:, channel_index]
            residual_spectra, _ = _stft_channel(residual, n_fft, hop_size)
            non_target_magnitude = np.abs(residual_spectra) ** mask_power
        mask = target_magnitude / (target_magnitude + non_target_magnitude + EPSILON)
        oracle_spectra = source_spectra * mask
        output_channels.append(_istft_channel(oracle_spectra, starts, frame_count, n_fft))
    return np.stack(output_channels, axis=1)


def _window_projection(residual_window: np.ndarray, source_window: np.ndarray) -> tuple[float, float]:
    residual = np.ravel(residual_window).astype(np.float64)
    source = np.ravel(source_window).astype(np.float64)
    residual_norm = float(np.linalg.norm(residual))
    source_norm = float(np.linalg.norm(source))
    if residual_norm <= EPSILON or source_norm <= EPSILON:
        return 0.0, 0.0
    correlation = float(np.dot(residual, source) / (residual_norm * source_norm))
    projection_gain = float(np.dot(residual, source) / (source_norm * source_norm + EPSILON))
    return correlation, projection_gain


def _harmonic_residual_profile(audio_window: np.ndarray) -> tuple[float, float]:
    mono = np.mean(audio_window, axis=1).astype(np.float64)
    if mono.size < 8 or _rms(mono) <= EPSILON:
        return 1.0, 0.0
    window = np.hanning(mono.size)
    magnitude = np.abs(np.fft.rfft(mono * window))[1:] + EPSILON
    if magnitude.size == 0:
        return 1.0, 0.0
    flatness = float(np.exp(np.mean(np.log(magnitude))) / (np.mean(magnitude) + EPSILON))
    peak_to_mean_db = float(20.0 * math.log10(float(np.max(magnitude)) / (float(np.mean(magnitude)) + EPSILON)))
    return flatness, peak_to_mean_db


def _merge_phrase_windows(windows: list[dict[str, Any]], min_window_seconds: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        grouped.setdefault(str(window["source"]), []).append(window)

    merged: list[dict[str, Any]] = []
    for source, source_windows in grouped.items():
        source_windows.sort(key=lambda item: float(item["start_seconds"]))
        current: dict[str, Any] | None = None
        for window in source_windows:
            if current is None:
                current = dict(window)
                continue
            if float(window["start_seconds"]) <= float(current["end_seconds"]):
                current["end_seconds"] = max(float(current["end_seconds"]), float(window["end_seconds"]))
                current["duration_seconds"] = float(current["end_seconds"]) - float(current["start_seconds"])
                current["max_abs_correlation"] = max(
                    float(current["max_abs_correlation"]),
                    float(window["max_abs_correlation"]),
                )
                current["max_projection_gain"] = max(
                    float(current["max_projection_gain"]),
                    float(window["max_projection_gain"]),
                )
                current["peak_residual_rms_dbfs"] = max(
                    float(current["peak_residual_rms_dbfs"]),
                    float(window["peak_residual_rms_dbfs"]),
                )
                current["window_count"] = int(current["window_count"]) + int(window["window_count"])
            else:
                if float(current["duration_seconds"]) >= min_window_seconds:
                    merged.append(current)
                current = dict(window)
        if current is not None and float(current["duration_seconds"]) >= min_window_seconds:
            merged.append(current)
    merged.sort(
        key=lambda item: (
            float(item["duration_seconds"]),
            float(item["max_abs_correlation"]),
            float(item["peak_residual_rms_dbfs"]),
        ),
        reverse=True,
    )
    return merged


def detect_phrase_level_leakage(
    *,
    candidate_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    reference_stems: dict[str, np.ndarray],
    sample_rate: int,
    window_seconds: float = 2.0,
    hop_seconds: float = 0.5,
    min_window_seconds: float = 1.5,
    min_abs_correlation: float = 0.45,
    min_residual_rms_dbfs: float = -55.0,
    max_windows: int = 20,
) -> list[dict[str, Any]]:
    validate_audio_compatible(
        source_mix_audio=candidate_audio,
        target_reference_audio=target_reference_audio,
        reference_stems=reference_stems,
    )
    frame_count = min(candidate_audio.shape[0], target_reference_audio.shape[0])
    candidate_audio = candidate_audio[:frame_count]
    target_reference_audio = target_reference_audio[:frame_count]
    residual = candidate_audio - target_reference_audio
    window_frames = max(1, int(round(window_seconds * sample_rate)))
    hop_frames = max(1, int(round(hop_seconds * sample_rate)))
    starts = _frame_starts(frame_count, min(window_frames, frame_count), hop_frames)
    raw_windows: list[dict[str, Any]] = []
    for start in starts:
        end = min(start + window_frames, frame_count)
        residual_window = residual[start:end]
        residual_rms_dbfs = _dbfs_from_rms(_rms(residual_window))
        if residual_rms_dbfs < min_residual_rms_dbfs:
            continue
        for source_name, source_audio in reference_stems.items():
            source_window = fit_length(source_audio, frame_count)[start:end]
            correlation, projection_gain = _window_projection(residual_window, source_window)
            if abs(correlation) < min_abs_correlation:
                continue
            raw_windows.append(
                {
                    "label": "bar_spanning_drop_melody_leakage",
                    "source": source_name,
                    "start_seconds": float(start / sample_rate),
                    "end_seconds": float(end / sample_rate),
                    "duration_seconds": float((end - start) / sample_rate),
                    "max_abs_correlation": float(abs(correlation)),
                    "max_projection_gain": float(abs(projection_gain)),
                    "peak_residual_rms_dbfs": float(residual_rms_dbfs),
                    "window_count": 1,
                }
            )
        flatness, peak_to_mean_db = _harmonic_residual_profile(residual_window)
        if flatness <= 0.22 and peak_to_mean_db >= 16.0:
            raw_windows.append(
                {
                    "label": "bar_spanning_drop_melody_leakage",
                    "source": "unattributed_harmonic_residual",
                    "start_seconds": float(start / sample_rate),
                    "end_seconds": float(end / sample_rate),
                    "duration_seconds": float((end - start) / sample_rate),
                    "max_abs_correlation": float(1.0 - flatness),
                    "max_projection_gain": float(peak_to_mean_db),
                    "peak_residual_rms_dbfs": float(residual_rms_dbfs),
                    "window_count": 1,
                }
            )
    return _merge_phrase_windows(raw_windows, min_window_seconds)[:max_windows]


def build_reference_assisted_contract() -> dict[str, Any]:
    return {
        "purpose": "diagnostic_oracle_upper_bound_only",
        "target_reference_assisted": True,
        "target_reference_used_at_application": True,
        "application_reference_free": False,
        "delivery_allowed_without_reference": False,
        "no_reference_production_claim": False,
        "fallback_used": False,
        "hidden_inference": False,
        "per_track_hardcode_allowed": False,
        "not_timestamp_hardcoded": True,
    }


def evaluate_oracle_upper_bound(
    *,
    source_mix_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    reference_stems: dict[str, np.ndarray],
    sample_rate: int,
    stem_name: str,
    n_fft: int = 2048,
    hop_size: int = 512,
    reference_likeness_threshold: float = 95.0,
    candidate_audio: np.ndarray | None = None,
) -> dict[str, Any]:
    validate_stem_name(stem_name)
    validate_non_target_reference_stems(stem_name, reference_stems)
    validate_audio_compatible(
        source_mix_audio=source_mix_audio,
        target_reference_audio=target_reference_audio,
        reference_stems=reference_stems,
    )
    frame_count = min(source_mix_audio.shape[0], target_reference_audio.shape[0])
    source_mix_audio = source_mix_audio[:frame_count]
    target_reference_audio = target_reference_audio[:frame_count]
    fitted_stems = {
        validate_stem_name(name): fit_length(audio, frame_count)
        for name, audio in reference_stems.items()
    }
    oracle_audio = build_ideal_ratio_mask_oracle(
        source_mix_audio=source_mix_audio,
        target_reference_audio=target_reference_audio,
        reference_stems=fitted_stems,
        n_fft=n_fft,
        hop_size=hop_size,
    )
    oracle_metrics = calculate_reference_metrics(target_reference_audio, oracle_audio)
    source_mix_metrics = calculate_reference_metrics(target_reference_audio, source_mix_audio)
    status = (
        "oracle_passed"
        if oracle_metrics["reference_likeness_percent"] >= reference_likeness_threshold
        else "oracle_failed"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "oracle_stem_upper_bound",
        "stem_name": stem_name,
        "sample_rate": int(sample_rate),
        "frame_count": int(frame_count),
        "contract": build_reference_assisted_contract(),
        "oracle_candidates": {
            "ideal_ratio_mask": {
                "description": "Reference-assisted ideal ratio mask applied to source mix STFT.",
                "metrics": oracle_metrics,
            }
        },
        "source_mix_as_candidate": {
            "metrics": source_mix_metrics,
        },
        "upper_bound_decision": {
            "status": status,
            "reference_likeness_threshold": float(reference_likeness_threshold),
            "reference_likeness_percent": float(oracle_metrics["reference_likeness_percent"]),
        },
        "interpretation_limits": [
            "This is a reference-assisted diagnostic upper bound, not a production no-reference route.",
            "A passing oracle proves feasibility of a stronger mask/router; it does not produce a deliverable stem.",
            "A failing oracle indicates phase, mix, reference, or model-stack limits that must be resolved before routing.",
        ],
    }
    if candidate_audio is not None:
        fitted_candidate = fit_length(candidate_audio, frame_count)
        report["candidate_diagnostics"] = {
            "metrics": calculate_reference_metrics(target_reference_audio, fitted_candidate),
            "phrase_level_leakage_windows": detect_phrase_level_leakage(
                candidate_audio=fitted_candidate,
                target_reference_audio=target_reference_audio,
                reference_stems=fitted_stems,
                sample_rate=sample_rate,
            ),
        }
    return report


def parse_reference_stem_arg(raw_value: str) -> tuple[str, Path]:
    if "=" not in raw_value:
        raise ValueError("--reference-stem must use NAME=PATH")
    name, raw_path = raw_value.split("=", 1)
    return validate_stem_name(name), Path(raw_path)


def load_reference_stems(stem_args: list[str], expected_sample_rate: int, frame_count: int) -> dict[str, np.ndarray]:
    stems: dict[str, np.ndarray] = {}
    for raw in stem_args:
        name, stem_path = parse_reference_stem_arg(raw)
        audio, sample_rate = read_audio(stem_path)
        if sample_rate != expected_sample_rate:
            raise ValueError(
                f"reference stem {name!r} sample rate mismatch: {sample_rate} != {expected_sample_rate}"
            )
        stems[name] = fit_length(audio, frame_count)
    return stems


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reference-assisted oracle upper-bound diagnostics for one stem.")
    parser.add_argument("--source-mix", required=True, type=Path)
    parser.add_argument("--target-reference", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--reference-stem", action="append", default=[])
    parser.add_argument("--stem-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--reference-likeness-threshold", type=float, default=95.0)
    parser.add_argument("--phrase-window-seconds", type=float, default=2.0)
    parser.add_argument("--phrase-hop-seconds", type=float, default=0.5)
    parser.add_argument("--phrase-min-window-seconds", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="oracle_stem_upper_bound.py",
        entrypoint_path="/app/tools/oracle_stem_upper_bound.py",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    source_mix, source_sample_rate = read_audio(args.source_mix)
    target_reference, target_sample_rate = read_audio(args.target_reference)
    if source_sample_rate != target_sample_rate:
        raise ValueError(f"sample rate mismatch: source={source_sample_rate}, target={target_sample_rate}")
    frame_count = min(source_mix.shape[0], target_reference.shape[0])
    source_mix = source_mix[:frame_count]
    target_reference = target_reference[:frame_count]
    reference_stems = load_reference_stems(args.reference_stem, source_sample_rate, frame_count)
    candidate_audio = None
    if args.candidate is not None:
        candidate_audio, candidate_sample_rate = read_audio(args.candidate)
        if candidate_sample_rate != source_sample_rate:
            raise ValueError(
                f"candidate sample rate mismatch: {candidate_sample_rate} != {source_sample_rate}"
            )
        candidate_audio = fit_length(candidate_audio, frame_count)

    oracle_audio = build_ideal_ratio_mask_oracle(
        source_mix_audio=source_mix,
        target_reference_audio=target_reference,
        reference_stems=reference_stems,
        n_fft=args.n_fft,
        hop_size=args.hop_size,
    )
    report = evaluate_oracle_upper_bound(
        source_mix_audio=source_mix,
        target_reference_audio=target_reference,
        reference_stems=reference_stems,
        sample_rate=source_sample_rate,
        stem_name=args.stem_name,
        n_fft=args.n_fft,
        hop_size=args.hop_size,
        reference_likeness_threshold=args.reference_likeness_threshold,
        candidate_audio=candidate_audio,
    )
    if candidate_audio is not None:
        report["candidate_diagnostics"]["phrase_level_leakage_windows"] = detect_phrase_level_leakage(
            candidate_audio=candidate_audio,
            target_reference_audio=target_reference,
            reference_stems=reference_stems,
            sample_rate=source_sample_rate,
            window_seconds=args.phrase_window_seconds,
            hop_seconds=args.phrase_hop_seconds,
            min_window_seconds=args.phrase_min_window_seconds,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = args.output_dir / "01_ideal_ratio_mask_oracle.flac"
    report_path = args.output_dir / "oracle_upper_bound_report.json"
    write_audio(oracle_path, oracle_audio, source_sample_rate)
    report["oracle_candidates"]["ideal_ratio_mask"]["path"] = str(oracle_path)
    write_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
