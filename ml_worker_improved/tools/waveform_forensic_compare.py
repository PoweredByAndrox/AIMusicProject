from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402

try:
    from scipy import signal as scipy_signal
except Exception:  # pragma: no cover - exercised only in minimal environments without scipy.
    scipy_signal = None


EPSILON = 1.0e-12
DB_FLOOR = -180.0
DEFAULT_DEFECT_THRESHOLD_DBFS = -60.0
DEFAULT_MIN_DEFECT_DURATION_SEC = 0.025
DEFAULT_ENVELOPE_WINDOW_SEC = 0.005
DEFAULT_MAX_DEFECT_RANGES = 12
DEFAULT_MAX_SVG_POINTS = 1600
DEFAULT_OFFSET_ANALYSIS_FRAMES = 262_144
MAX_SAFE_OFFSET_SAMPLES = 262_144
SAFE_STEM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _to_mono_float64(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float64, copy=False)
    raise ValueError(f"expected mono or stereo audio, got shape={audio.shape}")


def _align_to_shortest(reference_audio: np.ndarray, candidate_audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = _to_mono_float64(reference_audio)
    candidate = _to_mono_float64(candidate_audio)
    frame_count = min(reference.shape[0], candidate.shape[0])
    if frame_count <= 0:
        raise ValueError("reference and candidate must contain at least one frame")
    return reference[:frame_count], candidate[:frame_count]


def _slice_for_candidate_offset(
    reference: np.ndarray,
    candidate: np.ndarray,
    candidate_offset_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_offset_samples >= 0:
        frame_count = min(reference.shape[0], candidate.shape[0] - candidate_offset_samples)
        if frame_count <= 0:
            return reference[:0], candidate[:0]
        return reference[:frame_count], candidate[candidate_offset_samples : candidate_offset_samples + frame_count]

    reference_start = -candidate_offset_samples
    frame_count = min(reference.shape[0] - reference_start, candidate.shape[0])
    if frame_count <= 0:
        return reference[:0], candidate[:0]
    return reference[reference_start : reference_start + frame_count], candidate[:frame_count]


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return 1.0 if np.max(np.abs(left - right)) <= EPSILON else 0.0
    return float(np.dot(left, right) / denominator)


def _find_best_candidate_offset(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_offset_samples: int,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    if max_offset_samples <= 0:
        aligned_reference, aligned_candidate = _align_to_shortest(reference, candidate)
        return 0, aligned_reference, aligned_candidate, _cosine_similarity(aligned_reference, aligned_candidate)

    best_offset = _estimate_candidate_offset(reference, candidate, max_offset_samples)
    best_reference, best_candidate = _slice_for_candidate_offset(reference, candidate, best_offset)
    best_cosine = _cosine_similarity(best_reference, best_candidate)
    return best_offset, best_reference, best_candidate, best_cosine


def _estimate_candidate_offset(reference: np.ndarray, candidate: np.ndarray, max_offset_samples: int) -> int:
    if scipy_signal is None:
        return _estimate_candidate_offset_by_bounded_scan(reference, candidate, max_offset_samples)

    reference_frames = min(reference.shape[0], DEFAULT_OFFSET_ANALYSIS_FRAMES)
    candidate_frames = min(candidate.shape[0], DEFAULT_OFFSET_ANALYSIS_FRAMES + max_offset_samples)
    if reference_frames <= 0 or candidate_frames <= 0:
        return 0

    reference_window = reference[:reference_frames] - float(np.mean(reference[:reference_frames]))
    candidate_window = candidate[:candidate_frames] - float(np.mean(candidate[:candidate_frames]))
    if np.max(np.abs(reference_window)) <= EPSILON or np.max(np.abs(candidate_window)) <= EPSILON:
        return 0

    correlation = scipy_signal.correlate(candidate_window, reference_window, mode="full", method="fft")
    lags = scipy_signal.correlation_lags(candidate_window.size, reference_window.size, mode="full")
    allowed = np.abs(lags) <= max_offset_samples
    if not np.any(allowed):
        return 0
    allowed_lags = lags[allowed]
    allowed_correlation = correlation[allowed]
    best_index = int(np.argmax(np.abs(allowed_correlation)))
    return int(allowed_lags[best_index])


def _estimate_candidate_offset_by_bounded_scan(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_offset_samples: int,
) -> int:
    analysis_frames = min(reference.shape[0], candidate.shape[0], 16_384)
    if analysis_frames <= 0:
        return 0
    reference_window = reference[:analysis_frames]
    candidate_window = candidate[: analysis_frames + max_offset_samples]
    best_offset = 0
    best_score = -1.0
    for candidate_offset in range(-max_offset_samples, max_offset_samples + 1):
        aligned_reference, aligned_candidate = _slice_for_candidate_offset(
            reference_window,
            candidate_window,
            candidate_offset,
        )
        if aligned_reference.shape[0] <= 0:
            continue
        score = abs(_cosine_similarity(aligned_reference, aligned_candidate))
        if score > best_score + EPSILON or (
            abs(score - best_score) <= EPSILON and abs(candidate_offset) < abs(best_offset)
        ):
            best_offset = candidate_offset
            best_score = score
    return best_offset


def _dbfs_from_amplitude(amplitude: float) -> float:
    if amplitude <= EPSILON:
        return DB_FLOOR
    return float(max(DB_FLOOR, 20.0 * np.log10(amplitude)))


def _rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return DB_FLOOR
    return _dbfs_from_amplitude(float(np.sqrt(np.mean(np.square(audio)))))


def _moving_rms(audio: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames <= 1:
        return np.abs(audio)
    squared = np.square(audio)
    window = np.ones(window_frames, dtype=np.float64) / float(window_frames)
    return np.sqrt(np.convolve(squared, window, mode="same"))


def _histogram_summary(abs_error: np.ndarray) -> dict[str, float]:
    if abs_error.size == 0:
        return {
            "median_abs_error": 0.0,
            "p95_abs_error": 0.0,
            "p99_abs_error": 0.0,
            "max_abs_error": 0.0,
        }
    return {
        "median_abs_error": float(np.percentile(abs_error, 50.0)),
        "p95_abs_error": float(np.percentile(abs_error, 95.0)),
        "p99_abs_error": float(np.percentile(abs_error, 99.0)),
        "max_abs_error": float(np.max(abs_error)),
    }


def _histogram_bins(abs_error: np.ndarray, bin_count: int = 64) -> dict[str, list[float]]:
    if abs_error.size == 0:
        return {"bin_edges": [0.0], "counts": [0.0]}
    counts, bin_edges = np.histogram(abs_error, bins=bin_count)
    return {
        "bin_edges": [float(value) for value in bin_edges.tolist()],
        "counts": [float(value) for value in counts.tolist()],
    }


def _build_defect_ranges(
    *,
    residual_envelope: np.ndarray,
    residual: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    min_defect_duration_sec: float,
    max_ranges: int,
    include_named_reference_leakage: bool,
) -> list[dict[str, Any]]:
    threshold_amplitude = 10.0 ** (threshold_dbfs / 20.0)
    active = residual_envelope > threshold_amplitude
    min_frames = max(1, int(round(min_defect_duration_sec * sample_rate)))
    ranges: list[dict[str, Any]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        if start is not None and (not is_active or index == len(active) - 1):
            end = index + 1 if is_active and index == len(active) - 1 else index
            if end - start >= min_frames:
                segment = residual[start:end]
                labels = ["sample_residual_above_threshold"]
                if include_named_reference_leakage:
                    labels.append("named_reference_leakage")
                ranges.append(
                    {
                        "start_sample": int(start),
                        "end_sample": int(end),
                        "start_seconds": float(start / sample_rate),
                        "end_seconds": float(end / sample_rate),
                        "duration_seconds": float((end - start) / sample_rate),
                        "residual_peak_dbfs": _dbfs_from_amplitude(float(np.max(np.abs(segment)))),
                        "residual_rms_dbfs": _rms_dbfs(segment),
                        "labels": labels,
                    }
                )
            start = None

    ranges.sort(key=lambda item: item["residual_peak_dbfs"], reverse=True)
    return ranges[:max_ranges]


def _gain_to_match_reference(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.dot(candidate, candidate))
    if denominator <= EPSILON:
        return 1.0
    return float(np.dot(candidate, reference) / denominator)


def _align_stem_to_analysis(
    stem_audio: np.ndarray,
    *,
    candidate_offset_samples: int,
    aligned_frame_count: int,
) -> np.ndarray:
    stem = _to_mono_float64(stem_audio)
    if candidate_offset_samples >= 0:
        end = min(stem.shape[0], aligned_frame_count)
        aligned = stem[:end]
    else:
        start = -candidate_offset_samples
        end = min(stem.shape[0], start + aligned_frame_count)
        aligned = stem[start:end]
    if aligned.shape[0] < aligned_frame_count:
        padded = np.zeros(aligned_frame_count, dtype=np.float64)
        padded[: aligned.shape[0]] = aligned
        return padded
    return aligned[:aligned_frame_count]


def _build_leakage_attribution(
    *,
    residual: np.ndarray,
    reference_stems: Mapping[str, np.ndarray] | None,
    candidate_offset_samples: int,
) -> dict[str, Any]:
    if not reference_stems:
        return {"status": "not_available", "sources": {}, "max_source": None}

    residual_rms = float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0
    sources: dict[str, Any] = {}
    max_source: str | None = None
    max_ratio = 0.0

    for source_name, source_audio in reference_stems.items():
        aligned_source = _align_stem_to_analysis(
            source_audio,
            candidate_offset_samples=candidate_offset_samples,
            aligned_frame_count=residual.shape[0],
        )
        denominator = float(np.dot(aligned_source, aligned_source))
        projection_gain = 0.0 if denominator <= EPSILON else float(np.dot(residual, aligned_source) / denominator)
        projected = projection_gain * aligned_source
        projected_rms = float(np.sqrt(np.mean(np.square(projected)))) if projected.size else 0.0
        ratio = float(projected_rms / max(residual_rms, EPSILON))
        cosine = _cosine_similarity(residual, aligned_source)
        sources[str(source_name)] = {
            "projection_gain": projection_gain,
            "residual_projection_ratio": ratio,
            "residual_source_cosine": cosine,
            "projected_rms_dbfs": _rms_dbfs(projected),
        }
        if ratio > max_ratio:
            max_ratio = ratio
            max_source = str(source_name)

    return {
        "status": "available",
        "max_source": max_source,
        "max_residual_projection_ratio": max_ratio,
        "sources": sources,
    }


def _build_analysis_arrays(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    max_offset_samples: int,
    apply_gain_correction: bool,
) -> dict[str, Any]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    safe_max_offset_samples = _validate_max_offset_samples(max_offset_samples)

    raw_reference, raw_candidate = _align_to_shortest(reference_audio, candidate_audio)
    raw_cosine = _cosine_similarity(raw_reference, raw_candidate)

    reference_full = _to_mono_float64(reference_audio)
    candidate_full = _to_mono_float64(candidate_audio)
    candidate_offset, reference, offset_candidate, offset_cosine = _find_best_candidate_offset(
        reference_full,
        candidate_full,
        max_offset_samples=safe_max_offset_samples,
    )

    inverted_cosine = _cosine_similarity(reference, -offset_candidate)
    polarity_was_corrected = inverted_cosine > offset_cosine
    polarity_candidate = -offset_candidate if polarity_was_corrected else offset_candidate
    polarity_cosine = inverted_cosine if polarity_was_corrected else offset_cosine

    gain_to_reference = _gain_to_match_reference(reference, polarity_candidate)
    candidate = gain_to_reference * polarity_candidate if apply_gain_correction else polarity_candidate
    aligned_cosine = _cosine_similarity(reference, candidate)
    residual = candidate - reference
    abs_error = np.abs(residual)

    return {
        "reference": reference,
        "raw_candidate": raw_candidate,
        "candidate": candidate,
        "residual": residual,
        "abs_error": abs_error,
        "raw_cosine": raw_cosine,
        "offset_cosine": offset_cosine,
        "polarity_cosine": polarity_cosine,
        "aligned_cosine": aligned_cosine,
        "candidate_offset_samples": int(candidate_offset),
        "offset_was_corrected": bool(candidate_offset != 0),
        "polarity": "inverted" if polarity_was_corrected else "normal",
        "polarity_was_corrected": bool(polarity_was_corrected),
        "gain_to_reference": float(gain_to_reference),
        "gain_was_corrected": bool(apply_gain_correction and abs(gain_to_reference - 1.0) > 1.0e-6),
    }


def compare_waveforms_forensically(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    defect_threshold_dbfs: float = DEFAULT_DEFECT_THRESHOLD_DBFS,
    min_defect_duration_sec: float = DEFAULT_MIN_DEFECT_DURATION_SEC,
    envelope_window_sec: float = DEFAULT_ENVELOPE_WINDOW_SEC,
    max_defect_ranges: int = DEFAULT_MAX_DEFECT_RANGES,
    max_offset_samples: int = 0,
    apply_gain_correction: bool = False,
    reference_stems: Mapping[str, np.ndarray] | None = None,
    analysis_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    arrays = (
        dict(analysis_arrays)
        if analysis_arrays is not None
        else _build_analysis_arrays(
            reference_audio=reference_audio,
            candidate_audio=candidate_audio,
            sample_rate=sample_rate,
            max_offset_samples=max_offset_samples,
            apply_gain_correction=apply_gain_correction,
        )
    )
    residual = arrays["residual"]
    abs_error = arrays["abs_error"]
    envelope_frames = max(1, int(round(envelope_window_sec * sample_rate)))
    residual_envelope = _moving_rms(residual, envelope_frames)
    leakage_attribution = _build_leakage_attribution(
        residual=residual,
        reference_stems=reference_stems,
        candidate_offset_samples=arrays["candidate_offset_samples"],
    )
    named_leakage_available = leakage_attribution.get("status") == "available" and bool(
        leakage_attribution.get("max_source")
    )
    defect_ranges = _build_defect_ranges(
        residual_envelope=residual_envelope,
        residual=residual,
        sample_rate=sample_rate,
        threshold_dbfs=defect_threshold_dbfs,
        min_defect_duration_sec=min_defect_duration_sec,
        max_ranges=max_defect_ranges,
        include_named_reference_leakage=named_leakage_available,
    )

    return {
        "schema_version": 1,
        "status": "failed" if defect_ranges else "passed",
        "config": {
            "defect_threshold_dbfs": float(defect_threshold_dbfs),
            "min_defect_duration_sec": float(min_defect_duration_sec),
            "envelope_window_sec": float(envelope_window_sec),
            "max_defect_ranges": int(max_defect_ranges),
            "max_offset_samples": int(max_offset_samples),
            "apply_gain_correction": bool(apply_gain_correction),
        },
        "alignment": {
            "polarity": arrays["polarity"],
            "polarity_was_corrected": arrays["polarity_was_corrected"],
            "candidate_offset_samples": arrays["candidate_offset_samples"],
            "offset_was_corrected": arrays["offset_was_corrected"],
            "gain_to_reference": arrays["gain_to_reference"],
            "gain_was_corrected": arrays["gain_was_corrected"],
            "raw_cosine_similarity": arrays["raw_cosine"],
            "offset_cosine_similarity": arrays["offset_cosine"],
            "polarity_cosine_similarity": arrays["polarity_cosine"],
            "aligned_cosine_similarity": arrays["aligned_cosine"],
            "aligned_frame_count": int(arrays["reference"].shape[0]),
        },
        "residual": {
            "peak_dbfs": _dbfs_from_amplitude(float(np.max(abs_error))) if abs_error.size else DB_FLOOR,
            "rms_dbfs": _rms_dbfs(residual),
            "envelope_peak_dbfs": _dbfs_from_amplitude(float(np.max(residual_envelope)))
            if residual_envelope.size
            else DB_FLOOR,
        },
        "histogram": _histogram_summary(abs_error),
        "histogram_bins": _histogram_bins(abs_error),
        "leakage_attribution": leakage_attribution,
        "defect_ranges": defect_ranges,
    }


def _downsample_for_svg(audio: np.ndarray, max_points: int = DEFAULT_MAX_SVG_POINTS) -> list[float]:
    if audio.size == 0:
        return []
    if audio.size <= max_points:
        return [float(value) for value in audio.tolist()]
    frame_edges = np.linspace(0, audio.size, max_points + 1, dtype=np.int64)
    points: list[float] = []
    for start, end in zip(frame_edges[:-1], frame_edges[1:]):
        segment = audio[start:end]
        points.append(float(segment[np.argmax(np.abs(segment))]) if segment.size else 0.0)
    return points


def _polyline(points: list[float], *, width: int, height: int, color: str, y_center: float, gain: float) -> str:
    if not points:
        return ""
    if len(points) == 1:
        coordinates = f"0,{y_center:.2f}"
    else:
        coordinates = " ".join(
            f"{(index / (len(points) - 1)) * width:.2f},{(y_center - value * gain):.2f}"
            for index, value in enumerate(points)
        )
    return f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="1.2" />'


def _write_waveform_overlay_svg(
    *,
    path: Path,
    reference: np.ndarray,
    candidate: np.ndarray,
    residual: np.ndarray,
) -> None:
    width = 1200
    height = 420
    max_amplitude = max(
        float(np.max(np.abs(reference))) if reference.size else 0.0,
        float(np.max(np.abs(candidate))) if candidate.size else 0.0,
        float(np.max(np.abs(residual))) if residual.size else 0.0,
        EPSILON,
    )
    gain = 150.0 / max_amplitude
    reference_points = _downsample_for_svg(reference)
    candidate_points = _downsample_for_svg(candidate)
    residual_points = _downsample_for_svg(residual)
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            '<text x="20" y="28" font-family="Arial" font-size="16" fill="#111111">Waveform overlay: reference, candidate, residual</text>',
            '<line x1="0" x2="1200" y1="120" y2="120" stroke="#dddddd" stroke-width="1" />',
            '<line x1="0" x2="1200" y1="260" y2="260" stroke="#dddddd" stroke-width="1" />',
            _polyline(reference_points, width=width, height=height, color="#0f9d58", y_center=120.0, gain=gain),
            _polyline(candidate_points, width=width, height=height, color="#1a73e8", y_center=120.0, gain=gain),
            _polyline(residual_points, width=width, height=height, color="#d93025", y_center=300.0, gain=gain),
            '<text x="20" y="395" font-family="Arial" font-size="12" fill="#0f9d58">green=reference</text>',
            '<text x="160" y="395" font-family="Arial" font-size="12" fill="#1a73e8">blue=candidate</text>',
            '<text x="310" y="395" font-family="Arial" font-size="12" fill="#d93025">red=residual</text>',
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def _write_residual_histogram_svg(*, path: Path, abs_error: np.ndarray) -> None:
    histogram = _histogram_bins(abs_error, bin_count=48)
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    width = 900
    height = 360
    max_count = float(np.max(counts)) if counts.size else 1.0
    bars: list[str] = []
    for index, count in enumerate(counts):
        bar_width = width / max(len(counts), 1)
        bar_height = 0.0 if max_count <= EPSILON else (count / max_count) * 260.0
        x = index * bar_width
        y = 320.0 - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1.0, bar_width - 1.0):.2f}" '
            f'height="{bar_height:.2f}" fill="#d93025" />'
        )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            '<text x="20" y="28" font-family="Arial" font-size="16" fill="#111111">Residual absolute-error histogram</text>',
            *bars,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def _relative_artifact_paths(output_dir: Path, paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: path.relative_to(output_dir).as_posix() for name, path in paths.items()}


def _validate_stem_name(stem_name: str) -> str:
    if not stem_name:
        raise ValueError("stem_name must not be empty")
    if Path(stem_name).name != stem_name or "/" in stem_name or "\\" in stem_name:
        raise ValueError(f"stem_name must be a file-name token, got {stem_name!r}")
    if not SAFE_STEM_NAME_PATTERN.match(stem_name):
        raise ValueError(
            f"stem_name must contain only ASCII letters, digits, dot, underscore, or hyphen; got {stem_name!r}"
        )
    return stem_name


def _validate_max_offset_samples(max_offset_samples: int) -> int:
    offset = int(max_offset_samples)
    if offset < 0:
        raise ValueError(f"max_offset_samples must be non-negative, got {max_offset_samples}")
    if offset > MAX_SAFE_OFFSET_SAMPLES:
        raise ValueError(
            f"max_offset_samples={offset} exceeds safe limit {MAX_SAFE_OFFSET_SAMPLES}; "
            "use a pre-aligned candidate or add a bounded offset-estimation stage"
        )
    return offset


def write_waveform_forensic_evidence_pack(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    output_dir: str | Path,
    stem_name: str,
    defect_threshold_dbfs: float = DEFAULT_DEFECT_THRESHOLD_DBFS,
    min_defect_duration_sec: float = DEFAULT_MIN_DEFECT_DURATION_SEC,
    envelope_window_sec: float = DEFAULT_ENVELOPE_WINDOW_SEC,
    max_defect_ranges: int = DEFAULT_MAX_DEFECT_RANGES,
    max_offset_samples: int = 0,
    apply_gain_correction: bool = False,
    reference_stems: Mapping[str, np.ndarray] | None = None,
    max_evidence_frames: int | None = None,
    analysis_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safe_stem_name = _validate_stem_name(stem_name)
    safe_max_offset_samples = _validate_max_offset_samples(max_offset_samples)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    arrays = (
        dict(analysis_arrays)
        if analysis_arrays is not None
        else _build_analysis_arrays(
            reference_audio=reference_audio,
            candidate_audio=candidate_audio,
            sample_rate=sample_rate,
            max_offset_samples=safe_max_offset_samples,
            apply_gain_correction=apply_gain_correction,
        )
    )
    report = compare_waveforms_forensically(
        reference_audio=reference_audio,
        candidate_audio=candidate_audio,
        sample_rate=sample_rate,
        defect_threshold_dbfs=defect_threshold_dbfs,
        min_defect_duration_sec=min_defect_duration_sec,
        envelope_window_sec=envelope_window_sec,
        max_defect_ranges=max_defect_ranges,
        max_offset_samples=safe_max_offset_samples,
        apply_gain_correction=apply_gain_correction,
        reference_stems=reference_stems,
        analysis_arrays=arrays,
    )
    aligned_frame_count = int(arrays["reference"].shape[0])
    if max_evidence_frames is not None and int(max_evidence_frames) <= 0:
        raise ValueError(f"max_evidence_frames must be positive when provided, got {max_evidence_frames}")
    artifact_frame_count = (
        min(aligned_frame_count, int(max_evidence_frames))
        if max_evidence_frames is not None
        else aligned_frame_count
    )
    artifact_reference = arrays["reference"][:artifact_frame_count]
    artifact_candidate = arrays["candidate"][:artifact_frame_count]
    artifact_residual = arrays["residual"][:artifact_frame_count]
    artifact_abs_error = arrays["abs_error"][:artifact_frame_count]
    artifact_scope = {
        "mode": "head_truncated" if artifact_frame_count < aligned_frame_count else "full_aligned",
        "aligned_frame_count": aligned_frame_count,
        "artifact_frame_count": int(artifact_frame_count),
        "max_evidence_frames": int(max_evidence_frames) if max_evidence_frames is not None else None,
    }

    trace_path = target_dir / f"{safe_stem_name}_waveform_trace.npz"
    residual_audio_path = target_dir / f"{safe_stem_name}_residual_aligned.wav"
    overlay_svg_path = target_dir / f"{safe_stem_name}_waveform_overlay.svg"
    histogram_svg_path = target_dir / f"{safe_stem_name}_residual_histogram.svg"
    report_json_path = target_dir / f"{safe_stem_name}_waveform_forensic_report.json"

    np.savez_compressed(
        trace_path,
        reference_aligned=artifact_reference.astype(np.float32),
        candidate_aligned=artifact_candidate.astype(np.float32),
        residual=artifact_residual.astype(np.float32),
        abs_error=artifact_abs_error.astype(np.float32),
    )
    sf.write(residual_audio_path, artifact_residual.astype(np.float32), sample_rate, subtype="FLOAT")
    _write_waveform_overlay_svg(
        path=overlay_svg_path,
        reference=artifact_reference,
        candidate=artifact_candidate,
        residual=artifact_residual,
    )
    _write_residual_histogram_svg(path=histogram_svg_path, abs_error=arrays["abs_error"])

    artifact_paths = {
        "report_json": report_json_path,
        "trace_npz": trace_path,
        "residual_audio_wav": residual_audio_path,
        "waveform_overlay_svg": overlay_svg_path,
        "residual_histogram_svg": histogram_svg_path,
    }
    report_with_artifacts = {
        "schema_version": 1,
        "stem_name": safe_stem_name,
        "sample_rate": int(sample_rate),
        "artifact_scope": artifact_scope,
        "waveform_forensics": report,
        "evidence_artifacts": _relative_artifact_paths(target_dir, artifact_paths),
    }
    report_json_path.write_text(json.dumps(report_with_artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return report_with_artifacts


def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    return np.asarray(audio), int(sample_rate)


def _load_named_reference_stems(stem_args: list[str] | None, *, expected_sample_rate: int) -> dict[str, np.ndarray]:
    stems: dict[str, np.ndarray] = {}
    for stem_arg in stem_args or []:
        if "=" not in stem_arg:
            raise ValueError(f"expected NAME=PATH for --reference-stem, got {stem_arg!r}")
        name, raw_path = stem_arg.split("=", 1)
        _validate_stem_name(name)
        audio, sample_rate = _load_audio(raw_path)
        if sample_rate != expected_sample_rate:
            raise ValueError(
                f"sample-rate mismatch for reference stem {name!r}: "
                f"expected={expected_sample_rate}, actual={sample_rate}"
            )
        stems[name] = audio
    return stems


def main() -> int:
    require_docker_runtime(
        tool_name="waveform_forensic_compare.py",
        entrypoint_path="/app/tools/waveform_forensic_compare.py",
    )
    parser = argparse.ArgumentParser(description="Write sample-accurate waveform forensic evidence for a stem candidate.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem-name", default="stem")
    parser.add_argument("--defect-threshold-dbfs", type=float, default=DEFAULT_DEFECT_THRESHOLD_DBFS)
    parser.add_argument("--min-defect-duration-sec", type=float, default=DEFAULT_MIN_DEFECT_DURATION_SEC)
    parser.add_argument("--envelope-window-sec", type=float, default=DEFAULT_ENVELOPE_WINDOW_SEC)
    parser.add_argument("--max-defect-ranges", type=int, default=DEFAULT_MAX_DEFECT_RANGES)
    parser.add_argument("--max-offset-samples", type=int, default=0)
    parser.add_argument("--apply-gain-correction", action="store_true")
    parser.add_argument("--reference-stem", action="append", default=[], help="Named source stem in NAME=PATH form.")
    args = parser.parse_args()

    reference, reference_sample_rate = _load_audio(args.reference)
    candidate, candidate_sample_rate = _load_audio(args.candidate)
    if reference_sample_rate != candidate_sample_rate:
        raise ValueError(
            f"sample-rate mismatch: reference={reference_sample_rate}, candidate={candidate_sample_rate}"
        )
    report = write_waveform_forensic_evidence_pack(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=reference_sample_rate,
        output_dir=args.output_dir,
        stem_name=args.stem_name,
        defect_threshold_dbfs=args.defect_threshold_dbfs,
        min_defect_duration_sec=args.min_defect_duration_sec,
        envelope_window_sec=args.envelope_window_sec,
        max_defect_ranges=args.max_defect_ranges,
        max_offset_samples=args.max_offset_samples,
        apply_gain_correction=args.apply_gain_correction,
        reference_stems=_load_named_reference_stems(
            args.reference_stem,
            expected_sample_rate=reference_sample_rate,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
