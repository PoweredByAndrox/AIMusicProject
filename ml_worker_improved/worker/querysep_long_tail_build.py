from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

TASK_TYPE = "querysep_long_tail_build"
BACKEND_BANQUET_BYOQ = "banquet_byoq"
DEFAULT_PHASE_ID = "long_tail_querysep"
DEFAULT_TIMEOUT_SECONDS = 7200
DEFAULT_MIN_OUTPUT_BYTES = 1024
DEFAULT_MIN_QUALITY_DURATION_SECONDS = 1.0
DEFAULT_MAX_DURATION_DELTA_SECONDS = 2.0
DEFAULT_MIN_LONG_TAIL_RMS_DBFS = -50.0
DEFAULT_MIN_SOURCE_RMS_RATIO_DB = -42.0
DEFAULT_MIN_ACTIVE_PRESENCE_WINDOWS = 3
DEFAULT_MIN_ACTIVE_PRESENCE_RMS_DBFS = -50.0
DEFAULT_ACTIVE_PRESENCE_WINDOW_SECONDS = 5.0
DEFAULT_MAX_PROXY_CORRELATION = 0.85
DEFAULT_MAX_WINDOWED_PROXY_CORRELATION = 0.55
DEFAULT_WINDOWED_PROXY_CORRELATION_SECONDS = 5.0
DEFAULT_MIN_WINDOWED_PROXY_RMS_DBFS = -58.0
DEFAULT_MAX_POSSIBLE_OTHER_PROXY_CORRELATION = 0.42
DEFAULT_MIN_POSSIBLE_OTHER_PROXY_RMS_RATIO_DB = -18.0
PROXY_REFERENCE_STEMS = ("other", "melody", "guitar", "piano")
LONG_TAIL_ROUTER_MODE = "long_tail_model_router_v1"
WINDOWED_PROXY_DECORRELATION_CANDIDATE_TYPE = "windowed_proxy_decorrelated_candidate"
HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE = "banquet_byoq_harmonic_residual_query"
HARMONIC_RESIDUAL_QUERY_MODE = "harmonic_residual_single"
SPARSE_HARMONIC_RESIDUAL_CANDIDATE_TYPE = "sparse_harmonic_residual_candidate"
HARMONIC_RESIDUAL_QUERY_STEMS = (
    "strings_orchestra",
    "synth_lead",
    "synth_pad",
    "arps_plucks",
    "brass_woodwinds",
    "organ_reeds",
)
HARMONIC_RESIDUAL_PROJECTION_STRENGTHS = {
    "drums": 0.95,
    "bass": 0.95,
    "vocals": 0.9,
    "guitar": 0.75,
    "piano": 0.75,
    "melody": 0.75,
}
MIN_HARMONIC_RESIDUAL_REFERENCES = 2
MIN_SPARSE_HARMONIC_RESIDUAL_REFERENCES = 4

SUPPORTED_LONG_TAIL_STEMS = (
    "strings_orchestra",
    "synth_lead",
    "synth_pad",
    "arps_plucks",
    "brass_woodwinds",
    "organ_reeds",
    "fx_risers_impacts",
)


@dataclass(frozen=True)
class PathMapping:
    local_prefix: str
    querysep_prefix: str


@dataclass(frozen=True)
class LongTailQualityThresholds:
    min_duration_sec: float
    max_duration_delta_sec: float
    min_rms_dbfs: float
    min_source_rms_ratio_db: float
    min_active_presence_windows: int
    min_active_presence_rms_dbfs: float
    active_presence_window_sec: float
    max_proxy_correlation: float
    max_windowed_proxy_correlation: float
    windowed_proxy_correlation_sec: float
    min_windowed_proxy_rms_dbfs: float
    max_possible_other_proxy_correlation: float
    min_possible_other_proxy_rms_ratio_db: float


class LongTailQualityGateError(RuntimeError):
    def __init__(self, target_stem: str, stem_result: dict[str, Any]) -> None:
        self.target_stem = target_stem
        self.stem_result = stem_result
        findings = stem_result.get("quality_gate", {}).get("findings", [])
        super().__init__(
            f"querysep-worker output for {target_stem} rejected by long-tail quality gate: {findings}"
        )


def _slash_path(value: str | Path) -> str:
    return str(value).replace("\\", "/")


def _parse_path_mappings(raw_mappings: Any) -> tuple[PathMapping, ...]:
    if raw_mappings in (None, "", []):
        return ()
    if not isinstance(raw_mappings, list):
        raise ValueError("path_mappings must be a list of {local_prefix, querysep_prefix} objects")

    mappings: list[PathMapping] = []
    for raw_mapping in raw_mappings:
        if not isinstance(raw_mapping, dict):
            raise ValueError("path_mappings entries must be objects")
        local_prefix = _slash_path(str(raw_mapping.get("local_prefix") or "")).rstrip("/")
        querysep_prefix = _slash_path(str(raw_mapping.get("querysep_prefix") or "")).rstrip("/")
        if local_prefix == "" or querysep_prefix == "":
            raise ValueError("path_mappings entries require local_prefix and querysep_prefix")
        mappings.append(PathMapping(local_prefix=local_prefix, querysep_prefix=querysep_prefix))
    return tuple(mappings)


def _translate_path(value: str | Path, mappings: tuple[PathMapping, ...], *, to_querysep: bool) -> str:
    normalized_value = _slash_path(value)
    for mapping in mappings:
        source_prefix = mapping.local_prefix if to_querysep else mapping.querysep_prefix
        target_prefix = mapping.querysep_prefix if to_querysep else mapping.local_prefix
        if normalized_value == source_prefix:
            return target_prefix
        if normalized_value.startswith(source_prefix + "/"):
            suffix = normalized_value[len(source_prefix):].lstrip("/")
            return f"{target_prefix}/{suffix}" if suffix else target_prefix
    return normalized_value


def _normalize_service_url(raw_url: str | None) -> str:
    url = str(raw_url or os.getenv("QUERYSEP_WORKER_URL", "")).strip().rstrip("/")
    if url == "":
        raise ValueError("querysep_worker_url or QUERYSEP_WORKER_URL is required")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
        raise ValueError(f"Invalid querysep_worker_url: {url}")
    return url


def _parse_target_stems(raw_target_stems: Any) -> list[str]:
    if raw_target_stems is None:
        stems = ["strings_orchestra"]
    elif isinstance(raw_target_stems, str):
        stems = [part.strip() for part in raw_target_stems.split(",") if part.strip()]
    elif isinstance(raw_target_stems, list):
        stems = [str(part).strip() for part in raw_target_stems if str(part).strip()]
    else:
        raise ValueError("target_stems must be a list or comma-separated string")

    if not stems:
        raise ValueError("target_stems must not be empty")

    unsupported = [stem for stem in stems if stem not in SUPPORTED_LONG_TAIL_STEMS]
    if unsupported:
        supported = ", ".join(SUPPORTED_LONG_TAIL_STEMS)
        raise ValueError(f"Unsupported long-tail stems: {unsupported}. Supported: {supported}")
    return stems


def _parse_bool(raw_value: Any, *, field_name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value in (None, ""):
        return False
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"false", "0", "no", "off"}:
            return False
        if normalized in {"true", "1", "yes", "on"}:
            return True
    raise ValueError(f"{field_name} must be a boolean")


def _parse_float(raw_value: Any, *, field_name: str, default: float) -> float:
    if raw_value in (None, ""):
        return float(default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _parse_int(raw_value: Any, *, field_name: str, default: int) -> int:
    if raw_value in (None, ""):
        return int(default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return value


def _configured_value(raw_value: Any, env_name: str) -> Any:
    if raw_value not in (None, ""):
        return raw_value
    return os.getenv(env_name)


def _sanitize_candidate_id(raw_value: str) -> str:
    candidate_id = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(raw_value).strip()
    ).strip("_")
    return candidate_id or "candidate"


def _build_quality_thresholds(
    *,
    min_quality_duration_sec: Any = None,
    max_duration_delta_sec: Any = None,
    min_quality_rms_dbfs: Any = None,
    min_source_rms_ratio_db: Any = None,
    min_active_presence_windows: Any = None,
    min_active_presence_rms_dbfs: Any = None,
    active_presence_window_sec: Any = None,
    max_proxy_correlation: Any = None,
    max_windowed_proxy_correlation: Any = None,
    windowed_proxy_correlation_sec: Any = None,
    min_windowed_proxy_rms_dbfs: Any = None,
    max_possible_other_proxy_correlation: Any = None,
    min_possible_other_proxy_rms_ratio_db: Any = None,
) -> LongTailQualityThresholds:
    thresholds = LongTailQualityThresholds(
        min_duration_sec=_parse_float(
            _configured_value(min_quality_duration_sec, "QUERYSEP_LONG_TAIL_MIN_DURATION_SEC"),
            field_name="min_quality_duration_sec",
            default=DEFAULT_MIN_QUALITY_DURATION_SECONDS,
        ),
        max_duration_delta_sec=_parse_float(
            _configured_value(max_duration_delta_sec, "QUERYSEP_LONG_TAIL_MAX_DURATION_DELTA_SEC"),
            field_name="max_duration_delta_sec",
            default=DEFAULT_MAX_DURATION_DELTA_SECONDS,
        ),
        min_rms_dbfs=_parse_float(
            _configured_value(min_quality_rms_dbfs, "QUERYSEP_LONG_TAIL_MIN_RMS_DBFS"),
            field_name="min_quality_rms_dbfs",
            default=DEFAULT_MIN_LONG_TAIL_RMS_DBFS,
        ),
        min_source_rms_ratio_db=_parse_float(
            _configured_value(min_source_rms_ratio_db, "QUERYSEP_LONG_TAIL_MIN_SOURCE_RMS_RATIO_DB"),
            field_name="min_source_rms_ratio_db",
            default=DEFAULT_MIN_SOURCE_RMS_RATIO_DB,
        ),
        min_active_presence_windows=_parse_int(
            _configured_value(min_active_presence_windows, "QUERYSEP_LONG_TAIL_MIN_ACTIVE_PRESENCE_WINDOWS"),
            field_name="min_active_presence_windows",
            default=DEFAULT_MIN_ACTIVE_PRESENCE_WINDOWS,
        ),
        min_active_presence_rms_dbfs=_parse_float(
            _configured_value(min_active_presence_rms_dbfs, "QUERYSEP_LONG_TAIL_MIN_ACTIVE_PRESENCE_RMS_DBFS"),
            field_name="min_active_presence_rms_dbfs",
            default=DEFAULT_MIN_ACTIVE_PRESENCE_RMS_DBFS,
        ),
        active_presence_window_sec=_parse_float(
            _configured_value(active_presence_window_sec, "QUERYSEP_LONG_TAIL_ACTIVE_PRESENCE_WINDOW_SEC"),
            field_name="active_presence_window_sec",
            default=DEFAULT_ACTIVE_PRESENCE_WINDOW_SECONDS,
        ),
        max_proxy_correlation=_parse_float(
            _configured_value(max_proxy_correlation, "QUERYSEP_LONG_TAIL_MAX_PROXY_CORRELATION"),
            field_name="max_proxy_correlation",
            default=DEFAULT_MAX_PROXY_CORRELATION,
        ),
        max_windowed_proxy_correlation=_parse_float(
            _configured_value(max_windowed_proxy_correlation, "QUERYSEP_LONG_TAIL_MAX_WINDOWED_PROXY_CORRELATION"),
            field_name="max_windowed_proxy_correlation",
            default=DEFAULT_MAX_WINDOWED_PROXY_CORRELATION,
        ),
        windowed_proxy_correlation_sec=_parse_float(
            _configured_value(windowed_proxy_correlation_sec, "QUERYSEP_LONG_TAIL_WINDOWED_PROXY_CORRELATION_SEC"),
            field_name="windowed_proxy_correlation_sec",
            default=DEFAULT_WINDOWED_PROXY_CORRELATION_SECONDS,
        ),
        min_windowed_proxy_rms_dbfs=_parse_float(
            _configured_value(min_windowed_proxy_rms_dbfs, "QUERYSEP_LONG_TAIL_MIN_WINDOWED_PROXY_RMS_DBFS"),
            field_name="min_windowed_proxy_rms_dbfs",
            default=DEFAULT_MIN_WINDOWED_PROXY_RMS_DBFS,
        ),
        max_possible_other_proxy_correlation=_parse_float(
            _configured_value(
                max_possible_other_proxy_correlation,
                "QUERYSEP_LONG_TAIL_MAX_POSSIBLE_OTHER_PROXY_CORRELATION",
            ),
            field_name="max_possible_other_proxy_correlation",
            default=DEFAULT_MAX_POSSIBLE_OTHER_PROXY_CORRELATION,
        ),
        min_possible_other_proxy_rms_ratio_db=_parse_float(
            _configured_value(
                min_possible_other_proxy_rms_ratio_db,
                "QUERYSEP_LONG_TAIL_MIN_POSSIBLE_OTHER_PROXY_RMS_RATIO_DB",
            ),
            field_name="min_possible_other_proxy_rms_ratio_db",
            default=DEFAULT_MIN_POSSIBLE_OTHER_PROXY_RMS_RATIO_DB,
        ),
    )
    if thresholds.min_duration_sec <= 0:
        raise ValueError("min_quality_duration_sec must be positive")
    if thresholds.max_duration_delta_sec < 0:
        raise ValueError("max_duration_delta_sec must be non-negative")
    if thresholds.max_proxy_correlation <= 0.0 or thresholds.max_proxy_correlation > 1.0:
        raise ValueError("max_proxy_correlation must be in (0, 1]")
    if thresholds.min_active_presence_windows <= 0:
        raise ValueError("min_active_presence_windows must be positive")
    if thresholds.active_presence_window_sec <= 0.0:
        raise ValueError("active_presence_window_sec must be positive")
    if thresholds.max_windowed_proxy_correlation <= 0.0 or thresholds.max_windowed_proxy_correlation > 1.0:
        raise ValueError("max_windowed_proxy_correlation must be in (0, 1]")
    if thresholds.windowed_proxy_correlation_sec <= 0.0:
        raise ValueError("windowed_proxy_correlation_sec must be positive")
    if thresholds.max_possible_other_proxy_correlation <= 0.0 or thresholds.max_possible_other_proxy_correlation > 1.0:
        raise ValueError("max_possible_other_proxy_correlation must be in (0, 1]")
    return thresholds


def _safe_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _dbfs(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), 1e-12)))


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    rms = _safe_rms(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "frames": int(audio.shape[0]) if audio.ndim >= 1 else 0,
        "duration_sec": round(float(audio.shape[0] / sample_rate), 6) if sample_rate > 0 else 0.0,
        "rms": round(float(rms), 9),
        "rms_dbfs": round(_dbfs(rms), 3),
        "peak": round(float(peak), 9),
        "peak_dbfs": round(_dbfs(peak), 3),
    }


def _mono_for_comparison(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    return np.mean(audio, axis=1, dtype=np.float64)


def _rms_ratio_db(candidate_audio: np.ndarray, reference_audio: np.ndarray) -> float:
    candidate_rms = _safe_rms(candidate_audio)
    reference_rms = _safe_rms(reference_audio)
    return float(20.0 * math.log10(max(candidate_rms, 1e-12) / max(reference_rms, 1e-12)))


def _audio_correlation(candidate_audio: np.ndarray, reference_audio: np.ndarray) -> float:
    candidate_mono = _mono_for_comparison(candidate_audio)
    reference_mono = _mono_for_comparison(reference_audio)
    frame_count = min(candidate_mono.shape[0], reference_mono.shape[0])
    if frame_count < 2:
        return 0.0
    candidate_window = candidate_mono[:frame_count] - float(np.mean(candidate_mono[:frame_count]))
    reference_window = reference_mono[:frame_count] - float(np.mean(reference_mono[:frame_count]))
    denominator = float(np.linalg.norm(candidate_window) * np.linalg.norm(reference_window))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(candidate_window, reference_window) / denominator)


def _windowed_proxy_correlation_report(
    *,
    output_audio: np.ndarray,
    proxy_audio: np.ndarray,
    sample_rate: int,
    thresholds: LongTailQualityThresholds,
) -> dict[str, Any]:
    window_frames = max(2, int(round(thresholds.windowed_proxy_correlation_sec * sample_rate)))
    minimum_window_frames = max(2, min(window_frames, int(round(0.5 * sample_rate))))
    frame_count = min(output_audio.shape[0], proxy_audio.shape[0])
    if sample_rate <= 0 or frame_count < minimum_window_frames:
        return {
            "status": "insufficient_duration",
            "window_sec": thresholds.windowed_proxy_correlation_sec,
            "min_windowed_proxy_rms_dbfs": thresholds.min_windowed_proxy_rms_dbfs,
            "active_window_count": 0,
            "max_active_abs_correlation": 0.0,
            "top_windows": [],
        }

    active_windows: list[dict[str, Any]] = []
    for start in range(0, frame_count, window_frames):
        end = min(start + window_frames, frame_count)
        if end - start < minimum_window_frames:
            continue
        output_window = output_audio[start:end]
        output_rms_dbfs = round(_dbfs(_safe_rms(output_window)), 3)
        if output_rms_dbfs < thresholds.min_windowed_proxy_rms_dbfs:
            continue
        proxy_window = proxy_audio[start:end]
        correlation = _audio_correlation(output_window, proxy_window)
        active_windows.append(
            {
                "start_sec": round(float(start / sample_rate), 3),
                "end_sec": round(float(end / sample_rate), 3),
                "output_rms_dbfs": output_rms_dbfs,
                "correlation": round(float(correlation), 6),
                "abs_correlation": round(abs(float(correlation)), 6),
            }
        )

    top_windows = sorted(active_windows, key=lambda window: float(window["abs_correlation"]), reverse=True)[:8]
    max_active_abs_correlation = float(top_windows[0]["abs_correlation"]) if top_windows else 0.0
    return {
        "status": "compared",
        "window_sec": thresholds.windowed_proxy_correlation_sec,
        "min_windowed_proxy_rms_dbfs": thresholds.min_windowed_proxy_rms_dbfs,
        "max_windowed_proxy_correlation": thresholds.max_windowed_proxy_correlation,
        "active_window_count": len(active_windows),
        "max_active_abs_correlation": round(max_active_abs_correlation, 6),
        "top_windows": top_windows,
    }


def _active_presence_report(
    *,
    audio: np.ndarray,
    sample_rate: int,
    thresholds: LongTailQualityThresholds,
) -> dict[str, Any]:
    window_frames = max(2, int(round(thresholds.active_presence_window_sec * sample_rate)))
    minimum_window_frames = max(2, min(window_frames, int(round(0.5 * sample_rate))))
    if sample_rate <= 0 or audio.shape[0] < minimum_window_frames:
        return {
            "status": "insufficient_duration",
            "window_sec": thresholds.active_presence_window_sec,
            "min_active_presence_rms_dbfs": thresholds.min_active_presence_rms_dbfs,
            "min_active_presence_windows": thresholds.min_active_presence_windows,
            "required_active_windows": 0,
            "active_window_count": 0,
            "window_count": 0,
            "top_windows": [],
        }

    windows: list[dict[str, Any]] = []
    for start in range(0, audio.shape[0], window_frames):
        end = min(start + window_frames, audio.shape[0])
        if end - start < minimum_window_frames:
            continue
        window_rms_dbfs = round(_dbfs(_safe_rms(audio[start:end])), 3)
        windows.append(
            {
                "start_sec": round(float(start / sample_rate), 3),
                "end_sec": round(float(end / sample_rate), 3),
                "rms_dbfs": window_rms_dbfs,
            }
        )

    active_windows = [
        window for window in windows if float(window["rms_dbfs"]) >= thresholds.min_active_presence_rms_dbfs
    ]
    required_active_windows = min(thresholds.min_active_presence_windows, max(1, len(windows)))
    top_windows = sorted(windows, key=lambda window: float(window["rms_dbfs"]), reverse=True)[:8]
    return {
        "status": "measured",
        "window_sec": thresholds.active_presence_window_sec,
        "min_active_presence_rms_dbfs": thresholds.min_active_presence_rms_dbfs,
        "min_active_presence_windows": thresholds.min_active_presence_windows,
        "required_active_windows": required_active_windows,
        "active_window_count": len(active_windows),
        "window_count": len(windows),
        "active_window_ratio": round(float(len(active_windows) / max(1, len(windows))), 6),
        "top_windows": top_windows,
    }


def _quality_gate_has_windowed_proxy_leakage(quality_gate: dict[str, Any]) -> bool:
    findings = quality_gate.get("findings") if isinstance(quality_gate.get("findings"), list) else []
    return any(finding.get("label") == "windowed_proxy_correlation_too_high" for finding in findings)


def _quality_gate_has_proxy_copy(quality_gate: dict[str, Any]) -> bool:
    findings = quality_gate.get("findings") if isinstance(quality_gate.get("findings"), list) else []
    return any(finding.get("label") == "proxy_copy_correlation_too_high" for finding in findings)


def _quality_gate_has_presence_failure(quality_gate: dict[str, Any]) -> bool:
    findings = quality_gate.get("findings") if isinstance(quality_gate.get("findings"), list) else []
    presence_labels = {
        "long_tail_presence_too_weak",
        "long_tail_active_presence_too_sparse",
        "long_tail_presence_too_weak_vs_source",
    }
    return any(finding.get("label") in presence_labels for finding in findings)


def _decorrelation_strength(abs_correlation: float, threshold: float) -> float:
    excess = max(0.0, float(abs_correlation) - float(threshold))
    return float(min(1.0, max(0.35, 0.35 + excess / 0.35)))


def _decorrelation_envelope(frame_count: int, sample_rate: int) -> np.ndarray:
    envelope = np.ones((frame_count, 1), dtype=np.float32)
    fade_frames = min(frame_count // 4, max(1, int(round(0.05 * sample_rate))))
    if fade_frames <= 1:
        return envelope
    fade_in = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32).reshape(-1, 1)
    envelope[:fade_frames] = fade_in
    envelope[-fade_frames:] = fade_in[::-1]
    return envelope


def _write_windowed_proxy_decorrelated_candidate(
    *,
    target_stem: str,
    source_candidate: dict[str, Any],
    output_dir: Path,
    build_from_paths: dict[str, str],
    thresholds: LongTailQualityThresholds,
) -> dict[str, Any] | None:
    source_candidate_id = str(source_candidate.get("candidate_id") or "").strip()
    source_output_path = Path(str(source_candidate.get("local_output_path") or ""))
    if source_candidate_id == "" or not source_output_path.exists():
        return None
    if source_candidate.get("production_eligible") is not True:
        return None
    quality_gate = source_candidate.get("quality_gate", {})
    if _quality_gate_has_proxy_copy(quality_gate):
        return None
    if not _quality_gate_has_windowed_proxy_leakage(quality_gate):
        return None

    try:
        repaired_audio, sample_rate = _read_audio(source_output_path)
    except RuntimeError:
        return None
    if repaired_audio.size == 0 or sample_rate <= 0:
        return None

    window_frames = max(2, int(round(thresholds.windowed_proxy_correlation_sec * sample_rate)))
    minimum_window_frames = max(2, min(window_frames, int(round(0.5 * sample_rate))))
    repaired_audio = repaired_audio.astype(np.float32, copy=True)
    repaired_windows: list[dict[str, Any]] = []

    for proxy_stem in PROXY_REFERENCE_STEMS:
        raw_proxy_path = build_from_paths.get(proxy_stem)
        if not raw_proxy_path:
            continue
        proxy_path = Path(raw_proxy_path)
        try:
            proxy_audio, proxy_sample_rate = _read_audio(proxy_path)
        except RuntimeError:
            continue
        if proxy_sample_rate != sample_rate:
            continue
        frame_count = min(repaired_audio.shape[0], proxy_audio.shape[0])
        proxy_audio = _copy_aligned_audio_shape(
            proxy_audio,
            frame_count=frame_count,
            channels=repaired_audio.shape[1],
        )
        for start in range(0, frame_count, window_frames):
            end = min(start + window_frames, frame_count)
            if end - start < minimum_window_frames:
                continue
            output_window = repaired_audio[start:end]
            output_rms_dbfs = _dbfs(_safe_rms(output_window))
            if output_rms_dbfs < thresholds.min_windowed_proxy_rms_dbfs:
                continue
            proxy_window = proxy_audio[start:end]
            correlation = _audio_correlation(output_window, proxy_window)
            abs_correlation = abs(float(correlation))
            if abs_correlation <= thresholds.max_windowed_proxy_correlation:
                continue

            centered_proxy = proxy_window - np.mean(proxy_window, axis=0, keepdims=True)
            centered_output = output_window - np.mean(output_window, axis=0, keepdims=True)
            denominator = np.sum(centered_proxy * centered_proxy, axis=0, keepdims=True) + 1e-12
            projection = centered_proxy * (np.sum(centered_output * centered_proxy, axis=0, keepdims=True) / denominator)
            strength = _decorrelation_strength(abs_correlation, thresholds.max_windowed_proxy_correlation)
            envelope = _decorrelation_envelope(end - start, sample_rate)
            repaired_audio[start:end] = (
                repaired_audio[start:end] - projection.astype(np.float32) * float(strength) * envelope
            ).astype(np.float32)
            repaired_windows.append(
                {
                    "proxy_stem": proxy_stem,
                    "start_sec": round(float(start / sample_rate), 3),
                    "end_sec": round(float(end / sample_rate), 3),
                    "correlation_before": round(float(correlation), 6),
                    "abs_correlation_before": round(abs_correlation, 6),
                    "decorrelation_strength": round(float(strength), 6),
                    "output_rms_dbfs_before": round(float(output_rms_dbfs), 3),
                }
            )

    if not repaired_windows:
        return None

    peak = float(np.max(np.abs(repaired_audio))) if repaired_audio.size else 0.0
    if peak > 1.0:
        repaired_audio = (repaired_audio / peak).astype(np.float32)

    candidate_dir = output_dir / "router_candidates" / target_stem / "windowed_proxy_decorrelated"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    safe_source_id = _sanitize_candidate_id(source_candidate_id)
    candidate_path = candidate_dir / f"{safe_source_id}.flac"
    sf.write(str(candidate_path), repaired_audio, sample_rate, subtype="PCM_24")
    return {
        "candidate_id": f"{safe_source_id}_proxy_decorrelated",
        "candidate_type": WINDOWED_PROXY_DECORRELATION_CANDIDATE_TYPE,
        "production_eligible": True,
        "status": "materialized",
        "local_output_path": str(candidate_path),
        "source_candidate_id": source_candidate_id,
        "source_candidate_type": source_candidate.get("candidate_type"),
        "repaired_windows": repaired_windows,
        "repair_window_count": len(repaired_windows),
        "notes": (
            "Production repair candidate derived from a production model output by removing only local "
            "linear projection against active proxy-bus leakage windows. Selection still requires the full "
            "long-tail acoustic quality gate."
        ),
    }


def _copy_aligned_audio_shape(audio: np.ndarray, *, frame_count: int, channels: int) -> np.ndarray:
    aligned = audio[:frame_count]
    if aligned.ndim == 1:
        aligned = aligned.reshape(-1, 1)
    if aligned.shape[1] == channels:
        return aligned.astype(np.float32, copy=False)
    if aligned.shape[1] == 1 and channels == 2:
        return np.repeat(aligned, 2, axis=1).astype(np.float32, copy=False)
    if aligned.shape[1] > channels:
        return aligned[:, :channels].astype(np.float32, copy=False)
    repeats = int(math.ceil(channels / aligned.shape[1]))
    return np.tile(aligned, (1, repeats))[:, :channels].astype(np.float32, copy=False)


def _remove_linear_projection(
    base_audio: np.ndarray,
    reference_audio: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    frame_count = min(base_audio.shape[0], reference_audio.shape[0])
    if frame_count < 2:
        return base_audio
    aligned_reference = _copy_aligned_audio_shape(
        reference_audio,
        frame_count=frame_count,
        channels=base_audio.shape[1],
    )
    output = base_audio.astype(np.float32, copy=True)
    centered_reference = aligned_reference - np.mean(aligned_reference, axis=0, keepdims=True)
    centered_output = output[:frame_count] - np.mean(output[:frame_count], axis=0, keepdims=True)
    denominator = np.sum(centered_reference * centered_reference, axis=0, keepdims=True) + 1e-12
    coefficient = np.sum(centered_output * centered_reference, axis=0, keepdims=True) / denominator
    coefficient = np.clip(coefficient, -2.0, 2.0)
    output[:frame_count] = (
        output[:frame_count] - centered_reference * coefficient.astype(np.float32) * float(strength)
    ).astype(np.float32)
    return output


def _write_harmonic_residual_query_bus(
    *,
    target_stem: str,
    output_dir: Path,
    source_audio_path: Path,
    build_from_paths: dict[str, str],
) -> dict[str, Any] | None:
    if target_stem not in HARMONIC_RESIDUAL_QUERY_STEMS:
        return None

    try:
        residual_audio, sample_rate = _read_audio(source_audio_path)
    except RuntimeError:
        return None
    if residual_audio.size == 0 or sample_rate <= 0:
        return None

    used_references: list[dict[str, Any]] = []
    for stem_name, strength in HARMONIC_RESIDUAL_PROJECTION_STRENGTHS.items():
        raw_path = build_from_paths.get(stem_name)
        if not raw_path:
            continue
        stem_path = Path(raw_path)
        try:
            reference_audio, reference_sample_rate = _read_audio(stem_path)
        except RuntimeError:
            continue
        if reference_sample_rate != sample_rate or reference_audio.size == 0 or _safe_rms(reference_audio) <= 1e-8:
            continue
        residual_audio = _remove_linear_projection(
            residual_audio,
            reference_audio,
            strength=float(strength),
        )
        used_references.append(
            {
                "stem": stem_name,
                "path": str(stem_path),
                "projection_strength": float(strength),
            }
        )

    if len(used_references) < MIN_HARMONIC_RESIDUAL_REFERENCES:
        return {
            "candidate_id": "harmonic_residual_query_bus",
            "candidate_type": "query_auxiliary_bus",
            "production_eligible": False,
            "status": "skipped",
            "skip_reason": "insufficient_projection_references",
            "used_reference_count": len(used_references),
            "min_reference_count": MIN_HARMONIC_RESIDUAL_REFERENCES,
            "used_references": used_references,
        }

    peak = float(np.max(np.abs(residual_audio))) if residual_audio.size else 0.0
    if peak > 1.0:
        residual_audio = (residual_audio / peak).astype(np.float32)

    candidate_dir = output_dir / "router_auxiliary" / target_stem / "harmonic_residual_query"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    query_bus_path = candidate_dir / "harmonic_residual.flac"
    sf.write(str(query_bus_path), residual_audio.astype(np.float32, copy=False), sample_rate, subtype="PCM_24")
    return {
        "candidate_id": "harmonic_residual_query_bus",
        "candidate_type": "query_auxiliary_bus",
        "production_eligible": False,
        "status": "materialized",
        "local_output_path": str(query_bus_path),
        "used_references": used_references,
        "metrics": _audio_metrics(residual_audio, sample_rate),
        "notes": (
            "Auxiliary query bus for a second Banquet BYOQ pass. This file is never selected as a stem; "
            "it only gives querysep-worker a provenance-visible harmonic_residual component."
        ),
    }


def _write_sparse_harmonic_residual_candidate(
    *,
    target_stem: str,
    output_dir: Path,
    harmonic_query_bus: dict[str, Any],
) -> dict[str, Any] | None:
    if target_stem not in HARMONIC_RESIDUAL_QUERY_STEMS:
        return None
    if harmonic_query_bus.get("status") != "materialized":
        return None
    used_references = harmonic_query_bus.get("used_references") if isinstance(harmonic_query_bus.get("used_references"), list) else []
    if len(used_references) < MIN_SPARSE_HARMONIC_RESIDUAL_REFERENCES:
        return {
            "candidate_id": "sparse_harmonic_residual",
            "candidate_type": SPARSE_HARMONIC_RESIDUAL_CANDIDATE_TYPE,
            "production_eligible": False,
            "status": "skipped",
            "skip_reason": "insufficient_sparse_harmonic_references",
            "used_reference_count": len(used_references),
            "min_reference_count": MIN_SPARSE_HARMONIC_RESIDUAL_REFERENCES,
        }

    raw_query_path = harmonic_query_bus.get("local_output_path")
    if not raw_query_path:
        return {
            "candidate_id": "sparse_harmonic_residual",
            "candidate_type": SPARSE_HARMONIC_RESIDUAL_CANDIDATE_TYPE,
            "production_eligible": True,
            "status": "rejected",
            "skip_reason": "harmonic_residual_query_bus_path_missing",
        }

    query_bus_path = Path(str(raw_query_path))
    try:
        residual_audio, sample_rate = _read_audio(query_bus_path)
    except RuntimeError as exc:
        return {
            "candidate_id": "sparse_harmonic_residual",
            "candidate_type": SPARSE_HARMONIC_RESIDUAL_CANDIDATE_TYPE,
            "production_eligible": True,
            "status": "rejected",
            "skip_reason": "harmonic_residual_query_bus_unreadable",
            "error": str(exc),
        }
    if residual_audio.size == 0 or sample_rate <= 0 or _safe_rms(residual_audio) <= 1e-8:
        return {
            "candidate_id": "sparse_harmonic_residual",
            "candidate_type": SPARSE_HARMONIC_RESIDUAL_CANDIDATE_TYPE,
            "production_eligible": True,
            "status": "rejected",
            "skip_reason": "harmonic_residual_query_bus_empty",
            "metrics": _audio_metrics(residual_audio, sample_rate) if sample_rate > 0 else {},
        }

    peak = float(np.max(np.abs(residual_audio))) if residual_audio.size else 0.0
    if peak > 1.0:
        residual_audio = (residual_audio / peak).astype(np.float32)

    candidate_dir = output_dir / "router_candidates" / target_stem / "sparse_harmonic_residual"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / "sparse_harmonic_residual.flac"
    sf.write(str(candidate_path), residual_audio.astype(np.float32, copy=False), sample_rate, subtype="PCM_24")
    return {
        "candidate_id": "sparse_harmonic_residual",
        "candidate_type": SPARSE_HARMONIC_RESIDUAL_CANDIDATE_TYPE,
        "production_eligible": True,
        "status": "materialized",
        "local_output_path": str(candidate_path),
        "source_auxiliary_bus_id": harmonic_query_bus.get("candidate_id"),
        "source_auxiliary_path": str(query_bus_path),
        "used_references": used_references,
        "metrics": _audio_metrics(residual_audio, sample_rate),
        "notes": (
            "Production candidate materialized from the model-guided harmonic residual query bus after the "
            "harmonic Banquet pass produced a presence-failed output. Selection still requires the full "
            "long-tail acoustic quality gate; the auxiliary bus itself remains non-selectable."
        ),
    }


def _write_diagnostic_bus_sum_candidate(
    *,
    target_stem: str,
    output_dir: Path,
    build_from_paths: dict[str, str],
) -> dict[str, Any] | None:
    loaded_sources: list[tuple[str, Path, np.ndarray, int]] = []
    for proxy_stem in PROXY_REFERENCE_STEMS:
        raw_path = build_from_paths.get(proxy_stem)
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            audio, sample_rate = _read_audio(path)
        except RuntimeError:
            continue
        if audio.size == 0 or _safe_rms(audio) <= 1e-8:
            continue
        loaded_sources.append((proxy_stem, path, audio, sample_rate))

    if len(loaded_sources) < 2:
        return None

    sample_rates = {sample_rate for _, _, _, sample_rate in loaded_sources}
    if len(sample_rates) != 1:
        return {
            "candidate_id": "diagnostic_harmonic_bus_sum",
            "candidate_type": "diagnostic_bus_sum",
            "production_eligible": False,
            "status": "skipped",
            "skip_reason": "source_bus_sample_rate_mismatch",
            "source_sample_rates": {
                stem_name: sample_rate for stem_name, _, _, sample_rate in loaded_sources
            },
        }

    frame_count = min(audio.shape[0] for _, _, audio, _ in loaded_sources)
    channels = max(audio.shape[1] if audio.ndim == 2 else 1 for _, _, audio, _ in loaded_sources)
    if frame_count <= 0:
        return None

    summed = np.zeros((frame_count, channels), dtype=np.float32)
    source_paths: dict[str, str] = {}
    for stem_name, path, audio, _ in loaded_sources:
        summed += _copy_aligned_audio_shape(audio, frame_count=frame_count, channels=channels)
        source_paths[stem_name] = str(path)

    peak = float(np.max(np.abs(summed))) if summed.size else 0.0
    if peak > 1.0:
        summed = (summed / peak).astype(np.float32)

    candidate_dir = output_dir / "router_candidates" / target_stem
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / "diagnostic_harmonic_bus_sum.flac"
    sf.write(str(candidate_path), summed, next(iter(sample_rates)), subtype="PCM_24")
    return {
        "candidate_id": "diagnostic_harmonic_bus_sum",
        "candidate_type": "diagnostic_bus_sum",
        "production_eligible": False,
        "status": "materialized",
        "local_output_path": str(candidate_path),
        "source_paths": source_paths,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return loaded


def _extract_manifest_stem_paths(source_stem_manifest_path: Path) -> dict[str, str]:
    if not source_stem_manifest_path.exists():
        raise FileNotFoundError(f"Source stem manifest is missing: {source_stem_manifest_path}")

    manifest = _read_json(source_stem_manifest_path)
    stems = manifest.get("stems")
    if not isinstance(stems, dict) or stems == {}:
        raise ValueError(f"Source stem manifest has no stems: {source_stem_manifest_path}")

    stem_paths: dict[str, str] = {}
    for stem_name, raw_stem in stems.items():
        if not isinstance(raw_stem, dict):
            continue
        raw_path = str(raw_stem.get("path") or "").strip()
        if raw_path == "":
            continue
        local_path = Path(raw_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Source stem file from manifest is missing: {local_path}")
        stem_paths[str(stem_name)] = str(local_path)

    if stem_paths == {}:
        raise ValueError(f"Source stem manifest has no usable stem paths: {source_stem_manifest_path}")
    return stem_paths


def _resolve_build_from_paths(
    *,
    explicit_build_from_paths: Any,
    source_stem_manifest_path: str | Path | None,
) -> dict[str, str]:
    if explicit_build_from_paths not in (None, "", []):
        if not isinstance(explicit_build_from_paths, dict):
            raise ValueError("build_from_paths must be an object mapping stem name to local path")
        resolved: dict[str, str] = {}
        for stem_name, raw_path in explicit_build_from_paths.items():
            local_path = Path(str(raw_path))
            if not local_path.exists():
                raise FileNotFoundError(f"Explicit build_from_paths stem file is missing: {local_path}")
            resolved[str(stem_name)] = str(local_path)
        if resolved == {}:
            raise ValueError("build_from_paths must not be empty")
        return resolved

    if source_stem_manifest_path in (None, ""):
        raise ValueError("source_stem_manifest_path or build_from_paths is required")
    return _extract_manifest_stem_paths(Path(str(source_stem_manifest_path)))


def _resolve_external_candidate_paths(raw_candidate_paths: Any, target_stem: str) -> dict[str, str]:
    if raw_candidate_paths in (None, "", []):
        return {}
    if not isinstance(raw_candidate_paths, dict):
        raise ValueError("long_tail_candidate_paths must be an object")

    target_candidates = raw_candidate_paths.get(target_stem)
    if isinstance(target_candidates, dict):
        candidate_items = target_candidates.items()
    else:
        candidate_items = raw_candidate_paths.items()

    resolved: dict[str, str] = {}
    for raw_name, raw_path in candidate_items:
        if isinstance(raw_path, dict):
            raw_path = raw_path.get("path")
        candidate_path = Path(str(raw_path or ""))
        if str(candidate_path).strip() == "":
            continue
        if not candidate_path.exists():
            raise FileNotFoundError(f"Long-tail external candidate is missing: {candidate_path}")
        candidate_id = _sanitize_candidate_id(str(raw_name))
        resolved[candidate_id] = str(candidate_path)
    return resolved


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"querysep-worker returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"querysep-worker is unreachable: {exc}") from exc

    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError("querysep-worker response root must be an object")
    return decoded


def _evaluate_long_tail_quality(
    *,
    target_stem: str,
    output_path: Path,
    source_audio_path: Path,
    build_from_paths: dict[str, str],
    thresholds: LongTailQualityThresholds,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    output_audio, output_sample_rate = _read_audio(output_path)
    output_metrics = _audio_metrics(output_audio, output_sample_rate)
    active_presence = _active_presence_report(
        audio=output_audio,
        sample_rate=output_sample_rate,
        thresholds=thresholds,
    )

    if output_sample_rate <= 0:
        findings.append({"severity": "blocker", "label": "invalid_output_sample_rate", "evidence": output_sample_rate})
    if output_audio.ndim != 2 or output_audio.shape[1] < 1:
        findings.append({"severity": "blocker", "label": "invalid_output_channel_shape", "evidence": list(output_audio.shape)})
    if not np.all(np.isfinite(output_audio)):
        findings.append({"severity": "blocker", "label": "non_finite_output_audio_samples"})
    if float(output_metrics["duration_sec"]) < thresholds.min_duration_sec:
        findings.append(
            {
                "severity": "blocker",
                "label": "output_too_short",
                "evidence": {
                    "duration_sec": output_metrics["duration_sec"],
                    "min_duration_sec": thresholds.min_duration_sec,
                },
            }
        )
    if float(output_metrics["rms_dbfs"]) < thresholds.min_rms_dbfs:
        findings.append(
            {
                "severity": "blocker",
                "label": "long_tail_presence_too_weak",
                "evidence": {
                    "rms_dbfs": output_metrics["rms_dbfs"],
                    "min_rms_dbfs": thresholds.min_rms_dbfs,
                },
            }
        )
    if int(active_presence.get("active_window_count", 0)) < int(active_presence.get("required_active_windows", 0)):
        findings.append(
            {
                "severity": "blocker",
                "label": "long_tail_active_presence_too_sparse",
                "evidence": {
                    "active_window_count": active_presence["active_window_count"],
                    "required_active_windows": active_presence["required_active_windows"],
                    "window_count": active_presence["window_count"],
                    "min_active_presence_rms_dbfs": thresholds.min_active_presence_rms_dbfs,
                    "active_presence_window_sec": thresholds.active_presence_window_sec,
                    "top_windows": active_presence["top_windows"],
                },
            }
        )

    reference_reports: dict[str, Any] = {}
    source_audio, source_sample_rate = _read_audio(source_audio_path)
    source_metrics = _audio_metrics(source_audio, source_sample_rate)
    duration_delta = abs(float(output_metrics["duration_sec"]) - float(source_metrics["duration_sec"]))
    source_rms_ratio_db = _rms_ratio_db(output_audio, source_audio)
    reference_reports["source_audio"] = {
        "path": str(source_audio_path),
        "metrics": source_metrics,
        "output_rms_ratio_db": round(float(source_rms_ratio_db), 3),
        "duration_delta_sec": round(float(duration_delta), 6),
    }
    if duration_delta > thresholds.max_duration_delta_sec:
        findings.append(
            {
                "severity": "blocker",
                "label": "duration_mismatch",
                "evidence": {
                    "duration_delta_sec": round(float(duration_delta), 6),
                    "max_duration_delta_sec": thresholds.max_duration_delta_sec,
                },
            }
        )
    if source_rms_ratio_db < thresholds.min_source_rms_ratio_db:
        findings.append(
            {
                "severity": "blocker",
                "label": "long_tail_presence_too_weak_vs_source",
                "evidence": {
                    "rms_ratio_db": round(float(source_rms_ratio_db), 3),
                    "min_source_rms_ratio_db": thresholds.min_source_rms_ratio_db,
                },
            }
        )

    proxy_reports: dict[str, Any] = {}
    for proxy_stem in PROXY_REFERENCE_STEMS:
        raw_proxy_path = build_from_paths.get(proxy_stem)
        if not raw_proxy_path:
            continue
        proxy_path = Path(raw_proxy_path)
        try:
            proxy_audio, proxy_sample_rate = _read_audio(proxy_path)
        except RuntimeError as exc:
            proxy_reports[proxy_stem] = {
                "path": str(proxy_path),
                "status": "unreadable",
                "error": str(exc),
            }
            continue
        if proxy_sample_rate != output_sample_rate:
            proxy_reports[proxy_stem] = {
                "path": str(proxy_path),
                "status": "rejected_sample_rate_mismatch",
                "sample_rate": int(proxy_sample_rate),
                "output_sample_rate": int(output_sample_rate),
            }
            findings.append(
                {
                    "severity": "blocker",
                    "label": "proxy_sample_rate_mismatch",
                    "evidence": {
                        "proxy_stem": proxy_stem,
                        "sample_rate": int(proxy_sample_rate),
                        "output_sample_rate": int(output_sample_rate),
                    },
                }
            )
            continue
        correlation = _audio_correlation(output_audio, proxy_audio)
        proxy_rms_ratio_db = _rms_ratio_db(output_audio, proxy_audio)
        windowed_correlation = _windowed_proxy_correlation_report(
            output_audio=output_audio,
            proxy_audio=proxy_audio,
            sample_rate=output_sample_rate,
            thresholds=thresholds,
        )
        proxy_reports[proxy_stem] = {
            "path": str(proxy_path),
            "status": "compared",
            "correlation": round(float(correlation), 6),
            "abs_correlation": round(abs(float(correlation)), 6),
            "output_rms_ratio_db": round(float(proxy_rms_ratio_db), 3),
            "windowed_correlation": windowed_correlation,
        }
        if abs(correlation) > thresholds.max_proxy_correlation:
            findings.append(
                {
                    "severity": "blocker",
                    "label": "proxy_copy_correlation_too_high",
                    "evidence": {
                        "proxy_stem": proxy_stem,
                        "abs_correlation": round(abs(float(correlation)), 6),
                        "max_proxy_correlation": thresholds.max_proxy_correlation,
                    },
                }
            )
        if float(windowed_correlation.get("max_active_abs_correlation", 0.0)) > thresholds.max_windowed_proxy_correlation:
            top_window = (windowed_correlation.get("top_windows") or [{}])[0]
            findings.append(
                {
                    "severity": "blocker",
                    "label": "windowed_proxy_correlation_too_high",
                    "evidence": {
                        "proxy_stem": proxy_stem,
                        "max_active_abs_correlation": windowed_correlation["max_active_abs_correlation"],
                        "max_windowed_proxy_correlation": thresholds.max_windowed_proxy_correlation,
                        "window_sec": thresholds.windowed_proxy_correlation_sec,
                        "min_windowed_proxy_rms_dbfs": thresholds.min_windowed_proxy_rms_dbfs,
                        "active_window_count": windowed_correlation["active_window_count"],
                        "top_window": top_window,
                    },
                }
            )
        max_possible_other_abs_correlation = max(
            abs(float(correlation)),
            float(windowed_correlation.get("max_active_abs_correlation", 0.0)),
        )
        if (
            proxy_stem == "other"
            and abs(float(correlation)) <= thresholds.max_proxy_correlation
            and float(windowed_correlation.get("max_active_abs_correlation", 0.0))
            <= thresholds.max_windowed_proxy_correlation
            and max_possible_other_abs_correlation > thresholds.max_possible_other_proxy_correlation
            and proxy_rms_ratio_db >= thresholds.min_possible_other_proxy_rms_ratio_db
        ):
            findings.append(
                {
                    "severity": "blocker",
                    "label": "possible_other_proxy_leakage",
                    "evidence": {
                        "proxy_stem": proxy_stem,
                        "max_possible_other_abs_correlation": round(float(max_possible_other_abs_correlation), 6),
                        "max_possible_other_proxy_correlation": thresholds.max_possible_other_proxy_correlation,
                        "output_rms_ratio_db": round(float(proxy_rms_ratio_db), 3),
                        "min_possible_other_proxy_rms_ratio_db": thresholds.min_possible_other_proxy_rms_ratio_db,
                        "full_track_abs_correlation": round(abs(float(correlation)), 6),
                        "windowed_max_active_abs_correlation": windowed_correlation["max_active_abs_correlation"],
                    },
                }
            )

    status = "pass" if not any(finding.get("severity") == "blocker" for finding in findings) else "rejected"
    return {
        "status": status,
        "mode": "long_tail_acoustic_quality_gate_v1",
        "target_stem": target_stem,
        "thresholds": {
            "min_duration_sec": thresholds.min_duration_sec,
            "max_duration_delta_sec": thresholds.max_duration_delta_sec,
            "min_rms_dbfs": thresholds.min_rms_dbfs,
            "min_source_rms_ratio_db": thresholds.min_source_rms_ratio_db,
            "min_active_presence_windows": thresholds.min_active_presence_windows,
            "min_active_presence_rms_dbfs": thresholds.min_active_presence_rms_dbfs,
            "active_presence_window_sec": thresholds.active_presence_window_sec,
            "max_proxy_correlation": thresholds.max_proxy_correlation,
            "max_windowed_proxy_correlation": thresholds.max_windowed_proxy_correlation,
            "windowed_proxy_correlation_sec": thresholds.windowed_proxy_correlation_sec,
            "min_windowed_proxy_rms_dbfs": thresholds.min_windowed_proxy_rms_dbfs,
            "max_possible_other_proxy_correlation": thresholds.max_possible_other_proxy_correlation,
            "min_possible_other_proxy_rms_ratio_db": thresholds.min_possible_other_proxy_rms_ratio_db,
        },
        "metrics": {
            "output": output_metrics,
            "active_presence": active_presence,
            "references": reference_reports,
            "proxy_references": proxy_reports,
        },
        "findings": findings,
        "objective_metrics": {
            "sdr": None,
            "si_sdr": None,
            "sir": None,
            "sar": None,
            "reason": "isolated_ground_truth_for_target_stem_not_available",
        },
    }


def _candidate_score(quality_gate: dict[str, Any]) -> float:
    metrics = quality_gate.get("metrics") if isinstance(quality_gate.get("metrics"), dict) else {}
    output_metrics = metrics.get("output") if isinstance(metrics.get("output"), dict) else {}
    references = metrics.get("references") if isinstance(metrics.get("references"), dict) else {}
    source_reference = references.get("source_audio") if isinstance(references.get("source_audio"), dict) else {}
    proxy_references = metrics.get("proxy_references") if isinstance(metrics.get("proxy_references"), dict) else {}

    rms_dbfs = float(output_metrics.get("rms_dbfs", -120.0))
    source_ratio_db = float(source_reference.get("output_rms_ratio_db", -120.0))
    max_proxy_abs_correlation = 0.0
    for proxy_report in proxy_references.values():
        if not isinstance(proxy_report, dict):
            continue
        windowed_correlation = proxy_report.get("windowed_correlation")
        max_proxy_abs_correlation = max(
            max_proxy_abs_correlation,
            float(proxy_report.get("abs_correlation", 0.0) or 0.0),
            float(windowed_correlation.get("max_active_abs_correlation", 0.0) or 0.0)
            if isinstance(windowed_correlation, dict)
            else 0.0,
        )

    return round((rms_dbfs + 120.0) * 0.35 + (source_ratio_db + 120.0) * 0.55 - max_proxy_abs_correlation * 20.0, 6)


def _build_candidate_record(
    *,
    target_stem: str,
    candidate_id: str,
    candidate_type: str,
    local_output_path: Path,
    source_audio_path: Path,
    build_from_paths: dict[str, str],
    quality_thresholds: LongTailQualityThresholds,
    production_eligible: bool,
    metadata: dict[str, Any] | None = None,
    notes: Any = None,
) -> dict[str, Any]:
    quality_gate = _evaluate_long_tail_quality(
        target_stem=target_stem,
        output_path=local_output_path,
        source_audio_path=source_audio_path,
        build_from_paths=build_from_paths,
        thresholds=quality_thresholds,
    )
    passed = quality_gate["status"] == "pass"
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "production_eligible": bool(production_eligible),
        "status": "accepted" if passed and production_eligible else ("diagnostic_pass" if passed else "rejected"),
        "local_output_path": str(local_output_path),
        "output_bytes": local_output_path.stat().st_size if local_output_path.exists() else 0,
        "metadata": metadata or {},
        "quality_gate": quality_gate,
        "score": _candidate_score(quality_gate),
        "notes": notes,
    }


def _select_long_tail_candidate(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_candidates = [
        candidate
        for candidate in candidate_records
        if candidate.get("production_eligible") is True and candidate.get("quality_gate", {}).get("status") == "pass"
    ]
    selected = max(accepted_candidates, key=lambda candidate: float(candidate.get("score", -1e9)), default=None)
    return {
        "mode": LONG_TAIL_ROUTER_MODE,
        "status": "accepted" if selected is not None else "rejected",
        "selected_candidate_id": selected.get("candidate_id") if selected else None,
        "selected_candidate_type": selected.get("candidate_type") if selected else None,
        "candidate_count": len(candidate_records),
        "accepted_candidate_count": len(accepted_candidates),
        "rejected_candidate_count": len([candidate for candidate in candidate_records if candidate.get("status") == "rejected"]),
        "diagnostic_candidate_count": len([candidate for candidate in candidate_records if candidate.get("production_eligible") is False]),
        "candidate_ranking": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_type": candidate.get("candidate_type"),
                "production_eligible": candidate.get("production_eligible"),
                "status": candidate.get("status"),
                "score": candidate.get("score"),
                "quality_status": candidate.get("quality_gate", {}).get("status"),
            }
            for candidate in sorted(candidate_records, key=lambda candidate: float(candidate.get("score", -1e9)), reverse=True)
        ],
    }


def _validate_querysep_result(
    *,
    target_stem: str,
    result: dict[str, Any],
    path_mappings: tuple[PathMapping, ...],
    min_output_bytes: int,
    source_audio_path: Path,
    build_from_paths: dict[str, str],
    quality_thresholds: LongTailQualityThresholds,
    raise_on_quality_reject: bool = True,
) -> dict[str, Any]:
    status = str(result.get("status") or "")
    if status != "completed":
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        error = str(metadata.get("error") or result.get("notes") or status)
        raise RuntimeError(f"querysep-worker did not complete {target_stem}: {error}")

    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    routing_mode = str(metadata.get("routing_mode") or "")
    if routing_mode != BACKEND_BANQUET_BYOQ:
        raise RuntimeError(
            f"querysep-worker returned non-Banquet route for {target_stem}: routing_mode={routing_mode}"
        )

    named_outputs = result.get("named_outputs")
    if not isinstance(named_outputs, dict) or target_stem not in named_outputs:
        raise RuntimeError(f"querysep-worker completed {target_stem} without named output")

    querysep_output_path = str(named_outputs[target_stem])
    local_output_path = Path(_translate_path(querysep_output_path, path_mappings, to_querysep=False))
    if not local_output_path.exists():
        raise FileNotFoundError(
            "querysep-worker output is not visible to ml-worker. "
            f"querysep_path={querysep_output_path}, local_path={local_output_path}"
        )
    output_bytes = local_output_path.stat().st_size
    if output_bytes < min_output_bytes:
        raise RuntimeError(
            f"querysep-worker output for {target_stem} is too small: {local_output_path} ({output_bytes} bytes)"
        )

    quality_gate = _evaluate_long_tail_quality(
        target_stem=target_stem,
        output_path=local_output_path,
        source_audio_path=source_audio_path,
        build_from_paths=build_from_paths,
        thresholds=quality_thresholds,
    )
    stem_result = {
        "target_stem": target_stem,
        "status": "accepted" if quality_gate["status"] == "pass" else "rejected",
        "backend": BACKEND_BANQUET_BYOQ,
        "routing_mode": routing_mode,
        "fallback_used": False,
        "querysep_output_path": querysep_output_path,
        "local_output_path": str(local_output_path),
        "output_bytes": output_bytes,
        "metadata": metadata,
        "quality_gate": quality_gate,
        "score": _candidate_score(quality_gate),
        "notes": result.get("notes"),
    }
    if raise_on_quality_reject and quality_gate["status"] != "pass":
        raise LongTailQualityGateError(target_stem, stem_result)
    return stem_result


def build_querysep_long_tail_stems(
    *,
    audio_path: str | Path,
    output_dir: str | Path,
    build_id: str,
    target_stems: Any = None,
    source_stem_manifest_path: str | Path | None = None,
    build_from_paths: Any = None,
    querysep_worker_url: str | None = None,
    querysep_input_path: str | None = None,
    querysep_output_root: str | None = None,
    path_mappings: Any = None,
    track_name: str | None = None,
    phase_id: str = DEFAULT_PHASE_ID,
    backend: str = BACKEND_BANQUET_BYOQ,
    dry_run: Any = False,
    request_timeout_seconds: int | None = None,
    min_output_bytes: int | None = None,
    min_quality_duration_sec: Any = None,
    max_duration_delta_sec: Any = None,
    min_quality_rms_dbfs: Any = None,
    min_source_rms_ratio_db: Any = None,
    min_active_presence_windows: Any = None,
    min_active_presence_rms_dbfs: Any = None,
    active_presence_window_sec: Any = None,
    max_proxy_correlation: Any = None,
    max_windowed_proxy_correlation: Any = None,
    windowed_proxy_correlation_sec: Any = None,
    min_windowed_proxy_rms_dbfs: Any = None,
    max_possible_other_proxy_correlation: Any = None,
    min_possible_other_proxy_rms_ratio_db: Any = None,
    long_tail_candidate_paths: Any = None,
) -> dict[str, Any]:
    service_url = _normalize_service_url(querysep_worker_url)
    normalized_backend = str(backend or "").strip().lower()
    if normalized_backend != BACKEND_BANQUET_BYOQ:
        raise ValueError("querysep_long_tail_build requires backend='banquet_byoq'; proxy/fallback backends are not allowed")
    if _parse_bool(dry_run, field_name="dry_run"):
        raise ValueError("querysep_long_tail_build requires dry_run=false")

    resolved_audio_path = Path(audio_path)
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Input audio is missing: {resolved_audio_path}")

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    normalized_build_id = str(build_id or "").strip()
    if normalized_build_id == "":
        raise ValueError("build_id is required")

    targets = _parse_target_stems(target_stems)
    mappings = _parse_path_mappings(path_mappings)
    timeout_seconds = int(request_timeout_seconds or os.getenv("QUERYSEP_REQUEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    minimum_output_bytes = int(min_output_bytes or os.getenv("QUERYSEP_MIN_OUTPUT_BYTES", DEFAULT_MIN_OUTPUT_BYTES))
    if timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if minimum_output_bytes <= 0:
        raise ValueError("min_output_bytes must be positive")
    quality_thresholds = _build_quality_thresholds(
        min_quality_duration_sec=min_quality_duration_sec,
        max_duration_delta_sec=max_duration_delta_sec,
        min_quality_rms_dbfs=min_quality_rms_dbfs,
        min_source_rms_ratio_db=min_source_rms_ratio_db,
        min_active_presence_windows=min_active_presence_windows,
        min_active_presence_rms_dbfs=min_active_presence_rms_dbfs,
        active_presence_window_sec=active_presence_window_sec,
        max_proxy_correlation=max_proxy_correlation,
        max_windowed_proxy_correlation=max_windowed_proxy_correlation,
        windowed_proxy_correlation_sec=windowed_proxy_correlation_sec,
        min_windowed_proxy_rms_dbfs=min_windowed_proxy_rms_dbfs,
        max_possible_other_proxy_correlation=max_possible_other_proxy_correlation,
        min_possible_other_proxy_rms_ratio_db=min_possible_other_proxy_rms_ratio_db,
    )

    local_build_from_paths = _resolve_build_from_paths(
        explicit_build_from_paths=build_from_paths,
        source_stem_manifest_path=source_stem_manifest_path,
    )
    service_build_from_paths = {
        stem_name: _translate_path(local_path, mappings, to_querysep=True)
        for stem_name, local_path in sorted(local_build_from_paths.items())
    }

    service_input_path = querysep_input_path or _translate_path(resolved_audio_path, mappings, to_querysep=True)
    service_output_root = querysep_output_root or _translate_path(resolved_output_dir, mappings, to_querysep=True)
    resolved_track_name = str(track_name or resolved_audio_path.stem)

    results: dict[str, Any] = {}
    raw_querysep_results: dict[str, Any] = {}
    request_summaries: dict[str, Any] = {}
    manifest_path = resolved_output_dir / "querysep_long_tail_manifest.json"

    def write_manifest(status: str, failure: dict[str, Any] | None = None) -> Path:
        manifest = {
            "task_type": TASK_TYPE,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_id": normalized_build_id,
            "track_name": resolved_track_name,
            "backend": BACKEND_BANQUET_BYOQ,
            "fallback_used": False,
            "dry_run": False,
            "phase_id": phase_id,
            "querysep_worker_url": service_url,
            "source_audio_path": str(resolved_audio_path),
            "querysep_input_path": service_input_path,
            "output_dir": str(resolved_output_dir),
            "querysep_output_root": service_output_root,
            "source_stem_manifest_path": str(source_stem_manifest_path) if source_stem_manifest_path else None,
            "build_from_paths": local_build_from_paths,
            "querysep_build_from_paths": service_build_from_paths,
            "target_stems": targets,
            "quality_gate": {
                "mode": "long_tail_acoustic_quality_gate_v1",
                "thresholds": {
                    "min_duration_sec": quality_thresholds.min_duration_sec,
                    "max_duration_delta_sec": quality_thresholds.max_duration_delta_sec,
                    "min_rms_dbfs": quality_thresholds.min_rms_dbfs,
                    "min_source_rms_ratio_db": quality_thresholds.min_source_rms_ratio_db,
                    "min_active_presence_windows": quality_thresholds.min_active_presence_windows,
                    "min_active_presence_rms_dbfs": quality_thresholds.min_active_presence_rms_dbfs,
                    "active_presence_window_sec": quality_thresholds.active_presence_window_sec,
                    "max_proxy_correlation": quality_thresholds.max_proxy_correlation,
                    "max_windowed_proxy_correlation": quality_thresholds.max_windowed_proxy_correlation,
                    "windowed_proxy_correlation_sec": quality_thresholds.windowed_proxy_correlation_sec,
                    "min_windowed_proxy_rms_dbfs": quality_thresholds.min_windowed_proxy_rms_dbfs,
                    "max_possible_other_proxy_correlation": quality_thresholds.max_possible_other_proxy_correlation,
                    "min_possible_other_proxy_rms_ratio_db": quality_thresholds.min_possible_other_proxy_rms_ratio_db,
                },
                "objective_metrics": {
                    "sdr": None,
                    "si_sdr": None,
                    "sir": None,
                    "sar": None,
                    "reason": "isolated_ground_truth_for_target_stem_not_available",
                },
            },
            "stems": results,
            "requests": request_summaries,
            "querysep_results": raw_querysep_results,
        }
        if failure is not None:
            manifest["failure"] = failure
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest_path

    for index, target_stem in enumerate(targets, start=1):
        request_payload = {
            "job_id": normalized_build_id,
            "phase_id": phase_id,
            "step_index": index,
            "track_name": resolved_track_name,
            "input_path": service_input_path,
            "output_root": service_output_root,
            "target_stem": target_stem,
            "task": "query_separation",
            "model": BACKEND_BANQUET_BYOQ,
            "prompt": target_stem,
            "build_from_paths": service_build_from_paths,
            "raw_step": {
                "backend": BACKEND_BANQUET_BYOQ,
                "prompt": target_stem,
                "target_stem": target_stem,
            },
            "dry_run": False,
        }
        result = _post_json(f"{service_url}/run", request_payload, timeout_seconds=timeout_seconds)
        raw_querysep_results[target_stem] = result
        request_summaries[target_stem] = request_payload
        primary_result = _validate_querysep_result(
            target_stem=target_stem,
            result=result,
            path_mappings=mappings,
            min_output_bytes=minimum_output_bytes,
            source_audio_path=resolved_audio_path,
            build_from_paths=local_build_from_paths,
            quality_thresholds=quality_thresholds,
            raise_on_quality_reject=False,
        )
        primary_candidate = {
            **primary_result,
            "candidate_id": "banquet_byoq_primary",
            "candidate_type": "banquet_byoq",
            "production_eligible": True,
        }

        candidate_records = [primary_candidate]
        if primary_result.get("quality_gate", {}).get("status") != "pass":
            harmonic_query_bus = _write_harmonic_residual_query_bus(
                target_stem=target_stem,
                output_dir=resolved_output_dir,
                source_audio_path=resolved_audio_path,
                build_from_paths=local_build_from_paths,
            )
            if harmonic_query_bus and harmonic_query_bus.get("status") == "materialized":
                harmonic_query_phase_id = f"{phase_id}_harmonic_residual_query"
                harmonic_query_build_from_paths = {
                    **local_build_from_paths,
                    "harmonic_residual": str(harmonic_query_bus["local_output_path"]),
                }
                service_harmonic_query_build_from_paths = {
                    stem_name: _translate_path(local_path, mappings, to_querysep=True)
                    for stem_name, local_path in sorted(harmonic_query_build_from_paths.items())
                }
                harmonic_request_payload = {
                    "job_id": normalized_build_id,
                    "phase_id": harmonic_query_phase_id,
                    "step_index": index,
                    "track_name": resolved_track_name,
                    "input_path": service_input_path,
                    "output_root": service_output_root,
                    "target_stem": target_stem,
                    "task": "query_separation",
                    "model": BACKEND_BANQUET_BYOQ,
                    "prompt": target_stem,
                    "build_from_paths": service_harmonic_query_build_from_paths,
                    "raw_step": {
                        "backend": BACKEND_BANQUET_BYOQ,
                        "prompt": target_stem,
                        "target_stem": target_stem,
                        "candidate_type": HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE,
                        "query_mode": HARMONIC_RESIDUAL_QUERY_MODE,
                    },
                    "dry_run": False,
                }
                request_summaries[f"{target_stem}:harmonic_residual_query"] = harmonic_request_payload
                harmonic_query_key = f"{target_stem}:harmonic_residual_query"
                try:
                    harmonic_query_result = _post_json(
                        f"{service_url}/run",
                        harmonic_request_payload,
                        timeout_seconds=timeout_seconds,
                    )
                    raw_querysep_results[harmonic_query_key] = harmonic_query_result
                    harmonic_validated_result = _validate_querysep_result(
                        target_stem=target_stem,
                        result=harmonic_query_result,
                        path_mappings=mappings,
                        min_output_bytes=minimum_output_bytes,
                        source_audio_path=resolved_audio_path,
                        build_from_paths=local_build_from_paths,
                        quality_thresholds=quality_thresholds,
                        raise_on_quality_reject=False,
                    )
                    candidate_records.append(
                        {
                            **harmonic_validated_result,
                            "candidate_id": "banquet_byoq_harmonic_residual_query",
                            "candidate_type": HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE,
                            "production_eligible": True,
                            "metadata": {
                                **(
                                    harmonic_validated_result.get("metadata")
                                    if isinstance(harmonic_validated_result.get("metadata"), dict)
                                    else {}
                                ),
                                "query_auxiliary_bus": harmonic_query_bus,
                            },
                        }
                    )
                    if _quality_gate_has_presence_failure(harmonic_validated_result.get("quality_gate", {})):
                        sparse_harmonic_candidate = _write_sparse_harmonic_residual_candidate(
                            target_stem=target_stem,
                            output_dir=resolved_output_dir,
                            harmonic_query_bus=harmonic_query_bus,
                        )
                        if sparse_harmonic_candidate and sparse_harmonic_candidate.get("status") == "materialized":
                            candidate_records.append(
                                _build_candidate_record(
                                    target_stem=target_stem,
                                    candidate_id=str(sparse_harmonic_candidate["candidate_id"]),
                                    candidate_type=str(sparse_harmonic_candidate["candidate_type"]),
                                    local_output_path=Path(str(sparse_harmonic_candidate["local_output_path"])),
                                    source_audio_path=resolved_audio_path,
                                    build_from_paths=local_build_from_paths,
                                    quality_thresholds=quality_thresholds,
                                    production_eligible=True,
                                    metadata={
                                        "source_candidate_id": "banquet_byoq_harmonic_residual_query",
                                        "source_candidate_type": HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE,
                                        "source_auxiliary_bus_id": sparse_harmonic_candidate.get("source_auxiliary_bus_id"),
                                        "source_auxiliary_path": sparse_harmonic_candidate.get("source_auxiliary_path"),
                                        "used_references": sparse_harmonic_candidate.get("used_references", []),
                                        "activation_reason": "harmonic_residual_banquet_presence_failure",
                                        "source_metrics": sparse_harmonic_candidate.get("metrics", {}),
                                    },
                                    notes=sparse_harmonic_candidate.get("notes"),
                                )
                            )
                        elif sparse_harmonic_candidate:
                            candidate_records.append(sparse_harmonic_candidate)
                except Exception as exc:  # noqa: BLE001
                    raw_querysep_results[harmonic_query_key] = {
                        "status": "failed",
                        "metadata": {
                            "error": str(exc),
                            "candidate_type": HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE,
                            "query_auxiliary_bus": harmonic_query_bus,
                        },
                    }
                    candidate_records.append(
                        {
                            "candidate_id": "banquet_byoq_harmonic_residual_query",
                            "candidate_type": HARMONIC_RESIDUAL_QUERY_CANDIDATE_TYPE,
                            "production_eligible": True,
                            "status": "rejected",
                            "local_output_path": None,
                            "output_bytes": 0,
                            "metadata": {
                                "error": str(exc),
                                "query_auxiliary_bus": harmonic_query_bus,
                            },
                            "quality_gate": {
                                "status": "rejected",
                                "mode": "long_tail_acoustic_quality_gate_v1",
                                "target_stem": target_stem,
                                "findings": [
                                    {
                                        "severity": "blocker",
                                        "label": "harmonic_residual_query_candidate_failed",
                                        "evidence": str(exc),
                                    }
                                ],
                                "metrics": {},
                                "objective_metrics": {
                                    "sdr": None,
                                    "si_sdr": None,
                                    "sir": None,
                                    "sar": None,
                                    "reason": "candidate_failed_before_quality_evaluation",
                                },
                            },
                            "score": _candidate_score({}),
                            "notes": "Harmonic-residual Banquet candidate failed before selection and was not used.",
                        }
                    )
            elif harmonic_query_bus:
                candidate_records.append(harmonic_query_bus)

        external_candidates = _resolve_external_candidate_paths(long_tail_candidate_paths, target_stem)
        for candidate_id, candidate_path in sorted(external_candidates.items()):
            candidate_records.append(
                _build_candidate_record(
                    target_stem=target_stem,
                    candidate_id=f"external_{candidate_id}",
                    candidate_type="external_model_candidate",
                    local_output_path=Path(candidate_path),
                    source_audio_path=resolved_audio_path,
                    build_from_paths=local_build_from_paths,
                    quality_thresholds=quality_thresholds,
                    production_eligible=True,
                    metadata={"provided_by": "long_tail_candidate_paths"},
                )
            )

        for source_candidate in list(candidate_records):
            decorrelated_candidate = _write_windowed_proxy_decorrelated_candidate(
                target_stem=target_stem,
                source_candidate=source_candidate,
                output_dir=resolved_output_dir,
                build_from_paths=local_build_from_paths,
                thresholds=quality_thresholds,
            )
            if not decorrelated_candidate:
                continue
            candidate_records.append(
                _build_candidate_record(
                    target_stem=target_stem,
                    candidate_id=str(decorrelated_candidate["candidate_id"]),
                    candidate_type=str(decorrelated_candidate["candidate_type"]),
                    local_output_path=Path(str(decorrelated_candidate["local_output_path"])),
                    source_audio_path=resolved_audio_path,
                    build_from_paths=local_build_from_paths,
                    quality_thresholds=quality_thresholds,
                    production_eligible=True,
                    metadata={
                        "source_candidate_id": decorrelated_candidate.get("source_candidate_id"),
                        "source_candidate_type": decorrelated_candidate.get("source_candidate_type"),
                        "repair_window_count": decorrelated_candidate.get("repair_window_count"),
                        "repaired_windows": decorrelated_candidate.get("repaired_windows", []),
                    },
                    notes=decorrelated_candidate.get("notes"),
                )
            )

        diagnostic_bus_sum = _write_diagnostic_bus_sum_candidate(
            target_stem=target_stem,
            output_dir=resolved_output_dir,
            build_from_paths=local_build_from_paths,
        )
        if diagnostic_bus_sum and diagnostic_bus_sum.get("status") == "materialized":
            candidate_records.append(
                _build_candidate_record(
                    target_stem=target_stem,
                    candidate_id=str(diagnostic_bus_sum["candidate_id"]),
                    candidate_type=str(diagnostic_bus_sum["candidate_type"]),
                    local_output_path=Path(str(diagnostic_bus_sum["local_output_path"])),
                    source_audio_path=resolved_audio_path,
                    build_from_paths=local_build_from_paths,
                    quality_thresholds=quality_thresholds,
                    production_eligible=False,
                    metadata={"source_paths": diagnostic_bus_sum.get("source_paths", {})},
                    notes="Diagnostic proxy material only; never selected as production long-tail stem.",
                )
            )
        elif diagnostic_bus_sum:
            candidate_records.append(diagnostic_bus_sum)

        router_decision = _select_long_tail_candidate(candidate_records)
        selected_candidate = next(
            (
                candidate
                for candidate in candidate_records
                if candidate.get("candidate_id") == router_decision.get("selected_candidate_id")
            ),
            None,
        )

        if selected_candidate is None:
            primary_result["router"] = router_decision
            primary_result["candidates"] = candidate_records
            results[target_stem] = primary_result
            rejected_manifest_path = write_manifest(
                "rejected",
                failure={
                    "target_stem": target_stem,
                    "reason": "long_tail_model_router_rejected_all_candidates",
                    "router": router_decision,
                    "quality_gate": primary_result.get("quality_gate"),
                },
            )
            primary_findings = primary_result.get("quality_gate", {}).get("findings", [])
            raise RuntimeError(
                f"long-tail model router rejected all production candidates for {target_stem}: "
                f"{primary_findings}; manifest_path={rejected_manifest_path}"
            )

        selected_result = {
            **primary_result,
            "status": "accepted",
            "selected_candidate_id": selected_candidate.get("candidate_id"),
            "selected_candidate_type": selected_candidate.get("candidate_type"),
            "selected_output_origin": selected_candidate.get("candidate_type"),
            "selected_output_path": selected_candidate.get("local_output_path"),
            "primary_querysep_output_path": primary_result.get("querysep_output_path"),
            "querysep_output_path": selected_candidate.get("querysep_output_path"),
            "local_output_path": selected_candidate.get("local_output_path"),
            "output_bytes": selected_candidate.get("output_bytes"),
            "quality_gate": selected_candidate.get("quality_gate"),
            "score": selected_candidate.get("score"),
            "router": router_decision,
            "candidates": candidate_records,
        }
        results[target_stem] = selected_result

    write_manifest("completed")

    return {
        "task_type": TASK_TYPE,
        "status": "completed",
        "build_id": normalized_build_id,
        "backend": BACKEND_BANQUET_BYOQ,
        "fallback_used": False,
        "output_dir": str(resolved_output_dir),
        "manifest_path": str(manifest_path),
        "target_stems": targets,
        "stems": results,
    }
