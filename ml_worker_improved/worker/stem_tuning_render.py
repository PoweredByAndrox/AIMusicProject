from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy import signal

from . import high_quality_stem_export as high_quality_stem_export_module
from .high_quality_stem_export import (
    FINAL_STEM_EXPORT_SUBTYPE,
    SEPARATION_SAMPLE_RATE,
    _limit_peak,
    _safe_audio_rms,
    _split_harmonic_percussive,
    export_aggressive_melody_first_from_stem_dir,
    export_vocal_preview_from_audio,
    export_track_stems_aggressive_melody_first,
    export_track_stems_full_res,
)


def _db_to_gain(db: float) -> float:
    return float(np.power(10.0, float(db) / 20.0))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _lerp(start: float, end: float, amount: float) -> float:
    t = _clamp(float(amount), 0.0, 1.0)
    return float(start) + (float(end) - float(start)) * t


def _focus_mask(
    frames: int,
    sample_rate: int,
    focus_window: dict[str, Any],
) -> np.ndarray:
    if frames <= 0 or not bool(focus_window.get("enabled", False)):
        return np.zeros((frames,), dtype=np.float32)

    start_sec = max(0.0, float(focus_window.get("startSec", 0.0)))
    end_sec = max(start_sec, float(focus_window.get("endSec", start_sec)))
    xfade_sec = max(0.0, float(focus_window.get("xfadeSec", 0.0)))
    times = np.arange(frames, dtype=np.float32) / float(sample_rate)

    if xfade_sec <= 0:
        return np.where((times >= start_sec) & (times <= end_sec), 1.0, 0.0).astype(np.float32)

    mask = np.zeros((frames,), dtype=np.float32)
    fade_in_start = start_sec - xfade_sec
    fade_out_end = end_sec + xfade_sec

    inside = (times >= start_sec) & (times <= end_sec)
    fade_in = (times >= fade_in_start) & (times < start_sec)
    fade_out = (times > end_sec) & (times <= fade_out_end)

    mask[inside] = 1.0
    mask[fade_in] = np.clip((times[fade_in] - fade_in_start) / xfade_sec, 0.0, 1.0)
    mask[fade_out] = np.clip(1.0 - ((times[fade_out] - end_sec) / xfade_sec), 0.0, 1.0)
    return mask.astype(np.float32)


def _apply_filter(audio: np.ndarray, *, low_hz: float | None = None, high_hz: float | None = None) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    nyquist = SEPARATION_SAMPLE_RATE * 0.5
    if low_hz is not None and high_hz is not None:
        lo = _clamp(float(low_hz) / nyquist, 1e-5, 0.999)
        hi = _clamp(float(high_hz) / nyquist, lo + 1e-5, 0.999)
        sos = signal.butter(4, [lo, hi], btype="bandpass", output="sos")
    elif low_hz is not None:
        cutoff = _clamp(float(low_hz) / nyquist, 1e-5, 0.999)
        sos = signal.butter(4, cutoff, btype="highpass", output="sos")
    elif high_hz is not None:
        cutoff = _clamp(float(high_hz) / nyquist, 1e-5, 0.999)
        sos = signal.butter(4, cutoff, btype="lowpass", output="sos")
    else:
        return audio.astype(np.float32, copy=False)

    filtered = np.zeros_like(audio, dtype=np.float32)
    for channel in range(audio.shape[1]):
        filtered[:, channel] = signal.sosfiltfilt(sos, audio[:, channel]).astype(np.float32)
    return filtered


def _stem_summary(audio: np.ndarray) -> dict[str, Any]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "frames": int(audio.shape[0]),
        "duration_sec": round(float(audio.shape[0] / SEPARATION_SAMPLE_RATE), 6),
        "rms": round(_safe_audio_rms(audio), 6),
        "peak": round(peak, 6),
    }


def _normalize_processor_config(candidate: dict[str, Any] | None) -> dict[str, float | bool]:
    raw = candidate if isinstance(candidate, dict) else {}
    body_low = _clamp(float(raw.get("bodyLowHz", 90.0)), 40.0, 400.0)
    bleed_low = _clamp(float(raw.get("bleedLowHz", 700.0)), 120.0, 2500.0)

    return {
        "bypass": bool(raw.get("bypass", False)),
        "harmonicMargin": _clamp(float(raw.get("harmonicMargin", 1.0)), 0.5, 4.0),
        "percussiveMargin": _clamp(float(raw.get("percussiveMargin", 4.0)), 0.5, 6.0),
        "transientCutoffHz": _clamp(float(raw.get("transientCutoffHz", 2500.0)), 800.0, 8000.0),
        "bodyLowHz": body_low,
        "bodyHighHz": max(body_low + 20.0, _clamp(float(raw.get("bodyHighHz", 320.0)), body_low + 20.0, 1200.0)),
        "bleedLowHz": bleed_low,
        "bleedHighHz": max(bleed_low + 100.0, _clamp(float(raw.get("bleedHighHz", 3500.0)), bleed_low + 100.0, 9000.0)),
    }


def _normalize_hybrid_build_config(candidate: dict[str, Any] | None) -> dict[str, Any]:
    raw = candidate if isinstance(candidate, dict) else {}
    model_name = str(raw.get("modelName", "hybrid_htdemucs_ft_6s"))
    if model_name not in {"hybrid_htdemucs_ft_6s", "hybrid_htdemucs_ft_bs_roformer_6s"}:
        model_name = "hybrid_htdemucs_ft_6s"

    return {
        "modelName": model_name,
        "sharpenBleed": bool(raw.get("sharpenBleed", True)),
        "includeMelodyStem": bool(raw.get("includeMelodyStem", False)),
        "aggressiveMelodyFirst": bool(raw.get("aggressiveMelodyFirst", False)),
    }


def _normalize_stem_rewrite_config(candidate: dict[str, Any] | None) -> dict[str, Any]:
    raw = candidate if isinstance(candidate, dict) else {}
    return {
        "otherMove": _clamp(float(raw.get("otherMove", 0.45)), 0.0, 1.5),
        "guitarMove": _clamp(float(raw.get("guitarMove", 1.0)), 0.0, 1.5),
        "pianoMove": _clamp(float(raw.get("pianoMove", 1.0)), 0.0, 1.5),
        "drumsMaskedMove": _clamp(float(raw.get("drumsMaskedMove", 0.12)), 0.0, 1.0),
        "targetMelodyRms": _clamp(float(raw.get("targetMelodyRms", 0.026)), 0.001, 0.1),
        "maxMelodyGain": _clamp(float(raw.get("maxMelodyGain", 1.45)), 1.0, 3.0),
        "stageDrumHarmonicMove": _clamp(float(raw.get("stageDrumHarmonicMove", 0.42)), 0.0, 1.0),
        "stageHarmonicStemPercussiveMove": _clamp(float(raw.get("stageHarmonicStemPercussiveMove", 0.22)), 0.0, 1.0),
        "stageCleanupDrumHarmonicMove": _clamp(float(raw.get("stageCleanupDrumHarmonicMove", 0.05)), 0.0, 1.0),
        "stageCleanupHarmonicStemPercussiveMove": _clamp(float(raw.get("stageCleanupHarmonicStemPercussiveMove", 0.28)), 0.0, 1.0),
        "harmonicCount": int(max(1, min(8, int(raw.get("harmonicCount", 3))))),
        "confidenceFloor": _clamp(float(raw.get("confidenceFloor", 0.3)), 0.0, 1.0),
    }


def _normalize_vocals_rewrite_config(candidate: dict[str, Any] | None) -> dict[str, Any]:
    raw = candidate if isinstance(candidate, dict) else {}
    task_mode = str(raw.get("taskMode", "custom"))
    if task_mode not in {"custom", "clean_vocals", "minus_from_vocals", "artifact_repair", "aggressive_vocal_suppression", "detail_preserve"}:
        task_mode = "custom"
    candidate_mode = str(raw.get("candidateMode", "hybrid_vocals"))
    if candidate_mode not in {"ft_vocals", "bs_vocals", "hybrid_vocals"}:
        candidate_mode = "hybrid_vocals"

    return {
        "taskMode": task_mode,
        "vocalIsolationIntensity": _clamp(float(raw.get("vocalIsolationIntensity", 0.0)), 0.0, 1.0),
        "bleedRemovalIntensity": _clamp(float(raw.get("bleedRemovalIntensity", 0.0)), 0.0, 1.0),
        "airRestore": _clamp(float(raw.get("airRestore", 0.0)), 0.0, 1.0),
        "bodyPreserve": _clamp(float(raw.get("bodyPreserve", 0.0)), 0.0, 1.0),
        "artifactSuppression": _clamp(float(raw.get("artifactSuppression", 0.0)), 0.0, 1.0),
        "centerVocalSuppression": _clamp(float(raw.get("centerVocalSuppression", 0.0)), 0.0, 1.0),
        "candidateMode": candidate_mode,
        "lowBsWeight": _clamp(float(raw.get("lowBsWeight", 0.75)), 0.0, 1.0),
        "highBsWeight": _clamp(float(raw.get("highBsWeight", 0.25)), 0.0, 1.0),
        "inactiveLowBsWeight": _clamp(float(raw.get("inactiveLowBsWeight", 0.97)), 0.0, 1.0),
        "inactiveHighBsWeight": _clamp(float(raw.get("inactiveHighBsWeight", 0.9)), 0.0, 1.0),
        "crossoverHz": _clamp(float(raw.get("crossoverHz", 3000.0)), 200.0, 12000.0),
        "activityAbsFloorRatio": _clamp(float(raw.get("activityAbsFloorRatio", 0.08)), 0.0, 1.0),
        "activityRelFloor": _clamp(float(raw.get("activityRelFloor", 0.35)), 0.0, 1.0),
        "activitySmoothingFrames": int(max(1, min(101, int(raw.get("activitySmoothingFrames", 11))))),
        "vocalSharpenAlpha": _clamp(float(raw.get("vocalSharpenAlpha", 0.0)), 0.0, 4.0),
        "tailGateStrength": _clamp(float(raw.get("tailGateStrength", 0.0)), 0.0, 1.0),
        "tailGateFloorRatio": _clamp(float(raw.get("tailGateFloorRatio", 0.08)), 0.0, 1.0),
        "tailGateSmoothingFrames": int(max(1, min(101, int(raw.get("tailGateSmoothingFrames", 11))))),
        "subtractiveStrength": _clamp(float(raw.get("subtractiveStrength", 0.0)), 0.0, 4.0),
        "subtractivePower": _clamp(float(raw.get("subtractivePower", 1.0)), 0.1, 8.0),
        "subtractiveResidualFloor": _clamp(float(raw.get("subtractiveResidualFloor", 0.12)), 0.0, 1.0),
        "subtractiveActiveRestore": _clamp(float(raw.get("subtractiveActiveRestore", 0.0)), 0.0, 1.0),
        "subtractiveTargetRmsRatio": _clamp(float(raw.get("subtractiveTargetRmsRatio", 0.0)), 0.0, 4.0),
        "subtractiveMaxGain": _clamp(float(raw.get("subtractiveMaxGain", 1.35)), 1.0, 4.0),
        "hardMaskThreshold": _clamp(float(raw.get("hardMaskThreshold", 0.0)), 0.0, 1.0),
        "hardMaskPower": _clamp(float(raw.get("hardMaskPower", 3.0)), 1.0, 12.0),
        "hardMaskResidualFloor": _clamp(float(raw.get("hardMaskResidualFloor", 0.02)), 0.0, 1.0),
    }


def _vocals_rewrite_mode_seed(task_mode: str) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {
        "custom": {
            "taskMode": "custom",
            "vocalIsolationIntensity": 0.0,
            "bleedRemovalIntensity": 0.0,
            "airRestore": 0.0,
            "bodyPreserve": 0.0,
            "artifactSuppression": 0.0,
            "centerVocalSuppression": 0.0,
            "candidateMode": "hybrid_vocals",
            "lowBsWeight": 0.75,
            "highBsWeight": 0.25,
            "inactiveLowBsWeight": 0.97,
            "inactiveHighBsWeight": 0.9,
            "crossoverHz": 3000.0,
            "activityAbsFloorRatio": 0.08,
            "activityRelFloor": 0.35,
            "activitySmoothingFrames": 11,
            "vocalSharpenAlpha": 0.0,
            "tailGateStrength": 0.0,
            "tailGateFloorRatio": 0.08,
            "tailGateSmoothingFrames": 11,
            "subtractiveStrength": 0.0,
            "subtractivePower": 1.0,
            "subtractiveResidualFloor": 0.12,
            "subtractiveActiveRestore": 0.0,
            "subtractiveTargetRmsRatio": 0.0,
            "subtractiveMaxGain": 1.35,
            "hardMaskThreshold": 0.0,
            "hardMaskPower": 3.0,
            "hardMaskResidualFloor": 0.02,
        },
        "clean_vocals": {
            "taskMode": "clean_vocals",
            "vocalIsolationIntensity": 0.58,
            "bleedRemovalIntensity": 0.52,
            "airRestore": 0.46,
            "bodyPreserve": 0.42,
            "artifactSuppression": 0.38,
            "centerVocalSuppression": 0.42,
            "candidateMode": "hybrid_vocals",
            "lowBsWeight": 0.82,
            "highBsWeight": 0.34,
            "inactiveLowBsWeight": 0.98,
            "inactiveHighBsWeight": 0.93,
            "crossoverHz": 3200.0,
            "activityAbsFloorRatio": 0.06,
            "activityRelFloor": 0.3,
            "activitySmoothingFrames": 9,
            "vocalSharpenAlpha": 0.08,
            "tailGateStrength": 0.12,
            "tailGateFloorRatio": 0.06,
            "tailGateSmoothingFrames": 9,
            "subtractiveStrength": 0.22,
            "subtractivePower": 1.18,
            "subtractiveResidualFloor": 0.14,
            "subtractiveActiveRestore": 0.04,
            "subtractiveTargetRmsRatio": 0.98,
            "subtractiveMaxGain": 1.28,
            "hardMaskThreshold": 0.0,
            "hardMaskPower": 3.0,
            "hardMaskResidualFloor": 0.02,
        },
        "minus_from_vocals": {
            "taskMode": "minus_from_vocals",
            "vocalIsolationIntensity": 0.9,
            "bleedRemovalIntensity": 0.85,
            "airRestore": 0.2,
            "bodyPreserve": 0.2,
            "artifactSuppression": 0.62,
            "centerVocalSuppression": 0.92,
            "candidateMode": "hybrid_vocals",
            "lowBsWeight": 0.88,
            "highBsWeight": 0.42,
            "inactiveLowBsWeight": 0.99,
            "inactiveHighBsWeight": 0.96,
            "crossoverHz": 3400.0,
            "activityAbsFloorRatio": 0.04,
            "activityRelFloor": 0.24,
            "activitySmoothingFrames": 7,
            "vocalSharpenAlpha": 0.04,
            "tailGateStrength": 0.22,
            "tailGateFloorRatio": 0.05,
            "tailGateSmoothingFrames": 7,
            "subtractiveStrength": 0.42,
            "subtractivePower": 1.35,
            "subtractiveResidualFloor": 0.08,
            "subtractiveActiveRestore": 0.01,
            "subtractiveTargetRmsRatio": 0.85,
            "subtractiveMaxGain": 1.12,
            "hardMaskThreshold": 0.0,
            "hardMaskPower": 3.0,
            "hardMaskResidualFloor": 0.02,
        },
        "artifact_repair": {
            "taskMode": "artifact_repair",
            "vocalIsolationIntensity": 0.3,
            "bleedRemovalIntensity": 0.38,
            "airRestore": 0.34,
            "bodyPreserve": 0.72,
            "artifactSuppression": 0.86,
            "centerVocalSuppression": 0.18,
            "candidateMode": "ft_vocals",
            "lowBsWeight": 0.58,
            "highBsWeight": 0.18,
            "inactiveLowBsWeight": 0.88,
            "inactiveHighBsWeight": 0.78,
            "crossoverHz": 2800.0,
            "activityAbsFloorRatio": 0.09,
            "activityRelFloor": 0.42,
            "activitySmoothingFrames": 15,
            "vocalSharpenAlpha": 0.0,
            "tailGateStrength": 0.05,
            "tailGateFloorRatio": 0.08,
            "tailGateSmoothingFrames": 15,
            "subtractiveStrength": 0.1,
            "subtractivePower": 1.05,
            "subtractiveResidualFloor": 0.2,
            "subtractiveActiveRestore": 0.08,
            "subtractiveTargetRmsRatio": 1.05,
            "subtractiveMaxGain": 1.4,
            "hardMaskThreshold": 0.0,
            "hardMaskPower": 3.0,
            "hardMaskResidualFloor": 0.02,
        },
        "aggressive_vocal_suppression": {
            "taskMode": "aggressive_vocal_suppression",
            "vocalIsolationIntensity": 0.96,
            "bleedRemovalIntensity": 0.96,
            "airRestore": 0.12,
            "bodyPreserve": 0.1,
            "artifactSuppression": 0.72,
            "centerVocalSuppression": 0.98,
            "candidateMode": "bs_vocals",
            "lowBsWeight": 0.95,
            "highBsWeight": 0.62,
            "inactiveLowBsWeight": 1.0,
            "inactiveHighBsWeight": 0.98,
            "crossoverHz": 3600.0,
            "activityAbsFloorRatio": 0.03,
            "activityRelFloor": 0.18,
            "activitySmoothingFrames": 5,
            "vocalSharpenAlpha": 0.02,
            "tailGateStrength": 0.18,
            "tailGateFloorRatio": 0.04,
            "tailGateSmoothingFrames": 5,
            "subtractiveStrength": 0.0,
            "subtractivePower": 1.0,
            "subtractiveResidualFloor": 0.12,
            "subtractiveActiveRestore": 0.0,
            "subtractiveTargetRmsRatio": 0.0,
            "subtractiveMaxGain": 1.2,
            "hardMaskThreshold": 0.58,
            "hardMaskPower": 4.8,
            "hardMaskResidualFloor": 0.015,
        },
        "detail_preserve": {
            "taskMode": "detail_preserve",
            "vocalIsolationIntensity": 0.34,
            "bleedRemovalIntensity": 0.18,
            "airRestore": 0.82,
            "bodyPreserve": 0.86,
            "artifactSuppression": 0.22,
            "centerVocalSuppression": 0.16,
            "candidateMode": "hybrid_vocals",
            "lowBsWeight": 0.68,
            "highBsWeight": 0.24,
            "inactiveLowBsWeight": 0.94,
            "inactiveHighBsWeight": 0.88,
            "crossoverHz": 2600.0,
            "activityAbsFloorRatio": 0.08,
            "activityRelFloor": 0.36,
            "activitySmoothingFrames": 13,
            "vocalSharpenAlpha": 0.14,
            "tailGateStrength": 0.0,
            "tailGateFloorRatio": 0.08,
            "tailGateSmoothingFrames": 13,
            "subtractiveStrength": 0.08,
            "subtractivePower": 1.0,
            "subtractiveResidualFloor": 0.18,
            "subtractiveActiveRestore": 0.12,
            "subtractiveTargetRmsRatio": 1.08,
            "subtractiveMaxGain": 1.55,
            "hardMaskThreshold": 0.0,
            "hardMaskPower": 3.0,
            "hardMaskResidualFloor": 0.02,
        },
    }
    return dict(profiles.get(task_mode, profiles["custom"]))


def _resolved_vocals_rewrite_cleanup_family(config: dict[str, Any], preferred: str | None = None) -> str:
    if preferred == "hard_mask":
        return "hard_mask"
    if preferred == "subtractive":
        return "subtractive"
    if float(config.get("hardMaskThreshold", 0.0)) > 1e-6 and float(config.get("subtractiveStrength", 0.0)) <= 1e-6:
        return "hard_mask"
    if float(config.get("subtractiveStrength", 0.0)) > 1e-6:
        return "subtractive"
    if float(config.get("hardMaskThreshold", 0.0)) > 1e-6:
        return "hard_mask"
    return "none"


def _resolve_vocals_rewrite_runtime(candidate: dict[str, Any] | None) -> dict[str, Any]:
    raw = _normalize_vocals_rewrite_config(candidate)
    custom_seed = _vocals_rewrite_mode_seed("custom")
    mode_seed = _vocals_rewrite_mode_seed(str(raw.get("taskMode", "custom")))

    effective = dict(mode_seed)
    effective["taskMode"] = str(raw.get("taskMode", "custom"))
    for key, raw_value in raw.items():
        if key == "taskMode":
            continue
        custom_value = custom_seed.get(key)
        if isinstance(raw_value, float):
            if abs(float(raw_value) - float(custom_value)) > 1e-6:
                effective[key] = raw_value
        elif raw_value != custom_value:
            effective[key] = raw_value

    isolation = float(effective["vocalIsolationIntensity"])
    bleed = float(effective["bleedRemovalIntensity"])
    air = float(effective["airRestore"])
    body = float(effective["bodyPreserve"])
    artifact = float(effective["artifactSuppression"])
    center = float(effective["centerVocalSuppression"])

    resolved = dict(effective)
    resolved["lowBsWeight"] = float(resolved["lowBsWeight"]) + 0.12 * isolation + 0.08 * center - 0.08 * artifact - 0.1 * body
    resolved["highBsWeight"] = float(resolved["highBsWeight"]) + 0.18 * isolation + 0.16 * air + 0.1 * center - 0.12 * artifact
    resolved["inactiveLowBsWeight"] = float(resolved["inactiveLowBsWeight"]) + 0.03 * isolation + 0.02 * center - 0.08 * artifact - 0.06 * body
    resolved["inactiveHighBsWeight"] = float(resolved["inactiveHighBsWeight"]) + 0.05 * isolation + 0.12 * air + 0.02 * center - 0.1 * artifact
    resolved["crossoverHz"] = float(resolved["crossoverHz"]) + 900.0 * body - 700.0 * air + 300.0 * center
    resolved["activityAbsFloorRatio"] = float(resolved["activityAbsFloorRatio"]) - 0.03 * isolation - 0.03 * center + 0.05 * artifact
    resolved["activityRelFloor"] = float(resolved["activityRelFloor"]) - 0.12 * isolation - 0.1 * center + 0.12 * artifact
    resolved["activitySmoothingFrames"] = int(round(float(resolved["activitySmoothingFrames"]) + 12.0 * artifact - 4.0 * isolation))
    resolved["vocalSharpenAlpha"] = float(resolved["vocalSharpenAlpha"]) + 0.4 * air - 0.15 * artifact + 0.08 * isolation
    resolved["tailGateStrength"] = float(resolved["tailGateStrength"]) + 0.28 * bleed + 0.18 * artifact - 0.1 * air - 0.08 * body
    resolved["tailGateFloorRatio"] = float(resolved["tailGateFloorRatio"]) + 0.02 * artifact - 0.02 * air
    resolved["tailGateSmoothingFrames"] = int(round(float(resolved["tailGateSmoothingFrames"]) + 12.0 * artifact))

    preferred_cleanup_family = None
    if effective["taskMode"] == "aggressive_vocal_suppression":
        preferred_cleanup_family = "hard_mask"
    elif effective["taskMode"] != "custom":
        preferred_cleanup_family = "subtractive"

    if preferred_cleanup_family == "hard_mask":
        resolved["hardMaskThreshold"] = float(resolved["hardMaskThreshold"]) + 0.22 * bleed + 0.15 * center + 0.08 * isolation - 0.12 * body
        resolved["hardMaskPower"] = float(resolved["hardMaskPower"]) + 2.2 * bleed + 1.4 * artifact + 0.8 * center
        resolved["hardMaskResidualFloor"] = float(resolved["hardMaskResidualFloor"]) + 0.05 * artifact - 0.08 * bleed
        resolved["subtractiveStrength"] = 0.0
    else:
        resolved["subtractiveStrength"] = float(resolved["subtractiveStrength"]) + 0.85 * bleed + 0.35 * isolation + 0.25 * center - 0.22 * body - 0.18 * air
        resolved["subtractivePower"] = float(resolved["subtractivePower"]) + 0.5 * bleed + 0.3 * artifact
        resolved["subtractiveResidualFloor"] = float(resolved["subtractiveResidualFloor"]) + 0.1 * artifact - 0.12 * bleed
        resolved["subtractiveActiveRestore"] = float(resolved["subtractiveActiveRestore"]) + 0.35 * body + 0.2 * air
        resolved["subtractiveTargetRmsRatio"] = float(resolved["subtractiveTargetRmsRatio"]) + 0.65 * body + 0.35 * air - 0.2 * bleed
        resolved["subtractiveMaxGain"] = float(resolved["subtractiveMaxGain"]) + 0.8 * body + 0.3 * air
        resolved["hardMaskThreshold"] = 0.0

    resolved = _normalize_vocals_rewrite_config(resolved)
    cleanup_family = _resolved_vocals_rewrite_cleanup_family(resolved, preferred_cleanup_family)
    if cleanup_family == "subtractive":
        resolved["hardMaskThreshold"] = 0.0
    elif cleanup_family == "hard_mask":
        resolved["subtractiveStrength"] = 0.0
    else:
        resolved["subtractiveStrength"] = 0.0
        resolved["hardMaskThreshold"] = 0.0
    resolved = _normalize_vocals_rewrite_config(resolved)

    postprocess = {
        "airBoostDb": _clamp(6.0 * air - 1.2 * artifact, 0.0, 7.5),
        "bodyBoostDb": _clamp(5.0 * body - 1.0 * bleed, 0.0, 6.0),
        "artifactDampAmount": _clamp(0.75 * artifact + 0.2 * bleed, 0.0, 1.0),
        "sideAttenuation": _clamp(0.92 * center, 0.0, 0.95),
    }

    return {
        "raw": raw,
        "effectiveMacroConfig": {
            "taskMode": resolved["taskMode"],
            "vocalIsolationIntensity": resolved["vocalIsolationIntensity"],
            "bleedRemovalIntensity": resolved["bleedRemovalIntensity"],
            "airRestore": resolved["airRestore"],
            "bodyPreserve": resolved["bodyPreserve"],
            "artifactSuppression": resolved["artifactSuppression"],
            "centerVocalSuppression": resolved["centerVocalSuppression"],
        },
        "resolvedMicroConfig": {
            key: value
            for key, value in resolved.items()
            if key not in {
                "taskMode",
                "vocalIsolationIntensity",
                "bleedRemovalIntensity",
                "airRestore",
                "bodyPreserve",
                "artifactSuppression",
                "centerVocalSuppression",
            }
        },
        "cleanupFamily": cleanup_family,
        "postprocess": postprocess,
    }


def _apply_vocals_macro_postprocess(vocals_audio: np.ndarray, postprocess: dict[str, Any]) -> np.ndarray:
    processed = np.asarray(vocals_audio, dtype=np.float32)
    if processed.size == 0:
        return processed

    air_boost_db = float(postprocess.get("airBoostDb", 0.0))
    if air_boost_db > 1e-6:
        air_band = _apply_filter(processed, low_hz=4500.0, high_hz=12000.0)
        processed = processed + air_band * (_db_to_gain(air_boost_db) - 1.0)

    body_boost_db = float(postprocess.get("bodyBoostDb", 0.0))
    if body_boost_db > 1e-6:
        body_band = _apply_filter(processed, low_hz=120.0, high_hz=420.0)
        processed = processed + body_band * (_db_to_gain(body_boost_db) - 1.0)

    artifact_damp_amount = float(postprocess.get("artifactDampAmount", 0.0))
    if artifact_damp_amount > 1e-6:
        harsh_band = _apply_filter(processed, low_hz=2600.0, high_hz=9500.0)
        processed = processed - harsh_band * (0.65 * artifact_damp_amount)

    side_attenuation = float(postprocess.get("sideAttenuation", 0.0))
    if side_attenuation > 1e-6 and processed.ndim == 2 and processed.shape[1] == 2:
        mid = (processed[:, 0] + processed[:, 1]) * 0.5
        side = (processed[:, 0] - processed[:, 1]) * 0.5
        side *= max(0.0, 1.0 - side_attenuation)
        processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    return _limit_peak(processed.astype(np.float32))


def _normalize_unsafe_model_internals(candidate: dict[str, Any] | None) -> dict[str, Any]:
    raw = candidate if isinstance(candidate, dict) else {}
    hybrid_raw = raw.get("hybrid", {}) if isinstance(raw.get("hybrid"), dict) else {}
    bs_raw = raw.get("bsRoformer", {}) if isinstance(raw.get("bsRoformer"), dict) else {}

    return {
        "hybrid": {
            "requestedDevice": str(hybrid_raw.get("requestedDevice", "auto")).strip() or "auto",
            "stage1Model": str(hybrid_raw.get("stage1Model", "htdemucs_ft")).strip() or "htdemucs_ft",
            "stage2Model": str(hybrid_raw.get("stage2Model", "htdemucs_6s")).strip() or "htdemucs_6s",
            "stage1RequestedDevice": str(hybrid_raw.get("stage1RequestedDevice", hybrid_raw.get("requestedDevice", "auto"))).strip() or "auto",
            "stage2RequestedDevice": str(hybrid_raw.get("stage2RequestedDevice", hybrid_raw.get("requestedDevice", "auto"))).strip() or "auto",
            "stage1Split": bool(hybrid_raw.get("stage1Split", True)),
            "stage1Progress": bool(hybrid_raw.get("stage1Progress", True)),
            "stage1Shifts": int(max(1, min(8, int(hybrid_raw.get("stage1Shifts", 1))))),
            "stage1Overlap": _clamp(float(hybrid_raw.get("stage1Overlap", 0.25)), 0.0, 0.99),
            "stage1Segment": _clamp(float(hybrid_raw.get("stage1Segment", 0.0)), 0.0, 600.0),
            "stage2Split": bool(hybrid_raw.get("stage2Split", True)),
            "stage2Progress": bool(hybrid_raw.get("stage2Progress", True)),
            "stage2Shifts": int(max(1, min(8, int(hybrid_raw.get("stage2Shifts", 1))))),
            "stage2Overlap": _clamp(float(hybrid_raw.get("stage2Overlap", 0.25)), 0.0, 0.99),
            "stage2Segment": _clamp(float(hybrid_raw.get("stage2Segment", 0.0)), 0.0, 600.0),
        },
        "bsRoformer": {
            "requestedDevice": str(bs_raw.get("requestedDevice", "auto")).strip() or "auto",
            "deviceIds": [
                str(value).strip()
                for value in (bs_raw.get("deviceIds", ["0"]) if isinstance(bs_raw.get("deviceIds", ["0"]), list) else ["0"])
                if str(value).strip()
            ],
            "forceCpu": bool(bs_raw.get("forceCpu", False)),
            "disableDetailedPbar": bool(bs_raw.get("disableDetailedPbar", False)),
            "repoPath": str(bs_raw.get("repoPath", "/tmp/msst_zfturbo")).strip() or "/tmp/msst_zfturbo",
            "configPath": str(bs_raw.get("configPath", "/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml")).strip() or "/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml",
            "checkpointPath": str(bs_raw.get("checkpointPath", "/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt")).strip() or "/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt",
            "useTta": bool(bs_raw.get("useTta", True)),
            "extractInstrumental": bool(bs_raw.get("extractInstrumental", True)),
            "filenameTemplate": str(bs_raw.get("filenameTemplate", "{instr}")).strip() or "{instr}",
            "configOverrideYaml": str(bs_raw.get("configOverrideYaml", "")),
            "extraCliArgs": [
                str(value).strip()
                for value in (bs_raw.get("extraCliArgs", []) if isinstance(bs_raw.get("extraCliArgs", []), list) else [])
                if str(value).strip()
            ],
        },
    }


def _humanize_stem_key(stem_key: str) -> str:
    return stem_key.replace("_", " ").replace("-", " ").title()


def _load_source_manifest(source_stem_manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(source_stem_manifest_path).read_text(encoding="utf-8"))


def _stem_dir_from_manifest(source_manifest: dict[str, Any], source_stem_manifest_path: str | Path) -> Path:
    stem_paths: list[str] = []
    stems_payload = source_manifest.get("stems", {})
    if isinstance(stems_payload, dict):
        for stem_payload in stems_payload.values():
            if not isinstance(stem_payload, dict):
                continue
            stem_path = stem_payload.get("path")
            if isinstance(stem_path, str) and stem_path.strip():
                stem_paths.append(str(Path(stem_path).resolve()))

    if stem_paths:
        try:
            return Path(os.path.commonpath(stem_paths))
        except ValueError:
            pass

    return Path(source_stem_manifest_path).resolve().parent


def _patch_hq_progress(
    report: Callable[[int, str, dict[str, Any] | None], None],
    *,
    start_progress: int,
    end_progress: int,
) -> Callable[[], None]:
    progress_state = {"value": start_progress}
    original_emit_progress = high_quality_stem_export_module._emit_progress

    def bridged_emit_progress(message: str) -> None:
        progress_state["value"] = min(end_progress, progress_state["value"] + 1)
        report(progress_state["value"], message, {"stage": "hq_export"})

    high_quality_stem_export_module._emit_progress = bridged_emit_progress

    def restore() -> None:
        high_quality_stem_export_module._emit_progress = original_emit_progress

    return restore


def _finalize_external_render_manifest(
    export_manifest: dict[str, Any],
    *,
    source_stem_manifest_path: str | Path,
    source_manifest: dict[str, Any],
    output_dir: str | Path,
    latest_pointer_path: str | Path,
    preset: dict[str, Any],
    render_id: str,
    mode: str,
    mode_config: dict[str, Any],
    raw_reference_path: str | Path | None,
    report: Callable[[int, str, dict[str, Any] | None], None],
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_stems = source_manifest.get("stems", {}) if isinstance(source_manifest.get("stems"), dict) else {}
    rendered_stems = export_manifest.get("stems", {}) if isinstance(export_manifest.get("stems"), dict) else {}
    layer_states = preset.get("layerStates", {}) if isinstance(preset.get("layerStates"), dict) else {}

    report(90, "Summing rendered stems into render mix", {"stage": "sum_mix"})
    enabled_stem_keys = [
        stem_key
        for stem_key, layer_state in layer_states.items()
        if isinstance(stem_key, str)
        and isinstance(layer_state, dict)
        and bool(layer_state.get("enabled", True))
        and stem_key in rendered_stems
    ]
    if not enabled_stem_keys:
        enabled_stem_keys = [
            stem_key
            for stem_key in rendered_stems.keys()
            if stem_key != "melody"
        ]

    mix_components: list[np.ndarray] = []
    for stem_key in enabled_stem_keys:
        stem_payload = rendered_stems.get(stem_key)
        if not isinstance(stem_payload, dict):
            continue
        stem_path = Path(str(stem_payload.get("path", "")))
        if not stem_path.exists():
            continue
        stem_audio, sample_rate = sf.read(str(stem_path), always_2d=True, dtype="float32")
        if int(sample_rate) != SEPARATION_SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate in {stem_path}: {sample_rate}")
        mix_components.append(stem_audio.astype(np.float32))

    if mix_components:
        render_mix = np.sum(np.stack(mix_components, axis=0), axis=0).astype(np.float32)
    else:
        first_stem_payload = next(iter(rendered_stems.values()), None)
        if not isinstance(first_stem_payload, dict):
            raise ValueError("Rendered manifest has no readable stems for mix synthesis")
        first_stem_path = Path(str(first_stem_payload.get("path", "")))
        first_audio, sample_rate = sf.read(str(first_stem_path), always_2d=True, dtype="float32")
        if int(sample_rate) != SEPARATION_SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate in {first_stem_path}: {sample_rate}")
        render_mix = np.zeros_like(first_audio, dtype=np.float32)

    render_mix = _limit_peak(render_mix)
    mix_path = output_path / "mix.flac"
    report(94, "Writing render mix file", {"stage": "write_mix", "mix_path": str(mix_path)})
    sf.write(str(mix_path), render_mix, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)

    manifest_stems: dict[str, Any] = {}
    for stem_key, stem_payload in rendered_stems.items():
        if not isinstance(stem_key, str) or not isinstance(stem_payload, dict):
            continue

        source_stem_payload = source_stems.get(stem_key) if isinstance(source_stems.get(stem_key), dict) else {}
        path = str(stem_payload.get("path", ""))
        manifest_stems[stem_key] = {
            "path": path,
            "label": str(stem_payload.get("label") or source_stem_payload.get("label") or _humanize_stem_key(stem_key)),
            "description": str(stem_payload.get("description") or f"Rendered stem from tuning preset ({render_id})."),
            "family": str(stem_payload.get("family") or "tuning_render"),
            "kind": str(stem_payload.get("kind") or ("rendered_melody" if stem_key == "melody" else "rendered_stem")),
            "default_enabled": bool(
                layer_states.get(stem_key, {}).get("enabled", False)
                if isinstance(layer_states.get(stem_key), dict)
                else stem_key != "melody"
            ),
            **{
                key: value
                for key, value in stem_payload.items()
                if key not in {"label", "description", "family", "kind", "default_enabled"}
            },
        }

    manifest = {
        "render_id": render_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stem_manifest": str(source_stem_manifest_path),
        "audio_path": export_manifest.get("audio_path", source_manifest.get("audio_path")),
        "model_name": export_manifest.get("model_name", source_manifest.get("model_name")),
        "sample_rate": SEPARATION_SAMPLE_RATE,
        "export_subtype": export_manifest.get("export_subtype", FINAL_STEM_EXPORT_SUBTYPE),
        "pipeline": {
            "mode": f"stem_tuning_{mode}_v1",
            "render_strategy": mode,
            "source_pipeline": export_manifest.get("pipeline"),
            "raw_reference_path": str(raw_reference_path) if raw_reference_path else None,
        },
        "preset": preset,
        "mode_config": mode_config,
        "mix": {
            "path": str(mix_path),
            **_stem_summary(render_mix),
        },
        "stems": manifest_stems,
    }

    manifest_path = output_path / "stem_manifest.json"
    report(97, "Writing render manifest", {"stage": "write_manifest", "manifest_path": str(manifest_path)})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_pointer = Path(latest_pointer_path)
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    report(99, "Updating latest render pointer", {"stage": "update_latest_pointer", "latest_pointer_path": str(latest_pointer)})
    latest_pointer.write_text(json.dumps({
        "render_id": render_id,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    report(100, "Render finished", {"stage": "completed", "manifest_path": str(manifest_path), "mix_path": str(mix_path)})
    return {
        "render_id": render_id,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_path),
        "mix_path": str(mix_path),
    }


def _render_external_manifest_with_post_dsp(
    export_manifest: dict[str, Any],
    output_dir: str | Path,
    preset: dict[str, Any],
    *,
    render_id: str,
    latest_pointer_path: str | Path,
    mode: str,
    mode_config: dict[str, Any],
    raw_reference_path: str | Path | None = None,
    dsp_device: str = "cuda",
    progress_hook: Callable[[int, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    export_manifest_path = export_manifest.get("manifest_path")
    if not isinstance(export_manifest_path, str) or export_manifest_path.strip() == "":
        raise ValueError("External render manifest is missing manifest_path")

    resolved_export_manifest_path = Path(export_manifest_path)
    if not resolved_export_manifest_path.exists():
        raise FileNotFoundError(f"External render manifest does not exist: {resolved_export_manifest_path}")

    def report(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        if progress_hook is None:
            return
        progress_hook(int(progress), message, extra or None)

    def post_dsp_progress(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        scaled_progress = 89 + int(round((max(0, min(100, int(progress))) / 100) * 10))
        payload = dict(extra or {})
        payload.setdefault("stage", "post_dsp_from_external_manifest")
        payload.setdefault("external_manifest_path", str(resolved_export_manifest_path))
        report(scaled_progress, f"Applying post-DSP render pass: {message}", payload)

    final_render = render_stem_tuning_from_manifest(
        resolved_export_manifest_path,
        output_dir,
        preset,
        render_id=render_id,
        latest_pointer_path=latest_pointer_path,
        raw_reference_path=raw_reference_path,
        dsp_device=dsp_device,
        progress_hook=post_dsp_progress,
    )

    final_manifest_path = Path(str(final_render["manifest_path"]))
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    existing_pipeline = final_manifest.get("pipeline")
    final_manifest["pipeline"] = {
        **(existing_pipeline if isinstance(existing_pipeline, dict) else {}),
        "mode": f"stem_tuning_{mode}_plus_post_dsp_v1",
        "render_strategy": f"{mode}_plus_post_dsp",
        "source_pipeline": export_manifest.get("pipeline"),
        "external_manifest_path": str(resolved_export_manifest_path),
        "dsp_device": str(dsp_device),
        "raw_reference_path": str(raw_reference_path) if raw_reference_path else None,
    }
    final_manifest["mode_config"] = mode_config
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report(
        100,
        "Render finished",
        {
            "stage": "completed",
            "manifest_path": str(final_manifest_path),
            "mix_path": str(final_render["mix_path"]),
        },
    )

    return final_render


def _apply_layer_processors(
    audio: np.ndarray,
    layer_state: dict[str, Any],
    *,
    dsp_device: str,
    processor_config: dict[str, Any] | None = None,
) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    config = _normalize_processor_config(processor_config)
    harmonic, percussive = _split_harmonic_percussive(
        audio,
        harmonic_margin=float(config["harmonicMargin"]),
        percussive_margin=float(config["percussiveMargin"]),
        dsp_device=dsp_device,
    )

    harmonic_gain = _db_to_gain(float(layer_state.get("harmonicDuckDb", 0.0)))
    harmonic_processed = harmonic * harmonic_gain

    bleed_gain = _db_to_gain(float(layer_state.get("bleedDuckDb", 0.0)))
    bleed_band = _apply_filter(
        harmonic_processed,
        low_hz=float(config["bleedLowHz"]),
        high_hz=float(config["bleedHighHz"]),
    )
    harmonic_processed = harmonic_processed - bleed_band + (bleed_band * bleed_gain)

    percussive_processed = percussive.copy()

    transient_gain = _db_to_gain(float(layer_state.get("transientDb", 0.0)))
    transient_band = _apply_filter(percussive, low_hz=float(config["transientCutoffHz"]))
    percussive_processed += transient_band * float(transient_gain - 1.0)

    body_gain = _db_to_gain(float(layer_state.get("bodyDb", 0.0)))
    body_band = _apply_filter(
        percussive,
        low_hz=float(config["bodyLowHz"]),
        high_hz=float(config["bodyHighHz"]),
    )
    percussive_processed += body_band * float(body_gain - 1.0)

    return _limit_peak((harmonic_processed + percussive_processed).astype(np.float32))


def _apply_gain_and_focus(
    audio: np.ndarray,
    layer_key: str,
    layer_state: dict[str, Any],
    preset: dict[str, Any],
    focus_mask: np.ndarray,
) -> np.ndarray:
    base_gain_db = float(layer_state.get("gainDb", 0.0))
    processor_config = preset.get("processorConfig", {}) if isinstance(preset.get("processorConfig"), dict) else {}
    if bool(processor_config.get("bypass", False)):
        gain_curve = np.full((audio.shape[0],), _db_to_gain(base_gain_db), dtype=np.float32)
        return _limit_peak(audio * gain_curve[:, np.newaxis])

    focus_window = preset.get("focusWindow", {}) if isinstance(preset.get("focusWindow"), dict) else {}
    target_key = str(focus_window.get("targetLayerKey") or preset.get("sourceKey") or layer_key)
    if bool(focus_window.get("enabled", False)):
        extra_db = (
            float(focus_window.get("layerGainDb", 0.0))
            if layer_key == target_key
            else float(focus_window.get("otherLayersDuckDb", 0.0))
        )
        gain_curve = _db_to_gain(base_gain_db) * np.power(10.0, (focus_mask * extra_db) / 20.0)
    else:
        gain_curve = np.full((audio.shape[0],), _db_to_gain(base_gain_db), dtype=np.float32)

    return _limit_peak(audio * gain_curve[:, np.newaxis].astype(np.float32))


def render_stem_tuning_hybrid_build(
    source_stem_manifest_path: str | Path,
    output_dir: str | Path,
    preset: dict[str, Any],
    *,
    render_id: str,
    latest_pointer_path: str | Path,
    source_audio_path: str | Path | None,
    raw_reference_path: str | Path | None = None,
    dsp_device: str = "cuda",
    progress_hook: Callable[[int, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    def report(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        if progress_hook is None:
            return
        progress_hook(int(progress), message, extra or None)

    source_manifest = _load_source_manifest(source_stem_manifest_path)
    hybrid_build_config = _normalize_hybrid_build_config(
        preset.get("hybridBuildConfig", {}) if isinstance(preset.get("hybridBuildConfig"), dict) else {}
    )
    unsafe_model_internals = _normalize_unsafe_model_internals(
        preset.get("unsafeModelInternals", {}) if isinstance(preset.get("unsafeModelInternals"), dict) else {}
    )
    resolved_audio_path = Path(str(source_audio_path or source_manifest.get("audio_path") or "")).resolve()
    if not resolved_audio_path.exists():
        raise FileNotFoundError("Hybrid build render requires a readable source audio path")

    report(4, "Preparing hybrid build render", {"stage": "prepare_hybrid_build", "audio_path": str(resolved_audio_path)})
    external_output_dir = Path(output_dir) / "_external_source"
    restore_progress = _patch_hq_progress(report, start_progress=10, end_progress=88)
    try:
        if bool(hybrid_build_config["aggressiveMelodyFirst"]):
            export_manifest = export_track_stems_aggressive_melody_first(
                resolved_audio_path,
                external_output_dir,
                model_name=str(hybrid_build_config["modelName"]),
                device=None,
                sharpen_bleed=bool(hybrid_build_config["sharpenBleed"]),
                hybrid_stage_models=unsafe_model_internals["hybrid"],
                bs_roformer_runtime=unsafe_model_internals["bsRoformer"],
            )
        else:
            export_manifest = export_track_stems_full_res(
                resolved_audio_path,
                external_output_dir,
                model_name=str(hybrid_build_config["modelName"]),
                device=None,
                sharpen_bleed=bool(hybrid_build_config["sharpenBleed"]),
                include_melody_stem=bool(hybrid_build_config["includeMelodyStem"]),
                hybrid_stage_models=unsafe_model_internals["hybrid"],
                bs_roformer_runtime=unsafe_model_internals["bsRoformer"],
            )
    finally:
        restore_progress()

    return _render_external_manifest_with_post_dsp(
        export_manifest,
        output_dir=output_dir,
        preset=preset,
        render_id=render_id,
        latest_pointer_path=latest_pointer_path,
        mode="hybrid_build",
        mode_config={
            **hybrid_build_config,
            "unsafeModelInternals": unsafe_model_internals,
        },
        raw_reference_path=raw_reference_path,
        dsp_device=dsp_device,
        progress_hook=progress_hook,
    )


def render_stem_tuning_stem_rewrite(
    source_stem_manifest_path: str | Path,
    output_dir: str | Path,
    preset: dict[str, Any],
    *,
    render_id: str,
    latest_pointer_path: str | Path,
    raw_reference_path: str | Path | None = None,
    dsp_device: str = "cuda",
    progress_hook: Callable[[int, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    def report(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        if progress_hook is None:
            return
        progress_hook(int(progress), message, extra or None)

    source_manifest = _load_source_manifest(source_stem_manifest_path)
    stem_rewrite_config = _normalize_stem_rewrite_config(
        preset.get("stemRewriteConfig", {}) if isinstance(preset.get("stemRewriteConfig"), dict) else {}
    )
    resolved_stem_dir = _stem_dir_from_manifest(source_manifest, source_stem_manifest_path)
    if not resolved_stem_dir.exists():
        raise FileNotFoundError(f"Stem rewrite render requires a readable source stem directory: {resolved_stem_dir}")

    report(4, "Preparing stem rewrite render", {"stage": "prepare_stem_rewrite", "stem_dir": str(resolved_stem_dir)})
    external_output_dir = Path(output_dir) / "_external_source"
    restore_progress = _patch_hq_progress(report, start_progress=10, end_progress=88)
    try:
        export_manifest = export_aggressive_melody_first_from_stem_dir(
            resolved_stem_dir,
            external_output_dir,
            start_sec=None,
            end_sec=None,
            other_move=float(stem_rewrite_config["otherMove"]),
            guitar_move=float(stem_rewrite_config["guitarMove"]),
            piano_move=float(stem_rewrite_config["pianoMove"]),
            drums_masked_move=float(stem_rewrite_config["drumsMaskedMove"]),
            target_melody_rms=float(stem_rewrite_config["targetMelodyRms"]),
            max_melody_gain=float(stem_rewrite_config["maxMelodyGain"]),
            stage_drum_harmonic_move=float(stem_rewrite_config["stageDrumHarmonicMove"]),
            stage_harmonic_stem_percussive_move=float(stem_rewrite_config["stageHarmonicStemPercussiveMove"]),
            stage_cleanup_drum_harmonic_move=float(stem_rewrite_config["stageCleanupDrumHarmonicMove"]),
            stage_cleanup_harmonic_stem_percussive_move=float(stem_rewrite_config["stageCleanupHarmonicStemPercussiveMove"]),
            harmonic_count=int(stem_rewrite_config["harmonicCount"]),
            confidence_floor=float(stem_rewrite_config["confidenceFloor"]),
            dsp_device=dsp_device,
        )
    finally:
        restore_progress()

    return _render_external_manifest_with_post_dsp(
        export_manifest,
        output_dir=output_dir,
        preset=preset,
        render_id=render_id,
        latest_pointer_path=latest_pointer_path,
        mode="stem_rewrite",
        mode_config=stem_rewrite_config,
        raw_reference_path=raw_reference_path,
        dsp_device=dsp_device,
        progress_hook=progress_hook,
    )


def render_stem_tuning_vocals_rewrite(
    source_stem_manifest_path: str | Path,
    output_dir: str | Path,
    preset: dict[str, Any],
    *,
    render_id: str,
    latest_pointer_path: str | Path,
    source_audio_path: str | Path | None,
    raw_reference_path: str | Path | None = None,
    dsp_device: str = "cuda",
    progress_hook: Callable[[int, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    def report(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        if progress_hook is None:
            return
        progress_hook(int(progress), message, extra or None)

    source_manifest = _load_source_manifest(source_stem_manifest_path)
    vocals_rewrite_runtime = _resolve_vocals_rewrite_runtime(
        preset.get("vocalsRewriteConfig", {}) if isinstance(preset.get("vocalsRewriteConfig"), dict) else {}
    )
    vocals_rewrite_raw_config = vocals_rewrite_runtime["raw"]
    vocals_rewrite_config = vocals_rewrite_runtime["resolvedMicroConfig"]
    unsafe_model_internals = _normalize_unsafe_model_internals(
        preset.get("unsafeModelInternals", {}) if isinstance(preset.get("unsafeModelInternals"), dict) else {}
    )
    resolved_audio_path = Path(str(source_audio_path or source_manifest.get("audio_path") or "")).resolve()
    if not resolved_audio_path.exists():
        raise FileNotFoundError("Vocals rewrite render requires a readable source audio path")

    report(4, "Preparing vocals rewrite render", {"stage": "prepare_vocals_rewrite", "audio_path": str(resolved_audio_path)})
    external_output_dir = Path(output_dir) / "_external_source"
    preview_output_dir = external_output_dir / "_vocals_preview"
    preview_output_dir.mkdir(parents=True, exist_ok=True)
    restore_progress = _patch_hq_progress(report, start_progress=10, end_progress=88)
    try:
        vocal_manifest = export_vocal_preview_from_audio(
            resolved_audio_path,
            preview_output_dir,
            candidate_mode=str(vocals_rewrite_config["candidateMode"]),
            start_sec=None,
            end_sec=None,
            device=None,
            low_bs_weight=float(vocals_rewrite_config["lowBsWeight"]),
            high_bs_weight=float(vocals_rewrite_config["highBsWeight"]),
            inactive_low_bs_weight=float(vocals_rewrite_config["inactiveLowBsWeight"]),
            inactive_high_bs_weight=float(vocals_rewrite_config["inactiveHighBsWeight"]),
            crossover_hz=float(vocals_rewrite_config["crossoverHz"]),
            activity_abs_floor_ratio=float(vocals_rewrite_config["activityAbsFloorRatio"]),
            activity_rel_floor=float(vocals_rewrite_config["activityRelFloor"]),
            activity_smoothing_frames=int(vocals_rewrite_config["activitySmoothingFrames"]),
            vocal_sharpen_alpha=float(vocals_rewrite_config["vocalSharpenAlpha"]),
            tail_gate_strength=float(vocals_rewrite_config["tailGateStrength"]),
            tail_gate_floor_ratio=float(vocals_rewrite_config["tailGateFloorRatio"]),
            tail_gate_smoothing_frames=int(vocals_rewrite_config["tailGateSmoothingFrames"]),
            subtractive_strength=float(vocals_rewrite_config["subtractiveStrength"]),
            subtractive_power=float(vocals_rewrite_config["subtractivePower"]),
            subtractive_residual_floor=float(vocals_rewrite_config["subtractiveResidualFloor"]),
            subtractive_active_restore=float(vocals_rewrite_config["subtractiveActiveRestore"]),
            subtractive_target_rms_ratio=float(vocals_rewrite_config["subtractiveTargetRmsRatio"]),
            subtractive_max_gain=float(vocals_rewrite_config["subtractiveMaxGain"]),
            hard_mask_threshold=float(vocals_rewrite_config["hardMaskThreshold"]),
            hard_mask_power=float(vocals_rewrite_config["hardMaskPower"]),
            hard_mask_residual_floor=float(vocals_rewrite_config["hardMaskResidualFloor"]),
            hybrid_stage_models=unsafe_model_internals["hybrid"],
            bs_roformer_runtime=unsafe_model_internals["bsRoformer"],
        )
    finally:
        restore_progress()

    vocal_stem_payload = vocal_manifest.get("stems", {}).get("vocals", {}) if isinstance(vocal_manifest.get("stems"), dict) else {}
    vocal_path = Path(str(vocal_stem_payload.get("path", "")))
    if not vocal_path.exists():
        raise FileNotFoundError("Vocals rewrite render did not produce a readable vocals stem")

    vocal_postprocess = vocals_rewrite_runtime["postprocess"]
    if any(float(vocal_postprocess.get(key, 0.0)) > 1e-6 for key in ("airBoostDb", "bodyBoostDb", "artifactDampAmount", "sideAttenuation")):
        report(
            89,
            "Applying vocals rewrite macro postprocess",
            {
                "stage": "vocals_macro_postprocess",
                "air_boost_db": float(vocal_postprocess.get("airBoostDb", 0.0)),
                "body_boost_db": float(vocal_postprocess.get("bodyBoostDb", 0.0)),
                "artifact_damp_amount": float(vocal_postprocess.get("artifactDampAmount", 0.0)),
                "side_attenuation": float(vocal_postprocess.get("sideAttenuation", 0.0)),
            },
        )
        rewritten_vocals, sample_rate = sf.read(str(vocal_path), always_2d=True, dtype="float32")
        if int(sample_rate) != SEPARATION_SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate in rewritten vocals stem: {sample_rate}")
        rewritten_vocals = _apply_vocals_macro_postprocess(rewritten_vocals, vocal_postprocess)
        sf.write(str(vocal_path), rewritten_vocals, sample_rate, subtype=FINAL_STEM_EXPORT_SUBTYPE)

    source_stems = source_manifest.get("stems", {}) if isinstance(source_manifest.get("stems"), dict) else {}
    rebuilt_manifest: dict[str, Any] = {
        "audio_path": str(resolved_audio_path),
        "model_name": source_manifest.get("model_name"),
        "sample_rate": SEPARATION_SAMPLE_RATE,
        "export_subtype": FINAL_STEM_EXPORT_SUBTYPE,
        "pipeline": {
            "mode": "vocals_rewrite_from_audio",
            "candidate_mode": str(vocals_rewrite_config["candidateMode"]),
            "base_stem_manifest": str(Path(source_stem_manifest_path).resolve()),
            "vocals_preview_manifest_path": str(vocal_manifest.get("manifest_path", "")),
            "instrumental_strategy": "preserve_source_stems_exact",
            "vocals_strategy": "rewrite_from_source_audio",
            "vocals_pipeline": vocal_manifest.get("pipeline"),
            "unsafe_model_internals": unsafe_model_internals,
        },
        "tuning": {
            "raw": vocals_rewrite_raw_config,
            "resolved_micro_config": vocals_rewrite_config,
            "effective_macro_config": vocals_rewrite_runtime["effectiveMacroConfig"],
            "cleanup_family": vocals_rewrite_runtime["cleanupFamily"],
            "macro_postprocess": vocal_postprocess,
        },
        "stems": {},
    }

    for stem_key, stem_payload in source_stems.items():
        if not isinstance(stem_key, str) or not isinstance(stem_payload, dict):
            continue

        stem_entry = {
            "path": str(vocal_path) if stem_key == "vocals" else str(stem_payload.get("path", "")),
            "label": stem_payload.get("label") or _humanize_stem_key(stem_key),
            "description": stem_payload.get("description") or (
                "Vocals rebuilt from source audio for tuning render."
                if stem_key == "vocals"
                else f"Preserved source stem for vocals rewrite render ({render_id})."
            ),
            "family": stem_payload.get("family") or "tuning_render",
            "kind": stem_payload.get("kind") or ("rendered_vocals" if stem_key == "vocals" else "stem"),
            "default_enabled": bool(stem_payload.get("default_enabled", stem_key != "melody")),
        }

        if stem_key == "vocals":
            stem_entry.update({
                key: value
                for key, value in vocal_stem_payload.items()
                if key not in {"path"}
            })

        rebuilt_manifest["stems"][stem_key] = stem_entry

    external_manifest_path = external_output_dir / "stem_manifest.json"
    external_output_dir.mkdir(parents=True, exist_ok=True)
    external_manifest_path.write_text(json.dumps(rebuilt_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rebuilt_manifest["manifest_path"] = str(external_manifest_path)

    return _render_external_manifest_with_post_dsp(
        rebuilt_manifest,
        output_dir=output_dir,
        preset=preset,
        render_id=render_id,
        latest_pointer_path=latest_pointer_path,
        mode="vocals_rewrite",
        mode_config={
            **vocals_rewrite_raw_config,
            "resolvedMicroConfig": vocals_rewrite_config,
            "effectiveMacroConfig": vocals_rewrite_runtime["effectiveMacroConfig"],
            "cleanupFamily": vocals_rewrite_runtime["cleanupFamily"],
            "macroPostprocess": vocal_postprocess,
            "unsafeModelInternals": unsafe_model_internals,
        },
        raw_reference_path=raw_reference_path,
        dsp_device=dsp_device,
        progress_hook=progress_hook,
    )


def render_stem_tuning_from_manifest(
    source_stem_manifest_path: str | Path,
    output_dir: str | Path,
    preset: dict[str, Any],
    *,
    render_id: str,
    latest_pointer_path: str | Path,
    raw_reference_path: str | Path | None = None,
    dsp_device: str = "cuda",
    progress_hook: Callable[[int, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    def report(progress: int, message: str, **extra: Any) -> None:
        if progress_hook is None:
            return
        progress_hook(int(progress), message, extra or None)

    report(2, "Loading source stem manifest", stage="load_manifest")
    source_manifest_path = Path(source_stem_manifest_path)
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Source stem manifest is missing: {source_manifest_path}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    stems_payload = source_manifest.get("stems", {})
    if not isinstance(stems_payload, dict) or not stems_payload:
        raise ValueError(f"Source stem manifest has no stems: {source_manifest_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report(5, "Preparing output directory", stage="prepare_output", output_dir=str(output_path))

    preset_path = output_path / "preset.json"
    preset_path.write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")

    layer_states = preset.get("layerStates", {}) if isinstance(preset.get("layerStates"), dict) else {}
    focus_window = preset.get("focusWindow", {}) if isinstance(preset.get("focusWindow"), dict) else {}
    processor_config = preset.get("processorConfig", {}) if isinstance(preset.get("processorConfig"), dict) else {}

    rendered_stems: dict[str, np.ndarray] = {}
    manifest_stems: dict[str, Any] = {}

    ordered_keys = list(stems_payload.keys())
    loaded_any = False
    focus_mask: np.ndarray | None = None
    stem_count = max(1, len(ordered_keys))
    report(8, f"Rendering {stem_count} stem files", stage="render_stems", stem_count=stem_count)

    for stem_index, stem_key in enumerate(ordered_keys, start=1):
        stem_payload = stems_payload.get(stem_key)
        if not isinstance(stem_payload, dict):
            continue

        source_path = Path(str(stem_payload.get("path", "")))
        if not source_path.exists():
            continue

        stem_base_progress = 8 + int(((stem_index - 1) / stem_count) * 74)
        report(
            stem_base_progress,
            f"Loading stem {stem_index}/{stem_count}: {stem_key}",
            stage="load_stem",
            stem_key=stem_key,
            stem_index=stem_index,
            stem_count=stem_count,
        )
        audio, sample_rate = sf.read(str(source_path), always_2d=True, dtype="float32")
        if int(sample_rate) != SEPARATION_SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate in {source_path}: {sample_rate}")

        loaded_any = True
        if focus_mask is None:
            focus_mask = _focus_mask(audio.shape[0], int(sample_rate), focus_window)

        manifest_default_enabled = bool(stem_payload.get("default_enabled", stem_key != "melody"))
        layer_state = layer_states.get(stem_key)
        if not isinstance(layer_state, dict):
            layer_state = {
                "enabled": manifest_default_enabled,
                "gainDb": 0.0,
                "transientDb": 0.0,
                "bodyDb": 0.0,
                "harmonicDuckDb": 0.0,
                "bleedDuckDb": 0.0,
            }

        report(
            stem_base_progress + 8,
            f"Processing stem {stem_index}/{stem_count}: {stem_key}",
            stage="process_stem",
            stem_key=stem_key,
            stem_index=stem_index,
            stem_count=stem_count,
        )
        processed = audio.astype(np.float32)
        if not bool(processor_config.get("bypass", False)):
            processed = _apply_layer_processors(
                processed,
                layer_state,
                dsp_device=dsp_device,
                processor_config=processor_config,
            )
        processed = _apply_gain_and_focus(processed, stem_key, layer_state, preset, focus_mask if focus_mask is not None else np.zeros((audio.shape[0],), dtype=np.float32))
        processed = _limit_peak(processed)

        target_path = output_path / f"{stem_key}.flac"
        report(
            stem_base_progress + 16,
            f"Writing stem {stem_index}/{stem_count}: {stem_key}",
            stage="write_stem",
            stem_key=stem_key,
            stem_index=stem_index,
            stem_count=stem_count,
            target_path=str(target_path),
        )
        sf.write(str(target_path), processed, sample_rate, subtype=FINAL_STEM_EXPORT_SUBTYPE)

        rendered_stems[stem_key] = processed
        manifest_stems[stem_key] = {
            "path": str(target_path),
            "label": str(stem_payload.get("label") or stem_key.replace("_", " ").title()),
            "description": f"Rendered stem from tuning preset ({render_id}).",
            "family": "tuning_render",
            "kind": "rendered_stem",
            "default_enabled": bool(layer_state.get("enabled", True)),
            **_stem_summary(processed),
        }

    if not loaded_any:
        raise ValueError(f"No readable stems were found in manifest: {source_manifest_path}")

    report(84, "Summing enabled stems into render mix", stage="sum_mix")
    mix_components = [
        audio
        for stem_key, audio in rendered_stems.items()
        if bool(
            (layer_states.get(stem_key) or {}).get(
                "enabled",
                bool(
                    (stems_payload.get(stem_key) or {}).get("default_enabled", stem_key != "melody")
                    if isinstance(stems_payload.get(stem_key), dict)
                    else stem_key != "melody"
                ),
            )
        )
    ]

    if mix_components:
        render_mix = np.sum(np.stack(mix_components, axis=0), axis=0).astype(np.float32)
    else:
        shape = next(iter(rendered_stems.values())).shape
        render_mix = np.zeros(shape, dtype=np.float32)

    render_mix = _limit_peak(render_mix)
    mix_path = output_path / "mix.flac"
    report(90, "Writing render mix file", stage="write_mix", mix_path=str(mix_path))
    sf.write(str(mix_path), render_mix, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)

    manifest = {
        "render_id": render_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stem_manifest": str(source_manifest_path),
        "audio_path": source_manifest.get("audio_path"),
        "model_name": source_manifest.get("model_name"),
        "sample_rate": SEPARATION_SAMPLE_RATE,
        "export_subtype": FINAL_STEM_EXPORT_SUBTYPE,
        "pipeline": {
            "mode": "stem_tuning_render_v1",
            "render_strategy": "container_stem_layer_preset",
            "dsp_device": str(dsp_device),
            "raw_reference_path": str(raw_reference_path) if raw_reference_path else None,
        },
        "preset": preset,
        "mix": {
            "path": str(mix_path),
            **_stem_summary(render_mix),
        },
        "stems": manifest_stems,
    }

    manifest_path = output_path / "stem_manifest.json"
    report(95, "Writing render manifest", stage="write_manifest", manifest_path=str(manifest_path))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_pointer = Path(latest_pointer_path)
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    report(98, "Updating latest render pointer", stage="update_latest_pointer", latest_pointer_path=str(latest_pointer))
    latest_pointer.write_text(json.dumps({
        "render_id": render_id,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    report(100, "Render finished", stage="completed", manifest_path=str(manifest_path), mix_path=str(mix_path))

    return {
        "render_id": render_id,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_path),
        "mix_path": str(mix_path),
    }


def render_stem_tuning(
    source_stem_manifest_path: str | Path,
    output_dir: str | Path,
    preset: dict[str, Any],
    *,
    render_id: str,
    latest_pointer_path: str | Path,
    source_audio_path: str | Path | None = None,
    raw_reference_path: str | Path | None = None,
    dsp_device: str = "cuda",
    progress_hook: Callable[[int, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    render_mode = str(preset.get("renderMode") or "post_dsp")

    if render_mode == "hybrid_build":
        return render_stem_tuning_hybrid_build(
            source_stem_manifest_path,
            output_dir,
            preset,
            render_id=render_id,
            latest_pointer_path=latest_pointer_path,
            source_audio_path=source_audio_path,
            raw_reference_path=raw_reference_path,
            dsp_device=dsp_device,
            progress_hook=progress_hook,
        )

    if render_mode == "stem_rewrite":
        return render_stem_tuning_stem_rewrite(
            source_stem_manifest_path,
            output_dir,
            preset,
            render_id=render_id,
            latest_pointer_path=latest_pointer_path,
            raw_reference_path=raw_reference_path,
            dsp_device=dsp_device,
            progress_hook=progress_hook,
        )

    if render_mode == "vocals_rewrite":
        return render_stem_tuning_vocals_rewrite(
            source_stem_manifest_path,
            output_dir,
            preset,
            render_id=render_id,
            latest_pointer_path=latest_pointer_path,
            source_audio_path=source_audio_path,
            raw_reference_path=raw_reference_path,
            dsp_device=dsp_device,
            progress_hook=progress_hook,
        )

    return render_stem_tuning_from_manifest(
        source_stem_manifest_path,
        output_dir,
        preset,
        render_id=render_id,
        latest_pointer_path=latest_pointer_path,
        raw_reference_path=raw_reference_path,
        dsp_device=dsp_device,
        progress_hook=progress_hook,
    )
