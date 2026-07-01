from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import (  # noqa: E402
    SEMANTIC_JUDGE_INPUT_DELIVERY_STATUS,
    SEMANTIC_JUDGE_INPUT_PRESENTATION_STATUS,
    SEMANTIC_JUDGE_INPUT_RELEASE_STATUS,
    ArtifactStatusView,
    sha256_file,
    status_view_from_manifest,
)
from tools.run_no_reference_result_judge import (  # noqa: E402
    judge_candidate_row as judge_no_reference_candidate_row,
    resolve_path as resolve_no_reference_path,
)


JUDGE_JSONL_NAME = "no_reference_stem_semantic_judge.jsonl"
SUMMARY_JSON_NAME = "no_reference_stem_semantic_judge_summary.json"
SUMMARY_MD_NAME = "no_reference_stem_semantic_judge_summary.md"
PROGRESS_JSONL_NAME = "no_reference_stem_semantic_judge_progress.jsonl"

SUPPORTED_STEMS = {"melody", "drums", "bass", "other", "guitar", "piano"}
MODE_SPECIFIC_STEMS = {"vocal", "minus"}
MELODY_MIN_DOMINANT_FREQUENCY_HZ = 80.0
MELODY_BASS_DOMINANT_LOW_BAND_RATIO = 0.70
MELODY_BASS_DOMINANT_MAX_MID_BAND_RATIO = 0.35

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "release_ready": False,
}

SEMANTIC_JUDGE_INPUT_GATE_KEYS = (
    "audio_integrity_gate",
    "gpu_provenance_gate",
    "source_contract_gate",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def source_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
    }


def nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safety_from_manifest(payload: dict[str, Any]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for source in (payload, nested_dict(payload, "safety"), nested_dict(payload, "safety_flags")):
        for key in safety:
            safety[key] = bool(safety[key] or source.get(key) is True)
    return safety


def safety_violation(safety: dict[str, bool]) -> str:
    for key in sorted(safety):
        if safety.get(key) is True:
            return key
    return ""


def gate_passed(payload: dict[str, Any], gate_name: str) -> bool:
    gate = nested_dict(payload, gate_name)
    return str(gate.get("status") or "") == "passed"


def is_semantic_judge_input_manifest(payload: dict[str, Any], view: ArtifactStatusView) -> bool:
    if payload.get("semantic_judge_input") is not True:
        return False
    if str(payload.get("manifest_type") or "") != "no_reference_stem_listen_preview":
        return False
    if view.delivery_allowed:
        return False
    if view.delivery_status != SEMANTIC_JUDGE_INPUT_DELIVERY_STATUS:
        return False
    if view.presentation_status != SEMANTIC_JUDGE_INPUT_PRESENTATION_STATUS:
        return False
    if view.release_status != SEMANTIC_JUDGE_INPUT_RELEASE_STATUS:
        return False
    if payload.get("semantic_quality_verified") is True:
        return False
    if any(payload.get(key) is True for key in ("release_ready", "can_mark_release_ready", "can_publish_as_release")):
        return False
    return all(gate_passed(payload, gate_name) for gate_name in SEMANTIC_JUDGE_INPUT_GATE_KEYS)


def read_audio_mono(path: Path) -> tuple[np.ndarray, int, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if data.size == 0:
        return np.zeros(0, dtype=np.float32), int(sample_rate or 0), 0
    mono = np.mean(data, axis=1).astype(np.float32)
    return mono, int(sample_rate or 0), int(data.shape[1])


def analysis_signal(mono: np.ndarray, sample_rate: int, *, max_samples: int = 262_144) -> tuple[np.ndarray, float]:
    if mono.size <= max_samples:
        return mono.astype(np.float32), float(sample_rate)
    step = int(math.ceil(mono.size / max_samples))
    return mono[::step].astype(np.float32), float(sample_rate) / float(step)


def spectral_metrics(mono: np.ndarray, sample_rate: int) -> dict[str, float]:
    signal, effective_sr = analysis_signal(mono, sample_rate)
    if signal.size < 64 or effective_sr <= 0:
        return {
            "dominant_frequency_hz": 0.0,
            "pitch_confidence": 0.0,
            "spectral_flatness": 1.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }
    signal = signal - float(np.mean(signal))
    window = np.hanning(signal.size).astype(np.float32)
    spectrum = np.fft.rfft(signal * window)
    power = np.abs(spectrum).astype(np.float64) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / effective_sr)
    if power.size <= 1 or float(np.sum(power[1:])) <= 0.0:
        return {
            "dominant_frequency_hz": 0.0,
            "pitch_confidence": 0.0,
            "spectral_flatness": 1.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }
    power_no_dc = power.copy()
    power_no_dc[0] = 0.0
    total = float(np.sum(power_no_dc))
    peak_index = int(np.argmax(power_no_dc))
    peak_power = float(power_no_dc[peak_index])
    dominant_frequency = float(freqs[peak_index])
    nonzero_power = power_no_dc[power_no_dc > 1e-20]
    flatness = float(np.exp(np.mean(np.log(nonzero_power))) / (np.mean(nonzero_power) + 1e-20))
    low_band = float(np.sum(power_no_dc[(freqs >= 30.0) & (freqs <= 250.0)])) / total
    mid_band = float(np.sum(power_no_dc[(freqs > 250.0) & (freqs <= 4000.0)])) / total
    high_band = float(np.sum(power_no_dc[freqs > 4000.0])) / total
    pitch_confidence = clip01((peak_power / total) * 60.0)
    return {
        "dominant_frequency_hz": round(dominant_frequency, 6),
        "pitch_confidence": round(pitch_confidence, 6),
        "spectral_flatness": round(clip01(flatness), 6),
        "low_band_ratio": round(clip01(low_band), 6),
        "mid_band_ratio": round(clip01(mid_band), 6),
        "high_band_ratio": round(clip01(high_band), 6),
    }


def transient_metrics(mono: np.ndarray, sample_rate: int) -> dict[str, float]:
    if mono.size < 1024 or sample_rate <= 0:
        return {"transient_density": 0.0, "crest_factor": 0.0}
    frame = 1024
    hop = 256
    frame_count = 1 + max(0, (mono.size - frame) // hop)
    if frame_count <= 1:
        return {"transient_density": 0.0, "crest_factor": 0.0}
    rms_values = np.empty(frame_count, dtype=np.float64)
    for index in range(frame_count):
        start = index * hop
        chunk = mono[start : start + frame].astype(np.float64)
        rms_values[index] = math.sqrt(float(np.mean(chunk * chunk)) + 1e-20)
    diff = np.diff(rms_values)
    positive = diff[diff > 0.0]
    if positive.size:
        threshold = float(np.mean(positive) + np.std(positive))
        transient_density = float(np.mean(positive > threshold))
    else:
        transient_density = 0.0
    rms_total = math.sqrt(float(np.mean(mono.astype(np.float64) ** 2)) + 1e-20)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    crest_factor = float(peak / (rms_total + 1e-12))
    return {
        "transient_density": round(clip01(transient_density * 5.0), 6),
        "crest_factor": round(crest_factor, 6),
    }


def audio_metrics(path: Path) -> dict[str, Any]:
    mono, sample_rate, channels = read_audio_mono(path)
    duration_sec = float(mono.size / sample_rate) if sample_rate else 0.0
    rms = math.sqrt(float(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    spectrum = spectral_metrics(mono, sample_rate)
    transients = transient_metrics(mono, sample_rate)
    metrics: dict[str, Any] = {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": int(mono.size),
        "duration_sec": round(duration_sec, 6),
        "rms": round(rms, 10),
        "peak": round(peak, 10),
    }
    metrics.update(spectrum)
    metrics.update(transients)
    return metrics


def stem_semantic_result(stem_type: str, metrics: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    blockers: list[str] = []
    reasons: list[str] = []
    rms = as_float(metrics.get("rms"))
    peak = as_float(metrics.get("peak"))
    duration = as_float(metrics.get("duration_sec"))
    pitch_confidence = as_float(metrics.get("pitch_confidence"))
    dominant_frequency_hz = as_float(metrics.get("dominant_frequency_hz"))
    transient_density = as_float(metrics.get("transient_density"))
    crest_factor = as_float(metrics.get("crest_factor"))
    low_band_ratio = as_float(metrics.get("low_band_ratio"))
    mid_band_ratio = as_float(metrics.get("mid_band_ratio"))
    flatness = as_float(metrics.get("spectral_flatness"))

    if duration < 1.0:
        blockers.append("stem_duration_too_short")
    if rms < 1e-4 or peak < 1e-4:
        blockers.append("stem_too_silent")
    if peak > 1.01:
        blockers.append("stem_peak_above_ceiling")

    if stem_type == "melody":
        if pitch_confidence < 0.35:
            blockers.append("melody_low_pitch_confidence")
        if (
            dominant_frequency_hz > 0.0
            and dominant_frequency_hz < MELODY_MIN_DOMINANT_FREQUENCY_HZ
            and low_band_ratio > MELODY_BASS_DOMINANT_LOW_BAND_RATIO
            and mid_band_ratio < MELODY_BASS_DOMINANT_MAX_MID_BAND_RATIO
        ):
            blockers.append("melody_bass_dominant")
        score = 0.30 + 0.45 * pitch_confidence + 0.15 * (1.0 - flatness) + 0.10 * clip01(rms / 0.01)
    elif stem_type == "drums":
        if (transient_density < 0.08 and crest_factor < 6.0) or (pitch_confidence > 0.85 and crest_factor < 4.0):
            blockers.append("drums_low_transient_density")
        score = 0.25 + 0.45 * max(transient_density, clip01((crest_factor - 2.0) / 8.0)) + 0.20 * clip01(peak / 0.2) + 0.10 * clip01(rms / 0.01)
    elif stem_type == "bass":
        if low_band_ratio < 0.35:
            blockers.append("bass_low_band_energy_ratio")
        score = 0.25 + 0.50 * low_band_ratio + 0.15 * clip01(rms / 0.01) + 0.10 * clip01(peak / 0.2)
    elif stem_type in {"guitar", "piano"}:
        transient_support = max(transient_density, clip01((crest_factor - 4.0) / 8.0))
        tonal_support = max(
            pitch_confidence,
            clip01(0.48 * (1.0 - flatness) + 0.34 * mid_band_ratio + 0.18 * transient_support),
        )
        if tonal_support < 0.58 or flatness > 0.55:
            blockers.append(f"{stem_type}_low_tonal_confidence")
        if mid_band_ratio < 0.25:
            blockers.append(f"{stem_type}_low_mid_band_energy_ratio")
        if low_band_ratio > 0.82 and mid_band_ratio < 0.35:
            blockers.append(f"{stem_type}_bass_dominant")
        score = (
            0.25
            + 0.34 * tonal_support
            + 0.18 * mid_band_ratio
            + 0.12 * (1.0 - flatness)
            + 0.06 * clip01(rms / 0.01)
            + 0.05 * clip01(peak / 0.2)
        )
    elif stem_type == "other":
        score = 0.35 + 0.30 * clip01(rms / 0.005) + 0.20 * clip01(peak / 0.05) + 0.15 * (1.0 - clip01(low_band_ratio))
    else:
        blockers.append("unsupported_stem_type")
        score = 0.0

    if not blockers:
        reasons.append(f"{stem_type}_semantic_proxy_pass")
    reasons.extend(blockers)
    if blockers:
        score = min(score, 0.69)
    return round(clip01(score), 6), blockers, reasons


def blocked_row(
    *,
    listen_manifest_json: Path,
    payload: dict[str, Any],
    stem_type: str,
    selected_audio_path: str,
    reason: str,
    safety: dict[str, bool],
) -> dict[str, Any]:
    return {
        "judge_mode": "no_reference_stem_semantic",
        "score_mode": "stem_semantic",
        "decision": "blocked_hard_case",
        "stem_semantic_decision": "blocked_hard_case",
        "stem_type": stem_type,
        "track_id": str(payload.get("track_id") or ""),
        "model_name": str(payload.get("model_name") or ""),
        "selected_audio_path": selected_audio_path,
        "stem_semantic_score": 0.0,
        "stem_semantic_blockers": [reason],
        "reasons": [reason],
        "semantic_quality_verified": False,
        "semantic_quality_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "semantic_judge_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "source_listen_manifest": source_ref(listen_manifest_json),
        "safety": safety,
    }


def mode_specific_blocked_row(
    *,
    listen_manifest_json: Path,
    payload: dict[str, Any],
    stem_type: str,
    selected_audio_path: str,
    reason: str,
    safety: dict[str, bool],
) -> dict[str, Any]:
    row = blocked_row(
        listen_manifest_json=listen_manifest_json,
        payload=payload,
        stem_type=stem_type,
        selected_audio_path=selected_audio_path,
        reason=reason,
        safety=safety,
    )
    row["score_mode"] = stem_type
    row["mode_specific_quality_verified"] = False
    row["mode_specific_judge"] = {
        "judge_mode": "no_reference_mode_specific",
        "score_mode": stem_type,
        "status": "blocked",
        "reason": reason,
    }
    return row


def normalized_mode_specific_candidate_row(payload: dict[str, Any], stem_type: str) -> tuple[dict[str, Any] | None, str]:
    candidate = payload.get("mode_specific_candidate_row")
    if not isinstance(candidate, dict):
        return None, "mode_specific_judge_evidence_missing"
    normalized = dict(candidate)
    normalized["track_id"] = str(normalized.get("track_id") or payload.get("track_id") or "")
    normalized["candidate_id"] = str(
        normalized.get("candidate_id")
        or payload.get("candidate_id")
        or f"{normalized['track_id'] or 'track'}_{stem_type}_pair"
    )
    if not normalized.get("instrumental_path") and normalized.get("minus_path"):
        normalized["instrumental_path"] = normalized["minus_path"]
    required = ["track_id", "candidate_id", "mix_path", "vocal_path", "instrumental_path"]
    if any(not str(normalized.get(key) or "") for key in required):
        return None, "mode_specific_judge_evidence_missing"
    for key in ("mix_path", "vocal_path", "instrumental_path"):
        if not resolve_no_reference_path(str(normalized[key])).exists():
            return None, "mode_specific_judge_evidence_missing"
    return normalized, ""


def mode_specific_judge_row(
    *,
    listen_manifest_json: Path,
    payload: dict[str, Any],
    stem_type: str,
    selected_audio_path: str,
    safety: dict[str, bool],
) -> dict[str, Any]:
    candidate_row, missing_reason = normalized_mode_specific_candidate_row(payload, stem_type)
    if candidate_row is None:
        return mode_specific_blocked_row(
            listen_manifest_json=listen_manifest_json,
            payload=payload,
            stem_type=stem_type,
            selected_audio_path=selected_audio_path,
            reason=missing_reason,
            safety=safety,
        )
    try:
        judged = judge_no_reference_candidate_row(candidate_row, score_mode=stem_type)
    except Exception as exc:
        return mode_specific_blocked_row(
            listen_manifest_json=listen_manifest_json,
            payload=payload,
            stem_type=stem_type,
            selected_audio_path=selected_audio_path,
            reason=f"mode_specific_judge_failed:{type(exc).__name__}",
            safety=safety,
        )
    decision_key = f"{stem_type}_decision"
    score_key = f"{stem_type}_score"
    blockers_key = f"{stem_type}_blockers"
    defect_key = f"{stem_type}_dominant_defect"
    decision = str(judged.get(decision_key) or judged.get("decision") or "blocked_hard_case")
    blockers = [str(item) for item in judged.get(blockers_key) or [] if str(item)]
    reasons = blockers or [str(item) for item in judged.get("reasons") or [] if str(item)] or ["mode_specific_quality_not_verified"]
    verified = decision == "accept_candidate" and not blockers
    return {
        "judge_mode": "no_reference_stem_semantic",
        "score_mode": stem_type,
        "decision": decision,
        "stem_semantic_decision": decision,
        "stem_type": stem_type,
        "track_id": str(payload.get("track_id") or judged.get("track_id") or ""),
        "model_name": str(payload.get("model_name") or ""),
        "selected_audio_path": selected_audio_path,
        "stem_semantic_score": as_float(judged.get(score_key)),
        "stem_semantic_blockers": blockers,
        "reasons": reasons,
        "semantic_quality_verified": verified,
        "semantic_quality_status": "mode_specific_proxy_pass" if verified else "mode_specific_proxy_blocked",
        "mode_specific_quality_verified": verified,
        "mode_specific_judge": {
            "judge_mode": "no_reference_mode_specific",
            "score_mode": stem_type,
            "candidate_id": str(judged.get("candidate_id") or ""),
            "decision": decision,
            "score": as_float(judged.get(score_key)),
            "blockers": blockers,
            "dominant_defect": judged.get(defect_key),
            "mix_path": str(judged.get("mix_path") or ""),
            "vocal_path": str(judged.get("vocal_path") or ""),
            "instrumental_path": str(judged.get("instrumental_path") or ""),
        },
        "stem_semantic_metrics": judged,
        "delivery_allowed": False,
        "presentation_status": "semantic_judge_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "source_listen_manifest": source_ref(listen_manifest_json),
        "safety": safety,
    }


def judge_listen_manifest(listen_manifest_json: Path) -> dict[str, Any]:
    payload = read_json(listen_manifest_json)
    view = status_view_from_manifest(listen_manifest_json)
    raw_stem_type = str(payload.get("stem_type") or payload.get("target_stem") or "").strip().lower()
    stem_type = raw_stem_type if view.stem_type == "unknown" and raw_stem_type else view.stem_type
    selected_audio_path = str(payload.get("selected_audio_path") or payload.get("selected_stem_path") or view.audio_path or "")
    safety = safety_from_manifest(payload)
    safety_reason = safety_violation(safety)
    if safety_reason:
        return blocked_row(
            listen_manifest_json=listen_manifest_json,
            payload=payload,
            stem_type=stem_type,
            selected_audio_path=selected_audio_path,
            reason=f"source_safety_violation:{safety_reason}",
            safety=safety,
        )
    if not view.listenable and not is_semantic_judge_input_manifest(payload, view):
        return blocked_row(
            listen_manifest_json=listen_manifest_json,
            payload=payload,
            stem_type=stem_type,
            selected_audio_path=selected_audio_path,
            reason="listen_manifest_not_accepted",
            safety=safety,
        )
    audio_path = Path(selected_audio_path)
    if not selected_audio_path or not audio_path.exists():
        return blocked_row(
            listen_manifest_json=listen_manifest_json,
            payload=payload,
            stem_type=stem_type,
            selected_audio_path=selected_audio_path,
            reason="audio_path_missing",
            safety=safety,
        )
    if stem_type in MODE_SPECIFIC_STEMS:
        return mode_specific_judge_row(
            listen_manifest_json=listen_manifest_json,
            payload=payload,
            stem_type=stem_type,
            selected_audio_path=selected_audio_path,
            safety=safety,
        )
    if stem_type not in SUPPORTED_STEMS:
        return blocked_row(
            listen_manifest_json=listen_manifest_json,
            payload=payload,
            stem_type=stem_type,
            selected_audio_path=selected_audio_path,
            reason="unsupported_stem_type",
            safety=safety,
        )
    metrics = audio_metrics(audio_path)
    score, blockers, reasons = stem_semantic_result(stem_type, metrics)
    decision = "accept_candidate" if not blockers else "blocked_hard_case"
    return {
        "judge_mode": "no_reference_stem_semantic",
        "score_mode": "stem_semantic",
        "decision": decision,
        "stem_semantic_decision": decision,
        "stem_type": stem_type,
        "track_id": str(payload.get("track_id") or ""),
        "model_name": str(payload.get("model_name") or ""),
        "selected_audio_path": selected_audio_path,
        "stem_semantic_score": score,
        "stem_semantic_blockers": blockers,
        "reasons": reasons,
        "semantic_quality_verified": decision == "accept_candidate",
        "semantic_quality_status": "proxy_semantic_pass" if decision == "accept_candidate" else "proxy_semantic_blocked",
        "stem_semantic_metrics": metrics,
        "delivery_allowed": False,
        "presentation_status": "semantic_judge_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "source_listen_manifest": source_ref(listen_manifest_json),
        "safety": safety,
    }


def discover_listen_manifests(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("no_reference_stem_listen_preview_manifest.json"))


def aggregate_safety(rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for row in rows:
        row_safety = row.get("safety")
        if not isinstance(row_safety, dict):
            continue
        for key in safety:
            safety[key] = bool(safety[key] or row_safety.get(key) is True)
    return safety


def build_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision") or "unknown")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    return {
        "judge_mode": "no_reference_stem_semantic",
        "score_mode": "stem_semantic",
        "candidate_count": len(rows),
        "decision_counts": decision_counts,
        "accepted_candidates": decision_counts.get("accept_candidate", 0),
        "blocked_hard_case": decision_counts.get("blocked_hard_case", 0),
        "semantic_quality_verified_count": sum(1 for row in rows if row.get("semantic_quality_verified") is True),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "stem_semantic_judge_reads_existing_audio_only",
        "safety": aggregate_safety(rows),
    }


def run_stem_semantic_judge(listen_manifest_paths: Sequence[Path]) -> dict[str, Any]:
    rows = [judge_listen_manifest(path) for path in listen_manifest_paths]
    return {
        "rows": rows,
        "summary": build_summary(rows),
    }


def write_summary_md(path: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# No-Reference Stem Semantic Judge Summary",
        "",
        f"- judge_mode: {summary.get('judge_mode')}",
        f"- score_mode: {summary.get('score_mode')}",
        f"- candidate_count: {summary.get('candidate_count')}",
        f"- accepted_candidates: {summary.get('accepted_candidates')}",
        f"- blocked_hard_case: {summary.get('blocked_hard_case')}",
        f"- semantic_quality_verified_count: {summary.get('semantic_quality_verified_count')}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
        f"- acceleration_mode: {summary.get('acceleration_mode')}",
        f"- gpu_inference_skipped: {str(summary.get('gpu_inference_skipped') is True).lower()}",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row.get('stem_type')} "
            f"decision={row.get('decision')} "
            f"score={row.get('stem_semantic_score')} "
            f"blockers={','.join(row.get('stem_semantic_blockers') or [])} "
            f"path={row.get('selected_audio_path')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_root: Path, result: dict[str, Any], source_paths: Sequence[Path], *, elapsed_sec: float) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    summary = dict(result["summary"])
    summary["source_listen_manifests"] = [source_ref(path) for path in source_paths]
    summary["elapsed_sec"] = round(elapsed_sec, 6)
    write_jsonl(output_root / JUDGE_JSONL_NAME, rows)
    write_json(output_root / SUMMARY_JSON_NAME, summary)
    write_summary_md(output_root / SUMMARY_MD_NAME, summary, rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run no-reference semantic proxy judge for non-vocal stems.")
    parser.add_argument("--listen-manifest-json", type=Path, action="append", default=[])
    parser.add_argument("--listen-manifest-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.time()
    output_root = args.output_root
    progress_path = output_root / PROGRESS_JSONL_NAME
    manifests = list(args.listen_manifest_json)
    if args.listen_manifest_root is not None:
        manifests.extend(discover_listen_manifests(args.listen_manifest_root))
    manifests = sorted(dict.fromkeys(manifests))
    append_jsonl(
        progress_path,
        {
            "event": "no_reference_stem_semantic_judge_start",
            "candidate_count": len(manifests),
            "acceleration_mode": "external_only_cpu_dsp_scoring",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "stem_semantic_judge_reads_existing_audio_only",
        },
    )
    rows: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(manifests, start=1):
        row = judge_listen_manifest(manifest_path)
        rows.append(row)
        append_jsonl(
            progress_path,
            {
                "event": "no_reference_stem_semantic_judge_progress",
                "candidate_index": index,
                "candidate_count": len(manifests),
                "stem_type": row.get("stem_type"),
                "decision": row.get("decision"),
                "source_manifest": str(manifest_path),
            },
        )
    result = {"rows": rows, "summary": build_summary(rows)}
    elapsed_sec = time.time() - started
    write_outputs(output_root, result, manifests, elapsed_sec=elapsed_sec)
    append_jsonl(
        progress_path,
        {
            "event": "no_reference_stem_semantic_judge_end",
            "candidate_count": len(rows),
            "elapsed_sec": round(elapsed_sec, 6),
            "decision_counts": result["summary"]["decision_counts"],
        },
    )
    print(
        json.dumps(
            {
                "candidate_count": result["summary"]["candidate_count"],
                "accepted_candidates": result["summary"]["accepted_candidates"],
                "blocked_hard_case": result["summary"]["blocked_hard_case"],
                "semantic_quality_verified_count": result["summary"]["semantic_quality_verified_count"],
                "release_ready": result["summary"]["release_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
