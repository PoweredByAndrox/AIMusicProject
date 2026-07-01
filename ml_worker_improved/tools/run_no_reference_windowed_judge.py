from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.run_no_reference_result_judge import (  # noqa: E402
    apply_minus_detail_gate_override,
    as_list,
    build_minus_musical_detail_preservation_gate,
    clip01,
    judge_audio,
    resample_audio,
    resolve_path,
    stable_unique,
)
from tools.no_reference_score_matrix import pair_compatibility_score, score_candidate, validate_mode  # noqa: E402


METRIC_VERSION = "no_reference_windowed_v2"

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_copy_audio_windows",
]

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
    "audio_windows_exported": False,
}

SEVERE_THRESHOLDS = {
    "bad_mixture_reconstruction": ("mixture_consistency_score", "lt", 0.55),
    "vocal_leakage_in_minus": ("vocal_leakage_in_minus_score", "gt", 0.60),
    "instrumental_leakage_in_vocal": ("instrumental_leakage_in_vocal_score", "gt", 0.72),
    "silence_hallucination": ("silence_hallucination_score", "gt", 0.75),
    "robotic_or_noise_artifact": ("robotic_artifact_score", "gt", 0.82),
    "thin_or_overcleaned_vocal": ("thin_vocal_score", "gt", 0.78),
    "transient_damage": ("transient_damage_score", "gt", 0.82),
    "phase_or_stereo_inconsistency": ("phase_consistency_score", "lt", 0.30),
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def audio_info(path: Path) -> sf.SoundFile:
    return sf.info(str(path))


def audio_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"file_size": int(stat.st_size), "file_mtime": round(float(stat.st_mtime), 6)}


def audio_duration(path: Path) -> float:
    info = audio_info(path)
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


AudioWindowCache = OrderedDict[str, tuple[np.ndarray, int]]


def read_window(path: Path, start_sec: float, duration_sec: float) -> tuple[np.ndarray, int]:
    info = audio_info(path)
    start = max(0, int(round(start_sec * info.samplerate)))
    frames = max(1, int(round(duration_sec * info.samplerate)))
    audio, sample_rate = sf.read(str(path), start=start, frames=frames, always_2d=True, dtype="float32")
    if audio.size == 0:
        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return audio, int(sample_rate)


def read_window_cached(
    path: Path,
    start_sec: float,
    duration_sec: float,
    audio_cache: AudioWindowCache | None,
    *,
    max_cache_entries: int = 96,
) -> tuple[np.ndarray, int]:
    if audio_cache is None:
        return read_window(path, start_sec, duration_sec)
    key = f"{path}|{round(start_sec, 6)}|{round(duration_sec, 6)}"
    if key in audio_cache:
        audio_cache.move_to_end(key)
        return audio_cache[key]
    value = read_window(path, start_sec, duration_sec)
    audio_cache[key] = value
    audio_cache.move_to_end(key)
    while len(audio_cache) > max_cache_entries:
        audio_cache.popitem(last=False)
    return value


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + 1.0e-12)


def coarse_rms_windows(path: Path, window_seconds: float) -> list[tuple[float, float]]:
    info = audio_info(path)
    block_frames = max(1, int(round(window_seconds * info.samplerate)))
    result: list[tuple[float, float]] = []
    start_frame = 0
    for block in sf.blocks(str(path), blocksize=block_frames, always_2d=True, dtype="float32"):
        start_sec = start_frame / float(info.samplerate)
        result.append((round(start_sec, 6), rms(block)))
        start_frame += block.shape[0]
    return result or [(0.0, 0.0)]


def dense_grid_windows(
    *,
    duration: float,
    window_seconds: float,
    window_step_seconds: float,
    max_windows: int,
) -> list[dict[str, float | str]]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if window_step_seconds <= 0:
        raise ValueError("window_step_seconds must be > 0 for dense grid selection")
    max_start = max(0.0, duration - window_seconds)
    starts: list[float] = []
    current = 0.0
    while current < max_start:
        starts.append(round(current, 6))
        current += window_step_seconds
    if not starts or abs(starts[-1] - max_start) > 1.0e-6:
        starts.append(round(max_start, 6))
    unique_starts = list(dict.fromkeys(starts))
    if max_windows > 0:
        unique_starts = unique_starts[:max_windows]
    return [
        {
            "kind": f"grid_{index:04d}",
            "start_sec": round(float(start), 6),
            "duration_sec": round(window_seconds, 6),
        }
        for index, start in enumerate(unique_starts)
    ]


def select_windows(
    mix_path: Path,
    window_seconds: float = 12.0,
    max_windows: int = 5,
    window_step_seconds: float = 0.0,
) -> list[dict[str, float | str]]:
    duration = audio_duration(mix_path)
    if duration <= 0:
        return [{"kind": "full", "start_sec": 0.0, "duration_sec": 0.0}]
    if duration <= window_seconds:
        return [{"kind": "full", "start_sec": 0.0, "duration_sec": round(duration, 6)}]
    if window_step_seconds > 0:
        return dense_grid_windows(
            duration=duration,
            window_seconds=window_seconds,
            window_step_seconds=window_step_seconds,
            max_windows=max_windows,
        )

    coarse = coarse_rms_windows(mix_path, window_seconds)
    max_rms = max(value for _, value in coarse)
    active_threshold = max(max_rms * 0.08, 1.0e-5)
    first_active = next((start for start, value in coarse if value >= active_threshold), 0.0)
    high_energy = max(coarse, key=lambda item: (item[1], -item[0]))[0]
    low_energy = min(coarse, key=lambda item: (item[1], item[0]))[0]
    max_start = max(0.0, duration - window_seconds)

    candidates = [
        ("start", first_active),
        ("middle", max_start / 2.0),
        ("tail", max_start),
        ("high_energy", high_energy),
        ("low_energy", low_energy),
    ]
    windows = [
        {
            "kind": kind,
            "start_sec": round(max(0.0, min(float(start), max_start)), 6),
            "duration_sec": round(window_seconds, 6),
        }
        for kind, start in candidates
    ]
    return windows[: max(1, max_windows)]


def metric_cache_key(row: dict[str, Any], window: dict[str, Any]) -> str:
    mix_path = resolve_path(row["mix_path"])
    vocal_path = resolve_path(row["vocal_path"])
    instrumental_path = resolve_path(row.get("instrumental_path") or row.get("minus_path"))
    payload = {
        "metric_version": METRIC_VERSION,
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "mix_path": str(row.get("mix_path") or ""),
        "vocal_path": str(row.get("vocal_path") or ""),
        "instrumental_path": str(row.get("instrumental_path") or row.get("minus_path") or ""),
        "window_start_sec": window["start_sec"],
        "window_duration_sec": window["duration_sec"],
        "metric_name": "window_metrics",
        "mix_stat": audio_stat(mix_path),
        "vocal_stat": audio_stat(vocal_path),
        "instrumental_stat": audio_stat(instrumental_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {str(row.get("cache_key")): row for row in iter_jsonl(path) if row.get("cache_key")}


def cache_row(row: dict[str, Any], window: dict[str, Any], metrics: dict[str, Any], cache_key: str) -> dict[str, Any]:
    mix_path = resolve_path(row["mix_path"])
    stat = audio_stat(mix_path)
    return {
        "cache_key": cache_key,
        "audio_path": str(row["mix_path"]),
        "file_size": stat["file_size"],
        "file_mtime": stat["file_mtime"],
        "metric_version": METRIC_VERSION,
        "window_start_sec": window["start_sec"],
        "window_duration_sec": window["duration_sec"],
        "metric_name": "window_metrics",
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "vocal_path": str(row.get("vocal_path") or ""),
        "instrumental_path": str(row.get("instrumental_path") or row.get("minus_path") or ""),
        "metrics": metrics,
    }


def judge_window(
    row: dict[str, Any],
    window: dict[str, Any],
    audio_cache: AudioWindowCache | None = None,
) -> dict[str, Any]:
    mix_path = resolve_path(row["mix_path"])
    vocal_path = resolve_path(row["vocal_path"])
    instrumental_path = resolve_path(row.get("instrumental_path") or row.get("minus_path"))
    mix, sample_rate = read_window_cached(mix_path, float(window["start_sec"]), float(window["duration_sec"]), audio_cache)
    vocal, vocal_rate = read_window_cached(vocal_path, float(window["start_sec"]), float(window["duration_sec"]), audio_cache)
    instrumental, instrumental_rate = read_window_cached(instrumental_path, float(window["start_sec"]), float(window["duration_sec"]), audio_cache)
    vocal = resample_audio(vocal, vocal_rate, sample_rate)
    instrumental = resample_audio(instrumental, instrumental_rate, sample_rate)
    return judge_audio(mix, vocal, instrumental, sample_rate)


def aggregate_vocal_activity_confidence(activity_values: Sequence[float]) -> float:
    if not activity_values:
        return 0.0
    ordered = sorted(clip01(float(value)) for value in activity_values)
    if len(ordered) < 4:
        return ordered[0]
    return ordered[1]


def aggregate_metrics(window_metrics: list[dict[str, Any]], *, score_mode: str = "both") -> dict[str, Any]:
    score_mode = validate_mode(score_mode)

    def values(key: str) -> list[float]:
        return [float(metrics.get(key) or 0.0) for metrics in window_metrics]

    severe_reasons = stable_unique(reason for metrics in window_metrics for reason in as_list(metrics.get("reasons")))
    severe_reasons = [reason for reason in severe_reasons if reason != "no_reference_proxy_passed"]
    mixture = min(values("mixture_consistency_score"))
    phase = min(values("phase_consistency_score"))
    mix_phase = min(values("mix_phase_consistency_score")) if any("mix_phase_consistency_score" in metrics for metrics in window_metrics) else phase
    vocal_phase = min(values("vocal_phase_consistency_score")) if any("vocal_phase_consistency_score" in metrics for metrics in window_metrics) else phase
    instrumental_phase = (
        min(values("instrumental_phase_consistency_score"))
        if any("instrumental_phase_consistency_score" in metrics for metrics in window_metrics)
        else phase
    )
    pair_phase = min(values("pair_phase_consistency_score")) if any("pair_phase_consistency_score" in metrics for metrics in window_metrics) else phase
    vocal_activity_values = values("vocal_activity_confidence")
    vocal_activity = aggregate_vocal_activity_confidence(vocal_activity_values)
    vocal_leakage = max(values("vocal_leakage_in_minus_score"))
    inst_leakage = max(values("instrumental_leakage_in_vocal_score"))
    hallucination = max(values("silence_hallucination_score"))
    robotic = max(values("robotic_artifact_score"))
    thin = max(values("thin_vocal_score"))
    transient = max(values("transient_damage_score"))
    overall = clip01(
        0.30 * mixture
        + 0.14 * (1.0 - vocal_leakage)
        + 0.12 * (1.0 - inst_leakage)
        + 0.10 * vocal_activity
        + 0.09 * (1.0 - hallucination)
        + 0.08 * (1.0 - robotic)
        + 0.07 * (1.0 - thin)
        + 0.05 * (1.0 - transient)
        + 0.05 * phase
    )
    confidence = clip01(min(values("confidence")) if values("confidence") else 0.0)
    reasons = stable_unique(severe_reasons or ["no_reference_proxy_passed"])
    aggregated = {
        "mixture_consistency_score": mixture,
        "vocal_leakage_in_minus_score": vocal_leakage,
        "instrumental_leakage_in_vocal_score": inst_leakage,
        "vocal_activity_confidence": vocal_activity,
        "silence_hallucination_score": hallucination,
        "robotic_artifact_score": robotic,
        "thin_vocal_score": thin,
        "transient_damage_score": transient,
        "phase_consistency_score": phase,
        "mix_phase_consistency_score": mix_phase,
        "vocal_phase_consistency_score": vocal_phase,
        "instrumental_phase_consistency_score": instrumental_phase,
        "pair_phase_consistency_score": pair_phase,
        "overall_no_reference_score": overall,
        "confidence": confidence,
        "reasons": reasons,
        "vocal_activity_min": min(vocal_activity_values) if vocal_activity_values else 0.0,
        "vocal_low_activity_window_ratio": (
            sum(1 for value in vocal_activity_values if value < 0.25) / len(vocal_activity_values)
            if vocal_activity_values
            else 0.0
        ),
        "vocal_score": clip01(1.0 - max(inst_leakage, hallucination * 0.70, thin * 0.45)),
        "minus_score": clip01(1.0 - vocal_leakage),
        "artifact_score": clip01(1.0 - max(robotic, transient, max(0.0, 0.55 - phase))),
        "consistency_score": clip01(0.70 * mixture + 0.30 * phase),
        "defect_reduction_score": clip01(1.0 - max(vocal_leakage, inst_leakage, hallucination, robotic, transient, max(0.0, 0.70 - mixture))),
    }
    vocal_result = score_candidate(aggregated, "vocal")
    minus_result = score_candidate(aggregated, "minus")
    pair_result = pair_compatibility_score(aggregated, aggregated)
    aggregated.update(vocal_result)
    aggregated.update(minus_result)
    aggregated.update(pair_result)
    if score_mode == "minus":
        aggregated["minus_musical_detail_preservation_gate"] = build_minus_musical_detail_preservation_gate(aggregated)
        apply_minus_detail_gate_override(aggregated)
    if score_mode == "vocal":
        decision = str(vocal_result["vocal_decision"])
        reasons = stable_unique(as_list(vocal_result.get("vocal_blockers")) or reasons)
    elif score_mode == "minus":
        decision = str(aggregated.get("minus_decision") or minus_result["minus_decision"])
        reasons = stable_unique(as_list(aggregated.get("minus_blockers")) or reasons)
    else:
        if severe_reasons:
            decision = "blocked_hard_case"
        elif pair_result["pair_decision"] == "pair_compatible":
            decision = "accept_candidate"
        elif pair_result["pair_decision"] == "needs_pair_reconciliation":
            decision = "needs_pair_reconciliation"
        else:
            decision = "blocked_hard_case"
            pair_blockers = as_list(pair_result.get("pair_blockers"))
            reasons = stable_unique(reasons + (pair_blockers or ["low_no_reference_score"]))
    aggregated["score_mode"] = score_mode
    aggregated["decision"] = decision
    aggregated["reasons"] = reasons
    return aggregated


def window_metric_summary(window: dict[str, Any], metrics: dict[str, Any], *, score_mode: str = "both") -> dict[str, Any]:
    inst_leakage = float(metrics.get("instrumental_leakage_in_vocal_score") or 0.0)
    hallucination = float(metrics.get("silence_hallucination_score") or 0.0)
    thin = float(metrics.get("thin_vocal_score") or 0.0)
    vocal_leakage = float(metrics.get("vocal_leakage_in_minus_score") or 0.0)
    robotic = float(metrics.get("robotic_artifact_score") or 0.0)
    transient = float(metrics.get("transient_damage_score") or 0.0)
    mixture = float(metrics.get("mixture_consistency_score") or 0.0)
    phase = float(metrics.get("phase_consistency_score") or 1.0)
    mix_phase = float(metrics.get("mix_phase_consistency_score", phase) or phase)
    vocal_phase = float(metrics.get("vocal_phase_consistency_score", phase) or phase)
    instrumental_phase = float(metrics.get("instrumental_phase_consistency_score", phase) or phase)
    pair_phase = float(metrics.get("pair_phase_consistency_score", phase) or phase)
    vocal_score = clip01(1.0 - max(inst_leakage, hallucination * 0.70, thin * 0.45))
    minus_score = clip01(1.0 - vocal_leakage)
    artifact_score = clip01(1.0 - max(robotic, transient, max(0.0, 0.55 - phase)))
    consistency_score = clip01(0.70 * mixture + 0.30 * phase)
    defect_reduction_score = clip01(1.0 - max(vocal_leakage, inst_leakage, hallucination, robotic, transient, max(0.0, 0.70 - mixture)))
    window_fitness = clip01(
        0.30 * min(vocal_score, minus_score)
        + 0.25 * artifact_score
        + 0.25 * consistency_score
        + 0.20 * defect_reduction_score
    )
    summary = {
        "kind": str(window.get("kind") or ""),
        "start_sec": float(window.get("start_sec") or 0.0),
        "duration_sec": float(window.get("duration_sec") or 0.0),
        "mixture_consistency_score": round(mixture, 6),
        "vocal_leakage_in_minus_score": round(vocal_leakage, 6),
        "instrumental_leakage_in_vocal_score": round(inst_leakage, 6),
        "vocal_activity_confidence": round(float(metrics.get("vocal_activity_confidence") or 0.0), 6),
        "silence_hallucination_score": round(hallucination, 6),
        "robotic_artifact_score": round(robotic, 6),
        "thin_vocal_score": round(thin, 6),
        "transient_damage_score": round(transient, 6),
        "phase_consistency_score": round(phase, 6),
        "mix_phase_consistency_score": round(mix_phase, 6),
        "vocal_phase_consistency_score": round(vocal_phase, 6),
        "instrumental_phase_consistency_score": round(instrumental_phase, 6),
        "pair_phase_consistency_score": round(pair_phase, 6),
        "overall_no_reference_score": round(float(metrics.get("overall_no_reference_score") or 0.0), 6),
        "confidence": round(float(metrics.get("confidence") or 0.0), 6),
        "decision": str(metrics.get("decision") or ""),
        "reasons": as_list(metrics.get("reasons")),
        "vocal_score": vocal_score,
        "minus_score": minus_score,
        "artifact_score": artifact_score,
        "consistency_score": consistency_score,
        "defect_reduction_score": defect_reduction_score,
        "window_fitness": window_fitness,
    }
    summary.update(score_candidate(summary, "vocal"))
    summary.update(score_candidate(summary, "minus"))
    summary.update(pair_compatibility_score(summary, summary))
    score_mode = validate_mode(score_mode)
    if score_mode == "minus":
        summary["minus_musical_detail_preservation_gate"] = build_minus_musical_detail_preservation_gate(summary)
        apply_minus_detail_gate_override(summary)
    if score_mode == "vocal":
        summary["window_fitness"] = (
            0.0
            if summary.get("vocal_decision") == "blocked_hard_case"
            else summary["vocal_score"]
        )
    elif score_mode == "minus":
        summary["window_fitness"] = 0.0 if summary.get("minus_decision") == "blocked_hard_case" else summary["minus_score"]
    else:
        summary["window_fitness"] = (
            0.0
            if summary.get("pair_decision") == "blocked_pair_incompatible"
            else summary["pair_compatibility_score"]
        )
    pair_window_result = pair_compatibility_score(summary, summary)
    summary["pair_window_fitness"] = (
        0.0
        if pair_window_result.get("pair_decision") == "blocked_pair_incompatible"
        else pair_window_result["pair_compatibility_score"]
    )
    summary["score_mode"] = score_mode
    return summary


def judge_candidate_windowed(
    row: dict[str, Any],
    *,
    cache: dict[str, dict[str, Any]],
    cache_rows: list[dict[str, Any]],
    window_cache: dict[str, list[dict[str, Any]]],
    audio_cache: AudioWindowCache | None = None,
    window_seconds: float,
    max_windows_per_candidate: int,
    window_step_seconds: float = 0.0,
    score_mode: str = "both",
) -> tuple[dict[str, Any], int, int, int]:
    score_mode = validate_mode(score_mode)
    required = ["track_id", "candidate_id", "mix_path", "vocal_path"]
    missing = [key for key in required if not row.get(key)]
    if not row.get("instrumental_path") and not row.get("minus_path"):
        missing.append("instrumental_path")
    if missing:
        raise ValueError(f"Candidate row is missing required windowed no-reference fields {missing}: {row}")

    mix_path = resolve_path(row["mix_path"])
    mix_key = json.dumps(
        {
            "mix_path": str(mix_path),
            "window_seconds": round(float(window_seconds), 6),
            "max_windows_per_candidate": int(max_windows_per_candidate),
            "window_step_seconds": round(float(window_step_seconds), 6),
        },
        sort_keys=True,
    )
    if mix_key not in window_cache:
        window_cache[mix_key] = select_windows(
            mix_path,
            window_seconds=window_seconds,
            max_windows=max_windows_per_candidate,
            window_step_seconds=window_step_seconds,
        )
    windows = window_cache[mix_key]
    metrics_by_window: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    for window in windows:
        key = metric_cache_key(row, window)
        if key in cache:
            metrics = dict(cache[key]["metrics"])
            cache_hits += 1
        else:
            metrics = judge_window(row, window, audio_cache)
            entry = cache_row(row, window, metrics, key)
            cache[key] = entry
            cache_rows.append(entry)
            cache_misses += 1
        metrics_by_window.append(metrics)
        window_metric_rows.append(window_metric_summary(window, metrics, score_mode=score_mode))
    aggregated = aggregate_metrics(metrics_by_window, score_mode=score_mode)
    provenance_fields = (
        "stem_type",
        "source_stem",
        "base_candidate_id",
        "alternate_candidate_id",
        "window_start_sec",
        "window_duration_sec",
        "selected_vocal_path",
        "selected_minus_path",
        "route_id",
        "route_family",
        "model_id",
        "vocal_route_id",
        "minus_route_id",
        "pair_route_id",
        "vocal_model_id",
        "minus_model_id",
    )
    provenance = {
        field: row[field]
        for field in provenance_fields
        if row.get(field) not in (None, "")
    }
    judged = {
        "track_id": str(row["track_id"]),
        "candidate_id": str(row["candidate_id"]),
        "mix_path": str(row["mix_path"]),
        "vocal_path": str(row["vocal_path"]),
        "instrumental_path": str(row.get("instrumental_path") or row.get("minus_path")),
        "minus_path": str(row.get("minus_path") or row.get("instrumental_path")),
        "candidate_manifest": str(row.get("candidate_manifest") or row.get("source_manifest") or ""),
        "source_manifest": str(row.get("source_manifest") or row.get("candidate_manifest") or ""),
        "judge_mode": "no_reference_windowed",
        "score_mode": score_mode,
        "window_seconds": float(window_seconds),
        "max_windows_per_candidate": int(max_windows_per_candidate),
        "window_step_seconds": float(window_step_seconds),
        "window_count": len(windows),
        "windows": windows,
        "window_metrics": window_metric_rows,
        **provenance,
        **aggregated,
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + as_list(row.get("forbidden_actions"))),
    }
    return judged, len(windows), cache_hits, cache_misses


def judge_candidate_rows_windowed(
    rows: list[dict[str, Any]],
    *,
    cache_jsonl: Path | None,
    window_seconds: float = 12.0,
    max_windows_per_candidate: int = 5,
    window_step_seconds: float = 0.0,
    score_mode: str = "both",
) -> dict[str, Any]:
    score_mode = validate_mode(score_mode)
    cache = load_cache(cache_jsonl)
    cache_rows = list(cache.values())
    window_cache: dict[str, list[dict[str, Any]]] = {}
    audio_cache: AudioWindowCache = OrderedDict()
    judged_rows: list[dict[str, Any]] = []
    total_windows = 0
    total_hits = 0
    total_misses = 0
    for row in sorted(rows, key=lambda item: (str(item.get("track_id", "")), str(item.get("candidate_id", "")), str(item.get("vocal_path", "")), str(item.get("instrumental_path") or item.get("minus_path") or ""))):
        judged, windows, hits, misses = judge_candidate_windowed(
            row,
            cache=cache,
            cache_rows=cache_rows,
            window_cache=window_cache,
            audio_cache=audio_cache,
            window_seconds=window_seconds,
            max_windows_per_candidate=max_windows_per_candidate,
            window_step_seconds=window_step_seconds,
            score_mode=score_mode,
        )
        judged_rows.append(judged)
        total_windows += windows
        total_hits += hits
        total_misses += misses
    summary = build_summary(judged_rows, total_windows, total_hits, total_misses)
    result = {"rows": judged_rows, "summary": summary, "cache_rows": sorted(cache.values(), key=lambda entry: str(entry["cache_key"]))}
    if cache_jsonl is not None:
        write_jsonl(cache_jsonl, result["cache_rows"])
    return result


def path_fingerprint(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {"path": ""}
    try:
        path = resolve_path(path_text)
        stat = path.stat()
        return {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {"path": str(path_text), "missing": True}


def candidate_checkpoint_key(
    row: dict[str, Any],
    *,
    score_mode: str | None = None,
    window_seconds: float | None = None,
    max_windows_per_candidate: int | None = None,
    window_step_seconds: float | None = None,
) -> str:
    resolved_score_mode = validate_mode(score_mode or str(row.get("score_mode") or "both"))
    resolved_window_seconds = float(window_seconds if window_seconds is not None else row.get("window_seconds") or 0.0)
    resolved_max_windows = int(max_windows_per_candidate if max_windows_per_candidate is not None else row.get("max_windows_per_candidate") or 0)
    resolved_window_step = float(window_step_seconds if window_step_seconds is not None else row.get("window_step_seconds") or 0.0)
    payload = {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "score_mode": resolved_score_mode,
        "window_seconds": round(resolved_window_seconds, 6),
        "max_windows_per_candidate": resolved_max_windows,
        "window_step_seconds": round(resolved_window_step, 6),
        "metric_version": METRIC_VERSION,
        "mix_path": path_fingerprint(str(row.get("mix_path") or "")),
        "vocal_path": path_fingerprint(str(row.get("vocal_path") or "")),
        "instrumental_path": path_fingerprint(str(row.get("instrumental_path") or row.get("minus_path") or "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def sorted_candidate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            str(item.get("track_id", "")),
            str(item.get("candidate_id", "")),
            str(item.get("vocal_path", "")),
            str(item.get("instrumental_path") or item.get("minus_path") or ""),
        ),
    )


def load_existing_output_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def write_outputs_streaming(
    rows: list[dict[str, Any]],
    *,
    output_jsonl: Path,
    summary_md: Path,
    cache_jsonl: Path,
    window_seconds: float,
    max_windows_per_candidate: int,
    window_step_seconds: float = 0.0,
    score_mode: str = "both",
) -> dict[str, Any]:
    score_mode = validate_mode(score_mode)
    cache = load_cache(cache_jsonl)
    cache_rows = list(cache.values())
    window_cache: dict[str, list[dict[str, Any]]] = {}
    audio_cache: AudioWindowCache = OrderedDict()
    existing_rows = load_existing_output_rows(output_jsonl)
    existing_keys = {candidate_checkpoint_key(row) for row in existing_rows}
    judged_rows: list[dict[str, Any]] = []
    total_windows = 0
    total_hits = 0
    total_misses = 0
    started = time.monotonic()
    ordered_rows = sorted_candidate_rows(rows)
    pending_rows = [
        row
        for row in ordered_rows
        if candidate_checkpoint_key(
            row,
            score_mode=score_mode,
            window_seconds=window_seconds,
            max_windows_per_candidate=max_windows_per_candidate,
            window_step_seconds=window_step_seconds,
        )
        not in existing_keys
    ]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not output_jsonl.exists():
        output_jsonl.write_text("", encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "no_reference_windowed_judge_start",
                "route": "cached_windowed_no_reference_judge",
                "total_candidates": len(ordered_rows),
                "existing_checkpoint_rows": len(existing_rows),
                "pending_candidates": len(pending_rows),
                "window_seconds": window_seconds,
                "max_windows_per_candidate": max_windows_per_candidate,
                "window_step_seconds": window_step_seconds,
                "score_mode": score_mode,
                "output_jsonl": str(output_jsonl),
                "summary_md": str(summary_md),
                "cache_jsonl": str(cache_jsonl),
                "acceleration_mode": "external_only_cpu_dsp_scoring",
                "gpu_inference_skipped": True,
                "gpu_inference_skip_reason": "no_reference_windowed_judge_reads_existing_audio_only",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    for index, row in enumerate(pending_rows, start=1):
        judged, windows, hits, misses = judge_candidate_windowed(
            row,
            cache=cache,
            cache_rows=cache_rows,
            window_cache=window_cache,
            audio_cache=audio_cache,
            window_seconds=window_seconds,
            max_windows_per_candidate=max_windows_per_candidate,
            window_step_seconds=window_step_seconds,
            score_mode=score_mode,
        )
        judged["window_cache_hits"] = hits
        judged["window_cache_misses"] = misses
        judged_rows.append(judged)
        total_windows += windows
        total_hits += hits
        total_misses += misses
        append_jsonl(output_jsonl, judged)
        write_jsonl(cache_jsonl, sorted(cache.values(), key=lambda entry: str(entry["cache_key"])))
        print(
            json.dumps(
                {
                    "event": "no_reference_windowed_judge_progress",
                    "index": index,
                    "total": len(pending_rows),
                    "track_id": judged["track_id"],
                    "candidate_id": judged["candidate_id"],
                    "window_count": windows,
                    "cache_hits": hits,
                    "cache_misses": misses,
                    "elapsed_sec": round(time.monotonic() - started, 3),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    all_rows = existing_rows + judged_rows
    if all_rows and all("window_cache_hits" in row and "window_cache_misses" in row for row in all_rows):
        summary_windows = sum(int(row.get("window_count") or 0) for row in all_rows)
        summary_hits = sum(int(row.get("window_cache_hits") or 0) for row in all_rows)
        summary_misses = sum(int(row.get("window_cache_misses") or 0) for row in all_rows)
    else:
        summary_windows = total_windows
        summary_hits = total_hits
        summary_misses = total_misses
    summary = build_summary(all_rows, summary_windows, summary_hits, summary_misses)
    if pending_rows or not summary_md.exists():
        write_summary_md(summary_md, summary)
    print(
        json.dumps(
            {
                "event": "no_reference_windowed_judge_end",
                "route": "cached_windowed_no_reference_judge",
                "candidates_judged": len(all_rows),
                "new_candidates_judged": len(judged_rows),
                "existing_checkpoint_rows": len(existing_rows),
                "windows_evaluated": summary_windows,
                "cache_hits": summary_hits,
                "cache_misses": summary_misses,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "output_jsonl": str(output_jsonl),
                "cache_jsonl": str(cache_jsonl),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return {"rows": all_rows, "summary": summary, "cache_rows": sorted(cache.values(), key=lambda entry: str(entry["cache_key"]))}


def build_summary(rows: list[dict[str, Any]], windows: int, cache_hits: int, cache_misses: int) -> dict[str, Any]:
    decisions = Counter(str(row.get("decision") or "") for row in rows)
    reasons = Counter(reason for row in rows for reason in as_list(row.get("reasons")))
    return {
        "candidates_judged": len(rows),
        "windows_evaluated": windows,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "average_windows_per_candidate": round(windows / len(rows), 6) if rows else 0.0,
        "decision_counts": dict(sorted(decisions.items())),
        "top_failure_reasons": dict(reasons.most_common(10)),
        "safety": dict(SAFETY_FLAGS),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Windowed Judge Summary",
        "",
        f"- candidates_judged: {summary['candidates_judged']}",
        f"- windows_evaluated: {summary['windows_evaluated']}",
        f"- cache_hits: {summary['cache_hits']}",
        f"- cache_misses: {summary['cache_misses']}",
        f"- average_windows_per_candidate: {summary['average_windows_per_candidate']}",
        "",
        "## Decisions",
        "",
    ]
    for key, value in summary["decision_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Failure Reasons", ""])
    for key, value in summary["top_failure_reasons"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], *, output_jsonl: Path, summary_md: Path, cache_jsonl: Path) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_summary_md(summary_md, result["summary"])
    write_jsonl(cache_jsonl, result["cache_rows"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cached windowed no-reference judging for generated vocal/minus candidates.")
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--cache-jsonl", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--max-windows-per-candidate", type=int, default=5)
    parser.add_argument("--window-step-seconds", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--score-mode", choices=("vocal", "minus", "both"), default="both")
    args = parser.parse_args(argv)
    rows = list(iter_jsonl(args.candidates_jsonl))
    if args.max_candidates > 0:
        rows = rows[: args.max_candidates]
    result = write_outputs_streaming(
        rows,
        output_jsonl=args.output_jsonl,
        summary_md=args.summary_md,
        cache_jsonl=args.cache_jsonl,
        window_seconds=args.window_seconds,
        max_windows_per_candidate=args.max_windows_per_candidate,
        window_step_seconds=args.window_step_seconds,
        score_mode=args.score_mode,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
