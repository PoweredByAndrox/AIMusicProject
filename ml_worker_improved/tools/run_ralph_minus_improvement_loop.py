from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from scipy import signal

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


EPSILON = 1e-12
ROUTE_NAME = "ralph_minus_improvement_loop"
PROGRESS_LOG_NAME = "ralph_minus_improvement_progress.jsonl"
WORD_AUDIT_MODEL_ENV = "AIMUSIC_WORD_AUDIT_MODEL"
DEFAULT_WORD_AUDIT_MODEL = "base.en"


@dataclass(frozen=True)
class MinusCandidateRecipe:
    name: str
    tool_path: Path
    output_artifact_key: str
    arguments: tuple[str, ...]
    listen_label: str


@dataclass(frozen=True)
class MinusAcceptancePolicy:
    minimum_delta_rms_db: float = -48.0
    minimum_average_vocal_improvement_db: float = 0.12
    minimum_peak_corr_improvement: float = 0.0015
    maximum_rms_dip_db: float = -0.12
    maximum_presence_dip_db: float = -0.15
    maximum_air_dip_db: float = -0.12
    maximum_rms_boost_db: float = 6.0
    maximum_presence_boost_db: float = 8.0
    maximum_air_boost_db: float = 8.0


def default_candidate_recipes(
    *,
    focus_windows_jsonl: str = "",
    focus_track_id: str = "",
    focus_candidate_id: str = "",
    max_focus_windows: int = 16,
    focus_window_count: int = 0,
    focus_include_low_confidence_accepted: bool = False,
    focus_max_accept_probability: float = 0.0,
    focus_min_block_probability: float = 0.0,
) -> tuple[MinusCandidateRecipe, ...]:
    recipes: list[MinusCandidateRecipe] = [
        MinusCandidateRecipe(
            name="actual_masked_residue_guarded",
            tool_path=TOOLS_DIR / "apply_actual_masked_residue_minus_cleanup.py",
            output_artifact_key="minus_actual_masked_residue_flac",
            listen_label="02_candidate_product_minus.flac",
            arguments=(
                "--strength",
                "0.18",
                "--mask-power",
                "1.35",
                "--base-overlap-power",
                "1.5",
                "--smoothing-sec",
                "0.10",
                "--include-low-priority-windows",
                "--min-low-priority-score",
                "0.12",
            ),
        ),
        MinusCandidateRecipe(
            name="windowed_guarded",
            tool_path=TOOLS_DIR / "apply_windowed_hybrid_minus_cleanup.py",
            output_artifact_key="minus_windowed_hybrid_flac",
            listen_label="02_candidate_product_minus.flac",
            arguments=(
                "--strength",
                "0.90",
                "--mask-power",
                "1.35",
                "--guide-power",
                "1.20",
                "--guide-floor",
                "0.12",
                "--score-percentile",
                "70",
                "--gate-smooth-width",
                "51",
                "--gate-apply-threshold",
                "0.14",
            ),
        ),
        MinusCandidateRecipe(
            name="windowed_stronger",
            tool_path=TOOLS_DIR / "apply_windowed_hybrid_minus_cleanup.py",
            output_artifact_key="minus_windowed_hybrid_flac",
            listen_label="02_candidate_product_minus.flac",
            arguments=(
                "--strength",
                "0.96",
                "--mask-power",
                "1.25",
                "--guide-power",
                "1.20",
                "--guide-floor",
                "0.24",
                "--score-percentile",
                "62",
                "--gate-smooth-width",
                "65",
                "--gate-apply-threshold",
                "0.10",
            ),
        ),
        MinusCandidateRecipe(
            name="hybrid_balanced",
            tool_path=TOOLS_DIR / "apply_hybrid_minus_guided_cleanup.py",
            output_artifact_key="minus_hybrid_flac",
            listen_label="02_candidate_product_minus.flac",
            arguments=(
                "--strength",
                "0.88",
                "--mask-power",
                "1.25",
                "--guide-power",
                "1.15",
                "--activity-threshold",
                "0.04",
            ),
        ),
        MinusCandidateRecipe(
            name="hybrid_detail_safe",
            tool_path=TOOLS_DIR / "apply_hybrid_minus_guided_cleanup.py",
            output_artifact_key="minus_hybrid_flac",
            listen_label="02_candidate_product_minus.flac",
            arguments=(
                "--strength",
                "0.28",
                "--mask-power",
                "1.25",
                "--guide-power",
                "1.15",
                "--activity-threshold",
                "0.04",
            ),
        ),
    ]
    if focus_windows_jsonl and focus_window_count > 0:
        focus_args = [
            "--focus-windows-jsonl",
            str(focus_windows_jsonl),
            "--focus-track-id",
            str(focus_track_id or ""),
            "--focus-candidate-id",
            str(focus_candidate_id or ""),
            "--max-focus-windows",
            str(int(max_focus_windows)),
        ]
        if focus_include_low_confidence_accepted:
            focus_args.extend(
                [
                    "--focus-include-low-confidence-accepted",
                    "--focus-max-accept-probability",
                    str(float(focus_max_accept_probability)),
                    "--focus-min-block-probability",
                    str(float(focus_min_block_probability)),
                ]
            )
        recipes.append(
            MinusCandidateRecipe(
                name="focus_window_guarded",
                tool_path=TOOLS_DIR / "apply_focus_window_minus_cleanup.py",
                output_artifact_key="minus_focus_window_cleanup_flac",
                listen_label="02_candidate_product_minus.flac",
                arguments=(
                    *focus_args,
                    "--strength",
                    "0.025",
                    "--mask-power",
                    "1.35",
                    "--smoothing-sec",
                    "0.08",
                ),
            )
        )
    return tuple(recipes)


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


def calculate_band_ratio_db(
    candidate_window: np.ndarray,
    base_window: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> float:
    segment_length = min(4096, candidate_window.shape[0])
    frequencies, candidate_power = signal.welch(
        candidate_window,
        fs=sample_rate,
        nperseg=segment_length,
    )
    _, base_power = signal.welch(base_window, fs=sample_rate, nperseg=segment_length)
    band_mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(band_mask):
        return 0.0
    return float(
        10.0
        * math.log10(
            (float(candidate_power[band_mask].mean()) + EPSILON)
            / (float(base_power[band_mask].mean()) + EPSILON)
        )
    )


def calculate_windowed_minus_summary(
    *,
    candidate_audio: np.ndarray,
    base_audio: np.ndarray,
    vocal_audio: np.ndarray,
    sample_rate: int,
    window_sec: float = 5.0,
) -> dict[str, Any]:
    candidate_audio, base_audio, vocal_audio = match_length(
        [candidate_audio, base_audio, vocal_audio]
    )
    candidate_mono = candidate_audio.mean(axis=1)
    base_mono = base_audio.mean(axis=1)
    vocal_mono = vocal_audio.mean(axis=1)
    window_size = max(int(sample_rate * window_sec), 1)

    windows: list[dict[str, float]] = []
    for start in range(0, candidate_mono.shape[0], window_size):
        end = min(candidate_mono.shape[0], start + window_size)
        if end - start < sample_rate:
            continue
        candidate_window = candidate_mono[start:end]
        base_window = base_mono[start:end]
        vocal_window = vocal_mono[start:end]

        candidate_norm = float(np.linalg.norm(candidate_window))
        vocal_norm = float(np.linalg.norm(vocal_window))
        vocal_corr = abs(
            float(
                np.dot(candidate_window, vocal_window)
                / (candidate_norm * vocal_norm + EPSILON)
            )
        )
        vocal_projection_scale = float(
            np.dot(candidate_window, vocal_window)
            / (np.dot(vocal_window, vocal_window) + EPSILON)
        )
        vocal_projection = vocal_projection_scale * vocal_window
        vocal_projection_ratio_db = float(
            20.0
            * math.log10(
                (float(np.sqrt(np.mean(vocal_projection * vocal_projection))) + EPSILON)
                / (float(np.sqrt(np.mean(candidate_window * candidate_window))) + EPSILON)
            )
        )
        rms_drift_db = float(
            20.0
            * math.log10(
                (float(np.sqrt(np.mean(candidate_window * candidate_window))) + EPSILON)
                / (float(np.sqrt(np.mean(base_window * base_window))) + EPSILON)
            )
        )
        windows.append(
            {
                "start_sec": round(float(start) / float(sample_rate), 6),
                "end_sec": round(float(end) / float(sample_rate), 6),
                "vocal_residue_corr": vocal_corr,
                "vocal_projection_ratio_db": vocal_projection_ratio_db,
                "rms_drift_db_vs_base": rms_drift_db,
                "presence_band_ratio_drift_db": calculate_band_ratio_db(
                    candidate_window,
                    base_window,
                    sample_rate=sample_rate,
                    low_hz=1800.0,
                    high_hz=4500.0,
                ),
                "air_band_ratio_drift_db": calculate_band_ratio_db(
                    candidate_window,
                    base_window,
                    sample_rate=sample_rate,
                    low_hz=6000.0,
                    high_hz=12000.0,
                ),
            }
        )

    if not windows:
        raise ValueError("No analysis windows were produced.")

    delta_audio = candidate_audio - base_audio
    base_rms = float(np.sqrt(np.mean(base_audio * base_audio))) + EPSILON
    delta_rms = float(np.sqrt(np.mean(delta_audio * delta_audio))) + EPSILON

    return {
        "rms_delta_db_vs_base": float(20.0 * math.log10(delta_rms / base_rms)),
        "avg_vocal_residue_corr": float(
            np.mean([window["vocal_residue_corr"] for window in windows])
        ),
        "max_vocal_residue_corr": float(
            np.max([window["vocal_residue_corr"] for window in windows])
        ),
        "avg_vocal_projection_ratio_db": float(
            np.mean([window["vocal_projection_ratio_db"] for window in windows])
        ),
        "worst_rms_dip_db": float(np.min([window["rms_drift_db_vs_base"] for window in windows])),
        "max_rms_boost_db": float(np.max([window["rms_drift_db_vs_base"] for window in windows])),
        "worst_presence_dip_db": float(
            np.min([window["presence_band_ratio_drift_db"] for window in windows])
        ),
        "max_presence_boost_db": float(
            np.max([window["presence_band_ratio_drift_db"] for window in windows])
        ),
        "worst_air_dip_db": float(
            np.min([window["air_band_ratio_drift_db"] for window in windows])
        ),
        "max_air_boost_db": float(np.max([window["air_band_ratio_drift_db"] for window in windows])),
        "top_vocal_residue_windows": sorted(
            windows,
            key=lambda item: item["vocal_residue_corr"],
            reverse=True,
        )[:8],
        "top_dip_windows": sorted(windows, key=lambda item: item["rms_drift_db_vs_base"])[:8],
    }


def evaluate_minus_candidate(
    *,
    candidate_name: str,
    candidate_summary: dict[str, Any],
    base_summary: dict[str, Any],
    policy: MinusAcceptancePolicy,
    cumulative_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    average_vocal_improvement_db = float(base_summary["avg_vocal_projection_ratio_db"]) - float(
        candidate_summary["avg_vocal_projection_ratio_db"]
    )
    peak_corr_improvement = float(base_summary["max_vocal_residue_corr"]) - float(
        candidate_summary["max_vocal_residue_corr"]
    )

    blockers: list[dict[str, Any]] = []
    if float(candidate_summary["rms_delta_db_vs_base"]) < policy.minimum_delta_rms_db:
        blockers.append(
            {
                "label": "no_meaningful_minus_delta",
                "reason": "candidate_too_close_to_base_to_be_audible_fix",
                "value": float(candidate_summary["rms_delta_db_vs_base"]),
                "threshold": policy.minimum_delta_rms_db,
            }
        )
    if (
        average_vocal_improvement_db < policy.minimum_average_vocal_improvement_db
        and peak_corr_improvement < policy.minimum_peak_corr_improvement
    ):
        blockers.append(
            {
                "label": "insufficient_vocal_residue_improvement",
                "reason": "candidate_does_not_reduce_vocal_proxy_enough",
                "average_vocal_improvement_db": average_vocal_improvement_db,
                "peak_corr_improvement": peak_corr_improvement,
            }
        )
    if float(candidate_summary["worst_rms_dip_db"]) < policy.maximum_rms_dip_db:
        blockers.append(
            {
                "label": "rms_dip_damage",
                "reason": "candidate_creates_audible_energy_dip_risk",
                "value": float(candidate_summary["worst_rms_dip_db"]),
                "threshold": policy.maximum_rms_dip_db,
            }
        )
    if float(candidate_summary["worst_presence_dip_db"]) < policy.maximum_presence_dip_db:
        blockers.append(
            {
                "label": "presence_dip_damage",
                "reason": "candidate_creates_presence_band_dip_risk",
                "value": float(candidate_summary["worst_presence_dip_db"]),
                "threshold": policy.maximum_presence_dip_db,
            }
        )
    if float(candidate_summary["worst_air_dip_db"]) < policy.maximum_air_dip_db:
        blockers.append(
            {
                "label": "air_dip_damage",
                "reason": "candidate_creates_air_band_dip_risk",
                "value": float(candidate_summary["worst_air_dip_db"]),
                "threshold": policy.maximum_air_dip_db,
            }
        )
    if float(candidate_summary.get("max_rms_boost_db", 0.0)) > policy.maximum_rms_boost_db:
        blockers.append(
            {
                "label": "rms_boost_damage",
                "reason": "candidate_creates_excessive_energy_boost_risk",
                "value": float(candidate_summary.get("max_rms_boost_db", 0.0)),
                "threshold": policy.maximum_rms_boost_db,
            }
        )
    if float(candidate_summary.get("max_presence_boost_db", 0.0)) > policy.maximum_presence_boost_db:
        blockers.append(
            {
                "label": "presence_boost_damage",
                "reason": "candidate_creates_excessive_presence_boost_risk",
                "value": float(candidate_summary.get("max_presence_boost_db", 0.0)),
                "threshold": policy.maximum_presence_boost_db,
            }
        )
    if float(candidate_summary.get("max_air_boost_db", 0.0)) > policy.maximum_air_boost_db:
        blockers.append(
            {
                "label": "air_boost_damage",
                "reason": "candidate_creates_excessive_air_boost_risk",
                "value": float(candidate_summary.get("max_air_boost_db", 0.0)),
                "threshold": policy.maximum_air_boost_db,
            }
        )
    if cumulative_summary is not None:
        if float(cumulative_summary["worst_rms_dip_db"]) < policy.maximum_rms_dip_db:
            blockers.append(
                {
                    "label": "cumulative_rms_dip_damage",
                    "reason": "iterative_candidate_exceeds_total_rms_dip_budget",
                    "value": float(cumulative_summary["worst_rms_dip_db"]),
                    "threshold": policy.maximum_rms_dip_db,
                }
            )
        if float(cumulative_summary["worst_presence_dip_db"]) < policy.maximum_presence_dip_db:
            blockers.append(
                {
                    "label": "cumulative_presence_dip_damage",
                    "reason": "iterative_candidate_exceeds_total_presence_dip_budget",
                    "value": float(cumulative_summary["worst_presence_dip_db"]),
                    "threshold": policy.maximum_presence_dip_db,
                }
            )
        if float(cumulative_summary["worst_air_dip_db"]) < policy.maximum_air_dip_db:
            blockers.append(
                {
                    "label": "cumulative_air_dip_damage",
                    "reason": "iterative_candidate_exceeds_total_air_dip_budget",
                    "value": float(cumulative_summary["worst_air_dip_db"]),
                    "threshold": policy.maximum_air_dip_db,
                }
            )
        if float(cumulative_summary.get("max_rms_boost_db", 0.0)) > policy.maximum_rms_boost_db:
            blockers.append(
                {
                    "label": "cumulative_rms_boost_damage",
                    "reason": "iterative_candidate_exceeds_total_rms_boost_budget",
                    "value": float(cumulative_summary.get("max_rms_boost_db", 0.0)),
                    "threshold": policy.maximum_rms_boost_db,
                }
            )
        if float(cumulative_summary.get("max_presence_boost_db", 0.0)) > policy.maximum_presence_boost_db:
            blockers.append(
                {
                    "label": "cumulative_presence_boost_damage",
                    "reason": "iterative_candidate_exceeds_total_presence_boost_budget",
                    "value": float(cumulative_summary.get("max_presence_boost_db", 0.0)),
                    "threshold": policy.maximum_presence_boost_db,
                }
            )
        if float(cumulative_summary.get("max_air_boost_db", 0.0)) > policy.maximum_air_boost_db:
            blockers.append(
                {
                    "label": "cumulative_air_boost_damage",
                    "reason": "iterative_candidate_exceeds_total_air_boost_budget",
                    "value": float(cumulative_summary.get("max_air_boost_db", 0.0)),
                    "threshold": policy.maximum_air_boost_db,
                }
            )

    dip_damage = abs(min(float(candidate_summary["worst_rms_dip_db"]), 0.0))
    dip_damage += abs(min(float(candidate_summary["worst_presence_dip_db"]), 0.0))
    dip_damage += abs(min(float(candidate_summary["worst_air_dip_db"]), 0.0))
    edit_loudness = max(float(candidate_summary["rms_delta_db_vs_base"]) + 48.0, 0.0)
    score = (
        average_vocal_improvement_db
        + peak_corr_improvement * 25.0
        - dip_damage * 2.0
        - edit_loudness * 0.015
    )

    status = "accepted" if not blockers else "rejected"
    return {
        "candidate_name": candidate_name,
        "status": status,
        "score": round(float(score), 6),
        "average_vocal_improvement_db": round(float(average_vocal_improvement_db), 6),
        "peak_corr_improvement": round(float(peak_corr_improvement), 6),
        "blockers": blockers,
        "summary": candidate_summary,
        "cumulative_summary": cumulative_summary,
    }


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_zero_label(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _is_low_confidence_accept(
    row: dict[str, Any],
    *,
    max_accept_probability: float = 0.0,
    min_block_probability: float = 0.0,
) -> bool:
    if row.get("critic_decision") != "accept_candidate":
        return False
    accept_probability = _as_float(row.get("predicted_accept_probability"), 1.0)
    block_probability = _as_float(row.get("predicted_block_probability"), 0.0)
    if max_accept_probability > 0.0 and accept_probability <= max_accept_probability:
        return True
    if min_block_probability > 0.0 and block_probability >= min_block_probability:
        return True
    return False


def _focus_window_risk_key(window: dict[str, Any]) -> tuple[float, float, float, float]:
    blocked_priority = 1.0 if window.get("focus_reason") == "blocked_or_negative" else 0.0
    block_probability = _as_float(window.get("predicted_block_probability"), 0.0)
    accept_probability = _as_float(window.get("predicted_accept_probability"), 1.0)
    start_sec = _as_float(window.get("start_sec"), 0.0)
    return (-blocked_priority, -block_probability, accept_probability, start_sec)


def load_focus_windows(
    path: Path | None,
    *,
    track_id: str = "",
    candidate_id: str = "",
    max_windows: int = 0,
    include_low_confidence_accepts: bool = False,
    max_accept_probability: float = 0.0,
    min_block_probability: float = 0.0,
) -> list[dict[str, Any]]:
    if path is None:
        return []
    windows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if track_id and str(row.get("track_id") or "") != track_id:
            continue
        if candidate_id and str(row.get("candidate_id") or "") != candidate_id:
            continue
        is_blocked = row.get("critic_decision") == "blocked_hard_case" or _is_zero_label(row.get("true_label"))
        is_soft_hardcase = include_low_confidence_accepts and _is_low_confidence_accept(
            row,
            max_accept_probability=max_accept_probability,
            min_block_probability=min_block_probability,
        )
        if not (is_blocked or is_soft_hardcase):
            continue
        start = _as_float(row.get("window_start_sec"))
        end = _as_float(row.get("window_end_sec"))
        if end <= start:
            continue
        windows.append(
            {
                "track_id": row.get("track_id"),
                "candidate_id": row.get("candidate_id"),
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "critic_decision": row.get("critic_decision"),
                "predicted_blockers": row.get("predicted_blockers") or [],
                "predicted_accept_probability": row.get("predicted_accept_probability"),
                "predicted_block_probability": row.get("predicted_block_probability"),
                "focus_reason": (
                    "low_confidence_accept"
                    if is_soft_hardcase and not is_blocked
                    else "blocked_or_negative"
                ),
                "true_label": row.get("true_label"),
            }
        )
    windows.sort(key=_focus_window_risk_key)
    if max_windows > 0:
        return windows[:max_windows]
    return windows


def calculate_focus_window_summary(
    *,
    candidate_audio: np.ndarray,
    base_audio: np.ndarray,
    vocal_audio: np.ndarray,
    sample_rate: int,
    focus_windows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not focus_windows:
        return {"window_count": 0, "windows": []}
    candidate_audio, base_audio, vocal_audio = match_length([candidate_audio, base_audio, vocal_audio])
    candidate_mono = candidate_audio.mean(axis=1)
    base_mono = base_audio.mean(axis=1)
    vocal_mono = vocal_audio.mean(axis=1)
    rows: list[dict[str, Any]] = []
    for focus in focus_windows:
        start = _as_float(focus.get("start_sec"))
        end = _as_float(focus.get("end_sec"))
        start_frame = max(0, min(int(round(start * sample_rate)), candidate_mono.shape[0]))
        end_frame = max(start_frame, min(int(round(end * sample_rate)), candidate_mono.shape[0]))
        if end_frame <= start_frame:
            continue
        candidate_window = candidate_mono[start_frame:end_frame]
        base_window = base_mono[start_frame:end_frame]
        vocal_window = vocal_mono[start_frame:end_frame]
        candidate_norm = float(np.linalg.norm(candidate_window))
        vocal_norm = float(np.linalg.norm(vocal_window))
        vocal_corr = abs(float(np.dot(candidate_window, vocal_window) / (candidate_norm * vocal_norm + EPSILON)))
        vocal_projection_scale = float(np.dot(candidate_window, vocal_window) / (np.dot(vocal_window, vocal_window) + EPSILON))
        vocal_projection = vocal_projection_scale * vocal_window
        vocal_projection_ratio_db = float(
            20.0
            * math.log10(
                (float(np.sqrt(np.mean(vocal_projection * vocal_projection))) + EPSILON)
                / (float(np.sqrt(np.mean(candidate_window * candidate_window))) + EPSILON)
            )
        )
        rms_drift_db = float(
            20.0
            * math.log10(
                (float(np.sqrt(np.mean(candidate_window * candidate_window))) + EPSILON)
                / (float(np.sqrt(np.mean(base_window * base_window))) + EPSILON)
            )
        )
        rows.append(
            {
                "start_sec": start,
                "end_sec": end,
                "vocal_residue_corr": vocal_corr,
                "vocal_projection_ratio_db": vocal_projection_ratio_db,
                "rms_drift_db_vs_base": rms_drift_db,
                "predicted_blockers": focus.get("predicted_blockers") or [],
                "critic_decision": focus.get("critic_decision"),
            }
        )
    if not rows:
        return {"window_count": 0, "windows": []}
    return {
        "window_count": len(rows),
        "avg_vocal_residue_corr": float(np.mean([row["vocal_residue_corr"] for row in rows])),
        "max_vocal_residue_corr": float(np.max([row["vocal_residue_corr"] for row in rows])),
        "avg_vocal_projection_ratio_db": float(np.mean([row["vocal_projection_ratio_db"] for row in rows])),
        "worst_rms_dip_db": float(np.min([row["rms_drift_db_vs_base"] for row in rows])),
        "windows": rows,
    }


def compare_focus_summaries(candidate_summary: dict[str, Any], base_summary: dict[str, Any]) -> dict[str, Any]:
    if not candidate_summary.get("window_count") or not base_summary.get("window_count"):
        return {"available": False, "reason": "missing_focus_windows"}
    return {
        "available": True,
        "window_count": int(candidate_summary["window_count"]),
        "average_vocal_improvement_db": round(
            float(base_summary["avg_vocal_projection_ratio_db"]) - float(candidate_summary["avg_vocal_projection_ratio_db"]),
            6,
        ),
        "peak_corr_improvement": round(
            float(base_summary["max_vocal_residue_corr"]) - float(candidate_summary["max_vocal_residue_corr"]),
            6,
        ),
        "worst_rms_dip_db": round(float(candidate_summary["worst_rms_dip_db"]), 6),
        "windows": candidate_summary.get("windows") or [],
    }


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate, subtype="PCM_24")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_fingerprint(audio: np.ndarray) -> str:
    array = np.ascontiguousarray(audio.astype(np.float32, copy=False))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def word_like_token_count(text: str) -> int:
    tokens = text.replace("'", " ").replace("-", " ").split()
    return sum(1 for token in tokens if any(character.isalpha() for character in token))


def _as_word_count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def extract_word_audit(
    payload: dict[str, Any] | None,
    *,
    role: str,
    audio_path: Path | None = None,
) -> dict[str, Any] | None:
    if not payload:
        return None
    candidates: list[Any] = []
    results = payload.get("results")
    if isinstance(results, dict):
        candidates.append(results.get(role))
        if audio_path is not None:
            target_path = str(audio_path)
            for item in results.values():
                if isinstance(item, dict) and str(item.get("path") or "") == target_path:
                    candidates.append(item)
    candidates.append(payload.get(role))
    if role == "accepted_product_minus":
        candidates.append(payload.get("minus_word_audit"))
    for candidate in candidates:
        if isinstance(candidate, dict):
            word_count = _as_word_count(candidate.get("word_like_token_count"))
            if word_count is None:
                word_count = word_like_token_count(str(candidate.get("text") or ""))
            return {
                "status": candidate.get("status", "available"),
                "role": role,
                "path": str(candidate.get("path") or audio_path or ""),
                "text": str(candidate.get("text") or ""),
                "word_like_token_count": word_count,
                "source": "provided_content_identity_audit",
            }
    word_count = _as_word_count(payload.get("word_like_token_count"))
    if word_count is not None:
        return {
            "status": payload.get("status", "available"),
            "role": role,
            "path": str(payload.get("path") or audio_path or ""),
            "text": str(payload.get("text") or ""),
            "word_like_token_count": word_count,
            "source": "provided_content_identity_audit",
        }
    return None


def run_word_content_audit(audio_path: Path, *, role: str, model_name: str = "") -> dict[str, Any]:
    model_name = model_name or DEFAULT_WORD_AUDIT_MODEL
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime image
        return {
            "status": "unavailable",
            "role": role,
            "path": str(audio_path),
            "word_like_token_count": None,
            "text": "",
            "blocked_reason": "faster_whisper_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "model": model_name,
        }

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        rows: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            rows.append(
                {
                    "start": round(float(getattr(segment, "start", 0.0)), 3),
                    "end": round(float(getattr(segment, "end", 0.0)), 3),
                    "text": text,
                    "avg_logprob": round(float(getattr(segment, "avg_logprob", 0.0)), 4),
                    "no_speech_prob": round(float(getattr(segment, "no_speech_prob", 0.0)), 4),
                    "compression_ratio": round(float(getattr(segment, "compression_ratio", 0.0)), 4),
                }
            )
            if text:
                text_parts.append(text)
        full_text = " ".join(text_parts).strip()
        return {
            "status": "available",
            "role": role,
            "path": str(audio_path),
            "model": model_name,
            "language": getattr(info, "language", None),
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
            "text": full_text,
            "word_like_token_count": word_like_token_count(full_text),
            "segments": rows,
            "source": "faster_whisper_runtime_content_audit",
        }
    except Exception as exc:  # pragma: no cover - depends on runtime model/cache/network
        return {
            "status": "failed",
            "role": role,
            "path": str(audio_path),
            "word_like_token_count": None,
            "text": "",
            "blocked_reason": "word_content_audit_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "model": model_name,
        }


def resolve_word_audit(
    *,
    payload: dict[str, Any] | None,
    role: str,
    audio_path: Path,
    run_if_missing: bool,
    model_name: str = "",
) -> dict[str, Any]:
    provided = extract_word_audit(payload, role=role, audio_path=audio_path)
    if provided is not None:
        return provided
    if not run_if_missing:
        return {
            "status": "missing",
            "role": role,
            "path": str(audio_path),
            "word_like_token_count": None,
            "text": "",
            "blocked_reason": "missing_word_content_audit",
        }
    return run_word_content_audit(audio_path, role=role, model_name=model_name)


def word_audit_has_words(audit: dict[str, Any]) -> bool | None:
    word_count = _as_word_count(audit.get("word_like_token_count"))
    if word_count is None:
        return None
    return word_count > 0


def reference_filename_for_content(role_prefix: str, audit: dict[str, Any] | None) -> tuple[str, str]:
    if audit is None:
        return f"{role_prefix}_unverified_reference.flac", f"{role_prefix}_unverified_reference"
    has_words = word_audit_has_words(audit)
    if has_words is True:
        return f"{role_prefix}_vocal_reference.flac", f"{role_prefix}_vocal_reference"
    if has_words is False:
        return f"{role_prefix}_residual_reference.flac", f"{role_prefix}_residual_reference"
    return f"{role_prefix}_unverified_reference.flac", f"{role_prefix}_unverified_reference"


def input_content_identity_blockers(input_word_audit: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    raw_audit = input_word_audit.get("raw_mix") or {}
    raw_has_words = word_audit_has_words(raw_audit)
    for role in ("primary_vocal_reference", "secondary_vocal_reference"):
        audit = input_word_audit.get(role) or {}
        has_words = word_audit_has_words(audit)
        if has_words is None:
            blockers.append(
                {
                    "label": f"{role}_word_audit_missing_or_failed",
                    "role": role,
                    "reason": audit.get("blocked_reason", "missing_word_content_audit"),
                }
            )
        elif raw_has_words is True and has_words is False and role == "primary_vocal_reference":
            blockers.append(
                {
                    "label": "primary_vocal_reference_has_no_detected_words",
                    "role": role,
                    "reason": "raw_mix_has_words_but_primary_vocal_reference_has_none",
                    "word_like_token_count": audit.get("word_like_token_count"),
                }
            )
    if raw_has_words is None:
        blockers.append(
            {
                "label": "raw_mix_word_audit_missing_or_failed",
                "role": "raw_mix",
                "reason": raw_audit.get("blocked_reason", "missing_word_content_audit"),
            }
        )
    return blockers


def selected_minus_content_blockers(audit: dict[str, Any]) -> list[dict[str, Any]]:
    has_words = word_audit_has_words(audit)
    if has_words is None:
        return [
            {
                "label": "selected_minus_word_audit_missing_or_failed",
                "reason": audit.get("blocked_reason", "missing_word_content_audit"),
            }
        ]
    if has_words:
        return [
            {
                "label": "selected_minus_contains_detected_words",
                "reason": "accepted_product_minus_requires_zero_word_like_tokens",
                "word_like_token_count": audit.get("word_like_token_count"),
                "text": audit.get("text", ""),
            }
        ]
    return []


def validate_prepared_input_roles(provenance: dict[str, dict[str, Any]]) -> None:
    collision_pairs = (
        ("base_product_minus", "guide_product_minus"),
        ("primary_vocal_reference", "secondary_vocal_reference"),
    )
    blockers: list[dict[str, Any]] = []
    for left_role, right_role in collision_pairs:
        left = provenance[left_role]
        right = provenance[right_role]
        if left["normalized_sha256"] != right["normalized_sha256"]:
            continue
        if left["intended_audio_fingerprint"] == right["intended_audio_fingerprint"]:
            continue
        blockers.append(
            {
                "left_role": left_role,
                "right_role": right_role,
                "left_source_path": left["source_path"],
                "right_source_path": right["source_path"],
                "normalized_sha256": left["normalized_sha256"],
                "reason": "distinct_normalized_audio_roles_share_one_output_file_hash",
            }
        )
    if blockers:
        raise RuntimeError(
            "normalized_input_role_collision: " + json.dumps(blockers, ensure_ascii=False)
        )


def emit_progress(progress_path: Path, **payload: Any) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "route_name": ROUTE_NAME,
        "acceleration_mode": "external_only_cpu_scoring",
        "gpu_inference_skipped": True,
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def prepare_recipe_inputs(
    *,
    raw_audio_path: Path,
    base_minus_path: Path,
    guide_minus_path: Path,
    primary_vocals_path: Path,
    secondary_vocals_path: Path,
    output_dir: Path,
    input_word_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_audio, sample_rate = load_audio(base_minus_path)
    raw_audio, _ = load_audio(raw_audio_path, sample_rate)
    guide_minus, _ = load_audio(guide_minus_path, sample_rate)
    primary_vocals, _ = load_audio(primary_vocals_path, sample_rate)
    secondary_vocals, _ = load_audio(secondary_vocals_path, sample_rate)
    raw_audio, base_audio, guide_minus, primary_vocals, secondary_vocals = match_length(
        [raw_audio, base_audio, guide_minus, primary_vocals, secondary_vocals]
    )
    primary_filename, primary_contract_role = reference_filename_for_content(
        "primary",
        (input_word_audit or {}).get("primary_vocal_reference"),
    )
    secondary_filename, secondary_contract_role = reference_filename_for_content(
        "secondary",
        (input_word_audit or {}).get("secondary_vocal_reference"),
    )
    role_inputs = {
        "raw_mix": {
            "source_path": raw_audio_path,
            "normalized_path": output_dir / "raw_mix.flac",
            "audio": raw_audio,
            "contract_role": "source_mix",
        },
        "base_product_minus": {
            "source_path": base_minus_path,
            "normalized_path": output_dir / "base_product_minus.flac",
            "audio": base_audio,
            "contract_role": "baseline_product_minus",
        },
        "guide_product_minus": {
            "source_path": guide_minus_path,
            "normalized_path": output_dir / "guide_product_minus.flac",
            "audio": guide_minus,
            "contract_role": "guide_product_minus",
        },
        "primary_vocal_reference": {
            "source_path": primary_vocals_path,
            "normalized_path": output_dir / primary_filename,
            "audio": primary_vocals,
            "contract_role": primary_contract_role,
        },
        "secondary_vocal_reference": {
            "source_path": secondary_vocals_path,
            "normalized_path": output_dir / secondary_filename,
            "audio": secondary_vocals,
            "contract_role": secondary_contract_role,
        },
    }
    for item in role_inputs.values():
        write_audio(Path(item["normalized_path"]), item["audio"], sample_rate)
    provenance: dict[str, dict[str, Any]] = {}
    for role, item in role_inputs.items():
        source_path = Path(item["source_path"])
        normalized_path = Path(item["normalized_path"])
        normalized_audio, _ = load_audio(normalized_path, sample_rate)
        provenance[role] = {
            "role": role,
            "contract_role": item["contract_role"],
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "normalized_path": str(normalized_path),
            "normalized_filename": normalized_path.name,
            "normalized_sha256": sha256_file(normalized_path),
            "intended_audio_fingerprint": audio_fingerprint(item["audio"]),
            "normalized_audio_fingerprint": audio_fingerprint(normalized_audio),
            "sample_rate": sample_rate,
            "word_content_audit": (input_word_audit or {}).get(role),
        }
    validate_prepared_input_roles(provenance)
    normalized_input_manifest = output_dir / "inputs_manifest.json"
    normalized_input_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "ralph_normalized_inputs",
                "sample_rate": sample_rate,
                "provenance": provenance,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "sample_rate": sample_rate,
        "raw_audio_path": Path(role_inputs["raw_mix"]["normalized_path"]),
        "base_minus_path": Path(role_inputs["base_product_minus"]["normalized_path"]),
        "guide_minus_path": Path(role_inputs["guide_product_minus"]["normalized_path"]),
        "primary_vocals_path": Path(role_inputs["primary_vocal_reference"]["normalized_path"]),
        "secondary_vocals_path": Path(role_inputs["secondary_vocal_reference"]["normalized_path"]),
        "normalized_input_manifest": normalized_input_manifest,
        "provenance": provenance,
    }


def run_candidate_recipe(
    *,
    recipe: MinusCandidateRecipe,
    raw_audio_path: Path,
    base_minus_path: Path,
    guide_minus_path: Path,
    primary_vocals_path: Path,
    secondary_vocals_path: Path,
    output_dir: Path,
    listen_dir: Path | None,
) -> dict[str, Any]:
    candidate_output_dir = output_dir / "candidates" / recipe.name
    candidate_listen_dir = listen_dir / recipe.name if listen_dir is not None else None
    command = [
        sys.executable,
        str(recipe.tool_path),
        str(raw_audio_path),
        str(base_minus_path),
        str(guide_minus_path),
        str(primary_vocals_path),
        str(secondary_vocals_path),
        str(candidate_output_dir),
    ]
    if candidate_listen_dir is not None:
        command.extend(["--listen-dir", str(candidate_listen_dir), "--listen-label", recipe.listen_label])
    command.extend(recipe.arguments)
    run_command(command)
    manifest_path = candidate_output_dir / "stem_manifest.json"
    manifest = load_json(manifest_path)
    artifact_path = Path(manifest["artifacts"][recipe.output_artifact_key])
    return {
        "recipe": recipe,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "artifact_path": artifact_path,
        "output_dir": candidate_output_dir,
        "listen_dir": candidate_listen_dir,
    }


def write_selected_artifacts(
    *,
    selected_candidate: dict[str, Any],
    output_dir: Path,
    listen_dir: Path | None,
    selected_word_audit: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    content_blockers = selected_minus_content_blockers(selected_word_audit)
    if content_blockers:
        blocked_output_path = output_dir / "blocked_product_minus_candidate.flac"
        shutil.copyfile(selected_candidate["artifact_path"], blocked_output_path)
        candidate_manifest = selected_candidate.get("manifest") or {}
        return {
            "stem_type": "minus",
            "artifact_role": "blocked_product_minus_candidate",
            "blocked_product_minus_candidate_flac": str(blocked_output_path),
            "source_candidate_path": str(selected_candidate["artifact_path"]),
            "source_candidate_manifest": str(selected_candidate["manifest_path"]),
            "source_candidate_artifact_role": candidate_manifest.get(
                "artifact_role", "candidate_product_minus"
            ),
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "release_status": "blocked_for_listening",
            "content_identity_audit": selected_word_audit,
            "content_identity_blockers": content_blockers,
        }
    selected_output_path = output_dir / "accepted_product_minus.flac"
    shutil.copyfile(selected_candidate["artifact_path"], selected_output_path)
    candidate_manifest = selected_candidate.get("manifest") or {}
    selected_artifacts: dict[str, Any] = {
        "stem_type": "minus",
        "artifact_role": "accepted_product_minus",
        "selected_minus_flac": str(selected_output_path),
        "accepted_product_minus_flac": str(selected_output_path),
        "source_candidate_path": str(selected_candidate["artifact_path"]),
        "source_candidate_manifest": str(selected_candidate["manifest_path"]),
        "source_candidate_artifact_role": candidate_manifest.get(
            "artifact_role", "candidate_product_minus"
        ),
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "release_ready",
        "content_identity_audit": selected_word_audit,
        "content_identity_blockers": [],
    }
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        listen_output_path = listen_dir / "accepted_product_minus.flac"
        shutil.copyfile(selected_output_path, listen_output_path)
        selected_artifacts["listen_output_path"] = str(listen_output_path)
    return selected_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_audio_path")
    parser.add_argument("base_minus_path")
    parser.add_argument("guide_minus_path")
    parser.add_argument("primary_vocals_path")
    parser.add_argument("secondary_vocals_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--reference-base-path")
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--focus-windows-jsonl")
    parser.add_argument("--focus-track-id", default="")
    parser.add_argument("--focus-candidate-id", default="")
    parser.add_argument("--max-focus-windows", type=int, default=16)
    parser.add_argument("--focus-include-low-confidence-accepted", action="store_true")
    parser.add_argument("--focus-max-accept-probability", type=float, default=0.0)
    parser.add_argument("--focus-min-block-probability", type=float, default=0.0)
    parser.add_argument("--content-identity-audit-json")
    parser.add_argument(
        "--skip-runtime-word-content-audit",
        action="store_true",
        help="Do not run runtime ASR when the provided content audit is missing a role; missing audit blocks delivery.",
    )
    return parser.parse_args(argv)


def exit_code_for_manifest(manifest: dict[str, Any]) -> int:
    if (
        manifest.get("delivery_status") == "accepted"
        and manifest.get("delivery_allowed") is True
        and manifest.get("release_status") == "release_ready"
    ):
        return 0
    return 1


def run_ralph_minus_improvement_loop(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    raw_audio_path = Path(args.raw_audio_path)
    base_minus_path = Path(args.base_minus_path)
    guide_minus_path = Path(args.guide_minus_path)
    primary_vocals_path = Path(args.primary_vocals_path)
    secondary_vocals_path = Path(args.secondary_vocals_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None
    reference_base_path = (
        Path(args.reference_base_path) if args.reference_base_path else base_minus_path
    )
    reference_base_was_base = reference_base_path == base_minus_path
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_LOG_NAME
    if progress_path.exists():
        progress_path.unlink()
    manifest_path = output_dir / "ralph_minus_improvement_manifest.json"
    emit_progress(
        progress_path,
        phase="start",
        reason="external_only_cpu_dsp_scoring",
        raw_audio_path=str(raw_audio_path),
        base_minus_path=str(base_minus_path),
        guide_minus_path=str(guide_minus_path),
        primary_vocals_path=str(primary_vocals_path),
        secondary_vocals_path=str(secondary_vocals_path),
        output_manifest_path=str(manifest_path),
    )
    emit_progress(
        progress_path,
        phase="input_normalization_start",
        elapsed_sec=round(time.perf_counter() - started_at, 3),
    )
    content_identity_audit_path = (
        Path(args.content_identity_audit_json) if args.content_identity_audit_json else None
    )
    content_identity_audit = (
        load_json(content_identity_audit_path) if content_identity_audit_path else None
    )
    run_word_audit_if_missing = not bool(args.skip_runtime_word_content_audit)
    word_audit_model = os.environ.get(WORD_AUDIT_MODEL_ENV, DEFAULT_WORD_AUDIT_MODEL)
    input_word_audit = {
        "raw_mix": resolve_word_audit(
            payload=content_identity_audit,
            role="raw_mix",
            audio_path=raw_audio_path,
            run_if_missing=run_word_audit_if_missing,
            model_name=word_audit_model,
        ),
        "primary_vocal_reference": resolve_word_audit(
            payload=content_identity_audit,
            role="primary_vocal_reference",
            audio_path=primary_vocals_path,
            run_if_missing=run_word_audit_if_missing,
            model_name=word_audit_model,
        ),
        "secondary_vocal_reference": resolve_word_audit(
            payload=content_identity_audit,
            role="secondary_vocal_reference",
            audio_path=secondary_vocals_path,
            run_if_missing=run_word_audit_if_missing,
            model_name=word_audit_model,
        ),
    }
    prepared_inputs = prepare_recipe_inputs(
        raw_audio_path=raw_audio_path,
        base_minus_path=base_minus_path,
        guide_minus_path=guide_minus_path,
        primary_vocals_path=primary_vocals_path,
        secondary_vocals_path=secondary_vocals_path,
        output_dir=output_dir / "normalized_inputs",
        input_word_audit=input_word_audit,
    )
    input_content_blockers = input_content_identity_blockers(input_word_audit)
    emit_progress(
        progress_path,
        phase="input_normalization_end",
        sample_rate=prepared_inputs["sample_rate"],
        raw_audio_path=str(prepared_inputs["raw_audio_path"]),
        base_minus_path=str(prepared_inputs["base_minus_path"]),
        guide_minus_path=str(prepared_inputs["guide_minus_path"]),
        primary_vocals_path=str(prepared_inputs["primary_vocals_path"]),
        secondary_vocals_path=str(prepared_inputs["secondary_vocals_path"]),
        normalized_input_manifest=str(prepared_inputs["normalized_input_manifest"]),
        input_content_blocker_count=len(input_content_blockers),
        elapsed_sec=round(time.perf_counter() - started_at, 3),
    )
    raw_audio_path = Path(prepared_inputs["raw_audio_path"])
    base_minus_path = Path(prepared_inputs["base_minus_path"])
    guide_minus_path = Path(prepared_inputs["guide_minus_path"])
    primary_vocals_path = Path(prepared_inputs["primary_vocals_path"])
    secondary_vocals_path = Path(prepared_inputs["secondary_vocals_path"])
    if reference_base_was_base:
        reference_base_path = base_minus_path
    if input_content_blockers:
        selected_artifacts = {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "release_status": "blocked_for_listening",
            "artifact_role": "blocked_input_content_identity",
            "content_identity_blockers": input_content_blockers,
        }
        manifest = {
            "schema_version": 1,
            "mode": "ralph_minus_improvement_loop",
            "ralph_contract": {
                "reference_free_application": True,
                "no_per_track_hardcoded_timestamps": True,
                "no_fallback_as_success": True,
                "produce_only_release_ready": True,
                "content_identity_gate_required": True,
            },
            "inputs": {
                "raw_audio_path": str(raw_audio_path),
                "base_minus_path": str(base_minus_path),
                "reference_base_path": str(reference_base_path),
                "guide_minus_path": str(guide_minus_path),
                "primary_vocals_path": str(primary_vocals_path),
                "secondary_vocals_path": str(secondary_vocals_path),
                "normalized_inputs": {
                    key: str(value)
                    for key, value in prepared_inputs.items()
                    if key.endswith("_path")
                },
                "normalized_input_manifest": str(prepared_inputs["normalized_input_manifest"]),
                "normalized_input_provenance": prepared_inputs["provenance"],
                "normalized_sample_rate": prepared_inputs["sample_rate"],
                "word_content_audit": input_word_audit,
                "content_identity_audit_path": str(content_identity_audit_path or ""),
            },
            "input_content_identity_blockers": input_content_blockers,
            "focus_window_source": str(args.focus_windows_jsonl or ""),
            "focus_windows": [],
            "focus_base_summary": {"window_count": 0, "windows": []},
            "focus_evaluation_used_for_delivery": False,
            "focus_candidate_recipe_enabled": False,
            "policy": MinusAcceptancePolicy().__dict__,
            "base_summary": {},
            "candidates": [],
            "selected_candidate": None,
            "release_status": "blocked_for_listening",
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "selected_artifacts": selected_artifacts,
            "progress_log": str(progress_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        emit_progress(
            progress_path,
            phase="end",
            release_status=manifest["release_status"],
            delivery_status=manifest["delivery_status"],
            delivery_allowed=manifest["delivery_allowed"],
            selected_candidate=manifest["selected_candidate"],
            output_manifest_path=str(manifest_path),
            elapsed_sec=round(time.perf_counter() - started_at, 3),
        )
        return manifest

    base_audio, sample_rate = load_audio(base_minus_path)
    reference_base_audio, _ = load_audio(reference_base_path, sample_rate)
    primary_vocals, _ = load_audio(primary_vocals_path, sample_rate)
    base_audio, reference_base_audio, primary_vocals = match_length(
        [base_audio, reference_base_audio, primary_vocals]
    )
    base_summary = calculate_windowed_minus_summary(
        candidate_audio=base_audio,
        base_audio=base_audio,
        vocal_audio=primary_vocals,
        sample_rate=sample_rate,
        window_sec=float(args.window_sec),
    )
    focus_windows = load_focus_windows(
        Path(args.focus_windows_jsonl) if args.focus_windows_jsonl else None,
        track_id=str(args.focus_track_id or ""),
        candidate_id=str(args.focus_candidate_id or ""),
        max_windows=int(args.max_focus_windows),
        include_low_confidence_accepts=bool(args.focus_include_low_confidence_accepted),
        max_accept_probability=float(args.focus_max_accept_probability),
        min_block_probability=float(args.focus_min_block_probability),
    )
    focus_base_summary = calculate_focus_window_summary(
        candidate_audio=base_audio,
        base_audio=base_audio,
        vocal_audio=primary_vocals,
        sample_rate=sample_rate,
        focus_windows=focus_windows,
    )

    policy = MinusAcceptancePolicy()
    candidate_records: list[dict[str, Any]] = []
    recipes = default_candidate_recipes(
        focus_windows_jsonl=str(args.focus_windows_jsonl or ""),
        focus_track_id=str(args.focus_track_id or ""),
        focus_candidate_id=str(args.focus_candidate_id or ""),
        max_focus_windows=int(args.max_focus_windows),
        focus_window_count=len(focus_windows),
        focus_include_low_confidence_accepted=bool(args.focus_include_low_confidence_accepted),
        focus_max_accept_probability=float(args.focus_max_accept_probability),
        focus_min_block_probability=float(args.focus_min_block_probability),
    )
    for index, recipe in enumerate(recipes, start=1):
        emit_progress(
            progress_path,
            phase="candidate_start",
            candidate_index=index,
            candidate_count=len(recipes),
            candidate_name=recipe.name,
            elapsed_sec=round(time.perf_counter() - started_at, 3),
        )
        candidate_record = run_candidate_recipe(
            recipe=recipe,
            raw_audio_path=raw_audio_path,
            base_minus_path=base_minus_path,
            guide_minus_path=guide_minus_path,
            primary_vocals_path=primary_vocals_path,
            secondary_vocals_path=secondary_vocals_path,
            output_dir=output_dir,
            listen_dir=listen_dir,
        )
        candidate_audio, _ = load_audio(candidate_record["artifact_path"], sample_rate)
        (
            candidate_audio,
            base_for_candidate,
            reference_base_for_candidate,
            vocals_for_candidate,
        ) = match_length(
            [candidate_audio, base_audio, reference_base_audio, primary_vocals]
        )
        candidate_summary = calculate_windowed_minus_summary(
            candidate_audio=candidate_audio,
            base_audio=base_for_candidate,
            vocal_audio=vocals_for_candidate,
            sample_rate=sample_rate,
            window_sec=float(args.window_sec),
        )
        cumulative_summary = (
            calculate_windowed_minus_summary(
                candidate_audio=candidate_audio,
                base_audio=reference_base_for_candidate,
                vocal_audio=vocals_for_candidate,
                sample_rate=sample_rate,
                window_sec=float(args.window_sec),
            )
            if reference_base_path != base_minus_path
            else None
        )
        evaluation = evaluate_minus_candidate(
            candidate_name=recipe.name,
            candidate_summary=candidate_summary,
            base_summary=base_summary,
            policy=policy,
            cumulative_summary=cumulative_summary,
        )
        focus_summary = calculate_focus_window_summary(
            candidate_audio=candidate_audio,
            base_audio=base_for_candidate,
            vocal_audio=vocals_for_candidate,
            sample_rate=sample_rate,
            focus_windows=focus_windows,
        )
        evaluation["focus_evaluation"] = compare_focus_summaries(focus_summary, focus_base_summary)
        candidate_record["evaluation"] = evaluation
        candidate_records.append(candidate_record)
        emit_progress(
            progress_path,
            phase="candidate_end",
            candidate_index=index,
            candidate_count=len(recipes),
            candidate_name=recipe.name,
            candidate_status=evaluation["status"],
            blocker_count=len(evaluation["blockers"]),
            score=evaluation["score"],
            elapsed_sec=round(time.perf_counter() - started_at, 3),
        )

    accepted_candidates = [
        candidate for candidate in candidate_records if candidate["evaluation"]["status"] == "accepted"
    ]
    selected_candidate = (
        max(accepted_candidates, key=lambda item: item["evaluation"]["score"])
        if accepted_candidates
        else None
    )
    if selected_candidate is not None:
        selected_word_audit = resolve_word_audit(
            payload=content_identity_audit,
            role="accepted_product_minus",
            audio_path=Path(selected_candidate["artifact_path"]),
            run_if_missing=run_word_audit_if_missing,
            model_name=word_audit_model,
        )
        selected_artifacts = write_selected_artifacts(
            selected_candidate=selected_candidate,
            output_dir=output_dir,
            listen_dir=listen_dir,
            selected_word_audit=selected_word_audit,
        )
        overall_status = str(selected_artifacts["release_status"])
    else:
        selected_artifacts = {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "release_status": "blocked_for_listening",
        }
        overall_status = "blocked_for_listening"
    delivery_allowed = bool(selected_artifacts.get("delivery_allowed") is True)
    delivery_status = str(selected_artifacts.get("delivery_status") or "blocked")
    presentation_status = str(
        selected_artifacts.get("presentation_status") or "blocked_from_listen_pack"
    )

    manifest = {
        "schema_version": 1,
        "mode": "ralph_minus_improvement_loop",
        "ralph_contract": {
            "reference_free_application": True,
            "no_per_track_hardcoded_timestamps": True,
            "no_fallback_as_success": True,
            "produce_only_release_ready": True,
            "content_identity_gate_required": True,
        },
        "inputs": {
            "raw_audio_path": str(raw_audio_path),
            "base_minus_path": str(base_minus_path),
            "reference_base_path": str(reference_base_path),
            "guide_minus_path": str(guide_minus_path),
            "primary_vocals_path": str(primary_vocals_path),
            "secondary_vocals_path": str(secondary_vocals_path),
            "normalized_inputs": {
                key: str(value)
                for key, value in prepared_inputs.items()
                if key.endswith("_path")
            },
            "normalized_input_manifest": str(prepared_inputs["normalized_input_manifest"]),
            "normalized_input_provenance": prepared_inputs["provenance"],
            "normalized_sample_rate": prepared_inputs["sample_rate"],
            "word_content_audit": input_word_audit,
            "content_identity_audit_path": str(content_identity_audit_path or ""),
        },
        "focus_window_source": str(args.focus_windows_jsonl or ""),
        "focus_window_filters": {
            "track_id": str(args.focus_track_id or ""),
            "candidate_id": str(args.focus_candidate_id or ""),
            "max_focus_windows": int(args.max_focus_windows),
            "include_low_confidence_accepted": bool(args.focus_include_low_confidence_accepted),
            "max_accept_probability": float(args.focus_max_accept_probability),
            "min_block_probability": float(args.focus_min_block_probability),
        },
        "focus_windows": focus_windows,
        "focus_base_summary": focus_base_summary,
        "focus_evaluation_used_for_delivery": False,
        "focus_candidate_recipe_enabled": any(recipe.name == "focus_window_guarded" for recipe in recipes),
        "policy": policy.__dict__,
        "base_summary": base_summary,
        "candidates": [
            {
                "candidate_name": candidate["recipe"].name,
                "stem_type": candidate["manifest"].get("stem_type", "minus"),
                "artifact_role": candidate["manifest"].get(
                    "artifact_role", "candidate_product_minus"
                ),
                "manifest_path": str(candidate["manifest_path"]),
                "artifact_path": str(candidate["artifact_path"]),
                "evaluation": candidate["evaluation"],
            }
            for candidate in candidate_records
        ],
        "selected_candidate": selected_candidate["recipe"].name if selected_candidate else None,
        "release_status": overall_status,
        "delivery_status": delivery_status,
        "delivery_allowed": delivery_allowed,
        "presentation_status": presentation_status,
        "selected_artifacts": selected_artifacts,
        "progress_log": str(progress_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    emit_progress(
        progress_path,
        phase="end",
        release_status=manifest["release_status"],
        delivery_status=manifest["delivery_status"],
        delivery_allowed=manifest["delivery_allowed"],
        selected_candidate=manifest["selected_candidate"],
        output_manifest_path=str(manifest_path),
        elapsed_sec=round(time.perf_counter() - started_at, 3),
    )
    if listen_dir is not None and delivery_allowed:
        (listen_dir / "ralph_minus_improvement_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_ralph_minus_improvement_loop.py",
        entrypoint_path="/app/tools/run_ralph_minus_improvement_loop.py",
    )
    args = parse_args(argv)
    manifest = run_ralph_minus_improvement_loop(args)
    print(
        json.dumps(
            {
                "release_status": manifest["release_status"],
                "selected_candidate": manifest["selected_candidate"],
                "selected_artifacts": manifest["selected_artifacts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return exit_code_for_manifest(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
