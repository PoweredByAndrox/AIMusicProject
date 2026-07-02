from __future__ import annotations

import argparse
import json
import math
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


def default_candidate_recipes() -> tuple[MinusCandidateRecipe, ...]:
    return (
        MinusCandidateRecipe(
            name="actual_masked_residue_guarded",
            tool_path=TOOLS_DIR / "apply_actual_masked_residue_minus_cleanup.py",
            output_artifact_key="minus_actual_masked_residue_flac",
            listen_label="02_actual_masked_residue_guarded_minus.flac",
            arguments=(
                "--strength",
                "0.10",
                "--mask-power",
                "1.35",
                "--smoothing-sec",
                "0.10",
            ),
        ),
        MinusCandidateRecipe(
            name="windowed_guarded",
            tool_path=TOOLS_DIR / "apply_windowed_hybrid_minus_cleanup.py",
            output_artifact_key="minus_windowed_hybrid_flac",
            listen_label="02_windowed_guarded_minus.flac",
            arguments=(
                "--strength",
                "0.90",
                "--mask-power",
                "1.35",
                "--guide-power",
                "1.20",
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
            listen_label="02_windowed_stronger_minus.flac",
            arguments=(
                "--strength",
                "0.96",
                "--mask-power",
                "1.25",
                "--guide-power",
                "1.20",
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
            listen_label="02_hybrid_balanced_minus.flac",
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
    )


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
        "worst_presence_dip_db": float(
            np.min([window["presence_band_ratio_drift_db"] for window in windows])
        ),
        "worst_air_dip_db": float(
            np.min([window["air_band_ratio_drift_db"] for window in windows])
        ),
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


def load_focus_windows(
    path: Path | None,
    *,
    track_id: str = "",
    candidate_id: str = "",
    max_windows: int = 0,
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
        if not is_blocked:
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
                "true_label": row.get("true_label"),
            }
        )
        if max_windows > 0 and len(windows) >= max_windows:
            break
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
    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate)
    return path


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
) -> dict[str, Any]:
    base_audio, sample_rate = load_audio(base_minus_path)
    raw_audio, _ = load_audio(raw_audio_path, sample_rate)
    guide_minus, _ = load_audio(guide_minus_path, sample_rate)
    primary_vocals, _ = load_audio(primary_vocals_path, sample_rate)
    secondary_vocals, _ = load_audio(secondary_vocals_path, sample_rate)
    raw_audio, base_audio, guide_minus, primary_vocals, secondary_vocals = match_length(
        [raw_audio, base_audio, guide_minus, primary_vocals, secondary_vocals]
    )
    return {
        "sample_rate": sample_rate,
        "raw_audio_path": write_audio(output_dir / "raw_audio.flac", raw_audio, sample_rate),
        "base_minus_path": write_audio(output_dir / "base_minus.flac", base_audio, sample_rate),
        "guide_minus_path": write_audio(output_dir / "guide_minus.flac", guide_minus, sample_rate),
        "primary_vocals_path": write_audio(output_dir / "primary_vocals.flac", primary_vocals, sample_rate),
        "secondary_vocals_path": write_audio(
            output_dir / "secondary_vocals.flac",
            secondary_vocals,
            sample_rate,
        ),
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
) -> dict[str, Any]:
    selected_output_path = output_dir / "02_adaptive_selected_minus.flac"
    shutil.copyfile(selected_candidate["artifact_path"], selected_output_path)
    selected_artifacts: dict[str, Any] = {
        "selected_minus_flac": str(selected_output_path),
        "source_candidate_path": str(selected_candidate["artifact_path"]),
        "source_candidate_manifest": str(selected_candidate["manifest_path"]),
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "release_ready",
    }
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        listen_output_path = listen_dir / "02_adaptive_selected_minus.flac"
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
    prepared_inputs = prepare_recipe_inputs(
        raw_audio_path=raw_audio_path,
        base_minus_path=base_minus_path,
        guide_minus_path=guide_minus_path,
        primary_vocals_path=primary_vocals_path,
        secondary_vocals_path=secondary_vocals_path,
        output_dir=output_dir / "normalized_inputs",
    )
    emit_progress(
        progress_path,
        phase="input_normalization_end",
        sample_rate=prepared_inputs["sample_rate"],
        raw_audio_path=str(prepared_inputs["raw_audio_path"]),
        base_minus_path=str(prepared_inputs["base_minus_path"]),
        guide_minus_path=str(prepared_inputs["guide_minus_path"]),
        primary_vocals_path=str(prepared_inputs["primary_vocals_path"]),
        secondary_vocals_path=str(prepared_inputs["secondary_vocals_path"]),
        elapsed_sec=round(time.perf_counter() - started_at, 3),
    )
    raw_audio_path = Path(prepared_inputs["raw_audio_path"])
    base_minus_path = Path(prepared_inputs["base_minus_path"])
    guide_minus_path = Path(prepared_inputs["guide_minus_path"])
    primary_vocals_path = Path(prepared_inputs["primary_vocals_path"])
    secondary_vocals_path = Path(prepared_inputs["secondary_vocals_path"])
    if reference_base_was_base:
        reference_base_path = base_minus_path

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
    recipes = default_candidate_recipes()
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
        selected_artifacts = write_selected_artifacts(
            selected_candidate=selected_candidate,
            output_dir=output_dir,
            listen_dir=listen_dir,
        )
        overall_status = "release_ready"
    else:
        selected_artifacts = {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "release_status": "blocked_for_listening",
        }
        overall_status = "blocked_for_listening"

    manifest = {
        "schema_version": 1,
        "mode": "ralph_minus_improvement_loop",
        "ralph_contract": {
            "reference_free_application": True,
            "no_per_track_hardcoded_timestamps": True,
            "no_fallback_as_success": True,
            "produce_only_release_ready": True,
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
            "normalized_sample_rate": prepared_inputs["sample_rate"],
        },
        "focus_window_source": str(args.focus_windows_jsonl or ""),
        "focus_window_filters": {
            "track_id": str(args.focus_track_id or ""),
            "candidate_id": str(args.focus_candidate_id or ""),
            "max_focus_windows": int(args.max_focus_windows),
        },
        "focus_windows": focus_windows,
        "focus_base_summary": focus_base_summary,
        "focus_evaluation_used_for_delivery": False,
        "policy": policy.__dict__,
        "base_summary": base_summary,
        "candidates": [
            {
                "candidate_name": candidate["recipe"].name,
                "manifest_path": str(candidate["manifest_path"]),
                "artifact_path": str(candidate["artifact_path"]),
                "evaluation": candidate["evaluation"],
            }
            for candidate in candidate_records
        ],
        "selected_candidate": selected_candidate["recipe"].name if selected_candidate else None,
        "release_status": overall_status,
        "delivery_status": "accepted" if selected_candidate else "blocked",
        "delivery_allowed": selected_candidate is not None,
        "presentation_status": (
            "accepted_for_listen_pack" if selected_candidate else "blocked_from_listen_pack"
        ),
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
    if listen_dir is not None and selected_candidate is not None:
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
