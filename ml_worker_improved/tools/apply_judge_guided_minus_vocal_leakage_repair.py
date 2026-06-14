from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import soundfile as sf
from scipy import signal


EPS = 1.0e-9
MAX_ALLOWED_PROJECTION_GAIN = 512.0
LEAKAGE_BLOCKERS = {"severe_vocal_bleed_in_minus", "vocal_leakage_in_minus"}
SELF_JUDGE_LEAKAGE_ISSUES = {"vocal_projection_proxy", "vocal_stem_corr_proxy"}


def progress(phase: str, **fields: Any) -> None:
    payload = {"phase": phase, "elapsed_sec": round(time.monotonic() - START_TIME, 3), **fields}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


START_TIME = time.monotonic()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def self_judge_manifest_row(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    candidate_label = str(candidate.get("label") or candidate.get("candidate_id") or payload.get("candidate_id") or "")
    minus_path = str(candidate.get("instrumental_path") or candidate.get("minus_path") or "")
    return {
        "candidate_id": candidate_label,
        "candidate_label": candidate_label,
        "instrumental_path": minus_path,
        "minus_path": minus_path,
        "vocal_path": str(candidate.get("vocal_path") or ""),
        "source_audio_path": str(payload.get("source_audio_path") or ""),
        "judge_source": "instrumental_self_judge_manifest",
        "window_metrics": [segment for segment in segments if isinstance(segment, dict)],
    }


def read_judge_rows(path: Path) -> list[dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8-sig")
    stripped = raw_text.lstrip()
    if stripped.startswith("{"):
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}")
        if isinstance(payload.get("segments"), list):
            return [self_judge_manifest_row(payload)]
        return [payload]
    return read_jsonl(path)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def normalize_path_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def candidate_matches(row: dict[str, Any], *, candidate_id: str | None, minus_path: str | Path | None) -> bool:
    if candidate_id and str(row.get("candidate_id") or "") == candidate_id:
        return True
    if candidate_id and str(row.get("candidate_label") or "") == candidate_id:
        return True
    requested_path = normalize_path_text(minus_path)
    if not requested_path:
        return False
    row_paths = [
        normalize_path_text(row.get("minus_path")),
        normalize_path_text(row.get("instrumental_path")),
    ]
    return requested_path in row_paths


def has_leakage_blocker(window: dict[str, Any]) -> bool:
    labels = (
        set(as_list(window.get("minus_blockers")))
        | set(as_list(window.get("reasons")))
        | set(as_list(window.get("issues")))
    )
    return bool(labels & (LEAKAGE_BLOCKERS | SELF_JUDGE_LEAKAGE_ISSUES))


def self_judge_leakage_score(window: dict[str, Any]) -> float:
    labels = set(as_list(window.get("issues")))
    if not labels & SELF_JUDGE_LEAKAGE_ISSUES:
        return 0.0
    severities = window.get("issue_severities")
    severity_values: list[float] = []
    if isinstance(severities, dict):
        for key in ("vocal_projection", "vocal_stem_corr"):
            value = severities.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                severity_values.append(float(value))
    severity_score = window.get("severity_score")
    if isinstance(severity_score, (int, float)) and not isinstance(severity_score, bool):
        severity_values.append(float(severity_score))
    return max(severity_values, default=1.0e-6)


def window_bounds(window: dict[str, Any]) -> tuple[float, float] | None:
    if "start_sec" in window and "duration_sec" in window:
        start_sec = float(window.get("start_sec") or 0.0)
        duration_sec = float(window.get("duration_sec") or 0.0)
        return (start_sec, duration_sec) if duration_sec > 0.0 else None

    window_sec = window.get("window_sec")
    if (
        isinstance(window_sec, list)
        and len(window_sec) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in window_sec)
    ):
        start_sec = float(window_sec[0])
        duration_sec = float(window_sec[1]) - start_sec
        return (start_sec, duration_sec) if duration_sec > 0.0 else None
    return None


def repair_window_score(window: dict[str, Any], *, leak_threshold: float) -> float:
    leakage = float(window.get("vocal_leakage_in_minus_score") or 0.0)
    if leakage > leak_threshold and has_leakage_blocker(window):
        return leakage
    if bool(window.get("blocker")):
        return self_judge_leakage_score(window)
    return 0.0


def normalize_repair_window(window: dict[str, Any], *, score: float) -> dict[str, Any]:
    bounds = window_bounds(window)
    if bounds is None:
        raise ValueError("Cannot normalize repair window without valid timing")
    start_sec, duration_sec = bounds
    normalized = dict(window)
    normalized["start_sec"] = start_sec
    normalized["duration_sec"] = duration_sec
    if "vocal_leakage_in_minus_score" not in normalized:
        normalized["target_window_source"] = "instrumental_self_judge_gate"
        normalized["vocal_leakage_issue_score"] = score
    return normalized


def select_target_windows(
    rows: Iterable[dict[str, Any]],
    *,
    candidate_id: str | None,
    minus_path: str | Path | None,
    leak_threshold: float,
) -> list[dict[str, Any]]:
    selected: dict[tuple[float, float], dict[str, Any]] = {}
    for row in rows:
        if not candidate_matches(row, candidate_id=candidate_id, minus_path=minus_path):
            continue
        windows = row.get("window_metrics")
        window_rows = windows if isinstance(windows, list) else [row]
        for window in window_rows:
            if not isinstance(window, dict):
                continue
            score = repair_window_score(window, leak_threshold=leak_threshold)
            if score <= 0.0 or not has_leakage_blocker(window):
                continue
            bounds = window_bounds(window)
            if bounds is None:
                continue
            start_sec, duration_sec = bounds
            key = (round(start_sec, 6), round(duration_sec, 6))
            previous = selected.get(key)
            previous_score = repair_window_score(previous, leak_threshold=leak_threshold) if previous else 0.0
            if previous is None or score > previous_score:
                selected[key] = normalize_repair_window(window, score=score)
    return [selected[key] for key in sorted(selected)]


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return audio.astype(np.float64), int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float64)


def align_channels_and_length(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    channels = max(array.shape[1] for array in arrays)
    expanded: list[np.ndarray] = []
    for array in arrays:
        if array.shape[1] == channels:
            expanded.append(array)
        elif array.shape[1] == 1:
            expanded.append(np.repeat(array, channels, axis=1))
        else:
            expanded.append(array[:, :channels])
    frame_count = min(array.shape[0] for array in expanded)
    return [array[:frame_count].astype(np.float64, copy=False) for array in expanded]


def build_time_gate(
    *,
    frame_count: int,
    sample_rate: int,
    windows: Sequence[dict[str, Any]],
    margin_sec: float,
    smoothing_sec: float,
) -> np.ndarray:
    gate = np.zeros(frame_count, dtype=np.float64)
    for window in windows:
        start_sec = max(0.0, float(window.get("start_sec") or 0.0) - margin_sec)
        duration_sec = max(0.0, float(window.get("duration_sec") or 0.0) + (2.0 * margin_sec))
        start_frame = max(0, int(round(start_sec * sample_rate)))
        end_frame = min(frame_count, int(round((start_sec + duration_sec) * sample_rate)))
        if end_frame > start_frame:
            gate[start_frame:end_frame] = 1.0
    smoothing_frames = int(round(max(0.0, smoothing_sec) * sample_rate))
    if smoothing_frames > 1 and np.any(gate > 0.0):
        kernel = signal.windows.hann((smoothing_frames * 2) + 1)
        kernel /= float(np.sum(kernel))
        gate = np.convolve(gate, kernel, mode="same")
    return np.clip(gate, 0.0, 1.0)


def bandpass_audio(audio: np.ndarray, sample_rate: int, *, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = sample_rate * 0.5
    low = max(0.0, min(float(low_hz), nyquist - 1.0))
    high = max(low + 1.0, min(float(high_hz), nyquist - 1.0))
    if low <= 0.0 and high >= nyquist - 1.0:
        return audio.astype(np.float64, copy=True)
    if low <= 0.0:
        sos = signal.butter(4, high / nyquist, btype="lowpass", output="sos")
    else:
        sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float64)


def projection_gain(target: np.ndarray, basis: np.ndarray) -> float:
    target_flat = np.asarray(target, dtype=np.float64).reshape(-1)
    basis_flat = np.asarray(basis, dtype=np.float64).reshape(-1)
    return float(abs(np.dot(target_flat, basis_flat)) / (float(np.dot(basis_flat, basis_flat)) + EPS))


def signed_projection_gain(target: np.ndarray, basis: np.ndarray, *, max_projection_gain: float) -> float:
    target_flat = np.asarray(target, dtype=np.float64).reshape(-1)
    basis_flat = np.asarray(basis, dtype=np.float64).reshape(-1)
    gain = float(np.dot(target_flat, basis_flat) / (float(np.dot(basis_flat, basis_flat)) + EPS))
    return float(np.clip(gain, -max_projection_gain, max_projection_gain))


def limit_active_region_peaks(audio: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, float]:
    limited = np.asarray(audio, dtype=np.float64).copy()
    if active.shape[0] != limited.shape[0]:
        raise ValueError("active mask length must match audio frame count")
    if not np.any(active):
        return limited, 0.0
    active_peak = float(np.max(np.abs(limited[active]))) if limited[active].size else 0.0
    if active_peak > 1.0:
        limited[active] /= active_peak
    return limited, active_peak


def apply_projection_repair(
    *,
    base_minus: np.ndarray,
    vocal_audio: np.ndarray,
    sample_rate: int,
    gate: np.ndarray,
    strength: float,
    max_projection_gain: float,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if base_minus.shape != vocal_audio.shape:
        raise ValueError("base_minus and vocal_audio must have the same shape")
    if gate.shape[0] != base_minus.shape[0]:
        raise ValueError("gate length must match audio frame count")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if not 0.0 < max_projection_gain <= MAX_ALLOWED_PROJECTION_GAIN:
        raise ValueError(f"max_projection_gain must be in (0, {MAX_ALLOWED_PROJECTION_GAIN:g}]")

    active = gate > 1.0e-6
    if not np.any(active):
        return base_minus.copy(), {
            "projection_gain_before": 0.0,
            "projection_gain_after": 0.0,
            "applied_projection_gain": 0.0,
            "gate_active_ratio": 0.0,
            "status": "skipped_no_target_windows",
        }

    guide = bandpass_audio(vocal_audio, sample_rate, low_hz=low_hz, high_hz=high_hz)
    repaired = base_minus.astype(np.float64, copy=True)
    gate_column = gate.reshape(-1, 1)
    channel_gains: list[float] = []

    for channel_index in range(repaired.shape[1]):
        gain = signed_projection_gain(
            repaired[active, channel_index],
            guide[active, channel_index],
            max_projection_gain=max_projection_gain,
        )
        channel_gains.append(gain)
        repaired[:, channel_index] -= float(strength) * gain * guide[:, channel_index] * gate

    repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)
    repaired, active_peak = limit_active_region_peaks(repaired, active)

    before = projection_gain(base_minus[active], vocal_audio[active])
    after = projection_gain(repaired[active], vocal_audio[active])
    correction = (base_minus - repaired) * gate_column
    correction_rms = float(np.sqrt(np.mean(np.square(correction)))) if correction.size else 0.0
    return repaired.astype(np.float32), {
        "projection_gain_before": round(before, 9),
        "projection_gain_after": round(after, 9),
        "applied_projection_gain": round(float(max(abs(value) for value in channel_gains)), 9),
        "channel_projection_gains": [round(float(value), 9) for value in channel_gains],
        "gate_active_ratio": round(float(np.mean(gate > 1.0e-6)), 9),
        "correction_rms": round(correction_rms, 9),
        "strength": round(float(strength), 6),
        "max_projection_gain": round(float(max_projection_gain), 6),
        "band_low_hz": round(float(low_hz), 6),
        "band_high_hz": round(float(high_hz), 6),
        "active_peak_before_limit": round(float(active_peak), 9),
        "status": "projection_repair_applied",
        "projection_mode": "global",
    }


def apply_segment_local_projection_repair(
    *,
    base_minus: np.ndarray,
    vocal_audio: np.ndarray,
    sample_rate: int,
    windows: Sequence[dict[str, Any]],
    margin_sec: float,
    smoothing_sec: float,
    strength: float,
    max_projection_gain: float,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if base_minus.shape != vocal_audio.shape:
        raise ValueError("base_minus and vocal_audio must have the same shape")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if not 0.0 < max_projection_gain <= MAX_ALLOWED_PROJECTION_GAIN:
        raise ValueError(f"max_projection_gain must be in (0, {MAX_ALLOWED_PROJECTION_GAIN:g}]")

    guide = bandpass_audio(vocal_audio, sample_rate, low_hz=low_hz, high_hz=high_hz)
    repaired = base_minus.astype(np.float64, copy=True)
    combined_gate = np.zeros(base_minus.shape[0], dtype=np.float64)
    window_metrics: list[dict[str, Any]] = []

    for window_index, window in enumerate(windows):
        window_gate = build_time_gate(
            frame_count=base_minus.shape[0],
            sample_rate=sample_rate,
            windows=[window],
            margin_sec=margin_sec,
            smoothing_sec=smoothing_sec,
        )
        active = window_gate > 1.0e-6
        if not np.any(active):
            continue
        combined_gate = np.maximum(combined_gate, window_gate)
        channel_gains: list[float] = []
        before = projection_gain(repaired[active], vocal_audio[active])
        for channel_index in range(repaired.shape[1]):
            gain = signed_projection_gain(
                repaired[active, channel_index],
                guide[active, channel_index],
                max_projection_gain=max_projection_gain,
            )
            channel_gains.append(gain)
            repaired[:, channel_index] -= float(strength) * gain * guide[:, channel_index] * window_gate
        after = projection_gain(repaired[active], vocal_audio[active])
        window_metrics.append(
            {
                "window_index": int(window_index),
                "start_sec": round(float(window.get("start_sec") or 0.0), 6),
                "duration_sec": round(float(window.get("duration_sec") or 0.0), 6),
                "projection_gain_before": round(before, 9),
                "projection_gain_after": round(after, 9),
                "applied_projection_gain": round(float(max(abs(value) for value in channel_gains)), 9),
                "channel_projection_gains": [round(float(value), 9) for value in channel_gains],
            }
        )

    if not window_metrics:
        return base_minus.copy(), {
            "projection_gain_before": 0.0,
            "projection_gain_after": 0.0,
            "applied_projection_gain": 0.0,
            "window_projection_count": 0,
            "gate_active_ratio": 0.0,
            "status": "skipped_no_target_windows",
            "projection_mode": "segment_local",
        }

    repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)
    active = combined_gate > 1.0e-6
    repaired, active_peak = limit_active_region_peaks(repaired, active)
    before = projection_gain(base_minus[active], vocal_audio[active])
    after = projection_gain(repaired[active], vocal_audio[active])
    gate_column = combined_gate.reshape(-1, 1)
    correction = (base_minus - repaired) * gate_column
    correction_rms = float(np.sqrt(np.mean(np.square(correction)))) if correction.size else 0.0
    applied_gain = max(
        abs(value)
        for item in window_metrics
        for value in item.get("channel_projection_gains", [])
    )
    return repaired.astype(np.float32), {
        "projection_gain_before": round(before, 9),
        "projection_gain_after": round(after, 9),
        "applied_projection_gain": round(float(applied_gain), 9),
        "window_projection_count": len(window_metrics),
        "window_projection_metrics": window_metrics,
        "gate_active_ratio": round(float(np.mean(active)), 9),
        "correction_rms": round(correction_rms, 9),
        "strength": round(float(strength), 6),
        "max_projection_gain": round(float(max_projection_gain), 6),
        "band_low_hz": round(float(low_hz), 6),
        "band_high_hz": round(float(high_hz), 6),
        "active_peak_before_limit": round(float(active_peak), 9),
        "status": "segment_local_projection_repair_applied",
        "projection_mode": "segment_local",
    }


def build_manifest(
    *,
    track_id: str,
    candidate_id: str,
    raw_audio_path: Path,
    base_minus_path: Path,
    primary_vocals_path: Path,
    judge_jsonl_path: Path,
    output_path: Path,
    candidate_row_jsonl: Path,
    sample_rate: int,
    frames: int,
    selected_windows: Sequence[dict[str, Any]],
    repair_metrics: dict[str, Any],
) -> dict[str, Any]:
    candidate_row = {
        "track_id": str(track_id),
        "candidate_id": str(candidate_id),
        "mix_path": str(raw_audio_path),
        "source_audio_path": str(raw_audio_path),
        "vocal_path": str(primary_vocals_path),
        "selected_vocal_path": str(primary_vocals_path),
        "instrumental_path": str(output_path),
        "minus_path": str(output_path),
        "selected_minus_path": str(output_path),
        "base_minus_path": str(base_minus_path),
        "candidate_family": "judge_guided_minus_vocal_leakage_repair",
        "candidate_origin": "judge_guided_projection_repair_candidate",
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "is_complete_pair": True,
    }
    return {
        "schema_version": 1,
        "mode": "judge_guided_minus_vocal_leakage_repair",
        "stem_type": "minus",
        "track_id": str(track_id),
        "candidate_id": str(candidate_id),
        "candidate_only": True,
        "delivery_status": "candidate_only",
        "delivery_allowed": False,
        "presentation_status": "blocked_until_judged",
        "release_status": "not_release_ready",
        "listenable": False,
        "deliverable": False,
        "can_mark_release_ready": False,
        "requires_post_judge": True,
        "no_per_track_hardcoded_timestamps": True,
        "uses_judge_windows": True,
        "raw_audio_path": str(raw_audio_path),
        "base_minus_path": str(base_minus_path),
        "primary_vocals_path": str(primary_vocals_path),
        "judge_jsonl_path": str(judge_jsonl_path),
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
        "artifacts": {
            "minus_judge_guided_leakage_repair_flac": str(output_path),
            "minus_base_flac": str(base_minus_path),
            "primary_vocals_flac": str(primary_vocals_path),
            "raw_ref_audio": str(raw_audio_path),
        },
        "audio": {"sample_rate": int(sample_rate), "frames": int(frames)},
        "repair": {
            "selected_window_count": len(selected_windows),
            "selected_windows": list(selected_windows),
            "metrics": dict(repair_metrics),
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
            "release_ready": False,
        },
        "forbidden_actions": [
            "do_not_export_as_delivery_without_post_judge_acceptance",
            "do_not_mark_release_ready_from_repair_manifest",
            "do_not_relax_gates",
            "do_not_require_reference_stems",
            "do_not_start_training",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a judge-guided minus vocal-leakage repair candidate.")
    parser.add_argument("raw_audio_path", type=Path)
    parser.add_argument("base_minus_path", type=Path)
    parser.add_argument("primary_vocals_path", type=Path)
    parser.add_argument("windowed_judge_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--track-id")
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-label", default="minus_judge_guided_leakage_repair.flac")
    parser.add_argument("--leak-threshold", type=float, default=0.60)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--max-projection-gain", type=float, default=0.75)
    parser.add_argument("--gate-smoothing-sec", type=float, default=0.03)
    parser.add_argument("--window-margin-sec", type=float, default=0.04)
    parser.add_argument("--band-low-hz", type=float, default=160.0)
    parser.add_argument("--band-high-hz", type=float, default=8000.0)
    parser.add_argument("--projection-mode", choices=("global", "segment_local"), default="global")
    return parser.parse_args(argv)


def build_judge_guided_minus_vocal_leakage_repair(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "minus_judge_guided_leakage_repair.flac"
    manifest_path = output_dir / "judge_guided_minus_vocal_leakage_repair_manifest.json"
    candidate_row_jsonl = output_dir / "candidate_pair.jsonl"

    progress(
        "start",
        route="judge_guided_minus_vocal_leakage_repair",
        acceleration_mode="external_only_cpu_scoring",
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="route uses existing candidates and DSP projection repair only",
        output_manifest_path=str(manifest_path),
    )

    rows = read_judge_rows(args.windowed_judge_jsonl)
    selected_windows = select_target_windows(
        rows,
        candidate_id=args.candidate_id,
        minus_path=args.base_minus_path,
        leak_threshold=float(args.leak_threshold),
    )
    if not selected_windows:
        raise ValueError("No judge windows matched candidate/path and vocal leakage threshold")
    progress("judge_windows_selected", selected_window_count=len(selected_windows))

    raw, raw_rate = read_audio(args.raw_audio_path)
    base_minus, base_rate = read_audio(args.base_minus_path)
    primary_vocals, primary_rate = read_audio(args.primary_vocals_path)
    base_minus = resample_audio(base_minus, base_rate, raw_rate)
    primary_vocals = resample_audio(primary_vocals, primary_rate, raw_rate)
    raw, base_minus, primary_vocals = align_channels_and_length([raw, base_minus, primary_vocals])
    progress(
        "input_normalized",
        sample_rate=raw_rate,
        frames=int(base_minus.shape[0]),
        channels=int(base_minus.shape[1]),
    )

    gate = build_time_gate(
        frame_count=base_minus.shape[0],
        sample_rate=raw_rate,
        windows=selected_windows,
        margin_sec=float(args.window_margin_sec),
        smoothing_sec=float(args.gate_smoothing_sec),
    )
    progress("repair_start", gate_active_ratio=round(float(np.mean(gate > 1.0e-6)), 9))
    if str(args.projection_mode) == "segment_local":
        repaired, repair_metrics = apply_segment_local_projection_repair(
            base_minus=base_minus,
            vocal_audio=primary_vocals,
            sample_rate=raw_rate,
            windows=selected_windows,
            margin_sec=float(args.window_margin_sec),
            smoothing_sec=float(args.gate_smoothing_sec),
            strength=float(args.strength),
            max_projection_gain=float(args.max_projection_gain),
            low_hz=float(args.band_low_hz),
            high_hz=float(args.band_high_hz),
        )
    else:
        repaired, repair_metrics = apply_projection_repair(
            base_minus=base_minus,
            vocal_audio=primary_vocals,
            sample_rate=raw_rate,
            gate=gate,
            strength=float(args.strength),
            max_projection_gain=float(args.max_projection_gain),
            low_hz=float(args.band_low_hz),
            high_hz=float(args.band_high_hz),
        )
    sf.write(str(output_path), repaired, raw_rate, subtype="PCM_24")
    progress("output_written", output_audio_path=str(output_path), repair_metrics=repair_metrics)

    if args.listen_dir is not None:
        listen_dir = Path(args.listen_dir)
        listen_dir.mkdir(parents=True, exist_ok=True)
        listen_path = listen_dir / str(args.listen_label)
        shutil.copy2(output_path, listen_path)
        repair_metrics["diagnostic_listen_path"] = str(listen_path)
        progress("diagnostic_listen_written", diagnostic_listen_path=str(listen_path))

    manifest = build_manifest(
        track_id=str(args.track_id or Path(args.raw_audio_path).stem),
        candidate_id=f"{str(args.candidate_id or Path(args.base_minus_path).stem)}__judge_guided_leakage_repair",
        raw_audio_path=args.raw_audio_path,
        base_minus_path=args.base_minus_path,
        primary_vocals_path=args.primary_vocals_path,
        judge_jsonl_path=args.windowed_judge_jsonl,
        output_path=output_path,
        candidate_row_jsonl=candidate_row_jsonl,
        sample_rate=raw_rate,
        frames=int(repaired.shape[0]),
        selected_windows=selected_windows,
        repair_metrics=repair_metrics,
    )
    write_jsonl(candidate_row_jsonl, [manifest["candidate_row"]])
    write_json(manifest_path, manifest)
    progress(
        "end",
        output_manifest_path=str(manifest_path),
        delivery_status=manifest["delivery_status"],
        delivery_allowed=manifest["delivery_allowed"],
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="apply_judge_guided_minus_vocal_leakage_repair.py",
        entrypoint_path="/app/tools/apply_judge_guided_minus_vocal_leakage_repair.py",
    )
    args = parse_args(argv)
    manifest = build_judge_guided_minus_vocal_leakage_repair(args)
    print(
        json.dumps(
            {
                "manifest_path": str(Path(args.output_dir) / "judge_guided_minus_vocal_leakage_repair_manifest.json"),
                "candidate_audio_path": manifest["artifacts"]["minus_judge_guided_leakage_repair_flac"],
                "candidate_row_jsonl": manifest["candidate_row_jsonl"],
                "selected_window_count": manifest["repair"]["selected_window_count"],
                "delivery_status": manifest["delivery_status"],
                "delivery_allowed": manifest["delivery_allowed"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
