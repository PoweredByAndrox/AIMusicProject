from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from apply_judge_guided_minus_vocal_leakage_repair import (  # noqa: E402
    align_channels_and_length,
    as_list,
    build_time_gate,
    normalize_path_text,
    read_audio,
    read_jsonl,
    resample_audio,
    window_bounds,
)

import numpy as np
import soundfile as sf


RENDER_STATUS = "rendered_vocal_judge_guided_silence_repair_candidate"
ROUTE_NAME = "judge_guided_vocal_silence_hallucination_repair"
START_TIME = time.monotonic()
EPS = 1.0e-9


def progress(phase: str, **fields: Any) -> None:
    payload = {
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "route": ROUTE_NAME,
        "acceleration_mode": "external_only_cpu_dsp_repair",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "route uses existing vocal candidates and windowed judge rows only",
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def candidate_matches(row: dict[str, Any], *, candidate_id: str | None, vocal_path: str | Path | None) -> bool:
    if candidate_id and str(row.get("candidate_id") or "") == str(candidate_id):
        return True
    requested_path = normalize_path_text(vocal_path)
    if not requested_path:
        return False
    row_paths = [
        normalize_path_text(row.get("vocal_path")),
        normalize_path_text(row.get("selected_vocal_path")),
        normalize_path_text(row.get("rendered_vocal_candidate_path")),
    ]
    return requested_path in row_paths


def repair_window_score(window: dict[str, Any], *, hallucination_threshold: float) -> float:
    score = float(window.get("silence_hallucination_score") or 0.0)
    reasons = set(as_list(window.get("reasons")) + as_list(window.get("vocal_blockers")))
    if "silence_hallucination" in reasons:
        return max(score, 1.0e-6)
    return score if score >= float(hallucination_threshold) else 0.0


def normalize_repair_window(window: dict[str, Any], *, score: float) -> dict[str, Any]:
    bounds = window_bounds(window)
    if bounds is None:
        raise ValueError("Cannot normalize vocal repair window without valid timing")
    start_sec, duration_sec = bounds
    normalized = dict(window)
    normalized["start_sec"] = round(float(start_sec), 6)
    normalized["duration_sec"] = round(float(duration_sec), 6)
    normalized["target_window_source"] = "no_reference_vocal_windowed_judge"
    normalized["silence_hallucination_score"] = round(float(score), 9)
    return normalized


def select_target_windows(
    rows: Iterable[dict[str, Any]],
    *,
    candidate_id: str | None,
    vocal_path: str | Path | None,
    hallucination_threshold: float,
) -> list[dict[str, Any]]:
    selected: dict[tuple[float, float], dict[str, Any]] = {}
    for row in rows:
        if not candidate_matches(row, candidate_id=candidate_id, vocal_path=vocal_path):
            continue
        windows = row.get("window_metrics")
        window_rows = windows if isinstance(windows, list) else [row]
        for window in window_rows:
            if not isinstance(window, dict):
                continue
            score = repair_window_score(window, hallucination_threshold=hallucination_threshold)
            if score <= 0.0:
                continue
            bounds = window_bounds(window)
            if bounds is None:
                continue
            start_sec, duration_sec = bounds
            key = (round(float(start_sec), 6), round(float(duration_sec), 6))
            previous = selected.get(key)
            previous_score = repair_window_score(previous, hallucination_threshold=hallucination_threshold) if previous else 0.0
            if previous is None or score > previous_score:
                selected[key] = normalize_repair_window(window, score=score)
    return [selected[key] for key in sorted(selected)]


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return audio.mean(axis=1).astype(np.float64)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))) + EPS)


def frame_rms(audio: np.ndarray, window_frames: int) -> np.ndarray:
    values: list[float] = []
    for start in range(0, audio.size, window_frames):
        segment = audio[start : min(start + window_frames, audio.size)]
        if segment.size:
            values.append(rms(segment))
    return np.asarray(values or [0.0], dtype=np.float64)


def build_local_quiet_vocal_gate(
    *,
    source_audio: np.ndarray,
    base_vocal: np.ndarray,
    sample_rate: int,
    windows: Sequence[dict[str, Any]],
    margin_sec: float,
    smoothing_sec: float,
    subwindow_sec: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    gate = np.zeros(base_vocal.shape[0], dtype=np.float64)
    window_frames = max(1, int(round(float(sample_rate) * float(subwindow_sec))))
    window_metrics: list[dict[str, Any]] = []
    selected_subwindow_count = 0
    quiet_subwindow_count = 0
    active_vocal_subwindow_count = 0

    for window_index, window in enumerate(windows):
        start_sec = max(0.0, float(window.get("start_sec") or 0.0) - float(margin_sec))
        duration_sec = max(0.0, float(window.get("duration_sec") or 0.0) + (2.0 * float(margin_sec)))
        start_frame = max(0, int(round(start_sec * sample_rate)))
        end_frame = min(base_vocal.shape[0], int(round((start_sec + duration_sec) * sample_rate)))
        if end_frame <= start_frame:
            continue
        local_mix = source_audio[start_frame:end_frame]
        local_vocal = base_vocal[start_frame:end_frame]
        mix_mono = mono(local_mix)
        vocal_mono = mono(local_vocal)
        mix_windows = frame_rms(mix_mono, window_frames)
        vocal_windows = frame_rms(vocal_mono, window_frames)
        mix_active_threshold = max(float(np.percentile(mix_windows, 30)) * 0.6, rms(mix_mono) * 0.05, EPS)
        vocal_active_threshold = max(float(np.percentile(vocal_windows, 55)) * 0.7, rms(vocal_mono) * 0.08, EPS)
        quiet_mix = mix_windows <= mix_active_threshold
        active_vocal = vocal_windows > vocal_active_threshold
        target_mask = quiet_mix
        quiet_subwindow_count += int(np.count_nonzero(quiet_mix))
        active_vocal_subwindow_count += int(np.count_nonzero(active_vocal))
        selected_subwindow_count += int(np.count_nonzero(target_mask))
        for subwindow_index, selected in enumerate(target_mask.astype(bool).tolist()):
            if not selected:
                continue
            sub_start = start_frame + (subwindow_index * window_frames)
            sub_end = min(end_frame, sub_start + window_frames)
            if sub_end > sub_start:
                gate[sub_start:sub_end] = 1.0
        window_metrics.append(
            {
                "window_index": int(window_index),
                "start_sec": round(float(window.get("start_sec") or 0.0), 6),
                "duration_sec": round(float(window.get("duration_sec") or 0.0), 6),
                "mix_active_threshold": round(float(mix_active_threshold), 9),
                "vocal_active_threshold": round(float(vocal_active_threshold), 9),
                "quiet_subwindow_count": int(np.count_nonzero(quiet_mix)),
                "active_vocal_subwindow_count": int(np.count_nonzero(active_vocal)),
                "selected_subwindow_count": int(np.count_nonzero(target_mask)),
            }
        )

    smoothing_frames = int(round(max(0.0, float(smoothing_sec)) * sample_rate))
    if smoothing_frames > 1 and np.any(gate > 0.0):
        kernel = np.hanning((smoothing_frames * 2) + 1)
        kernel /= float(np.sum(kernel))
        gate = np.convolve(gate, kernel, mode="same")
    return np.clip(gate, 0.0, 1.0), {
        "window_subwindow_metrics": window_metrics,
        "quiet_subwindow_count": int(quiet_subwindow_count),
        "active_vocal_subwindow_count": int(active_vocal_subwindow_count),
        "selected_subwindow_count": int(selected_subwindow_count),
        "subwindow_sec": round(float(subwindow_sec), 6),
    }


def apply_vocal_silence_attenuation(
    *,
    source_audio: np.ndarray,
    base_vocal: np.ndarray,
    sample_rate: int,
    windows: Sequence[dict[str, Any]],
    margin_sec: float,
    smoothing_sec: float,
    attenuation_gain: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_gain = float(np.clip(float(attenuation_gain), 0.0, 1.0))
    gate, gate_metrics = build_local_quiet_vocal_gate(
        source_audio=source_audio,
        base_vocal=base_vocal,
        sample_rate=sample_rate,
        windows=windows,
        margin_sec=float(margin_sec),
        smoothing_sec=float(smoothing_sec),
    )
    active = gate > 1.0e-6
    if not np.any(active):
        return base_vocal.astype(np.float32, copy=True), {
            "status": "skipped_no_target_windows",
            "selected_window_count": len(windows),
            "gate_active_ratio": 0.0,
            "attenuation_gain": round(target_gain, 6),
            **gate_metrics,
        }

    gain = 1.0 - gate.reshape(-1, 1) * (1.0 - target_gain)
    repaired = (base_vocal * gain).astype(np.float64)
    correction = (base_vocal - repaired) * gate.reshape(-1, 1)
    before = float(np.sqrt(np.mean(np.square(base_vocal[active]))) + 1.0e-12)
    after = float(np.sqrt(np.mean(np.square(repaired[active]))) + 1.0e-12)
    correction_rms = float(np.sqrt(np.mean(np.square(correction))) + 1.0e-12)
    active_peak = float(np.max(np.abs(repaired[active]))) if repaired[active].size else 0.0
    if active_peak > 1.0:
        repaired[active] /= active_peak
    return repaired.astype(np.float32), {
        "status": "vocal_silence_hallucination_attenuation_applied",
        "selected_window_count": len(windows),
        "gate_active_ratio": round(float(np.mean(active)), 9),
        "attenuation_gain": round(target_gain, 6),
        "active_vocal_rms_before": round(before, 9),
        "active_vocal_rms_after": round(after, 9),
        "active_vocal_rms_ratio": round(after / (before + 1.0e-12), 9),
        "correction_rms": round(correction_rms, 9),
        "active_peak_before_limit": round(active_peak, 9),
        "window_margin_sec": round(float(margin_sec), 6),
        "gate_smoothing_sec": round(float(smoothing_sec), 6),
        **gate_metrics,
    }


def build_manifest(
    *,
    track_id: str,
    candidate_id: str,
    raw_audio_path: Path,
    base_vocal_path: Path,
    instrumental_path: Path,
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
        "vocal_path": str(output_path),
        "selected_vocal_path": str(output_path),
        "base_vocal_path": str(base_vocal_path),
        "instrumental_path": str(instrumental_path),
        "minus_path": str(instrumental_path),
        "selected_minus_path": str(instrumental_path),
        "candidate_family": ROUTE_NAME,
        "candidate_origin": "windowed_judge_guided_silence_repair_candidate",
        "render_status": RENDER_STATUS,
        "rendered_vocal_candidate_path": str(output_path),
        "requires_post_render_vocal_judge": True,
        "requires_dense_post_reconciliation_judge": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "is_complete_vocal_candidate": True,
        "score_mode": "vocal",
        "window_decisions": list(selected_windows),
    }
    return {
        "schema_version": 1,
        "mode": ROUTE_NAME,
        "stem_type": "vocal",
        "track_id": str(track_id),
        "candidate_id": str(candidate_id),
        "candidate_only": True,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "listenable": False,
        "deliverable": False,
        "can_mark_release_ready": False,
        "can_publish_as_result": False,
        "render_status": RENDER_STATUS,
        "rendered_vocal_candidate_path": str(output_path),
        "requires_post_judge": True,
        "requires_post_render_vocal_judge": True,
        "no_per_track_hardcoded_timestamps": True,
        "uses_judge_windows": True,
        "raw_audio_path": str(raw_audio_path),
        "base_vocal_path": str(base_vocal_path),
        "instrumental_path": str(instrumental_path),
        "judge_jsonl_path": str(judge_jsonl_path),
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
        "artifacts": {
            "vocal_judge_guided_silence_repair_wav": str(output_path),
            "base_vocal_wav": str(base_vocal_path),
            "instrumental_flac": str(instrumental_path),
            "raw_ref_audio": str(raw_audio_path),
        },
        "audio": {"sample_rate": int(sample_rate), "frames": int(frames)},
        "repair": {
            "selected_window_count": len(selected_windows),
            "selected_windows": list(selected_windows),
            "metrics": dict(repair_metrics),
        },
        "execution_contract": {
            "route_name": ROUTE_NAME,
            "docker_runtime_required": True,
            "expected_acceleration_mode": "external_only_cpu_dsp_repair",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "route consumes precomputed audio and windowed vocal judge rows",
            "per_track_hardcode_allowed": False,
            "fallback_used": False,
            "proxy_only_scoring": True,
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


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Judge-Guided Vocal Silence Hallucination Repair",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- render_status: {manifest.get('render_status')}",
        f"- candidate_id: {manifest.get('candidate_id')}",
        f"- selected_window_count: {manifest.get('repair', {}).get('selected_window_count')}",
        f"- rendered_vocal_candidate_path: {manifest.get('rendered_vocal_candidate_path')}",
        "",
        "## Safety",
        "",
    ]
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {}
    for key in sorted(safety):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_judge_guided_vocal_silence_hallucination_repair(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vocal_judge_guided_silence_repair_candidate.wav"
    manifest_path = output_dir / "judge_guided_vocal_silence_hallucination_repair_manifest.json"
    summary_path = output_dir / "judge_guided_vocal_silence_hallucination_repair_summary.md"
    candidate_row_jsonl = output_dir / "candidate_pair.jsonl"

    progress("start", output_manifest_path=str(manifest_path))
    rows = read_jsonl(Path(args.windowed_judge_jsonl))
    selected_windows = select_target_windows(
        rows,
        candidate_id=args.candidate_id,
        vocal_path=args.base_vocal_path,
        hallucination_threshold=float(args.hallucination_threshold),
    )
    if not selected_windows:
        raise ValueError("No vocal judge windows matched candidate/path and silence hallucination evidence")
    progress("judge_windows_selected", selected_window_count=len(selected_windows))

    raw, raw_rate = read_audio(Path(args.raw_audio_path))
    base_vocal, vocal_rate = read_audio(Path(args.base_vocal_path))
    instrumental, instrumental_rate = read_audio(Path(args.instrumental_path))
    base_vocal = resample_audio(base_vocal, vocal_rate, raw_rate)
    instrumental = resample_audio(instrumental, instrumental_rate, raw_rate)
    raw, base_vocal, instrumental = align_channels_and_length([raw, base_vocal, instrumental])
    progress(
        "input_normalized",
        sample_rate=raw_rate,
        frames=int(base_vocal.shape[0]),
        channels=int(base_vocal.shape[1]),
    )

    repaired, repair_metrics = apply_vocal_silence_attenuation(
        source_audio=raw,
        base_vocal=base_vocal,
        sample_rate=raw_rate,
        windows=selected_windows,
        margin_sec=float(args.window_margin_sec),
        smoothing_sec=float(args.gate_smoothing_sec),
        attenuation_gain=float(args.attenuation_gain),
    )
    sf.write(str(output_path), repaired, raw_rate, subtype="FLOAT")
    progress("output_written", output_audio_path=str(output_path), repair_metrics=repair_metrics)

    candidate_id = f"{str(args.candidate_id or Path(args.base_vocal_path).stem)}__judge_guided_silence_repair"
    manifest = build_manifest(
        track_id=str(args.track_id or Path(args.raw_audio_path).stem),
        candidate_id=candidate_id,
        raw_audio_path=Path(args.raw_audio_path),
        base_vocal_path=Path(args.base_vocal_path),
        instrumental_path=Path(args.instrumental_path),
        judge_jsonl_path=Path(args.windowed_judge_jsonl),
        output_path=output_path,
        candidate_row_jsonl=candidate_row_jsonl,
        sample_rate=raw_rate,
        frames=int(repaired.shape[0]),
        selected_windows=selected_windows,
        repair_metrics=repair_metrics,
    )
    write_jsonl(candidate_row_jsonl, [manifest["candidate_row"]])
    write_json(manifest_path, manifest)
    write_summary(summary_path, manifest)
    progress(
        "end",
        output_manifest_path=str(manifest_path),
        delivery_status=manifest["delivery_status"],
        delivery_allowed=manifest["delivery_allowed"],
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a judge-guided vocal silence hallucination repair candidate.")
    parser.add_argument("raw_audio_path", type=Path)
    parser.add_argument("base_vocal_path", type=Path)
    parser.add_argument("instrumental_path", type=Path)
    parser.add_argument("windowed_judge_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--track-id")
    parser.add_argument("--hallucination-threshold", type=float, default=0.75)
    parser.add_argument("--attenuation-gain", type=float, default=0.0)
    parser.add_argument("--gate-smoothing-sec", type=float, default=0.03)
    parser.add_argument("--window-margin-sec", type=float, default=0.04)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="apply_judge_guided_vocal_silence_hallucination_repair.py",
        entrypoint_path="/app/tools/apply_judge_guided_vocal_silence_hallucination_repair.py",
    )
    args = parse_args(argv)
    manifest = build_judge_guided_vocal_silence_hallucination_repair(args)
    print(
        json.dumps(
            {
                "manifest_path": str(Path(args.output_dir) / "judge_guided_vocal_silence_hallucination_repair_manifest.json"),
                "candidate_audio_path": manifest["artifacts"]["vocal_judge_guided_silence_repair_wav"],
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
