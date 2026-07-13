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

from artifact_name_policy import indexed_audio_filename  # noqa: E402
from docker_runtime import cli_runtime_contract, require_docker_runtime  # noqa: E402
from apply_judge_guided_minus_vocal_leakage_repair import (  # noqa: E402
    align_channels_and_length,
    as_list,
    bandpass_audio,
    build_time_gate,
    candidate_matches,
    limit_active_region_peaks,
    projection_gain,
    read_audio,
    read_jsonl,
    resample_audio,
    signed_projection_gain,
    window_bounds,
)

import numpy as np
import soundfile as sf


DETAIL_ISSUES = {"air_loss_proxy", "presence_loss_proxy", "low_rms_muffling_proxy"}
UNSAFE_ISSUES = {"vocal_projection_proxy", "vocal_stem_corr_proxy", "full_mix_like_risk_proxy"}
START_TIME = time.monotonic()


def progress(phase: str, **fields: Any) -> None:
    payload = {
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "route": "judge_guided_minus_air_detail_repair",
        "acceleration_mode": "external_only_cpu_dsp_repair",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "route uses existing candidates and CPU/DSP detail repair only",
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio)) + 1.0e-12))


def detail_issue_score(window: dict[str, Any]) -> float:
    issues = set(as_list(window.get("issues")))
    if not issues & DETAIL_ISSUES or issues & UNSAFE_ISSUES:
        return 0.0
    severities = window.get("issue_severities")
    values: list[float] = []
    if isinstance(severities, dict):
        for key in ("air_loss", "presence_loss", "low_rms_ratio"):
            value = severities.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    severity_score = window.get("severity_score")
    if isinstance(severity_score, (int, float)) and not isinstance(severity_score, bool):
        values.append(float(severity_score))
    return max(values, default=1.0e-6)


def normalize_repair_window(window: dict[str, Any], *, score: float) -> dict[str, Any]:
    bounds = window_bounds(window)
    if bounds is None:
        raise ValueError("Cannot normalize repair window without valid timing")
    start_sec, duration_sec = bounds
    normalized = dict(window)
    normalized["start_sec"] = start_sec
    normalized["duration_sec"] = duration_sec
    normalized["target_window_source"] = "instrumental_self_judge_gate"
    normalized["detail_issue_score"] = score
    return normalized


def select_target_windows(
    rows: Iterable[dict[str, Any]],
    *,
    candidate_id: str | None,
    minus_path: str | Path | None,
) -> list[dict[str, Any]]:
    selected: dict[tuple[float, float], dict[str, Any]] = {}
    for row in rows:
        if not candidate_matches(row, candidate_id=candidate_id, minus_path=minus_path):
            continue
        windows = row.get("window_metrics")
        window_rows = windows if isinstance(windows, list) else [row]
        for window in window_rows:
            if not isinstance(window, dict) or not bool(window.get("blocker")):
                continue
            score = detail_issue_score(window)
            if score <= 0.0:
                continue
            bounds = window_bounds(window)
            if bounds is None:
                continue
            start_sec, duration_sec = bounds
            key = (round(start_sec, 6), round(duration_sec, 6))
            previous = selected.get(key)
            previous_score = detail_issue_score(previous) if previous else 0.0
            if previous is None or score > previous_score:
                selected[key] = normalize_repair_window(window, score=score)
    return [selected[key] for key in sorted(selected)]


def _window_rms_gain(
    window: dict[str, Any],
    *,
    base_gain_strength: float,
    low_rms_target_ratio: float,
    max_base_gain: float,
) -> float:
    metrics = window.get("metrics")
    ratio = None
    if isinstance(metrics, dict) and isinstance(metrics.get("rms_ratio"), (int, float)):
        ratio = float(metrics["rms_ratio"])
    if ratio is None or ratio >= low_rms_target_ratio or low_rms_target_ratio <= 0.0:
        return 1.0
    required = low_rms_target_ratio / max(ratio, 1.0e-6)
    blended = 1.0 + max(0.0, required - 1.0) * max(0.0, base_gain_strength)
    return float(min(max_base_gain, max(1.0, blended)))


def _suppress_vocal_projection(
    layer: np.ndarray,
    vocal_audio: np.ndarray,
    gate: np.ndarray,
    *,
    strength: float,
    max_projection_gain: float,
) -> tuple[np.ndarray, dict[str, float]]:
    active = gate > 1.0e-6
    if strength <= 0.0 or not np.any(active):
        return layer, {
            "vocal_projection_gain_before": 0.0,
            "vocal_projection_gain_after": 0.0,
            "vocal_projection_gain_removed": 0.0,
        }
    if max_projection_gain <= 0.0:
        raise ValueError("max_vocal_projection_gain must be positive when vocal suppression is enabled")
    suppressed = layer.astype(np.float64, copy=True)
    before = projection_gain(suppressed[active], vocal_audio[active])
    removed_gains: list[float] = []
    for channel_index in range(suppressed.shape[1]):
        gain = signed_projection_gain(
            suppressed[active, channel_index],
            vocal_audio[active, channel_index],
            max_projection_gain=max_projection_gain,
        )
        removed_gains.append(abs(gain))
        suppressed[:, channel_index] -= (
            float(strength) * gain * vocal_audio[:, channel_index] * gate
        )
    after = projection_gain(suppressed[active], vocal_audio[active])
    return suppressed, {
        "vocal_projection_gain_before": round(before, 9),
        "vocal_projection_gain_after": round(after, 9),
        "vocal_projection_gain_removed": round(float(max(removed_gains, default=0.0)), 9),
    }


def apply_air_detail_repair(
    *,
    source_audio: np.ndarray,
    base_minus: np.ndarray,
    vocal_audio: np.ndarray | None,
    sample_rate: int,
    windows: Sequence[dict[str, Any]],
    margin_sec: float,
    smoothing_sec: float,
    presence_delta_strength: float,
    air_delta_strength: float,
    body_delta_strength: float,
    base_gain_strength: float,
    low_rms_target_ratio: float,
    max_base_gain: float,
    vocal_suppression_strength: float,
    max_vocal_projection_gain: float,
    presence_low_hz: float,
    presence_high_hz: float,
    air_low_hz: float,
    air_high_hz: float,
    body_low_hz: float,
    body_high_hz: float,
    vocal_suppression_mode: str = "global",
) -> tuple[np.ndarray, dict[str, Any]]:
    if source_audio.shape != base_minus.shape:
        raise ValueError("source_audio and base_minus must have the same shape")
    if vocal_audio is not None and vocal_audio.shape != base_minus.shape:
        raise ValueError("vocal_audio must match base_minus shape")
    if vocal_suppression_mode not in {"global", "segment_local"}:
        raise ValueError("vocal_suppression_mode must be 'global' or 'segment_local'")
    if not windows:
        return base_minus.copy(), {
            "status": "skipped_no_target_windows",
            "selected_window_count": 0,
            "gate_active_ratio": 0.0,
        }

    source_delta = source_audio.astype(np.float64) - base_minus.astype(np.float64)
    detail_basis = (
        float(presence_delta_strength)
        * bandpass_audio(source_delta, sample_rate, low_hz=presence_low_hz, high_hz=presence_high_hz)
        + float(air_delta_strength)
        * bandpass_audio(source_delta, sample_rate, low_hz=air_low_hz, high_hz=air_high_hz)
        + float(body_delta_strength)
        * bandpass_audio(source_delta, sample_rate, low_hz=body_low_hz, high_hz=body_high_hz)
    )
    added_layer = np.zeros_like(base_minus, dtype=np.float64)
    unsuppressed_added_layer = np.zeros_like(base_minus, dtype=np.float64)
    gain_envelope = np.ones(base_minus.shape[0], dtype=np.float64)
    combined_gate = np.zeros(base_minus.shape[0], dtype=np.float64)
    window_metrics: list[dict[str, Any]] = []
    window_vocal_metrics: list[dict[str, Any]] = []
    vocal_detail = None
    if vocal_audio is not None:
        vocal_detail = bandpass_audio(
            vocal_audio,
            sample_rate,
            low_hz=min(body_low_hz, presence_low_hz, air_low_hz),
            high_hz=max(body_high_hz, presence_high_hz, air_high_hz),
        )

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
        window_layer = detail_basis * window_gate.reshape(-1, 1)
        unsuppressed_added_layer += window_layer
        if vocal_detail is not None and vocal_suppression_mode == "segment_local":
            window_layer, local_vocal_metrics = _suppress_vocal_projection(
                window_layer,
                vocal_detail,
                window_gate,
                strength=vocal_suppression_strength,
                max_projection_gain=max_vocal_projection_gain,
            )
            window_vocal_metrics.append(
                {
                    "window_index": int(window_index),
                    "start_sec": round(float(window.get("start_sec") or 0.0), 6),
                    "duration_sec": round(float(window.get("duration_sec") or 0.0), 6),
                    **local_vocal_metrics,
                }
            )
        added_layer += window_layer
        base_gain = _window_rms_gain(
            window,
            base_gain_strength=base_gain_strength,
            low_rms_target_ratio=low_rms_target_ratio,
            max_base_gain=max_base_gain,
        )
        gain_envelope = np.maximum(gain_envelope, 1.0 + ((base_gain - 1.0) * window_gate))
        window_metrics.append(
            {
                "window_index": int(window_index),
                "start_sec": round(float(window.get("start_sec") or 0.0), 6),
                "duration_sec": round(float(window.get("duration_sec") or 0.0), 6),
                "base_gain": round(float(base_gain), 9),
                "detail_issue_score": round(float(window.get("detail_issue_score") or 0.0), 9),
            }
        )

    active = combined_gate > 1.0e-6
    vocal_metrics = {
        "vocal_projection_gain_before": 0.0,
        "vocal_projection_gain_after": 0.0,
        "vocal_projection_gain_removed": 0.0,
    }
    if vocal_detail is not None and vocal_suppression_mode == "global":
        added_layer, vocal_metrics = _suppress_vocal_projection(
            added_layer,
            vocal_detail,
            combined_gate,
            strength=vocal_suppression_strength,
            max_projection_gain=max_vocal_projection_gain,
        )
    elif vocal_detail is not None and vocal_suppression_mode == "segment_local":
        vocal_metrics = {
            "vocal_projection_gain_before": round(
                projection_gain(unsuppressed_added_layer[active], vocal_detail[active]), 9
            ),
            "vocal_projection_gain_after": round(
                projection_gain(added_layer[active], vocal_detail[active]), 9
            ),
            "vocal_projection_gain_removed": round(
                max(
                    (
                        float(item.get("vocal_projection_gain_removed") or 0.0)
                        for item in window_vocal_metrics
                    ),
                    default=0.0,
                ),
                9,
            ),
        }

    repaired = (base_minus.astype(np.float64) * gain_envelope.reshape(-1, 1)) + added_layer
    repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)
    repaired, active_peak = limit_active_region_peaks(repaired, active)
    repair_delta = (repaired - base_minus.astype(np.float64)) * combined_gate.reshape(-1, 1)
    metrics = {
        "status": "air_detail_repair_applied" if window_metrics else "skipped_no_active_target_windows",
        "selected_window_count": len(window_metrics),
        "window_metrics": window_metrics,
        "gate_active_ratio": round(float(np.mean(active)), 9),
        "added_layer_rms": round(rms(added_layer[active]) if np.any(active) else 0.0, 9),
        "repair_delta_rms": round(rms(repair_delta[active]) if np.any(active) else 0.0, 9),
        "max_base_gain": round(float(np.max(gain_envelope)), 9),
        "active_peak_before_limit": round(float(active_peak), 9),
        "presence_delta_strength": round(float(presence_delta_strength), 6),
        "air_delta_strength": round(float(air_delta_strength), 6),
        "body_delta_strength": round(float(body_delta_strength), 6),
        "base_gain_strength": round(float(base_gain_strength), 6),
        "vocal_suppression_mode": str(vocal_suppression_mode),
        "window_vocal_projection_count": len(window_vocal_metrics),
        "window_vocal_projection_metrics": window_vocal_metrics,
        **vocal_metrics,
    }
    return repaired.astype(np.float32), metrics


def build_manifest(
    *,
    track_id: str,
    candidate_id: str,
    raw_audio_path: Path,
    base_minus_path: Path,
    vocal_guide_path: Path | None,
    judge_jsonl_path: Path,
    output_path: Path,
    candidate_row_jsonl: Path,
    sample_rate: int,
    frames: int,
    selected_windows: Sequence[dict[str, Any]],
    repair_metrics: dict[str, Any],
    runtime_environment: str = "docker",
    native_runtime_allowed: bool = False,
) -> dict[str, Any]:
    candidate_row = {
        "track_id": str(track_id),
        "candidate_id": str(candidate_id),
        "mix_path": str(raw_audio_path),
        "source_audio_path": str(raw_audio_path),
        "vocal_path": str(vocal_guide_path) if vocal_guide_path is not None else "",
        "selected_vocal_path": str(vocal_guide_path) if vocal_guide_path is not None else "",
        "instrumental_path": str(output_path),
        "minus_path": str(output_path),
        "selected_minus_path": str(output_path),
        "base_minus_path": str(base_minus_path),
        "candidate_family": "judge_guided_minus_air_detail_repair",
        "candidate_origin": "self_judge_guided_air_detail_repair_candidate",
        "render_status": "rendered_minus_air_detail_repair_candidate",
        "rendered_minus_candidate_path": str(output_path),
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "delivery_allowed": False,
        "delivery_status": "candidate_only",
        "presentation_status": "blocked_until_judged",
        "release_status": "not_release_ready",
        "is_complete_pair": True,
    }
    return {
        "schema_version": 1,
        "mode": "judge_guided_minus_air_detail_repair",
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
        "render_status": "rendered_minus_air_detail_repair_candidate",
        "rendered_minus_candidate_path": str(output_path),
        "requires_post_judge": True,
        "requires_post_render_minus_self_judge": True,
        "no_per_track_hardcoded_timestamps": True,
        "uses_judge_windows": True,
        "raw_audio_path": str(raw_audio_path),
        "base_minus_path": str(base_minus_path),
        "vocal_guide_path": str(vocal_guide_path) if vocal_guide_path is not None else None,
        "judge_jsonl_path": str(judge_jsonl_path),
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
        "artifacts": {
            "minus_air_detail_repair_flac": str(output_path),
            "minus_base_flac": str(base_minus_path),
            "vocal_guide_flac": str(vocal_guide_path) if vocal_guide_path is not None else None,
            "raw_ref_audio": str(raw_audio_path),
        },
        "audio": {"sample_rate": int(sample_rate), "frames": int(frames)},
        "repair": {
            "selected_window_count": len(selected_windows),
            "selected_windows": list(selected_windows),
            "vocal_suppression_mode": str(repair_metrics.get("vocal_suppression_mode") or "global"),
            "metrics": dict(repair_metrics),
        },
        "execution_contract": {
            "route_name": "judge_guided_minus_air_detail_repair",
            "docker_runtime_required": not bool(native_runtime_allowed),
            "expected_acceleration_mode": "external_only_cpu_dsp_repair",
            "runtime_environment": str(runtime_environment),
            "native_runtime_allowed": bool(native_runtime_allowed),
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "route consumes precomputed audio and self-judge rows",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a judge-guided minus air/detail repair candidate.")
    parser.add_argument("raw_audio_path", type=Path)
    parser.add_argument("base_minus_path", type=Path)
    parser.add_argument("self_judge_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--track-id")
    parser.add_argument("--vocal-guide-path", type=Path)
    parser.add_argument("--gate-smoothing-sec", type=float, default=0.03)
    parser.add_argument("--window-margin-sec", type=float, default=0.04)
    parser.add_argument("--presence-delta-strength", type=float, default=0.18)
    parser.add_argument("--air-delta-strength", type=float, default=0.34)
    parser.add_argument("--body-delta-strength", type=float, default=0.04)
    parser.add_argument("--base-gain-strength", type=float, default=0.18)
    parser.add_argument("--low-rms-target-ratio", type=float, default=0.52)
    parser.add_argument("--max-base-gain", type=float, default=1.18)
    parser.add_argument("--vocal-suppression-strength", type=float, default=0.75)
    parser.add_argument("--max-vocal-projection-gain", type=float, default=8.0)
    parser.add_argument(
        "--vocal-suppression-mode",
        choices=("global", "segment_local"),
        default="global",
    )
    parser.add_argument("--presence-low-hz", type=float, default=1800.0)
    parser.add_argument("--presence-high-hz", type=float, default=6000.0)
    parser.add_argument("--air-low-hz", type=float, default=6000.0)
    parser.add_argument("--air-high-hz", type=float, default=12000.0)
    parser.add_argument("--body-low-hz", type=float, default=120.0)
    parser.add_argument("--body-high-hz", type=float, default=1800.0)
    parser.add_argument("--short-output-names", action="store_true")
    parser.add_argument(
        "--allow-native-runtime",
        action="store_true",
        help="Explicitly allow native Mac CPU-DSP diagnostic execution outside Docker.",
    )
    return parser.parse_args(argv)


def build_judge_guided_minus_air_detail_repair(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    short_output_names = bool(getattr(args, "short_output_names", False))
    output_path = output_dir / (indexed_audio_filename("m", 1) if short_output_names else "minus_judge_guided_air_detail_repair.flac")
    manifest_path = output_dir / ("manifest.json" if short_output_names else "judge_guided_minus_air_detail_repair_manifest.json")
    candidate_row_jsonl = output_dir / ("rows.jsonl" if short_output_names else "candidate_pair.jsonl")

    progress("start", output_manifest_path=str(manifest_path))
    rows = read_jsonl(args.self_judge_jsonl)
    selected_windows = select_target_windows(
        rows,
        candidate_id=args.candidate_id,
        minus_path=args.base_minus_path,
    )
    if not selected_windows:
        raise ValueError("No self-judge air/detail windows matched candidate/path without unsafe vocal/full-mix issues")
    progress("judge_windows_selected", selected_window_count=len(selected_windows))

    raw, raw_rate = read_audio(args.raw_audio_path)
    base_minus, base_rate = read_audio(args.base_minus_path)
    base_minus = resample_audio(base_minus, base_rate, raw_rate)
    arrays = [raw, base_minus]
    vocal_audio = None
    if args.vocal_guide_path is not None:
        vocal_audio, vocal_rate = read_audio(args.vocal_guide_path)
        arrays.append(resample_audio(vocal_audio, vocal_rate, raw_rate))
    aligned = align_channels_and_length(arrays)
    raw = aligned[0]
    base_minus = aligned[1]
    vocal_audio = aligned[2] if len(aligned) > 2 else None
    progress(
        "input_normalized",
        sample_rate=raw_rate,
        frames=int(base_minus.shape[0]),
        channels=int(base_minus.shape[1]),
        vocal_guide_used=vocal_audio is not None,
    )

    repaired, repair_metrics = apply_air_detail_repair(
        source_audio=raw,
        base_minus=base_minus,
        vocal_audio=vocal_audio,
        sample_rate=raw_rate,
        windows=selected_windows,
        margin_sec=float(args.window_margin_sec),
        smoothing_sec=float(args.gate_smoothing_sec),
        presence_delta_strength=float(args.presence_delta_strength),
        air_delta_strength=float(args.air_delta_strength),
        body_delta_strength=float(args.body_delta_strength),
        base_gain_strength=float(args.base_gain_strength),
        low_rms_target_ratio=float(args.low_rms_target_ratio),
        max_base_gain=float(args.max_base_gain),
        vocal_suppression_strength=float(args.vocal_suppression_strength),
        max_vocal_projection_gain=float(args.max_vocal_projection_gain),
        presence_low_hz=float(args.presence_low_hz),
        presence_high_hz=float(args.presence_high_hz),
        air_low_hz=float(args.air_low_hz),
        air_high_hz=float(args.air_high_hz),
        body_low_hz=float(args.body_low_hz),
        body_high_hz=float(args.body_high_hz),
        vocal_suppression_mode=str(args.vocal_suppression_mode),
    )
    sf.write(str(output_path), repaired, raw_rate, subtype="PCM_24")
    progress("output_written", output_audio_path=str(output_path), repair_metrics=repair_metrics)

    manifest = build_manifest(
        track_id=str(args.track_id or Path(args.raw_audio_path).stem),
        candidate_id=f"{str(args.candidate_id or Path(args.base_minus_path).stem)}__judge_guided_air_detail_repair",
        raw_audio_path=args.raw_audio_path,
        base_minus_path=args.base_minus_path,
        vocal_guide_path=args.vocal_guide_path,
        judge_jsonl_path=args.self_judge_jsonl,
        output_path=output_path,
        candidate_row_jsonl=candidate_row_jsonl,
        sample_rate=raw_rate,
        frames=int(repaired.shape[0]),
        selected_windows=selected_windows,
        repair_metrics=repair_metrics,
        runtime_environment=str(getattr(args, "runtime_environment", "docker")),
        native_runtime_allowed=bool(getattr(args, "native_runtime_allowed", False)),
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["short_output_names"] = short_output_names
    if short_output_names:
        manifest["artifact_name_policy"] = {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_filename": output_path.name,
            "manifest_filename": manifest_path.name,
            "rows_filename": candidate_row_jsonl.name,
            "raw_track_or_candidate_ids_in_path": False,
            "provenance_in_manifest": True,
        }
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
    args = parse_args(argv)
    runtime = cli_runtime_contract(allow_native_runtime=bool(args.allow_native_runtime))
    require_docker_runtime(
        tool_name="apply_judge_guided_minus_air_detail_repair.py",
        entrypoint_path="/app/tools/apply_judge_guided_minus_air_detail_repair.py",
        allow_native_macos_diagnostic=bool(args.allow_native_runtime),
    )
    args.runtime_environment = str(runtime["runtime_environment"])
    args.native_runtime_allowed = bool(runtime["native_runtime_allowed"])
    manifest = build_judge_guided_minus_air_detail_repair(args)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest.get("manifest_path") or Path(args.output_dir) / "judge_guided_minus_air_detail_repair_manifest.json"),
                "candidate_audio_path": manifest["artifacts"]["minus_air_detail_repair_flac"],
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
