from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from apply_flexible_minus_repair import (  # noqa: E402
    audio_metrics,
    match_length,
    read_audio,
    repair_metric_summary,
    resample_to_sample_rate,
    sha256_file,
    write_flac,
)

EPSILON = 1e-12
START_TIME = time.monotonic()

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


def progress(phase: str, **fields: Any) -> None:
    payload = {"phase": phase, "elapsed_sec": round(time.monotonic() - START_TIME, 3), **fields}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


@dataclass(frozen=True)
class VocalGuidedMinusTuning:
    vocal_subtract_gain: float
    base_blend_active: float
    base_blend_inactive: float
    activity_floor_ratio: float
    activity_smoothing_sec: float
    max_peak: float


def _ensure_channel_count(audio: np.ndarray, channel_count: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.shape[1] == channel_count:
        return audio.astype(np.float32, copy=False)
    if audio.shape[1] == 1 and channel_count > 1:
        return np.repeat(audio, channel_count, axis=1).astype(np.float32, copy=False)
    if audio.shape[1] > channel_count:
        return audio[:, :channel_count].astype(np.float32, copy=False)
    repeats = int(math.ceil(channel_count / float(audio.shape[1])))
    return np.tile(audio, (1, repeats))[:, :channel_count].astype(np.float32, copy=False)


def _match_to_reference(
    audio: np.ndarray,
    sample_rate: int,
    *,
    reference_audio: np.ndarray,
    reference_sample_rate: int,
) -> np.ndarray:
    matched = resample_to_sample_rate(audio, sample_rate, reference_sample_rate)
    return _ensure_channel_count(matched, int(reference_audio.shape[1])).astype(np.float32, copy=False)


def _frame_rms(mono: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    if mono.size == 0:
        return np.zeros(0, dtype=np.float32)
    frame_count = max(1, int(math.ceil(max(1, mono.shape[0] - frame_length) / float(hop_length))) + 1)
    values = np.zeros(frame_count, dtype=np.float32)
    for frame_index in range(frame_count):
        start = frame_index * hop_length
        end = min(mono.shape[0], start + frame_length)
        if end <= start:
            break
        frame = mono[start:end]
        values[frame_index] = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)) + EPSILON))
    return values


def build_vocal_activity_envelope(
    vocals: np.ndarray,
    sample_rate: int,
    *,
    floor_ratio: float,
    smoothing_sec: float,
) -> np.ndarray:
    mono = vocals.mean(axis=1).astype(np.float32, copy=False)
    if mono.size == 0:
        return np.zeros((0, vocals.shape[1]), dtype=np.float32)

    frame_length = min(max(512, int(round(0.046 * sample_rate))), max(512, mono.shape[0]))
    hop_length = max(128, frame_length // 4)
    frame_energy = _frame_rms(mono, frame_length=frame_length, hop_length=hop_length)
    reference = float(np.percentile(frame_energy, 95)) if frame_energy.size else 0.0
    if reference <= EPSILON:
        activity = np.zeros_like(frame_energy, dtype=np.float32)
    else:
        floor = max(reference * float(floor_ratio), EPSILON)
        activity = np.clip((frame_energy - floor) / max(reference - floor, EPSILON), 0.0, 1.0)
        activity = np.square(activity).astype(np.float32)

    smoothing_frames = max(1, int(round(float(smoothing_sec) * float(sample_rate) / float(hop_length))))
    if smoothing_frames > 1 and activity.size:
        kernel = np.ones(smoothing_frames, dtype=np.float32) / float(smoothing_frames)
        activity = np.convolve(activity, kernel, mode="same").astype(np.float32)

    frame_positions = np.arange(activity.size, dtype=np.float32) * float(hop_length)
    sample_positions = np.arange(mono.shape[0], dtype=np.float32)
    sample_activity = np.interp(
        sample_positions,
        frame_positions,
        activity,
        left=float(activity[0]) if activity.size else 0.0,
        right=float(activity[-1]) if activity.size else 0.0,
    ).astype(np.float32)
    return np.repeat(sample_activity[:, np.newaxis], vocals.shape[1], axis=1).astype(np.float32)


def _scale_to_peak(audio: np.ndarray, max_peak: float) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= float(max_peak) or peak <= EPSILON:
        return audio.astype(np.float32, copy=False), 1.0
    scale = float(max_peak) / peak
    return (audio * scale).astype(np.float32), scale


def build_vocal_guided_minus_audio(
    *,
    raw_audio: np.ndarray,
    selected_vocals: np.ndarray,
    base_minus: np.ndarray | None,
    sample_rate: int,
    tuning: VocalGuidedMinusTuning,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_minus = (raw_audio - selected_vocals * float(tuning.vocal_subtract_gain)).astype(np.float32)
    vocal_activity = build_vocal_activity_envelope(
        selected_vocals,
        sample_rate,
        floor_ratio=float(tuning.activity_floor_ratio),
        smoothing_sec=float(tuning.activity_smoothing_sec),
    )

    if base_minus is None:
        guided_minus = raw_minus
        blend_envelope = np.ones_like(raw_minus, dtype=np.float32)
        strategy = "raw_minus_from_selected_vocal"
    else:
        active_blend = float(tuning.base_blend_active)
        inactive_blend = float(tuning.base_blend_inactive)
        blend_envelope = inactive_blend + (active_blend - inactive_blend) * vocal_activity
        blend_envelope = np.clip(blend_envelope, 0.0, 1.0).astype(np.float32)
        guided_minus = (base_minus * (1.0 - blend_envelope) + raw_minus * blend_envelope).astype(np.float32)
        strategy = "base_minus_blended_with_raw_minus_from_selected_vocal"

    guided_minus, peak_scale = _scale_to_peak(guided_minus, float(tuning.max_peak))
    raw_minus, raw_minus_peak_scale = _scale_to_peak(raw_minus, float(tuning.max_peak))
    debug = {
        "strategy": strategy,
        "vocal_activity_mean": round(float(np.mean(vocal_activity)) if vocal_activity.size else 0.0, 6),
        "vocal_activity_max": round(float(np.max(vocal_activity)) if vocal_activity.size else 0.0, 6),
        "blend_mean": round(float(np.mean(blend_envelope)) if blend_envelope.size else 0.0, 6),
        "blend_max": round(float(np.max(blend_envelope)) if blend_envelope.size else 0.0, 6),
        "guided_minus_peak_scale": round(float(peak_scale), 6),
        "raw_minus_peak_scale": round(float(raw_minus_peak_scale), 6),
    }
    return guided_minus, raw_minus, debug


def _artifact(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    return write_flac(path, audio, sample_rate)


def _copy_to_listen(listen_dir: Path, source_path: Path, target_name: str, mapping: dict[str, str]) -> None:
    target_path = listen_dir / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    mapping[target_name] = str(source_path)


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    raw_audio_path = Path(args.raw_audio_path)
    selected_vocals_path = Path(args.selected_vocals_path)
    base_minus_path = Path(args.base_minus_path) if args.base_minus_path else None
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None
    manifest_path = output_dir / "vocal_guided_minus_manifest.json"

    progress(
        "start",
        route="vocal_guided_minus_builder",
        acceleration_mode="external_only_cpu_dsp",
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="route uses existing vocal/minus candidates and CPU/DSP subtraction only",
        output_manifest_path=str(manifest_path),
    )

    if not raw_audio_path.exists():
        raise FileNotFoundError(raw_audio_path)
    if not selected_vocals_path.exists():
        raise FileNotFoundError(selected_vocals_path)
    if base_minus_path is not None and not base_minus_path.exists():
        raise FileNotFoundError(base_minus_path)

    raw_audio, raw_sample_rate = read_audio(raw_audio_path)
    selected_vocals, vocals_sample_rate = read_audio(selected_vocals_path)
    selected_vocals = _match_to_reference(
        selected_vocals,
        vocals_sample_rate,
        reference_audio=raw_audio,
        reference_sample_rate=raw_sample_rate,
    )
    raw_audio, selected_vocals = match_length(raw_audio, selected_vocals)

    base_minus: np.ndarray | None = None
    base_sample_rate: int | None = None
    if base_minus_path is not None:
        base_audio, base_sample_rate = read_audio(base_minus_path)
        base_minus = _match_to_reference(
            base_audio,
            base_sample_rate,
            reference_audio=raw_audio,
            reference_sample_rate=raw_sample_rate,
        )
        raw_audio, base_minus = match_length(raw_audio, base_minus)
        selected_vocals = selected_vocals[: raw_audio.shape[0]]

    progress(
        "input_normalized",
        sample_rate=int(raw_sample_rate),
        frames=int(raw_audio.shape[0]),
        channels=int(raw_audio.shape[1]),
        has_base_minus=base_minus is not None,
    )

    tuning = VocalGuidedMinusTuning(
        vocal_subtract_gain=float(args.vocal_subtract_gain),
        base_blend_active=float(args.base_blend_active),
        base_blend_inactive=float(args.base_blend_inactive),
        activity_floor_ratio=float(args.activity_floor_ratio),
        activity_smoothing_sec=float(args.activity_smoothing_sec),
        max_peak=float(args.max_peak),
    )
    guided_minus, raw_minus, debug = build_vocal_guided_minus_audio(
        raw_audio=raw_audio,
        selected_vocals=selected_vocals,
        base_minus=base_minus,
        sample_rate=raw_sample_rate,
        tuning=tuning,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {
        "raw_ref_flac": _artifact(output_dir / "00_raw_ref.flac", raw_audio, raw_sample_rate),
        "selected_vocals_flac": _artifact(output_dir / "01_selected_vocals.flac", selected_vocals, raw_sample_rate),
        "raw_minus_from_selected_vocal_flac": _artifact(
            output_dir / "02_raw_minus_from_selected_vocal.flac",
            raw_minus,
            raw_sample_rate,
        ),
        "vocal_guided_minus_flac": _artifact(output_dir / "03_vocal_guided_minus.flac", guided_minus, raw_sample_rate),
    }
    if base_minus is not None:
        artifacts["base_minus_flac"] = _artifact(output_dir / "04_base_minus_reference.flac", base_minus, raw_sample_rate)
    progress(
        "output_written",
        output_dir=str(output_dir),
        vocal_guided_minus_path=str(artifacts["vocal_guided_minus_flac"]["path"]),
        raw_minus_path=str(artifacts["raw_minus_from_selected_vocal_flac"]["path"]),
    )

    raw_minus_before_limiter = (
        raw_audio - selected_vocals * float(tuning.vocal_subtract_gain)
    ).astype(np.float32)
    raw_subtraction_error = raw_audio - (raw_minus_before_limiter + selected_vocals)
    post_limiter_reconstruction_error = raw_audio - (raw_minus + selected_vocals)
    post_limiter_delta = raw_minus - raw_minus_before_limiter
    progress(
        "metrics_start",
        metric_count=7 if base_minus is None else 8,
        frames=int(raw_audio.shape[0]),
        sample_rate=int(raw_sample_rate),
    )
    raw_ref_metrics = audio_metrics(raw_audio, raw_sample_rate)
    progress("metric_computed", metric="raw_ref")
    selected_vocals_metrics = audio_metrics(selected_vocals, raw_sample_rate)
    progress("metric_computed", metric="selected_vocals")
    raw_minus_metrics = audio_metrics(raw_minus, raw_sample_rate)
    progress("metric_computed", metric="raw_minus_from_selected_vocal")
    vocal_guided_minus_metrics = audio_metrics(guided_minus, raw_sample_rate)
    progress("metric_computed", metric="vocal_guided_minus")
    reconstruction_error_metrics = audio_metrics(raw_subtraction_error, raw_sample_rate)
    progress("metric_computed", metric="reconstruction_error")
    post_limiter_reconstruction_error_metrics = audio_metrics(post_limiter_reconstruction_error, raw_sample_rate)
    progress("metric_computed", metric="post_limiter_reconstruction_error")
    post_limiter_delta_metrics = audio_metrics(post_limiter_delta, raw_sample_rate)
    progress("metric_computed", metric="post_limiter_delta")
    base_to_guided_minus_metrics = None
    if base_minus is not None:
        base_to_guided_minus_metrics = repair_metric_summary(base_minus, guided_minus, raw_sample_rate)
        progress("metric_computed", metric="base_to_guided_minus")
    progress("metrics_end")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "vocal_guided_minus_builder",
        "stem_type": "minus",
        "candidate_only": True,
        "delivery_status": "candidate_only",
        "delivery_allowed": False,
        "presentation_status": "blocked_until_judged",
        "release_status": "not_release_ready",
        "listenable": False,
        "deliverable": False,
        "can_mark_release_ready": False,
        "requires_post_judge": True,
        "is_clean_minus_baseline": False,
        "baseline_allowed": False,
        "route_allowed_as": "audited_candidate_only",
        "no_per_track_hardcoded_timestamps": True,
        "uses_existing_vocal_candidate": True,
        "raw_audio_path": str(raw_audio_path),
        "selected_vocals_path": str(selected_vocals_path),
        "base_minus_path": None if base_minus_path is None else str(base_minus_path),
        "output_dir": str(output_dir),
        "listen_dir": None if listen_dir is None else str(listen_dir),
        "tuning": {
            "vocal_subtract_gain": tuning.vocal_subtract_gain,
            "base_blend_active": tuning.base_blend_active,
            "base_blend_inactive": tuning.base_blend_inactive,
            "activity_floor_ratio": tuning.activity_floor_ratio,
            "activity_smoothing_sec": tuning.activity_smoothing_sec,
            "max_peak": tuning.max_peak,
        },
        "audio": {
            "raw_sample_rate": int(raw_sample_rate),
            "selected_vocals_original_sample_rate": int(vocals_sample_rate),
            "base_minus_original_sample_rate": None if base_sample_rate is None else int(base_sample_rate),
            "frames": int(guided_minus.shape[0]),
            "duration_sec": round(float(guided_minus.shape[0]) / float(raw_sample_rate), 6),
            "channels": int(guided_minus.shape[1]),
        },
        "metrics": {
            "raw_ref": raw_ref_metrics,
            "selected_vocals": selected_vocals_metrics,
            "raw_minus_from_selected_vocal": raw_minus_metrics,
            "vocal_guided_minus": vocal_guided_minus_metrics,
            "reconstruction_error_rms_dbfs": reconstruction_error_metrics["rms_dbfs"],
            "post_limiter_reconstruction_error_rms_dbfs": post_limiter_reconstruction_error_metrics["rms_dbfs"],
            "post_limiter_delta_rms_dbfs": post_limiter_delta_metrics["rms_dbfs"],
            "base_to_guided_minus": base_to_guided_minus_metrics,
        },
        "debug": debug,
        "artifacts": artifacts,
        "safety": dict(SAFETY_FLAGS),
        "forbidden_actions": [
            "do_not_use_as_clean_minus_baseline",
            "do_not_export_as_delivery_without_post_judge_acceptance",
            "do_not_mark_release_ready_from_builder_manifest",
            "do_not_relax_gates",
            "do_not_require_reference_stems",
            "do_not_start_training",
        ],
        "artifact_contract": {
            "01_selected_vocals.flac": "The selected upstream vocal stem, resampled/matched to raw mix.",
            "02_raw_minus_from_selected_vocal.flac": "Raw mix minus selected vocal. Highest instrumental preservation, highest risk if vocal estimate misses words.",
            "03_vocal_guided_minus.flac": "Final selected minus. If base minus is provided, this is a bounded blend between base minus and raw-minus-from-vocal.",
            "04_base_minus_reference.flac": "Optional matched base minus reference copied only when --base-minus-path is provided.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        listen_map: dict[str, str] = {}
        _copy_to_listen(listen_dir, Path(str(artifacts["raw_ref_flac"]["path"])), "00_raw_ref.flac", listen_map)
        _copy_to_listen(listen_dir, Path(str(artifacts["selected_vocals_flac"]["path"])), "01_selected_vocals.flac", listen_map)
        _copy_to_listen(
            listen_dir,
            Path(str(artifacts["vocal_guided_minus_flac"]["path"])),
            str(args.listen_minus_label),
            listen_map,
        )
        _copy_to_listen(
            listen_dir,
            Path(str(artifacts["raw_minus_from_selected_vocal_flac"]["path"])),
            "03_raw_minus_from_selected_vocal.flac",
            listen_map,
        )
        if "base_minus_flac" in artifacts:
            _copy_to_listen(listen_dir, Path(str(artifacts["base_minus_flac"]["path"])), "04_base_minus_reference.flac", listen_map)
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n"
            "00 = raw reference.\n"
            "01 = selected vocals used as the subtraction guide.\n"
            "02 = final vocal-guided minus.\n"
            "03 = raw mix minus selected vocal, audit candidate.\n"
            "04 = matched base minus reference when available.\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(listen_map, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest["listen_dir"] = str(listen_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    progress(
        "end",
        output_manifest_path=str(manifest_path),
        delivery_status=manifest["delivery_status"],
        delivery_allowed=manifest["delivery_allowed"],
        requires_post_judge=manifest["requires_post_judge"],
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "selected_vocals_flac": str(artifacts["selected_vocals_flac"]["path"]),
                "vocal_guided_minus_flac": str(artifacts["vocal_guided_minus_flac"]["path"]),
                "listen_dir": None if listen_dir is None else str(listen_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a full-track vocal-guided minus from raw audio and a selected vocal stem.",
    )
    parser.add_argument("raw_audio_path")
    parser.add_argument("selected_vocals_path")
    parser.add_argument("output_dir")
    parser.add_argument("--base-minus-path")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-minus-label", default="02_vocal_guided_minus.flac")
    parser.add_argument("--vocal-subtract-gain", type=float, default=1.0)
    parser.add_argument("--base-blend-active", type=float, default=0.72)
    parser.add_argument("--base-blend-inactive", type=float, default=0.18)
    parser.add_argument("--activity-floor-ratio", type=float, default=0.035)
    parser.add_argument("--activity-smoothing-sec", type=float, default=0.12)
    parser.add_argument("--max-peak", type=float, default=0.999)
    return parser


def main() -> int:
    require_docker_runtime(
        tool_name="build_vocal_guided_minus.py",
        entrypoint_path="/app/tools/build_vocal_guided_minus.py",
    )
    run_build(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
