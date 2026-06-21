#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime

import numpy as np
import soundfile as sf
from scipy import signal


START_TIME = time.monotonic()
EPS = 1.0e-12


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
MANIFEST_NAME = "pair_reconstruction_repair_manifest.json"
CANDIDATE_ROW_NAME = "candidate_pair.jsonl"
MINUS_AUDIO_NAME = "pair_reconstruction_minus.flac"
SHORT_MANIFEST_NAME = "manifest.json"
SHORT_CANDIDATE_ROW_NAME = "rows.jsonl"
SHORT_MINUS_AUDIO_NAME = "m001.flac"


def progress(phase: str, **fields: Any) -> None:
    payload = {"phase": phase, "elapsed_sec": round(time.monotonic() - START_TIME, 3), **fields}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(path)
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


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def peak_limit(audio: np.ndarray, *, max_peak: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 < max_peak <= 1.0:
        raise ValueError("max_peak must be in (0, 1]")
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_before <= max_peak + EPS:
        return audio.astype(np.float32), {
            "peak_before_limit": round(peak_before, 9),
            "peak_after_limit": round(peak_before, 9),
            "peak_limited": False,
            "peak_scale": 1.0,
            "max_peak": round(float(max_peak), 9),
        }
    scale = float(max_peak) / (peak_before + EPS)
    limited = audio * scale
    return limited.astype(np.float32), {
        "peak_before_limit": round(peak_before, 9),
        "peak_after_limit": round(float(np.max(np.abs(limited))) if limited.size else 0.0, 9),
        "peak_limited": True,
        "peak_scale": round(scale, 9),
        "max_peak": round(float(max_peak), 9),
    }


def build_pair_reconstruction_minus(
    *,
    raw_audio: np.ndarray,
    selected_vocal: np.ndarray,
    residual_strength: float,
    max_peak: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if raw_audio.shape != selected_vocal.shape:
        raise ValueError("raw_audio and selected_vocal must have the same shape")
    if not 0.0 <= residual_strength <= 1.0:
        raise ValueError("residual_strength must be in [0, 1]")

    residual_minus = raw_audio - selected_vocal
    candidate_before_limiter = residual_minus * float(residual_strength)
    reconstruction_error = raw_audio - (candidate_before_limiter + selected_vocal)
    candidate, limiting = peak_limit(candidate_before_limiter, max_peak=max_peak)
    post_limiter_reconstruction_error = raw_audio - (candidate.astype(np.float64) + selected_vocal)
    metrics = {
        "residual_strength": round(float(residual_strength), 9),
        "raw_residual_rms": round(rms(residual_minus), 12),
        "candidate_rms": round(rms(candidate), 12),
        "reconstruction_error_rms": round(rms(reconstruction_error), 12),
        "post_limiter_reconstruction_error_rms": round(rms(post_limiter_reconstruction_error), 12),
        **limiting,
    }
    return candidate.astype(np.float32), metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(
    *,
    track_id: str,
    candidate_id: str,
    raw_audio_path: Path,
    selected_vocal_path: Path,
    output_path: Path,
    candidate_row_jsonl: Path,
    sample_rate: int,
    frames: int,
    metrics: dict[str, Any],
    base_minus_path: Path | None = None,
    windowed_pair_judge_jsonl: Path | None = None,
) -> dict[str, Any]:
    candidate_row = {
        "track_id": str(track_id),
        "candidate_id": str(candidate_id),
        "mix_path": str(raw_audio_path),
        "source_audio_path": str(raw_audio_path),
        "vocal_path": str(selected_vocal_path),
        "selected_vocal_path": str(selected_vocal_path),
        "instrumental_path": str(output_path),
        "minus_path": str(output_path),
        "selected_minus_path": str(output_path),
        "base_minus_path": "" if base_minus_path is None else str(base_minus_path),
        "windowed_pair_judge_jsonl": "" if windowed_pair_judge_jsonl is None else str(windowed_pair_judge_jsonl),
        "candidate_family": "no_reference_pair_reconstruction_repair",
        "candidate_origin": "source_mix_minus_accepted_vocal_pair_reconstruction",
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "is_complete_pair": True,
        "forbidden_actions": [
            "do_not_export_as_delivery_without_post_judge_acceptance",
            "do_not_mark_release_ready_from_repair_manifest",
            "do_not_relax_gates",
            "do_not_require_reference_stems",
        ],
    }
    return {
        "schema_version": 1,
        "mode": "no_reference_pair_reconstruction_repair",
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
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "no_per_track_hardcoded_timestamps": True,
        "uses_existing_vocal_candidate": True,
        "uses_reference_stem": False,
        "raw_audio_path": str(raw_audio_path),
        "selected_vocal_path": str(selected_vocal_path),
        "base_minus_path": None if base_minus_path is None else str(base_minus_path),
        "windowed_pair_judge_jsonl": None if windowed_pair_judge_jsonl is None else str(windowed_pair_judge_jsonl),
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
        "artifacts": {
            "pair_reconstruction_minus_flac": str(output_path),
            "selected_vocal_audio": str(selected_vocal_path),
            "raw_ref_audio": str(raw_audio_path),
            "base_minus_audio": None if base_minus_path is None else str(base_minus_path),
        },
        "audio": {
            "sample_rate": int(sample_rate),
            "frames": int(frames),
            "duration_sec": round(float(frames) / float(sample_rate), 6) if sample_rate else 0.0,
        },
        "repair": {
            "strategy": "source_mix_minus_selected_vocal_residual_complement",
            "metrics": dict(metrics),
        },
        "safety": dict(SAFETY_FLAGS),
        "forbidden_actions": [
            "do_not_export_as_delivery_without_post_judge_acceptance",
            "do_not_mark_release_ready_from_repair_manifest",
            "do_not_relax_gates",
            "do_not_require_reference_stems",
            "do_not_start_training",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a no-reference pair reconstruction repair minus candidate.")
    parser.add_argument("raw_audio_path", type=Path)
    parser.add_argument("selected_vocal_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-minus-path", type=Path)
    parser.add_argument("--windowed-pair-judge-jsonl", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--track-id")
    parser.add_argument("--residual-strength", type=float, default=1.0)
    parser.add_argument("--max-peak", type=float, default=0.999)
    parser.add_argument("--short-output-names", action="store_true")
    return parser.parse_args(argv)


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    raw_audio_path = Path(args.raw_audio_path)
    selected_vocal_path = Path(args.selected_vocal_path)
    output_dir = Path(args.output_dir)
    base_minus_path = Path(args.base_minus_path) if args.base_minus_path else None
    windowed_pair_judge_jsonl = (
        Path(args.windowed_pair_judge_jsonl) if args.windowed_pair_judge_jsonl else None
    )
    short_output_names = bool(getattr(args, "short_output_names", False))
    manifest_path = output_dir / (SHORT_MANIFEST_NAME if short_output_names else MANIFEST_NAME)
    candidate_row_jsonl = output_dir / (SHORT_CANDIDATE_ROW_NAME if short_output_names else CANDIDATE_ROW_NAME)
    output_path = output_dir / (SHORT_MINUS_AUDIO_NAME if short_output_names else MINUS_AUDIO_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(
        "start",
        route="no_reference_pair_reconstruction_repair",
        acceleration_mode="external_only_cpu_dsp_scoring",
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="route uses existing accepted vocal candidate and deterministic residual complement",
        output_manifest_path=str(manifest_path),
    )

    raw_audio, raw_rate = read_audio(raw_audio_path)
    selected_vocal, vocal_rate = read_audio(selected_vocal_path)
    selected_vocal = resample_audio(selected_vocal, vocal_rate, raw_rate)
    arrays = [raw_audio, selected_vocal]
    base_metrics: dict[str, Any] = {}
    if base_minus_path is not None:
        base_minus, base_rate = read_audio(base_minus_path)
        base_minus = resample_audio(base_minus, base_rate, raw_rate)
        arrays.append(base_minus)
        base_metrics["base_minus_original_sample_rate"] = int(base_rate)
    aligned = align_channels_and_length(arrays)
    raw_audio, selected_vocal = aligned[0], aligned[1]
    base_minus = aligned[2] if len(aligned) > 2 else None
    progress(
        "input_normalized",
        sample_rate=int(raw_rate),
        frames=int(raw_audio.shape[0]),
        channels=int(raw_audio.shape[1]),
        has_base_minus=base_minus is not None,
        has_windowed_pair_judge=windowed_pair_judge_jsonl is not None,
    )

    candidate, metrics = build_pair_reconstruction_minus(
        raw_audio=raw_audio,
        selected_vocal=selected_vocal,
        residual_strength=float(args.residual_strength),
        max_peak=float(args.max_peak),
    )
    if base_minus is not None:
        metrics["base_to_pair_reconstruction_delta_rms"] = round(rms(base_minus - candidate), 12)
        metrics.update(base_metrics)
    sf.write(str(output_path), candidate, raw_rate, subtype="PCM_24")
    progress("output_written", output_audio_path=str(output_path), metrics=metrics)

    track_id = str(args.track_id or raw_audio_path.stem)
    source_candidate_id = str(args.candidate_id or selected_vocal_path.stem)
    candidate_id = f"{source_candidate_id}__pair_reconstruction_repair"
    manifest = build_manifest(
        track_id=track_id,
        candidate_id=candidate_id,
        raw_audio_path=raw_audio_path,
        selected_vocal_path=selected_vocal_path,
        output_path=output_path,
        candidate_row_jsonl=candidate_row_jsonl,
        sample_rate=raw_rate,
        frames=int(candidate.shape[0]),
        metrics=metrics,
        base_minus_path=base_minus_path,
        windowed_pair_judge_jsonl=windowed_pair_judge_jsonl,
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["short_output_names"] = short_output_names
    manifest["artifact_name_policy"] = {
        "path_segment_limit": 32,
        "raw_ids_in_paths": False,
        "provenance_in_manifest": True,
    }
    write_jsonl(candidate_row_jsonl, [manifest["candidate_row"]])
    write_json(manifest_path, manifest)
    progress(
        "end",
        output_manifest_path=str(manifest_path),
        candidate_row_jsonl=str(candidate_row_jsonl),
        delivery_status=manifest["delivery_status"],
        delivery_allowed=manifest["delivery_allowed"],
        requires_post_judge=manifest["requires_post_judge"],
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="apply_no_reference_pair_reconstruction_repair.py",
        entrypoint_path="/app/tools/apply_no_reference_pair_reconstruction_repair.py",
    )
    args = parse_args(argv)
    manifest = run_build(args)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest["manifest_path"]),
                "candidate_audio_path": manifest["artifacts"]["pair_reconstruction_minus_flac"],
                "candidate_row_jsonl": manifest["candidate_row_jsonl"],
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
