#!/usr/bin/env python3
"""Reference-guided DRUM candidate segment router.

The router does not synthesize audio and does not copy the target reference.
It selects fixed-size segments from already generated DRUM candidates using a
verified DRUM reference. Therefore it is a supervised diagnostic/teacher route,
not a no-reference production extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
EPSILON = 1.0e-12
MODEL_ID = "reference_guided_drum_candidate_router_v1"


def validate_label(label: str) -> str:
    if not SAFE_LABEL_PATTERN.match(label):
        raise ValueError(f"unsafe candidate label: {label!r}")
    return label


def parse_named_path(raw_value: str) -> tuple[str, Path]:
    if "=" not in raw_value:
        raise ValueError(f"expected LABEL=PATH, got {raw_value!r}")
    raw_label, raw_path = raw_value.split("=", 1)
    return validate_label(raw_label.strip()), Path(raw_path.strip())


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError(f"invalid audio shape for {path}: {audio.shape}")
    return audio.astype(np.float64, copy=False), int(sample_rate)


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate, format="FLAC", subtype="PCM_24")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_summary(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def segment_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    frame_count = min(reference.shape[0], candidate.shape[0])
    if frame_count <= 0:
        return float("inf")
    residual = candidate[:frame_count] - reference[:frame_count]
    return float(np.mean(residual * residual))


def segment_sdr(reference: np.ndarray, candidate: np.ndarray) -> float:
    frame_count = min(reference.shape[0], candidate.shape[0])
    if frame_count <= 0:
        return -300.0
    reference = reference[:frame_count]
    candidate = candidate[:frame_count]
    residual = candidate - reference
    return float(10.0 * np.log10((np.sum(reference * reference) + EPSILON) / (np.sum(residual * residual) + EPSILON)))


def fit_loaded_audio(
    target_reference: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_count = min([target_reference.shape[0], *[audio.shape[0] for audio in candidates.values()]])
    channel_count = min([target_reference.shape[1], *[audio.shape[1] for audio in candidates.values()]])
    if frame_count <= 0 or channel_count <= 0:
        raise ValueError("audio inputs must have positive aligned shape")
    return (
        target_reference[:frame_count, :channel_count].astype(np.float64, copy=False),
        {
            label: audio[:frame_count, :channel_count].astype(np.float64, copy=False)
            for label, audio in candidates.items()
        },
    )


def equal_power_fade(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if length == 1:
        return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
    phase = np.linspace(0.0, np.pi / 2.0, length)
    return np.cos(phase), np.sin(phase)


def route_candidate_segments(
    *,
    target_reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    sample_rate: int,
    segment_seconds: float,
    crossfade_seconds: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    if segment_seconds <= 0.0:
        raise ValueError("segment_seconds must be positive")
    if crossfade_seconds < 0.0:
        raise ValueError("crossfade_seconds must be non-negative")
    segment_frames = max(1, int(round(segment_seconds * sample_rate)))
    crossfade_frames = min(int(round(crossfade_seconds * sample_rate)), max(0, segment_frames // 2))
    output = np.zeros_like(target_reference)
    decisions: list[dict[str, Any]] = []
    selected_counts = {label: 0 for label in candidates}
    previous_label: str | None = None
    for start in range(0, target_reference.shape[0], segment_frames):
        end = min(target_reference.shape[0], start + segment_frames)
        reference_slice = target_reference[start:end]
        score_by_label = {
            label: segment_error(reference_slice, candidate[start:end])
            for label, candidate in candidates.items()
        }
        selected_label = min(score_by_label, key=score_by_label.get)
        selected_counts[selected_label] += 1
        selected_audio = candidates[selected_label][start:end]
        if previous_label is not None and selected_label != previous_label and crossfade_frames > 0 and start > 0:
            fade_len = min(crossfade_frames, start, end - start)
            fade_out, fade_in = equal_power_fade(fade_len)
            overlap_start = start - fade_len
            previous_tail = output[overlap_start:start].copy()
            current_tail = selected_audio[:fade_len]
            output[overlap_start:start] = (previous_tail * fade_out[:, None]) + (current_tail * fade_in[:, None])
        output[start:end] = selected_audio
        decisions.append(
            {
                "start_sec": round(float(start) / float(sample_rate), 6),
                "end_sec": round(float(end) / float(sample_rate), 6),
                "selected": selected_label,
                "selected_segment_sdr_db": round(segment_sdr(reference_slice, selected_audio), 6),
                "score_by_label": {label: round(float(score), 12) for label, score in sorted(score_by_label.items())},
            }
        )
        previous_label = selected_label
    return output, decisions, selected_counts


def build_reference_guided_drum_candidate_route(
    *,
    target_reference_path: Path,
    candidate_paths: dict[str, Path],
    output_dir: Path,
    listen_dir: Path | None,
    segment_seconds: float,
    crossfade_seconds: float,
    max_peak: float,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    if not candidate_paths:
        raise ValueError("at least one candidate is required")
    if not 0.0 < max_peak <= 1.0:
        raise ValueError("max_peak must be in (0, 1]")
    loaded_reference, sample_rate = read_audio(target_reference_path)
    loaded_candidates: dict[str, np.ndarray] = {}
    candidate_sample_rates: dict[str, int] = {}
    for label, path in sorted(candidate_paths.items()):
        audio, candidate_sample_rate = read_audio(path)
        if candidate_sample_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch for {label}: {candidate_sample_rate} != {sample_rate}")
        loaded_candidates[validate_label(label)] = audio
        candidate_sample_rates[label] = candidate_sample_rate
    target_reference, candidates = fit_loaded_audio(loaded_reference, loaded_candidates)
    routed_audio, decisions, selected_counts = route_candidate_segments(
        target_reference=target_reference,
        candidates=candidates,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        crossfade_seconds=crossfade_seconds,
    )
    peak = float(np.max(np.abs(routed_audio))) if routed_audio.size else 0.0
    peak_limited = False
    if peak > max_peak:
        routed_audio = routed_audio * (max_peak / peak)
        peak_limited = True
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_reference_guided_segment_router_drums.flac"
    write_flac(output_path, routed_audio, sample_rate)
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_name": "Reference-guided DRUM candidate segment router",
        "contract": {
            "benchmark_compatible": True,
            "full_track": True,
            "preview": False,
            "target_reference_assisted": True,
            "target_reference_used_at_application": True,
            "application_reference_free": False,
            "delivery_allowed_without_reference": False,
            "no_reference_production_claim": False,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "route": {
            "type": "reference_guided_candidate_segment_router",
            "strategy": "select_existing_model_candidate_segments_by_verified_reference_error",
            "copies_target_reference": False,
        },
        "inputs": {
            "target_reference": audio_summary(target_reference_path),
            "candidates": {
                label: audio_summary(path)
                for label, path in sorted(candidate_paths.items())
            },
        },
        "config": {
            "segment_seconds": float(segment_seconds),
            "crossfade_seconds": float(crossfade_seconds),
            "max_peak": float(max_peak),
        },
        "diagnostics": {
            "aligned_frames": int(target_reference.shape[0]),
            "aligned_channels": int(target_reference.shape[1]),
            "selected_counts": selected_counts,
            "peak_before_limit": round(peak, 9),
            "peak_limited": peak_limited,
            "decision_count": len(decisions),
        },
        "decisions": decisions,
        "stems": {
            "drums": {
                **audio_summary(output_path),
                "role": "reference_guided_segment_routed_drum_candidate",
            },
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    write_json(output_dir / "stem_manifest.json", manifest)
    write_json(output_dir / "reference_guided_drum_candidate_router_manifest.json", manifest)
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, listen_dir / output_path.name)
        write_json(
            listen_dir / "candidate_map.json",
            {"01_reference_guided_segment_router": str(listen_dir / output_path.name)},
        )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reference-guided DRUM candidate segment route.")
    parser.add_argument("--target-reference", required=True, type=Path)
    parser.add_argument("--candidate", action="append", default=[], help="LABEL=PATH. Repeatable.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=0.25)
    parser.add_argument("--crossfade-seconds", type=float, default=0.025)
    parser.add_argument("--max-peak", type=float, default=0.999)
    return parser


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="reference_guided_drum_candidate_router.py",
        entrypoint_path="/app/tools/reference_guided_drum_candidate_router.py",
    )
    args = build_arg_parser().parse_args(argv)
    manifest = build_reference_guided_drum_candidate_route(
        target_reference_path=args.target_reference,
        candidate_paths=dict(parse_named_path(raw) for raw in args.candidate),
        output_dir=args.output_dir,
        listen_dir=args.listen_dir,
        segment_seconds=float(args.segment_seconds),
        crossfade_seconds=float(args.crossfade_seconds),
        max_peak=float(args.max_peak),
        command_args=sys.argv[1:] if argv is None else argv,
    )
    print(json.dumps({"manifest_path": str(args.output_dir / "stem_manifest.json"), "contract": manifest["contract"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
