from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


STARTED_AT = time.monotonic()
VARIANTS = (
    ("minus_mid_mono", 0.0),
    ("minus_side_attenuate_25", 0.25),
    ("minus_side_attenuate_50", 0.50),
)
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


def progress(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "elapsed_sec": round(time.monotonic() - STARTED_AT, 3),
                **fields,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio.astype(np.float32, copy=False), int(sample_rate)


def align_audio(audio_items: Sequence[np.ndarray]) -> list[np.ndarray]:
    frame_count = min(audio.shape[0] for audio in audio_items)
    channel_count = min(audio.shape[1] for audio in audio_items)
    return [audio[:frame_count, :channel_count].astype(np.float32, copy=False) for audio in audio_items]


def clip_peak(audio: np.ndarray, max_peak: float) -> tuple[np.ndarray, float, float]:
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_before > max_peak > 0:
        audio = audio * (max_peak / peak_before)
    peak_after = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio.astype(np.float32, copy=False), peak_before, peak_after


def cosine_abs(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 1.0
    return abs(float(np.dot(left, right) / denom))


def phase_consistency(audio: np.ndarray) -> float:
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 1.0
    return round(max(0.0, min(1.0, (cosine_abs(audio[:, 0], audio[:, 1]) + 0.15) / 1.15)), 6)


def variant_audio(minus: np.ndarray, side_gain: float) -> np.ndarray:
    mid = np.mean(minus[:, :2], axis=1, keepdims=True)
    side = (minus[:, :1] - minus[:, 1:2]) * 0.5
    if side_gain <= 0:
        return np.repeat(mid, 2, axis=1).astype(np.float32)
    return np.concatenate([mid + side * side_gain, mid - side * side_gain], axis=1).astype(np.float32)


def build_candidate_row(
    *,
    mix_path: Path,
    vocal_path: Path,
    minus_path: Path,
    track_id: str,
    candidate_id: str,
    minus_route_id: str,
    variant_name: str,
) -> dict[str, Any]:
    return {
        "schema": "no_reference_minus_phase_repair_candidate.v1",
        "track_id": track_id,
        "candidate_id": f"{candidate_id}__{variant_name}",
        "mix_path": str(mix_path),
        "vocal_path": str(vocal_path),
        "minus_path": str(minus_path),
        "instrumental_path": str(minus_path),
        "vocal_route_id": "source_pool_vocal_context_not_selected",
        "minus_route_id": f"{minus_route_id}__{variant_name}",
        "minus_phase_repair_variant": variant_name,
        "route_contract": "no_reference_minus_phase_repair_candidate",
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "phase_repair_candidate_not_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_publish_as_release": False,
        "can_mark_release_ready": False,
    }


def build_minus_phase_repair(
    *,
    mix_path: Path,
    vocal_path: Path,
    minus_path: Path,
    output_dir: Path,
    track_id: str,
    candidate_id: str,
    minus_route_id: str,
    max_peak: float = 0.98,
) -> dict[str, Any]:
    progress(
        "minus_phase_repair_start",
        acceleration_mode="external_only_cpu_dsp_audio_transform",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="minus_phase_repair_is_stereo_dsp_only",
        mix_path=str(mix_path),
        vocal_path=str(vocal_path),
        minus_path=str(minus_path),
        output_dir=str(output_dir),
    )
    mix_audio, mix_rate = read_audio(mix_path)
    vocal_audio, vocal_rate = read_audio(vocal_path)
    minus_audio, minus_rate = read_audio(minus_path)
    if len({mix_rate, vocal_rate, minus_rate}) != 1:
        raise ValueError(f"sample_rate_mismatch:{mix_rate},{vocal_rate},{minus_rate}")
    mix_audio, vocal_audio, minus_audio = align_audio([mix_audio, vocal_audio, minus_audio])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    source_phase = phase_consistency(minus_audio)
    for variant_name, side_gain in VARIANTS:
        repaired = variant_audio(minus_audio, side_gain)
        repaired, peak_before, peak_after = clip_peak(repaired, max_peak)
        target_path = output_dir / f"{variant_name}.flac"
        sf.write(str(target_path), repaired, mix_rate, subtype="PCM_24")
        target_phase = phase_consistency(repaired)
        rows.append(
            build_candidate_row(
                mix_path=mix_path,
                vocal_path=vocal_path,
                minus_path=target_path,
                track_id=track_id,
                candidate_id=candidate_id,
                minus_route_id=minus_route_id,
                variant_name=variant_name,
            )
        )
        variant = {
            "variant": variant_name,
            "side_gain": side_gain,
            "path": str(target_path),
            "phase_consistency_before": source_phase,
            "phase_consistency_after": target_phase,
            "peak_before": round(peak_before, 6),
            "peak_after": round(peak_after, 6),
        }
        variants.append(variant)
        progress("minus_phase_repair_variant_written", **variant)

    candidate_pair_jsonl = output_dir / "candidate_pair.jsonl"
    write_jsonl(candidate_pair_jsonl, rows)
    manifest = {
        "schema": "no_reference_minus_phase_repair_manifest.v1",
        "mode": "no_reference_minus_phase_repair",
        "track_id": track_id,
        "source_candidate_id": candidate_id,
        "source_minus_route_id": minus_route_id,
        "source_mix_path": str(mix_path),
        "source_vocal_path": str(vocal_path),
        "source_minus_path": str(minus_path),
        "candidate_pair_jsonl": str(candidate_pair_jsonl),
        "variant_count": len(rows),
        "variants": variants,
        "target_stem": "minus",
        "target_stem_artifact": {
            "target_stem": "minus",
            "path": variants[1]["path"] if len(variants) > 1 else (variants[0]["path"] if variants else ""),
            "source": "minus_phase_repair_variant_pool",
            "source_stems": ["minus"],
        },
        "acceleration_mode": "external_only_cpu_dsp_audio_transform",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "minus_phase_repair_is_stereo_dsp_only",
        "safety": dict(SAFETY_FLAGS),
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "phase_repair_candidate_not_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_publish_as_release": False,
        "can_mark_release_ready": False,
        "downstream_gates_required": [
            "no_reference_windowed_judge_minus_mode",
            "mode_specific_score_matrix",
            "artifact_status_contract",
        ],
    }
    write_json(output_dir / "minus_phase_repair_manifest.json", manifest)
    progress(
        "minus_phase_repair_end",
        variant_count=len(rows),
        candidate_pair_jsonl=str(candidate_pair_jsonl),
        manifest_json=str(output_dir / "minus_phase_repair_manifest.json"),
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build no-reference minus phase repair candidate variants.")
    parser.add_argument("--mix-path", type=Path, required=True)
    parser.add_argument("--vocal-path", type=Path, required=True)
    parser.add_argument("--minus-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--minus-route-id", required=True)
    parser.add_argument("--max-peak", type=float, default=0.98)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_no_reference_minus_phase_repair.py",
        entrypoint_path="/app/tools/run_no_reference_minus_phase_repair.py",
    )
    args = build_arg_parser().parse_args(argv)
    manifest = build_minus_phase_repair(
        mix_path=args.mix_path,
        vocal_path=args.vocal_path,
        minus_path=args.minus_path,
        output_dir=args.output_dir,
        track_id=str(args.track_id),
        candidate_id=str(args.candidate_id),
        minus_route_id=str(args.minus_route_id),
        max_peak=float(args.max_peak),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / "minus_phase_repair_manifest.json"),
                "candidate_pair_jsonl": manifest["candidate_pair_jsonl"],
                "variant_count": manifest["variant_count"],
                "delivery_allowed": manifest["delivery_allowed"],
                "release_ready": manifest["release_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
