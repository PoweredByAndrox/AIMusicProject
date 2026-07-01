#!/usr/bin/env python3
"""Train a small Mac/MPS stem refiner on a focused synthetic leak defect."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mac_melband_pitch_guard import pitch_shift_resample_same_length
from tools.stem_region_repair_common import ISSUE_VOCAL_LEAK, make_region_mask, write_audio, write_json
from tools.stem_region_repair_model import (
    StemRegionRepairTrainConfig,
    apply_region_repair_model,
    resolve_region_repair_training_device,
    save_region_repair_checkpoint,
    train_region_repair_model_from_arrays,
    write_training_history,
)


SAMPLE_RATE = 16_000


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, ensure_ascii=False) + "\n")
        handle.flush()


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


def _mse_region(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    frames = min(left.shape[0], right.shape[0], mask.shape[0])
    weights = mask[:frames, None].astype(np.float32)
    denom = max(float(np.sum(weights)) * float(left.shape[1]), 1.0)
    return float(np.sum(np.square((left[:frames] - right[:frames]) * weights)) / denom)


def _projection_energy(audio: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    frames = min(audio.shape[0], source.shape[0], mask.shape[0])
    tested = np.mean(audio[:frames], axis=1).astype(np.float64)
    src = np.mean(source[:frames], axis=1).astype(np.float64)
    tested_hp = np.diff(tested)
    src_hp = np.diff(src)
    mask_hp = np.minimum(mask[:frames][1:], mask[:frames][:-1]).astype(np.float64)
    tested_hp *= mask_hp
    src_hp *= mask_hp
    denom = float(np.sum(np.square(src_hp)) + 1.0e-8)
    coeff = float(np.sum(tested_hp * src_hp) / denom)
    projected = coeff * src_hp
    return float(np.sum(np.square(projected)) / max(float(np.sum(mask_hp)), 1.0))


def _non_vocal_clean_stem(stem: str, seconds: float) -> np.ndarray:
    frames = int(round(float(seconds) * SAMPLE_RATE))
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    stem = str(stem or "drums").strip().lower()
    if stem == "drums":
        mono = np.zeros(frames, dtype=np.float32)
        impulse = np.exp(-np.linspace(0.0, 7.0, int(SAMPLE_RATE * 0.035), dtype=np.float32))
        for start in range(0, frames, int(SAMPLE_RATE * 0.16)):
            end = min(frames, start + len(impulse))
            mono[start:end] += 0.18 * impulse[: end - start]
        side = 0.018 * np.sin(2.0 * math.pi * 740.0 * t)
    elif stem == "bass":
        mono = 0.14 * np.sin(2.0 * math.pi * 73.42 * t) + 0.045 * np.sin(2.0 * math.pi * 146.84 * t)
        side = 0.010 * np.sin(2.0 * math.pi * 91.0 * t)
    else:
        mono = 0.09 * np.sin(2.0 * math.pi * 293.66 * t) + 0.045 * np.sin(2.0 * math.pi * 587.33 * t)
        side = 0.018 * np.sin(2.0 * math.pi * 311.0 * t)
    return np.stack([mono + side, mono - side], axis=1).astype(np.float32)


def _voice_like(seconds: float) -> np.ndarray:
    frames = int(round(float(seconds) * SAMPLE_RATE))
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    f0 = 185.0 + 18.0 * np.sin(2.0 * math.pi * 4.7 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(SAMPLE_RATE)).astype(np.float32)
    mono = np.zeros(frames, dtype=np.float32)
    for harmonic, gain in ((1, 0.15), (2, 0.10), (3, 0.07), (5, 0.04), (7, 0.024), (9, 0.015)):
        mono += gain * np.sin(float(harmonic) * phase + 0.09 * harmonic)
    mono *= (0.48 + 0.52 * np.sin(2.0 * math.pi * 2.3 * t) ** 2).astype(np.float32)
    side = 0.012 * np.sin(2.0 * math.pi * 410.0 * t)
    return np.stack([mono + side, mono - side], axis=1).astype(np.float32)


def run_training_cycle(
    *,
    output_dir: Path,
    target_stem: str = "drums",
    seconds: float = 2.0,
    epochs: int = 160,
    require_mps: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start", require_mps=bool(require_mps))
    device = resolve_region_repair_training_device("mps")
    if bool(require_mps) and device.type != "mps":
        manifest = {
            "schema": "mac_stem_refiner_training_cycle.v1",
            "status": "blocked_mps_unavailable",
            "delivery_allowed": False,
            "presentation_status": "training_diagnostic_only",
            "resolved_device": str(device),
        }
        write_json(output_dir / "manifest.json", manifest)
        _append_progress(progress_path, "preflight", "blocked_mps_unavailable", resolved_device=str(device))
        return manifest
    _append_progress(progress_path, "preflight", "end", resolved_device=str(device))

    _append_progress(progress_path, "build_synthetic_case", "start", target_stem=target_stem)
    clean = _non_vocal_clean_stem(target_stem, seconds)
    vocal_source = _voice_like(seconds)
    shifted_vocal = pitch_shift_resample_same_length(vocal_source, semitones=-5.0)
    region_mask = make_region_mask(
        frame_count=clean.shape[0],
        sample_rate=SAMPLE_RATE,
        start_sec=max(0.0, float(seconds) * 0.24),
        end_sec=min(float(seconds), float(seconds) * 0.78),
        fade_sec=0.025,
    )
    bad = clean + shifted_vocal * region_mask[:, None] * 0.42
    sibling = 0.05 * _voice_like(seconds)
    mix = clean + sibling
    audio_dir = output_dir / "audio"
    clean_path = write_audio(audio_dir / "clean.flac", clean, SAMPLE_RATE)
    bad_path = write_audio(audio_dir / "bad.flac", bad, SAMPLE_RATE)
    mix_path = write_audio(audio_dir / "mix.flac", mix, SAMPLE_RATE)
    source_path = write_audio(audio_dir / "src.flac", shifted_vocal, SAMPLE_RATE)
    _append_progress(progress_path, "build_synthetic_case", "end")

    before_mse = _mse_region(bad, clean, region_mask)
    before_projection = _projection_energy(bad, shifted_vocal, region_mask)
    _append_progress(progress_path, "train", "start", epochs=int(epochs), device=str(device))
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=clean,
        forensic_source_audio=shifted_vocal,
        region_mask=region_mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_VOCAL_LEAK,
        config=StemRegionRepairTrainConfig(
            hidden_channels=24,
            kernel_size=31,
            epochs=int(epochs),
            learning_rate=0.004,
            region_loss_weight=1.0,
            outside_loss_weight=0.35,
            forensic_source_conditioning=True,
            forensic_source_decorrelation_loss_weight=0.15,
            forensic_source_envelope_loss_weight=0.05,
            max_delta=0.6,
            device=str(device),
        ),
    )
    checkpoint_path = save_region_repair_checkpoint(model=model, path=output_dir / "model.pt", training_history=history)
    write_training_history(output_dir / "history.json", history)
    _append_progress(progress_path, "train", "end", final_loss=history.get("final_loss"), resolved_device=history.get("resolved_device"))

    repaired, apply_diag = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        source_condition_audio=shifted_vocal,
        region_mask=region_mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_VOCAL_LEAK,
    )
    repaired_path = write_audio(audio_dir / "repaired.flac", repaired, SAMPLE_RATE)
    after_mse = _mse_region(repaired, clean, region_mask)
    after_projection = _projection_energy(repaired, shifted_vocal, region_mask)
    target_issue_improved = after_mse < before_mse and after_projection < before_projection
    manifest = {
        "schema": "mac_stem_refiner_training_cycle.v1",
        "status": "trained" if target_issue_improved else "trained_but_not_improved",
        "profile": "mac_stem_refiner_v1",
        "target_stem": str(target_stem),
        "issue_type": ISSUE_VOCAL_LEAK,
        "checkpoint": str(checkpoint_path),
        "history_json": str(output_dir / "history.json"),
        "audio": {
            "mix": str(mix_path),
            "clean_target": str(clean_path),
            "bad_stem": str(bad_path),
            "forensic_source": str(source_path),
            "repaired": str(repaired_path),
        },
        "metrics": {
            "before_region_mse": before_mse,
            "after_region_mse": after_mse,
            "before_source_projection_energy": before_projection,
            "after_source_projection_energy": after_projection,
            "region_mse_ratio": after_mse / max(before_mse, 1.0e-12),
            "source_projection_ratio": after_projection / max(before_projection, 1.0e-12),
            "target_issue_improved": bool(target_issue_improved),
            "apply_diagnostics": apply_diag,
            "bad_rms": _rms(bad),
            "repaired_rms": _rms(repaired),
        },
        "training": history,
        "contract": {
            "weights_updated": True,
            "synthetic_ground_truth_only": True,
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "usable_for_promotion": False,
        },
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "training_diagnostic_only",
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", run_status=manifest["status"], target_issue_improved=target_issue_improved)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-stem", default="drums")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_training_cycle(
        output_dir=args.out,
        target_stem=args.target_stem,
        seconds=float(args.seconds),
        epochs=int(args.epochs),
        require_mps=bool(args.require_mps),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
