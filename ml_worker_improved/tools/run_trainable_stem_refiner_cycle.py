#!/usr/bin/env python3
"""Run a minimal trainable audio refiner cycle for one localized stem defect."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.stem_region_repair_common import (
        ISSUE_BAD_RECONSTRUCTION,
        ISSUE_INSTRUMENT_LEAK,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_VOCAL_LEAK,
        make_region_mask,
        validate_issue_type,
        write_audio,
        write_json,
    )
    from tools.stem_region_repair_model import (
        StemRegionRepairTrainConfig,
        apply_region_repair_model,
        save_region_repair_checkpoint,
        train_region_repair_model_from_arrays,
        write_training_history,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import (  # type: ignore
        ISSUE_BAD_RECONSTRUCTION,
        ISSUE_INSTRUMENT_LEAK,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_VOCAL_LEAK,
        make_region_mask,
        validate_issue_type,
        write_audio,
        write_json,
    )
    from stem_region_repair_model import (  # type: ignore
        StemRegionRepairTrainConfig,
        apply_region_repair_model,
        save_region_repair_checkpoint,
        train_region_repair_model_from_arrays,
        write_training_history,
    )


def _sine(freq: float, *, frames: int, sample_rate: int, amp: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    return (amp * np.sin(2.0 * math.pi * float(freq) * t + float(phase))).astype(np.float32)


def _stereo_from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def _build_clean_stem(*, frames: int, sample_rate: int, stem: str) -> np.ndarray:
    base_freq = {
        "vocals": 185.0,
        "drums": 92.0,
        "bass": 55.0,
        "guitar": 330.0,
        "piano": 247.0,
        "other": 410.0,
        "minus": 140.0,
    }.get(str(stem), 180.0)
    mid = (
        _sine(base_freq, frames=frames, sample_rate=sample_rate, amp=0.12)
        + _sine(base_freq * 2.0, frames=frames, sample_rate=sample_rate, amp=0.035, phase=0.2)
    )
    side = _sine(base_freq * 1.37, frames=frames, sample_rate=sample_rate, amp=0.025, phase=0.7)
    return _stereo_from_mid_side(mid, side)


def _build_vocal_like(*, frames: int, sample_rate: int, amp: float) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    f0 = 180.0 + 16.0 * np.sin(2.0 * math.pi * 5.0 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(sample_rate)).astype(np.float32)
    mid = np.zeros(frames, dtype=np.float32)
    for harmonic, gain in ((1, 1.0), (2, 0.55), (3, 0.34), (5, 0.18)):
        mid += amp * gain * np.sin(float(harmonic) * phase + 0.07 * harmonic).astype(np.float32)
    side = _sine(510.0, frames=frames, sample_rate=sample_rate, amp=amp * 0.12, phase=0.3)
    return _stereo_from_mid_side(mid, side)


def _build_instrument_detail(*, frames: int, sample_rate: int, amp: float) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    envelope = 0.55 + 0.45 * np.sin(2.0 * math.pi * 7.0 * t) ** 2
    mid = amp * envelope * (
        0.7 * np.sin(2.0 * math.pi * 660.0 * t)
        + 0.4 * np.sin(2.0 * math.pi * 990.0 * t + 0.2)
    )
    side = amp * 0.3 * envelope * np.sin(2.0 * math.pi * 740.0 * t + 0.6)
    return _stereo_from_mid_side(mid.astype(np.float32), side.astype(np.float32))


def _make_bad_target_mix(
    *,
    clean: np.ndarray,
    sibling: np.ndarray,
    mask: np.ndarray,
    sample_rate: int,
    issue_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frames = clean.shape[0]
    mask_2d = mask[:, None].astype(np.float32)
    if issue_type == ISSUE_VOCAL_LEAK:
        leak = _build_vocal_like(frames=frames, sample_rate=sample_rate, amp=0.06)
        bad = clean + leak * mask_2d
        target = clean
        corruption = {"type": "synthetic_vocal_leak", "leak_gain": 0.06}
    elif issue_type == ISSUE_INSTRUMENT_LEAK:
        leak = _build_instrument_detail(frames=frames, sample_rate=sample_rate, amp=0.055)
        bad = clean + leak * mask_2d
        target = clean
        corruption = {"type": "synthetic_instrument_leak", "leak_gain": 0.055}
    elif issue_type == ISSUE_STEREO_COLLAPSE:
        mid = np.mean(clean, axis=1, keepdims=True)
        collapsed = np.repeat(mid, 2, axis=1)
        bad = clean * (1.0 - mask_2d) + collapsed * mask_2d
        target = clean
        corruption = {"type": "synthetic_stereo_collapse"}
    elif issue_type == ISSUE_BAD_RECONSTRUCTION:
        detail = _build_instrument_detail(frames=frames, sample_rate=sample_rate, amp=0.05)
        target = clean + detail * mask_2d
        bad = clean
        corruption = {"type": "synthetic_missing_detail", "detail_gain": 0.05}
    else:
        raise ValueError(f"unsupported trainable refiner issue_type: {issue_type}")
    mix = target + sibling
    return bad.astype(np.float32), target.astype(np.float32), {"corruption": corruption, "mix_source": "target_plus_sibling"}


def _region_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    weights = mask[:, None].astype(np.float32)
    denom = max(float(np.sum(weights)) * float(a.shape[1]), 1.0)
    return float(np.sum(np.square((a - b) * weights)) / denom)


def _outside_delta_mse(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    weights = (1.0 - mask[:, None]).astype(np.float32)
    denom = max(float(np.sum(weights)) * float(before.shape[1]), 1.0)
    return float(np.sum(np.square((after - before) * weights)) / denom)


def _trainable_parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _state_abs_sum(model: torch.nn.Module) -> float:
    return float(sum(torch.sum(torch.abs(value)).item() for value in model.state_dict().values()))


def run_trainable_stem_refiner_cycle(
    *,
    output_dir: Path,
    stem: str,
    issue_type: str,
    epochs: int = 160,
    hidden_channels: int = 24,
    learning_rate: float = 0.006,
    device: str = "cuda",
    sample_rate: int = 16_000,
    seconds: float = 0.75,
) -> dict[str, Any]:
    issue_type = validate_issue_type(issue_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(round(float(sample_rate) * float(seconds))))
    start_sec = max(0.05, float(seconds) * 0.22)
    end_sec = min(float(seconds) - 0.05, float(seconds) * 0.78)
    mask = make_region_mask(frame_count=frames, sample_rate=sample_rate, start_sec=start_sec, end_sec=end_sec)
    clean = _build_clean_stem(frames=frames, sample_rate=sample_rate, stem=stem)
    sibling = _build_clean_stem(frames=frames, sample_rate=sample_rate, stem="minus" if stem != "minus" else "other") * 0.45
    bad, target, defect = _make_bad_target_mix(
        clean=clean,
        sibling=sibling,
        mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
    )
    mix = target + sibling

    before_mse = _region_mse(bad, target, mask)
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=target,
        sibling_stem_audio=sibling,
        target_mix_audio=mix,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
        config=StemRegionRepairTrainConfig(
            epochs=int(epochs),
            hidden_channels=int(hidden_channels),
            learning_rate=float(learning_rate),
            mix_conservation_loss_weight=1.0,
            stereo_preserve_loss_weight=1.0 if issue_type == ISSUE_STEREO_COLLAPSE else 0.0,
            device=str(device),
        ),
    )
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
    )
    after_mse = _region_mse(repaired, target, mask)
    pair_before_mse = _region_mse(bad + sibling, mix, mask)
    pair_after_mse = _region_mse(repaired + sibling, mix, mask)
    outside_delta_mse = _outside_delta_mse(bad, repaired, mask)
    weights_abs_sum = _state_abs_sum(model)

    audio_dir = output_dir / "audio"
    mix_path = write_audio(audio_dir / "mix.flac", mix, sample_rate)
    bad_path = write_audio(audio_dir / "bad.flac", bad, sample_rate)
    target_path = write_audio(audio_dir / "target.flac", target, sample_rate)
    repaired_path = write_audio(audio_dir / "repaired.flac", repaired, sample_rate)
    sibling_path = write_audio(audio_dir / "sibling.flac", sibling, sample_rate)
    history = {
        **history,
        "region": {"start_sec": float(start_sec), "end_sec": float(end_sec)},
        "weights_abs_sum_after_training": weights_abs_sum,
        "trainable_parameter_count": _trainable_parameter_count(model),
    }
    checkpoint_path = save_region_repair_checkpoint(model=model, path=output_dir / "model" / "model.pt", training_history=history)
    history_path = write_training_history(output_dir / "history.json", history)
    before_after = {
        "schema": "trainable_stem_refiner_before_after.v1",
        "stem": str(stem),
        "issue_type": issue_type,
        "target_region_mse_before": before_mse,
        "target_region_mse_after": after_mse,
        "target_region_improvement_ratio": float(after_mse / max(before_mse, 1.0e-12)),
        "pair_region_mse_before": pair_before_mse,
        "pair_region_mse_after": pair_after_mse,
        "pair_region_improvement_ratio": float(pair_after_mse / max(pair_before_mse, 1.0e-12)),
        "outside_region_mse_delta": outside_delta_mse,
        "diagnostics": diagnostics,
    }
    before_after_path = write_json(output_dir / "before_after.json", before_after)
    manifest = {
        "schema": "trainable_stem_refiner_cycle_manifest.v1",
        "output_dir": str(output_dir),
        "stem": str(stem),
        "issue_type": issue_type,
        "sample_rate": int(sample_rate),
        "region": {"start_sec": float(start_sec), "end_sec": float(end_sec)},
        "mix_path": str(mix_path),
        "bad_stem_path": str(bad_path),
        "target_stem_path": str(target_path),
        "sibling_stem_path": str(sibling_path),
        "repaired_stem_path": str(repaired_path),
        "checkpoint": str(checkpoint_path),
        "history_json": str(history_path),
        "before_after_json": str(before_after_path),
        "weights_changed": weights_abs_sum > 0.0 and float(history.get("final_loss", 0.0)) < float(history.get("initial_loss", 0.0)),
        "weights_abs_sum_after_training": weights_abs_sum,
        "trainable_parameter_count": _trainable_parameter_count(model),
        "target_issue_improved": after_mse < before_mse,
        "pair_reconstruction_improved": pair_after_mse < pair_before_mse,
        "artifact_rag_write_allowed": False,
        "listen_pack_created": False,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "defect": defect,
        "contract": {
            "trainable_audio_weights": True,
            "raw_audio_delivery_path": False,
            "generated_audio_delivery_allowed": False,
            "synthetic_ground_truth_only": True,
            "no_threshold_relaxation": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stem", default="drums")
    parser.add_argument("--issue-type", default=ISSUE_VOCAL_LEAK)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seconds", type=float, default=0.75)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = run_trainable_stem_refiner_cycle(
        output_dir=args.out,
        stem=args.stem,
        issue_type=args.issue_type,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        learning_rate=args.learning_rate,
        device=args.device,
        sample_rate=args.sample_rate,
        seconds=args.seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
