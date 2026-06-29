#!/usr/bin/env python3
"""Train a small learned stem-region repair model from repair rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.stem_region_repair_common import make_region_mask, read_audio_stereo, validate_region_repair_row, write_json
    from tools.stem_region_repair_model import (
        StemRegionRepairTrainConfig,
        save_region_repair_checkpoint,
        train_region_repair_model_from_arrays,
        write_training_history,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import make_region_mask, read_audio_stereo, validate_region_repair_row, write_json
    from stem_region_repair_model import (
        StemRegionRepairTrainConfig,
        save_region_repair_checkpoint,
        train_region_repair_model_from_arrays,
        write_training_history,
    )


def _is_forensic_only_training_row(row: dict[str, Any], config: StemRegionRepairTrainConfig) -> bool:
    has_forensic_source = bool(str(row.get("forensic_source_path") or "").strip())
    has_forensic_loss = (
        float(config.forensic_source_decorrelation_loss_weight) > 0.0
        or float(config.forensic_source_envelope_loss_weight) > 0.0
        or float(config.forensic_source_active_energy_loss_weight) > 0.0
    )
    return (
        not bool(row.get("is_ground_truth"))
        and not bool(row.get("usable_for_promotion", False))
        and has_forensic_source
        and has_forensic_loss
        and float(config.region_loss_weight) == 0.0
    )


def _is_weak_teacher_region_loss_training_row(row: dict[str, Any], config: StemRegionRepairTrainConfig) -> bool:
    if not bool(config.allow_weak_teacher_region_loss):
        return False
    if bool(row.get("is_ground_truth")) or bool(row.get("usable_for_promotion", False)):
        return False
    if str(row.get("label_type") or "") != "weak_forensic_teacher":
        return False
    if float(config.region_loss_weight) <= 0.0:
        return False
    manifest_path = Path(str(row.get("weak_teacher_manifest_path") or ""))
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract") or {}
    outputs = manifest.get("outputs") or {}
    return (
        str(contract.get("label_type") or "") == "weak_forensic_teacher"
        and not bool(contract.get("is_ground_truth"))
        and not bool(contract.get("usable_as_ground_truth"))
        and bool(contract.get("usable_for_training"))
        and not bool(contract.get("usable_for_promotion"))
        and not bool(contract.get("delivery_allowed"))
        and str(outputs.get("target_vocals_path") or "") == str(row.get("target_stem_path") or "")
    )


def _load_rows(path: Path, config: StemRegionRepairTrainConfig) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("repair training rows file is empty")
    for row in rows:
        validate_region_repair_row(row)
        if (
            not row.get("is_ground_truth")
            and not _is_forensic_only_training_row(row, config)
            and not _is_weak_teacher_region_loss_training_row(row, config)
        ):
            raise ValueError(f"training row {row.get('row_id')} is not ground truth")
        if not row.get("target_stem_path"):
            raise ValueError(f"training row {row.get('row_id')} missing target_stem_path")
    return rows


def _fit_like(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio.astype(np.float32)
    if audio.shape[0] > frames:
        return audio[:frames].astype(np.float32)
    pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad], axis=0)


def _load_sibling_sum(row: dict[str, Any], *, frames: int, sample_rate: int) -> np.ndarray | None:
    sibling_paths = row.get("sibling_stem_paths")
    if not isinstance(sibling_paths, dict) or not sibling_paths:
        return None
    sibling_sum: np.ndarray | None = None
    for path in sibling_paths.values():
        audio, rate = read_audio_stereo(path)
        if int(rate) != int(sample_rate):
            raise ValueError(f"sample rate mismatch for sibling stem: {path}")
        audio = _fit_like(audio, frames)
        sibling_sum = audio if sibling_sum is None else sibling_sum + audio
    return sibling_sum


def _load_optional_forensic_source(row: dict[str, Any], *, frames: int, sample_rate: int) -> np.ndarray | None:
    source_path = str(row.get("forensic_source_path") or "").strip()
    if not source_path:
        return None
    audio, rate = read_audio_stereo(source_path)
    if int(rate) != int(sample_rate):
        raise ValueError(f"sample rate mismatch for forensic_source_path: {source_path}")
    return _fit_like(audio, frames)


def train_stem_region_repair_model(
    *,
    rows_jsonl: Path,
    output_dir: Path,
    config: StemRegionRepairTrainConfig,
) -> dict[str, Any]:
    rows = _load_rows(rows_jsonl, config)
    # V1 intentionally trains one patch checkpoint per focused defect family. This
    # keeps the first training target small enough for 12GB VRAM and avoids hiding
    # unrelated issues inside a broad model.
    if len(rows) != 1:
        raise ValueError("v1 trains exactly one focused repair row per checkpoint")
    row = rows[0]
    forensic_only_non_ground_truth_training = _is_forensic_only_training_row(row, config)
    weak_teacher_region_loss_training = _is_weak_teacher_region_loss_training_row(row, config)
    mix, sample_rate = read_audio_stereo(row["mix_path"])
    bad, bad_rate = read_audio_stereo(row["bad_stem_path"])
    target, target_rate = read_audio_stereo(row["target_stem_path"])
    if len({sample_rate, bad_rate, target_rate}) != 1:
        raise ValueError("sample rate mismatch between mix, bad stem, and target")
    region = row.get("region") or {}
    frames = min(mix.shape[0], bad.shape[0], target.shape[0])
    sibling_sum = _load_sibling_sum(row, frames=frames, sample_rate=sample_rate)
    forensic_source = _load_optional_forensic_source(row, frames=frames, sample_rate=sample_rate)
    mask = make_region_mask(
        frame_count=frames,
        sample_rate=sample_rate,
        start_sec=float(region.get("start_sec", 0.0)),
        end_sec=float(region.get("end_sec", frames / float(sample_rate))),
    )
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix[:frames],
        bad_stem_audio=bad[:frames],
        target_stem_audio=target[:frames],
        sibling_stem_audio=sibling_sum,
        target_mix_audio=mix[:frames] if sibling_sum is not None else None,
        forensic_source_audio=forensic_source,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=str(row["issue_type"]),
        config=config,
    )
    history = {
        **history,
        "region": {
            "start_sec": float(region.get("start_sec", 0.0)),
            "end_sec": float(region.get("end_sec", frames / float(sample_rate))),
        },
        "training_row": {
            "row_id": str(row.get("row_id") or ""),
            "root_cause_row_id": str(row.get("root_cause_row_id") or ""),
            "target_stem": str(row.get("target_stem") or row.get("stem") or ""),
            "reconstruction_surface": str(row.get("reconstruction_surface") or ""),
            "target_surface": str(row.get("target_surface") or ""),
            "forensic_source_path": str(row.get("forensic_source_path") or ""),
        },
    }
    checkpoint_path = save_region_repair_checkpoint(
        model=model,
        path=output_dir / "model.pt",
        training_history=history,
    )
    write_training_history(output_dir / "history.json", history)
    manifest = {
        "schema": "stem_region_repair_model_manifest.v1",
        "model_id": "stem_region_repair_v1",
        "checkpoint": str(checkpoint_path),
        "history_json": str(output_dir / "history.json"),
        "rows_jsonl": str(rows_jsonl),
        "row_count": len(rows),
        "issue_type": row["issue_type"],
        "stem": row.get("target_stem") or row.get("stem"),
        "shadow_only": True,
        "delivery_allowed": False,
        "reconstruction_aware": sibling_sum is not None and float(config.mix_conservation_loss_weight) > 0.0,
        "forensic_source_conditioned": forensic_source is not None and bool(config.forensic_source_conditioning),
        "contract": {
            "training_first": True,
            "single_focused_defect_v1": True,
            "no_threshold_relaxation": True,
            "forensic_only_non_ground_truth_training": bool(forensic_only_non_ground_truth_training),
            "weak_teacher_region_loss_training": bool(weak_teacher_region_loss_training),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=StemRegionRepairTrainConfig.epochs)
    parser.add_argument("--hidden-channels", type=int, default=StemRegionRepairTrainConfig.hidden_channels)
    parser.add_argument("--learning-rate", type=float, default=StemRegionRepairTrainConfig.learning_rate)
    parser.add_argument("--region-loss-weight", type=float, default=StemRegionRepairTrainConfig.region_loss_weight)
    parser.add_argument("--outside-loss-weight", type=float, default=StemRegionRepairTrainConfig.outside_loss_weight)
    parser.add_argument(
        "--mix-conservation-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.mix_conservation_loss_weight,
    )
    parser.add_argument(
        "--sibling-preserve-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.sibling_preserve_loss_weight,
    )
    parser.add_argument("--stem-sum-loss-weight", type=float, default=StemRegionRepairTrainConfig.stem_sum_loss_weight)
    parser.add_argument(
        "--stereo-preserve-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.stereo_preserve_loss_weight,
    )
    parser.add_argument(
        "--target-delta-active-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.target_delta_active_loss_weight,
    )
    parser.add_argument(
        "--forensic-source-decorrelation-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.forensic_source_decorrelation_loss_weight,
    )
    parser.add_argument(
        "--forensic-source-envelope-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.forensic_source_envelope_loss_weight,
    )
    parser.add_argument(
        "--forensic-source-active-energy-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.forensic_source_active_energy_loss_weight,
    )
    parser.add_argument(
        "--forensic-source-target-loss-weight",
        type=float,
        default=StemRegionRepairTrainConfig.forensic_source_target_loss_weight,
    )
    parser.add_argument(
        "--forensic-envelope-window-ms",
        type=float,
        default=StemRegionRepairTrainConfig.forensic_envelope_window_ms,
    )
    parser.add_argument("--forensic-source-conditioning", action="store_true")
    parser.add_argument("--allow-weak-teacher-region-loss", action="store_true")
    parser.add_argument("--source-skip-init-from-target", action="store_true")
    parser.add_argument("--max-delta", type=float, default=StemRegionRepairTrainConfig.max_delta)
    parser.add_argument("--device", default=StemRegionRepairTrainConfig.device)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = train_stem_region_repair_model(
        rows_jsonl=args.rows_jsonl,
        output_dir=args.out,
        config=StemRegionRepairTrainConfig(
            epochs=args.epochs,
            hidden_channels=args.hidden_channels,
            learning_rate=args.learning_rate,
            region_loss_weight=args.region_loss_weight,
            outside_loss_weight=args.outside_loss_weight,
            mix_conservation_loss_weight=args.mix_conservation_loss_weight,
            sibling_preserve_loss_weight=args.sibling_preserve_loss_weight,
            stem_sum_loss_weight=args.stem_sum_loss_weight,
            stereo_preserve_loss_weight=args.stereo_preserve_loss_weight,
            target_delta_active_loss_weight=args.target_delta_active_loss_weight,
            forensic_source_decorrelation_loss_weight=args.forensic_source_decorrelation_loss_weight,
            forensic_source_envelope_loss_weight=args.forensic_source_envelope_loss_weight,
            forensic_source_active_energy_loss_weight=args.forensic_source_active_energy_loss_weight,
            forensic_source_target_loss_weight=args.forensic_source_target_loss_weight,
            forensic_envelope_window_ms=args.forensic_envelope_window_ms,
            forensic_source_conditioning=bool(args.forensic_source_conditioning),
            allow_weak_teacher_region_loss=bool(args.allow_weak_teacher_region_loss),
            source_skip_init_from_target=bool(args.source_skip_init_from_target),
            max_delta=args.max_delta,
            device=args.device,
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
