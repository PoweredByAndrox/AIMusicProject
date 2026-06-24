#!/usr/bin/env python3
"""Train a small learned stem-region repair model from repair rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("repair training rows file is empty")
    for row in rows:
        validate_region_repair_row(row)
        if not row.get("is_ground_truth"):
            raise ValueError(f"training row {row.get('row_id')} is not ground truth")
        if not row.get("target_stem_path"):
            raise ValueError(f"training row {row.get('row_id')} missing target_stem_path")
    return rows


def train_stem_region_repair_model(
    *,
    rows_jsonl: Path,
    output_dir: Path,
    config: StemRegionRepairTrainConfig,
) -> dict[str, Any]:
    rows = _load_rows(rows_jsonl)
    # V1 intentionally trains one patch checkpoint per focused defect family. This
    # keeps the first training target small enough for 12GB VRAM and avoids hiding
    # unrelated issues inside a broad model.
    if len(rows) != 1:
        raise ValueError("v1 trains exactly one focused repair row per checkpoint")
    row = rows[0]
    mix, sample_rate = read_audio_stereo(row["mix_path"])
    bad, bad_rate = read_audio_stereo(row["bad_stem_path"])
    target, target_rate = read_audio_stereo(row["target_stem_path"])
    if len({sample_rate, bad_rate, target_rate}) != 1:
        raise ValueError("sample rate mismatch between mix, bad stem, and target")
    region = row.get("region") or {}
    frames = min(mix.shape[0], bad.shape[0], target.shape[0])
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
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=str(row["issue_type"]),
        config=config,
    )
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
        "contract": {
            "training_first": True,
            "single_focused_defect_v1": True,
            "no_threshold_relaxation": True,
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
            device=args.device,
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
