"""Train a focused synthetic bass stereo-collapse repair checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_stem_region_repair_dataset import build_stem_region_repair_dataset
from tools.stem_region_repair_common import ISSUE_STEREO_COLLAPSE
from tools.stem_region_repair_model import StemRegionRepairTrainConfig
from tools.train_stem_region_repair_model import train_stem_region_repair_model
from tools.validate_stem_repair_checkpoint import validate_stem_repair_checkpoint


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_progress(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def _make_synthetic_bass_fixture(output_dir: Path, *, sample_rate: int, duration_sec: float) -> dict[str, str]:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260628)
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)

    base = np.sin(2.0 * np.pi * 58.0 * t) + 0.45 * np.sin(2.0 * np.pi * 116.0 * t)
    width_mod = 0.18 * np.sin(2.0 * np.pi * 1.7 * t)
    left = 0.34 * base * (1.0 + width_mod)
    right = 0.34 * np.sin(2.0 * np.pi * 58.0 * t + 0.42) + 0.15 * np.sin(2.0 * np.pi * 116.0 * t - 0.27)
    bass = np.stack([left, right], axis=1).astype(np.float32)

    drums = np.zeros_like(bass)
    for onset in np.arange(0.25, duration_sec, 0.5):
        start = int(onset * sample_rate)
        frames = min(int(0.035 * sample_rate), len(t) - start)
        envelope = np.exp(-np.linspace(0.0, 5.0, frames, dtype=np.float32))
        transient = 0.08 * rng.standard_normal(frames).astype(np.float32) * envelope
        drums[start : start + frames, 0] += transient
        drums[start : start + frames, 1] -= transient * 0.7
    pad = 0.03 * np.stack(
        [
            np.sin(2.0 * np.pi * 330.0 * t),
            np.sin(2.0 * np.pi * 333.0 * t + 0.2),
        ],
        axis=1,
    ).astype(np.float32)
    siblings = np.clip(drums + pad, -1.0, 1.0).astype(np.float32)
    mix = np.clip(bass + siblings, -1.0, 1.0).astype(np.float32)

    bass_path = audio_dir / "bass.flac"
    mix_path = audio_dir / "mix.flac"
    siblings_path = audio_dir / "siblings.flac"
    sf.write(bass_path, bass, sample_rate)
    sf.write(mix_path, mix, sample_rate)
    sf.write(siblings_path, siblings, sample_rate)
    return {
        "clean_bass_path": str(bass_path),
        "mix_path": str(mix_path),
        "siblings_path": str(siblings_path),
    }


def run_cycle(
    *,
    output_dir: Path,
    epochs: int,
    device: str,
    hidden_channels: int,
    learning_rate: float,
    root_cause_row_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _write_progress(progress_path, "start", output_dir=str(output_dir), device=device, epochs=epochs)

    fixture = _make_synthetic_bass_fixture(output_dir, sample_rate=44100, duration_sec=6.0)
    _write_progress(progress_path, "fixture_written", **fixture)

    dataset_manifest = build_stem_region_repair_dataset(
        clean_stem_path=Path(fixture["clean_bass_path"]),
        mix_path=Path(fixture["mix_path"]),
        output_dir=output_dir / "ds",
        stem="bass",
        issue_type=ISSUE_STEREO_COLLAPSE,
        start_sec=0.5,
        end_sec=5.5,
        sibling_stem_paths={"siblings": Path(fixture["siblings_path"])},
        root_cause_row_id=root_cause_row_id,
    )
    _write_progress(progress_path, "dataset_built", rows_jsonl=dataset_manifest["rows_jsonl"])

    model_manifest = train_stem_region_repair_model(
        rows_jsonl=Path(str(dataset_manifest["rows_jsonl"])),
        output_dir=output_dir / "model",
        config=StemRegionRepairTrainConfig(
            epochs=int(epochs),
            hidden_channels=int(hidden_channels),
            learning_rate=float(learning_rate),
            region_loss_weight=10.0,
            outside_loss_weight=1.0,
            mix_conservation_loss_weight=4.0,
            sibling_preserve_loss_weight=2.0,
            stem_sum_loss_weight=4.0,
            stereo_preserve_loss_weight=8.0,
            target_delta_active_loss_weight=2.0,
            max_delta=1.0,
            device=device,
        ),
    )
    _write_progress(progress_path, "model_trained", checkpoint=model_manifest["checkpoint"])

    validation_manifest = validate_stem_repair_checkpoint(
        checkpoint_path=Path(str(model_manifest["checkpoint"])),
        benchmark_rows_jsonl=Path(str(dataset_manifest["rows_jsonl"])),
        output_dir=output_dir / "val",
    )
    _write_progress(
        progress_path,
        "validated",
        improved_count=validation_manifest["improved_count"],
        evaluated_count=validation_manifest["evaluated_count"],
    )

    manifest = {
        "schema": "bass_stereo_repair_training_cycle.v1",
        "issue_type": ISSUE_STEREO_COLLAPSE,
        "target_stem": "bass",
        "fixture": fixture,
        "dataset_manifest": str(output_dir / "ds" / "manifest.json"),
        "model_manifest": str(output_dir / "model" / "manifest.json"),
        "validation_manifest": str(output_dir / "val" / "manifest.json"),
        "checkpoint": model_manifest["checkpoint"],
        "validation": {
            "evaluated_count": validation_manifest["evaluated_count"],
            "improved_count": validation_manifest["improved_count"],
        },
        "contract": {
            "synthetic_ground_truth_only": True,
            "generated_separator_ground_truth_allowed": False,
            "delivery_allowed": False,
            "presentation_status": "training_checkpoint_only",
            "requires_runtime_gates_before_listen": True,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_progress(progress_path, "end", checkpoint=manifest["checkpoint"])
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--root-cause-row-id", default="rc000046")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = run_cycle(
        output_dir=args.out,
        epochs=args.epochs,
        device=args.device,
        hidden_channels=args.hidden_channels,
        learning_rate=args.learning_rate,
        root_cause_row_id=args.root_cause_row_id,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
