#!/usr/bin/env python3
"""Validate a stem-region repair checkpoint on hardcase benchmark rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from tools.stem_region_repair_common import make_region_mask, read_audio_stereo, write_audio, write_json, write_jsonl
    from tools.stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import make_region_mask, read_audio_stereo, write_audio, write_json, write_jsonl
    from stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _fit_frames(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    frames = min(array.shape[0] for array in arrays)
    return tuple(array[:frames].astype(np.float32) for array in arrays)


def _mse_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    weights = mask[:, None].astype(np.float32)
    denom = max(float(np.sum(weights)) * float(a.shape[1]), 1.0)
    return float(np.sum(np.square((a - b) * weights)) / denom)


def _validate_row(
    row: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    required_paths = [row.get("mix_path"), row.get("bad_stem_path"), row.get("target_stem_path")]
    if any(not path for path in required_paths):
        return {
            "row_id": str(row.get("row_id") or ""),
            "evaluated": False,
            "improved": False,
            "reason": "missing_mix_bad_or_target_path",
        }
    mix_path = Path(str(row["mix_path"]))
    bad_path = Path(str(row["bad_stem_path"]))
    target_path = Path(str(row["target_stem_path"]))
    if not mix_path.exists() or not bad_path.exists() or not target_path.exists():
        return {
            "row_id": str(row.get("row_id") or ""),
            "evaluated": False,
            "improved": False,
            "reason": "audio_path_missing",
        }

    mix, sample_rate = read_audio_stereo(mix_path)
    bad, bad_rate = read_audio_stereo(bad_path)
    target, target_rate = read_audio_stereo(target_path)
    if len({sample_rate, bad_rate, target_rate}) != 1:
        return {
            "row_id": str(row.get("row_id") or ""),
            "evaluated": False,
            "improved": False,
            "reason": "sample_rate_mismatch",
        }
    mix, bad, target = _fit_frames(mix, bad, target)
    region = row.get("region") if isinstance(row.get("region"), Mapping) else {}
    mask = make_region_mask(
        frame_count=bad.shape[0],
        sample_rate=sample_rate,
        start_sec=float(region.get("start_sec", 0.0)),
        end_sec=float(region.get("end_sec", bad.shape[0] / float(sample_rate))),
    )
    model, checkpoint_payload = load_region_repair_checkpoint(checkpoint_path)
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=str(row.get("issue_type") or checkpoint_payload.get("training_history", {}).get("issue_type") or "unknown"),
    )
    before_error = _mse_region(bad, target, mask)
    after_error = _mse_region(repaired, target, mask)
    repaired_path = write_audio(output_dir / "audio" / f"{str(row.get('row_id') or 'row')}.flac", repaired, sample_rate)
    return {
        "row_id": str(row.get("row_id") or ""),
        "evaluated": True,
        "improved": after_error < before_error,
        "before_error": before_error,
        "after_error": after_error,
        "repaired_audio_path": str(repaired_path),
        "diagnostics": diagnostics,
    }


def validate_stem_repair_checkpoint(
    *,
    checkpoint_path: Path,
    benchmark_rows_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = _read_jsonl(benchmark_rows_jsonl)
    results = [
        _validate_row(row, checkpoint_path=checkpoint_path, output_dir=output_dir)
        for row in rows
    ]
    before_after_jsonl = output_dir / "before_after.jsonl"
    write_jsonl(before_after_jsonl, results)
    manifest = {
        "schema": "stem_repair_checkpoint_validation.v1",
        "checkpoint": str(checkpoint_path),
        "benchmark_rows_jsonl": str(benchmark_rows_jsonl),
        "row_count": len(rows),
        "evaluated_count": sum(1 for row in results if row.get("evaluated")),
        "improved_count": sum(1 for row in results if row.get("improved")),
        "before_after_jsonl": str(before_after_jsonl),
        "delivery_allowed": False,
        "promotion_allowed": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = validate_stem_repair_checkpoint(
        checkpoint_path=args.checkpoint,
        benchmark_rows_jsonl=args.rows_jsonl,
        output_dir=args.out,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
