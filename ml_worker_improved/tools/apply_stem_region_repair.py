#!/usr/bin/env python3
"""Apply a trained stem-region repair checkpoint to one localized defect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.stem_region_repair_common import make_region_mask, read_audio_stereo, validate_issue_type, write_audio, write_json
    from tools.stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import make_region_mask, read_audio_stereo, validate_issue_type, write_audio, write_json
    from stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint


def apply_stem_region_repair(
    *,
    checkpoint_path: Path,
    mix_path: Path,
    bad_stem_path: Path,
    output_dir: Path,
    stem: str,
    issue_type: str,
    start_sec: float,
    end_sec: float,
    forensic_source_path: Path | None = None,
) -> dict[str, Any]:
    issue_type = validate_issue_type(issue_type)
    model, checkpoint = load_region_repair_checkpoint(checkpoint_path)
    mix, sample_rate = read_audio_stereo(mix_path)
    bad, bad_rate = read_audio_stereo(bad_stem_path)
    if bad_rate != sample_rate:
        raise ValueError(f"sample rate mismatch: mix={sample_rate} bad_stem={bad_rate}")
    frames = min(mix.shape[0], bad.shape[0])
    mix = mix[:frames]
    bad = bad[:frames]
    forensic_source = None
    if forensic_source_path is not None:
        forensic_source, forensic_rate = read_audio_stereo(forensic_source_path)
        if forensic_rate != sample_rate:
            raise ValueError(f"sample rate mismatch: mix={sample_rate} forensic_source={forensic_rate}")
        if forensic_source.shape[0] > frames:
            forensic_source = forensic_source[:frames]
        elif forensic_source.shape[0] < frames:
            forensic_source = np.concatenate(
                [
                    forensic_source,
                    np.zeros((frames - forensic_source.shape[0], forensic_source.shape[1]), dtype=np.float32),
                ],
                axis=0,
            )
    mask = make_region_mask(
        frame_count=frames,
        sample_rate=sample_rate,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
        source_condition_audio=forensic_source,
    )
    repaired_path = write_audio(output_dir / "audio" / "repaired.flac", repaired, sample_rate)
    manifest = {
        "schema": "stem_region_repair_apply_manifest.v1",
        "stem": str(stem),
        "issue_type": issue_type,
        "checkpoint": str(checkpoint_path),
        "checkpoint_schema": checkpoint.get("schema"),
        "mix_path": str(mix_path),
        "bad_stem_path": str(bad_stem_path),
        "forensic_source_path": str(forensic_source_path) if forensic_source_path is not None else "",
        "repaired_stem_path": str(repaired_path),
        "region": {"start_sec": float(start_sec), "end_sec": float(end_sec)},
        "diagnostics": diagnostics,
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "contract": {
            "training_first": True,
            "localized_region_patch": True,
            "no_delivery_from_apply_stage": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--bad-stem", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--issue-type", required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--forensic-source", type=Path, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = apply_stem_region_repair(
        checkpoint_path=args.checkpoint,
        mix_path=args.mix,
        bad_stem_path=args.bad_stem,
        output_dir=args.out,
        stem=args.stem,
        issue_type=args.issue_type,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        forensic_source_path=args.forensic_source,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
