#!/usr/bin/env python3
"""Build synthetic training rows for learned stem-region repair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.stem_region_repair_common import (
        ISSUE_INSTRUMENT_LEAK,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_VOCAL_LEAK,
        LABEL_TYPE_SYNTHETIC_CLEAN,
        make_region_mask,
        read_audio_stereo,
        validate_region_repair_row,
        write_audio,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import (
        ISSUE_INSTRUMENT_LEAK,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_VOCAL_LEAK,
        LABEL_TYPE_SYNTHETIC_CLEAN,
        make_region_mask,
        read_audio_stereo,
        validate_region_repair_row,
        write_audio,
        write_json,
        write_jsonl,
    )


def _fit_like(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio.astype(np.float32)
    if audio.shape[0] > frames:
        return audio[:frames].astype(np.float32)
    pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad], axis=0)


def _synthetic_vocal_like(frames: int, sample_rate: int, amp: float) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    f0 = 185.0 + 22.0 * np.sin(2.0 * math.pi * 4.2 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(sample_rate)).astype(np.float32)
    mono = np.zeros(frames, dtype=np.float32)
    for harmonic, gain in ((1, 1.0), (2, 0.62), (3, 0.42), (5, 0.24), (7, 0.16)):
        mono += float(amp) * gain * np.sin(float(harmonic) * phase + 0.1 * harmonic)
    stereo = np.stack([mono, mono], axis=1)
    return stereo.astype(np.float32)


def _synthetic_instrument_like(frames: int, sample_rate: int, amp: float) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    pluck_spacing = max(1, int(round(sample_rate * 0.18)))
    envelope = np.zeros(frames, dtype=np.float32)
    decay = np.exp(-np.linspace(0.0, 5.0, max(8, int(sample_rate * 0.12)), dtype=np.float32))
    for start in range(0, frames, pluck_spacing):
        end = min(frames, start + decay.shape[0])
        envelope[start:end] += decay[: end - start]
    tone = (
        0.72 * np.sin(2.0 * math.pi * 330.0 * t)
        + 0.44 * np.sin(2.0 * math.pi * 495.0 * t + 0.17)
        + 0.23 * np.sin(2.0 * math.pi * 660.0 * t + 0.29)
    )
    side = 0.22 * np.sin(2.0 * math.pi * 417.0 * t + 0.37)
    left = float(amp) * envelope * (tone + side)
    right = float(amp) * envelope * (tone - side)
    return np.stack([left, right], axis=1).astype(np.float32)


def corrupt_clean_stem(
    *,
    clean_stem: np.ndarray,
    sample_rate: int,
    issue_type: str,
    region_mask: np.ndarray,
    foreign_source: np.ndarray | None = None,
    contamination_gain: float = 0.18,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean = clean_stem.astype(np.float32, copy=True)
    mask = region_mask[:, None].astype(np.float32)
    if issue_type == ISSUE_VOCAL_LEAK:
        leak = foreign_source
        if leak is None:
            leak = _synthetic_vocal_like(clean.shape[0], sample_rate, contamination_gain)
        leak = _fit_like(leak, clean.shape[0])
        bad = clean + leak * mask
        return bad.astype(np.float32), {
            "corruption": "vocal_leak_injection",
            "contamination_gain": float(contamination_gain),
        }
    if issue_type == ISSUE_INSTRUMENT_LEAK:
        leak = foreign_source
        if leak is None:
            leak = _synthetic_instrument_like(clean.shape[0], sample_rate, contamination_gain)
        leak = _fit_like(leak, clean.shape[0])
        bad = clean + leak * mask
        return bad.astype(np.float32), {
            "corruption": "instrument_leak_injection",
            "contamination_gain": float(contamination_gain),
        }
    if issue_type == ISSUE_STEREO_COLLAPSE:
        mid = np.mean(clean, axis=1, keepdims=True)
        collapsed = np.repeat(mid, 2, axis=1)
        bad = clean * (1.0 - mask) + collapsed * mask
        return bad.astype(np.float32), {"corruption": "region_stereo_collapse"}
    raise ValueError(f"unsupported synthetic repair issue_type: {issue_type}")


def build_stem_region_repair_dataset(
    *,
    clean_stem_path: Path,
    mix_path: Path,
    output_dir: Path,
    stem: str,
    issue_type: str,
    start_sec: float,
    end_sec: float,
    foreign_source_path: Path | None = None,
) -> dict[str, Any]:
    clean, sample_rate = read_audio_stereo(clean_stem_path)
    mix, mix_rate = read_audio_stereo(mix_path)
    if mix_rate != sample_rate:
        raise ValueError(f"sample rate mismatch: mix={mix_rate} clean={sample_rate}")
    frames = min(clean.shape[0], mix.shape[0])
    clean = clean[:frames]
    mix = mix[:frames]
    foreign = None
    if foreign_source_path is not None:
        foreign, foreign_rate = read_audio_stereo(foreign_source_path)
        if foreign_rate != sample_rate:
            raise ValueError(f"sample rate mismatch: foreign={foreign_rate} clean={sample_rate}")
        foreign = _fit_like(foreign, frames)
    mask = make_region_mask(frame_count=frames, sample_rate=sample_rate, start_sec=start_sec, end_sec=end_sec)
    bad, corruption = corrupt_clean_stem(
        clean_stem=clean,
        sample_rate=sample_rate,
        issue_type=issue_type,
        region_mask=mask,
        foreign_source=foreign,
    )

    audio_dir = output_dir / "audio"
    mix_out = write_audio(audio_dir / "mix.flac", mix, sample_rate)
    bad_out = write_audio(audio_dir / "bad.flac", bad, sample_rate)
    target_out = write_audio(audio_dir / "target.flac", clean, sample_rate)
    row = {
        "schema": "stem_region_repair_row.v1",
        "row_id": "rr000001",
        "stem": stem,
        "target_stem": stem,
        "issue_type": issue_type,
        "label_type": LABEL_TYPE_SYNTHETIC_CLEAN,
        "is_ground_truth": True,
        "usable_for_promotion": False,
        "mix_path": str(mix_out),
        "bad_stem_path": str(bad_out),
        "target_stem_path": str(target_out),
        "region": {"start_sec": float(start_sec), "end_sec": float(end_sec)},
        "corruption": corruption,
    }
    validate_region_repair_row(row)
    write_jsonl(output_dir / "rows.jsonl", [row])
    manifest = {
        "schema": "stem_region_repair_dataset.v1",
        "row_count": 1,
        "rows_jsonl": str(output_dir / "rows.jsonl"),
        "issue_type": issue_type,
        "stem": stem,
        "contract": {
            "synthetic_ground_truth": True,
            "generated_separator_ground_truth_allowed": False,
            "delivery_allowed": False,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-stem", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--issue-type", choices=[ISSUE_VOCAL_LEAK, ISSUE_STEREO_COLLAPSE], required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--foreign-source", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_stem_region_repair_dataset(
        clean_stem_path=args.clean_stem,
        mix_path=args.mix,
        output_dir=args.out,
        stem=args.stem,
        issue_type=args.issue_type,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        foreign_source_path=args.foreign_source,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
