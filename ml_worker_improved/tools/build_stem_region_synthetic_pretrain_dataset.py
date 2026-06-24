#!/usr/bin/env python3
"""Build synthetic supervised pretrain rows from hard-case issue metadata."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.build_stem_region_repair_dataset import corrupt_clean_stem
    from tools.stem_region_repair_common import (
        ISSUE_STEREO_COLLAPSE,
        ISSUE_VOCAL_LEAK,
        LABEL_TYPE_SYNTHETIC_CLEAN,
        make_region_mask,
        read_audio_stereo,
        validate_issue_type,
        validate_region_repair_row,
        write_audio,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from build_stem_region_repair_dataset import corrupt_clean_stem
    from stem_region_repair_common import (
        ISSUE_STEREO_COLLAPSE,
        ISSUE_VOCAL_LEAK,
        LABEL_TYPE_SYNTHETIC_CLEAN,
        make_region_mask,
        read_audio_stereo,
        validate_issue_type,
        validate_region_repair_row,
        write_audio,
        write_json,
        write_jsonl,
    )


SUPPORTED_ISSUES = {ISSUE_STEREO_COLLAPSE, ISSUE_VOCAL_LEAK}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reference_shape(path: str, duration_sec: float) -> tuple[int, int]:
    if path and Path(path).exists():
        audio, sample_rate = read_audio_stereo(path)
        frames = min(audio.shape[0], max(1, int(round(float(duration_sec) * sample_rate))))
        return frames, sample_rate
    sample_rate = 44_100
    return max(1, int(round(float(duration_sec) * sample_rate))), sample_rate


def _stem_clean_signal(stem: str, frames: int, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    stem = str(stem or "").strip().lower()
    if stem == "bass":
        mono = 0.12 * np.sin(2.0 * math.pi * 73.0 * t) + 0.045 * np.sin(2.0 * math.pi * 146.0 * t)
        side = 0.018 * np.sin(2.0 * math.pi * 91.0 * t)
    elif stem == "drums":
        mono = np.zeros(frames, dtype=np.float32)
        spacing = max(1, int(sample_rate * 0.115))
        impulse = np.exp(-np.linspace(0.0, 7.0, max(8, int(sample_rate * 0.035)), dtype=np.float32))
        for start in range(0, frames, spacing):
            end = min(frames, start + impulse.shape[0])
            mono[start:end] += 0.16 * impulse[: end - start]
        mono += 0.015 * rng.standard_normal(frames).astype(np.float32)
        side = 0.012 * rng.standard_normal(frames).astype(np.float32)
    else:
        base_freq = 220.0 + float((abs(hash(stem)) % 180))
        mono = 0.09 * np.sin(2.0 * math.pi * base_freq * t) + 0.035 * np.sin(2.0 * math.pi * base_freq * 1.5 * t)
        side = 0.012 * np.sin(2.0 * math.pi * (base_freq + 17.0) * t)
    stereo = np.stack([mono + side, mono - side], axis=1)
    return np.clip(stereo, -0.8, 0.8).astype(np.float32)


def _synthetic_mix(target: np.ndarray, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 11)
    frames = target.shape[0]
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    pad = 0.035 * np.sin(2.0 * math.pi * 330.0 * t) + 0.02 * np.sin(2.0 * math.pi * 515.0 * t)
    side = 0.006 * rng.standard_normal(frames).astype(np.float32)
    context = np.stack([pad + side, pad - side], axis=1).astype(np.float32)
    return np.clip(target + context, -1.0, 1.0).astype(np.float32)


def build_stem_region_synthetic_pretrain_dataset(
    *,
    hardcase_rows_jsonl: Path,
    output_dir: Path,
    max_rows: int = 8,
    duration_sec: float = 2.0,
) -> dict[str, Any]:
    hardcase_rows = _read_jsonl(hardcase_rows_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for hardcase in hardcase_rows:
        if len(rows) >= int(max_rows):
            break
        issue_type = validate_issue_type(hardcase.get("issue_type"))
        if issue_type not in SUPPORTED_ISSUES:
            skipped.append({"row_id": str(hardcase.get("row_id") or ""), "reason": "unsupported_issue"})
            continue
        if str(hardcase.get("training_state") or "") != "data_gap":
            skipped.append({"row_id": str(hardcase.get("row_id") or ""), "reason": "not_data_gap"})
            continue
        stem = str(hardcase.get("affected_stem") or hardcase.get("target_stem") or "").strip().lower()
        frames, sample_rate = _reference_shape(str(hardcase.get("bad_stem_path") or ""), duration_sec)
        row_index = len(rows)
        region_mask = make_region_mask(
            frame_count=frames,
            sample_rate=sample_rate,
            start_sec=0.0,
            end_sec=frames / float(sample_rate),
        )
        target = _stem_clean_signal(stem, frames, sample_rate, seed=20260624 + row_index)
        bad, corruption = corrupt_clean_stem(
            clean_stem=target,
            sample_rate=sample_rate,
            issue_type=issue_type,
            region_mask=region_mask,
        )
        mix = _synthetic_mix(target, sample_rate, seed=20260624 + row_index)
        mix_path = write_audio(audio_dir / f"m{row_index:03d}.flac", mix, sample_rate)
        bad_path = write_audio(audio_dir / f"b{row_index:03d}.flac", bad, sample_rate)
        target_path = write_audio(audio_dir / f"t{row_index:03d}.flac", target, sample_rate)
        row = {
            "schema": "stem_region_repair_row.v1",
            "row_id": f"sp{row_index:06d}",
            "source_hardcase_row_id": str(hardcase.get("row_id") or ""),
            "source_hardcase_bad_stem_path": str(hardcase.get("bad_stem_path") or ""),
            "stem": stem,
            "target_stem": stem,
            "issue_type": issue_type,
            "label_type": LABEL_TYPE_SYNTHETIC_CLEAN,
            "is_ground_truth": True,
            "usable_for_promotion": False,
            "mix_path": str(mix_path),
            "bad_stem_path": str(bad_path),
            "target_stem_path": str(target_path),
            "region": {"start_sec": 0.0, "end_sec": frames / float(sample_rate)},
            "corruption": corruption,
            "training_state": "synthetic_pretrain_ready",
            "delivery_allowed": False,
        }
        validate_region_repair_row(row)
        rows.append(row)

    issue_counts = Counter(str(row["issue_type"]) for row in rows)
    stem_counts = Counter(str(row["target_stem"]) for row in rows)
    rows_path = write_jsonl(output_dir / "rows.jsonl", rows)
    manifest = {
        "schema": "stem_region_synthetic_pretrain_dataset.v1",
        "hardcase_rows_jsonl": str(hardcase_rows_jsonl),
        "rows_jsonl": str(rows_path),
        "row_count": len(rows),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "synthetic_ground_truth_count": len(rows),
        "issue_type_counts": dict(sorted(issue_counts.items())),
        "stem_counts": dict(sorted(stem_counts.items())),
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "training_pretrain_only",
        "contract": {
            "synthetic_ground_truth": True,
            "hardcase_audio_not_self_labeled": True,
            "usable_for_promotion": False,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardcase-rows-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_stem_region_synthetic_pretrain_dataset(
        hardcase_rows_jsonl=args.hardcase_rows_jsonl,
        output_dir=args.out,
        max_rows=args.max_rows,
        duration_sec=args.duration_sec,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
