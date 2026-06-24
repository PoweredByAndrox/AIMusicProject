#!/usr/bin/env python3
"""Apply a stem-region repair checkpoint to hard-case rows as diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.stem_region_repair_common import (
        ISSUE_STEREO_COLLAPSE,
        make_region_mask,
        read_audio_stereo,
        validate_issue_type,
        write_audio,
        write_json,
        write_jsonl,
    )
    from tools.stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import (
        ISSUE_STEREO_COLLAPSE,
        make_region_mask,
        read_audio_stereo,
        validate_issue_type,
        write_audio,
        write_json,
        write_jsonl,
    )
    from stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint


EPSILON = 1.0e-8


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _infer_window_mix_path(row: dict[str, Any]) -> str:
    explicit = str(row.get("mix_path") or "")
    if explicit and Path(explicit).exists():
        return explicit
    for key in ("source_bad_stem_path", "bad_stem_path"):
        source = str(row.get(key) or "")
        if not source:
            continue
        parts = Path(source).parts
        for index, part in enumerate(parts):
            if part.startswith("w") and part[1:].isdigit():
                candidate = Path(*parts[: index + 1]) / "mix.flac"
                if candidate.exists():
                    return str(candidate)
    return ""


def _stereo_width(audio: np.ndarray) -> float:
    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    mid_rms = float(np.sqrt(np.mean(np.square(mid))) + EPSILON)
    side_rms = float(np.sqrt(np.mean(np.square(side))))
    return side_rms / mid_rms


def _issue_diagnostics(issue_type: str, before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    if issue_type == ISSUE_STEREO_COLLAPSE:
        before_width = _stereo_width(before)
        after_width = _stereo_width(after)
        return {
            "stereo_width_before": before_width,
            "stereo_width_after": after_width,
            "target_issue_improved": after_width > before_width,
        }
    return {"target_issue_improved": False, "unsupported_issue_metric": issue_type}


def apply_stem_region_repair_pack(
    *,
    checkpoint_path: Path,
    hardcase_rows_jsonl: Path,
    output_dir: Path,
    max_rows: int = 8,
) -> dict[str, Any]:
    model, checkpoint = load_region_repair_checkpoint(checkpoint_path)
    checkpoint_issue = validate_issue_type((checkpoint.get("training_history") or {}).get("issue_type"))
    hardcase_rows = _read_jsonl(hardcase_rows_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"

    rows_out: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in hardcase_rows:
        if len(rows_out) >= int(max_rows):
            break
        issue_type = validate_issue_type(row.get("issue_type"))
        if issue_type != checkpoint_issue:
            skipped.append({"row_id": str(row.get("row_id") or ""), "reason": "issue_mismatch"})
            continue
        bad_path = str(row.get("bad_stem_path") or "")
        mix_path = _infer_window_mix_path(row)
        if not bad_path or not Path(bad_path).exists():
            skipped.append({"row_id": str(row.get("row_id") or ""), "reason": "missing_bad_stem"})
            continue
        if not mix_path:
            skipped.append({"row_id": str(row.get("row_id") or ""), "reason": "missing_mix"})
            continue
        bad, sample_rate = read_audio_stereo(bad_path)
        mix, mix_rate = read_audio_stereo(mix_path)
        if mix_rate != sample_rate:
            skipped.append({"row_id": str(row.get("row_id") or ""), "reason": "sample_rate_mismatch"})
            continue
        frames = min(bad.shape[0], mix.shape[0])
        bad = bad[:frames]
        mix = mix[:frames]
        mask = make_region_mask(frame_count=frames, sample_rate=sample_rate, start_sec=0.0, end_sec=frames / float(sample_rate))
        repaired, apply_diagnostics = apply_region_repair_model(
            model=model,
            mix_audio=mix,
            bad_stem_audio=bad,
            region_mask=mask,
            sample_rate=sample_rate,
            issue_type=issue_type,
        )
        index = len(rows_out)
        repaired_path = write_audio(audio_dir / f"r{index:03d}.flac", repaired, sample_rate)
        issue_diagnostics = _issue_diagnostics(issue_type, bad, repaired)
        rows_out.append(
            {
                "schema": "stem_region_repair_apply_pack_row.v1",
                "row_id": f"ap{index:06d}",
                "source_hardcase_row_id": str(row.get("row_id") or ""),
                "issue_type": issue_type,
                "affected_stem": str(row.get("affected_stem") or ""),
                "bad_stem_path": bad_path,
                "mix_path": mix_path,
                "repaired_stem_path": str(repaired_path),
                "checkpoint": str(checkpoint_path),
                "diagnostics": {**apply_diagnostics, **issue_diagnostics},
                "delivery_status": "diagnostic_only",
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
            }
        )

    issue_counts = Counter(str(row["issue_type"]) for row in rows_out)
    improved_count = sum(1 for row in rows_out if bool((row.get("diagnostics") or {}).get("target_issue_improved")))
    rows_path = write_jsonl(output_dir / "rows.jsonl", rows_out)
    manifest = {
        "schema": "stem_region_repair_apply_pack.v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_issue_type": checkpoint_issue,
        "hardcase_rows_jsonl": str(hardcase_rows_jsonl),
        "rows_jsonl": str(rows_path),
        "row_count": len(rows_out),
        "improved_count": improved_count,
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "issue_type_counts": dict(sorted(issue_counts.items())),
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "contract": {
            "diagnostic_only": True,
            "rejected_candidates_are_not_results": True,
            "no_delivery_from_repair_apply": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hardcase-rows-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=8)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = apply_stem_region_repair_pack(
        checkpoint_path=args.checkpoint,
        hardcase_rows_jsonl=args.hardcase_rows_jsonl,
        output_dir=args.out,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
