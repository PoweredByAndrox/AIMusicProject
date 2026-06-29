#!/usr/bin/env python3
"""Build a locked hardcase benchmark from training-first root-cause rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.stem_region_repair_common import validate_region_repair_row, write_json, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import validate_region_repair_row, write_json, write_jsonl


BLOCKED_MINING_LABEL = "blocked_candidate_mining"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable_split(row: Mapping[str, Any], index: int) -> str:
    if index == 1:
        return "locked_holdout"
    key = "|".join(
        [
            str(row.get("row_id") or ""),
            str(row.get("route_signature") or row.get("route_family") or ""),
            str(row.get("issue_type") or ""),
        ]
    )
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 4
    return "locked_holdout" if bucket == 0 else "train_mining"


def _benchmark_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    label_type = str(row.get("label_type") or BLOCKED_MINING_LABEL)
    if label_type == BLOCKED_MINING_LABEL and bool(row.get("is_ground_truth")):
        raise ValueError("blocked candidate cannot be ground truth")
    out = {
        "schema": "stem_repair_hardcase_row.v1",
        "row_id": f"hb{index:06d}",
        "source_queue_row_id": str(row.get("row_id") or ""),
        "source_run_json": str(row.get("source_run_json") or ""),
        "root_cause": str(row.get("root_cause") or ""),
        "issue_type": str(row.get("issue_type") or ""),
        "affected_stem": str(row.get("affected_stem") or ""),
        "route_signature": str(row.get("route_signature") or row.get("route_family") or ""),
        "bad_stem_path": str(row.get("bad_stem_path") or ""),
        "target_stem_path": str(row.get("target_stem_path") or ""),
        "mix_path": str(row.get("mix_path") or ""),
        "region": row.get("residual_region") or row.get("region") or {"start_sec": 0.0, "end_sec": 0.0},
        "split": _stable_split(row, index),
        "label_type": label_type,
        "is_ground_truth": False,
        "usable_for_promotion": False,
        "usable_for_training": False,
        "usable_for_mining": True,
    }
    if out["affected_stem"]:
        validate_region_repair_row(
            {
                "row_id": out["row_id"],
                "label_type": out["label_type"],
                "is_ground_truth": out["is_ground_truth"],
                "usable_for_promotion": out["usable_for_promotion"],
                "issue_type": out["issue_type"],
                "target_stem": out["affected_stem"],
                "region": out["region"],
            }
        )
    return out


def build_stem_repair_hardcase_benchmark(
    *,
    queue_jsonl: Path,
    source_run_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_rows = _read_jsonl(queue_jsonl)
    rows = [_benchmark_row(row, index=index) for index, row in enumerate(source_rows, start=1)]
    split_counts = Counter(str(row["split"]) for row in rows)
    root_counts = Counter(str(row["root_cause"]) for row in rows)
    issue_counts = Counter(str(row["issue_type"]) for row in rows)
    before_after_rows = [
        {
            "row_id": row["row_id"],
            "source_queue_row_id": row["source_queue_row_id"],
            "evaluated": False,
            "improved": False,
            "reason": "checkpoint_not_evaluated",
        }
        for row in rows
    ]

    rows_jsonl = output_dir / "rows.jsonl"
    splits_json = output_dir / "splits.json"
    before_after_jsonl = output_dir / "before_after.jsonl"
    write_jsonl(rows_jsonl, rows)
    write_jsonl(before_after_jsonl, before_after_rows)
    write_json(
        splits_json,
        {
            "schema": "stem_repair_hardcase_splits.v1",
            "split_counts": dict(sorted(split_counts.items())),
            "locked_split_policy": "hash_by_queue_row_route_issue",
        },
    )
    manifest = {
        "schema": "stem_repair_hardcase_benchmark.v1",
        "source_queue_jsonl": str(queue_jsonl),
        "source_run_json": str(source_run_json),
        "row_count": len(rows),
        "rows_jsonl": str(rows_jsonl),
        "splits_json": str(splits_json),
        "before_after_jsonl": str(before_after_jsonl),
        "unsafe_promotion_row_count": sum(1 for row in rows if row.get("usable_for_promotion") or row.get("is_ground_truth")),
        "root_cause_counts": dict(sorted(root_counts.items())),
        "issue_type_counts": dict(sorted(issue_counts.items())),
        "contract": {
            "delivery_allowed": False,
            "presentation_status": "training_benchmark_only",
            "blocked_candidates_are_ground_truth": False,
            "locked_holdout_required": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--source-run-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_stem_repair_hardcase_benchmark(
        queue_jsonl=args.queue_jsonl,
        source_run_json=args.source_run_json,
        output_dir=args.out,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
