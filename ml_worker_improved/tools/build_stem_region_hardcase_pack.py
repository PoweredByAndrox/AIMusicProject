#!/usr/bin/env python3
"""Build a short hard-case pack from training-first root-cause queue rows."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from tools.stem_contamination_common import normalize_stem_name
    from tools.stem_region_repair_common import (
        ISSUE_TYPES,
        ROOT_CAUSE_REPAIR_GAP,
        validate_issue_type,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_contamination_common import normalize_stem_name
    from stem_region_repair_common import (
        ISSUE_TYPES,
        ROOT_CAUSE_REPAIR_GAP,
        validate_issue_type,
        write_json,
        write_jsonl,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _copy_optional(src: str, dst: Path) -> str:
    if not src:
        return ""
    source = Path(src)
    if not source.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    return str(dst)


def _wanted_issue_types(issue_types: Iterable[str] | None) -> set[str]:
    if not issue_types:
        return set(ISSUE_TYPES)
    return {validate_issue_type(issue_type) for issue_type in issue_types}


def build_stem_region_hardcase_pack(
    *,
    queue_jsonl: Path,
    output_dir: Path,
    issue_types: list[str] | None = None,
    max_rows: int = 32,
) -> dict[str, Any]:
    wanted = _wanted_issue_types(issue_types)
    rows_in = _read_jsonl(queue_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    score_dir = output_dir / "score"

    rows_out: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in rows_in:
        if len(rows_out) >= int(max_rows):
            break
        issue_type = validate_issue_type(row.get("issue_type"))
        if issue_type not in wanted:
            continue
        if str(row.get("root_cause") or "") != ROOT_CAUSE_REPAIR_GAP:
            continue
        bad_stem_path = str(row.get("bad_stem_path") or "")
        if not bad_stem_path or not Path(bad_stem_path).exists():
            skipped.append({"row_id": str(row.get("row_id") or ""), "reason": "missing_bad_stem"})
            continue
        index = len(rows_out)
        bad_alias = audio_dir / f"b{index:03d}.flac"
        score_alias = score_dir / f"s{index:03d}.json"
        copied_bad = _copy_optional(bad_stem_path, bad_alias)
        copied_score = _copy_optional(str(row.get("score_path") or ""), score_alias)
        target_path = str(row.get("target_stem_path") or "")
        training_state = "supervised_ready" if target_path and Path(target_path).exists() else "data_gap"
        rows_out.append(
            {
                "schema": "stem_region_hardcase_row.v1",
                "row_id": f"hc{index:06d}",
                "source_queue_row_id": str(row.get("row_id") or ""),
                "root_cause": str(row.get("root_cause") or ""),
                "issue_type": issue_type,
                "affected_stem": normalize_stem_name(row.get("affected_stem") or row.get("target_stem") or ""),
                "foreign_source": normalize_stem_name(row.get("foreign_source") or ""),
                "repeat_count": int(row.get("repeat_count") or 0),
                "bad_stem_path": copied_bad,
                "target_stem_path": target_path if training_state == "supervised_ready" else "",
                "score_path": copied_score,
                "blocker_reason": str(row.get("blocker_reason") or ""),
                "source_run_json": str(row.get("source_run_json") or ""),
                "source_bad_stem_path": bad_stem_path,
                "source_score_path": str(row.get("score_path") or ""),
                "training_state": training_state,
                "training_action": "attach_verified_target" if training_state == "data_gap" else "train_region_repair",
                "is_ground_truth": False,
                "usable_for_promotion": False,
                "delivery_allowed": False,
            }
        )

    state_counts = Counter(str(row["training_state"]) for row in rows_out)
    issue_counts = Counter(str(row["issue_type"]) for row in rows_out)
    rows_path = write_jsonl(output_dir / "rows.jsonl", rows_out)
    manifest = {
        "schema": "stem_region_hardcase_pack.v1",
        "queue_jsonl": str(queue_jsonl),
        "rows_jsonl": str(rows_path),
        "output_dir": str(output_dir),
        "row_count": len(rows_out),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "data_gap_count": int(state_counts.get("data_gap", 0)),
        "supervised_ready_count": int(state_counts.get("supervised_ready", 0)),
        "issue_type_counts": dict(sorted(issue_counts.items())),
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "training_mining_only",
        "contract": {
            "short_alias_paths": True,
            "rejected_candidates_are_not_results": True,
            "requires_verified_target_before_training": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--issue-type", action="append", dest="issue_types")
    parser.add_argument("--max-rows", type=int, default=32)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_stem_region_hardcase_pack(
        queue_jsonl=args.queue_jsonl,
        output_dir=args.out,
        issue_types=args.issue_types,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
