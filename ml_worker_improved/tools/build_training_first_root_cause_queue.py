#!/usr/bin/env python3
"""Build a training-first root-cause/mining queue from blocked stem-search candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.stem_contamination_common import normalize_stem_name
    from tools.stem_region_repair_common import (
        ISSUE_BAD_RECONSTRUCTION,
        ISSUE_INFRA,
        ISSUE_INSTRUMENT_LEAK,
        ISSUE_MISSING_DETAIL,
        ISSUE_PHASE_DAMAGE,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_THIN_BODY,
        ISSUE_TRANSIENT_LOSS,
        ISSUE_UNKNOWN,
        ISSUE_VOCAL_LEAK,
        ROOT_CAUSE_INFRA_GAP,
        ROOT_CAUSE_MODEL_GAP,
        ROOT_CAUSE_REPAIR_GAP,
        ROOT_CAUSE_ROUTING_GAP,
        validate_region_repair_row,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_contamination_common import normalize_stem_name
    from stem_region_repair_common import (
        ISSUE_BAD_RECONSTRUCTION,
        ISSUE_INFRA,
        ISSUE_INSTRUMENT_LEAK,
        ISSUE_MISSING_DETAIL,
        ISSUE_PHASE_DAMAGE,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_THIN_BODY,
        ISSUE_TRANSIENT_LOSS,
        ISSUE_UNKNOWN,
        ISSUE_VOCAL_LEAK,
        ROOT_CAUSE_INFRA_GAP,
        ROOT_CAUSE_MODEL_GAP,
        ROOT_CAUSE_REPAIR_GAP,
        ROOT_CAUSE_ROUTING_GAP,
        validate_region_repair_row,
        write_json,
        write_jsonl,
    )


FOREIGN_RECOVERED_RE = re.compile(r"foreign_source_recovered:([^:,]+):([^,]+)")
STEREO_COLLAPSE_RE = re.compile(r"stereo_width_collapse:([^,]+)")


def _candidate_reason(candidate: Mapping[str, Any]) -> str:
    next_action = candidate.get("next_action") if isinstance(candidate.get("next_action"), Mapping) else {}
    reason = str(next_action.get("reason") or "").strip()
    if reason:
        return reason
    if candidate.get("error"):
        return "failed_runtime_error"
    reconstruction = candidate.get("reconstruction") if isinstance(candidate.get("reconstruction"), Mapping) else {}
    pair = reconstruction.get("delivery_pair") if isinstance(reconstruction.get("delivery_pair"), Mapping) else {}
    blockers = pair.get("blockers") if isinstance(pair.get("blockers"), list) else []
    if blockers:
        return ",".join(str(blocker) for blocker in blockers)
    return str(candidate.get("verdict") or "unknown")


def _route_signature(candidate: Mapping[str, Any]) -> str:
    composition = candidate.get("target_specific_composition")
    if not isinstance(composition, Mapping):
        return ""
    routes = composition.get("routes")
    if not isinstance(routes, list):
        return ""
    parts = []
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        lane = str(route.get("lane") or "").strip()
        route_id = str(route.get("route_id") or route.get("route_alias") or "").strip()
        if lane or route_id:
            parts.append(f"{lane}:{route_id}")
    return "|".join(parts)


def classify_blocker(candidate: Mapping[str, Any]) -> dict[str, Any]:
    reason = _candidate_reason(candidate)
    error = candidate.get("error") if isinstance(candidate.get("error"), Mapping) else {}
    if candidate.get("execution_status") == "failed_runtime_error" or error:
        return {
            "root_cause": ROOT_CAUSE_INFRA_GAP,
            "issue_type": ISSUE_INFRA,
            "affected_stem": "",
            "foreign_source": "",
            "training_action": "fix_or_disable_infra_route",
            "blocker_reason": reason,
        }

    foreign_match = FOREIGN_RECOVERED_RE.search(reason)
    if foreign_match:
        affected_stem = normalize_stem_name(foreign_match.group(1))
        foreign_source = normalize_stem_name(foreign_match.group(2))
        issue_type = ISSUE_VOCAL_LEAK if foreign_source == "vocals" else ISSUE_INSTRUMENT_LEAK
        return {
            "root_cause": ROOT_CAUSE_REPAIR_GAP,
            "issue_type": issue_type,
            "affected_stem": affected_stem,
            "foreign_source": foreign_source,
            "training_action": "mine_region_repair_example",
            "blocker_reason": reason,
        }

    stereo_match = STEREO_COLLAPSE_RE.search(reason)
    if stereo_match:
        return {
            "root_cause": ROOT_CAUSE_REPAIR_GAP,
            "issue_type": ISSUE_STEREO_COLLAPSE,
            "affected_stem": normalize_stem_name(stereo_match.group(1)),
            "foreign_source": "",
            "training_action": "mine_stereo_repair_example",
            "blocker_reason": reason,
        }

    lowered = reason.lower()
    if "transient" in lowered:
        issue_type = ISSUE_TRANSIENT_LOSS
        root_cause = ROOT_CAUSE_REPAIR_GAP
    elif "phase" in lowered:
        issue_type = ISSUE_PHASE_DAMAGE
        root_cause = ROOT_CAUSE_REPAIR_GAP
    elif "thin" in lowered:
        issue_type = ISSUE_THIN_BODY
        root_cause = ROOT_CAUSE_REPAIR_GAP
    elif "missing" in lowered or "detail" in lowered:
        issue_type = ISSUE_MISSING_DETAIL
        root_cause = ROOT_CAUSE_REPAIR_GAP
    elif "mix_conservation" in lowered or "reconstruction" in lowered:
        issue_type = ISSUE_BAD_RECONSTRUCTION
        root_cause = ROOT_CAUSE_ROUTING_GAP
    else:
        issue_type = ISSUE_UNKNOWN
        root_cause = ROOT_CAUSE_MODEL_GAP

    return {
        "root_cause": root_cause,
        "issue_type": issue_type,
        "affected_stem": "",
        "foreign_source": "",
        "training_action": "mine_model_gap_example" if root_cause == ROOT_CAUSE_MODEL_GAP else "mine_region_repair_example",
        "blocker_reason": reason,
    }


def _queue_signature(candidate: Mapping[str, Any], classification: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(candidate.get("window_id") or ""),
        _route_signature(candidate),
        str(classification.get("root_cause") or ""),
        str(classification.get("issue_type") or ""),
        str(classification.get("blocker_reason") or ""),
    )


def build_root_cause_queue(
    *,
    run_json_path: Path,
    output_dir: Path,
    min_repeats: int = 2,
) -> dict[str, Any]:
    payload = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("run_json must contain candidates list")

    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    classifications: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    blocked_candidate_count = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("verdict") or "").strip() == "auto_accept":
            continue
        blocked_candidate_count += 1
        classification = classify_blocker(candidate)
        signature = _queue_signature(candidate, classification)
        grouped[signature].append(candidate)
        classifications[signature] = classification

    rows: list[dict[str, Any]] = []
    for index, (signature, group) in enumerate(sorted(grouped.items(), key=lambda item: item[0]), start=1):
        classification = classifications[signature]
        root_cause = str(classification["root_cause"])
        repeat_count = len(group)
        if root_cause != ROOT_CAUSE_INFRA_GAP and repeat_count < int(min_repeats):
            continue
        first = group[0]
        row = {
            "schema": "training_first_root_cause_sample.v1",
            "row_id": f"rc{index:06d}",
            "root_cause": root_cause,
            "issue_type": classification["issue_type"],
            "affected_stem": classification["affected_stem"],
            "foreign_source": classification["foreign_source"],
            "training_action": classification["training_action"],
            "blocker_reason": classification["blocker_reason"],
            "repeat_count": repeat_count,
            "candidate_ids": [str(candidate.get("candidate_id") or "") for candidate in group],
            "window_ids": sorted({str(candidate.get("window_id") or "") for candidate in group}),
            "route_signature": signature[1],
            "source_run_json": str(run_json_path),
            "score_path": str(first.get("score_path") or ""),
            "bad_stem_path": _bad_stem_path(first, str(classification["affected_stem"])),
            "label_type": "blocked_candidate_mining",
            "is_ground_truth": False,
            "usable_for_promotion": False,
        }
        rows.append(row)

    root_counts = Counter(str(row["root_cause"]) for row in rows)
    issue_counts = Counter(str(row["issue_type"]) for row in rows)
    manifest = {
        "schema": "training_first_root_cause_queue.v1",
        "source_run_json": str(run_json_path),
        "candidate_count": len(candidates),
        "blocked_candidate_count": blocked_candidate_count,
        "queue_row_count": len(rows),
        "min_repeats": int(min_repeats),
        "root_cause_counts": dict(sorted(root_counts.items())),
        "issue_type_counts": dict(sorted(issue_counts.items())),
        "queue_jsonl": str(output_dir / "queue.jsonl"),
        "contract": {
            "delivery_allowed": False,
            "presentation_status": "training_mining_only",
            "rejected_candidates_are_not_results": True,
        },
    }
    write_jsonl(output_dir / "queue.jsonl", rows)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _bad_stem_path(candidate: Mapping[str, Any], affected_stem: str) -> str:
    stem_paths = candidate.get("stem_paths")
    if not isinstance(stem_paths, Mapping) or not affected_stem:
        return ""
    for stem, path in stem_paths.items():
        if normalize_stem_name(stem) == affected_stem:
            return str(path)
    return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-repeats", type=int, default=2)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_root_cause_queue(
        run_json_path=args.run_json,
        output_dir=args.output_dir,
        min_repeats=args.min_repeats,
    )
    # Validate only rows that can become repair rows later; root-cause rows are mining rows.
    for row_line in Path(manifest["queue_jsonl"]).read_text(encoding="utf-8").splitlines():
        row = json.loads(row_line)
        if row["root_cause"] == ROOT_CAUSE_REPAIR_GAP and row["affected_stem"]:
            validate_region_repair_row(
                {
                    "row_id": row["row_id"],
                    "label_type": "blocked_candidate_mining",
                    "is_ground_truth": False,
                    "issue_type": row["issue_type"],
                    "target_stem": row["affected_stem"],
                    "region": {"start_sec": 0.0, "end_sec": 0.0},
                }
            )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
