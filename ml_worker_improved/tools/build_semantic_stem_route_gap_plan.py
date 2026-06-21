#!/usr/bin/env python3
"""Build a route-gap plan from semantic stem review blockers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "gap.json"
ROWS_NAME = "gaps.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "separator_training_started",
    "cleanup_routes_added",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_route_gap_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "semantic_route_gap_plan_reads_review_artifacts_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "semantic_route_gap_plan_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    return failed


def safety_flags() -> dict[str, bool]:
    return {
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "release_ready": False,
    }


def recommended_action(stem_type: str, blockers: Sequence[str]) -> str:
    blocker_set = set(blockers)
    if stem_type == "vocal":
        return "add_or_run_vocal_semantic_quality_gate"
    if stem_type == "minus":
        return "add_or_run_minus_bleed_phase_quality_gate"
    if stem_type in {"piano", "guitar"}:
        return "add_or_run_instrument_semantic_proxy_gate"
    if "unsupported_stem_type" in blocker_set:
        return "define_supported_semantic_gate_for_stem"
    return "generate_or_select_better_stem_candidate"


def row_priority(stem_type: str) -> str:
    if stem_type in {"vocal", "minus"}:
        return "P0"
    if stem_type in {"piano", "guitar"}:
        return "P1"
    return "P2"


def build_gap_rows(review_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        if row.get("semantic_quality_verified") is True or row.get("review_status") == "accepted":
            continue
        grouped[str(row.get("stem_type") or "unknown")].append(row)
    gap_rows: list[dict[str, Any]] = []
    for index, (stem_type, rows) in enumerate(sorted(grouped.items()), start=1):
        blockers = stable_unique(
            blocker
            for row in rows
            for blocker in (row.get("blockers") or ["semantic_quality_not_verified"])
        )
        gap_rows.append(
            {
                "schema": "semantic_stem_route_gap.v1",
                "gap_id": f"g{index:03d}",
                "stem_type": stem_type,
                "blocked_count": len(rows),
                "blocked_group_aliases": sorted({str(row.get("group_alias") or "") for row in rows if str(row.get("group_alias") or "")}),
                "blockers": blockers,
                "recommended_action": recommended_action(stem_type, blockers),
                "priority": row_priority(stem_type),
                "next_wave_required": stem_type in {"vocal", "minus", "piano", "guitar"},
                "training_started": False,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    gap_rows.sort(key=lambda row: (row["priority"], row["stem_type"]))
    return gap_rows


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# semantic stem route gaps",
        "",
        f"- gap_status: `{manifest['gap_status']}`",
        f"- gap_count: `{manifest['gap_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Gaps",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['gap_id']}` `{row['stem_type']}` -> `{row['recommended_action']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    return "\n".join(lines) + "\n"


def build_semantic_stem_route_gap_plan(
    *,
    semantic_review_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", semantic_review_json=str(semantic_review_json))

    review = read_json(semantic_review_json)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("semantic_review", review))
    if review.get("schema") != "semantic_stem_quality_review.v1":
        failed.append("semantic_review_schema_unexpected")
    if review.get("review_status") != "completed":
        failed.append("semantic_review_not_completed")
    if int(review.get("pending_item_count") or 0) > 0:
        failed.append("semantic_review_pending_items")
    rows_jsonl = Path(str(review.get("rows_jsonl") or ""))
    if not rows_jsonl.exists():
        failed.append("semantic_review_rows_missing")
        review_rows: list[dict[str, Any]] = []
    else:
        review_rows = list(iter_jsonl(rows_jsonl))
    failed = stable_unique(failed)
    gap_rows = [] if failed else build_gap_rows(review_rows)
    status = "route_gaps_ready" if gap_rows and not failed else ("no_route_gaps" if not failed else "blocked")
    manifest = {
        "schema": "semantic_stem_route_gap_plan.v1",
        "gap_status": status,
        "semantic_review_json": str(semantic_review_json),
        "review_status": str(review.get("review_status") or ""),
        "review_total_item_count": int(review.get("total_item_count") or 0),
        "review_verified_item_count": int(review.get("semantic_quality_verified_item_count") or 0),
        "gap_count": len(gap_rows),
        "failed_checks": failed,
        "gaps_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "semantic_route_gap_plan_not_listen_result",
        "safety": safety_flags(),
        "not_claimed": ["release_ready", "publishable_result", "route_implemented", "training_started"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, gap_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, gap_rows), encoding="utf-8")
    progress(progress_path, "finish", gap_status=status, gap_count=len(gap_rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a semantic stem route-gap plan from review blockers.")
    parser.add_argument("--semantic-review-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_semantic_stem_route_gap_plan.py",
        entrypoint_path="/app/tools/build_semantic_stem_route_gap_plan.py",
    )
    args = parse_args(argv)
    manifest = build_semantic_stem_route_gap_plan(
        semantic_review_json=args.semantic_review_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "gap_status": manifest["gap_status"],
                "gap_count": manifest["gap_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["gap_status"] in {"route_gaps_ready", "no_route_gaps"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
