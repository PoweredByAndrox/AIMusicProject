#!/usr/bin/env python3
"""Convert a blocked Mac readiness handoff into a full-pack loop queue."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
QUEUE_NAME = "queue.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
ROUTE_INTENT = "mac_promising_blocked_handoff"
READY_QUEUE_STATUS = "ready_for_full_pack_loop"
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
    "delivery_allowed",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
    "separator_training_started",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    return path


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_handoff_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "mac_promising_handoff_planner_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "mac_promising_handoff_planner_no_training_started",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[str]) -> list[str]:
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
    for nested_key in ("safety", "contract"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            for key in BLOCKED_TRUE_FLAGS:
                if nested.get(key) is True:
                    failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def route_status_counts(routes: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        status = str(route.get("route_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def required_actions(routes: Sequence[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for route in routes:
        next_action = str(route.get("next_action") or "")
        if str(route.get("route_status") or "") == "missing_implementation":
            route_name = str(route.get("route") or "route")
            actions.append(f"design_{route_name}_route")
        elif next_action:
            actions.append(next_action)
    actions.extend(["minus_audit_gate", "holdout_eval_gate", "train_ready_decision_gate", "full_pack_gate_rebuild"])
    return stable_unique(actions)


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Mac Promising Blocked Handoff",
        "",
        f"- queue_status: `{manifest['queue_status']}`",
        f"- queue_item_count: `{manifest['queue_item_count']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
        "## Items",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['queue_item_id']}` `{row['route_intent']}` `{','.join(row['issue_categories'])}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_row(summary: dict[str, Any], handoff: dict[str, Any], readiness_summary_json: Path) -> dict[str, Any]:
    routes = [dict(route) for route in handoff.get("recommended_routes") or [] if isinstance(route, dict)]
    return {
        "schema": "short_listen_repair_queue_item.v1",
        "queue_item_id": "mph001",
        "route_intent": ROUTE_INTENT,
        "stem_type": "minus",
        "audio_path": str(summary.get("source_audio_path") or ""),
        "source_audio_path": str(summary.get("source_audio_path") or ""),
        "readiness_summary_json": str(readiness_summary_json),
        "source_manifest": str(summary.get("manifest") or ""),
        "source_schema": str(summary.get("source_schema") or ""),
        "source_status": str(summary.get("source_status") or ""),
        "source_track_identity": dict(summary.get("source_track_identity") or {}) if isinstance(summary.get("source_track_identity"), dict) else {},
        "issue_tags": [str(tag) for tag in handoff.get("issue_tags") or [] if str(tag)],
        "issue_categories": [str(category) for category in handoff.get("issue_categories") or [] if str(category)],
        "objective_summary": dict(summary.get("objective_summary") or {}) if isinstance(summary.get("objective_summary"), dict) else {},
        "readiness_blocker_reasons": [str(reason) for reason in summary.get("blocker_reasons") or [] if str(reason)],
        "recommended_routes": routes,
        "route_status_counts": route_status_counts(routes),
        "required_actions": required_actions(routes),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
    }


def build_mac_promising_blocked_handoff_queue(
    *,
    mac_readiness_summary_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", mac_readiness_summary_json=str(mac_readiness_summary_json))

    summary = read_json(mac_readiness_summary_json)
    failed = collect_true_flag_failures("readiness_summary", summary)
    if summary.get("schema") != "mac_candidate_readiness_summary.v1":
        failed.append("readiness_summary_schema_unexpected")
    if summary.get("delivery_allowed") is not False:
        failed.append("readiness_summary_delivery_allowed_not_false")
    if summary.get("listen_result_allowed") is not False:
        failed.append("readiness_summary_listen_result_allowed_not_false")
    handoff = summary.get("promising_blocked_handoff") if isinstance(summary.get("promising_blocked_handoff"), dict) else {}
    failed.extend(collect_true_flag_failures("promising_blocked_handoff", dict(handoff)))
    if handoff.get("status") != "promising_but_blocked":
        failed.append("promising_blocked_handoff_status_unexpected")
    if handoff.get("delivery_allowed") is not False:
        failed.append("promising_blocked_handoff_delivery_allowed_not_false")
    if str(handoff.get("presentation_status") or "") != "diagnostic_only_handoff":
        failed.append("promising_blocked_handoff_presentation_status_unexpected")
    routes = [route for route in handoff.get("recommended_routes") or [] if isinstance(route, dict)]
    if not routes:
        failed.append("promising_blocked_handoff_recommended_routes_missing")

    failed = stable_unique(failed)
    rows = [] if failed else [build_row(summary, dict(handoff), mac_readiness_summary_json)]
    manifest = {
        "schema": "short_listen_repair_queue_manifest.v1",
        "queue_status": READY_QUEUE_STATUS if not failed else "blocked",
        "queue_source": "mac_promising_blocked_handoff",
        "queue_item_count": len(rows),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "mac_readiness_summary_json": str(mac_readiness_summary_json),
        "route_status_counts": route_status_counts(routes) if not failed else {},
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "mac_promising_handoff_queue_not_for_listen_pack",
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "not_release_artifact": True,
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / QUEUE_NAME, manifest)
    write_jsonl(output_dir / ROWS_NAME, rows)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", queue_status=manifest["queue_status"], queue_item_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-readiness-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="plan_mac_promising_blocked_handoff.py",
        entrypoint_path="/app/tools/plan_mac_promising_blocked_handoff.py",
    )
    args = parse_args(argv)
    manifest = build_mac_promising_blocked_handoff_queue(
        mac_readiness_summary_json=args.mac_readiness_summary_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / QUEUE_NAME),
                "queue_status": manifest["queue_status"],
                "queue_item_count": manifest["queue_item_count"],
                "failed_checks": manifest["failed_checks"],
                "delivery_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
