#!/usr/bin/env python3
"""Run feedback manifest, goal audit, and repair queue from one Docker command."""

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

from tools.audit_short_listen_goal_status import audit_short_listen_goal_status  # noqa: E402
from tools.build_short_listen_feedback_manifest import (  # noqa: E402
    MANIFEST_NAME as FEEDBACK_MANIFEST_NAME,
    build_short_listen_feedback_manifest,
)
from tools.build_short_listen_repair_queue import (  # noqa: E402
    MANIFEST_NAME as REPAIR_MANIFEST_NAME,
    ROWS_NAME as REPAIR_ROWS_NAME,
    build_short_listen_repair_queue,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402

START_TIME = time.monotonic()
MANIFEST_NAME = "cycle.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected JSON object rows")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_feedback_cycle_orchestration",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_feedback_cycle_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_feedback_cycle_no_training",
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
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def stage_failed(stage_name: str, payload: dict[str, Any]) -> list[str]:
    failed = [str(item) for item in payload.get("failed_checks") or [] if str(item)]
    return [f"{stage_name}:{item}" for item in failed]


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# short listen feedback cycle",
        "",
        f"- cycle_status: `{manifest['cycle_status']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Stages",
        "",
    ]
    for name, payload in manifest["stages"].items():
        status = (
            payload.get("overall_listen_verdict")
            or payload.get("goal_status")
            or payload.get("queue_status")
            or "unknown"
        )
        lines.append(f"- `{name}`: `{status}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def run_short_listen_feedback_cycle(
    *,
    pack_manifest_json: Path,
    aggregate_validation_json: Path,
    feedback_jsonl: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", pack_manifest_json=str(pack_manifest_json), feedback_jsonl=str(feedback_jsonl))

    stages: dict[str, dict[str, Any]] = {}
    failed: list[str] = []

    feedback_dir = output_dir / "feedback"
    feedback = build_short_listen_feedback_manifest(
        pack_manifest_json=pack_manifest_json,
        output_dir=feedback_dir,
        feedback_items=read_jsonl(feedback_jsonl),
        command_args=[
            "--pack-manifest-json",
            str(pack_manifest_json),
            "--feedback-jsonl",
            str(feedback_jsonl),
            "--output-dir",
            str(feedback_dir),
        ],
    )
    stages["feedback_manifest"] = {
        "overall_listen_verdict": feedback.get("overall_listen_verdict"),
        "feedback_item_count": feedback.get("feedback_item_count"),
        "review_coverage_status": feedback.get("review_coverage_status"),
        "failed_checks": feedback.get("failed_checks") or [],
        "manifest_json": str(feedback_dir / FEEDBACK_MANIFEST_NAME),
    }
    failed.extend(stage_failed("feedback_manifest", feedback))

    if not failed:
        audit_dir = output_dir / "goal_audit"
        audit = audit_short_listen_goal_status(
            pack_manifest_json=pack_manifest_json,
            validation_json=aggregate_validation_json,
            output_dir=audit_dir,
            human_listen_verdict_json=feedback_dir / FEEDBACK_MANIFEST_NAME,
        )
        stages["goal_audit"] = {
            "goal_status": audit.get("goal_status"),
            "goal_complete": audit.get("goal_complete"),
            "remaining_blockers": audit.get("remaining_blockers") or [],
            "failed_checks": [],
            "audit_json": str(audit_dir / "goal_status_audit.json"),
        }
        if audit.get("goal_status") == "failed_safety_contract":
            failed.append("goal_audit:failed_safety_contract")

    if not failed and stages.get("goal_audit", {}).get("goal_status") == "human_listen_rejected":
        repair_dir = output_dir / "repair_queue"
        repair = build_short_listen_repair_queue(
            feedback_manifest_json=feedback_dir / FEEDBACK_MANIFEST_NAME,
            output_dir=repair_dir,
            command_args=[
                "--feedback-manifest-json",
                str(feedback_dir / FEEDBACK_MANIFEST_NAME),
                "--output-dir",
                str(repair_dir),
            ],
        )
        stages["repair_queue"] = {
            "queue_status": repair.get("queue_status"),
            "queue_item_count": repair.get("queue_item_count"),
            "route_intent_counts": repair.get("route_intent_counts") or {},
            "failed_checks": repair.get("failed_checks") or [],
            "manifest_json": str(repair_dir / REPAIR_MANIFEST_NAME),
            "rows_jsonl": str(repair_dir / REPAIR_ROWS_NAME),
        }
        failed.extend(stage_failed("repair_queue", repair))

    failed = stable_unique(failed)
    goal_status = stages.get("goal_audit", {}).get("goal_status")
    if failed:
        cycle_status = "failed_safety_contract"
    elif goal_status == "human_listen_rejected":
        cycle_status = "repair_queue_ready"
    elif goal_status == "machine_and_human_accepted":
        cycle_status = "human_accepted_no_repair_needed"
    else:
        cycle_status = str(goal_status or "feedback_recorded")

    manifest = {
        "schema": "short_listen_feedback_cycle_manifest.v1",
        "cycle_status": cycle_status,
        "pack_manifest_json": str(pack_manifest_json),
        "aggregate_validation_json": str(aggregate_validation_json),
        "feedback_jsonl": str(feedback_jsonl),
        "output_dir": str(output_dir),
        "stages": stages,
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "feedback_cycle_not_for_listen_pack",
        "progress_jsonl": str(progress_path),
        "outputs": {
            "feedback_cycle_manifest_json": str(output_dir / MANIFEST_NAME),
            "feedback_cycle_summary_md": str(output_dir / SUMMARY_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
        "not_claimed": ["release_ready", "publishable_result", "audio_repaired", "training_started"],
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", cycle_status=cycle_status, failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short listen feedback manifest, audit, and repair queue from one command.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--aggregate-validation-json", type=Path, required=True)
    parser.add_argument("--feedback-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_short_listen_feedback_cycle.py",
        entrypoint_path="/app/tools/run_short_listen_feedback_cycle.py",
    )
    args = parse_args(argv)
    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=args.pack_manifest_json,
        aggregate_validation_json=args.aggregate_validation_json,
        feedback_jsonl=args.feedback_jsonl,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "cycle_status": manifest["cycle_status"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
