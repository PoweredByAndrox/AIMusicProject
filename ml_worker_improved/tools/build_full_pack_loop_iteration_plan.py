#!/usr/bin/env python3
"""Build a machine-only ARIS iteration plan from a blocked full-pack queue."""

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
PLAN_NAME = "plan.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
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
    "cleanup_routes_added",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
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
        "expected_acceleration_mode": "external_only_cpu_loop_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "full_pack_loop_iteration_plan_reads_existing_queue_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "full_pack_loop_iteration_plan_no_training_started",
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


def normalize_stem(value: Any) -> str:
    return str(value or "").strip().lower().replace("vocals", "vocal").replace("instrumental", "minus")


def canonical_path(raw_path: Any, base_dir: Path | None = None) -> Path:
    text = str(raw_path or "")
    if not text:
        return Path("")
    path = Path(text)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


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


def job_stems(job: dict[str, Any]) -> list[str]:
    raw = job.get("stem_types")
    if isinstance(raw, list):
        return sorted({normalize_stem(stem) for stem in raw if normalize_stem(stem)})
    stem = normalize_stem(job.get("stem_type") or job.get("stem"))
    return [stem] if stem else []


def load_training_jobs(training_plan_json: Path | None) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    if training_plan_json is None:
        return {}, ["training_plan_missing"]
    if not training_plan_json.exists():
        return {}, ["training_plan_file_missing"]
    plan = read_json(training_plan_json)
    failed = collect_true_flag_failures("training_plan", plan)
    if plan.get("schema") != "generic_audio_finetune_job_plan.v1":
        failed.append("training_plan_schema_unexpected")
    jobs_by_stem: dict[str, list[dict[str, Any]]] = {}
    for job in plan.get("jobs") if isinstance(plan.get("jobs"), list) else []:
        if not isinstance(job, dict):
            continue
        for stem in job_stems(job):
            jobs_by_stem.setdefault(stem, []).append(job)
    return jobs_by_stem, stable_unique(failed)


def route_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("route_intent") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def first_ready_job(jobs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for job in jobs:
        if job.get("status") == "ready_for_gpu_training":
            return job
    return None


def docker_commands(job: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = []
    for example in job.get("example_jobs") if isinstance(job.get("example_jobs"), list) else []:
        if isinstance(example, dict) and isinstance(example.get("command"), list):
            commands.append([str(part) for part in example["command"]])
    return commands


def blocked_status_from_jobs(jobs: Sequence[dict[str, Any]]) -> str:
    statuses = {str(job.get("status") or "unknown") for job in jobs}
    if "blocked_insufficient_unique_tracks" in statuses:
        return "blocked_insufficient_unique_tracks"
    if statuses:
        return "blocked_training_job"
    return "blocked_missing_training_job"


def training_job_statuses(jobs: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(job.get("status") or "unknown") for job in jobs})


def stem_training_row(queue_item: dict[str, Any], jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stem = normalize_stem(queue_item.get("stem_type"))
    ready_job = first_ready_job(jobs)
    base = {
        "schema": "full_pack_loop_iteration_action.v1",
        "queue_item_id": str(queue_item.get("queue_item_id") or ""),
        "stem_type": stem,
        "route_intent": str(queue_item.get("route_intent") or ""),
        "issue_tags": [str(tag) for tag in queue_item.get("issue_tags") or [] if str(tag)],
        "source_audio_path": str(queue_item.get("audio_path") or ""),
        "training_job_statuses": training_job_statuses(jobs),
        "required_next_actions": [],
        "docker_commands": [],
        "expected_acceleration_mode": "external_only_cpu_blocker_analysis",
        "execution_started": False,
        "training_started": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }
    if ready_job is not None:
        return {
            **base,
            "action_status": "ready_for_gpu_training",
            "scope": str(ready_job.get("scope") or ""),
            "unique_track_count": ready_job.get("unique_track_count"),
            "min_unique_tracks": ready_job.get("min_unique_tracks"),
            "required_next_actions": ["gpu_teacher_mask_train", "holdout_eval_gate", "train_ready_decision_gate", "full_pack_gate_rebuild"],
            "docker_commands": docker_commands(ready_job),
            "expected_acceleration_mode": "gpu_training",
        }
    status = blocked_status_from_jobs(jobs)
    unique_counts = [job.get("unique_track_count") for job in jobs if job.get("unique_track_count") is not None]
    min_counts = [job.get("min_unique_tracks") for job in jobs if job.get("min_unique_tracks") is not None]
    required = ["verified_reference_corpus_expand"]
    if status == "blocked_missing_training_job":
        required.append("rerun_generic_audio_finetune_job_plan")
    required.extend(["holdout_eval_gate", "train_ready_decision_gate", "full_pack_gate_rebuild"])
    return {
        **base,
        "action_status": status,
        "unique_track_count": min(unique_counts) if unique_counts else None,
        "min_unique_tracks": max(min_counts) if min_counts else None,
        "required_next_actions": stable_unique(required),
    }


def minus_repair_row(queue_item: dict[str, Any], jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    training_status = blocked_status_from_jobs(jobs)
    actions = ["run_minus_repair_route", "minus_audit_gate", "verified_reference_corpus_expand", "train_ready_decision_gate", "full_pack_gate_rebuild"]
    unique_counts = [job.get("unique_track_count") for job in jobs if job.get("unique_track_count") is not None]
    min_counts = [job.get("min_unique_tracks") for job in jobs if job.get("min_unique_tracks") is not None]
    status = "blocked_minus_repair_and_training_coverage"
    if training_status == "blocked_missing_training_job":
        status = "blocked_minus_repair_missing_training_job"
    return {
        "schema": "full_pack_loop_iteration_action.v1",
        "queue_item_id": str(queue_item.get("queue_item_id") or ""),
        "stem_type": normalize_stem(queue_item.get("stem_type")),
        "route_intent": str(queue_item.get("route_intent") or ""),
        "issue_tags": [str(tag) for tag in queue_item.get("issue_tags") or [] if str(tag)],
        "source_audio_path": str(queue_item.get("audio_path") or ""),
        "action_status": status,
        "training_job_statuses": training_job_statuses(jobs),
        "unique_track_count": min(unique_counts) if unique_counts else None,
        "min_unique_tracks": max(min_counts) if min_counts else None,
        "required_next_actions": actions,
        "docker_commands": [],
        "expected_acceleration_mode": "gpu_inference_then_gpu_training",
        "execution_started": False,
        "training_started": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def build_action_rows(queue_rows: Sequence[dict[str, Any]], jobs_by_stem: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for queue_item in queue_rows:
        failed.extend(collect_true_flag_failures("repair_queue_item", queue_item))
        intent = str(queue_item.get("route_intent") or "")
        stem = normalize_stem(queue_item.get("stem_type"))
        jobs = jobs_by_stem.get(stem, [])
        if intent == "stem_training_coverage_gap":
            rows.append(stem_training_row(queue_item, jobs))
        elif intent == "minus_repair":
            rows.append(minus_repair_row(queue_item, jobs))
        else:
            failed.append(f"unsupported_route_intent:{intent}")
    return rows, stable_unique(failed)


def plan_status(rows: Sequence[dict[str, Any]], failed: Sequence[str]) -> str:
    if failed:
        return "blocked"
    if any(row.get("action_status") == "ready_for_gpu_training" for row in rows):
        if all(str(row.get("action_status") or "").startswith("ready_") for row in rows):
            return "ready_for_gpu_training_actions"
        return "partial_ready_with_blockers"
    if any(row.get("route_intent") == "minus_repair" for row in rows):
        return "blocked_waiting_for_verified_reference_corpus_and_minus_repair"
    return "blocked_waiting_for_verified_reference_corpus"


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Full Pack Loop Iteration Plan",
        "",
        f"- plan_status: `{manifest['plan_status']}`",
        f"- action_count: `{manifest['action_count']}`",
        f"- ready_action_count: `{manifest['ready_action_count']}`",
        f"- blocked_action_count: `{manifest['blocked_action_count']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
        "## Actions",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['queue_item_id']}` `{row['stem_type']}` `{row['action_status']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_full_pack_loop_iteration_plan(
    *,
    repair_queue_manifest_json: Path,
    output_dir: Path,
    training_plan_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", repair_queue_manifest_json=str(repair_queue_manifest_json), training_plan_json=str(training_plan_json or ""))

    failed: list[str] = []
    queue_manifest = read_json(repair_queue_manifest_json)
    failed.extend(collect_true_flag_failures("repair_queue", queue_manifest))
    if queue_manifest.get("schema") != "short_listen_repair_queue_manifest.v1":
        failed.append("repair_queue_schema_unexpected")
    if queue_manifest.get("queue_status") != READY_QUEUE_STATUS:
        failed.append("repair_queue_not_full_pack_loop_ready")
    if queue_manifest.get("failed_checks") not in ([], None):
        failed.append("repair_queue_failed_checks_not_empty")

    rows_path = canonical_path(queue_manifest.get("rows_jsonl"), repair_queue_manifest_json.parent)
    if not str(rows_path) or not rows_path.exists():
        failed.append("repair_queue_rows_jsonl_missing")
        queue_rows: list[dict[str, Any]] = []
    else:
        queue_rows = list(iter_jsonl(rows_path))

    jobs_by_stem, training_failed = load_training_jobs(training_plan_json)
    failed.extend(training_failed)
    rows, row_failed = build_action_rows(queue_rows, jobs_by_stem)
    failed.extend(row_failed)
    failed = stable_unique(failed)
    if failed:
        rows = []

    ready_count = sum(1 for row in rows if str(row.get("action_status") or "").startswith("ready_"))
    blocked_count = len(rows) - ready_count
    status = plan_status(rows, failed)
    manifest = {
        "schema": "full_pack_loop_iteration_plan.v1",
        "plan_status": status,
        "repair_queue_manifest_json": str(repair_queue_manifest_json),
        "training_plan_json": str(training_plan_json or ""),
        "queue_status": str(queue_manifest.get("queue_status") or ""),
        "queue_source": str(queue_manifest.get("queue_source") or ""),
        "action_count": len(rows),
        "ready_action_count": ready_count,
        "blocked_action_count": blocked_count,
        "route_intent_counts": route_counts(rows),
        "failed_checks": failed,
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "execution_started": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "delivery_allowed": False,
        "presentation_status": "machine_loop_plan_not_for_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "not_claimed": ["accepted_stems", "listen_result", "audio_repaired", "training_started", "model_promoted"],
        "command_args": command_args or [],
    }
    write_json(output_dir / PLAN_NAME, manifest)
    write_jsonl(output_dir / ROWS_NAME, rows)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", plan_status=status, action_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a machine-only full-pack ARIS iteration plan from a repair queue.")
    parser.add_argument("--repair-queue-manifest-json", type=Path, required=True)
    parser.add_argument("--training-plan-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_loop_iteration_plan.py",
        entrypoint_path="/app/tools/build_full_pack_loop_iteration_plan.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_loop_iteration_plan(
        repair_queue_manifest_json=args.repair_queue_manifest_json,
        training_plan_json=args.training_plan_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / PLAN_NAME),
                "plan_status": manifest["plan_status"],
                "action_count": manifest["action_count"],
                "ready_action_count": manifest["ready_action_count"],
                "blocked_action_count": manifest["blocked_action_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["ready_action_count"] > 0 and not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
