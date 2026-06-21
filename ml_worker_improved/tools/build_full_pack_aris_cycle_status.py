#!/usr/bin/env python3
"""Aggregate full-pack ARIS loop artifacts into one machine next-action status."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "status.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
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
EXPECTED_SCHEMAS = {
    "machine_gate": "machine_stem_acceptance_gate.v1",
    "repair_queue": "short_listen_repair_queue_manifest.v1",
    "full_pack_loop_plan": "full_pack_loop_iteration_plan.v1",
    "reference_gap": "full_pack_reference_corpus_gap_plan.v1",
    "reference_intake": "full_pack_reference_corpus_intake_manifest.v1",
    "verified_corpus": "verified_audio_training_corpus_manifest.v1",
    "training_plan": "generic_audio_finetune_job_plan.v1",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_cycle_status_reconciliation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "cycle_status_reads_existing_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "cycle_status_no_training_started",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Sequence[str]) -> list[str]:
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
        if not isinstance(nested, dict):
            continue
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def int_value(payload: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(payload.get(key) or default)
    except (TypeError, ValueError):
        return default


def load_checked(path: Path, *, label: str) -> tuple[dict[str, Any], list[str]]:
    payload = read_json(path)
    failed = collect_true_flag_failures(label, payload)
    expected = EXPECTED_SCHEMAS[label]
    if payload.get("schema") != expected:
        failed.append(f"{label}_schema_unexpected")
    if payload.get("failed_checks") not in ([], None):
        # Failed checks are state evidence, not an input-contract failure, except for
        # artifacts whose own safety contract already failed above.
        pass
    return payload, failed


def next_action_from_state(
    *,
    failed_checks: Sequence[str],
    gap: dict[str, Any],
    intake: dict[str, Any],
    corpus: dict[str, Any],
    training: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[str, str]:
    if failed_checks:
        return "blocked_invalid_cycle_inputs", "fix_cycle_input_contracts"
    remaining_refs = int_value(intake, "remaining_reference_stem_count") or int_value(gap, "missing_reference_stem_count")
    if remaining_refs > 0:
        return "blocked_waiting_for_verified_reference_corpus", "collect_verified_reference_corpus"
    summary = training.get("summary") if isinstance(training.get("summary"), dict) else {}
    ready_jobs = int_value(summary, "ready_job_count")
    if ready_jobs > 0:
        return "ready_for_gpu_training", "run_gpu_training_and_holdout_eval"
    if corpus.get("corpus_validation_status") != "passed":
        return "blocked_waiting_for_corpus_validation", "rerun_or_fix_verified_corpus_validation"
    if str(plan.get("plan_status") or "").startswith("blocked"):
        return "blocked_waiting_for_full_pack_loop_actions", "execute_full_pack_loop_actions"
    return "needs_full_pack_gate_rebuild", "full_pack_gate_rebuild"


def summary_text(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack ARIS Cycle Status",
            "",
            f"- cycle_status: `{manifest['cycle_status']}`",
            f"- next_action: `{manifest['next_action']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            f"- training_started: `{str(manifest['training_started']).lower()}`",
            "",
        ]
    )


def build_full_pack_aris_cycle_status(
    *,
    machine_gate_json: Path,
    repair_queue_json: Path,
    full_pack_loop_plan_json: Path,
    reference_gap_plan_json: Path,
    reference_intake_json: Path,
    verified_corpus_manifest_json: Path,
    training_plan_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", output_dir=str(output_dir))

    gate, gate_failed = load_checked(machine_gate_json, label="machine_gate")
    queue, queue_failed = load_checked(repair_queue_json, label="repair_queue")
    plan, plan_failed = load_checked(full_pack_loop_plan_json, label="full_pack_loop_plan")
    gap, gap_failed = load_checked(reference_gap_plan_json, label="reference_gap")
    intake, intake_failed = load_checked(reference_intake_json, label="reference_intake")
    corpus, corpus_failed = load_checked(verified_corpus_manifest_json, label="verified_corpus")
    training, training_failed = load_checked(training_plan_json, label="training_plan")
    failed_checks = stable_unique(
        [
            *gate_failed,
            *queue_failed,
            *plan_failed,
            *gap_failed,
            *intake_failed,
            *corpus_failed,
            *training_failed,
        ]
    )
    cycle_status, next_action = next_action_from_state(
        failed_checks=failed_checks,
        gap=gap,
        intake=intake,
        corpus=corpus,
        training=training,
        plan=plan,
    )
    training_summary = training.get("summary") if isinstance(training.get("summary"), dict) else {}
    manifest = {
        "schema": "full_pack_aris_cycle_status.v1",
        "manifest_type": "full_pack_aris_cycle_status",
        "cycle_status": cycle_status,
        "next_action": next_action,
        "failed_checks": failed_checks,
        "inputs": {
            "machine_gate_json": str(machine_gate_json),
            "repair_queue_json": str(repair_queue_json),
            "full_pack_loop_plan_json": str(full_pack_loop_plan_json),
            "reference_gap_plan_json": str(reference_gap_plan_json),
            "reference_intake_json": str(reference_intake_json),
            "verified_corpus_manifest_json": str(verified_corpus_manifest_json),
            "training_plan_json": str(training_plan_json),
        },
        "source_status": {
            "machine_gate": str(gate.get("gate_status") or ""),
            "repair_queue": str(queue.get("queue_status") or ""),
            "full_pack_loop_plan": str(plan.get("plan_status") or ""),
            "reference_gap": str(gap.get("gap_status") or ""),
            "reference_intake": str(intake.get("intake_status") or ""),
            "verified_corpus": str(corpus.get("corpus_validation_status") or ""),
        },
        "blocking_summary": {
            "remaining_reference_stem_count": int_value(intake, "remaining_reference_stem_count")
            or int_value(gap, "missing_reference_stem_count"),
            "missing_training_job_count": int_value(gap, "missing_training_job_count"),
            "ready_training_job_count": int_value(training_summary, "ready_job_count"),
            "blocked_training_job_count": int_value(training_summary, "blocked_job_count"),
            "verified_corpus_failed_checks": [str(item) for item in corpus.get("failed_checks") or []],
            "machine_gate_failed_checks": [str(item) for item in gate.get("failed_checks") or []],
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "full_pack_cycle_status_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "not_claimed": ["accepted_stems", "listen_result", "training_started", "model_promoted"],
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "training_started": False,
            "not_release_artifact": True,
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(progress_path, "finish", cycle_status=cycle_status, next_action=next_action, failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full-pack ARIS cycle status manifest.")
    parser.add_argument("--machine-gate-json", type=Path, required=True)
    parser.add_argument("--repair-queue-json", type=Path, required=True)
    parser.add_argument("--full-pack-loop-plan-json", type=Path, required=True)
    parser.add_argument("--reference-gap-plan-json", type=Path, required=True)
    parser.add_argument("--reference-intake-json", type=Path, required=True)
    parser.add_argument("--verified-corpus-manifest-json", type=Path, required=True)
    parser.add_argument("--training-plan-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_aris_cycle_status.py",
        entrypoint_path="/app/tools/build_full_pack_aris_cycle_status.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_aris_cycle_status(
        machine_gate_json=args.machine_gate_json,
        repair_queue_json=args.repair_queue_json,
        full_pack_loop_plan_json=args.full_pack_loop_plan_json,
        reference_gap_plan_json=args.reference_gap_plan_json,
        reference_intake_json=args.reference_intake_json,
        verified_corpus_manifest_json=args.verified_corpus_manifest_json,
        training_plan_json=args.training_plan_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "cycle_status": manifest["cycle_status"],
                "next_action": manifest["next_action"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
