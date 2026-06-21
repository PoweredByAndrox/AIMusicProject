#!/usr/bin/env python3
"""Audit whether the ARIS full-pack goal can be closed by machine evidence."""

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


AUDIT_NAME = "audit.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()
REQUIRED_STEMS = {"vocal", "minus", "drums", "bass", "other", "guitar", "piano"}
REQUIRED_MACHINE_GATES = {
    "aggregate_pack_validation_passed",
    "full_required_stem_coverage",
    "required_stem_machine_quality_evidence",
    "source_vocal_minus_master_pair_compatible",
    "minus_quality_audit_passed_without_warnings",
    "required_stem_train_ready_model_evidence",
}
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "cleanup_routes_added",
)


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
        "expected_acceleration_mode": "external_only_cpu_completion_audit",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "completion_audit_reads_existing_manifests_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "completion_audit_no_training_started",
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


def int_value(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def normalize_stem(value: Any) -> str:
    stem = str(value or "").strip().lower().replace(" ", "_")
    if stem == "vocals":
        return "vocal"
    if stem == "instrumental":
        return "minus"
    return stem


def collect_true_flag_blockers(prefix: str, payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            blockers.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                blockers.append(f"{prefix}_safety_{key}_true")
    return blockers


def check(check_id: str, status: str, evidence: dict[str, Any], blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status,
        "evidence": evidence,
        "blockers": stable_unique(list(blockers)),
    }


def machine_gate_check(gate: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if gate.get("schema") != "machine_stem_acceptance_gate.v1":
        blockers.append("machine_gate_schema_unexpected")
    if gate.get("gate_status") != "accepted":
        blockers.append("machine_gate_not_accepted")
    if gate.get("final_result_allowed") is not True:
        blockers.append("machine_gate_final_result_not_allowed")
    if gate.get("delivery_allowed") is not True:
        blockers.append("machine_gate_delivery_not_allowed")
    if gate.get("presentation_status") != "accepted_for_listen_pack":
        blockers.append("machine_gate_presentation_not_accepted")
    failed_checks = [str(item) for item in gate.get("failed_checks") or []]
    blockers.extend(f"machine_gate_failed:{item}" for item in failed_checks)
    missing = [str(item) for item in gate.get("missing_required_stems") or []]
    blockers.extend(f"required_stem_missing:{stem}" for stem in missing)
    gate_required = {normalize_stem(item) for item in gate.get("required_stems") or []}
    for stem in sorted(REQUIRED_STEMS - gate_required):
        blockers.append(f"required_stem_missing_from_gate_contract:{stem}")
    gate_stems = {normalize_stem(item) for item in gate.get("stem_types") or []}
    for stem in sorted(REQUIRED_STEMS - gate_stems):
        blockers.append(f"required_stem_missing_from_gate_pack:{stem}")
    machine_gates = {str(item) for item in gate.get("required_machine_gates") or []}
    for gate_name in sorted(REQUIRED_MACHINE_GATES - machine_gates):
        blockers.append(f"machine_gate_required_gate_missing:{gate_name}")
    if gate.get("require_training_coverage") is not True:
        blockers.append("machine_gate_training_coverage_not_required")
    contract = gate.get("contract") if isinstance(gate.get("contract"), dict) else {}
    if contract.get("training_coverage_required") is not True:
        blockers.append("machine_gate_contract_training_coverage_not_required")
    if contract.get("train_ready_model_evidence_required") is not True:
        blockers.append("machine_gate_contract_train_ready_model_evidence_not_required")
    train_ready = {normalize_stem(item) for item in gate.get("train_ready_stems") or []}
    for stem in sorted(REQUIRED_STEMS - train_ready):
        blockers.append(f"required_stem_train_ready_missing:{stem}")
    blockers.extend(collect_true_flag_blockers("machine_gate", gate))
    return check(
        "machine_gate",
        "proved" if not blockers else "blocked",
        {
            "gate_status": gate.get("gate_status"),
            "final_result_allowed": gate.get("final_result_allowed"),
            "delivery_allowed": gate.get("delivery_allowed"),
            "presentation_status": gate.get("presentation_status"),
            "required_stems": sorted(gate_required),
            "stem_types": sorted(gate_stems),
            "required_machine_gates": sorted(machine_gates),
            "require_training_coverage": gate.get("require_training_coverage"),
            "train_ready_stems": sorted(train_ready),
            "failed_checks": failed_checks,
        },
        blockers,
    )


def cycle_check(cycle: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if cycle.get("schema") != "full_pack_aris_cycle_status.v1":
        blockers.append("cycle_status_schema_unexpected")
    next_action = str(cycle.get("next_action") or "")
    if next_action not in {"final_delivery_audit", "none", "complete"}:
        blockers.append(f"cycle_next_action:{next_action or 'missing'}")
    cycle_status = str(cycle.get("cycle_status") or "")
    if cycle_status.startswith("blocked"):
        blockers.append(f"cycle_status:{cycle_status}")
    failed_checks = [str(item) for item in cycle.get("failed_checks") or []]
    blockers.extend(f"cycle_failed:{item}" for item in failed_checks)
    summary = cycle.get("blocking_summary") if isinstance(cycle.get("blocking_summary"), dict) else {}
    if int_value(summary, "remaining_reference_stem_count") > 0:
        blockers.append("cycle_remaining_reference_stems")
    if int_value(summary, "blocked_training_job_count") > 0:
        blockers.append("cycle_blocked_training_jobs")
    blockers.extend(collect_true_flag_blockers("cycle", cycle))
    return check(
        "cycle_status",
        "proved" if not blockers else "blocked",
        {"cycle_status": cycle_status, "next_action": next_action, "blocking_summary": summary},
        blockers,
    )


def collection_check(queue: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if queue.get("schema") != "full_pack_reference_collection_queue_manifest.v1":
        blockers.append("collection_queue_schema_unexpected")
    if int_value(queue, "collection_item_count") > 0:
        blockers.append("reference_collection_pending")
    status = str(queue.get("queue_status") or "")
    if status == "ready_for_reference_collection":
        blockers.append("reference_collection_queue_ready")
    failed_checks = [str(item) for item in queue.get("failed_checks") or []]
    blockers.extend(f"collection_queue_failed:{item}" for item in failed_checks)
    blockers.extend(collect_true_flag_blockers("collection_queue", queue))
    return check(
        "reference_collection",
        "proved" if not blockers else "blocked",
        {
            "queue_status": status,
            "collection_item_count": queue.get("collection_item_count"),
            "remaining_by_stem": queue.get("remaining_by_stem") or {},
        },
        blockers,
    )


def corpus_check(corpus: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if corpus.get("schema") != "verified_audio_training_corpus_manifest.v1":
        blockers.append("verified_corpus_schema_unexpected")
    if corpus.get("corpus_validation_status") != "passed":
        blockers.append("verified_corpus_not_passed")
    failed_checks = [str(item) for item in corpus.get("failed_checks") or []]
    blockers.extend(f"verified_corpus_failed:{item}" for item in failed_checks)
    if int_value(corpus, "unique_track_count") < int_value(corpus, "min_unique_tracks"):
        blockers.append("verified_corpus_insufficient_unique_tracks")
    blockers.extend(collect_true_flag_blockers("verified_corpus", corpus))
    return check(
        "verified_corpus",
        "proved" if not blockers else "blocked",
        {
            "corpus_validation_status": corpus.get("corpus_validation_status"),
            "unique_track_count": corpus.get("unique_track_count"),
            "min_unique_tracks": corpus.get("min_unique_tracks"),
            "failed_checks": failed_checks,
        },
        blockers,
    )


def training_check(training: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if training.get("schema") != "generic_audio_finetune_job_plan.v1":
        blockers.append("training_plan_schema_unexpected")
    summary = training.get("summary") if isinstance(training.get("summary"), dict) else {}
    if int_value(summary, "blocked_job_count") > 0:
        blockers.append("training_jobs_blocked")
    if int_value(summary, "ready_job_count") <= 0:
        blockers.append("training_ready_jobs_missing")
    failed_checks = [str(item) for item in training.get("failed_checks") or []]
    blockers.extend(f"training_plan_failed:{item}" for item in failed_checks)
    blockers.extend(collect_true_flag_blockers("training_plan", training))
    return check(
        "training_plan",
        "proved" if not blockers else "blocked",
        {"summary": summary, "failed_checks": failed_checks},
        blockers,
    )


def build_summary(audit: dict[str, Any]) -> str:
    blockers = audit["remaining_blockers"]
    lines = [
        "# Full Pack Goal Completion Audit",
        "",
        f"- completion_status: `{audit['completion_status']}`",
        f"- goal_complete: `{str(audit['goal_complete']).lower()}`",
        f"- result_delivery_allowed: `{str(audit['result_delivery_allowed']).lower()}`",
        f"- remaining_blocker_count: `{len(blockers)}`",
        "",
    ]
    if blockers:
        lines.append("## Blockers")
        lines.extend(f"- `{blocker}`" for blocker in blockers)
        lines.append("")
    return "\n".join(lines)


def audit_full_pack_goal_completion(
    *,
    machine_gate_json: Path,
    cycle_status_json: Path,
    collection_queue_json: Path,
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

    gate = read_json(machine_gate_json)
    cycle = read_json(cycle_status_json)
    queue = read_json(collection_queue_json)
    corpus = read_json(verified_corpus_manifest_json)
    training = read_json(training_plan_json)
    checks = [
        machine_gate_check(gate),
        cycle_check(cycle),
        collection_check(queue),
        corpus_check(corpus),
        training_check(training),
    ]
    remaining_blockers = stable_unique([blocker for item in checks for blocker in item["blockers"]])
    goal_complete = not remaining_blockers
    result_delivery_allowed = goal_complete and gate.get("delivery_allowed") is True and gate.get("final_result_allowed") is True
    completion_status = "machine_accepted" if goal_complete else "machine_evidence_incomplete"
    audit = {
        "schema": "full_pack_goal_completion_audit.v1",
        "manifest_type": "full_pack_goal_completion_audit",
        "completion_status": completion_status,
        "goal_complete": goal_complete,
        "result_delivery_allowed": result_delivery_allowed,
        "result_presentation_status": "accepted_for_listen_pack" if result_delivery_allowed else "not_accepted",
        "remaining_blockers": remaining_blockers,
        "checks": checks,
        "inputs": {
            "machine_gate_json": str(machine_gate_json),
            "cycle_status_json": str(cycle_status_json),
            "collection_queue_json": str(collection_queue_json),
            "verified_corpus_manifest_json": str(verified_corpus_manifest_json),
            "training_plan_json": str(training_plan_json),
        },
        "outputs": {
            "audit_json": str(output_dir / AUDIT_NAME),
            "summary_md": str(output_dir / SUMMARY_NAME),
            "progress_jsonl": str(progress_path),
        },
        "delivery_allowed": False,
        "presentation_status": "completion_audit_not_for_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "not_claimed": [
            "audio_created",
            "listen_pack_created",
            "model_promoted",
            "release_ready",
        ],
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "training_started": False,
            "requires_machine_accepted_gate": True,
            "not_release_artifact": True,
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / AUDIT_NAME, audit)
    (output_dir / SUMMARY_NAME).write_text(build_summary(audit), encoding="utf-8")
    progress(progress_path, "finish", completion_status=completion_status, goal_complete=goal_complete, remaining_blockers=remaining_blockers)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether full-pack ARIS goal completion is machine-proved.")
    parser.add_argument("--machine-gate-json", type=Path, required=True)
    parser.add_argument("--cycle-status-json", type=Path, required=True)
    parser.add_argument("--collection-queue-json", type=Path, required=True)
    parser.add_argument("--verified-corpus-manifest-json", type=Path, required=True)
    parser.add_argument("--training-plan-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="audit_full_pack_goal_completion.py",
        entrypoint_path="/app/tools/audit_full_pack_goal_completion.py",
    )
    args = parse_args(argv)
    audit = audit_full_pack_goal_completion(
        machine_gate_json=args.machine_gate_json,
        cycle_status_json=args.cycle_status_json,
        collection_queue_json=args.collection_queue_json,
        verified_corpus_manifest_json=args.verified_corpus_manifest_json,
        training_plan_json=args.training_plan_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "audit_path": str(args.output_dir / AUDIT_NAME),
                "completion_status": audit["completion_status"],
                "goal_complete": audit["goal_complete"],
                "remaining_blockers": audit["remaining_blockers"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if audit["goal_complete"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
