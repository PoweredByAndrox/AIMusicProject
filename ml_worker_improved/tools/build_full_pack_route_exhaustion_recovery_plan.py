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
PLAN_NAME = "plan.json"
ACTIONS_NAME = "actions.jsonl"
PROGRESS_NAME = "progress.jsonl"
SUMMARY_NAME = "summary.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_unique(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_route_exhaustion_recovery_plan",
        "expected_acceleration_mode": "external_only_cpu_metadata_planning",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in (
        "delivery_allowed",
        "release_ready",
        "can_publish_as_result",
        "final_result_allowed",
        "model_registry_edited",
        "thresholds_relaxed",
    ):
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    return failed


def int_value(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def bool_value(payload: dict[str, Any], key: str) -> bool:
    return payload.get(key) is True


def normalized_stem_list(values: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(values, list):
        return result
    for value in values:
        text = str(value or "").strip().lower().replace(" ", "_")
        if text == "vocals":
            text = "vocal"
        elif text == "instrumental":
            text = "minus"
        if text and text not in result:
            result.append(text)
    return result


def stem_suffixes(checks: Sequence[str], prefix: str) -> list[str]:
    stems: list[str] = []
    for check in checks:
        if not check.startswith(prefix):
            continue
        stem = check.split(":", 1)[1] if ":" in check else ""
        if stem:
            stems.append(stem)
    return normalized_stem_list(stems)


def route_summary(route_expansion: dict[str, Any]) -> dict[str, Any]:
    summary = route_expansion.get("summary")
    return summary if isinstance(summary, dict) else {}


def reference_intake_evidence(reference_intake: dict[str, Any] | None, candidate_roots: Sequence[str]) -> dict[str, Any]:
    if reference_intake is None:
        return {
            "reference_intake_status": "",
            "reference_intake_accepted_reference_rows": 0,
            "reference_intake_rejected_reference_rows": 0,
            "reference_intake_rejection_reason_counts": {},
            "reference_intake_candidate_roots_consumed": False,
        }
    intake_roots = stable_unique([str(item) for item in reference_intake.get("candidate_roots") or []])
    root_overlap = bool(set(candidate_roots) & set(intake_roots))
    accepted_rows = int_value(reference_intake, "accepted_reference_rows")
    rejected_rows = int_value(reference_intake, "rejected_reference_rows")
    reason_counts = reference_intake.get("rejection_reason_counts")
    return {
        "reference_intake_status": str(reference_intake.get("intake_status") or ""),
        "reference_intake_accepted_reference_rows": accepted_rows,
        "reference_intake_rejected_reference_rows": rejected_rows,
        "reference_intake_rejection_reason_counts": reason_counts if isinstance(reason_counts, dict) else {},
        "reference_intake_candidate_roots_consumed": bool(
            candidate_roots
            and root_overlap
            and accepted_rows == 0
            and rejected_rows > 0
        ),
    }


def collect_summary_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in (
        "cleanup_routes_added",
        "human_csv_required",
        "melody_transcription_ran",
        "model_downloaded",
        "model_registry_edited",
        "recommendation_index_updated",
        "reference_required",
        "release_ready",
        "separator_training_started",
        "thresholds_relaxed",
        "training_started",
    ):
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    return failed


def blocking_evidence(
    *,
    machine_gate: dict[str, Any],
    minus_audit: dict[str, Any],
    route_expansion: dict[str, Any],
    reference_gap: dict[str, Any],
    raw_reference_handoff: dict[str, Any],
) -> dict[str, Any]:
    summary = route_summary(route_expansion)
    machine_failed_checks = [str(item) for item in machine_gate.get("failed_checks") or []]
    required_stems = normalized_stem_list(machine_gate.get("required_stems"))
    present_stems = normalized_stem_list(machine_gate.get("stem_types"))
    missing_required_stems = normalized_stem_list(machine_gate.get("missing_required_stems"))
    if not missing_required_stems and required_stems and present_stems:
        missing_required_stems = [stem for stem in required_stems if stem not in present_stems]
    train_ready_stems = normalized_stem_list(machine_gate.get("train_ready_stems"))
    missing_train_ready_stems = stem_suffixes(machine_failed_checks, "required_stem_train_ready_missing:")
    if not missing_train_ready_stems and required_stems:
        missing_train_ready_stems = [stem for stem in required_stems if stem not in train_ready_stems]
    return {
        "machine_gate_failed_checks": machine_failed_checks,
        "minus_audit_failed_checks": [str(item) for item in minus_audit.get("failed_checks") or []],
        "full_pack_required_stems": required_stems,
        "full_pack_present_stems": present_stems,
        "full_pack_missing_required_stems": missing_required_stems,
        "full_pack_train_ready_stems": train_ready_stems,
        "full_pack_missing_train_ready_stems": missing_train_ready_stems,
        "blocked_minus_count": int_value(minus_audit, "blocked_minus_count"),
        "minus_warning_count": int_value(minus_audit, "warning_count"),
        "route_space_exhausted": bool_value(summary, "route_space_exhausted"),
        "accepted_minus_candidates": int_value(summary, "accepted_minus_candidates"),
        "minus_candidates_generated": int_value(summary, "minus_candidates_generated"),
        "jobs_attempted": int_value(summary, "jobs_attempted"),
        "jobs_failed": int_value(summary, "jobs_failed"),
        "severe_vocal_bleed_in_minus_count": int_value(summary, "severe_vocal_bleed_in_minus_count"),
        "severe_phase_collapse_count": int_value(summary, "severe_phase_collapse_count"),
        "selected_route_ids": [str(item) for item in summary.get("selected_route_ids") or []],
        "rejected_route_ids": [str(item) for item in summary.get("rejected_route_ids") or []],
        "reference_gap_count": int_value(reference_gap, "gap_count"),
        "missing_reference_stem_count": int_value(reference_gap, "missing_reference_stem_count"),
        "missing_training_job_count": int_value(reference_gap, "missing_training_job_count"),
        "ready_raw_reference_handoff_count": int_value(raw_reference_handoff, "ready_handoff_count"),
        "blocked_raw_reference_handoff_count": int_value(raw_reference_handoff, "blocked_handoff_count"),
    }


def action_row(
    *,
    action_id: str,
    action_type: str,
    target_stem_type: str,
    priority: str,
    reason: str,
    evidence: dict[str, Any],
    candidate_roots: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema": "full_pack_route_exhaustion_recovery_action.v1",
        "action_id": action_id,
        "action_type": action_type,
        "target_stem_type": target_stem_type,
        "priority": priority,
        "reason": reason,
        "candidate_roots": list(candidate_roots),
        "blocking_evidence": evidence,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
    }


def build_actions(
    *,
    evidence: dict[str, Any],
    candidate_roots: Sequence[str],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    if evidence["accepted_minus_candidates"] > 0:
        failed.append("accepted_minus_candidate_present_use_gate_route_instead")
    if not evidence["route_space_exhausted"]:
        failed.append("minus_route_space_not_exhausted")
    if evidence["jobs_failed"] > 0:
        failed.append("minus_route_expansion_jobs_failed")
    if failed:
        return "blocked_route_space_not_exhausted", [], failed

    if candidate_roots and not evidence["reference_intake_candidate_roots_consumed"]:
        actions = [
            action_row(
                action_id="a001",
                action_type="ingest_ready_reference_material",
                target_stem_type="minus",
                priority="high",
                reason="ready_reference_material_exists_after_route_exhaustion",
                evidence=evidence,
                candidate_roots=candidate_roots,
            ),
            action_row(
                action_id="a002",
                action_type="rerun_full_pack_gates_after_new_candidate",
                target_stem_type="minus",
                priority="high",
                reason="rerun_required_after_reference_intake_or_new_candidate_generation",
                evidence=evidence,
            ),
        ]
        return "ready_for_reference_material_intake", actions, []

    actions = [
        action_row(
            action_id="a001",
            action_type="collect_verified_reference_material",
            target_stem_type="minus",
            priority="high",
            reason="existing_minus_routes_exhausted_without_accepted_candidate",
            evidence=evidence,
        ),
        action_row(
            action_id="a002",
            action_type="design_new_minus_route_or_training_candidate",
            target_stem_type="minus",
            priority="high",
            reason="registered_route_space_exhausted_and_reference_gap_remains",
            evidence=evidence,
        ),
        action_row(
            action_id="a003",
            action_type="rerun_full_pack_gates_after_new_candidate",
            target_stem_type="minus",
            priority="high",
            reason="full_pack_delivery_requires_machine_gate_acceptance_after_candidate_generation",
            evidence=evidence,
        ),
    ]
    return "blocked_waiting_for_reference_or_new_route_material", actions, []


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Route Exhaustion Recovery Plan",
            "",
            f"- plan_status: `{manifest['plan_status']}`",
            f"- route_space_exhausted: `{str(manifest['route_space_exhausted']).lower()}`",
            f"- accepted_minus_candidates: `{manifest['accepted_minus_candidates']}`",
            f"- full_pack_required_stems: `{', '.join(manifest['full_pack_required_stems'])}`",
            f"- full_pack_missing_train_ready_stems: `{', '.join(manifest['full_pack_missing_train_ready_stems']) or 'none'}`",
            f"- action_count: `{manifest['action_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_full_pack_route_exhaustion_recovery_plan(
    *,
    machine_gate_json: Path,
    minus_audit_json: Path,
    minus_route_expansion_json: Path,
    reference_gap_plan_json: Path,
    raw_reference_handoff_json: Path,
    output_dir: Path,
    reference_intake_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", minus_route_expansion_json=str(minus_route_expansion_json))

    machine_gate = read_json(machine_gate_json)
    minus_audit = read_json(minus_audit_json)
    route_expansion = read_json(minus_route_expansion_json)
    reference_gap = read_json(reference_gap_plan_json)
    raw_reference_handoff = read_json(raw_reference_handoff_json)
    reference_intake = read_json(reference_intake_json) if reference_intake_json else None

    failed = []
    failed.extend(collect_true_flag_failures("machine_gate", machine_gate))
    failed.extend(collect_true_flag_failures("minus_audit", minus_audit))
    failed.extend(collect_true_flag_failures("minus_route_expansion", route_expansion))
    failed.extend(collect_true_flag_failures("reference_gap", reference_gap))
    failed.extend(collect_true_flag_failures("raw_reference_handoff", raw_reference_handoff))
    if reference_intake is not None:
        failed.extend(collect_true_flag_failures("reference_intake", reference_intake))
    if machine_gate.get("schema") != "machine_stem_acceptance_gate.v1":
        failed.append("machine_gate_schema_unexpected")
    if minus_audit.get("schema") != "short_minus_quality_audit.v1":
        failed.append("minus_audit_schema_unexpected")
    if reference_gap.get("schema") != "full_pack_reference_corpus_gap_plan.v1":
        failed.append("reference_gap_schema_unexpected")
    if raw_reference_handoff.get("schema") != "full_pack_raw_reference_recovery_material_handoff_manifest.v1":
        failed.append("raw_reference_handoff_schema_unexpected")
    if reference_intake is not None and reference_intake.get("schema") != "full_pack_reference_corpus_intake_manifest.v1":
        failed.append("reference_intake_schema_unexpected")
    if reference_intake is not None and reference_intake.get("failed_checks") not in ([], None):
        failed.append("reference_intake_failed_checks_not_empty")
    summary = route_summary(route_expansion)
    if not summary:
        failed.append("minus_route_expansion_summary_missing")
    failed.extend(collect_summary_true_flag_failures("minus_route_expansion_summary", summary))

    evidence = blocking_evidence(
        machine_gate=machine_gate,
        minus_audit=minus_audit,
        route_expansion=route_expansion,
        reference_gap=reference_gap,
        raw_reference_handoff=raw_reference_handoff,
    )
    candidate_roots = stable_unique([str(item) for item in raw_reference_handoff.get("candidate_roots") or []])
    evidence.update(reference_intake_evidence(reference_intake, candidate_roots))
    action_status, actions, action_failed = build_actions(evidence=evidence, candidate_roots=candidate_roots)
    failed.extend(action_failed)
    failed = stable_unique(failed)
    if failed and action_failed != failed:
        actions = []
        plan_status = "blocked"
    else:
        plan_status = action_status

    manifest = {
        "schema": "full_pack_route_exhaustion_recovery_plan.v1",
        "manifest_type": "full_pack_route_exhaustion_recovery_plan",
        "plan_status": plan_status,
        "failed_checks": failed,
        "action_count": len(actions),
        "actions_jsonl": str(output_dir / ACTIONS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "machine_gate_json": str(machine_gate_json),
        "minus_audit_json": str(minus_audit_json),
        "minus_route_expansion_json": str(minus_route_expansion_json),
        "reference_gap_plan_json": str(reference_gap_plan_json),
        "raw_reference_handoff_json": str(raw_reference_handoff_json),
        "reference_intake_json": str(reference_intake_json or ""),
        "route_space_exhausted": bool(evidence["route_space_exhausted"]),
        "full_pack_required_stems": evidence["full_pack_required_stems"],
        "full_pack_present_stems": evidence["full_pack_present_stems"],
        "full_pack_missing_required_stems": evidence["full_pack_missing_required_stems"],
        "full_pack_train_ready_stems": evidence["full_pack_train_ready_stems"],
        "full_pack_missing_train_ready_stems": evidence["full_pack_missing_train_ready_stems"],
        "accepted_minus_candidates": evidence["accepted_minus_candidates"],
        "minus_candidates_generated": evidence["minus_candidates_generated"],
        "jobs_attempted": evidence["jobs_attempted"],
        "jobs_failed": evidence["jobs_failed"],
        "selected_route_ids": evidence["selected_route_ids"],
        "rejected_route_ids": evidence["rejected_route_ids"],
        "ready_raw_reference_handoff_count": evidence["ready_raw_reference_handoff_count"],
        "blocked_raw_reference_handoff_count": evidence["blocked_raw_reference_handoff_count"],
        "reference_intake_status": evidence["reference_intake_status"],
        "reference_intake_accepted_reference_rows": evidence["reference_intake_accepted_reference_rows"],
        "reference_intake_rejected_reference_rows": evidence["reference_intake_rejected_reference_rows"],
        "reference_intake_rejection_reason_counts": evidence["reference_intake_rejection_reason_counts"],
        "reference_intake_candidate_roots_consumed": evidence["reference_intake_candidate_roots_consumed"],
        "reference_gap_count": evidence["reference_gap_count"],
        "missing_reference_stem_count": evidence["missing_reference_stem_count"],
        "missing_training_job_count": evidence["missing_training_job_count"],
        "delivery_allowed": False,
        "presentation_status": "route_exhaustion_recovery_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "not_claimed": ["accepted_stems", "listen_result", "audio_repaired", "model_promoted"],
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "gpu_inference_skipped": True,
            "training_started": False,
            "not_release_artifact": True,
            "fail_closed_until_machine_gate_accepts": True,
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / PLAN_NAME, manifest)
    write_jsonl(output_dir / ACTIONS_NAME, actions)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", plan_status=plan_status, action_count=len(actions), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a recovery plan after minus route expansion is exhausted.")
    parser.add_argument("--machine-gate-json", type=Path, required=True)
    parser.add_argument("--minus-audit-json", type=Path, required=True)
    parser.add_argument("--minus-route-expansion-json", type=Path, required=True)
    parser.add_argument("--reference-gap-plan-json", type=Path, required=True)
    parser.add_argument("--raw-reference-handoff-json", type=Path, required=True)
    parser.add_argument("--reference-intake-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_route_exhaustion_recovery_plan.py",
        entrypoint_path="/app/tools/build_full_pack_route_exhaustion_recovery_plan.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_route_exhaustion_recovery_plan(
        machine_gate_json=args.machine_gate_json,
        minus_audit_json=args.minus_audit_json,
        minus_route_expansion_json=args.minus_route_expansion_json,
        reference_gap_plan_json=args.reference_gap_plan_json,
        raw_reference_handoff_json=args.raw_reference_handoff_json,
        reference_intake_json=args.reference_intake_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / PLAN_NAME),
                "plan_status": manifest["plan_status"],
                "action_count": manifest["action_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
