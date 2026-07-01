#!/usr/bin/env python3
"""Audit whether a short listen pack satisfies the active listen-goal contract."""

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

from tools.artifact_name_policy import validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
AUDIT_NAME = "goal_status_audit.json"
SUMMARY_NAME = "goal_status_summary.md"
PROGRESS_NAME = "progress.jsonl"

REQUIRED_6S_STEMS = {"vocal", "minus", "drums", "bass", "other", "guitar", "piano"}
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        "expected_acceleration_mode": "external_only_cpu_contract_audit",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_goal_audit_no_audio_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_goal_audit_no_training",
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


def resolve_artifact_path(pack_root: Path, raw_path: Any, default_name: str) -> Path:
    path = Path(str(raw_path or default_name))
    if not path.is_absolute():
        path = pack_root / path
    return path


def check(status_id: str, status: str, evidence: dict[str, Any] | None = None, blockers: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": status_id,
        "status": status,
        "evidence": evidence or {},
        "blockers": blockers or [],
    }


def collect_true_flag_blockers(prefix: str, payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            if key == "release_ready":
                blockers.append("release_ready_true")
            blockers.append(f"{prefix}_{key}_true")
    for nested_key in ("safety", "contract"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            for key in BLOCKED_TRUE_FLAGS:
                if nested.get(key) is True:
                    blockers.append(f"{prefix}_{nested_key}_{key}_true")
    return blockers


def path_segment_failures(items: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in items:
        audio_path = str(item.get("audio_path") or "")
        if not audio_path or Path(audio_path).is_absolute():
            blockers.append("absolute_or_empty_audio_path")
            continue
        for segment in Path(audio_path).parts:
            try:
                validate_artifact_segment(segment)
            except ValueError:
                blockers.append("long_or_unsafe_audio_path_segment")
                break
    return stable_unique(blockers)


def policy_blockers(pack: dict[str, Any]) -> list[str]:
    policy = pack.get("artifact_name_policy")
    if not isinstance(policy, dict):
        return ["short_name_policy_missing"]
    blockers: list[str] = []
    if policy.get("schema") != "short_windows_safe_artifact_names.v1":
        blockers.append("short_name_policy_schema_unexpected")
    if policy.get("raw_track_or_stem_ids_in_path") is not False:
        blockers.append("raw_ids_allowed_in_path")
    if policy.get("provenance_in_manifest") is not True:
        blockers.append("provenance_not_required_in_manifest")
    return blockers


def provenance_blockers(items: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in items:
        if not str(item.get("source_manifest") or ""):
            blockers.append("source_manifest_missing")
        if not str(item.get("source_audio_path") or ""):
            blockers.append("source_audio_path_missing")
        if not isinstance(item.get("source_row"), dict):
            blockers.append("source_row_missing")
    return stable_unique(blockers)


def delivery_blockers(pack: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    expected = {
        "schema": "aggregate_short_listen_pack_manifest.v1",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "deliverable": False,
    }
    for key, expected_value in expected.items():
        if pack.get(key) != expected_value:
            blockers.append(f"{key}_unexpected")
    if pack.get("failed_checks") not in ([], None):
        blockers.append("pack_failed_checks_not_empty")
    return blockers


def validation_blockers(validation: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if validation.get("validation_status") != "passed":
        blockers.append("aggregate_pack_validation_failed")
    if validation.get("failed_checks") not in ([], None):
        blockers.append("aggregate_pack_validation_failed_checks_not_empty")
    if validation.get("release_ready") is True:
        blockers.append("validation_release_ready_true")
    if validation.get("can_publish_as_result") is True:
        blockers.append("validation_can_publish_as_result_true")
    return blockers


def machine_gate_blockers(
    machine_gate_json: Path | None,
    *,
    pack_manifest_json: Path,
    validation_json: Path,
) -> tuple[dict[str, Any], list[str]]:
    evidence = {"machine_gate_json": str(machine_gate_json) if machine_gate_json else ""}
    if machine_gate_json is None or not machine_gate_json.exists():
        return evidence, ["machine_acceptance_gate_missing"]
    gate = read_json(machine_gate_json)
    evidence.update(
        {
            "gate_status": gate.get("gate_status"),
            "final_result_allowed": gate.get("final_result_allowed"),
            "semantic_review_json": gate.get("semantic_review_json"),
            "semantic_review_evidence_count": gate.get("semantic_review_evidence_count"),
            "minus_audit_json": gate.get("minus_audit_json"),
            "minus_audit_status": gate.get("minus_audit_status"),
            "failed_checks": gate.get("failed_checks") or [],
        }
    )
    blockers: list[str] = []
    if gate.get("schema") != "machine_stem_acceptance_gate.v1":
        blockers.append("machine_acceptance_gate_schema_unexpected")
    if gate.get("pack_manifest_json") and str(gate.get("pack_manifest_json")) != str(pack_manifest_json):
        blockers.append("machine_acceptance_gate_pack_mismatch")
    if gate.get("validation_json") and str(gate.get("validation_json")) != str(validation_json):
        blockers.append("machine_acceptance_gate_validation_mismatch")
    if gate.get("gate_status") != "accepted":
        blockers.append("machine_acceptance_gate_not_accepted")
    if gate.get("final_result_allowed") is not True:
        blockers.append("machine_acceptance_gate_final_result_not_allowed")
    if gate.get("failed_checks") not in ([], None):
        blockers.append("machine_acceptance_gate_failed_checks_not_empty")
    if not str(gate.get("semantic_review_json") or "") or int(gate.get("semantic_review_evidence_count") or 0) <= 0:
        blockers.append("machine_acceptance_gate_semantic_review_missing")
    if not str(gate.get("minus_audit_json") or "") or str(gate.get("minus_audit_status") or "") != "completed":
        blockers.append("machine_acceptance_gate_minus_audit_missing")
    return evidence, blockers


def six_stem_coverage(pack: dict[str, Any], coverage: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    groups = coverage.get("stem_groups")
    if not isinstance(groups, list):
        return ["coverage_groups_missing"], []
    complete_groups: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        stems = {str(stem) for stem in group.get("stem_types") or []}
        if REQUIRED_6S_STEMS.issubset(stems):
            complete_groups.append(
                {
                    "group_alias": str(group.get("group_alias") or ""),
                    "item_count": int(group.get("item_count") or 0),
                    "stem_types": sorted(stems),
                    "semantic_quality_verified": group.get("semantic_quality_verified") is True,
                }
            )
    blockers: list[str] = []
    if len(complete_groups) < 3:
        blockers.append("six_stem_coverage_incomplete")
    for group in complete_groups:
        if group["semantic_quality_verified"] is True:
            blockers.append("six_stem_group_claims_semantic_ground_truth")
    if int(pack.get("item_count") or 0) < 25:
        blockers.append("aggregate_item_count_too_small_for_multi_stem_pack")
    return blockers, complete_groups


def full_vocal_candidate_count(items: list[dict[str, Any]]) -> int:
    return sum(1 for item in items if item.get("stem_type") == "vocal")


def minus_preview_count(items: list[dict[str, Any]]) -> int:
    count = 0
    for item in items:
        stem_type = str(item.get("stem_type") or "")
        source_row = item.get("source_row") if isinstance(item.get("source_row"), dict) else {}
        route_intent = str(source_row.get("route_intent") or item.get("route_intent") or "")
        if stem_type.startswith("minus") or route_intent == "minus_repair":
            count += 1
    return count


def load_human_verdict(path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, ["human_listen_verdict_missing"]
    if not path.exists():
        return None, ["human_listen_verdict_missing"]
    verdict = read_json(path)
    if verdict.get("failed_checks") not in ([], None):
        return verdict, ["human_listen_verdict_failed_checks_not_empty"]
    if verdict.get("release_ready") is True:
        return verdict, ["human_listen_verdict_release_ready_true"]
    if verdict.get("can_publish_as_result") is True:
        return verdict, ["human_listen_verdict_can_publish_as_result_true"]
    if verdict.get("overall_listen_verdict") not in ("accepted", "rejected"):
        return verdict, ["human_listen_verdict_unrecognized"]
    if verdict.get("overall_listen_verdict") != "accepted":
        return verdict, ["human_listen_verdict_rejected"]
    return verdict, []


def rejected_audio_paths(verdict: dict[str, Any] | None) -> list[str]:
    if not isinstance(verdict, dict):
        return []
    explicit = verdict.get("rejected_audio_paths")
    if isinstance(explicit, list):
        return [str(path) for path in explicit if str(path)]
    items = verdict.get("items")
    if not isinstance(items, list):
        return []
    return [str(item.get("audio_path") or "") for item in items if isinstance(item, dict) and item.get("verdict") == "rejected" and str(item.get("audio_path") or "")]


def rejected_items(verdict: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(verdict, dict):
        return []
    items = verdict.get("items")
    if not isinstance(items, list):
        return []
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("verdict") != "rejected":
            continue
        result.append(
            {
                "audio_path": str(item.get("audio_path") or ""),
                "issue_tags": [str(tag) for tag in item.get("issue_tags") or [] if str(tag)],
                "note": str(item.get("note") or ""),
                "source_alias": str(item.get("source_alias") or ""),
                "stem_type": str(item.get("stem_type") or ""),
                "track_alias": str(item.get("track_alias") or ""),
            }
        )
    return result


def issue_tag_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in item.get("issue_tags") or []:
            key = str(tag)
            if key:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def human_feedback_summary(verdict: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(verdict, dict):
        return {
            "overall_listen_verdict": "missing",
            "review_coverage_status": "none",
            "reviewed_item_count": 0,
            "rejected_item_count": 0,
            "rejected_items": [],
            "issue_tag_counts": {},
        }
    rejected = rejected_items(verdict)
    return {
        "overall_listen_verdict": str(verdict.get("overall_listen_verdict") or "unrecognized"),
        "review_coverage_status": str(verdict.get("review_coverage_status") or "unknown"),
        "reviewed_item_count": int(verdict.get("reviewed_item_count") or verdict.get("feedback_item_count") or 0),
        "rejected_item_count": len(rejected),
        "rejected_items": rejected,
        "issue_tag_counts": issue_tag_counts(rejected),
    }


def build_summary(audit: dict[str, Any]) -> str:
    lines = [
        f"# {audit['audit_id']}",
        "",
        f"- goal_status: `{audit['goal_status']}`",
        f"- goal_complete: `{str(audit['goal_complete']).lower()}`",
        f"- release_ready: `{str(audit['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(audit['can_publish_as_result']).lower()}`",
        "",
        "## Checks",
        "",
    ]
    for item in audit["checks"]:
        lines.append(f"- `{item['id']}`: `{item['status']}`")
        if item["blockers"]:
            lines.append(f"  blockers: `{', '.join(item['blockers'])}`")
    lines.extend(["", "## Remaining Blockers", ""])
    for blocker in audit["remaining_blockers"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Not Claimed", ""])
    for claim in audit["not_claimed"]:
        lines.append(f"- `{claim}`")
    lines.append("")
    return "\n".join(lines)


def audit_short_listen_goal_status(
    *,
    pack_manifest_json: Path,
    validation_json: Path,
    output_dir: Path,
    human_listen_verdict_json: Path | None = None,
    machine_gate_json: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        pack_manifest_json=str(pack_manifest_json),
        validation_json=str(validation_json),
        machine_gate_json=str(machine_gate_json or ""),
    )

    pack = read_json(pack_manifest_json)
    pack_root = pack_manifest_json.parent
    validation = read_json(validation_json)
    coverage = read_json(resolve_artifact_path(pack_root, pack.get("coverage_json"), "coverage.json"))
    items = [item for item in pack.get("items") or [] if isinstance(item, dict)]

    checks: list[dict[str, Any]] = []
    validation_failed = validation_blockers(validation)
    checks.append(
        check(
            "aggregate_pack_validation",
            "proved" if not validation_failed else "blocked",
            {"validation_status": validation.get("validation_status"), "failed_checks": validation.get("failed_checks") or []},
            validation_failed,
        )
    )

    delivery_failed = delivery_blockers(pack)
    checks.append(
        check(
            "delivery_contract",
            "proved" if not delivery_failed else "blocked",
            {"delivery_status": pack.get("delivery_status"), "presentation_status": pack.get("presentation_status")},
            delivery_failed,
        )
    )

    short_failed = stable_unique([*policy_blockers(pack), *path_segment_failures(items)])
    checks.append(
        check(
            "short_windows_safe_names",
            "proved" if not short_failed else "not_proved",
            {"item_count": len(items), "artifact_name_policy": pack.get("artifact_name_policy") or {}},
            short_failed,
        )
    )

    provenance_failed = provenance_blockers(items)
    checks.append(
        check(
            "provenance_manifest",
            "proved" if not provenance_failed else "not_proved",
            {"item_count": len(items)},
            provenance_failed,
        )
    )

    six_failed, complete_six_groups = six_stem_coverage(pack, coverage)
    checks.append(
        check(
            "six_stem_coverage",
            "proved" if not six_failed else "not_proved",
            {"complete_group_count": len(complete_six_groups), "required_stems": sorted(REQUIRED_6S_STEMS), "complete_groups": complete_six_groups},
            six_failed,
        )
    )

    vocal_count = full_vocal_candidate_count(items)
    vocal_failed = [] if vocal_count >= 3 else ["full_vocal_candidate_count_too_small"]
    checks.append(check("full_vocal_candidate_coverage", "proved" if not vocal_failed else "not_proved", {"count": vocal_count}, vocal_failed))

    minus_count = minus_preview_count(items)
    minus_failed = [] if minus_count >= 1 else ["minus_preview_missing"]
    checks.append(check("minus_preview_present", "proved" if not minus_failed else "not_proved", {"count": minus_count}, minus_failed))

    safety_failed = stable_unique(
        [
            *collect_true_flag_blockers("pack", pack),
            *collect_true_flag_blockers("coverage", coverage),
            *collect_true_flag_blockers("validation", validation),
        ]
    )
    checks.append(check("safety_contract", "proved" if not safety_failed else "blocked", {}, safety_failed))

    machine_gate_evidence, machine_gate_failed = machine_gate_blockers(
        machine_gate_json,
        pack_manifest_json=pack_manifest_json,
        validation_json=validation_json,
    )
    checks.append(
        check(
            "machine_acceptance_gate",
            "proved" if not machine_gate_failed else "blocked",
            machine_gate_evidence,
            machine_gate_failed,
        )
    )

    human_verdict, human_failed = load_human_verdict(human_listen_verdict_json)
    human_feedback = human_feedback_summary(human_verdict)
    human_evidence = {
        "human_verdict_json": str(human_listen_verdict_json) if human_listen_verdict_json else "",
        "verdict": human_verdict or {},
        "rejected_audio_paths": [item["audio_path"] for item in human_feedback["rejected_items"]],
    }
    checks.append(
        check(
            "human_listen_verdict",
            "proved" if not human_failed else "blocked",
            human_evidence,
            human_failed,
        )
    )

    remaining_blockers = stable_unique(blocker for item in checks for blocker in item["blockers"])
    only_human_missing = remaining_blockers == ["human_listen_verdict_missing"]
    goal_complete = not remaining_blockers
    if goal_complete:
        goal_status = "machine_and_human_accepted"
        goal_status_reason = "all_required_evidence_present"
    elif only_human_missing:
        goal_status = "machine_pass_human_listen_pending"
        goal_status_reason = "needs_human_listen_verdict"
    elif "human_listen_verdict_rejected" in remaining_blockers:
        goal_status = "human_listen_rejected"
        goal_status_reason = "listen_feedback_requires_repair_cycle"
    elif any(item["id"] == "aggregate_pack_validation" and item["status"] == "blocked" for item in checks):
        goal_status = "failed_safety_contract"
        goal_status_reason = "aggregate_pack_validation_failed"
    else:
        goal_status = "machine_evidence_incomplete"
        goal_status_reason = "non_human_machine_blockers_present"

    not_claimed = stable_unique(
        [
            "release_ready",
            "publishable_result",
            "semantic_clean_ground_truth_for_6s_stems",
            "semantic_clean_ground_truth_for_unverified_stems",
            *(coverage.get("not_claimed") if isinstance(coverage.get("not_claimed"), list) else []),
        ]
    )
    audit = {
        "schema": "short_listen_goal_status_audit.v1",
        "audit_id": output_dir.name,
        "pack_manifest_json": str(pack_manifest_json),
        "validation_json": str(validation_json),
        "machine_gate_json": str(machine_gate_json or ""),
        "goal_status": goal_status,
        "goal_status_reason": goal_status_reason,
        "goal_complete": goal_complete,
        "remaining_blockers": remaining_blockers,
        "checks": checks,
        "human_feedback": human_feedback,
        "not_claimed": not_claimed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "audit_artifact_not_for_listen_pack",
        "progress_jsonl": str(progress_path),
        "outputs": {
            "goal_status_audit_json": str(output_dir / AUDIT_NAME),
            "goal_status_summary_md": str(output_dir / SUMMARY_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
    }
    write_json(output_dir / AUDIT_NAME, audit)
    (output_dir / SUMMARY_NAME).write_text(build_summary(audit), encoding="utf-8")
    progress(progress_path, "finish", goal_status=goal_status, goal_complete=goal_complete, remaining_blockers=remaining_blockers)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit active short listen goal status from manifests.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-listen-verdict-json", type=Path)
    parser.add_argument("--machine-gate-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="audit_short_listen_goal_status.py",
        entrypoint_path="/app/tools/audit_short_listen_goal_status.py",
    )
    args = parse_args(argv)
    audit = audit_short_listen_goal_status(
        pack_manifest_json=args.pack_manifest_json,
        validation_json=args.validation_json,
        output_dir=args.output_dir,
        human_listen_verdict_json=args.human_listen_verdict_json,
        machine_gate_json=args.machine_gate_json,
    )
    print(
        json.dumps(
            {
                "audit_path": str(args.output_dir / AUDIT_NAME),
                "goal_status": audit["goal_status"],
                "goal_complete": audit["goal_complete"],
                "remaining_blockers": audit["remaining_blockers"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    success_statuses = {"machine_pass_human_listen_pending", "machine_and_human_accepted"}
    return 0 if audit["goal_status"] in success_statuses else 2


if __name__ == "__main__":
    raise SystemExit(main())
