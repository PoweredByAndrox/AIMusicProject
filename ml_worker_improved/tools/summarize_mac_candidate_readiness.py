#!/usr/bin/env python3
"""Summarize whether a Mac audio candidate is allowed to be presented as a listen result."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ACCEPTED_PRESENTATION_STATUS = "accepted_for_listen_pack"
LISTEN_FEEDBACK_SCHEMAS = {"short_listen_feedback_manifest.v1"}
VOCAL_LEAK_TAG_MARKERS = (
    "residual_vocal",
    "vocal_leak",
    "leaked_vocal",
    "background_vocal",
    "pitch_shifted_vocal",
)
DETAIL_LOSS_TAG_MARKERS = (
    "detail_loss",
    "missing_detail",
    "muffled",
    "presence_loss",
    "air_detail_loss",
)
PROMISING_HANDOFF_ROUTES = {
    "detail_loss": {
        "route": "judge_guided_minus_air_detail_repair",
        "route_status": "documented_route",
        "tool": "tools/apply_judge_guided_minus_air_detail_repair.py",
        "next_action": "run_judge_guided_minus_air_detail_repair",
    },
    "residual_vocal_leak": {
        "route": "detail_preserving_residual_vocal_cleanup",
        "route_status": "missing_implementation",
        "tool": "",
        "next_action": "run_detail_preserving_residual_vocal_cleanup",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _int_value(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _score_region_count(score: Mapping[str, Any]) -> int:
    if str(score.get("verdict") or "") == "pass":
        return 0
    regions = score.get("regions")
    return _int_value(score.get("total_fail_regions"), score.get("all_region_count"), score.get("selected_region_count"), len(regions) if isinstance(regions, list) else None)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _delivery_contract_passes(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("delivery_status") or "") == "accepted"
        and bool(payload.get("delivery_allowed"))
        and str(payload.get("presentation_status") or "") == ACCEPTED_PRESENTATION_STATUS
    )


def _audibility_audit_failed(audibility_audit: Mapping[str, Any] | None) -> bool:
    if audibility_audit is None:
        return False
    return str(audibility_audit.get("verdict") or "") != "pass"


def _audibility_summary(audibility_audit: Mapping[str, Any] | None) -> dict[str, Any]:
    if audibility_audit is None:
        return {}
    metrics = _mapping_value(audibility_audit, "metrics")
    return {
        "status": str(audibility_audit.get("status") or ""),
        "verdict": str(audibility_audit.get("verdict") or ""),
        "diff_rel_db_vs_source": metrics.get("diff_rel_db_vs_source"),
        "active_changed_sec": metrics.get("active_changed_sec"),
        "max_window_diff_ratio": metrics.get("max_window_diff_ratio"),
    }


def _listen_feedback_issue_categories(issue_tags: list[str]) -> list[str]:
    categories: set[str] = set()
    for raw_tag in issue_tags:
        tag = raw_tag.lower()
        if any(marker in tag for marker in VOCAL_LEAK_TAG_MARKERS):
            categories.add("residual_vocal_leak")
        if any(marker in tag for marker in DETAIL_LOSS_TAG_MARKERS):
            categories.add("detail_loss")
    return sorted(categories)


def _listen_feedback_summary(feedback: Mapping[str, Any] | None, feedback_path: Path | None) -> dict[str, Any]:
    if feedback is None:
        return {"status": "missing", "manifest": "", "blocks_delivery": False, "blocker_reasons": [], "next_actions": []}

    schema = str(feedback.get("schema") or "")
    overall = str(feedback.get("overall_listen_verdict") or "")
    failed_checks = [str(value) for value in feedback.get("failed_checks") or [] if str(value)]
    items = [dict(item) for item in feedback.get("items") or [] if isinstance(item, Mapping)]
    rejected_items = [item for item in items if str(item.get("verdict") or "") == "rejected"]
    issue_tags = sorted({str(tag) for item in rejected_items for tag in item.get("issue_tags") or [] if str(tag)})
    issue_categories = _listen_feedback_issue_categories(issue_tags)

    blockers: list[str] = []
    next_actions: list[str] = []
    if schema not in LISTEN_FEEDBACK_SCHEMAS:
        blockers.append("listen_feedback_schema_invalid")
        next_actions.append("rebuild_listen_feedback_manifest")
    if failed_checks or overall == "invalid":
        blockers.append("listen_feedback_invalid")
        next_actions.append("rebuild_listen_feedback_manifest")
    if bool(feedback.get("delivery_allowed")) or bool(feedback.get("release_ready")) or bool(feedback.get("can_publish_as_result")):
        blockers.append("listen_feedback_contract_violation")
        next_actions.append("rebuild_listen_feedback_manifest")
    if overall == "rejected" or rejected_items:
        blockers.append("human_listen_feedback_rejected")
        next_actions.append("route_human_feedback_to_failure_taxonomy")
    if "residual_vocal_leak" in issue_categories:
        blockers.append("human_feedback_residual_vocal_leak")
        next_actions.append("run_detail_preserving_residual_vocal_cleanup")
    if "detail_loss" in issue_categories:
        blockers.append("human_feedback_detail_loss")
        next_actions.append("run_judge_guided_minus_air_detail_repair")

    if "listen_feedback_invalid" in blockers or "listen_feedback_schema_invalid" in blockers or "listen_feedback_contract_violation" in blockers:
        status = "invalid"
    elif "human_listen_feedback_rejected" in blockers:
        status = "rejected"
    elif overall == "accepted":
        status = "accepted"
    else:
        status = "provided"

    return {
        "status": status,
        "manifest": str(feedback_path or ""),
        "source_schema": schema,
        "overall_listen_verdict": overall,
        "review_coverage_status": str(feedback.get("review_coverage_status") or ""),
        "reviewed_item_count": _int_value(feedback.get("reviewed_item_count")),
        "rejected_item_count": len(rejected_items),
        "issue_tags": issue_tags,
        "issue_categories": issue_categories,
        "failed_checks": failed_checks,
        "blocks_delivery": bool(blockers),
        "blocker_reasons": _unique(blockers),
        "next_actions": _unique(next_actions),
        "contract": {
            "diagnostic_feedback_only": True,
            "does_not_create_listen_result": True,
            "rejected_feedback_blocks_delivery": True,
        },
    }


def _promising_blocked_handoff(listen_feedback_summary: Mapping[str, Any]) -> dict[str, Any]:
    if str(listen_feedback_summary.get("status") or "") != "rejected":
        return {
            "status": "not_applicable",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_only_handoff",
            "issue_categories": [],
            "issue_tags": [],
            "recommended_routes": [],
            "contract": {
                "diagnostic_only": True,
                "does_not_create_listen_result": True,
                "requires_objective_gate_before_delivery": True,
            },
        }
    issue_categories = [str(value) for value in listen_feedback_summary.get("issue_categories") or [] if str(value)]
    routes: list[dict[str, Any]] = []
    for category in issue_categories:
        route = PROMISING_HANDOFF_ROUTES.get(category)
        if route:
            routes.append({"issue_category": category, **route})
    return {
        "status": "promising_but_blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_only_handoff",
        "issue_categories": issue_categories,
        "issue_tags": [str(value) for value in listen_feedback_summary.get("issue_tags") or [] if str(value)],
        "recommended_routes": routes,
        "contract": {
            "diagnostic_only": True,
            "does_not_create_listen_result": True,
            "requires_objective_gate_before_delivery": True,
        },
    }


def _mapping_value(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _track_id(identity: Mapping[str, Any]) -> str:
    return str(identity.get("track_id") or identity.get("source_track_id") or identity.get("id") or "").strip()


def _slug(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def _same_path_text(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return str(left) == str(right)


def _reference_track_identity(reference_audit: Mapping[str, Any] | None) -> dict[str, Any]:
    if reference_audit is None:
        return {}
    identity = _mapping_value(reference_audit, "reference_track_identity") or _mapping_value(reference_audit, "source_track_identity")
    if identity:
        return identity
    track_name = str(reference_audit.get("track_name") or reference_audit.get("title") or "")
    if track_name:
        return {"track_id": _slug(track_name), "title": track_name, "identity_status": "derived_from_reference_track_name"}
    return {}


def _reference_audit_status(reference_audit: Mapping[str, Any] | None, *, source_track_identity: Mapping[str, Any]) -> str:
    if reference_audit is None:
        return "missing"
    source_id = _track_id(source_track_identity)
    reference_identity = _reference_track_identity(reference_audit)
    reference_id = _track_id(reference_identity)
    if source_id and reference_id and source_id != reference_id:
        return "mismatched_to_source"
    verdict = str(reference_audit.get("verdict") or reference_audit.get("status") or "")
    if verdict in {"pass", "passed", "accepted", "exact_reference_passed"}:
        return "passed"
    return "failed"


def _objective_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    before_score = payload.get("before_score") if isinstance(payload.get("before_score"), Mapping) else {}
    after_score = payload.get("after_score") if isinstance(payload.get("after_score"), Mapping) else {}
    guard = _mapping_value(payload, "guard")
    before_count = _int_value(
        payload.get("before_region_count"),
        before_score.get("total_fail_regions"),
        before_score.get("all_region_count"),
        guard.get("score_region_count"),
        guard.get("before_region_count"),
    )
    after_count = _int_value(
        payload.get("after_region_count"),
        after_score.get("total_fail_regions"),
        after_score.get("all_region_count"),
        guard.get("after_repair_region_count"),
        guard.get("after_region_count"),
        _score_region_count(after_score),
    )
    verdict = str(after_score.get("verdict") or "")
    if not verdict:
        verdict = str(guard.get("after_repair_verdict") or guard.get("score_verdict") or "")
    if not verdict:
        verdict = "pass" if before_count > 0 and after_count == 0 else "fail" if after_count > 0 else "unknown"
    return {
        "before_fail_regions": before_count,
        "after_fail_regions": after_count,
        "region_delta": before_count - after_count,
        "pitch_guard_verdict": verdict,
    }


def _status_from_blockers(*, accepted: bool, selection: bool, failed_objective: bool, blockers: list[str]) -> str:
    if accepted:
        return "accepted_for_listen_pack"
    if "audibility_audit_missing" in blockers:
        return "blocked_missing_audibility_audit"
    if "audibility_delta_below_gate" in blockers or "source_candidate_same_path" in blockers:
        return "blocked_inaudible_candidate"
    if "listen_feedback_invalid" in blockers or "listen_feedback_schema_invalid" in blockers or "listen_feedback_contract_violation" in blockers:
        return "blocked_invalid_listen_feedback"
    if "human_listen_feedback_rejected" in blockers:
        return "blocked_human_listen_feedback"
    if "full_holdout_validation_failed" in blockers:
        return "blocked_failed_full_holdout_validation"
    if "reference_bundle_mismatched_to_source" in blockers:
        return "blocked_source_reference_mismatch"
    if failed_objective:
        return "blocked_failed_objective_gate"
    if selection and "full_holdout_not_evaluated" in blockers:
        return "blocked_needs_full_holdout_and_reference"
    if "exact_reference_audit_missing" in blockers:
        return "blocked_missing_reference"
    return "blocked_not_accepted_for_listen_pack"


def summarize_manifest(
    manifest_path: Path,
    *,
    reference_audit_path: Path | None = None,
    audibility_audit_path: Path | None = None,
    listen_feedback_path: Path | None = None,
    full_holdout_validation_path: Path | None = None,
    require_audibility_audit: bool = False,
    source_audio_path_override: str = "",
    source_track_id: str = "",
) -> dict[str, Any]:
    payload = _load_json(manifest_path)
    reference_audit = _load_json(reference_audit_path) if reference_audit_path else None
    audibility_audit = _load_json(audibility_audit_path) if audibility_audit_path else None
    listen_feedback = _load_json(listen_feedback_path) if listen_feedback_path else None
    full_holdout_validation = _load_json(full_holdout_validation_path) if full_holdout_validation_path else None
    audibility_failed = _audibility_audit_failed(audibility_audit)
    audibility_missing = bool(require_audibility_audit and audibility_audit is None)
    listen_feedback_summary = _listen_feedback_summary(listen_feedback, listen_feedback_path)
    listen_feedback_blocks_delivery = bool(listen_feedback_summary.get("blocks_delivery"))
    accepted = _delivery_contract_passes(payload) and not audibility_failed and not audibility_missing and not listen_feedback_blocks_delivery
    schema = str(payload.get("schema") or "")
    presentation_status = str(payload.get("presentation_status") or "")
    delivery_allowed = bool(payload.get("delivery_allowed"))
    listen_pack_created = bool(payload.get("listen_pack_created"))
    failed_checks = [str(value) for value in payload.get("failed_checks") or []]
    selection = schema == "mac_residual_pareto_selection.v1"
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    selected_candidate_id = str(payload.get("selected_candidate_id") or selected.get("candidate_id") or "")
    source_track_identity = _mapping_value(selected, "source_track_identity") or _mapping_value(payload, "source_track_identity")
    if source_track_id:
        source_track_identity = {**source_track_identity, "track_id": source_track_id, "identity_status": str(source_track_identity.get("identity_status") or "provided_override")}
    source_audio_path = str(source_audio_path_override or selected.get("source_audio_path") or payload.get("source_audio_path") or "")
    reference_track_identity = _reference_track_identity(reference_audit)
    reference_source_audio_path = str(reference_audit.get("source_audio_path") or reference_audit.get("reference_audio_path") or "") if reference_audit else ""
    ref_status = _reference_audit_status(reference_audit, source_track_identity=source_track_identity)
    objective = _objective_summary(payload)
    guard = _mapping_value(payload, "guard")
    guard_blockers = [str(value) for value in guard.get("blocker_reasons") or [] if str(value)]

    blockers: list[str] = []
    next_actions: list[str] = []
    if not accepted:
        if not delivery_allowed:
            blockers.append("delivery_not_allowed")
        if not listen_pack_created:
            blockers.append("listen_pack_not_created")
        if presentation_status == "pareto_selection_diagnostic_only":
            blockers.append("selection_diagnostic_only")
        elif presentation_status and presentation_status != ACCEPTED_PRESENTATION_STATUS:
            blockers.append("diagnostic_presentation_status")
        elif not presentation_status:
            blockers.append("presentation_status_missing")

    if audibility_failed:
        blockers.extend([str(value) for value in audibility_audit.get("blocker_reasons") or []] or ["audibility_delta_below_gate"])
        next_actions.append("run_full_region_repair_or_reject_inaudible_candidate")
    elif audibility_missing:
        blockers.append("audibility_audit_missing")
        next_actions.append("run_audibility_delta_audit_for_candidate")
    if listen_feedback_blocks_delivery:
        blockers.extend([str(value) for value in listen_feedback_summary.get("blocker_reasons") or []])
        next_actions.extend([str(value) for value in listen_feedback_summary.get("next_actions") or []])

    selected_full_holdout_status = str(selected.get("full_holdout_resolution_status") or payload.get("full_holdout_resolution_status") or "")
    holdout_validation_status = str(full_holdout_validation.get("status") or "") if full_holdout_validation else "missing"
    holdout_validation_schema = str(full_holdout_validation.get("schema") or "") if full_holdout_validation else ""
    selected_manifest = str(selected.get("manifest") or selected.get("source_manifest") or "")
    holdout_source_candidate_manifest = str(full_holdout_validation.get("source_candidate_manifest") or "") if full_holdout_validation else ""
    holdout_validation_linked = bool(
        full_holdout_validation is not None
        and holdout_validation_schema == "mac_residual_candidate_full_holdout_validation.v1"
        and selected_manifest
        and _same_path_text(holdout_source_candidate_manifest, selected_manifest)
    )
    holdout_validation_passed = bool(
        full_holdout_validation is not None
        and holdout_validation_linked
        and holdout_validation_status == "validation_passed"
        and str(full_holdout_validation.get("full_holdout_resolution_status") or "") == "passed"
        and not bool(full_holdout_validation.get("delivery_allowed"))
        and not bool(full_holdout_validation.get("listen_pack_created"))
    )
    if selection and selected_candidate_id:
        if full_holdout_validation is not None and not holdout_validation_passed:
            blockers.append("full_holdout_validation_failed")
            if holdout_validation_schema != "mac_residual_candidate_full_holdout_validation.v1":
                blockers.append("full_holdout_validation_schema_invalid")
            elif not holdout_validation_linked:
                blockers.append("full_holdout_validation_not_linked_to_selected_candidate")
            next_actions.append("rerun_or_attach_independent_full_holdout_evidence")
        elif selected_full_holdout_status != "passed" and not holdout_validation_passed:
            blockers.append("full_holdout_not_evaluated")
            next_actions.append("run_full_holdout_resolution_for_selected_candidate")
        if str(selected.get("presentation_status") or "") and str(selected.get("presentation_status") or "") != ACCEPTED_PRESENTATION_STATUS:
            blockers.append("selected_candidate_diagnostic_only")

    failed_objective = "pitch_guard_failed" in failed_checks or objective["after_fail_regions"] > 0 or objective["pitch_guard_verdict"] == "fail"
    if failed_objective:
        blockers.extend(failed_checks or ["pitch_guard_failed"])
        blockers.extend(guard_blockers)
        next_actions.append("continue_model_routing_or_training_until_objective_gate_passes")

    if not accepted:
        if ref_status == "missing":
            blockers.append("exact_reference_audit_missing")
            next_actions.append("attach_verified_reference_bundle_for_actual_source")
        elif ref_status == "mismatched_to_source":
            blockers.append("reference_bundle_mismatched_to_source")
            next_actions.append("attach_verified_reference_bundle_for_actual_source")
        elif ref_status == "failed":
            blockers.append("exact_reference_audit_failed")
            next_actions.append("repair_source_identity_or_reference_pairing")

    blockers = _unique(blockers)
    next_actions = _unique(next_actions)
    if not accepted and not next_actions:
        next_actions.append("inspect_blockers_and_continue_repair_loop")

    candidate_ready_for_full_holdout = bool(
        selection
        and selected_candidate_id
        and selected.get("pitch_pass") is True
        and selected.get("damage_pass") is True
        and selected.get("eligible_for_full_holdout", selected.get("eligible_for_promotion")) is True
    )
    return {
        "schema": "mac_candidate_readiness_summary.v1",
        "manifest": str(manifest_path),
        "source_schema": schema,
        "source_status": str(payload.get("status") or ""),
        "source_presentation_status": presentation_status,
        "source_audio_path": source_audio_path,
        "source_track_identity": source_track_identity,
        "reference_track_identity": reference_track_identity,
        "reference_source_audio_path": reference_source_audio_path,
        "selected_candidate_id": selected_candidate_id,
        "candidate_ready_for_full_holdout": candidate_ready_for_full_holdout,
        "reference_bundle_status": ref_status,
        "delivery_allowed": accepted,
        "listen_result_allowed": accepted,
        "status": _status_from_blockers(accepted=accepted, selection=selection, failed_objective=failed_objective, blockers=blockers),
        "blocker_reasons": blockers,
        "next_actions": next_actions,
        "objective_summary": objective,
        "reference_audit_status": ref_status,
        "audibility_audit": _audibility_summary(audibility_audit),
        "audibility_audit_status": str(audibility_audit.get("verdict") or "missing") if audibility_audit is not None else "missing",
        "listen_feedback": listen_feedback_summary,
        "listen_feedback_status": str(listen_feedback_summary.get("status") or "missing"),
        "promising_blocked_handoff": _promising_blocked_handoff(listen_feedback_summary),
        "full_holdout_validation_status": holdout_validation_status,
        "full_holdout_validation": {
            "manifest": str(full_holdout_validation_path) if full_holdout_validation_path else "",
            "source_candidate_manifest": holdout_source_candidate_manifest,
            "full_holdout_resolution_status": str(full_holdout_validation.get("full_holdout_resolution_status") or "") if full_holdout_validation else "",
            "failed_checks": [str(value) for value in full_holdout_validation.get("failed_checks") or []] if full_holdout_validation else [],
            "linked_to_selected_candidate": holdout_validation_linked,
            "passed": holdout_validation_passed,
        },
        "contract": {
            "accepted_requires_delivery_status": "accepted",
            "accepted_requires_delivery_allowed": True,
            "accepted_requires_presentation_status": ACCEPTED_PRESENTATION_STATUS,
            "provided_audibility_audit_must_pass": True,
            "provided_listen_feedback_must_not_reject": True,
            "provided_full_holdout_validation_must_pass": True,
            "required_audibility_audit": bool(require_audibility_audit),
            "diagnostic_artifacts_are_not_listen_results": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path)
    parser.add_argument("--audibility-audit", type=Path)
    parser.add_argument("--listen-feedback", type=Path)
    parser.add_argument("--full-holdout-validation", type=Path)
    parser.add_argument("--require-audibility-audit", action="store_true")
    parser.add_argument("--source-audio-path", default="")
    parser.add_argument("--source-track-id", default="")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_manifest(
        args.manifest,
        reference_audit_path=args.reference_audit,
        audibility_audit_path=args.audibility_audit,
        listen_feedback_path=args.listen_feedback,
        full_holdout_validation_path=args.full_holdout_validation,
        require_audibility_audit=bool(args.require_audibility_audit),
        source_audio_path_override=args.source_audio_path,
        source_track_id=args.source_track_id,
    )
    path = _write_json(args.out / "summary.json", summary)
    print(json.dumps({"summary": str(path), "status": summary["status"], "listen_result_allowed": summary["listen_result_allowed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
