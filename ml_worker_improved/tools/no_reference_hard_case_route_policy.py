from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import iter_jsonl, stable_unique  # noqa: E402
from tools.no_reference_candidate_job_registry import (  # noqa: E402
    CandidateJobConfig,
    build_candidate_job_registry,
)


PREFERRED_CONDITIONAL_ROUTE_ID = "local_registered__htdemucs_6s__both__shared"
MEASUREMENT_SAFE_ROUTE_IDS = {PREFERRED_CONDITIONAL_ROUTE_ID}
HARD_CASE_BLOCKERS = {
    "missing_safe_mode_candidate",
    "severe_vocal_bleed_in_minus",
    "severe_phase_collapse",
    "severe_phase_comb_filtering",
}
MATRIX_TRIGGER_ROUTE = {
    "missing_safe_mode_candidate": "generate_minus_focused_variants",
    "severe_vocal_bleed_in_minus": "generate_minus_focused_variants",
    "severe_phase_collapse": "generate_phase_safe_variants",
    "severe_phase_comb_filtering": "generate_phase_safe_variants",
}
MATRIX_TRIGGER_PRIORITY = (
    "severe_phase_collapse",
    "severe_phase_comb_filtering",
    "severe_vocal_bleed_in_minus",
    "missing_safe_mode_candidate",
)
MATRIX_GENERATE_NEW_CANDIDATE_ACTIONS = {
    "blocked_generate_or_select_better_minus_candidate",
}
MATRIX_SELF_BLOCKER_ACTIONS = {
    "blocked_generate_or_select_better_minus_candidate",
    "run_segment_router_for_self_blockers",
}
VOCAL_LEAKAGE_PROXY_ISSUES = {
    "vocal_projection_proxy",
    "vocal_stem_corr_proxy",
    "vocal_leakage_proxy",
    "vocal_bleed_proxy",
}
PHASE_PROXY_ISSUES = {
    "phase_or_stereo_inconsistency",
    "phase_consistency_proxy",
    "phase_collapse_proxy",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in iter_jsonl(path) if isinstance(row, dict)]


def normalize_trigger(reason: str) -> str:
    text = str(reason or "").strip()
    if text == "phase_comb_filtering":
        return "severe_phase_comb_filtering"
    return text


def row_values(row: dict[str, Any], keys: Iterable[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item or "").strip())
        elif isinstance(value, dict):
            values.extend(str(item) for item in value if str(item or "").strip())
        elif value:
            values.append(str(value))
    return values


def is_accepted_minus(row: dict[str, Any]) -> bool:
    decisions = row_values(
        row,
        (
            "minus_decision",
            "decision",
            "script_decision",
            "final_guarded_decision",
            "judge_decision",
            "status",
        ),
    )
    accepted = any(value in {"accept_candidate", "accepted", "accepted_for_listen_pack"} for value in decisions)
    blockers = row_values(row, ("minus_blockers", "blockers", "script_blockers", "failure_reasons"))
    return accepted and not blockers and row.get("release_ready") is not True


def accepted_minus_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in rows if is_accepted_minus(row))


def trigger_reasons_for(run_dir: Path) -> list[str]:
    run_manifest = read_json(run_dir / "run_manifest.json")
    minus_rows = read_jsonl(run_dir / "minus_windowed_judge.jsonl")
    candidate_rows = read_jsonl(run_dir / "candidate_pool.jsonl")
    selected_rows = read_jsonl(run_dir / "selected_candidates.jsonl")
    expansion = read_json(run_dir / "minus_route_expansion_manifest.json")
    expansion_summary = expansion.get("summary") if isinstance(expansion.get("summary"), dict) else {}

    triggers: list[str] = []
    for reason in row_values(run_manifest, ("failure_reason", "safety_blockers")):
        normalized = normalize_trigger(reason)
        if normalized in HARD_CASE_BLOCKERS:
            triggers.append(normalized)
    for row in minus_rows:
        for reason in row_values(
            row,
            (
                "minus_blockers",
                "blockers",
                "failure_reasons",
                "minus_dominant_defect",
                "dominant_defect",
                "failure_reason",
                "judge_reason",
            ),
        ):
            normalized = normalize_trigger(reason)
            if normalized in HARD_CASE_BLOCKERS:
                triggers.append(normalized)

    accepted_minus = accepted_minus_count(minus_rows)
    if minus_rows and accepted_minus == 0:
        triggers.append("no_accepted_minus_candidates")
    if not selected_rows and str(run_manifest.get("output_status") or "") == "needs_more_candidates":
        triggers.append("missing_safe_mode_candidate")

    route_families = {
        str(row.get("route_family") or "").strip()
        for row in candidate_rows
        if (row.get("can_use_for_minus") is not False) and str(row.get("route_family") or "").strip()
    }
    if candidate_rows and accepted_minus == 0 and len(route_families) <= 1:
        triggers.append("only_one_minus_route_family_used")

    route_space_limited = (
        expansion_summary.get("route_space_limited") is True
        or run_manifest.get("route_space_limited") is True
        or (isinstance(run_manifest.get("candidate_wave_summary"), dict) and run_manifest["candidate_wave_summary"].get("route_space_limited") is True)
    )
    if route_space_limited and (accepted_minus == 0 or str(run_manifest.get("output_status") or "") in {"needs_more_candidates", "blocked_hard_case"}):
        triggers.append("route_space_limited")

    return stable_unique(triggers)


def issue_names_from(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key, count in value.items() if count}
    if isinstance(value, list):
        return {str(item) for item in value if str(item or "").strip()}
    if value:
        return {str(value)}
    return set()


def matrix_paths_from(candidate_score_matrix_json: Path | Sequence[Path] | None) -> list[Path]:
    if candidate_score_matrix_json is None:
        return []
    if isinstance(candidate_score_matrix_json, (str, Path)):
        return [Path(candidate_score_matrix_json)]
    return [Path(path) for path in candidate_score_matrix_json]


def iter_matrix_rows(candidate_score_matrix_json: Path | Sequence[Path] | None) -> Iterable[tuple[dict[str, Any], list[dict[str, Any]]]]:
    for path in matrix_paths_from(candidate_score_matrix_json):
        if not path.exists():
            continue
        payload = read_json(path)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        yield summary, [row for row in rows if isinstance(row, dict)]


def matrix_trigger_reasons_for(candidate_score_matrix_json: Path | Sequence[Path] | None) -> list[str]:
    triggers: list[str] = []
    for summary, rows in iter_matrix_rows(candidate_score_matrix_json):
        top_next_action = str(summary.get("top_next_action") or "")
        if top_next_action in MATRIX_GENERATE_NEW_CANDIDATE_ACTIONS:
            triggers.append("missing_safe_mode_candidate")
        for row in rows:
            next_action = str(row.get("next_action") or "")
            if next_action in MATRIX_GENERATE_NEW_CANDIDATE_ACTIONS:
                triggers.append("missing_safe_mode_candidate")
            result_judge = row.get("result_judge") if isinstance(row.get("result_judge"), dict) else {}
            for reason in row_values(result_judge, ("minus_blockers", "blockers", "failure_reasons", "reasons")):
                normalized = normalize_trigger(reason)
                if normalized in HARD_CASE_BLOCKERS:
                    triggers.append(normalized)
            self_judge = row.get("self_judge") if isinstance(row.get("self_judge"), dict) else {}
            issue_names = issue_names_from(row.get("issue_counts")) | issue_names_from(self_judge.get("issue_counts"))
            self_status = str(self_judge.get("status") or row.get("self_judge_status") or "")
            if next_action in MATRIX_SELF_BLOCKER_ACTIONS or self_status in {"proxy_blocked", "blocked"}:
                if issue_names & VOCAL_LEAKAGE_PROXY_ISSUES:
                    triggers.append("severe_vocal_bleed_in_minus")
                if issue_names & PHASE_PROXY_ISSUES:
                    triggers.append("severe_phase_collapse")
    return stable_unique(triggers)


def matrix_failed_model_names_for(candidate_score_matrix_json: Path | Sequence[Path] | None) -> list[str]:
    failed: list[str] = []
    for _, rows in iter_matrix_rows(candidate_score_matrix_json):
        for row in rows:
            if row.get("delivery_allowed") is True or row.get("release_ready") is True:
                continue
            next_action = str(row.get("next_action") or "")
            if next_action not in MATRIX_SELF_BLOCKER_ACTIONS and next_action not in MATRIX_GENERATE_NEW_CANDIDATE_ACTIONS:
                continue
            model_name = str(row.get("model_name") or "").strip()
            if model_name:
                failed.append(model_name)
    return stable_unique(failed)


def previous_quality_route_ids(run_dir: Path) -> list[str]:
    route_ids: list[str] = []
    for path in (
        run_dir / "minus_route_expansion_manifest.json",
        run_dir / "stem2_validation.json",
        APP_ROOT.parent / "lab_artifacts" / "no_reference_minus_route_expansion" / "stem2_validation.json",
    ):
        payload = read_json(path)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
        values = summary.get("quality_improved_route_ids") if isinstance(summary, dict) else []
        if isinstance(values, list):
            route_ids.extend(str(value) for value in values if str(value or "").strip())
    return stable_unique(route_ids)


def route_available(route_id: str, registry_configs: Iterable[CandidateJobConfig]) -> bool:
    for config in registry_configs:
        if route_id not in {str(config.route_id), str(config.candidate_config_id)}:
            continue
        if config.available is not True:
            return False
        if config.entrypoint_type == "unavailable":
            return False
        if config.produces_minus is not True:
            return False
        if config.can_run_without_download is not True:
            return False
        if config.can_run_without_registry_edit is not True:
            return False
        if str(config.forbidden_reason or "").strip():
            return False
        return True
    return False


def config_route_available(config: CandidateJobConfig) -> bool:
    return (
        config.available is True
        and config.entrypoint_type != "unavailable"
        and config.produces_minus is True
        and config.can_run_without_download is True
        and config.can_run_without_registry_edit is True
        and not str(config.forbidden_reason or "").strip()
    )


def pair_capable_route_id(
    config: CandidateJobConfig,
    registry_configs: Iterable[CandidateJobConfig],
    *,
    failed_model_names: Iterable[str] = (),
) -> str:
    failed_models = set(failed_model_names)
    if config.produces_both is True and config.produces_vocal is True:
        return str(config.route_id or config.candidate_config_id)
    for other in sorted(registry_configs, key=lambda item: str(item.route_id or item.candidate_config_id)):
        if str(other.source_model_id or "") != str(config.source_model_id or ""):
            continue
        if str(other.source_model_id or "") in failed_models:
            continue
        if other.produces_both is not True or other.produces_vocal is not True:
            continue
        if config_route_available(other):
            return str(other.route_id or other.candidate_config_id)
    return str(config.route_id or config.candidate_config_id)


def matrix_selected_route_ids(
    triggers: Iterable[str],
    registry_configs: Iterable[CandidateJobConfig],
    *,
    failed_model_names: Iterable[str] = (),
) -> list[str]:
    trigger_set = set(triggers)
    failed_models = set(failed_model_names)
    for trigger in MATRIX_TRIGGER_PRIORITY:
        if trigger not in trigger_set:
            continue
        hard_case_route = MATRIX_TRIGGER_ROUTE.get(trigger)
        if not hard_case_route:
            continue
        matching_configs = sorted(
            (
                config
                for config in registry_configs
                if hard_case_route in set(config.hard_case_routes)
                and config_route_available(config)
                and str(config.source_model_id or "") not in failed_models
            ),
            key=lambda item: str(item.route_id or item.candidate_config_id),
        )
        matches = stable_unique(
            pair_capable_route_id(config, registry_configs, failed_model_names=failed_models)
            for config in matching_configs
        )
        if matches:
            return matches
    return []


def build_hard_case_route_policy(
    *,
    run_dir: Path,
    candidate_score_matrix_json: Path | Sequence[Path] | None = None,
    registry_configs: Iterable[CandidateJobConfig] | None = None,
    conditional_route_policy: str = "disabled",
) -> dict[str, Any]:
    run_manifest = read_json(run_dir / "run_manifest.json")
    candidate_rows = read_jsonl(run_dir / "candidate_pool.jsonl")
    track_id = str(run_manifest.get("track_id") or "")
    if not track_id and candidate_rows:
        track_id = str(candidate_rows[0].get("track_id") or "")
    if not track_id:
        track_id = Path(str(run_manifest.get("input_audio") or run_dir.name)).stem

    matrix_paths = matrix_paths_from(candidate_score_matrix_json)
    matrix_triggers = matrix_trigger_reasons_for(matrix_paths)
    matrix_failed_model_names = matrix_failed_model_names_for(matrix_paths)
    triggers = stable_unique([*trigger_reasons_for(run_dir), *matrix_triggers])
    hard_case_detected = bool(triggers)
    registry_profile = "exploration" if matrix_triggers else "balanced"
    configs = (
        list(registry_configs)
        if registry_configs is not None
        else build_candidate_job_registry(profile=registry_profile)
    )
    quality_routes = set(previous_quality_route_ids(run_dir))
    measurement_safe_routes = quality_routes | MEASUREMENT_SAFE_ROUTE_IDS
    matrix_route_ids = matrix_selected_route_ids(
        matrix_triggers,
        configs,
        failed_model_names=matrix_failed_model_names,
    )

    selected_route_ids: list[str] = []
    route_allowed = False
    route_scope = "disabled"
    route_reason = ""
    route_block_reason = ""

    if conditional_route_policy == "disabled":
        route_block_reason = "conditional_route_policy_disabled"
    elif conditional_route_policy != "hard_case_only":
        route_block_reason = "unsupported_conditional_route_policy"
    elif not hard_case_detected:
        route_block_reason = "no_hard_case_trigger"
    elif matrix_triggers and matrix_route_ids:
        selected_route_ids = [matrix_route_ids[0]]
        route_allowed = True
        route_scope = "hard_case_only"
        route_reason = "matrix_hard_case_route_for_candidate_score_blocker"
    elif matrix_triggers and matrix_failed_model_names:
        route_block_reason = "matrix_hard_case_routes_already_failed_or_unavailable"
    elif PREFERRED_CONDITIONAL_ROUTE_ID not in measurement_safe_routes:
        route_block_reason = "conditional_route_not_stem2_quality_improved_or_measurement_safe"
    elif not route_available(PREFERRED_CONDITIONAL_ROUTE_ID, configs):
        route_block_reason = "conditional_route_not_available_locally"
    else:
        selected_route_ids = [PREFERRED_CONDITIONAL_ROUTE_ID]
        route_allowed = True
        route_scope = "hard_case_only"
        route_reason = "stem2_quality_improved_route_for_hard_case"

    return {
        "run_id": str(run_manifest.get("run_id") or run_dir.name),
        "track_id": track_id,
        "hard_case_detected": hard_case_detected,
        "trigger_reasons": triggers,
        "matrix_candidate_score_evidence_used": any(path.exists() for path in matrix_paths),
        "matrix_candidate_score_trigger_reasons": matrix_triggers,
        "matrix_candidate_score_selected_route_ids": matrix_route_ids,
        "matrix_candidate_score_failed_model_names": matrix_failed_model_names,
        "selected_conditional_route_ids": selected_route_ids,
        "route_scope": route_scope,
        "route_reason": route_reason,
        "route_allowed": route_allowed,
        "route_block_reason": route_block_reason,
        "registry_profile": registry_profile,
        "default_profile_changed": False,
        "release_ready": False,
    }


def write_policy_summary(path: Path, policy: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Hard-Case Route Policy",
        "",
        f"- run_id: {policy.get('run_id') or ''}",
        f"- track_id: {policy.get('track_id') or ''}",
        f"- hard_case_detected: {str(policy.get('hard_case_detected') is True).lower()}",
        f"- trigger_reasons: {', '.join(policy.get('trigger_reasons') or [])}",
        f"- matrix_candidate_score_evidence_used: {str(policy.get('matrix_candidate_score_evidence_used') is True).lower()}",
        f"- matrix_candidate_score_trigger_reasons: {', '.join(policy.get('matrix_candidate_score_trigger_reasons') or [])}",
        f"- matrix_candidate_score_selected_route_ids: {', '.join(policy.get('matrix_candidate_score_selected_route_ids') or [])}",
        f"- matrix_candidate_score_failed_model_names: {', '.join(policy.get('matrix_candidate_score_failed_model_names') or [])}",
        f"- registry_profile: {policy.get('registry_profile') or ''}",
        f"- selected_conditional_route_ids: {', '.join(policy.get('selected_conditional_route_ids') or [])}",
        f"- route_scope: {policy.get('route_scope') or ''}",
        f"- route_reason: {policy.get('route_reason') or ''}",
        f"- route_allowed: {str(policy.get('route_allowed') is True).lower()}",
        f"- route_block_reason: {policy.get('route_block_reason') or ''}",
        "- default_profile_changed: false",
        "- release_ready: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hard_case_route_policy(
    *,
    run_dir: Path,
    output_json: Path,
    summary_md: Path,
    candidate_score_matrix_json: Path | Sequence[Path] | None = None,
    registry_configs: Iterable[CandidateJobConfig] | None = None,
    conditional_route_policy: str = "disabled",
) -> dict[str, Any]:
    policy = build_hard_case_route_policy(
        run_dir=run_dir,
        candidate_score_matrix_json=candidate_score_matrix_json,
        registry_configs=registry_configs,
        conditional_route_policy=conditional_route_policy,
    )
    write_json(output_json, policy)
    write_policy_summary(summary_md, policy)
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a no-reference conditional hard-case route policy artifact.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--candidate-score-matrix-json", type=Path, action="append", default=[])
    parser.add_argument("--conditional-route-policy", choices=("hard_case_only", "disabled"), default="disabled")
    args = parser.parse_args(argv)
    policy = run_hard_case_route_policy(
        run_dir=args.run_dir,
        output_json=args.output_json,
        summary_md=args.summary_md,
        candidate_score_matrix_json=args.candidate_score_matrix_json,
        conditional_route_policy=args.conditional_route_policy,
    )
    print(json.dumps({"route_allowed": policy["route_allowed"], "release_ready": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
