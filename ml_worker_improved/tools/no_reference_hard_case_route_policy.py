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


def build_hard_case_route_policy(
    *,
    run_dir: Path,
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

    triggers = trigger_reasons_for(run_dir)
    hard_case_detected = bool(triggers)
    configs = list(registry_configs) if registry_configs is not None else build_candidate_job_registry(profile="balanced")
    quality_routes = set(previous_quality_route_ids(run_dir))
    measurement_safe_routes = quality_routes | MEASUREMENT_SAFE_ROUTE_IDS

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
        "selected_conditional_route_ids": selected_route_ids,
        "route_scope": route_scope,
        "route_reason": route_reason,
        "route_allowed": route_allowed,
        "route_block_reason": route_block_reason,
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
    registry_configs: Iterable[CandidateJobConfig] | None = None,
    conditional_route_policy: str = "disabled",
) -> dict[str, Any]:
    policy = build_hard_case_route_policy(
        run_dir=run_dir,
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
    parser.add_argument("--conditional-route-policy", choices=("hard_case_only", "disabled"), default="disabled")
    args = parser.parse_args(argv)
    policy = run_hard_case_route_policy(
        run_dir=args.run_dir,
        output_json=args.output_json,
        summary_md=args.summary_md,
        conditional_route_policy=args.conditional_route_policy,
    )
    print(json.dumps({"route_allowed": policy["route_allowed"], "release_ready": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
