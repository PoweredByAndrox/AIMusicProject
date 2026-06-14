from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import stable_hash  # noqa: E402


SEVERE_BLOCKERS = ("severe_phase_collapse", "severe_vocal_bleed_in_minus")
FORBIDDEN_CHANGE_CHECK = {
    "thresholds_relaxed": False,
    "cleanup_debleed_denoise": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "python_source_code_modified": False,
    "release_ready_enabled": False,
    "safety_gate_removed": False,
    "blockers_bypassed": False,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def payload_for(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        return payload
    return {}


def route_for_blocker(blocker: str, router_events: list[dict[str, Any]]) -> str:
    for event in router_events:
        payload = payload_for(event)
        if str(payload.get("blocker_type") or "") == blocker:
            return str(payload.get("recommended_route") or "")
    return ""


def evidence_for_blocker(blocker: str, run_events: list[dict[str, Any]], router_events: list[dict[str, Any]], failure_events: list[dict[str, Any]]) -> dict[str, Any]:
    count_key = f"{blocker}_count"
    return {
        "run_roots": sorted({str(event.get("run_root") or "") for event in run_events if event.get("run_root")}),
        "track_ids": sorted({str(event.get("track_id") or "") for event in run_events if event.get("track_id")}),
        "total_blocker_count": int(sum(int(event.get(count_key) or 0) for event in run_events)),
        "failure_reasons": sorted({str(event.get("failure_reason") or "") for event in run_events if event.get("failure_reason")}),
        "router_routes": sorted({str(payload_for(event).get("recommended_route") or "") for event in router_events if str(payload_for(event).get("blocker_type") or "") == blocker}),
        "failed_candidate_count": len(failure_events),
    }


def diagnosis_for(blocker: str, evidence: dict[str, Any]) -> str:
    if blocker == "severe_phase_collapse":
        return "Minus candidates preserve unsafe phase structure across replayed routes; current phase-safe route space did not remove collapse signatures."
    if blocker == "severe_vocal_bleed_in_minus":
        return "Minus candidates still contain vocal bleed after replayed routing; current minus route space is not orthogonal enough to isolate non-vocal content."
    if blocker == "missing_safe_mode_candidate":
        return "Selector could not find a script-safe candidate in replayed outputs; candidate diversity or route ordering is insufficient."
    if blocker == "route_space_exhausted":
        return "Available supported routes were exhausted without producing an accepted candidate; harness should stop or request a different supported route family."
    return f"Replay events repeatedly report {blocker}; harness should treat this as a hard failure signature, not a publishable result."


def recommendation_for(blocker: str, evidence: dict[str, Any]) -> str:
    if blocker in {"severe_phase_collapse", "severe_vocal_bleed_in_minus"} and evidence.get("total_blocker_count", 0) > 0:
        return "add_different_supported_minus_route_or_mark_route_space_exhausted"
    if blocker == "missing_safe_mode_candidate":
        return "increase_safe_candidate_diversity_with_existing_registered_configs"
    if blocker == "route_space_exhausted":
        return "mark_route_space_exhausted_and_stop_without_preview"
    if blocker == "failed_candidate":
        return "prioritize_configs_with_successful_job_manifest_history"
    return "record_failure_signature_and_keep_safety_blockers_hard"


def target_module_for(blocker: str) -> str:
    if blocker in {"severe_phase_collapse", "severe_vocal_bleed_in_minus", "route_space_exhausted"}:
        return "planning.hard_case_router"
    if blocker == "missing_safe_mode_candidate":
        return "planning.candidate_job_scheduler"
    if blocker == "failed_candidate":
        return "tool_use.local_separator_job_runner"
    return "reasoning.failure_reflection"


def mutation_patch_for(blocker: str, recommendation: str) -> dict[str, Any]:
    if recommendation == "add_different_supported_minus_route_or_mark_route_space_exhausted":
        return {
            "mutations": {"hard_case_route_order": "prefer_distinct_supported_minus_routes"},
            "stage_graph_policy": "keep_pair_compatibility_and_post_render_judge_required",
        }
    if recommendation == "increase_safe_candidate_diversity_with_existing_registered_configs":
        return {
            "mutations": {"candidate_diversity_profile": "balanced", "max_candidates": "increase_within_runtime_budget"},
            "stage_graph_policy": "keep_safety_gate_required",
        }
    if recommendation == "mark_route_space_exhausted_and_stop_without_preview":
        return {
            "mutations": {"hard_case_route_order": "mark_route_space_exhausted"},
            "stage_graph_policy": "stop_before_preview_render_when_no_safe_pair",
        }
    if recommendation == "prioritize_configs_with_successful_job_manifest_history":
        return {
            "mutations": {"job_priority_order": "prefer_successful_existing_configs"},
            "stage_graph_policy": "keep_all_required_nodes",
        }
    return {"mutations": {"window_selection_policy": "record_failure_signature"}, "stage_graph_policy": "no_safety_bypass"}


def make_reflection(*, blocker: str, failed_route: str, evidence: dict[str, Any]) -> dict[str, Any]:
    recommendation = recommendation_for(blocker, evidence)
    return {
        "failure_id": "failure_" + stable_hash({"blocker": blocker, "failed_route": failed_route, "evidence": evidence})[:16],
        "harness_candidate_id": "",
        "parent_id": "",
        "blocker_type": blocker,
        "failed_route": failed_route,
        "evidence": evidence,
        "natural_language_diagnosis": diagnosis_for(blocker, evidence),
        "recommended_harness_change": recommendation,
        "target_module": target_module_for(blocker),
        "target_stage_graph_policy": mutation_patch_for(blocker, recommendation)["stage_graph_policy"],
        "candidate_mutation_patch": mutation_patch_for(blocker, recommendation)["mutations"],
        "forbidden_change_check": dict(FORBIDDEN_CHANGE_CHECK),
        "release_ready": False,
    }


def reflection_blockers(events: list[dict[str, Any]]) -> set[str]:
    blockers: set[str] = set()
    for event in events:
        if event.get("event_type") != "run_summary":
            continue
        for blocker in SEVERE_BLOCKERS:
            if int(event.get(f"{blocker}_count") or 0) > 0:
                blockers.add(blocker)
        if str(event.get("failure_reason") or "") == "missing_safe_mode_candidate":
            blockers.add("missing_safe_mode_candidate")
        if str(event.get("failure_reason") or "") == "minus_hard_case_route_exhausted":
            blockers.add("route_space_exhausted")
    if any(event.get("event_type") == "candidate_wave_failures" for event in events):
        blockers.add("failed_candidate")
    return blockers


def write_summary(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    route_exhausted = sum(1 for row in rows if row.get("blocker_type") == "route_space_exhausted")
    lines = [
        "# No-Reference Failure Reflections",
        "",
        f"- replay_bundle: {manifest.get('bundle_id') or 'unknown'}",
        f"- reflection_count: {len(rows)}",
        f"- route_space_exhausted: {route_exhausted}",
        f"- release_ready: false",
        f"- forbidden_threshold_relaxation: false",
        f"- forbidden_cleanup_debleed_denoise: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_failure_reflections(*, replay_bundle_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(replay_bundle_dir / "replay_bundle_manifest.json")
    events = read_jsonl(replay_bundle_dir / "replay_bundle_events.jsonl")
    run_events = [event for event in events if event.get("event_type") == "run_summary"]
    router_events = [event for event in events if event.get("event_type") == "minus_hard_case_router"]
    failure_events = [event for event in events if event.get("event_type") == "candidate_wave_failures"]
    rows = [
        make_reflection(
            blocker=blocker,
            failed_route=route_for_blocker(blocker, router_events),
            evidence=evidence_for_blocker(blocker, run_events, router_events, failure_events),
        )
        for blocker in sorted(reflection_blockers(events))
    ]
    write_jsonl(output_dir / "failure_reflections.jsonl", rows)
    write_summary(output_dir / "failure_reflections_summary.md", rows, manifest)
    return {
        "reflection_count": len(rows),
        "failure_reflections_jsonl": str(output_dir / "failure_reflections.jsonl"),
        "failure_reflections_summary_md": str(output_dir / "failure_reflections_summary.md"),
        "release_ready": False,
        **FORBIDDEN_CHANGE_CHECK,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic failure reflections from no-reference replay bundle events.")
    parser.add_argument("--replay-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_failure_reflections(replay_bundle_dir=args.replay_bundle_dir, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
