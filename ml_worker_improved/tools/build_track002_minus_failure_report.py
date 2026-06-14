from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.analyze_minus_window_blockers import SAFETY_FLAGS, read_json, write_json  # noqa: E402
from tools.build_no_reference_ml_dataset import iter_jsonl  # noqa: E402


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return list(iter_jsonl(path))


def choose_recommended_next_step(decision: dict[str, Any]) -> str:
    if decision["normalization_issue_suspected"]:
        return "audit_shape_normalization_contract"
    if decision["segment_blend_candidate"]:
        return "plan_segment_level_minus_repair"
    if decision["route_space_exhausted"]:
        return "blocked_no_safe_local_route_available"
    if decision["route_specific_failure"]:
        return "inspect_route_specific_candidate_config"
    if decision["local_route_candidate_needed"]:
        return "design_local_route_candidate_without_download"
    return "blocked_no_safe_action"


def build_decision(profile_summary: dict[str, Any], comparison_summary: dict[str, Any]) -> dict[str, Any]:
    route_space_exhausted = comparison_summary.get("route_space_exhausted") is True
    segment_blend_candidate = comparison_summary.get("segment_blend_possible") is True
    normalization_issue_suspected = (
        comparison_summary.get("normalization_issue_suspected") is True
        or profile_summary.get("normalization_correlated") is True
    )
    any_local_improvement = any(
        comparison_summary.get(key) is True
        for key in (
            "any_route_has_clean_window",
            "any_route_reduces_phase_collapse",
            "any_route_reduces_vocal_bleed",
            "segment_blend_possible",
        )
    )
    unsupported_new_model_needed = bool(route_space_exhausted or not any_local_improvement)
    route_specific_failure = bool(profile_summary.get("route_specific"))
    local_route_candidate_needed = bool(
        not segment_blend_candidate
        and not route_space_exhausted
        and not normalization_issue_suspected
        and not route_specific_failure
    )
    decision = {
        "segment_blend_candidate": segment_blend_candidate,
        "local_route_candidate_needed": local_route_candidate_needed,
        "route_space_exhausted": route_space_exhausted,
        "normalization_issue_suspected": normalization_issue_suspected,
        "judge_threshold_issue_suspected": False,
        "input_track_hard_case": bool(profile_summary.get("severe_phase_collapse_total", 0) or profile_summary.get("severe_vocal_bleed_in_minus_total", 0)),
        "unsupported_new_model_needed": unsupported_new_model_needed,
        "unsupported_new_model_needed_execution": False,
        "blocked_no_safe_action": bool(route_space_exhausted and unsupported_new_model_needed),
        "route_specific_failure": route_specific_failure,
        "recommend_threshold_relaxation": False,
        "recommend_cleanup_debleed_denoise": False,
        "new_model_download_started": False,
        "can_publish_as_result": False,
        "release_ready": False,
        **SAFETY_FLAGS,
    }
    decision["recommended_next_step"] = choose_recommended_next_step(decision)
    return decision


def decision_category_for(decision: dict[str, Any]) -> str:
    for key in (
        "segment_blend_candidate",
        "normalization_issue_suspected",
        "route_space_exhausted",
        "route_specific_failure",
        "local_route_candidate_needed",
        "unsupported_new_model_needed",
        "input_track_hard_case",
    ):
        if decision.get(key) is True:
            return key
    return "blocked_no_safe_action"


def build_failure_report(forensics_dir: Path) -> dict[str, Any]:
    clusters = read_json(forensics_dir / "minus_window_blocker_clusters.json")
    profile_summary = clusters.get("summary") if isinstance(clusters.get("summary"), dict) else {}
    comparison_summary = read_json(forensics_dir / "minus_route_window_comparison_summary.json")
    comparison_rows = read_jsonl_if_exists(forensics_dir / "minus_route_window_comparison.jsonl")
    decision = build_decision(profile_summary, comparison_summary)
    report = {
        "report_type": "track002_minus_failure_forensics_v1",
        "forensics_dir": str(forensics_dir),
        "candidates_analyzed": int(profile_summary.get("candidates_analyzed") or 0),
        "routes_analyzed": int(comparison_summary.get("routes_analyzed") or profile_summary.get("routes_analyzed") or 0),
        "windows_analyzed": int(comparison_summary.get("windows_analyzed") or profile_summary.get("windows_analyzed") or 0),
        "severe_phase_collapse_total": int(profile_summary.get("severe_phase_collapse_total") or 0),
        "severe_vocal_bleed_in_minus_total": int(profile_summary.get("severe_vocal_bleed_in_minus_total") or 0),
        "severe_phase_comb_filtering_total": int(profile_summary.get("severe_phase_comb_filtering_total") or 0),
        "blocker_signatures": list(profile_summary.get("blocker_signatures") or []),
        "all_routes_same_blocker_signature": comparison_summary.get("all_routes_same_blocker_signature") is True,
        "any_route_has_clean_window": comparison_summary.get("any_route_has_clean_window") is True,
        "any_route_reduces_phase_collapse": comparison_summary.get("any_route_reduces_phase_collapse") is True,
        "any_route_reduces_vocal_bleed": comparison_summary.get("any_route_reduces_vocal_bleed") is True,
        "segment_blend_possible": comparison_summary.get("segment_blend_possible") is True,
        "segment_blend_all_windows_covered": comparison_summary.get("segment_blend_all_windows_covered") is True,
        "normalization_issue_suspected": decision["normalization_issue_suspected"],
        "route_space_exhausted": decision["route_space_exhausted"],
        "unsupported_new_model_needed": decision["unsupported_new_model_needed"],
        "recommended_next_step": decision["recommended_next_step"],
        "decision_category": decision_category_for(decision),
        "decision": decision,
        "route_comparison_rows": comparison_rows,
        "input_artifacts": profile_summary.get("input_artifacts") if isinstance(profile_summary.get("input_artifacts"), dict) else {},
        "can_publish_as_result": False,
        "release_ready": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        **SAFETY_FLAGS,
    }
    return report


def build_decision_payload(report: dict[str, Any]) -> dict[str, Any]:
    decision = dict(report["decision"])
    return {
        "decision_category": report["decision_category"],
        "recommended_next_step": report["recommended_next_step"],
        "candidates_analyzed": report["candidates_analyzed"],
        "routes_analyzed": report["routes_analyzed"],
        "windows_analyzed": report["windows_analyzed"],
        "all_routes_same_blocker_signature": report["all_routes_same_blocker_signature"],
        "any_route_has_clean_window": report["any_route_has_clean_window"],
        "segment_blend_possible": report["segment_blend_possible"],
        "route_space_exhausted": report["route_space_exhausted"],
        "unsupported_new_model_needed": report["unsupported_new_model_needed"],
        "unsupported_new_model_needed_execution": False,
        "can_publish_as_result": False,
        "release_ready": False,
        **decision,
    }


def build_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track002 Minus Failure Report",
        "",
        f"- candidates_analyzed: {report['candidates_analyzed']}",
        f"- routes_analyzed: {report['routes_analyzed']}",
        f"- windows_analyzed: {report['windows_analyzed']}",
        f"- severe_phase_collapse_total: {report['severe_phase_collapse_total']}",
        f"- severe_vocal_bleed_in_minus_total: {report['severe_vocal_bleed_in_minus_total']}",
        f"- blocker_signatures: {', '.join(report.get('blocker_signatures') or [])}",
        f"- all_routes_same_blocker_signature: {str(report['all_routes_same_blocker_signature']).lower()}",
        f"- any_route_has_clean_window: {str(report['any_route_has_clean_window']).lower()}",
        f"- any_route_reduces_phase_collapse: {str(report['any_route_reduces_phase_collapse']).lower()}",
        f"- any_route_reduces_vocal_bleed: {str(report['any_route_reduces_vocal_bleed']).lower()}",
        f"- segment_blend_possible: {str(report['segment_blend_possible']).lower()}",
        f"- segment_blend_all_windows_covered: {str(report['segment_blend_all_windows_covered']).lower()}",
        f"- normalization_issue_suspected: {str(report['normalization_issue_suspected']).lower()}",
        f"- route_space_exhausted: {str(report['route_space_exhausted']).lower()}",
        f"- unsupported_new_model_needed: {str(report['unsupported_new_model_needed']).lower()}",
        f"- recommended_next_step: {report['recommended_next_step']}",
        "- recommend_threshold_relaxation: false",
        "- recommend_cleanup_debleed_denoise: false",
        "- unsupported_new_model_needed_execution: false",
        "- can_publish_as_result: false",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "track002_minus_failure_report.json", report)
    write_json(output_dir / "track002_next_route_decision.json", build_decision_payload(report))
    (output_dir / "track002_minus_failure_report.md").write_text(build_report_markdown(report), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Track002 no-reference minus failure decision report.")
    parser.add_argument("--forensics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_failure_report(args.forensics_dir)
    write_outputs(report, args.output_dir)
    print(json.dumps(build_decision_payload(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
