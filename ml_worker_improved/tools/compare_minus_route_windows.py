from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.analyze_minus_window_blockers import (  # noqa: E402
    SAFETY_FLAGS,
    build_window_blocker_profile,
    write_json,
)
from tools.build_no_reference_ml_dataset import write_jsonl  # noqa: E402


def is_clean_window(row: dict[str, Any]) -> bool:
    if int(row.get("blocker_count") or 0) == 0:
        return True
    return str(row.get("minus_judge_decision") or "") == "accept_candidate"


def route_rows(profile_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        grouped[str(row.get("route_id") or "unknown_route")].append(row)
    return dict(sorted(grouped.items()))


def median_score(rows: list[dict[str, Any]]) -> float:
    scores = [float(row.get("minus_score") or 0.0) for row in rows]
    if not scores:
        return 0.0
    return round(float(median(scores)), 6)


def score_range(rows: list[dict[str, Any]]) -> tuple[float, float]:
    scores = [float(row.get("minus_score") or 0.0) for row in rows]
    if not scores:
        return 0.0, 0.0
    return round(max(scores), 6), round(min(scores), 6)


def clean_coverage_by_window(profile_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = defaultdict(list)
    for row in profile_rows:
        if not is_clean_window(row):
            continue
        key = str(row.get("window_index"))
        value = f"{row.get('candidate_id')}|{row.get('route_id')}"
        coverage[key].append(value)
    return {key: sorted(values) for key, values in sorted(coverage.items(), key=lambda item: int(item[0]))}


def route_window_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("route_id") or ""), int(row.get("window_index") or 0)


def build_baseline_by_window(grouped: dict[str, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    if not grouped:
        return {}
    baseline_route = next(iter(grouped))
    return {int(row.get("window_index") or 0): row for row in grouped[baseline_route]}


def route_improves_any_window(rows: list[dict[str, Any]], baseline_by_window: dict[int, dict[str, Any]]) -> bool:
    if any(is_clean_window(row) for row in rows):
        return True
    for row in rows:
        baseline = baseline_by_window.get(int(row.get("window_index") or 0))
        if not baseline:
            continue
        if int(row.get("blocker_count") or 0) < int(baseline.get("blocker_count") or 0):
            return True
        if float(row.get("minus_score") or 0.0) > float(baseline.get("minus_score") or 0.0):
            return True
    return False


def route_worse_than_baseline(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> bool:
    if not baseline_rows:
        return False
    return (
        sum(1 for row in rows if not is_clean_window(row))
        > sum(1 for row in baseline_rows if not is_clean_window(row))
    ) or median_score(rows) < median_score(baseline_rows)


def compact_profile_summary(run_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "candidates_analyzed": profile["summary"].get("candidates_analyzed", 0),
        "routes_analyzed": profile["summary"].get("routes_analyzed", 0),
        "windows_analyzed": profile["summary"].get("windows_analyzed", 0),
        "severe_phase_collapse_total": profile["summary"].get("severe_phase_collapse_total", 0),
        "severe_vocal_bleed_in_minus_total": profile["summary"].get("severe_vocal_bleed_in_minus_total", 0),
        "blocker_signatures": profile["summary"].get("blocker_signatures", []),
        "release_ready": False,
    }


def compare_minus_routes(
    run_dir: Path,
    control_run_dir: Path | None = None,
    extra_run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    profile = build_window_blocker_profile(run_dir)
    profile_rows = profile["rows"]
    grouped = route_rows(profile_rows)
    baseline_by_window = build_baseline_by_window(grouped)
    baseline_rows = next(iter(grouped.values())) if grouped else []

    rows: list[dict[str, Any]] = []
    for route_id, items in grouped.items():
        best_score, worst_score = score_range(items)
        rows.append(
            {
                "route_id": route_id,
                "windows_total": len(items),
                "windows_accepted": sum(1 for row in items if is_clean_window(row)),
                "windows_blocked": sum(1 for row in items if not is_clean_window(row)),
                "severe_phase_collapse_count": sum(1 for row in items if row.get("severe_phase_collapse") is True),
                "severe_vocal_bleed_count": sum(1 for row in items if row.get("severe_vocal_bleed_in_minus") is True),
                "phase_comb_filtering_count": sum(1 for row in items if row.get("severe_phase_comb_filtering") is True),
                "best_window_score": best_score,
                "worst_window_score": worst_score,
                "median_window_score": median_score(items),
                "unique_blocker_signatures": sorted({str(row.get("blocker_signature") or "clean") for row in items}),
                "route_improves_any_window": route_improves_any_window(items, baseline_by_window),
                "route_worse_than_baseline": route_worse_than_baseline(items, baseline_rows),
                "release_ready": False,
            }
        )

    route_signature_sets = [tuple(row["unique_blocker_signatures"]) for row in rows]
    all_routes_same_blocker_signature = bool(route_signature_sets) and len(set(route_signature_sets)) == 1
    any_route_has_clean_window = any(row["windows_accepted"] > 0 for row in rows)
    phase_counts = [int(row["severe_phase_collapse_count"]) for row in rows]
    bleed_counts = [int(row["severe_vocal_bleed_count"]) for row in rows]
    clean_coverage = clean_coverage_by_window(profile_rows)
    all_window_keys = {str(row.get("window_index")) for row in profile_rows}
    clean_routes = {
        str(profile_row.get("route_id") or "")
        for profile_row in profile_rows
        if is_clean_window(profile_row)
    }
    segment_blend_possible = bool(clean_coverage) and len(clean_routes) >= 2
    segment_blend_all_windows_covered = bool(all_window_keys) and all_window_keys.issubset(set(clean_coverage))
    all_windows_blocked = bool(profile_rows) and all(not is_clean_window(row) for row in profile_rows)
    summary = {
        "run_dir": str(run_dir),
        "routes_analyzed": len(rows),
        "windows_analyzed": len(profile_rows),
        "all_routes_same_blocker_signature": all_routes_same_blocker_signature,
        "any_route_has_clean_window": any_route_has_clean_window,
        "any_route_reduces_phase_collapse": bool(phase_counts) and min(phase_counts) < max(phase_counts),
        "any_route_reduces_vocal_bleed": bool(bleed_counts) and min(bleed_counts) < max(bleed_counts),
        "segment_blend_possible": segment_blend_possible,
        "segment_blend_all_windows_covered": segment_blend_all_windows_covered,
        "segment_blend_clean_window_coverage": clean_coverage,
        "route_space_exhausted": all_routes_same_blocker_signature and all_windows_blocked and not any_route_has_clean_window,
        "normalization_issue_suspected": profile["summary"].get("normalization_correlated") is True,
        "release_ready": False,
        **SAFETY_FLAGS,
    }
    if control_run_dir is not None and control_run_dir.exists():
        control_profile = build_window_blocker_profile(control_run_dir)
        summary["control_run"] = compact_profile_summary(control_run_dir, control_profile)
    extra_summaries = []
    for extra_run_dir in extra_run_dirs or []:
        if extra_run_dir.exists():
            extra_summaries.append(compact_profile_summary(extra_run_dir, build_window_blocker_profile(extra_run_dir)))
    summary["extra_comparison_runs"] = extra_summaries
    return {"rows": rows, "summary": summary, "profile_summary": profile["summary"]}


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Minus Route Window Comparison",
        "",
        f"- routes_analyzed: {summary.get('routes_analyzed', 0)}",
        f"- windows_analyzed: {summary.get('windows_analyzed', 0)}",
        f"- all_routes_same_blocker_signature: {str(summary.get('all_routes_same_blocker_signature') is True).lower()}",
        f"- any_route_has_clean_window: {str(summary.get('any_route_has_clean_window') is True).lower()}",
        f"- any_route_reduces_phase_collapse: {str(summary.get('any_route_reduces_phase_collapse') is True).lower()}",
        f"- any_route_reduces_vocal_bleed: {str(summary.get('any_route_reduces_vocal_bleed') is True).lower()}",
        f"- segment_blend_possible: {str(summary.get('segment_blend_possible') is True).lower()}",
        f"- segment_blend_all_windows_covered: {str(summary.get('segment_blend_all_windows_covered') is True).lower()}",
        f"- route_space_exhausted: {str(summary.get('route_space_exhausted') is True).lower()}",
        f"- normalization_issue_suspected: {str(summary.get('normalization_issue_suspected') is True).lower()}",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "minus_route_window_comparison.jsonl", result["rows"])
    write_json(output_dir / "minus_route_window_comparison_summary.json", result["summary"])
    (output_dir / "minus_route_window_comparison_summary.md").write_text(
        build_summary_markdown(result["summary"]),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare no-reference minus routes at window level.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control-run-dir", type=Path)
    parser.add_argument("--extra-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_minus_routes(args.run_dir, args.control_run_dir, args.extra_run_dir)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
