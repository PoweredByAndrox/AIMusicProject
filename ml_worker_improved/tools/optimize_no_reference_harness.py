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
from tools.generate_no_reference_failure_reflections import generate_failure_reflections  # noqa: E402
from tools.replay_no_reference_harness_candidate import (  # noqa: E402
    default_harness_candidate,
    parse_bool,
    replay_harness_candidate,
    validate_harness_candidate,
)


POSITIVE_OBJECTIVES = (
    "preview_ready_count",
    "minus_accepted_count",
    "pair_compatible_count",
    "vocal_accepted_count",
    "per_track_floor_score",
)
NEGATIVE_OBJECTIVES = (
    "severe_vocal_bleed_in_minus_count",
    "severe_phase_collapse_count",
    "failed_run_count",
    "timeout_count",
    "incomplete_replay_count",
    "estimated_runtime_sec",
    "complexity_score",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def quality_signature(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(metric(row, key) for key in POSITIVE_OBJECTIVES + NEGATIVE_OBJECTIVES[:-2])


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    no_worse = True
    strictly_better = False
    for key in POSITIVE_OBJECTIVES:
        if metric(a, key) < metric(b, key):
            no_worse = False
        if metric(a, key) > metric(b, key):
            strictly_better = True
    for key in NEGATIVE_OBJECTIVES:
        if metric(a, key) > metric(b, key):
            no_worse = False
        if metric(a, key) < metric(b, key):
            strictly_better = True
    return no_worse and strictly_better


def is_safety_rejected(row: dict[str, Any]) -> bool:
    return (
        row.get("safety_valid") is not True
        or metric(row, "failed_safety_contract_count") > 0
        or metric(row, "release_ready_count") > 0
        or row.get("release_ready") is True
    )


def is_specialist(row: dict[str, Any], baseline_eval: dict[str, Any] | None) -> bool:
    if not baseline_eval or row.get("harness_candidate_id") == baseline_eval.get("harness_candidate_id"):
        return False
    scores = row.get("per_track_scores") or {}
    baseline_scores = baseline_eval.get("per_track_scores") or {}
    if not isinstance(scores, dict) or not isinstance(baseline_scores, dict):
        return False
    common = set(scores) & set(baseline_scores)
    if not common:
        return False
    better = any(float(scores[key]) > float(baseline_scores[key]) for key in common)
    worse = any(float(scores[key]) < float(baseline_scores[key]) for key in common)
    return better and worse


def select_pareto_frontier(rows: list[dict[str, Any]], baseline_eval: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    frontier_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    specialist_rows: list[dict[str, Any]] = []
    safety_valid_rows: list[dict[str, Any]] = []
    for row in rows:
        base = {**row}
        if is_safety_rejected(base):
            rejected_rows.append({**base, "selection_decision": "candidate_rejected_safety"})
        elif is_specialist(base, baseline_eval):
            specialist_rows.append({**base, "selection_decision": "candidate_specialist_route_only"})
        else:
            safety_valid_rows.append(base)
    for row in safety_valid_rows:
        dominators = [other for other in safety_valid_rows if other is not row and dominates(other, row)]
        if dominators:
            decision = "candidate_rejected_complexity" if any(quality_signature(other) == quality_signature(row) for other in dominators) else "candidate_rejected_pareto_dominated"
            rejected_rows.append({**row, "selection_decision": decision})
        else:
            decision = "baseline_keep" if row.get("mutation_type") == "baseline_keep" else "candidate_pareto_selected"
            frontier_rows.append({**row, "selection_decision": decision})
    frontier_rows = sorted(frontier_rows, key=lambda row: (metric(row, "complexity_score"), metric(row, "estimated_runtime_sec"), str(row.get("harness_candidate_id"))))
    rejected_rows = sorted(rejected_rows, key=lambda row: str(row.get("harness_candidate_id")))
    specialist_rows = sorted(specialist_rows, key=lambda row: str(row.get("harness_candidate_id")))
    return {"frontier_rows": frontier_rows, "rejected_rows": rejected_rows, "specialist_rows": specialist_rows}


def mutate_candidate(parent: dict[str, Any], index: int) -> dict[str, Any]:
    candidate = {**parent, "parent_id": str(parent.get("harness_candidate_id") or ""), "mutations": {}}
    if index == 0:
        candidate["harness_candidate_id"] = str(parent.get("harness_candidate_id") or "baseline_safe_harness")
        candidate["mutation_type"] = "baseline_keep"
        return candidate
    variants = (
        ("balanced_candidate_budget", {"candidate_profile": "balanced", "candidate_diversity_profile": "balanced", "max_candidates": max(12, int(parent.get("max_candidates") or 0)), "mutations": {"candidate_profile": "balanced", "candidate_diversity_profile": "balanced", "max_candidates": max(12, int(parent.get("max_candidates") or 0))}}),
        ("phase_first_hard_case_route", {"hard_case_route_order": ["generate_phase_safe_variants", "generate_minus_focused_variants"], "max_hard_case_extra_jobs": max(4, int(parent.get("max_hard_case_extra_jobs") or 0)), "mutations": {"hard_case_route_order": ["generate_phase_safe_variants", "generate_minus_focused_variants"]}}),
        ("lower_runtime_same_policy", {"timeout_budget_policy": {"mode": "bounded", "max_runtime_sec": 900}, "max_candidates": min(8, int(parent.get("max_candidates") or 8)), "mutations": {"timeout_budget_allocation": 900, "max_candidates": min(8, int(parent.get("max_candidates") or 8))}}),
        ("window_policy_floor_guard", {"window_judge_policy": "floor_guarded_windowed_no_reference", "mutations": {"window_selection_policy": "floor_guarded"}}),
    )
    name, changes = variants[(index - 1) % len(variants)]
    candidate.update(changes)
    candidate["mutation_type"] = name
    candidate["harness_candidate_id"] = "harness_" + stable_hash({"parent": parent.get("harness_candidate_id"), "index": index, "changes": changes})[:16]
    return candidate


def build_population(baseline: dict[str, Any], population_size: int) -> list[dict[str, Any]]:
    return [mutate_candidate(baseline, index) for index in range(max(1, int(population_size)))]


def merge_missing_baseline_defaults(candidate: dict[str, Any]) -> dict[str, Any]:
    defaults = default_harness_candidate()
    merged = {**defaults, **candidate}
    for key in ("module_config", "stage_graph", "timeout_budget_policy"):
        if not isinstance(candidate.get(key), dict):
            merged[key] = defaults[key]
    if not isinstance(candidate.get("forbidden_actions"), list):
        merged["forbidden_actions"] = defaults["forbidden_actions"]
    return merged


def load_or_create_baseline(path: Path) -> dict[str, Any]:
    if path.exists():
        baseline = merge_missing_baseline_defaults(read_json(path))
        if baseline != read_json(path):
            write_json(path, baseline)
        return baseline
    baseline = default_harness_candidate()
    write_json(path, baseline)
    return baseline


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Meta-Harness Optimizer Summary",
        "",
        f"- evaluated_candidates: {summary.get('evaluated_candidates')}",
        f"- pareto_candidates_selected: {summary.get('pareto_candidates_selected')}",
        f"- rejected_safety_candidates: {summary.get('rejected_safety_candidates')}",
        f"- rejected_complexity_candidates: {summary.get('rejected_complexity_candidates')}",
        f"- specialist_routes_discovered: {summary.get('specialist_routes_discovered')}",
        f"- best_baseline_candidate: {summary.get('best_baseline_candidate') or 'none'}",
        f"- best_conditional_specialist_candidate: {summary.get('best_conditional_specialist_candidate') or 'none'}",
        f"- estimated_preview_ready_gain: {summary.get('estimated_preview_ready_gain')}",
        f"- estimated_minus_accepted_gain: {summary.get('estimated_minus_accepted_gain')}",
        f"- severe_blocker_change: {summary.get('severe_blocker_change')}",
        f"- estimated_runtime_change: {summary.get('estimated_runtime_change')}",
        f"- failure_reflections_generated: {summary.get('failure_reflections_generated')}",
        f"- release_ready_count: {summary.get('release_ready_count')}",
        f"- llm_proposer_used: {str(summary.get('llm_proposer_used') is True).lower()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def optimize_harness(
    *,
    replay_bundle_dir: Path,
    baseline_harness_json: Path,
    output_dir: Path,
    generations: int,
    population_size: int,
    use_llm_proposer: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_or_create_baseline(baseline_harness_json)
    population = build_population(baseline, population_size)
    eval_rows: list[dict[str, Any]] = []
    for candidate in population:
        validation = validate_harness_candidate(candidate)
        row = replay_harness_candidate(candidate=candidate, replay_bundle_dir=replay_bundle_dir)
        row["generations_requested"] = int(generations)
        row["llm_proposer_used"] = False
        row["candidate_config"] = candidate
        if not validation["valid"]:
            row["safety_valid"] = False
            row["validation_rejection_reasons"] = validation["rejection_reasons"]
        eval_rows.append(row)
    baseline_eval = next((row for row in eval_rows if row.get("mutation_type") == "baseline_keep"), eval_rows[0] if eval_rows else None)
    selection = select_pareto_frontier(eval_rows, baseline_eval=baseline_eval)
    frontier_output_rows = selection["frontier_rows"] + selection["specialist_rows"]
    write_jsonl(output_dir / "harness_candidate_eval.jsonl", eval_rows)
    write_jsonl(output_dir / "harness_pareto_frontier.jsonl", frontier_output_rows)
    write_jsonl(output_dir / "harness_rejected_candidates.jsonl", selection["rejected_rows"])
    reflection_result = generate_failure_reflections(replay_bundle_dir=replay_bundle_dir, output_dir=output_dir / "failure_reflections")
    best_baseline = next((row for row in selection["frontier_rows"] if row.get("mutation_type") == "baseline_keep"), None)
    best_specialist = selection["specialist_rows"][0] if selection["specialist_rows"] else None
    baseline_candidate_id = str((baseline_eval or {}).get("harness_candidate_id") or "")
    base_preview = int((baseline_eval or {}).get("preview_ready_count") or 0)
    base_minus = int((baseline_eval or {}).get("minus_accepted_count") or 0)
    best_frontier = selection["frontier_rows"][0] if selection["frontier_rows"] else (baseline_eval or {})
    severe_base = int((baseline_eval or {}).get("severe_vocal_bleed_in_minus_count") or 0) + int((baseline_eval or {}).get("severe_phase_collapse_count") or 0)
    severe_best = int(best_frontier.get("severe_vocal_bleed_in_minus_count") or 0) + int(best_frontier.get("severe_phase_collapse_count") or 0)
    summary = {
        "schema_version": 1,
        "replay_bundle_dir": str(replay_bundle_dir),
        "baseline_harness_json": str(baseline_harness_json),
        "generations": int(generations),
        "population_size": int(population_size),
        "evaluated_candidates": len(eval_rows),
        "pareto_candidates_selected": len(selection["frontier_rows"]),
        "rejected_safety_candidates": sum(1 for row in selection["rejected_rows"] if row.get("selection_decision") == "candidate_rejected_safety"),
        "rejected_complexity_candidates": sum(1 for row in selection["rejected_rows"] if row.get("selection_decision") == "candidate_rejected_complexity"),
        "specialist_routes_discovered": len(selection["specialist_rows"]),
        "best_baseline_candidate": str(best_baseline.get("harness_candidate_id")) if best_baseline else baseline_candidate_id,
        "baseline_selection_decision": "baseline_keep" if best_baseline else "baseline_pareto_dominated",
        "best_conditional_specialist_candidate": str(best_specialist.get("harness_candidate_id")) if best_specialist else "",
        "estimated_preview_ready_gain": int(best_frontier.get("preview_ready_count") or 0) - base_preview,
        "estimated_minus_accepted_gain": int(best_frontier.get("minus_accepted_count") or 0) - base_minus,
        "severe_blocker_change": severe_best - severe_base,
        "estimated_runtime_change": int(best_frontier.get("estimated_runtime_sec") or 0) - int((baseline_eval or {}).get("estimated_runtime_sec") or 0),
        "release_ready_count": sum(int(row.get("release_ready_count") or 0) for row in eval_rows),
        "failure_reflections_generated": int(reflection_result.get("reflection_count") or 0),
        "failure_reflections_jsonl": str(reflection_result.get("failure_reflections_jsonl") or ""),
        "failure_reflections_summary_md": str(reflection_result.get("failure_reflections_summary_md") or ""),
        "llm_proposer_used": False,
        "codex_required_at_runtime": False,
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "model_downloaded": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "release_ready": False,
    }
    if use_llm_proposer:
        summary["llm_proposer_request_ignored_reason"] = "local_llm_proposer_not_configured"
    write_summary(output_dir / "harness_optimizer_summary.md", summary)
    write_json(output_dir / "harness_optimizer_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministically optimize no-reference harness configs by replaying existing traces.")
    parser.add_argument("--replay-bundle-dir", type=Path, required=True)
    parser.add_argument("--baseline-harness-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--use-llm-proposer", type=parse_bool, default=False)
    args = parser.parse_args(argv)
    baseline_harness_json = args.baseline_harness_json or (args.output_dir.parent / "baseline_harness.json")
    result = optimize_harness(
        replay_bundle_dir=args.replay_bundle_dir,
        baseline_harness_json=baseline_harness_json,
        output_dir=args.output_dir,
        generations=args.generations,
        population_size=args.population_size,
        use_llm_proposer=args.use_llm_proposer,
    )
    print(json.dumps({"optimizer_summary": str(args.output_dir / "harness_optimizer_summary.md"), **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
