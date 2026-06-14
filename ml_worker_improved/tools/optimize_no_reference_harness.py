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


QUALITY_POSITIVE_OBJECTIVES = (
    "audio_preview_ready_count",
    "rendered_preview_audio_count",
    "accepted_minus_count",
    "minus_accepted_count",
    "pair_compatible_count",
    "analysis_ready_stems_count",
)
QUALITY_NEGATIVE_OBJECTIVES = (
    "severe_phase_collapse_count",
    "severe_vocal_bleed_in_minus_count",
    "blocked_stems_count",
)
SAFETY_NEGATIVE_OBJECTIVES = (
    "preview_render_failed_count",
    "preview_validation_failed_count",
    "preview_contract_missing_count",
    "failed_safety_contract_count",
    "failed_run_count",
    "timeout_count",
    "incomplete_replay_count",
)
EFFICIENCY_OBJECTIVES = (
    "estimated_runtime_sec",
)
COMPLEXITY_OBJECTIVES = (
    "complexity_score",
)
POSITIVE_OBJECTIVES = QUALITY_POSITIVE_OBJECTIVES
NEGATIVE_OBJECTIVES = QUALITY_NEGATIVE_OBJECTIVES + SAFETY_NEGATIVE_OBJECTIVES + EFFICIENCY_OBJECTIVES + COMPLEXITY_OBJECTIVES


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
    return tuple(metric(row, key) for key in QUALITY_POSITIVE_OBJECTIVES + QUALITY_NEGATIVE_OBJECTIVES + SAFETY_NEGATIVE_OBJECTIVES)


def quality_improvement_reasons(row: dict[str, Any], baseline_eval: dict[str, Any] | None) -> list[str]:
    if not baseline_eval or row.get("harness_candidate_id") == baseline_eval.get("harness_candidate_id"):
        return []
    reasons: list[str] = []
    for key in QUALITY_POSITIVE_OBJECTIVES:
        if metric(row, key) > metric(baseline_eval, key):
            reasons.append(f"{key}_increased")
    for key in QUALITY_NEGATIVE_OBJECTIVES:
        if metric(row, key) < metric(baseline_eval, key):
            reasons.append(f"{key}_decreased")
    return reasons


def quality_regression_reasons(row: dict[str, Any], baseline_eval: dict[str, Any] | None) -> list[str]:
    if not baseline_eval or row.get("harness_candidate_id") == baseline_eval.get("harness_candidate_id"):
        return []
    reasons: list[str] = []
    for key in QUALITY_POSITIVE_OBJECTIVES:
        if metric(row, key) < metric(baseline_eval, key):
            reasons.append(f"{key}_decreased")
    for key in QUALITY_NEGATIVE_OBJECTIVES:
        if metric(row, key) > metric(baseline_eval, key):
            reasons.append(f"{key}_increased")
    return reasons


def safety_regression_reasons(row: dict[str, Any], baseline_eval: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if row.get("release_ready") is True or metric(row, "release_ready_count") > 0:
        reasons.append("release_ready_forbidden")
    if row.get("safety_valid") is not True:
        reasons.append("candidate_safety_invalid")
    for key in SAFETY_NEGATIVE_OBJECTIVES:
        baseline_value = metric(baseline_eval or {}, key)
        if metric(row, key) > baseline_value:
            reasons.append(f"{key}_increased")
    return reasons


def efficiency_improvement_detected(row: dict[str, Any], baseline_eval: dict[str, Any] | None) -> bool:
    if not baseline_eval or row.get("harness_candidate_id") == baseline_eval.get("harness_candidate_id"):
        return False
    return any(metric(row, key) > 0.0 and metric(baseline_eval, key) > 0.0 and metric(row, key) < metric(baseline_eval, key) for key in EFFICIENCY_OBJECTIVES)


def with_gate_fields(
    row: dict[str, Any],
    *,
    baseline_eval: dict[str, Any] | None,
    selection_decision: str,
    promotion_eligible: bool,
    promotion_block_reason: str = "",
) -> dict[str, Any]:
    quality_reasons = quality_improvement_reasons(row, baseline_eval)
    safety_reasons = safety_regression_reasons(row, baseline_eval)
    quality_regressions = quality_regression_reasons(row, baseline_eval)
    return {
        **row,
        "selection_decision": selection_decision,
        "decision_class": selection_decision,
        "quality_improvement_detected": bool(quality_reasons),
        "quality_improvement_reasons": quality_reasons,
        "quality_regression_reasons": quality_regressions,
        "efficiency_improvement_detected": efficiency_improvement_detected(row, baseline_eval),
        "safety_regression_detected": bool(safety_reasons),
        "safety_regression_reasons": safety_reasons,
        "promotion_eligible": bool(promotion_eligible),
        "promotion_block_reason": promotion_block_reason,
    }


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
    efficiency_only_rows: list[dict[str, Any]] = []
    no_op_rows: list[dict[str, Any]] = []
    safety_valid_rows: list[dict[str, Any]] = []
    for row in rows:
        base = {**row}
        if is_safety_rejected(base):
            rejected_rows.append(
                with_gate_fields(
                    base,
                    baseline_eval=baseline_eval,
                    selection_decision="candidate_rejected_safety",
                    promotion_eligible=False,
                    promotion_block_reason="safety_regression",
                )
            )
        elif metric(base, "preview_contract_missing_count") > 0:
            rejected_rows.append(
                with_gate_fields(
                    base,
                    baseline_eval=baseline_eval,
                    selection_decision="candidate_rejected_preview_contract_missing",
                    promotion_eligible=False,
                    promotion_block_reason="preview_contract_missing",
                )
            )
        elif baseline_eval is None:
            safety_valid_rows.append(base)
        elif is_specialist(base, baseline_eval):
            specialist_rows.append(
                with_gate_fields(
                    base,
                    baseline_eval=baseline_eval,
                    selection_decision="candidate_specialist_only",
                    promotion_eligible=False,
                    promotion_block_reason="specialist_only_not_global",
                )
            )
        elif base.get("mutation_type") == "baseline_keep":
            safety_valid_rows.append(base)
        elif quality_improvement_reasons(base, baseline_eval):
            safety_valid_rows.append(base)
        elif quality_regression_reasons(base, baseline_eval) or safety_regression_reasons(base, baseline_eval):
            rejected_rows.append(
                with_gate_fields(
                    base,
                    baseline_eval=baseline_eval,
                    selection_decision="candidate_rejected_runtime",
                    promotion_eligible=False,
                    promotion_block_reason="quality_or_safety_regression",
                )
            )
        elif efficiency_improvement_detected(base, baseline_eval):
            efficiency_only_rows.append(
                with_gate_fields(
                    base,
                    baseline_eval=baseline_eval,
                    selection_decision="candidate_efficiency_only",
                    promotion_eligible=False,
                    promotion_block_reason="efficiency_only_no_quality_improvement",
                )
            )
        else:
            no_op_rows.append(
                with_gate_fields(
                    base,
                    baseline_eval=baseline_eval,
                    selection_decision="candidate_no_op",
                    promotion_eligible=False,
                    promotion_block_reason="no_quality_or_efficiency_improvement",
                )
            )
    for row in safety_valid_rows:
        dominators = [other for other in safety_valid_rows if other is not row and dominates(other, row)]
        if dominators:
            decision = "candidate_rejected_complexity" if any(quality_signature(other) == quality_signature(row) for other in dominators) else "candidate_rejected_pareto_dominated"
            rejected_rows.append(
                with_gate_fields(
                    row,
                    baseline_eval=baseline_eval,
                    selection_decision=decision,
                    promotion_eligible=False,
                    promotion_block_reason=decision,
                )
            )
        else:
            if row.get("mutation_type") == "baseline_keep":
                decision = "baseline_keep"
                eligible = False
                block_reason = "baseline"
            else:
                decision = "candidate_quality_improved"
                eligible = True
                block_reason = ""
            frontier_rows.append(
                with_gate_fields(
                    row,
                    baseline_eval=baseline_eval,
                    selection_decision=decision,
                    promotion_eligible=eligible,
                    promotion_block_reason=block_reason,
                )
            )
    frontier_rows = sorted(frontier_rows, key=lambda row: (metric(row, "complexity_score"), metric(row, "estimated_runtime_sec"), str(row.get("harness_candidate_id"))))
    rejected_rows = sorted(rejected_rows, key=lambda row: str(row.get("harness_candidate_id")))
    specialist_rows = sorted(specialist_rows, key=lambda row: str(row.get("harness_candidate_id")))
    efficiency_only_rows = sorted(efficiency_only_rows, key=lambda row: (metric(row, "estimated_runtime_sec"), str(row.get("harness_candidate_id"))))
    no_op_rows = sorted(no_op_rows, key=lambda row: str(row.get("harness_candidate_id")))
    return {
        "frontier_rows": frontier_rows,
        "rejected_rows": rejected_rows,
        "specialist_rows": specialist_rows,
        "efficiency_only_rows": efficiency_only_rows,
        "no_op_rows": no_op_rows,
    }


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
        f"- quality_improved_candidates: {summary.get('quality_improved_candidates')}",
        f"- efficiency_only_candidates: {summary.get('efficiency_only_candidates')}",
        f"- no_op_candidates: {summary.get('no_op_candidates')}",
        f"- pareto_candidates_selected: {summary.get('pareto_candidates_selected')}",
        f"- rejected_safety_candidates: {summary.get('rejected_safety_candidates')}",
        f"- rejected_runtime_candidates: {summary.get('rejected_runtime_candidates')}",
        f"- rejected_complexity_candidates: {summary.get('rejected_complexity_candidates')}",
        f"- rejected_preview_contract_missing_candidates: {summary.get('rejected_preview_contract_missing_candidates')}",
        f"- specialist_routes_discovered: {summary.get('specialist_routes_discovered')}",
        f"- best_global_candidate: {summary.get('best_global_candidate') or 'none'}",
        f"- efficiency_only_best_candidate: {summary.get('efficiency_only_best_candidate') or 'none'}",
        f"- best_baseline_candidate: {summary.get('best_baseline_candidate') or 'none'}",
        f"- best_conditional_specialist_candidate: {summary.get('best_conditional_specialist_candidate') or 'none'}",
        f"- estimated_audio_preview_ready_gain: {summary.get('estimated_audio_preview_ready_gain')}",
        f"- estimated_rendered_preview_audio_gain: {summary.get('estimated_rendered_preview_audio_gain')}",
        f"- estimated_minus_accepted_gain: {summary.get('estimated_minus_accepted_gain')}",
        f"- preview_render_failed_count: {summary.get('preview_render_failed_count')}",
        f"- preview_validation_failed_count: {summary.get('preview_validation_failed_count')}",
        f"- preview_contract_missing_count: {summary.get('preview_contract_missing_count')}",
        f"- severe_blocker_change: {summary.get('severe_blocker_change')}",
        f"- estimated_runtime_change: {summary.get('estimated_runtime_change')}",
        f"- failure_reflections_generated: {summary.get('failure_reflections_generated')}",
        f"- release_ready_count: {summary.get('release_ready_count')}",
        f"- llm_proposer_used: {str(summary.get('llm_proposer_used') is True).lower()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_promotion_gate_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Promotion Gate Summary",
        "",
        f"- best_global_candidate: {summary.get('best_global_candidate') or 'none'}",
        f"- quality_improved_candidates: {summary.get('quality_improved_candidates')}",
        f"- efficiency_only_candidates: {summary.get('efficiency_only_candidates')}",
        f"- efficiency_only_best_candidate: {summary.get('efficiency_only_best_candidate') or 'none'}",
        f"- no_op_candidates: {summary.get('no_op_candidates')}",
        f"- promotion_gate_decision: {summary.get('promotion_gate_decision')}",
        f"- promotion_block_reason: {summary.get('promotion_block_reason')}",
        f"- release_ready_count: {summary.get('release_ready_count')}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
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
    write_jsonl(output_dir / "efficiency_only_candidates.jsonl", selection["efficiency_only_rows"])
    write_jsonl(output_dir / "no_op_candidates.jsonl", selection["no_op_rows"])
    reflection_result = generate_failure_reflections(replay_bundle_dir=replay_bundle_dir, output_dir=output_dir / "failure_reflections")
    best_baseline = next((row for row in selection["frontier_rows"] if row.get("mutation_type") == "baseline_keep"), None)
    best_specialist = selection["specialist_rows"][0] if selection["specialist_rows"] else None
    quality_frontier_rows = [row for row in selection["frontier_rows"] if row.get("selection_decision") == "candidate_quality_improved"]
    best_global = quality_frontier_rows[0] if quality_frontier_rows else None
    best_efficiency = selection["efficiency_only_rows"][0] if selection["efficiency_only_rows"] else None
    baseline_candidate_id = str((baseline_eval or {}).get("harness_candidate_id") or "")
    base_audio_preview = int((baseline_eval or {}).get("audio_preview_ready_count") or 0)
    base_rendered_preview = int((baseline_eval or {}).get("rendered_preview_audio_count") or 0)
    base_minus = int((baseline_eval or {}).get("minus_accepted_count") or 0)
    best_frontier = best_global or best_baseline or (baseline_eval or {})
    severe_base = int((baseline_eval or {}).get("severe_vocal_bleed_in_minus_count") or 0) + int((baseline_eval or {}).get("severe_phase_collapse_count") or 0)
    severe_best = int(best_frontier.get("severe_vocal_bleed_in_minus_count") or 0) + int(best_frontier.get("severe_phase_collapse_count") or 0)
    promotion_block_reason = "" if best_global else "no_quality_improved_candidate"
    summary = {
        "schema_version": 1,
        "replay_bundle_dir": str(replay_bundle_dir),
        "baseline_harness_json": str(baseline_harness_json),
        "generations": int(generations),
        "population_size": int(population_size),
        "evaluated_candidates": len(eval_rows),
        "quality_improved_candidates": len(quality_frontier_rows),
        "efficiency_only_candidates": len(selection["efficiency_only_rows"]),
        "no_op_candidates": len(selection["no_op_rows"]),
        "pareto_candidates_selected": len(quality_frontier_rows),
        "rejected_safety_candidates": sum(1 for row in selection["rejected_rows"] if row.get("selection_decision") == "candidate_rejected_safety"),
        "rejected_runtime_candidates": sum(1 for row in selection["rejected_rows"] if row.get("selection_decision") == "candidate_rejected_runtime"),
        "rejected_complexity_candidates": sum(1 for row in selection["rejected_rows"] if row.get("selection_decision") == "candidate_rejected_complexity"),
        "rejected_preview_contract_missing_candidates": sum(
            1 for row in selection["rejected_rows"] if row.get("selection_decision") == "candidate_rejected_preview_contract_missing"
        ),
        "specialist_routes_discovered": len(selection["specialist_rows"]),
        "best_baseline_candidate": str(best_baseline.get("harness_candidate_id")) if best_baseline else baseline_candidate_id,
        "best_global_candidate": str(best_global.get("harness_candidate_id")) if best_global else "",
        "efficiency_only_best_candidate": str(best_efficiency.get("harness_candidate_id")) if best_efficiency else "",
        "promotion_gate_decision": "candidate_quality_improved" if best_global else "keep_baseline",
        "promotion_block_reason": promotion_block_reason,
        "baseline_selection_decision": "baseline_keep" if best_baseline else "baseline_pareto_dominated",
        "best_conditional_specialist_candidate": str(best_specialist.get("harness_candidate_id")) if best_specialist else "",
        "estimated_audio_preview_ready_gain": int(best_frontier.get("audio_preview_ready_count") or 0) - base_audio_preview,
        "estimated_preview_ready_gain": int(best_frontier.get("audio_preview_ready_count") or 0) - base_audio_preview,
        "estimated_rendered_preview_audio_gain": int(best_frontier.get("rendered_preview_audio_count") or 0) - base_rendered_preview,
        "estimated_minus_accepted_gain": int(best_frontier.get("minus_accepted_count") or 0) - base_minus,
        "preview_render_failed_count": sum(int(row.get("preview_render_failed_count") or 0) for row in eval_rows),
        "preview_validation_failed_count": sum(int(row.get("preview_validation_failed_count") or 0) for row in eval_rows),
        "preview_contract_missing_count": sum(int(row.get("preview_contract_missing_count") or 0) for row in eval_rows),
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
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "release_ready": False,
    }
    if use_llm_proposer:
        summary["llm_proposer_request_ignored_reason"] = "local_llm_proposer_not_configured"
    write_summary(output_dir / "harness_optimizer_summary.md", summary)
    write_promotion_gate_summary(output_dir / "promotion_gate_summary.md", summary)
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
