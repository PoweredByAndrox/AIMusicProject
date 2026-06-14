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


REQUIRED_CANDIDATE_FIELDS = (
    "harness_candidate_id",
    "parent_id",
    "mutation_type",
    "candidate_profile",
    "candidate_diversity_profile",
    "job_priority_policy",
    "shared_reuse_policy",
    "hard_case_route_order",
    "minus_selector_policy",
    "vocal_selector_policy",
    "pair_compatibility_policy",
    "window_judge_policy",
    "learned_script_hybrid_weight",
    "timeout_budget_policy",
    "max_candidates",
    "max_hard_case_extra_jobs",
    "forbidden_actions",
    "module_config",
    "stage_graph",
)
REQUIRED_MODULE_CONFIG = {
    "planning": ("candidate_job_scheduler", "profile_escalation", "hard_case_router", "timeout_budget_policy"),
    "reasoning": ("no_reference_windowed_judge", "mode_specific_score_matrix", "learned_critic", "failure_reflection"),
    "tool_use": ("local_separator_job_runner", "shape_normalizer", "preview_renderer", "post_render_judge"),
    "memory": ("replay_bundle", "learning_outcomes", "model_manifest", "route_outcome_history"),
}
ALLOWED_STAGE_NODES = {
    "candidate_wave",
    "windowed_judge_vocal",
    "windowed_judge_minus",
    "learned_critic",
    "selector",
    "pair_compatibility",
    "hard_case_router",
    "segment_blend_planner",
    "dry_run",
    "preview_render",
    "post_render_judge",
}
REQUIRED_STAGE_NODES = {
    "candidate_wave",
    "windowed_judge_vocal",
    "windowed_judge_minus",
    "learned_critic",
    "selector",
    "pair_compatibility",
    "dry_run",
    "post_render_judge",
}
ALLOWED_STAGE_MUTATIONS = {
    "enable_optional_hard_case_route",
    "disable_optional_hard_case_route",
    "reorder_hard_case_routes",
    "adjust_fallback_edge",
    "adjust_profile_escalation",
    "adjust_retry_count",
}
FORBIDDEN_STAGE_MUTATIONS = {
    "remove_safety_gate",
    "skip_pair_compatibility",
    "skip_post_render_judge",
    "enable_release_ready",
    "bypass_blockers",
}
ALLOWED_MUTATION_FIELDS = {
    "job_priority_order",
    "hard_case_route_order",
    "candidate_profile",
    "candidate_diversity_profile",
    "timeout_budget_allocation",
    "max_candidates",
    "learned_script_hybrid_weight",
    "selector_preference_order",
    "window_selection_policy",
}
FORBIDDEN_MUTATION_FIELDS = {
    "safety_blockers",
    "release_ready_rules",
    "threshold_relaxation",
    "cleanup_debleed_denoise",
    "model_downloads",
    "unsupported_models",
    "source_code_generation",
    "model_registry_edits",
}
FORBIDDEN_ACTIONS = (
    "do_not_require_reference_stems",
    "do_not_require_human_csv",
    "do_not_train_separator",
    "do_not_download_models",
    "do_not_add_unsupported_models",
    "do_not_modify_source_code_automatically",
    "do_not_relax_thresholds",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_mark_release_ready",
    "do_not_publish_final_result",
)
SAFETY_FLAG_REJECTIONS = {
    "release_ready": "release_ready_forbidden",
    "thresholds_relaxed": "threshold_relaxation_forbidden",
    "cleanup_routes_added": "cleanup_debleed_denoise_forbidden",
    "model_downloaded": "model_download_forbidden",
    "model_registry_edited": "model_registry_edit_forbidden",
    "training_started": "training_forbidden",
    "separator_training_started": "separator_training_forbidden",
    "reference_required": "reference_required_forbidden",
    "human_csv_required": "human_csv_required_forbidden",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def default_harness_candidate() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "harness_candidate_id": "baseline_safe_harness",
        "parent_id": "",
        "mutation_type": "baseline_keep",
        "candidate_profile": "safe",
        "candidate_diversity_profile": "safe",
        "job_priority_policy": "registry_default",
        "shared_reuse_policy": "reuse_shared_outputs",
        "hard_case_route_order": ["generate_phase_safe_variants", "generate_minus_focused_variants"],
        "minus_selector_policy": "script_guarded_then_hybrid_score",
        "vocal_selector_policy": "script_guarded_then_hybrid_score",
        "pair_compatibility_policy": "strict_no_bypass",
        "window_judge_policy": "default_windowed_no_reference",
        "learned_script_hybrid_weight": 0.5,
        "timeout_budget_policy": {"mode": "bounded", "max_runtime_sec": 1800},
        "max_candidates": 12,
        "max_hard_case_extra_jobs": 4,
        "module_config": {
            "planning": {
                "candidate_job_scheduler": "registry_default",
                "profile_escalation": "safe_to_balanced_only",
                "hard_case_router": "minus_hard_case_router",
                "timeout_budget_policy": "bounded",
            },
            "reasoning": {
                "no_reference_windowed_judge": "mode_specific",
                "mode_specific_score_matrix": "strict_script_blockers",
                "learned_critic": "shadow_hybrid_no_bypass",
                "failure_reflection": "deterministic_replay_reflection",
            },
            "tool_use": {
                "local_separator_job_runner": "registered_existing_models_only",
                "shape_normalizer": "safe_shape_normalization",
                "preview_renderer": "bounded_preview_only",
                "post_render_judge": "required_after_preview",
            },
            "memory": {
                "replay_bundle": "metadata_only",
                "learning_outcomes": "verified_outcomes_only",
                "model_manifest": "record_version_only",
                "route_outcome_history": "replay_events",
            },
        },
        "stage_graph": {
            "nodes": [
                "candidate_wave",
                "windowed_judge_vocal",
                "windowed_judge_minus",
                "learned_critic",
                "selector",
                "pair_compatibility",
                "hard_case_router",
                "segment_blend_planner",
                "dry_run",
                "preview_render",
                "post_render_judge",
            ],
            "edges": [
                ["candidate_wave", "windowed_judge_vocal"],
                ["candidate_wave", "windowed_judge_minus"],
                ["windowed_judge_vocal", "learned_critic"],
                ["windowed_judge_minus", "learned_critic"],
                ["learned_critic", "selector"],
                ["selector", "pair_compatibility"],
                ["pair_compatibility", "dry_run"],
                ["dry_run", "preview_render"],
                ["preview_render", "post_render_judge"],
            ],
            "optional_mutations": ["enable_optional_hard_case_route", "reorder_hard_case_routes"],
        },
        "forbidden_actions": list(FORBIDDEN_ACTIONS),
        "mutations": {},
        "release_ready": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "model_downloaded": False,
        "model_registry_edited": False,
        "training_started": False,
        "separator_training_started": False,
        "reference_required": False,
        "human_csv_required": False,
    }


def validate_harness_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for field in REQUIRED_CANDIDATE_FIELDS:
        if field not in candidate:
            reasons.append(f"missing_required_field:{field}")
    for key, reason in SAFETY_FLAG_REJECTIONS.items():
        if candidate.get(key) is True:
            reasons.append(reason)
    mutations = candidate.get("mutations") or {}
    if not isinstance(mutations, dict):
        reasons.append("mutations_must_be_object")
        mutations = {}
    for field in sorted(mutations):
        if field in FORBIDDEN_MUTATION_FIELDS:
            reasons.append(f"forbidden_mutation_field:{field}")
        elif field not in ALLOWED_MUTATION_FIELDS:
            reasons.append(f"unsupported_mutation_field:{field}")
    try:
        hybrid_weight = float(candidate.get("learned_script_hybrid_weight", -1.0))
    except (TypeError, ValueError):
        hybrid_weight = -1.0
    if not 0.0 <= hybrid_weight <= 1.0:
        reasons.append("learned_script_hybrid_weight_out_of_range")
    if str(candidate.get("candidate_profile") or "") not in {"safe", "balanced", "exploration"}:
        reasons.append("candidate_profile_not_allowed")
    if str(candidate.get("candidate_diversity_profile") or "") not in {"safe", "balanced", "exploration"}:
        reasons.append("candidate_diversity_profile_not_allowed")
    module_config = candidate.get("module_config") or {}
    if not isinstance(module_config, dict):
        reasons.append("module_config_must_be_object")
        module_config = {}
    for group, modules in REQUIRED_MODULE_CONFIG.items():
        group_config = module_config.get(group)
        if not isinstance(group_config, dict):
            reasons.append(f"missing_module_group:{group}")
            continue
        for module_name in modules:
            if module_name not in group_config:
                reasons.append(f"missing_module:{group}.{module_name}")
    stage_graph = candidate.get("stage_graph") or {}
    if not isinstance(stage_graph, dict):
        reasons.append("stage_graph_must_be_object")
        stage_graph = {}
    nodes = {str(node) for node in stage_graph.get("nodes") or []}
    for node in sorted(nodes - ALLOWED_STAGE_NODES):
        reasons.append(f"stage_graph_unsupported_node:{node}")
    for node in sorted(REQUIRED_STAGE_NODES - nodes):
        reasons.append(f"stage_graph_missing_required_node:{node}")
    mutations_requested = {str(item) for item in stage_graph.get("optional_mutations") or []}
    for item in sorted(mutations_requested & FORBIDDEN_STAGE_MUTATIONS):
        reasons.append(f"stage_graph_forbidden_mutation:{item}")
    for item in sorted(mutations_requested - ALLOWED_STAGE_MUTATIONS - FORBIDDEN_STAGE_MUTATIONS):
        reasons.append(f"stage_graph_unsupported_mutation:{item}")
    edges = stage_graph.get("edges") or []
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for index, edge in enumerate(edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            reasons.append(f"stage_graph_invalid_edge:{index}")
            continue
        source, target = str(edge[0]), str(edge[1])
        if source not in ALLOWED_STAGE_NODES:
            reasons.append(f"stage_graph_edge_unsupported_source:{source}")
        if target not in ALLOWED_STAGE_NODES:
            reasons.append(f"stage_graph_edge_unsupported_target:{target}")
        if source in nodes and target in nodes:
            adjacency.setdefault(source, set()).add(target)
        if target == "preview_render" and source != "dry_run":
            reasons.append(f"stage_graph_unsafe_edge_to_preview_render:{source}")
        if target == "post_render_judge" and source != "preview_render":
            reasons.append(f"stage_graph_unsafe_edge_to_post_render_judge:{source}")

    def reachable(start: str, target: str) -> bool:
        if start not in nodes or target not in nodes:
            return False
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(sorted(adjacency.get(node, set()) - seen))
        return False

    for node in sorted(REQUIRED_STAGE_NODES):
        if node in nodes and not reachable("candidate_wave", node):
            reasons.append(f"stage_graph_required_node_unreachable:{node}")
    for source, target in (
        ("selector", "pair_compatibility"),
        ("pair_compatibility", "dry_run"),
        ("dry_run", "preview_render"),
        ("preview_render", "post_render_judge"),
    ):
        if source in nodes and target in nodes and not reachable(source, target):
            reasons.append(f"stage_graph_required_path_missing:{source}->{target}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def has_cycle(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for child in adjacency.get(node, set()):
            if has_cycle(child):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(has_cycle(node) for node in sorted(nodes)):
        reasons.append("stage_graph_cycle_detected")
    mutation_type = str(candidate.get("mutation_type") or "").lower()
    for token, reason in (
        ("release_ready", "release_ready_forbidden"),
        ("threshold", "threshold_relaxation_forbidden"),
        ("cleanup", "cleanup_debleed_denoise_forbidden"),
        ("debleed", "cleanup_debleed_denoise_forbidden"),
        ("denoise", "cleanup_debleed_denoise_forbidden"),
        ("download", "model_download_forbidden"),
        ("registry", "model_registry_edit_forbidden"),
        ("source_code", "source_code_generation_forbidden"),
    ):
        if token in mutation_type and reason not in reasons:
            reasons.append(reason)
    return {
        "valid": not reasons,
        "rejection_reasons": reasons,
        "harness_candidate_id": str(candidate.get("harness_candidate_id") or ""),
        "release_ready": False,
    }


def numeric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def track_score(row: dict[str, Any]) -> float:
    return (
        numeric(row, "preview_ready_count") * 4.0
        + numeric(row, "minus_accepted_count") * 3.0
        + numeric(row, "pair_compatible_count") * 3.0
        + numeric(row, "vocal_accepted_count")
        - numeric(row, "severe_vocal_bleed_in_minus_count") * 2.0
        - numeric(row, "severe_phase_collapse_count") * 2.0
        - numeric(row, "failed_safety_contract_count") * 5.0
        - numeric(row, "failed_run_count") * 4.0
        - numeric(row, "incomplete_replay_count") * 3.0
        - numeric(row, "timeout_count")
    )


def run_summary_events(replay_bundle_dir: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(replay_bundle_dir / "replay_bundle_events.jsonl") if row.get("event_type") == "run_summary"]


def complexity_for(candidate: dict[str, Any], run_rows: list[dict[str, Any]]) -> int:
    route_count = len(candidate.get("hard_case_route_order") or [])
    selector_complexity = 1 if "hybrid" in str(candidate.get("minus_selector_policy") or "") else 0
    candidate_count = int(candidate.get("max_candidates") or 0)
    extra_jobs = int(candidate.get("max_hard_case_extra_jobs") or 0)
    observed_complexity = sum(int(row.get("complexity_score") or row.get("candidate_count") or 0) for row in run_rows)
    return max(1, route_count + selector_complexity + candidate_count + extra_jobs + observed_complexity // max(1, len(run_rows) or 1))


def replay_harness_candidate(*, candidate: dict[str, Any], replay_bundle_dir: Path, output_jsonl: Path | None = None) -> dict[str, Any]:
    validation = validate_harness_candidate(candidate)
    run_rows = run_summary_events(replay_bundle_dir)
    per_track_scores = {str(row.get("track_id") or row.get("run_root") or index): track_score(row) for index, row in enumerate(run_rows)}
    aggregate = {
        "schema_version": 1,
        "harness_candidate_id": str(candidate.get("harness_candidate_id") or ""),
        "parent_id": str(candidate.get("parent_id") or ""),
        "mutation_type": str(candidate.get("mutation_type") or ""),
        "safety_valid": validation["valid"],
        "validation_rejection_reasons": validation["rejection_reasons"],
        "run_count": len(run_rows),
        "accepted_pair_count": int(sum(numeric(row, "accepted_pair_count") for row in run_rows)),
        "preview_ready_count": int(sum(numeric(row, "preview_ready_count") for row in run_rows)),
        "final_safe_candidate_count": int(sum(numeric(row, "final_safe_candidate_count") for row in run_rows)),
        "pair_compatible_count": int(sum(numeric(row, "pair_compatible_count") for row in run_rows)),
        "minus_accepted_count": int(sum(numeric(row, "minus_accepted_count") for row in run_rows)),
        "vocal_accepted_count": int(sum(numeric(row, "vocal_accepted_count") for row in run_rows)),
        "severe_vocal_bleed_in_minus_count": int(sum(numeric(row, "severe_vocal_bleed_in_minus_count") for row in run_rows)),
        "severe_phase_collapse_count": int(sum(numeric(row, "severe_phase_collapse_count") for row in run_rows)),
        "failed_safety_contract_count": int(sum(numeric(row, "failed_safety_contract_count") for row in run_rows)) + (0 if validation["valid"] else 1),
        "release_ready_count": 0,
        "estimated_gpu_jobs": int(sum(numeric(row, "estimated_gpu_jobs") for row in run_rows)),
        "estimated_runtime_sec": int(sum(numeric(row, "estimated_runtime_sec") for row in run_rows)),
        "duplicate_jobs_avoided": int(sum(numeric(row, "duplicate_jobs_avoided") for row in run_rows)),
        "candidate_count": int(sum(numeric(row, "candidate_count") for row in run_rows)),
        "complexity_score": complexity_for(candidate, run_rows),
        "per_track_scores": per_track_scores,
        "per_track_floor_score": min(per_track_scores.values()) if per_track_scores else 0.0,
        "mean_track_score": sum(per_track_scores.values()) / len(per_track_scores) if per_track_scores else 0.0,
        "failed_run_count": int(sum(numeric(row, "failed_run_count") for row in run_rows)),
        "timeout_count": int(sum(numeric(row, "timeout_count") for row in run_rows)),
        "incomplete_replay_count": int(sum(numeric(row, "incomplete_replay_count") for row in run_rows)),
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
    if output_jsonl is not None:
        write_jsonl(output_jsonl, [aggregate])
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a no-reference harness candidate config against a replay bundle.")
    parser.add_argument("--replay-bundle-dir", type=Path, required=True)
    parser.add_argument("--harness-candidate-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args(argv)
    candidate = read_json(args.harness_candidate_json)
    result = replay_harness_candidate(candidate=candidate, replay_bundle_dir=args.replay_bundle_dir, output_jsonl=args.output_jsonl)
    print(json.dumps({"harness_candidate_eval_jsonl": str(args.output_jsonl), **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
