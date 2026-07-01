from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.no_reference_score_matrix import pair_compatibility_score  # noqa: E402


DEFAULT_MIN_DELTA = 0.03

SEVERE_BLOCKERS = {
    "bad_mixture_reconstruction",
    "vocal_leakage_in_minus",
    "instrumental_leakage_in_vocal",
    "silence_hallucination",
    "robotic_or_noise_artifact",
    "thin_or_overcleaned_vocal",
    "transient_damage",
    "phase_or_stereo_inconsistency",
    "low_no_reference_score",
}

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_random_cleanup_chains",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_best_of_bad",
    "do_not_use_hard_vad_gate",
    "do_not_execute_segment_blend_without_approval",
]

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "segment_blend_executed": False,
}

DEFECT_ACTIONS = {
    "vocal_bleed_in_minus": "choose_lower_vocal_leakage_in_minus_candidate",
    "instrumental_bleed_in_vocal": "choose_cleaner_vocal_candidate",
    "vocal_body_loss": "rollback_to_fuller_vocal_candidate",
    "vocal_tail_loss": "tail_preserving_candidate",
    "robotic_musical_noise": "rollback_to_less_aggressive_candidate",
    "silence_hallucination": "choose_candidate_with_lower_silence_hallucination",
    "transient_smearing": "choose_candidate_with_lower_transient_damage",
    "phase_comb_filtering": "phase_alignment_correction_plan",
    "high_band_damage": "body_preserving_blend_plan",
    "mix_reconstruction_drift": "gain_phase_alignment_correction_plan",
}


def clip01(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def reasons(row: dict[str, Any]) -> list[str]:
    return as_list(row.get("reasons"))


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def severe_reasons(row: dict[str, Any]) -> set[str]:
    return {reason for reason in reasons(row) if reason in SEVERE_BLOCKERS}


def detect_defects(row: dict[str, Any]) -> list[str]:
    row_reasons = set(reasons(row))
    defects: list[str] = []
    if metric(row, "vocal_leakage_in_minus_score") >= 0.45 or "vocal_leakage_in_minus" in row_reasons:
        defects.append("vocal_bleed_in_minus")
    if metric(row, "instrumental_leakage_in_vocal_score") >= 0.55 or "instrumental_leakage_in_vocal" in row_reasons:
        defects.append("instrumental_bleed_in_vocal")
    if metric(row, "thin_vocal_score") >= 0.55 or row_reasons.intersection({"thin_or_overcleaned_vocal", "thin_or_overcleaned"}):
        defects.append("vocal_body_loss")
    if "vocal_tail_loss" in row_reasons:
        defects.append("vocal_tail_loss")
    if metric(row, "robotic_artifact_score") >= 0.55 or row_reasons.intersection({"robotic_or_noise_artifact", "robotic_artifact", "musical_noise"}):
        defects.append("robotic_musical_noise")
    if metric(row, "silence_hallucination_score") >= 0.55 or "silence_hallucination" in row_reasons:
        defects.append("silence_hallucination")
    if metric(row, "transient_damage_score") >= 0.55 or "transient_damage" in row_reasons:
        defects.append("transient_smearing")
    if metric(row, "phase_consistency_score", 1.0) <= 0.55 or "phase_or_stereo_inconsistency" in row_reasons:
        defects.append("phase_comb_filtering")
    if row_reasons.intersection({"high_band_damage", "air_detail_loss", "presence_loss"}):
        defects.append("high_band_damage")
    if metric(row, "mixture_consistency_score", 1.0) <= 0.70 or "bad_mixture_reconstruction" in row_reasons:
        defects.append("mix_reconstruction_drift")
    return stable_unique(defects)


def component_scores(row: dict[str, Any], *, baseline: dict[str, Any] | None = None) -> dict[str, float]:
    vocal_score = clip01(
        1.0
        - max(
            metric(row, "instrumental_leakage_in_vocal_score"),
            metric(row, "silence_hallucination_score") * 0.70,
            metric(row, "thin_vocal_score") * 0.45,
        )
    )
    minus_score = clip01(1.0 - metric(row, "vocal_leakage_in_minus_score"))
    artifact_score = clip01(
        1.0
        - max(
            metric(row, "robotic_artifact_score"),
            metric(row, "transient_damage_score"),
            max(0.0, 0.55 - metric(row, "phase_consistency_score", 1.0)),
        )
    )
    consistency_score = clip01(
        0.70 * metric(row, "mixture_consistency_score", 0.0)
        + 0.30 * metric(row, "phase_consistency_score", 1.0)
    )
    if baseline is None:
        defect_reduction_score = clip01(1.0 - defect_burden(row))
    else:
        defect_reduction_score = clip01(0.50 + defect_burden(baseline) - defect_burden(row))
    final_fitness = clip01(
        0.30 * min(vocal_score, minus_score)
        + 0.25 * artifact_score
        + 0.25 * consistency_score
        + 0.20 * defect_reduction_score
    )
    return {
        "vocal_score": vocal_score,
        "minus_score": minus_score,
        "artifact_score": artifact_score,
        "consistency_score": consistency_score,
        "defect_reduction_score": defect_reduction_score,
        "final_fitness": final_fitness,
    }


def defect_burden(row: dict[str, Any]) -> float:
    row_reasons = set(reasons(row))
    burden = max(
        metric(row, "vocal_leakage_in_minus_score"),
        metric(row, "instrumental_leakage_in_vocal_score"),
        metric(row, "silence_hallucination_score"),
        metric(row, "robotic_artifact_score"),
        metric(row, "thin_vocal_score") * 0.75,
        metric(row, "transient_damage_score"),
        max(0.0, 0.70 - metric(row, "mixture_consistency_score", 1.0)),
        max(0.0, 0.65 - metric(row, "phase_consistency_score", 1.0)),
    )
    if row_reasons.intersection({"thin_or_overcleaned_vocal", "thin_or_overcleaned"}):
        burden = max(burden, 0.70)
    if "vocal_tail_loss" in row_reasons:
        burden = max(burden, 0.70)
    if severe_reasons(row):
        burden = max(burden, 0.65)
    return clip01(burden)


def choose_input_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def priority(row: dict[str, Any]) -> tuple[int, str]:
        cid = candidate_id(row).lower()
        role = str(row.get("candidate_role") or row.get("role") or "").lower()
        if row.get("is_input_candidate") is True:
            return (0, cid)
        if role in {"input", "original", "baseline"}:
            return (1, cid)
        if cid in {"original", "baseline", "input"}:
            return (2, cid)
        return (3, cid)

    return sorted(rows, key=priority)[0]


def route_action(defects: list[str]) -> str:
    if not defects:
        return "preserve_original_candidate"
    return DEFECT_ACTIONS.get(defects[0], "select_existing_candidate_variant")


def reject_reason(baseline: dict[str, Any], candidate: dict[str, Any], baseline_scores: dict[str, float], candidate_scores: dict[str, float]) -> str | None:
    if candidate_scores["minus_score"] <= baseline_scores["minus_score"] - 0.25 or metric(candidate, "vocal_leakage_in_minus_score") >= 0.75:
        return "minus_regression_severe"
    if candidate_scores["vocal_score"] <= baseline_scores["vocal_score"] - 0.25 or metric(candidate, "instrumental_leakage_in_vocal_score") >= 0.80:
        return "vocal_regression_severe"
    new_blockers = severe_reasons(candidate) - severe_reasons(baseline)
    if new_blockers:
        return "new_severe_blocker"
    if candidate_scores["artifact_score"] < baseline_scores["artifact_score"] - 0.05 and candidate_scores["defect_reduction_score"] < 0.85:
        return "artifact_score_regression"
    return None


def optimize_track(track_id: str, rows: list[dict[str, Any]], min_delta: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: candidate_id(row))
    baseline = choose_input_candidate(ordered)
    alternatives = [row for row in ordered if row is not baseline]
    baseline_defects = detect_defects(baseline)
    baseline_scores = component_scores(baseline)
    action = route_action(baseline_defects)
    actions_tried = stable_unique([action] if alternatives else [])
    actions_rejected: list[dict[str, Any]] = []
    scored_candidates: list[tuple[float, dict[str, Any], dict[str, float]]] = []

    for candidate in alternatives:
        candidate_scores = component_scores(candidate, baseline=baseline)
        rejection = reject_reason(baseline, candidate, baseline_scores, candidate_scores)
        if rejection is not None:
            actions_rejected.append(
                {
                    "candidate_id": candidate_id(candidate),
                    "action": action,
                    "reason": rejection,
                }
            )
            continue
        scored_candidates.append((candidate_scores["final_fitness"], candidate, candidate_scores))

    selected = baseline
    selected_scores = baseline_scores
    rollback_reason = ""
    final_decision = "original_candidate_kept"
    segment_blend_plan: dict[str, Any] | None = None

    if scored_candidates:
        scored_candidates.sort(key=lambda item: (-item[0], candidate_id(item[1])))
        best_fitness, best_candidate, best_scores = scored_candidates[0]
        improvement_delta = round(best_fitness - baseline_scores["final_fitness"], 6)
        if improvement_delta >= min_delta and severe_reasons(best_candidate) == set():
            selected = best_candidate
            selected_scores = best_scores
            final_decision = "optimized_candidate_selected"
        elif improvement_delta >= min_delta and any(reason.startswith("segment_winner_") for reason in reasons(best_candidate) + reasons(baseline)):
            final_decision = "needs_segment_blend"
            rollback_reason = "segment_candidates_need_plan_only"
            segment_blend_plan = {
                "status": "plan_only_not_executed",
                "candidate_ids": [candidate_id(baseline), candidate_id(best_candidate)],
                "reason": "candidate_strength_differs_by_segment",
            }
        else:
            rollback_reason = "fitness_improvement_below_min_delta"
    elif alternatives:
        rollback_reason = "all_candidate_variants_rejected"
    elif baseline_scores["final_fitness"] < 0.45 and severe_reasons(baseline):
        final_decision = "blocked_hard_case"
        rollback_reason = "input_candidate_has_no_safe_variant"

    if final_decision == "original_candidate_kept" and not rollback_reason:
        rollback_reason = "no_candidate_variant_supplied"

    fitness_before = baseline_scores["final_fitness"]
    fitness_after = selected_scores["final_fitness"] if final_decision == "optimized_candidate_selected" else fitness_before
    improvement_delta = round(fitness_after - fitness_before, 6)

    return {
        "track_id": track_id,
        "candidate_id": candidate_id(selected) if final_decision != "blocked_hard_case" else "",
        "mix_path": str(selected.get("mix_path") or "") if final_decision != "blocked_hard_case" else "",
        "vocal_path": str(selected.get("vocal_path") or "") if final_decision != "blocked_hard_case" else "",
        "instrumental_path": str(selected.get("instrumental_path") or selected.get("minus_path") or "") if final_decision != "blocked_hard_case" else "",
        "minus_path": str(selected.get("instrumental_path") or selected.get("minus_path") or "") if final_decision != "blocked_hard_case" else "",
        "input_candidate_id": candidate_id(baseline),
        "selected_candidate_id": candidate_id(selected) if final_decision != "blocked_hard_case" else None,
        "optimization_iterations": len(alternatives),
        "dominant_defects_before": baseline_defects,
        "actions_tried": actions_tried,
        "actions_rejected": actions_rejected,
        "fitness_before": fitness_before,
        "fitness_after": fitness_after,
        "fitness_components_before": baseline_scores,
        "fitness_components_after": selected_scores if final_decision == "optimized_candidate_selected" else baseline_scores,
        "improvement_delta": improvement_delta,
        "final_decision": final_decision,
        "selected_vocal_path": str(selected.get("vocal_path") or "") if final_decision != "blocked_hard_case" else "",
        "selected_minus_path": str(selected.get("instrumental_path") or "") if final_decision != "blocked_hard_case" else "",
        "reasons": stable_unique(
            [
                f"dominant_defect:{defect}" for defect in baseline_defects[:3]
            ]
            + (["fitness_improved"] if final_decision == "optimized_candidate_selected" else [])
            + (["no_best_of_bad"] if final_decision == "blocked_hard_case" else [])
        ),
        "rollback_reason": rollback_reason,
        "segment_blend_plan": segment_blend_plan,
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + [action for row in ordered for action in as_list(row.get("forbidden_actions"))]),
    }


def optimize_tracks(rows: list[dict[str, Any]], min_delta: float = DEFAULT_MIN_DELTA) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        track_id = str(row.get("track_id") or "")
        if not track_id:
            raise ValueError(f"Judge row is missing track_id: {row}")
        grouped[track_id].append(row)
    optimized = [optimize_track(track_id, grouped[track_id], min_delta) for track_id in sorted(grouped)]
    return {"rows": optimized, "summary": build_summary(optimized)}


def mode_field(mode: str, suffix: str) -> str:
    if mode == "vocal":
        return f"vocal_{suffix}"
    if mode == "minus":
        return f"minus_{suffix}"
    raise ValueError(f"Unsupported optimizer mode: {mode}")


def mode_schema_blockers(row: dict[str, Any], mode: str) -> list[str]:
    score_key = mode_field(mode, "score")
    decision_key = mode_field(mode, "decision")
    blocker_key = mode_field(mode, "blockers")
    blockers: list[str] = []
    if score_key not in row or row.get(score_key) in (None, ""):
        blockers.append(f"missing_mode_specific_score:{score_key}")
    if decision_key not in row or row.get(decision_key) in (None, ""):
        blockers.append(f"missing_mode_specific_decision:{decision_key}")
    if blocker_key not in row or row.get(blocker_key) is None:
        blockers.append(f"missing_mode_specific_blockers:{blocker_key}")
    return blockers


def mode_score(row: dict[str, Any], mode: str) -> float:
    score_key = mode_field(mode, "score")
    if score_key not in row or row.get(score_key) in (None, ""):
        return 0.0
    return metric(row, score_key)


def mode_decision(row: dict[str, Any], mode: str) -> str:
    decision_key = mode_field(mode, "decision")
    if decision_key not in row or row.get(decision_key) in (None, ""):
        return "blocked_hard_case"
    return str(row.get(decision_key) or "")


def mode_blockers(row: dict[str, Any], mode: str) -> list[str]:
    blocker_key = mode_field(mode, "blockers")
    schema_blockers = mode_schema_blockers(row, mode)
    explicit = as_list(row.get(blocker_key))
    if schema_blockers:
        inherited_reasons = [
            reason
            for reason in reasons(row)
            if reason != "no_reference_proxy_passed"
        ]
        return stable_unique(schema_blockers + explicit + inherited_reasons)
    if blocker_key in row and row.get(blocker_key) is not None:
        return explicit
    return explicit + [reason for reason in reasons(row) if reason != "no_reference_proxy_passed"]


def mode_rejection_blockers(rows: list[dict[str, Any]], mode: str) -> list[str]:
    return stable_unique(blocker for row in rows for blocker in mode_blockers(row, mode))


def select_mode_candidate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    acceptable = [
        row
        for row in rows
        if mode_decision(row, mode) == "accept_candidate"
        and not mode_blockers(row, mode)
        and mode_score(row, mode) >= 0.72
    ]
    if not acceptable:
        return None
    return sorted(acceptable, key=lambda row: (-mode_score(row, mode), candidate_id(row), str(row.get("vocal_path") or ""), str(row.get("instrumental_path") or row.get("minus_path") or "")))[0]


def mode_output_path(row: dict[str, Any], mode: str) -> str:
    if mode == "vocal":
        return str(row.get("vocal_path") or "")
    if mode == "minus":
        return str(row.get("instrumental_path") or row.get("minus_path") or "")
    raise ValueError(f"Unsupported optimizer path mode: {mode}")


def select_preview_mode_candidate(
    rows: list[dict[str, Any]],
    mode: str,
    *,
    counterpart_path: str = "",
) -> dict[str, Any] | None:
    previewable = [
        row
        for row in rows
        if mode_decision(row, mode) == "select_alternative_candidate"
        and not mode_blockers(row, mode)
        and mode_output_path(row, mode)
    ]
    if not previewable:
        return None
    counterpart_mode = "minus" if mode == "vocal" else "vocal"
    counterpart_matched = [
        row
        for row in previewable
        if counterpart_path and mode_output_path(row, counterpart_mode) == counterpart_path
    ]
    candidates = counterpart_matched or previewable
    return sorted(
        candidates,
        key=lambda row: (
            -mode_score(row, mode),
            candidate_id(row),
            str(row.get("vocal_path") or ""),
            str(row.get("instrumental_path") or row.get("minus_path") or ""),
        ),
    )[0]


def optimize_track_mode_specific(track_id: str, vocal_rows: list[dict[str, Any]], minus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_vocal = select_mode_candidate(vocal_rows, "vocal")
    selected_minus = select_mode_candidate(minus_rows, "minus")
    actions_rejected: list[dict[str, Any]] = []
    if selected_vocal is None:
        actions_rejected.append(
            {
                "mode": "vocal",
                "reason": "no_accepted_vocal_candidate",
                "mode_blockers": mode_rejection_blockers(vocal_rows, "vocal"),
            }
        )
    if selected_minus is None:
        actions_rejected.append(
            {
                "mode": "minus",
                "reason": "no_accepted_minus_candidate",
                "mode_blockers": mode_rejection_blockers(minus_rows, "minus"),
            }
        )
    if selected_vocal is None or selected_minus is None:
        preview_vocal = selected_vocal
        preview_minus = selected_minus
        if preview_vocal is None and selected_minus is not None:
            preview_vocal = select_preview_mode_candidate(
                vocal_rows,
                "vocal",
                counterpart_path=mode_output_path(selected_minus, "minus"),
            )
        if preview_minus is None and selected_vocal is not None:
            preview_minus = select_preview_mode_candidate(
                minus_rows,
                "minus",
                counterpart_path=mode_output_path(selected_vocal, "vocal"),
            )
        if (selected_vocal is None) ^ (selected_minus is None) and preview_vocal is not None and preview_minus is not None:
            preview_vocal_path = mode_output_path(preview_vocal, "vocal")
            preview_minus_path = mode_output_path(preview_minus, "minus")
            preview_base = preview_vocal if selected_vocal is None else preview_minus
            return {
                "track_id": track_id,
                "candidate_id": f"preview_mode_specific_pair:{candidate_id(preview_vocal)}:{candidate_id(preview_minus)}",
                "mix_path": str(preview_vocal.get("mix_path") or preview_minus.get("mix_path") or ""),
                "vocal_path": preview_vocal_path,
                "instrumental_path": preview_minus_path,
                "minus_path": preview_minus_path,
                "input_candidate_id": candidate_id(preview_base),
                "selected_candidate_id": None,
                "selected_vocal_candidate_id": candidate_id(preview_vocal),
                "selected_minus_candidate_id": candidate_id(preview_minus),
                "same_candidate_pair": candidate_id(preview_vocal) == candidate_id(preview_minus),
                "pair_compatibility_decision": "preview_only_not_release",
                "needs_pair_reconciliation": True,
                "segment_blend_needed": True,
                "optimization_iterations": max(len(vocal_rows), len(minus_rows)),
                "dominant_defects_before": stable_unique(
                    as_list(preview_vocal.get("vocal_dominant_defect"))
                    + as_list(preview_minus.get("minus_dominant_defect"))
                ),
                "actions_tried": ["mode_specific_candidate_selection", "preview_only_segment_blend_routing"],
                "actions_rejected": actions_rejected,
                "fitness_before": 0.0,
                "fitness_after": 0.0,
                "improvement_delta": 0.0,
                "final_decision": "needs_segment_blend",
                "selected_vocal_path": preview_vocal_path,
                "selected_minus_path": preview_minus_path,
                "selected_vocal_score": mode_score(preview_vocal, "vocal"),
                "selected_minus_score": mode_score(preview_minus, "minus"),
                "reasons": stable_unique(
                    [
                        "no_best_of_bad",
                        "preview_only_not_release",
                        "missing_accepted_mode_candidate",
                    ]
                ),
                "rollback_reason": "preview_requires_segment_plan_and_post_render_judge",
                "segment_blend_plan": {
                    "status": "preview_only_not_executed",
                    "reason": "missing_accepted_mode_candidate",
                    "can_publish_as_result": False,
                    "can_mark_release_ready": False,
                    "requires_post_render_judge": True,
                },
                "forbidden_actions": stable_unique(
                    BASE_FORBIDDEN_ACTIONS
                    + as_list(preview_vocal.get("forbidden_actions"))
                    + as_list(preview_minus.get("forbidden_actions"))
                ),
            }
        return {
            "track_id": track_id,
            "candidate_id": "",
            "mix_path": "",
            "vocal_path": "",
            "instrumental_path": "",
            "minus_path": "",
            "input_candidate_id": None,
            "selected_candidate_id": None,
            "selected_vocal_candidate_id": candidate_id(selected_vocal) if selected_vocal else None,
            "selected_minus_candidate_id": candidate_id(selected_minus) if selected_minus else None,
            "same_candidate_pair": False,
            "pair_compatibility_decision": "blocked_pair_incompatible",
            "needs_pair_reconciliation": False,
            "segment_blend_needed": False,
            "optimization_iterations": max(len(vocal_rows), len(minus_rows)),
            "dominant_defects_before": [],
            "actions_tried": ["mode_specific_candidate_selection"],
            "actions_rejected": actions_rejected,
            "fitness_before": 0.0,
            "fitness_after": 0.0,
            "improvement_delta": 0.0,
            "final_decision": "blocked_hard_case",
            "selected_vocal_path": "",
            "selected_minus_path": "",
            "reasons": ["no_best_of_bad"],
            "rollback_reason": "missing_accepted_mode_candidate",
            "segment_blend_plan": None,
            "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS),
        }
    pair = pair_compatibility_score(selected_vocal, selected_minus)
    same_candidate = candidate_id(selected_vocal) == candidate_id(selected_minus)
    if pair["pair_decision"] == "pair_compatible":
        final_decision = "mode_specific_pair_selected"
        rollback_reason = ""
        needs_pair_reconciliation = False
    else:
        final_decision = "needs_pair_reconciliation"
        rollback_reason = "selected_vocal_minus_pair_requires_reconciliation"
        needs_pair_reconciliation = True
    vocal_score = mode_score(selected_vocal, "vocal")
    minus_score = mode_score(selected_minus, "minus")
    pair_score = metric(pair, "pair_compatibility_score")
    selected_vocal_path = str(selected_vocal.get("vocal_path") or "")
    selected_minus_path = str(selected_minus.get("instrumental_path") or selected_minus.get("minus_path") or "")
    mix_path = str(selected_vocal.get("mix_path") or selected_minus.get("mix_path") or "")
    pair_candidate_id = (
        candidate_id(selected_vocal)
        if same_candidate
        else f"mode_specific_pair:{candidate_id(selected_vocal)}:{candidate_id(selected_minus)}"
    )
    return {
        "track_id": track_id,
        "candidate_id": pair_candidate_id,
        "mix_path": mix_path,
        "vocal_path": selected_vocal_path,
        "instrumental_path": selected_minus_path,
        "minus_path": selected_minus_path,
        "input_candidate_id": None,
        "selected_candidate_id": candidate_id(selected_vocal) if same_candidate and final_decision == "mode_specific_pair_selected" else None,
        "selected_vocal_candidate_id": candidate_id(selected_vocal),
        "selected_minus_candidate_id": candidate_id(selected_minus),
        "same_candidate_pair": same_candidate,
        "pair_compatibility_decision": pair["pair_decision"],
        "pair_compatibility_score": pair_score,
        "pair_blockers": pair["pair_blockers"],
        "needs_pair_reconciliation": needs_pair_reconciliation,
        "segment_blend_needed": False,
        "optimization_iterations": max(len(vocal_rows), len(minus_rows)),
        "dominant_defects_before": stable_unique(
            as_list(selected_vocal.get("vocal_dominant_defect"))
            + as_list(selected_minus.get("minus_dominant_defect"))
        ),
        "actions_tried": ["mode_specific_vocal_selection", "mode_specific_minus_selection", "pair_compatibility_check"],
        "actions_rejected": actions_rejected,
        "fitness_before": 0.0,
        "fitness_after": round(0.45 * vocal_score + 0.45 * minus_score + 0.10 * pair_score, 6),
        "improvement_delta": 0.0,
        "final_decision": final_decision,
        "selected_vocal_path": selected_vocal_path,
        "selected_minus_path": selected_minus_path,
        "selected_vocal_score": vocal_score,
        "selected_minus_score": minus_score,
        "reasons": stable_unique(
            ["mode_specific_selection"]
            + (["same_candidate_pair"] if same_candidate else ["mixed_candidate_pair"])
            + ([] if not needs_pair_reconciliation else ["needs_pair_reconciliation"])
        ),
        "rollback_reason": rollback_reason,
        "segment_blend_plan": None,
        "forbidden_actions": stable_unique(
            BASE_FORBIDDEN_ACTIONS
            + as_list(selected_vocal.get("forbidden_actions"))
            + as_list(selected_minus.get("forbidden_actions"))
        ),
    }


def optimize_tracks_mode_specific(vocal_rows: list[dict[str, Any]], minus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vocal_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    minus_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vocal_rows:
        track_id = str(row.get("track_id") or "")
        if not track_id:
            raise ValueError(f"Vocal judge row is missing track_id: {row}")
        vocal_grouped[track_id].append(row)
    for row in minus_rows:
        track_id = str(row.get("track_id") or "")
        if not track_id:
            raise ValueError(f"Minus judge row is missing track_id: {row}")
        minus_grouped[track_id].append(row)
    track_ids = sorted(set(vocal_grouped) | set(minus_grouped))
    rows = [
        optimize_track_mode_specific(track_id, vocal_grouped.get(track_id, []), minus_grouped.get(track_id, []))
        for track_id in track_ids
    ]
    return {"rows": rows, "summary": build_summary(rows)}


def average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key) or 0.0) for row in rows) / len(rows), 6)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(row.get("final_decision") or "") for row in rows)
    defects = Counter(defect for row in rows for defect in as_list(row.get("dominant_defects_before")))
    rejected = Counter(action.get("reason") for row in rows for action in row.get("actions_rejected", []) if isinstance(action, dict))
    usable = decisions.get("optimized_candidate_selected", 0) + decisions.get("original_candidate_kept", 0) + decisions.get("mode_specific_pair_selected", 0)
    return {
        "total_tracks_optimized": len(rows),
        "optimized_candidate_selected": decisions.get("optimized_candidate_selected", 0),
        "mode_specific_pair_selected": decisions.get("mode_specific_pair_selected", 0),
        "original_candidate_kept": decisions.get("original_candidate_kept", 0),
        "needs_segment_blend": decisions.get("needs_segment_blend", 0),
        "needs_pair_reconciliation": decisions.get("needs_pair_reconciliation", 0),
        "blocked_hard_case": decisions.get("blocked_hard_case", 0),
        "average_fitness_before": average(rows, "fitness_before"),
        "average_fitness_after": average(rows, "fitness_after"),
        "average_improvement_delta": average(rows, "improvement_delta"),
        "top_dominant_defects": dict(defects.most_common(10)),
        "top_rejected_actions": dict(rejected.most_common(10)),
        "usable_without_stems": usable,
        "safety": dict(SAFETY_FLAGS),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Optimizer Summary",
        "",
        f"- total_tracks_optimized: {summary['total_tracks_optimized']}",
        f"- optimized_candidate_selected: {summary['optimized_candidate_selected']}",
        f"- mode_specific_pair_selected: {summary['mode_specific_pair_selected']}",
        f"- original_candidate_kept: {summary['original_candidate_kept']}",
        f"- needs_segment_blend: {summary['needs_segment_blend']}",
        f"- needs_pair_reconciliation: {summary['needs_pair_reconciliation']}",
        f"- blocked_hard_case: {summary['blocked_hard_case']}",
        f"- average_fitness_before: {summary['average_fitness_before']}",
        f"- average_fitness_after: {summary['average_fitness_after']}",
        f"- average_improvement_delta: {summary['average_improvement_delta']}",
        f"- usable_without_stems: {summary['usable_without_stems']}",
        "",
        "## Top Dominant Defects",
        "",
    ]
    for key, value in summary["top_dominant_defects"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Rejected Actions", ""])
    for key, value in summary["top_rejected_actions"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "no_reference_optimizer_result.jsonl", result["rows"])
    write_summary_md(output_dir / "no_reference_optimizer_summary.md", result["summary"])


def write_outputs_to_paths(result: dict[str, Any], output_jsonl: Path, summary_md: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, result["rows"])
    write_summary_md(summary_md, result["summary"])


def load_rows_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.judge_mode == "mode_specific":
        raise SystemExit("mode_specific rows are loaded separately")
    if args.judge_mode == "windowed":
        if args.windowed_judge_jsonl is None:
            raise SystemExit("--windowed-judge-jsonl is required when --judge-mode=windowed")
        return list(iter_jsonl(args.windowed_judge_jsonl))
    if args.judge_jsonl is not None:
        return list(iter_jsonl(args.judge_jsonl))
    from tools.run_no_reference_result_judge import judge_candidate_row, load_candidate_rows

    candidate_jsonl = args.candidate_jsonl or args.candidates_jsonl
    candidate_rows = load_candidate_rows(candidate_jsonl, args.candidate_manifest)
    if not candidate_rows:
        raise SystemExit("No candidate rows supplied")
    judged_rows: list[dict[str, Any]] = []
    total = len(candidate_rows)
    for index, row in enumerate(sorted(candidate_rows, key=lambda item: (str(item.get("track_id", "")), str(item.get("candidate_id", "")))), start=1):
        print(
            json.dumps(
                {
                    "event": "no_reference_optimizer_judge_progress",
                    "index": index,
                    "total": total,
                    "track_id": str(row.get("track_id") or ""),
                    "candidate_id": str(row.get("candidate_id") or ""),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        judged_rows.append(judge_candidate_row(row))
    return judged_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize existing generated vocal/minus candidates without reference stems.")
    parser.add_argument("--judge-jsonl", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path)
    parser.add_argument("--candidates-jsonl", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--judge-mode", choices=("full", "windowed", "mode_specific"), default="full")
    parser.add_argument("--windowed-judge-jsonl", type=Path)
    parser.add_argument("--vocal-judge-jsonl", type=Path)
    parser.add_argument("--minus-judge-jsonl", type=Path)
    args = parser.parse_args(argv)
    if args.judge_mode == "mode_specific":
        if args.vocal_judge_jsonl is None or args.minus_judge_jsonl is None:
            raise SystemExit("--vocal-judge-jsonl and --minus-judge-jsonl are required when --judge-mode=mode_specific")
        result = optimize_tracks_mode_specific(
            list(iter_jsonl(args.vocal_judge_jsonl)),
            list(iter_jsonl(args.minus_judge_jsonl)),
        )
    else:
        rows = load_rows_from_args(args)
        result = optimize_tracks(rows, min_delta=args.min_delta)
    if args.output_jsonl is not None or args.summary_md is not None:
        if args.output_jsonl is None or args.summary_md is None:
            raise SystemExit("--output-jsonl and --summary-md must be provided together")
        write_outputs_to_paths(result, args.output_jsonl, args.summary_md)
    elif args.output_dir is not None:
        write_outputs(result, args.output_dir)
    else:
        raise SystemExit("Provide either --output-dir or both --output-jsonl and --summary-md")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
