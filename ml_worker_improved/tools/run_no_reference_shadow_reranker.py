from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import (  # noqa: E402
    as_float,
    as_list,
    extract_features,
    iter_jsonl,
    normalize_mode,
    score_for_mode,
    stable_unique,
    write_jsonl,
)
from tools.no_reference_score_matrix import ACCEPT_THRESHOLD  # noqa: E402
from tools.run_no_reference_learned_critic import load_model_dir, safety_reasons  # noqa: E402
from tools.train_no_reference_critic import predict_binary_score, predict_policy_action  # noqa: E402


ACCEPT_DECISIONS = {"accept_candidate", "pair_compatible"}
DISAGREEMENT_TYPES = {
    "script_accept_learned_reject",
    "script_reject_learned_accept",
    "rank_order_disagreement",
    "no_disagreement",
}

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "separator_training_started": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
    "audio_read_performed": False,
    "audio_exported": False,
    "diagnostic_audio_exported": False,
    "can_publish_as_result": False,
    "can_mark_release_ready": False,
}


def candidate_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("track_id") or ""), str(row.get("candidate_id") or "")


def candidate_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("track_id") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("vocal_path") or ""),
        str(row.get("instrumental_path") or row.get("minus_path") or ""),
    )


def selected_mode(row: dict[str, Any]) -> str:
    return normalize_mode(row.get("score_mode")) or normalize_mode(row.get("source_stem")) or "both"


def is_accept_decision(decision: Any) -> bool:
    return str(decision or "") in ACCEPT_DECISIONS


def normalized_script_decision(row: dict[str, Any], mode: str) -> str:
    if mode == "vocal":
        return str(row.get("vocal_decision") or row.get("decision") or "")
    if mode == "minus":
        return str(row.get("minus_decision") or row.get("decision") or "")
    if mode == "both":
        pair_decision = str(row.get("pair_decision") or "")
        if pair_decision == "pair_compatible":
            return "accept_candidate"
        if pair_decision:
            return pair_decision
    return str(row.get("decision") or "")


def explicit_blockers(row: dict[str, Any], mode: str) -> list[str]:
    if mode == "vocal":
        mode_blockers = as_list(row.get("vocal_blockers"))
    elif mode == "minus":
        mode_blockers = as_list(row.get("minus_blockers"))
    else:
        mode_blockers = as_list(row.get("pair_blockers"))
    return stable_unique(mode_blockers)


def script_blockers(row: dict[str, Any], mode: str, script_decision: str) -> list[str]:
    blockers = explicit_blockers(row, mode)
    blockers.extend(safety_reasons(row))
    if script_decision and not is_accept_decision(script_decision):
        blockers.append(f"script_decision:{script_decision}")
    return stable_unique(blockers)


def learned_scores(row: dict[str, Any], bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    features = extract_features(row)
    vocal_score = predict_binary_score(bundles["vocal_critic"], features)
    minus_score = predict_binary_score(bundles["minus_critic"], features)
    pair_score = predict_binary_score(bundles["pair_critic"], features)
    policy_action, policy_confidence = predict_policy_action(bundles["policy_controller"], features)
    return {
        "learned_vocal_score": round(vocal_score, 6),
        "learned_minus_score": round(minus_score, 6),
        "learned_pair_score": round(pair_score, 6),
        "learned_policy_action": policy_action,
        "learned_policy_confidence": round(policy_confidence, 6),
    }


def learned_score_for_mode(scores: dict[str, Any], mode: str) -> float:
    if mode == "vocal":
        return as_float(scores.get("learned_vocal_score"))
    if mode == "minus":
        return as_float(scores.get("learned_minus_score"))
    return as_float(scores.get("learned_pair_score"))


def decision_from_score(score: float, threshold: float = ACCEPT_THRESHOLD) -> str:
    return "accept_candidate" if float(score) >= float(threshold) else "blocked_hard_case"


def learned_decision_from_score(score: float) -> str:
    return "accept_candidate" if float(score) >= 0.5 else "blocked_hard_case"


def dominant_defect(row: dict[str, Any], mode: str, blockers: list[str]) -> str:
    for key in (
        f"{mode}_dominant_defect",
        "dominant_defect",
        "dominant_failure_reason",
        "dominant_failure_class",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    reasons = [reason for reason in as_list(row.get("reasons")) if reason != "no_reference_proxy_passed"]
    if reasons:
        return reasons[0]
    return blockers[0] if blockers else "none"


def disagreement_type(script_decision: str, learned_decision: str) -> str:
    script_accept = is_accept_decision(script_decision)
    learned_accept = is_accept_decision(learned_decision)
    if script_accept and not learned_accept:
        return "script_accept_learned_reject"
    if not script_accept and learned_accept:
        return "script_reject_learned_accept"
    return "no_disagreement"


def recommended_next_action(row: dict[str, Any], kind: str) -> str:
    if row.get("script_blockers") or kind == "script_reject_learned_accept":
        return "keep_script_decision"
    if kind == "script_accept_learned_reject":
        return "inspect_with_window_judge"
    if kind == "rank_order_disagreement":
        return "add_to_hard_case_pool"
    return "candidate_for_preview_render"


def disagreement_case(row: dict[str, Any], kind: str, *, detail: str = "") -> dict[str, Any]:
    if kind not in DISAGREEMENT_TYPES or kind == "no_disagreement":
        raise ValueError(f"Unsupported disagreement case type: {kind}")
    blockers = as_list(row.get("script_blockers"))
    reasons = [reason for reason in as_list(row.get("reasons")) if reason != "no_reference_proxy_passed"]
    script_reason = detail or ";".join(blockers or reasons or [str(row.get("script_decision") or "unknown_script_reason")])
    return {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "disagreement_type": kind,
        "script_reason": script_reason,
        "learned_confidence": round(as_float(row.get("learned_score")), 6),
        "dominant_defect": str(row.get("dominant_defect") or "none"),
        "blocker_labels": blockers,
        "recommended_next_action": recommended_next_action(row, kind),
        "script_score": round(as_float(row.get("script_score")), 6),
        "learned_score": round(as_float(row.get("learned_score")), 6),
        "hybrid_score": round(as_float(row.get("hybrid_score")), 6),
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }


def enrich_windowed_rows(
    *,
    candidate_pool_rows: Iterable[dict[str, Any]],
    windowed_judge_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(candidate_pool_rows, key=candidate_sort_key):
        key = candidate_key(row)
        if key[0] and key[1] and key not in candidates_by_key:
            candidates_by_key[key] = dict(row)
    enriched: list[dict[str, Any]] = []
    for row in windowed_judge_rows:
        payload = dict(candidates_by_key.get(candidate_key(row), {}))
        payload.update(row)
        enriched.append(payload)
    return sorted(enriched, key=candidate_sort_key)


def optimizer_context_by_track(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (str(item.get("track_id") or ""), str(item.get("selected_candidate_id") or ""))):
        track_id = str(row.get("track_id") or "")
        if track_id and track_id not in result:
            result[track_id] = dict(row)
    return result


def attach_optimizer_context(row: dict[str, Any], optimizer_by_track: dict[str, dict[str, Any]]) -> dict[str, Any]:
    optimizer_row = optimizer_by_track.get(str(row.get("track_id") or ""), {})
    if not optimizer_row:
        return row
    return {
        **row,
        "optimizer_final_decision": str(optimizer_row.get("final_decision") or ""),
        "optimizer_selected_candidate_id": str(optimizer_row.get("selected_candidate_id") or ""),
        "optimizer_input_candidate_id": str(optimizer_row.get("input_candidate_id") or ""),
        "optimizer_rollback_reason": str(optimizer_row.get("rollback_reason") or ""),
        "optimizer_reasons": as_list(optimizer_row.get("reasons")),
    }


def score_shadow_row(
    row: dict[str, Any],
    *,
    bundles: dict[str, dict[str, Any]],
    optimizer_by_track: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mode = selected_mode(row)
    mode_for_script = "both" if mode == "window" else mode
    script_decision = normalized_script_decision(row, mode_for_script)
    script_score = round(score_for_mode(row, mode), 6)
    model_scores = learned_scores(row, bundles)
    learned_score = round(learned_score_for_mode(model_scores, mode), 6)
    hybrid_score = round(0.5 * script_score + 0.5 * learned_score, 6)
    learned_decision = learned_decision_from_score(learned_score)
    hybrid_decision = decision_from_score(hybrid_score)
    blockers = script_blockers(row, mode_for_script, script_decision)
    script_accept = is_accept_decision(script_decision)
    if blockers or not script_accept:
        final_decision = script_decision if script_decision and not script_accept else "blocked_hard_case"
    else:
        final_decision = hybrid_decision
    safety_override_applied = (
        (is_accept_decision(learned_decision) or is_accept_decision(hybrid_decision))
        and not is_accept_decision(final_decision)
        and (bool(blockers) or not script_accept)
    )
    payload = {
        **row,
        **model_scores,
        "score_mode": mode,
        "script_score": script_score,
        "learned_score": learned_score,
        "hybrid_score": hybrid_score,
        "script_decision": script_decision,
        "learned_decision": learned_decision,
        "hybrid_decision": hybrid_decision,
        "final_guarded_decision": final_decision,
        "script_blockers": blockers,
        "learned_blocker_probability": round(1.0 - learned_score, 6),
        "safety_override_applied": bool(safety_override_applied),
        "disagreement_type": disagreement_type(script_decision, learned_decision),
        "dominant_defect": dominant_defect(row, mode_for_script, blockers),
        "delivery_allowed": False,
        "presentation_status": "diagnostic_shadow_reranker_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "separator_training_started": False,
        "thresholds_relaxed": False,
        "audio_exported": False,
        "diagnostic_audio_exported": False,
    }
    return attach_optimizer_context(payload, optimizer_by_track)


def top_by(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: (-as_float(row.get(score_key)), candidate_sort_key(row)))[0]


def apply_rank_disagreements(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("track_id") or "")].append(row)
    cases: list[dict[str, Any]] = []
    hybrid_changed = 0
    for track_id in sorted(grouped):
        track_rows = grouped[track_id]
        script_top = top_by(track_rows, "script_score")
        learned_top = top_by(track_rows, "learned_score")
        hybrid_top = top_by(track_rows, "hybrid_score")
        if script_top is None:
            continue
        script_top_id = str(script_top.get("candidate_id") or "")
        if learned_top is not None and str(learned_top.get("candidate_id") or "") != script_top_id:
            if learned_top.get("disagreement_type") == "no_disagreement":
                learned_top["disagreement_type"] = "rank_order_disagreement"
            cases.append(
                disagreement_case(
                    learned_top,
                    "rank_order_disagreement",
                    detail=f"learned_top_candidate_differs_from_script_top:{script_top_id}",
                )
            )
        if hybrid_top is not None and str(hybrid_top.get("candidate_id") or "") != script_top_id:
            hybrid_changed += 1
            if hybrid_top.get("disagreement_type") == "no_disagreement":
                hybrid_top["disagreement_type"] = "rank_order_disagreement"
            cases.append(
                {
                    **disagreement_case(
                        hybrid_top,
                        "rank_order_disagreement",
                        detail=f"hybrid_top_candidate_differs_from_script_top:{script_top_id}",
                    ),
                    "hybrid_top_candidate_changed": True,
                }
            )
    return rows, cases, hybrid_changed


def collect_disagreements(rows: list[dict[str, Any]], rank_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = [
        disagreement_case(row, str(row["disagreement_type"]))
        for row in rows
        if row.get("disagreement_type") in {"script_accept_learned_reject", "script_reject_learned_accept"}
    ]
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for case in cases + rank_cases:
        key = (
            str(case.get("track_id") or ""),
            str(case.get("candidate_id") or ""),
            str(case.get("disagreement_type") or ""),
            str(case.get("script_reason") or ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(case)
    return sorted(unique, key=lambda row: (str(row["track_id"]), str(row["candidate_id"]), str(row["disagreement_type"]), str(row["script_reason"])))


def build_summary(rows: list[dict[str, Any]], disagreements: list[dict[str, Any]], *, hybrid_top_candidate_changed_count: int) -> dict[str, Any]:
    disagreement_counts = Counter(str(row.get("disagreement_type") or "no_disagreement") for row in rows)
    return {
        "rows_scored": len(rows),
        "learned_accept_count": sum(1 for row in rows if is_accept_decision(row.get("learned_decision"))),
        "learned_reject_count": sum(1 for row in rows if not is_accept_decision(row.get("learned_decision"))),
        "script_accept_count": sum(1 for row in rows if is_accept_decision(row.get("script_decision"))),
        "script_reject_count": sum(1 for row in rows if not is_accept_decision(row.get("script_decision"))),
        "hybrid_accept_count": sum(1 for row in rows if is_accept_decision(row.get("hybrid_decision"))),
        "safety_overrides_applied": sum(1 for row in rows if row.get("safety_override_applied") is True),
        "script_accept_learned_reject_count": disagreement_counts.get("script_accept_learned_reject", 0),
        "script_reject_learned_accept_count": disagreement_counts.get("script_reject_learned_accept", 0),
        "rank_order_disagreement_count": sum(1 for row in disagreements if row.get("disagreement_type") == "rank_order_disagreement"),
        "hybrid_top_candidate_changed_count": int(hybrid_top_candidate_changed_count),
        "false_accept_blocked_by_script_count": sum(
            1
            for row in rows
            if is_accept_decision(row.get("learned_decision"))
            and not is_accept_decision(row.get("final_guarded_decision"))
            and (bool(row.get("script_blockers")) or not is_accept_decision(row.get("script_decision")))
        ),
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "release_ready_count": sum(1 for row in rows if row.get("release_ready") is True),
        "disagreement_type_counts": dict(sorted(Counter(str(row.get("disagreement_type") or "") for row in disagreements).items())),
        "safety": dict(SAFETY_FLAGS),
    }


def build_shadow_rerank(
    *,
    candidate_pool_rows: Iterable[dict[str, Any]],
    windowed_judge_rows: Iterable[dict[str, Any]],
    optimizer_result_rows: Iterable[dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    optimizer_by_track = optimizer_context_by_track(optimizer_result_rows)
    rows = [
        score_shadow_row(row, bundles=bundles, optimizer_by_track=optimizer_by_track)
        for row in enrich_windowed_rows(candidate_pool_rows=candidate_pool_rows, windowed_judge_rows=windowed_judge_rows)
    ]
    rows, rank_cases, hybrid_changed = apply_rank_disagreements(rows)
    rows = sorted(rows, key=candidate_sort_key)
    disagreements = collect_disagreements(rows, rank_cases)
    return {
        "rows": rows,
        "disagreements": disagreements,
        "summary": build_summary(rows, disagreements, hybrid_top_candidate_changed_count=hybrid_changed),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# No-Reference Shadow Reranker Summary", ""]
    for key in (
        "rows_scored",
        "learned_accept_count",
        "learned_reject_count",
        "script_accept_count",
        "script_reject_count",
        "hybrid_accept_count",
        "safety_overrides_applied",
        "script_accept_learned_reject_count",
        "script_reject_learned_accept_count",
        "rank_order_disagreement_count",
        "hybrid_top_candidate_changed_count",
        "false_accept_blocked_by_script_count",
        "can_publish_as_result_count",
        "can_mark_release_ready_count",
        "release_ready_count",
    ):
        lines.append(f"- {key}: {summary[key]}")
    lines.extend(["", "## Disagreement Types", ""])
    for key, value in summary["disagreement_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    result: dict[str, Any],
    *,
    output_jsonl: Path,
    summary_md: Path,
    disagreements_jsonl: Path,
) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_jsonl(disagreements_jsonl, result["disagreements"])
    write_summary_md(summary_md, result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run shadow learned critic reranking without changing script-gated decisions.")
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--optimizer-result-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--disagreements-jsonl", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "no_reference_shadow_reranker_start",
                "route": "shadow_learned_critic_reranker",
                "candidate_pool_jsonl": str(args.candidate_pool_jsonl),
                "windowed_judge_jsonl": str(args.windowed_judge_jsonl),
                "optimizer_result_jsonl": str(args.optimizer_result_jsonl),
                "model_dir": str(args.model_dir),
                "output_jsonl": str(args.output_jsonl),
                "summary_md": str(args.summary_md),
                "disagreements_jsonl": str(args.disagreements_jsonl),
                "acceleration_mode": "external_only_cpu_metadata_scoring",
                "gpu_inference_skipped": True,
                "gpu_inference_skip_reason": "shadow_reranker_scores_existing_metadata_with_local_sklearn_critics_only",
                "audio_exported": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    result = build_shadow_rerank(
        candidate_pool_rows=list(iter_jsonl(args.candidate_pool_jsonl)),
        windowed_judge_rows=list(iter_jsonl(args.windowed_judge_jsonl)),
        optimizer_result_rows=list(iter_jsonl(args.optimizer_result_jsonl)),
        bundles=load_model_dir(args.model_dir),
    )
    write_outputs(
        result,
        output_jsonl=args.output_jsonl,
        summary_md=args.summary_md,
        disagreements_jsonl=args.disagreements_jsonl,
    )
    print(
        json.dumps(
            {
                "event": "no_reference_shadow_reranker_end",
                "rows_scored": result["summary"]["rows_scored"],
                "safety_overrides_applied": result["summary"]["safety_overrides_applied"],
                "disagreement_cases": len(result["disagreements"]),
                "elapsed_sec": round(time.monotonic() - started, 3),
                "output_jsonl": str(args.output_jsonl),
                "summary_md": str(args.summary_md),
                "disagreements_jsonl": str(args.disagreements_jsonl),
                "can_publish_as_result_count": result["summary"]["can_publish_as_result_count"],
                "can_mark_release_ready_count": result["summary"]["can_mark_release_ready_count"],
                "release_ready": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
