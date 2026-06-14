from __future__ import annotations

import json
from pathlib import Path

from tools.build_no_reference_ml_dataset import FEATURE_NAMES
from tools.run_no_reference_shadow_reranker import build_shadow_rerank, write_outputs


def fallback_bundles(threshold: float = 0.5) -> dict:
    binary = {
        "model_type": "threshold_fallback",
        "threshold": threshold,
        "feature_list": list(FEATURE_NAMES),
    }
    return {
        "vocal_critic": dict(binary),
        "minus_critic": dict(binary),
        "pair_critic": dict(binary),
        "policy_controller": {
            "model_type": "majority_policy_fallback",
            "majority_action": "accept_candidate",
            "feature_list": list(FEATURE_NAMES),
        },
    }


def candidate_row(
    track_id: str,
    candidate_id: str,
    *,
    score_mode: str = "both",
    decision: str = "accept_candidate",
    overall: float = 0.8,
    vocal_score: float | None = None,
    minus_score: float | None = None,
    pair_score: float | None = None,
    reasons: list[str] | None = None,
) -> dict:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "score_mode": score_mode,
        "mix_path": f"/tmp/{track_id}/mix.wav",
        "vocal_path": f"/tmp/{track_id}/{candidate_id}_vocal.wav",
        "instrumental_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "minus_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "overall_no_reference_score": overall,
        "vocal_score": vocal_score if vocal_score is not None else overall,
        "minus_score": minus_score if minus_score is not None else overall,
        "pair_compatibility_score": pair_score if pair_score is not None else overall,
        "confidence": overall,
        "decision": decision,
        "reasons": reasons or ["no_reference_proxy_passed"],
        "forbidden_actions": ["do_not_train", "do_not_export_diagnostic_audio_as_result"],
        "window_count": 1,
    }


def test_learned_accept_cannot_bypass_script_blocker() -> None:
    blocked = candidate_row(
        "track-a",
        "unsafe-minus",
        score_mode="minus",
        decision="blocked_hard_case",
        overall=0.96,
        minus_score=0.93,
        reasons=["vocal_leakage_in_minus"],
    )

    result = build_shadow_rerank(
        candidate_pool_rows=[],
        windowed_judge_rows=[blocked],
        optimizer_result_rows=[],
        bundles=fallback_bundles(threshold=0.5),
    )

    row = result["rows"][0]
    assert row["learned_decision"] == "accept_candidate"
    assert row["script_decision"] == "blocked_hard_case"
    assert row["final_guarded_decision"] == "blocked_hard_case"
    assert row["safety_override_applied"] is True
    assert row["script_blockers"]
    assert row["disagreement_type"] == "script_reject_learned_accept"
    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False
    assert row["release_ready"] is False
    assert row["audio_exported"] is False
    assert result["summary"]["false_accept_blocked_by_script_count"] == 1
    assert result["summary"]["can_publish_as_result_count"] == 0
    assert result["summary"]["can_mark_release_ready_count"] == 0


def test_disagreements_are_written_and_hybrid_score_is_exact(tmp_path: Path) -> None:
    script_accept_learned_reject = candidate_row(
        "track-b",
        "script-accept",
        score_mode="minus",
        decision="accept_candidate",
        overall=0.2,
        minus_score=0.98,
    )
    script_reject_learned_accept = candidate_row(
        "track-b",
        "script-reject",
        score_mode="both",
        decision="blocked_hard_case",
        overall=0.94,
        pair_score=0.94,
        reasons=["phase_or_stereo_inconsistency"],
    )
    optimizer_rows = [
        {
            "track_id": "track-b",
            "final_decision": "original_candidate_kept",
            "selected_candidate_id": "script-accept",
        }
    ]

    result = build_shadow_rerank(
        candidate_pool_rows=[],
        windowed_judge_rows=[script_reject_learned_accept, script_accept_learned_reject],
        optimizer_result_rows=optimizer_rows,
        bundles=fallback_bundles(threshold=0.5),
    )
    write_outputs(
        result,
        output_jsonl=tmp_path / "shadow.jsonl",
        summary_md=tmp_path / "summary.md",
        disagreements_jsonl=tmp_path / "disagreements.jsonl",
    )

    rows = [json.loads(line) for line in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()]
    disagreements = [json.loads(line) for line in (tmp_path / "disagreements.jsonl").read_text(encoding="utf-8").splitlines()]
    by_candidate = {row["candidate_id"]: row for row in rows}
    accept_row = by_candidate["script-accept"]
    reject_row = by_candidate["script-reject"]

    assert accept_row["hybrid_score"] == round(0.5 * accept_row["script_score"] + 0.5 * accept_row["learned_score"], 6)
    assert reject_row["hybrid_score"] == round(0.5 * reject_row["script_score"] + 0.5 * reject_row["learned_score"], 6)
    assert accept_row["disagreement_type"] == "script_accept_learned_reject"
    assert reject_row["disagreement_type"] == "script_reject_learned_accept"
    assert {row["disagreement_type"] for row in disagreements} >= {
        "script_accept_learned_reject",
        "script_reject_learned_accept",
        "rank_order_disagreement",
    }
    rank_cases = [row for row in disagreements if row["disagreement_type"] == "rank_order_disagreement"]
    assert rank_cases
    assert all("top_candidate_differs_from_script_top" in row["script_reason"] for row in rank_cases)
    assert all(row["recommended_next_action"] in {
        "inspect_with_window_judge",
        "add_to_hard_case_pool",
        "candidate_for_preview_render",
        "keep_script_decision",
    } for row in disagreements)
    assert "hybrid_top_candidate_changed_count" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_output_is_deterministic_and_never_exports_audio(tmp_path: Path) -> None:
    rows = [
        candidate_row("track-c", "b", score_mode="vocal", overall=0.7, vocal_score=0.73),
        candidate_row("track-c", "a", score_mode="vocal", overall=0.8, vocal_score=0.84),
    ]
    first = build_shadow_rerank(
        candidate_pool_rows=[],
        windowed_judge_rows=list(reversed(rows)),
        optimizer_result_rows=[],
        bundles=fallback_bundles(threshold=0.5),
    )
    second = build_shadow_rerank(
        candidate_pool_rows=[],
        windowed_judge_rows=rows,
        optimizer_result_rows=[],
        bundles=fallback_bundles(threshold=0.5),
    )

    assert first == second
    assert all(row["can_publish_as_result"] is False for row in first["rows"])
    assert all(row["can_mark_release_ready"] is False for row in first["rows"])
    assert first["summary"]["safety"]["audio_exported"] is False
    assert first["summary"]["safety"]["training_started"] is False
    assert first["summary"]["safety"]["models_changed"] is False

    output_dir = tmp_path / "out"
    write_outputs(
        first,
        output_jsonl=output_dir / "shadow.jsonl",
        summary_md=output_dir / "summary.md",
        disagreements_jsonl=output_dir / "disagreements.jsonl",
    )
    assert not list(output_dir.rglob("*.wav"))
    assert not list(output_dir.rglob("*.flac"))


def test_false_accept_blocked_by_script_counts_only_script_guards() -> None:
    script_accept_hybrid_reject = candidate_row(
        "track-d",
        "script-accept-low-hybrid",
        score_mode="minus",
        decision="accept_candidate",
        overall=0.96,
        minus_score=0.2,
    )

    result = build_shadow_rerank(
        candidate_pool_rows=[],
        windowed_judge_rows=[script_accept_hybrid_reject],
        optimizer_result_rows=[],
        bundles=fallback_bundles(threshold=0.5),
    )

    row = result["rows"][0]
    assert row["script_decision"] == "accept_candidate"
    assert row["learned_decision"] == "accept_candidate"
    assert row["hybrid_decision"] == "blocked_hard_case"
    assert row["final_guarded_decision"] == "blocked_hard_case"
    assert row["safety_override_applied"] is False
    assert result["summary"]["false_accept_blocked_by_script_count"] == 0


def test_rank_order_disagreement_writes_only_detailed_case() -> None:
    learned_top_by_tie = candidate_row(
        "track-e",
        "a-learned-tie-top",
        score_mode="minus",
        decision="accept_candidate",
        overall=0.9,
        minus_score=0.74,
    )
    script_top = candidate_row(
        "track-e",
        "b-script-top",
        score_mode="minus",
        decision="accept_candidate",
        overall=0.9,
        minus_score=0.91,
    )

    result = build_shadow_rerank(
        candidate_pool_rows=[],
        windowed_judge_rows=[script_top, learned_top_by_tie],
        optimizer_result_rows=[],
        bundles=fallback_bundles(threshold=0.5),
    )

    rank_cases = [row for row in result["disagreements"] if row["disagreement_type"] == "rank_order_disagreement"]
    assert len(rank_cases) == 1
    assert rank_cases[0]["candidate_id"] == "a-learned-tie-top"
    assert rank_cases[0]["script_reason"] == "learned_top_candidate_differs_from_script_top:b-script-top"
