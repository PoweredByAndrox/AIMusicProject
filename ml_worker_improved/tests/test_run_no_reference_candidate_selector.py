from __future__ import annotations

import json
from pathlib import Path

from tools.run_no_reference_candidate_selector import select_candidates, write_outputs


def judge_row(
    candidate_id: str,
    *,
    track_id: str = "track-a",
    score: float,
    confidence: float = 0.8,
    decision: str = "accept_candidate",
    severe: bool = False,
    reasons: list[str] | None = None,
    score_mode: str = "both",
    vocal_score: float | None = None,
    minus_score: float | None = None,
    vocal_decision: str | None = None,
    minus_decision: str | None = None,
    vocal_blockers: list[str] | None = None,
    minus_blockers: list[str] | None = None,
    minus_detail_gate: dict | None = None,
) -> dict:
    row = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": f"/tmp/{track_id}/mix.wav",
        "vocal_path": f"/tmp/{track_id}/{candidate_id}_vocal.wav",
        "instrumental_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "judge_mode": "no_reference",
        "score_mode": score_mode,
        "overall_no_reference_score": score,
        "vocal_score": score if vocal_score is None else vocal_score,
        "minus_score": score if minus_score is None else minus_score,
        "vocal_decision": decision if vocal_decision is None else vocal_decision,
        "minus_decision": decision if minus_decision is None else minus_decision,
        "vocal_blockers": [] if vocal_blockers is None else vocal_blockers,
        "minus_blockers": [] if minus_blockers is None else minus_blockers,
        "confidence": confidence,
        "decision": decision,
        "reasons": reasons or ([] if not severe else ["vocal_leakage_in_minus"]),
        "forbidden_actions": ["do_not_train"],
    }
    if minus_detail_gate is not None:
        row["minus_musical_detail_preservation_gate"] = minus_detail_gate
    return row


def test_best_candidate_is_selected_when_one_is_consistently_better() -> None:
    result = select_candidates(
        [
            judge_row("baseline", score=0.55, decision="select_alternative_candidate", reasons=["thin_vocal"]),
            judge_row("candidate-b", score=0.84),
        ]
    )
    row = result["rows"][0]
    assert row["decision"] == "select_alternative_candidate"
    assert row["selected_candidate_id"] == "candidate-b"
    assert row["accepted_candidate_count"] == 1


def test_best_of_bad_is_not_accepted() -> None:
    result = select_candidates(
        [
            judge_row("bad-a", score=0.40, confidence=0.9, decision="blocked_hard_case", severe=True),
            judge_row("bad-b", score=0.50, confidence=0.9, decision="select_alternative_candidate", reasons=["bad_mixture_reconstruction"]),
        ]
    )
    row = result["rows"][0]
    assert row["decision"] == "blocked_hard_case"
    assert row["selected_candidate_id"] is None
    assert "no_acceptable_candidate" in row["reasons"]


def test_low_confidence_blocks_even_with_good_score() -> None:
    result = select_candidates([judge_row("uncertain", score=0.86, confidence=0.40)])
    row = result["rows"][0]
    assert row["decision"] == "blocked_hard_case"
    assert row["selected_candidate_id"] is None
    assert "low_confidence" in row["reasons"]


def test_close_complementary_candidates_emit_segment_blend_plan_only() -> None:
    result = select_candidates(
        [
            judge_row("a", score=0.81, reasons=["segment_winner_low_band"]),
            judge_row("b", score=0.80, reasons=["segment_winner_high_band"]),
        ]
    )
    row = result["rows"][0]
    assert row["decision"] == "needs_segment_blend"
    assert row["selected_candidate_id"] is None
    assert row["segment_blend_plan"]["status"] == "plan_only_not_executed"


def test_vocal_selector_uses_vocal_matrix_not_overall_score() -> None:
    result = select_candidates(
        [
            judge_row(
                "overall-winner-bad-vocal",
                score=0.95,
                score_mode="vocal",
                vocal_score=0.20,
                vocal_decision="blocked_hard_case",
                vocal_blockers=["severe_instrumental_bleed_in_vocal"],
                minus_score=0.95,
                reasons=["vocal_leakage_in_minus"],
            ),
            judge_row("vocal-winner", score=0.73, score_mode="vocal", vocal_score=0.84, minus_score=0.30),
        ],
        score_mode="vocal",
    )

    row = result["rows"][0]
    assert row["selected_candidate_id"] == "vocal-winner"
    assert row["ranked_candidates"][0]["candidate_id"] == "vocal-winner"
    assert row["ranked_candidates"][0]["selection_score"] == 0.84
    assert row["score_mode"] == "vocal"


def test_minus_selector_uses_minus_matrix_not_overall_score() -> None:
    result = select_candidates(
        [
            judge_row(
                "overall-winner-bad-minus",
                score=0.95,
                score_mode="minus",
                vocal_score=0.95,
                minus_score=0.20,
                minus_decision="blocked_hard_case",
                minus_blockers=["severe_vocal_bleed_in_minus"],
            ),
            judge_row("minus-winner", score=0.73, score_mode="minus", vocal_score=0.30, minus_score=0.86),
        ],
        score_mode="minus",
    )

    row = result["rows"][0]
    assert row["selected_candidate_id"] == "minus-winner"
    assert row["ranked_candidates"][0]["candidate_id"] == "minus-winner"
    assert row["ranked_candidates"][0]["selection_score"] == 0.86
    assert row["score_mode"] == "minus"


def test_vocal_selector_rejects_missing_mode_schema_despite_generic_accept() -> None:
    legacy_vocal = judge_row("legacy-vocal", score=0.96, score_mode="vocal")
    del legacy_vocal["vocal_score"]
    del legacy_vocal["vocal_decision"]
    del legacy_vocal["vocal_blockers"]

    result = select_candidates([legacy_vocal], score_mode="vocal")

    row = result["rows"][0]
    assert row["decision"] == "blocked_hard_case"
    assert row["selected_candidate_id"] is None
    assert row["accepted_candidate_count"] == 0
    assert row["ranked_candidates"][0]["selection_score"] == 0.0
    assert row["ranked_candidates"][0]["decision"] == "blocked_hard_case"
    assert {
        "missing_mode_specific_score:vocal_score",
        "missing_mode_specific_decision:vocal_decision",
        "missing_mode_specific_blockers:vocal_blockers",
    }.issubset(set(row["reasons"]))


def test_minus_selector_rejects_missing_mode_blockers() -> None:
    minus_without_blockers = judge_row("minus-no-blockers", score=0.90, score_mode="minus")
    del minus_without_blockers["minus_blockers"]

    result = select_candidates([minus_without_blockers], score_mode="minus")

    row = result["rows"][0]
    assert row["decision"] == "blocked_hard_case"
    assert row["selected_candidate_id"] is None
    assert row["accepted_candidate_count"] == 0
    assert "missing_mode_specific_blockers:minus_blockers" in row["reasons"]


def test_minus_selector_blocks_proxy_blocked_detail_gate_even_when_minus_score_accepts() -> None:
    result = select_candidates(
        [
            judge_row(
                "phase-collapsed-minus",
                score=0.81,
                score_mode="minus",
                minus_score=0.95,
                minus_decision="accept_candidate",
                minus_blockers=[],
                reasons=[],
                minus_detail_gate={
                    "status": "proxy_blocked",
                    "blockers": [
                        {
                            "label": "phase_stereo_stability_below_detail_gate",
                            "reason": "minus_candidate_loses_music_detail",
                        }
                    ],
                },
            )
        ],
        score_mode="minus",
    )

    row = result["rows"][0]
    assert row["decision"] == "blocked_hard_case"
    assert row["selected_candidate_id"] is None
    assert "minus_detail_gate_blocked" in row["reasons"]
    assert result["summary"]["accepted_candidates"] == 0


def test_output_is_deterministic_and_summary_matches(tmp_path: Path) -> None:
    rows = [judge_row("a", score=0.81), judge_row("b", score=0.52, decision="blocked_hard_case", severe=True)]
    first = select_candidates(rows)
    second = select_candidates(rows)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]
    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "no_reference_candidate_selection.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(output_rows) == first["summary"]["total_tracks"]
    assert not list(output_dir.rglob("*.wav"))
    assert first["summary"]["safety"]["reference_required"] is False
    assert first["summary"]["safety"]["human_csv_required"] is False
    assert first["summary"]["safety"]["training_started"] is False
    assert first["summary"]["safety"]["models_changed"] is False
    assert first["summary"]["safety"]["cleanup_routes_added"] is False
    assert first["summary"]["safety"]["thresholds_relaxed"] is False
