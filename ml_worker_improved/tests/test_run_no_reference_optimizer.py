from __future__ import annotations

import json
from pathlib import Path

from tools.run_no_reference_optimizer import main, optimize_tracks, optimize_tracks_mode_specific, write_outputs


def judge_row(
    candidate_id: str,
    *,
    track_id: str = "track-a",
    mix_score: float = 0.86,
    vocal_leakage: float = 0.12,
    inst_leakage: float = 0.12,
    activity: float = 0.75,
    hallucination: float = 0.05,
    robotic: float = 0.04,
    thin: float = 0.04,
    transient: float = 0.05,
    phase: float = 0.85,
    overall: float | None = None,
    decision: str = "accept_candidate",
    reasons: list[str] | None = None,
    is_input: bool = False,
) -> dict:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": f"/tmp/{track_id}/mix.wav",
        "vocal_path": f"/tmp/{track_id}/{candidate_id}_vocal.wav",
        "instrumental_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "judge_mode": "no_reference",
        "mixture_consistency_score": mix_score,
        "vocal_leakage_in_minus_score": vocal_leakage,
        "instrumental_leakage_in_vocal_score": inst_leakage,
        "vocal_activity_confidence": activity,
        "silence_hallucination_score": hallucination,
        "robotic_artifact_score": robotic,
        "thin_vocal_score": thin,
        "transient_damage_score": transient,
        "phase_consistency_score": phase,
        "overall_no_reference_score": overall if overall is not None else mix_score,
        "confidence": 0.85,
        "decision": decision,
        "reasons": reasons or ["no_reference_proxy_passed"],
        "forbidden_actions": ["do_not_train"],
        "is_input_candidate": is_input,
    }


def test_optimizer_keeps_original_when_no_candidate_improves() -> None:
    result = optimize_tracks(
        [
            judge_row("original", mix_score=0.90, overall=0.82),
            judge_row("tiny-delta", mix_score=0.91, overall=0.83),
        ],
        min_delta=0.03,
    )
    row = result["rows"][0]
    assert row["final_decision"] == "original_candidate_kept"
    assert row["selected_candidate_id"] == "original"
    assert row["candidate_id"] == "original"
    assert row["mix_path"] == "/tmp/track-a/mix.wav"
    assert row["vocal_path"].endswith("original_vocal.wav")
    assert row["instrumental_path"].endswith("original_minus.wav")
    assert row["minus_path"].endswith("original_minus.wav")
    assert row["improvement_delta"] < 0.03
    assert row["rollback_reason"] == "fitness_improvement_below_min_delta"


def test_optimizer_selects_improved_existing_candidate() -> None:
    result = optimize_tracks(
        [
            judge_row("original", vocal_leakage=0.70, overall=0.58, decision="blocked_hard_case", reasons=["vocal_leakage_in_minus"]),
            judge_row("cleaner-minus", vocal_leakage=0.10, overall=0.86),
        ],
        min_delta=0.03,
    )
    row = result["rows"][0]
    assert row["final_decision"] == "optimized_candidate_selected"
    assert row["selected_candidate_id"] == "cleaner-minus"
    assert row["candidate_id"] == "cleaner-minus"
    assert row["mix_path"] == "/tmp/track-a/mix.wav"
    assert row["vocal_path"].endswith("cleaner-minus_vocal.wav")
    assert row["instrumental_path"].endswith("cleaner-minus_minus.wav")
    assert row["minus_path"].endswith("cleaner-minus_minus.wav")
    assert "vocal_bleed_in_minus" in row["dominant_defects_before"]
    assert row["improvement_delta"] >= 0.03
    assert row["selected_minus_path"].endswith("cleaner-minus_minus.wav")


def test_optimizer_rejects_vocal_improvement_that_destroys_minus() -> None:
    result = optimize_tracks(
        [
            judge_row("original", inst_leakage=0.72, vocal_leakage=0.12, overall=0.58, reasons=["instrumental_leakage_in_vocal"]),
            judge_row("bad-minus", inst_leakage=0.10, vocal_leakage=0.88, overall=0.86, reasons=["vocal_leakage_in_minus"]),
        ],
        min_delta=0.03,
    )
    row = result["rows"][0]
    assert row["final_decision"] == "original_candidate_kept"
    assert row["selected_candidate_id"] == "original"
    assert any(action["candidate_id"] == "bad-minus" and action["reason"] == "minus_regression_severe" for action in row["actions_rejected"])


def test_optimizer_rejects_candidate_with_new_severe_artifact() -> None:
    result = optimize_tracks(
        [
            judge_row("original", vocal_leakage=0.55, overall=0.60, reasons=["vocal_leakage_in_minus"]),
            judge_row("robotic", vocal_leakage=0.05, robotic=0.90, overall=0.88, reasons=["robotic_or_noise_artifact"]),
        ],
        min_delta=0.03,
    )
    row = result["rows"][0]
    assert row["final_decision"] == "original_candidate_kept"
    assert any(action["candidate_id"] == "robotic" and action["reason"] == "new_severe_blocker" for action in row["actions_rejected"])


def test_optimizer_routes_vocal_body_loss_away_from_stronger_cleanup() -> None:
    result = optimize_tracks(
        [
            judge_row("thin", thin=0.82, overall=0.55, reasons=["thin_or_overcleaned_vocal"], is_input=True),
            judge_row("fuller", thin=0.08, overall=0.84),
        ],
        min_delta=0.03,
    )
    row = result["rows"][0]
    assert row["final_decision"] == "optimized_candidate_selected"
    assert "vocal_body_loss" in row["dominant_defects_before"]
    assert "rollback_to_fuller_vocal_candidate" in row["actions_tried"]
    assert "stronger_debleed" not in json.dumps(row)
    assert "highpass" not in json.dumps(row)
    assert "stronger_denoise" not in json.dumps(row)


def test_optimizer_routes_vocal_tail_loss_away_from_hard_gate() -> None:
    result = optimize_tracks(
        [
            judge_row("tail-cut", overall=0.55, reasons=["vocal_tail_loss"], is_input=True),
            judge_row("tail-safe", overall=0.83, reasons=["no_reference_proxy_passed"]),
        ],
        min_delta=0.03,
    )
    row = result["rows"][0]
    assert row["final_decision"] == "optimized_candidate_selected"
    assert "vocal_tail_loss" in row["dominant_defects_before"]
    assert "tail_preserving_candidate" in row["actions_tried"]
    assert "hard_vad_gate" not in row["actions_tried"]
    assert "do_not_use_hard_vad_gate" in row["forbidden_actions"]


def test_optimizer_never_requires_stems_or_training(tmp_path: Path) -> None:
    result = optimize_tracks([judge_row("original", overall=0.82)], min_delta=0.03)
    row = result["rows"][0]
    assert "reference_path" not in row
    assert result["summary"]["safety"]["reference_required"] is False
    assert result["summary"]["safety"]["human_csv_required"] is False
    assert result["summary"]["safety"]["training_started"] is False
    assert result["summary"]["safety"]["fine_tune_started"] is False
    assert result["summary"]["safety"]["models_changed"] is False
    assert result["summary"]["safety"]["cleanup_routes_added"] is False

    output_dir = tmp_path / "out"
    write_outputs(result, output_dir)
    assert (output_dir / "no_reference_optimizer_result.jsonl").exists()
    assert (output_dir / "no_reference_optimizer_summary.md").exists()
    assert not list(output_dir.rglob("*.wav"))
    assert not list(output_dir.rglob("*.flac"))


def test_optimizer_output_is_deterministic() -> None:
    rows = [
        judge_row("b", vocal_leakage=0.20, overall=0.75),
        judge_row("a", vocal_leakage=0.60, overall=0.55, reasons=["vocal_leakage_in_minus"], is_input=True),
    ]
    first = optimize_tracks(rows, min_delta=0.03)
    second = optimize_tracks(list(reversed(rows)), min_delta=0.03)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]


def test_optimizer_can_consume_windowed_judge_jsonl(tmp_path: Path) -> None:
    judge_jsonl = tmp_path / "windowed.jsonl"
    rows = [
        judge_row("original", vocal_leakage=0.70, overall=0.58, reasons=["vocal_leakage_in_minus"], is_input=True),
        judge_row("cleaner-minus", vocal_leakage=0.10, overall=0.86),
    ]
    judge_jsonl.write_text(
        "\n".join(json.dumps({**row, "judge_mode": "no_reference_windowed"}, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "optimizer.jsonl"
    summary_md = tmp_path / "optimizer.md"

    exit_code = main(
        [
            "--judge-mode",
            "windowed",
            "--windowed-judge-jsonl",
            str(judge_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    output_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert output_rows[0]["final_decision"] == "optimized_candidate_selected"
    assert output_rows[0]["selected_candidate_id"] == "cleaner-minus"


def mode_row(candidate_id: str, *, mode: str, score: float, decision: str = "accept_candidate", track_id: str = "track-a", reasons: list[str] | None = None) -> dict:
    prefix = "vocal" if mode == "vocal" else "minus"
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": f"/tmp/{track_id}/mix.wav",
        "vocal_path": f"/tmp/{track_id}/{candidate_id}_vocal.wav",
        "instrumental_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "minus_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "score_mode": mode,
        f"{prefix}_score": score,
        f"{prefix}_decision": decision,
        f"{prefix}_blockers": [] if decision == "accept_candidate" else [f"{prefix}_blocked"],
        "reasons": reasons or ["no_reference_proxy_passed"],
        "mixture_consistency_score": 0.92,
        "phase_consistency_score": 0.92,
        "vocal_leakage_in_minus_score": 0.06,
        "instrumental_leakage_in_vocal_score": 0.06,
        "confidence": 0.8,
        "forbidden_actions": ["do_not_train"],
    }


def test_mode_specific_optimizer_selects_separate_best_vocal_and_minus() -> None:
    result = optimize_tracks_mode_specific(
        [
            mode_row("vocal-best", mode="vocal", score=0.91),
            mode_row("minus-best", mode="vocal", score=0.72),
        ],
        [
            mode_row("vocal-best", mode="minus", score=0.70),
            mode_row("minus-best", mode="minus", score=0.94),
        ],
    )

    row = result["rows"][0]
    assert row["selected_vocal_candidate_id"] == "vocal-best"
    assert row["selected_minus_candidate_id"] == "minus-best"
    assert row["same_candidate_pair"] is False
    assert row["selected_candidate_id"] is None
    assert row["candidate_id"] == "mode_specific_pair:vocal-best:minus-best"
    assert row["vocal_path"].endswith("vocal-best_vocal.wav")
    assert row["instrumental_path"].endswith("minus-best_minus.wav")
    assert row["mix_path"].endswith("mix.wav")
    assert row["final_decision"] in {"mode_specific_pair_selected", "needs_pair_reconciliation"}


def test_mode_specific_optimizer_does_not_average_vocal_and_minus() -> None:
    result = optimize_tracks_mode_specific(
        [
            mode_row("avg-good", mode="vocal", score=0.99),
            mode_row("balanced", mode="vocal", score=0.78),
        ],
        [
            mode_row("avg-good", mode="minus", score=0.20, decision="blocked_hard_case"),
            mode_row("balanced", mode="minus", score=0.80),
        ],
    )

    row = result["rows"][0]
    assert row["selected_vocal_candidate_id"] == "avg-good"
    assert row["selected_minus_candidate_id"] == "balanced"
    assert row["same_candidate_pair"] is False
    assert row["selected_minus_candidate_id"] != "avg-good"


def test_mode_specific_minus_selection_ignores_vocal_only_reasons() -> None:
    result = optimize_tracks_mode_specific(
        [
            mode_row("vocal-best", mode="vocal", score=0.80),
        ],
        [
            mode_row("minus-best", mode="minus", score=0.88, reasons=["silence_hallucination"]),
        ],
    )

    row = result["rows"][0]
    assert row["selected_minus_candidate_id"] == "minus-best"
    assert row["final_decision"] != "blocked_hard_case"


def test_mode_specific_legacy_rows_without_blockers_fall_back_to_reasons() -> None:
    result = optimize_tracks_mode_specific(
        [
            mode_row("vocal-best", mode="vocal", score=0.80),
        ],
        [
            {
                **mode_row("minus-leaky", mode="minus", score=0.88, reasons=["vocal_leakage_in_minus"]),
                "minus_blockers": None,
            },
        ],
    )

    row = result["rows"][0]
    assert row["selected_minus_candidate_id"] is None
    assert row["final_decision"] == "blocked_hard_case"


def test_mode_specific_optimizer_rejects_missing_mode_schema_despite_generic_accept() -> None:
    legacy_vocal = {
        **mode_row("legacy-vocal", mode="vocal", score=0.91),
        "overall_no_reference_score": 0.96,
        "decision": "accept_candidate",
    }
    del legacy_vocal["vocal_score"]
    del legacy_vocal["vocal_decision"]
    del legacy_vocal["vocal_blockers"]

    result = optimize_tracks_mode_specific(
        [legacy_vocal],
        [mode_row("accepted-minus", mode="minus", score=0.88)],
    )

    row = result["rows"][0]
    assert row["selected_vocal_candidate_id"] is None
    assert row["selected_minus_candidate_id"] == "accepted-minus"
    assert row["final_decision"] == "blocked_hard_case"
    assert {
        "missing_mode_specific_score:vocal_score",
        "missing_mode_specific_decision:vocal_decision",
        "missing_mode_specific_blockers:vocal_blockers",
    }.issubset(set(row["actions_rejected"][0]["mode_blockers"]))


def test_mode_specific_optimizer_rejects_missing_mode_blockers() -> None:
    vocal_without_blockers = mode_row("vocal-no-blockers", mode="vocal", score=0.91)
    del vocal_without_blockers["vocal_blockers"]

    result = optimize_tracks_mode_specific(
        [vocal_without_blockers],
        [mode_row("accepted-minus", mode="minus", score=0.88)],
    )

    row = result["rows"][0]
    assert row["selected_vocal_candidate_id"] is None
    assert row["selected_minus_candidate_id"] == "accepted-minus"
    assert row["final_decision"] == "blocked_hard_case"
    assert "missing_mode_specific_blockers:vocal_blockers" in row["actions_rejected"][0]["mode_blockers"]


def test_mode_specific_optimizer_routes_near_vocal_to_preview_segment_plan_only() -> None:
    near_vocal = {
        **mode_row("near-vocal", mode="vocal", score=0.69, decision="select_alternative_candidate"),
        "vocal_blockers": [],
        "reasons": ["silence_hallucination"],
    }
    accepted_minus = mode_row("accepted-minus", mode="minus", score=0.86)

    result = optimize_tracks_mode_specific([near_vocal], [accepted_minus])

    row = result["rows"][0]
    assert row["final_decision"] == "needs_segment_blend"
    assert row["selected_candidate_id"] is None
    assert row["selected_vocal_candidate_id"] == "near-vocal"
    assert row["selected_minus_candidate_id"] == "accepted-minus"
    assert row["segment_blend_needed"] is True
    assert row["segment_blend_plan"]["status"] == "preview_only_not_executed"
    assert row["segment_blend_plan"]["can_publish_as_result"] is False
    assert row["segment_blend_plan"]["can_mark_release_ready"] is False
    assert "no_best_of_bad" in row["reasons"]
    assert result["summary"]["needs_segment_blend"] == 1
    assert result["summary"]["usable_without_stems"] == 0
