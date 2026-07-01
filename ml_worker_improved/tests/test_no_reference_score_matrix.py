from __future__ import annotations

import math

import pytest

from tools.no_reference_score_matrix import (
    SCORE_WEIGHTS,
    allowed_actions_for_defect,
    detect_blockers,
    dominant_defect,
    pair_compatibility_score,
    score_candidate,
)


def good_metrics(**overrides: float | int | str | list[str]) -> dict:
    metrics = {
        "mixture_consistency_score": 0.94,
        "vocal_leakage_in_minus_score": 0.08,
        "instrumental_leakage_in_vocal_score": 0.08,
        "vocal_activity_confidence": 0.86,
        "silence_hallucination_score": 0.04,
        "robotic_artifact_score": 0.04,
        "thin_vocal_score": 0.08,
        "transient_damage_score": 0.06,
        "phase_consistency_score": 0.92,
        "confidence": 0.82,
        "reasons": ["no_reference_proxy_passed"],
        "sample_rate": 44_100,
        "channels": 2,
        "duration_sec": 180.0,
    }
    metrics.update(overrides)
    return metrics


@pytest.mark.parametrize("mode", ["vocal", "minus", "both"])
def test_weights_sum_to_one(mode: str) -> None:
    assert round(sum(SCORE_WEIGHTS[mode].values()), 6) == 1.0


def test_invalid_mode_fails_with_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported no-reference score mode"):
        score_candidate(good_metrics(), "shared")


def test_vocal_mode_detects_severe_instrumental_bleed_in_vocal() -> None:
    result = score_candidate(good_metrics(instrumental_leakage_in_vocal_score=0.86), "vocal")

    assert result["score_mode"] == "vocal"
    assert "severe_instrumental_bleed_in_vocal" in result["vocal_blockers"]
    assert result["vocal_decision"] == "blocked_hard_case"
    assert dominant_defect(good_metrics(instrumental_leakage_in_vocal_score=0.86), "vocal") == "instrumental_bleed_in_vocal"


def test_minus_mode_detects_severe_vocal_bleed_in_minus() -> None:
    result = score_candidate(good_metrics(vocal_leakage_in_minus_score=0.82), "minus")

    assert result["score_mode"] == "minus"
    assert "severe_vocal_bleed_in_minus" in result["minus_blockers"]
    assert result["minus_decision"] == "blocked_hard_case"
    assert dominant_defect(good_metrics(vocal_leakage_in_minus_score=0.82), "minus") == "vocal_bleed_in_minus"


def test_minus_mode_blocks_missing_vocal_leakage_evidence() -> None:
    metrics = good_metrics()
    metrics.pop("vocal_leakage_in_minus_score")

    result = score_candidate(metrics, "minus")

    assert "missing_minus_vocal_leakage_evidence" in result["minus_blockers"]
    assert result["minus_decision"] == "blocked_hard_case"


def test_minus_mode_blocks_missing_instrumental_integrity_evidence() -> None:
    metrics = good_metrics()
    metrics.pop("mixture_consistency_score")

    result = score_candidate(metrics, "minus")

    assert "missing_minus_instrumental_integrity_evidence" in result["minus_blockers"]
    assert result["minus_decision"] == "blocked_hard_case"


@pytest.mark.parametrize("bad_value", [None, "", math.nan, math.inf, -math.inf])
def test_minus_mode_blocks_invalid_vocal_leakage_evidence(bad_value: object) -> None:
    result = score_candidate(good_metrics(vocal_leakage_in_minus_score=bad_value), "minus")

    assert "missing_minus_vocal_leakage_evidence" in result["minus_blockers"]
    assert result["minus_decision"] == "blocked_hard_case"


@pytest.mark.parametrize("bad_value", [None, "", math.nan, math.inf, -math.inf])
def test_minus_mode_blocks_invalid_audio_info_integrity_evidence(bad_value: object) -> None:
    metrics = good_metrics()
    metrics.pop("mixture_consistency_score")
    metrics["audio_info"] = {"mixture_consistency_score": bad_value}

    result = score_candidate(metrics, "minus")

    assert "missing_minus_instrumental_integrity_evidence" in result["minus_blockers"]
    assert result["minus_decision"] == "blocked_hard_case"


def test_minus_mode_accepts_required_evidence_from_audio_info() -> None:
    metrics = good_metrics()
    metrics.pop("vocal_leakage_in_minus_score")
    metrics.pop("mixture_consistency_score")
    metrics["audio_info"] = {
        "vocal_leakage_in_minus_score": 0.04,
        "mixture_consistency_score": 0.94,
    }

    result = score_candidate(metrics, "minus")

    assert "missing_minus_vocal_leakage_evidence" not in result["minus_blockers"]
    assert "missing_minus_instrumental_integrity_evidence" not in result["minus_blockers"]
    assert result["minus_decision"] == "accept_candidate"


def test_both_mode_detects_sample_rate_mismatch() -> None:
    result = pair_compatibility_score(
        good_metrics(candidate_id="vocal-a", sample_rate=44_100),
        good_metrics(candidate_id="minus-b", sample_rate=48_000),
    )

    assert result["score_mode"] == "both"
    assert "sample_rate_mismatch" in result["pair_blockers"]
    assert result["pair_decision"] == "blocked_pair_incompatible"


def test_candidate_can_pass_vocal_and_fail_minus() -> None:
    metrics = good_metrics(vocal_leakage_in_minus_score=0.84, instrumental_leakage_in_vocal_score=0.04)

    assert score_candidate(metrics, "vocal")["vocal_decision"] == "accept_candidate"
    assert score_candidate(metrics, "minus")["minus_decision"] == "blocked_hard_case"


def test_candidate_can_pass_minus_and_fail_vocal() -> None:
    metrics = good_metrics(vocal_leakage_in_minus_score=0.04, instrumental_leakage_in_vocal_score=0.84)

    assert score_candidate(metrics, "minus")["minus_decision"] == "accept_candidate"
    assert score_candidate(metrics, "vocal")["vocal_decision"] == "blocked_hard_case"


def test_vocal_mode_uses_vocal_phase_not_minus_phase() -> None:
    metrics = good_metrics(
        phase_consistency_score=0.12,
        pair_phase_consistency_score=0.12,
        vocal_phase_consistency_score=0.94,
        instrumental_phase_consistency_score=0.12,
    )

    result = score_candidate(metrics, "vocal")

    assert result["vocal_decision"] == "accept_candidate"
    assert "severe_phase_comb_filtering" not in result["vocal_blockers"]


def test_vocal_mode_ignores_pair_phase_reason_when_vocal_phase_is_good() -> None:
    metrics = good_metrics(
        phase_consistency_score=0.12,
        pair_phase_consistency_score=0.12,
        vocal_phase_consistency_score=0.94,
        instrumental_phase_consistency_score=0.12,
        reasons=["phase_or_stereo_inconsistency"],
    )

    result = score_candidate(metrics, "vocal")

    assert result["vocal_decision"] == "accept_candidate"
    assert "severe_phase_comb_filtering" not in result["vocal_blockers"]


def test_minus_mode_uses_instrumental_phase_not_vocal_phase() -> None:
    metrics = good_metrics(
        phase_consistency_score=0.12,
        pair_phase_consistency_score=0.12,
        vocal_phase_consistency_score=0.12,
        instrumental_phase_consistency_score=0.94,
    )

    result = score_candidate(metrics, "minus")

    assert result["minus_decision"] == "accept_candidate"
    assert "severe_phase_collapse" not in result["minus_blockers"]


def test_allowed_actions_are_mode_specific_and_safe() -> None:
    vocal_actions = allowed_actions_for_defect("vocal_body_loss", "vocal")
    minus_actions = allowed_actions_for_defect("vocal_bleed_in_minus", "minus")
    phase_actions = allowed_actions_for_defect("phase_comb_filtering", "vocal")

    assert "rollback_to_fuller_vocal_candidate" in vocal_actions["allowed_actions"]
    assert "stronger_debleed" in vocal_actions["forbidden_actions"]
    assert "choose_candidate_with_lower_vocal_leakage" in minus_actions["allowed_actions"]
    assert "stronger_vocal_cleanup" in minus_actions["forbidden_actions"]
    assert "bounded_phase_alignment_candidate" in phase_actions["allowed_actions"]
    assert "threshold_relaxation" in phase_actions["forbidden_actions"]


def test_detect_blockers_and_score_are_deterministic() -> None:
    metrics = good_metrics(vocal_leakage_in_minus_score=0.84)

    assert detect_blockers(metrics, "minus") == detect_blockers(metrics, "minus")
    assert score_candidate(metrics, "minus") == score_candidate(dict(reversed(list(metrics.items()))), "minus")
