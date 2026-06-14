from __future__ import annotations

import json
from pathlib import Path

from tools.autonomous_audio_judge import judge_case


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def benchmark_row(
    idx: int,
    *,
    source_manifest: str = "",
    split: str = "train",
    reason: str = "selected_candidate_still_has_severe_issue",
    labels: list[str] | None = None,
    failed_checks: list[str] | None = None,
    track_hash: str | None = None,
) -> dict:
    return {
        "benchmark_case_id": f"bench-{idx:03d}",
        "source_hard_case_id": f"hard-{idx:03d}",
        "track_hash": track_hash or f"track-{idx:03d}",
        "split": split,
        "stem_type": "vocal",
        "dominant_failure_reason": reason,
        "failure_labels": labels if labels is not None else [reason],
        "failed_checks": failed_checks if failed_checks is not None else [],
        "severity_score": 95,
        "window_start_sec": float(idx),
        "window_end_sec": float(idx + 5),
        "source_manifest": source_manifest,
        "recommended_review_type": "autonomous_control_plane_review",
    }


def review_row(row: dict, *, allowed_next_step: str = "finetune_candidate") -> dict:
    return {
        "benchmark_case_id": row["benchmark_case_id"],
        "track_hash": row["track_hash"],
        "stem_type": row["stem_type"],
        "dominant_failure_reason": row["dominant_failure_reason"],
        "failure_labels": row["failure_labels"],
        "severity_score": row["severity_score"],
        "recommended_review_type": row["recommended_review_type"],
        "review_priority": "P0",
        "expected_audio_problem": "test problem",
        "human_rating": None,
        "human_notes": None,
        "reviewer_decision": None,
        "allowed_next_step": allowed_next_step,
    }


def test_reference_aware_promotable_manifest_can_be_accepted(tmp_path: Path) -> None:
    manifest = write_json(
        tmp_path / "accepted.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "stem_type": "vocal",
            "metrics": {
                "sdr_db": 18.0,
                "si_sdr_db": 17.0,
                "sir_db": 22.0,
                "sar_db": 19.0,
                "reference_likeness_percent": 97.5,
                "residual_rms_dbfs": -70.0,
                "residual_peak_dbfs": -51.0,
            },
        },
    )
    row = benchmark_row(1, source_manifest=str(manifest), reason="accepted", labels=[], failed_checks=[])

    decision = judge_case(row, review_row(row, allowed_next_step="listen_pack"), golden_track_hashes=set())

    assert decision["judge_mode"] == "reference_aware"
    assert decision["decision"] == "accepted_for_listen_pack"
    assert decision["confidence_band"] in {"high", "medium"}
    assert decision["human_review_required"] is False


def test_low_confidence_manifest_only_never_accepted_or_release_ready() -> None:
    row = benchmark_row(2, reason="unknown_blocked_reason", labels=[], failed_checks=[])

    decision = judge_case(row, None, golden_track_hashes=set())

    assert decision["judge_mode"] == "manifest_only"
    assert decision["confidence"] <= 0.65
    assert decision["decision"] in {"blocked_from_listen_pack", "hard_case"}
    assert decision["decision"] not in {"accepted_for_listen_pack", "release_ready"}
    assert decision["human_review_required"] is False


def test_delivery_allowed_false_never_promotes(tmp_path: Path) -> None:
    manifest = write_json(
        tmp_path / "blocked.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": False,
            "presentation_status": "accepted_for_listen_pack",
            "dominant_failure_reason": "accepted_looking_but_blocked",
        },
    )
    row = benchmark_row(3, source_manifest=str(manifest), reason="accepted", labels=[])

    decision = judge_case(row, review_row(row), golden_track_hashes=set())

    assert decision["decision"] == "blocked_from_listen_pack"
    assert "delivery_allowed_false" in decision["reasons"]


def test_diagnostic_only_artifact_is_never_accepted(tmp_path: Path) -> None:
    manifest = write_json(
        tmp_path / "diagnostic.json",
        {
            "delivery_status": "diagnostic_only",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_only",
        },
    )
    row = benchmark_row(4, source_manifest=str(manifest), labels=["diagnostic_only"])

    decision = judge_case(row, review_row(row), golden_track_hashes=set())

    assert decision["decision"] == "blocked_from_listen_pack"
    assert decision["decision"] not in {"accepted_for_listen_pack", "release_ready"}


def test_release_ready_with_delivery_allowed_false_is_blocked(tmp_path: Path) -> None:
    manifest = write_json(
        tmp_path / "release_blocked.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": False,
            "release_status": "release_ready",
        },
    )
    row = benchmark_row(5, source_manifest=str(manifest), reason="accepted", labels=[])

    decision = judge_case(row, review_row(row), golden_track_hashes=set())

    assert decision["decision"] == "blocked_from_listen_pack"


def test_candidate_space_exhausted_returns_hard_case_and_forbids_cleanup() -> None:
    row = benchmark_row(
        6,
        reason="candidate_space_exhausted",
        labels=["candidate_space_exhausted"],
        failed_checks=["candidate_space_exhausted"],
    )

    decision = judge_case(row, review_row(row, allowed_next_step="manual_ab_only"), golden_track_hashes=set())

    assert decision["decision"] == "hard_case"
    assert decision["confidence_band"] == "high"
    assert any("cleanup" in action for action in decision["forbidden_actions"])
    assert decision["human_review_required"] is False


def test_no_accepted_candidate_does_not_export_audio() -> None:
    row = benchmark_row(7, reason="no_accepted_candidate", labels=["no_accepted_candidate"])

    decision = judge_case(row, review_row(row, allowed_next_step="manual_ab_only"), golden_track_hashes=set())

    assert decision["decision"] in {"hard_case", "finetune_candidate"}
    assert decision["decision"] not in {"accepted_for_listen_pack", "release_ready"}
    assert "no_audio_export" in decision["forbidden_actions"]


def test_holdout_cannot_become_finetune_candidate() -> None:
    row = benchmark_row(8, split="holdout", reason="thin_or_overcleaned", labels=["thin_or_overcleaned"])

    decision = judge_case(row, review_row(row, allowed_next_step="finetune_candidate"), golden_track_hashes=set())

    assert decision["decision"] != "finetune_candidate"
    assert "holdout_cannot_enter_finetune" in decision["reasons"]


def test_golden_track_cannot_become_training_or_policy_candidate() -> None:
    row = benchmark_row(
        9,
        reason="selected_candidate_has_unstable_vocal_consistency",
        labels=["selected_candidate_has_unstable_vocal_consistency"],
        track_hash="golden-track",
    )

    decision = judge_case(row, review_row(row, allowed_next_step="policy_fix_candidate"), golden_track_hashes={"golden-track"})

    assert decision["decision"] not in {"finetune_candidate", "needs_policy_fix"}
    assert "golden_track_protected_from_training_or_tuning" in decision["reasons"]
