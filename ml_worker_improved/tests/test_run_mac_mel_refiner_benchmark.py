from __future__ import annotations

from tools.run_mac_mel_refiner_benchmark import (
    ALLOWED_BENCHMARK_OUTCOMES,
    _score_clean_row,
    decide_benchmark_outcome,
    summarize_benchmark_taxonomy,
    validate_benchmark_row,
)


def _row(
    *,
    row_id: str,
    row_type: str,
    checkpoint_id: str = "g1.5",
    before_regions: int = 1,
    after_regions: int = 0,
    improved: bool = True,
    damage_regressed: bool = False,
    delivery_allowed: bool = False,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "row_type": row_type,
        "checkpoint_id": checkpoint_id,
        "before_regions": before_regions,
        "after_regions": after_regions,
        "before_best_score": 0.6,
        "after_best_score": 0.2,
        "target_issue_improved": improved,
        "damage_regressed": damage_regressed,
        "delivery_allowed": delivery_allowed,
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "valid_for_decision": True,
    }


def test_benchmark_rejects_pseudo_teacher_ground_truth() -> None:
    row = _row(row_id="r001", row_type="source_hardcase")
    row["is_ground_truth"] = True

    try:
        validate_benchmark_row(row)
    except ValueError as exc:
        assert "pseudo_teacher" in str(exc)
    else:
        raise AssertionError("pseudo_teacher row marked as ground truth was accepted")


def test_benchmark_outcome_enum_and_no_delivery() -> None:
    rows = [
        _row(row_id="src", row_type="source_hardcase"),
        _row(row_id="var", row_type="synthetic_variant"),
        _row(row_id="clean", row_type="clean_negative", before_regions=0, after_regions=0, improved=False),
    ]

    result = decide_benchmark_outcome(rows, checkpoint_order=["g1.5", "g2.0", "g2.5"])

    assert result["outcome"] in ALLOWED_BENCHMARK_OUTCOMES
    assert result["outcome"] == "promote_to_next_diagnostic_stage"
    assert result["selected_checkpoint_id"] == "g1.5"
    assert result["delivery_allowed"] is False


def test_candidate_delivery_allowed_is_error() -> None:
    row = _row(row_id="r001", row_type="source_hardcase", delivery_allowed=True)

    try:
        validate_benchmark_row(row)
    except ValueError as exc:
        assert "delivery_allowed" in str(exc)
    else:
        raise AssertionError("benchmark row with delivery_allowed=true was accepted")


def test_clean_no_leak_regression_prevents_promotion() -> None:
    rows = [
        _row(row_id="src", row_type="source_hardcase"),
        _row(row_id="var", row_type="synthetic_variant"),
        _row(
            row_id="clean",
            row_type="clean_negative",
            before_regions=0,
            after_regions=1,
            improved=False,
            damage_regressed=True,
        ),
    ]

    result = decide_benchmark_outcome(rows, checkpoint_order=["g1.5"])

    assert result["outcome"] == "model_gap_requires_new_training_objective"
    assert result["selected_checkpoint_id"] is None
    assert result["delivery_allowed"] is False


def test_source_only_success_is_rejected_as_overfit() -> None:
    rows = [
        _row(row_id="src", row_type="source_hardcase"),
        _row(row_id="var", row_type="synthetic_variant", after_regions=1, improved=False),
        _row(row_id="clean", row_type="clean_negative", before_regions=0, after_regions=0, improved=False),
    ]

    result = decide_benchmark_outcome(rows, checkpoint_order=["g1.5"])

    assert result["outcome"] == "reject_as_overfit"
    assert result["selected_checkpoint_id"] is None
    assert result["delivery_allowed"] is False


def test_leak_row_without_before_leak_prevents_promotion() -> None:
    rows = [
        _row(row_id="src", row_type="source_hardcase"),
        _row(row_id="reg", row_type="region_hardcase", before_regions=0, after_regions=0, improved=False),
        _row(row_id="var", row_type="synthetic_variant"),
        _row(row_id="clean", row_type="clean_negative", before_regions=0, after_regions=0, improved=False),
    ]

    result = decide_benchmark_outcome(rows, checkpoint_order=["g1.5"])

    assert result["outcome"] == "reject_as_overfit"
    assert result["selected_checkpoint_id"] is None
    assert result["invalid_row_count"] == 1


def test_clean_row_uses_uncapped_failure_region_counts(monkeypatch, tmp_path) -> None:
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir()
    (guard_dir / "manifest.json").write_text(
        """{
  "audio": {"vocals": "v.flac", "instrumental": "i.flac"},
  "score": {"config": {"max_regions": 1}}
}
""",
        encoding="utf-8",
    )

    def fake_score(*, output_dir, **_kwargs):
        is_after = output_dir.name == "after"
        total = 2 if is_after else 5
        return {
            "regions": [{"start": 0.0, "end": 0.25, "score": 0.8}],
            "total_fail_regions": total,
            "selected_region_count": 1,
            "regions_capped": True,
            "failure_taxonomy": {
                "total_fail_regions": total,
                "selected_region_count": 1,
                "by_fail_mode": {"standard_score": total},
                "by_probe": {"-12.0": total},
            },
            "best_probe": {"score": 0.7 if is_after else 0.9},
        }

    monkeypatch.setattr("tools.run_mac_mel_refiner_benchmark.score_pitch_shifted_vocal_leak", fake_score)

    result = _score_clean_row(row={"guard_dir": str(guard_dir)}, output_dir=tmp_path / "out")

    assert result["before_regions"] == 5
    assert result["after_regions"] == 2
    assert result["damage_regressed"] is False
    assert result["before_failure_taxonomy"]["total_fail_regions"] == 5
    assert result["after_failure_taxonomy"]["by_probe"] == {"-12.0": 2}


def test_clean_row_derives_best_score_from_region_only_scores(monkeypatch, tmp_path) -> None:
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir()
    (guard_dir / "manifest.json").write_text(
        """{
  "audio": {"vocals": "v.flac", "instrumental": "i.flac"},
  "score": {"config": {"max_regions": 1}}
}
""",
        encoding="utf-8",
    )

    def fake_score(*, output_dir, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "regions": [{"start": 0.0, "end": 0.25, "score": 0.62 if is_after else 0.81}],
            "total_fail_regions": 1,
            "selected_region_count": 1,
            "failure_taxonomy": {"total_fail_regions": 1, "selected_region_count": 1},
        }

    monkeypatch.setattr("tools.run_mac_mel_refiner_benchmark.score_pitch_shifted_vocal_leak", fake_score)

    result = _score_clean_row(row={"guard_dir": str(guard_dir)}, output_dir=tmp_path / "out")

    assert result["before_best_score"] == 0.81
    assert result["after_best_score"] == 0.62


def test_benchmark_row_from_replay_recomputes_strict_improvement_from_scores() -> None:
    from tools.run_mac_mel_refiner_benchmark import _benchmark_row_from_replay

    row = {"row_id": "src", "row_type": "source_hardcase"}
    replay_manifest = {
        "manifest_path": "/tmp/replay/manifest.json",
        "metrics": {
            "before_region_count": 9,
            "after_region_count": 0,
            "before_best_score": 0.0,
            "after_best_score": 0.0,
            "target_issue_improved": True,
        },
        "before_score": {
            "verdict": "fail",
            "regions": [{"score": 0.81}],
            "total_fail_regions": 1,
        },
        "after_score": {
            "verdict": "fail",
            "regions": [{"score": 0.62}],
            "total_fail_regions": 1,
        },
    }

    bench_row = _benchmark_row_from_replay(row=row, checkpoint_id="g1.5", replay_manifest=replay_manifest)

    assert bench_row["before_regions"] == 1
    assert bench_row["after_regions"] == 1
    assert bench_row["before_best_score"] == 0.81
    assert bench_row["after_best_score"] == 0.62
    assert bench_row["target_issue_improved"] is False


def test_benchmark_taxonomy_summary_aggregates_and_ranks_rows() -> None:
    rows = [
        {
            **_row(row_id="src", row_type="source_hardcase", checkpoint_id="g1.5", before_regions=16, after_regions=12),
            "before_failure_taxonomy": {
                "total_fail_regions": 16,
                "by_fail_mode": {"standard_score": 10, "strong_projection_stft": 6},
                "by_probe": {"-12.0": 11, "-5.0": 5},
            },
            "after_failure_taxonomy": {
                "total_fail_regions": 12,
                "by_fail_mode": {"standard_score": 9, "strong_projection_stft": 3},
                "by_probe": {"-12.0": 8, "-5.0": 4},
            },
        },
        {
            **_row(row_id="var", row_type="synthetic_variant", checkpoint_id="g1.5", before_regions=4, after_regions=0),
            "before_failure_taxonomy": {"total_fail_regions": 4, "by_fail_mode": {"standard_score": 4}, "by_probe": {"-5.0": 4}},
            "after_failure_taxonomy": {"total_fail_regions": 0, "by_fail_mode": {}, "by_probe": {}},
        },
        {
            **_row(
                row_id="clean",
                row_type="clean_negative",
                checkpoint_id="g1.5",
                before_regions=0,
                after_regions=2,
                improved=False,
                damage_regressed=True,
            ),
            "before_failure_taxonomy": {"total_fail_regions": 0, "by_fail_mode": {}, "by_probe": {}},
            "after_failure_taxonomy": {"total_fail_regions": 2, "by_fail_mode": {"standard_score": 2}, "by_probe": {"-5.0": 2}},
        },
        {
            **_row(row_id="src", row_type="source_hardcase", checkpoint_id="g2.0", before_regions=16, after_regions=8),
            "before_failure_taxonomy": {"total_fail_regions": 16, "by_fail_mode": {"standard_score": 16}, "by_probe": {"-12.0": 16}},
            "after_failure_taxonomy": {"total_fail_regions": 8, "by_fail_mode": {"standard_score": 8}, "by_probe": {"-12.0": 8}},
        },
    ]

    summary = summarize_benchmark_taxonomy(rows, checkpoint_order=["g1.5", "g2.0"])

    assert summary["delivery_allowed"] is False
    assert summary["total_before_regions"] == 36
    assert summary["total_after_regions"] == 22
    assert summary["after_failure_taxonomy"]["by_probe"] == {"-12.0": 16, "-5.0": 6}
    assert summary["after_failure_taxonomy"]["by_fail_mode"] == {"standard_score": 19, "strong_projection_stft": 3}
    assert summary["checkpoint_rank"][0]["checkpoint_id"] == "g2.0"
    assert summary["checkpoint_rank"][0]["after_regions"] == 8
    assert summary["checkpoint_rank"][1]["checkpoint_id"] == "g1.5"
    assert summary["unresolved_rows"][0]["row_id"] == "src"
    assert summary["unresolved_rows"][0]["checkpoint_id"] == "g1.5"
    assert summary["by_row_type"]["clean_negative"]["damage_regression_rows"] == 1


def test_benchmark_taxonomy_summary_does_not_mark_passing_clean_negative_unresolved() -> None:
    rows = [
        _row(row_id="src", row_type="source_hardcase", before_regions=4, after_regions=0),
        _row(
            row_id="clean",
            row_type="clean_negative",
            before_regions=0,
            after_regions=0,
            improved=False,
            damage_regressed=False,
        ),
    ]

    summary = summarize_benchmark_taxonomy(rows, checkpoint_order=["g1.5"])

    assert summary["unresolved_rows"] == []
