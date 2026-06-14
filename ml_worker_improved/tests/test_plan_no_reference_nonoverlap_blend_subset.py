from __future__ import annotations

import json
from pathlib import Path

from tools.plan_no_reference_nonoverlap_blend_subset import (
    build_nonoverlap_subset,
    main,
    write_outputs,
)


def plan_row(
    *,
    track_id: str = "track-a",
    source_stem: str = "vocal",
    start: float = 0.0,
    duration: float = 1.0,
    delta: float = 0.10,
    alt: str = "alt",
    blocker_passed: bool = True,
    reason: str | None = None,
) -> dict:
    if reason is None:
        reason = {
            "vocal": "alternate_improves_vocal_window",
            "minus": "alternate_improves_minus_window",
            "both": "vocal_and_minus_window_scores_improve",
        }[source_stem]
    return {
        "track_id": track_id,
        "base_candidate_id": "base",
        "alternate_candidate_id": alt,
        "source_stem": source_stem,
        "window_start_sec": start,
        "window_duration_sec": duration,
        "window_kind": "middle",
        "improvement_delta": delta,
        "blocker_check_passed": blocker_passed,
        "reason_for_blend": reason,
        "base_vocal_path": f"/tmp/{track_id}/base_vocal.flac",
        "alternate_vocal_path": f"/tmp/{track_id}/{alt}_vocal.flac",
        "base_minus_path": f"/tmp/{track_id}/base_minus.flac",
        "alternate_minus_path": f"/tmp/{track_id}/{alt}_minus.flac",
        "forbidden_actions": ["do_not_train"],
    }


def dry_row(plan: dict, *, status: str = "render_ready", pre_roll_ms: int = 100, post_roll_ms: int = 100) -> dict:
    return {
        **plan,
        "render_status": status,
        "can_render_later": status == "render_ready",
        "pre_roll_ms": pre_roll_ms,
        "post_roll_ms": post_roll_ms,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "forbidden_actions": ["do_not_render_audio"],
    }


def preview_row(*, decision: str = "preview_improved_window", next_step: str = "expand_preview_policy") -> dict:
    return {
        "track_id": "track-a",
        "preview_decision": decision,
        "allowed_next_step": next_step,
        "score_delta": 0.012,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }


def preview_row_for_plan(
    plan: dict,
    *,
    decision: str = "preview_improved_window",
    next_step: str = "expand_preview_policy",
) -> dict:
    return {
        **preview_row(decision=decision, next_step=next_step),
        "track_id": plan["track_id"],
        "base_candidate_id": plan["base_candidate_id"],
        "alternate_candidate_id": plan["alternate_candidate_id"],
        "source_stem": plan["source_stem"],
        "window_start_sec": plan["window_start_sec"],
        "window_duration_sec": plan["window_duration_sec"],
        "base_vocal_path": plan["base_vocal_path"],
        "alt_vocal_path": plan["alternate_vocal_path"],
        "base_minus_path": plan["base_minus_path"],
        "alt_minus_path": plan["alternate_minus_path"],
    }


def test_selects_highest_score_from_overlapping_group() -> None:
    rows = [
        plan_row(start=0.0, delta=0.10, alt="low"),
        plan_row(start=0.5, delta=0.25, alt="high"),
    ]

    result = build_nonoverlap_subset(
        rows,
        dry_run_rows=[dry_row(row) for row in rows],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    statuses = {row["alternate_candidate_id"]: row["conflict_resolution_status"] for row in result["rows"]}
    assert statuses == {
        "high": "selected_nonoverlap",
        "low": "rejected_lower_score_overlap",
    }
    selected = [row for row in result["rows"] if row["conflict_resolution_status"] == "selected_nonoverlap"]
    assert selected[0]["expected_improvement_delta"] == 0.25
    assert selected[0]["selected_rank_in_group"] == 1


def test_keeps_non_overlapping_windows() -> None:
    rows = [
        plan_row(start=0.0, delta=0.10, alt="left"),
        plan_row(start=2.0, delta=0.12, alt="right"),
    ]

    result = build_nonoverlap_subset(
        rows,
        dry_run_rows=[dry_row(row) for row in rows],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    assert [row["conflict_resolution_status"] for row in result["rows"]] == [
        "selected_nonoverlap",
        "selected_nonoverlap",
    ]


def test_blocks_expansion_if_preview_did_not_improve() -> None:
    row = plan_row()

    result = build_nonoverlap_subset(
        [row],
        dry_run_rows=[dry_row(row)],
        preview_window_judge_rows=[preview_row(decision="preview_inconclusive", next_step="keep_preview_for_diagnostics_only")],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    assert result["rows"][0]["conflict_resolution_status"] == "rejected_preview_policy_not_met"
    assert result["rows"][0]["can_render_preview_later"] is False


def test_preview_regression_feedback_blocks_matching_window_and_selects_alternative() -> None:
    regressed_high = plan_row(start=0.0, delta=0.25, alt="same_id")
    safer_low = {
        **plan_row(start=0.0, delta=0.20, alt="same_id"),
        "alternate_vocal_path": "/tmp/track-a/safer_low_vocal.flac",
    }

    result = build_nonoverlap_subset(
        [regressed_high, safer_low],
        dry_run_rows=[dry_row(regressed_high), dry_row(safer_low)],
        preview_window_judge_rows=[
            preview_row_for_plan(regressed_high, decision="preview_regressed_window", next_step="discard_preview_and_fix_planner"),
            preview_row_for_plan(safer_low),
        ],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    statuses = {
        row["expected_improvement_delta"]: row["conflict_resolution_status"]
        for row in result["rows"]
    }
    assert statuses == {
        0.25: "rejected_preview_policy_not_met",
        0.20: "selected_nonoverlap",
    }
    rejected = next(row for row in result["rows"] if row["expected_improvement_delta"] == 0.25)
    assert "preview_feedback=preview_regressed_window" in rejected["selection_reason"]


def test_regressed_preview_feedback_blocks_same_candidate_window_path_variant() -> None:
    regressed = plan_row(start=0.0, delta=0.25, alt="same_id")
    path_variant = {
        **plan_row(start=0.0, delta=0.20, alt="same_id"),
        "alternate_vocal_path": "/tmp/track-a/path_variant_vocal.flac",
    }
    improved_seed = plan_row(track_id="seed-track", start=4.0, delta=0.05, alt="seed")

    result = build_nonoverlap_subset(
        [regressed, path_variant],
        dry_run_rows=[dry_row(regressed), dry_row(path_variant)],
        preview_window_judge_rows=[
            preview_row_for_plan(improved_seed),
            preview_row_for_plan(regressed, decision="preview_regressed_window", next_step="discard_preview_and_fix_planner"),
        ],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    statuses = {
        row["expected_improvement_delta"]: row["conflict_resolution_status"]
        for row in result["rows"]
    }
    assert statuses == {
        0.25: "rejected_preview_policy_not_met",
        0.20: "rejected_preview_policy_not_met",
    }
    variant = next(row for row in result["rows"] if row["expected_improvement_delta"] == 0.20)
    assert "preview_feedback_identity=preview_regressed_window" in variant["selection_reason"]


def test_pair_rejection_feedback_blocks_same_candidate_window_identity() -> None:
    pair_rejected = plan_row(start=0.0, delta=0.25, alt="pair_bad")
    fallback = plan_row(start=2.0, delta=0.20, alt="fallback")
    pair_feedback = {
        "track_id": pair_rejected["track_id"],
        "base_candidate_id": pair_rejected["base_candidate_id"],
        "alternate_candidate_id": pair_rejected["alternate_candidate_id"],
        "source_stem": pair_rejected["source_stem"],
        "window_start_sec": pair_rejected["window_start_sec"],
        "window_duration_sec": pair_rejected["window_duration_sec"],
        "score_mode": "both",
        "decision": "blocked_hard_case",
        "pair_decision": "blocked_pair_incompatible",
        "pair_blockers": ["phase_collapse"],
    }

    result = build_nonoverlap_subset(
        [pair_rejected, fallback],
        dry_run_rows=[dry_row(pair_rejected), dry_row(fallback)],
        preview_window_judge_rows=[preview_row_for_plan(pair_rejected), preview_row_for_plan(fallback)],
        windowed_judge_rows=[pair_feedback],
        candidate_pool_rows=[],
    )

    statuses = {
        row["alternate_candidate_id"]: row["conflict_resolution_status"]
        for row in result["rows"]
    }
    assert statuses == {
        "pair_bad": "rejected_preview_policy_not_met",
        "fallback": "selected_nonoverlap",
    }
    rejected = next(row for row in result["rows"] if row["alternate_candidate_id"] == "pair_bad")
    assert "pair_feedback=blocked_pair_incompatible" in rejected["selection_reason"]
    assert result["summary"]["rejected_pair_feedback"] == 1


def test_score_matrix_minus_guard_blocks_leaky_alternate_candidate() -> None:
    leaky = plan_row(start=0.0, delta=0.25, alt="leaky_matrix")
    fallback = plan_row(start=2.0, delta=0.20, alt="fallback")
    matrix_row = {
        "track_id": leaky["track_id"],
        "candidate_id": leaky["alternate_candidate_id"],
        "vocal_leakage_in_minus_score": 1.0,
        "mixture_consistency_score": 0.90,
        "phase_consistency_score": 0.90,
        "instrumental_phase_consistency_score": 0.90,
        "reasons": ["vocal_leakage_in_minus"],
    }

    result = build_nonoverlap_subset(
        [leaky, fallback],
        dry_run_rows=[dry_row(leaky), dry_row(fallback)],
        preview_window_judge_rows=[preview_row_for_plan(leaky), preview_row_for_plan(fallback)],
        windowed_judge_rows=[matrix_row],
        candidate_pool_rows=[],
    )

    statuses = {
        row["alternate_candidate_id"]: row["conflict_resolution_status"]
        for row in result["rows"]
    }
    assert statuses == {
        "leaky_matrix": "rejected_preview_policy_not_met",
        "fallback": "selected_nonoverlap",
    }
    rejected = next(row for row in result["rows"] if row["alternate_candidate_id"] == "leaky_matrix")
    assert "matrix_minus_guard=severe_vocal_bleed_in_minus" in rejected["selection_reason"]
    assert result["summary"]["rejected_matrix_guard"] == 1


def test_dry_run_contract_matching_is_path_sensitive_for_duplicate_candidate_ids() -> None:
    blocked_high = plan_row(start=0.0, delta=0.25, alt="same_id")
    renderable_low = {
        **plan_row(start=0.0, delta=0.20, alt="same_id"),
        "alternate_vocal_path": "/tmp/track-a/renderable_low_vocal.flac",
    }

    result = build_nonoverlap_subset(
        [blocked_high, renderable_low],
        dry_run_rows=[dry_row(blocked_high, status="blocked_noop_self_blend"), dry_row(renderable_low)],
        preview_window_judge_rows=[preview_row_for_plan(renderable_low)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    statuses = {
        row["expected_improvement_delta"]: row["conflict_resolution_status"]
        for row in result["rows"]
    }
    assert statuses == {
        0.25: "rejected_safety_blocker",
        0.20: "selected_nonoverlap",
    }
    blocked = next(row for row in result["rows"] if row["expected_improvement_delta"] == 0.25)
    assert "render_status=blocked_noop_self_blend" in blocked["selection_reason"]


def test_rejects_safety_blockers() -> None:
    row = plan_row(blocker_passed=False)

    result = build_nonoverlap_subset(
        [row],
        dry_run_rows=[dry_row(row, status="blocked_missing_source_path")],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    assert result["rows"][0]["conflict_resolution_status"] == "rejected_safety_blocker"
    assert result["summary"]["rejected_safety_blocker"] == 1


def test_missing_dry_run_contract_is_safety_blocker() -> None:
    row = plan_row()

    result = build_nonoverlap_subset(
        [row],
        dry_run_rows=[],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    assert result["rows"][0]["conflict_resolution_status"] == "rejected_safety_blocker"
    assert "missing_dry_run_contract" in result["rows"][0]["selection_reason"]


def test_rejects_source_stem_that_does_not_match_improved_metric() -> None:
    row = plan_row(source_stem="both", reason="alternate_improves_vocal_window")

    result = build_nonoverlap_subset(
        [row],
        dry_run_rows=[dry_row(row)],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    assert result["rows"][0]["conflict_resolution_status"] == "rejected_safety_blocker"
    assert "source_stem_does_not_match_improved_metric" in result["rows"][0]["selection_reason"]


def test_rejects_minus_window_when_target_minus_path_is_identical() -> None:
    row = plan_row(source_stem="minus")
    row["alternate_minus_path"] = row["base_minus_path"]

    result = build_nonoverlap_subset(
        [row],
        dry_run_rows=[dry_row(row)],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    assert result["rows"][0]["conflict_resolution_status"] == "rejected_safety_blocker"
    assert "target_minus_path_identity_noop_risk" in result["rows"][0]["selection_reason"]
    assert result["rows"][0]["can_render_preview_later"] is False


def test_rejects_both_window_when_any_target_stem_path_is_identical() -> None:
    row = plan_row(source_stem="both")
    row["alternate_minus_path"] = row["base_minus_path"]

    result = build_nonoverlap_subset(
        [row],
        dry_run_rows=[dry_row(row)],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    assert result["rows"][0]["conflict_resolution_status"] == "rejected_safety_blocker"
    assert "target_minus_path_identity_noop_risk" in result["rows"][0]["selection_reason"]
    assert result["rows"][0]["can_render_preview_later"] is False


def test_publish_release_false_and_no_audio_rendered(tmp_path: Path) -> None:
    row = plan_row()
    result = build_nonoverlap_subset(
        [row],
        dry_run_rows=[dry_row(row)],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    selected = result["rows"][0]
    assert selected["can_publish_as_result"] is False
    assert selected["can_mark_release_ready"] is False
    assert selected["can_render_preview_later"] is True
    assert result["summary"]["safety"]["audio_rendered"] is False
    assert result["summary"]["safety"]["audio_exported"] is False
    assert not list(tmp_path.rglob("*.flac"))
    assert not list(tmp_path.rglob("*.wav"))


def test_selected_rows_are_render_contract_compatible() -> None:
    selected_row = plan_row(start=0.0, delta=0.25, alt="selected")
    rejected_row = plan_row(start=0.5, delta=0.10, alt="rejected")

    result = build_nonoverlap_subset(
        [selected_row, rejected_row],
        dry_run_rows=[dry_row(selected_row), dry_row(rejected_row)],
        preview_window_judge_rows=[preview_row_for_plan(selected_row)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )

    selected = next(row for row in result["rows"] if row["alternate_candidate_id"] == "selected")
    rejected = next(row for row in result["rows"] if row["alternate_candidate_id"] == "rejected")
    assert selected["conflict_resolution_status"] == "selected_nonoverlap"
    assert selected["render_status"] == "render_ready"
    assert selected["base_vocal_path"] == selected_row["base_vocal_path"]
    assert selected["base_minus_path"] == selected_row["base_minus_path"]
    assert selected["alt_vocal_path"] == selected_row["alternate_vocal_path"]
    assert selected["alt_minus_path"] == selected_row["alternate_minus_path"]
    assert selected["crossfade_ms"] == 80
    assert selected["can_render_later"] is True
    assert rejected["conflict_resolution_status"] == "rejected_lower_score_overlap"
    assert rejected["render_status"] != "render_ready"
    assert rejected["can_render_later"] is False


def test_output_is_deterministic_and_summary_counts_match_jsonl(tmp_path: Path) -> None:
    rows = [
        plan_row(start=0.5, delta=0.25, alt="high"),
        plan_row(start=0.0, delta=0.10, alt="low"),
        plan_row(start=2.0, delta=0.12, alt="right", source_stem="minus"),
    ]

    first = build_nonoverlap_subset(
        rows,
        dry_run_rows=[dry_row(row) for row in rows],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )
    second = build_nonoverlap_subset(
        list(reversed(rows)),
        dry_run_rows=[dry_row(row) for row in reversed(rows)],
        preview_window_judge_rows=[preview_row()],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
    )
    assert first == second

    output_jsonl = tmp_path / "subset.jsonl"
    summary_md = tmp_path / "summary.md"
    write_outputs(first, output_jsonl=output_jsonl, summary_md=summary_md)
    output_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(output_rows) == first["summary"]["total_planned_windows"]
    assert f"- selected_nonoverlap: {first['summary']['selected_nonoverlap']}" in summary_md.read_text(encoding="utf-8")


def test_cli_writes_outputs(tmp_path: Path) -> None:
    row = plan_row()
    plan_jsonl = tmp_path / "plan.jsonl"
    dry_jsonl = tmp_path / "dry.jsonl"
    preview_jsonl = tmp_path / "preview.jsonl"
    windowed_jsonl = tmp_path / "windowed.jsonl"
    pool_jsonl = tmp_path / "pool.jsonl"
    out_jsonl = tmp_path / "out.jsonl"
    summary_md = tmp_path / "summary.md"
    plan_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    dry_jsonl.write_text(json.dumps(dry_row(row), sort_keys=True) + "\n", encoding="utf-8")
    preview_jsonl.write_text(json.dumps(preview_row(), sort_keys=True) + "\n", encoding="utf-8")
    windowed_jsonl.write_text("", encoding="utf-8")
    pool_jsonl.write_text("", encoding="utf-8")

    exit_code = main(
        [
            "--segment-blend-plan-jsonl",
            str(plan_jsonl),
            "--dry-run-jsonl",
            str(dry_jsonl),
            "--preview-window-judge-jsonl",
            str(preview_jsonl),
            "--windowed-judge-jsonl",
            str(windowed_jsonl),
            "--candidate-pool-jsonl",
            str(pool_jsonl),
            "--output-jsonl",
            str(out_jsonl),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    assert out_jsonl.exists()
    assert summary_md.exists()


def test_cli_accepts_multiple_preview_feedback_files(tmp_path: Path) -> None:
    regressed_high = plan_row(start=0.0, delta=0.25, alt="same_id")
    safer_low = {
        **plan_row(start=0.0, delta=0.20, alt="same_id"),
        "alternate_vocal_path": "/tmp/track-a/safer_low_vocal.flac",
    }
    plan_jsonl = tmp_path / "plan.jsonl"
    dry_jsonl = tmp_path / "dry.jsonl"
    preview_regressed_jsonl = tmp_path / "preview-regressed.jsonl"
    preview_improved_jsonl = tmp_path / "preview-improved.jsonl"
    windowed_jsonl = tmp_path / "windowed.jsonl"
    pool_jsonl = tmp_path / "pool.jsonl"
    out_jsonl = tmp_path / "out.jsonl"
    summary_md = tmp_path / "summary.md"
    plan_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [regressed_high, safer_low]) + "\n",
        encoding="utf-8",
    )
    dry_jsonl.write_text(
        "\n".join(json.dumps(dry_row(row), sort_keys=True) for row in [regressed_high, safer_low]) + "\n",
        encoding="utf-8",
    )
    preview_regressed_jsonl.write_text(
        json.dumps(preview_row_for_plan(regressed_high, decision="preview_regressed_window", next_step="discard_preview_and_fix_planner"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preview_improved_jsonl.write_text(json.dumps(preview_row_for_plan(safer_low), sort_keys=True) + "\n", encoding="utf-8")
    windowed_jsonl.write_text("", encoding="utf-8")
    pool_jsonl.write_text("", encoding="utf-8")

    exit_code = main(
        [
            "--segment-blend-plan-jsonl",
            str(plan_jsonl),
            "--dry-run-jsonl",
            str(dry_jsonl),
            "--preview-window-judge-jsonl",
            str(preview_regressed_jsonl),
            "--preview-window-judge-jsonl",
            str(preview_improved_jsonl),
            "--windowed-judge-jsonl",
            str(windowed_jsonl),
            "--candidate-pool-jsonl",
            str(pool_jsonl),
            "--output-jsonl",
            str(out_jsonl),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    output_rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    statuses = {
        row["expected_improvement_delta"]: row["conflict_resolution_status"]
        for row in output_rows
    }
    assert statuses == {
        0.25: "rejected_preview_policy_not_met",
        0.20: "selected_nonoverlap",
    }


def test_cli_accepts_multiple_windowed_judge_files_for_matrix_and_pair_feedback(tmp_path: Path) -> None:
    leaky = plan_row(start=0.0, delta=0.25, alt="leaky_matrix")
    pair_rejected = plan_row(start=2.0, delta=0.20, alt="pair_bad")
    fallback = plan_row(start=4.0, delta=0.10, alt="fallback")
    plan_jsonl = tmp_path / "plan.jsonl"
    dry_jsonl = tmp_path / "dry.jsonl"
    preview_jsonl = tmp_path / "preview.jsonl"
    matrix_jsonl = tmp_path / "matrix.jsonl"
    pair_feedback_jsonl = tmp_path / "pair_feedback.jsonl"
    pool_jsonl = tmp_path / "pool.jsonl"
    out_jsonl = tmp_path / "out.jsonl"
    summary_md = tmp_path / "summary.md"
    rows = [leaky, pair_rejected, fallback]
    plan_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    dry_jsonl.write_text("\n".join(json.dumps(dry_row(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    preview_jsonl.write_text("\n".join(json.dumps(preview_row_for_plan(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    matrix_jsonl.write_text(
        json.dumps(
            {
                "track_id": leaky["track_id"],
                "candidate_id": leaky["alternate_candidate_id"],
                "vocal_leakage_in_minus_score": 1.0,
                "mixture_consistency_score": 0.90,
                "phase_consistency_score": 0.90,
                "instrumental_phase_consistency_score": 0.90,
                "reasons": ["vocal_leakage_in_minus"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pair_feedback_jsonl.write_text(
        json.dumps(
            {
                "track_id": pair_rejected["track_id"],
                "base_candidate_id": pair_rejected["base_candidate_id"],
                "alternate_candidate_id": pair_rejected["alternate_candidate_id"],
                "source_stem": pair_rejected["source_stem"],
                "window_start_sec": pair_rejected["window_start_sec"],
                "window_duration_sec": pair_rejected["window_duration_sec"],
                "score_mode": "both",
                "pair_decision": "blocked_pair_incompatible",
                "pair_blockers": ["phase_collapse"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pool_jsonl.write_text("", encoding="utf-8")

    exit_code = main(
        [
            "--segment-blend-plan-jsonl",
            str(plan_jsonl),
            "--dry-run-jsonl",
            str(dry_jsonl),
            "--preview-window-judge-jsonl",
            str(preview_jsonl),
            "--windowed-judge-jsonl",
            str(matrix_jsonl),
            "--windowed-judge-jsonl",
            str(pair_feedback_jsonl),
            "--candidate-pool-jsonl",
            str(pool_jsonl),
            "--output-jsonl",
            str(out_jsonl),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    output_rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    statuses = {row["alternate_candidate_id"]: row["conflict_resolution_status"] for row in output_rows}
    assert statuses == {
        "leaky_matrix": "rejected_preview_policy_not_met",
        "pair_bad": "rejected_preview_policy_not_met",
        "fallback": "selected_nonoverlap",
    }
