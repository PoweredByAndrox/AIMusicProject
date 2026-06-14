from __future__ import annotations

import json
from pathlib import Path

from tools.plan_no_reference_segment_blends import (
    build_segment_blend_plan,
    main,
    write_outputs,
)


BASE_FORBIDDEN = ["do_not_train"]


def metric_row(
    candidate_id: str,
    *,
    track_id: str = "track-a",
    vocal_path: str | None = None,
    instrumental_path: str | None = None,
    window_fitness: float = 0.50,
    vocal_score: float = 0.50,
    minus_score: float = 0.50,
    reasons: list[str] | None = None,
    window_metrics: list[dict] | None = None,
) -> dict:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": f"/tmp/{track_id}/mix.wav",
        "vocal_path": vocal_path or f"/tmp/{track_id}/{candidate_id}_vocal.wav",
        "instrumental_path": instrumental_path or f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "judge_mode": "no_reference_windowed",
        "forbidden_actions": BASE_FORBIDDEN,
        "window_metrics": window_metrics
        or [
            window_metric(
                "middle",
                window_fitness=window_fitness,
                vocal_score=vocal_score,
                minus_score=minus_score,
                reasons=reasons,
            )
        ],
    }


def window_metric(
    kind: str,
    *,
    start_sec: float = 12.0,
    duration_sec: float = 12.0,
    window_fitness: float = 0.50,
    vocal_score: float = 0.50,
    minus_score: float = 0.50,
    pair_compatibility_score: float = 0.90,
    pair_decision: str = "pair_compatible",
    reasons: list[str] | None = None,
) -> dict:
    return {
        "kind": kind,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "vocal_score": vocal_score,
        "minus_score": minus_score,
        "artifact_score": 0.90,
        "consistency_score": 0.90,
        "defect_reduction_score": 0.70,
        "window_fitness": window_fitness,
        "pair_compatibility_score": pair_compatibility_score,
        "pair_decision": pair_decision,
        "pair_blockers": [] if pair_decision == "pair_compatible" else ["severe_mixture_reconstruction_drift"],
        "reasons": reasons or ["no_reference_proxy_passed"],
        "vocal_leakage_in_minus_score": 1.0 - minus_score,
        "instrumental_leakage_in_vocal_score": 1.0 - vocal_score,
        "robotic_artifact_score": 0.05,
        "transient_damage_score": 0.05,
        "phase_consistency_score": 0.90,
    }


def optimizer_row(
    *,
    track_id: str = "track-a",
    input_candidate_id: str = "base",
    selected_candidate_id: str = "base",
    rejected_reason: str = "minus_regression_severe",
) -> dict:
    return {
        "track_id": track_id,
        "input_candidate_id": input_candidate_id,
        "selected_candidate_id": selected_candidate_id,
        "final_decision": "original_candidate_kept",
        "actions_rejected": [{"candidate_id": "alt", "reason": rejected_reason}],
    }


def candidate_pool_rows() -> list[dict]:
    return [
        {"track_id": "track-a", "candidate_id": "base", "is_complete_pair": True},
        {"track_id": "track-a", "candidate_id": "alt", "is_complete_pair": True},
    ]


def test_proposes_both_stem_segment_when_alternate_window_improves_without_blocker() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("base", window_fitness=0.50, vocal_score=0.50, minus_score=0.50),
            metric_row("alt", window_fitness=0.72, vocal_score=0.72, minus_score=0.75),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["segment_blend_action"] == "use_alt_both_window"
    assert row["source_stem"] == "both"
    assert row["can_execute_blend_now"] is False
    assert row["requires_crossfade"] is True
    assert row["improvement_delta"] == 0.22


def test_rejects_segment_when_alternate_has_severe_blocker() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("base", window_fitness=0.50, vocal_score=0.50, minus_score=0.50),
            metric_row("alt", window_fitness=0.82, vocal_score=0.82, minus_score=0.82, reasons=["vocal_leakage_in_minus"]),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert result["rows"] == []
    assert result["summary"]["blocked_segment_candidates"] == 1


def test_proposes_vocal_only_when_minus_regresses() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("base", window_fitness=0.50, vocal_score=0.45, minus_score=0.80),
            metric_row("alt", window_fitness=0.58, vocal_score=0.75, minus_score=0.68),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert result["rows"][0]["segment_blend_action"] == "use_alt_vocal_window"
    assert result["rows"][0]["source_stem"] == "vocal"


def test_proposes_minus_only_when_vocal_regresses() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("base", window_fitness=0.50, vocal_score=0.80, minus_score=0.45),
            metric_row("alt", window_fitness=0.58, vocal_score=0.68, minus_score=0.75),
        ],
        [optimizer_row(rejected_reason="vocal_regression_severe")],
        min_segment_delta=0.05,
    )

    assert result["rows"][0]["segment_blend_action"] == "use_alt_minus_window"
    assert result["rows"][0]["source_stem"] == "minus"


def test_vocal_segment_uses_vocal_matrix_not_shared_window_fitness() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("base", window_fitness=0.90, vocal_score=0.45, minus_score=0.95),
            metric_row("alt", window_fitness=0.60, vocal_score=0.75, minus_score=0.68),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["segment_blend_action"] == "use_alt_vocal_window"
    assert result["rows"][0]["source_stem"] == "vocal"
    assert result["rows"][0]["improvement_delta"] == 0.30


def test_minus_segment_uses_minus_matrix_not_shared_window_fitness() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("base", window_fitness=0.90, vocal_score=0.95, minus_score=0.45),
            metric_row("alt", window_fitness=0.60, vocal_score=0.68, minus_score=0.75),
        ],
        [optimizer_row(rejected_reason="vocal_regression_severe")],
        min_segment_delta=0.05,
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["segment_blend_action"] == "use_alt_minus_window"
    assert result["rows"][0]["source_stem"] == "minus"
    assert result["rows"][0]["improvement_delta"] == 0.30


def test_both_segment_requires_pair_compatibility() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("base", window_fitness=0.50, vocal_score=0.50, minus_score=0.50),
            metric_row(
                "alt",
                window_metrics=[
                    window_metric(
                        "middle",
                        window_fitness=0.82,
                        vocal_score=0.82,
                        minus_score=0.82,
                        pair_compatibility_score=0.40,
                        pair_decision="blocked_pair_incompatible",
                    )
                ],
            ),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert result["rows"] == []


def test_outputs_are_deterministic_and_do_not_export_audio(tmp_path: Path) -> None:
    first = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row("alt", window_fitness=0.72, vocal_score=0.72, minus_score=0.75),
            metric_row("base", window_fitness=0.50, vocal_score=0.50, minus_score=0.50),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )
    second = build_segment_blend_plan(
        list(reversed(candidate_pool_rows())),
        [
            metric_row("base", window_fitness=0.50, vocal_score=0.50, minus_score=0.50),
            metric_row("alt", window_fitness=0.72, vocal_score=0.72, minus_score=0.75),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert first == second
    output_jsonl = tmp_path / "plan.jsonl"
    summary_md = tmp_path / "summary.md"
    write_outputs(first, output_jsonl=output_jsonl, summary_md=summary_md)
    assert output_jsonl.exists()
    assert summary_md.exists()
    assert not list(tmp_path.rglob("*.wav"))
    assert not list(tmp_path.rglob("*.flac"))
    assert first["summary"]["safety"]["segment_blend_executed"] is False
    assert first["summary"]["safety"]["audio_exported"] is False


def test_cli_writes_summary_counts_matching_jsonl(tmp_path: Path) -> None:
    pool_jsonl = tmp_path / "pool.jsonl"
    judge_jsonl = tmp_path / "judge.jsonl"
    optimizer_jsonl = tmp_path / "optimizer.jsonl"
    output_jsonl = tmp_path / "out.jsonl"
    summary_md = tmp_path / "summary.md"
    pool_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in candidate_pool_rows()) + "\n", encoding="utf-8")
    judge_rows = [
        metric_row("base", window_fitness=0.50, vocal_score=0.50, minus_score=0.50),
        metric_row("alt", window_fitness=0.72, vocal_score=0.72, minus_score=0.75),
    ]
    judge_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in judge_rows) + "\n", encoding="utf-8")
    optimizer_jsonl.write_text(json.dumps(optimizer_row(), sort_keys=True) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "--candidate-pool-jsonl",
            str(pool_jsonl),
            "--windowed-judge-jsonl",
            str(judge_jsonl),
            "--optimizer-result-jsonl",
            str(optimizer_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-md",
            str(summary_md),
            "--min-segment-delta",
            "0.05",
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert "- total_segment_blend_windows: 1" in summary_md.read_text(encoding="utf-8")


def test_duplicate_candidate_ids_are_kept_path_distinct() -> None:
    pool_rows = [
        {"track_id": "track-a", "candidate_id": "base", "vocal_path": "/tmp/base_vocal.wav", "instrumental_path": "/tmp/base_minus.wav", "is_complete_pair": True},
        {"track_id": "track-a", "candidate_id": "alt", "vocal_path": "/tmp/alt_1_vocal.wav", "instrumental_path": "/tmp/alt_1_minus.wav", "is_complete_pair": True},
        {"track_id": "track-a", "candidate_id": "alt", "vocal_path": "/tmp/alt_2_vocal.wav", "instrumental_path": "/tmp/alt_2_minus.wav", "is_complete_pair": True},
    ]
    result = build_segment_blend_plan(
        pool_rows,
        [
            metric_row("base", vocal_path="/tmp/base_vocal.wav", instrumental_path="/tmp/base_minus.wav", window_fitness=0.50, vocal_score=0.50, minus_score=0.50),
            metric_row("alt", vocal_path="/tmp/alt_1_vocal.wav", instrumental_path="/tmp/alt_1_minus.wav", window_fitness=0.70, vocal_score=0.70, minus_score=0.75),
            metric_row("alt", vocal_path="/tmp/alt_2_vocal.wav", instrumental_path="/tmp/alt_2_minus.wav", window_fitness=0.74, vocal_score=0.74, minus_score=0.76),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert len(result["rows"]) == 2
    assert {row["alternate_vocal_path"] for row in result["rows"]} == {"/tmp/alt_1_vocal.wav", "/tmp/alt_2_vocal.wav"}


def test_duplicate_time_windows_are_matched_by_kind() -> None:
    result = build_segment_blend_plan(
        candidate_pool_rows(),
        [
            metric_row(
                "base",
                window_metrics=[
                    window_metric("low_energy", start_sec=0.0, window_fitness=0.60, vocal_score=0.60, minus_score=0.60),
                    window_metric("start", start_sec=0.0, window_fitness=0.10, vocal_score=0.10, minus_score=0.10),
                ],
            ),
            metric_row(
                "alt",
                window_metrics=[
                    window_metric("low_energy", start_sec=0.0, window_fitness=0.72, vocal_score=0.72, minus_score=0.75),
                    window_metric("start", start_sec=0.0, window_fitness=0.12, vocal_score=0.12, minus_score=0.12),
                ],
            ),
        ],
        [optimizer_row()],
        min_segment_delta=0.05,
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["window_kind"] == "low_energy"
    assert result["rows"][0]["base_window_fitness"] == 0.60
