from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.prepare_no_reference_segment_blend_dry_run import (
    build_dry_run_contract,
    main,
    write_outputs,
)


def write_wav(path: Path, *, sample_rate: int = 16_000, seconds: float = 3.0, channels: int = 1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate * seconds)
    audio = np.zeros((frames, channels), dtype=np.float32)
    sf.write(path, audio, sample_rate)
    return str(path)


def plan_row(
    tmp_path: Path,
    *,
    track_id: str = "track-a",
    source_stem: str = "vocal",
    start: float = 0.5,
    duration: float = 1.0,
    alt_exists: bool = True,
    alt_sr: int = 16_000,
    alt_id: str = "alt",
    improvement_delta: float = 0.10,
    same_alt_vocal: bool = False,
    same_alt_minus: bool = False,
) -> dict:
    base_vocal = write_wav(tmp_path / track_id / "base_vocal.wav")
    base_minus = write_wav(tmp_path / track_id / "base_minus.wav")
    alt_vocal = (
        base_vocal
        if same_alt_vocal
        else write_wav(tmp_path / track_id / "alt_vocal.wav", sample_rate=alt_sr)
        if alt_exists
        else str(tmp_path / track_id / "missing_alt_vocal.wav")
    )
    alt_minus = (
        base_minus
        if same_alt_minus
        else write_wav(tmp_path / track_id / "alt_minus.wav", sample_rate=alt_sr)
        if alt_exists
        else str(tmp_path / track_id / "missing_alt_minus.wav")
    )
    return {
        "track_id": track_id,
        "base_candidate_id": "base",
        "alternate_candidate_id": alt_id,
        "source_stem": source_stem,
        "window_start_sec": start,
        "window_duration_sec": duration,
        "improvement_delta": improvement_delta,
        "base_vocal_path": base_vocal,
        "base_instrumental_path": base_minus,
        "alternate_vocal_path": alt_vocal,
        "alternate_instrumental_path": alt_minus,
        "forbidden_actions": ["do_not_train"],
    }


def test_dry_run_creates_render_contract_for_valid_segment(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="vocal")],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
        crossfade_ms=80,
        pre_roll_ms=120,
        post_roll_ms=120,
    )

    row = result["rows"][0]
    assert row["render_status"] == "render_ready"
    assert row["can_render_later"] is True
    assert row["selected_target_stem"] == "vocal"
    assert row["selected_source_path_for_window"] == row["alt_vocal_path"]
    assert row["sample_rate_compatible"] is True
    assert row["channels_compatible"] is True
    assert row["duration_compatible"] is True
    assert row["boundary_safe"] is True
    assert row["overlaps_with_other_planned_window"] is False
    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False


def test_both_stem_segment_uses_structured_source_paths(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="both")],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
    )

    row = result["rows"][0]
    assert row["selected_target_stem"] == "both"
    assert row["selected_source_path_for_window"] == {
        "vocal": row["alt_vocal_path"],
        "minus": row["alt_minus_path"],
    }


def test_incompatible_sample_rate_blocks(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="minus", alt_sr=22_050)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
    )

    row = result["rows"][0]
    assert row["render_status"] == "blocked_incompatible_audio_shape"
    assert row["sample_rate_compatible"] is False
    assert row["can_render_later"] is False


def test_sample_rate_mismatch_can_be_render_ready_with_explicit_resample_flag(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="minus", alt_sr=22_050)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
        allow_sample_rate_resample=True,
    )

    row = result["rows"][0]
    assert row["render_status"] == "render_ready"
    assert row["sample_rate_compatible"] is False
    assert row["sample_rate_resample_allowed"] is True
    assert row["sample_rate_requires_resample"] is True
    assert row["can_render_later"] is True


def test_minus_self_blend_is_blocked_before_render(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="minus", same_alt_minus=True)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
    )

    row = result["rows"][0]
    assert row["active_source_path_distinct"] is False
    assert row["render_status"] == "blocked_noop_self_blend"
    assert row["can_render_later"] is False
    assert result["summary"]["blocked_noop_self_blend"] == 1


def test_both_stem_render_allowed_when_one_active_path_differs(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="both", same_alt_minus=True)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
    )

    row = result["rows"][0]
    assert row["active_source_path_distinct"] is True
    assert row["render_status"] == "render_ready"
    assert row["can_render_later"] is True


def test_overlap_resolution_skips_high_score_noop_candidate(tmp_path: Path) -> None:
    rows = [
        plan_row(
            tmp_path,
            track_id="track-a",
            source_stem="minus",
            start=0.5,
            duration=1.0,
            alt_id="noop-high",
            improvement_delta=0.95,
            same_alt_minus=True,
        ),
        plan_row(
            tmp_path,
            track_id="track-a",
            source_stem="minus",
            start=1.0,
            duration=1.0,
            alt_id="distinct-lower",
            improvement_delta=0.25,
        ),
    ]

    result = build_dry_run_contract(
        rows,
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
        resolve_overlap_conflicts=True,
    )

    statuses = {row["alternate_candidate_id"]: row["render_status"] for row in result["rows"]}
    assert statuses == {
        "distinct-lower": "render_ready",
        "noop-high": "blocked_noop_self_blend",
    }
    assert result["summary"]["render_ready"] == 1
    assert result["summary"]["blocked_noop_self_blend"] == 1


def test_overlapping_windows_are_detected(tmp_path: Path) -> None:
    rows = [
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=0.5, duration=1.0),
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=1.0, duration=1.0),
    ]

    result = build_dry_run_contract(rows, windowed_judge_rows=[], candidate_pool_rows=[], optimizer_rows=[])

    assert {row["render_status"] for row in result["rows"]} == {"blocked_overlap_conflict"}
    assert all(row["overlaps_with_other_planned_window"] is True for row in result["rows"])


def test_overlap_conflicts_can_select_one_render_ready_row_with_explicit_resolve_flag(tmp_path: Path) -> None:
    rows = [
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=0.5, duration=1.0, alt_id="low", improvement_delta=0.10),
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=1.0, duration=1.0, alt_id="high", improvement_delta=0.25),
    ]

    result = build_dry_run_contract(
        rows,
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
        resolve_overlap_conflicts=True,
    )

    statuses = {row["alternate_candidate_id"]: row["render_status"] for row in result["rows"]}
    assert statuses == {"high": "render_ready", "low": "blocked_overlap_conflict"}
    selected = [row for row in result["rows"] if row["render_status"] == "render_ready"]
    assert selected[0]["overlap_conflict_resolution"] == "selected_highest_improvement_nonoverlap"
    assert result["summary"]["render_ready"] == 1


def test_overlap_resolution_skips_unsafe_boundary_candidate(tmp_path: Path) -> None:
    rows = [
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=0.05, duration=1.0, alt_id="unsafe-high", improvement_delta=0.95),
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=0.5, duration=1.0, alt_id="safe-lower", improvement_delta=0.25),
    ]

    result = build_dry_run_contract(
        rows,
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
        resolve_overlap_conflicts=True,
    )

    statuses = {row["alternate_candidate_id"]: row["render_status"] for row in result["rows"]}
    assert statuses["safe-lower"] == "render_ready"
    assert statuses["unsafe-high"] != "render_ready"
    assert {row["alternate_candidate_id"]: row["can_render_later"] for row in result["rows"]}["unsafe-high"] is False
    assert result["summary"]["render_ready"] == 1


def test_vocal_and_minus_windows_on_same_interval_do_not_overlap(tmp_path: Path) -> None:
    rows = [
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=0.5, duration=1.0),
        plan_row(tmp_path, track_id="track-a", source_stem="minus", start=0.5, duration=1.0),
    ]

    result = build_dry_run_contract(rows, windowed_judge_rows=[], candidate_pool_rows=[], optimizer_rows=[])

    assert {row["render_status"] for row in result["rows"]} == {"render_ready"}
    assert all(row["overlaps_with_other_planned_window"] is False for row in result["rows"])


def test_both_stem_window_conflicts_with_single_stem_window(tmp_path: Path) -> None:
    rows = [
        plan_row(tmp_path, track_id="track-a", source_stem="both", start=0.5, duration=1.0),
        plan_row(tmp_path, track_id="track-a", source_stem="minus", start=0.5, duration=1.0),
    ]

    result = build_dry_run_contract(rows, windowed_judge_rows=[], candidate_pool_rows=[], optimizer_rows=[])

    assert {row["render_status"] for row in result["rows"]} == {"blocked_overlap_conflict"}
    assert all(row["overlaps_with_other_planned_window"] is True for row in result["rows"])


def test_missing_alt_path_blocks(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="vocal", alt_exists=False)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
    )

    row = result["rows"][0]
    assert row["render_status"] == "blocked_missing_source_path"
    assert row["can_render_later"] is False


def test_unsafe_boundary_blocks(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="vocal", start=0.05, duration=1.0)],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
    )

    row = result["rows"][0]
    assert row["render_status"] == "blocked_unsafe_boundary"
    assert row["boundary_safe"] is False
    assert row["can_render_later"] is False


def test_status_priority_keeps_missing_path_before_overlap(tmp_path: Path) -> None:
    rows = [
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=0.5, duration=1.0, alt_exists=False),
        plan_row(tmp_path, track_id="track-a", source_stem="vocal", start=0.5, duration=1.0),
    ]

    result = build_dry_run_contract(rows, windowed_judge_rows=[], candidate_pool_rows=[], optimizer_rows=[])

    statuses = {row["alternate_candidate_id"] + row["alt_vocal_path"]: row["render_status"] for row in result["rows"]}
    assert "blocked_missing_source_path" in statuses.values()
    assert "blocked_overlap_conflict" in statuses.values()


def test_publish_and_release_flags_are_always_false(tmp_path: Path) -> None:
    result = build_dry_run_contract(
        [plan_row(tmp_path, source_stem="both")],
        windowed_judge_rows=[],
        candidate_pool_rows=[],
        optimizer_rows=[],
    )

    row = result["rows"][0]
    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False
    assert result["summary"]["can_publish_as_result_count"] == 0
    assert result["summary"]["can_mark_release_ready_count"] == 0
    assert result["summary"]["safety"]["audio_rendered"] is False
    assert result["summary"]["safety"]["audio_exported"] is False


def test_output_is_deterministic_and_no_audio_exported(tmp_path: Path) -> None:
    rows = [
        plan_row(tmp_path, track_id="b", source_stem="minus"),
        plan_row(tmp_path, track_id="a", source_stem="vocal"),
    ]
    first = build_dry_run_contract(list(reversed(rows)), windowed_judge_rows=[], candidate_pool_rows=[], optimizer_rows=[])
    second = build_dry_run_contract(rows, windowed_judge_rows=[], candidate_pool_rows=[], optimizer_rows=[])

    assert first == second
    output_jsonl = tmp_path / "dry.jsonl"
    render_contract_jsonl = tmp_path / "render.jsonl"
    summary_md = tmp_path / "summary.md"
    write_outputs(first, output_jsonl=output_jsonl, render_contract_jsonl=render_contract_jsonl, summary_md=summary_md)
    assert output_jsonl.exists()
    assert render_contract_jsonl.exists()
    assert summary_md.exists()
    assert not list(tmp_path.glob("*.flac"))
    assert not list(tmp_path.glob("*.mp3"))


def test_cli_summary_counts_match_jsonl(tmp_path: Path) -> None:
    plan_jsonl = tmp_path / "plan.jsonl"
    output_jsonl = tmp_path / "dry.jsonl"
    render_contract_jsonl = tmp_path / "render.jsonl"
    summary_md = tmp_path / "summary.md"
    plan_jsonl.write_text(json.dumps(plan_row(tmp_path), sort_keys=True) + "\n", encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    exit_code = main(
        [
            "--segment-blend-plan-jsonl",
            str(plan_jsonl),
            "--windowed-judge-jsonl",
            str(empty),
            "--candidate-pool-jsonl",
            str(empty),
            "--optimizer-result-jsonl",
            str(empty),
            "--output-jsonl",
            str(output_jsonl),
            "--render-contract-jsonl",
            str(render_contract_jsonl),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    contracts = [json.loads(line) for line in render_contract_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert len(contracts) == 1
    assert "- total_planned_windows: 1" in summary_md.read_text(encoding="utf-8")
