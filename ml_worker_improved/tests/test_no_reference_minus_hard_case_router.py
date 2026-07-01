from __future__ import annotations

import json
from pathlib import Path

from tools.no_reference_candidate_job_registry import CandidateJobConfig
from tools.no_reference_minus_hard_case_router import run_minus_hard_case_router


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def unit_config(
    config_id: str,
    *,
    family: str = "minus_focused_existing_model",
    routes: tuple[str, ...] = ("generate_minus_focused_variants",),
    entrypoint_type: str = "python_script",
) -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id=config_id,
        candidate_family=family,
        target_mode="minus",
        source_model_id="unit-model",
        source_tool="unit-tool",
        entrypoint_type=entrypoint_type,
        command_template=("unit",),
        expected_outputs={"vocal_path": "", "minus_path": ""},
        gpu_required=False,
        timeout_sec=60,
        max_retries=0,
        profile_allowed=("balanced",),
        safety_notes=("unit",),
        supported_parameters=("target_stem",),
        shared_execution_group_id=config_id,
        produces_vocal=False,
        produces_minus=True,
        materialized_candidate_modes=("minus",),
        reuse_outputs_across_modes=False,
        target_stem="minus",
        hard_case_routes=routes,
    )


def make_hard_case_run(tmp_path: Path, *, blockers: list[str]) -> Path:
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "run_manifest.json",
        {
            "output_status": "needs_more_candidates",
            "failure_reason": "missing_safe_mode_candidate",
            "safety_blockers": ["missing_safe_mode_candidate"],
            "release_ready": False,
        },
    )
    write_jsonl(run_dir / "selected_candidates.jsonl", [])
    write_jsonl(run_dir / "pair_compatibility.jsonl", [])
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-002",
                "candidate_id": "pair-a",
                "candidate_config_id": "local_registered__htdemucs_ft__both__shared",
                "candidate_family": "balanced_pair_default",
                "can_use_for_minus": True,
                "minus_path": "/tmp/minus-a.wav",
                "instrumental_path": "/tmp/minus-a.wav",
            }
        ],
    )
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track-002",
                "candidate_id": "pair-a",
                "candidate_config_id": "local_registered__htdemucs_ft__both__shared",
                "candidate_family": "balanced_pair_default",
                "minus_decision": "blocked_hard_case",
                "minus_blockers": blockers,
            }
        ],
    )
    return run_dir


def test_router_detects_severe_vocal_bleed_and_low_diversity(tmp_path: Path) -> None:
    run_dir = make_hard_case_run(tmp_path, blockers=["severe_vocal_bleed_in_minus"])

    result = run_minus_hard_case_router(
        run_dir=run_dir,
        output_jsonl=run_dir / "minus_hard_case_router.jsonl",
        summary_md=run_dir / "minus_hard_case_router_summary.md",
        registry_configs=[unit_config("minus-supported")],
    )

    rows = [json.loads(line) for line in (run_dir / "minus_hard_case_router.jsonl").read_text(encoding="utf-8").splitlines()]
    blocker_types = {row["blocker_type"] for row in rows}
    assert "severe_vocal_bleed_in_minus" in blocker_types
    assert "missing_safe_mode_candidate" in blocker_types
    assert "low_minus_candidate_diversity" in blocker_types
    assert any(row["recommended_route"] == "generate_minus_focused_variants" for row in rows)
    assert result["recommended_route"] == "generate_minus_focused_variants"
    assert result["can_run_without_new_models"] is True
    assert result["release_ready"] is False


def test_router_recommends_phase_safe_route_for_phase_collapse(tmp_path: Path) -> None:
    run_dir = make_hard_case_run(tmp_path, blockers=["severe_phase_collapse"])

    result = run_minus_hard_case_router(
        run_dir=run_dir,
        output_jsonl=run_dir / "minus_hard_case_router.jsonl",
        summary_md=run_dir / "minus_hard_case_router_summary.md",
        registry_configs=[
            unit_config(
                "phase-supported",
                family="phase_safe_existing_model",
                routes=("generate_phase_safe_variants",),
            )
        ],
    )

    assert result["recommended_route"] == "generate_phase_safe_variants"
    assert result["available_route_configs"] == ["phase-supported"]


def test_router_prioritizes_separator_objective_when_pitch_regions_stall(tmp_path: Path) -> None:
    run_dir = make_hard_case_run(
        tmp_path,
        blockers=[
            "severe_vocal_bleed_in_minus",
            "scalar_score_only_pitch_guard_progress",
        ],
    )

    result = run_minus_hard_case_router(
        run_dir=run_dir,
        output_jsonl=run_dir / "minus_hard_case_router.jsonl",
        summary_md=run_dir / "minus_hard_case_router_summary.md",
        registry_configs=[unit_config("minus-focused")],
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "minus_hard_case_router.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    scalar_row = next(row for row in rows if row["blocker_type"] == "scalar_score_only_pitch_guard_progress")
    assert scalar_row["recommended_route"] == "train_separator_attribution_objective"
    assert result["recommended_route"] == "train_separator_attribution_objective"
    assert result["can_run_without_new_models"] is False
    assert result["release_ready"] is False


def test_router_derives_scalar_only_pitch_guard_progress_from_region_metrics(tmp_path: Path) -> None:
    run_dir = make_hard_case_run(tmp_path, blockers=[])
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track-002",
                "candidate_id": "pair-a",
                "candidate_config_id": "local_registered__htdemucs_ft__both__shared",
                "candidate_family": "balanced_pair_default",
                "minus_decision": "blocked_hard_case",
                "pitch_guard_before_region_count": 16,
                "pitch_guard_after_region_count": 16,
                "pitch_guard_before_best_score": 0.702988,
                "pitch_guard_after_best_score": 0.63693,
            }
        ],
    )

    result = run_minus_hard_case_router(
        run_dir=run_dir,
        output_jsonl=run_dir / "minus_hard_case_router.jsonl",
        summary_md=run_dir / "minus_hard_case_router_summary.md",
        registry_configs=[unit_config("minus-focused")],
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "minus_hard_case_router.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["blocker_type"] == "scalar_score_only_pitch_guard_progress" for row in rows)
    assert result["recommended_route"] == "train_separator_attribution_objective"
    assert result["release_ready"] is False


def test_router_derives_full_region_stall_without_score_delta(tmp_path: Path) -> None:
    run_dir = make_hard_case_run(tmp_path, blockers=[])
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track-002",
                "candidate_id": "pair-a",
                "candidate_config_id": "local_registered__htdemucs_ft__both__shared",
                "candidate_family": "balanced_pair_default",
                "minus_decision": "blocked_hard_case",
                "before_region_count": 16,
                "after_region_count": 16,
            }
        ],
    )

    result = run_minus_hard_case_router(
        run_dir=run_dir,
        output_jsonl=run_dir / "minus_hard_case_router.jsonl",
        summary_md=run_dir / "minus_hard_case_router_summary.md",
        registry_configs=[unit_config("minus-focused")],
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "minus_hard_case_router.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["blocker_type"] == "pitch_guard_full_region_stalled" for row in rows)
    assert result["recommended_route"] == "train_separator_attribution_objective"
    assert result["release_ready"] is False


def test_router_derives_pitch_stall_from_candidate_pool_metrics(tmp_path: Path) -> None:
    run_dir = make_hard_case_run(tmp_path, blockers=[])
    write_jsonl(run_dir / "minus_windowed_judge.jsonl", [])
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-002",
                "candidate_id": "pair-a",
                "candidate_config_id": "local_registered__htdemucs_ft__both__shared",
                "candidate_family": "balanced_pair_default",
                "can_use_for_minus": True,
                "minus_path": "/tmp/minus-a.wav",
                "instrumental_path": "/tmp/minus-a.wav",
                "pitch_guard_before_region_count": 16,
                "pitch_guard_after_region_count": 16,
                "pitch_guard_before_best_score": 0.702988,
                "pitch_guard_after_best_score": 0.63693,
            }
        ],
    )

    result = run_minus_hard_case_router(
        run_dir=run_dir,
        output_jsonl=run_dir / "minus_hard_case_router.jsonl",
        summary_md=run_dir / "minus_hard_case_router_summary.md",
        registry_configs=[unit_config("minus-focused")],
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "minus_hard_case_router.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["blocker_type"] == "scalar_score_only_pitch_guard_progress" for row in rows)
    assert result["recommended_route"] == "train_separator_attribution_objective"
    assert result["release_ready"] is False


def test_router_reports_unsupported_route_unavailable(tmp_path: Path) -> None:
    run_dir = make_hard_case_run(tmp_path, blockers=["severe_phase_collapse"])

    result = run_minus_hard_case_router(
        run_dir=run_dir,
        output_jsonl=run_dir / "minus_hard_case_router.jsonl",
        summary_md=run_dir / "minus_hard_case_router_summary.md",
        registry_configs=[
            unit_config(
                "phase-unavailable",
                family="phase_safe_existing_model",
                routes=("generate_phase_safe_variants",),
                entrypoint_type="unavailable",
            )
        ],
    )

    rows = [json.loads(line) for line in (run_dir / "minus_hard_case_router.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["recommended_route"] == "generate_phase_safe_variants"
    assert result["can_run_without_new_models"] is False
    assert result["available_route_configs"] == []
    assert all(row["release_ready"] is False for row in rows)
    assert "unavailable" in (run_dir / "minus_hard_case_router_summary.md").read_text(encoding="utf-8")
