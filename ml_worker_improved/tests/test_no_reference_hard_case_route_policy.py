from __future__ import annotations

import json
from pathlib import Path

from tools.no_reference_candidate_job_registry import CandidateJobConfig


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def htdemucs_6s_both_config(*, available: bool = True) -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id="local_registered__htdemucs_6s__both__shared",
        route_id="local_registered__htdemucs_6s__both__shared",
        candidate_family="artifact_safe_default",
        target_mode="both",
        source_model_id="htdemucs_6s",
        source_tool="tools/run_registered_model_candidate.py",
        route_family="htdemucs_6s",
        entrypoint_type="python_script" if available else "unavailable",
        command_template=("python", "/app/tools/run_registered_model_candidate.py"),
        expected_outputs={"vocal_path": "", "minus_path": ""},
        gpu_required=True,
        timeout_sec=900,
        max_retries=0,
        profile_allowed=("balanced", "exploration"),
        safety_notes=("uses_existing_model_registry_entry", "no_model_download"),
        supported_parameters=("source_audio_path", "output_dir", "model_id", "device", "target_stem"),
        shared_execution_group_id="local_registered__htdemucs_6s__default",
        produces_vocal=True,
        produces_minus=True,
        produces_both=True,
        materialized_candidate_modes=("vocal", "minus", "both"),
        reuse_outputs_across_modes=True,
        available=available,
        can_run_without_download=available,
        can_run_without_registry_edit=True,
    )


def write_policy_inputs(
    run_dir: Path,
    *,
    blockers: list[str] | None = None,
    failure_reason: str = "",
    route_family: str = "htdemucs_ft",
    route_space_limited: bool = True,
    accepted_minus: bool = False,
) -> None:
    blockers = blockers or []
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run-a",
            "output_status": "needs_more_candidates" if blockers or failure_reason else "audio_preview_ready",
            "failure_reason": failure_reason,
            "safety_blockers": [failure_reason] if failure_reason else [],
            "release_ready": False,
        },
    )
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "candidate-a",
                "route_id": "local_registered__htdemucs_ft__both__shared",
                "route_family": route_family,
                "minus_path": "/tmp/minus.wav",
                "can_use_for_minus": True,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "candidate-a",
                "minus_decision": "accept_candidate" if accepted_minus else "blocked_hard_case",
                "decision": "accept_candidate" if accepted_minus else "blocked_hard_case",
                "minus_blockers": blockers,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(run_dir / "selected_candidates.jsonl", [] if blockers or failure_reason else [{"candidate_id": "candidate-a"}])
    write_jsonl(run_dir / "pair_compatibility.jsonl", [] if blockers or failure_reason else [{"pair_decision": "pair_compatible"}])
    write_json(
        run_dir / "minus_route_expansion_manifest.json",
        {
            "summary": {
                "route_space_limited": route_space_limited,
                "quality_improved_route_ids": ["local_registered__htdemucs_6s__both__shared"],
                "accepted_minus_candidates": 1,
                "release_ready_count": 0,
            },
            "release_ready": False,
        },
    )


def test_policy_detects_severe_vocal_bleed_and_selects_hard_case_route(tmp_path: Path) -> None:
    from tools.no_reference_hard_case_route_policy import build_hard_case_route_policy

    write_policy_inputs(tmp_path, blockers=["severe_vocal_bleed_in_minus"])

    policy = build_hard_case_route_policy(
        run_dir=tmp_path,
        registry_configs=[htdemucs_6s_both_config()],
        conditional_route_policy="hard_case_only",
    )

    assert policy["hard_case_detected"] is True
    assert "severe_vocal_bleed_in_minus" in policy["trigger_reasons"]
    assert policy["selected_conditional_route_ids"] == ["local_registered__htdemucs_6s__both__shared"]
    assert policy["route_scope"] == "hard_case_only"
    assert policy["route_allowed"] is True
    assert policy["default_profile_changed"] is False
    assert policy["release_ready"] is False


def test_policy_detects_phase_collapse(tmp_path: Path) -> None:
    from tools.no_reference_hard_case_route_policy import build_hard_case_route_policy

    write_policy_inputs(tmp_path, blockers=["severe_phase_collapse"])

    policy = build_hard_case_route_policy(
        run_dir=tmp_path,
        registry_configs=[htdemucs_6s_both_config()],
        conditional_route_policy="hard_case_only",
    )

    assert "severe_phase_collapse" in policy["trigger_reasons"]
    assert policy["route_allowed"] is True


def test_policy_detects_missing_safe_mode_candidate(tmp_path: Path) -> None:
    from tools.no_reference_hard_case_route_policy import build_hard_case_route_policy

    write_policy_inputs(tmp_path, failure_reason="missing_safe_mode_candidate")

    policy = build_hard_case_route_policy(
        run_dir=tmp_path,
        registry_configs=[htdemucs_6s_both_config()],
        conditional_route_policy="hard_case_only",
    )

    assert "missing_safe_mode_candidate" in policy["trigger_reasons"]
    assert policy["route_allowed"] is True


def test_policy_does_not_select_without_hard_case_trigger(tmp_path: Path) -> None:
    from tools.no_reference_hard_case_route_policy import build_hard_case_route_policy

    write_policy_inputs(tmp_path, accepted_minus=True, route_space_limited=False)

    policy = build_hard_case_route_policy(
        run_dir=tmp_path,
        registry_configs=[htdemucs_6s_both_config()],
        conditional_route_policy="hard_case_only",
    )

    assert policy["hard_case_detected"] is False
    assert policy["selected_conditional_route_ids"] == []
    assert policy["route_allowed"] is False
    assert policy["route_scope"] == "disabled"


def test_policy_blocks_route_when_not_locally_available(tmp_path: Path) -> None:
    from tools.no_reference_hard_case_route_policy import build_hard_case_route_policy

    write_policy_inputs(tmp_path, blockers=["severe_phase_collapse"])

    policy = build_hard_case_route_policy(
        run_dir=tmp_path,
        registry_configs=[htdemucs_6s_both_config(available=False)],
        conditional_route_policy="hard_case_only",
    )

    assert policy["hard_case_detected"] is True
    assert policy["selected_conditional_route_ids"] == []
    assert policy["route_allowed"] is False
    assert policy["route_block_reason"] == "conditional_route_not_available_locally"
