from __future__ import annotations

import json
from pathlib import Path

from tools.no_reference_candidate_job_registry import CandidateJobConfig
from tools.no_reference_candidate_job_scheduler import build_candidate_job_plan, write_candidate_job_plan


def unit_config(
    config_id: str,
    *,
    target_mode: str,
    family: str,
    group_id: str,
    produces_vocal: bool,
    produces_minus: bool,
    materialized_modes: tuple[str, ...],
    timeout_sec: int = 100,
    entrypoint_type: str = "python_script",
) -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id=config_id,
        candidate_family=family,
        target_mode=target_mode,
        source_model_id="unit-model",
        source_tool="unit-tool",
        entrypoint_type=entrypoint_type,
        command_template=("unit-runner", "{input_audio}", "{output_dir}"),
        expected_outputs={"vocal_path": "{output_dir}/vocals.wav", "minus_path": "{output_dir}/minus.wav"},
        gpu_required=False,
        timeout_sec=timeout_sec,
        max_retries=0,
        profile_allowed=("safe", "balanced"),
        safety_notes=("unit",),
        supported_parameters=("target_stem",),
        target_stem="minus" if produces_minus else "vocal",
        shared_execution_group_id=group_id,
        produces_vocal=produces_vocal,
        produces_minus=produces_minus,
        materialized_candidate_modes=materialized_modes,
        reuse_outputs_across_modes=True,
        hard_case_routes=("generate_minus_focused_variants",) if target_mode == "minus" else (),
    )


def test_scheduler_prioritizes_shared_both_job_and_dedupes_group(tmp_path: Path) -> None:
    configs = [
        unit_config("unit-vocal", target_mode="vocal", family="vocal_focused_default", group_id="shared-a", produces_vocal=True, produces_minus=False, materialized_modes=("vocal",)),
        unit_config("unit-minus", target_mode="minus", family="minus_focused_default", group_id="shared-a", produces_vocal=False, produces_minus=True, materialized_modes=("minus",)),
        unit_config("unit-both", target_mode="both", family="balanced_pair_default", group_id="shared-a", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both")),
    ]

    plan = build_candidate_job_plan(
        input_audio_path=tmp_path / "song.wav",
        registry_configs=configs,
        candidate_profile="safe",
        candidate_diversity_profile="safe",
        max_candidates=6,
        max_runtime_sec=300,
    )

    runnable = [row for row in plan if row["can_run"] is True]
    skipped = [row for row in plan if row["can_run"] is False]

    assert [row["candidate_config_id"] for row in runnable] == ["unit-both"]
    assert runnable[0]["shared_execution_group_id"] == "shared-a"
    assert runnable[0]["target_modes_produced"] == ["vocal", "minus", "both"]
    assert runnable[0]["candidate_families_materialized"] == ["balanced_pair_default"]
    assert {row["reason_skipped"] for row in skipped} == {"duplicate_shared_execution_group"}


def test_scheduler_writes_plan_and_skips_lower_priority_when_budget_is_low(tmp_path: Path) -> None:
    configs = [
        unit_config("unit-both-a", target_mode="both", family="balanced_pair_default", group_id="group-a", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both"), timeout_sec=80),
        unit_config("unit-both-b", target_mode="both", family="artifact_safe_default", group_id="group-b", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both"), timeout_sec=80),
    ]
    output = tmp_path / "candidate_job_plan.jsonl"

    plan = build_candidate_job_plan(
        input_audio_path=tmp_path / "song.wav",
        registry_configs=configs,
        candidate_profile="balanced",
        candidate_diversity_profile="balanced",
        max_candidates=8,
        max_runtime_sec=100,
    )
    write_candidate_job_plan(output, plan)
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["can_run"] for row in plan] == [True, False]
    assert plan[1]["reason_skipped"] == "insufficient_runtime_budget"
    assert written == plan


def test_scheduler_can_select_minus_hard_case_route(tmp_path: Path) -> None:
    configs = [
        unit_config("unit-both", target_mode="both", family="balanced_pair_default", group_id="group-a", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both")),
        unit_config("unit-minus", target_mode="minus", family="minus_focused_existing_model", group_id="group-b", produces_vocal=False, produces_minus=True, materialized_modes=("minus",)),
    ]

    plan = build_candidate_job_plan(
        input_audio_path=tmp_path / "song.wav",
        registry_configs=configs,
        candidate_profile="balanced",
        candidate_diversity_profile="balanced",
        max_candidates=4,
        max_runtime_sec=300,
        candidate_route_filter="generate_minus_focused_variants",
    )

    runnable = [row for row in plan if row["can_run"] is True]
    route_skips = [row for row in plan if row["reason_skipped"] == "route_not_selected"]
    assert [row["candidate_config_id"] for row in runnable] == ["unit-minus"]
    assert runnable[0]["target_modes_produced"] == ["minus"]
    assert route_skips[0]["candidate_config_id"] == "unit-both"
