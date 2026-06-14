from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import stable_hash  # noqa: E402
from tools.no_reference_candidate_job_registry import (  # noqa: E402
    CandidateJobConfig,
    build_candidate_job_registry,
)


DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00Z"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def config_priority(config: CandidateJobConfig) -> tuple[int, int, int, str]:
    produces_both_rank = 0 if config.produces_both else 1
    target_rank = {"both": 0, "minus": 1, "vocal": 2}.get(config.target_mode, 9)
    unavailable_rank = 1 if config.entrypoint_type == "unavailable" else 0
    return (unavailable_rank, produces_both_rank, target_rank, config.candidate_config_id)


def expected_runtime_for(config: CandidateJobConfig, previous_runtime_estimates: dict[str, float] | None) -> int:
    estimates = previous_runtime_estimates or {}
    estimate = estimates.get(config.shared_execution_group_id, estimates.get(config.candidate_config_id, config.timeout_sec))
    return max(1, int(round(float(estimate))))


def target_modes_produced(config: CandidateJobConfig) -> list[str]:
    modes = list(config.materialized_candidate_modes or (config.target_mode,))
    if config.produces_vocal and "vocal" not in modes:
        modes.append("vocal")
    if config.produces_minus and "minus" not in modes:
        modes.append("minus")
    if config.produces_vocal and config.produces_minus and "both" not in modes:
        modes.append("both")
    order = {"vocal": 0, "minus": 1, "both": 2}
    return sorted({mode for mode in modes if mode in order}, key=lambda mode: order[mode])


def plan_id(input_audio_path: Path, config: CandidateJobConfig) -> str:
    return "job_plan_" + stable_hash(
        {
            "input_audio_path": str(input_audio_path),
            "candidate_config_id": config.candidate_config_id,
            "shared_execution_group_id": config.shared_execution_group_id,
            "source_config_hash": config.source_config_hash,
        }
    )[:16]


def build_candidate_job_plan(
    *,
    input_audio_path: Path,
    registry_configs: Iterable[CandidateJobConfig],
    candidate_profile: str,
    candidate_diversity_profile: str,
    max_candidates: int,
    max_runtime_sec: int,
    previous_runtime_estimates: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    configs = sorted(list(registry_configs), key=config_priority)
    limit = max(0, int(max_candidates))
    selected_groups: set[str] = set()
    consumed_runtime = 0
    runnable_count = 0
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs, start=1):
        expected_runtime_sec = expected_runtime_for(config, previous_runtime_estimates)
        group_id = str(config.shared_execution_group_id or config.candidate_config_id)
        can_run = True
        reason_selected = "selected"
        reason_skipped = ""
        if config.entrypoint_type == "unavailable":
            can_run = False
            reason_selected = ""
            reason_skipped = "unavailable_entrypoint"
        elif group_id in selected_groups:
            can_run = False
            reason_selected = ""
            reason_skipped = "duplicate_shared_execution_group"
        elif limit and runnable_count >= limit:
            can_run = False
            reason_selected = ""
            reason_skipped = "max_candidates_limit"
        elif max_runtime_sec > 0 and consumed_runtime + expected_runtime_sec > int(max_runtime_sec):
            can_run = False
            reason_selected = ""
            reason_skipped = "insufficient_runtime_budget"

        if can_run:
            selected_groups.add(group_id)
            consumed_runtime += expected_runtime_sec
            runnable_count += 1
            if config.produces_both:
                reason_selected = "produces_complete_pair"

        rows.append(
            {
                "job_plan_id": plan_id(input_audio_path, config),
                "candidate_config_id": config.candidate_config_id,
                "shared_execution_group_id": group_id,
                "priority": index,
                "expected_runtime_sec": expected_runtime_sec,
                "timeout_sec": max(int(config.timeout_sec), expected_runtime_sec),
                "target_modes_produced": target_modes_produced(config),
                "candidate_families_materialized": [config.candidate_family],
                "reason_selected": reason_selected,
                "reason_skipped": reason_skipped,
                "can_run": bool(can_run),
                "candidate_profile": candidate_profile,
                "candidate_diversity_profile": candidate_diversity_profile,
                "source_config_hash": config.source_config_hash,
                "created_at": DETERMINISTIC_CREATED_AT,
            }
        )
    return rows


def write_candidate_job_plan(path: Path, plan_rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, plan_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a no-reference local separator candidate job plan.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-profile", choices=("safe", "balanced", "exploration"), default="safe")
    parser.add_argument("--candidate-diversity-profile", choices=("safe", "balanced", "exploration"), default="safe")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-runtime-sec", type=int, default=600)
    args = parser.parse_args(argv)
    configs = build_candidate_job_registry(profile=args.candidate_diversity_profile)
    plan = build_candidate_job_plan(
        input_audio_path=args.input_audio,
        registry_configs=configs,
        candidate_profile=args.candidate_profile,
        candidate_diversity_profile=args.candidate_diversity_profile,
        max_candidates=args.max_candidates,
        max_runtime_sec=args.max_runtime_sec,
    )
    write_candidate_job_plan(args.output_jsonl, plan)
    print(json.dumps({"candidate_job_plan_jsonl": str(args.output_jsonl), "planned_rows": len(plan)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
