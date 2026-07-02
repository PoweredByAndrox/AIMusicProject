#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.hard_case_route_policy_controls import (  # noqa: E402
    hard_case_allowed_route_ids,
    hard_case_artifact_safe_route_required,
    hard_case_policy_applies,
    hard_case_policy_fields,
    hard_case_training_objective_required,
)
from tools.train_generic_finetuning_controller_model import (  # noqa: E402
    example_accepted,
    example_route,
    example_stem,
    iter_jsonl,
    stable_hash,
)


MANIFEST_NAME = "generic_finetuning_controller_runtime_smoke_manifest.json"
REPORT_NAME = "generic_finetuning_controller_runtime_smoke_report.json"
DECISIONS_NAME = "generic_finetuning_controller_runtime_smoke_decisions.jsonl"
PROGRESS_NAME = "generic_finetuning_controller_runtime_smoke_progress.jsonl"

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}

START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return read_json(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "controller_runtime_smoke_uses_existing_metadata_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_runtime_smoke",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def trainable_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("trainable") is not False]


def expected_action(row: dict[str, Any]) -> str:
    target = row.get("target")
    if isinstance(target, dict) and target.get("action"):
        return str(target["action"])
    return "reuse_route_candidate" if example_accepted(row) else "avoid_route_and_generate_alternative"


def route_priors(model: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["stem_type"]), str(row["route_id"])): row
        for row in model.get("route_priors", [])
        if isinstance(row, dict) and row.get("stem_type") is not None and row.get("route_id") is not None
    }


def replay_decision(
    row: dict[str, Any],
    prior: dict[str, Any] | None,
    *,
    accept_threshold: float,
    hard_case_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = hard_case_policy or {}
    stem = example_stem(row)
    route = example_route(row)
    expected = expected_action(row)
    allowed_route_ids = set(hard_case_allowed_route_ids(policy))
    if hard_case_training_objective_required(policy):
        predicted = "require_separator_training_objective"
        posterior = None
        recommendation = str(policy.get("route_block_reason") or "hard_case_requires_separator_training_objective")
    elif hard_case_artifact_safe_route_required(policy) and route not in allowed_route_ids:
        predicted = "avoid_route_and_generate_artifact_safe_alternative"
        posterior = None
        recommendation = "hard_case_policy_route_mismatch"
    elif prior is None:
        predicted = "manual_review_missing_route"
        posterior = None
        recommendation = "missing_route"
    else:
        posterior = float(prior.get("posterior_accept_rate") or 0.0)
        predicted = "reuse_route_candidate" if posterior >= accept_threshold else "avoid_route_and_generate_alternative"
        recommendation = str(prior.get("recommendation") or "")
    return {
        "schema": "generic_finetuning_controller_runtime_smoke_decision.v1",
        "id": stable_hash(["runtime_smoke", row.get("id"), stem, route, expected])[:16],
        "source_example_id": str(row.get("id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "stem_type": stem,
        "route_id": route,
        "posterior_accept_rate": posterior,
        "route_recommendation": recommendation,
        "predicted_action": predicted,
        "expected_action": expected,
        "action_correct": predicted == expected,
        "missing_route": prior is None,
        "hard_case_policy_blocked": predicted
        in {"require_separator_training_objective", "avoid_route_and_generate_artifact_safe_alternative"},
        **hard_case_policy_fields(policy),
    }


def promotion_status(
    *,
    report: dict[str, Any],
    min_replay_examples: int,
    min_action_accuracy: float,
) -> tuple[str, str, bool]:
    if report.get("hard_case_training_objective_required") is True:
        return (
            "blocked_hard_case_requires_separator_training_objective",
            "hard-case policy requires separator training objective before runtime smoke promotion",
            False,
        )
    if int(report.get("hard_case_policy_blocked_examples") or 0) > 0:
        return (
            "blocked_hard_case_policy_route_mismatch",
            "hard-case policy blocked replay examples outside the allowed artifact-safe route set",
            False,
        )
    replayed = int(report.get("replayed_examples") or 0)
    missing = int(report.get("missing_route_examples") or 0)
    accuracy = float(report.get("action_accuracy") or 0.0)
    if replayed < min_replay_examples:
        return ("blocked_insufficient_replay_examples", "not enough runtime smoke examples", False)
    if missing:
        return ("blocked_missing_routes", "runtime smoke examples include routes missing from controller model", False)
    if accuracy < min_action_accuracy:
        return ("blocked_action_accuracy_below_threshold", "runtime smoke action accuracy below threshold", False)
    return ("runtime_smoke_replay_passed", "runtime smoke replay passed metadata action gates", True)


def run_runtime_smoke_replay(
    *,
    controller_model_json: Path,
    replay_examples_jsonl: Path,
    output_dir: Path,
    min_replay_examples: int = 20,
    min_action_accuracy: float = 0.9,
    accept_threshold: float = 0.5,
    hard_case_route_policy_json: Path | None = None,
) -> dict[str, Any]:
    if min_replay_examples <= 0:
        raise ValueError("min_replay_examples must be > 0")
    if not 0.0 <= min_action_accuracy <= 1.0:
        raise ValueError("min_action_accuracy must be between 0 and 1")
    if not 0.0 <= accept_threshold <= 1.0:
        raise ValueError("accept_threshold must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(
        progress_path,
        "start",
        controller_model_json=str(controller_model_json),
        replay_examples_jsonl=str(replay_examples_jsonl),
        hard_case_route_policy_json=str(hard_case_route_policy_json or ""),
        min_replay_examples=min_replay_examples,
        min_action_accuracy=min_action_accuracy,
        accept_threshold=accept_threshold,
    )
    model = read_json(controller_model_json)
    hard_case_policy = read_optional_json(hard_case_route_policy_json)
    priors = route_priors(model)
    rows = trainable_rows(iter_jsonl(replay_examples_jsonl))
    decisions = [
        replay_decision(
            row,
            priors.get((example_stem(row), example_route(row))),
            accept_threshold=accept_threshold,
            hard_case_policy=hard_case_policy,
        )
        for row in rows
    ]
    replayed = len(decisions)
    missing = sum(1 for row in decisions if row["missing_route"])
    policy_blocked = sum(1 for row in decisions if row["hard_case_policy_blocked"])
    correct = sum(1 for row in decisions if row["action_correct"])
    accuracy = round(correct / replayed, 6) if replayed else 0.0
    report = {
        "schema": "generic_finetuning_controller_runtime_smoke_report.v1",
        "replayed_examples": replayed,
        "correct_actions": correct,
        "missing_route_examples": missing,
        "hard_case_policy_blocked_examples": policy_blocked,
        "hard_case_route_policy_applied": hard_case_policy_applies(hard_case_policy),
        "hard_case_training_objective_required": hard_case_training_objective_required(hard_case_policy),
        "hard_case_artifact_safe_route_required": hard_case_artifact_safe_route_required(hard_case_policy),
        "hard_case_allowed_route_ids": hard_case_allowed_route_ids(hard_case_policy),
        "action_accuracy": accuracy,
        "min_replay_examples": min_replay_examples,
        "min_action_accuracy": min_action_accuracy,
        "accept_threshold": accept_threshold,
    }
    status, reason, can_stage = promotion_status(
        report=report,
        min_replay_examples=min_replay_examples,
        min_action_accuracy=min_action_accuracy,
    )
    fingerprint = stable_hash(
        {
            "controller_model_json": str(controller_model_json),
            "replay_examples_jsonl": str(replay_examples_jsonl),
            "hard_case_route_policy_json": str(hard_case_route_policy_json or ""),
            "hard_case_route_policy": hard_case_policy,
            "decisions": [
                {
                    "source_example_id": row["source_example_id"],
                    "route_id": row["route_id"],
                    "predicted_action": row["predicted_action"],
                    "expected_action": row["expected_action"],
                }
                for row in decisions
            ],
            "min_replay_examples": min_replay_examples,
            "min_action_accuracy": min_action_accuracy,
            "accept_threshold": accept_threshold,
        }
    )
    manifest = {
        "schema": "generic_finetuning_controller_runtime_smoke_manifest.v1",
        "manifest_type": "generic_finetuning_controller_runtime_smoke",
        "controller_model_json": str(controller_model_json),
        "replay_examples_jsonl": str(replay_examples_jsonl),
        "hard_case_route_policy_json": str(hard_case_route_policy_json or ""),
        "hard_case_route_policy": hard_case_policy,
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "runtime_smoke": report,
        "promotion": {
            "status": status,
            "reason": reason,
            "can_stage_controller_candidate": can_stage,
            "can_edit_model_registry": False,
            "next_required_step": "controlled_runtime_integration_or_registry_review" if can_stage else "fix_runtime_smoke_blockers",
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "controller_runtime_smoke_not_for_listen_pack",
        "release_status": "runtime_smoke_artifact_only",
        "release_ready": False,
        "safety": {
            **SAFETY_FLAGS,
            "metadata_only": True,
            "not_release_artifact": True,
            "registry_edit_blocked": True,
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "report_json": str(output_dir / REPORT_NAME),
            "decisions_jsonl": str(output_dir / DECISIONS_NAME),
            "progress_jsonl": str(progress_path),
        },
    }
    progress(progress_path, "write_outputs", **report, status=status)
    write_jsonl(output_dir / DECISIONS_NAME, decisions)
    write_json(output_dir / REPORT_NAME, report)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a controller model on metadata examples as a runtime smoke gate.")
    parser.add_argument("--controller-model-json", type=Path, required=True)
    parser.add_argument("--replay-examples-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-route-policy-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-replay-examples", type=int, default=20)
    parser.add_argument("--min-action-accuracy", type=float, default=0.9)
    parser.add_argument("--accept-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_generic_finetuning_controller_runtime_smoke_replay.py",
        entrypoint_path="/app/tools/run_generic_finetuning_controller_runtime_smoke_replay.py",
    )
    args = parse_args(argv)
    manifest = run_runtime_smoke_replay(
        controller_model_json=args.controller_model_json,
        replay_examples_jsonl=args.replay_examples_jsonl,
        hard_case_route_policy_json=args.hard_case_route_policy_json,
        output_dir=args.output_dir,
        min_replay_examples=int(args.min_replay_examples),
        min_action_accuracy=float(args.min_action_accuracy),
        accept_threshold=float(args.accept_threshold),
    )
    print(json.dumps({"runtime_smoke": manifest["runtime_smoke"], "promotion": manifest["promotion"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
