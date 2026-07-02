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
from tools.no_reference_score_matrix import validate_mode  # noqa: E402
from tools.run_no_reference_candidate_selector import (  # noqa: E402
    is_acceptable,
    iter_jsonl,
    row_confidence,
    row_reasons,
    row_score,
    select_candidates,
    stable_unique,
)
from tools.train_generic_finetuning_controller_model import stable_hash  # noqa: E402


MANIFEST_NAME = "generic_finetuning_controller_selector_integration_manifest.json"
SELECTOR_DECISIONS_NAME = "generic_finetuning_controller_selector_decisions.jsonl"
INTEGRATION_DECISIONS_NAME = "generic_finetuning_controller_selector_integration_decisions.jsonl"
PROGRESS_NAME = "generic_finetuning_controller_selector_integration_progress.jsonl"

SELECTOR_USABLE_DECISIONS = {"accept_candidate", "select_alternative_candidate"}

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
        "gpu_inference_skip_reason": "controller_selector_integration_uses_existing_metadata_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_selector_integration",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def route_priors(model: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["stem_type"]), str(row["route_id"])): row
        for row in model.get("route_priors", [])
        if isinstance(row, dict) and row.get("stem_type") is not None and row.get("route_id") is not None
    }


def aggregate_alias_prior(
    model: dict[str, Any],
    *,
    stem_type: str,
    route_id: str,
) -> dict[str, Any] | None:
    if not route_id:
        return None
    matches = [
        row
        for row in model.get("route_priors", [])
        if isinstance(row, dict)
        and str(row.get("stem_type") or "") == stem_type
        and str(row.get("route_id") or "").endswith(f"__{route_id}")
    ]
    if not matches:
        return None
    hyper = model.get("hyperparameters") if isinstance(model.get("hyperparameters"), dict) else {}
    alpha = float(hyper.get("alpha") or 1.0)
    beta = float(hyper.get("beta") or 1.0)
    accepted = sum(int(row.get("accepted_count") or 0) for row in matches)
    rejected = sum(int(row.get("rejected_count") or 0) for row in matches)
    total = accepted + rejected
    posterior = (accepted + alpha) / (total + alpha + beta) if total + alpha + beta else 0.0
    return {
        "stem_type": stem_type,
        "route_id": f"alias:{route_id}",
        "example_count": total,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "posterior_accept_rate": round(posterior, 6),
        "recommendation": "prefer" if posterior >= 0.5 else "avoid_or_explore",
        "source_route_ids": sorted(str(row.get("route_id") or "") for row in matches),
        "route_match_kind": "alias_aggregate",
    }


def resolve_controller_prior(
    model: dict[str, Any],
    priors: dict[tuple[str, str], dict[str, Any]],
    *,
    stem_type: str,
    route_id: str,
) -> tuple[dict[str, Any] | None, str, list[str]]:
    exact = priors.get((stem_type, route_id))
    if exact is not None:
        return exact, "exact", [route_id]
    alias = aggregate_alias_prior(model, stem_type=stem_type, route_id=route_id)
    if alias is not None:
        return alias, "alias_aggregate", list(alias["source_route_ids"])
    return None, "missing", []


def stem_for_selector_row(row: dict[str, Any], score_mode: str) -> str:
    mode = validate_mode(score_mode)
    if mode != "both":
        return mode
    selected = row.get("selected_candidate")
    if isinstance(selected, dict) and selected.get("stem_type"):
        return str(selected["stem_type"])
    return "both"


def selected_route_id(row: dict[str, Any]) -> str:
    selected = row.get("selected_candidate")
    if isinstance(selected, dict):
        for key in ("model_id", "route_id", "candidate_id", "base_candidate_id", "selected_candidate_id"):
            value = selected.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = row.get("selected_candidate_id")
    return str(value or "").strip()


def candidate_route_id(row: dict[str, Any]) -> str:
    for key in ("model_id", "route_id", "candidate_id", "base_candidate_id", "selected_candidate_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def selector_is_usable(row: dict[str, Any]) -> bool:
    return str(row.get("decision") or "") in SELECTOR_USABLE_DECISIONS and bool(selected_route_id(row))


def candidate_stem_type(row: dict[str, Any], score_mode: str) -> str:
    mode = validate_mode(score_mode)
    if mode != "both":
        return mode
    return str(row.get("stem_type") or "both")


def controller_candidate_evaluation(
    candidate: dict[str, Any],
    *,
    model: dict[str, Any],
    priors: dict[tuple[str, str], dict[str, Any]],
    score_mode: str,
    accept_threshold: float,
    rank: int,
    hard_case_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = hard_case_policy or {}
    stem = candidate_stem_type(candidate, score_mode)
    route_id = candidate_route_id(candidate)
    prior, match_kind, source_route_ids = resolve_controller_prior(model, priors, stem_type=stem, route_id=route_id)
    posterior = float(prior.get("posterior_accept_rate") or 0.0) if prior else None
    allowed_route_ids = set(hard_case_allowed_route_ids(policy))
    if hard_case_training_objective_required(policy):
        action = "require_separator_training_objective"
    elif hard_case_artifact_safe_route_required(policy) and route_id not in allowed_route_ids:
        action = "controller_block_hard_case_policy_route_mismatch"
    elif prior is None:
        action = "manual_review_missing_controller_route"
    elif float(posterior or 0.0) >= accept_threshold:
        action = "allow_selector_candidate"
    else:
        action = "controller_block_selector_candidate"
    return {
        "rank": rank,
        "candidate_id": route_id,
        "stem_type": stem,
        "selection_score": row_score(candidate, score_mode),
        "confidence": row_confidence(candidate),
        "controller_action": action,
        "controller_route_match_kind": match_kind,
        "controller_matched_route_id": str(prior.get("route_id") or "") if prior else "",
        "controller_source_route_ids": source_route_ids,
        "posterior_accept_rate": posterior,
        **hard_case_policy_fields(policy),
    }


def apply_controller_aware_rerank(
    selector_row: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    *,
    model: dict[str, Any],
    priors: dict[tuple[str, str], dict[str, Any]],
    score_mode: str,
    accept_threshold: float,
    hard_case_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = hard_case_policy or {}
    original_route_id = selected_route_id(selector_row)
    base = dict(selector_row)
    base["selector_original_selected_candidate_id"] = original_route_id
    base["controller_rerank_applied"] = False
    base["controller_rerank_from_candidate_id"] = ""
    base["controller_rerank_evaluated_candidates"] = []
    base.update(hard_case_policy_fields(policy))
    if not selector_is_usable(selector_row):
        return base
    if hard_case_training_objective_required(policy):
        return base

    ordered = sorted(
        candidates,
        key=lambda row: (-row_score(row, score_mode), -row_confidence(row), str(row.get("candidate_id") or "")),
    )
    acceptable = [row for row in ordered if is_acceptable(row, score_mode)]
    allowed_route_ids = set(hard_case_allowed_route_ids(policy))
    if hard_case_artifact_safe_route_required(policy):
        acceptable = [row for row in acceptable if candidate_route_id(row) in allowed_route_ids]
    evaluations: list[dict[str, Any]] = []
    for index, candidate in enumerate(acceptable, start=1):
        evaluation = controller_candidate_evaluation(
            candidate,
            model=model,
            priors=priors,
            score_mode=score_mode,
            accept_threshold=accept_threshold,
            rank=index,
            hard_case_policy=policy,
        )
        evaluations.append(evaluation)
        if evaluation["controller_action"] != "allow_selector_candidate":
            continue
        route_id = str(evaluation["candidate_id"])
        base["controller_rerank_evaluated_candidates"] = evaluations
        if route_id == original_route_id:
            return base
        reranked = dict(base)
        reranked["selected_candidate_id"] = route_id
        reranked["selected_candidate"] = candidate
        reranked["decision"] = "select_alternative_candidate"
        reranked["confidence"] = row_confidence(candidate)
        reranked["reasons"] = stable_unique(
            ["controller_allowed_alternative_after_blocked_top", "selected_by_no_reference_score"] + row_reasons(candidate)
        )
        reranked["controller_rerank_applied"] = True
        reranked["controller_rerank_from_candidate_id"] = original_route_id
        reranked["controller_rerank_evaluated_candidates"] = evaluations
        return reranked

    base["controller_rerank_evaluated_candidates"] = evaluations
    return base


def integration_decision(
    selector_row: dict[str, Any],
    *,
    model: dict[str, Any],
    priors: dict[tuple[str, str], dict[str, Any]],
    score_mode: str,
    accept_threshold: float,
    hard_case_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = hard_case_policy or {}
    stem = stem_for_selector_row(selector_row, score_mode)
    route_id = selected_route_id(selector_row)
    selector_decision = str(selector_row.get("decision") or "")
    selector_usable = selector_is_usable(selector_row)
    prior, match_kind, source_route_ids = resolve_controller_prior(model, priors, stem_type=stem, route_id=route_id)
    posterior = float(prior.get("posterior_accept_rate") or 0.0) if prior else None
    allowed_route_ids = set(hard_case_allowed_route_ids(policy))
    if hard_case_training_objective_required(policy):
        action = "require_separator_training_objective"
    elif hard_case_artifact_safe_route_required(policy) and route_id not in allowed_route_ids:
        action = "controller_block_hard_case_policy_route_mismatch"
    elif selector_decision == "needs_segment_blend":
        action = "respect_selector_segment_blend_plan"
    elif not selector_usable:
        action = "respect_selector_block"
    elif prior is None:
        action = "manual_review_missing_controller_route"
    elif float(posterior or 0.0) >= accept_threshold:
        action = "allow_controller_reranked_candidate" if selector_row.get("controller_rerank_applied") else "allow_selector_candidate"
    else:
        action = "controller_block_selector_candidate"
    return {
        "schema": "generic_finetuning_controller_selector_integration_decision.v1",
        "id": stable_hash(["controller_selector_integration", selector_row.get("track_id"), stem, route_id, selector_decision])[:16],
        "track_id": str(selector_row.get("track_id") or ""),
        "score_mode": validate_mode(score_mode),
        "stem_type": stem,
        "selector_decision": selector_decision,
        "selector_original_selected_candidate_id": str(selector_row.get("selector_original_selected_candidate_id") or route_id),
        "selected_candidate_id": route_id,
        "selector_candidate_count": int(selector_row.get("candidate_count") or 0),
        "selector_confidence": float(selector_row.get("confidence") or 0.0),
        "selector_usable": selector_usable,
        "controller_action": action,
        "controller_rerank_applied": bool(selector_row.get("controller_rerank_applied")),
        "controller_rerank_from_candidate_id": str(selector_row.get("controller_rerank_from_candidate_id") or ""),
        "controller_rerank_evaluated_candidates": list(selector_row.get("controller_rerank_evaluated_candidates") or []),
        "controller_missing_route": selector_usable and prior is None,
        "controller_route_match_kind": match_kind,
        "controller_matched_route_id": str(prior.get("route_id") or "") if prior else "",
        "controller_source_route_ids": source_route_ids,
        "posterior_accept_rate": posterior,
        "route_recommendation": str(prior.get("recommendation") or "") if prior else "",
        **hard_case_policy_fields(policy),
        "can_execute_without_audio_render": True,
        "delivery_allowed": False,
        "presentation_status": "controller_selector_integration_not_for_listen_pack",
        "release_status": "staging_artifact_only",
    }


def build_summary(
    *,
    selector_rows: Sequence[dict[str, Any]],
    integration_rows: Sequence[dict[str, Any]],
    min_selector_usable: int,
) -> dict[str, Any]:
    selector_usable = sum(1 for row in selector_rows if selector_is_usable(row))
    checked = sum(1 for row in integration_rows if row["selector_usable"])
    allowed = sum(
        1
        for row in integration_rows
        if row["controller_action"] in {"allow_selector_candidate", "allow_controller_reranked_candidate"}
    )
    reranked = sum(1 for row in integration_rows if row["controller_action"] == "allow_controller_reranked_candidate")
    blocked = sum(1 for row in integration_rows if row["controller_action"] == "controller_block_selector_candidate")
    policy_blocked = sum(
        1
        for row in integration_rows
        if row["controller_action"] in {"require_separator_training_objective", "controller_block_hard_case_policy_route_mismatch"}
    )
    missing = sum(1 for row in integration_rows if row["controller_missing_route"])
    return {
        "tracks": len(selector_rows),
        "min_selector_usable": min_selector_usable,
        "selector_usable_tracks": selector_usable,
        "controller_checked_candidates": checked,
        "controller_allowed_candidates": allowed,
        "controller_reranked_candidates": reranked,
        "controller_blocked_candidates": blocked,
        "hard_case_policy_blocked_candidates": policy_blocked,
        "controller_missing_routes": missing,
        "controller_override_count": blocked + reranked + policy_blocked,
        "selector_blocked_or_blend_tracks": len(selector_rows) - selector_usable,
        "hard_case_route_policy_applied": any(row.get("hard_case_route_policy_applied") is True for row in integration_rows),
        "hard_case_training_objective_required": any(
            row.get("hard_case_training_objective_required") is True for row in integration_rows
        ),
        "hard_case_artifact_safe_route_required": any(
            row.get("hard_case_artifact_safe_route_required") is True for row in integration_rows
        ),
    }


def promotion_status(summary: dict[str, Any]) -> tuple[str, str, bool]:
    if summary.get("hard_case_training_objective_required") is True:
        return (
            "blocked_hard_case_requires_separator_training_objective",
            "hard-case policy requires separator training objective before generic route staging",
            False,
        )
    if int(summary.get("hard_case_policy_blocked_candidates") or 0) > 0:
        return (
            "blocked_hard_case_policy_route_mismatch",
            "hard-case policy blocked selector routes outside the allowed artifact-safe route set",
            False,
        )
    if int(summary.get("selector_usable_tracks") or 0) < int(summary.get("min_selector_usable") or 0):
        return ("blocked_insufficient_selector_usable_tracks", "not enough selector-usable tracks for staging", False)
    if int(summary.get("controller_checked_candidates") or 0) <= 0:
        return ("blocked_no_controller_checked_candidates", "no selector candidates were checked by controller", False)
    if int(summary.get("controller_missing_routes") or 0) > 0:
        return ("blocked_missing_controller_routes", "selected selector routes are missing from controller model", False)
    return ("controller_selector_integration_staged", "controller overlay can be staged for controlled runtime review", True)


def stage_controller_with_selector(
    *,
    controller_model_json: Path,
    judge_jsonl: Path,
    output_dir: Path,
    score_mode: str = "both",
    min_selector_usable: int = 1,
    accept_threshold: float = 0.5,
    hard_case_route_policy_json: Path | None = None,
) -> dict[str, Any]:
    if min_selector_usable <= 0:
        raise ValueError("min_selector_usable must be > 0")
    if not 0.0 <= accept_threshold <= 1.0:
        raise ValueError("accept_threshold must be between 0 and 1")
    score_mode = validate_mode(score_mode)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    progress(
        progress_path,
        "start",
        controller_model_json=str(controller_model_json),
        judge_jsonl=str(judge_jsonl),
        hard_case_route_policy_json=str(hard_case_route_policy_json or ""),
        score_mode=score_mode,
        min_selector_usable=min_selector_usable,
        accept_threshold=accept_threshold,
    )
    model = read_json(controller_model_json)
    hard_case_policy = read_optional_json(hard_case_route_policy_json)
    judge_rows = list(iter_jsonl(judge_jsonl))
    selector_result = select_candidates(judge_rows, score_mode=score_mode)
    raw_selector_rows = selector_result["rows"]
    grouped_judge_rows: dict[str, list[dict[str, Any]]] = {}
    for row in judge_rows:
        grouped_judge_rows.setdefault(str(row.get("track_id") or ""), []).append(row)
    priors = route_priors(model)
    selector_rows = [
        apply_controller_aware_rerank(
            row,
            grouped_judge_rows.get(str(row.get("track_id") or ""), []),
            model=model,
            priors=priors,
            score_mode=score_mode,
            accept_threshold=accept_threshold,
            hard_case_policy=hard_case_policy,
        )
        for row in raw_selector_rows
    ]
    integration_rows = [
        integration_decision(
            row,
            model=model,
            priors=priors,
            score_mode=score_mode,
            accept_threshold=accept_threshold,
            hard_case_policy=hard_case_policy,
        )
        for row in selector_rows
    ]
    summary = build_summary(
        selector_rows=selector_rows,
        integration_rows=integration_rows,
        min_selector_usable=min_selector_usable,
    )
    status, reason, can_stage = promotion_status(summary)
    fingerprint = stable_hash(
        {
            "controller_model_json": str(controller_model_json),
            "judge_jsonl": str(judge_jsonl),
            "hard_case_route_policy_json": str(hard_case_route_policy_json or ""),
            "hard_case_route_policy": hard_case_policy,
            "score_mode": score_mode,
            "min_selector_usable": min_selector_usable,
            "accept_threshold": accept_threshold,
            "integration": [
                {
                    "track_id": row["track_id"],
                    "stem_type": row["stem_type"],
                    "selected_candidate_id": row["selected_candidate_id"],
                    "controller_action": row["controller_action"],
                }
                for row in integration_rows
            ],
        }
    )
    manifest = {
        "schema": "generic_finetuning_controller_selector_integration_manifest.v1",
        "manifest_type": "generic_finetuning_controller_selector_integration",
        "controller_model_json": str(controller_model_json),
        "judge_jsonl": str(judge_jsonl),
        "hard_case_route_policy_json": str(hard_case_route_policy_json or ""),
        "hard_case_route_policy": hard_case_policy,
        "output_dir": str(output_dir),
        "score_mode": score_mode,
        "fingerprint": fingerprint,
        "selector_summary": selector_result["summary"],
        "integration_summary": summary,
        "promotion": {
            "status": status,
            "reason": reason,
            "can_stage_controller_candidate": can_stage,
            "can_edit_model_registry": False,
            "next_required_step": "shadow_runtime_batch_or_policy_review" if can_stage else "fix_controller_selector_integration_blockers",
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "controller_selector_integration_not_for_listen_pack",
        "release_status": "staging_artifact_only",
        "release_ready": False,
        "safety": {
            **SAFETY_FLAGS,
            "metadata_only": True,
            "not_release_artifact": True,
            "registry_edit_blocked": True,
            "audio_inference_started": False,
            "audio_render_started": False,
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "selector_decisions_jsonl": str(output_dir / SELECTOR_DECISIONS_NAME),
            "integration_decisions_jsonl": str(output_dir / INTEGRATION_DECISIONS_NAME),
            "progress_jsonl": str(progress_path),
        },
    }
    progress(progress_path, "write_outputs", **summary, status=status)
    write_jsonl(output_dir / SELECTOR_DECISIONS_NAME, raw_selector_rows)
    write_jsonl(output_dir / INTEGRATION_DECISIONS_NAME, integration_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage a generic controller overlay on no-reference selector decisions.")
    parser.add_argument("--controller-model-json", type=Path, required=True)
    parser.add_argument("--judge-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-route-policy-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-mode", choices=("vocal", "minus", "both"), default="both")
    parser.add_argument("--min-selector-usable", type=int, default=1)
    parser.add_argument("--accept-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="stage_generic_finetuning_controller_with_no_reference_selector.py",
        entrypoint_path="/app/tools/stage_generic_finetuning_controller_with_no_reference_selector.py",
    )
    args = parse_args(argv)
    manifest = stage_controller_with_selector(
        controller_model_json=args.controller_model_json,
        judge_jsonl=args.judge_jsonl,
        hard_case_route_policy_json=args.hard_case_route_policy_json,
        output_dir=args.output_dir,
        score_mode=args.score_mode,
        min_selector_usable=int(args.min_selector_usable),
        accept_threshold=float(args.accept_threshold),
    )
    print(
        json.dumps(
            {
                "selector_summary": manifest["selector_summary"],
                "integration_summary": manifest["integration_summary"],
                "promotion": manifest["promotion"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
