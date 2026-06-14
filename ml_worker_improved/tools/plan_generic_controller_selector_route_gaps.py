#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.train_generic_finetuning_controller_model import stable_hash  # noqa: E402


ROUTE_GAPS_NAME = "generic_controller_selector_route_gaps.jsonl"
GAP_MANIFEST_NAME = "generic_controller_selector_route_gap_plan_manifest.json"
GAP_PROGRESS_NAME = "generic_controller_selector_route_gap_plan_progress.jsonl"

START_TIME = time.monotonic()

SAFETY_FLAGS = {
    "metadata_only": True,
    "audio_inference_started": False,
    "audio_render_started": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "delivery_allowed": False,
    "release_ready": False,
    "not_release_artifact": True,
    "thresholds_relaxed": False,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


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
        "gpu_inference_skip_reason": "controller_selector_route_gap_planner_uses_existing_metadata_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_route_gap_planning",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integration_decision_path(item: dict[str, Any]) -> Path:
    manifest_path = Path(str(item.get("single_manifest_json") or ""))
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict) and outputs.get("integration_decisions_jsonl"):
        return Path(str(outputs["integration_decisions_jsonl"]))
    return manifest_path.parent / "generic_finetuning_controller_selector_integration_decisions.jsonl"


def candidate_rows_from_decision(decision: dict[str, Any]) -> list[dict[str, Any]]:
    evaluated = decision.get("controller_rerank_evaluated_candidates")
    if isinstance(evaluated, list) and evaluated:
        return [row for row in evaluated if isinstance(row, dict)]
    return [
        {
            "candidate_id": decision.get("selected_candidate_id"),
            "stem_type": decision.get("stem_type"),
            "controller_action": decision.get("controller_action"),
            "controller_route_match_kind": decision.get("controller_route_match_kind"),
            "posterior_accept_rate": decision.get("posterior_accept_rate"),
            "selection_score": None,
            "confidence": decision.get("selector_confidence"),
        }
    ]


def action_bucket(action: str) -> str:
    if action in {"allow_selector_candidate", "allow_controller_reranked_candidate"}:
        return "allowed"
    if action == "controller_block_selector_candidate":
        return "blocked"
    if action == "manual_review_missing_controller_route":
        return "missing"
    return "other"


def route_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("stem_type") or "unknown"), str(row.get("candidate_id") or row.get("selected_candidate_id") or ""))


def collect_route_observations(items_jsonl: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for item_index, item in enumerate(iter_jsonl(items_jsonl), start=1):
        decision_path = integration_decision_path(item)
        for decision_index, decision in enumerate(iter_jsonl(decision_path), start=1):
            for candidate in candidate_rows_from_decision(decision):
                stem_type, route_id = route_key(candidate)
                if not route_id:
                    continue
                action = str(candidate.get("controller_action") or "")
                observations.append(
                    {
                        "item_index": item_index,
                        "decision_index": decision_index,
                        "judge_jsonl": str(item.get("judge_jsonl") or ""),
                        "integration_decisions_jsonl": str(decision_path),
                        "track_id": str(decision.get("track_id") or ""),
                        "stem_type": stem_type,
                        "route_id": route_id,
                        "controller_action": action,
                        "action_bucket": action_bucket(action),
                        "controller_route_match_kind": str(candidate.get("controller_route_match_kind") or ""),
                        "posterior_accept_rate": as_float_or_none(candidate.get("posterior_accept_rate")),
                        "selection_score": as_float_or_none(candidate.get("selection_score")),
                        "confidence": as_float_or_none(candidate.get("confidence")),
                    }
                )
    return observations


def average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def recommended_action(*, allowed: int, blocked: int, missing: int, min_repeated_blocked: int) -> tuple[str, str, bool]:
    if allowed:
        return ("keep_candidate_route_available", "P2", False)
    if blocked >= min_repeated_blocked:
        return ("generate_new_candidate_wave", "P0", True)
    if missing:
        return ("collect_controller_prior_evidence", "P1", False)
    if blocked:
        return ("keep_blocked_until_more_evidence", "P2", False)
    return ("monitor_only", "P3", False)


def build_route_gap_rows(observations: Sequence[dict[str, Any]], *, min_repeated_blocked: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(row["stem_type"], row["route_id"])].append(row)

    route_rows: list[dict[str, Any]] = []
    for (stem_type, route_id), rows in grouped.items():
        allowed = sum(1 for row in rows if row["action_bucket"] == "allowed")
        blocked = sum(1 for row in rows if row["action_bucket"] == "blocked")
        missing = sum(1 for row in rows if row["action_bucket"] == "missing")
        other = len(rows) - allowed - blocked - missing
        action, priority, next_wave = recommended_action(
            allowed=allowed,
            blocked=blocked,
            missing=missing,
            min_repeated_blocked=min_repeated_blocked,
        )
        posterior_values = [float(row["posterior_accept_rate"]) for row in rows if row["posterior_accept_rate"] is not None]
        score_values = [float(row["selection_score"]) for row in rows if row["selection_score"] is not None]
        confidence_values = [float(row["confidence"]) for row in rows if row["confidence"] is not None]
        source_files = sorted({str(row["integration_decisions_jsonl"]) for row in rows})
        match_kinds = sorted({str(row["controller_route_match_kind"]) for row in rows if row["controller_route_match_kind"]})
        route_rows.append(
            {
                "schema": "generic_controller_selector_route_gap.v1",
                "id": stable_hash(["controller_selector_route_gap", stem_type, route_id, rows])[:16],
                "stem_type": stem_type,
                "route_id": route_id,
                "observation_count": len(rows),
                "allowed_count": allowed,
                "blocked_count": blocked,
                "missing_count": missing,
                "other_count": other,
                "controller_route_match_kinds": match_kinds,
                "posterior_accept_rate_mean": average(posterior_values),
                "selection_score_mean": average(score_values),
                "confidence_mean": average(confidence_values),
                "recommended_action": action,
                "priority": priority,
                "next_wave_required": next_wave,
                "source_integration_decisions_jsonls": source_files,
                "forbidden_actions": [
                    "do_not_mark_release_ready_from_gap_plan",
                    "do_not_start_training_without_approved_audio_examples",
                    "do_not_relax_selector_or_controller_thresholds",
                    "do_not_edit_model_registry_from_gap_plan",
                ],
            }
        )
    route_rows.sort(key=lambda row: (row["priority"], row["stem_type"], row["route_id"]))
    return route_rows


def promotion_status(summary: dict[str, Any]) -> tuple[str, str, bool]:
    if int(summary.get("route_count") or 0) <= 0:
        return ("blocked_no_route_observations", "shadow batch did not contain route observations", False)
    if int(summary.get("next_wave_route_count") or 0) > 0:
        return ("candidate_wave_required", "blocked controller routes require a new candidate wave", False)
    if int(summary.get("missing_route_count") or 0) > 0:
        return ("controller_prior_evidence_required", "missing controller priors require evidence before training", False)
    return ("route_gap_plan_ready", "route gap plan has no blocking next-wave routes", False)


def plan_route_gaps(
    *,
    shadow_batch_manifest_json: Path,
    output_dir: Path,
    min_repeated_blocked: int = 2,
) -> dict[str, Any]:
    if min_repeated_blocked <= 0:
        raise ValueError("min_repeated_blocked must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / GAP_PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    progress(progress_path, "start", shadow_batch_manifest_json=str(shadow_batch_manifest_json), min_repeated_blocked=min_repeated_blocked)

    shadow_manifest = read_json(shadow_batch_manifest_json)
    outputs = shadow_manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get("items_jsonl"):
        raise ValueError("shadow batch manifest is missing outputs.items_jsonl")
    items_jsonl = Path(str(outputs["items_jsonl"]))
    observations = collect_route_observations(items_jsonl)
    route_rows = build_route_gap_rows(observations, min_repeated_blocked=min_repeated_blocked)
    summary = {
        "shadow_batch_manifest_json": str(shadow_batch_manifest_json),
        "items_jsonl": str(items_jsonl),
        "observation_count": len(observations),
        "route_count": len(route_rows),
        "allowed_route_count": sum(1 for row in route_rows if int(row["allowed_count"]) > 0),
        "blocked_route_count": sum(1 for row in route_rows if int(row["blocked_count"]) > 0),
        "repeated_blocked_route_count": sum(1 for row in route_rows if int(row["blocked_count"]) >= min_repeated_blocked),
        "missing_route_count": sum(1 for row in route_rows if int(row["missing_count"]) > 0),
        "next_wave_route_count": sum(1 for row in route_rows if bool(row["next_wave_required"])),
        "min_repeated_blocked": min_repeated_blocked,
    }
    status, reason, can_start_training = promotion_status(summary)
    fingerprint = stable_hash(
        {
            "shadow_batch_manifest_json": str(shadow_batch_manifest_json),
            "min_repeated_blocked": min_repeated_blocked,
            "routes": [
                {
                    "stem_type": row["stem_type"],
                    "route_id": row["route_id"],
                    "recommended_action": row["recommended_action"],
                    "observation_count": row["observation_count"],
                }
                for row in route_rows
            ],
        }
    )
    manifest = {
        "schema": "generic_controller_selector_route_gap_plan_manifest.v1",
        "manifest_type": "generic_controller_selector_route_gap_plan",
        "shadow_batch_manifest_json": str(shadow_batch_manifest_json),
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "summary": summary,
        "promotion": {
            "status": status,
            "reason": reason,
            "can_start_training": can_start_training,
            "can_edit_model_registry": False,
            "next_required_step": "generate_candidate_wave_or_collect_prior_evidence",
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "route_gap_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "safety": dict(SAFETY_FLAGS),
        "outputs": {
            "route_gaps_jsonl": str(output_dir / ROUTE_GAPS_NAME),
            "manifest_json": str(output_dir / GAP_MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
    }
    progress(progress_path, "write_outputs", **summary, status=status)
    write_jsonl(output_dir / ROUTE_GAPS_NAME, route_rows)
    write_json(output_dir / GAP_MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / GAP_MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan controller/selector route gaps from a shadow batch artifact.")
    parser.add_argument("--shadow-batch-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-repeated-blocked", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="plan_generic_controller_selector_route_gaps.py",
        entrypoint_path="/app/tools/plan_generic_controller_selector_route_gaps.py",
    )
    args = parse_args(argv)
    manifest = plan_route_gaps(
        shadow_batch_manifest_json=args.shadow_batch_manifest_json,
        output_dir=args.output_dir,
        min_repeated_blocked=int(args.min_repeated_blocked),
    )
    print(json.dumps({"summary": manifest["summary"], "promotion": manifest["promotion"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
