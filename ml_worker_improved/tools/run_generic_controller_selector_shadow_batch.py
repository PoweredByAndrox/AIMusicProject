#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.no_reference_score_matrix import validate_mode  # noqa: E402
from tools.stage_generic_finetuning_controller_with_no_reference_selector import (  # noqa: E402
    stage_controller_with_selector,
)
from tools.train_generic_finetuning_controller_model import stable_hash  # noqa: E402


BATCH_MANIFEST_NAME = "generic_controller_selector_shadow_batch_manifest.json"
BATCH_ITEMS_NAME = "generic_controller_selector_shadow_batch_items.jsonl"
BATCH_PROGRESS_NAME = "generic_controller_selector_shadow_batch_progress.jsonl"

START_TIME = time.monotonic()

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
    "metadata_only": True,
    "not_release_artifact": True,
    "registry_edit_blocked": True,
    "audio_inference_started": False,
    "audio_render_started": False,
}


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
        "gpu_inference_skip_reason": "controller_selector_shadow_batch_uses_existing_metadata_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_shadow_batch",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def discover_judge_jsonls(judge_root: Path, judge_glob: str) -> list[Path]:
    if not judge_root.exists():
        raise FileNotFoundError(judge_root)
    return sorted((path for path in judge_root.glob(judge_glob) if path.is_file()), key=lambda path: path.as_posix())


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:80] or "judge"


def single_case_output_dir(output_dir: Path, judge_jsonl: Path, index: int) -> Path:
    label = safe_slug(f"{judge_jsonl.parent.name}_{judge_jsonl.stem}")
    digest = stable_hash(str(judge_jsonl))[:12]
    return output_dir / "single_cases" / f"{index:04d}_{label}_{digest}"


def item_row(index: int, judge_jsonl: Path, single_manifest: dict[str, Any]) -> dict[str, Any]:
    integration = single_manifest["integration_summary"]
    promotion = single_manifest["promotion"]
    outputs = single_manifest["outputs"]
    return {
        "schema": "generic_controller_selector_shadow_batch_item.v1",
        "id": stable_hash(["controller_selector_shadow_batch_item", str(judge_jsonl), single_manifest["fingerprint"]])[:16],
        "index": index,
        "judge_jsonl": str(judge_jsonl),
        "single_output_dir": str(single_manifest["output_dir"]),
        "single_manifest_json": str(outputs["manifest_json"]),
        "single_promotion_status": str(promotion["status"]),
        "single_can_stage_controller_candidate": bool(promotion["can_stage_controller_candidate"]),
        "selector_usable_tracks": int(integration.get("selector_usable_tracks") or 0),
        "controller_checked_candidates": int(integration.get("controller_checked_candidates") or 0),
        "controller_allowed_candidates": int(integration.get("controller_allowed_candidates") or 0),
        "controller_reranked_candidates": int(integration.get("controller_reranked_candidates") or 0),
        "controller_blocked_candidates": int(integration.get("controller_blocked_candidates") or 0),
        "hard_case_policy_blocked_candidates": int(integration.get("hard_case_policy_blocked_candidates") or 0),
        "hard_case_route_policy_applied": bool(integration.get("hard_case_route_policy_applied")),
        "hard_case_training_objective_required": bool(integration.get("hard_case_training_objective_required")),
        "hard_case_artifact_safe_route_required": bool(integration.get("hard_case_artifact_safe_route_required")),
        "controller_missing_routes": int(integration.get("controller_missing_routes") or 0),
        "controller_override_count": int(integration.get("controller_override_count") or 0),
        "delivery_allowed": False,
        "release_status": "shadow_batch_staging_only",
        "presentation_status": "controller_selector_shadow_batch_not_for_listen_pack",
    }


def build_batch_summary(item_rows: Sequence[dict[str, Any]], min_batch_checked_candidates: int) -> dict[str, Any]:
    status_counts = Counter(str(row["single_promotion_status"]) for row in item_rows)
    return {
        "input_count": len(item_rows),
        "min_batch_checked_candidates": int(min_batch_checked_candidates),
        "status_counts": dict(sorted(status_counts.items())),
        "staging_ready_inputs": sum(1 for row in item_rows if row["single_can_stage_controller_candidate"]),
        "blocked_inputs": sum(1 for row in item_rows if not row["single_can_stage_controller_candidate"]),
        "selector_usable_tracks": sum(int(row["selector_usable_tracks"]) for row in item_rows),
        "controller_checked_candidates": sum(int(row["controller_checked_candidates"]) for row in item_rows),
        "controller_allowed_candidates": sum(int(row["controller_allowed_candidates"]) for row in item_rows),
        "controller_reranked_candidates": sum(int(row["controller_reranked_candidates"]) for row in item_rows),
        "controller_blocked_candidates": sum(int(row["controller_blocked_candidates"]) for row in item_rows),
        "hard_case_policy_blocked_candidates": sum(int(row["hard_case_policy_blocked_candidates"]) for row in item_rows),
        "hard_case_route_policy_applied": any(bool(row["hard_case_route_policy_applied"]) for row in item_rows),
        "hard_case_training_objective_required": any(bool(row["hard_case_training_objective_required"]) for row in item_rows),
        "hard_case_artifact_safe_route_required": any(bool(row["hard_case_artifact_safe_route_required"]) for row in item_rows),
        "controller_missing_routes": sum(int(row["controller_missing_routes"]) for row in item_rows),
        "controller_override_count": sum(int(row["controller_override_count"]) for row in item_rows),
    }


def promotion_status(summary: dict[str, Any]) -> tuple[str, str, bool]:
    if int(summary.get("input_count") or 0) <= 0:
        return ("blocked_no_judge_inputs", "no judge JSONL inputs were provided or discovered", False)
    if summary.get("hard_case_training_objective_required") is True:
        return (
            "blocked_hard_case_requires_separator_training_objective",
            "hard-case policy requires separator training objective before shadow-batch staging",
            False,
        )
    if int(summary.get("hard_case_policy_blocked_candidates") or 0) > 0:
        return (
            "blocked_hard_case_policy_route_mismatch",
            "hard-case policy blocked at least one selector route outside the allowed artifact-safe route set",
            False,
        )
    if int(summary.get("controller_checked_candidates") or 0) < int(summary.get("min_batch_checked_candidates") or 0):
        return ("blocked_insufficient_controller_checked_candidates", "not enough controller-checked candidates", False)
    if int(summary.get("controller_missing_routes") or 0) > 0:
        return ("blocked_missing_controller_routes", "at least one selector-chosen route is missing from controller model", False)
    return ("controller_selector_shadow_batch_staged", "shadow batch is ready for controlled policy review", True)


def run_shadow_batch(
    *,
    controller_model_json: Path,
    judge_jsonls: Sequence[Path],
    output_dir: Path,
    score_mode: str = "both",
    min_selector_usable_per_input: int = 1,
    min_batch_checked_candidates: int = 1,
    accept_threshold: float = 0.5,
    hard_case_route_policy_json: Path | None = None,
) -> dict[str, Any]:
    if min_selector_usable_per_input <= 0:
        raise ValueError("min_selector_usable_per_input must be > 0")
    if min_batch_checked_candidates <= 0:
        raise ValueError("min_batch_checked_candidates must be > 0")
    if not 0.0 <= accept_threshold <= 1.0:
        raise ValueError("accept_threshold must be between 0 and 1")
    score_mode = validate_mode(score_mode)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / BATCH_PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    judge_inputs = dedupe_paths(judge_jsonls)
    progress(
        progress_path,
        "start",
        controller_model_json=str(controller_model_json),
        hard_case_route_policy_json=str(hard_case_route_policy_json or ""),
        input_count=len(judge_inputs),
        score_mode=score_mode,
        min_selector_usable_per_input=min_selector_usable_per_input,
        min_batch_checked_candidates=min_batch_checked_candidates,
        accept_threshold=accept_threshold,
    )

    items: list[dict[str, Any]] = []
    for index, judge_jsonl in enumerate(judge_inputs, start=1):
        single_output = single_case_output_dir(output_dir, judge_jsonl, index)
        progress(progress_path, "input_start", index=index, judge_jsonl=str(judge_jsonl), single_output_dir=str(single_output))
        single_manifest = stage_controller_with_selector(
            controller_model_json=controller_model_json,
            judge_jsonl=judge_jsonl,
            output_dir=single_output,
            score_mode=score_mode,
            min_selector_usable=min_selector_usable_per_input,
            accept_threshold=accept_threshold,
            hard_case_route_policy_json=hard_case_route_policy_json,
        )
        item = item_row(index, judge_jsonl, single_manifest)
        items.append(item)
        progress(
            progress_path,
            "input_finish",
            index=index,
            judge_jsonl=str(judge_jsonl),
            single_promotion_status=item["single_promotion_status"],
            controller_checked_candidates=item["controller_checked_candidates"],
            controller_missing_routes=item["controller_missing_routes"],
            hard_case_policy_blocked_candidates=item["hard_case_policy_blocked_candidates"],
        )

    summary = build_batch_summary(items, min_batch_checked_candidates)
    status, reason, can_stage = promotion_status(summary)
    fingerprint = stable_hash(
        {
            "controller_model_json": str(controller_model_json),
            "hard_case_route_policy_json": str(hard_case_route_policy_json or ""),
            "judge_jsonls": [str(path) for path in judge_inputs],
            "score_mode": score_mode,
            "min_selector_usable_per_input": min_selector_usable_per_input,
            "min_batch_checked_candidates": min_batch_checked_candidates,
            "accept_threshold": accept_threshold,
            "items": [
                {
                    "judge_jsonl": row["judge_jsonl"],
                    "single_promotion_status": row["single_promotion_status"],
                    "controller_checked_candidates": row["controller_checked_candidates"],
                    "controller_missing_routes": row["controller_missing_routes"],
                }
                for row in items
            ],
        }
    )
    manifest = {
        "schema": "generic_controller_selector_shadow_batch_manifest.v1",
        "manifest_type": "generic_controller_selector_shadow_batch",
        "controller_model_json": str(controller_model_json),
        "hard_case_route_policy_json": str(hard_case_route_policy_json or ""),
        "judge_jsonls": [str(path) for path in judge_inputs],
        "output_dir": str(output_dir),
        "score_mode": score_mode,
        "fingerprint": fingerprint,
        "batch_summary": summary,
        "promotion": {
            "status": status,
            "reason": reason,
            "can_stage_controller_candidate": can_stage,
            "can_edit_model_registry": False,
            "next_required_step": "policy_review_or_shadow_batch_expansion" if can_stage else "fix_shadow_batch_blockers",
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "controller_selector_shadow_batch_not_for_listen_pack",
        "release_status": "shadow_batch_staging_only",
        "release_ready": False,
        "safety": dict(SAFETY_FLAGS),
        "outputs": {
            "manifest_json": str(output_dir / BATCH_MANIFEST_NAME),
            "items_jsonl": str(output_dir / BATCH_ITEMS_NAME),
            "progress_jsonl": str(progress_path),
            "single_cases_dir": str(output_dir / "single_cases"),
        },
    }
    progress(progress_path, "write_outputs", **summary, status=status)
    write_jsonl(output_dir / BATCH_ITEMS_NAME, items)
    write_json(output_dir / BATCH_MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / BATCH_MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a metadata-only controller/selector shadow batch.")
    parser.add_argument("--controller-model-json", type=Path, required=True)
    parser.add_argument("--hard-case-route-policy-json", type=Path, default=None)
    parser.add_argument("--judge-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--judge-root", type=Path)
    parser.add_argument("--judge-glob", default="**/no_reference_*_windowed_judge_results.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-mode", choices=["minus", "vocal", "both"], default="both")
    parser.add_argument("--min-selector-usable-per-input", type=int, default=1)
    parser.add_argument("--min-batch-checked-candidates", type=int, default=1)
    parser.add_argument("--accept-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def collect_judge_jsonls(args: argparse.Namespace) -> list[Path]:
    paths = list(args.judge_jsonl or [])
    if args.judge_root is not None:
        paths.extend(discover_judge_jsonls(Path(args.judge_root), str(args.judge_glob)))
    return dedupe_paths(paths)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_generic_controller_selector_shadow_batch.py",
        entrypoint_path="/app/tools/run_generic_controller_selector_shadow_batch.py",
    )
    args = parse_args(argv)
    manifest = run_shadow_batch(
        controller_model_json=args.controller_model_json,
        hard_case_route_policy_json=args.hard_case_route_policy_json,
        judge_jsonls=collect_judge_jsonls(args),
        output_dir=args.output_dir,
        score_mode=args.score_mode,
        min_selector_usable_per_input=int(args.min_selector_usable_per_input),
        min_batch_checked_candidates=int(args.min_batch_checked_candidates),
        accept_threshold=float(args.accept_threshold),
    )
    print(json.dumps({"batch_summary": manifest["batch_summary"], "promotion": manifest["promotion"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
