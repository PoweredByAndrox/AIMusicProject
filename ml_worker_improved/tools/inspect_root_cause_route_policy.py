"""Inspect root-cause route policy without running audio separation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_fully_automatic_stem_search import (
    MODEL_SPLIT_V2_PROFILE,
    _append_jsonl,
    _apply_root_cause_route_policy,
    _candidate_items,
    _parse_bool,
    _parse_stem_region_repair_checkpoint_spec,
    _repair_option_checkpoints,
    _write_json,
    build_default_from_scratch_recipe,
    build_model_split_v2_recipe,
)
from tools.stem_contamination_common import validate_no_artifact_rag_output


def _repair_option_count(candidates: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for candidate in candidates:
        options = candidate.get("model_split_composite_repair_options")
        if isinstance(options, list):
            count += sum(1 for option in options if isinstance(option, Mapping))
    return count


def _active_blocked_repair_options(
    candidates: Sequence[Mapping[str, Any]],
    *,
    blocked_repair_ids: set[str],
    blocked_repair_checkpoints: set[str],
) -> list[dict[str, str]]:
    active: list[dict[str, str]] = []
    for candidate in candidates:
        options = candidate.get("model_split_composite_repair_options")
        if not isinstance(options, list):
            continue
        for option in options:
            if not isinstance(option, Mapping):
                continue
            repair_id = str(option.get("repair_id") or "").strip()
            checkpoints = _repair_option_checkpoints(option)
            matched_checkpoint = next((checkpoint for checkpoint in sorted(checkpoints) if checkpoint in blocked_repair_checkpoints), "")
            if (repair_id and repair_id in blocked_repair_ids) or matched_checkpoint:
                active.append(
                    {
                        "candidate_id": str(candidate.get("candidate_id") or ""),
                        "repair_id": repair_id,
                        "matched_checkpoint": matched_checkpoint,
                    }
                )
    return active


def _load_recipe(
    *,
    recipe_path: Path | None,
    profile: str,
    demucs_available: bool | None,
    contamination_model_manifest: Path | str | None,
    stem_region_repair_checkpoints: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if recipe_path is not None:
        return json.loads(recipe_path.read_text(encoding="utf-8-sig"))
    if profile == MODEL_SPLIT_V2_PROFILE:
        return build_model_split_v2_recipe(
            demucs_available=demucs_available,
            contamination_model_manifest=contamination_model_manifest,
            stem_region_repair_checkpoints=stem_region_repair_checkpoints,
        )
    return build_default_from_scratch_recipe(
        demucs_available=demucs_available,
        contamination_model_manifest=contamination_model_manifest,
    )


def inspect_root_cause_route_policy(
    *,
    output_dir: Path,
    root_cause_queue_jsonl: Path | None,
    recipe_path: Path | None = None,
    profile: str = MODEL_SPLIT_V2_PROFILE,
    demucs_available: bool | None = None,
    contamination_model_manifest: Path | str | None = None,
    stem_region_repair_checkpoints: Sequence[Mapping[str, Any]] | None = None,
    no_artifact_rag: bool = True,
) -> dict[str, Any]:
    if no_artifact_rag:
        validate_no_artifact_rag_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    started = time.monotonic()
    _append_jsonl(
        progress_path,
        {
            "event": "start",
            "elapsed_sec": 0.0,
            "profile": profile or "default",
            "root_cause_queue_jsonl": str(root_cause_queue_jsonl or ""),
            "recipe_path": str(recipe_path or ""),
            "gpu_inference_skipped": True,
            "reason": "diagnostic_policy_inspection_only",
        },
    )

    recipe = _load_recipe(
        recipe_path=recipe_path,
        profile=profile,
        demucs_available=demucs_available,
        contamination_model_manifest=contamination_model_manifest,
        stem_region_repair_checkpoints=stem_region_repair_checkpoints,
    )
    candidates_before = _candidate_items(recipe)
    filtered_recipe = _apply_root_cause_route_policy(recipe, root_cause_queue_jsonl=root_cause_queue_jsonl)
    candidates_after = _candidate_items(filtered_recipe)
    policy = filtered_recipe.get("root_cause_route_policy") or {}
    blocked_repair_ids = {str(value) for value in policy.get("blocked_repair_ids") or []}
    blocked_repair_checkpoints = {str(value) for value in policy.get("blocked_repair_checkpoints") or []}
    active_blocked_repair_options = _active_blocked_repair_options(
        candidates_after,
        blocked_repair_ids=blocked_repair_ids,
        blocked_repair_checkpoints=blocked_repair_checkpoints,
    )
    manifest: dict[str, Any] = {
        "schema": "root_cause_route_policy_inspection.v1",
        "output_dir": str(output_dir),
        "profile": profile or str(recipe.get("profile") or "default"),
        "recipe_path": str(recipe_path or ""),
        "recipe_available": bool(recipe.get("available", True)),
        "root_cause_queue_jsonl": str(root_cause_queue_jsonl or ""),
        "root_cause_route_policy": policy,
        "candidate_count_before_policy": len(candidates_before),
        "candidate_count_after_policy": len(candidates_after),
        "repair_option_count_before_policy": _repair_option_count(candidates_before),
        "remaining_repair_option_count": _repair_option_count(candidates_after),
        "active_blocked_repair_option_count": len(active_blocked_repair_options),
        "active_blocked_repair_options": active_blocked_repair_options[:20],
        "candidate_ids_after_policy": [str(candidate.get("candidate_id") or "") for candidate in candidates_after],
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_policy_only",
        "listen_pack": {},
        "gpu_inference_skipped": True,
        "artifact_rag_write_allowed": not no_artifact_rag,
    }
    if recipe.get("available", True) is False:
        manifest["failed_checks"] = [str(recipe.get("unavailable_reason") or "recipe_unavailable")]
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "recipe_after_policy.json", filtered_recipe)
    _append_jsonl(
        progress_path,
        {
            "event": "end",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "candidate_count_before_policy": manifest["candidate_count_before_policy"],
            "candidate_count_after_policy": manifest["candidate_count_after_policy"],
            "skipped_candidate_count": int(policy.get("skipped_candidate_count") or 0),
            "skipped_repair_option_count": int(policy.get("skipped_repair_option_count") or 0),
            "active_blocked_repair_option_count": len(active_blocked_repair_options),
            "gpu_inference_skipped": True,
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root-cause-queue", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--profile", choices=("", MODEL_SPLIT_V2_PROFILE), default=MODEL_SPLIT_V2_PROFILE)
    parser.add_argument("--contamination-model-manifest")
    parser.add_argument("--demucs-available", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--no-artifact-rag", action="store_true")
    parser.add_argument(
        "--stem-region-repair-checkpoint",
        action="append",
        default=[],
        help="Learned repair checkpoint spec: stem:issue_type:checkpoint_path or JSON object",
    )
    parser.add_argument(
        "--echo-progress",
        dest="echo_progress",
        action="store_true",
        default=True,
        help="Mirror progress.jsonl events to stderr during CLI runs",
    )
    parser.add_argument(
        "--no-echo-progress",
        dest="echo_progress",
        action="store_false",
        help="Write progress only to progress.jsonl",
    )
    return parser.parse_args()


def _optional_bool(value: str) -> bool | None:
    if value == "auto":
        return None
    return _parse_bool(value)


def _manifest_cli_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    policy = manifest.get("root_cause_route_policy")
    if not isinstance(policy, Mapping):
        policy = {}
    return {
        "schema": manifest.get("schema"),
        "output_dir": manifest.get("output_dir"),
        "delivery_allowed": manifest.get("delivery_allowed"),
        "presentation_status": manifest.get("presentation_status"),
        "candidate_count_before_policy": manifest.get("candidate_count_before_policy"),
        "candidate_count_after_policy": manifest.get("candidate_count_after_policy"),
        "repair_option_count_before_policy": manifest.get("repair_option_count_before_policy"),
        "remaining_repair_option_count": manifest.get("remaining_repair_option_count"),
        "active_blocked_repair_option_count": manifest.get("active_blocked_repair_option_count"),
        "skipped_candidate_count": policy.get("skipped_candidate_count", 0),
        "skipped_repair_option_count": policy.get("skipped_repair_option_count", 0),
        "blocked_repair_id_count": len(policy.get("blocked_repair_ids") or []),
        "blocked_repair_checkpoint_count": len(policy.get("blocked_repair_checkpoints") or []),
        "manifest_path": str(Path(str(manifest.get("output_dir") or "")) / "manifest.json"),
    }


def main() -> None:
    args = parse_args()
    os.environ["STEM_SEARCH_ECHO_PROGRESS"] = "1" if args.echo_progress else "0"
    stem_region_repair_checkpoints = [
        _parse_stem_region_repair_checkpoint_spec(spec, index=index)
        for index, spec in enumerate(args.stem_region_repair_checkpoint or [], start=1)
    ]
    manifest = inspect_root_cause_route_policy(
        output_dir=args.out,
        root_cause_queue_jsonl=args.root_cause_queue,
        recipe_path=args.recipe,
        profile=args.profile,
        demucs_available=_optional_bool(args.demucs_available),
        contamination_model_manifest=args.contamination_model_manifest,
        stem_region_repair_checkpoints=stem_region_repair_checkpoints,
        no_artifact_rag=args.no_artifact_rag,
    )
    print(json.dumps(_manifest_cli_summary(manifest), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
