#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_generic_controller_external_eval_from_hard_cases import (  # noqa: E402
    DEFAULT_SPLITS,
    external_eval_example,
    failure_labels,
    iter_jsonl,
    row_split,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.train_generic_finetuning_controller_model import stable_hash  # noqa: E402


TRAIN_EXAMPLES_NAME = "augmented_controller_train_examples.jsonl"
EVAL_EXAMPLES_NAME = "hard_case_holdout_external_eval_examples.jsonl"
SKIPPED_NAME = "hard_case_train_eval_split_skipped.jsonl"
MANIFEST_NAME = "generic_controller_hard_case_train_eval_split_manifest.json"
PROGRESS_NAME = "generic_controller_hard_case_train_eval_split_progress.jsonl"

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
        "gpu_inference_skip_reason": "controller_hard_case_split_uses_existing_metadata_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_split",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def example_manifest_path(row: dict[str, Any]) -> str:
    input_payload = row.get("input")
    if isinstance(input_payload, dict) and input_payload.get("manifest_path"):
        return str(input_payload["manifest_path"])
    return str(row.get("manifest_path") or "")


def row_route(row: dict[str, Any]) -> str:
    return str(row.get("model_id") or "").strip()


def skipped_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    source_id = str(row.get("id") or "")
    manifest_path = str(row.get("manifest_path") or "")
    return {
        "schema": "generic_controller_hard_case_train_eval_split_skipped.v1",
        "id": stable_hash(["controller_hard_case_split_skipped", source_id, manifest_path, reason])[:16],
        "source_hard_eval_case_id": source_id,
        "track_id": str(row.get("track_id") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "manifest_path": manifest_path,
        "model_id": row_route(row),
        "source_split": row_split(row),
        "skip_reason": reason,
        "trainable": False,
    }


def hard_case_controller_example(row: dict[str, Any], *, route_id: str, split_role: str) -> dict[str, Any]:
    example = external_eval_example(row, route_id=route_id)
    example["id"] = stable_hash(["controller_hard_case", split_role, row.get("id"), row.get("manifest_path"), route_id])[:16]
    example["input"]["hard_case_split_role"] = split_role
    example["input"]["failure_labels"] = failure_labels(row)
    if split_role == "train":
        example["training_target"] = "llm_controller_hard_case_train"
        example["eval_only"] = False
    else:
        example["training_target"] = "llm_controller_external_eval"
        example["eval_only"] = True
    return example


def split_route_rows(rows: Sequence[dict[str, Any]], *, min_route_examples_for_holdout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (str(row.get("id") or ""), str(row.get("manifest_path") or "")))
    if len(ordered) < min_route_examples_for_holdout:
        return list(ordered), []
    return ordered[1:], [ordered[0]]


def build_hard_case_train_eval_split(
    *,
    base_controller_examples_jsonl: Path,
    hard_eval_jsonl: Path,
    output_dir: Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
    min_route_examples_for_holdout: int = 2,
) -> dict[str, Any]:
    if min_route_examples_for_holdout < 2:
        raise ValueError("min_route_examples_for_holdout must be >= 2")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(
        progress_path,
        "start",
        base_controller_examples_jsonl=str(base_controller_examples_jsonl),
        hard_eval_jsonl=str(hard_eval_jsonl),
        splits=list(splits),
        min_route_examples_for_holdout=min_route_examples_for_holdout,
    )
    base_rows = list(iter_jsonl(base_controller_examples_jsonl))
    base_manifest_paths = {example_manifest_path(row) for row in base_rows if example_manifest_path(row)}

    skipped_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_rows = list(iter_jsonl(hard_eval_jsonl))
    enabled_splits = tuple(str(split) for split in splits)
    for row in source_rows:
        if enabled_splits and row_split(row) not in enabled_splits:
            skipped_rows.append(skipped_row(row, reason="split_not_enabled"))
            continue
        route_id = row_route(row)
        if not route_id:
            skipped_rows.append(skipped_row(row, reason="missing_model_id"))
            continue
        manifest_path = str(row.get("manifest_path") or "")
        if manifest_path and manifest_path in base_manifest_paths:
            skipped_rows.append(skipped_row(row, reason="base_controller_manifest_overlap"))
            continue
        grouped[(str(row.get("stem_type") or "unknown"), route_id)].append(row)

    hard_train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for (_stem, route_id), rows in sorted(grouped.items()):
        train_source, holdout_source = split_route_rows(rows, min_route_examples_for_holdout=min_route_examples_for_holdout)
        hard_train_rows.extend(hard_case_controller_example(row, route_id=route_id, split_role="train") for row in train_source)
        holdout_rows.extend(hard_case_controller_example(row, route_id=route_id, split_role="holdout_eval") for row in holdout_source)

    augmented_train_rows = [*base_rows, *hard_train_rows]
    train_paths = {example_manifest_path(row) for row in augmented_train_rows if example_manifest_path(row)}
    eval_paths = {example_manifest_path(row) for row in holdout_rows if example_manifest_path(row)}
    overlap_paths = sorted(train_paths & eval_paths)
    skipped_reasons = Counter(str(row["skip_reason"]) for row in skipped_rows)
    holdout_routes = {
        (str(row.get("stem_type") or "unknown"), str(row.get("input", {}).get("model_id") or ""))
        for row in holdout_rows
    }
    train_routes = {
        (str(row.get("stem_type") or "unknown"), str(row.get("input", {}).get("model_id") or ""))
        for row in augmented_train_rows
    }
    missing_holdout_routes = sorted(f"{stem}:{route}" for stem, route in holdout_routes if (stem, route) not in train_routes)
    fingerprint = stable_hash(
        {
            "base_controller_examples_jsonl": str(base_controller_examples_jsonl),
            "hard_eval_jsonl": str(hard_eval_jsonl),
            "splits": list(enabled_splits),
            "min_route_examples_for_holdout": min_route_examples_for_holdout,
            "hard_train_ids": [row.get("id") for row in hard_train_rows],
            "holdout_ids": [row.get("id") for row in holdout_rows],
        }
    )
    summary = {
        "source_rows": len(source_rows),
        "base_controller_examples": len(base_rows),
        "hard_case_route_groups": len(grouped),
        "hard_case_train_examples": len(hard_train_rows),
        "augmented_controller_examples": len(augmented_train_rows),
        "holdout_eval_examples": len(holdout_rows),
        "skipped_rows": len(skipped_rows),
        "skipped_no_route": int(skipped_reasons.get("missing_model_id", 0)),
        "skipped_split": int(skipped_reasons.get("split_not_enabled", 0)),
        "skipped_base_overlap": int(skipped_reasons.get("base_controller_manifest_overlap", 0)),
        "train_eval_manifest_overlap_count": len(overlap_paths),
        "holdout_routes_missing_from_train": len(missing_holdout_routes),
    }
    progress(progress_path, "write_outputs", **summary)
    write_jsonl(output_dir / TRAIN_EXAMPLES_NAME, augmented_train_rows)
    write_jsonl(output_dir / EVAL_EXAMPLES_NAME, holdout_rows)
    write_jsonl(output_dir / SKIPPED_NAME, skipped_rows)

    manifest = {
        "schema": "generic_controller_hard_case_train_eval_split_manifest.v1",
        "manifest_type": "generic_controller_hard_case_train_eval_split",
        "base_controller_examples_jsonl": str(base_controller_examples_jsonl),
        "hard_eval_jsonl": str(hard_eval_jsonl),
        "output_dir": str(output_dir),
        "splits": list(enabled_splits),
        "min_route_examples_for_holdout": min_route_examples_for_holdout,
        "fingerprint": fingerprint,
        "summary": summary,
        "quality_gates": {
            "train_eval_manifest_overlap_paths": overlap_paths[:20],
            "holdout_routes_missing_from_train": missing_holdout_routes[:20],
            "can_use_holdout_for_external_eval": not overlap_paths and not missing_holdout_routes and bool(holdout_rows),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "controller_hard_case_split_not_for_listen_pack",
        "release_status": "training_eval_dataset_artifact_only",
        "release_ready": False,
        "safety": {
            **SAFETY_FLAGS,
            "metadata_only": True,
            "not_release_artifact": True,
            "disjoint_train_eval_required": True,
            "unknown_route_default_skipped": True,
        },
        "outputs": {
            "augmented_controller_examples_jsonl": str(output_dir / TRAIN_EXAMPLES_NAME),
            "holdout_eval_examples_jsonl": str(output_dir / EVAL_EXAMPLES_NAME),
            "skipped_jsonl": str(output_dir / SKIPPED_NAME),
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disjoint controller train/holdout examples from hard eval cases.")
    parser.add_argument("--base-controller-examples-jsonl", type=Path, required=True)
    parser.add_argument("--hard-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", dest="splits", help="Source split to include. Defaults to hard_eval.")
    parser.add_argument("--min-route-examples-for-holdout", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_generic_controller_hard_case_train_eval_split.py",
        entrypoint_path="/app/tools/build_generic_controller_hard_case_train_eval_split.py",
    )
    args = parse_args(argv)
    manifest = build_hard_case_train_eval_split(
        base_controller_examples_jsonl=args.base_controller_examples_jsonl,
        hard_eval_jsonl=args.hard_eval_jsonl,
        output_dir=args.output_dir,
        splits=tuple(args.splits) if args.splits else DEFAULT_SPLITS,
        min_route_examples_for_holdout=int(args.min_route_examples_for_holdout),
    )
    print(json.dumps({"summary": manifest["summary"], "quality_gates": manifest["quality_gates"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
