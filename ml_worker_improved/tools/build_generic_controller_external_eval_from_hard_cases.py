#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.train_generic_finetuning_controller_model import stable_hash  # noqa: E402


EXAMPLES_NAME = "generic_controller_external_eval_examples.jsonl"
SKIPPED_NAME = "generic_controller_external_eval_skipped.jsonl"
MANIFEST_NAME = "generic_controller_external_eval_manifest.json"
PROGRESS_NAME = "generic_controller_external_eval_progress.jsonl"
DEFAULT_SPLITS = ("hard_eval",)
UNKNOWN_ROUTE = "unknown_route"

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
        "gpu_inference_skip_reason": "controller_external_eval_builder_uses_existing_metadata_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_eval_builder",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def row_split(row: dict[str, Any]) -> str:
    return str(row.get("split") or "hard_eval")


def row_route(row: dict[str, Any], *, include_unknown_route: bool) -> str:
    route = str(row.get("model_id") or "").strip()
    if route:
        return route
    return UNKNOWN_ROUTE if include_unknown_route else ""


def failure_labels(row: dict[str, Any]) -> list[str]:
    labels = [str(item).strip() for item in as_list(row.get("failure_labels")) if str(item).strip()]
    if labels:
        return labels
    reason = str(row.get("dominant_failure_reason") or "").strip()
    return [reason] if reason else []


def external_eval_example(row: dict[str, Any], *, route_id: str) -> dict[str, Any]:
    source_id = str(row.get("id") or "")
    manifest_path = str(row.get("manifest_path") or "")
    labels = failure_labels(row)
    return {
        "schema": "generic_finetuning_controller_decision_example.v1",
        "id": stable_hash(["controller_external_eval", source_id, manifest_path, route_id])[:16],
        "track_id": str(row.get("track_id") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "input": {
            "manifest_path": manifest_path,
            "model_id": route_id,
            "original_model_id": str(row.get("model_id") or ""),
            "delivery_status": "blocked",
            "presentation_status": "blocked_from_listen_pack",
            "release_status": "external_eval_artifact_only",
            "failure_labels": labels,
            "dominant_failure_reason": str(row.get("dominant_failure_reason") or (labels[0] if labels else "")),
            "source_hard_eval_case_id": source_id,
            "source_split": row_split(row),
            "source_schema": str(row.get("schema") or ""),
        },
        "target": {
            "action": "avoid_route_and_generate_alternative",
            "chosen_model": route_id,
            "acceptance_tests": [
                "artifact_status_contract",
                "external_eval_replay",
                "training_overlap_guard",
            ],
        },
        "result": {
            "accepted": False,
            "human_verified": bool(row.get("human_verified")),
            "human_rating": row.get("human_rating") if row.get("human_rating") is not None else "",
            "reference_verified": False,
        },
        "trainable": True,
        "training_target": "llm_controller_external_eval",
        "eval_only": True,
    }


def skipped_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    source_id = str(row.get("id") or "")
    manifest_path = str(row.get("manifest_path") or "")
    return {
        "schema": "generic_controller_external_eval_skipped.v1",
        "id": stable_hash(["controller_external_eval_skipped", source_id, manifest_path, reason])[:16],
        "source_hard_eval_case_id": source_id,
        "track_id": str(row.get("track_id") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "manifest_path": manifest_path,
        "model_id": str(row.get("model_id") or ""),
        "source_split": row_split(row),
        "skip_reason": reason,
        "trainable": False,
    }


def build_external_eval_set(
    *,
    hard_eval_jsonl: Path,
    output_dir: Path,
    include_unknown_route: bool = False,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    enabled_splits = tuple(str(split) for split in splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(
        progress_path,
        "start",
        hard_eval_jsonl=str(hard_eval_jsonl),
        include_unknown_route=include_unknown_route,
        splits=list(enabled_splits),
    )

    source_rows = list(iter_jsonl(hard_eval_jsonl))
    eval_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for row in source_rows:
        if enabled_splits and row_split(row) not in enabled_splits:
            skipped_rows.append(skipped_row(row, reason="split_not_enabled"))
            continue
        route_id = row_route(row, include_unknown_route=include_unknown_route)
        if not route_id:
            skipped_rows.append(skipped_row(row, reason="missing_model_id"))
            continue
        eval_rows.append(external_eval_example(row, route_id=route_id))

    skipped_reasons = Counter(str(row["skip_reason"]) for row in skipped_rows)
    stem_counts = Counter(str(row.get("stem_type") or "unknown") for row in eval_rows)
    split_counts = Counter(str(row.get("input", {}).get("source_split") or "") for row in eval_rows)
    route_counts = Counter(str(row.get("input", {}).get("model_id") or "") for row in eval_rows)
    fingerprint = stable_hash(
        {
            "source": str(hard_eval_jsonl),
            "include_unknown_route": include_unknown_route,
            "splits": list(enabled_splits),
            "examples": [
                {
                    "id": row.get("id"),
                    "stem_type": row.get("stem_type"),
                    "route": row.get("input", {}).get("model_id"),
                    "accepted": row.get("result", {}).get("accepted"),
                    "failure_labels": row.get("input", {}).get("failure_labels"),
                }
                for row in eval_rows
            ],
        }
    )
    summary = {
        "source_rows": len(source_rows),
        "eval_examples": len(eval_rows),
        "skipped_rows": len(skipped_rows),
        "skipped_no_route": int(skipped_reasons.get("missing_model_id", 0)),
        "skipped_split": int(skipped_reasons.get("split_not_enabled", 0)),
        "unknown_route_examples": int(route_counts.get(UNKNOWN_ROUTE, 0)),
        "stem_counts": dict(sorted(stem_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "route_count": len(route_counts),
    }
    progress(progress_path, "write_outputs", **summary)
    write_jsonl(output_dir / EXAMPLES_NAME, eval_rows)
    write_jsonl(output_dir / SKIPPED_NAME, skipped_rows)

    manifest = {
        "schema": "generic_controller_external_eval_set_manifest.v1",
        "manifest_type": "generic_controller_external_eval_set",
        "hard_eval_jsonl": str(hard_eval_jsonl),
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "include_unknown_route": include_unknown_route,
        "splits": list(enabled_splits),
        "summary": summary,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "controller_external_eval_set_not_for_listen_pack",
        "release_status": "external_eval_artifact_only",
        "release_ready": False,
        "safety": {
            **SAFETY_FLAGS,
            "metadata_only": True,
            "not_release_artifact": True,
            "requires_external_eval_runner": True,
            "requires_training_overlap_guard": True,
            "unknown_route_default_skipped": not include_unknown_route,
        },
        "outputs": {
            "eval_examples_jsonl": str(output_dir / EXAMPLES_NAME),
            "skipped_jsonl": str(output_dir / SKIPPED_NAME),
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controller external eval examples from generic hard eval cases.")
    parser.add_argument("--hard-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-unknown-route", action="store_true")
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Source split to include. Defaults to hard_eval. Repeat to include multiple splits.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_generic_controller_external_eval_from_hard_cases.py",
        entrypoint_path="/app/tools/build_generic_controller_external_eval_from_hard_cases.py",
    )
    args = parse_args(argv)
    manifest = build_external_eval_set(
        hard_eval_jsonl=args.hard_eval_jsonl,
        output_dir=args.output_dir,
        include_unknown_route=bool(args.include_unknown_route),
        splits=tuple(args.splits) if args.splits else DEFAULT_SPLITS,
    )
    print(json.dumps({"summary": manifest["summary"], "outputs": manifest["outputs"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
