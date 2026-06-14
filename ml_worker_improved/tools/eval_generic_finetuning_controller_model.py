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
from tools.train_generic_finetuning_controller_model import (  # noqa: E402
    evaluate_model,
    example_accepted,
    example_failure_labels,
    example_route,
    example_stem,
    iter_jsonl,
    stable_hash,
)


MANIFEST_NAME = "generic_finetuning_controller_external_eval_manifest.json"
REPORT_NAME = "generic_finetuning_controller_external_eval_report.json"
PROGRESS_NAME = "generic_finetuning_controller_external_eval_progress.jsonl"

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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "controller_external_eval_uses_existing_metadata_examples",
        "expected_acceleration_mode": "external_only_cpu_metadata_eval",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def trainable_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("trainable") is not False]


def examples_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    return stable_hash(
        [
            {
                "id": row.get("id"),
                "track_id": row.get("track_id"),
                "stem_type": row.get("stem_type"),
                "target": row.get("target"),
                "result": row.get("result"),
            }
            for row in rows
        ]
    )


def training_style_fingerprint(rows: Sequence[dict[str, Any]], hyperparameters: dict[str, Any]) -> str:
    return stable_hash(
        {
            "examples": [
                {
                    "id": row.get("id"),
                    "stem": example_stem(row),
                    "route": example_route(row),
                    "accepted": example_accepted(row),
                    "failure_labels": example_failure_labels(row),
                }
                for row in sorted(rows, key=lambda item: str(item.get("id") or ""))
            ],
            "hyperparameters": hyperparameters,
        }
    )


def same_training_examples(
    eval_examples_jsonl: Path,
    training_manifest: dict[str, Any] | None,
    eval_rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
) -> bool:
    if not training_manifest:
        return False
    train_path = str(training_manifest.get("controller_examples_jsonl") or "")
    try:
        same_path = bool(train_path) and Path(train_path).resolve() == eval_examples_jsonl.resolve()
    except OSError:
        same_path = bool(train_path) and str(Path(train_path)) == str(eval_examples_jsonl)
    if same_path:
        return True
    hyperparameters = model.get("hyperparameters") if isinstance(model.get("hyperparameters"), dict) else {}
    return training_style_fingerprint(eval_rows, hyperparameters) == str(training_manifest.get("fingerprint") or "")


def example_manifest_path(row: dict[str, Any]) -> str:
    input_payload = row.get("input")
    if isinstance(input_payload, dict) and input_payload.get("manifest_path"):
        return str(input_payload["manifest_path"])
    return str(row.get("manifest_path") or "")


def training_overlap_report(
    *,
    eval_rows: Sequence[dict[str, Any]],
    training_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    if not training_manifest:
        return {
            "checked": False,
            "blocker": False,
            "reason": "training_manifest_not_provided",
            "overlap_count": 0,
            "overlap_manifest_paths": [],
        }
    train_path_text = str(training_manifest.get("controller_examples_jsonl") or "")
    if not train_path_text:
        return {
            "checked": False,
            "blocker": True,
            "reason": "training_manifest_missing_controller_examples_jsonl",
            "overlap_count": 0,
            "overlap_manifest_paths": [],
        }
    train_path = Path(train_path_text)
    if not train_path.exists():
        return {
            "checked": False,
            "blocker": True,
            "reason": "training_examples_jsonl_not_readable",
            "training_examples_jsonl": train_path_text,
            "overlap_count": 0,
            "overlap_manifest_paths": [],
        }

    training_rows = trainable_rows(iter_jsonl(train_path))
    training_paths = {example_manifest_path(row) for row in training_rows if example_manifest_path(row)}
    eval_paths = [example_manifest_path(row) for row in eval_rows if example_manifest_path(row)]
    overlap_paths = sorted({path for path in eval_paths if path in training_paths})
    return {
        "checked": True,
        "blocker": False,
        "reason": "checked_manifest_path_overlap",
        "training_examples_jsonl": train_path_text,
        "training_manifest_path_count": len(training_paths),
        "eval_manifest_path_count": len(set(eval_paths)),
        "overlap_count": len(overlap_paths),
        "overlap_ratio": round(len(overlap_paths) / len(set(eval_paths)), 6) if eval_paths else 0.0,
        "overlap_manifest_paths": overlap_paths[:20],
    }


def promotion_status(
    *,
    eval_report: dict[str, Any],
    same_training_data: bool,
    training_overlap: dict[str, Any],
    min_eval_examples: int,
    min_accuracy: float,
) -> tuple[str, str, bool]:
    evaluated = int(eval_report.get("evaluated_examples") or 0)
    missing = int(eval_report.get("missing_route_examples") or 0)
    accuracy = float(eval_report.get("accuracy") or 0.0)
    if same_training_data:
        return (
            "blocked_training_data_eval_only",
            "external eval cannot reuse the controller training dataset",
            False,
        )
    if bool(training_overlap.get("blocker")):
        return (
            "blocked_training_overlap_unchecked",
            str(training_overlap.get("reason") or "training overlap could not be checked"),
            False,
        )
    if int(training_overlap.get("overlap_count") or 0) > 0:
        return (
            "blocked_training_overlap_eval_only",
            "external eval examples overlap with controller training manifests",
            False,
        )
    if evaluated < min_eval_examples:
        return (
            "blocked_insufficient_eval_examples",
            "not enough external eval examples",
            False,
        )
    if missing:
        return (
            "blocked_missing_routes",
            "external eval contains routes unseen by the controller model",
            False,
        )
    if accuracy < min_accuracy:
        return (
            "blocked_accuracy_below_threshold",
            "external eval accuracy is below promotion threshold",
            False,
        )
    return ("external_eval_passed", "external eval passed metadata replay gates", True)


def eval_controller_model(
    *,
    controller_model_json: Path,
    eval_examples_jsonl: Path,
    output_dir: Path,
    training_manifest_json: Path | None = None,
    min_eval_examples: int = 20,
    min_accuracy: float = 0.7,
) -> dict[str, Any]:
    if min_eval_examples <= 0:
        raise ValueError("min_eval_examples must be > 0")
    if not 0.0 <= min_accuracy <= 1.0:
        raise ValueError("min_accuracy must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(
        progress_path,
        "start",
        controller_model_json=str(controller_model_json),
        eval_examples_jsonl=str(eval_examples_jsonl),
        min_eval_examples=min_eval_examples,
        min_accuracy=min_accuracy,
    )
    model = read_json(controller_model_json)
    eval_rows = trainable_rows(iter_jsonl(eval_examples_jsonl))
    training_manifest = read_json(training_manifest_json) if training_manifest_json else None
    eval_report = evaluate_model(model, eval_rows)
    same_data = same_training_examples(eval_examples_jsonl, training_manifest, eval_rows, model)
    overlap = training_overlap_report(eval_rows=eval_rows, training_manifest=training_manifest)
    status, reason, can_promote = promotion_status(
        eval_report=eval_report,
        same_training_data=same_data,
        training_overlap=overlap,
        min_eval_examples=min_eval_examples,
        min_accuracy=min_accuracy,
    )
    eval_report = {
        **eval_report,
        "eval_examples": len(eval_rows),
        "min_eval_examples": min_eval_examples,
        "min_accuracy": min_accuracy,
        "same_training_data": same_data,
        "training_overlap": overlap,
    }
    fingerprint = stable_hash(
        {
            "controller_model_json": str(controller_model_json),
            "eval_examples_fingerprint": examples_fingerprint(eval_rows),
            "min_eval_examples": min_eval_examples,
            "min_accuracy": min_accuracy,
        }
    )
    manifest = {
        "schema": "generic_finetuning_controller_external_eval_manifest.v1",
        "manifest_type": "generic_finetuning_controller_external_eval",
        "controller_model_json": str(controller_model_json),
        "eval_examples_jsonl": str(eval_examples_jsonl),
        "training_manifest_json": str(training_manifest_json or ""),
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "eval_report": eval_report,
        "promotion": {
            "status": status,
            "reason": reason,
            "can_promote_controller_model": can_promote,
            "can_edit_model_registry": False,
            "next_required_step": "runtime_smoke_replay" if can_promote else "fix_eval_blockers",
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "controller_external_eval_not_for_listen_pack",
        "release_status": "external_eval_artifact_only",
        "release_ready": False,
        "safety": {
            **SAFETY_FLAGS,
            "external_eval_required": True,
            "blocks_training_data_eval": True,
            "blocks_training_overlap_eval": True,
            "metadata_only": True,
            "not_release_artifact": True,
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "eval_report_json": str(output_dir / REPORT_NAME),
            "progress_jsonl": str(progress_path),
        },
    }
    write_json(output_dir / REPORT_NAME, eval_report)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", status=status, evaluated_examples=eval_report["evaluated_examples"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a generic fine-tuning controller model on external examples.")
    parser.add_argument("--controller-model-json", type=Path, required=True)
    parser.add_argument("--eval-examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-manifest-json", type=Path)
    parser.add_argument("--min-eval-examples", type=int, default=20)
    parser.add_argument("--min-accuracy", type=float, default=0.7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="eval_generic_finetuning_controller_model.py",
        entrypoint_path="/app/tools/eval_generic_finetuning_controller_model.py",
    )
    args = parse_args(argv)
    manifest = eval_controller_model(
        controller_model_json=args.controller_model_json,
        eval_examples_jsonl=args.eval_examples_jsonl,
        output_dir=args.output_dir,
        training_manifest_json=args.training_manifest_json,
        min_eval_examples=int(args.min_eval_examples),
        min_accuracy=float(args.min_accuracy),
    )
    print(json.dumps({"eval_report": manifest["eval_report"], "promotion": manifest["promotion"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
