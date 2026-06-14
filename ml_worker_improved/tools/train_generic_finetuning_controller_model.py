#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402

MODEL_NAME = "generic_finetuning_controller_model.json"
MANIFEST_NAME = "generic_finetuning_controller_training_manifest.json"
EVAL_NAME = "generic_finetuning_controller_eval_report.json"
PROGRESS_NAME = "generic_finetuning_controller_training_progress.jsonl"

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
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
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "controller_training_uses_existing_metadata_examples",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def example_stem(row: dict[str, Any]) -> str:
    return str(row.get("stem_type") or "unknown")


def example_route(row: dict[str, Any]) -> str:
    target = row.get("target")
    if isinstance(target, dict) and target.get("chosen_model"):
        return str(target["chosen_model"])
    input_payload = row.get("input")
    if isinstance(input_payload, dict) and input_payload.get("model_id"):
        return str(input_payload["model_id"])
    return "unknown_route"


def example_accepted(row: dict[str, Any]) -> bool:
    result = row.get("result")
    if isinstance(result, dict) and isinstance(result.get("accepted"), bool):
        return bool(result["accepted"])
    target = row.get("target")
    if isinstance(target, dict):
        return str(target.get("action") or "") == "reuse_route_candidate"
    return False


def example_failure_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    input_payload = row.get("input")
    if isinstance(input_payload, dict):
        labels.extend(str(item) for item in as_list(input_payload.get("failure_labels")) if str(item))
    return sorted(set(labels))


def row_is_trainable(row: dict[str, Any]) -> bool:
    return row.get("trainable") is not False


def route_prior_row(
    *,
    stem_type: str,
    route_id: str,
    rows: Sequence[dict[str, Any]],
    min_examples_per_route: int,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    accepted = sum(1 for row in rows if example_accepted(row))
    rejected = len(rows) - accepted
    posterior_accept_rate = (accepted + alpha) / (accepted + rejected + alpha + beta)
    failure_labels = Counter(label for row in rows for label in example_failure_labels(row))
    recommendation = "prefer" if len(rows) >= min_examples_per_route and posterior_accept_rate >= 0.5 else "avoid_or_explore"
    return {
        "schema": "generic_finetuning_controller_route_prior.v1",
        "stem_type": stem_type,
        "route_id": route_id,
        "example_count": len(rows),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "posterior_accept_rate": round(posterior_accept_rate, 6),
        "recommendation": recommendation,
        "top_failure_labels": [
            {"label": label, "count": count}
            for label, count in failure_labels.most_common(10)
        ],
    }


def build_model(
    examples: Sequence[dict[str, Any]],
    *,
    min_examples_per_route: int,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        if not row_is_trainable(row):
            continue
        grouped[(example_stem(row), example_route(row))].append(row)
    priors = [
        route_prior_row(
            stem_type=stem,
            route_id=route,
            rows=rows,
            min_examples_per_route=min_examples_per_route,
            alpha=alpha,
            beta=beta,
        )
        for (stem, route), rows in grouped.items()
    ]
    priors.sort(key=lambda row: (row["stem_type"], -float(row["posterior_accept_rate"]), row["route_id"]))
    best_by_stem: dict[str, dict[str, Any]] = {}
    for row in priors:
        best_by_stem.setdefault(str(row["stem_type"]), row)
    return {
        "schema": "generic_finetuning_controller_model.v1",
        "model_type": "bayesian_route_prior_controller",
        "route_priors": priors,
        "best_route_by_stem": best_by_stem,
        "hyperparameters": {
            "min_examples_per_route": min_examples_per_route,
            "alpha": alpha,
            "beta": beta,
        },
        "delivery_allowed": False,
        "presentation_status": "controller_model_not_for_listen_pack",
        "release_status": "training_artifact_only",
        "release_ready": False,
    }


def evaluate_model(model: dict[str, Any], examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    priors = {
        (str(row["stem_type"]), str(row["route_id"])): row
        for row in model.get("route_priors", [])
        if isinstance(row, dict)
    }
    correct = 0
    evaluated = 0
    missing = 0
    for row in examples:
        key = (example_stem(row), example_route(row))
        prior = priors.get(key)
        if prior is None:
            missing += 1
            continue
        predicted_accept = float(prior.get("posterior_accept_rate") or 0.0) >= 0.5
        if predicted_accept == example_accepted(row):
            correct += 1
        evaluated += 1
    accuracy = round(correct / evaluated, 6) if evaluated else 0.0
    return {
        "schema": "generic_finetuning_controller_eval_report.v1",
        "evaluated_examples": evaluated,
        "correct": correct,
        "missing_route_examples": missing,
        "accuracy": accuracy,
        "promotion_candidate": evaluated >= 20 and accuracy >= 0.7 and missing == 0,
    }


def train_controller_model(
    *,
    controller_examples_jsonl: Path,
    output_dir: Path,
    min_examples_per_route: int = 2,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, Any]:
    if min_examples_per_route <= 0:
        raise ValueError("min_examples_per_route must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(progress_path, "start", controller_examples_jsonl=str(controller_examples_jsonl))
    examples = list(iter_jsonl(controller_examples_jsonl))
    trainable = [row for row in examples if row_is_trainable(row)]
    progress(progress_path, "examples_loaded", input_examples=len(examples), trainable_examples=len(trainable))
    model = build_model(
        trainable,
        min_examples_per_route=min_examples_per_route,
        alpha=alpha,
        beta=beta,
    )
    eval_report = evaluate_model(model, trainable)
    fingerprint = stable_hash(
        {
            "examples": [
                {
                    "id": row.get("id"),
                    "stem": example_stem(row),
                    "route": example_route(row),
                    "accepted": example_accepted(row),
                    "failure_labels": example_failure_labels(row),
                }
                for row in sorted(trainable, key=lambda item: str(item.get("id") or ""))
            ],
            "hyperparameters": model["hyperparameters"],
        }
    )
    promotion_status = "candidate_ready_for_external_eval" if eval_report["promotion_candidate"] else "blocked"
    manifest = {
        "schema": "generic_finetuning_controller_training_manifest.v1",
        "manifest_type": "generic_finetuning_controller_training",
        "controller_examples_jsonl": str(controller_examples_jsonl),
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "training_started": True,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "controller_training_not_for_listen_pack",
        "release_status": "training_artifact_only",
        "release_ready": False,
        "summary": {
            "input_examples": len(examples),
            "trainable_examples": len(trainable),
            "route_prior_count": len(model["route_priors"]),
            "stem_count": len(model["best_route_by_stem"]),
        },
        "model": model,
        "eval_report": eval_report,
        "promotion": {
            "status": promotion_status,
            "reason": (
                "external_eval_required_before_promotion"
                if promotion_status != "blocked"
                else "insufficient_eval_accuracy_or_coverage_for_promotion"
            ),
            "can_promote": False,
        },
        "outputs": {
            "model_json": str(output_dir / MODEL_NAME),
            "eval_report_json": str(output_dir / EVAL_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": dict(SAFETY_FLAGS),
    }
    progress(progress_path, "write_outputs", route_prior_count=len(model["route_priors"]))
    write_json(output_dir / MODEL_NAME, model)
    write_json(output_dir / EVAL_NAME, eval_report)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a metadata controller model from generic fine-tuning examples.")
    parser.add_argument("--controller-examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-examples-per-route", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="train_generic_finetuning_controller_model.py",
        entrypoint_path="/app/tools/train_generic_finetuning_controller_model.py",
    )
    args = parse_args(argv)
    manifest = train_controller_model(
        controller_examples_jsonl=args.controller_examples_jsonl,
        output_dir=args.output_dir,
        min_examples_per_route=int(args.min_examples_per_route),
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    print(json.dumps({"summary": manifest["summary"], "promotion": manifest["promotion"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
