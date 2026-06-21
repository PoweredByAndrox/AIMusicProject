#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402

MANIFEST_NAME = "manifest.json"
TRAIN_NAME = "train.jsonl"
EVAL_NAME = "eval.jsonl"
EXCLUDED_NAME = "excluded.jsonl"
PROGRESS_NAME = "progress.jsonl"

SCHEMA_NAME = "qwen_controller_sft_example.v1"
SOURCE_SCHEMA_NAME = "generic_finetuning_controller_decision_example.v1"
TEMPLATE_VERSION = "qwen_controller_sft_template.v1"
TARGET_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

SYSTEM_PROMPT = (
    "You are the RALPH-AI route controller. Return only compact JSON with "
    "action, route_id, stem_type, and accepted. Never approve release, listen "
    "delivery, registry edits, or fallback masking from this training task."
)

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
    "blocks_train_eval_overlap": True,
    "dirty_stems_never_ground_truth": True,
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
        "gpu_inference_skip_reason": "qwen_sft_dataset_build_uses_existing_metadata_examples",
        "expected_acceleration_mode": "external_only_cpu_metadata_dataset_build",
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


def input_payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("input")
    return value if isinstance(value, dict) else {}


def target_payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("target")
    return value if isinstance(value, dict) else {}


def result_payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("result")
    return value if isinstance(value, dict) else {}


def example_manifest_path(row: dict[str, Any]) -> str:
    input_row = input_payload(row)
    if input_row.get("manifest_path"):
        return str(input_row["manifest_path"])
    return str(row.get("manifest_path") or "")


def canonical_manifest_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = posixpath.normpath(text.replace("\\", "/"))
    return "" if normalized == "." else normalized


def example_stem(row: dict[str, Any]) -> str:
    for value in (row.get("stem_type"), input_payload(row).get("stem_type"), row.get("target_stem")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def example_route(row: dict[str, Any]) -> str:
    target = target_payload(row)
    input_row = input_payload(row)
    for value in (
        target.get("chosen_model"),
        target.get("route_id"),
        input_row.get("model_id"),
        input_row.get("route_id"),
        row.get("model_id"),
        row.get("route_id"),
        row.get("candidate_model_id"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def example_action(row: dict[str, Any]) -> str:
    target = target_payload(row)
    action = target.get("action")
    if isinstance(action, str) and action.strip():
        return action.strip()
    return "reuse_route_candidate" if example_accepted(row) else "avoid_route_and_generate_alternative"


def example_accepted(row: dict[str, Any]) -> bool:
    result = result_payload(row)
    if isinstance(result.get("accepted"), bool):
        return bool(result["accepted"])
    return str(target_payload(row).get("action") or "") == "reuse_route_candidate"


def example_failure_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    input_row = input_payload(row)
    for value in (input_row.get("failure_labels"), row.get("failure_labels")):
        labels.extend(str(item) for item in as_list(value) if str(item))
    return sorted(set(labels))


def source_claims_delivery_or_release(row: dict[str, Any]) -> bool:
    input_row = input_payload(row)
    if row.get("delivery_allowed") is True or row.get("release_ready") is True:
        return True
    if input_row.get("delivery_allowed") is True or input_row.get("release_ready") is True:
        return True
    release_status = str(row.get("release_status") or input_row.get("release_status") or "").lower()
    return release_status in {"release_ready", "ready_for_release", "accepted_for_release"}


def exclusion_reason(row: dict[str, Any]) -> str:
    if str(row.get("schema") or "") != SOURCE_SCHEMA_NAME:
        return "unexpected_source_schema"
    if row.get("trainable") is False:
        return "source_row_not_trainable"
    if source_claims_delivery_or_release(row):
        return "source_row_claims_delivery_or_release"
    if not example_manifest_path(row):
        return "source_manifest_path_missing"
    if not example_route(row):
        return "source_route_id_missing"
    return ""


def assistant_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": example_accepted(row),
        "action": example_action(row),
        "route_id": example_route(row),
        "stem_type": example_stem(row),
    }


def user_payload(row: dict[str, Any]) -> dict[str, Any]:
    input_row = input_payload(row)
    result = result_payload(row)
    return {
        "task": "decide_generic_audio_route_action",
        "stem_type": example_stem(row),
        "route_id": example_route(row),
        "manifest_path": example_manifest_path(row),
        "delivery_status": str(input_row.get("delivery_status") or row.get("delivery_status") or ""),
        "presentation_status": str(input_row.get("presentation_status") or row.get("presentation_status") or ""),
        "failure_labels": example_failure_labels(row),
        "human_verified": result.get("human_verified") is True,
        "human_rating": str(result.get("human_rating") or ""),
        "reference_verified": result.get("reference_verified") is True,
        "controller_training_target": str(row.get("training_target") or "llm_controller"),
        "constraints": {
            "do_not_mark_release_ready": True,
            "do_not_create_listen_artifacts": True,
            "do_not_treat_rejected_audio_as_success": True,
            "metadata_only_training_example": True,
        },
    }


def sft_example(row: dict[str, Any], *, split: str, source_jsonl: Path) -> dict[str, Any]:
    row_id = str(row.get("id") or "")
    manifest_path = example_manifest_path(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload(row), ensure_ascii=False, sort_keys=True)},
        {"role": "assistant", "content": json.dumps(assistant_payload(row), ensure_ascii=False, sort_keys=True)},
    ]
    return {
        "schema": SCHEMA_NAME,
        "id": stable_hash(["qwen_controller_sft", split, row_id, manifest_path])[:16],
        "messages": messages,
        "metadata": {
            "split": split,
            "template_version": TEMPLATE_VERSION,
            "target_model_id": TARGET_MODEL_ID,
            "source": {
                "controller_examples_jsonl": str(source_jsonl),
                "source_schema": str(row.get("schema") or ""),
            },
            "provenance": {
                "source_row_id": row_id,
                "track_id": str(row.get("track_id") or ""),
                "stem_type": example_stem(row),
                "route_id": example_route(row),
                "manifest_path": manifest_path,
            },
        },
        "training_target": "qwen_controller_sft",
        "delivery_allowed": False,
        "presentation_status": "qwen_sft_example_not_for_listen_pack",
        "release_status": "training_dataset_row_only",
        "release_ready": False,
    }


def excluded_row(row: dict[str, Any], *, split: str, source_jsonl: Path, reason: str) -> dict[str, Any]:
    return {
        "schema": "qwen_controller_sft_excluded_row.v1",
        "id": stable_hash(["qwen_controller_sft_excluded", split, str(row.get("id") or ""), reason])[:16],
        "split": split,
        "source_row_id": str(row.get("id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "stem_type": example_stem(row),
        "route_id": example_route(row),
        "manifest_path": example_manifest_path(row),
        "source_jsonl": str(source_jsonl),
        "exclude_reason": reason,
    }


def excluded_sft_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    source_row_id = str(provenance.get("source_row_id") or "")
    manifest_path = str(provenance.get("manifest_path") or "")
    split = str(metadata.get("split") or "")
    return {
        "schema": "qwen_controller_sft_excluded_row.v1",
        "id": stable_hash(["qwen_controller_sft_excluded", split, source_row_id, manifest_path, reason])[:16],
        "split": split,
        "source_row_id": source_row_id,
        "track_id": str(provenance.get("track_id") or ""),
        "stem_type": str(provenance.get("stem_type") or ""),
        "route_id": str(provenance.get("route_id") or ""),
        "manifest_path": manifest_path,
        "source_jsonl": str(source.get("controller_examples_jsonl") or ""),
        "exclude_reason": reason,
    }


def convert_rows(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    source_jsonl: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        reason = exclusion_reason(row)
        if reason:
            excluded.append(excluded_row(row, split=split, source_jsonl=source_jsonl, reason=reason))
            continue
        converted.append(sft_example(row, split=split, source_jsonl=source_jsonl))
    return converted, excluded


def manifest_paths(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {
        canonical_manifest_path(str(row.get("metadata", {}).get("provenance", {}).get("manifest_path") or ""))
        for row in rows
        if canonical_manifest_path(str(row.get("metadata", {}).get("provenance", {}).get("manifest_path") or ""))
    }


def remove_overlap_rows(rows: Sequence[dict[str, Any]], overlap_paths: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        manifest_path = canonical_manifest_path(str(row.get("metadata", {}).get("provenance", {}).get("manifest_path") or ""))
        if manifest_path not in overlap_paths:
            result.append(row)
    return result


def overlap_rows(rows: Sequence[dict[str, Any]], overlap_paths: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        manifest_path = canonical_manifest_path(str(row.get("metadata", {}).get("provenance", {}).get("manifest_path") or ""))
        if manifest_path in overlap_paths:
            result.append(row)
    return result


def build_qwen_controller_sft_dataset(
    *,
    controller_examples_jsonl: Path,
    output_dir: Path,
    eval_examples_jsonl: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()

    progress(
        progress_path,
        "start",
        controller_examples_jsonl=str(controller_examples_jsonl),
        eval_examples_jsonl=str(eval_examples_jsonl or ""),
    )
    raw_train_rows = list(iter_jsonl(controller_examples_jsonl))
    raw_eval_rows = list(iter_jsonl(eval_examples_jsonl)) if eval_examples_jsonl else []

    train_rows, excluded_rows = convert_rows(raw_train_rows, split="train", source_jsonl=controller_examples_jsonl)
    eval_rows: list[dict[str, Any]] = []
    eval_excluded: list[dict[str, Any]] = []
    if eval_examples_jsonl:
        eval_rows, eval_excluded = convert_rows(raw_eval_rows, split="eval", source_jsonl=eval_examples_jsonl)
        excluded_rows.extend(eval_excluded)

    train_paths = manifest_paths(train_rows)
    eval_paths = manifest_paths(eval_rows)
    overlap_paths = sorted(train_paths.intersection(eval_paths))
    failed_checks: list[dict[str, Any]] = []
    if overlap_paths:
        overlap_path_set = set(overlap_paths)
        failed_checks.append(
            {
                "check": "train_eval_manifest_overlap",
                "overlap_count": len(overlap_paths),
                "overlap_manifest_paths": overlap_paths[:20],
            }
        )
        excluded_rows.extend(
            excluded_sft_row(row, reason="train_eval_manifest_overlap")
            for row in overlap_rows(eval_rows, overlap_path_set)
        )
        eval_rows = remove_overlap_rows(eval_rows, overlap_path_set)
    if not train_rows:
        failed_checks.append({"check": "no_train_examples", "reason": "no usable train examples after safety filtering"})

    dataset_status = "blocked" if failed_checks else "ready"
    fingerprint = stable_hash(
        {
            "template_version": TEMPLATE_VERSION,
            "target_model_id": TARGET_MODEL_ID,
            "train": [
                {
                    "id": row["id"],
                    "messages": row["messages"],
                    "provenance": row["metadata"]["provenance"],
                }
                for row in train_rows
            ],
            "eval": [
                {
                    "id": row["id"],
                    "messages": row["messages"],
                    "provenance": row["metadata"]["provenance"],
                }
                for row in eval_rows
            ],
            "failed_checks": failed_checks,
        }
    )
    summary = {
        "input_train_rows": len(raw_train_rows),
        "input_eval_rows": len(raw_eval_rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "excluded_examples": len(excluded_rows),
        "overlap_count": len(overlap_paths),
    }
    outputs = {
        "manifest_json": str(output_dir / MANIFEST_NAME),
        "train_jsonl": str(output_dir / TRAIN_NAME),
        "eval_jsonl": str(output_dir / EVAL_NAME),
        "excluded_jsonl": str(output_dir / EXCLUDED_NAME),
        "progress_jsonl": str(progress_path),
    }
    manifest = {
        "schema": "qwen_controller_sft_dataset_manifest.v1",
        "manifest_type": "qwen_controller_sft_dataset",
        "dataset_status": dataset_status,
        "controller_examples_jsonl": str(controller_examples_jsonl),
        "eval_examples_jsonl": str(eval_examples_jsonl or ""),
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "target_model_id": TARGET_MODEL_ID,
        "template_version": TEMPLATE_VERSION,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "qwen_controller_sft_dataset_not_for_listen_pack",
        "release_status": "training_dataset_artifact_only",
        "release_ready": False,
        "failed_checks": failed_checks,
        "summary": summary,
        "outputs": outputs,
        "safety": dict(SAFETY_FLAGS),
        "recommended_next_step": (
            "run_qwen_controller_sft_training_after_dataset_review"
            if dataset_status == "ready"
            else "fix_dataset_blockers_before_training"
        ),
    }

    progress(progress_path, "examples_loaded", **summary)
    progress(progress_path, "write_outputs", dataset_status=dataset_status, failed_checks=len(failed_checks))
    write_jsonl(output_dir / TRAIN_NAME, train_rows)
    write_jsonl(output_dir / EVAL_NAME, eval_rows)
    write_jsonl(output_dir / EXCLUDED_NAME, excluded_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen controller SFT JSONL from generic controller examples.")
    parser.add_argument("--controller-examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-examples-jsonl", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_qwen_controller_sft_dataset.py",
        entrypoint_path="/app/tools/build_qwen_controller_sft_dataset.py",
    )
    args = parse_args(argv)
    manifest = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=args.controller_examples_jsonl,
        output_dir=args.output_dir,
        eval_examples_jsonl=args.eval_examples_jsonl,
    )
    print(json.dumps({"dataset_status": manifest["dataset_status"], "summary": manifest["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["dataset_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
