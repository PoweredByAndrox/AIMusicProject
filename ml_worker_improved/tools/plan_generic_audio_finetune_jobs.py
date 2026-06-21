#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402

PLAN_NAME = "generic_audio_finetune_job_plan.json"
PROGRESS_NAME = "generic_audio_finetune_job_plan_progress.jsonl"
SHORT_NAMES = {
    "plan": "plan.json",
    "progress": "progress.jsonl",
}

TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}
VOCAL_STEMS = {"vocal", "vocals"}
MINUS_STEMS = {"minus", "instrumental"}

START_TIME = time.monotonic()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "planning_only_no_training_started",
        "expected_acceleration_mode": "gpu_training_required_for_ready_jobs",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def output_names(*, short_output_names: bool) -> dict[str, str]:
    if short_output_names:
        return dict(SHORT_NAMES)
    return {
        "plan": PLAN_NAME,
        "progress": PROGRESS_NAME,
    }


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_stem(stem: Any) -> str:
    value = str(stem or "").strip().lower()
    if value in VOCAL_STEMS:
        return "vocal"
    if value in MINUS_STEMS:
        return "minus"
    return value or "unknown"


def row_is_trainable_audio(row: dict[str, Any]) -> bool:
    source_audio_path = str(row.get("source_audio_path") or "")
    target_audio_path = str(row.get("target_audio_path") or "")
    return (
        row.get("trainable") is True
        and str(row.get("permission_status") or "").lower() in TRAINABLE_PERMISSION_STATUSES
        and bool(source_audio_path)
        and bool(target_audio_path)
        and Path(source_audio_path).exists()
        and Path(target_audio_path).exists()
    )


def scope_for_stem(stem: str) -> str:
    if stem == "vocal":
        return "vocal_teacher_mask_adapter"
    if stem == "minus":
        return "minus_teacher_mask_adapter"
    return "stem_teacher_mask_adapter"


def command_for_example(row: dict[str, Any], *, output_root: str) -> list[str]:
    stem = normalize_stem(row.get("stem_type"))
    job_id = stable_hash(["teacher_mask", row.get("id"), row.get("target_audio_sha256")])[:16]
    return [
        "python",
        "/app/tools/train_teacher_mask_separator.py",
        "--source-mix",
        str(row["source_audio_path"]),
        "--teacher-target",
        str(row["target_audio_path"]),
        "--output-dir",
        f"{output_root}/{stem}/{job_id}",
        "--stem",
        stem,
        "--device",
        "cuda",
    ]


def build_job(scope: str, rows: Sequence[dict[str, Any]], *, min_unique_tracks: int, output_root: str) -> dict[str, Any]:
    unique_tracks = sorted({str(row.get("track_id") or row.get("manifest_path") or row.get("target_audio_path")) for row in rows})
    stem_types = sorted({normalize_stem(row.get("stem_type")) for row in rows})
    status = "ready_for_gpu_training" if len(unique_tracks) >= min_unique_tracks else "blocked_insufficient_unique_tracks"
    blocker_reason = "" if status == "ready_for_gpu_training" else "audio fine-tune requires more unique verified tracks"
    example_jobs = [
        {
            "example_id": row.get("id"),
            "track_id": row.get("track_id"),
            "stem_type": normalize_stem(row.get("stem_type")),
            "source_audio_path": row.get("source_audio_path"),
            "target_audio_path": row.get("target_audio_path"),
            "target_audio_sha256": row.get("target_audio_sha256"),
            "command": command_for_example(row, output_root=output_root),
        }
        for row in rows
    ]
    return {
        "schema": "generic_audio_finetune_job.v1",
        "job_id": stable_hash(["generic_audio_finetune_job", scope, stem_types, unique_tracks])[:16],
        "scope": scope,
        "status": status,
        "blocker_reason": blocker_reason,
        "entrypoint": "/app/tools/train_teacher_mask_separator.py",
        "docker_service": "ml-worker",
        "accelerator_mode": "gpu_training_required",
        "device": "cuda",
        "example_count": len(rows),
        "unique_track_count": len(unique_tracks),
        "min_unique_tracks": min_unique_tracks,
        "stem_types": stem_types,
        "example_jobs": example_jobs,
        "training_started": False,
        "fine_tune_started": False,
        "release_ready": False,
        "delivery_allowed": False,
    }


def plan_generic_audio_finetune_jobs(
    *,
    audio_training_examples_jsonl: Path,
    output_dir: Path,
    min_unique_tracks: int = 3,
    training_output_root: str = "/app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_runs",
    short_output_names: bool = False,
) -> dict[str, Any]:
    if min_unique_tracks <= 0:
        raise ValueError("min_unique_tracks must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(short_output_names=short_output_names)
    progress_path = output_dir / names["progress"]
    progress(progress_path, "start", audio_training_examples_jsonl=str(audio_training_examples_jsonl))
    rows = [row for row in iter_jsonl(audio_training_examples_jsonl) if row_is_trainable_audio(row)]
    progress(progress_path, "examples_loaded", trainable_audio_examples=len(rows))

    by_scope: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stem = normalize_stem(row.get("stem_type"))
        by_scope.setdefault(scope_for_stem(stem), []).append(row)

    jobs = [
        build_job(scope, scope_rows, min_unique_tracks=min_unique_tracks, output_root=training_output_root)
        for scope, scope_rows in sorted(by_scope.items())
    ]
    ready_jobs = [job for job in jobs if job["status"] == "ready_for_gpu_training"]
    blocked_jobs = [job for job in jobs if job["status"] != "ready_for_gpu_training"]
    manifest = {
        "schema": "generic_audio_finetune_job_plan.v1",
        "manifest_type": "generic_audio_finetune_job_plan",
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "output_dir": str(output_dir),
        "fingerprint": stable_hash(
            {
                "rows": [
                    {
                        "id": row.get("id"),
                        "track_id": row.get("track_id"),
                        "stem_type": normalize_stem(row.get("stem_type")),
                        "target_audio_sha256": row.get("target_audio_sha256"),
                    }
                    for row in rows
                ],
                "min_unique_tracks": min_unique_tracks,
            }
        ),
        "summary": {
            "trainable_audio_examples": len(rows),
            "job_count": len(jobs),
            "ready_job_count": len(ready_jobs),
            "blocked_job_count": len(blocked_jobs),
            "min_unique_tracks": min_unique_tracks,
        },
        "jobs": jobs,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "audio_finetune_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "short_output_names": short_output_names,
        "safety": {
            "gpu_training_required_for_ready_jobs": True,
            "fail_closed_when_insufficient_unique_tracks": True,
            "requires_holdout_eval_before_promotion": True,
            "not_release_artifact": True,
        },
        "outputs": {
            "plan_json": str(output_dir / names["plan"]),
            "progress_jsonl": str(progress_path),
        },
    }
    write_json(output_dir / names["plan"], manifest)
    progress(progress_path, "finish", plan_json=str(output_dir / names["plan"]), **manifest["summary"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan GPU audio fine-tune jobs from generic audio training examples.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-unique-tracks", type=int, default=3)
    parser.add_argument(
        "--training-output-root",
        default="/app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_runs",
    )
    parser.add_argument("--short-output-names", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="plan_generic_audio_finetune_jobs.py",
        entrypoint_path="/app/tools/plan_generic_audio_finetune_jobs.py",
    )
    args = parse_args(argv)
    manifest = plan_generic_audio_finetune_jobs(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        output_dir=args.output_dir,
        min_unique_tracks=int(args.min_unique_tracks),
        training_output_root=str(args.training_output_root),
        short_output_names=bool(args.short_output_names),
    )
    print(json.dumps({"summary": manifest["summary"], "fingerprint": manifest["fingerprint"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
