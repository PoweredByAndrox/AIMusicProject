#!/usr/bin/env python3
"""Build exact verified-reference corpus gaps for a blocked full-pack loop."""

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


START_TIME = time.monotonic()
MANIFEST_NAME = "gap.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
    "delivery_allowed",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
    "separator_training_started",
    "cleanup_routes_added",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_reference_corpus_gap_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_corpus_gap_plan_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "reference_corpus_gap_plan_no_training_started",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def normalized_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "vocals":
        return "vocal"
    if text == "instrumental":
        return "minus"
    return text or "unknown"


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    for nested_key in ("safety", "contract"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def resolved_path(raw_path: Any, *, base_dir: Path) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return Path("")
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def default_min_unique_tracks(training_plan: dict[str, Any], corpus_manifest: dict[str, Any] | None) -> int:
    summary = training_plan.get("summary")
    if isinstance(summary, dict):
        value = int_or_none(summary.get("min_unique_tracks"))
        if value:
            return value
    if corpus_manifest is not None:
        value = int_or_none(corpus_manifest.get("min_unique_tracks"))
        if value:
            return value
    return 3


def required_actions_for_gap(action: dict[str, Any], *, missing_training_job: bool) -> list[str]:
    actions = ["collect_verified_reference_stem"]
    if missing_training_job:
        actions.append("rerun_generic_audio_finetune_job_plan")
    if action.get("route_intent") == "minus_repair":
        actions.append("run_minus_repair_route")
    actions.extend(["rerun_corpus_validation", "rerun_training_plan"])
    return stable_unique(actions)


def build_gap_rows(action_rows: Sequence[dict[str, Any]], *, default_min_unique: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(action_rows, start=1):
        required_next_actions = [str(item) for item in action.get("required_next_actions") or [] if str(item)]
        if "verified_reference_corpus_expand" not in required_next_actions:
            continue
        status = str(action.get("action_status") or "")
        if not status.startswith("blocked"):
            continue
        stem = normalized_stem(action.get("stem_type"))
        min_unique = int_or_none(action.get("min_unique_tracks")) or default_min_unique
        unique_count = int_or_none(action.get("unique_track_count")) or 0
        missing_training_job = status in {"blocked_missing_training_job", "blocked_minus_repair_missing_training_job"}
        missing_unique = max(min_unique - unique_count, 0)
        rows.append(
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": f"g{index:03d}",
                "queue_item_id": str(action.get("queue_item_id") or ""),
                "stem_type": stem,
                "route_intent": str(action.get("route_intent") or ""),
                "action_status": status,
                "training_job_statuses": [str(item) for item in action.get("training_job_statuses") or [] if str(item)],
                "unique_track_count": unique_count,
                "min_unique_tracks": min_unique,
                "missing_unique_track_count": missing_unique,
                "missing_training_job": missing_training_job,
                "required_reference_stem_count": missing_unique,
                "required_next_actions": required_actions_for_gap(action, missing_training_job=missing_training_job),
                "priority": "P0" if stem in {"vocal", "minus"} else ("P1" if stem in {"guitar", "piano"} else "P2"),
                "training_started": False,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    rows.sort(key=lambda row: (row["priority"], row["stem_type"], row["gap_id"]))
    return rows


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Full Pack Reference Corpus Gaps",
        "",
        f"- gap_status: `{manifest['gap_status']}`",
        f"- gap_count: `{manifest['gap_count']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
        "## Gaps",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['gap_id']}` `{row['stem_type']}` missing_tracks=`{row['missing_unique_track_count']}` "
            f"missing_training_job=`{str(row['missing_training_job']).lower()}`"
        )
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    return "\n".join(lines) + "\n"


def build_full_pack_reference_corpus_gap_plan(
    *,
    full_pack_loop_plan_json: Path,
    training_plan_json: Path,
    output_dir: Path,
    verified_corpus_manifest_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        full_pack_loop_plan_json=str(full_pack_loop_plan_json),
        training_plan_json=str(training_plan_json),
        verified_corpus_manifest_json=str(verified_corpus_manifest_json or ""),
    )

    failed: list[str] = []
    full_pack_plan = read_json(full_pack_loop_plan_json)
    failed.extend(collect_true_flag_failures("full_pack_loop_plan", full_pack_plan))
    if full_pack_plan.get("schema") != "full_pack_loop_iteration_plan.v1":
        failed.append("full_pack_loop_plan_schema_unexpected")
    if full_pack_plan.get("failed_checks") not in ([], None):
        failed.append("full_pack_loop_plan_failed_checks_not_empty")

    rows_path = resolved_path(full_pack_plan.get("rows_jsonl"), base_dir=full_pack_loop_plan_json.parent)
    if not str(full_pack_plan.get("rows_jsonl") or "").strip() or not rows_path.exists():
        failed.append("full_pack_loop_plan_rows_jsonl_missing")
        action_rows: list[dict[str, Any]] = []
    else:
        action_rows = list(iter_jsonl(rows_path))
        if any(row.get("schema") != "full_pack_loop_iteration_action.v1" for row in action_rows):
            failed.append("full_pack_loop_action_schema_unexpected")

    training_plan = read_json(training_plan_json)
    failed.extend(collect_true_flag_failures("training_plan", training_plan))
    if training_plan.get("schema") != "generic_audio_finetune_job_plan.v1":
        failed.append("training_plan_schema_unexpected")

    corpus_manifest = read_json(verified_corpus_manifest_json) if verified_corpus_manifest_json else None
    if corpus_manifest is not None:
        failed.extend(collect_true_flag_failures("verified_corpus", corpus_manifest))
        if corpus_manifest.get("schema") != "verified_audio_training_corpus_manifest.v1":
            failed.append("verified_corpus_schema_unexpected")

    failed = stable_unique(failed)
    rows = [] if failed else build_gap_rows(action_rows, default_min_unique=default_min_unique_tracks(training_plan, corpus_manifest))
    status = "blocked" if failed else ("blocked_waiting_for_verified_reference_corpus" if rows else "no_reference_corpus_gap_detected")
    manifest = {
        "schema": "full_pack_reference_corpus_gap_plan.v1",
        "manifest_type": "full_pack_reference_corpus_gap_plan",
        "gap_status": status,
        "gap_count": len(rows),
        "failed_checks": failed,
        "full_pack_loop_plan_json": str(full_pack_loop_plan_json),
        "training_plan_json": str(training_plan_json),
        "verified_corpus_manifest_json": str(verified_corpus_manifest_json or ""),
        "training_plan_summary": training_plan.get("summary") if isinstance(training_plan.get("summary"), dict) else {},
        "verified_corpus_status": str((corpus_manifest or {}).get("corpus_validation_status") or ""),
        "verified_corpus_failed_checks": [str(item) for item in (corpus_manifest or {}).get("failed_checks") or []],
        "missing_reference_stem_count": sum(int(row["required_reference_stem_count"]) for row in rows),
        "missing_training_job_count": sum(1 for row in rows if row["missing_training_job"]),
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "rows_jsonl": str(output_dir / ROWS_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "reference_corpus_gap_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "dataset_built": False,
            "training_started": False,
            "not_release_artifact": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", gap_status=status, gap_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-pack verified-reference corpus gap plan.")
    parser.add_argument("--full-pack-loop-plan-json", type=Path, required=True)
    parser.add_argument("--training-plan-json", type=Path, required=True)
    parser.add_argument("--verified-corpus-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_reference_corpus_gap_plan.py",
        entrypoint_path="/app/tools/build_full_pack_reference_corpus_gap_plan.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_reference_corpus_gap_plan(
        full_pack_loop_plan_json=args.full_pack_loop_plan_json,
        training_plan_json=args.training_plan_json,
        verified_corpus_manifest_json=args.verified_corpus_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "gap_status": manifest["gap_status"],
                "gap_count": manifest["gap_count"],
                "missing_reference_stem_count": manifest["missing_reference_stem_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
