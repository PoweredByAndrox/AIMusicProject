#!/usr/bin/env python3
"""Build the next ARIS action plan from raw-reference recovery queue rows."""

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


START_TIME = time.monotonic()
MANIFEST_NAME = "plan.json"
ITEMS_NAME = "items.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
SOURCE_SCHEMA = "full_pack_raw_reference_bundle_recovery_queue_manifest.v1"
ROW_SCHEMA = "full_pack_raw_reference_bundle_recovery_item.v1"
PLAN_SCHEMA = "full_pack_raw_reference_recovery_action_plan_manifest.v1"
ITEM_SCHEMA = "full_pack_raw_reference_recovery_action_item.v1"
BLOCKED_TRUE_FLAGS = (
    "accepted_as_ground_truth",
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
BLOCKED_ROW_OUTPUT_FIELDS = (
    "candidate_root",
    "ground_truth_manifest_json",
)
ROUTE_NEXT_GATE = {
    "collect_clean_reference_bundle": "collect_verified_clean_reference_bundle",
    "repair_or_replace_reconstruction_failed_bundle": "rerun_full_mix_reconstruction_audit",
    "replace_missing_requested_stem_candidate": "rerun_raw_reference_bundle_worklist",
    "inspect_blocked_raw_reference_bundle": "inspect_raw_reference_blockers",
}
ROUTE_NEXT_COMMAND_INTENT = {
    "collect_clean_reference_bundle": "collect_clean_project_reference_export_then_rerun_verified_bundle_bridge",
    "repair_or_replace_reconstruction_failed_bundle": "provide_clean_scoped_export_then_rerun_reconstruction_audit",
    "replace_missing_requested_stem_candidate": "collect_missing_requested_stem_then_rerun_raw_worklist",
    "inspect_blocked_raw_reference_bundle": "inspect_blockers_then_update_raw_reference_route",
}


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
        "expected_acceleration_mode": "external_only_cpu_raw_reference_action_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_action_plan_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_action_plan_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "")]


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


def resolved_rows_path(manifest: dict[str, Any], *, base_dir: Path) -> Path | None:
    outputs = manifest.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("rows_jsonl") or "")
    text = text or str(manifest.get("rows_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_recovery_rows(
    raw_reference_recovery_queue_json: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    rows_path = resolved_rows_path(manifest, base_dir=raw_reference_recovery_queue_json.parent)
    expected_count = int(manifest.get("queue_item_count") or 0)
    if rows_path is None or not rows_path.is_file():
        if expected_count > 0:
            failed.append("raw_reference_recovery_queue_rows_jsonl_missing")
        return [], failed
    rows = list(iter_jsonl(rows_path))
    for row_index, row in enumerate(rows, start=1):
        row_prefix = f"raw_reference_recovery_row_{row_index:03d}"
        if row.get("schema") != ROW_SCHEMA:
            failed.append(f"{row_prefix}_schema_unexpected")
        failed.extend(collect_true_flag_failures(row_prefix, row))
        for field in BLOCKED_ROW_OUTPUT_FIELDS:
            if str(row.get(field) or ""):
                failed.append(f"{row_prefix}_{field}_present")
    if expected_count != len(rows):
        failed.append("raw_reference_recovery_queue_item_count_mismatch")
    return rows if not failed else [], stable_unique(failed)


def next_gate(route_intent: str) -> str:
    return ROUTE_NEXT_GATE.get(route_intent, ROUTE_NEXT_GATE["inspect_blocked_raw_reference_bundle"])


def next_command_intent(route_intent: str) -> str:
    return ROUTE_NEXT_COMMAND_INTENT.get(route_intent, ROUTE_NEXT_COMMAND_INTENT["inspect_blocked_raw_reference_bundle"])


def target_stem(row: dict[str, Any]) -> str:
    stem = str(row.get("target_stem_type") or "")
    if stem:
        return stem
    stems = str_list(row.get("requested_stems"))
    return stems[0] if len(stems) == 1 else ""


def build_item(index: int, row: dict[str, Any]) -> dict[str, Any]:
    route = str(row.get("route_intent") or "inspect_blocked_raw_reference_bundle")
    requested_stems = str_list(row.get("requested_stems"))
    return {
        "schema": ITEM_SCHEMA,
        "action_item_id": f"a{index:03d}",
        "source_queue_item_id": str(row.get("queue_item_id") or ""),
        "source_bundle_alias": str(row.get("source_bundle_alias") or ""),
        "source_work_item_aliases": str_list(row.get("source_work_item_aliases")),
        "route_intent": route,
        "priority": str(row.get("priority") or "normal"),
        "next_gate": next_gate(route),
        "next_command_intent": next_command_intent(route),
        "execution_mode": "metadata_only_external_reference_recovery",
        "requires_verified_reference_material": True,
        "docker_required_for_next_rerun": True,
        "next_docker_service": "ml-worker",
        "requested_stems": requested_stems,
        "target_stem_type": target_stem(row),
        "verified_stems": str_list(row.get("verified_stems")),
        "blockers": str_list(row.get("blockers")),
        "issue_tags": str_list(row.get("issue_tags")),
        "required_actions": str_list(row.get("required_actions")),
        "candidate_directory": str(row.get("candidate_directory") or ""),
        "source_mix_candidate_path": str(row.get("source_mix_candidate_path") or ""),
        "source_mix_resolution": str(row.get("source_mix_resolution") or ""),
        "source_mix_manifest_json": str(row.get("source_mix_manifest_json") or ""),
        "diagnostic_bundle_root": str(row.get("diagnostic_bundle_root") or ""),
        "diagnostic_manifest_json": str(row.get("diagnostic_manifest_json") or ""),
        "full_mix_reconstruction_status": str(row.get("full_mix_reconstruction_status") or ""),
        "failure_detail": str(row.get("failure_detail") or ""),
        "rights_status": str(row.get("rights_status") or ""),
        "bundle_build_attempted": row.get("bundle_build_attempted") is True,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "diagnostic_context": row.get("diagnostic_context") if isinstance(row.get("diagnostic_context"), dict) else {},
    }


def count_by_stem(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(stem for row in rows for stem in row["requested_stems"])
    return dict(sorted(counts.items()))


def count_field(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "") for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def action_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(action for row in rows for action in row["required_actions"])
    return dict(sorted(counts.items()))


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Recovery Action Plan",
            "",
            f"- action_plan_status: `{manifest['action_plan_status']}`",
            f"- action_item_count: `{manifest['action_item_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_full_pack_raw_reference_recovery_action_plan(
    *,
    raw_reference_recovery_queue_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", raw_reference_recovery_queue_json=str(raw_reference_recovery_queue_json))

    source = read_json(raw_reference_recovery_queue_json)
    failed = collect_true_flag_failures("raw_reference_recovery_queue", source)
    if source.get("schema") != SOURCE_SCHEMA:
        failed.append("raw_reference_recovery_queue_schema_unexpected")
    queue_failed = [str(item) for item in source.get("failed_checks") or []]
    if queue_failed:
        failed.append("raw_reference_recovery_queue_failed_checks_not_empty")
    queue_status = str(source.get("queue_status") or "")
    if queue_status not in {"ready_for_raw_reference_recovery", "no_raw_reference_recovery_needed", "blocked"}:
        failed.append("raw_reference_recovery_queue_status_unexpected")
    if queue_status == "blocked":
        failed.append("raw_reference_recovery_queue_blocked")
    recovery_rows, row_failed = load_recovery_rows(raw_reference_recovery_queue_json, source)
    failed.extend(row_failed)
    failed = stable_unique(failed)

    rows = [] if failed else [build_item(index, row) for index, row in enumerate(recovery_rows, start=1)]
    if failed:
        status = "blocked"
        next_action = "inspect_failed_raw_reference_recovery_queue"
    elif rows:
        status = "ready_for_raw_reference_recovery_actions"
        next_action = "execute_raw_reference_recovery_actions"
    else:
        status = "no_raw_reference_recovery_actions_needed"
        next_action = "none"

    route_counts = Counter(row["route_intent"] for row in rows)
    next_gate_counts = Counter(row["next_gate"] for row in rows)
    manifest = {
        "schema": PLAN_SCHEMA,
        "manifest_type": "full_pack_raw_reference_recovery_action_plan",
        "action_plan_status": status,
        "next_action": next_action,
        "failed_checks": failed,
        "source_failed_checks": queue_failed,
        "raw_reference_recovery_queue_json": str(raw_reference_recovery_queue_json),
        "action_item_count": len(rows),
        "route_intent_counts": dict(sorted(route_counts.items())),
        "next_gate_counts": dict(sorted(next_gate_counts.items())),
        "required_action_counts": action_counts(rows),
        "blocked_by_stem": count_by_stem(rows),
        "source_mix_resolution_counts": count_field(rows, "source_mix_resolution"),
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "items_jsonl": str(output_dir / ITEMS_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "raw_reference_recovery_action_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "accepted_reference_rows": 0,
        "eligible_reference_material_rows": 0,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "dataset_built": False,
            "training_started": False,
            "ground_truth_manifest_written": False,
            "not_release_artifact": True,
            "accepted_rows_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", action_plan_status=status, action_item_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-reference-recovery-queue-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_raw_reference_recovery_action_plan.py",
        entrypoint_path="/app/tools/build_full_pack_raw_reference_recovery_action_plan.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_queue_json=args.raw_reference_recovery_queue_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "action_plan_status": manifest["action_plan_status"],
                "action_item_count": manifest["action_item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
