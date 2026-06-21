#!/usr/bin/env python3
"""Execute metadata-only handoff requests from a raw-reference recovery action plan."""

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
MANIFEST_NAME = "exec.json"
ITEMS_NAME = "items.jsonl"
BLOCKED_NAME = "blocked.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
TEMPLATE_DIR = "tpl"
SOURCE_SCHEMA = "full_pack_raw_reference_recovery_action_plan_manifest.v1"
ROW_SCHEMA = "full_pack_raw_reference_recovery_action_item.v1"
MANIFEST_SCHEMA = "full_pack_raw_reference_recovery_action_execution_manifest.v1"
ITEM_SCHEMA = "full_pack_raw_reference_recovery_action_execution_item.v1"
TEMPLATE_SCHEMA = "full_pack_raw_reference_recovery_execution_template.v1"
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
    "accepted_reference_manifest_json",
)
ROUTE_EXECUTION_STATUS = {
    "collect_clean_reference_bundle": "blocked_waiting_for_clean_reference_bundle",
    "repair_or_replace_reconstruction_failed_bundle": "blocked_waiting_for_clean_scoped_export",
    "replace_missing_requested_stem_candidate": "blocked_waiting_for_missing_requested_stem_candidate",
    "inspect_blocked_raw_reference_bundle": "blocked_waiting_for_raw_reference_triage",
}
ROUTE_REQUIRED_MATERIAL = {
    "collect_clean_reference_bundle": "clean_project_reference_bundle",
    "repair_or_replace_reconstruction_failed_bundle": "clean_scoped_reference_export",
    "replace_missing_requested_stem_candidate": "missing_requested_stem_candidate",
    "inspect_blocked_raw_reference_bundle": "raw_reference_triage_decision",
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
        "expected_acceleration_mode": "external_only_cpu_raw_reference_recovery_execution",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_recovery_execution_writes_metadata_requests_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_recovery_execution_never_starts_training",
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


def resolved_items_path(manifest: dict[str, Any], *, base_dir: Path) -> Path | None:
    outputs = manifest.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("items_jsonl") or "")
    text = text or str(manifest.get("items_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_action_items(plan_json: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    items_path = resolved_items_path(manifest, base_dir=plan_json.parent)
    expected_count = int(manifest.get("action_item_count") or 0)
    if items_path is None or not items_path.is_file():
        if expected_count > 0:
            failed.append("raw_reference_recovery_action_items_jsonl_missing")
        return [], failed
    rows = list(iter_jsonl(items_path))
    for row_index, row in enumerate(rows, start=1):
        row_prefix = f"raw_reference_recovery_action_row_{row_index:03d}"
        if row.get("schema") != ROW_SCHEMA:
            failed.append(f"{row_prefix}_schema_unexpected")
        failed.extend(collect_true_flag_failures(row_prefix, row))
        for field in BLOCKED_ROW_OUTPUT_FIELDS:
            if str(row.get(field) or ""):
                failed.append(f"{row_prefix}_{field}_present")
    if expected_count != len(rows):
        failed.append("raw_reference_recovery_action_item_count_mismatch")
    return rows if not failed else [], stable_unique(failed)


def execution_status(route_intent: str) -> str:
    return ROUTE_EXECUTION_STATUS.get(route_intent, ROUTE_EXECUTION_STATUS["inspect_blocked_raw_reference_bundle"])


def required_material(route_intent: str) -> str:
    return ROUTE_REQUIRED_MATERIAL.get(route_intent, ROUTE_REQUIRED_MATERIAL["inspect_blocked_raw_reference_bundle"])


def target_stem(row: dict[str, Any]) -> str:
    stem = str(row.get("target_stem_type") or "")
    if stem:
        return stem
    stems = str_list(row.get("requested_stems"))
    return stems[0] if len(stems) == 1 else ""


def template_payload(*, row: dict[str, Any], status: str) -> dict[str, Any]:
    stem = target_stem(row)
    route = str(row.get("route_intent") or "inspect_blocked_raw_reference_bundle")
    return {
        "schema": TEMPLATE_SCHEMA,
        "source_action_item_id": str(row.get("action_item_id") or ""),
        "source_bundle_alias": str(row.get("source_bundle_alias") or ""),
        "route_intent": route,
        "execution_status": status,
        "required_material": required_material(route),
        "target_stem_type": stem,
        "requested_stems": str_list(row.get("requested_stems")),
        "expected_files": {
            "source_mix": "source_mix.flac",
            "ground_truth_manifest": "ground_truth_manifest.json",
            "target_stem": f"stems/{stem}.flac" if stem else "",
        },
        "provenance_required": True,
        "trainable_rights_required": True,
        "exact_reference_waveform_audit_required": True,
        "ground_truth_manifest_written": False,
        "audio_placeholder_written": False,
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
    }


def build_execution_item(*, index: int, row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    route = str(row.get("route_intent") or "inspect_blocked_raw_reference_bundle")
    status = execution_status(route)
    action_id = f"x{index:03d}"
    template_path = output_dir / TEMPLATE_DIR / f"a{index:03d}.json"
    write_json(template_path, template_payload(row=row, status=status))
    return {
        "schema": ITEM_SCHEMA,
        "execution_item_id": action_id,
        "source_action_item_id": str(row.get("action_item_id") or ""),
        "source_queue_item_id": str(row.get("source_queue_item_id") or ""),
        "source_bundle_alias": str(row.get("source_bundle_alias") or ""),
        "source_work_item_aliases": str_list(row.get("source_work_item_aliases")),
        "route_intent": route,
        "next_gate": str(row.get("next_gate") or ""),
        "execution_status": status,
        "metadata_request_written": True,
        "template_json": str(template_path),
        "required_material": required_material(route),
        "requested_stems": str_list(row.get("requested_stems")),
        "target_stem_type": target_stem(row),
        "blockers": str_list(row.get("blockers")),
        "issue_tags": str_list(row.get("issue_tags")),
        "required_actions": str_list(row.get("required_actions")),
        "candidate_directory": str(row.get("candidate_directory") or ""),
        "source_mix_candidate_path": str(row.get("source_mix_candidate_path") or ""),
        "diagnostic_bundle_root": str(row.get("diagnostic_bundle_root") or ""),
        "diagnostic_manifest_json": str(row.get("diagnostic_manifest_json") or ""),
        "next_docker_service": str(row.get("next_docker_service") or "ml-worker"),
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def count_by_stem(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(stem for row in rows for stem in row["requested_stems"])
    return dict(sorted(counts.items()))


def count_field(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "") for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Recovery Action Execution",
            "",
            f"- execution_status: `{manifest['execution_status']}`",
            f"- action_item_count: `{manifest['action_item_count']}`",
            f"- completed_action_count: `{manifest['completed_action_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            "",
        ]
    )


def execute_full_pack_raw_reference_recovery_action_plan(
    *,
    raw_reference_recovery_action_plan_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", raw_reference_recovery_action_plan_json=str(raw_reference_recovery_action_plan_json))

    source = read_json(raw_reference_recovery_action_plan_json)
    failed = collect_true_flag_failures("raw_reference_recovery_action_plan", source)
    if source.get("schema") != SOURCE_SCHEMA:
        failed.append("raw_reference_recovery_action_plan_schema_unexpected")
    source_failed = [str(item) for item in source.get("failed_checks") or []]
    if source_failed:
        failed.append("raw_reference_recovery_action_plan_failed_checks_not_empty")
    plan_status = str(source.get("action_plan_status") or "")
    if plan_status not in {
        "ready_for_raw_reference_recovery_actions",
        "no_raw_reference_recovery_actions_needed",
        "blocked",
    }:
        failed.append("raw_reference_recovery_action_plan_status_unexpected")
    if plan_status == "blocked":
        failed.append("raw_reference_recovery_action_plan_blocked")
    action_rows, row_failed = load_action_items(raw_reference_recovery_action_plan_json, source)
    failed.extend(row_failed)
    failed = stable_unique(failed)

    rows = [] if failed else [
        build_execution_item(index=index, row=row, output_dir=output_dir)
        for index, row in enumerate(action_rows, start=1)
    ]
    status_counts = Counter(row["execution_status"] for row in rows)
    if failed:
        status = "blocked"
        next_action = "inspect_failed_raw_reference_recovery_action_plan"
    elif rows:
        status = "blocked_waiting_for_external_reference_material"
        next_action = "provide_external_reference_material_then_rerun_aris_loop"
    else:
        status = "no_raw_reference_recovery_execution_needed"
        next_action = "none"

    route_counts = Counter(row["route_intent"] for row in rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "full_pack_raw_reference_recovery_action_execution",
        "execution_status": status,
        "next_action": next_action,
        "failed_checks": failed,
        "source_failed_checks": source_failed,
        "raw_reference_recovery_action_plan_json": str(raw_reference_recovery_action_plan_json),
        "action_item_count": len(rows),
        "metadata_request_count": len(rows),
        "blocked_action_count": len(rows),
        "completed_action_count": 0,
        "route_intent_counts": dict(sorted(route_counts.items())),
        "execution_status_counts": dict(sorted(status_counts.items())),
        "blocked_by_stem": count_by_stem(rows),
        "required_material_counts": count_field(rows, "required_material"),
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "items_jsonl": str(output_dir / ITEMS_NAME),
            "blocked_jsonl": str(output_dir / BLOCKED_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
            "template_dir": str(output_dir / TEMPLATE_DIR),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "raw_reference_recovery_execution_not_for_listen_pack",
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
            "audio_placeholder_written": False,
            "not_release_artifact": True,
            "accepted_rows_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, rows)
    write_jsonl(output_dir / BLOCKED_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", execution_status=status, action_item_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-reference-recovery-action-plan-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="execute_full_pack_raw_reference_recovery_action_plan.py",
        entrypoint_path="/app/tools/execute_full_pack_raw_reference_recovery_action_plan.py",
    )
    args = parse_args(argv)
    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=args.raw_reference_recovery_action_plan_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "execution_status": manifest["execution_status"],
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
