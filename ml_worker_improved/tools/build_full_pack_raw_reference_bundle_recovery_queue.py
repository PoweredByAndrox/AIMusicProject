#!/usr/bin/env python3
"""Build recovery actions for blocked full-pack raw-reference bundle attempts."""

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
MANIFEST_NAME = "queue.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
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
LISTEN_PACK_SOURCE_BLOCKERS = {
    "source_mix_from_listen_pack_manifest_requires_clean_reference_bundle",
    "source_not_under_allowed_root",
}
RECONSTRUCTION_BLOCKERS = {
    "full_mix_reconstruction_not_accepted",
    "input_scope_not_clean",
}
MISSING_STEM_PREFIX = "verified_bundle_missing_requested_stem:"


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
        "expected_acceleration_mode": "external_only_cpu_raw_reference_recovery_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_recovery_queue_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_recovery_queue_no_training_started",
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


def resolved_blocked_path(manifest: dict[str, Any], *, base_dir: Path) -> Path | None:
    outputs = manifest.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("blocked_jsonl") or "")
    text = text or str(manifest.get("blocked_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_blocked_rows(raw_verified_bundles_json: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    blocked_path = resolved_blocked_path(manifest, base_dir=raw_verified_bundles_json.parent)
    blocked_count = int(manifest.get("blocked_bundle_count") or 0)
    if blocked_path is None or not blocked_path.is_file():
        if blocked_count > 0:
            failed.append("raw_verified_bundles_blocked_jsonl_missing")
        return [], failed
    rows = list(iter_jsonl(blocked_path))
    for row_index, row in enumerate(rows, start=1):
        row_prefix = f"raw_verified_bundle_row_{row_index:03d}"
        if row.get("schema") != "full_pack_raw_reference_verified_bundle_row.v1":
            failed.append(f"{row_prefix}_schema_unexpected")
        failed.extend(collect_true_flag_failures(row_prefix, row))
        for field in BLOCKED_ROW_OUTPUT_FIELDS:
            if str(row.get(field) or ""):
                failed.append(f"{row_prefix}_{field}_present")
    if blocked_count != len(rows):
        failed.append("raw_verified_bundles_blocked_count_mismatch")
    return rows if not failed else [], stable_unique(failed)


def missing_stem_from_blockers(blockers: Sequence[str]) -> str:
    for blocker in blockers:
        if blocker.startswith(MISSING_STEM_PREFIX):
            return blocker.removeprefix(MISSING_STEM_PREFIX)
    return ""


def route_intent(blockers: Sequence[str]) -> str:
    blocker_set = set(blockers)
    if "missing_raw_reference_candidate" in blocker_set or any(item.startswith(MISSING_STEM_PREFIX) for item in blockers):
        return "replace_missing_requested_stem_candidate"
    if blocker_set & LISTEN_PACK_SOURCE_BLOCKERS:
        return "collect_clean_reference_bundle"
    if blocker_set & RECONSTRUCTION_BLOCKERS or any(item.startswith("bundle_build_failed:") for item in blockers):
        return "repair_or_replace_reconstruction_failed_bundle"
    return "inspect_blocked_raw_reference_bundle"


def issue_tags(blockers: Sequence[str]) -> list[str]:
    tags: list[str] = []
    for blocker in blockers:
        if blocker == "source_mix_from_listen_pack_manifest_requires_clean_reference_bundle":
            tags.append("listen_pack_source_mix_provenance")
        elif blocker == "source_not_under_allowed_root":
            tags.append("source_outside_allowed_root")
        elif blocker == "full_mix_reconstruction_not_accepted":
            tags.append("full_mix_reconstruction_failed")
        elif blocker == "input_scope_not_clean":
            tags.append("input_scope_not_clean")
        elif blocker == "missing_raw_reference_candidate":
            tags.append("raw_reference_candidate_missing")
        elif blocker.startswith(MISSING_STEM_PREFIX):
            tags.append("requested_stem_missing")
        elif blocker.startswith("bundle_build_failed:"):
            tags.append("bundle_build_failed")
        else:
            tags.append(blocker)
    return stable_unique(tags)


def required_actions(intent: str) -> list[str]:
    if intent == "collect_clean_reference_bundle":
        return [
            "collect_clean_project_reference_export",
            "verify_source_provenance",
            "verify_trainable_rights",
            "write_verified_ground_truth_manifest",
            "rerun_raw_reference_verified_bundle_bridge",
        ]
    if intent == "replace_missing_requested_stem_candidate":
        return [
            "collect_requested_stem_candidate",
            "rerun_raw_reference_bundle_worklist",
            "rerun_raw_reference_verified_bundle_bridge",
        ]
    if intent == "repair_or_replace_reconstruction_failed_bundle":
        return [
            "provide_clean_scoped_export",
            "repair_or_replace_source_mix_reconstruction",
            "rerun_full_mix_reconstruction_audit",
            "rerun_raw_reference_verified_bundle_bridge",
        ]
    return [
        "inspect_raw_reference_blockers",
        "rerun_raw_reference_verified_bundle_bridge",
    ]


def target_stem(row: dict[str, Any], blockers: Sequence[str]) -> str:
    missing = missing_stem_from_blockers(blockers)
    if missing:
        return missing
    stems = str_list(row.get("requested_stems"))
    return stems[0] if len(stems) == 1 else ""


def build_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    blockers = str_list(row.get("blockers"))
    intent = route_intent(blockers)
    requested_stems = str_list(row.get("requested_stems"))
    return {
        "schema": "full_pack_raw_reference_bundle_recovery_item.v1",
        "queue_item_id": f"q{index:03d}",
        "source_bundle_alias": str(row.get("bundle_alias") or ""),
        "source_verified_bundle_status": str(row.get("verified_bundle_status") or ""),
        "route_intent": intent,
        "priority": "high" if intent != "inspect_blocked_raw_reference_bundle" else "normal",
        "requested_stems": requested_stems,
        "target_stem_type": target_stem(row, blockers),
        "verified_stems": str_list(row.get("verified_stems")),
        "blockers": blockers,
        "issue_tags": issue_tags(blockers),
        "required_actions": required_actions(intent),
        "source_work_item_aliases": str_list(row.get("source_work_item_aliases")),
        "source_work_item_count": int(row.get("source_work_item_count") or 0),
        "candidate_directory": str(row.get("candidate_directory") or ""),
        "source_mix_candidate_path": str(row.get("source_mix_candidate_path") or ""),
        "source_mix_resolution": str(row.get("source_mix_resolution") or ""),
        "source_mix_manifest_json": str(row.get("source_mix_manifest_json") or ""),
        "diagnostic_bundle_root": str(row.get("diagnostic_bundle_root") or ""),
        "diagnostic_manifest_json": str(row.get("diagnostic_manifest_json") or ""),
        "ground_truth_manifest_json": str(row.get("ground_truth_manifest_json") or ""),
        "candidate_root": str(row.get("candidate_root") or ""),
        "full_mix_reconstruction_status": str(row.get("full_mix_reconstruction_status") or ""),
        "failure_detail": str(row.get("failure_detail") or ""),
        "rights_status": str(row.get("rights_status") or ""),
        "bundle_build_attempted": row.get("bundle_build_attempted") is True,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "diagnostic_context": {
            "source_blockers": blockers,
            "source_mix_resolution": str(row.get("source_mix_resolution") or ""),
            "source_mix_manifest_json": str(row.get("source_mix_manifest_json") or ""),
            "diagnostic_manifest_json": str(row.get("diagnostic_manifest_json") or ""),
            "source_work_item_aliases": str_list(row.get("source_work_item_aliases")),
        },
    }


def count_by_stem(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(stem for row in rows for stem in row["requested_stems"])
    return dict(sorted(counts.items()))


def count_field(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "") for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def blocker_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(blocker for row in rows for blocker in row["blockers"])
    return dict(sorted(counts.items()))


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Bundle Recovery Queue",
            "",
            f"- queue_status: `{manifest['queue_status']}`",
            f"- queue_item_count: `{manifest['queue_item_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_full_pack_raw_reference_bundle_recovery_queue(
    *,
    raw_verified_bundles_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", raw_verified_bundles_json=str(raw_verified_bundles_json))

    source = read_json(raw_verified_bundles_json)
    failed = collect_true_flag_failures("raw_verified_bundles", source)
    if source.get("schema") != "full_pack_raw_reference_verified_bundles_manifest.v1":
        failed.append("raw_verified_bundles_schema_unexpected")
    if source.get("failed_checks") not in ([], None):
        failed.append("raw_verified_bundles_failed_checks_not_empty")
    blocked_rows, row_failed = load_blocked_rows(raw_verified_bundles_json, source)
    failed.extend(row_failed)
    failed = stable_unique(failed)

    rows = [] if failed else [build_row(index, row) for index, row in enumerate(blocked_rows, start=1)]
    if failed:
        status = "blocked"
    elif rows:
        status = "ready_for_raw_reference_recovery"
    else:
        status = "no_raw_reference_recovery_needed"
    route_counts = Counter(row["route_intent"] for row in rows)
    manifest = {
        "schema": "full_pack_raw_reference_bundle_recovery_queue_manifest.v1",
        "manifest_type": "full_pack_raw_reference_bundle_recovery_queue",
        "queue_status": status,
        "queue_source": "raw_reference_verified_bundles",
        "failed_checks": failed,
        "raw_verified_bundles_json": str(raw_verified_bundles_json),
        "queue_item_count": len(rows),
        "route_intent_counts": dict(sorted(route_counts.items())),
        "blocked_by_stem": count_by_stem(rows),
        "blocker_counts": blocker_counts(rows),
        "source_mix_resolution_counts": count_field(rows, "source_mix_resolution"),
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
        "presentation_status": "raw_reference_bundle_recovery_queue_not_for_listen_pack",
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
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", queue_status=status, queue_item_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recovery actions for blocked full-pack raw-reference bundle attempts.")
    parser.add_argument("--raw-verified-bundles-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_raw_reference_bundle_recovery_queue.py",
        entrypoint_path="/app/tools/build_full_pack_raw_reference_bundle_recovery_queue.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_raw_reference_bundle_recovery_queue(
        raw_verified_bundles_json=args.raw_verified_bundles_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "queue_status": manifest["queue_status"],
                "queue_item_count": manifest["queue_item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
