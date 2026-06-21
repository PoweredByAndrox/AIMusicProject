#!/usr/bin/env python3
"""Build a safe handoff for collecting missing full-pack reference bundles."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "handoff.json"
ITEMS_NAME = "items.jsonl"
TEMPLATE_NAME = "tpl.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
SAFE_ALIAS_RE = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
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
        "expected_acceleration_mode": "external_only_cpu_reference_bundle_handoff",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_bundle_handoff_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "reference_bundle_handoff_no_training_started",
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


def normalized_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "vocals":
        return "vocal"
    if text == "instrumental":
        return "minus"
    return text or "unknown"


def int_mapping(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in payload.items():
        stem = normalized_stem(key)
        if stem == "unknown":
            continue
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            result[stem] = count
    return dict(sorted(result.items()))


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


def resolved_items_path(queue: dict[str, Any], *, base_dir: Path) -> Path | None:
    outputs = queue.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("items_jsonl") or "")
    text = text or str(queue.get("items_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def safe_alias(value: Any, index: int) -> str:
    candidate = str(value or "").strip()
    if SAFE_ALIAS_RE.fullmatch(candidate):
        return candidate
    return f"r{index:03d}"


def manifest_template_for_stem(stem: str) -> dict[str, Any]:
    return {
        "track_name": "<new_unique_track_id>",
        "ground_truth_contract": {
            "verified": True,
            "source": "FL Studio split mixer tracks plus full mix supplied by user",
        },
        "mix": {"path": "source_mix.flac"},
        "stems": {stem: {"path": f"stems/{stem}.flac"}},
        "report_alignment_contract": {
            "rights_status": "user_owned|owned|licensed|cleared|project_owned",
        },
    }


def str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "")]


def sanitized_raw_reference_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "schema": "full_pack_raw_reference_evidence.v1",
        "source_worklist_bundle_alias": str(value.get("source_worklist_bundle_alias") or ""),
        "source_collection_id": str(value.get("source_collection_id") or ""),
        "work_item_status": str(value.get("work_item_status") or ""),
        "raw_reference_candidate_matched": value.get("raw_reference_candidate_matched") is True,
        "candidate_directory": str(value.get("candidate_directory") or ""),
        "source_mix_candidate_path": str(value.get("source_mix_candidate_path") or ""),
        "target_stem_candidate_path": str(value.get("target_stem_candidate_path") or ""),
        "source_mix_resolution": str(value.get("source_mix_resolution") or ""),
        "source_mix_manifest_json": str(value.get("source_mix_manifest_json") or ""),
        "track_id_hint": str(value.get("track_id_hint") or ""),
        "required_actions": str_list(value.get("required_actions")),
        "blockers": str_list(value.get("blockers")),
        "requires_clean_reference_bundle": True,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
    }


def handoff_template() -> dict[str, Any]:
    return {
        "schema": "full_pack_reference_bundle_template.v1",
        "required_directory": "refs/<alias>",
        "required_manifest_name": "ground_truth_manifest.json",
        "required_files": ["source_mix.flac", "stems/<stem>.flac", "ground_truth_manifest.json"],
        "ground_truth_manifest_template": {
            "track_name": "<new_unique_track_id>",
            "ground_truth_contract": {
                "verified": True,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {"<stem>": {"path": "stems/<stem>.flac"}},
            "report_alignment_contract": {
                "rights_status": "user_owned|owned|licensed|cleared|project_owned",
            },
        },
        "accepted_source_provenance": "FL Studio split mixer tracks plus full mix supplied by user",
        "rejected_sources": ["synthetic_fixture", "runtime_candidate", "dirty_stem", "duplicate_existing_track_stem"],
        "followup_command": (
            "python /app/tools/run_full_pack_aris_metadata_loop.py "
            "--candidate-root <handoff_root>/refs/<alias> --output-dir <next_loop_out> ..."
        ),
    }


def build_handoff_item(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    stem = normalized_stem(row.get("stem_type"))
    alias = safe_alias(row.get("collection_id"), index)
    required_files = row.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        required_files = ["source_mix.flac", f"stems/{stem}.flac", "ground_truth_manifest.json"]
    acceptance_criteria = row.get("acceptance_criteria")
    if not isinstance(acceptance_criteria, list):
        acceptance_criteria = []
    item = {
        "schema": "full_pack_reference_bundle_handoff_item.v1",
        "bundle_alias": alias,
        "source_collection_id": str(row.get("collection_id") or ""),
        "stem_type": stem,
        "slot_index": row.get("slot_index"),
        "gap_id": str(row.get("gap_id") or ""),
        "queue_item_id": str(row.get("queue_item_id") or ""),
        "priority": str(row.get("priority") or ""),
        "missing_training_job": row.get("missing_training_job") is True,
        "required_directory": f"refs/{alias}",
        "required_manifest_name": "ground_truth_manifest.json",
        "required_files": [str(item) for item in required_files],
        "acceptance_criteria": [str(item) for item in acceptance_criteria],
        "ground_truth_manifest_template": manifest_template_for_stem(stem),
        "next_intake_command_args": ["--candidate-root", f"<handoff_root>/refs/{alias}"],
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
    }
    raw_evidence = sanitized_raw_reference_evidence(row.get("raw_reference_evidence"))
    if raw_evidence:
        item["raw_reference_evidence"] = raw_evidence
    return item


def load_collection_items(collection_queue_json: Path, queue: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    items_path = resolved_items_path(queue, base_dir=collection_queue_json.parent)
    if items_path is None or not items_path.is_file():
        failed.append("collection_queue_items_jsonl_missing")
        return [], failed
    rows = list(iter_jsonl(items_path))
    for row in rows:
        if row.get("schema") != "full_pack_reference_collection_item.v1":
            failed.append("collection_queue_item_schema_unexpected")
        failed.extend(collect_true_flag_failures("collection_item", row))
        stem = normalized_stem(row.get("stem_type"))
        if stem == "unknown":
            failed.append("collection_item_stem_type_missing")
            continue
        required_files = row.get("required_files")
        expected_files = {"source_mix.flac", f"stems/{stem}.flac", "ground_truth_manifest.json"}
        if not isinstance(required_files, list) or not expected_files.issubset({str(item) for item in required_files}):
            failed.append("collection_item_required_files_missing")
        if row.get("required_manifest_name") != "ground_truth_manifest.json":
            failed.append("collection_item_required_manifest_unexpected")
        raw_evidence = row.get("raw_reference_evidence")
        if raw_evidence is not None:
            if not isinstance(raw_evidence, dict):
                failed.append("collection_item_raw_reference_evidence_unexpected")
            else:
                if raw_evidence.get("schema") != "full_pack_raw_reference_evidence.v1":
                    failed.append("collection_item_raw_reference_evidence_schema_unexpected")
                failed.extend(collect_true_flag_failures("collection_item_raw_reference_evidence", raw_evidence))
    return rows if not failed else [], stable_unique(failed)


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Reference Bundle Handoff",
            "",
            f"- handoff_status: `{manifest['handoff_status']}`",
            f"- handoff_item_count: `{manifest['handoff_item_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def evidence_status_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        evidence = item.get("raw_reference_evidence")
        if not isinstance(evidence, dict):
            continue
        status = str(evidence.get("work_item_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_full_pack_reference_bundle_handoff(
    *,
    collection_queue_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", collection_queue_json=str(collection_queue_json))

    queue = read_json(collection_queue_json)
    failed = collect_true_flag_failures("collection_queue", queue)
    if queue.get("schema") != "full_pack_reference_collection_queue_manifest.v1":
        failed.append("collection_queue_schema_unexpected")
    if queue.get("queue_status") != "ready_for_reference_collection":
        failed.append("collection_queue_status_not_ready")
    if queue.get("failed_checks") not in ([], None):
        failed.append("collection_queue_failed_checks_not_empty")
    queue_item_count = int(queue.get("collection_item_count") or 0)
    if queue_item_count <= 0:
        failed.append("collection_queue_empty")
    raw_items, item_failed = load_collection_items(collection_queue_json, queue)
    if not item_failed and queue_item_count != len(raw_items):
        failed.append("collection_queue_item_count_mismatch")
    failed.extend(item_failed)
    failed = stable_unique(failed)

    items = [] if failed else [build_handoff_item(row, index=index) for index, row in enumerate(raw_items, start=1)]
    status = "blocked" if failed else "ready_for_reference_bundle_collection"
    raw_evidence_item_count = sum(1 for item in items if isinstance(item.get("raw_reference_evidence"), dict))
    manifest = {
        "schema": "full_pack_reference_bundle_handoff_manifest.v1",
        "manifest_type": "full_pack_reference_bundle_handoff",
        "handoff_status": status,
        "failed_checks": failed,
        "source_collection_queue_json": str(collection_queue_json),
        "handoff_item_count": len(items),
        "remaining_by_stem": {} if failed else int_mapping(queue.get("remaining_by_stem")),
        "raw_reference_evidence_item_count": raw_evidence_item_count,
        "raw_reference_evidence_by_status": evidence_status_counts(items),
        "required_bundle_root_layout": "refs/<alias>/",
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "items_jsonl": str(output_dir / ITEMS_NAME),
            "template_json": str(output_dir / TEMPLATE_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "reference_bundle_handoff_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "dataset_built": False,
            "training_started": False,
            "not_release_artifact": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, items)
    write_json(output_dir / TEMPLATE_NAME, handoff_template())
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", handoff_status=status, handoff_item_count=len(items), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a handoff for missing full-pack verified reference bundles.")
    parser.add_argument("--collection-queue-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_reference_bundle_handoff.py",
        entrypoint_path="/app/tools/build_full_pack_reference_bundle_handoff.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_reference_bundle_handoff(
        collection_queue_json=args.collection_queue_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "handoff_status": manifest["handoff_status"],
                "handoff_item_count": manifest["handoff_item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
