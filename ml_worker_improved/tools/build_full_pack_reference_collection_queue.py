#!/usr/bin/env python3
"""Build collection slots for missing full-pack verified reference stems."""

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
MANIFEST_NAME = "queue.json"
ITEMS_NAME = "items.jsonl"
TEMPLATE_NAME = "tpl.json"
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
        "expected_acceleration_mode": "external_only_cpu_reference_collection_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_collection_queue_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "reference_collection_queue_no_training_started",
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


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def resolved_rows_path(plan: dict[str, Any], *, base_dir: Path) -> Path | None:
    outputs = plan.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("rows_jsonl") or "")
    text = text or str(plan.get("rows_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


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


def load_gap_rows(gap_plan_json: Path) -> tuple[list[dict[str, Any]], list[str]]:
    plan = read_json(gap_plan_json)
    failed = collect_true_flag_failures("reference_gap", plan)
    if plan.get("schema") != "full_pack_reference_corpus_gap_plan.v1":
        failed.append("reference_gap_schema_unexpected")
    if plan.get("failed_checks") not in ([], None):
        failed.append("reference_gap_failed_checks_not_empty")
    rows_path = resolved_rows_path(plan, base_dir=gap_plan_json.parent)
    if rows_path is None or not rows_path.is_file():
        failed.append("reference_gap_rows_jsonl_missing")
        return [], stable_unique(failed)
    rows = list(iter_jsonl(rows_path))
    if any(row.get("schema") != "full_pack_reference_corpus_gap.v1" for row in rows):
        failed.append("reference_gap_row_schema_unexpected")
    return rows if not failed else [], stable_unique(failed)


def remaining_counts(intake: dict[str, Any], gap_rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    raw = intake.get("remaining_by_stem")
    if isinstance(raw, dict):
        return {
            normalized_stem(stem): int_value(count)
            for stem, count in raw.items()
            if normalized_stem(stem) != "unknown" and int_value(count) > 0
        }
    counts: dict[str, int] = {}
    for row in gap_rows:
        stem = normalized_stem(row.get("stem_type"))
        count = int_value(row.get("required_reference_stem_count") or row.get("missing_unique_track_count"))
        if stem != "unknown" and count > 0:
            counts[stem] = count
    return counts


def gap_by_stem(gap_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in gap_rows:
        stem = normalized_stem(row.get("stem_type"))
        if stem != "unknown":
            result.setdefault(stem, row)
    return result


def priority_sort(stem: str, gap: dict[str, Any]) -> tuple[str, str]:
    return (str(gap.get("priority") or "P9"), stem)


def str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "")]


def raw_reference_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "full_pack_raw_reference_evidence.v1",
        "source_worklist_bundle_alias": str(row.get("bundle_alias") or ""),
        "source_collection_id": str(row.get("source_collection_id") or ""),
        "work_item_status": str(row.get("work_item_status") or ""),
        "raw_reference_candidate_matched": row.get("raw_reference_candidate_matched") is True,
        "candidate_directory": str(row.get("candidate_directory") or ""),
        "source_mix_candidate_path": str(row.get("source_mix_candidate_path") or ""),
        "target_stem_candidate_path": str(row.get("target_stem_candidate_path") or ""),
        "source_mix_resolution": str(row.get("source_mix_resolution") or ""),
        "source_mix_manifest_json": str(row.get("source_mix_manifest_json") or ""),
        "track_id_hint": str(row.get("track_id_hint") or ""),
        "required_actions": str_list(row.get("required_actions")),
        "blockers": str_list(row.get("blockers")),
        "requires_clean_reference_bundle": True,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
    }


def evidence_keys(row: dict[str, Any]) -> list[tuple[str, ...]]:
    stem = normalized_stem(row.get("stem_type"))
    source_collection_id = str(row.get("source_collection_id") or row.get("collection_id") or "")
    gap_id = str(row.get("gap_id") or "")
    queue_item_id = str(row.get("queue_item_id") or "")
    slot_index = str(row.get("slot_index") or "")
    keys: list[tuple[str, ...]] = []
    if source_collection_id and stem != "unknown" and slot_index:
        keys.append(("collection_stem_slot", source_collection_id, stem, slot_index))
    if stem != "unknown" and (gap_id or queue_item_id):
        keys.append(("slot", gap_id, queue_item_id, stem, slot_index))
    return keys


def load_raw_reference_evidence(raw_reference_worklist_json: Path | None) -> tuple[dict[tuple[str, ...], dict[str, Any]], list[str]]:
    if raw_reference_worklist_json is None:
        return {}, []
    manifest = read_json(raw_reference_worklist_json)
    failed = collect_true_flag_failures("raw_reference_worklist", manifest)
    if manifest.get("schema") != "full_pack_raw_reference_bundle_worklist.v1":
        failed.append("raw_reference_worklist_schema_unexpected")
    if manifest.get("failed_checks") not in ([], None):
        failed.append("raw_reference_worklist_failed_checks_not_empty")
    items_path = resolved_items_path(manifest, base_dir=raw_reference_worklist_json.parent)
    if items_path is None or not items_path.is_file():
        failed.append("raw_reference_worklist_items_jsonl_missing")
        return {}, stable_unique(failed)
    rows = list(iter_jsonl(items_path))
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != "full_pack_raw_reference_bundle_work_item.v1":
            failed.append("raw_reference_work_item_schema_unexpected")
        failed.extend(collect_true_flag_failures("raw_reference_work_item", row))
        evidence = raw_reference_evidence(row)
        for key in evidence_keys(row):
            by_key.setdefault(key, evidence)
    return ({} if failed else by_key), stable_unique(failed)


def evidence_for_item(item: dict[str, Any], by_key: dict[tuple[str, ...], dict[str, Any]]) -> dict[str, Any] | None:
    for key in evidence_keys(item):
        evidence = by_key.get(key)
        if evidence is not None:
            return dict(evidence)
    return None


def evidence_status_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        evidence = item.get("raw_reference_evidence")
        if not isinstance(evidence, dict):
            continue
        status = str(evidence.get("work_item_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_items(
    remaining: dict[str, int],
    gaps: dict[str, dict[str, Any]],
    *,
    raw_reference_evidence_by_key: dict[tuple[str, ...], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    evidence_by_key = raw_reference_evidence_by_key or {}
    ordered_stems = sorted(remaining, key=lambda stem: priority_sort(stem, gaps.get(stem, {})))
    items: list[dict[str, Any]] = []
    for stem in ordered_stems:
        gap = gaps.get(stem, {})
        for slot_index in range(1, remaining[stem] + 1):
            item = {
                "schema": "full_pack_reference_collection_item.v1",
                "collection_id": f"r{len(items) + 1:03d}",
                "stem_type": stem,
                "slot_index": slot_index,
                "gap_id": str(gap.get("gap_id") or ""),
                "queue_item_id": str(gap.get("queue_item_id") or ""),
                "priority": str(gap.get("priority") or ""),
                "missing_training_job": gap.get("missing_training_job") is True,
                "required_manifest_name": "ground_truth_manifest.json",
                "required_files": ["source_mix.flac", f"stems/{stem}.flac", "ground_truth_manifest.json"],
                "acceptance_criteria": [
                    "new_unique_track_id",
                    "new_track_stem_pair",
                    "ground_truth_contract_verified_true",
                    "user_supplied_flstudio_music_source",
                    "trainable_permission_status",
                    "source_and_target_audio_paths_exist",
                    "passes_full_pack_reference_corpus_intake",
                ],
                "candidate_intake_command_args": ["--candidate-root", "<candidate_bundle_root>"],
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            }
            evidence = evidence_for_item(item, evidence_by_key)
            if evidence is not None:
                item["raw_reference_evidence"] = evidence
            items.append(item)
    return items


def collection_template() -> dict[str, Any]:
    return {
        "schema": "full_pack_reference_collection_template.v1",
        "required_manifest_name": "ground_truth_manifest.json",
        "required_manifest_fields": {
            "track_name": "<new_unique_track_id>",
            "ground_truth_contract.verified": True,
            "ground_truth_contract.source": "FL Studio split mixer tracks plus full mix supplied by user",
            "mix.path": "source_mix.flac",
            "stems.<stem>.path": "stems/<stem>.flac",
            "report_alignment_contract.rights_status": "user_owned|owned|licensed|cleared|project_owned",
        },
        "accepted_source_provenance": "FL Studio split mixer tracks plus full mix supplied by user",
        "rejected_sources": ["synthetic_fixture", "runtime_candidate", "dirty_stem", "duplicate_existing_track_stem"],
        "followup_command": (
            "python /app/tools/build_full_pack_reference_corpus_intake.py "
            "--gap-plan-json <gap.json> --existing-audio-training-examples-jsonl <approved.jsonl> "
            "--candidate-root <candidate_bundle_root> --output-dir <intake_out>"
        ),
    }


def build_full_pack_reference_collection_queue(
    *,
    cycle_status_json: Path,
    reference_gap_plan_json: Path,
    reference_intake_json: Path,
    output_dir: Path,
    raw_reference_worklist_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        cycle_status_json=str(cycle_status_json),
        reference_gap_plan_json=str(reference_gap_plan_json),
        reference_intake_json=str(reference_intake_json),
        raw_reference_worklist_json=str(raw_reference_worklist_json or ""),
    )

    cycle = read_json(cycle_status_json)
    intake = read_json(reference_intake_json)
    gap_rows, gap_failed = load_gap_rows(reference_gap_plan_json)
    raw_evidence_by_key, raw_evidence_failed = load_raw_reference_evidence(raw_reference_worklist_json)
    failed = [
        *collect_true_flag_failures("cycle_status", cycle),
        *collect_true_flag_failures("reference_intake", intake),
        *gap_failed,
        *raw_evidence_failed,
    ]
    if cycle.get("schema") != "full_pack_aris_cycle_status.v1":
        failed.append("cycle_status_schema_unexpected")
    if cycle.get("failed_checks") not in ([], None):
        failed.append("cycle_status_failed_checks_not_empty")
    if cycle.get("next_action") != "collect_verified_reference_corpus":
        failed.append("cycle_status_next_action_not_reference_collection")
    if intake.get("schema") != "full_pack_reference_corpus_intake_manifest.v1":
        failed.append("reference_intake_schema_unexpected")
    if intake.get("failed_checks") not in ([], None):
        failed.append("reference_intake_failed_checks_not_empty")
    failed = stable_unique(failed)
    remaining = {} if failed else remaining_counts(intake, gap_rows)
    items = [] if failed else build_items(
        remaining,
        gap_by_stem(gap_rows),
        raw_reference_evidence_by_key=raw_evidence_by_key,
    )
    status = "blocked" if failed else ("ready_for_reference_collection" if items else "no_reference_collection_needed")
    raw_evidence_item_count = sum(1 for item in items if isinstance(item.get("raw_reference_evidence"), dict))
    manifest = {
        "schema": "full_pack_reference_collection_queue_manifest.v1",
        "manifest_type": "full_pack_reference_collection_queue",
        "queue_status": status,
        "failed_checks": failed,
        "cycle_status_json": str(cycle_status_json),
        "reference_gap_plan_json": str(reference_gap_plan_json),
        "reference_intake_json": str(reference_intake_json),
        "raw_reference_worklist_json": str(raw_reference_worklist_json or ""),
        "collection_item_count": len(items),
        "remaining_by_stem": remaining,
        "raw_reference_evidence_item_count": raw_evidence_item_count,
        "raw_reference_evidence_by_status": evidence_status_counts(items),
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
        "presentation_status": "reference_collection_queue_not_for_listen_pack",
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
    write_jsonl(output_dir / ITEMS_NAME, items)
    write_json(output_dir / TEMPLATE_NAME, collection_template())
    write_json(output_dir / MANIFEST_NAME, manifest)
    summary_lines = [
        "# Full Pack Reference Collection Queue",
        "",
        f"- queue_status: `{status}`",
        f"- collection_item_count: `{len(items)}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
    ]
    (output_dir / SUMMARY_NAME).write_text("\n".join(summary_lines), encoding="utf-8")
    progress(progress_path, "finish", queue_status=status, collection_item_count=len(items), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build collection slots for missing full-pack verified reference stems.")
    parser.add_argument("--cycle-status-json", type=Path, required=True)
    parser.add_argument("--reference-gap-plan-json", type=Path, required=True)
    parser.add_argument("--reference-intake-json", type=Path, required=True)
    parser.add_argument("--raw-reference-worklist-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_reference_collection_queue.py",
        entrypoint_path="/app/tools/build_full_pack_reference_collection_queue.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_reference_collection_queue(
        cycle_status_json=args.cycle_status_json,
        reference_gap_plan_json=args.reference_gap_plan_json,
        reference_intake_json=args.reference_intake_json,
        output_dir=args.output_dir,
        raw_reference_worklist_json=args.raw_reference_worklist_json,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "queue_status": manifest["queue_status"],
                "collection_item_count": manifest["collection_item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
