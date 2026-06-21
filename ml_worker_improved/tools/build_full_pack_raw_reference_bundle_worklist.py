#!/usr/bin/env python3
"""Build short work items from raw reference candidates without accepting them."""

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

from tools.build_full_pack_reference_corpus_intake import stable_unique  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "work.json"
ITEMS_NAME = "items.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
TEMPLATE_NAME = "tpl.json"
BUILDABLE_STATUS = "needs_verified_ground_truth_manifest"
MISSING_RAW_STATUS = "blocked_missing_raw_reference_candidate"
BLOCKED_PROVENANCE_STATUS = "blocked_raw_reference_provenance"
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
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
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
        "expected_acceleration_mode": "external_only_cpu_raw_reference_worklist",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_worklist_writes_metadata_templates_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_worklist_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    nested = payload.get("safety")
    if isinstance(nested, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    return failed


def resolved_rows_path(raw_candidates_json: Path, raw_manifest: dict[str, Any]) -> Path | None:
    outputs = raw_manifest.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("rows_jsonl") or "")
    text = text or str(raw_manifest.get("rows_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = raw_candidates_json.parent / path
    return path


def raw_rows(raw_candidates_json: Path, raw_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows_path = resolved_rows_path(raw_candidates_json, raw_manifest)
    if rows_path is None or not rows_path.is_file():
        return [], ["raw_candidates_rows_jsonl_missing"]
    rows = list(iter_jsonl(rows_path))
    failed: list[str] = []
    for row in rows:
        if row.get("schema") != "full_pack_raw_reference_candidate_row.v1":
            failed.append("raw_candidate_row_schema_unexpected")
        failed.extend(collect_true_flag_failures("raw_candidate_row", row))
        if row.get("triage_status") != "needs_verified_ground_truth_manifest":
            failed.append("raw_candidate_row_status_unexpected")
    return (rows if not failed else []), stable_unique(failed)


def normalize_stem(value: Any) -> str:
    stem = str(value or "").strip().lower().replace(" ", "_")
    if stem == "vocals":
        return "vocal"
    if stem == "instrumental":
        return "minus"
    return stem or "unknown"


def resolved_handoff_items_path(handoff_json: Path, handoff: dict[str, Any]) -> Path | None:
    outputs = handoff.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("items_jsonl") or "")
    text = text or str(handoff.get("items_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = handoff_json.parent / path
    return path


def handoff_rows(handoff_json: Path) -> tuple[list[dict[str, Any]], list[str]]:
    handoff = read_json(handoff_json)
    failed = collect_true_flag_failures("handoff", handoff)
    if handoff.get("schema") != "full_pack_reference_bundle_handoff_manifest.v1":
        failed.append("handoff_schema_unexpected")
    if handoff.get("handoff_status") != "ready_for_reference_bundle_collection":
        failed.append("handoff_status_not_ready")
    if handoff.get("failed_checks") not in ([], None):
        failed.append("handoff_failed_checks_not_empty")
    expected_count = int(handoff.get("handoff_item_count") or 0)
    if expected_count <= 0:
        failed.append("handoff_empty")
    items_path = resolved_handoff_items_path(handoff_json, handoff)
    if items_path is None or not items_path.is_file():
        failed.append("handoff_items_jsonl_missing")
        return [], stable_unique(failed)
    rows = list(iter_jsonl(items_path))
    if expected_count != len(rows):
        failed.append("handoff_item_count_mismatch")
    for row in rows:
        if row.get("schema") != "full_pack_reference_bundle_handoff_item.v1":
            failed.append("handoff_item_schema_unexpected")
        failed.extend(collect_true_flag_failures("handoff_item", row))
        stem = normalize_stem(row.get("stem_type"))
        if stem == "unknown":
            failed.append("handoff_item_stem_type_missing")
        alias = str(row.get("bundle_alias") or "")
        required_directory = str(row.get("required_directory") or "")
        if not alias:
            failed.append("handoff_item_bundle_alias_missing")
        if required_directory and required_directory != f"refs/{alias}":
            failed.append("handoff_item_required_directory_alias_mismatch")
    return (rows if not failed else []), stable_unique(failed)


def template_for(row: dict[str, Any], *, stem: str) -> dict[str, Any]:
    source_mix_resolution = str(
        row.get("source_mix_resolution")
        or ("same_directory_audio" if str(row.get("source_mix_candidate_path") or "") else "")
    )
    return {
        "track_name": "<new_unique_track_id>",
        "ground_truth_contract": {
            "verified": "<must_be_true_after_verification>",
            "source": "FL Studio split mixer tracks plus full mix supplied by user",
        },
        "mix": {"path": "source_mix.flac"},
        "stems": {stem: {"path": f"stems/{stem}.flac"}},
        "report_alignment_contract": {"rights_status": "<owned|licensed|cleared|project_owned|user_owned>"},
        "source_candidates": {
            "candidate_source_mix_path": str(row.get("source_mix_candidate_path") or ""),
            "candidate_source_mix_resolution": source_mix_resolution,
            "candidate_source_mix_manifest_json": str(row.get("source_mix_manifest_json") or ""),
            "candidate_target_stem_path": str(row.get("target_stem_candidate_path") or ""),
        },
        "required_before_intake": [
            "copy_or_render_source_mix_flac_from_candidate_after_rights_verification",
            "copy_or_render_target_stem_flac_from_candidate_after_rights_verification",
            "set_ground_truth_contract.verified=true only after provenance and waveform audit",
            "write this template as ground_truth_manifest.json only after verification",
        ],
    }


def slot_value(slot: dict[str, Any] | None, key: str, default: Any = "") -> Any:
    if not slot:
        return default
    value = slot.get(key)
    return default if value is None else value


def required_files_for(stem: str, slot: dict[str, Any] | None) -> list[str]:
    raw = slot_value(slot, "required_files", None)
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    return ["source_mix.flac", f"stems/{stem}.flac", "ground_truth_manifest.json"]


def raw_candidate_blockers(row: dict[str, Any]) -> list[str]:
    blockers = [str(item) for item in row.get("blockers") or [] if str(item or "")]
    blockers.extend(str(item) for item in row.get("provenance_blockers") or [] if str(item or ""))
    if str(row.get("source_mix_resolution") or "") == "pack_manifest_source_audio_path":
        blockers.append("source_mix_from_listen_pack_manifest_requires_clean_reference_bundle")
    return stable_unique(blockers)


def work_item(
    row: dict[str, Any],
    *,
    output_dir: Path,
    index: int,
    slot: dict[str, Any] | None = None,
    matched_raw_candidate: bool = True,
) -> dict[str, Any]:
    stem = normalize_stem(slot_value(slot, "stem_type", row.get("stem_type") or "unknown"))
    alias = str(slot_value(slot, "bundle_alias", "")) or f"r{index:03d}"
    bundle_dir = output_dir / "refs" / alias
    template_path = bundle_dir / TEMPLATE_NAME
    write_json(template_path, template_for(row, stem=stem))
    blockers = raw_candidate_blockers(row) if matched_raw_candidate else ["missing_raw_reference_candidate"]
    if not matched_raw_candidate:
        status = MISSING_RAW_STATUS
    elif blockers:
        status = BLOCKED_PROVENANCE_STATUS
    else:
        status = BUILDABLE_STATUS
    source_mix_resolution = str(
        row.get("source_mix_resolution")
        or ("same_directory_audio" if str(row.get("source_mix_candidate_path") or "") else "")
    )
    return {
        "schema": "full_pack_raw_reference_bundle_work_item.v1",
        "work_item_status": status,
        "bundle_alias": alias,
        "bundle_dir": str(bundle_dir),
        "template_json": str(template_path),
        "stem_type": stem,
        "source_collection_id": str(slot_value(slot, "source_collection_id", "")),
        "gap_id": str(slot_value(slot, "gap_id", "")),
        "queue_item_id": str(slot_value(slot, "queue_item_id", "")),
        "slot_index": slot_value(slot, "slot_index", None),
        "missing_training_job": slot_value(slot, "missing_training_job", False) is True,
        "raw_reference_candidate_matched": bool(matched_raw_candidate),
        "blockers": blockers,
        "track_id_hint": str(row.get("track_id_hint") or ""),
        "candidate_directory": str(row.get("candidate_directory") or ""),
        "source_mix_candidate_path": str(row.get("source_mix_candidate_path") or ""),
        "source_mix_resolution": source_mix_resolution,
        "source_mix_manifest_json": str(row.get("source_mix_manifest_json") or ""),
        "target_stem_candidate_path": str(row.get("target_stem_candidate_path") or ""),
        "required_files": required_files_for(stem, slot),
        "required_actions": [
            "verify_source_provenance",
            "verify_trainable_rights",
            "run_reference_waveform_audit",
            "copy_or_render_candidate_audio_after_verification",
            "write_verified_ground_truth_manifest",
            "rerun_full_pack_reference_bundle_preflight",
        ],
        "accepted_reference_row": False,
        "eligible_reference_material": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def rows_by_stem(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(normalize_stem(row.get("stem_type")), []).append(row)
    return result


def pop_preferred_raw_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for index, row in enumerate(candidates):
        if not raw_candidate_blockers(row):
            return candidates.pop(index)
    return candidates.pop(0) if candidates else {}


def slot_aware_items(
    rows: Sequence[dict[str, Any]],
    slots: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], int, int, int]:
    available = rows_by_stem(rows)
    items: list[dict[str, Any]] = []
    matched_count = 0
    missing_count = 0
    for index, slot in enumerate(slots, start=1):
        stem = normalize_stem(slot.get("stem_type"))
        candidates = available.get(stem) or []
        row = pop_preferred_raw_candidate(candidates)
        matched = bool(row)
        matched_count += 1 if matched else 0
        missing_count += 0 if matched else 1
        items.append(work_item(row, output_dir=output_dir, index=index, slot=slot, matched_raw_candidate=matched))
    unmatched_count = sum(len(items_for_stem) for items_for_stem in available.values())
    return items, matched_count, missing_count, unmatched_count


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Bundle Worklist",
            "",
            f"- worklist_status: `{manifest['worklist_status']}`",
            f"- work_item_count: `{manifest['work_item_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_full_pack_raw_reference_bundle_worklist(
    *,
    raw_candidates_json: Path,
    output_dir: Path,
    handoff_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", raw_candidates_json=str(raw_candidates_json), handoff_json=str(handoff_json or ""))

    raw_manifest = read_json(raw_candidates_json)
    failed = collect_true_flag_failures("raw_candidates", raw_manifest)
    if raw_manifest.get("schema") != "full_pack_raw_reference_candidates_audit.v1":
        failed.append("raw_candidates_schema_unexpected")
    if raw_manifest.get("failed_checks") not in ([], None):
        failed.append("raw_candidates_failed_checks_not_empty")
    raw_status = raw_manifest.get("raw_candidate_status")
    if raw_status not in ("raw_reference_candidates_need_verified_manifest", "blocked_no_raw_reference_candidates"):
        failed.append("raw_candidates_status_unexpected")
    rows, row_failed = raw_rows(raw_candidates_json, raw_manifest)
    failed.extend(row_failed)
    slots: list[dict[str, Any]] = []
    slot_failed: list[str] = []
    if handoff_json is not None:
        slots, slot_failed = handoff_rows(handoff_json)
        failed.extend(slot_failed)
    failed = stable_unique(failed)

    matched_raw_candidate_count = 0
    missing_raw_candidate_slot_count = 0
    unmatched_raw_candidate_count = 0
    if failed:
        items = []
    elif slots:
        items, matched_raw_candidate_count, missing_raw_candidate_slot_count, unmatched_raw_candidate_count = slot_aware_items(
            rows,
            slots,
            output_dir=output_dir,
        )
    else:
        items = [work_item(row, output_dir=output_dir, index=index) for index, row in enumerate(rows, start=1)]
        matched_raw_candidate_count = len(items)
    by_stem = Counter(item["stem_type"] for item in items)
    by_status = Counter(item["work_item_status"] for item in items)
    if failed:
        status = "blocked"
    elif items:
        status = "raw_reference_bundle_worklist_ready"
    else:
        status = "no_raw_reference_bundle_work_items"

    manifest = {
        "schema": "full_pack_raw_reference_bundle_worklist.v1",
        "manifest_type": "full_pack_raw_reference_bundle_worklist",
        "worklist_status": status,
        "failed_checks": failed,
        "raw_candidates_json": str(raw_candidates_json),
        "handoff_json": str(handoff_json or ""),
        "slot_aware": bool(slots),
        "formal_slot_count": len(slots),
        "matched_raw_candidate_count": matched_raw_candidate_count,
        "missing_raw_candidate_slot_count": missing_raw_candidate_slot_count,
        "unmatched_raw_candidate_count": unmatched_raw_candidate_count,
        "work_item_count": len(items),
        "work_items_by_stem": dict(sorted(by_stem.items())),
        "work_items_by_status": dict(sorted(by_status.items())),
        "accepted_reference_rows": 0,
        "eligible_reference_material_rows": 0,
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
        "presentation_status": "raw_reference_worklist_not_for_listen_pack",
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
            "ground_truth_manifest_written": False,
            "audio_placeholder_written": False,
            "not_release_artifact": True,
            "accepted_rows_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, items)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", worklist_status=status, work_item_count=len(items), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build worklist templates from raw reference candidates.")
    parser.add_argument("--raw-candidates-json", type=Path, required=True)
    parser.add_argument("--handoff-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_raw_reference_bundle_worklist.py",
        entrypoint_path="/app/tools/build_full_pack_raw_reference_bundle_worklist.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_raw_reference_bundle_worklist(
        raw_candidates_json=args.raw_candidates_json,
        handoff_json=args.handoff_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "worklist_status": manifest["worklist_status"],
                "work_item_count": manifest["work_item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
