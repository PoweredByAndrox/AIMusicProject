#!/usr/bin/env python3
"""Run one metadata-only ARIS full-pack loop refresh."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_full_pack_goal_completion import audit_full_pack_goal_completion  # noqa: E402
from tools.audit_full_pack_reference_material_discovery import audit_full_pack_reference_material_discovery  # noqa: E402
from tools.audit_full_pack_raw_reference_candidates import audit_full_pack_raw_reference_candidates  # noqa: E402
from tools.build_full_pack_aris_cycle_status import build_full_pack_aris_cycle_status  # noqa: E402
from tools.build_full_pack_aris_next_run_plan import build_full_pack_aris_next_run_plan  # noqa: E402
from tools.build_full_pack_raw_reference_bundle_worklist import build_full_pack_raw_reference_bundle_worklist  # noqa: E402
from tools.build_full_pack_raw_reference_bundle_recovery_queue import build_full_pack_raw_reference_bundle_recovery_queue  # noqa: E402
from tools.build_full_pack_raw_reference_recovery_material_handoff import build_full_pack_raw_reference_recovery_material_handoff  # noqa: E402
from tools.build_full_pack_raw_reference_recovery_material_preflight import build_full_pack_raw_reference_recovery_material_preflight  # noqa: E402
from tools.build_full_pack_raw_reference_recovery_material_responses import build_full_pack_raw_reference_recovery_material_responses  # noqa: E402
from tools.build_full_pack_raw_reference_recovery_action_plan import build_full_pack_raw_reference_recovery_action_plan  # noqa: E402
from tools.build_full_pack_raw_reference_verified_bundles import build_full_pack_raw_reference_verified_bundles  # noqa: E402
from tools.build_full_pack_reference_bundle_handoff import build_full_pack_reference_bundle_handoff  # noqa: E402
from tools.build_full_pack_reference_bundle_preflight import build_full_pack_reference_bundle_preflight  # noqa: E402
from tools.build_full_pack_reference_bundle_scaffold import build_full_pack_reference_bundle_scaffold  # noqa: E402
from tools.build_full_pack_reference_collection_queue import build_full_pack_reference_collection_queue  # noqa: E402
from tools.build_full_pack_reference_corpus_intake import build_full_pack_reference_corpus_intake  # noqa: E402
from tools.build_generic_audio_training_recovery_queue import build_generic_audio_training_recovery_queue  # noqa: E402
from tools.build_project_owned_audio_permission_decisions import build_project_owned_audio_permission_decisions  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.execute_full_pack_raw_reference_recovery_action_plan import execute_full_pack_raw_reference_recovery_action_plan  # noqa: E402
from tools.ingest_generic_audio_training_recovery_decisions import ingest_generic_audio_training_recovery_decisions  # noqa: E402
from tools.merge_generic_audio_training_examples import merge_generic_audio_training_examples  # noqa: E402
from tools.plan_generic_audio_finetune_jobs import plan_generic_audio_finetune_jobs  # noqa: E402
from tools.validate_verified_audio_training_corpus import validate_verified_audio_training_corpus  # noqa: E402


MANIFEST_NAME = "loop.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
RAW_BUNDLE_HANDOFF_SCHEMA = "full_pack_reference_bundle_handoff_manifest.v1"
RAW_BUNDLE_HANDOFF_READY_STATUS = "ready_for_reference_bundle_collection"
START_TIME = time.monotonic()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_loop",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_loop_or_recovery_sha256_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "metadata_loop_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def manifest_path_from_outputs(payload: dict[str, Any]) -> Path:
    outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
    path = outputs.get("manifest_json") or payload.get("manifest_json")
    if not path:
        raise ValueError(f"{payload.get('schema') or 'manifest'}: missing manifest_json output")
    return Path(str(path))


def default_next_loop_output_dir(output_dir: Path) -> Path:
    match = re.fullmatch(r"([A-Za-z_-]*?)(\d+)", output_dir.name)
    if match:
        prefix, digits = match.groups()
        return output_dir.with_name(f"{prefix}{int(digits) + 1:0{len(digits)}d}")
    return output_dir.parent / "fullloop_next"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object row")
        rows.append(row)
    return rows


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
        result[stem] = max(count, 0)
    return result


def track_stem_pair(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("track_id") or ""), normalized_stem(row.get("stem_type"))


def unique_paths(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def raw_reference_handoff_for_worklist(handoff_json: Path | None) -> tuple[Path | None, dict[str, Any]]:
    if handoff_json is None:
        return None, {
            "raw_reference_handoff_use_status": "not_supplied",
            "raw_reference_handoff_skip_reason": "",
            "raw_reference_handoff_schema": "",
            "raw_reference_handoff_status": "",
            "raw_reference_handoff_failed_checks": [],
        }
    if not handoff_json.is_file():
        return None, {
            "raw_reference_handoff_use_status": "skipped_incompatible_handoff",
            "raw_reference_handoff_skip_reason": "raw_reference_handoff_missing",
            "raw_reference_handoff_schema": "",
            "raw_reference_handoff_status": "",
            "raw_reference_handoff_failed_checks": ["raw_reference_handoff_missing"],
        }

    payload = read_json(handoff_json)
    schema = str(payload.get("schema") or "")
    status = str(payload.get("handoff_status") or "")
    raw_failed = payload.get("failed_checks")
    failed_checks = [str(item) for item in raw_failed] if isinstance(raw_failed, list) else []
    decision = {
        "raw_reference_handoff_schema": schema,
        "raw_reference_handoff_status": status,
        "raw_reference_handoff_failed_checks": failed_checks,
    }
    if schema == RAW_BUNDLE_HANDOFF_SCHEMA and status == RAW_BUNDLE_HANDOFF_READY_STATUS and raw_failed in ([], None):
        return handoff_json, {
            **decision,
            "raw_reference_handoff_use_status": "used_for_slot_aware_worklist",
            "raw_reference_handoff_skip_reason": "",
        }
    return None, {
        **decision,
        "raw_reference_handoff_use_status": "skipped_incompatible_handoff",
        "raw_reference_handoff_skip_reason": "raw_reference_handoff_not_ready_for_bundle_worklist",
    }


def build_skipped_raw_reference_worklist(
    *,
    raw_candidates: dict[str, Any],
    raw_candidates_json: Path,
    handoff_json: Path,
    handoff_decision: dict[str, Any],
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        raw_candidates_json=str(raw_candidates_json),
        handoff_json=str(handoff_json),
        **handoff_decision,
    )
    manifest = {
        "schema": "full_pack_raw_reference_bundle_worklist.v1",
        "manifest_type": "full_pack_raw_reference_bundle_worklist",
        "worklist_status": "no_raw_reference_bundle_work_items",
        "failed_checks": [],
        "raw_candidates_json": str(raw_candidates_json),
        "raw_candidate_status": raw_candidates.get("raw_candidate_status"),
        "raw_candidate_rows": int(raw_candidates.get("raw_candidate_rows") or 0),
        "handoff_json": str(handoff_json),
        "slot_aware": False,
        "formal_slot_count": 0,
        "matched_raw_candidate_count": 0,
        "missing_raw_candidate_slot_count": 0,
        "unmatched_raw_candidate_count": 0,
        "work_item_count": 0,
        "work_items_by_stem": {},
        "work_items_by_status": {},
        "accepted_reference_rows": 0,
        "eligible_reference_material_rows": 0,
        **handoff_decision,
        "outputs": {
            "manifest_json": str(output_dir / "work.json"),
            "items_jsonl": str(output_dir / "items.jsonl"),
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
    write_jsonl(output_dir / "items.jsonl", [])
    write_json(output_dir / "work.json", manifest)
    (output_dir / SUMMARY_NAME).write_text(
        "\n".join(
            [
                "# Full Pack Raw Reference Bundle Worklist",
                "",
                "- worklist_status: `no_raw_reference_bundle_work_items`",
                f"- raw_reference_handoff_use_status: `{handoff_decision['raw_reference_handoff_use_status']}`",
                f"- raw_reference_handoff_skip_reason: `{handoff_decision['raw_reference_handoff_skip_reason']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    progress(
        progress_path,
        "finish",
        worklist_status=manifest["worklist_status"],
        work_item_count=0,
        failed_checks=[],
        **handoff_decision,
    )
    return manifest


def split_new_recovered_rows(
    *,
    recovered_rows: Sequence[dict[str, Any]],
    existing_rows: Sequence[dict[str, Any]],
    intake_audio_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_pairs = {
        pair
        for pair in [*(track_stem_pair(row) for row in existing_rows), *(track_stem_pair(row) for row in intake_audio_rows)]
        if pair[0] and pair[1] != "unknown"
    }
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in recovered_rows:
        track_id, stem = track_stem_pair(row)
        if not track_id or stem == "unknown":
            excluded.append(
                {
                    "id": row.get("id"),
                    "track_id": track_id,
                    "stem_type": stem,
                    "reason": "missing_track_or_stem",
                }
            )
            continue
        if (track_id, stem) in seen_pairs:
            excluded.append(
                {
                    "id": row.get("id"),
                    "track_id": track_id,
                    "stem_type": stem,
                    "reason": "duplicate_existing_track_stem",
                }
            )
            continue
        seen_pairs.add((track_id, stem))
        accepted.append(row)
    return accepted, excluded


def build_recovered_intake_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "full_pack_reference_corpus_intake_row.v1",
        "id": str(row.get("id") or ""),
        "intake_status": "accepted_for_corpus_import",
        "rejection_reasons": [],
        "gap_id": "",
        "queue_item_id": "",
        "stem_type": normalized_stem(row.get("stem_type")),
        "track_id": str(row.get("track_id") or ""),
        "manifest_path": str(row.get("manifest_path") or ""),
        "source_domain": "project_owned_recovery_ingest",
        "permission_status": str(row.get("permission_status") or ""),
        "source_audio_path": str(row.get("source_audio_path") or ""),
        "target_audio_path": str(row.get("target_audio_path") or ""),
        "target_audio_sha256": str(row.get("target_audio_sha256") or ""),
        "reference_verified": row.get("reference_verified") is True,
        "required_reference_stem_count": 0,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def build_combined_reference_intake(
    *,
    intake: dict[str, Any],
    existing_audio_training_examples_jsonl: Path,
    recovered_audio_training_examples_jsonls: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", recovered_jsonls=[str(path) for path in recovered_audio_training_examples_jsonls])

    outputs = intake.get("outputs") if isinstance(intake.get("outputs"), dict) else {}
    intake_rows = iter_jsonl(Path(str(outputs.get("rows_jsonl") or ""))) if outputs.get("rows_jsonl") else []
    intake_audio_rows = iter_jsonl(Path(str(outputs.get("accepted_audio_training_examples_jsonl") or ""))) if outputs.get("accepted_audio_training_examples_jsonl") else []
    existing_rows = iter_jsonl(existing_audio_training_examples_jsonl)
    all_recovered_audio_rows = [row for path in recovered_audio_training_examples_jsonls for row in iter_jsonl(path)]
    recovered_audio_rows, excluded_recovered_rows = split_new_recovered_rows(
        recovered_rows=all_recovered_audio_rows,
        existing_rows=existing_rows,
        intake_audio_rows=intake_audio_rows,
    )
    recovered_intake_rows = [build_recovered_intake_row(row) for row in recovered_audio_rows]

    accepted_by_stem = Counter(int_mapping(intake.get("accepted_by_stem")))
    recovered_by_stem = Counter(normalized_stem(row.get("stem_type")) for row in recovered_audio_rows)
    recovered_by_stem.pop("unknown", None)
    accepted_by_stem.update(recovered_by_stem)

    remaining_by_stem = int_mapping(intake.get("remaining_by_stem"))
    for stem, recovered_count in recovered_by_stem.items():
        if stem in remaining_by_stem:
            remaining_by_stem[stem] = max(remaining_by_stem[stem] - recovered_count, 0)
    remaining_by_stem = dict(sorted((stem, count) for stem, count in remaining_by_stem.items() if count > 0))
    remaining_total = sum(remaining_by_stem.values())
    accepted_total = int(intake.get("accepted_reference_rows") or 0) + len(recovered_audio_rows)
    if intake.get("failed_checks"):
        status = "blocked"
    elif accepted_total and remaining_total:
        status = "accepted_with_remaining_gaps"
    elif accepted_total:
        status = "accepted_reference_corpus_complete"
    else:
        status = str(intake.get("intake_status") or "no_accepted_references")

    rows_path = output_dir / "rows.jsonl"
    audio_path = output_dir / "audio.jsonl"
    excluded_path = output_dir / "excluded.jsonl"
    manifest_path = output_dir / "intake.json"
    summary_path = output_dir / "summary.md"
    excluded_reason_counts = dict(Counter(str(row["reason"]) for row in excluded_recovered_rows))
    manifest = {
        "schema": "full_pack_reference_corpus_intake_manifest.v1",
        "manifest_type": "full_pack_reference_corpus_intake",
        "intake_status": status,
        "combined_from_recovery": True,
        "failed_checks": list(intake.get("failed_checks") or []),
        "source_intake_json": str(manifest_path_from_outputs(intake)),
        "recovered_audio_training_examples_jsonls": [str(path) for path in recovered_audio_training_examples_jsonls],
        "candidate_roots": list(intake.get("candidate_roots") or []),
        "accepted_reference_rows": accepted_total,
        "rejected_reference_rows": int(intake.get("rejected_reference_rows") or 0),
        "recovery_ingest_audio_training_examples": len(all_recovered_audio_rows),
        "recovered_audio_training_examples": len(recovered_audio_rows),
        "excluded_recovered_audio_training_examples": len(excluded_recovered_rows),
        "excluded_recovered_reason_counts": excluded_reason_counts,
        "accepted_audio_training_examples": len(intake_audio_rows) + len(recovered_audio_rows),
        "required_reference_stem_count": int(intake.get("required_reference_stem_count") or 0),
        "remaining_reference_stem_count": remaining_total,
        "accepted_by_stem": dict(sorted(accepted_by_stem.items())),
        "remaining_by_stem": remaining_by_stem,
        "rejection_reason_counts": dict(intake.get("rejection_reason_counts") or {}),
        "outputs": {
            "manifest_json": str(manifest_path),
            "rows_jsonl": str(rows_path),
            "accepted_audio_training_examples_jsonl": str(audio_path),
            "excluded_recovered_jsonl": str(excluded_path),
            "progress_jsonl": str(progress_path),
            "summary_md": str(summary_path),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "reference_corpus_intake_not_for_listen_pack",
        "release_status": "training_reference_intake_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": not bool(all_recovered_audio_rows),
            "metadata_and_sha256_only": True,
            "audio_read": bool(all_recovered_audio_rows),
            "audio_decoded": False,
            "audio_copied": False,
            "audio_sha256_read_by_recovery_ingest": bool(all_recovered_audio_rows),
            "not_release_artifact": True,
        },
    }
    write_jsonl(rows_path, [*intake_rows, *recovered_intake_rows])
    write_jsonl(audio_path, [*intake_audio_rows, *recovered_audio_rows])
    write_jsonl(excluded_path, excluded_recovered_rows)
    write_json(manifest_path, manifest)
    summary_path.write_text(
        "\n".join(
            [
                "# Combined Full Pack Reference Intake",
                "",
                f"- intake_status: `{status}`",
                f"- accepted_reference_rows: `{accepted_total}`",
                f"- remaining_reference_stem_count: `{remaining_total}`",
                f"- recovered_audio_training_examples: `{len(recovered_audio_rows)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    progress(progress_path, "finish", intake_status=status, accepted_reference_rows=accepted_total, remaining_reference_stem_count=remaining_total)
    return manifest


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack ARIS Metadata Loop",
            "",
            f"- loop_status: `{manifest['loop_status']}`",
            f"- goal_complete: `{str(manifest['goal_complete']).lower()}`",
            f"- accepted_reference_rows: `{manifest['accepted_reference_rows']}`",
            f"- remaining_reference_stem_count: `{manifest['remaining_reference_stem_count']}`",
            f"- cycle_status: `{manifest['cycle_status']}`",
            f"- next_action: `{manifest['next_action']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def run_full_pack_aris_metadata_loop(
    *,
    machine_gate_json: Path,
    repair_queue_json: Path,
    full_pack_loop_plan_json: Path,
    reference_gap_plan_json: Path,
    existing_audio_training_examples_jsonl: Path,
    candidate_roots: Sequence[Path],
    output_dir: Path,
    min_unique_tracks: int = 3,
    training_output_root: str = "/app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_runs",
    raw_reference_handoff_json: Path | None = None,
    raw_reference_allowed_source_roots: Sequence[Path] = (),
    raw_reference_recovery_material_candidate_roots: Sequence[Path] = (),
    raw_reference_rights_status: str = "",
    raw_reference_require_full_mix_reconstruction: bool = True,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    if min_unique_tracks <= 0:
        raise ValueError("min_unique_tracks must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        machine_gate_json=str(machine_gate_json),
        reference_gap_plan_json=str(reference_gap_plan_json),
        candidate_roots=[str(path) for path in candidate_roots],
        raw_reference_handoff_json=str(raw_reference_handoff_json or ""),
    )

    discovery = audit_full_pack_reference_material_discovery(
        gap_plan_json=reference_gap_plan_json,
        existing_audio_training_examples_jsonl=existing_audio_training_examples_jsonl,
        candidate_roots=candidate_roots,
        output_dir=output_dir / "disc",
        command_args=command_args or [],
    )
    progress(
        progress_path,
        "reference_material_discovery_done",
        discovery_status=discovery["discovery_status"],
        eligible_candidate_rows=discovery["eligible_candidate_rows"],
        candidate_manifest_count=discovery["candidate_manifest_count"],
    )

    raw_candidates = audit_full_pack_raw_reference_candidates(
        gap_plan_json=reference_gap_plan_json,
        candidate_roots=candidate_roots,
        output_dir=output_dir / "raw",
        command_args=command_args or [],
    )
    progress(
        progress_path,
        "raw_reference_candidate_triage_done",
        raw_candidate_status=raw_candidates["raw_candidate_status"],
        raw_candidate_rows=raw_candidates["raw_candidate_rows"],
        rejected_directory_count=raw_candidates["rejected_directory_count"],
    )

    raw_candidates_json = Path(str(raw_candidates["outputs"]["manifest_json"]))
    raw_worklist_handoff_json, raw_handoff_decision = raw_reference_handoff_for_worklist(
        raw_reference_handoff_json
    )
    if raw_reference_handoff_json is not None and raw_worklist_handoff_json is None:
        raw_worklist = build_skipped_raw_reference_worklist(
            raw_candidates=raw_candidates,
            raw_candidates_json=raw_candidates_json,
            output_dir=output_dir / "rwork",
            handoff_json=raw_reference_handoff_json,
            handoff_decision=raw_handoff_decision,
            command_args=command_args or [],
        )
    else:
        raw_worklist = build_full_pack_raw_reference_bundle_worklist(
            raw_candidates_json=raw_candidates_json,
            output_dir=output_dir / "rwork",
            handoff_json=raw_worklist_handoff_json,
            command_args=command_args or [],
        )
    progress(
        progress_path,
        "raw_reference_worklist_done",
        worklist_status=raw_worklist["worklist_status"],
        work_item_count=raw_worklist["work_item_count"],
        slot_aware=raw_worklist["slot_aware"],
        formal_slot_count=raw_worklist["formal_slot_count"],
        missing_raw_candidate_slot_count=raw_worklist["missing_raw_candidate_slot_count"],
        **raw_handoff_decision,
    )

    raw_verified_bundles: dict[str, Any]
    if raw_reference_allowed_source_roots and raw_reference_rights_status:
        raw_verified_bundles = build_full_pack_raw_reference_verified_bundles(
            raw_worklist_json=Path(str(raw_worklist["outputs"]["manifest_json"])),
            output_dir=output_dir / "rvb",
            allowed_source_roots=raw_reference_allowed_source_roots,
            rights_status=raw_reference_rights_status,
            require_full_mix_reconstruction=bool(raw_reference_require_full_mix_reconstruction),
            command_args=command_args or [],
        )
        progress(
            progress_path,
            "raw_reference_verified_bundles_done",
            bundle_status=raw_verified_bundles["bundle_status"],
            verified_bundle_count=raw_verified_bundles["verified_bundle_count"],
            blocked_bundle_count=raw_verified_bundles["blocked_bundle_count"],
        )
    else:
        raw_verified_bundles = {
            "bundle_status": "not_requested",
            "verified_bundle_count": 0,
            "blocked_bundle_count": 0,
            "candidate_roots": [],
            "outputs": {"manifest_json": ""},
            "safety": {
                "audio_read": False,
                "audio_copied": False,
                "audio_decoded": False,
                "ground_truth_manifest_written": False,
            },
        }
        progress(progress_path, "raw_reference_verified_bundles_skipped", bundle_status="not_requested")

    if raw_verified_bundles["outputs"]["manifest_json"]:
        raw_reference_recovery_queue = build_full_pack_raw_reference_bundle_recovery_queue(
            raw_verified_bundles_json=Path(str(raw_verified_bundles["outputs"]["manifest_json"])),
            output_dir=output_dir / "rrec",
            command_args=command_args or [],
        )
    else:
        raw_reference_recovery_queue = {
            "queue_status": "not_requested",
            "queue_item_count": 0,
            "route_intent_counts": {},
            "outputs": {"manifest_json": ""},
            "failed_checks": [],
        }
    progress(
        progress_path,
        "raw_reference_recovery_queue_done",
        queue_status=raw_reference_recovery_queue["queue_status"],
        queue_item_count=raw_reference_recovery_queue["queue_item_count"],
        failed_checks=raw_reference_recovery_queue["failed_checks"],
    )

    if raw_reference_recovery_queue["outputs"]["manifest_json"]:
        raw_reference_recovery_action_plan = build_full_pack_raw_reference_recovery_action_plan(
            raw_reference_recovery_queue_json=Path(str(raw_reference_recovery_queue["outputs"]["manifest_json"])),
            output_dir=output_dir / "ract",
            command_args=command_args or [],
        )
    else:
        raw_reference_recovery_action_plan = {
            "action_plan_status": "not_requested",
            "action_item_count": 0,
            "route_intent_counts": {},
            "outputs": {"manifest_json": ""},
            "failed_checks": [],
        }
    progress(
        progress_path,
        "raw_reference_recovery_action_plan_done",
        action_plan_status=raw_reference_recovery_action_plan["action_plan_status"],
        action_item_count=raw_reference_recovery_action_plan["action_item_count"],
        failed_checks=raw_reference_recovery_action_plan["failed_checks"],
    )

    if raw_reference_recovery_action_plan["outputs"]["manifest_json"]:
        raw_reference_recovery_execution = execute_full_pack_raw_reference_recovery_action_plan(
            raw_reference_recovery_action_plan_json=Path(
                str(raw_reference_recovery_action_plan["outputs"]["manifest_json"])
            ),
            output_dir=output_dir / "rexec",
            command_args=command_args or [],
        )
    else:
        raw_reference_recovery_execution = {
            "execution_status": "not_requested",
            "action_item_count": 0,
            "execution_status_counts": {},
            "outputs": {"manifest_json": ""},
            "failed_checks": [],
        }
    progress(
        progress_path,
        "raw_reference_recovery_execution_done",
        execution_status=raw_reference_recovery_execution["execution_status"],
        action_item_count=raw_reference_recovery_execution["action_item_count"],
        failed_checks=raw_reference_recovery_execution["failed_checks"],
    )

    recovery_material_allowed_roots = (
        raw_reference_recovery_material_candidate_roots or raw_reference_allowed_source_roots
    )
    recovery_material_candidate_roots = raw_reference_recovery_material_candidate_roots
    if (
        not recovery_material_candidate_roots
        and int(raw_reference_recovery_execution.get("action_item_count") or 0) > 0
    ):
        recovery_material_candidate_roots = recovery_material_allowed_roots
    if raw_reference_recovery_execution["outputs"]["manifest_json"]:
        raw_reference_recovery_material_responses = build_full_pack_raw_reference_recovery_material_responses(
            raw_reference_recovery_execution_json=Path(
                str(raw_reference_recovery_execution["outputs"]["manifest_json"])
            ),
            candidate_roots=recovery_material_candidate_roots,
            response_root=output_dir / "rexec" / "rsp",
            allowed_source_roots=recovery_material_allowed_roots,
            output_dir=output_dir / "rrsp",
            command_args=command_args or [],
        )
    else:
        raw_reference_recovery_material_responses = {
            "response_status": "not_requested",
            "action_item_count": 0,
            "ready_response_count": 0,
            "blocked_response_count": 0,
            "written_response_count": 0,
            "outputs": {"manifest_json": ""},
            "failed_checks": [],
        }
    progress(
        progress_path,
        "raw_reference_recovery_material_responses_done",
        response_status=raw_reference_recovery_material_responses["response_status"],
        action_item_count=raw_reference_recovery_material_responses["action_item_count"],
        ready_response_count=raw_reference_recovery_material_responses["ready_response_count"],
        blocked_response_count=raw_reference_recovery_material_responses["blocked_response_count"],
        failed_checks=raw_reference_recovery_material_responses["failed_checks"],
    )

    if raw_reference_recovery_execution["outputs"]["manifest_json"]:
        raw_reference_recovery_material_preflight = build_full_pack_raw_reference_recovery_material_preflight(
            raw_reference_recovery_execution_json=Path(
                str(raw_reference_recovery_execution["outputs"]["manifest_json"])
            ),
            response_root=output_dir / "rexec" / "rsp",
            allowed_source_roots=recovery_material_allowed_roots,
            output_dir=output_dir / "rpre",
            command_args=command_args or [],
        )
    else:
        raw_reference_recovery_material_preflight = {
            "preflight_status": "not_requested",
            "action_item_count": 0,
            "ready_action_count": 0,
            "blocked_action_count": 0,
            "candidate_roots": [],
            "outputs": {"manifest_json": ""},
            "failed_checks": [],
        }
    progress(
        progress_path,
        "raw_reference_recovery_material_preflight_done",
        preflight_status=raw_reference_recovery_material_preflight["preflight_status"],
        action_item_count=raw_reference_recovery_material_preflight["action_item_count"],
        ready_action_count=raw_reference_recovery_material_preflight["ready_action_count"],
        blocked_action_count=raw_reference_recovery_material_preflight["blocked_action_count"],
        failed_checks=raw_reference_recovery_material_preflight["failed_checks"],
    )

    if raw_reference_recovery_material_preflight["outputs"]["manifest_json"]:
        raw_reference_recovery_material_handoff = build_full_pack_raw_reference_recovery_material_handoff(
            raw_reference_recovery_material_preflight_json=Path(
                str(raw_reference_recovery_material_preflight["outputs"]["manifest_json"])
            ),
            output_dir=output_dir / "rhand",
            command_args=command_args or [],
        )
    else:
        raw_reference_recovery_material_handoff = {
            "handoff_status": "not_requested",
            "handoff_item_count": 0,
            "ready_handoff_count": 0,
            "blocked_handoff_count": 0,
            "candidate_roots": [],
            "outputs": {"manifest_json": ""},
            "failed_checks": [],
        }
    progress(
        progress_path,
        "raw_reference_recovery_material_handoff_done",
        handoff_status=raw_reference_recovery_material_handoff["handoff_status"],
        handoff_item_count=raw_reference_recovery_material_handoff["handoff_item_count"],
        ready_handoff_count=raw_reference_recovery_material_handoff["ready_handoff_count"],
        blocked_handoff_count=raw_reference_recovery_material_handoff["blocked_handoff_count"],
        failed_checks=raw_reference_recovery_material_handoff["failed_checks"],
    )

    raw_verified_candidate_roots = [Path(str(path)) for path in raw_verified_bundles.get("candidate_roots", [])]
    raw_recovery_material_candidate_roots = [
        Path(str(path)) for path in raw_reference_recovery_material_handoff.get("candidate_roots", [])
    ]
    intake_candidate_roots = unique_paths(
        [*candidate_roots, *raw_verified_candidate_roots, *raw_recovery_material_candidate_roots]
    )
    intake = build_full_pack_reference_corpus_intake(
        gap_plan_json=reference_gap_plan_json,
        existing_audio_training_examples_jsonl=existing_audio_training_examples_jsonl,
        candidate_roots=intake_candidate_roots,
        output_dir=output_dir / "int",
        command_args=command_args or [],
    )
    progress(progress_path, "intake_done", intake_status=intake["intake_status"], accepted_reference_rows=intake["accepted_reference_rows"])

    recovery_manifests: list[dict[str, Any]] = []
    permission_manifests: list[dict[str, Any]] = []
    recovery_ingest_manifests: list[dict[str, Any]] = []
    recovered_audio_jsonls: list[Path] = []
    for root_index, candidate_root in enumerate(candidate_roots, start=1):
        artifact_root = candidate_root.parent if candidate_root.is_file() else candidate_root
        alias = f"r{root_index:03d}"
        recovery = build_generic_audio_training_recovery_queue(
            artifact_root=artifact_root,
            output_dir=output_dir / "rec" / alias,
            full_pack_loop_plan_json=full_pack_loop_plan_json,
            short_output_names=True,
        )
        recovery_manifests.append(recovery)
        recovery_queue_jsonl = Path(str(recovery["outputs"]["queue_jsonl"]))
        permission = build_project_owned_audio_permission_decisions(
            recovery_queue_jsonl=recovery_queue_jsonl,
            output_dir=output_dir / "perm" / alias,
            allowed_source_roots=[artifact_root],
            short_output_names=True,
        )
        permission_manifests.append(permission)
        ingest = ingest_generic_audio_training_recovery_decisions(
            recovery_queue_jsonl=recovery_queue_jsonl,
            decisions_jsonl=Path(str(permission["decisions_jsonl"])),
            output_dir=output_dir / "ing" / alias,
            short_output_names=True,
        )
        recovery_ingest_manifests.append(ingest)
        approved_count = int(ingest["summary"]["approved_audio_training_examples"])
        if approved_count > 0:
            recovered_audio_jsonls.append(Path(str(ingest["outputs"]["approved_audio_training_examples_jsonl"])))
        progress(
            progress_path,
            "recovery_root_done",
            root_alias=alias,
            candidate_root=str(candidate_root),
            recovery_rows=recovery["summary"]["recovery_rows"],
            permission_decision_rows=permission["decision_rows"],
            recovered_audio_training_examples=approved_count,
            audio_sha256_read=approved_count > 0,
        )

    combined_intake = build_combined_reference_intake(
        intake=intake,
        existing_audio_training_examples_jsonl=existing_audio_training_examples_jsonl,
        recovered_audio_training_examples_jsonls=recovered_audio_jsonls,
        output_dir=output_dir / "cint",
    )
    progress(
        progress_path,
        "combined_intake_done",
        intake_status=combined_intake["intake_status"],
        accepted_reference_rows=combined_intake["accepted_reference_rows"],
        remaining_reference_stem_count=combined_intake["remaining_reference_stem_count"],
        recovery_ingest_audio_training_examples=combined_intake["recovery_ingest_audio_training_examples"],
        recovered_audio_training_examples=combined_intake["recovered_audio_training_examples"],
        excluded_recovered_audio_training_examples=combined_intake["excluded_recovered_audio_training_examples"],
    )

    combined_audio_jsonl = Path(str(combined_intake["outputs"]["accepted_audio_training_examples_jsonl"]))
    merge = merge_generic_audio_training_examples(
        base_audio_training_examples_jsonls=[existing_audio_training_examples_jsonl],
        approved_audio_training_examples_jsonls=[combined_audio_jsonl],
        output_dir=output_dir / "merge",
        short_output_names=True,
    )
    merged_jsonl = Path(str(merge["outputs"]["merged_audio_training_examples_jsonl"]))
    progress(progress_path, "merge_done", **merge["summary"])

    validation = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=merged_jsonl,
        output_dir=output_dir / "val",
        min_unique_tracks=min_unique_tracks,
        short_output_names=True,
    )
    progress(progress_path, "validation_done", corpus_validation_status=validation["corpus_validation_status"], failed_checks=validation["failed_checks"])

    training = plan_generic_audio_finetune_jobs(
        audio_training_examples_jsonl=merged_jsonl,
        output_dir=output_dir / "tr",
        min_unique_tracks=min_unique_tracks,
        training_output_root=training_output_root,
        short_output_names=True,
    )
    progress(progress_path, "training_plan_done", **training["summary"])

    cycle = build_full_pack_aris_cycle_status(
        machine_gate_json=machine_gate_json,
        repair_queue_json=repair_queue_json,
        full_pack_loop_plan_json=full_pack_loop_plan_json,
        reference_gap_plan_json=reference_gap_plan_json,
        reference_intake_json=manifest_path_from_outputs(combined_intake),
        verified_corpus_manifest_json=Path(str(validation["manifest_json"])),
        training_plan_json=Path(str(training["outputs"]["plan_json"])),
        output_dir=output_dir / "cycle",
        command_args=command_args or [],
    )
    progress(progress_path, "cycle_done", cycle_status=cycle["cycle_status"], next_action=cycle["next_action"])

    collection = build_full_pack_reference_collection_queue(
        cycle_status_json=manifest_path_from_outputs(cycle),
        reference_gap_plan_json=reference_gap_plan_json,
        reference_intake_json=manifest_path_from_outputs(combined_intake),
        output_dir=output_dir / "collect",
        raw_reference_worklist_json=manifest_path_from_outputs(raw_worklist),
        command_args=command_args or [],
    )
    progress(progress_path, "collection_done", queue_status=collection["queue_status"], collection_item_count=collection["collection_item_count"])

    handoff = build_full_pack_reference_bundle_handoff(
        collection_queue_json=manifest_path_from_outputs(collection),
        output_dir=output_dir / "hand",
        command_args=command_args or [],
    )
    progress(
        progress_path,
        "handoff_done",
        handoff_status=handoff["handoff_status"],
        handoff_item_count=handoff["handoff_item_count"],
    )

    scaffold = build_full_pack_reference_bundle_scaffold(
        handoff_json=manifest_path_from_outputs(handoff),
        handoff_root=output_dir / "hand",
        output_dir=output_dir / "scaf",
        command_args=command_args or [],
    )
    progress(
        progress_path,
        "scaffold_done",
        scaffold_status=scaffold["scaffold_status"],
        bundle_directory_count=scaffold["bundle_directory_count"],
    )

    preflight = build_full_pack_reference_bundle_preflight(
        handoff_json=manifest_path_from_outputs(handoff),
        handoff_root=output_dir / "hand",
        output_dir=output_dir / "pre",
        command_args=command_args or [],
    )
    progress(
        progress_path,
        "preflight_done",
        preflight_status=preflight["preflight_status"],
        ready_bundle_count=preflight["ready_bundle_count"],
        blocked_bundle_count=preflight["blocked_bundle_count"],
    )

    next_run = build_full_pack_aris_next_run_plan(
        preflight_json=manifest_path_from_outputs(preflight),
        machine_gate_json=machine_gate_json,
        repair_queue_json=repair_queue_json,
        full_pack_loop_plan_json=full_pack_loop_plan_json,
        reference_gap_plan_json=reference_gap_plan_json,
        existing_audio_training_examples_jsonl=existing_audio_training_examples_jsonl,
        raw_reference_recovery_material_handoff_json=manifest_path_from_outputs(raw_reference_recovery_material_handoff)
        if raw_reference_recovery_material_handoff["outputs"]["manifest_json"]
        else None,
        output_dir=output_dir / "next",
        next_output_dir=default_next_loop_output_dir(output_dir),
        min_unique_tracks=min_unique_tracks,
        training_output_root=training_output_root,
        command_args=command_args or [],
    )
    progress(
        progress_path,
        "next_run_plan_done",
        next_run_status=next_run["next_run_status"],
        next_run_candidate_root_count=next_run["next_run_candidate_root_count"],
        failed_checks=next_run["failed_checks"],
    )

    audit = audit_full_pack_goal_completion(
        machine_gate_json=machine_gate_json,
        cycle_status_json=manifest_path_from_outputs(cycle),
        collection_queue_json=manifest_path_from_outputs(collection),
        verified_corpus_manifest_json=Path(str(validation["manifest_json"])),
        training_plan_json=Path(str(training["outputs"]["plan_json"])),
        output_dir=output_dir / "goal",
        command_args=command_args or [],
    )

    goal_complete = audit["goal_complete"] is True
    loop_status = "goal_complete" if goal_complete else "blocked_continuing_aris_loop"
    manifest = {
        "schema": "full_pack_aris_metadata_loop_manifest.v1",
        "manifest_type": "full_pack_aris_metadata_loop",
        "loop_status": loop_status,
        "goal_complete": goal_complete,
        "result_delivery_allowed": audit["result_delivery_allowed"],
        "result_presentation_status": audit["result_presentation_status"],
        "accepted_reference_rows": combined_intake["accepted_reference_rows"],
        "remaining_reference_stem_count": combined_intake["remaining_reference_stem_count"],
        "recovery_ingest_audio_training_examples": combined_intake["recovery_ingest_audio_training_examples"],
        "recovered_audio_training_examples": combined_intake["recovered_audio_training_examples"],
        "excluded_recovered_audio_training_examples": combined_intake["excluded_recovered_audio_training_examples"],
        "cycle_status": cycle["cycle_status"],
        "next_action": cycle["next_action"],
        "collection_queue_status": collection["queue_status"],
        "reference_bundle_handoff_status": handoff["handoff_status"],
        "reference_bundle_scaffold_status": scaffold["scaffold_status"],
        "reference_bundle_preflight_status": preflight["preflight_status"],
        "reference_material_discovery_status": discovery["discovery_status"],
        "eligible_reference_material_rows": discovery["eligible_candidate_rows"],
        "raw_reference_candidate_status": raw_candidates["raw_candidate_status"],
        "raw_reference_candidate_rows": raw_candidates["raw_candidate_rows"],
        "raw_reference_worklist_status": raw_worklist["worklist_status"],
        "raw_reference_work_item_count": raw_worklist["work_item_count"],
        "raw_reference_handoff_use_status": raw_handoff_decision["raw_reference_handoff_use_status"],
        "raw_reference_handoff_skip_reason": raw_handoff_decision["raw_reference_handoff_skip_reason"],
        "raw_reference_verified_bundle_status": raw_verified_bundles["bundle_status"],
        "raw_reference_verified_bundle_count": raw_verified_bundles["verified_bundle_count"],
        "raw_reference_verified_blocked_bundle_count": raw_verified_bundles["blocked_bundle_count"],
        "raw_reference_recovery_queue_status": raw_reference_recovery_queue["queue_status"],
        "raw_reference_recovery_queue_item_count": raw_reference_recovery_queue["queue_item_count"],
        "raw_reference_recovery_route_intent_counts": raw_reference_recovery_queue["route_intent_counts"],
        "raw_reference_recovery_action_plan_status": raw_reference_recovery_action_plan["action_plan_status"],
        "raw_reference_recovery_action_item_count": raw_reference_recovery_action_plan["action_item_count"],
        "raw_reference_recovery_action_route_counts": raw_reference_recovery_action_plan["route_intent_counts"],
        "raw_reference_recovery_execution_status": raw_reference_recovery_execution["execution_status"],
        "raw_reference_recovery_execution_item_count": raw_reference_recovery_execution["action_item_count"],
        "raw_reference_recovery_execution_status_counts": raw_reference_recovery_execution["execution_status_counts"],
        "raw_reference_recovery_material_response_status": raw_reference_recovery_material_responses["response_status"],
        "raw_reference_recovery_material_response_ready_count": raw_reference_recovery_material_responses["ready_response_count"],
        "raw_reference_recovery_material_response_blocked_count": raw_reference_recovery_material_responses["blocked_response_count"],
        "raw_reference_recovery_material_preflight_status": raw_reference_recovery_material_preflight["preflight_status"],
        "raw_reference_recovery_material_ready_count": raw_reference_recovery_material_preflight["ready_action_count"],
        "raw_reference_recovery_material_blocked_count": raw_reference_recovery_material_preflight["blocked_action_count"],
        "raw_reference_recovery_material_handoff_status": raw_reference_recovery_material_handoff["handoff_status"],
        "raw_reference_recovery_material_handoff_ready_count": raw_reference_recovery_material_handoff["ready_handoff_count"],
        "raw_reference_recovery_material_handoff_blocked_count": raw_reference_recovery_material_handoff["blocked_handoff_count"],
        "ready_reference_bundle_count": preflight["ready_bundle_count"],
        "blocked_reference_bundle_count": preflight["blocked_bundle_count"],
        "next_run_status": next_run["next_run_status"],
        "next_run_candidate_root_count": next_run["next_run_candidate_root_count"],
        "next_run_failed_checks": next_run["failed_checks"],
        "verified_corpus_status": validation["corpus_validation_status"],
        "training_plan_summary": training["summary"],
        "remaining_blockers": audit["remaining_blockers"],
        "inputs": {
            "machine_gate_json": str(machine_gate_json),
            "repair_queue_json": str(repair_queue_json),
            "full_pack_loop_plan_json": str(full_pack_loop_plan_json),
            "reference_gap_plan_json": str(reference_gap_plan_json),
            "existing_audio_training_examples_jsonl": str(existing_audio_training_examples_jsonl),
            "candidate_roots": [str(path) for path in candidate_roots],
            "intake_candidate_roots": [str(path) for path in intake_candidate_roots],
            "min_unique_tracks": int(min_unique_tracks),
            "raw_reference_handoff_json": str(raw_reference_handoff_json or ""),
            "raw_reference_allowed_source_roots": [str(path) for path in raw_reference_allowed_source_roots],
            "raw_reference_recovery_material_candidate_roots": [
                str(path) for path in raw_reference_recovery_material_candidate_roots
            ],
            "raw_reference_rights_status": raw_reference_rights_status,
            "raw_reference_require_full_mix_reconstruction": bool(raw_reference_require_full_mix_reconstruction),
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "summary_md": str(output_dir / SUMMARY_NAME),
            "progress_jsonl": str(progress_path),
            "reference_material_discovery_json": str(Path(str(discovery["outputs"]["manifest_json"]))),
            "raw_reference_candidates_json": str(Path(str(raw_candidates["outputs"]["manifest_json"]))),
            "raw_reference_worklist_json": str(Path(str(raw_worklist["outputs"]["manifest_json"]))),
            "raw_reference_verified_bundles_json": str(Path(str(raw_verified_bundles["outputs"]["manifest_json"])))
            if raw_verified_bundles["outputs"]["manifest_json"]
            else "",
            "raw_reference_recovery_queue_json": str(Path(str(raw_reference_recovery_queue["outputs"]["manifest_json"])))
            if raw_reference_recovery_queue["outputs"]["manifest_json"]
            else "",
            "raw_reference_recovery_action_plan_json": str(
                Path(str(raw_reference_recovery_action_plan["outputs"]["manifest_json"]))
            )
            if raw_reference_recovery_action_plan["outputs"]["manifest_json"]
            else "",
            "raw_reference_recovery_execution_json": str(
                Path(str(raw_reference_recovery_execution["outputs"]["manifest_json"]))
            )
            if raw_reference_recovery_execution["outputs"]["manifest_json"]
            else "",
            "raw_reference_recovery_material_responses_json": str(
                Path(str(raw_reference_recovery_material_responses["outputs"]["manifest_json"]))
            )
            if raw_reference_recovery_material_responses["outputs"]["manifest_json"]
            else "",
            "raw_reference_recovery_material_preflight_json": str(
                Path(str(raw_reference_recovery_material_preflight["outputs"]["manifest_json"]))
            )
            if raw_reference_recovery_material_preflight["outputs"]["manifest_json"]
            else "",
            "raw_reference_recovery_material_handoff_json": str(
                Path(str(raw_reference_recovery_material_handoff["outputs"]["manifest_json"]))
            )
            if raw_reference_recovery_material_handoff["outputs"]["manifest_json"]
            else "",
            "intake_json": str(manifest_path_from_outputs(intake)),
            "combined_intake_json": str(manifest_path_from_outputs(combined_intake)),
            "recovery_queue_jsons": [str(manifest_path_from_outputs(item)) for item in recovery_manifests],
            "permission_decision_jsons": [str(item["manifest_json"]) for item in permission_manifests],
            "recovery_ingest_jsons": [str(manifest_path_from_outputs(item)) for item in recovery_ingest_manifests],
            "merge_json": str(manifest_path_from_outputs(merge)),
            "validation_json": str(validation["manifest_json"]),
            "training_plan_json": str(training["outputs"]["plan_json"]),
            "cycle_status_json": str(manifest_path_from_outputs(cycle)),
            "collection_queue_json": str(manifest_path_from_outputs(collection)),
            "reference_bundle_handoff_json": str(manifest_path_from_outputs(handoff)),
            "reference_bundle_scaffold_json": str(manifest_path_from_outputs(scaffold)),
            "reference_bundle_preflight_json": str(manifest_path_from_outputs(preflight)),
            "next_run_plan_json": str(manifest_path_from_outputs(next_run)),
            "completion_audit_json": str(Path(str(audit["outputs"]["audit_json"]))),
        },
        "delivery_allowed": False,
        "presentation_status": "full_pack_aris_metadata_loop_not_for_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "not_claimed": ["accepted_stems", "listen_result", "training_started", "model_promoted"],
        "safety": {
            "metadata_only": combined_intake["recovery_ingest_audio_training_examples"] == 0 and not raw_verified_bundles["safety"]["audio_read"],
            "metadata_and_sha256_only": True,
            "audio_read": combined_intake["recovery_ingest_audio_training_examples"] > 0 or raw_verified_bundles["safety"]["audio_read"],
            "audio_sha256_read_by_recovery_ingest": combined_intake["recovery_ingest_audio_training_examples"] > 0,
            "audio_copied": raw_verified_bundles["safety"]["audio_copied"],
            "audio_decoded": raw_verified_bundles["safety"]["audio_decoded"],
            "ground_truth_manifest_written": raw_verified_bundles["safety"]["ground_truth_manifest_written"],
            "training_started": False,
            "not_release_artifact": True,
            "fail_closed_until_completion_audit_accepts": True,
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", loop_status=loop_status, goal_complete=goal_complete, remaining_blockers=audit["remaining_blockers"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one metadata-only ARIS full-pack loop refresh.")
    parser.add_argument("--machine-gate-json", type=Path, required=True)
    parser.add_argument("--repair-queue-json", type=Path, required=True)
    parser.add_argument("--full-pack-loop-plan-json", type=Path, required=True)
    parser.add_argument("--reference-gap-plan-json", type=Path, required=True)
    parser.add_argument("--existing-audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-unique-tracks", type=int, default=3)
    parser.add_argument(
        "--training-output-root",
        default="/app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_runs",
    )
    parser.add_argument("--raw-reference-handoff-json", type=Path)
    parser.add_argument("--raw-reference-allowed-source-root", type=Path, action="append", default=[])
    parser.add_argument("--raw-reference-recovery-material-candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--raw-reference-rights-status", default="")
    parser.add_argument(
        "--raw-reference-require-full-mix-reconstruction",
        dest="raw_reference_require_full_mix_reconstruction",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--raw-reference-allow-diagnostic-rejected-full-mix-reconstruction",
        dest="raw_reference_require_full_mix_reconstruction",
        action="store_false",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_full_pack_aris_metadata_loop.py",
        entrypoint_path="/app/tools/run_full_pack_aris_metadata_loop.py",
    )
    args = parse_args(argv)
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=args.machine_gate_json,
        repair_queue_json=args.repair_queue_json,
        full_pack_loop_plan_json=args.full_pack_loop_plan_json,
        reference_gap_plan_json=args.reference_gap_plan_json,
        existing_audio_training_examples_jsonl=args.existing_audio_training_examples_jsonl,
        candidate_roots=args.candidate_root,
        output_dir=args.output_dir,
        min_unique_tracks=int(args.min_unique_tracks),
        training_output_root=str(args.training_output_root),
        raw_reference_handoff_json=args.raw_reference_handoff_json,
        raw_reference_allowed_source_roots=args.raw_reference_allowed_source_root,
        raw_reference_recovery_material_candidate_roots=args.raw_reference_recovery_material_candidate_root,
        raw_reference_rights_status=str(args.raw_reference_rights_status),
        raw_reference_require_full_mix_reconstruction=bool(args.raw_reference_require_full_mix_reconstruction),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "loop_status": manifest["loop_status"],
                "goal_complete": manifest["goal_complete"],
                "next_action": manifest["next_action"],
                "remaining_blocker_count": len(manifest["remaining_blockers"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["goal_complete"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
