#!/usr/bin/env python3
"""Build material response JSON files for raw-reference recovery requests."""

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

from tools.build_full_pack_raw_reference_recovery_material_preflight import (  # noqa: E402
    RESPONSE_SCHEMA,
    SOURCE_SCHEMA,
    TRAINABLE_PERMISSION_STATUSES,
    collect_true_flag_failures,
    expected_stems,
    load_execution_items,
    load_template,
    path_inside_roots,
    permission_status,
    read_json,
    response_path_for,
    source_domain_allowed,
    stable_unique,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.full_pack_artifact_scope import is_non_production_artifact_path  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "responses.json"
ITEMS_NAME = "items.jsonl"
READY_NAME = "ready.jsonl"
BLOCKED_NAME = "blocked.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
MANIFEST_SCHEMA = "full_pack_raw_reference_recovery_material_responses_manifest.v1"
ITEM_SCHEMA = "full_pack_raw_reference_recovery_material_response_builder_item.v1"


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_raw_reference_recovery_material_responses",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_recovery_material_response_builder_checks_metadata_and_paths_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_recovery_material_response_builder_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def iter_candidate_bundle_roots(candidate_roots: Sequence[Path]) -> list[Path]:
    roots: list[Path] = []
    for root in candidate_roots:
        if root.is_file() and root.name == "ground_truth_manifest.json":
            roots.append(root.parent)
            continue
        if (root / "ground_truth_manifest.json").is_file():
            roots.append(root)
            continue
        if root.is_dir():
            roots.extend(sorted(path.parent for path in root.rglob("ground_truth_manifest.json")))
    resolved_seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve(strict=False))
        if key not in resolved_seen:
            resolved_seen.add(key)
            unique.append(root)
    return unique


def normalize_allowed_roots(roots: Sequence[Path]) -> list[Path]:
    normalized: list[Path] = []
    for root in roots:
        normalized.append(root.parent if root.is_file() and root.name == "ground_truth_manifest.json" else root)
    resolved_seen: set[str] = set()
    unique: list[Path] = []
    for root in normalized:
        key = str(root.resolve(strict=False))
        if key not in resolved_seen:
            resolved_seen.add(key)
            unique.append(root)
    return unique


def rel_text(path: str) -> str:
    return path.replace("\\", "/")


def status_for(*, failed: Sequence[str], ready_count: int, blocked_count: int) -> str:
    if failed:
        return "blocked"
    if ready_count == 0 and blocked_count == 0:
        return "no_raw_reference_recovery_material_responses_needed"
    if blocked_count == 0:
        return "ready_for_raw_reference_recovery_material_preflight"
    if ready_count > 0:
        return "partial_ready_waiting_for_recovery_reference_material"
    return "blocked_waiting_for_recovery_reference_material"


def count_by_stem(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(stem for row in rows for stem in row["requested_stems"])
    counts.pop("", None)
    return dict(sorted(counts.items()))


def count_field(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "") for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def candidate_blockers(
    *,
    root: Path,
    requested_stems: Sequence[str],
    allowed_source_roots: Sequence[Path],
) -> tuple[list[str], dict[str, Any], str]:
    blockers: list[str] = []
    if not root.is_dir():
        return ["candidate_root_directory_missing"], {}, ""
    if is_non_production_artifact_path(root):
        blockers.append("candidate_root_non_production_artifact")
    if allowed_source_roots and not path_inside_roots(root, allowed_source_roots):
        blockers.append("candidate_root_outside_allowed_roots")
    if not allowed_source_roots:
        blockers.append("allowed_source_roots_missing")

    manifest_path = root / "ground_truth_manifest.json"
    if not manifest_path.is_file():
        return stable_unique([*blockers, "ground_truth_manifest_missing"]), {}, ""
    try:
        payload = read_json(manifest_path)
    except (json.JSONDecodeError, ValueError):
        return stable_unique([*blockers, "ground_truth_manifest_invalid_json"]), {}, ""

    blockers.extend(collect_true_flag_failures("ground_truth_manifest", payload))
    contract = payload.get("ground_truth_contract")
    if not isinstance(contract, dict) or contract.get("verified") is not True:
        blockers.append("ground_truth_not_verified")
    if not source_domain_allowed(payload):
        blockers.append("source_domain_not_allowed")

    rights = permission_status(payload)
    if not rights:
        blockers.append("missing_permission_status")
    elif rights not in TRAINABLE_PERMISSION_STATUSES:
        blockers.append("permission_not_trainable")

    if str(payload.get("source_mix_resolution") or "") == "pack_manifest_source_audio_path":
        blockers.append("listen_pack_source_mix_not_clean_reference")

    mix = payload.get("mix") if isinstance(payload.get("mix"), dict) else {}
    if mix.get("path") != "source_mix.flac":
        blockers.append("mix_path_unexpected")
    if not (root / str(mix.get("path") or "")).is_file():
        blockers.append("source_audio_not_found")

    stems_payload = payload.get("stems")
    if not isinstance(stems_payload, dict):
        blockers.append("stem_payload_missing")
    else:
        for stem in requested_stems:
            stem_payload = stems_payload.get(stem)
            if not isinstance(stem_payload, dict):
                blockers.append(f"candidate_missing_requested_stem:{stem}")
                continue
            expected_path = f"stems/{stem}.flac"
            if stem_payload.get("path") != expected_path:
                blockers.append(f"stem_path_unexpected:{stem}")
            if not (root / str(stem_payload.get("path") or "")).is_file():
                blockers.append(f"target_audio_not_found:{stem}")

    source_text = ""
    if isinstance(contract, dict):
        source_text = str(contract.get("source") or contract.get("verification_method") or "")
    return stable_unique(blockers), payload, rights


def build_response_payload(
    *,
    row: dict[str, Any],
    template: dict[str, Any],
    material_root: Path,
    rights_status: str,
    manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    requested_stems = expected_stems(row, template)
    contract = manifest_payload.get("ground_truth_contract")
    provenance_source = ""
    if isinstance(contract, dict):
        provenance_source = str(contract.get("source") or contract.get("verification_method") or "")
    return {
        "schema": RESPONSE_SCHEMA,
        "source_action_item_id": str(row.get("source_action_item_id") or row.get("execution_item_id") or ""),
        "source_bundle_alias": str(row.get("source_bundle_alias") or template.get("source_bundle_alias") or ""),
        "route_intent": str(row.get("route_intent") or template.get("route_intent") or ""),
        "required_material": str(row.get("required_material") or template.get("required_material") or ""),
        "target_stem_type": str(row.get("target_stem_type") or template.get("target_stem_type") or ""),
        "requested_stems": requested_stems,
        "material_status": "provided_for_preflight",
        "material_root": str(material_root),
        "ground_truth_manifest_json": "ground_truth_manifest.json",
        "source_mix_path": "source_mix.flac",
        "target_stem_paths": {stem: rel_text(f"stems/{stem}.flac") for stem in requested_stems},
        "rights_status": rights_status,
        "provenance_source": provenance_source,
        "verification_declared": True,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
    }


def build_item(
    *,
    row: dict[str, Any],
    candidate_bundle_roots: Sequence[Path],
    response_root: Path,
    allowed_source_roots: Sequence[Path],
) -> dict[str, Any]:
    template, blockers = load_template(row)
    requested_stems = expected_stems(row, template)
    if not requested_stems:
        blockers.append("requested_stems_missing")
    response_json, response_path_blockers = response_path_for(row, response_root=response_root)
    blockers.extend(response_path_blockers)
    candidate_failures: list[str] = []
    selected_root: Path | None = None
    selected_payload: dict[str, Any] = {}
    selected_rights = ""

    if requested_stems:
        for root in candidate_bundle_roots:
            root_blockers, payload, rights = candidate_blockers(
                root=root,
                requested_stems=requested_stems,
                allowed_source_roots=allowed_source_roots,
            )
            if not root_blockers:
                selected_root = root
                selected_payload = payload
                selected_rights = rights
                break
            candidate_failures.extend(root_blockers)

    if selected_root is None:
        blockers.extend(candidate_failures or ["no_matching_clean_reference_material"])
        blockers.append("no_matching_clean_reference_material")
        blockers = stable_unique(blockers)
        return {
            "schema": ITEM_SCHEMA,
            "source_action_item_id": str(row.get("source_action_item_id") or row.get("execution_item_id") or ""),
            "execution_item_id": str(row.get("execution_item_id") or ""),
            "source_bundle_alias": str(row.get("source_bundle_alias") or ""),
            "route_intent": str(row.get("route_intent") or ""),
            "required_material": str(row.get("required_material") or ""),
            "target_stem_type": str(row.get("target_stem_type") or ""),
            "requested_stems": requested_stems,
            "response_json": str(response_json),
            "material_root": "",
            "ground_truth_manifest_json": "",
            "response_builder_status": "blocked_missing_recovery_reference_material",
            "blockers": blockers,
            "response_written": False,
            "accepted_as_ground_truth": False,
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "training_started": False,
        }

    response_payload = build_response_payload(
        row=row,
        template=template,
        material_root=selected_root,
        rights_status=selected_rights,
        manifest_payload=selected_payload,
    )
    if blockers:
        status = "blocked_invalid_recovery_reference_material_request"
        response_written = False
    else:
        write_json(response_json, response_payload)
        status = "response_written_for_raw_reference_recovery_material_preflight"
        response_written = True

    return {
        "schema": ITEM_SCHEMA,
        "source_action_item_id": response_payload["source_action_item_id"],
        "execution_item_id": str(row.get("execution_item_id") or ""),
        "source_bundle_alias": response_payload["source_bundle_alias"],
        "route_intent": response_payload["route_intent"],
        "required_material": response_payload["required_material"],
        "target_stem_type": response_payload["target_stem_type"],
        "requested_stems": requested_stems,
        "response_json": str(response_json),
        "material_root": str(selected_root),
        "ground_truth_manifest_json": str(selected_root / "ground_truth_manifest.json"),
        "response_builder_status": status,
        "blockers": stable_unique(blockers),
        "response_written": response_written,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Recovery Material Responses",
            "",
            f"- response_status: `{manifest['response_status']}`",
            f"- ready_response_count: `{manifest['ready_response_count']}`",
            f"- blocked_response_count: `{manifest['blocked_response_count']}`",
            f"- written_response_count: `{manifest['written_response_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            "",
        ]
    )


def build_full_pack_raw_reference_recovery_material_responses(
    *,
    raw_reference_recovery_execution_json: Path,
    candidate_roots: Sequence[Path],
    response_root: Path,
    output_dir: Path,
    allowed_source_roots: Sequence[Path] = (),
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    effective_allowed_roots = normalize_allowed_roots(allowed_source_roots or candidate_roots)
    progress(
        progress_path,
        "start",
        raw_reference_recovery_execution_json=str(raw_reference_recovery_execution_json),
        candidate_roots=[str(path) for path in candidate_roots],
        response_root=str(response_root),
        allowed_source_roots=[str(path) for path in effective_allowed_roots],
    )

    source = read_json(raw_reference_recovery_execution_json)
    failed = collect_true_flag_failures("raw_reference_recovery_execution", source)
    if source.get("schema") != SOURCE_SCHEMA:
        failed.append("raw_reference_recovery_execution_schema_unexpected")
    source_failed = [str(item) for item in source.get("failed_checks") or []]
    if source_failed:
        failed.append("raw_reference_recovery_execution_failed_checks_not_empty")
    execution_status = str(source.get("execution_status") or "")
    if execution_status not in {
        "blocked_waiting_for_external_reference_material",
        "no_raw_reference_recovery_execution_needed",
    }:
        failed.append("raw_reference_recovery_execution_status_unexpected")
    action_rows, row_failed = load_execution_items(raw_reference_recovery_execution_json, source)
    failed.extend(row_failed)
    failed = stable_unique(failed)

    if action_rows:
        for row in action_rows:
            response_json, response_path_blockers = response_path_for(row, response_root=response_root)
            if response_path_blockers:
                continue
            if response_json.is_file():
                response_json.unlink()

    candidate_bundle_roots = [] if failed else iter_candidate_bundle_roots(candidate_roots)
    rows = [] if failed else [
        build_item(
            row=row,
            candidate_bundle_roots=candidate_bundle_roots,
            response_root=response_root,
            allowed_source_roots=effective_allowed_roots,
        )
        for row in action_rows
    ]
    ready_rows = [
        row
        for row in rows
        if row["response_builder_status"] == "response_written_for_raw_reference_recovery_material_preflight"
    ]
    blocked_rows = [
        row
        for row in rows
        if row["response_builder_status"] != "response_written_for_raw_reference_recovery_material_preflight"
    ]
    status = status_for(failed=failed, ready_count=len(ready_rows), blocked_count=len(blocked_rows))
    status_counts = Counter(str(row.get("response_builder_status") or "") for row in rows)
    status_counts.pop("", None)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "full_pack_raw_reference_recovery_material_responses",
        "response_status": status,
        "failed_checks": failed,
        "source_failed_checks": source_failed,
        "raw_reference_recovery_execution_json": str(raw_reference_recovery_execution_json),
        "candidate_roots": [str(path) for path in candidate_roots],
        "candidate_bundle_roots": [str(path) for path in candidate_bundle_roots],
        "response_root": str(response_root),
        "allowed_source_roots": [str(path) for path in effective_allowed_roots],
        "action_item_count": len(rows),
        "ready_response_count": len(ready_rows),
        "blocked_response_count": len(blocked_rows),
        "written_response_count": len(ready_rows),
        "ready_by_stem": count_by_stem(ready_rows),
        "blocked_by_stem": count_by_stem(blocked_rows),
        "route_intent_counts": count_field(rows, "route_intent"),
        "required_material_counts": count_field(rows, "required_material"),
        "response_builder_status_counts": dict(sorted(status_counts.items())),
        "response_paths": [row["response_json"] for row in ready_rows],
        "accepted_reference_rows": 0,
        "eligible_reference_material_rows": 0,
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "items_jsonl": str(output_dir / ITEMS_NAME),
            "ready_jsonl": str(output_dir / READY_NAME),
            "blocked_jsonl": str(output_dir / BLOCKED_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
            "response_root": str(response_root),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "raw_reference_recovery_material_responses_not_for_listen_pack",
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
            "responses_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, rows)
    write_jsonl(output_dir / READY_NAME, ready_rows)
    write_jsonl(output_dir / BLOCKED_NAME, blocked_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        response_status=status,
        ready_response_count=len(ready_rows),
        blocked_response_count=len(blocked_rows),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-reference-recovery-execution-json", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--response-root", type=Path, required=True)
    parser.add_argument("--allowed-source-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_raw_reference_recovery_material_responses.py",
        entrypoint_path="/app/tools/build_full_pack_raw_reference_recovery_material_responses.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=args.raw_reference_recovery_execution_json,
        candidate_roots=args.candidate_root,
        response_root=args.response_root,
        allowed_source_roots=args.allowed_source_root,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "response_status": manifest["response_status"],
                "ready_response_count": manifest["ready_response_count"],
                "blocked_response_count": manifest["blocked_response_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
