#!/usr/bin/env python3
"""Preflight external material supplied for raw-reference recovery requests."""

from __future__ import annotations

import argparse
import json
import re
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
MANIFEST_NAME = "preflight.json"
ITEMS_NAME = "items.jsonl"
READY_NAME = "ready.jsonl"
BLOCKED_NAME = "blocked.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
SOURCE_SCHEMA = "full_pack_raw_reference_recovery_action_execution_manifest.v1"
SOURCE_ROW_SCHEMA = "full_pack_raw_reference_recovery_action_execution_item.v1"
RESPONSE_SCHEMA = "full_pack_raw_reference_recovery_material_response.v1"
MANIFEST_SCHEMA = "full_pack_raw_reference_recovery_material_preflight_manifest.v1"
ITEM_SCHEMA = "full_pack_raw_reference_recovery_material_preflight_item.v1"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned", "user_owned"}
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
BLOCKED_SOURCE_ROW_OUTPUT_FIELDS = (
    "candidate_root",
    "ground_truth_manifest_json",
    "accepted_reference_manifest_json",
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
        "expected_acceleration_mode": "external_only_cpu_raw_reference_recovery_material_preflight",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_recovery_material_preflight_checks_metadata_and_paths_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_recovery_material_preflight_never_starts_training",
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
    for nested_key in ("safety", "contract", "ground_truth_contract"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def source_domain_allowed(*payloads: dict[str, Any]) -> bool:
    chunks: list[str] = []
    for payload in payloads:
        chunks.extend(
            [
                str(payload.get("provenance_source") or ""),
                str(payload.get("source") or ""),
                str(payload.get("verification_method") or ""),
            ]
        )
        contract = payload.get("ground_truth_contract")
        if isinstance(contract, dict):
            chunks.extend([str(contract.get("source") or ""), str(contract.get("verification_method") or "")])
    text = " ".join(chunks).lower()
    return "fl studio" in text or "supplied by user" in text


def permission_status(payload: dict[str, Any]) -> str:
    report = payload.get("report_alignment_contract")
    if not isinstance(report, dict):
        report = {}
    return str(report.get("rights_status") or payload.get("rights_status") or "").strip().lower()


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


def load_execution_items(exec_json: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    items_path = resolved_items_path(manifest, base_dir=exec_json.parent)
    expected_count = int(manifest.get("action_item_count") or 0)
    if items_path is None or not items_path.is_file():
        if expected_count > 0:
            failed.append("raw_reference_recovery_execution_items_jsonl_missing")
        return [], failed
    rows = list(iter_jsonl(items_path))
    for row_index, row in enumerate(rows, start=1):
        row_prefix = f"raw_reference_recovery_execution_row_{row_index:03d}"
        if row.get("schema") != SOURCE_ROW_SCHEMA:
            failed.append(f"{row_prefix}_schema_unexpected")
        failed.extend(collect_true_flag_failures(row_prefix, row))
        for field in BLOCKED_SOURCE_ROW_OUTPUT_FIELDS:
            if str(row.get(field) or ""):
                failed.append(f"{row_prefix}_{field}_present")
    if expected_count != len(rows):
        failed.append("raw_reference_recovery_execution_item_count_mismatch")
    return rows if not failed else [], stable_unique(failed)


def path_from_payload(raw_path: Any, *, base_dir: Path) -> Path | None:
    text = str(raw_path or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def path_inside_roots(path: Path, roots: Sequence[Path]) -> bool:
    if not roots:
        return False
    resolved_path = path.resolve(strict=False)
    for root in roots:
        try:
            resolved_path.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def expected_stems(row: dict[str, Any], template: dict[str, Any]) -> list[str]:
    stems = str_list(row.get("requested_stems"))
    if not stems:
        stems = str_list(template.get("requested_stems"))
    if not stems:
        stem = str(row.get("target_stem_type") or template.get("target_stem_type") or "")
        stems = [stem] if stem else []
    return stems


def response_path_for(row: dict[str, Any], *, response_root: Path) -> tuple[Path, list[str]]:
    source_action_id = str(row.get("source_action_item_id") or row.get("execution_item_id") or "")
    if not SAFE_ID_RE.fullmatch(source_action_id):
        return response_root / "invalid.json", ["source_action_item_id_unsafe_or_missing"]
    return response_root / f"{source_action_id}.json", []


def load_template(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    template_path = Path(str(row.get("template_json") or ""))
    if not str(template_path):
        return {}, ["template_json_missing"]
    if not template_path.is_file():
        return {}, ["template_json_not_found"]
    try:
        template = read_json(template_path)
    except (json.JSONDecodeError, ValueError):
        return {}, ["template_json_invalid"]
    if template.get("schema") != "full_pack_raw_reference_recovery_execution_template.v1":
        blockers.append("template_schema_unexpected")
    return template, blockers


def compare_response_to_request(
    *,
    response: dict[str, Any],
    row: dict[str, Any],
    template: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    fields = (
        "source_action_item_id",
        "source_bundle_alias",
        "route_intent",
        "required_material",
        "target_stem_type",
    )
    for field in fields:
        expected = str(row.get(field) or template.get(field) or "")
        actual = str(response.get(field) or "")
        if expected and actual != expected:
            blockers.append(f"response_{field}_mismatch")
    requested = expected_stems(row, template)
    if str_list(response.get("requested_stems")) != requested:
        blockers.append("response_requested_stems_mismatch")
    if response.get("material_status") != "provided_for_preflight":
        blockers.append("response_material_status_unexpected")
    return blockers


def declared_file_blockers(
    *,
    response: dict[str, Any],
    material_root: Path,
    allowed_source_roots: Sequence[Path],
    field: str,
    raw_path: Any,
) -> tuple[list[str], Path | None]:
    blockers: list[str] = []
    path = path_from_payload(raw_path, base_dir=material_root)
    if path is None:
        return [f"{field}_missing"], None
    if not path.is_file():
        blockers.append(f"{field}_not_found")
    if allowed_source_roots and not path_inside_roots(path, allowed_source_roots):
        blockers.append(f"{field}_outside_allowed_roots")
    if not allowed_source_roots:
        blockers.append("allowed_source_roots_missing")
    return blockers, path


def validate_ground_truth_manifest(
    *,
    payload: dict[str, Any],
    response: dict[str, Any],
    material_root: Path,
    stems: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    blockers.extend(collect_true_flag_failures("ground_truth_manifest", payload))
    contract = payload.get("ground_truth_contract")
    if not isinstance(contract, dict) or contract.get("verified") is not True:
        blockers.append("ground_truth_not_verified")
    if not source_domain_allowed(payload, response):
        blockers.append("source_domain_not_allowed")
    manifest_rights = permission_status(payload)
    response_rights = str(response.get("rights_status") or "").strip().lower()
    rights = response_rights or manifest_rights
    if not rights:
        blockers.append("missing_permission_status")
    elif rights not in TRAINABLE_PERMISSION_STATUSES:
        blockers.append("permission_not_trainable")
    if manifest_rights and response_rights and manifest_rights != response_rights:
        blockers.append("rights_status_mismatch")
    mix = payload.get("mix") if isinstance(payload.get("mix"), dict) else {}
    if mix.get("path") != "source_mix.flac":
        blockers.append("mix_path_unexpected")
    if not (material_root / str(mix.get("path") or "")).is_file():
        blockers.append("source_audio_not_found")
    stem_payloads = payload.get("stems")
    if not isinstance(stem_payloads, dict):
        blockers.append("stem_payload_missing")
    else:
        for stem in stems:
            stem_payload = stem_payloads.get(stem)
            if not isinstance(stem_payload, dict):
                blockers.append(f"stem_payload_missing:{stem}")
                continue
            expected_path = f"stems/{stem}.flac"
            if stem_payload.get("path") != expected_path:
                blockers.append(f"stem_path_unexpected:{stem}")
            if not (material_root / str(stem_payload.get("path") or "")).is_file():
                blockers.append(f"target_audio_not_found:{stem}")
    return blockers


def material_blockers(
    *,
    row: dict[str, Any],
    template: dict[str, Any],
    response: dict[str, Any],
    response_json: Path,
    allowed_source_roots: Sequence[Path],
) -> tuple[list[str], Path | None, Path | None]:
    blockers: list[str] = []
    blockers.extend(collect_true_flag_failures("response", response))
    blockers.extend(compare_response_to_request(response=response, row=row, template=template))
    if response.get("verification_declared") is not True:
        blockers.append("verification_not_declared")
    if not source_domain_allowed(response):
        blockers.append("response_source_domain_not_allowed")
    if str(response.get("source_mix_resolution") or "") == "pack_manifest_source_audio_path":
        blockers.append("listen_pack_source_mix_not_clean_reference")

    material_root = path_from_payload(response.get("material_root"), base_dir=response_json.parent)
    if material_root is None:
        return stable_unique([*blockers, "material_root_missing"]), None, None
    if not material_root.is_dir():
        blockers.append("material_root_directory_missing")
    if not allowed_source_roots:
        blockers.append("allowed_source_roots_missing")
    elif not path_inside_roots(material_root, allowed_source_roots):
        blockers.append("material_root_outside_allowed_roots")

    stems = expected_stems(row, template)
    gt_blockers, gt_path = declared_file_blockers(
        response=response,
        material_root=material_root,
        allowed_source_roots=allowed_source_roots,
        field="ground_truth_manifest_json",
        raw_path=response.get("ground_truth_manifest_json"),
    )
    blockers.extend(gt_blockers)
    mix_blockers, _mix_path = declared_file_blockers(
        response=response,
        material_root=material_root,
        allowed_source_roots=allowed_source_roots,
        field="source_mix_path",
        raw_path=response.get("source_mix_path"),
    )
    blockers.extend(mix_blockers)
    target_paths = response.get("target_stem_paths")
    if not isinstance(target_paths, dict):
        blockers.append("target_stem_paths_missing")
    else:
        for stem in stems:
            target_blockers, _target_path = declared_file_blockers(
                response=response,
                material_root=material_root,
                allowed_source_roots=allowed_source_roots,
                field=f"target_stem_path:{stem}",
                raw_path=target_paths.get(stem),
            )
            blockers.extend(target_blockers)

    if gt_path is not None and gt_path.is_file():
        try:
            gt_payload = read_json(gt_path)
        except (json.JSONDecodeError, ValueError):
            blockers.append("ground_truth_manifest_invalid_json")
        else:
            blockers.extend(
                validate_ground_truth_manifest(
                    payload=gt_payload,
                    response=response,
                    material_root=material_root,
                    stems=stems,
                )
            )
    return stable_unique(blockers), material_root, gt_path


def preflight_item(
    *,
    row: dict[str, Any],
    response_root: Path,
    allowed_source_roots: Sequence[Path],
) -> dict[str, Any]:
    template, blockers = load_template(row)
    response_json, response_path_blockers = response_path_for(row, response_root=response_root)
    blockers.extend(response_path_blockers)
    response: dict[str, Any] = {}
    material_root: Path | None = None
    ground_truth_manifest_json: Path | None = None
    if not response_json.is_file():
        blockers.append("response_json_missing")
        status = "blocked_missing_recovery_reference_material"
    else:
        try:
            response = read_json(response_json)
        except (json.JSONDecodeError, ValueError):
            blockers.append("response_json_invalid")
            status = "blocked_invalid_recovery_reference_material"
        else:
            if response.get("schema") != RESPONSE_SCHEMA:
                blockers.append("response_schema_unexpected")
            material_failed, material_root, ground_truth_manifest_json = material_blockers(
                row=row,
                template=template,
                response=response,
                response_json=response_json,
                allowed_source_roots=allowed_source_roots,
            )
            blockers.extend(material_failed)
            status = (
                "ready_for_raw_reference_recovery_verification"
                if not blockers
                else "blocked_invalid_recovery_reference_material"
            )
    blockers = stable_unique(blockers)
    stems = expected_stems(row, template)
    material_root_text = str(material_root) if material_root is not None else str(response.get("material_root") or "")
    return {
        "schema": ITEM_SCHEMA,
        "source_action_item_id": str(row.get("source_action_item_id") or ""),
        "execution_item_id": str(row.get("execution_item_id") or ""),
        "source_bundle_alias": str(row.get("source_bundle_alias") or ""),
        "route_intent": str(row.get("route_intent") or ""),
        "required_material": str(row.get("required_material") or ""),
        "target_stem_type": str(row.get("target_stem_type") or ""),
        "requested_stems": stems,
        "template_json": str(row.get("template_json") or ""),
        "response_json": str(response_json),
        "material_root": material_root_text,
        "candidate_root": material_root_text if status == "ready_for_raw_reference_recovery_verification" else "",
        "ground_truth_manifest_json": str(ground_truth_manifest_json) if ground_truth_manifest_json else "",
        "material_preflight_status": status,
        "blockers": blockers,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def count_by_stem(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(stem for row in rows for stem in row["requested_stems"])
    counts.pop("", None)
    return dict(sorted(counts.items()))


def count_field(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "") for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def status_for(*, failed: Sequence[str], ready_count: int, blocked_count: int) -> str:
    if failed:
        return "blocked"
    if ready_count == 0 and blocked_count == 0:
        return "no_raw_reference_recovery_material_preflight_needed"
    if blocked_count == 0:
        return "ready_for_raw_reference_recovery_verification"
    if ready_count > 0:
        return "partial_ready_waiting_for_recovery_reference_material"
    return "blocked_waiting_for_recovery_reference_material"


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Recovery Material Preflight",
            "",
            f"- preflight_status: `{manifest['preflight_status']}`",
            f"- ready_action_count: `{manifest['ready_action_count']}`",
            f"- blocked_action_count: `{manifest['blocked_action_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            "",
        ]
    )


def build_full_pack_raw_reference_recovery_material_preflight(
    *,
    raw_reference_recovery_execution_json: Path,
    response_root: Path,
    allowed_source_roots: Sequence[Path],
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
        raw_reference_recovery_execution_json=str(raw_reference_recovery_execution_json),
        response_root=str(response_root),
        allowed_source_roots=[str(path) for path in allowed_source_roots],
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

    rows = [] if failed else [
        preflight_item(row=row, response_root=response_root, allowed_source_roots=allowed_source_roots)
        for row in action_rows
    ]
    ready_rows = [
        row for row in rows if row["material_preflight_status"] == "ready_for_raw_reference_recovery_verification"
    ]
    blocked_rows = [
        row for row in rows if row["material_preflight_status"] != "ready_for_raw_reference_recovery_verification"
    ]
    status = status_for(failed=failed, ready_count=len(ready_rows), blocked_count=len(blocked_rows))
    route_counts = Counter(str(row.get("route_intent") or "") for row in rows)
    route_counts.pop("", None)
    status_counts = Counter(str(row.get("material_preflight_status") or "") for row in rows)
    status_counts.pop("", None)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "full_pack_raw_reference_recovery_material_preflight",
        "preflight_status": status,
        "failed_checks": failed,
        "source_failed_checks": source_failed,
        "raw_reference_recovery_execution_json": str(raw_reference_recovery_execution_json),
        "source_items_jsonl": str(resolved_items_path(source, base_dir=raw_reference_recovery_execution_json.parent) or ""),
        "template_dir": str((source.get("outputs") or {}).get("template_dir") or ""),
        "response_root": str(response_root),
        "allowed_source_roots": [str(path) for path in allowed_source_roots],
        "action_item_count": len(rows),
        "ready_action_count": len(ready_rows),
        "blocked_action_count": len(blocked_rows),
        "ready_by_stem": count_by_stem(ready_rows),
        "blocked_by_stem": count_by_stem(blocked_rows),
        "route_intent_counts": dict(sorted(route_counts.items())),
        "required_material_counts": count_field(rows, "required_material"),
        "material_preflight_status_counts": dict(sorted(status_counts.items())),
        "candidate_roots": [row["candidate_root"] for row in ready_rows],
        "accepted_reference_rows": 0,
        "eligible_reference_material_rows": len(ready_rows),
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "items_jsonl": str(output_dir / ITEMS_NAME),
            "ready_jsonl": str(output_dir / READY_NAME),
            "blocked_jsonl": str(output_dir / BLOCKED_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "raw_reference_recovery_material_preflight_not_for_listen_pack",
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
            "ready_rows_not_imported": True,
            "not_release_artifact": True,
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
        preflight_status=status,
        ready_action_count=len(ready_rows),
        blocked_action_count=len(blocked_rows),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-reference-recovery-execution-json", type=Path, required=True)
    parser.add_argument("--response-root", type=Path, required=True)
    parser.add_argument("--allowed-source-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_raw_reference_recovery_material_preflight.py",
        entrypoint_path="/app/tools/build_full_pack_raw_reference_recovery_material_preflight.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=args.raw_reference_recovery_execution_json,
        response_root=args.response_root,
        allowed_source_roots=args.allowed_source_root,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "preflight_status": manifest["preflight_status"],
                "ready_action_count": manifest["ready_action_count"],
                "blocked_action_count": manifest["blocked_action_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
