#!/usr/bin/env python3
"""Build an intake handoff from preflighted raw-reference recovery material."""

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
    TRAINABLE_PERMISSION_STATUSES,
    collect_true_flag_failures,
    path_inside_roots,
    permission_status,
    read_json,
    source_domain_allowed,
    stable_unique,
    str_list,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "handoff.json"
ITEMS_NAME = "items.jsonl"
READY_NAME = "ready.jsonl"
BLOCKED_NAME = "blocked.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
SOURCE_SCHEMA = "full_pack_raw_reference_recovery_material_preflight_manifest.v1"
SOURCE_ROW_SCHEMA = "full_pack_raw_reference_recovery_material_preflight_item.v1"
MANIFEST_SCHEMA = "full_pack_raw_reference_recovery_material_handoff_manifest.v1"
ITEM_SCHEMA = "full_pack_raw_reference_recovery_material_handoff_item.v1"
READY_PREFLIGHT_STATUS = "ready_for_raw_reference_recovery_verification"
READY_HANDOFF_STATUS = "ready_for_reference_corpus_intake"


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_raw_reference_recovery_material_handoff",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_recovery_material_handoff_checks_metadata_and_paths_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_recovery_material_handoff_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        yield payload


def resolved_output_path(manifest: dict[str, Any], key: str, *, base_dir: Path) -> Path | None:
    outputs = manifest.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get(key) or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_output_rows(
    *,
    preflight_json: Path,
    preflight: dict[str, Any],
    output_key: str,
    expected_count_key: str,
    failed: list[str],
) -> list[dict[str, Any]]:
    expected_count = int(preflight.get(expected_count_key) or 0)
    path = resolved_output_path(preflight, output_key, base_dir=preflight_json.parent)
    if path is None:
        if expected_count:
            failed.append(f"preflight_{output_key}_missing")
        return []
    if not path.is_file():
        if expected_count:
            failed.append(f"preflight_{output_key}_not_found")
        return []
    rows = list(iter_jsonl(path))
    if expected_count != len(rows):
        failed.append(f"preflight_{output_key}_count_mismatch")
    return rows


def status_for(*, failed: Sequence[str], ready_count: int, blocked_count: int) -> str:
    if failed:
        return "blocked"
    if ready_count == 0 and blocked_count == 0:
        return "no_raw_reference_recovery_material_handoff_needed"
    if ready_count and not blocked_count:
        return READY_HANDOFF_STATUS
    if ready_count:
        return "partial_ready_for_reference_corpus_intake"
    return "blocked_waiting_for_recovery_reference_material"


def count_by_stem(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(stem for row in rows for stem in row["requested_stems"])
    counts.pop("", None)
    return dict(sorted(counts.items()))


def path_from_text(raw_path: Any) -> Path | None:
    text = str(raw_path or "")
    return Path(text) if text else None


def validate_ground_truth(
    *,
    row: dict[str, Any],
    material_root: Path,
    ground_truth_manifest_json: Path,
) -> list[str]:
    blockers: list[str] = []
    try:
        payload = read_json(ground_truth_manifest_json)
    except (json.JSONDecodeError, ValueError):
        return ["ground_truth_manifest_invalid_json"]
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
    mix = payload.get("mix") if isinstance(payload.get("mix"), dict) else {}
    if mix.get("path") != "source_mix.flac":
        blockers.append("mix_path_unexpected")
    if not (material_root / str(mix.get("path") or "")).is_file():
        blockers.append("source_audio_not_found")
    stems_payload = payload.get("stems")
    requested_stems = str_list(row.get("requested_stems"))
    if not requested_stems:
        blockers.append("requested_stems_missing")
    if not isinstance(stems_payload, dict):
        blockers.append("stem_payload_missing")
    else:
        for stem in requested_stems:
            stem_payload = stems_payload.get(stem)
            if not isinstance(stem_payload, dict):
                blockers.append(f"stem_payload_missing:{stem}")
                continue
            expected_path = f"stems/{stem}.flac"
            if stem_payload.get("path") != expected_path:
                blockers.append(f"stem_path_unexpected:{stem}")
            if not (material_root / str(stem_payload.get("path") or "")).is_file():
                blockers.append(f"target_audio_not_found:{stem}")
    return stable_unique(blockers)


def ready_handoff_item(
    *,
    row: dict[str, Any],
    allowed_source_roots: Sequence[Path],
) -> dict[str, Any]:
    blockers: list[str] = []
    blockers.extend(collect_true_flag_failures("ready_row", row))
    if row.get("schema") != SOURCE_ROW_SCHEMA:
        blockers.append("ready_row_schema_unexpected")
    if row.get("material_preflight_status") != READY_PREFLIGHT_STATUS:
        blockers.append("material_preflight_status_not_ready")
    if row.get("blockers") not in ([], None):
        blockers.append("ready_row_blockers_not_empty")
    requested_stems = str_list(row.get("requested_stems"))
    if not requested_stems:
        blockers.append("requested_stems_missing")

    candidate_root = path_from_text(row.get("candidate_root") or row.get("material_root"))
    material_root = path_from_text(row.get("material_root") or row.get("candidate_root"))
    ground_truth_manifest_json = path_from_text(row.get("ground_truth_manifest_json"))
    if candidate_root is None:
        blockers.append("candidate_root_missing")
        candidate_root = Path("")
    if material_root is None:
        blockers.append("material_root_missing")
        material_root = Path("")
    if str(candidate_root) and not candidate_root.is_dir():
        blockers.append("candidate_root_directory_missing")
    if str(material_root) and not material_root.is_dir():
        blockers.append("material_root_directory_missing")
    if str(candidate_root) and str(material_root) and candidate_root.resolve(strict=False) != material_root.resolve(strict=False):
        blockers.append("candidate_root_material_root_mismatch")
    if not allowed_source_roots:
        blockers.append("allowed_source_roots_missing")
    elif str(candidate_root) and not path_inside_roots(candidate_root, allowed_source_roots):
        blockers.append("candidate_root_outside_allowed_roots")

    if ground_truth_manifest_json is None:
        blockers.append("ground_truth_manifest_json_missing")
        ground_truth_manifest_json = Path("")
    elif not ground_truth_manifest_json.is_file():
        blockers.append("ground_truth_manifest_json_not_found")
    elif allowed_source_roots and not path_inside_roots(ground_truth_manifest_json, allowed_source_roots):
        blockers.append("ground_truth_manifest_json_outside_allowed_roots")
    if str(material_root) and ground_truth_manifest_json.is_file():
        blockers.extend(
            validate_ground_truth(
                row=row,
                material_root=material_root,
                ground_truth_manifest_json=ground_truth_manifest_json,
            )
        )

    blockers = stable_unique(blockers)
    status = READY_HANDOFF_STATUS if not blockers else "blocked_invalid_recovery_reference_material"
    return {
        "schema": ITEM_SCHEMA,
        "source_action_item_id": str(row.get("source_action_item_id") or ""),
        "execution_item_id": str(row.get("execution_item_id") or ""),
        "source_bundle_alias": str(row.get("source_bundle_alias") or ""),
        "route_intent": str(row.get("route_intent") or ""),
        "required_material": str(row.get("required_material") or ""),
        "target_stem_type": str(row.get("target_stem_type") or ""),
        "requested_stems": requested_stems,
        "source_material_preflight_status": str(row.get("material_preflight_status") or ""),
        "handoff_status": status,
        "candidate_root": str(candidate_root),
        "material_root": str(material_root),
        "ground_truth_manifest_json": str(ground_truth_manifest_json),
        "blockers": blockers,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def blocked_handoff_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": ITEM_SCHEMA,
        "source_action_item_id": str(row.get("source_action_item_id") or ""),
        "execution_item_id": str(row.get("execution_item_id") or ""),
        "source_bundle_alias": str(row.get("source_bundle_alias") or ""),
        "route_intent": str(row.get("route_intent") or ""),
        "required_material": str(row.get("required_material") or ""),
        "target_stem_type": str(row.get("target_stem_type") or ""),
        "requested_stems": str_list(row.get("requested_stems")),
        "source_material_preflight_status": str(row.get("material_preflight_status") or ""),
        "handoff_status": "blocked_missing_recovery_reference_material",
        "candidate_root": "",
        "material_root": str(row.get("material_root") or ""),
        "ground_truth_manifest_json": str(row.get("ground_truth_manifest_json") or ""),
        "blockers": stable_unique(str_list(row.get("blockers")) or ["recovery_reference_material_not_ready"]),
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Recovery Material Handoff",
            "",
            f"- handoff_status: `{manifest['handoff_status']}`",
            f"- ready_handoff_count: `{manifest['ready_handoff_count']}`",
            f"- blocked_handoff_count: `{manifest['blocked_handoff_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            "",
        ]
    )


def build_full_pack_raw_reference_recovery_material_handoff(
    *,
    raw_reference_recovery_material_preflight_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", raw_reference_recovery_material_preflight_json=str(raw_reference_recovery_material_preflight_json))

    preflight = read_json(raw_reference_recovery_material_preflight_json)
    failed = collect_true_flag_failures("preflight", preflight)
    if preflight.get("schema") != SOURCE_SCHEMA:
        failed.append("preflight_schema_unexpected")
    preflight_failed = [str(item) for item in preflight.get("failed_checks") or []]
    if preflight_failed:
        failed.append("preflight_failed_checks_not_empty")
    source_status = str(preflight.get("preflight_status") or "")
    if source_status not in {
        READY_PREFLIGHT_STATUS,
        "partial_ready_waiting_for_recovery_reference_material",
        "blocked_waiting_for_recovery_reference_material",
        "no_raw_reference_recovery_material_preflight_needed",
    }:
        failed.append("preflight_status_unexpected")
    ready_rows = load_output_rows(
        preflight_json=raw_reference_recovery_material_preflight_json,
        preflight=preflight,
        output_key="ready_jsonl",
        expected_count_key="ready_action_count",
        failed=failed,
    )
    blocked_source_rows = load_output_rows(
        preflight_json=raw_reference_recovery_material_preflight_json,
        preflight=preflight,
        output_key="blocked_jsonl",
        expected_count_key="blocked_action_count",
        failed=failed,
    )
    failed = stable_unique(failed)

    allowed_source_roots = [Path(str(path)) for path in preflight.get("allowed_source_roots") or [] if str(path or "")]
    rows: list[dict[str, Any]] = [] if failed else [
        ready_handoff_item(row=row, allowed_source_roots=allowed_source_roots) for row in ready_rows
    ]
    if not failed:
        rows.extend(blocked_handoff_item(row) for row in blocked_source_rows)
    ready_handoffs = [row for row in rows if row["handoff_status"] == READY_HANDOFF_STATUS]
    blocked_handoffs = [row for row in rows if row["handoff_status"] != READY_HANDOFF_STATUS]
    status = status_for(failed=failed, ready_count=len(ready_handoffs), blocked_count=len(blocked_handoffs))
    status_counts = Counter(str(row.get("handoff_status") or "") for row in rows)
    status_counts.pop("", None)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "full_pack_raw_reference_recovery_material_handoff",
        "handoff_status": status,
        "failed_checks": failed,
        "preflight_failed_checks": preflight_failed,
        "raw_reference_recovery_material_preflight_json": str(raw_reference_recovery_material_preflight_json),
        "preflight_status": source_status,
        "allowed_source_roots": [str(path) for path in allowed_source_roots],
        "handoff_item_count": len(rows),
        "ready_handoff_count": len(ready_handoffs),
        "blocked_handoff_count": len(blocked_handoffs),
        "ready_by_stem": count_by_stem(ready_handoffs),
        "blocked_by_stem": count_by_stem(blocked_handoffs),
        "handoff_status_counts": dict(sorted(status_counts.items())),
        "candidate_roots": [row["candidate_root"] for row in ready_handoffs],
        "accepted_reference_rows": 0,
        "eligible_reference_material_rows": len(ready_handoffs),
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
        "presentation_status": "raw_reference_recovery_material_handoff_not_for_listen_pack",
        "release_status": "training_reference_material_handoff_only",
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
            "not_release_artifact": True,
            "accepted_rows_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, rows)
    write_jsonl(output_dir / READY_NAME, ready_handoffs)
    write_jsonl(output_dir / BLOCKED_NAME, blocked_handoffs)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        handoff_status=status,
        ready_handoff_count=len(ready_handoffs),
        blocked_handoff_count=len(blocked_handoffs),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-reference-recovery-material-preflight-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_raw_reference_recovery_material_handoff.py",
        entrypoint_path="/app/tools/build_full_pack_raw_reference_recovery_material_handoff.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_raw_reference_recovery_material_handoff(
        raw_reference_recovery_material_preflight_json=args.raw_reference_recovery_material_preflight_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "handoff_status": manifest["handoff_status"],
                "ready_handoff_count": manifest["ready_handoff_count"],
                "blocked_handoff_count": manifest["blocked_handoff_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
