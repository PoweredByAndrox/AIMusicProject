#!/usr/bin/env python3
"""Preflight full-pack reference bundles before ARIS reference intake."""

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
SAFE_ALIAS_RE = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned", "user_owned"}
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
        "expected_acceleration_mode": "external_only_cpu_reference_bundle_preflight",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_bundle_preflight_reads_metadata_and_checks_paths_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "reference_bundle_preflight_no_training_started",
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
    for nested_key in ("safety", "contract", "ground_truth_contract"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def resolved_items_path(handoff: dict[str, Any], *, base_dir: Path) -> Path | None:
    outputs = handoff.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("items_jsonl") or "")
    text = text or str(handoff.get("items_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_handoff_items(handoff_json: Path, handoff: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    items_path = resolved_items_path(handoff, base_dir=handoff_json.parent)
    if items_path is None or not items_path.is_file():
        failed.append("handoff_items_jsonl_missing")
        return [], failed
    rows = list(iter_jsonl(items_path))
    for row in rows:
        if row.get("schema") != "full_pack_reference_bundle_handoff_item.v1":
            failed.append("handoff_item_schema_unexpected")
        failed.extend(collect_true_flag_failures("handoff_item", row))
        alias = str(row.get("bundle_alias") or "")
        if not SAFE_ALIAS_RE.fullmatch(alias):
            failed.append("handoff_item_alias_unsafe")
        stem = normalized_stem(row.get("stem_type"))
        if stem == "unknown":
            failed.append("handoff_item_stem_type_missing")
            continue
        required_files = row.get("required_files")
        expected = {"source_mix.flac", f"stems/{stem}.flac", "ground_truth_manifest.json"}
        if not isinstance(required_files, list) or not expected.issubset({str(item) for item in required_files}):
            failed.append("handoff_item_required_files_missing")
        if row.get("required_manifest_name") != "ground_truth_manifest.json":
            failed.append("handoff_item_required_manifest_unexpected")
        if row.get("required_directory") != f"refs/{alias}":
            failed.append("handoff_item_required_directory_unexpected")
    return rows if not failed else [], stable_unique(failed)


def path_from_bundle(raw_path: Any, *, bundle_dir: Path) -> Path:
    path = Path(str(raw_path or ""))
    if not path.is_absolute():
        path = bundle_dir / path
    return path


def source_domain_allowed(payload: dict[str, Any]) -> bool:
    contract = payload.get("ground_truth_contract")
    if not isinstance(contract, dict):
        return False
    text = f"{contract.get('source') or ''} {contract.get('verification_method') or ''}".lower()
    return "fl studio" in text or "supplied by user" in text


def permission_status(payload: dict[str, Any]) -> str:
    report = payload.get("report_alignment_contract")
    if not isinstance(report, dict):
        report = {}
    return str(report.get("rights_status") or payload.get("rights_status") or "").strip().lower()


def bundle_blockers(*, item: dict[str, Any], handoff_root: Path) -> tuple[list[str], Path, dict[str, Any] | None]:
    blockers: list[str] = []
    alias = str(item.get("bundle_alias") or "")
    stem = normalized_stem(item.get("stem_type"))
    bundle_dir = handoff_root / str(item.get("required_directory") or f"refs/{alias}")
    if not bundle_dir.is_dir():
        return ["bundle_directory_missing"], bundle_dir, None
    for required_file in item.get("required_files") or []:
        if not (bundle_dir / str(required_file)).is_file():
            blockers.append(f"required_file_missing:{required_file}")
    manifest_path = bundle_dir / "ground_truth_manifest.json"
    payload: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            payload = read_json(manifest_path)
        except (json.JSONDecodeError, ValueError):
            blockers.append("ground_truth_manifest_invalid_json")
    else:
        blockers.append("ground_truth_manifest_missing")
    if payload is None:
        return stable_unique(blockers), bundle_dir, None

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
    if not path_from_bundle(mix.get("path"), bundle_dir=bundle_dir).is_file():
        blockers.append("source_audio_not_found")
    stems = payload.get("stems")
    stem_payload = stems.get(stem) if isinstance(stems, dict) else None
    if not isinstance(stem_payload, dict):
        blockers.append("stem_payload_missing")
    else:
        if stem_payload.get("path") != f"stems/{stem}.flac":
            blockers.append("stem_path_unexpected")
        if not path_from_bundle(stem_payload.get("path"), bundle_dir=bundle_dir).is_file():
            blockers.append("target_audio_not_found")
    return stable_unique(blockers), bundle_dir, payload


def preflight_item(*, item: dict[str, Any], handoff_root: Path) -> dict[str, Any]:
    blockers, bundle_dir, payload = bundle_blockers(item=item, handoff_root=handoff_root)
    status = "blocked" if blockers else "ready_for_reference_intake"
    return {
        "schema": "full_pack_reference_bundle_preflight_item.v1",
        "bundle_alias": str(item.get("bundle_alias") or ""),
        "stem_type": normalized_stem(item.get("stem_type")),
        "source_collection_id": str(item.get("source_collection_id") or ""),
        "required_directory": str(item.get("required_directory") or ""),
        "bundle_dir": str(bundle_dir),
        "candidate_root": str(bundle_dir) if status == "ready_for_reference_intake" else "",
        "manifest_path": str(bundle_dir / "ground_truth_manifest.json"),
        "track_id": str((payload or {}).get("track_name") or ""),
        "preflight_status": status,
        "blockers": blockers,
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
    }


def status_for(*, failed: Sequence[str], ready_count: int, blocked_count: int) -> str:
    if failed:
        return "blocked"
    if blocked_count == 0:
        return "ready_for_reference_intake"
    if ready_count > 0:
        return "partial_ready_waiting_for_reference_bundles"
    return "blocked_waiting_for_reference_bundles"


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Reference Bundle Preflight",
            "",
            f"- preflight_status: `{manifest['preflight_status']}`",
            f"- ready_bundle_count: `{manifest['ready_bundle_count']}`",
            f"- blocked_bundle_count: `{manifest['blocked_bundle_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_full_pack_reference_bundle_preflight(
    *,
    handoff_json: Path,
    handoff_root: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", handoff_json=str(handoff_json), handoff_root=str(handoff_root))

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
    raw_items, item_failed = load_handoff_items(handoff_json, handoff)
    if not item_failed and expected_count != len(raw_items):
        failed.append("handoff_item_count_mismatch")
    failed.extend(item_failed)
    failed = stable_unique(failed)

    rows = [] if failed else [preflight_item(item=item, handoff_root=handoff_root) for item in raw_items]
    ready_rows = [row for row in rows if row["preflight_status"] == "ready_for_reference_intake"]
    blocked_rows = [row for row in rows if row["preflight_status"] != "ready_for_reference_intake"]
    candidate_roots = [row["candidate_root"] for row in ready_rows]
    ready_by_stem = Counter(row["stem_type"] for row in ready_rows)
    blocked_by_stem = Counter(row["stem_type"] for row in blocked_rows)
    status = status_for(failed=failed, ready_count=len(ready_rows), blocked_count=len(blocked_rows))
    manifest = {
        "schema": "full_pack_reference_bundle_preflight_manifest.v1",
        "manifest_type": "full_pack_reference_bundle_preflight",
        "preflight_status": status,
        "failed_checks": failed,
        "source_handoff_json": str(handoff_json),
        "handoff_root": str(handoff_root),
        "handoff_item_count": len(raw_items) if not failed else 0,
        "ready_bundle_count": len(ready_rows),
        "blocked_bundle_count": len(blocked_rows),
        "ready_by_stem": dict(sorted(ready_by_stem.items())),
        "blocked_by_stem": dict(sorted(blocked_by_stem.items())),
        "remaining_by_stem": {} if failed else int_mapping(handoff.get("remaining_by_stem")),
        "candidate_roots": candidate_roots,
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
        "presentation_status": "reference_bundle_preflight_not_for_listen_pack",
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
    write_jsonl(output_dir / ITEMS_NAME, rows)
    write_jsonl(output_dir / READY_NAME, ready_rows)
    write_jsonl(output_dir / BLOCKED_NAME, blocked_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        preflight_status=status,
        ready_bundle_count=len(ready_rows),
        blocked_bundle_count=len(blocked_rows),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight full-pack reference bundles before ARIS intake.")
    parser.add_argument("--handoff-json", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_reference_bundle_preflight.py",
        entrypoint_path="/app/tools/build_full_pack_reference_bundle_preflight.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_reference_bundle_preflight(
        handoff_json=args.handoff_json,
        handoff_root=args.handoff_root,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "preflight_status": manifest["preflight_status"],
                "ready_bundle_count": manifest["ready_bundle_count"],
                "blocked_bundle_count": manifest["blocked_bundle_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
