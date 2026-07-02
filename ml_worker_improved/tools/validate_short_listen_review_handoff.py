#!/usr/bin/env python3
"""Validate a short listen review handoff against its pack and template."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import validate_artifact_relative_path  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.short_listen_gate_contract import (  # noqa: E402
    accepted_machine_gate_failures,
    staged_aggregate_pack_failures,
)

START_TIME = time.monotonic()
VALIDATION_NAME = "validation.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"

BLOCKED_TRUE_FLAGS = {
    "release_ready",
    "can_publish_as_result",
    "can_publish_as_release",
    "can_mark_release_ready",
    "thresholds_relaxed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
    "separator_training_started",
    "cleanup_routes_added",
}
LISTEN_AUDIO_FILE_PACK_SCHEMAS = {
    "short_listen_subset_pack_manifest.v1",
    "short_listen_repair_selection_pack_manifest.v1",
    "short_candidate_wave_listen_pack_manifest.v1",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected JSON object rows")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_review_handoff_validation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_review_handoff_validation_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_review_handoff_validation_no_training",
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
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    for nested_key in ("safety", "contract"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            for key in BLOCKED_TRUE_FLAGS:
                if nested.get(key) is True:
                    failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def check_short_relative_path(audio_path: str, *, prefix: str) -> list[str]:
    try:
        validate_artifact_relative_path(audio_path)
    except ValueError:
        if Path(str(audio_path or "")).is_absolute() or not str(audio_path or ""):
            return [f"{prefix}_audio_path_not_relative"]
        return [f"{prefix}_audio_path_not_short_windows_safe"]
    return []


def resolve_path(base: Path, value: Any, fallback_name: str) -> Path:
    raw = str(value or "")
    return Path(raw) if raw else base / fallback_name


def pack_items(pack: dict[str, Any]) -> list[dict[str, Any]]:
    if pack.get("schema") in LISTEN_AUDIO_FILE_PACK_SCHEMAS:
        rows = pack.get("listen_audio_files") if isinstance(pack.get("listen_audio_files"), list) else []
        return [
            {**row, "audio_path": str(row.get("artifact_relative_path") or row.get("audio_path") or "")}
            for row in rows
            if isinstance(row, dict)
        ]
    return [item for item in pack.get("items") or [] if isinstance(item, dict)]


def playlist_entries(path: Path) -> list[str]:
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(value.replace("\\", "/"))
    return entries


def playlist_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value.startswith("#EXTINF:"):
            continue
        labels.append(value.split(",", 1)[1].strip() if "," in value else value)
    return labels


def label_leaks_raw_or_long_id(label: str) -> bool:
    lower = label.lower()
    return (
        "candidate_id" in lower
        or len(label) > 64
        or bool(re.search(r"[0-9a-f]{16,}", lower))
        or "__" in label
    )


def checklist_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def expected_playlist_entry(audio_path: str, *, pack_root: Path, handoff_root: Path) -> str:
    normalized = str(audio_path or "").replace("\\", "/").strip()
    return os.path.relpath(pack_root / normalized, start=handoff_root).replace("\\", "/")


def playlist_target_path(entry: str, *, handoff_root: Path) -> Path:
    return (handoff_root / entry).resolve()


def build_summary(validation: dict[str, Any]) -> str:
    lines = [
        "# short listen review handoff validation",
        "",
        f"- validation_status: `{validation['validation_status']}`",
        f"- item_count: `{validation['item_count']}`",
        f"- release_ready: `{str(validation['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(validation['can_publish_as_result']).lower()}`",
        "",
        "## Failed Checks",
        "",
    ]
    if validation["failed_checks"]:
        lines.extend(f"- `{check}`" for check in validation["failed_checks"])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def validate_short_listen_review_handoff(
    *,
    handoff_manifest_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", handoff_manifest_json=str(handoff_manifest_json))

    handoff = read_json(handoff_manifest_json)
    handoff_root = handoff_manifest_json.parent
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("handoff", handoff))
    if handoff.get("schema") != "short_listen_review_handoff_manifest.v1":
        failed.append("handoff_schema_unexpected")
    if handoff.get("handoff_status") != "ready_for_listen_review":
        failed.append("handoff_status_not_ready")
    if handoff.get("failed_checks") not in ([], None):
        failed.append("handoff_failed_checks_not_empty")

    pack_path = Path(str(handoff.get("pack_manifest_json") or ""))
    template_path = Path(str(handoff.get("feedback_template_manifest_json") or ""))
    playlist_path = resolve_path(handoff_root, handoff.get("playlist_m3u"), "playlist.m3u")
    checklist_path = resolve_path(handoff_root, handoff.get("listen_checklist_md"), "checklist.md")
    for label, path in (
        ("pack_manifest", pack_path),
        ("feedback_template_manifest", template_path),
        ("playlist", playlist_path),
        ("checklist", checklist_path),
    ):
        if not path.exists():
            failed.append(f"{label}_missing")

    pack: dict[str, Any] = {}
    template: dict[str, Any] = {}
    template_rows: list[dict[str, Any]] = []
    if pack_path.exists():
        pack = read_json(pack_path)
        failed.extend(collect_true_flag_failures("pack", pack))
        failed.extend(staged_aggregate_pack_failures(pack))
        if pack.get("failed_checks") not in ([], None):
            failed.append("pack_failed_checks_not_empty")
    if template_path.exists():
        template = read_json(template_path)
        failed.extend(collect_true_flag_failures("feedback_template", template))
        if template.get("schema") != "short_listen_feedback_template_manifest.v1":
            failed.append("feedback_template_schema_unexpected")
        if template.get("template_status") != "ready_for_human_review":
            failed.append("feedback_template_status_not_ready")
        if template.get("failed_checks") not in ([], None):
            failed.append("feedback_template_failed_checks_not_empty")
        template_jsonl = Path(str(template.get("template_jsonl") or ""))
        if not template_jsonl.exists():
            failed.append("feedback_template_jsonl_missing")
        else:
            template_rows = read_jsonl(template_jsonl)
    if pack and template and str(pack.get("pack_id") or "") != str(template.get("pack_id") or ""):
        failed.append("feedback_template_pack_id_mismatch")
    if pack and str(pack.get("pack_id") or "") != str(handoff.get("pack_id") or ""):
        failed.append("handoff_pack_id_mismatch")
    if pack:
        gate_value = str(handoff.get("machine_gate_json") or template.get("machine_gate_json") or "")
        machine_gate_json = Path(gate_value) if gate_value else None
        failed.extend(
            accepted_machine_gate_failures(
                machine_gate_json=machine_gate_json,
                pack_manifest_json=pack_path,
                pack=pack,
            )
        )

    rows = [row for row in handoff.get("items") or [] if isinstance(row, dict)]
    for index, row in enumerate(rows, start=1):
        failed.extend(collect_true_flag_failures(f"handoff_item_{index}", row))
    item_paths = [str(row.get("audio_path") or "").replace("\\", "/").strip() for row in rows]
    failed.extend(check for path in item_paths for check in check_short_relative_path(path, prefix="handoff"))
    if len(item_paths) != len(set(item_paths)):
        failed.append("handoff_duplicate_audio_paths")
    if int(handoff.get("item_count") or 0) != len(rows):
        failed.append("handoff_item_count_mismatch")
    pack_paths = {str(item.get("audio_path") or "").replace("\\", "/").strip() for item in pack_items(pack)}
    template_paths = {str(item.get("audio_path") or "").replace("\\", "/").strip() for item in template_rows}
    if pack_paths and set(item_paths) != pack_paths:
        failed.append("handoff_items_do_not_match_pack")
    if template_paths and set(item_paths) != template_paths:
        failed.append("handoff_items_do_not_match_template")

    if playlist_path.exists():
        entries = playlist_entries(playlist_path)
        labels = playlist_labels(playlist_path)
        expected_entries = [
            expected_playlist_entry(path, pack_root=pack_path.parent, handoff_root=handoff_root)
            for path in item_paths
        ]
        if entries != expected_entries:
            failed.append("playlist_entries_do_not_match_manifest_items")
        pack_root_resolved = pack_path.parent.resolve()
        for entry in entries:
            target_path = playlist_target_path(entry, handoff_root=handoff_root)
            if not target_path.is_relative_to(pack_root_resolved):
                failed.append(f"playlist_target_outside_pack_root:{entry}")
                continue
            if not target_path.is_file():
                failed.append(f"playlist_target_missing:{entry}")
        if any("candidate_id" in entry or len(Path(entry).name) > 32 for entry in entries):
            failed.append("playlist_contains_raw_or_long_names")
        if any(label_leaks_raw_or_long_id(label) for label in labels):
            failed.append("playlist_contains_raw_or_long_labels")
    if checklist_path.exists():
        text = checklist_text(checklist_path)
        if any(path not in text for path in item_paths):
            failed.append("checklist_missing_manifest_audio_path")
        if "candidate_id" in text:
            failed.append("checklist_contains_raw_candidate_ids")

    failed = stable_unique(failed)
    validation = {
        "schema": "short_listen_review_handoff_validation.v1",
        "handoff_manifest_json": str(handoff_manifest_json),
        "validation_status": "passed" if not failed else "failed",
        "item_count": len(rows) if not failed else 0,
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "review_handoff_validation_not_for_listen_pack",
        "progress_jsonl": str(progress_path),
        "outputs": {
            "validation_json": str(output_dir / VALIDATION_NAME),
            "summary_md": str(output_dir / SUMMARY_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / VALIDATION_NAME, validation)
    (output_dir / SUMMARY_NAME).write_text(build_summary(validation), encoding="utf-8")
    progress(progress_path, "finish", validation_status=validation["validation_status"], item_count=validation["item_count"], failed_checks=failed)
    return validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a short listen review handoff.")
    parser.add_argument("--handoff-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="validate_short_listen_review_handoff.py",
        entrypoint_path="/app/tools/validate_short_listen_review_handoff.py",
    )
    args = parse_args(argv)
    validation = validate_short_listen_review_handoff(
        handoff_manifest_json=args.handoff_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "validation_path": str(args.output_dir / VALIDATION_NAME),
                "validation_status": validation["validation_status"],
                "item_count": validation["item_count"],
                "failed_checks": validation["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if validation["validation_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
