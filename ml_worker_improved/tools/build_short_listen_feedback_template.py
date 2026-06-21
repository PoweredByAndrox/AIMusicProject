#!/usr/bin/env python3
"""Build a short-path listen feedback template from an aggregate listen pack."""

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

from tools.artifact_name_policy import validate_artifact_relative_path
from tools.docker_runtime import require_docker_runtime

START_TIME = time.monotonic()
MANIFEST_NAME = "tpl.json"
TEMPLATE_NAME = "items.jsonl"
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

ISSUE_TAGS_BY_STEM = {
    "vocal": [
        "vocal_artifact",
        "vocal_missing_detail",
        "vocal_bleed",
        "backing_vocal_missing",
        "phase_or_stereo_issue",
    ],
    "minus": [
        "vocal_bleed_in_minus",
        "minus_muffled",
        "phase_or_stereo_issue",
        "missing_instrument_detail",
        "artifact_noise",
    ],
    "drums": ["drum_bleed", "transient_damage", "missing_instrument_detail", "artifact_noise"],
    "bass": ["bass_bleed", "bass_missing_detail", "phase_or_stereo_issue", "artifact_noise"],
    "piano": ["piano_bleed", "piano_missing_detail", "phase_or_stereo_issue", "artifact_noise"],
    "guitar": ["guitar_bleed", "guitar_missing_detail", "phase_or_stereo_issue", "artifact_noise"],
    "other": ["stem_bleed", "missing_instrument_detail", "phase_or_stereo_issue", "artifact_noise"],
}
LISTEN_AUDIO_FILE_PACK_SCHEMAS = {
    "short_listen_subset_pack_manifest.v1",
    "short_listen_repair_selection_pack_manifest.v1",
    "short_candidate_wave_listen_pack_manifest.v1",
}
ACCEPTED_PACK_SCHEMAS = {"aggregate_short_listen_pack_manifest.v1", *LISTEN_AUDIO_FILE_PACK_SCHEMAS}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_feedback_template",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_feedback_template_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_feedback_template_no_training",
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


def check_short_relative_path(audio_path: str) -> list[str]:
    try:
        validate_artifact_relative_path(audio_path)
    except ValueError:
        if Path(str(audio_path or "")).is_absolute() or not str(audio_path or ""):
            return ["template_audio_path_not_relative"]
        return ["template_audio_path_not_short_windows_safe"]
    return []


def suggested_issue_tags(stem_type: str) -> list[str]:
    return ISSUE_TAGS_BY_STEM.get(stem_type, ISSUE_TAGS_BY_STEM["other"])


def pack_items(pack: dict[str, Any]) -> list[dict[str, Any]]:
    if pack.get("schema") in LISTEN_AUDIO_FILE_PACK_SCHEMAS:
        rows = pack.get("listen_audio_files") if isinstance(pack.get("listen_audio_files"), list) else []
        return [
            {**row, "audio_path": str(row.get("artifact_relative_path") or row.get("audio_path") or "")}
            for row in rows
            if isinstance(row, dict)
        ]
    return [item for item in pack.get("items") or [] if isinstance(item, dict)]


def template_row(index: int, pack_item: dict[str, Any]) -> dict[str, Any]:
    stem_type = str(pack_item.get("stem_type") or "unknown")
    return {
        "schema": "short_listen_feedback_template_item.v1",
        "review_rank": index,
        "audio_path": str(pack_item.get("audio_path") or ""),
        "verdict": "",
        "allowed_verdicts": ["accepted", "rejected"],
        "issue_tags": [],
        "suggested_issue_tags": suggested_issue_tags(stem_type),
        "note": "",
        "source_alias": str(pack_item.get("source_alias") or ""),
        "stem_type": stem_type,
        "track_alias": str(pack_item.get("track_alias") or ""),
        "release_ready": False,
        "can_publish_as_result": False,
        "template_only": True,
    }


def build_summary(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short listen feedback template",
        "",
        f"- template_status: `{manifest['template_status']}`",
        f"- item_count: `{manifest['item_count']}`",
        f"- ready_for_feedback_manifest: `{str(manifest['ready_for_feedback_manifest']).lower()}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "Fill `verdict` with `accepted` or `rejected`. Rejected rows must include at least one `issue_tags` value.",
        "",
        "## Items",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['audio_path']}` `{row['stem_type']}` `{row['source_alias']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_short_listen_feedback_template(
    *,
    pack_manifest_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", pack_manifest_json=str(pack_manifest_json))

    pack = read_json(pack_manifest_json)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("pack", pack))
    if pack.get("schema") not in ACCEPTED_PACK_SCHEMAS:
        failed.append("pack_schema_unexpected")
    if pack.get("delivery_status") != "accepted":
        failed.append("pack_delivery_status_not_accepted")
    if pack.get("delivery_allowed") is not True:
        failed.append("pack_delivery_allowed_not_true")
    if pack.get("presentation_status") != "accepted_for_listen_pack":
        failed.append("pack_presentation_status_not_accepted")
    if pack.get("failed_checks") not in ([], None):
        failed.append("pack_failed_checks_not_empty")

    rows: list[dict[str, Any]] = []
    if not failed:
        rows_from_pack = pack_items(pack)
        if not rows_from_pack:
            failed.append("pack_items_empty")
        seen: set[str] = set()
        for index, item in enumerate(rows_from_pack, start=1):
            audio_path = str(item.get("audio_path") or "").replace("\\", "/").strip()
            failed.extend(check_short_relative_path(audio_path))
            if audio_path in seen:
                failed.append(f"template_duplicate_audio_path:{audio_path}")
                continue
            seen.add(audio_path)
            failed.extend(collect_true_flag_failures(f"pack_item_{index}", item))
            if not failed:
                item = {**item, "audio_path": audio_path}
                rows.append(template_row(index, item))

    failed = stable_unique(failed)
    if failed:
        rows = []
    manifest = {
        "schema": "short_listen_feedback_template_manifest.v1",
        "pack_id": str(pack.get("pack_id") or ""),
        "pack_manifest_json": str(pack_manifest_json),
        "template_status": "ready_for_human_review" if rows and not failed else "blocked",
        "ready_for_feedback_manifest": False,
        "template_jsonl": str(output_dir / TEMPLATE_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "item_count": len(rows),
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "feedback_template_not_for_listen_pack",
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
    write_jsonl(output_dir / TEMPLATE_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", template_status=manifest["template_status"], item_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short listen feedback template from an aggregate listen pack.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_feedback_template.py",
        entrypoint_path="/app/tools/build_short_listen_feedback_template.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_feedback_template(
        pack_manifest_json=args.pack_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "template_status": manifest["template_status"],
                "item_count": manifest["item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
