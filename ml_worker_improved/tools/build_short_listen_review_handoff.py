#!/usr/bin/env python3
"""Build a short-path listen review handoff from an aggregate pack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import validate_artifact_relative_path  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402

START_TIME = time.monotonic()
MANIFEST_NAME = "handoff.json"
CHECKLIST_NAME = "checklist.md"
PLAYLIST_NAME = "playlist.m3u"
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
ACCEPTED_PACK_SCHEMAS = {"aggregate_short_listen_pack_manifest.v1", *LISTEN_AUDIO_FILE_PACK_SCHEMAS}


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
        "expected_acceleration_mode": "external_only_cpu_listen_review_handoff",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_review_handoff_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_review_handoff_no_training",
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
            return ["handoff_audio_path_not_relative"]
        return ["handoff_audio_path_not_short_windows_safe"]
    return []


def relative_playlist_entry(audio_path: str, *, pack_root: Path, handoff_root: Path) -> str:
    normalized = audio_path.replace("\\", "/")
    target = pack_root / normalized
    return os.path.relpath(target, start=handoff_root).replace("\\", "/")


def pack_items(pack: dict[str, Any]) -> list[dict[str, Any]]:
    if pack.get("schema") in LISTEN_AUDIO_FILE_PACK_SCHEMAS:
        rows = pack.get("listen_audio_files") if isinstance(pack.get("listen_audio_files"), list) else []
        return [
            {**row, "audio_path": str(row.get("artifact_relative_path") or row.get("audio_path") or "")}
            for row in rows
            if isinstance(row, dict)
        ]
    return [item for item in pack.get("items") or [] if isinstance(item, dict)]


def build_playlist(rows: Sequence[dict[str, Any]]) -> str:
    lines = ["#EXTM3U"]
    for row in rows:
        label = f"{row['review_rank']:03d} {row['stem_type']} {row['source_alias']}"
        duration = int(float(row.get("duration_sec") or -1))
        lines.append(f"#EXTINF:{duration},{label}")
        lines.append(str(row["playlist_entry"]))
    return "\n".join(lines) + "\n"


def build_checklist(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short listen review handoff",
        "",
        f"- handoff_status: `{manifest['handoff_status']}`",
        f"- item_count: `{manifest['item_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "Mark each row in the feedback template as `accepted` or `rejected`.",
        "Rejected rows must include at least one issue tag.",
        "",
        "## Listen Order",
        "",
    ]
    for row in rows:
        tags = ", ".join(row.get("suggested_issue_tags") or [])
        lines.append(f"- [{row['review_rank']:03d}] `{row['audio_path']}` `{row['stem_type']}` `{row['source_alias']}`")
        if tags:
            lines.append(f"  suggested_issue_tags: `{tags}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def make_rows(
    pack: dict[str, Any],
    template_rows: Sequence[dict[str, Any]],
    failed: list[str],
    *,
    pack_root: Path,
    handoff_root: Path,
) -> list[dict[str, Any]]:
    items = pack_items(pack)
    items_by_path = {str(item.get("audio_path") or ""): item for item in items}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, template in enumerate(template_rows, start=1):
        audio_path = str(template.get("audio_path") or "").replace("\\", "/").strip()
        failed.extend(check_short_relative_path(audio_path))
        if audio_path in seen:
            failed.append(f"handoff_duplicate_audio_path:{audio_path}")
            continue
        seen.add(audio_path)
        pack_item = items_by_path.get(audio_path)
        if pack_item is None:
            failed.append(f"handoff_audio_path_not_in_pack:{audio_path}")
            continue
        failed.extend(collect_true_flag_failures(f"pack_item_{index}", pack_item))
        rows.append(
            {
                "schema": "short_listen_review_handoff_item.v1",
                "review_rank": int(template.get("review_rank") or index),
                "audio_path": audio_path,
                "playlist_entry": relative_playlist_entry(audio_path, pack_root=pack_root, handoff_root=handoff_root),
                "source_alias": str(pack_item.get("source_alias") or template.get("source_alias") or ""),
                "stem_type": str(pack_item.get("stem_type") or template.get("stem_type") or "unknown"),
                "track_alias": str(pack_item.get("track_alias") or ""),
                "duration_sec": pack_item.get("duration_sec"),
                "suggested_issue_tags": [str(tag) for tag in template.get("suggested_issue_tags") or [] if str(tag)],
                "verdict": "",
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    missing = sorted(set(items_by_path) - seen)
    failed.extend(f"handoff_pack_item_missing_from_template:{path}" for path in missing)
    rows.sort(key=lambda item: item["review_rank"])
    return rows


def build_short_listen_review_handoff(
    *,
    pack_manifest_json: Path,
    feedback_template_manifest_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", pack_manifest_json=str(pack_manifest_json), feedback_template_manifest_json=str(feedback_template_manifest_json))

    pack = read_json(pack_manifest_json)
    template = read_json(feedback_template_manifest_json)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("pack", pack))
    failed.extend(collect_true_flag_failures("feedback_template", template))
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
    if template.get("schema") != "short_listen_feedback_template_manifest.v1":
        failed.append("feedback_template_schema_unexpected")
    if template.get("template_status") != "ready_for_human_review":
        failed.append("feedback_template_status_not_ready")
    if template.get("ready_for_feedback_manifest") is not False:
        failed.append("feedback_template_ready_for_feedback_manifest_not_false")
    if template.get("failed_checks") not in ([], None):
        failed.append("feedback_template_failed_checks_not_empty")
    if str(template.get("pack_id") or "") != str(pack.get("pack_id") or ""):
        failed.append("feedback_template_pack_id_mismatch")

    rows: list[dict[str, Any]] = []
    if not failed:
        template_jsonl = Path(str(template.get("template_jsonl") or ""))
        if not template_jsonl.exists():
            failed.append("feedback_template_jsonl_missing")
        else:
            rows = make_rows(
                pack,
                read_jsonl(template_jsonl),
                failed,
                pack_root=pack_manifest_json.parent,
                handoff_root=output_dir,
            )
            if not rows:
                failed.append("handoff_items_empty")

    failed = stable_unique(failed)
    if failed:
        rows = []
    manifest = {
        "schema": "short_listen_review_handoff_manifest.v1",
        "pack_id": str(pack.get("pack_id") or ""),
        "pack_manifest_json": str(pack_manifest_json),
        "feedback_template_manifest_json": str(feedback_template_manifest_json),
        "handoff_status": "ready_for_listen_review" if rows and not failed else "blocked",
        "item_count": len(rows),
        "items": rows,
        "playlist_m3u": str(output_dir / PLAYLIST_NAME),
        "listen_checklist_md": str(output_dir / CHECKLIST_NAME),
        "progress_jsonl": str(progress_path),
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "review_handoff_not_for_listen_pack",
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
        "not_claimed": ["release_ready", "publishable_result", "human_listen_verdict", "semantic_ground_truth"],
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / PLAYLIST_NAME).write_text(build_playlist(rows), encoding="utf-8")
    (output_dir / CHECKLIST_NAME).write_text(build_checklist(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", handoff_status=manifest["handoff_status"], item_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short listen review handoff from a pack and feedback template.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--feedback-template-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_review_handoff.py",
        entrypoint_path="/app/tools/build_short_listen_review_handoff.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_review_handoff(
        pack_manifest_json=args.pack_manifest_json,
        feedback_template_manifest_json=args.feedback_template_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "handoff_status": manifest["handoff_status"],
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
