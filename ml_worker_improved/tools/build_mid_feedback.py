#!/usr/bin/env python3
"""Build repair-queue-compatible feedback from a minus middle audition pack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import validate_artifact_segment
from tools.build_short_listen_feedback_manifest import (
    ALLOWED_ITEM_VERDICTS,
    canonical_audio_path,
    check_short_relative_path,
    issue_tag_counts,
)
from tools.docker_runtime import require_docker_runtime

MANIFEST_NAME = "feedback.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "separator_training_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
    "cleanup_routes_added",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_mid_feedback_manifest",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "mid_feedback_manifest_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "mid_feedback_manifest_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    return failed


def select_feedback_rows(
    *,
    items: Sequence[dict[str, Any]],
    select_source_aliases: Sequence[str],
    select_audio_paths: Sequence[str],
    verdict: str,
    issue_tags: Sequence[str],
    note: str,
) -> list[dict[str, Any]]:
    aliases = {str(alias) for alias in select_source_aliases if str(alias)}
    paths = {canonical_audio_path(path) for path in select_audio_paths if str(path)}
    rows: list[dict[str, Any]] = []
    for item in items:
        audio_path = canonical_audio_path(item.get("audio_path"))
        alias = str(item.get("source_alias") or "")
        if (aliases and alias in aliases) or (paths and audio_path in paths):
            rows.append(
                {
                    "audio_path": audio_path,
                    "verdict": verdict,
                    "issue_tags": list(issue_tags),
                    "note": note,
                }
            )
    return rows


def enriched_source_row(mid_item: dict[str, Any], mid_pack_json: Path) -> dict[str, Any]:
    source_row = mid_item.get("source_row") if isinstance(mid_item.get("source_row"), dict) else {}
    source_row = dict(source_row)
    judge = source_row.get("judge_evidence") if isinstance(source_row.get("judge_evidence"), dict) else {}
    judge = dict(judge)
    judge["applied_windows"] = [
        {
            "window_id": str(mid_item.get("window_id") or ""),
            "start_sec": float(mid_item.get("window_start_sec") or 0.0),
            "duration_sec": float(mid_item.get("window_duration_sec") or 0.0),
            "reason": str(mid_item.get("window_reason") or ""),
            "diagnostic_audio_path": canonical_audio_path(mid_item.get("audio_path")),
            "diagnostic_source_alias": str(mid_item.get("source_alias") or ""),
        }
    ]
    judge["applied_window_count"] = 1
    judge["source_mid_pack_json"] = str(mid_pack_json)
    source_row["judge_evidence"] = judge
    source_row["diagnostic_mid_pack"] = {
        "mid_pack_json": str(mid_pack_json),
        "diagnostic_audio_path": canonical_audio_path(mid_item.get("audio_path")),
        "source_alias": str(mid_item.get("source_alias") or ""),
        "source_kind": str(mid_item.get("source_kind") or ""),
    }
    return source_row


def source_manifest_for(mid_item: dict[str, Any], mid_pack: dict[str, Any], mid_pack_json: Path) -> str:
    source_row = mid_item.get("source_row") if isinstance(mid_item.get("source_row"), dict) else {}
    for value in (
        mid_item.get("source_manifest"),
        source_row.get("source_manifest"),
        source_row.get("source_selection_manifest_json"),
        source_row.get("source_phase_repair_manifest_json"),
        mid_pack.get("source_pack_json"),
        str(mid_pack_json),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return str(mid_pack_json)


def summarize(manifest: dict[str, Any]) -> str:
    lines = [
        "# mid feedback",
        "",
        f"- overall_listen_verdict: `{manifest['overall_listen_verdict']}`",
        f"- feedback_item_count: `{manifest['feedback_item_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Items",
        "",
    ]
    for item in manifest["items"]:
        lines.append(f"- `{item['audio_path']}` `{item['verdict']}` `{', '.join(item['issue_tags'])}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_mid_feedback(
    *,
    mid_pack_json: Path,
    output_dir: Path,
    feedback_items: Sequence[dict[str, Any]],
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", mid_pack_json=str(mid_pack_json), feedback_item_count=len(feedback_items))

    mid_pack = read_json(mid_pack_json)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("mid_pack", mid_pack))
    if mid_pack.get("schema") != "minus_mid_audition_pack.v1":
        failed.append("mid_pack_schema_unexpected")
    if mid_pack.get("pack_status") != "ready_for_diagnostic_listen":
        failed.append("mid_pack_status_not_ready")
    if mid_pack.get("failed_checks") not in ([], None):
        failed.append("mid_pack_failed_checks_not_empty")
    if mid_pack.get("diagnostic_audio_exported") is not True:
        failed.append("mid_pack_diagnostic_audio_not_exported")

    mid_items = [item for item in mid_pack.get("items") or [] if isinstance(item, dict)]
    items_by_path = {canonical_audio_path(item.get("audio_path")): item for item in mid_items}
    selected: list[dict[str, Any]] = []
    rejected_audio_paths: list[str] = []
    accepted_audio_paths: list[str] = []
    seen_audio_paths: set[str] = set()

    for index, feedback in enumerate(feedback_items, start=1):
        audio_path = canonical_audio_path(feedback.get("audio_path"))
        failed.extend(check_short_relative_path(audio_path))
        if audio_path in seen_audio_paths:
            failed.append(f"feedback_duplicate_audio_path:{audio_path}")
            continue
        seen_audio_paths.add(audio_path)
        mid_item = items_by_path.get(audio_path)
        if mid_item is None:
            failed.append(f"feedback_audio_path_not_in_mid_pack:{audio_path}")
            continue
        clip_path = mid_pack_json.parent / audio_path
        if not clip_path.exists() or not clip_path.is_file():
            failed.append(f"feedback_mid_audio_missing:{audio_path}")
            continue
        verdict = str(feedback.get("verdict") or "")
        if verdict not in ALLOWED_ITEM_VERDICTS:
            failed.append(f"feedback_verdict_invalid:{audio_path or index}")
            continue
        issue_tags = [str(tag) for tag in feedback.get("issue_tags") or [] if str(tag)]
        if verdict == "rejected" and not issue_tags:
            failed.append(f"feedback_rejected_issue_tags_missing:{audio_path}")
            continue
        if mid_item.get("release_ready") is True:
            failed.append(f"feedback_mid_item_release_ready_true:{audio_path}")
        if mid_item.get("can_publish_as_result") is True:
            failed.append(f"feedback_mid_item_can_publish_as_result_true:{audio_path}")
        selected.append(
            {
                "schema": "short_listen_feedback_item.v1",
                "audio_path": audio_path,
                "diagnostic_clip_audio_path": audio_path,
                "verdict": verdict,
                "issue_tags": issue_tags,
                "note": str(feedback.get("note") or ""),
                "source_alias": str(mid_item.get("source_alias") or ""),
                "stem_type": str(mid_item.get("stem_type") or ""),
                "track_alias": str(mid_item.get("source_alias") or ""),
                "source_manifest": source_manifest_for(mid_item, mid_pack, mid_pack_json),
                "source_audio_path": str(mid_item.get("source_audio_path") or ""),
                "source_row": enriched_source_row(mid_item, mid_pack_json),
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
        if verdict == "rejected":
            rejected_audio_paths.append(audio_path)
        else:
            accepted_audio_paths.append(audio_path)

    if not feedback_items:
        failed.append("feedback_review_empty")

    reviewed_item_count = len(selected)
    pack_item_count = len(mid_items)
    if reviewed_item_count == 0:
        review_coverage_status = "none"
    elif reviewed_item_count == pack_item_count:
        review_coverage_status = "complete"
    else:
        review_coverage_status = "partial"
    if selected and not rejected_audio_paths and review_coverage_status != "complete":
        failed.append("feedback_accepted_review_incomplete")

    failed = stable_unique(failed)
    if failed:
        selected = []
        rejected_audio_paths = []
        accepted_audio_paths = []
        overall = "invalid"
        reviewed_item_count = 0
    elif rejected_audio_paths:
        overall = "rejected"
    else:
        overall = "accepted"

    manifest = {
        "schema": "short_listen_feedback_manifest.v1",
        "feedback_source_schema": "minus_mid_audition_pack.v1",
        "pack_id": str(mid_pack.get("source_pack_id") or mid_pack_json.parent.name),
        "pack_manifest_json": str(mid_pack_json),
        "mid_pack_json": str(mid_pack_json),
        "overall_listen_verdict": overall,
        "review_scope": "diagnostic_subset" if review_coverage_status != "complete" else "diagnostic_full_pack",
        "review_coverage_status": review_coverage_status,
        "reviewed_item_count": reviewed_item_count,
        "pack_item_count": pack_item_count,
        "feedback_item_count": len(selected),
        "items": selected,
        "rejected_audio_paths": rejected_audio_paths,
        "accepted_audio_paths": accepted_audio_paths,
        "issue_tag_counts": issue_tag_counts(selected),
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "diagnostic_feedback_not_for_listen_pack",
        "progress_jsonl": str(progress_path),
        "outputs": {
            "listen_feedback_json": str(output_dir / MANIFEST_NAME),
            "listen_feedback_summary_md": str(output_dir / SUMMARY_NAME),
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
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summarize(manifest), encoding="utf-8")
    progress(progress_path, "finish", overall_listen_verdict=overall, feedback_item_count=len(selected), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feedback from a minus middle audition pack.")
    parser.add_argument("--mid-pack-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feedback-jsonl", type=Path)
    parser.add_argument("--select-source", action="append", default=[])
    parser.add_argument("--select-audio-path", action="append", default=[])
    parser.add_argument("--verdict", choices=sorted(ALLOWED_ITEM_VERDICTS), default="rejected")
    parser.add_argument("--issue-tag", action="append", default=[])
    parser.add_argument("--note", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_mid_feedback.py", entrypoint_path="/app/tools/build_mid_feedback.py")
    args = parse_args(argv)
    mid_pack = read_json(args.mid_pack_json)
    mid_items = [item for item in mid_pack.get("items") or [] if isinstance(item, dict)]
    if args.feedback_jsonl:
        feedback_items = read_jsonl(args.feedback_jsonl)
    else:
        feedback_items = select_feedback_rows(
            items=mid_items,
            select_source_aliases=tuple(str(item) for item in args.select_source),
            select_audio_paths=tuple(str(item) for item in args.select_audio_path),
            verdict=str(args.verdict),
            issue_tags=tuple(str(item) for item in args.issue_tag),
            note=str(args.note),
        )
    manifest = build_mid_feedback(
        mid_pack_json=args.mid_pack_json,
        output_dir=args.output_dir,
        feedback_items=feedback_items,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "overall_listen_verdict": manifest["overall_listen_verdict"],
                "feedback_item_count": manifest["feedback_item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["overall_listen_verdict"] != "invalid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
