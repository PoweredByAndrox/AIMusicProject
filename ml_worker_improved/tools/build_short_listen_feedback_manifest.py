#!/usr/bin/env python3
"""Build item-level human listen feedback for a short aggregate listen pack."""

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

from tools.artifact_name_policy import validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.short_listen_gate_contract import (  # noqa: E402
    accepted_machine_gate_failures,
    staged_aggregate_pack_failures,
)


START_TIME = time.monotonic()
MANIFEST_NAME = "feedback.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
ALLOWED_ITEM_VERDICTS = {"accepted", "rejected"}
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def read_feedback_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
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
        "expected_acceleration_mode": "external_only_cpu_feedback_manifest",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_feedback_manifest_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_feedback_manifest_no_training",
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
    path = Path(str(audio_path or ""))
    if path.is_absolute() or not str(audio_path or ""):
        return ["feedback_audio_path_not_relative"]
    failed: list[str] = []
    for segment in path.parts:
        try:
            validate_artifact_segment(segment)
        except ValueError:
            failed.append("feedback_audio_path_not_short_windows_safe")
            break
    return failed


def canonical_audio_path(raw_path: Any) -> str:
    return str(raw_path or "").replace("\\", "/").strip()


def summarize(manifest: dict[str, Any]) -> str:
    lines = [
        "# short listen feedback",
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
        lines.append(f"- `{item['audio_path']}`: `{item['verdict']}` `{', '.join(item['issue_tags'])}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        for check in manifest["failed_checks"]:
            lines.append(f"- `{check}`")
    lines.append("")
    return "\n".join(lines)


def issue_tag_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in item.get("issue_tags") or []:
            key = str(tag)
            if key:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_short_listen_feedback_manifest(
    *,
    pack_manifest_json: Path,
    machine_gate_json: Path | None = None,
    output_dir: Path,
    feedback_items: Sequence[dict[str, Any]],
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        pack_manifest_json=str(pack_manifest_json),
        machine_gate_json=str(machine_gate_json or ""),
        feedback_item_count=len(feedback_items),
    )

    pack = read_json(pack_manifest_json)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("pack", pack))
    failed.extend(staged_aggregate_pack_failures(pack))
    failed.extend(
        accepted_machine_gate_failures(
            machine_gate_json=machine_gate_json,
            pack_manifest_json=pack_manifest_json,
            pack=pack,
        )
    )
    if pack.get("failed_checks") not in ([], None):
        failed.append("pack_failed_checks_not_empty")

    pack_items = [item for item in pack.get("items") or [] if isinstance(item, dict)]
    items_by_path = {str(item.get("audio_path") or ""): item for item in pack_items}
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
        pack_item = items_by_path.get(audio_path)
        if pack_item is None:
            failed.append(f"feedback_audio_path_not_in_pack:{audio_path}")
            continue
        verdict = str(feedback.get("verdict") or "")
        if verdict not in ALLOWED_ITEM_VERDICTS:
            failed.append(f"feedback_verdict_invalid:{audio_path or index}")
            continue
        issue_tags = [str(tag) for tag in feedback.get("issue_tags") or [] if str(tag)]
        if verdict == "rejected" and not issue_tags:
            failed.append(f"feedback_rejected_issue_tags_missing:{audio_path}")
            continue
        if pack_item.get("release_ready") is True:
            failed.append(f"feedback_pack_item_release_ready_true:{audio_path}")
        if pack_item.get("can_publish_as_result") is True:
            failed.append(f"feedback_pack_item_can_publish_as_result_true:{audio_path}")
        selected.append(
            {
                "schema": "short_listen_feedback_item.v1",
                "audio_path": audio_path,
                "verdict": verdict,
                "issue_tags": issue_tags,
                "note": str(feedback.get("note") or ""),
                "source_alias": str(pack_item.get("source_alias") or ""),
                "stem_type": str(pack_item.get("stem_type") or ""),
                "track_alias": str(pack_item.get("track_alias") or ""),
                "source_manifest": str(pack_item.get("source_manifest") or ""),
                "source_audio_path": str(pack_item.get("source_audio_path") or ""),
                "source_row": pack_item.get("source_row") if isinstance(pack_item.get("source_row"), dict) else {},
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
        if verdict == "rejected":
            rejected_audio_paths.append(audio_path)
        else:
            accepted_audio_paths.append(audio_path)

    pack_item_count = len(pack_items)
    reviewed_item_count = len(selected)
    if reviewed_item_count == 0:
        review_coverage_status = "none"
    elif reviewed_item_count == pack_item_count:
        review_coverage_status = "complete"
    else:
        review_coverage_status = "partial"
    if not feedback_items:
        failed.append("feedback_review_empty")
    if selected and not rejected_audio_paths and review_coverage_status != "complete":
        failed.append("feedback_accepted_review_incomplete")

    failed = stable_unique(failed)
    if failed:
        selected = []
        rejected_audio_paths = []
        accepted_audio_paths = []
        overall = "invalid"
    elif rejected_audio_paths:
        overall = "rejected"
    else:
        overall = "accepted"

    manifest = {
        "schema": "short_listen_feedback_manifest.v1",
        "pack_id": str(pack.get("pack_id") or ""),
        "pack_manifest_json": str(pack_manifest_json),
        "machine_gate_json": str(machine_gate_json or ""),
        "overall_listen_verdict": overall,
        "review_scope": "full_pack" if review_coverage_status == "complete" else "subset",
        "review_coverage_status": review_coverage_status,
        "reviewed_item_count": reviewed_item_count if not failed else 0,
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
        "presentation_status": "human_feedback_not_for_listen_pack",
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
    parser = argparse.ArgumentParser(description="Build short listen feedback manifest from item-level verdict rows.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--machine-gate-json", type=Path)
    parser.add_argument("--feedback-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_feedback_manifest.py",
        entrypoint_path="/app/tools/build_short_listen_feedback_manifest.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=args.pack_manifest_json,
        machine_gate_json=args.machine_gate_json,
        output_dir=args.output_dir,
        feedback_items=read_feedback_jsonl(args.feedback_jsonl),
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
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
