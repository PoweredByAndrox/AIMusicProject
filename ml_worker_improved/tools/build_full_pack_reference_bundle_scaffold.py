#!/usr/bin/env python3
"""Create local directory scaffolds for full-pack reference bundles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_full_pack_reference_bundle_preflight import (  # noqa: E402
    collect_true_flag_failures,
    load_handoff_items,
    read_json,
    stable_unique,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "scaffold.json"
ITEMS_NAME = "items.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
TEMPLATE_NAME = "tpl.json"


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
        "expected_acceleration_mode": "external_only_cpu_reference_bundle_scaffold",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_bundle_scaffold_writes_metadata_dirs_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "reference_bundle_scaffold_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stem_from_item(item: dict[str, Any]) -> str:
    return str(item.get("stem_type") or "unknown")


def default_template(item: dict[str, Any]) -> dict[str, Any]:
    stem = stem_from_item(item)
    return {
        "track_name": "<unique_track_id>",
        "ground_truth_contract": {
            "verified": True,
            "source": "FL Studio split mixer tracks plus full mix supplied by user",
        },
        "mix": {"path": "source_mix.flac"},
        "stems": {stem: {"path": f"stems/{stem}.flac"}},
        "report_alignment_contract": {"rights_status": "user_owned"},
    }


def scaffold_item(*, item: dict[str, Any], handoff_root: Path) -> dict[str, Any]:
    bundle_dir = handoff_root / str(item.get("required_directory") or f"refs/{item.get('bundle_alias')}")
    stems_dir = bundle_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    template = item.get("ground_truth_manifest_template")
    if not isinstance(template, dict):
        template = default_template(item)
    template_path = bundle_dir / TEMPLATE_NAME
    write_json(template_path, template)
    return {
        "schema": "full_pack_reference_bundle_scaffold_item.v1",
        "bundle_alias": str(item.get("bundle_alias") or ""),
        "stem_type": stem_from_item(item),
        "source_collection_id": str(item.get("source_collection_id") or ""),
        "required_directory": str(item.get("required_directory") or ""),
        "bundle_dir": str(bundle_dir),
        "template_json": str(template_path),
        "required_files": [str(value) for value in item.get("required_files") or []],
        "next_intake_command_args": list(item.get("next_intake_command_args") or []),
        "scaffold_status": "created",
        "ground_truth_manifest_written": False,
        "audio_placeholder_written": False,
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Reference Bundle Scaffold",
            "",
            f"- scaffold_status: `{manifest['scaffold_status']}`",
            f"- bundle_directory_count: `{manifest['bundle_directory_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_full_pack_reference_bundle_scaffold(
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

    rows = [] if failed else [scaffold_item(item=item, handoff_root=handoff_root) for item in raw_items]
    by_stem = Counter(row["stem_type"] for row in rows)
    status = "blocked" if failed else "created_reference_bundle_scaffold"
    manifest = {
        "schema": "full_pack_reference_bundle_scaffold_manifest.v1",
        "manifest_type": "full_pack_reference_bundle_scaffold",
        "scaffold_status": status,
        "failed_checks": failed,
        "source_handoff_json": str(handoff_json),
        "handoff_root": str(handoff_root),
        "bundle_directory_count": len(rows),
        "scaffold_by_stem": dict(sorted(by_stem.items())),
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "items_jsonl": str(output_dir / ITEMS_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "reference_bundle_scaffold_not_for_listen_pack",
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
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        scaffold_status=status,
        bundle_directory_count=len(rows),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create directory scaffolds for full-pack reference bundles.")
    parser.add_argument("--handoff-json", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_reference_bundle_scaffold.py",
        entrypoint_path="/app/tools/build_full_pack_reference_bundle_scaffold.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_reference_bundle_scaffold(
        handoff_json=args.handoff_json,
        handoff_root=args.handoff_root,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "scaffold_status": manifest["scaffold_status"],
                "bundle_directory_count": manifest["bundle_directory_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
