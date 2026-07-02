#!/usr/bin/env python3
"""Build a listen pack from accepted Music Intelligence stem manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from artifact_name_policy import ARTIFACT_AUDIO_DIR, relative_artifact_path, stem_audio_filename, track_alias  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "pack.json"
SELECTED_NAME = "selected.jsonl"
REVIEW_NAME = "review.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
REVIEW_QUEUE_DELIVERY_STATUS = "semantic_review_pending"
REVIEW_QUEUE_PRESENTATION_STATUS = "semantic_review_queue_not_for_listen_pack"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_packaging",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "stem_listen_pack_copies_accepted_existing_audio",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "stem_listen_pack_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _manifest_stems(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stems = payload.get("stems")
    if isinstance(stems, list):
        return [stem for stem in stems if isinstance(stem, dict)]
    items = payload.get("items")
    if isinstance(items, list):
        return [stem for stem in items if isinstance(stem, dict)]
    return []


def _accepted_stem(row: dict[str, Any]) -> bool:
    return (
        row.get("analysis_allowed") is True
        and str(row.get("quality_status") or "") == "accepted"
        and row.get("release_ready") is not True
        and bool(str(row.get("audio_path") or ""))
    )


def _accepted_pack_row(row: dict[str, Any]) -> bool:
    return (
        row.get("release_ready") is not True
        and row.get("can_publish_as_result") is not True
        and row.get("delivery_status") in (None, "accepted")
        and row.get("delivery_allowed") in (None, True)
        and bool(str(row.get("listen_audio_path") or row.get("audio_path") or row.get("source_audio_path") or ""))
    )


def _aggregate_pack_rows(payload: dict[str, Any], pack_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    if payload.get("schema") != "aggregate_short_listen_pack_manifest.v1":
        failed.append("input_pack_schema_unexpected")
    if payload.get("delivery_status") != "accepted" or payload.get("delivery_allowed") is not True:
        failed.append("input_pack_not_accepted")
    if payload.get("release_ready") is True:
        failed.append("input_pack_release_ready_true")
    if payload.get("can_publish_as_result") is True:
        failed.append("input_pack_can_publish_as_result_true")
    rows: list[dict[str, Any]] = []
    for row in payload.get("items") or []:
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        normalized.setdefault("track_id", str(row.get("source_alias") or row.get("track_alias") or pack_path.parent.name))
        normalized.setdefault("stem_id", str(row.get("source_stem_id") or row.get("audio_path") or ""))
        normalized.setdefault("quality_status", str(row.get("quality_status") or "listen_preview_only"))
        normalized["_source_root"] = str(pack_path.parent)
        normalized["_source_input_type"] = "aggregate_short_listen_pack"
        rows.append(normalized)
    return rows, failed


def _resolve_audio_path(row: dict[str, Any], source_root: Path | None = None) -> Path:
    raw = str(row.get("listen_audio_path") or row.get("audio_path") or row.get("source_audio_path") or "")
    source = Path(raw)
    if not source.is_absolute() and source_root is not None:
        source = source_root / source
    return source


def _copy_row(
    row: dict[str, Any],
    *,
    output_dir: Path,
    source_manifest: Path,
    track_alias_value: str,
    duplicate_index: int,
    source_root: Path | None = None,
) -> dict[str, Any]:
    source = _resolve_audio_path(row, source_root=source_root)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"stem audio missing: {source}")
    track_id = str(row.get("track_id") or "unknown_track")
    stem_type = str(row.get("stem_type") or "unknown")
    artifact_filename = stem_audio_filename(stem_type, duplicate_index=duplicate_index)
    listen_path = output_dir / ARTIFACT_AUDIO_DIR / track_alias_value / artifact_filename
    listen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, listen_path)
    copied = {
        "schema": "music_intelligence_stem_listen_audio.v1",
        "track_id": track_id,
        "source_track_id": track_id,
        "source_alias": str(row.get("source_alias") or row.get("track_alias") or ""),
        "track_alias": track_alias_value,
        "stem_id": str(row.get("stem_id") or ""),
        "source_stem_id": str(row.get("stem_id") or ""),
        "stem_alias": artifact_filename.removesuffix(".flac"),
        "artifact_filename": artifact_filename,
        "artifact_relative_path": relative_artifact_path(listen_path, output_dir),
        "stem_type": stem_type,
        "instrument_hint": str(row.get("instrument_hint") or ""),
        "duration_sec": row.get("duration_sec"),
        "quality_status": str(row.get("quality_status") or ""),
        "analysis_allowed": True,
        "source_input_type": str(row.get("_source_input_type") or "music_intelligence_input_manifest"),
        "source_manifest": str(source_manifest),
        "upstream_source_manifest": str(row.get("source_manifest") or ""),
        "source_audio_path": str(source),
        "listen_audio_path": str(listen_path),
        "listen_audio_sha256": sha256_file(listen_path),
        "semantic_review_required": True,
        "semantic_quality_verified": False,
        "review_status": "semantic_review_pending",
        "delivery_status": REVIEW_QUEUE_DELIVERY_STATUS,
        "delivery_allowed": False,
        "presentation_status": REVIEW_QUEUE_PRESENTATION_STATUS,
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
    }
    source_row = row.get("source_row")
    if isinstance(source_row, dict):
        copied["source_row"] = source_row
    return copied


def _review_rows(selected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        rows.append(
            {
                "schema": "semantic_stem_quality_review_row.v1",
                "group_alias": str(item.get("track_alias") or ""),
                "artifact_relative_path": str(item.get("artifact_relative_path") or ""),
                "audio_path": str(item.get("artifact_relative_path") or ""),
                "stem_type": str(item.get("stem_type") or "unknown"),
                "semantic_review_required": True,
                "semantic_quality_verified": False,
                "semantic_quality_status": "pending",
                "review_status": "pending",
                "blockers": [],
                "evidence_paths": [],
                "reviewed_at_utc": "",
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    return rows


def _review_manifest(
    *,
    output_dir: Path,
    selected: Sequence[dict[str, Any]],
    failed_checks: Sequence[str],
    music_intelligence_manifests: Sequence[Path],
    aggregate_pack_manifests: Sequence[Path],
) -> dict[str, Any]:
    rows = _review_rows(selected if not failed_checks else [])
    verified = sum(1 for row in rows if row["semantic_quality_verified"] is True)
    pending = sum(1 for row in rows if row["semantic_quality_status"] == "pending")
    return {
        "schema": "semantic_stem_quality_review.v1",
        "review_status": "pending" if rows and not failed_checks else "blocked",
        "reviewer_kind": "human_or_machine_pending",
        "review_tool": "build_music_intelligence_stem_listen_pack.py",
        "reviewed_at_utc": "",
        "music_intelligence_manifests": [str(path) for path in music_intelligence_manifests],
        "input_pack_jsons": [str(path) for path in aggregate_pack_manifests],
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "semantic_quality_verified_item_count": verified,
        "pending_item_count": pending,
        "total_item_count": len(rows),
        "reviewed_group_aliases": sorted({str(row["group_alias"]) for row in rows}),
        "failed_checks": list(failed_checks),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "semantic_review_artifact_not_listen_result",
        "not_claimed": ["release_ready", "publishable_result", "semantic_clean_ground_truth"],
    }


def _summary_text(review: dict[str, Any]) -> str:
    lines = [
        "# semantic stem quality review",
        "",
        f"- review_status: `{review['review_status']}`",
        f"- total_item_count: `{review['total_item_count']}`",
        f"- pending_item_count: `{review['pending_item_count']}`",
        f"- semantic_quality_verified_item_count: `{review['semantic_quality_verified_item_count']}`",
        f"- release_ready: `{str(review['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(review['can_publish_as_result']).lower()}`",
        "",
        "## Failed Checks",
        "",
    ]
    if review["failed_checks"]:
        lines.extend(f"- `{item}`" for item in review["failed_checks"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_music_intelligence_stem_listen_pack(
    *,
    music_intelligence_manifests: Sequence[Path],
    aggregate_pack_manifests: Sequence[Path] = (),
    source_aliases: set[str] | None = None,
    output_dir: Path,
    required_stem_types: set[str] | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    listen_dir = output_dir / ARTIFACT_AUDIO_DIR
    if listen_dir.exists():
        shutil.rmtree(listen_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    required = {str(item) for item in (required_stem_types or set()) if str(item)}
    allowed_source_aliases = {str(item) for item in (source_aliases or set()) if str(item)}
    progress(
        progress_path,
        "start",
        manifests=[str(path) for path in music_intelligence_manifests],
        input_pack_jsons=[str(path) for path in aggregate_pack_manifests],
        source_aliases=sorted(allowed_source_aliases),
        required_stem_types=sorted(required),
    )
    selected: list[dict[str, Any]] = []
    failed_checks: list[str] = []
    seen_by_track: dict[str, set[str]] = {}
    aliases_by_track: dict[str, str] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}
    for manifest_path in music_intelligence_manifests:
        payload = read_json(manifest_path)
        if payload.get("release_ready") is True:
            failed_checks.append("source_manifest_release_ready_true")
            continue
        for row in _manifest_stems(payload):
            track_id = str(row.get("track_id") or "")
            stem_type = str(row.get("stem_type") or "")
            if required and stem_type not in required:
                continue
            if not _accepted_stem(row):
                continue
            try:
                track_alias_value = aliases_by_track.setdefault(track_id, track_alias(len(aliases_by_track) + 1))
                duplicate_key = (track_alias_value, stem_type)
                duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1
                copied = _copy_row(
                    row,
                    output_dir=output_dir,
                    source_manifest=manifest_path,
                    track_alias_value=track_alias_value,
                    duplicate_index=duplicate_counts[duplicate_key],
                )
            except FileNotFoundError:
                failed_checks.append(f"stem_audio_missing:{stem_type}")
                continue
            selected.append(copied)
            seen_by_track.setdefault(track_id, set()).add(stem_type)
    for pack_path in aggregate_pack_manifests:
        payload = read_json(pack_path)
        pack_rows, pack_failed = _aggregate_pack_rows(payload, pack_path)
        failed_checks.extend(pack_failed)
        if pack_failed:
            continue
        for row in pack_rows:
            if allowed_source_aliases and str(row.get("source_alias") or "") not in allowed_source_aliases:
                continue
            track_id = str(row.get("track_id") or "")
            stem_type = str(row.get("stem_type") or "")
            if required and stem_type not in required:
                continue
            if not _accepted_pack_row(row):
                continue
            try:
                track_alias_value = aliases_by_track.setdefault(track_id, track_alias(len(aliases_by_track) + 1))
                duplicate_key = (track_alias_value, stem_type)
                duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1
                copied = _copy_row(
                    row,
                    output_dir=output_dir,
                    source_manifest=pack_path,
                    track_alias_value=track_alias_value,
                    duplicate_index=duplicate_counts[duplicate_key],
                    source_root=Path(str(row.get("_source_root") or pack_path.parent)),
                )
            except FileNotFoundError:
                failed_checks.append(f"stem_audio_missing:{stem_type}")
                continue
            selected.append(copied)
            seen_by_track.setdefault(track_id, set()).add(stem_type)
    for track_id, stem_types in seen_by_track.items():
        for stem_type in required:
            if stem_type not in stem_types:
                failed_checks.append(f"track_required_stem_missing:{track_id}:{stem_type}")
    if required:
        all_types = {item["stem_type"] for item in selected}
        for stem_type in required:
            if stem_type not in all_types:
                failed_checks.append(f"required_stem_type_missing:{stem_type}")
    if not selected:
        failed_checks.append("listen_audio_missing")
    failed_checks = sorted(set(failed_checks))
    review_queue_ready = bool(selected) and not failed_checks
    stem_counts: dict[str, int] = {}
    for item in selected:
        stem_counts[item["stem_type"]] = stem_counts.get(item["stem_type"], 0) + 1
    manifest = {
        "schema": "music_intelligence_stem_listen_pack_manifest.v1",
        "delivery_status": REVIEW_QUEUE_DELIVERY_STATUS if review_queue_ready else "blocked",
        "delivery_allowed": False,
        "presentation_status": REVIEW_QUEUE_PRESENTATION_STATUS if review_queue_ready else "blocked_hard_case",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": False,
        "deliverable": False,
        "review_queue_ready": review_queue_ready,
        "failed_checks": failed_checks,
        "music_intelligence_manifests": [str(path) for path in music_intelligence_manifests],
        "input_pack_jsons": [str(path) for path in aggregate_pack_manifests],
        "source_aliases": sorted(allowed_source_aliases),
        "output_dir": str(output_dir),
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "review_json": str(output_dir / REVIEW_NAME),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "required_stem_types": sorted(required),
        "track_count": len(seen_by_track),
        "track_aliases": [{"track_alias": alias, "source_track_id": track} for track, alias in sorted(aliases_by_track.items(), key=lambda item: item[1])],
        "stem_counts": stem_counts,
        "listen_audio_count": len(selected) if review_queue_ready else 0,
        "listen_audio_files": selected if review_queue_ready else [],
        "semantic_review_required": True,
        "semantic_quality_verified": False,
        "review_status": "semantic_review_pending",
        "not_claimed": [
            "release_ready",
            "publishable_result",
            "semantic_clean_ground_truth",
            "human_listen_verdict",
        ],
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "track_dir_strategy": "tNNN",
            "filename_strategy": "stem_type_alias",
            "raw_track_or_stem_ids_in_path": False,
            "provenance_in_manifest": True,
        },
        "thresholds_relaxed": False,
        "runtime_gates_changed": False,
        "fallback_used": False,
        "model_registry_edited": False,
        "contract": {
            "only_accepted_analysis_allowed_stems_packaged": True,
            "aggregate_pack_rows_are_listen_review_inputs_only": True,
            "audio_copied_without_modification": True,
            "semantic_review_queue_not_listen_result": True,
            "semantic_review_required": True,
            "semantic_quality_verified": False,
            "release_ready_forced_false": True,
            "short_windows_safe_artifact_names": True,
            "raw_ids_kept_in_manifest_only": True,
        },
        "command_args": command_args or [],
    }
    review = _review_manifest(
        output_dir=output_dir,
        selected=selected,
        failed_checks=failed_checks,
        music_intelligence_manifests=music_intelligence_manifests,
        aggregate_pack_manifests=aggregate_pack_manifests,
    )
    write_jsonl(output_dir / SELECTED_NAME, selected if review_queue_ready else [])
    write_jsonl(output_dir / ROWS_NAME, _review_rows(selected if review_queue_ready else []))
    write_json(output_dir / REVIEW_NAME, review)
    (output_dir / SUMMARY_NAME).write_text(_summary_text(review), encoding="utf-8")
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], listen_audio_count=manifest["listen_audio_count"], failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a listen pack from accepted Music Intelligence stems.")
    parser.add_argument("--music-intelligence-manifest", type=Path, action="append", default=[])
    parser.add_argument("--input-pack-json", type=Path, action="append", default=[])
    parser.add_argument("--source-alias", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-stem-type", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_music_intelligence_stem_listen_pack.py", entrypoint_path="/app/tools/build_music_intelligence_stem_listen_pack.py")
    args = parse_args(argv)
    manifest = build_music_intelligence_stem_listen_pack(
        music_intelligence_manifests=args.music_intelligence_manifest,
        aggregate_pack_manifests=args.input_pack_json,
        source_aliases={str(item) for item in args.source_alias},
        output_dir=args.output_dir,
        required_stem_types={str(item) for item in args.required_stem_type},
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(json.dumps({"manifest_path": str(args.output_dir / MANIFEST_NAME), "delivery_status": manifest["delivery_status"], "listen_audio_count": manifest["listen_audio_count"]}, sort_keys=True))
    return 0 if manifest.get("review_queue_ready") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
