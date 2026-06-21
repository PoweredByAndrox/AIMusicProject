#!/usr/bin/env python3
"""Build short diagnostic middle-window minus audition clips."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import soundfile as sf

from tools.artifact_name_policy import ARTIFACT_AUDIO_DIR, relative_artifact_path, validate_artifact_segment
from tools.build_aggregate_short_listen_pack import audio_metadata, read_json, sha256_file, validate_alias, write_json, write_jsonl
from tools.docker_runtime import require_docker_runtime

MANIFEST_NAME = "pack.json"
ITEMS_NAME = "items.jsonl"
PLAYLIST_NAME = "playlist.m3u"
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


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_diagnostic_audio_slicing",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "minus_mid_pack_slices_existing_audio_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "minus_mid_pack_no_training",
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
    contract = payload.get("contract")
    if isinstance(contract, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if contract.get(key) is True:
                failed.append(f"{prefix}_contract_{key}_true")
    return failed


def parse_window_arg(value: str) -> dict[str, Any]:
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("--window must be ID=START:DURATION:REASON")
    window_id, rest = str(value).split("=", 1)
    parts = rest.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--window must be ID=START:DURATION:REASON")
    if not window_id.strip():
        raise argparse.ArgumentTypeError("--window id must be non-empty")
    validate_artifact_segment(window_id)
    try:
        start_sec = float(parts[0])
        duration_sec = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--window start and duration must be numeric") from exc
    if start_sec < 0.0:
        raise argparse.ArgumentTypeError("--window start must be >= 0")
    if duration_sec <= 0.0:
        raise argparse.ArgumentTypeError("--window duration must be > 0")
    return {
        "window_id": window_id,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "reason": parts[2],
    }


def parse_extra_source_arg(value: str) -> tuple[str, Path]:
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("--extra-source must be ALIAS=PATH")
    alias, raw_path = str(value).split("=", 1)
    alias = validate_alias(alias)
    return alias, Path(raw_path)


def parse_extra_source_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("--extra-source-manifest must be ALIAS=PATH")
    alias, raw_path = str(value).split("=", 1)
    alias = validate_alias(alias)
    return alias, Path(raw_path)


def source_audio_for_item(pack_root: Path, item: dict[str, Any]) -> Path:
    raw = str(item.get("audio_path") or item.get("listen_audio_path") or item.get("source_audio_path") or "")
    if not raw:
        raise FileNotFoundError("item has no audio path")
    path = Path(raw)
    if not path.is_absolute():
        path = pack_root / path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def source_alias_from_audio_path(audio_path: str) -> str:
    alias = Path(audio_path).stem
    return validate_alias(alias)


def collect_pack_sources(
    *,
    pack: dict[str, Any],
    pack_root: Path,
    source_alias: str,
    stem_type: str,
    failed: list[str],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    items = pack.get("items")
    if not isinstance(items, list):
        failed.append("source_pack_items_missing")
        return sources

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            failed.append(f"source_pack_item_invalid:{index}")
            continue
        if str(item.get("source_alias") or "") != source_alias:
            continue
        if str(item.get("stem_type") or "") != stem_type:
            continue
        audio_rel = str(item.get("audio_path") or "")
        try:
            alias = source_alias_from_audio_path(audio_rel)
        except ValueError as exc:
            failed.append(f"source_item_alias_invalid:{index}:{exc}")
            continue
        for check in collect_true_flag_failures(f"source_item", item):
            failed.append(f"{check}:{alias}")
        try:
            source_audio = source_audio_for_item(pack_root, item)
        except FileNotFoundError as exc:
            failed.append(f"source_item_audio_missing:{alias}:{exc}")
            continue
        sources.append(
            {
                "alias": alias,
                "source_kind": "candidate_pack_item",
                "source_audio_path": str(source_audio),
                "source_manifest": str(item.get("source_manifest") or ""),
                "source_pack_audio_path": audio_rel,
                "source_row": item.get("source_row") if isinstance(item.get("source_row"), dict) else item,
            }
        )
    if not sources:
        failed.append(f"source_items_empty:{source_alias}:{stem_type}")
    return sources


def collect_extra_sources(
    extra_sources: Sequence[tuple[str, Path]],
    extra_source_manifests: Sequence[tuple[str, Path]],
    failed: list[str],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    manifest_by_alias = {validate_alias(alias): Path(path) for alias, path in extra_source_manifests}
    for alias, raw_path in extra_sources:
        alias = validate_alias(alias)
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            failed.append(f"extra_source_missing:{alias}:{path}")
            continue
        manifest_path = manifest_by_alias.get(alias)
        if manifest_path is not None and (not manifest_path.exists() or not manifest_path.is_file()):
            failed.append(f"extra_source_manifest_missing:{alias}:{manifest_path}")
            continue
        sources.append(
            {
                "alias": alias,
                "source_kind": "extra_diagnostic_source",
                "source_audio_path": str(path),
                "source_manifest": str(manifest_path or ""),
                "source_pack_audio_path": "",
                "source_row": {
                    "extra_source_alias": alias,
                    "source_audio_path": str(path),
                    "source_manifest": str(manifest_path or ""),
                },
            }
        )
    return sources


def render_clip(*, source_audio: Path, dest: Path, start_sec: float, duration_sec: float) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(source_audio)) as handle:
        sample_rate = int(handle.samplerate)
        start_frame = int(round(start_sec * sample_rate))
        frame_count = int(round(duration_sec * sample_rate))
        if start_frame >= len(handle):
            raise ValueError(f"window starts after audio end: {source_audio}")
        handle.seek(start_frame)
        data = handle.read(frame_count, dtype="float32", always_2d=True)
    if len(data) == 0:
        raise ValueError(f"window produced no audio: {source_audio}")
    sf.write(str(dest), data, sample_rate, format="FLAC", subtype="PCM_24")
    metadata = audio_metadata(dest)
    metadata["actual_duration_sec"] = metadata["duration_sec"]
    return metadata


def build_summary(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# minus mid pack",
        "",
        "- diagnostic_only=true",
        "- delivery_allowed=false",
        "- release_ready=false",
        "- can_publish_as_result=false",
        f"- pack_status={manifest['pack_status']}",
        f"- item_count={len(rows)}",
        "",
        "## Clips",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['audio_path']}` {row['source_alias']} {row['window_id']} {row['window_start_sec']:.3f}s")
    lines.append("")
    return "\n".join(lines)


def blocked_manifest(
    *,
    output_dir: Path,
    candidate_pack_json: Path,
    failed_checks: list[str],
    command_args: list[str] | None,
) -> dict[str, Any]:
    manifest = {
        "schema": "minus_mid_audition_pack.v1",
        "source_pack_json": str(candidate_pack_json),
        "pack_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_audition_only_blocked",
        "release_ready": False,
        "can_publish_as_result": False,
        "diagnostic_audio_exported": False,
        "item_count": 0,
        "source_count": 0,
        "window_count": 0,
        "items": [],
        "failed_checks": stable_unique(failed_checks),
        "items_jsonl": str(output_dir / ITEMS_NAME),
        "playlist_m3u": str(output_dir / PLAYLIST_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(output_dir / PROGRESS_NAME),
        "safety": safety_flags(),
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, [])
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / PLAYLIST_NAME).write_text("#EXTM3U\n", encoding="utf-8")
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest, []), encoding="utf-8")
    return manifest


def safety_flags() -> dict[str, bool]:
    return {
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "release_ready": False,
    }


def build_minus_mid_pack(
    *,
    candidate_pack_json: Path,
    output_dir: Path,
    windows: Sequence[dict[str, Any]],
    source_alias: str = "mpr",
    stem_type: str = "minus",
    extra_sources: Sequence[tuple[str, Path]] = (),
    extra_source_manifests: Sequence[tuple[str, Path]] = (),
    handoff_playlist_m3u: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", candidate_pack_json=str(candidate_pack_json), window_count=len(windows))

    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)

    failed: list[str] = []
    source_alias = validate_alias(source_alias)
    stem_type = str(stem_type or "").strip()
    if stem_type != "minus":
        failed.append(f"unsupported_stem_type:{stem_type}")
    if not windows:
        failed.append("windows_empty")
    for window in windows:
        window_id = str(window.get("window_id") or "").strip()
        try:
            if not window_id:
                failed.append("window_id_missing")
            validate_artifact_segment(window_id)
            if float(window.get("start_sec") or 0.0) < 0.0:
                failed.append(f"window_start_negative:{window.get('window_id')}")
            if float(window.get("duration_sec") or 0.0) <= 0.0:
                failed.append(f"window_duration_nonpositive:{window.get('window_id')}")
        except (TypeError, ValueError) as exc:
            failed.append(f"window_invalid:{window}:{exc}")

    pack: dict[str, Any] = {}
    if not candidate_pack_json.exists():
        failed.append(f"source_pack_missing:{candidate_pack_json}")
    else:
        pack = read_json(candidate_pack_json)
        if pack.get("schema") != "aggregate_short_listen_pack_manifest.v1":
            failed.append("source_pack_schema_unexpected")
        if pack.get("delivery_status") != "accepted":
            failed.append("source_pack_delivery_status_not_accepted")
        if pack.get("delivery_allowed") is not True:
            failed.append("source_pack_delivery_allowed_not_true")
        if pack.get("presentation_status") != "accepted_for_listen_pack":
            failed.append("source_pack_presentation_not_accepted")
        if pack.get("failed_checks") not in ([], None):
            failed.append("source_pack_failed_checks_not_empty")
        failed.extend(collect_true_flag_failures("source_pack", pack))

    sources: list[dict[str, Any]] = []
    if pack:
        sources.extend(
            collect_pack_sources(
                pack=pack,
                pack_root=candidate_pack_json.parent,
                source_alias=source_alias,
                stem_type=stem_type,
                failed=failed,
            )
        )
    sources.extend(collect_extra_sources(extra_sources, extra_source_manifests, failed))

    aliases = [str(source["alias"]) for source in sources]
    duplicate_aliases = sorted({alias for alias in aliases if aliases.count(alias) > 1})
    for alias in duplicate_aliases:
        failed.append(f"duplicate_source_alias:{alias}")

    if failed:
        manifest = blocked_manifest(
            output_dir=output_dir,
            candidate_pack_json=candidate_pack_json,
            failed_checks=failed,
            command_args=command_args,
        )
        progress(progress_path, "finish", pack_status=manifest["pack_status"], item_count=0, failed_checks=manifest["failed_checks"])
        return manifest

    rows: list[dict[str, Any]] = []
    render_failed: list[str] = []
    for source in sources:
        source_path = Path(str(source["source_audio_path"]))
        source_hash = sha256_file(source_path)
        for window in windows:
            window_id = str(window["window_id"])
            dest = output_dir / ARTIFACT_AUDIO_DIR / str(source["alias"]) / f"{window_id}.flac"
            try:
                validate_artifact_segment(str(source["alias"]))
                validate_artifact_segment(dest.name)
                metadata = render_clip(
                    source_audio=source_path,
                    dest=dest,
                    start_sec=float(window["start_sec"]),
                    duration_sec=float(window["duration_sec"]),
                )
            except (OSError, ValueError) as exc:
                render_failed.append(f"render_failed:{source['alias']}:{window_id}:{exc}")
                continue
            rows.append(
                {
                    "schema": "minus_mid_audition_item.v1",
                    "source_alias": str(source["alias"]),
                    "source_kind": str(source["source_kind"]),
                    "source_audio_path": str(source_path),
                    "source_audio_sha256": source_hash,
                    "source_manifest": str(source.get("source_manifest") or ""),
                    "source_pack_audio_path": str(source.get("source_pack_audio_path") or ""),
                    "source_row": source["source_row"],
                    "stem_type": stem_type,
                    "window_id": window_id,
                    "window_start_sec": float(window["start_sec"]),
                    "window_duration_sec": float(window["duration_sec"]),
                    "window_reason": str(window.get("reason") or ""),
                    "audio_path": relative_artifact_path(dest, output_dir),
                    "sample_rate": metadata["sample_rate"],
                    "channels": metadata["channels"],
                    "duration_sec": metadata["duration_sec"],
                    "sha256": sha256_file(dest),
                    "delivery_allowed": False,
                    "presentation_status": "diagnostic_audition_only",
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            )
            progress(progress_path, "clip", source_alias=source["alias"], window_id=window_id, audio_path=relative_artifact_path(dest, output_dir))

    if render_failed:
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
        manifest = blocked_manifest(
            output_dir=output_dir,
            candidate_pack_json=candidate_pack_json,
            failed_checks=render_failed,
            command_args=command_args,
        )
        progress(progress_path, "finish", pack_status=manifest["pack_status"], item_count=0, failed_checks=manifest["failed_checks"])
        return manifest

    manifest = {
        "schema": "minus_mid_audition_pack.v1",
        "source_pack_json": str(candidate_pack_json),
        "source_pack_id": str(pack.get("pack_id") or ""),
        "pack_status": "ready_for_diagnostic_listen",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_audition_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "diagnostic_audio_exported": True,
        "item_count": len(rows),
        "source_count": len(sources),
        "window_count": len(windows),
        "items": rows,
        "failed_checks": [],
        "items_jsonl": str(output_dir / ITEMS_NAME),
        "playlist_m3u": str(output_dir / PLAYLIST_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "source_alias_strategy": "candidate_file_stem_or_explicit_short_alias",
            "raw_track_or_stem_ids_in_path": False,
            "provenance_in_manifest": True,
            "metadata_filenames": [MANIFEST_NAME, ITEMS_NAME, PLAYLIST_NAME, SUMMARY_NAME, PROGRESS_NAME],
        },
        "safety": safety_flags(),
        "not_claimed": ["release_ready", "publishable_result", "semantic_clean_minus", "human_listen_verdict"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ITEMS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / PLAYLIST_NAME).write_text("\n".join(["#EXTM3U", *[row["audio_path"] for row in rows]]) + "\n", encoding="utf-8")
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest, rows), encoding="utf-8")
    if handoff_playlist_m3u is not None:
        handoff_playlist_m3u.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#EXTM3U"]
        for row in rows:
            target = output_dir / str(row["audio_path"])
            lines.append(Path(os.path.relpath(target, handoff_playlist_m3u.parent)).as_posix())
        handoff_playlist_m3u.write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(progress_path, "finish", pack_status=manifest["pack_status"], item_count=len(rows), failed_checks=[])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build short diagnostic minus middle-window audition clips.")
    parser.add_argument("--candidate-pack-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", dest="windows", type=parse_window_arg, action="append", required=True)
    parser.add_argument("--source-alias", default="mpr")
    parser.add_argument("--stem-type", default="minus")
    parser.add_argument("--extra-source", dest="extra_sources", type=parse_extra_source_arg, action="append", default=[])
    parser.add_argument(
        "--extra-source-manifest",
        dest="extra_source_manifests",
        type=parse_extra_source_manifest_arg,
        action="append",
        default=[],
    )
    parser.add_argument("--handoff-playlist-m3u", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_minus_mid_pack.py",
        entrypoint_path="/app/tools/build_minus_mid_pack.py",
    )
    args = parse_args(argv)
    manifest = build_minus_mid_pack(
        candidate_pack_json=args.candidate_pack_json,
        output_dir=args.output_dir,
        windows=args.windows,
        source_alias=args.source_alias,
        stem_type=args.stem_type,
        extra_sources=args.extra_sources,
        extra_source_manifests=args.extra_source_manifests,
        handoff_playlist_m3u=args.handoff_playlist_m3u,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "pack_status": manifest["pack_status"],
                "item_count": manifest["item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["pack_status"] == "ready_for_diagnostic_listen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
