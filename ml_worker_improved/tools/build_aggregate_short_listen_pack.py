#!/usr/bin/env python3
"""Build a reproducible aggregate listen pack with short Windows-safe names."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import (  # noqa: E402
    ARTIFACT_AUDIO_DIR,
    relative_artifact_path,
    stem_audio_filename,
    validate_artifact_segment,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.full_pack_stem_contract import (  # noqa: E402
    missing_required_stems,
    normalize_stem_list,
    present_stems_from_items,
    required_stem_missing_checks,
    required_stems_from_cli,
)


START_TIME = time.monotonic()
MANIFEST_NAME = "pack.json"
COVERAGE_NAME = "coverage.json"
SELECTED_NAME = "selected.jsonl"
PROGRESS_NAME = "progress.jsonl"
README_NAME = "README.md"

SOURCE_LEVEL_BLOCKERS = (
    "release_ready",
    "can_publish_as_result",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
    "separator_training_started",
    "cleanup_routes_added",
)
ROW_LEVEL_BLOCKERS = ("release_ready", "can_publish_as_result")


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


def audio_metadata(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(duration, 6),
        "frames": int(info.frames),
    }


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_packaging",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "aggregate_short_listen_pack_copies_existing_audio",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "aggregate_short_listen_pack_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def validate_alias(alias: str) -> str:
    normalized = str(alias or "").strip()
    if not normalized:
        raise ValueError("source alias cannot be empty")
    validate_artifact_segment(normalized)
    if not normalized.replace("_", "").replace("-", "").isalnum() or not normalized.isascii():
        raise ValueError(f"source alias must be short ascii alphanumeric/dash/underscore: {alias}")
    return normalized


def source_failed_checks(alias: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if payload.get("delivery_status") is None:
        failed.append(f"source_delivery_status_missing:{alias}")
    elif payload.get("delivery_status") != "accepted":
        failed.append(f"source_delivery_status_not_accepted:{alias}")
    if payload.get("delivery_allowed") is None:
        failed.append(f"source_delivery_allowed_missing:{alias}")
    elif payload.get("delivery_allowed") is not True:
        failed.append(f"source_delivery_allowed_not_true:{alias}")
    if payload.get("presentation_status") is None:
        failed.append(f"source_presentation_status_missing:{alias}")
    elif payload.get("presentation_status") != "accepted_for_listen_pack":
        failed.append(f"source_presentation_not_accepted:{alias}")
    for key in SOURCE_LEVEL_BLOCKERS:
        if payload.get(key) is True:
            failed.append(f"source_{key}_true:{alias}")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in SOURCE_LEVEL_BLOCKERS:
            if safety.get(key) is True:
                failed.append(f"source_safety_{key}_true:{alias}")
    return failed


def source_rows(alias: str, source_manifest: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("listen_audio_files", "stems", "items", "accepted_listen_preview_rows"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    if rows:
        return rows
    if payload.get("selected_audio_path"):
        return [
            {
                "schema": str(payload.get("manifest_type") or payload.get("schema") or ""),
                "listen_audio_path": str(payload.get("selected_audio_path") or ""),
                "stem_type": str(payload.get("stem_type") or payload.get("artifact_kind") or "audio"),
                "track_alias": alias,
                "quality_status": "listen_preview_only",
                "semantic_quality_verified": False,
                "release_ready": False,
                "can_publish_as_result": False,
                "judge_evidence": payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {},
                "source_manifest": str(source_manifest),
            }
        ]
    if payload.get("audio_path"):
        return [
            {
                "schema": str(payload.get("schema") or ""),
                "artifact_relative_path": str(payload.get("audio_path") or ""),
                "listen_audio_path": str(payload.get("audio_path") or ""),
                "stem_type": str(payload.get("artifact_kind") or "audio"),
                "track_alias": alias,
                "quality_status": "listen_check_preview_only",
                "semantic_quality_verified": False,
                "release_ready": False,
                "can_publish_as_result": False,
                "judge_evidence": payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {},
                "source_manifest": str(source_manifest),
            }
        ]
    return []


def resolve_source_audio(row: dict[str, Any], source_manifest: Path) -> Path:
    raw = str(row.get("listen_audio_path") or row.get("audio_path") or row.get("source_audio_path") or row.get("selected_audio_path") or "")
    if not raw:
        raise FileNotFoundError("source row has no audio path")
    source = Path(raw)
    if not source.is_absolute():
        source = source_manifest.parent / source
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(str(source))
    return source


def normalize_source_relative_path(row: dict[str, Any], source_audio: Path, source_manifest: Path) -> Path:
    raw = str(row.get("artifact_relative_path") or row.get("audio_path") or "")
    if not raw and row.get("selected_audio_path"):
        raw = stem_audio_filename(str(row.get("stem_type") or "unknown"))
    if not raw:
        raw = source_audio.name
    rel = Path(raw)
    if rel.is_absolute():
        rel = Path(source_audio.name)
    parts = list(rel.parts)
    if parts and parts[0] == ARTIFACT_AUDIO_DIR:
        parts = parts[1:]
    if not parts:
        parts = [source_audio.name]
    if parts[-1].lower() != source_audio.name.lower() and "." not in parts[-1]:
        parts.append(source_audio.name)
    return Path(*parts)


def destination_relative_path(alias: str, row: dict[str, Any], source_audio: Path, source_manifest: Path) -> Path:
    rel = normalize_source_relative_path(row, source_audio, source_manifest)
    parts = [ARTIFACT_AUDIO_DIR, alias, *rel.parts]
    for part in parts:
        validate_artifact_segment(str(part))
    return Path(*parts)


def row_failed_checks(alias: str, row: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in ROW_LEVEL_BLOCKERS:
        if row.get(key) is True:
            failed.append(f"row_{key}_true:{alias}")
    if row.get("delivery_status") is not None and row.get("delivery_status") != "accepted":
        failed.append(f"row_delivery_status_not_accepted:{alias}")
    if row.get("delivery_allowed") is not None and row.get("delivery_allowed") is not True:
        failed.append(f"row_delivery_allowed_not_true:{alias}")
    return failed


def source_arg(value: str) -> tuple[str, Path]:
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("--source-manifest must be ALIAS=PATH")
    alias, path = str(value).split("=", 1)
    return validate_alias(alias), Path(path)


def build_coverage(
    *,
    pack_id: str,
    output_dir: Path,
    selected: list[dict[str, Any]],
    failed_checks: list[str],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for item in selected:
        alias = str(item["source_alias"])
        group = groups.setdefault(
            alias,
            {
                "group_alias": alias,
                "status": "listen_preview_only",
                "semantic_quality_verified": False,
                "item_count": 0,
                "stem_types": [],
                "files": [],
            },
        )
        group["item_count"] += 1
        stem_type = str(item.get("stem_type") or "unknown")
        if stem_type not in group["stem_types"]:
            group["stem_types"].append(stem_type)
        group["files"].append(str(item["audio_path"]))
        if item.get("semantic_quality_verified") is True:
            group["semantic_quality_verified"] = True
    return {
        "schema": "aggregate_short_listen_pack_coverage.v1",
        "pack_id": pack_id,
        "output_dir": str(output_dir),
        "release_ready": False,
        "can_publish_as_result": False,
        "stem_groups": list(groups.values()),
        "failed_checks": failed_checks,
        "not_claimed": [
            "release_ready",
            "publishable_result",
            "semantic_clean_ground_truth_for_unverified_stems",
        ],
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
        },
    }


def build_readme(pack_id: str, selected: list[dict[str, Any]]) -> str:
    lines = [
        f"# {pack_id}",
        "",
        "Short Windows-safe aggregate listen pack.",
        "",
        "- `pack.json`: machine-readable manifest",
        "- `coverage.json`: stem coverage and safety status",
        "- `selected.jsonl`: copied audio rows",
        "- `progress.jsonl`: reproducible packaging progress",
        "- `release_ready=false`, `can_publish_as_result=false`",
        "",
        "## Audio",
        "",
    ]
    for item in selected:
        lines.append(f"- `{item['audio_path']}`")
    lines.append("")
    return "\n".join(lines)


def build_aggregate_short_listen_pack(
    *,
    sources: Sequence[tuple[str, Path]],
    output_dir: Path,
    pack_id: str,
    required_stems: Sequence[str] | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()

    normalized_sources = [(validate_alias(alias), Path(path)) for alias, path in sources]
    normalized_required_stems = required_stems_from_cli(requested=required_stems, require_full_pack=False)
    progress(
        progress_path,
        "start",
        pack_id=pack_id,
        required_stems=normalized_required_stems,
        sources=[{"alias": alias, "manifest": str(path)} for alias, path in normalized_sources],
    )
    selected: list[dict[str, Any]] = []
    failed_checks: list[str] = []
    used_destinations: set[str] = set()

    for alias, manifest_path in normalized_sources:
        if not manifest_path.exists():
            failed_checks.append(f"source_manifest_missing:{alias}")
            continue
        payload = read_json(manifest_path)
        source_required_stems = normalize_stem_list(payload.get("required_stems") if isinstance(payload.get("required_stems"), list) else [])
        normalized_required_stems = normalize_stem_list([*normalized_required_stems, *source_required_stems])
        failed_checks.extend(source_failed_checks(alias, payload))
        for index, row in enumerate(source_rows(alias, manifest_path, payload), start=1):
            failed_checks.extend(row_failed_checks(alias, row))
            try:
                source_audio = resolve_source_audio(row, manifest_path)
                dest_relative = destination_relative_path(alias, row, source_audio, manifest_path)
            except (FileNotFoundError, ValueError) as exc:
                failed_checks.append(f"source_row_invalid:{alias}:{index}:{exc}")
                continue
            dest_key = dest_relative.as_posix()
            if dest_key in used_destinations:
                failed_checks.append(f"duplicate_output_path:{dest_key}")
                continue
            used_destinations.add(dest_key)
            dest_path = output_dir / dest_relative
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_audio, dest_path)
            metadata = audio_metadata(dest_path)
            selected.append(
                {
                    "schema": "aggregate_short_listen_pack_audio.v1",
                    "source_alias": alias,
                    "source_manifest": str(manifest_path),
                    "source_audio_path": str(source_audio),
                    "audio_path": relative_artifact_path(dest_path, output_dir),
                    "stem_type": str(row.get("stem_type") or row.get("artifact_kind") or "unknown"),
                    "track_alias": str(row.get("track_alias") or alias),
                    "quality_status": str(row.get("quality_status") or "listen_preview_only"),
                    "semantic_quality_verified": row.get("semantic_quality_verified") is True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "sample_rate": metadata["sample_rate"],
                    "channels": metadata["channels"],
                    "duration_sec": metadata["duration_sec"],
                    "sha256": sha256_file(dest_path),
                    "source_row": row,
                }
            )

    present_stems = present_stems_from_items(selected, normalized_required_stems)
    missing_stems = missing_required_stems(normalized_required_stems, present_stems)
    failed_checks.extend(required_stem_missing_checks(missing_stems))
    failed_checks = sorted(set(failed_checks))
    accepted = bool(selected) and not failed_checks
    if not accepted:
        selected = []
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

    coverage = build_coverage(pack_id=pack_id, output_dir=output_dir, selected=selected, failed_checks=failed_checks)
    manifest = {
        "schema": "aggregate_short_listen_pack_manifest.v1",
        "pack_id": pack_id,
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": accepted,
        "deliverable": False,
        "item_count": len(selected) if accepted else 0,
        "items": selected if accepted else [],
        "required_stems": normalized_required_stems,
        "present_stems": present_stems,
        "missing_required_stems": missing_stems,
        "failed_checks": failed_checks,
        "source_manifests": [{"source_alias": alias, "manifest_path": str(path)} for alias, path in normalized_sources],
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "progress_jsonl": str(progress_path),
        "coverage_json": str(output_dir / COVERAGE_NAME),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "source_alias_strategy": "caller_supplied_validated_short_alias",
            "raw_track_or_stem_ids_in_path": False,
            "provenance_in_manifest": True,
            "metadata_filenames": [MANIFEST_NAME, COVERAGE_NAME, SELECTED_NAME, PROGRESS_NAME, README_NAME],
        },
        "contract": {
            "audio_copied_without_modification": True,
            "no_inference_run": True,
            "training_started": False,
            "fine_tune_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready_forced_false": True,
            "short_windows_safe_artifact_names": True,
            "raw_ids_kept_in_manifest_only": True,
            "required_stem_contract_active": bool(normalized_required_stems),
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / SELECTED_NAME, selected if accepted else [])
    write_json(output_dir / COVERAGE_NAME, coverage)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / README_NAME).write_text(build_readme(pack_id, selected if accepted else []), encoding="utf-8")
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], item_count=manifest["item_count"], failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aggregate short-name listen pack from accepted source manifests.")
    parser.add_argument("--source-manifest", type=source_arg, action="append", required=True, metavar="ALIAS=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--required-stem", action="append", default=[])
    parser.add_argument("--require-full-pack-stems", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_aggregate_short_listen_pack.py",
        entrypoint_path="/app/tools/build_aggregate_short_listen_pack.py",
    )
    args = parse_args(argv)
    manifest = build_aggregate_short_listen_pack(
        sources=args.source_manifest,
        output_dir=args.output_dir,
        pack_id=str(args.pack_id),
        required_stems=required_stems_from_cli(
            requested=tuple(str(item) for item in args.required_stem),
            require_full_pack=bool(args.require_full_pack_stems),
        ),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "item_count": manifest["item_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
