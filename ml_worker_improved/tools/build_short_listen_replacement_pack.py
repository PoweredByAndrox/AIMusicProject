#!/usr/bin/env python3
"""Build an aggregate short listen pack with accepted replacement items."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import ARTIFACT_AUDIO_DIR, validate_artifact_segment  # noqa: E402
from tools.build_aggregate_short_listen_pack import (  # noqa: E402
    COVERAGE_NAME,
    MANIFEST_NAME,
    PROGRESS_NAME,
    README_NAME,
    SELECTED_NAME,
    audio_metadata,
    build_coverage,
    build_readme,
    sha256_file,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
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
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_packaging",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_replacement_pack_copies_existing_audio",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_replacement_pack_no_training",
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


def resolve_audio(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        return path
    if path.is_absolute():
        return path
    return base / path


def existing_audio(raw: Any, *, base: Path) -> Path:
    path = resolve_audio(raw, base=base)
    if not str(path) or not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def path_key(raw: Any) -> str:
    path = Path(str(raw or ""))
    if not str(path):
        return ""
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def source_dicts(row: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(current: dict[str, Any]) -> None:
        marker = id(current)
        if marker in seen:
            return
        seen.add(marker)
        values.append(current)
        for key in ("source_row", "source_phase_repair_manifest"):
            nested = current.get(key)
            if isinstance(nested, dict):
                visit(nested)

    visit(row)
    return values


def replacement_source_minus_path(row: dict[str, Any]) -> str:
    sources = source_dicts(row)
    for source in sources:
        for key in ("source_minus_path", "base_minus_path"):
            value = str(source.get(key) or "")
            if value:
                return value
    for source in sources[1:]:
        value = str(source.get("source_audio_path") or "")
        if value:
            return value
    return ""


def track_minus_key(row: dict[str, Any]) -> str:
    if str(row.get("stem_type") or "").lower() != "minus":
        return ""
    for source in source_dicts(row):
        for key in ("source_track_id", "track_id"):
            value = str(source.get(key) or "")
            if value:
                return f"track:{value}:minus"
    return ""


def replacement_match_key(row: dict[str, Any]) -> str:
    source_minus = replacement_source_minus_path(row)
    if source_minus:
        return path_key(source_minus)
    return track_minus_key(row)


def base_match_keys(item: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for source in source_dicts(item):
        for field in ("source_minus_path", "base_minus_path", "source_audio_path"):
            key = path_key(source.get(field))
            if key:
                keys.append(key)
    keys.append(track_minus_key(item))
    return stable_unique(key for key in keys if key)


def replacement_audio_path(row: dict[str, Any], manifest_path: Path) -> Path:
    return existing_audio(
        row.get("listen_audio_path") or row.get("source_audio_path") or row.get("artifact_relative_path"),
        base=manifest_path.parent,
    )


def replacement_rows(manifest_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("listen_audio_files", "items", "stems"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def replacement_failed_checks(alias: str, payload: dict[str, Any]) -> list[str]:
    failed = collect_true_flag_failures(f"replacement:{alias}", payload)
    if payload.get("delivery_status") != "accepted":
        failed.append(f"replacement_delivery_status_not_accepted:{alias}")
    if payload.get("delivery_allowed") is not True:
        failed.append(f"replacement_delivery_allowed_not_true:{alias}")
    if payload.get("presentation_status") != "accepted_for_listen_pack":
        failed.append(f"replacement_presentation_not_accepted:{alias}")
    failed_checks = payload.get("failed_checks")
    if isinstance(failed_checks, list) and failed_checks:
        failed.append(f"replacement_failed_checks_not_empty:{alias}")
    return failed


def check_short_relative_path(relative_path: str) -> list[str]:
    path = Path(str(relative_path or ""))
    if path.is_absolute() or not str(path):
        return ["invalid_relative_audio_path"]
    failed: list[str] = []
    for segment in path.parts:
        try:
            validate_artifact_segment(segment)
        except ValueError:
            failed.append("long_path_segment")
            break
    return failed


def base_audio_path(item: dict[str, Any], base_pack_json: Path) -> Path:
    try:
        return existing_audio(item.get("audio_path"), base=base_pack_json.parent)
    except FileNotFoundError:
        return existing_audio(item.get("source_audio_path"), base=base_pack_json.parent)


def build_replacement_map(
    replacement_pack_jsons: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    mapping: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    failed: list[str] = []
    for index, manifest_path in enumerate(replacement_pack_jsons, start=1):
        alias = f"r{index:03d}"
        if not manifest_path.exists():
            failed.append(f"replacement_manifest_missing:{alias}")
            continue
        payload = read_json(manifest_path)
        manifests.append({"replacement_alias": alias, "manifest_path": str(manifest_path)})
        failed.extend(replacement_failed_checks(alias, payload))
        for row_index, row in enumerate(replacement_rows(manifest_path, payload), start=1):
            if str(row.get("stem_type") or "").lower() != "minus":
                continue
            failed.extend(collect_true_flag_failures(f"replacement_row:{alias}:{row_index}", row))
            key = replacement_match_key(row)
            if not key:
                failed.append(f"replacement_match_key_missing:{alias}:{row_index}")
                continue
            if key in mapping:
                failed.append(f"duplicate_replacement_source_minus:{alias}:{row_index}")
                continue
            try:
                audio_path = replacement_audio_path(row, manifest_path)
            except FileNotFoundError as exc:
                failed.append(f"replacement_audio_missing:{alias}:{row_index}:{exc}")
                continue
            mapping[key] = {
                "replacement_alias": alias,
                "replacement_manifest": str(manifest_path),
                "replacement_row": row,
                "replacement_audio_path": audio_path,
                "replacement_match_key": key,
                "source_minus_path": replacement_source_minus_path(row),
                "used": False,
            }
    return mapping, manifests, failed


def build_short_listen_replacement_pack(
    *,
    base_pack_json: Path,
    replacement_pack_jsons: Sequence[Path],
    output_dir: Path,
    pack_id: str,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        pack_id=pack_id,
        base_pack_json=str(base_pack_json),
        replacement_pack_jsons=[str(path) for path in replacement_pack_jsons],
    )

    failed: list[str] = []
    if not base_pack_json.exists():
        failed.append("base_pack_missing")
        base_pack: dict[str, Any] = {}
    else:
        base_pack = read_json(base_pack_json)
    failed.extend(collect_true_flag_failures("base_pack", base_pack))
    if base_pack.get("schema") != "aggregate_short_listen_pack_manifest.v1":
        failed.append("base_pack_schema_unexpected")
    if base_pack.get("delivery_status") != "accepted" or base_pack.get("delivery_allowed") is not True:
        failed.append("base_pack_not_accepted")
    base_items = [item for item in base_pack.get("items") or [] if isinstance(item, dict)]

    replacements, replacement_manifests, replacement_failed = build_replacement_map(replacement_pack_jsons)
    failed.extend(replacement_failed)

    selected: list[dict[str, Any]] = []
    used_destinations: set[str] = set()
    replacement_count = 0
    for index, item in enumerate(base_items, start=1):
        failed.extend(collect_true_flag_failures(f"base_item:{index}", item))
        relative_path = str(item.get("audio_path") or "")
        failed.extend(check_short_relative_path(relative_path))
        replacement = None
        if str(item.get("stem_type") or "").lower() == "minus":
            for source_key in base_match_keys(item):
                replacement = replacements.get(source_key)
                if replacement:
                    break
        try:
            source_audio = replacement["replacement_audio_path"] if replacement else base_audio_path(item, base_pack_json)
        except FileNotFoundError as exc:
            failed.append(f"source_audio_missing:{index}:{exc}")
            continue
        destination = output_dir / relative_path
        destination_key = Path(relative_path).as_posix()
        if destination_key in used_destinations:
            failed.append(f"duplicate_output_path:{destination_key}")
            continue
        used_destinations.add(destination_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_audio, destination)
        metadata = audio_metadata(destination)
        row = dict(item)
        row.update(
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "audio_path": destination_key,
                "source_audio_path": str(source_audio),
                "release_ready": False,
                "can_publish_as_result": False,
                "sample_rate": metadata["sample_rate"],
                "channels": metadata["channels"],
                "duration_sec": metadata["duration_sec"],
                "sha256": sha256_file(destination),
                "replacement_applied": bool(replacement),
            }
        )
        if replacement:
            replacement["used"] = True
            replacement_count += 1
            replacement_row = replacement["replacement_row"]
            row.update(
                {
                    "source_manifest": str(replacement["replacement_manifest"]),
                    "source_row": replacement_row,
                    "replacement_source_audio_path": str(source_audio),
                    "replacement_source_manifest": str(replacement["replacement_manifest"]),
                    "replacement_source_minus_path": str(replacement["source_minus_path"]),
                    "replacement_match_key": str(replacement["replacement_match_key"]),
                    "base_source_audio_path": str(item.get("source_audio_path") or ""),
                    "semantic_quality_verified": replacement_row.get("semantic_quality_verified") is True,
                    "quality_status": str(replacement_row.get("release_status") or "listen_preview_only"),
                }
            )
        selected.append(row)

    for key, replacement in replacements.items():
        if not replacement.get("used"):
            failed.append(f"unused_replacement_source_minus:{key}")

    failed = stable_unique(failed)
    accepted = bool(selected) and not failed
    if not accepted:
        selected = []
        replacement_count = 0
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

    coverage = build_coverage(pack_id=pack_id, output_dir=output_dir, selected=selected, failed_checks=failed)
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
        "replacement_count": replacement_count,
        "items": selected if accepted else [],
        "failed_checks": failed,
        "base_pack_json": str(base_pack_json),
        "replacement_manifests": replacement_manifests,
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
            "replacement_packaging_only": True,
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
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / SELECTED_NAME, selected if accepted else [])
    write_json(output_dir / COVERAGE_NAME, coverage)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / README_NAME).write_text(build_readme(pack_id, selected if accepted else []), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        delivery_status=manifest["delivery_status"],
        item_count=manifest["item_count"],
        replacement_count=replacement_count,
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short listen pack with accepted replacement rows.")
    parser.add_argument("--base-pack-json", type=Path, required=True)
    parser.add_argument("--replacement-pack-json", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_replacement_pack.py",
        entrypoint_path="/app/tools/build_short_listen_replacement_pack.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_replacement_pack(
        base_pack_json=args.base_pack_json,
        replacement_pack_jsons=args.replacement_pack_json,
        output_dir=args.output_dir,
        pack_id=str(args.pack_id),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "item_count": manifest["item_count"],
                "replacement_count": manifest["replacement_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
