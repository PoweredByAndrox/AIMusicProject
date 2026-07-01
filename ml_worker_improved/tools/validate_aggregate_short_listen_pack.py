#!/usr/bin/env python3
"""Validate aggregate short-name listen packs fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.full_pack_stem_contract import (  # noqa: E402
    missing_required_stems,
    normalize_stem_list,
    present_stems_from_items,
    required_stem_quality_missing_checks,
    required_stem_missing_checks,
    required_stems_from_cli,
)


START_TIME = time.monotonic()
MANIFEST_NAME = "validation.json"
PROGRESS_NAME = "progress.jsonl"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
        "expected_acceleration_mode": "external_only_cpu_metadata_validation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "aggregate_short_listen_pack_validation_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "aggregate_short_listen_pack_validation_no_training",
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


def resolve_artifact_path(pack_root: Path, raw_path: Any, default_name: str) -> Path:
    raw = str(raw_path or default_name)
    path = Path(raw)
    if not path.is_absolute():
        path = pack_root / path
    return path


def check_short_relative_path(relative_path: str) -> list[str]:
    failed: list[str] = []
    path = Path(str(relative_path or ""))
    if path.is_absolute() or not str(relative_path or ""):
        return ["invalid_relative_audio_path"]
    for segment in path.parts:
        try:
            validate_artifact_segment(segment)
        except ValueError:
            failed.append("long_path_segment")
            break
    return failed


def list_failed_checks(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed_checks = payload.get("failed_checks")
    if failed_checks is None:
        return []
    if not isinstance(failed_checks, list):
        return [f"{prefix}_failed_checks_invalid"]
    if failed_checks:
        return [f"{prefix}_failed_checks_not_empty"]
    return []


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


def check_policy(policy: Any) -> list[str]:
    if not isinstance(policy, dict) or policy.get("schema") != "short_windows_safe_artifact_names.v1":
        return ["short_name_policy_missing"]
    failed: list[str] = []
    if policy.get("audio_dir") != "audio":
        failed.append("short_name_policy_audio_dir_unexpected")
    if policy.get("raw_track_or_stem_ids_in_path") is not False:
        failed.append("short_name_policy_raw_ids_not_false")
    if policy.get("provenance_in_manifest") is not True:
        failed.append("short_name_policy_provenance_not_true")
    if policy.get("source_alias_strategy") != "caller_supplied_validated_short_alias":
        failed.append("short_name_policy_alias_strategy_unexpected")
    expected_metadata = ["pack.json", "coverage.json", "selected.jsonl", "progress.jsonl", "README.md"]
    if policy.get("metadata_filenames") != expected_metadata:
        failed.append("short_name_policy_metadata_filenames_unexpected")
    return failed


def check_audio_integrity(pack_root: Path, item: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    audio_path = str(item.get("audio_path") or "")
    resolved_audio = pack_root / audio_path
    if not resolved_audio.exists() or not resolved_audio.is_file():
        return ["audio_missing"]

    expected_sha = str(item.get("sha256") or "")
    if not expected_sha:
        failed.append("audio_sha256_missing")
    else:
        try:
            if sha256_file(resolved_audio) != expected_sha:
                failed.append("audio_sha256_mismatch")
        except OSError:
            failed.append("audio_sha256_unreadable")

    try:
        metadata = audio_metadata(resolved_audio)
    except Exception:
        failed.append("audio_metadata_unreadable")
        return failed

    if item.get("sample_rate") is not None and int(item.get("sample_rate") or 0) != metadata["sample_rate"]:
        failed.append("audio_sample_rate_mismatch")
    if item.get("channels") is not None and int(item.get("channels") or 0) != metadata["channels"]:
        failed.append("audio_channels_mismatch")
    if item.get("duration_sec") is not None:
        expected_duration = float(item.get("duration_sec") or 0.0)
        if abs(expected_duration - float(metadata["duration_sec"])) > 0.001:
            failed.append("audio_duration_mismatch")
    return failed


def check_selected_jsonl(pack_root: Path, pack: dict[str, Any], items: list[dict[str, Any]]) -> list[str]:
    failed: list[str] = []
    selected_path = resolve_artifact_path(pack_root, pack.get("selected_jsonl"), "selected.jsonl")
    if not selected_path.exists():
        return ["selected_jsonl_missing"]
    try:
        selected_rows = read_jsonl(selected_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["selected_jsonl_invalid"]
    if len(selected_rows) != len(items):
        failed.append("selected_jsonl_count_mismatch")
    selected_audio = [str(row.get("audio_path") or "") for row in selected_rows]
    item_audio = [str(item.get("audio_path") or "") for item in items]
    if selected_audio != item_audio:
        failed.append("selected_jsonl_audio_sequence_mismatch")
    return failed


def check_progress_jsonl(pack_root: Path, pack: dict[str, Any]) -> list[str]:
    progress_path = resolve_artifact_path(pack_root, pack.get("progress_jsonl"), "progress.jsonl")
    if not progress_path.exists():
        return ["progress_jsonl_missing"]
    try:
        rows = read_jsonl(progress_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["progress_jsonl_invalid"]
    if not rows:
        return ["progress_jsonl_empty"]
    failed: list[str] = []
    if rows[0].get("event") != "start":
        failed.append("progress_jsonl_missing_start")
    finish = rows[-1]
    if finish.get("event") != "finish":
        failed.append("progress_jsonl_missing_finish")
    if finish.get("delivery_status") != "accepted":
        failed.append("progress_jsonl_finish_not_accepted")
    if finish.get("failed_checks") not in ([], None):
        failed.append("progress_jsonl_finish_failed_checks_not_empty")
    return failed


def check_readme(pack_root: Path, items: list[dict[str, Any]]) -> list[str]:
    readme_path = pack_root / "README.md"
    if not readme_path.exists():
        return ["readme_missing"]
    try:
        text = readme_path.read_text(encoding="utf-8")
    except OSError:
        return ["readme_unreadable"]
    failed: list[str] = []
    for required in ("pack.json", "coverage.json", "selected.jsonl", "progress.jsonl"):
        if required not in text:
            failed.append("readme_metadata_listing_incomplete")
            break
    for item in items:
        if str(item.get("audio_path") or "") not in text:
            failed.append("readme_audio_listing_incomplete")
            break
    return failed


def validate_aggregate_short_listen_pack(
    *,
    pack_manifest_json: Path,
    output_dir: Path,
    required_stems: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", pack_manifest_json=str(pack_manifest_json))

    pack = read_json(pack_manifest_json)
    pack_root = pack_manifest_json.parent
    failed: list[str] = []
    requested_required_stems = required_stems_from_cli(requested=required_stems, require_full_pack=False)
    manifest_required_stems = normalize_stem_list(pack.get("required_stems") if isinstance(pack.get("required_stems"), list) else [])
    contract_required_stems = normalize_stem_list([*manifest_required_stems, *requested_required_stems])
    failed.extend(collect_true_flag_failures("pack", pack))
    failed.extend(list_failed_checks("pack", pack))
    if pack.get("schema") != "aggregate_short_listen_pack_manifest.v1":
        failed.append("unexpected_schema")
    if pack.get("delivery_status") != "accepted":
        failed.append("pack_delivery_status_not_accepted")
    if pack.get("delivery_allowed") is not True:
        failed.append("pack_delivery_allowed_not_true")
    if pack.get("presentation_status") != "accepted_for_listen_pack":
        failed.append("pack_presentation_status_not_accepted")
    if pack.get("release_status") != "listen_preview_only":
        failed.append("pack_release_status_not_listen_preview")
    failed.extend(check_policy(pack.get("artifact_name_policy")))

    items = pack.get("items")
    if not isinstance(items, list) or not items:
        failed.append("items_missing")
        items = []
    if int(pack.get("item_count") or 0) != len(items):
        failed.append("item_count_mismatch")

    audio_paths: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            failed.append("invalid_item")
            continue
        failed.extend(collect_true_flag_failures("item", item))
        audio_path = str(item.get("audio_path") or "")
        failed.extend(check_short_relative_path(audio_path))
        if audio_path in audio_paths:
            failed.append("duplicate_audio_path")
        audio_paths.add(audio_path)
        failed.extend(check_audio_integrity(pack_root, item))
        if not str(item.get("stem_type") or ""):
            failed.append("stem_type_missing")
        if item.get("semantic_quality_verified") is True and str(item.get("quality_status") or "").endswith("not_semantic_verified"):
            failed.append("semantic_quality_contradiction")

    present_stems = present_stems_from_items(items, contract_required_stems)
    missing_stems = missing_required_stems(contract_required_stems, present_stems)
    failed.extend(required_stem_missing_checks(missing_stems))
    failed.extend(required_stem_quality_missing_checks(items, contract_required_stems))

    coverage_path = resolve_artifact_path(pack_root, pack.get("coverage_json"), "coverage.json")
    coverage: dict[str, Any] = {}
    if not coverage_path.exists():
        failed.append("coverage_missing")
    else:
        coverage = read_json(coverage_path)
        failed.extend(collect_true_flag_failures("coverage", coverage))
        failed.extend(list_failed_checks("coverage", coverage))
        if coverage.get("schema") != "aggregate_short_listen_pack_coverage.v1":
            failed.append("coverage_schema_unexpected")
        coverage_files: list[str] = []
        groups = coverage.get("stem_groups")
        if not isinstance(groups, list) or not groups:
            failed.append("coverage_groups_missing")
        else:
            group_aliases: set[str] = set()
            item_count_by_alias = Counter(str(item.get("source_alias") or "") for item in items if isinstance(item, dict))
            stem_types_by_alias: dict[str, set[str]] = {}
            for item in items:
                if isinstance(item, dict):
                    stem_types_by_alias.setdefault(str(item.get("source_alias") or ""), set()).add(str(item.get("stem_type") or ""))
            for group in groups:
                if not isinstance(group, dict):
                    failed.append("coverage_group_invalid")
                    continue
                alias = str(group.get("group_alias") or "")
                if not alias:
                    failed.append("coverage_group_alias_missing")
                elif alias in group_aliases:
                    failed.append("coverage_group_alias_duplicate")
                group_aliases.add(alias)
                if group.get("status") != "listen_preview_only":
                    failed.append("coverage_group_status_unexpected")
                files = group.get("files")
                if isinstance(files, list):
                    coverage_files.extend(str(path) for path in files)
                    if int(group.get("item_count") or 0) != len(files):
                        failed.append("coverage_group_item_count_mismatch")
                    if alias and item_count_by_alias.get(alias, 0) != len(files):
                        failed.append("coverage_group_pack_count_mismatch")
                    if alias:
                        expected_stems = stem_types_by_alias.get(alias, set())
                        actual_stems = {str(stem) for stem in group.get("stem_types") or []}
                        if expected_stems and actual_stems != expected_stems:
                            failed.append("coverage_group_stem_types_mismatch")
                else:
                    failed.append("coverage_group_files_invalid")
        if audio_paths and set(coverage_files) != audio_paths:
            failed.append("coverage_missing_audio_paths")
        if len(coverage_files) != len(set(coverage_files)):
            failed.append("coverage_duplicate_audio_paths")

    failed.extend(check_selected_jsonl(pack_root, pack, items))
    failed.extend(check_progress_jsonl(pack_root, pack))
    failed.extend(check_readme(pack_root, items))

    failed = stable_unique(failed)
    status = "passed" if not failed else "failed"
    manifest = {
        "schema": "aggregate_short_listen_pack_validation.v1",
        "validation_status": status,
        "pack_manifest_json": str(pack_manifest_json),
        "output_dir": str(output_dir),
        "failed_checks": failed,
        "audio_file_count": len(items) if status == "passed" else 0,
        "coverage_group_count": len(coverage.get("stem_groups") or []) if isinstance(coverage, dict) else 0,
        "required_stems": contract_required_stems,
        "present_stems": present_stems,
        "missing_required_stems": missing_stems,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "validation_artifact_not_for_listen_pack",
        "progress_jsonl": str(progress_path),
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
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", validation_status=status, failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an aggregate short-name listen pack.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-stem", action="append", default=[])
    parser.add_argument("--require-full-pack-stems", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="validate_aggregate_short_listen_pack.py",
        entrypoint_path="/app/tools/validate_aggregate_short_listen_pack.py",
    )
    args = parse_args(argv)
    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=args.pack_manifest_json,
        output_dir=args.output_dir,
        required_stems=required_stems_from_cli(
            requested=tuple(str(item) for item in args.required_stem),
            require_full_pack=bool(args.require_full_pack_stems),
        ),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "validation_status": manifest["validation_status"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["validation_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
