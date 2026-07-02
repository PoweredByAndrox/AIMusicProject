#!/usr/bin/env python3
"""Copy accepted registered stem-manifest audio into a short-name listen pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import ARTIFACT_AUDIO_DIR, relative_artifact_path, stem_audio_filename, track_alias  # noqa: E402
from docker_runtime import require_docker_runtime  # noqa: E402
from full_pack_stem_contract import (  # noqa: E402
    missing_required_stems,
    present_stems_from_items,
    required_stem_missing_checks,
    required_stems_from_cli,
)
from path_prefix_mapping import parse_path_prefix_map, resolve_audio_path  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "pack.json"
SELECTED_NAME = "selected.jsonl"
PROGRESS_NAME = "progress.jsonl"
DEFAULT_STEMS = ("vocal", "minus", "drums", "bass", "other", "guitar", "piano")
STEM_SOURCE_KEYS = {
    "vocal": ("vocal", "vocals"),
    "minus": ("minus", "instrumental"),
    "drums": ("drums",),
    "bass": ("bass",),
    "other": ("other",),
    "guitar": ("guitar",),
    "piano": ("piano",),
    "melody": ("melody",),
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
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
        "gpu_inference_skip_reason": "stem_manifest_pack_copies_existing_registered_audio",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "stem_manifest_pack_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def normalize_stem(value: Any) -> str:
    stem = str(value or "").strip().lower().replace(" ", "_")
    if stem == "vocals":
        return "vocal"
    if stem == "instrumental":
        return "minus"
    return stem


def nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def audio_metadata(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(duration, 6),
        "frames": int(info.frames),
    }


def source_contract_checks(source_manifest: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if source_manifest.get("release_ready") is True:
        failed.append("source_release_ready_true")
    if source_manifest.get("delivery_allowed") is True:
        failed.append("source_delivery_allowed_true")
    contract = nested_dict(source_manifest, "contract")
    if not contract:
        failed.append("source_contract_missing")
    elif contract.get("registered_model_candidate") is not True:
        failed.append("source_contract_not_registered_model_candidate")
    if contract.get("fallback_used") is True:
        failed.append("source_contract_fallback_used")
    if contract.get("hidden_inference") is True:
        failed.append("source_contract_hidden_inference")
    if contract.get("per_track_hardcode_allowed") is True:
        failed.append("source_contract_per_track_hardcode")
    if contract.get("not_timestamp_hardcoded") is False:
        failed.append("source_contract_timestamp_hardcode")
    execution = nested_dict(source_manifest, "execution_contract")
    cuda = nested_dict(execution, "cuda_preflight")
    if execution.get("gpu_inference_required") is True:
        if execution.get("gpu_inference_skipped") is True:
            failed.append("source_gpu_inference_skipped")
        if cuda.get("cuda_available") is not True or str(cuda.get("resolved_device") or "").lower() != "cuda":
            failed.append("source_gpu_proof_missing")
    for key in (
        "training_started",
        "fine_tune_started",
        "models_changed",
        "model_registry_edited",
        "cleanup_routes_added",
        "thresholds_relaxed",
    ):
        if source_manifest.get(key) is True:
            failed.append(f"source_{key}_true")
    return failed


def selected_stems(requested_stems: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for stem in requested_stems:
        normalized = normalize_stem(stem)
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result or DEFAULT_STEMS)


def source_stem_payload(source_manifest: dict[str, Any], stem_type: str) -> tuple[str, dict[str, Any]]:
    stems = nested_dict(source_manifest, "stems")
    for key in STEM_SOURCE_KEYS.get(stem_type, (stem_type,)):
        value = stems.get(key)
        if isinstance(value, dict):
            return key, value
    return "", {}


def source_role_safe_metadata(
    *,
    source_manifest: dict[str, Any],
    payload: dict[str, Any],
    stem_type: str,
    source_key: str,
    source_manifest_path: Path,
    source_path: Path,
) -> dict[str, str]:
    contract = nested_dict(source_manifest, "role_safe_stem_contract")
    target_artifact = nested_dict(source_manifest, "target_stem_artifact")
    source_manifest_target_stem = str(
        payload.get("source_manifest_target_stem")
        or contract.get("source_manifest_target_stem")
        or source_manifest.get("target_stem")
        or source_key
        or stem_type
    ).strip().lower()
    canonical_stem_type = str(
        payload.get("canonical_stem_type")
        or contract.get("canonical_stem_type")
        or source_manifest_target_stem
        or source_key
        or stem_type
    ).strip().lower()
    product_stem_type = str(
        payload.get("product_stem_type")
        or contract.get("product_stem_type")
        or source_manifest.get("target_product_stem")
        or stem_type
    ).strip().lower()
    model_output_stem_type = str(
        payload.get("model_output_stem_type")
        or contract.get("model_output_stem_type")
        or target_artifact.get("model_output_stem_type")
        or canonical_stem_type
        or source_key
    ).strip().lower()
    manifest_stem_name = str(
        payload.get("manifest_stem_name")
        or contract.get("manifest_stem_name")
        or target_artifact.get("manifest_stem_name")
        or source_key
    ).strip().lower()
    naming_contract = str(
        payload.get("naming_contract")
        or contract.get("naming_contract")
        or target_artifact.get("naming_contract")
        or "source_manifest_stem_matches_product_role"
    )
    return {
        "source_manifest_target_stem": source_manifest_target_stem,
        "canonical_stem_type": canonical_stem_type,
        "user_facing_stem_role": str(
            payload.get("user_facing_stem_role")
            or contract.get("user_facing_stem_role")
            or target_artifact.get("user_facing_stem_role")
            or product_stem_type
        ).strip().lower(),
        "manifest_stem_name": manifest_stem_name,
        "product_stem_type": product_stem_type,
        "model_output_stem_type": model_output_stem_type,
        "source_manifest_stem_type": source_manifest_target_stem,
        "naming_contract": naming_contract,
        "product_audio_path": str(payload.get("product_audio_path") or source_manifest_path),
        "model_output_audio_path": str(
            payload.get("model_output_audio_path")
            or target_artifact.get("model_output_audio_path")
            or source_manifest.get("target_model_output_audio_path")
            or source_manifest.get("target_audio_path")
            or source_manifest_path
        ),
        "copied_product_audio_path": str(source_path),
    }


def parse_override_stem_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"override stem path must use stem=path: {raw}")
    stem, path = raw.split("=", 1)
    normalized = normalize_stem(stem)
    if not normalized:
        raise ValueError(f"override stem is empty: {raw}")
    if not path.strip():
        raise ValueError(f"override path is empty for stem {normalized}")
    return normalized, Path(path)


def normalize_override_stem_paths(overrides: dict[str, Path] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for stem, path in (overrides or {}).items():
        normalized = normalize_stem(stem)
        if normalized:
            result[normalized] = Path(path)
    return result


def build_stem_manifest_audio_pack(
    *,
    stem_manifest_json: Path,
    output_dir: Path,
    track_id: str,
    stems: Sequence[str] = DEFAULT_STEMS,
    override_stem_paths: dict[str, Path] | None = None,
    path_prefix_map: Sequence[tuple[str, str]] = (),
    require_full_pack_stems: bool = False,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    source_manifest = read_json(stem_manifest_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()

    required_stems = required_stems_from_cli(requested=(), require_full_pack=require_full_pack_stems)
    requested_stems = selected_stems(required_stems or stems)
    overrides = normalize_override_stem_paths(override_stem_paths)
    progress(
        progress_path,
        "start",
        stem_manifest_json=str(stem_manifest_json),
        requested_stems=list(requested_stems),
        required_stems=required_stems,
    )
    failed_checks = source_contract_checks(source_manifest)
    selected: list[dict[str, Any]] = []
    alias = track_alias(1)
    if not failed_checks:
        for stem_type in requested_stems:
            source_key, payload = source_stem_payload(source_manifest, stem_type)
            if not payload:
                failed_checks.append(f"source_stem_missing:{stem_type}")
                continue
            source_manifest_path = Path(str(payload.get("path") or ""))
            mapped_path_text, path_resolution = resolve_audio_path(str(source_manifest_path), path_prefix_map)
            source_path = overrides.get(stem_type, Path(mapped_path_text))
            if stem_type in overrides:
                path_resolution = {
                    **path_resolution,
                    "override_applied": True,
                    "override_path": str(source_path),
                }
            if not source_path.exists() or not source_path.is_file():
                failed_checks.append(f"source_stem_audio_missing:{stem_type}")
                continue
            artifact_filename = stem_audio_filename(stem_type)
            listen_path = output_dir / ARTIFACT_AUDIO_DIR / alias / artifact_filename
            listen_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, listen_path)
            metadata = audio_metadata(listen_path)
            selected.append(
                {
                    "schema": "stem_manifest_audio_pack_stem.v1",
                    "track_alias": alias,
                    "track_id": track_id,
                    "source_track_id": track_id,
                    "stem_type": stem_type,
                    "source_stem_key": source_key,
                    "source_audio_path": str(source_path),
                    "source_manifest_audio_path": str(source_manifest_path),
                    **source_role_safe_metadata(
                        source_manifest=source_manifest,
                        payload=payload,
                        stem_type=stem_type,
                        source_key=source_key,
                        source_manifest_path=source_manifest_path,
                        source_path=source_path,
                    ),
                    "audio_override_applied": stem_type in overrides,
                    "source_audio_path_resolution": path_resolution,
                    "listen_audio_path": str(listen_path),
                    "artifact_filename": artifact_filename,
                    "artifact_relative_path": relative_artifact_path(listen_path, output_dir),
                    "listen_audio_sha256": sha256_file(listen_path),
                    "sample_rate": metadata["sample_rate"],
                    "channels": metadata["channels"],
                    "duration_sec": metadata["duration_sec"],
                    "quality_status": "listen_preview_only_not_semantic_verified",
                    "semantic_quality_verified": False,
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "release_status": "listen_preview_only",
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            )

    present_stems = present_stems_from_items(selected, required_stems)
    missing_stems = missing_required_stems(required_stems, present_stems)
    failed_checks.extend(required_stem_missing_checks(missing_stems))
    failed_checks = sorted(set(failed_checks))
    accepted = bool(selected) and not failed_checks
    if not accepted:
        selected = []
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
    manifest = {
        "schema": "stem_manifest_audio_pack_manifest.v1",
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": accepted,
        "deliverable": False,
        "semantic_quality_verified": False,
        "stem_manifest_json": str(stem_manifest_json),
        "source_audio_path": str(source_manifest.get("audio_path") or ""),
        "source_model_name": str(source_manifest.get("model_name") or ""),
        "track_id": track_id,
        "track_aliases": [{"track_alias": alias, "source_track_id": track_id}],
        "requested_stems": list(requested_stems),
        "required_stems": required_stems,
        "present_stems": present_stems,
        "missing_required_stems": missing_stems,
        "source_audio_overrides": {stem: str(path) for stem, path in sorted(overrides.items())},
        "path_prefix_map": [{"from": source, "to": target} for source, target in path_prefix_map],
        "stem_count": len(selected) if accepted else 0,
        "stems": selected if accepted else [],
        "failed_checks": failed_checks,
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "progress_jsonl": str(progress_path),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "track_dir_strategy": "tNNN",
            "filename_strategy": "stem_type_alias",
            "raw_track_or_stem_ids_in_path": False,
            "provenance_in_manifest": True,
        },
        "contract": {
            "source_registered_stem_manifest_required": True,
            "audio_copied_without_modification": True,
            "explicit_audio_override_allowed_with_provenance": True,
            "no_inference_run": True,
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "release_ready_forced_false": True,
            "short_windows_safe_artifact_names": True,
            "raw_ids_kept_in_manifest_only": True,
            "required_stem_contract_active": bool(required_stems),
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / SELECTED_NAME, selected if accepted else [])
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], stem_count=manifest["stem_count"], failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short-name listen pack from a registered stem_manifest.json.")
    parser.add_argument("--stem-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--stem", action="append", default=[])
    parser.add_argument("--require-full-pack-stems", action="store_true")
    parser.add_argument(
        "--override-stem-path",
        action="append",
        default=[],
        help="Explicit existing audio override in stem=/path form, for example minus=/app/run/normalized/minus.flac.",
    )
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Map source manifest stem paths into the local artifact mirror before copying.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_stem_manifest_audio_pack.py", entrypoint_path="/app/tools/build_stem_manifest_audio_pack.py")
    args = parse_args(argv)
    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=args.stem_manifest_json,
        output_dir=args.output_dir,
        track_id=str(args.track_id),
        stems=tuple(str(item) for item in args.stem) or DEFAULT_STEMS,
        override_stem_paths=dict(parse_override_stem_path(str(item)) for item in args.override_stem_path),
        path_prefix_map=parse_path_prefix_map(args.path_prefix_map),
        require_full_pack_stems=bool(args.require_full_pack_stems),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(json.dumps({"manifest_path": str(args.output_dir / MANIFEST_NAME), "delivery_status": manifest["delivery_status"], "stem_count": manifest["stem_count"]}, sort_keys=True))
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
