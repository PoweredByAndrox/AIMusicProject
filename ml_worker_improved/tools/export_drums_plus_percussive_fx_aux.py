#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from export_hq_stems_full import _ensure_hybrid_bs_runtime as ensure_hybrid_bs_runtime  # noqa: E402
from provenance_contract import assert_no_reference_lineage  # noqa: E402
from worker.high_quality_stem_export import export_track_stems_aggressive_melody_first  # noqa: E402


TARGET_STEM = "drums_plus_percussive_fx_aux"
SOURCE_MODEL_STEM = "drums"
DEFAULT_ROUTE_NAME = "htdemucs_6s_drums_plus_percussive_fx_aux"
BASE_MANIFEST_ROUTE_NAME = "drums_plus_percussive_fx_aux_from_base_manifest"
REQUIRED_NO_REFERENCE_CONTRACT: dict[str, bool] = {
    "target_reference_assisted": False,
    "target_reference_used_at_application": False,
    "application_reference_free": True,
    "fallback_used": False,
    "per_track_hardcode_allowed": False,
    "not_timestamp_hardcoded": True,
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_listen_file(*, target_path: Path, listen_dir: Path, listen_label: str) -> dict[str, Any]:
    listen_dir.mkdir(parents=True, exist_ok=True)
    listen_path = listen_dir / listen_label
    shutil.copy2(target_path, listen_path)
    return {
        "listen_dir": str(listen_dir),
        "files": {
            TARGET_STEM: str(listen_path),
        },
    }


def _manifest_model_id(base_manifest: dict[str, Any]) -> str:
    model_id = base_manifest.get("model_id") or base_manifest.get("model_name")
    if not model_id and isinstance(base_manifest.get("model"), dict):
        model_id = base_manifest["model"].get("model_id") or base_manifest["model"].get("base_model")
    return str(model_id or "unknown_model")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


def _resolve_manifest_relative_path(path_value: str, manifest_dir: Path | None) -> Path:
    candidate_path = Path(path_value)
    if candidate_path.is_absolute() or manifest_dir is None:
        return candidate_path
    return manifest_dir / candidate_path


def _validated_base_drum_path(
    base_manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    manifest_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    lineage_report = assert_no_reference_lineage(
        base_manifest,
        manifest_path=manifest_path,
        role="base export",
        required_contract=REQUIRED_NO_REFERENCE_CONTRACT,
    )
    contract = base_manifest.get("contract", {})
    if bool(contract.get("target_reference_assisted")):
        raise ValueError("base export is reference-assisted and cannot seed a no-reference candidate")
    if bool(contract.get("target_reference_used_at_application")) or bool(contract.get("target_reference_used")):
        raise ValueError("base export used a target reference at application time")
    if contract.get("application_reference_free") is False:
        raise ValueError("base export is not application-reference-free")
    if bool(contract.get("fallback_used")):
        raise ValueError("base export used fallback and cannot seed a production candidate")
    if bool(contract.get("per_track_hardcode_allowed")):
        raise ValueError("base export allows per-track hardcodes and cannot seed a production candidate")
    if contract.get("not_timestamp_hardcoded") is False:
        raise ValueError("base export may contain timestamp hardcodes and cannot seed a production candidate")

    stem_entry = base_manifest.get("stems", {}).get(SOURCE_MODEL_STEM)
    if not isinstance(stem_entry, dict):
        raise ValueError(f"base export does not contain required stem: {SOURCE_MODEL_STEM}")

    stem_path_value = stem_entry.get("path")
    if not stem_path_value:
        raise ValueError(f"base export stem has no path: {SOURCE_MODEL_STEM}")

    stem_path = _resolve_manifest_relative_path(str(stem_path_value), manifest_dir)
    if not stem_path.exists():
        raise FileNotFoundError(f"base export stem file is missing: {stem_path}")
    return stem_path, lineage_report


def build_drums_plus_percussive_fx_aux_manifest(
    *,
    base_manifest: dict[str, Any],
    output_dir: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
    source_candidate_manifest: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_drum_path, lineage_report = _validated_base_drum_path(
        base_manifest,
        manifest_path=source_candidate_manifest,
        manifest_dir=source_candidate_manifest.parent if source_candidate_manifest is not None else None,
    )
    target_path = output_dir / f"{TARGET_STEM}.flac"
    shutil.copy2(source_drum_path, target_path)

    source_stem_entry = dict(base_manifest["stems"][SOURCE_MODEL_STEM])
    source_stem_entry.pop("path", None)
    source_candidate_model_id = _manifest_model_id(base_manifest)
    manifest = {
        "schema_version": 1,
        "source_audio_path": base_manifest.get("audio_path") or base_manifest.get("source_audio_path"),
        "model_id": route_name,
        "model_name": base_manifest.get("model_name"),
        "model": {
            "model_id": route_name,
            "base_model": base_manifest.get("model_name"),
            "source_candidate_model_id": source_candidate_model_id,
            "source_model_stem": SOURCE_MODEL_STEM,
            "target_stem": TARGET_STEM,
        },
        "contract": {
            "benchmark_compatible": True,
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "reference_teacher_pack": False,
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "no_reference_production_claim": True,
        },
        "pipeline": {
            "route": route_name,
            "base_export_manifest": base_manifest.get("manifest_path"),
            "source_candidate_manifest": str(source_candidate_manifest) if source_candidate_manifest is not None else None,
            "source_candidate_model_id": source_candidate_model_id,
            "base_pipeline": base_manifest.get("pipeline", {}),
            "target_definition": TARGET_STEM,
            "source_model_stem": SOURCE_MODEL_STEM,
            "reference_audio_used": False,
            "track_specific_state_used": False,
            "provenance_contract": lineage_report,
        },
        "stems": {
            TARGET_STEM: {
                "path": str(target_path),
                "source_model_stem": SOURCE_MODEL_STEM,
                "source_candidate_model_id": source_candidate_model_id,
                "target_definition": TARGET_STEM,
                "derived": False,
                **source_stem_entry,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_drums_plus_percussive_fx_aux_from_base_manifest(
    *,
    base_manifest_path: Path,
    output_dir: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
) -> dict[str, Any]:
    base_manifest_path = base_manifest_path.resolve()
    base_manifest = _load_json(base_manifest_path)
    base_manifest["manifest_path"] = str(base_manifest_path)
    return build_drums_plus_percussive_fx_aux_manifest(
        base_manifest=base_manifest,
        output_dir=output_dir,
        route_name=route_name,
        source_candidate_manifest=base_manifest_path,
    )


def _route_name_for_model(model_name: str) -> str:
    normalized_model_name = str(model_name).strip()
    if normalized_model_name == "htdemucs_6s":
        return "htdemucs_6s_drums_plus_percussive_fx_aux"
    if normalized_model_name == "htdemucs_ft":
        return "htdemucs_ft_drums_plus_percussive_fx_aux"
    safe_model_name = "".join(
        character if character.isalnum() or character in ("_", "-") else "_"
        for character in normalized_model_name.lower()
    ).strip("_")
    if not safe_model_name:
        safe_model_name = "unknown_model"
    return f"{safe_model_name}_{TARGET_STEM}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a no-reference drums+percussive-FX-aux candidate from a full mix."
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--route-name")
    parser.add_argument("--model-name", default="htdemucs_6s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-label", default="01_drums_plus_percussive_fx_aux.flac")
    return parser.parse_args()


def _resolve_input_paths(args: argparse.Namespace) -> tuple[Path | None, Path]:
    if args.base_manifest is not None:
        if args.output_dir is not None:
            if args.paths:
                raise ValueError("base-manifest mode accepts --output-dir or one positional output_dir, not both")
            return None, args.output_dir
        if len(args.paths) != 1:
            raise ValueError("base-manifest mode requires exactly one output directory")
        return None, args.paths[0]

    if args.output_dir is not None:
        if len(args.paths) != 1:
            raise ValueError("full-mix mode with --output-dir requires exactly one source audio path")
        return args.paths[0], args.output_dir
    if len(args.paths) != 2:
        raise ValueError("full-mix mode requires source_audio_path and output_dir")
    return args.paths[0], args.paths[1]


def main() -> int:
    require_docker_runtime(
        tool_name="export_drums_plus_percussive_fx_aux.py",
        entrypoint_path="/app/tools/export_drums_plus_percussive_fx_aux.py",
    )
    args = parse_args()
    source_audio_path, output_dir_argument = _resolve_input_paths(args)
    output_dir = output_dir_argument.resolve()

    if args.base_manifest is not None:
        started_at = time.perf_counter()
        route_name = str(args.route_name or BASE_MANIFEST_ROUTE_NAME)
        manifest = build_drums_plus_percussive_fx_aux_from_base_manifest(
            base_manifest_path=args.base_manifest,
            output_dir=output_dir,
            route_name=route_name,
        )
        manifest["execution"] = {
            "tool": "ml_worker_improved/tools/export_drums_plus_percussive_fx_aux.py",
            "mode": "base_manifest",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
            "device": None,
        }
        if args.listen_dir is not None:
            target_path = Path(manifest["stems"][TARGET_STEM]["path"])
            manifest["listen"] = _copy_listen_file(
                target_path=target_path,
                listen_dir=args.listen_dir.resolve(),
                listen_label=str(args.listen_label),
            )
        manifest_path = output_dir / "stem_manifest.json"
        _write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    if source_audio_path is None:
        raise ValueError("source_audio_path is required outside base-manifest mode")
    source_audio_path = source_audio_path.resolve()
    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")

    started_at = time.perf_counter()
    base_export_dir = output_dir / "_base_aggressive_melody_first_export"
    if str(args.model_name).startswith("hybrid_"):
        ensure_hybrid_bs_runtime()
    base_manifest = export_track_stems_aggressive_melody_first(
        source_audio_path,
        base_export_dir,
        model_name=str(args.model_name),
        device=str(args.device),
        sharpen_bleed=True,
    )
    route_name = str(args.route_name or _route_name_for_model(str(args.model_name)))
    manifest = build_drums_plus_percussive_fx_aux_manifest(
        base_manifest=base_manifest,
        output_dir=output_dir,
        route_name=route_name,
    )
    manifest["execution"] = {
        "tool": "ml_worker_improved/tools/export_drums_plus_percussive_fx_aux.py",
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "device": str(args.device),
    }
    if args.listen_dir is not None:
        target_path = Path(manifest["stems"][TARGET_STEM]["path"])
        manifest["listen"] = _copy_listen_file(
            target_path=target_path,
            listen_dir=args.listen_dir.resolve(),
            listen_label=str(args.listen_label),
        )
    manifest_path = output_dir / "stem_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
