#!/usr/bin/env python3
"""Apply damage-limited repair to an existing residual training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mac_residual_route_utils import (
    append_progress as _append_progress,
    blocked_route_manifest,
    load_json as _load_json,
    path_from_mapping as _path_from_mapping,
    write_json as _write_json,
)
from tools.train_mac_refiner_from_fulltrack_residuals import run_damage_limited_repair

AUDIT_ROUTE_SCHEMA = "mac_residual_damage_audit_route.v1"
AUDIT_ROUTE_PRESENTATION_STATUS = "damage_audit_route_diagnostic_only"
PITCH_PASS_TRAINING_STATUS = "trained_repair_passed_pitch_guard"


def _blocked_manifest(
    *,
    residual_manifest_path: Path,
    output_dir: Path,
    status: str,
    failed_checks: list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return blocked_route_manifest(
        schema="mac_residual_damage_limited_repair_route.v1",
        presentation_status="damage_limited_repair_diagnostic_only",
        residual_manifest_path=residual_manifest_path,
        output_dir=output_dir,
        status=status,
        failed_checks=failed_checks,
        extra=extra,
    )


def _optional_path(value: Any) -> Path | None:
    text = str(value or "")
    return Path(text) if text else None


def run_existing_residual_damage_limited_repair(
    *,
    residual_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(
        progress_path,
        "preflight",
        "start",
        residual_manifest=str(residual_manifest_path),
        acceleration_mode="cpu_dsp_scoring",
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="posthoc_damage_limited_repair_uses_existing_residual_audio",
    )
    residual_manifest = _load_json(Path(residual_manifest_path))
    if bool(residual_manifest.get("delivery_allowed")) or bool(residual_manifest.get("listen_pack_created")):
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_source_residual_delivery_artifact",
            failed_checks=["source_residual_delivery_artifact"],
        )
        _append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    source_damage_audit_route_manifest = ""
    source_residual_manifest_path = Path(residual_manifest_path)
    source_residual_manifest = residual_manifest
    guard_manifest_path_override: Path | None = None
    source_training_manifest_override = ""
    if str(residual_manifest.get("schema") or "") == AUDIT_ROUTE_SCHEMA:
        if str(residual_manifest.get("presentation_status") or "") != AUDIT_ROUTE_PRESENTATION_STATUS:
            manifest = _blocked_manifest(
                residual_manifest_path=residual_manifest_path,
                output_dir=output_dir,
                status="blocked_source_damage_audit_route_contract",
                failed_checks=["source_damage_audit_route_contract"],
                extra={"source_presentation_status": residual_manifest.get("presentation_status")},
            )
            _append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
            return manifest
        source_damage_audit_route_manifest = str(residual_manifest_path)
        try:
            source_residual_manifest_path = _path_from_mapping(residual_manifest, "source_residual_manifest")
            source_residual_manifest = _load_json(source_residual_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            manifest = _blocked_manifest(
                residual_manifest_path=residual_manifest_path,
                output_dir=output_dir,
                status="blocked_source_damage_audit_route_manifest",
                failed_checks=["source_damage_audit_route_manifest"],
                extra={
                    "source_damage_audit_route_manifest": source_damage_audit_route_manifest,
                    "blocker_detail": str(exc),
                },
            )
            _append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
            return manifest
        guard_manifest_path_override = _optional_path(residual_manifest.get("source_guard_manifest"))
        source_training_manifest_override = str(residual_manifest.get("source_training_manifest") or "")
        if bool(source_residual_manifest.get("delivery_allowed")) or bool(source_residual_manifest.get("listen_pack_created")):
            manifest = _blocked_manifest(
                residual_manifest_path=residual_manifest_path,
                output_dir=output_dir,
                status="blocked_source_residual_delivery_artifact",
                failed_checks=["source_residual_delivery_artifact"],
                extra={
                    "source_damage_audit_route_manifest": source_damage_audit_route_manifest,
                    "source_residual_manifest": str(source_residual_manifest_path),
                },
            )
            _append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
            return manifest

    damage_audit = residual_manifest.get("damage_audit")
    if not isinstance(damage_audit, Mapping) or str(damage_audit.get("verdict")) != "fail":
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_source_residual_not_damage_failed",
            failed_checks=["source_residual_not_damage_failed"],
            extra={"source_status": residual_manifest.get("status")},
        )
        _append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    source_training_status = str(
        residual_manifest.get("source_training_status")
        or source_residual_manifest.get("training_status")
        or residual_manifest.get("training_status")
        or ""
    )
    if source_training_status != PITCH_PASS_TRAINING_STATUS:
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_source_residual_not_pitch_passed",
            failed_checks=["source_residual_not_pitch_passed"],
            extra={
                "source_status": source_residual_manifest.get("status"),
                "source_training_status": source_training_status,
            },
        )
        _append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    guard_manifest_path = guard_manifest_path_override or _path_from_mapping(source_residual_manifest, "guard_manifest")
    guard_manifest = _load_json(guard_manifest_path)
    guard_audio = guard_manifest.get("audio")
    residual_audio = source_residual_manifest.get("audio")
    if not isinstance(guard_audio, Mapping) or not isinstance(residual_audio, Mapping):
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_missing_audio_manifest",
            failed_checks=["missing_audio_manifest"],
        )
        _append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    repair_manifest = run_damage_limited_repair(
        reference_vocals_path=_path_from_mapping(guard_audio, "vocals"),
        reference_instrumental_path=_path_from_mapping(guard_audio, "instrumental"),
        reference_raw_path=_path_from_mapping(guard_audio, "raw_ref"),
        candidate_instrumental_path=_path_from_mapping(residual_audio, "instrumental"),
        guard_score=dict(guard_manifest.get("score") or {}),
        initial_damage_audit=damage_audit,
        output_dir=output_dir,
        progress_path=progress_path,
    )
    manifest = {
        **dict(repair_manifest),
        "schema": "mac_residual_damage_limited_repair_route.v1",
        "source_damage_audit_route_manifest": source_damage_audit_route_manifest,
        "source_residual_manifest": str(source_residual_manifest_path),
        "source_residual_status": source_residual_manifest.get("status"),
        "source_training_status": source_training_status,
        "source_guard_manifest": str(guard_manifest_path),
        "source_training_manifest": source_training_manifest_override or str(source_residual_manifest.get("training_manifest") or ""),
        "full_holdout_resolution_status": "not_evaluated",
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "damage_limited_repair_diagnostic_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        "manifest",
        "end",
        route_status=manifest["status"],
        pitch_verdict=(manifest.get("after_score") or {}).get("verdict") if isinstance(manifest.get("after_score"), Mapping) else "",
        damage_verdict=(manifest.get("damage_audit") or {}).get("verdict") if isinstance(manifest.get("damage_audit"), Mapping) else "",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_existing_residual_damage_limited_repair(
        residual_manifest_path=args.residual_manifest,
        output_dir=args.out,
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
