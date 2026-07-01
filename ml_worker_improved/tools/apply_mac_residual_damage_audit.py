#!/usr/bin/env python3
"""Run post-hoc damage audit for an existing pitch-passing residual manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mac_audio_damage_audit import run_damage_audit
from tools.mac_residual_route_utils import append_progress, blocked_route_manifest, load_json, path_from_mapping, write_json

SCHEMA = "mac_residual_damage_audit_route.v1"
PRESENTATION_STATUS = "damage_audit_route_diagnostic_only"
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
        schema=SCHEMA,
        presentation_status=PRESENTATION_STATUS,
        residual_manifest_path=residual_manifest_path,
        output_dir=output_dir,
        status=status,
        failed_checks=failed_checks,
        extra=extra,
    )


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _source_is_pitch_passed(residual_manifest: Mapping[str, Any]) -> bool:
    return (
        str(residual_manifest.get("training_status") or "") == PITCH_PASS_TRAINING_STATUS
        and _int_value(residual_manifest.get("before_region_count")) > 0
        and _int_value(residual_manifest.get("after_region_count")) == 0
    )


def _copy_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _write_score_surface(*, guard_manifest: Mapping[str, Any], guard_manifest_path: Path, score_json_path: Path) -> dict[str, Any]:
    score = guard_manifest.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("missing guard score")
    regions = score.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("missing guard score regions")
    payload = {
        "schema": "mac_residual_damage_audit_score_surface.v1",
        "source_guard_manifest": str(guard_manifest_path),
        "source_score_schema": str(score.get("schema") or ""),
        "regions": list(regions),
        "config": dict(score.get("config") or {}) if isinstance(score.get("config"), Mapping) else {},
    }
    write_json(score_json_path, payload)
    return payload


def _status_from_damage(damage_audit: Mapping[str, Any]) -> tuple[str, list[str], str]:
    verdict = str(damage_audit.get("verdict") or "")
    if verdict == "pass":
        return "residual_damage_audit_passed", [], "run_full_holdout_validation"
    if verdict == "fail":
        return "residual_damage_audit_failed", ["damage_audit_failed"], "run_damage_limited_repair"
    blockers = list(damage_audit.get("blocker_reasons") or [])
    return "residual_damage_audit_blocked", blockers or ["damage_audit_blocked"], "inspect_damage_audit_blocker"


def run_existing_residual_damage_audit(
    *,
    residual_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    append_progress(
        progress_path,
        "preflight",
        "start",
        residual_manifest=str(residual_manifest_path),
        acceleration_mode="cpu_dsp_scoring",
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="posthoc_residual_damage_audit_uses_existing_audio",
    )
    residual_manifest = load_json(Path(residual_manifest_path))
    if bool(residual_manifest.get("delivery_allowed")) or bool(residual_manifest.get("listen_pack_created")):
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_source_residual_delivery_artifact",
            failed_checks=["source_residual_delivery_artifact"],
        )
        append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    existing_damage = residual_manifest.get("damage_audit")
    if isinstance(existing_damage, Mapping) and str(existing_damage.get("verdict") or ""):
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_source_residual_already_damage_audited",
            failed_checks=["source_residual_already_damage_audited"],
            extra={
                "source_status": residual_manifest.get("status"),
                "source_damage_verdict": existing_damage.get("verdict"),
                "next_action": "run_damage_limited_repair" if str(existing_damage.get("verdict") or "") == "fail" else "run_full_holdout_validation",
            },
        )
        append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    if not _source_is_pitch_passed(residual_manifest):
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_source_residual_not_pitch_passed",
            failed_checks=["source_residual_not_pitch_passed"],
            extra={
                "source_status": residual_manifest.get("status"),
                "source_training_status": residual_manifest.get("training_status"),
                "before_region_count": _int_value(residual_manifest.get("before_region_count")),
                "after_region_count": _int_value(residual_manifest.get("after_region_count")),
            },
        )
        append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    try:
        guard_manifest_path = path_from_mapping(residual_manifest, "guard_manifest")
        guard_manifest = load_json(guard_manifest_path)
        guard_audio = guard_manifest.get("audio")
        residual_audio = residual_manifest.get("audio")
        if not isinstance(guard_audio, Mapping) or not isinstance(residual_audio, Mapping):
            raise ValueError("missing audio manifest")
        reference_instrumental_path = path_from_mapping(guard_audio, "instrumental")
        candidate_instrumental_path = path_from_mapping(residual_audio, "instrumental")
        score_json_path = output_dir / "score.json"
        _write_score_surface(guard_manifest=guard_manifest, guard_manifest_path=guard_manifest_path, score_json_path=score_json_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest = _blocked_manifest(
            residual_manifest_path=residual_manifest_path,
            output_dir=output_dir,
            status="blocked_missing_guard_or_audio_manifest",
            failed_checks=["missing_guard_or_audio_manifest"],
            extra={"blocker_detail": str(exc)},
        )
        append_progress(progress_path, "manifest", "end", route_status=manifest["status"])
        return manifest

    append_progress(progress_path, "damage_audit", "start", score_json=str(score_json_path))
    damage_output_dir = output_dir / "damage"
    damage_audit = run_damage_audit(
        reference_instrumental_path=reference_instrumental_path,
        candidate_instrumental_path=candidate_instrumental_path,
        score_json_path=score_json_path,
        output_dir=damage_output_dir,
    )
    damage_manifest_path = damage_output_dir / "damage.json"
    if not damage_manifest_path.is_file():
        write_json(damage_manifest_path, damage_audit)

    status, failed_checks, next_action = _status_from_damage(damage_audit)
    manifest = {
        "schema": SCHEMA,
        "status": status,
        "failed_checks": failed_checks,
        "next_action": next_action,
        "source_residual_manifest": str(residual_manifest_path),
        "source_residual_status": residual_manifest.get("status"),
        "source_training_status": residual_manifest.get("training_status"),
        "training_status": residual_manifest.get("training_status"),
        "source_guard_manifest": str(guard_manifest_path),
        "source_training_manifest": str(residual_manifest.get("training_manifest") or ""),
        "source_audio_path": str(residual_manifest.get("source_audio_path") or ""),
        "source_track_identity": _copy_mapping(residual_manifest, "source_track_identity"),
        "reference_bundle_status": str(residual_manifest.get("reference_bundle_status") or "missing_for_actual_source"),
        "reference_mismatch_status": str(residual_manifest.get("reference_mismatch_status") or ""),
        "objective_profile": str(residual_manifest.get("objective_profile") or ""),
        "before_region_count": _int_value(residual_manifest.get("before_region_count")),
        "after_region_count": _int_value(residual_manifest.get("after_region_count")),
        "damage_audit": dict(damage_audit),
        "damage_audit_manifest": str(damage_manifest_path),
        "score_json": str(score_json_path),
        "full_holdout_resolution_status": "not_evaluated",
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": PRESENTATION_STATUS,
    }
    write_json(output_dir / "manifest.json", manifest)
    append_progress(
        progress_path,
        "manifest",
        "end",
        route_status=manifest["status"],
        damage_verdict=str(damage_audit.get("verdict") or ""),
        failed_region_count=_int_value(damage_audit.get("failed_region_count")),
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_existing_residual_damage_audit(
        residual_manifest_path=args.residual_manifest,
        output_dir=args.out,
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
