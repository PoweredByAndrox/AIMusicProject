#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_audio_error_dataset_provenance import audit_audio_error_dataset_provenance  # noqa: E402
from tools.build_audio_error_grounding_dataset import build_audio_error_grounding_dataset  # noqa: E402
from tools.build_audio_error_training_views import build_audio_error_training_views  # noqa: E402
from tools.mac_windows_mount_preflight import ALLOWED_PROJECT_ARTIFACT_ROOTS, DEFAULT_WINDOWS_PROJECT_ROOT  # noqa: E402
from tools.mac_windows_mount_preflight import SCHEMA as MOUNT_PREFLIGHT_SCHEMA  # noqa: E402
from tools.mac_windows_mount_preflight import preflight_mac_windows_project_mount  # noqa: E402
from tools.repair_audio_error_dataset_provenance import repair_audio_error_dataset_provenance  # noqa: E402
from tools.validate_audio_error_dataset_provenance_repair import (  # noqa: E402
    validate_audio_error_dataset_provenance_repair,
)
from tools.validate_audio_error_training_views import validate_audio_error_training_views  # noqa: E402


SAFETY_FLAGS = {
    "delivery_allowed": False,
    "can_publish_as_result": False,
    "listen_delivery_artifact": False,
    "release_ready": False,
    "training_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "models_changed": False,
    "thresholds_relaxed": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
    "diagnostic_artifacts_promoted": False,
}
OWNED_OUTPUT_FILES = ("bridge_manifest.json", "summary.md", "progress.jsonl")
FORBIDDEN_TRUE_FLAGS = (
    "delivery_allowed",
    "can_publish_as_result",
    "listen_delivery_artifact",
    "release_ready",
    "training_started",
    "separator_training_started",
)
FORBIDDEN_LEGACY_CORPUS_MARKERS = ("testing_flstudio_music",)
RUNTIME_INPUT_FILES = (
    "candidate_pool.jsonl",
    "minus_windowed_judge.jsonl",
    "minus_judge.jsonl",
    "no_reference_minus_windowed_judge.jsonl",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def progress(output_dir: Path, phase: str, **payload: Any) -> None:
    append_jsonl(
        output_dir / "progress.jsonl",
        {
            "phase": phase,
            "elapsed_sec": round(float(time.monotonic() - START_MONOTONIC), 3),
            "gpu_inference_skipped": True,
            "expected_acceleration_mode": "cpu_dsp_error_map_extraction_then_metadata_training_view_build",
            **payload,
        },
    )


START_MONOTONIC = time.monotonic()


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OWNED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def not_requested_mount_preflight(output_dir: Path, windows_project_root: str) -> dict[str, Any]:
    return {
        "schema": MOUNT_PREFLIGHT_SCHEMA,
        "preflight_status": "not_requested",
        "blocked_reason": "",
        "can_write_route_artifacts": False,
        "mac_windows_project_mount": "",
        "mac_output_path": "",
        "container_output_path": str(output_dir),
        "windows_project_root": windows_project_root,
        "windows_output_path": "",
        "allowed_project_artifact_roots": list(ALLOWED_PROJECT_ARTIFACT_ROOTS),
        "probe_status": "not_run",
        "probe_path": "",
    }


def load_mount_preflight(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_mount_preflight(
    *,
    output_dir: Path,
    mount_preflight: dict[str, Any] | None = None,
    mount_preflight_path: Path | None = None,
    mac_windows_project_mount: Path | str | None = None,
    mac_output_path: Path | str | None = None,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
) -> dict[str, Any]:
    if mount_preflight is not None:
        payload = dict(mount_preflight)
    elif mount_preflight_path is not None:
        payload = load_mount_preflight(mount_preflight_path)
    elif mac_windows_project_mount is not None:
        payload = preflight_mac_windows_project_mount(
            mac_windows_project_mount=mac_windows_project_mount,
            output_path=mac_output_path or output_dir,
            windows_project_root=windows_project_root,
        )
    else:
        payload = not_requested_mount_preflight(output_dir, windows_project_root)
    payload.setdefault("container_output_path", str(output_dir))
    payload.setdefault("windows_project_root", windows_project_root)
    payload.setdefault("windows_output_path", "")
    payload.setdefault("mac_output_path", "")
    payload.setdefault("probe_status", "not_run")
    return payload


def mount_preflight_validation_errors(
    *,
    mount_preflight: dict[str, Any],
    require_mount_preflight: bool,
) -> list[str]:
    if not require_mount_preflight:
        return []
    if mount_preflight.get("preflight_status") != "passed":
        reason = mount_preflight.get("blocked_reason") or mount_preflight.get("preflight_status") or "not_passed"
        return [f"mount_preflight:{reason}"]
    if mount_preflight.get("can_write_route_artifacts") is not True:
        return ["mount_preflight:can_write_route_artifacts_false"]
    if not mount_preflight.get("windows_output_path"):
        return ["mount_preflight:windows_output_path_missing"]
    if mount_preflight.get("probe_status") != "write_read_delete_passed":
        return [f"mount_preflight:probe_status:{mount_preflight.get('probe_status') or 'missing'}"]
    return []


def mount_preflight_manifest_fields(mount_preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "mount_preflight": mount_preflight,
        "mount_preflight_status": mount_preflight.get("preflight_status", ""),
        "mount_probe_status": mount_preflight.get("probe_status", ""),
        "mac_windows_project_mount": mount_preflight.get("mac_windows_project_mount", ""),
        "mac_output_path": mount_preflight.get("mac_output_path", ""),
        "container_output_path": mount_preflight.get("container_output_path", ""),
        "windows_output_path": mount_preflight.get("windows_output_path", ""),
    }


def runtime_input_files(run_roots: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in run_roots:
        if root.is_file() and root.name in RUNTIME_INPUT_FILES:
            files.add(root)
            continue
        if not root.exists():
            continue
        for name in RUNTIME_INPUT_FILES:
            direct = root / name
            if direct.exists():
                files.add(direct)
            for path in root.rglob(name):
                files.add(path)
    return sorted(files, key=lambda path: str(path))


def forbidden_legacy_corpus_violations(run_roots: Sequence[Path]) -> list[str]:
    violations: list[str] = []
    for path in runtime_input_files(run_roots):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in FORBIDDEN_LEGACY_CORPUS_MARKERS:
            if marker in text:
                violations.append(f"forbidden_legacy_corpus_marker:{marker}:{path}")
    return violations


def blocked_manifest(
    *,
    output_dir: Path,
    run_roots: Sequence[Path],
    output_status: str,
    validation_errors: list[str],
    command_args: Sequence[str] | None,
    mount_preflight: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema": "minus_hardcase_audio_error_bridge.v1",
        "run_roots": [str(path) for path in run_roots],
        "run_root_count": len(run_roots),
        "grounding_dir": str(output_dir / "ground"),
        "audit_dir": str(output_dir / "audit"),
        "repaired_dir": str(output_dir / "repaired"),
        "views_dir": str(output_dir / "views"),
        "grounding_windows_total": 0,
        "grounding_safe_for_training_count": 0,
        "grounding_error_maps_written": 0,
        "audit_rows_audited": 0,
        "audit_rows_repairable": 0,
        "repair_safe_for_training_count_after": 0,
        "repair_validation_ok": False,
        "training_views_validation_ok": False,
        "source_run_rows_added": 0,
        "strict_negative_rows_found": 0,
        "strict_positive_rows_found": 0,
        "can_train_minus_critic": False,
        "can_train_vocal_critic": False,
        "can_train_pair_critic": False,
        "can_train_overall_accept": False,
        "validation_errors": validation_errors,
        "output_status": output_status,
        "next_step": "fix_runtime_roots_or_provenance_before_training",
        "presentation_status": "training_evidence_not_for_listen_pack",
        "command_args": list(command_args or []),
        **mount_preflight_manifest_fields(mount_preflight),
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "bridge_manifest.json", manifest)
    write_summary(output_dir / "summary.md", manifest)
    progress(output_dir, "complete", output_status=output_status, validation_errors=validation_errors)
    return manifest


def has_forbidden_true_flag(payload: dict[str, Any]) -> list[str]:
    return [flag for flag in FORBIDDEN_TRUE_FLAGS if payload.get(flag) is True]


def validation_errors_for(
    *,
    bridge_manifest: dict[str, Any],
    repaired_validation: dict[str, Any],
    views_validation: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for flag in has_forbidden_true_flag(bridge_manifest):
        errors.append(f"forbidden_true_flag:bridge_manifest:{flag}")
    if repaired_validation.get("ok") is not True:
        errors.extend(f"repaired_dataset:{item}" for item in repaired_validation.get("errors", []) or ["not_ok"])
    if views_validation.get("ok") is not True:
        errors.extend(f"training_views:{item}" for item in views_validation.get("errors", []) or ["not_ok"])
    if int(bridge_manifest.get("source_run_rows_added") or 0) <= 0:
        errors.append("no_runtime_rows_added")
    return errors


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Minus Hardcase Audio Error Bridge",
        "",
        f"- schema: `{manifest['schema']}`",
        f"- output_status: `{manifest['output_status']}`",
        f"- run_root_count: {manifest['run_root_count']}",
        f"- grounding_windows_total: {manifest['grounding_windows_total']}",
        f"- grounding_safe_for_training_count: {manifest['grounding_safe_for_training_count']}",
        f"- source_run_rows_added: {manifest['source_run_rows_added']}",
        f"- strict_negative_rows_found: {manifest['strict_negative_rows_found']}",
        f"- strict_positive_rows_found: {manifest['strict_positive_rows_found']}",
        f"- can_train_minus_critic: {str(manifest['can_train_minus_critic']).lower()}",
        f"- mount_preflight_status: `{manifest.get('mount_preflight_status', '')}`",
        f"- mount_probe_status: `{manifest.get('mount_probe_status', '')}`",
        f"- mac_output_path: `{manifest.get('mac_output_path', '')}`",
        f"- windows_output_path: `{manifest.get('windows_output_path', '')}`",
        "- delivery_allowed: false",
        "- listen_delivery_artifact: false",
        "- release_ready: false",
        f"- next_step: {manifest['next_step']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def bridge_minus_hardcase_audio_error_views(
    *,
    run_roots: Sequence[Path],
    output_dir: Path,
    window_sec: float = 12.0,
    hop_sec: float = 12.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    map_max_bins: int = 128,
    map_max_frames: int = 256,
    max_waveform_samples: int = 4096,
    mount_preflight: dict[str, Any] | None = None,
    mount_preflight_path: Path | None = None,
    mac_windows_project_mount: Path | str | None = None,
    mac_output_path: Path | str | None = None,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
    require_mount_preflight: bool = False,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    prepare_output_dir(output_dir)
    run_roots = [Path(path) for path in run_roots]
    mount_preflight_payload = resolve_mount_preflight(
        output_dir=output_dir,
        mount_preflight=mount_preflight,
        mount_preflight_path=mount_preflight_path,
        mac_windows_project_mount=mac_windows_project_mount,
        mac_output_path=mac_output_path,
        windows_project_root=windows_project_root,
    )
    ground_dir = output_dir / "ground"
    audit_dir = output_dir / "audit"
    repaired_dir = output_dir / "repaired"
    views_dir = output_dir / "views"

    progress(
        output_dir,
        "start",
        run_roots=[str(path) for path in run_roots],
        bridge_output_dir=str(output_dir),
        mount_preflight_status=mount_preflight_payload.get("preflight_status"),
        mount_probe_status=mount_preflight_payload.get("probe_status"),
        windows_output_path=mount_preflight_payload.get("windows_output_path"),
    )
    preflight_errors = mount_preflight_validation_errors(
        mount_preflight=mount_preflight_payload,
        require_mount_preflight=require_mount_preflight,
    )
    if preflight_errors:
        progress(output_dir, "blocked_mount_preflight", validation_errors=preflight_errors)
        return blocked_manifest(
            output_dir=output_dir,
            run_roots=run_roots,
            output_status="blocked_mount_preflight",
            validation_errors=preflight_errors,
            command_args=command_args,
            mount_preflight=mount_preflight_payload,
        )
    legacy_violations = forbidden_legacy_corpus_violations(run_roots)
    if legacy_violations:
        progress(output_dir, "blocked_forbidden_legacy_corpus", validation_errors=legacy_violations)
        return blocked_manifest(
            output_dir=output_dir,
            run_roots=run_roots,
            output_status="blocked_forbidden_legacy_corpus",
            validation_errors=legacy_violations,
            command_args=command_args,
            mount_preflight=mount_preflight_payload,
        )
    progress(output_dir, "grounding_start", phase_output_dir=str(ground_dir))
    grounding_manifest = build_audio_error_grounding_dataset(
        run_roots=run_roots,
        output_dir=ground_dir,
        window_sec=window_sec,
        hop_sec=hop_sec,
        n_fft=n_fft,
        hop_length=hop_length,
        map_max_bins=map_max_bins,
        map_max_frames=map_max_frames,
        max_waveform_samples=max_waveform_samples,
    )
    progress(
        output_dir,
        "grounding_done",
        windows_total=grounding_manifest.get("windows_total"),
        safe_for_training_count=grounding_manifest.get("safe_for_training_count"),
    )

    progress(output_dir, "provenance_audit_start", phase_output_dir=str(audit_dir))
    audit_manifest = audit_audio_error_dataset_provenance(
        dataset_dir=ground_dir,
        run_roots=run_roots,
        output_dir=audit_dir,
    )
    progress(
        output_dir,
        "provenance_audit_done",
        rows_audited=audit_manifest.get("rows_audited"),
        rows_repairable=audit_manifest.get("rows_repairable"),
    )

    progress(output_dir, "provenance_repair_start", phase_output_dir=str(repaired_dir))
    repair_report = repair_audio_error_dataset_provenance(
        dataset_dir=ground_dir,
        audit_dir=audit_dir,
        output_dir=repaired_dir,
    )
    repaired_validation = validate_audio_error_dataset_provenance_repair(repaired_dir)
    progress(
        output_dir,
        "provenance_repair_done",
        safe_for_training_count_after=repair_report.get("safe_for_training_count_after"),
        validation_ok=repaired_validation.get("ok"),
    )

    progress(output_dir, "training_views_start", phase_output_dir=str(views_dir))
    views_manifest = build_audio_error_training_views(
        dataset_dir=repaired_dir,
        output_dir=views_dir,
        run_roots=run_roots,
    )
    views_validation = validate_audio_error_training_views(views_dir)
    progress(
        output_dir,
        "training_views_done",
        source_run_rows_added=views_manifest.get("source_run_rows_added"),
        validation_ok=views_validation.get("ok"),
    )

    manifest: dict[str, Any] = {
        "schema": "minus_hardcase_audio_error_bridge.v1",
        "run_roots": [str(path) for path in run_roots],
        "run_root_count": len(run_roots),
        "grounding_dir": str(ground_dir),
        "audit_dir": str(audit_dir),
        "repaired_dir": str(repaired_dir),
        "views_dir": str(views_dir),
        "grounding_windows_total": int(grounding_manifest.get("windows_total") or 0),
        "grounding_safe_for_training_count": int(grounding_manifest.get("safe_for_training_count") or 0),
        "grounding_error_maps_written": int(grounding_manifest.get("error_maps_written") or 0),
        "audit_rows_audited": int(audit_manifest.get("rows_audited") or 0),
        "audit_rows_repairable": int(audit_manifest.get("rows_repairable") or 0),
        "repair_safe_for_training_count_after": int(repair_report.get("safe_for_training_count_after") or 0),
        "repair_validation_ok": repaired_validation.get("ok") is True,
        "training_views_validation_ok": views_validation.get("ok") is True,
        "source_run_rows_added": int(views_manifest.get("source_run_rows_added") or 0),
        "strict_negative_rows_found": int(views_manifest.get("strict_negative_rows_found") or 0),
        "strict_positive_rows_found": int(views_manifest.get("strict_positive_rows_found") or 0),
        "can_train_minus_critic": views_validation.get("can_train_minus_critic") is True,
        "can_train_vocal_critic": views_validation.get("can_train_vocal_critic") is True,
        "can_train_pair_critic": views_validation.get("can_train_pair_critic") is True,
        "can_train_overall_accept": views_validation.get("can_train_overall_accept") is True,
        "grounding_manifest": grounding_manifest,
        "audit_manifest": audit_manifest,
        "repair_report": repair_report,
        "repaired_validation": repaired_validation,
        "training_views_manifest": views_manifest,
        "training_views_validation": views_validation,
        "command_args": list(command_args or []),
        "next_step": "train_or_evaluate_minus_critic_on_approved_runtime" if views_validation.get("can_train_minus_critic") is True else "fix_runtime_roots_or_provenance_before_training",
        "presentation_status": "training_evidence_not_for_listen_pack",
        **mount_preflight_manifest_fields(mount_preflight_payload),
        **SAFETY_FLAGS,
    }
    errors = validation_errors_for(
        bridge_manifest=manifest,
        repaired_validation=repaired_validation,
        views_validation=views_validation,
    )
    manifest["validation_errors"] = errors
    manifest["output_status"] = "training_views_ready" if not errors else "blocked_validation_failed"
    write_json(output_dir / "bridge_manifest.json", manifest)
    write_summary(output_dir / "summary.md", manifest)
    progress(output_dir, "complete", output_status=manifest["output_status"], validation_errors=errors)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge minus hardcase runtime roots into audio-error grounding and training views.",
    )
    parser.add_argument("--run-root", action="append", required=True, help="Runtime root with candidate_pool/judge JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for bridge evidence artifacts.")
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--hop-sec", type=float, default=12.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--map-max-bins", type=int, default=128)
    parser.add_argument("--map-max-frames", type=int, default=256)
    parser.add_argument("--max-waveform-samples", type=int, default=4096)
    parser.add_argument("--mount-preflight-json", help="JSON evidence from mac_windows_mount_preflight.")
    parser.add_argument("--mac-windows-project-mount", help="Mac path to the mounted Windows project workspace.")
    parser.add_argument("--mac-output-dir", help="Mac-mounted path corresponding to --output-dir.")
    parser.add_argument("--windows-project-root", default=DEFAULT_WINDOWS_PROJECT_ROOT)
    parser.add_argument("--require-mount-preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = bridge_minus_hardcase_audio_error_views(
        run_roots=[Path(path) for path in args.run_root],
        output_dir=Path(args.output_dir),
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        map_max_bins=int(args.map_max_bins),
        map_max_frames=int(args.map_max_frames),
        max_waveform_samples=int(args.max_waveform_samples),
        mount_preflight_path=Path(args.mount_preflight_json) if args.mount_preflight_json else None,
        mac_windows_project_mount=args.mac_windows_project_mount,
        mac_output_path=args.mac_output_dir,
        windows_project_root=str(args.windows_project_root),
        require_mount_preflight=bool(args.require_mount_preflight),
        command_args=sys.argv[1:] if argv is None else list(argv),
    )
    print(json.dumps({"manifest_path": str(Path(args.output_dir) / "bridge_manifest.json"), "output_status": manifest["output_status"]}, indent=2))


if __name__ == "__main__":
    main()
