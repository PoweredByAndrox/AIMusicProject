from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.no_reference_candidate_job_registry import (  # noqa: E402
    CandidateJobConfig,
    build_candidate_job_registry,
    get_candidate_job_config,
    registered_model_device,
)
from tools.normalize_no_reference_candidate_audio import normalize_candidate_audio_to_input_shape  # noqa: E402


CommandRunner = Callable[[list[str], int], None]
DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00Z"
SAFETY_FLAGS = {
    "codex_required_at_runtime": False,
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "separator_training_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def audio_info(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def validate_input_audio(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "reason": "input_audio_missing", "path": str(path)}
    if path.stat().st_size <= 0:
        return {"ok": False, "reason": "input_audio_empty", "path": str(path)}
    try:
        return {"ok": True, **audio_info(path)}
    except Exception as exc:  # pragma: no cover - defensive corrupt input path.
        return {"ok": False, "reason": f"input_audio_unreadable:{type(exc).__name__}", "path": str(path)}


def render_template(value: str, replacements: dict[str, str]) -> str:
    rendered = value
    for key, replacement in replacements.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def render_command(config: CandidateJobConfig, replacements: dict[str, str]) -> list[str]:
    return [render_template(part, replacements) for part in config.command_template]


def default_command_runner(command: list[str], timeout_sec: int) -> None:
    subprocess.run(command, check=True, timeout=int(timeout_sec))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def stem_path_from_manifest_entry(entry: Any, manifest_dir: Path) -> str:
    if not isinstance(entry, dict):
        return ""
    raw_path = str(entry.get("path") or entry.get("audio_path") or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path)


def paths_from_registered_execution_manifest(registered_output_dir: Path, target_mode: str) -> dict[str, str]:
    execution_manifest = registered_output_dir / "registry_candidate_execution_manifest.json"
    if not execution_manifest.exists():
        return {"vocal_path": "", "minus_path": ""}
    payload = load_json(execution_manifest)
    candidate_manifest = payload.get("candidate_manifest") if isinstance(payload.get("candidate_manifest"), dict) else {}
    output_paths = {"vocal_path": "", "minus_path": ""}
    target_audio_path = str(candidate_manifest.get("target_audio_path") or "").strip()
    if target_audio_path and target_mode == "vocal":
        output_paths["vocal_path"] = target_audio_path
    if target_audio_path and target_mode == "minus":
        output_paths["minus_path"] = target_audio_path
    manifest_path = Path(str(candidate_manifest.get("path") or ""))
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        stems = manifest.get("stems") if isinstance(manifest.get("stems"), dict) else {}
        manifest_dir = manifest_path.parent
        if not output_paths["vocal_path"]:
            output_paths["vocal_path"] = stem_path_from_manifest_entry(stems.get("vocal") or stems.get("vocals"), manifest_dir)
        if not output_paths["minus_path"]:
            output_paths["minus_path"] = stem_path_from_manifest_entry(stems.get("minus") or stems.get("instrumental"), manifest_dir)
    return output_paths


def expected_output_paths(config: CandidateJobConfig, output_dir: Path, registered_output_dir: Path) -> dict[str, str]:
    replacements = {
        "output_dir": str(output_dir),
        "registered_output_dir": str(registered_output_dir),
    }
    paths = {
        "vocal_path": render_template(str(config.expected_outputs.get("vocal_path") or ""), replacements),
        "minus_path": render_template(str(config.expected_outputs.get("minus_path") or ""), replacements),
    }
    registered_paths = paths_from_registered_execution_manifest(registered_output_dir, config.target_mode)
    for key, value in registered_paths.items():
        if value and not paths[key]:
            paths[key] = value
    if config.target_mode == "vocal":
        paths["minus_path"] = ""
    if config.target_mode == "minus":
        paths["vocal_path"] = ""
    return paths


def required_output_keys(target_mode: str) -> tuple[str, ...]:
    if target_mode == "vocal":
        return ("vocal_path",)
    if target_mode == "minus":
        return ("minus_path",)
    return ("vocal_path", "minus_path")


def validate_output_paths(paths: dict[str, str], target_mode: str) -> list[str]:
    missing: list[str] = []
    for key in required_output_keys(target_mode):
        path_text = str(paths.get(key) or "").strip()
        if not path_text or not Path(path_text).exists():
            missing.append(key)
    return missing


def validate_audio_shape(input_audio: Path, paths: dict[str, str], target_mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    input_info = audio_info(input_audio)
    output_info: dict[str, Any] = {}
    blockers: list[str] = []
    for key in required_output_keys(target_mode):
        path_text = str(paths.get(key) or "").strip()
        if not path_text:
            continue
        try:
            info = audio_info(Path(path_text))
        except Exception as exc:
            blockers.append(f"{key}_unreadable:{type(exc).__name__}")
            continue
        output_info[key] = info
        if int(info["sample_rate"]) != int(input_info["sample_rate"]):
            blockers.append(f"{key}_sample_rate_mismatch")
        duration_delta = abs(float(info["duration_sec"]) - float(input_info["duration_sec"]))
        if duration_delta > max(0.25, float(input_info["duration_sec"]) * 0.02):
            blockers.append(f"{key}_duration_mismatch")
        if int(info["channels"]) <= 0:
            blockers.append(f"{key}_invalid_channels")
    return not blockers, blockers, {"input": input_info, "outputs": output_info}


def normalized_output_path(output_dir: Path, raw_path: Path, stem_name: str) -> Path:
    suffix = raw_path.suffix if raw_path.suffix else ".wav"
    return output_dir / "normalized" / f"{stem_name}{suffix}"


def primary_output_info(paths: dict[str, str], target_mode: str) -> dict[str, Any]:
    for key in required_output_keys(target_mode):
        path_text = str(paths.get(key) or "").strip()
        if path_text and Path(path_text).exists():
            return audio_info(Path(path_text))
    return {"sample_rate": 0, "channels": 0, "duration_sec": 0.0, "frames": 0}


def attach_shape_contract(
    manifest: dict[str, Any],
    *,
    input_audio: Path,
    raw_paths: dict[str, str],
    normalized_paths: dict[str, str],
    target_mode: str,
    applied: bool,
    reason: str,
    reports: dict[str, dict[str, Any]] | None = None,
) -> None:
    input_info = audio_info(input_audio)
    raw_info = primary_output_info(raw_paths, target_mode)
    normalized_info = primary_output_info(normalized_paths, target_mode)
    manifest.update(
        {
            "input_sample_rate": int(input_info["sample_rate"]),
            "input_channels": int(input_info["channels"]),
            "input_duration_sec": float(input_info["duration_sec"]),
            "raw_output_sample_rate": int(raw_info.get("sample_rate") or 0),
            "raw_output_channels": int(raw_info.get("channels") or 0),
            "raw_output_duration_sec": float(raw_info.get("duration_sec") or 0.0),
            "normalized_output_sample_rate": int(normalized_info.get("sample_rate") or 0),
            "normalized_output_channels": int(normalized_info.get("channels") or 0),
            "normalized_output_duration_sec": float(normalized_info.get("duration_sec") or 0.0),
            "shape_normalization_applied": bool(applied),
            "shape_normalization_reason": str(reason),
            "raw_vocal_path": str(raw_paths.get("vocal_path") or ""),
            "raw_minus_path": str(raw_paths.get("minus_path") or ""),
            "normalized_vocal_path": str(normalized_paths.get("vocal_path") or ""),
            "normalized_minus_path": str(normalized_paths.get("minus_path") or ""),
            "shape_normalization_outputs": reports or {},
            "original_files_overwritten": False,
        }
    )


def base_manifest(
    *,
    input_audio: Path,
    output_dir: Path,
    job_manifest_json: Path,
    job_progress_jsonl: Path,
    config: CandidateJobConfig,
    command: list[str],
    timeout_sec: int,
) -> dict[str, Any]:
    return {
        "created_at": DETERMINISTIC_CREATED_AT,
        "input_audio_path": str(input_audio),
        "output_dir": str(output_dir),
        "job_manifest_path": str(job_manifest_json),
        "job_progress_path": str(job_progress_jsonl),
        "candidate_config_id": config.candidate_config_id,
        "route_id": config.route_id,
        "route_family": config.route_family,
        "candidate_family": config.candidate_family,
        "target_mode": config.target_mode,
        "source_model_id": config.source_model_id,
        "source_tool": config.source_tool,
        "entrypoint_type": config.entrypoint_type,
        "source_config_hash": config.source_config_hash,
        "can_run_without_download": bool(config.can_run_without_download),
        "can_run_without_registry_edit": bool(config.can_run_without_registry_edit),
        "forbidden_reason": config.forbidden_reason,
        "command": command,
        "timeout_sec": int(timeout_sec),
        "max_retries": int(config.max_retries),
        "output_paths": {"vocal_path": "", "minus_path": ""},
        "generation_status": "failed_runtime_error",
        "skip_reason": "",
        "audio_rendered": False,
        "input_sample_rate": 0,
        "input_channels": 0,
        "input_duration_sec": 0.0,
        "raw_output_sample_rate": 0,
        "raw_output_channels": 0,
        "raw_output_duration_sec": 0.0,
        "normalized_output_sample_rate": 0,
        "normalized_output_channels": 0,
        "normalized_output_duration_sec": 0.0,
        "shape_normalization_applied": False,
        "shape_normalization_reason": "",
        "raw_vocal_path": "",
        "raw_minus_path": "",
        "normalized_vocal_path": "",
        "normalized_minus_path": "",
        "shape_normalization_outputs": {},
        "original_files_overwritten": False,
        **SAFETY_FLAGS,
    }


def run_local_separator_candidate_job(
    *,
    input_audio: Path,
    candidate_config_id: str,
    output_dir: Path,
    job_manifest_json: Path,
    job_progress_jsonl: Path,
    registry_configs: Iterable[CandidateJobConfig] | None = None,
    profile: str | None = None,
    timeout_sec: int | None = None,
    command_runner: CommandRunner | None = None,
    allow_shape_normalization: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if job_progress_jsonl.exists():
        job_progress_jsonl.unlink()
    config = get_candidate_job_config(candidate_config_id, registry_configs=registry_configs, profile=profile)
    registered_output_dir = output_dir / "registered"
    replacements = {
        "input_audio": str(input_audio),
        "output_dir": str(output_dir),
        "registered_output_dir": str(registered_output_dir),
        "candidate_config_id": config.candidate_config_id,
        "source_model_id": config.source_model_id,
        "device": registered_model_device(),
        "target_stem": config.target_stem or config.target_mode,
        "job_manifest_json": str(job_manifest_json),
        "job_progress_jsonl": str(job_progress_jsonl),
    }
    command = render_command(config, replacements)
    effective_timeout = int(timeout_sec if timeout_sec is not None else config.timeout_sec)
    start = time.monotonic()
    manifest = base_manifest(
        input_audio=input_audio,
        output_dir=output_dir,
        job_manifest_json=job_manifest_json,
        job_progress_jsonl=job_progress_jsonl,
        config=config,
        command=command,
        timeout_sec=effective_timeout,
    )
    append_jsonl(job_progress_jsonl, {"phase": "start", "status": "started", "candidate_config_id": config.candidate_config_id})
    input_validation = validate_input_audio(input_audio)
    append_jsonl(job_progress_jsonl, {"phase": "validate_input_audio", "status": "completed" if input_validation["ok"] else "failed", **input_validation})
    if not input_validation["ok"]:
        manifest.update({"generation_status": "failed_invalid_audio_shape_unrecoverable", "skip_reason": str(input_validation["reason"])})
        write_json(job_manifest_json, manifest)
        return manifest
    if config.entrypoint_type == "unavailable":
        manifest.update({"generation_status": "skipped_unavailable_entrypoint", "skip_reason": "unavailable_entrypoint"})
        append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "skipped_unavailable_entrypoint"})
        write_json(job_manifest_json, manifest)
        return manifest
    runner = command_runner or default_command_runner
    try:
        append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "started", "gpu_required": config.gpu_required})
        runner(command, effective_timeout)
    except subprocess.TimeoutExpired as exc:
        output_paths = expected_output_paths(config, output_dir, registered_output_dir)
        manifest["output_paths"] = output_paths
        missing = validate_output_paths(output_paths, config.target_mode)
        present_count = len(required_output_keys(config.target_mode)) - len(missing)
        elapsed_sec = round(time.monotonic() - start, 6)
        if present_count <= 0:
            manifest.update(
                {
                    "generation_status": "failed_timeout_before_any_output",
                    "skip_reason": "failed_timeout_before_any_output",
                    "elapsed_sec": elapsed_sec,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "failed_timeout_before_any_output"})
            write_json(job_manifest_json, manifest)
            return manifest
        if missing:
            manifest.update(
                {
                    "generation_status": "partial_output_timeout",
                    "skip_reason": "partial_output_timeout",
                    "missing_outputs": missing,
                    "elapsed_sec": elapsed_sec,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "partial_output_timeout", "missing_outputs": missing})
            write_json(job_manifest_json, manifest)
            return manifest
        shape_ok, shape_blockers, shape_report = validate_audio_shape(input_audio, output_paths, config.target_mode)
        manifest["audio_shape"] = shape_report
        if not shape_ok:
            attach_shape_contract(
                manifest,
                input_audio=input_audio,
                raw_paths=output_paths,
                normalized_paths=output_paths,
                target_mode=config.target_mode,
                applied=False,
                reason="late_timeout_shape_invalid:" + ",".join(shape_blockers),
            )
            manifest.update(
                {
                    "generation_status": "partial_output_timeout",
                    "skip_reason": "partial_output_timeout",
                    "shape_blockers": shape_blockers,
                    "elapsed_sec": elapsed_sec,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "partial_output_timeout", "shape_blockers": shape_blockers})
            write_json(job_manifest_json, manifest)
            return manifest
        attach_shape_contract(
            manifest,
            input_audio=input_audio,
            raw_paths=output_paths,
            normalized_paths=output_paths,
            target_mode=config.target_mode,
            applied=False,
            reason="shape_already_valid",
        )
        manifest.update(
            {
                "generation_status": "generated_with_late_timeout",
                "skip_reason": "",
                "audio_rendered": True,
                "elapsed_sec": elapsed_sec,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "generated_with_late_timeout"})
        append_jsonl(job_progress_jsonl, {"phase": "validate_outputs", "status": "completed_after_late_timeout"})
        append_jsonl(job_progress_jsonl, {"phase": "complete", "status": "generated_with_late_timeout"})
        write_json(job_manifest_json, manifest)
        return manifest
    except Exception as exc:
        manifest.update({"generation_status": "failed_runtime_error", "skip_reason": "failed_runtime_error", "error": {"type": type(exc).__name__, "message": str(exc)}})
        append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "failed_runtime_error"})
        write_json(job_manifest_json, manifest)
        return manifest
    append_jsonl(job_progress_jsonl, {"phase": "execute_separator_command", "status": "completed"})
    output_paths = expected_output_paths(config, output_dir, registered_output_dir)
    manifest["output_paths"] = output_paths
    raw_output_paths = dict(output_paths)
    missing = validate_output_paths(output_paths, config.target_mode)
    if missing:
        manifest.update({"generation_status": "failed_missing_output", "skip_reason": "failed_missing_output", "missing_outputs": missing})
        append_jsonl(job_progress_jsonl, {"phase": "validate_outputs", "status": "failed_missing_output", "missing_outputs": missing})
        write_json(job_manifest_json, manifest)
        return manifest
    shape_ok, shape_blockers, shape_report = validate_audio_shape(input_audio, output_paths, config.target_mode)
    manifest["audio_shape"] = shape_report
    if not shape_ok:
        if not allow_shape_normalization:
            attach_shape_contract(
                manifest,
                input_audio=input_audio,
                raw_paths=raw_output_paths,
                normalized_paths=raw_output_paths,
                target_mode=config.target_mode,
                applied=False,
                reason="shape_normalization_disabled:" + ",".join(shape_blockers),
            )
            manifest.update(
                {
                    "generation_status": "failed_invalid_audio_shape_unrecoverable",
                    "skip_reason": "failed_invalid_audio_shape_unrecoverable",
                    "shape_blockers": shape_blockers,
                }
            )
            append_jsonl(job_progress_jsonl, {"phase": "validate_outputs", "status": "failed_invalid_audio_shape_unrecoverable", "shape_blockers": shape_blockers})
            write_json(job_manifest_json, manifest)
            return manifest
        append_jsonl(job_progress_jsonl, {"phase": "shape_normalization", "status": "started", "shape_blockers": shape_blockers, "gpu_inference_skipped": True, "reason": "cpu_shape_only_resample_channel_duration_alignment"})
        normalized_paths = dict(output_paths)
        reports: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        try:
            for key in required_output_keys(config.target_mode):
                raw_path_text = str(output_paths.get(key) or "").strip()
                if not raw_path_text:
                    continue
                stem_name = "vocals" if key == "vocal_path" else "minus"
                report = normalize_candidate_audio_to_input_shape(
                    input_audio_path=input_audio,
                    raw_output_path=Path(raw_path_text),
                    normalized_output_path=normalized_output_path(output_dir, Path(raw_path_text), stem_name),
                )
                reports[key] = report
                normalized_paths[key] = str(report["normalized_output_path"])
                reasons.append(str(report.get("shape_normalization_reason") or ""))
        except Exception as exc:
            attach_shape_contract(
                manifest,
                input_audio=input_audio,
                raw_paths=raw_output_paths,
                normalized_paths=raw_output_paths,
                target_mode=config.target_mode,
                applied=False,
                reason=f"shape_normalization_failed:{type(exc).__name__}",
                reports=reports,
            )
            manifest.update(
                {
                    "generation_status": "failed_invalid_audio_shape_unrecoverable",
                    "skip_reason": "failed_invalid_audio_shape_unrecoverable",
                    "shape_blockers": shape_blockers,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            append_jsonl(job_progress_jsonl, {"phase": "shape_normalization", "status": "failed_invalid_audio_shape_unrecoverable", "error_type": type(exc).__name__})
            write_json(job_manifest_json, manifest)
            return manifest
        normalized_shape_ok, normalized_blockers, normalized_shape_report = validate_audio_shape(input_audio, normalized_paths, config.target_mode)
        manifest["audio_shape"] = normalized_shape_report
        if not normalized_shape_ok:
            attach_shape_contract(
                manifest,
                input_audio=input_audio,
                raw_paths=raw_output_paths,
                normalized_paths=normalized_paths,
                target_mode=config.target_mode,
                applied=any(bool(report.get("shape_normalization_applied")) for report in reports.values()),
                reason="shape_normalization_failed:" + ",".join(normalized_blockers),
                reports=reports,
            )
            manifest.update(
                {
                    "generation_status": "failed_invalid_audio_shape_unrecoverable",
                    "skip_reason": "failed_invalid_audio_shape_unrecoverable",
                    "shape_blockers": normalized_blockers,
                }
            )
            append_jsonl(job_progress_jsonl, {"phase": "shape_normalization", "status": "failed_invalid_audio_shape_unrecoverable", "shape_blockers": normalized_blockers})
            write_json(job_manifest_json, manifest)
            return manifest
        output_paths = normalized_paths
        manifest["output_paths"] = output_paths
        attach_shape_contract(
            manifest,
            input_audio=input_audio,
            raw_paths=raw_output_paths,
            normalized_paths=normalized_paths,
            target_mode=config.target_mode,
            applied=any(bool(report.get("shape_normalization_applied")) for report in reports.values()),
            reason=",".join(sorted({reason for reason in reasons if reason})),
            reports=reports,
        )
        append_jsonl(job_progress_jsonl, {"phase": "shape_normalization", "status": "completed", "normalized_outputs": normalized_paths})
        generation_status = "generated_with_shape_normalization"
    else:
        attach_shape_contract(
            manifest,
            input_audio=input_audio,
            raw_paths=raw_output_paths,
            normalized_paths=raw_output_paths,
            target_mode=config.target_mode,
            applied=False,
            reason="shape_already_valid",
        )
        generation_status = "generated"
    manifest.update(
        {
            "generation_status": generation_status,
            "skip_reason": "",
            "audio_rendered": True,
            "elapsed_sec": round(time.monotonic() - start, 6),
        }
    )
    append_jsonl(job_progress_jsonl, {"phase": "validate_outputs", "status": "completed"})
    append_jsonl(job_progress_jsonl, {"phase": "complete", "status": generation_status})
    write_json(job_manifest_json, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one local no-reference separator candidate job.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--candidate-config-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-manifest-json", type=Path, required=True)
    parser.add_argument("--job-progress-jsonl", type=Path, required=True)
    parser.add_argument("--profile", choices=("safe", "balanced", "exploration"), default="safe")
    parser.add_argument("--timeout-sec", type=int)
    parser.add_argument("--allow-shape-normalization", type=parse_bool, default=True)
    args = parser.parse_args(argv)
    configs = build_candidate_job_registry(profile=args.profile)
    manifest = run_local_separator_candidate_job(
        input_audio=args.input_audio,
        candidate_config_id=args.candidate_config_id,
        output_dir=args.output_dir,
        job_manifest_json=args.job_manifest_json,
        job_progress_jsonl=args.job_progress_jsonl,
        registry_configs=configs,
        profile=args.profile,
        timeout_sec=args.timeout_sec,
        allow_shape_normalization=args.allow_shape_normalization,
    )
    print(json.dumps({"generation_status": manifest["generation_status"], "job_manifest_json": str(args.job_manifest_json), "release_ready": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
