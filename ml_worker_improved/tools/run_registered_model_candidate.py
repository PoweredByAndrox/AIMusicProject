from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.model_registry import ModelRegistry, default_model_registry  # noqa: E402


CommandRunner = Callable[[list[str]], Any]
EXECUTION_MANIFEST_NAME = "registry_candidate_execution_manifest.json"
STDOUT_LOG_NAME = "registered_candidate_stdout.log"
STDERR_LOG_NAME = "registered_candidate_stderr.log"
PROGRESS_LOG_NAME = "registered_candidate_progress.jsonl"
VOCAL_STEM_ALIASES = {"vocal", "vocals", "voice"}
MINUS_STEM_ALIASES = {"minus", "instrumental", "instrumentals"}
START_TIME = time.monotonic()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def progress(path: Path, event: str, *, device: str, **fields: Any) -> None:
    uses_gpu = str(device).strip().lower() == "cuda"
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": (
            "gpu_registered_model_inference" if uses_gpu else "cpu_registered_model_inference"
        ),
        "gpu_inference_required": uses_gpu,
        "gpu_inference_skipped": not uses_gpu,
    }
    if not uses_gpu:
        payload["gpu_skip_reason"] = "registered_candidate_device_not_cuda"
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def output_artifact_count(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(1 for child in output_dir.iterdir() if child.is_file())


def expected_manifest_path(
    *,
    expected_manifest_template: str,
    source_audio_path: Path,
    output_dir: Path,
    listen_dir: Path | None,
    device: str,
) -> Path:
    return Path(
        expected_manifest_template.format(
            source_audio_path=str(source_audio_path),
            output_dir=str(output_dir),
            listen_dir=str(listen_dir) if listen_dir is not None else "",
            device=device,
        )
    )


def build_base_execution_manifest(
    *,
    model_id: str,
    source_audio_path: Path,
    output_dir: Path,
    device: str,
    command: Sequence[str],
    registry: ModelRegistry,
) -> dict[str, Any]:
    entry = registry.get(model_id)
    availability = registry.availability(entry)
    return {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/tools/run_registered_model_candidate.py",
            "intended_runtime": "docker ml-worker",
        },
        "contract": {
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_required": True,
            "expected_manifest_required": True,
            "per_track_hardcode_allowed": False,
        },
        "model": {
            "model_id": entry.model_id,
            "display_name": entry.display_name,
            "family": entry.family,
            "target_stems": list(entry.target_stems),
            "output_stems": list(entry.output_stems),
        },
        "inputs": {
            "source_audio_path": str(source_audio_path),
            "output_dir": str(output_dir),
            "device": device,
        },
        "availability": {
            "status": availability.status,
            "missing_paths": list(availability.missing_paths),
            "missing_python_modules": list(availability.missing_python_modules),
        },
        "command": list(command),
    }


def unavailable_reason(availability: dict[str, Any]) -> str:
    missing_paths = availability.get("missing_paths") or []
    missing_python_modules = availability.get("missing_python_modules") or []
    if missing_paths and missing_python_modules:
        return "missing_required_dependencies"
    if missing_python_modules:
        return "missing_required_python_modules"
    return "missing_required_paths"


def command_log_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "stdout_path": output_dir / STDOUT_LOG_NAME,
        "stderr_path": output_dir / STDERR_LOG_NAME,
    }


def log_manifest_entry(log_paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in log_paths.items()}


def normalize_target_stem(stem: str | None) -> str | None:
    value = str(stem or "").strip().lower()
    if not value:
        return None
    if value in VOCAL_STEM_ALIASES:
        return "vocal"
    if value in MINUS_STEM_ALIASES:
        return "minus"
    return value


def command_with_target_stem_runtime_flags(command: list[str], target_stem: str | None) -> list[str]:
    normalized_target = normalize_target_stem(target_stem)
    if normalized_target not in {"minus", "vocal"}:
        return command
    if "--no-melody-stem" in command:
        return command
    if not any(str(part).endswith("export_hq_stems_full.py") for part in command):
        return command
    return [*command, "--no-melody-stem"]


def target_stem_aliases(stem: str) -> tuple[str, ...]:
    normalized = normalize_target_stem(stem)
    if normalized == "vocal":
        return ("vocal", "vocals")
    if normalized == "minus":
        return ("minus", "instrumental")
    return (str(normalized),)


def stem_audio_path(entry: Any, *, manifest_dir: Path) -> Path:
    if isinstance(entry, dict):
        raw_path = entry.get("path") or entry.get("audio_path") or entry.get("file")
    else:
        raw_path = entry
    if not raw_path:
        raise ValueError("stem entry does not contain an audio path")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def find_manifest_stem(stems: dict[str, Any], stem: str) -> tuple[str, Any] | None:
    aliases = set(target_stem_aliases(stem))
    for name, entry in stems.items():
        if str(name).strip().lower() in aliases:
            return str(name), entry
    return None


def load_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"candidate manifest must be a JSON object: {path}")
    return payload


def materialize_minus_from_non_vocal_stems(
    *,
    stems: dict[str, Any],
    manifest_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_paths: list[tuple[str, Path]] = []
    for stem_name, stem_entry in stems.items():
        normalized_name = normalize_target_stem(str(stem_name))
        if normalized_name == "vocal" or normalized_name == "minus":
            continue
        source_path = stem_audio_path(stem_entry, manifest_dir=manifest_dir)
        if source_path.exists():
            source_paths.append((str(stem_name), source_path))
    if not source_paths:
        raise RuntimeError("minus target requires instrumental stem or non-vocal stems")

    summed_audio: np.ndarray | None = None
    expected_sample_rate: int | None = None
    expected_shape: tuple[int, ...] | None = None
    for _stem_name, source_path in source_paths:
        audio, sample_rate = sf.read(str(source_path), always_2d=True, dtype="float32")
        if expected_sample_rate is None:
            expected_sample_rate = int(sample_rate)
            expected_shape = tuple(audio.shape)
            summed_audio = np.zeros_like(audio, dtype=np.float32)
        if int(sample_rate) != expected_sample_rate:
            raise RuntimeError(f"sample rate mismatch while materializing minus: {source_path}")
        if tuple(audio.shape) != expected_shape:
            raise RuntimeError(f"audio shape mismatch while materializing minus: {source_path}")
        assert summed_audio is not None
        summed_audio += audio

    target_path = output_dir / "minus.flac"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    assert summed_audio is not None
    sf.write(str(target_path), summed_audio, int(expected_sample_rate or 0), subtype="PCM_24")
    return {
        "target_stem": "minus",
        "path": str(target_path),
        "source": "sum_non_vocal_stems",
        "source_stems": [stem_name for stem_name, _ in source_paths],
        "source_paths": [str(source_path) for _, source_path in source_paths],
    }


def ensure_target_stem_artifact(
    *,
    candidate_manifest_path: Path,
    target_stem: str | None,
    output_dir: Path,
) -> dict[str, Any] | None:
    normalized_target = normalize_target_stem(target_stem)
    if normalized_target is None:
        return None

    candidate_manifest = load_candidate_manifest(candidate_manifest_path)
    stems = candidate_manifest.setdefault("stems", {})
    if not isinstance(stems, dict):
        raise RuntimeError("candidate manifest stems must be an object")

    manifest_dir = candidate_manifest_path.parent
    existing_target = find_manifest_stem(stems, normalized_target)
    if existing_target is not None:
        matched_name, stem_entry = existing_target
        target_path = stem_audio_path(stem_entry, manifest_dir=manifest_dir)
        if not target_path.exists():
            raise FileNotFoundError(f"target stem audio is missing: {target_path}")
        artifact = {
            "target_stem": normalized_target,
            "path": str(target_path),
            "source": "existing_manifest_stem",
            "source_stems": [matched_name],
        }
    elif normalized_target == "minus":
        artifact = materialize_minus_from_non_vocal_stems(
            stems=stems,
            manifest_dir=manifest_dir,
            output_dir=output_dir,
        )
    else:
        raise RuntimeError(f"target stem is missing from candidate manifest: {normalized_target}")

    stems[normalized_target] = {
        "path": artifact["path"],
        "source": artifact["source"],
        "source_stems": list(artifact.get("source_stems") or []),
    }
    if normalized_target == "minus" and "instrumental" not in stems:
        stems["instrumental"] = dict(stems[normalized_target])
    candidate_manifest.update(
        {
            "target_stem": normalized_target,
            "target_stem_artifact": artifact,
            "delivery_allowed": False,
            "presentation_status": "diagnostic_candidate_not_for_listen_pack",
            "release_status": "not_release_ready",
            "release_ready": False,
        }
    )
    write_json(candidate_manifest_path, candidate_manifest)
    return artifact


def run_command_with_logs(command: list[str], log_paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    for path in log_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with log_paths["stdout_path"].open("w", encoding="utf-8") as stdout_handle:
        with log_paths["stderr_path"].open("w", encoding="utf-8") as stderr_handle:
            return subprocess.run(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=True,
            )


def run_registered_model_candidate(
    *,
    model_id: str,
    source_audio_path: Path,
    output_dir: Path,
    registry: ModelRegistry | None = None,
    command_runner: CommandRunner | None = None,
    device: str = "cuda",
    listen_dir: Path | None = None,
    fail_on_unavailable: bool = True,
    target_stem: str | None = None,
) -> dict[str, Any]:
    registry = registry or default_model_registry()
    entry = registry.get(model_id)
    source_audio_path = source_audio_path.resolve()
    output_dir = output_dir.resolve()
    listen_dir = listen_dir.resolve() if listen_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    command = entry.render_command(
        source_audio_path=str(source_audio_path),
        output_dir=str(output_dir),
        listen_dir=str(listen_dir) if listen_dir is not None else None,
        device=device,
    )
    command = command_with_target_stem_runtime_flags(command, target_stem)
    execution_manifest_path = output_dir / EXECUTION_MANIFEST_NAME
    progress_path = output_dir / PROGRESS_LOG_NAME
    if progress_path.exists():
        progress_path.unlink()
    candidate_manifest_path = expected_manifest_path(
        expected_manifest_template=entry.expected_manifest,
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        listen_dir=listen_dir,
        device=device,
    )
    manifest = build_base_execution_manifest(
        model_id=model_id,
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        device=device,
        command=command,
        registry=registry,
    )
    manifest["progress_jsonl"] = str(progress_path)
    progress(
        progress_path,
        "start",
        device=device,
        phase="input_validation",
        model_id=model_id,
        target_stem=normalize_target_stem(target_stem),
        source_audio_path=str(source_audio_path),
        output_dir=str(output_dir),
        execution_manifest_path=str(execution_manifest_path),
        expected_candidate_manifest_path=str(candidate_manifest_path),
        output_artifact_count=output_artifact_count(output_dir),
    )

    if not source_audio_path.exists():
        manifest.update(
            {
                "status": "failed",
                "reason": "source_audio_missing",
                "candidate_manifest": None,
            }
        )
        progress(
            progress_path,
            "failed",
            device=device,
            phase="input_validation",
            model_id=model_id,
            reason="source_audio_missing",
            source_audio_path=str(source_audio_path),
            output_artifact_count=output_artifact_count(output_dir),
        )
        write_json(execution_manifest_path, manifest)
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")

    progress(
        progress_path,
        "availability_checked",
        device=device,
        phase="availability",
        model_id=model_id,
        availability_status=manifest["availability"]["status"],
        missing_paths=list(manifest["availability"].get("missing_paths") or []),
        missing_python_modules=list(manifest["availability"].get("missing_python_modules") or []),
        output_artifact_count=output_artifact_count(output_dir),
    )
    if manifest["availability"]["status"] != "available":
        manifest.update(
            {
                "status": "unavailable",
                "reason": unavailable_reason(manifest["availability"]),
                "candidate_manifest": None,
            }
        )
        progress(
            progress_path,
            "failed",
            device=device,
            phase="availability",
            model_id=model_id,
            reason=manifest["reason"],
            output_artifact_count=output_artifact_count(output_dir),
        )
        write_json(execution_manifest_path, manifest)
        if fail_on_unavailable:
            raise RuntimeError(f"Registered model is unavailable: {model_id}")
        return manifest

    log_paths = command_log_paths(output_dir)
    if command_runner is None:
        manifest["logs"] = log_manifest_entry(log_paths)
    runner = command_runner or (lambda rendered_command: run_command_with_logs(rendered_command, log_paths))
    try:
        progress(
            progress_path,
            "command_start",
            device=device,
            phase="registered_model_inference",
            model_id=model_id,
            command=list(command),
            stdout_path=str(log_paths["stdout_path"]),
            stderr_path=str(log_paths["stderr_path"]),
            output_artifact_count=output_artifact_count(output_dir),
        )
        runner(command)
        progress(
            progress_path,
            "command_finish",
            device=device,
            phase="registered_model_inference",
            model_id=model_id,
            candidate_manifest_exists=candidate_manifest_path.exists(),
            output_artifact_count=output_artifact_count(output_dir),
        )
    except Exception as exc:
        if command_runner is not None:
            existing_log_paths = {
                name: path for name, path in log_paths.items() if path.exists()
            }
            if existing_log_paths:
                manifest["logs"] = log_manifest_entry(existing_log_paths)
        manifest.update(
            {
                "status": "failed",
                "reason": "execution_failed",
                "candidate_manifest": {
                    "path": str(candidate_manifest_path),
                    "exists": candidate_manifest_path.exists(),
                },
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        progress(
            progress_path,
            "failed",
            device=device,
            phase="registered_model_inference",
            model_id=model_id,
            reason="execution_failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            candidate_manifest_exists=candidate_manifest_path.exists(),
            output_artifact_count=output_artifact_count(output_dir),
        )
        write_json(execution_manifest_path, manifest)
        raise

    if not candidate_manifest_path.exists():
        manifest.update(
            {
                "status": "failed",
                "reason": "expected_manifest_missing",
                "candidate_manifest": {
                    "path": str(candidate_manifest_path),
                    "exists": False,
                },
            }
        )
        progress(
            progress_path,
            "failed",
            device=device,
            phase="candidate_manifest_validation",
            model_id=model_id,
            reason="expected_manifest_missing",
            expected_candidate_manifest_path=str(candidate_manifest_path),
            output_artifact_count=output_artifact_count(output_dir),
        )
        write_json(execution_manifest_path, manifest)
        raise RuntimeError(f"expected candidate manifest was not created: {candidate_manifest_path}")

    candidate_manifest_entry: dict[str, Any] = {
        "path": str(candidate_manifest_path),
        "exists": True,
    }
    try:
        progress(
            progress_path,
            "target_stem_materialization_start",
            device=device,
            phase="target_stem_materialization",
            model_id=model_id,
            target_stem=normalize_target_stem(target_stem),
            candidate_manifest_path=str(candidate_manifest_path),
            output_artifact_count=output_artifact_count(output_dir),
        )
        target_artifact = ensure_target_stem_artifact(
            candidate_manifest_path=candidate_manifest_path,
            target_stem=target_stem,
            output_dir=output_dir,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "reason": "target_stem_materialization_failed",
                "candidate_manifest": candidate_manifest_entry,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        progress(
            progress_path,
            "failed",
            device=device,
            phase="target_stem_materialization",
            model_id=model_id,
            reason="target_stem_materialization_failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            candidate_manifest_path=str(candidate_manifest_path),
            output_artifact_count=output_artifact_count(output_dir),
        )
        write_json(execution_manifest_path, manifest)
        raise
    if target_artifact is not None:
        candidate_manifest_entry.update(
            {
                "target_stem": str(target_artifact["target_stem"]),
                "target_audio_path": str(target_artifact["path"]),
                "target_source": str(target_artifact["source"]),
                "target_source_stems": list(target_artifact.get("source_stems") or []),
            }
        )
        manifest["inputs"]["target_stem"] = str(target_artifact["target_stem"])
    progress(
        progress_path,
        "target_stem_materialization_finish",
        device=device,
        phase="target_stem_materialization",
        model_id=model_id,
        target_stem=normalize_target_stem(target_stem),
        target_audio_path=str(target_artifact["path"]) if target_artifact is not None else None,
        output_artifact_count=output_artifact_count(output_dir),
    )

    manifest.update(
        {
            "status": "completed",
            "reason": "candidate_manifest_created",
            "candidate_manifest": candidate_manifest_entry,
        }
    )
    progress(
        progress_path,
        "finish",
        device=device,
        phase="execution_manifest_write",
        model_id=model_id,
        status="completed",
        reason="candidate_manifest_created",
        execution_manifest_path=str(execution_manifest_path),
        candidate_manifest_path=str(candidate_manifest_path),
        output_artifact_count=output_artifact_count(output_dir),
    )
    write_json(execution_manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one registered model candidate and verify its manifest.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    parser.add_argument("--target-stem")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_registered_model_candidate.py",
        entrypoint_path="/app/tools/run_registered_model_candidate.py",
    )
    args = build_arg_parser().parse_args(argv)
    registry = default_model_registry(runtime_root=args.runtime_root)
    manifest = run_registered_model_candidate(
        model_id=args.model_id,
        source_audio_path=args.source_audio_path,
        output_dir=args.output_dir,
        registry=registry,
        device=str(args.device),
        listen_dir=args.listen_dir,
        fail_on_unavailable=True,
        target_stem=args.target_stem,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
