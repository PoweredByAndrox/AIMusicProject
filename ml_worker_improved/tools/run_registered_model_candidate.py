from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    execution_manifest_path = output_dir / EXECUTION_MANIFEST_NAME
    manifest = build_base_execution_manifest(
        model_id=model_id,
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        device=device,
        command=command,
        registry=registry,
    )

    if not source_audio_path.exists():
        manifest.update(
            {
                "status": "failed",
                "reason": "source_audio_missing",
                "candidate_manifest": None,
            }
        )
        write_json(execution_manifest_path, manifest)
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")

    if manifest["availability"]["status"] != "available":
        manifest.update(
            {
                "status": "unavailable",
                "reason": unavailable_reason(manifest["availability"]),
                "candidate_manifest": None,
            }
        )
        write_json(execution_manifest_path, manifest)
        if fail_on_unavailable:
            raise RuntimeError(f"Registered model is unavailable: {model_id}")
        return manifest

    candidate_manifest_path = expected_manifest_path(
        expected_manifest_template=entry.expected_manifest,
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        listen_dir=listen_dir,
        device=device,
    )
    log_paths = command_log_paths(output_dir)
    if command_runner is None:
        manifest["logs"] = log_manifest_entry(log_paths)
    runner = command_runner or (lambda rendered_command: run_command_with_logs(rendered_command, log_paths))
    try:
        runner(command)
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
        write_json(execution_manifest_path, manifest)
        raise RuntimeError(f"expected candidate manifest was not created: {candidate_manifest_path}")

    manifest.update(
        {
            "status": "completed",
            "reason": "candidate_manifest_created",
            "candidate_manifest": {
                "path": str(candidate_manifest_path),
                "exists": True,
            },
        }
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
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
