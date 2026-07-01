#!/usr/bin/env python3
"""Run a metadata-only project smoke inside Apple container on macOS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mac_windows_mount_preflight import (  # noqa: E402
    DEFAULT_WINDOWS_PROJECT_ROOT,
    preflight_mac_windows_project_mount,
)


SCHEMA = "apple_container_project_smoke.v1"
MANIFEST_NAME = "manifest.json"
PROGRESS_NAME = "progress.jsonl"
STDOUT_NAME = "stdout.txt"
STDERR_NAME = "stderr.txt"
COMMAND_NAME = "cmd.json"
DEFAULT_CONTAINER_CLI = Path("/usr/local/bin/container")
DEFAULT_IMAGE = "docker.io/library/python:3.11-alpine"
DEFAULT_MOUNT_TARGET = "/work"
DEFAULT_WORKDIR = "/work/ml_worker_improved"
DEFAULT_TIMEOUT_SEC = 600
DEFAULT_TEST_SELECTORS = (
    "tests.test_docker_runtime_policy.TestDockerRuntimePolicy."
    "test_cli_runtime_contract_preserves_explicit_apple_container_runtime_marker",
    "tests.test_docker_runtime_policy.TestDockerRuntimePolicy."
    "test_cli_runtime_contract_reports_apple_container_on_macos_without_native_diagnostic",
    "tests.test_docker_runtime_policy.TestDockerRuntimePolicy."
    "test_shared_runtime_guard_rejects_macos_host_with_apple_container_command",
)
PRESENTATION_STATUS = "diagnostic_apple_container_project_smoke_only"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_progress(path: Path, phase: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "phase": phase,
        "time_unix": round(time.time(), 6),
        "expected_acceleration_mode": "external_only_container_metadata_test",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "Apple-container smoke runs metadata-only Python tests",
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def build_apple_container_command(
    *,
    project_root: Path,
    container_cli: Path,
    image: str,
    test_selectors: Sequence[str],
    mount_target: str = DEFAULT_MOUNT_TARGET,
    workdir: str = DEFAULT_WORKDIR,
) -> list[str]:
    return [
        str(container_cli),
        "run",
        "--rm",
        "--mount",
        f"type=bind,source={project_root},target={mount_target},readonly",
        "-e",
        "AIMUSIC_CONTAINER_RUNTIME=apple_container",
        "--workdir",
        workdir,
        image,
        "python",
        "-m",
        "unittest",
        *list(test_selectors),
    ]


def base_manifest(
    *,
    output_dir: Path,
    project_root: Path,
    container_cli: Path,
    image: str,
    test_selectors: Sequence[str],
    mount_target: str = DEFAULT_MOUNT_TARGET,
    workdir: str = DEFAULT_WORKDIR,
    command: Sequence[str] = (),
    mount_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA,
        "mode": "apple_container_project_smoke",
        "project_root": str(project_root),
        "container_cli": str(container_cli),
        "container_runtime": "apple_container",
        "runtime_environment": "apple_container_project_smoke",
        "image": image,
        "mount_target": mount_target,
        "workdir": workdir,
        "test_selectors": list(test_selectors),
        "command": list(command),
        "manifest_path": str(output_dir / MANIFEST_NAME),
        "progress_jsonl": str(output_dir / PROGRESS_NAME),
        "stdout_path": str(output_dir / STDOUT_NAME),
        "stderr_path": str(output_dir / STDERR_NAME),
        "command_json": str(output_dir / COMMAND_NAME),
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": PRESENTATION_STATUS,
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "expected_acceleration_mode": "external_only_container_metadata_test",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "Apple-container smoke runs metadata-only Python tests",
        "audio_generated": False,
        "listen_pack": {},
    }
    if mount_preflight is not None:
        manifest["mount_preflight"] = mount_preflight
        manifest["mac_windows_project_mount"] = str(mount_preflight.get("mac_windows_project_mount") or "")
        manifest["windows_output_path"] = str(mount_preflight.get("windows_output_path") or "")
    return manifest


def write_terminal_outputs(output_dir: Path, *, stdout: str = "", stderr: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / STDOUT_NAME).write_text(stdout, encoding="utf-8")
    (output_dir / STDERR_NAME).write_text(stderr, encoding="utf-8")


def blocked_manifest(
    *,
    output_dir: Path,
    project_root: Path,
    container_cli: Path,
    image: str,
    test_selectors: Sequence[str],
    blocked_reason: str,
    returncode: int,
    mount_target: str = DEFAULT_MOUNT_TARGET,
    workdir: str = DEFAULT_WORKDIR,
    command: Sequence[str] = (),
    stderr: str = "",
    mount_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        **base_manifest(
            output_dir=output_dir,
            project_root=project_root,
            container_cli=container_cli,
            image=image,
            test_selectors=test_selectors,
            mount_target=mount_target,
            workdir=workdir,
            command=command,
            mount_preflight=mount_preflight,
        ),
        "status": "blocked",
        "blocked_reason": blocked_reason,
        "returncode": int(returncode),
    }
    write_terminal_outputs(output_dir, stderr=stderr)
    write_json(output_dir / COMMAND_NAME, {"command": list(command)})
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_progress(output_dir / PROGRESS_NAME, "finish", status="blocked", blocked_reason=blocked_reason)
    return manifest


def run_apple_container_project_smoke(
    *,
    project_root: Path,
    output_dir: Path,
    container_cli: Path = DEFAULT_CONTAINER_CLI,
    image: str = DEFAULT_IMAGE,
    test_selectors: Sequence[str] = DEFAULT_TEST_SELECTORS,
    mac_windows_project_mount: Path | None = None,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
    mount_target: str = DEFAULT_MOUNT_TARGET,
    workdir: str = DEFAULT_WORKDIR,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    mount_preflight = preflight_mac_windows_project_mount(
        mac_windows_project_mount=mac_windows_project_mount,
        output_path=output_dir,
        windows_project_root=windows_project_root,
    )
    if mount_preflight.get("preflight_status") != "passed":
        manifest = {
            **base_manifest(
                output_dir=output_dir,
                project_root=project_root,
                container_cli=container_cli,
                image=image,
                test_selectors=test_selectors,
                mount_target=mount_target,
                workdir=workdir,
                mount_preflight=mount_preflight,
            ),
            "status": "blocked",
            "blocked_reason": str(mount_preflight.get("blocked_reason") or ""),
            "returncode": 78,
        }
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    started = time.monotonic()
    write_progress(progress_path, "start", project_root=str(project_root), image=image)

    command = build_apple_container_command(
        project_root=project_root,
        container_cli=container_cli,
        image=image,
        test_selectors=test_selectors,
        mount_target=mount_target,
        workdir=workdir,
    )
    if not project_root.exists():
        return blocked_manifest(
            output_dir=output_dir,
            project_root=project_root,
            container_cli=container_cli,
            image=image,
            test_selectors=test_selectors,
            blocked_reason="project_root_missing",
            returncode=66,
            mount_target=mount_target,
            workdir=workdir,
            command=command,
            mount_preflight=mount_preflight,
        )
    if not container_cli.exists():
        return blocked_manifest(
            output_dir=output_dir,
            project_root=project_root,
            container_cli=container_cli,
            image=image,
            test_selectors=test_selectors,
            blocked_reason="apple_container_cli_missing",
            returncode=127,
            mount_target=mount_target,
            workdir=workdir,
            command=command,
            mount_preflight=mount_preflight,
        )

    write_json(output_dir / COMMAND_NAME, {"command": command})
    write_progress(progress_path, "container_command_start", command=command)
    try:
        completed = runner(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = int(completed.returncode)
    except FileNotFoundError as exc:
        return blocked_manifest(
            output_dir=output_dir,
            project_root=project_root,
            container_cli=container_cli,
            image=image,
            test_selectors=test_selectors,
            blocked_reason="apple_container_cli_unavailable",
            returncode=127,
            mount_target=mount_target,
            workdir=workdir,
            command=command,
            stderr=str(exc),
            mount_preflight=mount_preflight,
        )
    except OSError as exc:
        return blocked_manifest(
            output_dir=output_dir,
            project_root=project_root,
            container_cli=container_cli,
            image=image,
            test_selectors=test_selectors,
            blocked_reason="apple_container_cli_unavailable",
            returncode=126,
            mount_target=mount_target,
            workdir=workdir,
            command=command,
            stderr=f"{type(exc).__name__}: {exc}",
            mount_preflight=mount_preflight,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = (exc.stderr or "").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        returncode = 124

    write_terminal_outputs(output_dir, stdout=stdout, stderr=stderr)
    status = "passed" if returncode == 0 else "failed"
    manifest = {
        **base_manifest(
            output_dir=output_dir,
            project_root=project_root,
            container_cli=container_cli,
            image=image,
            test_selectors=test_selectors,
            mount_target=mount_target,
            workdir=workdir,
            command=command,
            mount_preflight=mount_preflight,
        ),
        "status": status,
        "delivery_status": "diagnostic_only" if status == "passed" else "blocked",
        "returncode": returncode,
        "elapsed_sec": round(time.monotonic() - started, 6),
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_progress(progress_path, "finish", status=status, returncode=returncode, manifest_path=str(output_dir / MANIFEST_NAME))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container-cli", type=Path, default=DEFAULT_CONTAINER_CLI)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--mac-windows-project-mount", type=Path)
    parser.add_argument("--windows-project-root", default=DEFAULT_WINDOWS_PROJECT_ROOT)
    parser.add_argument("--mount-target", default=DEFAULT_MOUNT_TARGET)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--test-selector", action="append", dest="test_selectors")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_apple_container_project_smoke(
        project_root=args.project_root,
        output_dir=args.output_dir,
        container_cli=args.container_cli,
        image=args.image,
        test_selectors=tuple(args.test_selectors or DEFAULT_TEST_SELECTORS),
        mac_windows_project_mount=args.mac_windows_project_mount,
        windows_project_root=args.windows_project_root,
        mount_target=args.mount_target,
        workdir=args.workdir,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({"manifest_path": manifest["manifest_path"], "status": manifest["status"]}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if manifest.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
