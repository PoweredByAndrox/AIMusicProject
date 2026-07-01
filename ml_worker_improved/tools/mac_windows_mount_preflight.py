"""Fail-closed Mac-to-Windows project mount preflight helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "mac_windows_project_mount_preflight.v1"
DEFAULT_WINDOWS_PROJECT_ROOT = r"D:\AIMusicProject"
ALLOWED_PROJECT_ARTIFACT_ROOTS = ("artifacts", "lab_artifacts", "artifact_rag")
PROBE_DIR_NAME = ".aimusic_mount_probe"


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except FileNotFoundError:
        return path.expanduser().absolute()


def _blocked(
    *,
    reason: str,
    mac_windows_project_mount: Path | None,
    output_path: Path,
    windows_project_root: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "preflight_status": "blocked",
        "blocked_reason": reason,
        "can_write_route_artifacts": False,
        "mac_windows_project_mount": str(mac_windows_project_mount or ""),
        "mac_output_path": str(_safe_resolve(output_path)),
        "windows_project_root": str(windows_project_root),
        "windows_output_path": "",
        "allowed_project_artifact_roots": list(ALLOWED_PROJECT_ARTIFACT_ROOTS),
        "probe_status": "not_run",
        "probe_path": "",
    }
    if extra:
        payload.update(extra)
    return payload


def _windows_join(root: str, parts: Iterable[str]) -> str:
    normalized = root.rstrip("\\/")
    suffix = "\\".join(str(part).strip("\\/") for part in parts if str(part).strip("\\/"))
    return f"{normalized}\\{suffix}" if suffix else normalized


def preflight_mac_windows_project_mount(
    *,
    mac_windows_project_mount: Path | str | None,
    output_path: Path | str,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
) -> dict[str, Any]:
    output = _safe_resolve(Path(output_path))
    if mac_windows_project_mount is None or str(mac_windows_project_mount).strip() == "":
        return _blocked(
            reason="mac_windows_project_mount_missing",
            mac_windows_project_mount=None,
            output_path=output,
            windows_project_root=windows_project_root,
        )

    mount = _safe_resolve(Path(mac_windows_project_mount))
    if not mount.exists() or not mount.is_dir():
        return _blocked(
            reason="mac_windows_project_mount_missing",
            mac_windows_project_mount=mount,
            output_path=output,
            windows_project_root=windows_project_root,
        )

    try:
        relative_output = output.relative_to(mount)
    except ValueError:
        return _blocked(
            reason="mac_output_path_outside_windows_project_mount",
            mac_windows_project_mount=mount,
            output_path=output,
            windows_project_root=windows_project_root,
        )

    if not relative_output.parts or relative_output.parts[0] not in ALLOWED_PROJECT_ARTIFACT_ROOTS:
        return _blocked(
            reason="mac_output_path_not_in_allowed_project_artifact_root",
            mac_windows_project_mount=mount,
            output_path=output,
            windows_project_root=windows_project_root,
            extra={"relative_output_path": relative_output.as_posix()},
        )

    probe_dir = mount / PROBE_DIR_NAME
    probe_path = probe_dir / f"probe_{os.getpid()}_{time.time_ns()}.txt"
    probe_payload = f"{SCHEMA}:{time.time_ns()}"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(probe_payload, encoding="utf-8")
        if probe_path.read_text(encoding="utf-8") != probe_payload:
            raise OSError("mount probe readback mismatch")
        probe_path.unlink()
        try:
            probe_dir.rmdir()
        except OSError:
            pass
    except OSError as exc:
        return _blocked(
            reason="mac_windows_project_mount_probe_failed",
            mac_windows_project_mount=mount,
            output_path=output,
            windows_project_root=windows_project_root,
            extra={"probe_status": "failed", "probe_path": str(probe_path), "probe_error": str(exc)},
        )

    return {
        "schema": SCHEMA,
        "preflight_status": "passed",
        "blocked_reason": "",
        "can_write_route_artifacts": True,
        "mac_windows_project_mount": str(mount),
        "mac_output_path": str(output),
        "windows_project_root": str(windows_project_root),
        "windows_output_path": _windows_join(windows_project_root, relative_output.parts),
        "allowed_project_artifact_roots": list(ALLOWED_PROJECT_ARTIFACT_ROOTS),
        "relative_output_path": relative_output.as_posix(),
        "probe_status": "write_read_delete_passed",
        "probe_path": str(probe_path),
    }
