from __future__ import annotations

import os
import platform
from pathlib import Path


def running_inside_docker() -> bool:
    return Path("/.dockerenv").exists() or Path("/app").exists()


def _native_macos_diagnostic_allowed(*, explicit: bool = False) -> bool:
    return (
        (explicit or os.environ.get("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC") == "1")
        and platform.system() == "Darwin"
    )


def cli_runtime_contract(*, allow_native_runtime: bool) -> dict[str, object]:
    inside_docker = running_inside_docker()
    native_allowed = not inside_docker and bool(allow_native_runtime)
    return {
        "runtime_environment": "docker" if inside_docker else ("native_mac_cpu_dsp" if native_allowed else "host"),
        "native_runtime_allowed": native_allowed,
    }


def require_docker_runtime(
    *,
    tool_name: str,
    entrypoint_path: str,
    allow_native_macos_diagnostic: bool = False,
) -> None:
    if running_inside_docker():
        return
    if _native_macos_diagnostic_allowed(explicit=allow_native_macos_diagnostic):
        return
    raise RuntimeError(
        f"{tool_name} is Docker-only. Run it through "
        "`docker compose run --rm --no-deps -T ml-worker python "
        f"{entrypoint_path} ...`."
    )
