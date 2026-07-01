from __future__ import annotations

import os
import platform
from pathlib import Path

APPLE_CONTAINER_CLI = "/usr/local/bin/container"
FORBIDDEN_MAC_CONTAINER_ROUTES = (
    "Docker Desktop",
    "Docker daemon",
    "docker compose",
    "Docker VMM",
)
SUPPORTED_CONTAINER_RUNTIME_MARKERS = ("apple_container", "docker")


def running_inside_docker() -> bool:
    return Path("/.dockerenv").exists() or Path("/app").exists()


def _native_macos_diagnostic_allowed(*, explicit: bool = False) -> bool:
    return (
        (explicit or os.environ.get("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC") == "1")
        and platform.system() == "Darwin"
    )


def explicit_container_runtime_marker() -> str:
    marker = os.environ.get("AIMUSIC_CONTAINER_RUNTIME", "").strip().lower()
    return marker if marker in SUPPORTED_CONTAINER_RUNTIME_MARKERS else ""


def cli_runtime_contract(*, allow_native_runtime: bool) -> dict[str, object]:
    inside_docker = running_inside_docker()
    is_macos = platform.system() == "Darwin"
    native_allowed = not inside_docker and _native_macos_diagnostic_allowed(explicit=bool(allow_native_runtime))
    container_runtime = explicit_container_runtime_marker() or "docker"
    runtime_environment = container_runtime if inside_docker else ("native_mac_cpu_dsp" if native_allowed else "host")
    if is_macos and not inside_docker:
        container_runtime = "apple_container"
        if not native_allowed:
            runtime_environment = "apple_container_orchestration"
    return {
        "runtime_environment": runtime_environment,
        "container_runtime": container_runtime,
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
    if platform.system() == "Darwin":
        forbidden_routes = ", ".join(FORBIDDEN_MAC_CONTAINER_ROUTES)
        raise RuntimeError(
            f"{tool_name} requires the approved Mac project container route. "
            f"Use Apple container at `{APPLE_CONTAINER_CLI}` with a project image, for example "
            f"`{APPLE_CONTAINER_CLI} run --rm <project-image> python {entrypoint_path} ...`. "
            f"Mac routes using {forbidden_routes} are not approved; do not use docker compose on Mac."
        )
    raise RuntimeError(
        f"{tool_name} is Docker-only. Run it through "
        "`docker compose run --rm --no-deps -T ml-worker python "
        f"{entrypoint_path} ...`."
    )
