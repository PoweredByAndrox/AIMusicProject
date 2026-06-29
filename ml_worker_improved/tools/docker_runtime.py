from __future__ import annotations

import os
import platform
from pathlib import Path


def _native_macos_diagnostic_allowed() -> bool:
    return (
        os.environ.get("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC") == "1"
        and platform.system() == "Darwin"
    )


def require_docker_runtime(*, tool_name: str, entrypoint_path: str) -> None:
    if Path("/.dockerenv").exists():
        return
    if _native_macos_diagnostic_allowed():
        return
    raise RuntimeError(
        f"{tool_name} is Docker-only. Run it through "
        "`docker compose run --rm --no-deps -T ml-worker python "
        f"{entrypoint_path} ...`."
    )
