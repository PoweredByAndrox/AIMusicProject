from __future__ import annotations

from pathlib import Path


def require_docker_runtime(*, tool_name: str, entrypoint_path: str) -> None:
    if Path("/.dockerenv").exists():
        return
    raise RuntimeError(
        f"{tool_name} is Docker-only. Run it through "
        "`docker compose run --rm --no-deps -T ml-worker python "
        f"{entrypoint_path} ...`."
    )
