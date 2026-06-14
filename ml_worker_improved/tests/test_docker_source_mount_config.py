from __future__ import annotations

from pathlib import Path

import pytest


REQUIRED_MOUNT = "./testing_stuff_music:/host_testing_stuff_music:ro"


def read_compose_text() -> str:
    candidates = [
        Path("/app/docker-compose.yml"),
        Path(__file__).resolve().parents[2] / "docker-compose.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    pytest.skip("docker-compose.yml is not mounted into this container")


def ml_worker_service_block(compose_text: str) -> str:
    lines = compose_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line == "  ml-worker:":
            start = index
            break
    assert start is not None, "ml-worker service not found in docker-compose.yml"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            end = index
            break
    return "\n".join(lines[start:end])


def test_ml_worker_mounts_testing_stuff_music_read_only() -> None:
    service_block = ml_worker_service_block(read_compose_text())

    assert REQUIRED_MOUNT in service_block
    assert "./testing_stuff_music:/host_testing_stuff_music" not in service_block.replace(REQUIRED_MOUNT, "")
