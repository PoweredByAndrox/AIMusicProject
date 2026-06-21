"""Shared path-scope guards for full-pack reference artifacts."""

from __future__ import annotations

from pathlib import Path


ARTIFACT_SCOPE_SEGMENT = "lab_artifacts"
NON_PRODUCTION_ARTIFACT_SEGMENTS = {
    "dbg_loop",
    "dbg_loop_tmp",
    "tmp_querysep_windowed_test",
}
SCOPED_NON_PRODUCTION_ARTIFACT_SEGMENTS = {
    "debug",
    "debug_artifacts",
    "fixture",
    "test_artifacts",
}
NON_PRODUCTION_ARTIFACT_PREFIXES = ("test_", "debug_", "smoke_")
NON_PRODUCTION_ARTIFACT_CONTAINS = ("_smoke",)


def artifact_scoped_parts(path: Path) -> tuple[list[str], bool]:
    parts = [part.lower() for part in path.parts]
    if ARTIFACT_SCOPE_SEGMENT not in parts:
        return parts, False
    scope_start = len(parts) - 1 - list(reversed(parts)).index(ARTIFACT_SCOPE_SEGMENT)
    return parts[scope_start + 1 :], True


def is_non_production_artifact_path(path: Path) -> bool:
    parts, has_artifact_scope = artifact_scoped_parts(path)
    if any(part in NON_PRODUCTION_ARTIFACT_SEGMENTS for part in parts):
        return True
    if not has_artifact_scope:
        return False
    return any(
        part in SCOPED_NON_PRODUCTION_ARTIFACT_SEGMENTS
        or part.startswith(NON_PRODUCTION_ARTIFACT_PREFIXES)
        or any(marker in part for marker in NON_PRODUCTION_ARTIFACT_CONTAINS)
        for part in parts
    )
