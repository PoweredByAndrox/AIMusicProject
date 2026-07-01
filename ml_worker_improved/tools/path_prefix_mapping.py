"""Path prefix remapping for mirrored local-network artifact workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def parse_path_prefix_map(values: Sequence[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid path prefix map, expected FROM=TO: {text}")
        source, target = text.split("=", 1)
        source = source.strip().rstrip("/\\")
        target = target.strip().rstrip("/\\")
        if not source or not target:
            raise ValueError(f"Invalid path prefix map, expected non-empty FROM=TO: {text}")
        mappings.append((source, target))
    return mappings


def resolve_audio_path(audio_path: str, path_prefix_map: Sequence[tuple[str, str]]) -> tuple[str, dict[str, Any]]:
    original = str(audio_path or "").strip()
    resolution: dict[str, Any] = {
        "original_path": original,
        "resolved_path": original,
        "mapped": False,
        "mapping_from": "",
        "mapping_to": "",
    }
    if not original:
        return "", resolution
    normalized_original = original.replace("\\", "/")
    for source_prefix, target_prefix in path_prefix_map:
        normalized_source = str(source_prefix).replace("\\", "/").rstrip("/")
        if normalized_original != normalized_source and not normalized_original.startswith(normalized_source + "/"):
            continue
        suffix = normalized_original[len(normalized_source) :].lstrip("/")
        suffix_parts = [part for part in suffix.split("/") if part]
        resolved = str(Path(target_prefix).joinpath(*suffix_parts)) if suffix_parts else str(Path(target_prefix))
        resolution.update(
            {
                "resolved_path": resolved,
                "mapped": True,
                "mapping_from": source_prefix,
                "mapping_to": target_prefix,
            }
        )
        return resolved, resolution
    return original, resolution
