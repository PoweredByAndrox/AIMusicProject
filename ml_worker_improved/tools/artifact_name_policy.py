"""Short Windows-safe artifact naming helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


MAX_ARTIFACT_SEGMENT_CHARS = 32
ARTIFACT_AUDIO_DIR = "audio"
_WINDOWS_RESERVED_BASE_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}

_STEM_AUDIO_NAMES = {
    "vocal": "vocal",
    "minus": "minus",
    "instrumental": "inst",
    "piano": "piano",
    "bass": "bass",
    "drums": "drums",
    "guitar": "guitar",
    "melody": "melody",
    "other": "other",
    "unknown": "unk",
}


def indexed_alias(prefix: str, index: int) -> str:
    if index < 1:
        raise ValueError("index must be >= 1")
    if not prefix.isascii() or not prefix.isalnum():
        raise ValueError("prefix must be ascii alphanumeric")
    return f"{prefix}{index:03d}"


def indexed_audio_filename(prefix: str, index: int, *, suffix: str = ".flac") -> str:
    name = f"{indexed_alias(prefix, index)}{suffix}"
    validate_artifact_segment(name)
    return name


def track_alias(index: int) -> str:
    return indexed_alias("t", index)


def stem_audio_filename(stem_type: str, *, duplicate_index: int = 1) -> str:
    stem = str(stem_type or "unknown")
    base = _STEM_AUDIO_NAMES.get(stem, _STEM_AUDIO_NAMES["unknown"])
    name = f"{base}.flac" if duplicate_index == 1 else f"{base}{duplicate_index}.flac"
    validate_artifact_segment(name)
    return name


def relative_artifact_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root).as_posix())


def validate_artifact_segment(segment: str) -> None:
    if not segment:
        raise ValueError("artifact path segment must be non-empty")
    if segment in {".", ".."}:
        raise ValueError(f"artifact path segment cannot be traversal marker: {segment}")
    if len(segment) > MAX_ARTIFACT_SEGMENT_CHARS:
        raise ValueError(f"artifact path segment too long: {segment}")
    if any(ch in segment for ch in '<>:"/\\|?*'):
        raise ValueError(f"artifact path segment has Windows-unsafe character: {segment}")
    if any(ord(ch) < 32 for ch in segment):
        raise ValueError(f"artifact path segment has control character: {segment!r}")
    if segment.endswith((" ", ".")):
        raise ValueError(f"artifact path segment has Windows-unsafe suffix: {segment}")
    base_name = segment.split(".", 1)[0].upper()
    if base_name in _WINDOWS_RESERVED_BASE_NAMES or re.fullmatch(r"(COM|LPT)[1-9]", base_name):
        raise ValueError(f"artifact path segment uses Windows reserved device name: {segment}")


def validate_artifact_relative_path(relative_path: str | Path) -> None:
    raw = str(relative_path or "").replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", raw) or not raw:
        raise ValueError(f"artifact path must be non-empty relative path: {relative_path}")
    for segment in raw.split("/"):
        validate_artifact_segment(segment)


def invalid_artifact_segments(paths: Iterable[str | Path]) -> list[str]:
    failed: list[str] = []
    for raw_path in paths:
        try:
            validate_artifact_relative_path(raw_path)
        except ValueError:
            failed.append(str(raw_path))
    return failed
