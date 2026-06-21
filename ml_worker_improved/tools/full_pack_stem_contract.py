"""Shared full-pack stem coverage helpers."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


REQUIRED_FULL_PACK_STEMS = ("vocal", "minus", "drums", "bass", "other", "guitar", "piano")


def normalize_stem_type(value: Any) -> str:
    stem = str(value or "").strip().lower().replace(" ", "_")
    if not stem:
        return ""
    if stem in {"vocals", "voice"} or stem.startswith("vocal"):
        return "vocal"
    if stem in {"instrumental", "accompaniment"} or stem.startswith("minus"):
        return "minus"
    return stem


def normalize_stem_list(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        stem = normalize_stem_type(value)
        if stem and stem not in seen:
            seen.add(stem)
            result.append(stem)
    return result


def required_stems_from_cli(*, requested: Sequence[str] | None, require_full_pack: bool) -> list[str]:
    stems = normalize_stem_list(requested)
    if require_full_pack:
        return list(REQUIRED_FULL_PACK_STEMS)
    return stems


def stem_type_from_item(item: dict[str, Any]) -> str:
    return normalize_stem_type(item.get("stem_type") or item.get("artifact_kind") or item.get("source_stem_key"))


def present_stems_from_items(items: Iterable[dict[str, Any]], required_stems: Sequence[str] | None = None) -> list[str]:
    raw_present = normalize_stem_list(stem_type_from_item(item) for item in items if isinstance(item, dict))
    required = normalize_stem_list(required_stems)
    if not required:
        return raw_present
    present = set(raw_present)
    ordered = [stem for stem in required if stem in present]
    ordered.extend(stem for stem in raw_present if stem not in required)
    return ordered


def missing_required_stems(required_stems: Sequence[str] | None, present_stems: Sequence[str] | None) -> list[str]:
    required = normalize_stem_list(required_stems)
    present = set(normalize_stem_list(present_stems))
    return [stem for stem in required if stem not in present]


def required_stem_missing_checks(missing_stems: Sequence[str], *, prefix: str = "required_stem_missing") -> list[str]:
    return [f"{prefix}:{stem}" for stem in missing_stems]


def pack_items_for_stem_contract(pack: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "stems", "listen_audio_files", "accepted_listen_preview_rows"):
        value = pack.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []
