"""Shared contracts for fully automatic stem contamination checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf


STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
NON_VOCAL_STEMS = ("drums", "bass", "guitar", "piano", "other")
MINUS_STEM = "minus"

FORBIDDEN_SOURCES = {
    "vocals": ("drums", "bass", "guitar", "piano", "other"),
    "drums": ("vocals", "bass", "guitar", "piano", "other"),
    "bass": ("vocals", "drums", "guitar", "piano", "other"),
    "guitar": ("vocals", "drums", "bass", "piano", "other"),
    "piano": ("vocals", "drums", "bass", "guitar", "other"),
    "other": ("vocals", "drums", "bass", "guitar", "piano"),
    "minus": ("vocals",),
}

VERDICTS = (
    "auto_accept",
    "blocked_leak",
    "blocked_damage",
    "blocked_missing_stem",
    "blocked_bad_reconstruction",
    "blocked_unknown",
    "blocked_no_clean_candidate",
)

GROUND_TRUTH_LABEL_TYPES = {"synthetic_clean", "clean_multitrack_ref"}
GENERATED_LABEL_TYPES = {"generated_separator", "pseudo_teacher", "model_generated"}
DIRTY_REF_LABEL_TYPES = {"dirty_ref", "dirty_ref_masked"}


def normalize_stem_name(value: Any) -> str:
    stem = str(value or "").strip().lower().replace(" ", "_")
    if stem in {"vocal", "voice"}:
        return "vocals"
    if stem in {"instrumental", "accompaniment"}:
        return "minus"
    return stem


def missing_required_stems(stem_paths: Mapping[str, Any]) -> list[str]:
    present = {normalize_stem_name(stem) for stem in stem_paths}
    return [stem for stem in STEMS if stem not in present]


def validate_six_stem_paths(stem_paths: Mapping[str, Any]) -> None:
    missing = missing_required_stems(stem_paths)
    if missing:
        raise ValueError(f"missing required stems: {', '.join(missing)}")


def validate_no_artifact_rag_output(output_dir: Path | str) -> None:
    parts = {part.lower() for part in Path(output_dir).parts}
    if "artifact_rag" in parts:
        raise ValueError("fully automatic stem search must not write into artifact_rag")


def validate_forbidden_source_pair(tested_stem: Any, foreign_source: Any) -> tuple[str, str]:
    tested = normalize_stem_name(tested_stem)
    foreign = normalize_stem_name(foreign_source)
    if foreign not in FORBIDDEN_SOURCES.get(tested, ()):
        raise ValueError(f"invalid forbidden source pair: {tested}:{foreign}")
    return tested, foreign


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return path


def build_minus_from_non_vocal_stems(stem_paths: Mapping[str, Path | str], output_path: Path | str) -> Path:
    missing = [stem for stem in NON_VOCAL_STEMS if stem not in {normalize_stem_name(key) for key in stem_paths}]
    if missing:
        raise ValueError(f"missing non-vocal stems for minus: {', '.join(missing)}")

    normalized = {normalize_stem_name(key): Path(value) for key, value in stem_paths.items()}
    mixed: np.ndarray | None = None
    expected_rate: int | None = None
    expected_shape: tuple[int, int] | None = None

    for stem in NON_VOCAL_STEMS:
        audio, sample_rate = _read_audio(normalized[stem])
        if expected_rate is None:
            expected_rate = sample_rate
            expected_shape = audio.shape
            mixed = np.zeros_like(audio, dtype=np.float32)
        if sample_rate != expected_rate:
            raise ValueError(f"sample rate mismatch for {stem}: {sample_rate} != {expected_rate}")
        if audio.shape != expected_shape:
            raise ValueError(f"shape mismatch for {stem}: {audio.shape} != {expected_shape}")
        mixed = mixed + audio

    assert mixed is not None
    assert expected_rate is not None
    return _write_audio(Path(output_path), mixed, expected_rate)


def validate_contamination_label_row(row: Mapping[str, Any]) -> None:
    label_type = str(row.get("label_type") or "").strip().lower()
    is_ground_truth = bool(row.get("is_ground_truth"))
    usable_for_promotion = bool(row.get("usable_for_promotion"))

    if label_type in GENERATED_LABEL_TYPES and is_ground_truth:
        raise ValueError("generated separator output cannot be marked as ground truth")
    if label_type in DIRTY_REF_LABEL_TYPES and is_ground_truth:
        raise ValueError("dirty reference cannot be marked as exact ground truth")
    if label_type == "dirty_ref_masked" and usable_for_promotion:
        raise ValueError("dirty masked reference cannot drive promotion")
    if is_ground_truth and label_type not in GROUND_TRUTH_LABEL_TYPES:
        raise ValueError(f"label_type {label_type!r} is not allowed as ground truth")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path
