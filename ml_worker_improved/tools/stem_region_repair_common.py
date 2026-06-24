"""Shared contracts for training-first stem region repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

try:
    from tools.stem_contamination_common import STEMS, MINUS_STEM, normalize_stem_name
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_contamination_common import STEMS, MINUS_STEM, normalize_stem_name


ISSUE_VOCAL_LEAK = "vocal_leak"
ISSUE_INSTRUMENT_LEAK = "instrument_leak"
ISSUE_MISSING_DETAIL = "missing_detail"
ISSUE_THIN_BODY = "thin_body"
ISSUE_TRANSIENT_LOSS = "transient_loss"
ISSUE_STEREO_COLLAPSE = "stereo_collapse"
ISSUE_PHASE_DAMAGE = "phase_damage"
ISSUE_HALLUCINATION = "hallucination"
ISSUE_BAD_RECONSTRUCTION = "bad_reconstruction"
ISSUE_INFRA = "infra_failure"
ISSUE_UNKNOWN = "unknown"

ISSUE_TYPES = (
    ISSUE_VOCAL_LEAK,
    ISSUE_INSTRUMENT_LEAK,
    ISSUE_MISSING_DETAIL,
    ISSUE_THIN_BODY,
    ISSUE_TRANSIENT_LOSS,
    ISSUE_STEREO_COLLAPSE,
    ISSUE_PHASE_DAMAGE,
    ISSUE_HALLUCINATION,
    ISSUE_BAD_RECONSTRUCTION,
    ISSUE_INFRA,
    ISSUE_UNKNOWN,
)

ROOT_CAUSE_MODEL_GAP = "model_gap"
ROOT_CAUSE_ROUTING_GAP = "routing_gap"
ROOT_CAUSE_REPAIR_GAP = "repair_gap"
ROOT_CAUSE_INFRA_GAP = "infra_gap"
ROOT_CAUSE_DATA_GAP = "data_gap"

ROOT_CAUSES = (
    ROOT_CAUSE_MODEL_GAP,
    ROOT_CAUSE_ROUTING_GAP,
    ROOT_CAUSE_REPAIR_GAP,
    ROOT_CAUSE_INFRA_GAP,
    ROOT_CAUSE_DATA_GAP,
)

LABEL_TYPE_SYNTHETIC_CLEAN = "synthetic_clean"
LABEL_TYPE_REFERENCE_VERIFIED = "reference_verified"
LABEL_TYPE_HUMAN_VERIFIED = "human_verified"
LABEL_TYPE_PSEUDO_TEACHER = "pseudo_teacher"
LABEL_TYPE_GENERATED_SEPARATOR = "generated_separator"
LABEL_TYPE_MODEL_GENERATED = "model_generated"

GROUND_TRUTH_LABEL_TYPES = {
    LABEL_TYPE_SYNTHETIC_CLEAN,
    LABEL_TYPE_REFERENCE_VERIFIED,
    LABEL_TYPE_HUMAN_VERIFIED,
}
GENERATED_LABEL_TYPES = {
    LABEL_TYPE_PSEUDO_TEACHER,
    LABEL_TYPE_GENERATED_SEPARATOR,
    LABEL_TYPE_MODEL_GENERATED,
}

SAFE_ROW_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def validate_issue_type(value: Any) -> str:
    issue_type = str(value or "").strip().lower()
    if issue_type not in ISSUE_TYPES:
        raise ValueError(f"unsupported repair issue_type: {value!r}")
    return issue_type


def validate_root_cause(value: Any) -> str:
    root_cause = str(value or "").strip().lower()
    if root_cause not in ROOT_CAUSES:
        raise ValueError(f"unsupported root_cause: {value!r}")
    return root_cause


def validate_region(region: Mapping[str, Any] | None) -> dict[str, float]:
    if region is None:
        return {"start_sec": 0.0, "end_sec": 0.0}
    start_sec = float(region.get("start_sec", 0.0))
    end_sec = float(region.get("end_sec", 0.0))
    if start_sec < 0.0:
        raise ValueError("region start_sec must be non-negative")
    if end_sec < start_sec:
        raise ValueError("region end_sec must be >= start_sec")
    return {"start_sec": start_sec, "end_sec": end_sec}


def validate_region_repair_row(row: Mapping[str, Any]) -> None:
    row_id = str(row.get("row_id") or "").strip()
    if row_id and not SAFE_ROW_ID_RE.match(row_id):
        raise ValueError(f"unsafe row_id: {row_id!r}")

    label_type = str(row.get("label_type") or "").strip().lower()
    is_ground_truth = bool(row.get("is_ground_truth"))
    usable_for_promotion = bool(row.get("usable_for_promotion", False))

    if label_type in GENERATED_LABEL_TYPES and is_ground_truth:
        raise ValueError("generated separator or pseudo output cannot be marked as ground truth")
    if is_ground_truth and label_type not in GROUND_TRUTH_LABEL_TYPES:
        raise ValueError(f"label_type {label_type!r} is not allowed as ground truth")
    if label_type in GENERATED_LABEL_TYPES and usable_for_promotion:
        raise ValueError("generated or pseudo repair rows cannot drive promotion")
    if usable_for_promotion and label_type not in GROUND_TRUTH_LABEL_TYPES:
        raise ValueError(f"label_type {label_type!r} cannot drive promotion")

    validate_issue_type(row.get("issue_type"))
    stem = normalize_stem_name(row.get("target_stem") or row.get("affected_stem"))
    if stem not in set(STEMS) | {MINUS_STEM}:
        raise ValueError(f"unsupported target stem: {stem!r}")
    validate_region(row.get("region") if isinstance(row.get("region"), Mapping) else None)


def make_region_mask(
    *,
    frame_count: int,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    fade_sec: float = 0.0,
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if end_sec < start_sec:
        raise ValueError("end_sec must be >= start_sec")

    mask = np.zeros(int(frame_count), dtype=np.float32)
    start = max(0, min(int(round(float(start_sec) * float(sample_rate))), frame_count))
    end = max(start, min(int(round(float(end_sec) * float(sample_rate))), frame_count))
    if end <= start:
        return mask
    mask[start:end] = 1.0
    fade_frames = max(0, int(round(float(fade_sec) * float(sample_rate))))
    if fade_frames > 0:
        fade_frames = min(fade_frames, max(1, (end - start) // 2))
        ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        mask[start : start + fade_frames] *= ramp
        mask[end - fade_frames : end] *= ramp[::-1]
    return mask


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError(f"invalid audio shape: {arr.shape}")
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    if arr.shape[1] != 2:
        raise ValueError(f"expected mono/stereo audio, got {arr.shape[1]} channels")
    return arr.astype(np.float32, copy=False)


def read_audio_stereo(path: Path | str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return ensure_stereo(audio), int(sample_rate)


def write_audio(path: Path | str, audio: np.ndarray, sample_rate: int) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), ensure_stereo(audio), int(sample_rate))
    return out


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_jsonl(path: Path | str, rows: list[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return out
