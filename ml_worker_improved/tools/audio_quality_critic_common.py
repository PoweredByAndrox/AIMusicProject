from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DATASET_SCHEMA = "audio_quality_critic_dataset.v1"
DATASET_MANIFEST_SCHEMA = "audio_quality_critic_dataset_manifest.v1"
MODEL_MANIFEST_SCHEMA = "audio_quality_critic_model_manifest.v1"
SCORE_MANIFEST_SCHEMA = "audio_quality_critic_score_manifest.v1"
VALIDATION_SCHEMA = "audio_quality_critic_validation.v1"
LEARNING_CYCLE_SCHEMA = "audio_quality_learning_cycle_manifest.v1"

DATASET_FILE = "dataset.jsonl"
BLOCKED_FILE = "blocked.jsonl"
MANIFEST_FILE = "manifest.json"
PROGRESS_FILE = "progress.jsonl"
MODEL_FILE = "model.pt"
METRICS_FILE = "metrics.json"
SCORES_FILE = "scores.jsonl"
REVIEW_FILE = "review.jsonl"

LABEL_TYPES = {
    "reference_verified",
    "human_verified",
    "locked_holdout",
    "weak_proxy",
    "pseudo_teacher",
    "rejected_blocked",
}
PROMOTION_LABEL_TYPES = {"reference_verified", "human_verified", "locked_holdout"}
WEAK_LABEL_TYPES = {"weak_proxy", "pseudo_teacher"}
STEM_TYPES = {"vocal", "minus", "both", "pair", "window"}
OUTPUT_NAMES = {
    "dataset_jsonl": DATASET_FILE,
    "blocked_jsonl": BLOCKED_FILE,
    "manifest_json": MANIFEST_FILE,
    "progress_jsonl": PROGRESS_FILE,
}
TARGET_NAMES = (
    "accept_probability",
    "vocal_bleed_in_minus",
    "instrumental_bleed_in_vocal",
    "vocal_hallucination",
    "robotic_artifact",
    "thin_vocal",
    "music_damage",
    "transient_damage",
    "phase_stereo_collapse",
    "uncertainty",
)
SAFETY_FLAGS = {
    "delivery_allowed": False,
    "release_ready": False,
    "can_publish_as_result": False,
    "thresholds_relaxed": False,
    "model_registry_edited": False,
    "runtime_gates_changed": False,
}


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield payload


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_progress(path: Path, event: str, *, start_time: float, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - start_time, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_or_cpu_test_phase_unless_training_device_cuda",
        "release_ready": False,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_label_type(value: Any) -> str:
    label_type = str(value or "").strip().lower()
    return label_type if label_type in LABEL_TYPES else ""


def normalize_stem_type(value: Any) -> str:
    stem = str(value or "").strip().lower()
    if stem == "instrumental":
        return "minus"
    if stem == "vocals":
        return "vocal"
    return stem if stem in STEM_TYPES else "window"


def split_for_track(track_id: str) -> str:
    digest = stable_hash(track_id or "missing_track")
    bucket = int(digest[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "dev"
    return "test"


def load_rag_current(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"rag_id": "", "idx": "", "status": "missing_rag_current"}
    current = read_json(path)
    idx_value = str(current.get("idx") or "")
    idx_path = path.parent / idx_value
    idx_exists = bool(idx_value and idx_path.exists())
    return {
        "rag_id": str(current.get("rag_id") or ""),
        "idx": idx_value,
        "idx_exists": idx_exists,
        "query_hint": str(current.get("query_hint") or ""),
        "path": str(path),
    }


def row_identity(row: dict[str, Any]) -> str:
    return stable_hash(
        {
            "track_id": row.get("track_id"),
            "candidate_id": row.get("candidate_id"),
            "stem_type": row.get("stem_type"),
            "window_start_sec": row.get("window_start_sec"),
            "window_end_sec": row.get("window_end_sec"),
            "tensor_path": row.get("tensor_path"),
        }
    )[:16]


def target_payload(row: dict[str, Any]) -> dict[str, float]:
    label = 1.0 if int(row.get("label") or 0) == 1 else 0.0
    raw_targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    targets = {name: as_float(raw_targets.get(name), 0.0) for name in TARGET_NAMES}
    targets["accept_probability"] = as_float(raw_targets.get("accept_probability"), label)
    default_issue = 0.0 if label > 0.5 else 1.0
    for name in TARGET_NAMES[1:-1]:
        targets[name] = as_float(raw_targets.get(name), default_issue)
    targets["uncertainty"] = as_float(raw_targets.get("uncertainty"), 0.0)
    return {name: round(float(targets[name]), 6) for name in TARGET_NAMES}


def tensor_stats(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path)
    values: list[float] = []
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(payload.files):
        arr = np.asarray(payload[name], dtype=np.float32)
        arrays[name] = arr
        flat = arr.reshape(-1)
        if flat.size == 0:
            values.extend([0.0, 0.0, 0.0, 0.0])
            continue
        values.extend(
            [
                float(np.mean(flat)),
                float(np.std(flat)),
                float(np.max(np.abs(flat))),
                float(np.sqrt(np.mean(np.square(flat)))),
            ]
        )
    if {"mix", "vocal", "minus"}.issubset(arrays):
        residual = arrays["mix"] - arrays["vocal"] - arrays["minus"]
        flat = residual.reshape(-1)
        values.extend([float(np.mean(flat)), float(np.std(flat)), float(np.max(np.abs(flat))), float(np.sqrt(np.mean(np.square(flat))))])
    return [0.0 if not math.isfinite(value) else round(value, 8) for value in values]


def feature_vector_from_row(row: dict[str, Any]) -> list[float]:
    if row.get("tensor_path"):
        return tensor_stats(Path(str(row["tensor_path"])))
    features = row.get("features")
    if isinstance(features, dict):
        return [as_float(features[key]) for key in sorted(features)]
    if isinstance(features, list):
        return [as_float(value) for value in features]
    raise ValueError("missing_tensor_or_feature_vector")


def pad_feature_vectors(vectors: Sequence[Sequence[float]]) -> tuple[list[list[float]], int]:
    width = max((len(vector) for vector in vectors), default=0)
    return [[float(value) for value in vector] + [0.0] * (width - len(vector)) for vector in vectors], width


def label_type_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(str(row.get("label_type") or "") for row in rows).items())}


def split_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(str(row.get("split") or "") for row in rows).items())}
