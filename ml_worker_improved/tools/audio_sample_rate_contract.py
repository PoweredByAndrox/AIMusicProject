from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import yaml


def load_configured_model_sample_rate(config_path: Path) -> int:
    payload = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
    audio = payload.get("audio") or {}
    model = payload.get("model") or {}
    candidates = [
        int(value)
        for value in (audio.get("sample_rate"), model.get("sample_rate"))
        if value not in (None, "")
    ]
    if not candidates:
        raise ValueError(f"Model config is missing model sample_rate: {config_path}")
    if any(value <= 0 for value in candidates):
        raise ValueError(f"Model config has invalid model sample_rate: {candidates}")
    if len(set(candidates)) != 1:
        raise ValueError(f"Model config has conflicting model sample_rate values: {candidates}")
    return candidates[0]


def resample_audio_to_layout(
    audio: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
    target_frames: int,
) -> np.ndarray:
    source_sample_rate = int(source_sample_rate)
    target_sample_rate = int(target_sample_rate)
    target_frames = int(target_frames)
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("audio sample rates must be positive")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized[:, None]
    if normalized.ndim != 2:
        raise ValueError(f"audio must be frames x channels, got shape={normalized.shape}")
    if source_sample_rate != target_sample_rate:
        normalized = librosa.resample(
            normalized.T,
            orig_sr=source_sample_rate,
            target_sr=target_sample_rate,
            res_type="soxr_hq",
            axis=-1,
        ).T
    if normalized.shape[0] > target_frames:
        normalized = normalized[:target_frames]
    elif normalized.shape[0] < target_frames:
        normalized = np.pad(normalized, ((0, target_frames - normalized.shape[0]), (0, 0)))
    if not np.isfinite(normalized).all():
        raise RuntimeError("resampled audio contains non-finite samples")
    return np.ascontiguousarray(normalized, dtype=np.float32)
