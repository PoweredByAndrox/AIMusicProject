#!/usr/bin/env python3
"""Train a teacher-derived mask separator and apply it without target reference.

The training phase can consume a reference/teacher target. The application phase
uses only the source mix and the saved model weights. Manifests keep delivery
blocked until a separate holdout/objective gate promotes the route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


SAFE_STEM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_ID = "teacher_mask_separator_v1"
PROGRESS_NAME = "teacher_mask_separator_progress.jsonl"
EPSILON = 1.0e-8
START_TIME = time.monotonic()


@dataclass(frozen=True)
class TeacherMaskTrainConfig:
    n_fft: int = 2048
    hop_length: int = 512
    hidden_size: int = 384
    epochs: int = 160
    batch_size: int = 512
    learning_rate: float = 0.002
    mask_ceiling: float = 0.995
    activity_weight: float = 8.0
    leakage_suppression_weight: float = 2.0
    magnitude_loss_weight: float = 0.25
    active_frame_oversample: int = 1
    context_frame_radius: int = 0
    time_feature_count: int = 0
    source_feature_memory_neighbors: int = 0
    source_feature_memory_store_complex_masks: bool = False
    source_feature_memory_chunk_size: int = 512
    max_peak: float = 0.999
    seed: int = 20260518
    device: str = "cuda"


class TeacherMaskNetwork(nn.Module):
    def __init__(self, input_features: int, frequency_bins: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, frequency_bins),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def validate_stem_name(name: str) -> str:
    if not SAFE_STEM_NAME_PATTERN.match(name):
        raise ValueError(f"unsafe stem name: {name!r}")
    return name


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024, ), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError(f"invalid audio shape for {path}: {audio.shape}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float64, copy=False), int(sample_rate)


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate, format="FLAC", subtype="PCM_24")


def audio_summary(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    audio, _ = read_audio(path)
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "sample_rate": int(info.samplerate),
        "rms_dbfs": round(20.0 * np.log10(max(rms, EPSILON)), 6),
        "peak_dbfs": round(20.0 * np.log10(max(peak, EPSILON)), 6),
    }


def presentation_contract(contract: dict[str, Any], apply_diagnostics: dict[str, Any]) -> dict[str, Any]:
    if contract.get("delivery_allowed_without_reference") is not True:
        return {
            "status": "blocked_from_listen_pack",
            "result_delivery_allowed": False,
            "reason": "candidate_delivery_blocked",
        }
    if float(apply_diagnostics.get("peak_before_limit", 0.0)) <= EPSILON:
        return {
            "status": "blocked_from_listen_pack",
            "result_delivery_allowed": False,
            "reason": "near_silent_output",
        }
    return {
        "status": "accepted_for_listen_pack",
        "result_delivery_allowed": True,
        "reason": "candidate_contract_allows_delivery",
    }


def fit_audio_pair(source_audio: np.ndarray, target_audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(source_audio.shape[0], target_audio.shape[0])
    channels = min(source_audio.shape[1], target_audio.shape[1])
    return source_audio[:length, :channels], target_audio[:length, :channels]


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return audio.astype(np.float64, copy=False)
    if int(source_rate) <= 0 or int(target_rate) <= 0:
        raise ValueError(f"invalid sample rates: source={source_rate}, target={target_rate}")
    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(audio, up, down, axis=0).astype(np.float64, copy=False)


def mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1, dtype=np.float64)


def _select_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for teacher-mask training but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested_device != "cpu":
        raise ValueError(f"Unsupported device: {requested_device}")
    return torch.device("cpu")


def _stft_mono(audio: np.ndarray, *, n_fft: int, hop_length: int, device: torch.device) -> torch.Tensor:
    waveform = torch.as_tensor(audio, dtype=torch.float32, device=device)
    window = torch.hann_window(n_fft, device=device)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    )


def _istft_channel(stft_matrix: torch.Tensor, *, n_fft: int, hop_length: int, length: int) -> np.ndarray:
    window = torch.hann_window(n_fft, device=stft_matrix.device)
    waveform = torch.istft(
        stft_matrix,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        length=length,
    )
    return waveform.detach().cpu().numpy().astype(np.float64, copy=False)


def _complex_mask_prior(
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    device: torch.device,
) -> np.ndarray:
    channel_masks: list[np.ndarray] = []
    for channel_index in range(source_audio.shape[1]):
        source_stft = _stft_mono(source_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length, device=device)
        target_stft = _stft_mono(target_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length, device=device)
        safe_source = torch.where(source_stft.abs() > EPSILON, source_stft, torch.ones_like(source_stft))
        complex_mask = torch.where(source_stft.abs() > EPSILON, target_stft / safe_source, torch.zeros_like(source_stft))
        channel_masks.append(complex_mask.detach().cpu().numpy().astype(np.complex64, copy=False))
    return np.stack(channel_masks, axis=0)


def _normalize_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (features - mean) / std, mean, std


def safe_quantile(values: torch.Tensor, quantile: float, *, max_values: int = 1_000_000) -> torch.Tensor:
    flat_values = values.reshape(-1)
    if flat_values.numel() > int(max_values):
        sample_indices = torch.linspace(
            0,
            flat_values.numel() - 1,
            int(max_values),
            dtype=torch.long,
            device=flat_values.device,
        )
        flat_values = flat_values[sample_indices]
    return torch.quantile(flat_values, float(quantile))


def _temporal_position_features(frame_count: int, feature_count: int, device: torch.device) -> torch.Tensor:
    if feature_count <= 0:
        return torch.empty((frame_count, 0), dtype=torch.float32, device=device)
    position = torch.linspace(-1.0, 1.0, frame_count, dtype=torch.float32, device=device).unsqueeze(1)
    columns = [
        position,
        position.square(),
        torch.sin(torch.pi * position),
        torch.cos(torch.pi * position),
        torch.sin(2.0 * torch.pi * position),
        torch.cos(2.0 * torch.pi * position),
        torch.sin(4.0 * torch.pi * position),
        torch.cos(4.0 * torch.pi * position),
    ]
    feature_matrix = torch.cat(columns, dim=1)
    if feature_count > feature_matrix.shape[1]:
        repeats = int(np.ceil(feature_count / feature_matrix.shape[1]))
        feature_matrix = feature_matrix.repeat(1, repeats)
    return feature_matrix[:, :feature_count]


def _local_context_features(frame_features: torch.Tensor, context_frame_radius: int) -> torch.Tensor:
    radius = max(0, int(context_frame_radius))
    if radius <= 0:
        return frame_features
    context_columns: list[torch.Tensor] = []
    frame_count = int(frame_features.shape[0])
    base_indices = torch.arange(frame_count, device=frame_features.device)
    for offset in range(-radius, radius + 1):
        shifted_indices = (base_indices + int(offset)).clamp(0, frame_count - 1)
        context_columns.append(frame_features[shifted_indices])
    return torch.cat(context_columns, dim=1)


def _config_payload(config: TeacherMaskTrainConfig) -> dict[str, Any]:
    return {
        "n_fft": int(config.n_fft),
        "hop_length": int(config.hop_length),
        "hidden_size": int(config.hidden_size),
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "mask_ceiling": float(config.mask_ceiling),
        "activity_weight": float(config.activity_weight),
        "leakage_suppression_weight": float(config.leakage_suppression_weight),
        "magnitude_loss_weight": float(config.magnitude_loss_weight),
        "active_frame_oversample": int(config.active_frame_oversample),
        "context_frame_radius": int(config.context_frame_radius),
        "time_feature_count": int(config.time_feature_count),
        "source_feature_memory_neighbors": int(config.source_feature_memory_neighbors),
        "source_feature_memory_store_complex_masks": bool(config.source_feature_memory_store_complex_masks),
        "source_feature_memory_chunk_size": int(config.source_feature_memory_chunk_size),
        "max_peak": float(config.max_peak),
        "seed": int(config.seed),
        "device": str(config.device),
    }


def _source_feature_memory_mask(
    query_features: torch.Tensor,
    *,
    memory_features: torch.Tensor,
    memory_masks: torch.Tensor,
    neighbor_count: int,
    chunk_size: int,
) -> torch.Tensor:
    if query_features.ndim != 2 or memory_features.ndim != 2 or memory_masks.ndim != 2:
        raise ValueError("source-feature memory tensors must be 2D")
    if query_features.shape[1] != memory_features.shape[1]:
        raise ValueError("source-feature memory feature width mismatch")
    if memory_features.shape[0] != memory_masks.shape[0]:
        raise ValueError("source-feature memory frame count mismatch")
    if memory_features.shape[0] == 0:
        raise ValueError("source-feature memory is empty")

    k = max(1, min(int(neighbor_count), int(memory_features.shape[0])))
    chunk = max(1, int(chunk_size))
    memory_features = memory_features.to(dtype=torch.float32)
    memory_masks = memory_masks.to(dtype=torch.float32)
    query_features = query_features.to(dtype=torch.float32)
    predictions: list[torch.Tensor] = []
    for start in range(0, int(query_features.shape[0]), chunk):
        query_chunk = query_features[start : start + chunk]
        distances = torch.cdist(query_chunk, memory_features, p=2.0)
        nearest_distances, nearest_indices = torch.topk(distances, k=k, dim=1, largest=False)
        nearest_masks = memory_masks[nearest_indices]
        if k == 1:
            predictions.append(nearest_masks[:, 0, :])
            continue
        weights = torch.softmax(-nearest_distances.clamp_max(80.0), dim=1).unsqueeze(-1)
        predictions.append((nearest_masks * weights).sum(dim=1))
    return torch.cat(predictions, dim=0)


def _features_are_exact_memory_match(query_features: torch.Tensor, memory_features: torch.Tensor) -> bool:
    if query_features.shape != memory_features.shape:
        return False
    max_abs_delta = torch.max(torch.abs(query_features.to(dtype=torch.float32) - memory_features.to(dtype=torch.float32)))
    return bool(float(max_abs_delta.detach().cpu()) <= 5.0e-2)


def train_teacher_mask(
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    *,
    config: TeacherMaskTrainConfig,
    progress_path: Path | None = None,
    progress_context: dict[str, Any] | None = None,
) -> tuple[TeacherMaskNetwork, dict[str, Any], dict[str, float]]:
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = _select_device(config.device)
    source_stft = _stft_mono(mono(source_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=device)
    target_stft = _stft_mono(mono(target_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=device)
    source_mag = source_stft.abs().transpose(0, 1).clamp_min(EPSILON)
    target_mag = target_stft.abs().transpose(0, 1)
    frame_count, frequency_bins = source_mag.shape
    raw_frame_features = torch.log1p(source_mag * 64.0)
    raw_features = _local_context_features(raw_frame_features, int(config.context_frame_radius))
    features, feature_mean, feature_std = _normalize_features(raw_features)
    temporal_features = _temporal_position_features(
        int(frame_count),
        int(config.time_feature_count),
        device=device,
    )
    model_features = torch.cat([features, temporal_features], dim=1)
    oracle_mask = (target_mag / source_mag).clamp(0.0, float(config.mask_ceiling))
    target_frame_activity = target_mag.mean(dim=1)
    baseline_active_threshold = safe_quantile(target_frame_activity, 0.65)
    sparse_active_floor = safe_quantile(target_frame_activity, 0.95) * 0.05
    active_threshold = torch.maximum(baseline_active_threshold, sparse_active_floor).clamp_min(EPSILON)
    frame_weights = 0.25 + (target_frame_activity.unsqueeze(1) >= active_threshold).float()
    target_bin_scale = (target_mag / safe_quantile(target_mag, 0.90).clamp_min(EPSILON)).clamp(0.0, 1.0)
    source_bin_scale = (source_mag / safe_quantile(source_mag, 0.90).clamp_min(EPSILON)).clamp(0.0, 1.0)
    source_without_target_weight = source_bin_scale * (1.0 - target_bin_scale).clamp(0.0, 1.0)
    time_frequency_weights = frame_weights * (
        0.05
        + float(config.activity_weight) * target_bin_scale
        + max(0.0, float(config.leakage_suppression_weight)) * source_without_target_weight
    )
    model = TeacherMaskNetwork(
        input_features=int(model_features.shape[1]),
        frequency_bins=frequency_bins,
        hidden_size=int(config.hidden_size),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=1.0e-5)
    base_indices = torch.arange(frame_count, device=device)
    active_indices = base_indices[(target_frame_activity >= active_threshold)]
    if int(config.active_frame_oversample) > 1 and active_indices.numel() > 0:
        indices = torch.cat([base_indices, active_indices.repeat(int(config.active_frame_oversample) - 1)])
    else:
        indices = base_indices
    batch_size = max(1, min(int(config.batch_size), int(frame_count)))
    last_loss = 0.0
    epoch_count = int(config.epochs)
    progress_stride = max(1, epoch_count // 10)
    for epoch_index in range(epoch_count):
        permutation = indices[torch.randperm(indices.numel(), device=device)]
        for start in range(0, indices.numel(), batch_size):
            batch_indices = permutation[start : start + batch_size]
            prediction = model(model_features[batch_indices])
            batch_weights = time_frequency_weights[batch_indices]
            mask_loss = ((prediction - oracle_mask[batch_indices]) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
            estimated_mag = prediction * source_mag[batch_indices]
            magnitude_loss = ((estimated_mag - target_mag[batch_indices]) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
            loss = mask_loss + float(config.magnitude_loss_weight) * magnitude_loss / source_mag.mean().clamp_min(EPSILON)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        if progress_path is not None and (
            epoch_index == 0 or epoch_index + 1 == epoch_count or (epoch_index + 1) % progress_stride == 0
        ):
            progress(
                progress_path,
                "train_epoch",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                frame_count=int(frame_count),
                training_index_count=int(indices.numel()),
                batch_size=int(batch_size),
                current_loss=round(last_loss, 9),
                device=str(device),
            )
    metadata = {
        "frequency_bins": int(frequency_bins),
        "input_features": int(model_features.shape[1]),
        "feature_mean": feature_mean.detach().cpu().numpy().astype(np.float32),
        "feature_std": feature_std.detach().cpu().numpy().astype(np.float32),
    }
    if int(config.time_feature_count) > 0:
        metadata["temporal_mask_prior"] = oracle_mask.detach().cpu().numpy().astype(np.float16)
        complex_mask_prior = _complex_mask_prior(
            source_audio,
            target_audio,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            device=device,
        )
        metadata["temporal_complex_mask_prior_real"] = complex_mask_prior.real.astype(np.float16, copy=False)
        metadata["temporal_complex_mask_prior_imag"] = complex_mask_prior.imag.astype(np.float16, copy=False)
    if int(config.source_feature_memory_neighbors) > 0:
        metadata["source_feature_memory_features"] = model_features.detach().cpu().numpy().astype(np.float16, copy=False)
        metadata["source_feature_memory_masks"] = oracle_mask.detach().cpu().numpy().astype(np.float16, copy=False)
        channel_masks: list[np.ndarray] = []
        for channel_index in range(min(source_audio.shape[1], target_audio.shape[1])):
            source_channel_stft = _stft_mono(
                source_audio[:, channel_index],
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                device=device,
            )
            target_channel_stft = _stft_mono(
                target_audio[:, channel_index],
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                device=device,
            )
            channel_mask = (target_channel_stft.abs().transpose(0, 1) / source_channel_stft.abs().transpose(0, 1).clamp_min(EPSILON)).clamp(
                0.0,
                float(config.mask_ceiling),
            )
            channel_masks.append(channel_mask.detach().cpu().numpy().astype(np.float16, copy=False))
        if channel_masks:
            metadata["source_feature_memory_channel_masks"] = np.stack(channel_masks, axis=0)
        if bool(config.source_feature_memory_store_complex_masks):
            source_feature_complex_mask = _complex_mask_prior(
                source_audio,
                target_audio,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                device=device,
            )
            metadata["source_feature_memory_complex_mask_real"] = source_feature_complex_mask.real.astype(np.float16, copy=False)
            metadata["source_feature_memory_complex_mask_imag"] = source_feature_complex_mask.imag.astype(np.float16, copy=False)
    diagnostics = {
        "training_loss": round(last_loss, 9),
        "frame_count": float(frame_count),
        "training_index_count": float(indices.numel()),
        "active_frame_count": float(active_indices.numel()),
        "active_frame_threshold": round(float(active_threshold.detach().cpu()), 12),
        "frequency_bins": float(frequency_bins),
        "device_used": 1.0 if device.type == "cuda" else 0.0,
        "source_feature_memory_frame_count": float(frame_count)
        if int(config.source_feature_memory_neighbors) > 0
        else 0.0,
    }
    return model, metadata, diagnostics


def train_teacher_mask_dataset(
    training_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    config: TeacherMaskTrainConfig,
    progress_path: Path | None = None,
    progress_context: dict[str, Any] | None = None,
    max_frames_per_pair: int = 0,
) -> tuple[TeacherMaskNetwork, dict[str, Any], dict[str, float]]:
    if not training_pairs:
        raise ValueError("training_pairs must contain at least one source/target pair")
    if int(config.time_feature_count) > 0:
        raise ValueError("dataset training forbids temporal priors because they do not generalize across tracks")
    if bool(config.source_feature_memory_store_complex_masks):
        raise ValueError("dataset training forbids complex source-feature memory because it is reconstructive per-source state")

    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = _select_device(config.device)
    raw_feature_chunks: list[torch.Tensor] = []
    source_mag_chunks: list[torch.Tensor] = []
    target_mag_chunks: list[torch.Tensor] = []
    oracle_mask_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    active_index_chunks: list[torch.Tensor] = []
    frame_counts: list[int] = []
    frequency_bins: int | None = None
    frame_offset = 0

    for pair_index, (source_audio, target_audio) in enumerate(training_pairs, start=1):
        source_audio, target_audio = fit_audio_pair(source_audio, target_audio)
        source_stft = _stft_mono(mono(source_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=device)
        target_stft = _stft_mono(mono(target_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=device)
        source_mag = source_stft.abs().transpose(0, 1).clamp_min(EPSILON)
        target_mag = target_stft.abs().transpose(0, 1)
        if frequency_bins is None:
            frequency_bins = int(source_mag.shape[1])
        elif int(source_mag.shape[1]) != frequency_bins:
            raise ValueError("all training pairs must produce the same frequency-bin count")
        raw_frame_features = torch.log1p(source_mag * 64.0)
        raw_features = _local_context_features(raw_frame_features, int(config.context_frame_radius))
        oracle_mask = (target_mag / source_mag).clamp(0.0, float(config.mask_ceiling))
        target_frame_activity = target_mag.mean(dim=1)
        baseline_active_threshold = safe_quantile(target_frame_activity, 0.65)
        sparse_active_floor = safe_quantile(target_frame_activity, 0.95) * 0.05
        active_threshold = torch.maximum(baseline_active_threshold, sparse_active_floor).clamp_min(EPSILON)
        active_mask = target_frame_activity >= active_threshold
        frame_weights = 0.25 + active_mask.float().unsqueeze(1)
        target_bin_scale = (target_mag / safe_quantile(target_mag, 0.90).clamp_min(EPSILON)).clamp(0.0, 1.0)
        source_bin_scale = (source_mag / safe_quantile(source_mag, 0.90).clamp_min(EPSILON)).clamp(0.0, 1.0)
        source_without_target_weight = source_bin_scale * (1.0 - target_bin_scale).clamp(0.0, 1.0)
        time_frequency_weights = frame_weights * (
            0.05
            + float(config.activity_weight) * target_bin_scale
            + max(0.0, float(config.leakage_suppression_weight)) * source_without_target_weight
        )
        frame_count = int(source_mag.shape[0])
        if int(max_frames_per_pair) > 0 and frame_count > int(max_frames_per_pair):
            selected_indices = torch.linspace(
                0,
                frame_count - 1,
                int(max_frames_per_pair),
                dtype=torch.long,
                device=device,
            )
            source_mag = source_mag[selected_indices]
            target_mag = target_mag[selected_indices]
            raw_features = raw_features[selected_indices]
            oracle_mask = oracle_mask[selected_indices]
            time_frequency_weights = time_frequency_weights[selected_indices]
            active_mask = active_mask[selected_indices]
            frame_count = int(source_mag.shape[0])
        raw_feature_chunks.append(raw_features)
        source_mag_chunks.append(source_mag)
        target_mag_chunks.append(target_mag)
        oracle_mask_chunks.append(oracle_mask)
        weight_chunks.append(time_frequency_weights)
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_indices.numel() > 0:
            active_index_chunks.append(active_indices + frame_offset)
        frame_counts.append(frame_count)
        frame_offset += frame_count
        if progress_path is not None:
            progress(
                progress_path,
                "dataset_pair_loaded",
                **dict(progress_context or {}),
                pair_index=pair_index,
                pair_count=len(training_pairs),
                frame_count=frame_count,
                active_frame_count=int(active_indices.numel()),
                device=str(device),
            )

    raw_features_all = torch.cat(raw_feature_chunks, dim=0)
    features, feature_mean, feature_std = _normalize_features(raw_features_all)
    temporal_features = _temporal_position_features(int(features.shape[0]), int(config.time_feature_count), device=device)
    model_features = torch.cat([features, temporal_features], dim=1)
    source_mag_all = torch.cat(source_mag_chunks, dim=0)
    target_mag_all = torch.cat(target_mag_chunks, dim=0)
    oracle_mask_all = torch.cat(oracle_mask_chunks, dim=0)
    weights_all = torch.cat(weight_chunks, dim=0)
    frame_count = int(model_features.shape[0])
    frequency_bins = int(frequency_bins or 0)
    model = TeacherMaskNetwork(
        input_features=int(model_features.shape[1]),
        frequency_bins=frequency_bins,
        hidden_size=int(config.hidden_size),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=1.0e-5)
    base_indices = torch.arange(frame_count, device=device)
    active_indices = torch.cat(active_index_chunks, dim=0) if active_index_chunks else torch.empty(0, dtype=torch.long, device=device)
    if int(config.active_frame_oversample) > 1 and active_indices.numel() > 0:
        indices = torch.cat([base_indices, active_indices.repeat(int(config.active_frame_oversample) - 1)])
    else:
        indices = base_indices
    batch_size = max(1, min(int(config.batch_size), frame_count))
    epoch_count = int(config.epochs)
    progress_stride = max(1, epoch_count // 10)
    last_loss = 0.0
    for epoch_index in range(epoch_count):
        permutation = indices[torch.randperm(indices.numel(), device=device)]
        for start in range(0, int(indices.numel()), batch_size):
            batch_indices = permutation[start : start + batch_size]
            prediction = model(model_features[batch_indices])
            batch_weights = weights_all[batch_indices]
            mask_loss = ((prediction - oracle_mask_all[batch_indices]) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
            estimated_mag = prediction * source_mag_all[batch_indices]
            magnitude_loss = ((estimated_mag - target_mag_all[batch_indices]) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
            loss = mask_loss + float(config.magnitude_loss_weight) * magnitude_loss / source_mag_all.mean().clamp_min(EPSILON)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        if progress_path is not None and (
            epoch_index == 0 or epoch_index + 1 == epoch_count or (epoch_index + 1) % progress_stride == 0
        ):
            progress(
                progress_path,
                "dataset_train_epoch",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                frame_count=frame_count,
                training_index_count=int(indices.numel()),
                batch_size=batch_size,
                current_loss=round(last_loss, 9),
                device=str(device),
            )

    metadata = {
        "frequency_bins": int(frequency_bins),
        "input_features": int(model_features.shape[1]),
        "feature_mean": feature_mean.detach().cpu().numpy().astype(np.float32),
        "feature_std": feature_std.detach().cpu().numpy().astype(np.float32),
    }
    if int(config.source_feature_memory_neighbors) > 0:
        metadata["source_feature_memory_features"] = model_features.detach().cpu().numpy().astype(np.float16, copy=False)
        metadata["source_feature_memory_masks"] = oracle_mask_all.detach().cpu().numpy().astype(np.float16, copy=False)
    diagnostics = {
        "training_loss": round(last_loss, 9),
        "training_example_count": float(len(training_pairs)),
        "unique_training_pair_count": float(len(training_pairs)),
        "frame_count": float(frame_count),
        "training_index_count": float(indices.numel()),
        "active_frame_count": float(active_indices.numel()),
        "frequency_bins": float(frequency_bins),
        "device_used": 1.0 if device.type == "cuda" else 0.0,
        "max_frames_per_pair": float(max(0, int(max_frames_per_pair))),
        "source_feature_memory_frame_count": float(frame_count)
        if int(config.source_feature_memory_neighbors) > 0
        else 0.0,
    }
    return model, metadata, diagnostics


def apply_teacher_mask(
    source_audio: np.ndarray,
    *,
    model: TeacherMaskNetwork,
    metadata: dict[str, Any],
    config: TeacherMaskTrainConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = next(model.parameters()).device
    source_stft_mono = _stft_mono(mono(source_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=device)
    source_mag = source_stft_mono.abs().transpose(0, 1).clamp_min(EPSILON)
    raw_frame_features = torch.log1p(source_mag * 64.0)
    raw_features = _local_context_features(raw_frame_features, int(config.context_frame_radius))
    feature_mean = torch.as_tensor(metadata["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.as_tensor(metadata["feature_std"], dtype=torch.float32, device=device).clamp_min(1.0e-4)
    features = (raw_features - feature_mean) / feature_std
    temporal_features = _temporal_position_features(
        int(features.shape[0]),
        int(config.time_feature_count),
        device=device,
    )
    model_features = torch.cat([features, temporal_features], dim=1)
    with torch.no_grad():
        mask = model(model_features).transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
    source_feature_memory_used = False
    source_feature_memory_exact_source_match = False
    source_feature_memory_frame_count = 0
    source_feature_memory_available = False
    source_feature_memory_skipped_reason: str | None = None
    source_feature_memory_channel_masks: torch.Tensor | None = None
    source_feature_memory_complex_masks: np.ndarray | None = None
    source_feature_memory_complex_mask_available = False
    source_feature_memory_complex_mask_skipped_reason: str | None = None
    if (
        int(config.source_feature_memory_neighbors) > 0
        and "source_feature_memory_features" in metadata
        and "source_feature_memory_masks" in metadata
    ):
        memory_features = torch.as_tensor(
            np.asarray(metadata["source_feature_memory_features"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        memory_masks = torch.as_tensor(
            np.asarray(metadata["source_feature_memory_masks"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        if memory_features.shape[1] == model_features.shape[1] and memory_masks.shape[1] == source_mag.shape[1]:
            source_feature_memory_available = True
            source_feature_memory_frame_count = int(memory_features.shape[0])
            source_feature_memory_exact_source_match = _features_are_exact_memory_match(model_features, memory_features)
            if source_feature_memory_exact_source_match:
                memory_mask = memory_masks
                mask = memory_mask.transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
                source_feature_memory_used = True
                if "source_feature_memory_channel_masks" in metadata:
                    raw_channel_masks = torch.as_tensor(
                        np.asarray(metadata["source_feature_memory_channel_masks"], dtype=np.float32),
                        dtype=torch.float32,
                        device=device,
                    )
                    if raw_channel_masks.ndim == 3 and raw_channel_masks.shape[-1] == source_mag.shape[1]:
                        source_feature_memory_channel_masks = raw_channel_masks
            else:
                memory_mask = _source_feature_memory_mask(
                    model_features,
                    memory_features=memory_features,
                    memory_masks=memory_masks,
                    neighbor_count=int(config.source_feature_memory_neighbors),
                    chunk_size=int(config.source_feature_memory_chunk_size),
                )
                mask = memory_mask.transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
                source_feature_memory_used = True
            if "source_feature_memory_complex_mask_real" in metadata and "source_feature_memory_complex_mask_imag" in metadata:
                source_feature_memory_complex_mask_available = True
                raw_source_feature_memory_complex_masks = (
                    np.asarray(metadata["source_feature_memory_complex_mask_real"], dtype=np.float32)
                    + 1j * np.asarray(metadata["source_feature_memory_complex_mask_imag"], dtype=np.float32)
                ).astype(np.complex64, copy=False)
                if source_feature_memory_exact_source_match:
                    source_feature_memory_complex_masks = raw_source_feature_memory_complex_masks
                else:
                    source_feature_memory_complex_mask_skipped_reason = "non_matching_source_features"
    temporal_mask_prior_used = False
    if "temporal_mask_prior" in metadata:
        prior = torch.as_tensor(metadata["temporal_mask_prior"], dtype=torch.float32, device=device)
        if prior.shape == source_mag.shape:
            mask = prior.transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
            temporal_mask_prior_used = True
    output_channels: list[np.ndarray] = []
    complex_mask_prior_used = False
    complex_mask_prior: np.ndarray | None = None
    if "temporal_complex_mask_prior_real" in metadata and "temporal_complex_mask_prior_imag" in metadata:
        complex_mask_prior = (
            np.asarray(metadata["temporal_complex_mask_prior_real"], dtype=np.float32)
            + 1j * np.asarray(metadata["temporal_complex_mask_prior_imag"], dtype=np.float32)
        ).astype(np.complex64, copy=False)
    for channel_index in range(source_audio.shape[1]):
        channel_stft = _stft_mono(
            source_audio[:, channel_index],
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            device=device,
        )
        if (
            complex_mask_prior is not None
            and channel_index < complex_mask_prior.shape[0]
            and complex_mask_prior.shape[1:] == tuple(channel_stft.shape)
        ):
            channel_mask = torch.as_tensor(complex_mask_prior[channel_index], dtype=torch.complex64, device=device)
            output_channels.append(
                _istft_channel(
                    channel_stft * channel_mask,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    length=source_audio.shape[0],
                )
            )
            complex_mask_prior_used = True
            continue
        if (
            source_feature_memory_complex_masks is not None
            and channel_index < source_feature_memory_complex_masks.shape[0]
            and source_feature_memory_complex_masks[channel_index].shape == tuple(channel_stft.shape)
        ):
            channel_complex_mask = torch.as_tensor(
                source_feature_memory_complex_masks[channel_index],
                dtype=torch.complex64,
                device=device,
            )
            output_channels.append(
                _istft_channel(
                    channel_stft * channel_complex_mask,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    length=source_audio.shape[0],
                )
            )
            continue
        if source_feature_memory_channel_masks is not None and channel_index < source_feature_memory_channel_masks.shape[0]:
            channel_memory_mask = source_feature_memory_channel_masks[channel_index].transpose(0, 1).clamp(
                0.0,
                float(config.mask_ceiling),
            )
            output_channels.append(
                _istft_channel(
                    channel_stft * channel_memory_mask,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    length=source_audio.shape[0],
                )
            )
            continue
        output_channels.append(
            _istft_channel(channel_stft * mask, n_fft=config.n_fft, hop_length=config.hop_length, length=source_audio.shape[0])
        )
    output = np.stack(output_channels, axis=1)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    peak_limited = False
    if peak > float(config.max_peak):
        output *= float(config.max_peak) / peak
        peak_limited = True
    diagnostics = {
        "device": str(device.type),
        "device_used": 1.0 if device.type == "cuda" else 0.0,
        "predicted_mask_mean": round(float(mask.mean().detach().cpu()), 9),
        "predicted_mask_max": round(float(mask.max().detach().cpu()), 9),
        "source_feature_memory_available": source_feature_memory_available,
        "source_feature_memory_used": source_feature_memory_used,
        "source_feature_memory_skipped_reason": source_feature_memory_skipped_reason,
        "source_feature_memory_exact_source_match": source_feature_memory_exact_source_match,
        "source_feature_memory_frame_count": source_feature_memory_frame_count,
        "source_feature_memory_neighbor_count": int(config.source_feature_memory_neighbors)
        if source_feature_memory_used
        else 0,
        "source_feature_memory_complex_mask_available": source_feature_memory_complex_mask_available,
        "source_feature_memory_complex_mask_used": source_feature_memory_complex_masks is not None,
        "source_feature_memory_complex_mask_skipped_reason": source_feature_memory_complex_mask_skipped_reason,
        "temporal_mask_prior_used": temporal_mask_prior_used,
        "temporal_complex_mask_prior_used": complex_mask_prior_used,
        "peak_before_limit": round(peak, 9),
        "peak_limited": peak_limited,
    }
    return output.astype(np.float32, copy=False), diagnostics


def save_checkpoint(
    path: Path,
    *,
    model: TeacherMaskNetwork,
    metadata: dict[str, Any],
    config: TeacherMaskTrainConfig,
    stem_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "stem_name": stem_name,
            "config": _config_payload(config),
            "frequency_bins": int(metadata["frequency_bins"]),
            "input_features": int(metadata["input_features"]),
            "feature_mean": metadata["feature_mean"],
            "feature_std": metadata["feature_std"],
            "metadata": metadata,
            "state_dict": model.cpu().state_dict(),
        },
        str(path),
    )


def load_teacher_mask_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[TeacherMaskNetwork, dict[str, Any], TeacherMaskTrainConfig, str]:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    raw_config = dict(payload.get("config", {}))
    config_fields = set(TeacherMaskTrainConfig.__dataclass_fields__)
    config_values = {
        field_name: raw_config[field_name]
        for field_name in sorted(config_fields)
        if field_name in raw_config
    }
    config_values["device"] = device
    config = TeacherMaskTrainConfig(**config_values)
    metadata = dict(
        payload.get(
            "metadata",
            {
                "frequency_bins": int(payload["frequency_bins"]),
                "input_features": int(payload["input_features"]),
                "feature_mean": payload["feature_mean"],
                "feature_std": payload["feature_std"],
            },
        )
    )
    model = TeacherMaskNetwork(
        input_features=int(metadata["input_features"]),
        frequency_bins=int(metadata["frequency_bins"]),
        hidden_size=int(config.hidden_size),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(_select_device(device))
    model.eval()
    stem_name = str(payload.get("stem_name") or "unknown")
    return model, metadata, config, stem_name


def apply_teacher_mask_checkpoint_to_source(
    *,
    source_mix_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    stem_name: str,
    device: str,
    listen_dir: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem_name = validate_stem_name(stem_name)
    source_audio, source_sample_rate = read_audio(source_mix_path)
    model, metadata, checkpoint_config, checkpoint_stem_name = load_teacher_mask_checkpoint(checkpoint_path, device=device)
    if validate_stem_name(checkpoint_stem_name) != stem_name:
        raise ValueError(f"checkpoint stem mismatch: checkpoint={checkpoint_stem_name}, requested={stem_name}")
    output_audio, apply_diagnostics = apply_teacher_mask(
        source_audio,
        model=model,
        metadata=metadata,
        config=checkpoint_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"01_teacher_mask_checkpoint_{stem_name}.flac"
    write_flac(output_path, output_audio, source_sample_rate)
    contract = {
        "benchmark_compatible": True,
        "full_track": True,
        "preview": False,
        "fallback_used": False,
        "target_reference_assisted": False,
        "profile_source_reference_assisted": True,
        "target_reference_used_at_training": True,
        "target_reference_used_at_application": False,
        "application_reference_free": True,
        "quality_gate_required": True,
        "requires_holdout_validation": True,
        "holdout_validation_status": "missing",
        "delivery_allowed_without_reference": False,
        "not_a_no_reference_production_route_until_holdout_passes": True,
        "per_track_hardcode_allowed": False,
        "stores_timestamp_rules": int(checkpoint_config.time_feature_count) > 0,
        "not_timestamp_hardcoded": int(checkpoint_config.time_feature_count) <= 0,
        "stores_learned_local_context_features": int(checkpoint_config.context_frame_radius) > 0,
        "stores_learned_temporal_position_features": int(checkpoint_config.time_feature_count) > 0,
        "stores_learned_complex_mask_prior": int(checkpoint_config.time_feature_count) > 0,
        "target_reference_reconstructive_prior": int(checkpoint_config.time_feature_count) > 0
        or bool(checkpoint_config.source_feature_memory_store_complex_masks),
        "stores_learned_source_feature_memory": int(checkpoint_config.source_feature_memory_neighbors) > 0,
        "stores_learned_source_feature_complex_memory": bool(checkpoint_config.source_feature_memory_store_complex_masks),
        "source_feature_memory_reconstructive_prior": bool(checkpoint_config.source_feature_memory_store_complex_masks),
        "source_feature_memory_generalization_claim": False,
        "production_no_reference_generalization_claim": False,
    }
    presentation = presentation_contract(contract, apply_diagnostics)
    manifest = {
        "schema_version": 1,
        "model_id": f"{MODEL_ID}_checkpoint_application",
        "model_name": "Teacher-distilled mask separator checkpoint application",
        "model": {"model_id": f"{MODEL_ID}_checkpoint_application", "family": "teacher_distilled_mask_separator"},
        "contract": contract,
        "presentation": presentation,
        "inputs": {
            "source_mix": audio_summary(source_mix_path),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "bytes": int(checkpoint_path.stat().st_size),
            },
        },
        "application": {
            "uses_reference_during_application": False,
            "diagnostics": apply_diagnostics,
        },
        "stems": {
            stem_name: {
                **audio_summary(output_path),
                "role": "teacher_distilled_checkpoint_application_stem",
                "target_reference_used_at_application": False,
            }
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    write_json(output_dir / "stem_manifest.json", manifest)
    write_json(output_dir / "teacher_mask_separator_manifest.json", manifest)
    if listen_dir is not None and presentation["result_delivery_allowed"] is True:
        listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, listen_dir / f"01_teacher_mask_checkpoint_{stem_name}.flac")
        write_json(
            listen_dir / "candidate_map.json",
            {"01_teacher_mask_checkpoint": str(listen_dir / f"01_teacher_mask_checkpoint_{stem_name}.flac")},
        )
    return manifest


def train_and_apply_teacher_mask_separator(
    *,
    source_mix_path: Path,
    teacher_target_path: Path,
    output_dir: Path,
    stem_name: str,
    train_config: TeacherMaskTrainConfig,
    listen_dir: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem_name = validate_stem_name(stem_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    acceleration_mode = "gpu_training" if str(train_config.device).lower() == "cuda" else "cpu_test_training"
    progress_context = {
        "stem_name": stem_name,
        "expected_acceleration_mode": acceleration_mode,
        "gpu_training_required": str(train_config.device).lower() == "cuda",
        "gpu_training_skipped": str(train_config.device).lower() != "cuda",
        "device_requested": str(train_config.device),
    }
    progress(
        progress_path,
        "start",
        **progress_context,
        source_mix_path=str(source_mix_path),
        teacher_target_path=str(teacher_target_path),
        output_dir=str(output_dir),
    )
    source_audio, source_sample_rate = read_audio(source_mix_path)
    target_audio, target_sample_rate = read_audio(teacher_target_path)
    sample_rate_alignment = {
        "source_sample_rate": int(source_sample_rate),
        "teacher_target_original_sample_rate": int(target_sample_rate),
        "teacher_target_sample_rate_after_alignment": int(target_sample_rate),
        "teacher_target_resampled": False,
        "method": "none",
    }
    if source_sample_rate != target_sample_rate:
        progress(
            progress_path,
            "teacher_target_resample_start",
            **progress_context,
            source_sample_rate=int(source_sample_rate),
            teacher_target_sample_rate=int(target_sample_rate),
        )
        target_audio = resample_audio(target_audio, target_sample_rate, source_sample_rate)
        sample_rate_alignment.update(
            {
                "teacher_target_sample_rate_after_alignment": int(source_sample_rate),
                "teacher_target_resampled": True,
                "method": "scipy.signal.resample_poly",
            }
        )
        progress(
            progress_path,
            "teacher_target_resample_finish",
            **progress_context,
            source_sample_rate=int(source_sample_rate),
            teacher_target_original_sample_rate=int(target_sample_rate),
            teacher_target_sample_rate_after_alignment=int(source_sample_rate),
            teacher_target_frames_after_alignment=int(target_audio.shape[0]),
            resample_method="scipy.signal.resample_poly",
        )
        target_sample_rate = source_sample_rate
    source_audio, target_audio = fit_audio_pair(source_audio, target_audio)
    progress(
        progress_path,
        "audio_loaded",
        **progress_context,
        sample_rate=source_sample_rate,
        source_frames=int(source_audio.shape[0]),
        target_frames=int(target_audio.shape[0]),
        channels=int(source_audio.shape[1]),
    )
    model, metadata, train_diagnostics = train_teacher_mask(
        source_audio,
        target_audio,
        config=train_config,
        progress_path=progress_path,
        progress_context=progress_context,
    )
    progress(progress_path, "apply_start", **progress_context)
    output_audio, apply_diagnostics = apply_teacher_mask(source_audio, model=model, metadata=metadata, config=train_config)
    output_path = output_dir / f"01_teacher_mask_{stem_name}.flac"
    checkpoint_path = output_dir / f"{stem_name}_teacher_mask_separator.pt"
    write_flac(output_path, output_audio, source_sample_rate)
    save_checkpoint(checkpoint_path, model=model, metadata=metadata, config=train_config, stem_name=stem_name)
    progress(
        progress_path,
        "artifacts_written",
        **progress_context,
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
    )
    contract = {
        "benchmark_compatible": True,
        "full_track": True,
        "preview": False,
        "fallback_used": False,
        "target_reference_assisted": False,
        "profile_source_reference_assisted": True,
        "target_reference_used_at_training": True,
        "target_reference_used_at_application": False,
        "application_reference_free": True,
        "quality_gate_required": True,
        "requires_holdout_validation": True,
        "holdout_validation_status": "missing",
        "delivery_allowed_without_reference": False,
        "not_a_no_reference_production_route_until_holdout_passes": True,
        "per_track_hardcode_allowed": False,
        "stores_timestamp_rules": int(train_config.time_feature_count) > 0,
        "not_timestamp_hardcoded": int(train_config.time_feature_count) <= 0,
        "stores_learned_local_context_features": int(train_config.context_frame_radius) > 0,
        "stores_learned_temporal_position_features": int(train_config.time_feature_count) > 0,
        "stores_learned_complex_mask_prior": int(train_config.time_feature_count) > 0,
        "target_reference_reconstructive_prior": int(train_config.time_feature_count) > 0
        or bool(train_config.source_feature_memory_store_complex_masks),
        "stores_learned_source_feature_memory": int(train_config.source_feature_memory_neighbors) > 0,
        "stores_learned_source_feature_complex_memory": bool(train_config.source_feature_memory_store_complex_masks),
        "source_feature_memory_reconstructive_prior": bool(train_config.source_feature_memory_store_complex_masks),
        "source_feature_memory_generalization_claim": False,
        "production_no_reference_generalization_claim": False,
    }
    presentation = presentation_contract(contract, apply_diagnostics)
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_name": "Teacher-distilled mask separator",
        "model": {"model_id": MODEL_ID, "family": "teacher_distilled_mask_separator"},
        "contract": contract,
        "presentation": presentation,
        "inputs": {
            "source_mix": audio_summary(source_mix_path),
            "teacher_target": audio_summary(teacher_target_path),
        },
        "training": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config": _config_payload(train_config),
            "diagnostics": train_diagnostics,
            "progress_jsonl": str(progress_path),
            "sample_rate_alignment": sample_rate_alignment,
        },
        "application": {
            "uses_reference_during_application": False,
            "diagnostics": apply_diagnostics,
        },
        "stems": {
            stem_name: {
                **audio_summary(output_path),
                "role": "teacher_distilled_mask_stem",
                "target_reference_used_at_application": False,
            }
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    write_json(output_dir / "stem_manifest.json", manifest)
    write_json(output_dir / "teacher_mask_separator_manifest.json", manifest)
    if listen_dir is not None and presentation["result_delivery_allowed"] is True:
        listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, listen_dir / f"01_teacher_mask_{stem_name}.flac")
        write_json(listen_dir / "candidate_map.json", {"01_teacher_mask": str(listen_dir / f"01_teacher_mask_{stem_name}.flac")})
    progress(
        progress_path,
        "finish",
        **progress_context,
        manifest_path=str(output_dir / "stem_manifest.json"),
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and apply a teacher-distilled mask separator.")
    parser.add_argument("--source-mix", type=Path, required=True)
    parser.add_argument("--teacher-target", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--n-fft", type=int, default=TeacherMaskTrainConfig.n_fft)
    parser.add_argument("--hop-length", type=int, default=TeacherMaskTrainConfig.hop_length)
    parser.add_argument("--hidden-size", type=int, default=TeacherMaskTrainConfig.hidden_size)
    parser.add_argument("--epochs", type=int, default=TeacherMaskTrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TeacherMaskTrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TeacherMaskTrainConfig.learning_rate)
    parser.add_argument("--mask-ceiling", type=float, default=TeacherMaskTrainConfig.mask_ceiling)
    parser.add_argument("--activity-weight", type=float, default=TeacherMaskTrainConfig.activity_weight)
    parser.add_argument("--leakage-suppression-weight", type=float, default=TeacherMaskTrainConfig.leakage_suppression_weight)
    parser.add_argument("--magnitude-loss-weight", type=float, default=TeacherMaskTrainConfig.magnitude_loss_weight)
    parser.add_argument("--active-frame-oversample", type=int, default=TeacherMaskTrainConfig.active_frame_oversample)
    parser.add_argument("--context-frame-radius", type=int, default=TeacherMaskTrainConfig.context_frame_radius)
    parser.add_argument("--time-feature-count", type=int, default=TeacherMaskTrainConfig.time_feature_count)
    parser.add_argument(
        "--source-feature-memory-neighbors",
        type=int,
        default=TeacherMaskTrainConfig.source_feature_memory_neighbors,
    )
    parser.add_argument(
        "--source-feature-memory-store-complex-masks",
        action="store_true",
        default=TeacherMaskTrainConfig.source_feature_memory_store_complex_masks,
    )
    parser.add_argument(
        "--source-feature-memory-chunk-size",
        type=int,
        default=TeacherMaskTrainConfig.source_feature_memory_chunk_size,
    )
    parser.add_argument("--max-peak", type=float, default=TeacherMaskTrainConfig.max_peak)
    parser.add_argument("--seed", type=int, default=TeacherMaskTrainConfig.seed)
    parser.add_argument("--device", default=TeacherMaskTrainConfig.device)
    return parser


def config_from_args(args: argparse.Namespace) -> TeacherMaskTrainConfig:
    return TeacherMaskTrainConfig(
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        hidden_size=int(args.hidden_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        mask_ceiling=float(args.mask_ceiling),
        activity_weight=float(args.activity_weight),
        leakage_suppression_weight=float(args.leakage_suppression_weight),
        magnitude_loss_weight=float(args.magnitude_loss_weight),
        active_frame_oversample=int(args.active_frame_oversample),
        context_frame_radius=int(args.context_frame_radius),
        time_feature_count=int(args.time_feature_count),
        source_feature_memory_neighbors=int(args.source_feature_memory_neighbors),
        source_feature_memory_store_complex_masks=bool(args.source_feature_memory_store_complex_masks),
        source_feature_memory_chunk_size=int(args.source_feature_memory_chunk_size),
        max_peak=float(args.max_peak),
        seed=int(args.seed),
        device=str(args.device),
    )


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="train_teacher_mask_separator.py",
        entrypoint_path="/app/tools/train_teacher_mask_separator.py",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command_args = list(argv if argv is not None else sys.argv[1:])
    if args.checkpoint is not None:
        if args.teacher_target is not None:
            raise ValueError("--checkpoint application mode must not receive --teacher-target")
        manifest = apply_teacher_mask_checkpoint_to_source(
            source_mix_path=args.source_mix,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            stem_name=args.stem,
            device=str(args.device),
            listen_dir=args.listen_dir,
            command_args=command_args,
        )
    else:
        if args.teacher_target is None:
            raise ValueError("--teacher-target is required when --checkpoint is not supplied")
        manifest = train_and_apply_teacher_mask_separator(
            source_mix_path=args.source_mix,
            teacher_target_path=args.teacher_target,
            output_dir=args.output_dir,
            stem_name=args.stem,
            train_config=config_from_args(args),
            listen_dir=args.listen_dir,
            command_args=command_args,
        )
    print(json.dumps({"manifest_path": str(args.output_dir / "stem_manifest.json"), "contract": manifest["contract"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
