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
from typing import Any, Callable, Sequence

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
DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS = 50_000_000_000
DEFAULT_MIN_CUDA_FREE_MEMORY_MB = 4096
MASK_OUTPUT_MODE_MAGNITUDE = "magnitude"
MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL = "magnitude_residual"
MASK_OUTPUT_MODE_COMPLEX = "complex"
MASK_OUTPUT_MODES = {MASK_OUTPUT_MODE_MAGNITUDE, MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL, MASK_OUTPUT_MODE_COMPLEX}
CONDITIONING_FEATURE_SCHEMA = "symmetric_vocal_proxy_log_consensus_v1"
ProgressCallback = Callable[[str, dict[str, Any]], None]
ACCELERATOR_DEVICE_TYPES = {"cuda", "mps"}


def is_accelerator_device(device: torch.device | str) -> bool:
    return str(device).split(":", 1)[0].strip().lower() in ACCELERATOR_DEVICE_TYPES


def acceleration_mode(device: torch.device | str, *, phase: str) -> str:
    device_type = str(device).split(":", 1)[0].strip().lower()
    if device_type == "cuda":
        return f"gpu_{phase}"
    if device_type == "mps":
        return f"mps_gpu_{phase}"
    return f"cpu_diagnostic_{phase}"


def validate_mask_output_mode(value: str) -> str:
    mode = str(value or MASK_OUTPUT_MODE_MAGNITUDE).strip().lower()
    if mode not in MASK_OUTPUT_MODES:
        raise ValueError(f"unsupported mask output mode: {value!r}")
    return mode


def magnitude_residual_deviation(mask_ceiling: float) -> float:
    deviation = float(mask_ceiling) - 1.0
    if deviation <= 0.0:
        raise ValueError("magnitude_residual mask output mode requires mask_ceiling > 1.0")
    return deviation


@dataclass(frozen=True)
class TeacherMaskTrainConfig:
    n_fft: int = 2048
    hop_length: int = 512
    hidden_size: int = 384
    epochs: int = 160
    batch_size: int = 512
    learning_rate: float = 0.002
    mask_ceiling: float = 0.995
    mask_output_mode: str = MASK_OUTPUT_MODE_MAGNITUDE
    complex_mask_ceiling: float = 2.0
    activity_weight: float = 8.0
    leakage_suppression_weight: float = 2.0
    magnitude_loss_weight: float = 0.25
    source_preservation_weight: float = 0.0
    active_frame_oversample: int = 1
    context_frame_radius: int = 0
    time_feature_count: int = 0
    source_feature_memory_neighbors: int = 0
    source_feature_memory_store_complex_masks: bool = False
    source_feature_memory_chunk_size: int = 512
    source_feature_memory_projection_dim: int = 0
    source_feature_memory_max_frames: int = 0
    min_cuda_free_memory_mb: int = DEFAULT_MIN_CUDA_FREE_MEMORY_MB
    max_peak: float = 0.999
    seed: int = 20260518
    device: str = "cuda"


class TeacherMaskNetwork(nn.Module):
    def __init__(
        self,
        input_features: int,
        frequency_bins: int,
        hidden_size: int,
        *,
        mask_output_mode: str = MASK_OUTPUT_MODE_MAGNITUDE,
        mask_ceiling: float = 1.0,
        complex_mask_ceiling: float = 2.0,
    ) -> None:
        super().__init__()
        self.mask_output_mode = validate_mask_output_mode(mask_output_mode)
        self.frequency_bins = int(frequency_bins)
        self.mask_ceiling = float(mask_ceiling)
        self.complex_mask_ceiling = float(complex_mask_ceiling)
        if self.mask_output_mode == MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL:
            magnitude_residual_deviation(self.mask_ceiling)
        output_features = self.frequency_bins * 2 if self.mask_output_mode == MASK_OUTPUT_MODE_COMPLEX else self.frequency_bins
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_features),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.layers(features)
        if self.mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
            return torch.tanh(raw) * float(self.complex_mask_ceiling)
        if self.mask_output_mode == MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL:
            return 1.0 + torch.tanh(raw) * magnitude_residual_deviation(self.mask_ceiling)
        return torch.sigmoid(raw) * float(self.mask_ceiling)


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


def emit_progress(callback: ProgressCallback | None, event: str, **fields: Any) -> None:
    if callback is not None:
        callback(event, dict(fields))


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


def fit_audio_frames(audio: np.ndarray, frame_count: int) -> np.ndarray:
    target_frames = int(frame_count)
    if target_frames <= 0:
        raise ValueError("frame_count must be positive")
    if audio.shape[0] >= target_frames:
        return audio[:target_frames]
    padding = np.zeros((target_frames - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, padding], axis=0)


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
    normalized = str(requested_device or "").strip().lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for teacher-mask training but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if normalized == "mps":
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            raise RuntimeError("MPS was requested for teacher-mask training but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if normalized != "cpu":
        raise ValueError(f"Unsupported device: {requested_device}")
    return torch.device("cpu")


def _require_cuda_memory_available(
    device: torch.device,
    *,
    min_free_memory_mb: int,
    progress_path: Path | None = None,
    progress_context: dict[str, Any] | None = None,
    phase: str,
) -> dict[str, int]:
    if device.type != "cuda" or int(min_free_memory_mb) <= 0:
        return {
            "free_memory_mb": 0,
            "total_memory_mb": 0,
            "required_free_memory_mb": max(0, int(min_free_memory_mb)),
        }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime failure mode.
        if progress_path is not None:
            progress(
                progress_path,
                "cuda_memory_guard_failed",
                **dict(progress_context or {}),
                phase=str(phase),
                required_free_memory_mb=int(min_free_memory_mb),
                error=str(exc),
            )
        raise RuntimeError(f"CUDA memory guard failed during {phase}: {exc}") from exc
    free_mb = int(free_bytes // (1024 * 1024))
    total_mb = int(total_bytes // (1024 * 1024))
    snapshot = {
        "free_memory_mb": free_mb,
        "total_memory_mb": total_mb,
        "required_free_memory_mb": int(min_free_memory_mb),
    }
    if progress_path is not None:
        progress(
            progress_path,
            "cuda_memory_guard_checked",
            **dict(progress_context or {}),
            phase=str(phase),
            **snapshot,
        )
    if free_mb < int(min_free_memory_mb):
        if progress_path is not None:
            progress(
                progress_path,
                "cuda_memory_guard_blocked",
                **dict(progress_context or {}),
                phase=str(phase),
                **snapshot,
            )
        raise RuntimeError(
            "CUDA free memory is below required minimum: "
            f"free_memory_mb={free_mb}, required_free_memory_mb={int(min_free_memory_mb)}, phase={phase}"
        )
    return snapshot


def _progress_train_batch_phase(
    progress_path: Path | None,
    event: str,
    progress_context: dict[str, Any] | None,
    *,
    epoch: int,
    epoch_count: int,
    batch_index: int,
    batch_count: int,
    batch_start_index: int,
    batch_end_index: int,
    training_index_count: int,
    batch_frame_count: int,
    device: torch.device,
    **fields: Any,
) -> None:
    if progress_path is None:
        return
    progress(
        progress_path,
        event,
        **dict(progress_context or {}),
        epoch=int(epoch),
        epoch_count=int(epoch_count),
        batch_index=int(batch_index),
        batch_count=int(batch_count),
        batch_start_index=int(batch_start_index),
        batch_end_index=int(batch_end_index),
        training_index_count=int(training_index_count),
        batch_frame_count=int(batch_frame_count),
        device=str(device),
        **fields,
    )


def _synchronize_cuda_phase(
    progress_path: Path | None,
    progress_context: dict[str, Any] | None,
    *,
    event_prefix: str,
    device: torch.device,
    **fields: Any,
) -> None:
    if progress_path is None or not is_accelerator_device(device):
        return
    accelerator = str(device.type)
    progress(
        progress_path,
        f"{event_prefix}_{accelerator}_sync_start",
        **dict(progress_context or {}),
        device=str(device),
        **fields,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.mps.synchronize()
    progress(
        progress_path,
        f"{event_prefix}_{accelerator}_sync_finish",
        **dict(progress_context or {}),
        device=str(device),
        **fields,
    )


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
    ceiling: float,
) -> np.ndarray:
    channel_masks: list[np.ndarray] = []
    limit = float(ceiling)
    for channel_index in range(source_audio.shape[1]):
        source_stft = _stft_mono(source_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length, device=device)
        target_stft = _stft_mono(target_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length, device=device)
        safe_source = torch.where(source_stft.abs() > EPSILON, source_stft, torch.ones_like(source_stft))
        complex_mask = torch.where(source_stft.abs() > EPSILON, target_stft / safe_source, torch.zeros_like(source_stft))
        if limit > 0.0:
            scale = (limit / complex_mask.abs().clamp_min(EPSILON)).clamp(max=1.0)
            complex_mask = complex_mask * scale
        complex_mask = torch.nan_to_num(complex_mask, nan=0.0, posinf=0.0, neginf=0.0)
        channel_masks.append(complex_mask.detach().cpu().numpy().astype(np.complex64, copy=False))
    return np.stack(channel_masks, axis=0)


def _complex_ratio_mask_frames(
    source_stft: torch.Tensor,
    target_stft: torch.Tensor,
    *,
    ceiling: float,
) -> torch.Tensor:
    denominator = source_stft.abs().pow(2).clamp_min(EPSILON)
    ratio = target_stft * torch.conj(source_stft) / denominator
    limit = float(ceiling)
    if limit > 0.0:
        scale = (limit / ratio.abs().clamp_min(EPSILON)).clamp(max=1.0)
        ratio = ratio * scale
    return ratio.transpose(0, 1)


def _complex_prediction_frames(prediction: torch.Tensor, frequency_bins: int) -> torch.Tensor:
    bins = int(frequency_bins)
    if prediction.ndim != 2 or int(prediction.shape[1]) != bins * 2:
        raise ValueError(f"complex mask prediction shape mismatch: expected second dim {bins * 2}, got {tuple(prediction.shape)}")
    return torch.complex(prediction[:, :bins], prediction[:, bins:])


def _weighted_complex_mse(delta: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((delta.real.square() + delta.imag.square()) * weights).sum() / weights.sum().clamp_min(EPSILON)


def _scoped_source_preservation_loss(
    delta: torch.Tensor,
    weights: torch.Tensor,
    scope: torch.Tensor,
) -> torch.Tensor:
    scoped_weights = weights * scope
    squared_error = delta.real.square() + delta.imag.square() if delta.is_complex() else delta.square()
    return (squared_error * scoped_weights).sum() / scoped_weights.sum().clamp_min(EPSILON)


def _normalize_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (features - mean) / std, mean, std


def _feature_mean_std_from_chunks(chunks: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not chunks:
        raise ValueError("feature chunks must contain at least one tensor")
    feature_width = int(chunks[0].shape[1])
    total = torch.zeros(feature_width, dtype=torch.float64, device=torch.device("cpu"))
    total_sq = torch.zeros(feature_width, dtype=torch.float64, device=torch.device("cpu"))
    total_count = 0
    for chunk in chunks:
        if int(chunk.shape[1]) != feature_width:
            raise ValueError("all feature chunks must have the same feature width")
        chunk_float = chunk.detach().cpu().to(torch.float32)
        total += chunk_float.sum(dim=0, dtype=torch.float64)
        total_sq += chunk_float.square().sum(dim=0, dtype=torch.float64)
        total_count += int(chunk_float.shape[0])
        del chunk_float
    if total_count <= 0:
        raise ValueError("feature chunks must contain at least one frame")
    mean = total / float(total_count)
    if total_count > 1:
        variance = (total_sq - (total * total / float(total_count))) / float(total_count - 1)
    else:
        variance = torch.zeros_like(total)
    std = torch.sqrt(variance.clamp_min(1.0e-8)).clamp_min(1.0e-4)
    return mean.to(torch.float32).unsqueeze(0), std.to(torch.float32).unsqueeze(0)


def _evenly_spaced_long_indices(*, length: int, count: int, device: torch.device) -> torch.Tensor:
    length_int = int(length)
    count_int = int(count)
    if length_int <= 0 or count_int <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if count_int >= length_int:
        return torch.arange(length_int, dtype=torch.long, device=device)
    if count_int == 1:
        return torch.zeros((1,), dtype=torch.long, device=device)
    positions = torch.arange(count_int, dtype=torch.long, device=device)
    return (positions * (length_int - 1)).div(count_int - 1, rounding_mode="floor")


def safe_quantile(values: torch.Tensor, quantile: float, *, max_values: int = 1_000_000) -> torch.Tensor:
    flat_values = values.reshape(-1)
    if flat_values.numel() > int(max_values):
        sample_indices = _evenly_spaced_long_indices(
            length=int(flat_values.numel()),
            count=int(max_values),
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


def _conditioned_local_context_features(
    source_magnitude_frames: torch.Tensor,
    conditioning_magnitude_frames: Sequence[torch.Tensor],
    context_frame_radius: int,
) -> torch.Tensor:
    source_log = torch.log1p(source_magnitude_frames * 64.0)
    if not conditioning_magnitude_frames:
        return _local_context_features(source_log, context_frame_radius)
    for conditioning_magnitude in conditioning_magnitude_frames:
        if conditioning_magnitude.shape != source_magnitude_frames.shape:
            raise ValueError(
                "conditioning magnitude shape mismatch: "
                f"source={tuple(source_magnitude_frames.shape)}, conditioning={tuple(conditioning_magnitude.shape)}"
            )
    conditioning_logs = torch.stack(
        [torch.log1p(conditioning_magnitude.clamp_min(EPSILON) * 64.0) for conditioning_magnitude in conditioning_magnitude_frames],
        dim=0,
    )
    consensus_log = conditioning_logs.mean(dim=0)
    disagreement_log = conditioning_logs.amax(dim=0) - conditioning_logs.amin(dim=0)
    frame_features = torch.cat([source_log, consensus_log, disagreement_log], dim=1)
    return _local_context_features(frame_features, context_frame_radius)


def _config_payload(config: TeacherMaskTrainConfig) -> dict[str, Any]:
    return {
        "n_fft": int(config.n_fft),
        "hop_length": int(config.hop_length),
        "hidden_size": int(config.hidden_size),
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "mask_ceiling": float(config.mask_ceiling),
        "mask_output_mode": validate_mask_output_mode(config.mask_output_mode),
        "complex_mask_ceiling": float(config.complex_mask_ceiling),
        "activity_weight": float(config.activity_weight),
        "leakage_suppression_weight": float(config.leakage_suppression_weight),
        "magnitude_loss_weight": float(config.magnitude_loss_weight),
        "source_preservation_weight": float(config.source_preservation_weight),
        "active_frame_oversample": int(config.active_frame_oversample),
        "context_frame_radius": int(config.context_frame_radius),
        "time_feature_count": int(config.time_feature_count),
        "source_feature_memory_neighbors": int(config.source_feature_memory_neighbors),
        "source_feature_memory_store_complex_masks": bool(config.source_feature_memory_store_complex_masks),
        "source_feature_memory_chunk_size": int(config.source_feature_memory_chunk_size),
        "source_feature_memory_projection_dim": int(config.source_feature_memory_projection_dim),
        "source_feature_memory_max_frames": int(config.source_feature_memory_max_frames),
        "min_cuda_free_memory_mb": int(config.min_cuda_free_memory_mb),
        "max_peak": float(config.max_peak),
        "seed": int(config.seed),
        "device": str(config.device),
    }


def _source_feature_memory_projection_matrix(
    *,
    feature_width: int,
    projection_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor | None:
    dim = int(projection_dim)
    width = int(feature_width)
    if dim <= 0 or dim >= width:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 7919)
    matrix = torch.randn((width, dim), generator=generator, dtype=torch.float32) / math.sqrt(float(dim))
    return matrix.to(device=device)


def _source_feature_memory_selected_indices(*, frame_count: int, max_frames: int, device: torch.device) -> torch.Tensor | None:
    cap = int(max_frames)
    frames = int(frame_count)
    if cap <= 0 or frames <= cap:
        return None
    return _evenly_spaced_long_indices(length=frames, count=cap, device=device)


def _project_source_feature_memory_features(
    model_features: torch.Tensor,
    projection_matrix: torch.Tensor,
) -> torch.Tensor:
    return model_features.to(dtype=torch.float32) @ projection_matrix.to(
        device=model_features.device,
        dtype=torch.float32,
    )


def _source_feature_memory_mask(
    query_features: torch.Tensor,
    *,
    memory_features: torch.Tensor,
    memory_masks: torch.Tensor,
    neighbor_count: int,
    chunk_size: int,
    progress_callback: ProgressCallback | None = None,
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
    query_frame_count = int(query_features.shape[0])
    chunk_count = int(math.ceil(query_frame_count / chunk)) if query_frame_count else 0
    progress_stride = max(1, chunk_count // 10) if chunk_count else 1
    for start in range(0, int(query_features.shape[0]), chunk):
        query_chunk = query_features[start : start + chunk]
        distances = torch.cdist(query_chunk, memory_features, p=2.0)
        nearest_distances, nearest_indices = torch.topk(distances, k=k, dim=1, largest=False)
        nearest_masks = memory_masks[nearest_indices]
        chunk_index = (start // chunk) + 1
        if chunk_index == 1 or chunk_index == chunk_count or chunk_index % progress_stride == 0:
            emit_progress(
                progress_callback,
                "source_feature_memory_search_chunk",
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                chunk_start_frame=int(start),
                chunk_end_frame=int(min(start + chunk, query_frame_count)),
            )
        if k == 1:
            predictions.append(nearest_masks[:, 0, :])
            continue
        weights = torch.softmax(-nearest_distances.clamp_max(80.0), dim=1).unsqueeze(-1)
        predictions.append((nearest_masks * weights).sum(dim=1))
    return torch.cat(predictions, dim=0)


def _source_feature_memory_complex_mask(
    query_features: torch.Tensor,
    *,
    memory_features: torch.Tensor,
    memory_complex_masks: torch.Tensor,
    neighbor_count: int,
    chunk_size: int,
    progress_callback: ProgressCallback | None = None,
) -> torch.Tensor:
    if query_features.ndim != 2 or memory_features.ndim != 2 or memory_complex_masks.ndim != 2:
        raise ValueError("source-feature complex memory tensors must be 2D")
    if query_features.shape[1] != memory_features.shape[1]:
        raise ValueError("source-feature complex memory feature width mismatch")
    if memory_features.shape[0] != memory_complex_masks.shape[0]:
        raise ValueError("source-feature complex memory frame count mismatch")
    if memory_features.shape[0] == 0:
        raise ValueError("source-feature complex memory is empty")

    k = max(1, min(int(neighbor_count), int(memory_features.shape[0])))
    chunk = max(1, int(chunk_size))
    memory_features = memory_features.to(dtype=torch.float32)
    memory_complex_masks = memory_complex_masks.to(dtype=torch.complex64)
    query_features = query_features.to(dtype=torch.float32)
    predictions: list[torch.Tensor] = []
    query_frame_count = int(query_features.shape[0])
    chunk_count = int(math.ceil(query_frame_count / chunk)) if query_frame_count else 0
    progress_stride = max(1, chunk_count // 10) if chunk_count else 1
    for start in range(0, int(query_features.shape[0]), chunk):
        query_chunk = query_features[start : start + chunk]
        distances = torch.cdist(query_chunk, memory_features, p=2.0)
        nearest_distances, nearest_indices = torch.topk(distances, k=k, dim=1, largest=False)
        nearest_masks = memory_complex_masks[nearest_indices]
        chunk_index = (start // chunk) + 1
        if chunk_index == 1 or chunk_index == chunk_count or chunk_index % progress_stride == 0:
            emit_progress(
                progress_callback,
                "source_feature_memory_complex_search_chunk",
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                chunk_start_frame=int(start),
                chunk_end_frame=int(min(start + chunk, query_frame_count)),
            )
        if k == 1:
            predictions.append(nearest_masks[:, 0, :])
            continue
        weights = torch.softmax(-nearest_distances.clamp_max(80.0), dim=1).to(dtype=torch.float32).unsqueeze(-1)
        predictions.append((nearest_masks * weights).sum(dim=1))
    return torch.cat(predictions, dim=0)


def _features_are_exact_memory_match(query_features: torch.Tensor, memory_features: torch.Tensor) -> bool:
    if query_features.shape != memory_features.shape:
        return False
    max_abs_delta = torch.max(torch.abs(query_features.to(dtype=torch.float32) - memory_features.to(dtype=torch.float32)))
    return bool(float(max_abs_delta.detach().cpu()) <= 5.0e-2)


def _find_exact_memory_subsequence(
    query_features: torch.Tensor,
    memory_features: torch.Tensor,
    *,
    tolerance: float = 5.0e-2,
) -> int | None:
    if query_features.ndim != 2 or memory_features.ndim != 2:
        return None
    query_frame_count, feature_width = query_features.shape
    memory_frame_count, memory_feature_width = memory_features.shape
    if (
        int(query_frame_count) <= 0
        or int(memory_frame_count) <= 0
        or int(feature_width) != int(memory_feature_width)
        or int(query_frame_count) > int(memory_frame_count)
    ):
        return None
    if int(query_frame_count) == int(memory_frame_count):
        return 0 if _features_are_exact_memory_match(query_features, memory_features) else None

    query = query_features.to(dtype=torch.float32)
    memory = memory_features.to(dtype=torch.float32)
    candidate_count = int(memory_frame_count) - int(query_frame_count) + 1
    first_frame_delta = torch.max(torch.abs(memory[:candidate_count] - query[0].unsqueeze(0)), dim=1).values
    candidate_starts = torch.nonzero(first_frame_delta <= float(tolerance), as_tuple=False).flatten()
    for raw_start in candidate_starts.detach().cpu().tolist():
        start = int(raw_start)
        candidate = memory[start : start + int(query_frame_count)]
        max_abs_delta = torch.max(torch.abs(candidate - query))
        if bool(float(max_abs_delta.detach().cpu()) <= float(tolerance)):
            return start
    return None


def train_teacher_mask(
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    *,
    config: TeacherMaskTrainConfig,
    progress_path: Path | None = None,
    progress_context: dict[str, Any] | None = None,
) -> tuple[TeacherMaskNetwork, dict[str, Any], dict[str, float]]:
    mask_output_mode = validate_mask_output_mode(config.mask_output_mode)
    if mask_output_mode == MASK_OUTPUT_MODE_COMPLEX and int(config.source_feature_memory_neighbors) > 0:
        raise ValueError("complex mask output mode does not support source-feature memory")
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = _select_device(config.device)
    if device.type == "mps":
        return train_teacher_mask_dataset(
            [(source_audio, target_audio)],
            config=config,
            progress_path=progress_path,
            progress_context=progress_context,
        )
    _require_cuda_memory_available(
        device,
        min_free_memory_mb=int(config.min_cuda_free_memory_mb),
        progress_path=progress_path,
        progress_context=progress_context,
        phase="single_pair_training",
    )
    source_stft = _stft_mono(mono(source_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=device)
    target_stft = _stft_mono(mono(target_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=device)
    source_stft_frames = source_stft.transpose(0, 1)
    target_stft_frames = target_stft.transpose(0, 1)
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
    if mask_output_mode == MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL:
        residual_deviation = magnitude_residual_deviation(config.mask_ceiling)
        oracle_mask = (target_mag / source_mag).clamp(1.0 - residual_deviation, 1.0 + residual_deviation)
    else:
        oracle_mask = (target_mag / source_mag).clamp(0.0, float(config.mask_ceiling))
    complex_oracle_mask = _complex_ratio_mask_frames(
        source_stft,
        target_stft,
        ceiling=float(config.complex_mask_ceiling),
    )
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
        mask_output_mode=mask_output_mode,
        mask_ceiling=float(config.mask_ceiling),
        complex_mask_ceiling=float(config.complex_mask_ceiling),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=1.0e-5)
    base_indices = torch.arange(frame_count, device=torch.device("cpu"))
    active_mask_cpu = (target_frame_activity >= active_threshold).detach().cpu()
    active_indices = base_indices[active_mask_cpu]
    if int(config.active_frame_oversample) > 1 and active_indices.numel() > 0:
        indices = torch.cat([base_indices, active_indices.repeat(int(config.active_frame_oversample) - 1)])
    else:
        indices = base_indices
    batch_size = max(1, min(int(config.batch_size), int(frame_count)))
    batch_count = int(math.ceil(int(indices.numel()) / float(batch_size)))
    batch_progress_stride = max(1, batch_count // 10)
    last_loss = 0.0
    epoch_count = int(config.epochs)
    if progress_path is not None:
        progress(
            progress_path,
            "train_loop_start",
            **dict(progress_context or {}),
            epoch_count=epoch_count,
            batch_count=batch_count,
            batch_size=batch_size,
            frame_count=int(frame_count),
            training_index_count=int(indices.numel()),
            batch_progress_stride=batch_progress_stride,
            device=str(device),
        )
    _synchronize_cuda_phase(
        progress_path,
        progress_context,
        event_prefix="train_pre_loop",
        device=device,
        frame_count=int(frame_count),
        training_index_count=int(indices.numel()),
        batch_count=batch_count,
        batch_size=batch_size,
    )
    for epoch_index in range(epoch_count):
        if progress_path is not None:
            progress(
                progress_path,
                "train_epoch_start",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                frame_count=int(frame_count),
                training_index_count=int(indices.numel()),
                batch_size=batch_size,
                batch_count=batch_count,
                device=str(device),
            )
        permutation_device = torch.device("cpu")
        if progress_path is not None:
            progress(
                progress_path,
                "train_permutation_start",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                training_index_count=int(indices.numel()),
                training_device=str(device),
                permutation_device=str(permutation_device),
            )
        permutation_order = torch.randperm(indices.numel(), device=permutation_device)
        if progress_path is not None:
            progress(
                progress_path,
                "train_permutation_finish",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                training_index_count=int(indices.numel()),
                training_device=str(device),
                permutation_device=str(permutation_device),
            )
        for start in range(0, indices.numel(), batch_size):
            batch_index = (int(start) // int(batch_size)) + 1
            should_log_batch_progress = (
                batch_index == 1 or batch_index == batch_count or batch_index % batch_progress_stride == 0
            )
            if progress_path is not None and should_log_batch_progress:
                progress(
                    progress_path,
                    "train_batch_index_select_start",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    training_device=str(device),
                    permutation_device=str(permutation_device),
                    device=str(device),
                )
            batch_order = permutation_order[start : start + batch_size]
            batch_indices = indices[batch_order]
            if progress_path is not None and should_log_batch_progress:
                progress(
                    progress_path,
                    "train_batch_index_select_finish",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    batch_frame_count=int(batch_indices.numel()),
                    training_device=str(device),
                    permutation_device=str(permutation_device),
                    device=str(device),
                )
                progress(
                    progress_path,
                    "train_batch_start",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    batch_frame_count=int(batch_indices.numel()),
                    device=str(device),
                )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_feature_gather_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            batch_model_features = model_features[batch_indices]
            batch_weights = time_frequency_weights[batch_indices]
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_feature_gather_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_forward_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            prediction = model(batch_model_features)
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_forward_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_loss_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            if mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
                prediction_complex = _complex_prediction_frames(prediction, frequency_bins)
                mask_loss = _weighted_complex_mse(prediction_complex - complex_oracle_mask[batch_indices], batch_weights)
                estimated_stft = prediction_complex * source_stft_frames[batch_indices]
                stft_loss = _weighted_complex_mse(estimated_stft - target_stft_frames[batch_indices], batch_weights)
                loss = mask_loss + float(config.magnitude_loss_weight) * stft_loss / source_mag.mean().clamp_min(EPSILON)
            else:
                mask_loss = ((prediction - oracle_mask[batch_indices]) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
                estimated_mag = prediction * source_mag[batch_indices]
                magnitude_loss = ((estimated_mag - target_mag[batch_indices]) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
                loss = mask_loss + float(config.magnitude_loss_weight) * magnitude_loss / source_mag.mean().clamp_min(EPSILON)
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_loss_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_backward_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            loss.backward()
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_backward_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_optimizer_step_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            optimizer.step()
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "train_batch_optimizer_step_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            last_loss = float(loss.detach().cpu())
            if progress_path is not None and should_log_batch_progress:
                progress(
                    progress_path,
                    "train_batch_heartbeat",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    current_loss=round(last_loss, 9),
                    device=str(device),
                )
        if progress_path is not None:
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
        "mask_output_mode": mask_output_mode,
        "complex_mask_ceiling": float(config.complex_mask_ceiling),
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
            ceiling=float(config.complex_mask_ceiling),
        )
        metadata["temporal_complex_mask_prior_real"] = complex_mask_prior.real.astype(np.float16, copy=False)
        metadata["temporal_complex_mask_prior_imag"] = complex_mask_prior.imag.astype(np.float16, copy=False)
    source_feature_memory_stored_frame_count = 0
    source_feature_memory_projection_dim_used = 0
    source_feature_memory_selected_indices: torch.Tensor | None = None
    if int(config.source_feature_memory_neighbors) > 0:
        memory_features = model_features
        projection_matrix = _source_feature_memory_projection_matrix(
            feature_width=int(model_features.shape[1]),
            projection_dim=int(config.source_feature_memory_projection_dim),
            seed=int(config.seed),
            device=device,
        )
        if projection_matrix is not None:
            memory_features = _project_source_feature_memory_features(model_features, projection_matrix)
            source_feature_memory_projection_dim_used = int(projection_matrix.shape[1])
            metadata["source_feature_memory_projection_matrix"] = (
                projection_matrix.detach().cpu().numpy().astype(np.float16, copy=False)
            )
        source_feature_memory_selected_indices = _source_feature_memory_selected_indices(
            frame_count=int(memory_features.shape[0]),
            max_frames=int(config.source_feature_memory_max_frames),
            device=device,
        )
        if source_feature_memory_selected_indices is not None:
            memory_features = memory_features[source_feature_memory_selected_indices]
            memory_masks = oracle_mask[source_feature_memory_selected_indices]
            metadata["source_feature_memory_selected_indices"] = (
                source_feature_memory_selected_indices.detach().cpu().numpy().astype(np.int64, copy=False)
            )
        else:
            memory_masks = oracle_mask
        source_feature_memory_stored_frame_count = int(memory_features.shape[0])
        metadata["source_feature_memory_features"] = memory_features.detach().cpu().numpy().astype(np.float16, copy=False)
        metadata["source_feature_memory_masks"] = memory_masks.detach().cpu().numpy().astype(np.float16, copy=False)
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
            if source_feature_memory_selected_indices is not None:
                channel_mask = channel_mask[source_feature_memory_selected_indices]
            channel_masks.append(channel_mask.detach().cpu().numpy().astype(np.float16, copy=False))
        if channel_masks:
            metadata["source_feature_memory_channel_masks"] = np.stack(channel_masks, axis=0)
        if bool(config.source_feature_memory_store_complex_masks) and source_feature_memory_selected_indices is None:
            source_feature_complex_mask = _complex_mask_prior(
                source_audio,
                target_audio,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                device=device,
                ceiling=float(config.complex_mask_ceiling),
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
        "mask_output_mode": mask_output_mode,
        "complex_mask_ceiling": float(config.complex_mask_ceiling),
        "device_used": 1.0 if device.type == "cuda" else 0.0,
        "source_feature_memory_frame_count": float(source_feature_memory_stored_frame_count)
        if int(config.source_feature_memory_neighbors) > 0
        else 0.0,
        "source_feature_memory_original_frame_count": float(frame_count)
        if int(config.source_feature_memory_neighbors) > 0
        else 0.0,
        "source_feature_memory_projection_dim": float(source_feature_memory_projection_dim_used),
        "source_feature_memory_max_frames": float(int(config.source_feature_memory_max_frames)),
    }
    return model, metadata, diagnostics


def train_teacher_mask_dataset(
    training_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    config: TeacherMaskTrainConfig,
    progress_path: Path | None = None,
    progress_context: dict[str, Any] | None = None,
    max_frames_per_pair: int = 0,
    identity_pair_flags: Sequence[bool] | None = None,
    conditioning_audio_groups: Sequence[Sequence[np.ndarray]] | None = None,
) -> tuple[TeacherMaskNetwork, dict[str, Any], dict[str, float]]:
    if not training_pairs:
        raise ValueError("training_pairs must contain at least one source/target pair")
    mask_output_mode = validate_mask_output_mode(config.mask_output_mode)
    if int(config.time_feature_count) > 0:
        raise ValueError("dataset training forbids temporal priors because they do not generalize across tracks")
    if bool(config.source_feature_memory_store_complex_masks):
        raise ValueError("dataset training forbids complex source-feature memory because it is reconstructive per-source state")
    if identity_pair_flags is None:
        resolved_identity_pair_flags = [True] * len(training_pairs)
        source_preservation_scope = "all_pairs_backward_compatible"
    else:
        if len(identity_pair_flags) != len(training_pairs):
            raise ValueError("identity_pair_flags must match training_pairs length")
        resolved_identity_pair_flags = [bool(flag) for flag in identity_pair_flags]
        source_preservation_scope = "declared_identity_pairs_only"
    if conditioning_audio_groups is None:
        resolved_conditioning_groups: list[list[np.ndarray]] = [[] for _ in training_pairs]
    else:
        if len(conditioning_audio_groups) != len(training_pairs):
            raise ValueError("conditioning_audio_groups must match training_pairs length")
        resolved_conditioning_groups = [list(group) for group in conditioning_audio_groups]
    conditioning_audio_count = len(resolved_conditioning_groups[0]) if resolved_conditioning_groups else 0
    if any(len(group) != conditioning_audio_count for group in resolved_conditioning_groups):
        raise ValueError("all training pairs must use the same conditioning audio count")
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = _select_device(config.device)
    if device.type == "mps" and mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
        raise RuntimeError("MPS complex-mask training is unsupported; use magnitude or magnitude_residual mode")
    dsp_device = torch.device("cpu") if device.type == "mps" else device
    cpu_storage_dtype = torch.float16 if is_accelerator_device(device) else torch.float32
    needs_complex_training_tensors = mask_output_mode == MASK_OUTPUT_MODE_COMPLEX
    _require_cuda_memory_available(
        device,
        min_free_memory_mb=int(config.min_cuda_free_memory_mb),
        progress_path=progress_path,
        progress_context=progress_context,
        phase="dataset_training",
    )
    raw_feature_chunks: list[torch.Tensor] = []
    source_stft_chunks: list[torch.Tensor] = []
    target_stft_chunks: list[torch.Tensor] = []
    source_mag_chunks: list[torch.Tensor] = []
    target_mag_chunks: list[torch.Tensor] = []
    oracle_mask_chunks: list[torch.Tensor] = []
    complex_mask_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    preservation_scope_chunks: list[torch.Tensor] = []
    active_index_chunks: list[torch.Tensor] = []
    frame_counts: list[int] = []
    frequency_bins: int | None = None
    frame_offset = 0

    for pair_index, (source_audio, target_audio) in enumerate(training_pairs, start=1):
        source_audio, target_audio = fit_audio_pair(source_audio, target_audio)
        conditioning_group = [
            fit_audio_frames(conditioning_audio, int(source_audio.shape[0]))
            for conditioning_audio in resolved_conditioning_groups[pair_index - 1]
        ]
        source_stft = _stft_mono(
            mono(source_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=dsp_device
        )
        target_stft = _stft_mono(
            mono(target_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=dsp_device
        )
        source_stft_frames = source_stft.transpose(0, 1)
        target_stft_frames = target_stft.transpose(0, 1)
        source_mag = source_stft.abs().transpose(0, 1).clamp_min(EPSILON)
        target_mag = target_stft.abs().transpose(0, 1)
        conditioning_magnitudes = [
            _stft_mono(
                mono(conditioning_audio),
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                device=dsp_device,
            )
            .abs()
            .transpose(0, 1)
            .clamp_min(EPSILON)
            for conditioning_audio in conditioning_group
        ]
        if frequency_bins is None:
            frequency_bins = int(source_mag.shape[1])
        elif int(source_mag.shape[1]) != frequency_bins:
            raise ValueError("all training pairs must produce the same frequency-bin count")
        raw_features = _conditioned_local_context_features(
            source_mag,
            conditioning_magnitudes,
            int(config.context_frame_radius),
        )
        if mask_output_mode == MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL:
            residual_deviation = magnitude_residual_deviation(config.mask_ceiling)
            oracle_mask = (target_mag / source_mag).clamp(1.0 - residual_deviation, 1.0 + residual_deviation)
        else:
            oracle_mask = (target_mag / source_mag).clamp(0.0, float(config.mask_ceiling))
        complex_oracle_mask = _complex_ratio_mask_frames(
            source_stft,
            target_stft,
            ceiling=float(config.complex_mask_ceiling),
        )
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
            selected_indices = _evenly_spaced_long_indices(
                length=frame_count,
                count=int(max_frames_per_pair),
                device=source_mag.device,
            )
            source_mag = source_mag[selected_indices]
            target_mag = target_mag[selected_indices]
            source_stft_frames = source_stft_frames[selected_indices]
            target_stft_frames = target_stft_frames[selected_indices]
            raw_features = raw_features[selected_indices]
            oracle_mask = oracle_mask[selected_indices]
            complex_oracle_mask = complex_oracle_mask[selected_indices]
            time_frequency_weights = time_frequency_weights[selected_indices]
            active_mask = active_mask[selected_indices]
            frame_count = int(source_mag.shape[0])
        if progress_path is not None:
            progress(
                progress_path,
                "dataset_pair_tensor_cpu_transfer_start",
                **dict(progress_context or {}),
                pair_index=pair_index,
                pair_count=len(training_pairs),
                frame_count=frame_count,
                source_device=str(source_mag.device),
                target_storage_device="cpu",
                device=str(device),
            )
        raw_feature_chunks.append(raw_features.detach().cpu().to(cpu_storage_dtype))
        if needs_complex_training_tensors:
            source_stft_chunks.append(source_stft_frames.detach().cpu())
            target_stft_chunks.append(target_stft_frames.detach().cpu())
            complex_mask_chunks.append(complex_oracle_mask.detach().cpu())
        source_mag_chunks.append(source_mag.detach().cpu().to(cpu_storage_dtype))
        target_mag_chunks.append(target_mag.detach().cpu().to(cpu_storage_dtype))
        oracle_mask_chunks.append(oracle_mask.detach().cpu().to(cpu_storage_dtype))
        weight_chunks.append(time_frequency_weights.detach().cpu().to(cpu_storage_dtype))
        preservation_scope_chunks.append(
            torch.full_like(
                time_frequency_weights,
                1.0 if resolved_identity_pair_flags[pair_index - 1] else 0.0,
            )
            .detach()
            .cpu()
            .to(cpu_storage_dtype)
        )
        if progress_path is not None:
            progress(
                progress_path,
                "dataset_pair_tensor_cpu_transfer_finish",
                **dict(progress_context or {}),
                pair_index=pair_index,
                pair_count=len(training_pairs),
                frame_count=frame_count,
                training_tensor_storage_device="cpu",
                training_tensor_storage_dtype=str(cpu_storage_dtype),
                complex_training_tensor_stored=needs_complex_training_tensors,
                device=str(device),
            )
        del (
            source_stft,
            target_stft,
            source_stft_frames,
            target_stft_frames,
            source_mag,
            target_mag,
            raw_features,
            oracle_mask,
            complex_oracle_mask,
            time_frequency_weights,
            conditioning_magnitudes,
            conditioning_group,
        )
        active_indices = torch.nonzero(active_mask.detach().cpu(), as_tuple=False).flatten()
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
                conditioning_audio_count=conditioning_audio_count,
                device=str(device),
            )

    model_feature_width = int(raw_feature_chunks[0].shape[1]) + int(config.time_feature_count)
    frequency_bins = int(frequency_bins or 0)
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_model_init_start",
            **dict(progress_context or {}),
            hidden_size=int(config.hidden_size),
            input_features=model_feature_width,
            frequency_bins=frequency_bins,
            mask_output_mode=mask_output_mode,
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_network_construct_start",
            **dict(progress_context or {}),
            hidden_size=int(config.hidden_size),
            input_features=model_feature_width,
            frequency_bins=frequency_bins,
            mask_output_mode=mask_output_mode,
            device=str(device),
        )
    model = TeacherMaskNetwork(
        input_features=model_feature_width,
        frequency_bins=frequency_bins,
        hidden_size=int(config.hidden_size),
        mask_output_mode=mask_output_mode,
        mask_ceiling=float(config.mask_ceiling),
        complex_mask_ceiling=float(config.complex_mask_ceiling),
    )
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_network_construct_finish",
            **dict(progress_context or {}),
            hidden_size=int(config.hidden_size),
            input_features=model_feature_width,
            frequency_bins=frequency_bins,
            mask_output_mode=mask_output_mode,
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_model_to_device_start",
            **dict(progress_context or {}),
            device=str(device),
        )
    model = model.to(device)
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_model_to_device_finish",
            **dict(progress_context or {}),
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_optimizer_init_start",
            **dict(progress_context or {}),
            learning_rate=float(config.learning_rate),
            device=str(device),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=1.0e-5)
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_optimizer_init_finish",
            **dict(progress_context or {}),
            learning_rate=float(config.learning_rate),
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_model_init_finish",
            **dict(progress_context or {}),
            hidden_size=int(config.hidden_size),
            input_features=model_feature_width,
            frequency_bins=frequency_bins,
            mask_output_mode=mask_output_mode,
            device=str(device),
        )

    if progress_path is not None:
        progress(
            progress_path,
            "dataset_feature_stack_start",
            **dict(progress_context or {}),
            pair_count=len(training_pairs),
            chunk_count=len(raw_feature_chunks),
            total_pair_frame_count=int(sum(frame_counts)),
            device=str(device),
        )
    raw_feature_chunk_count = len(raw_feature_chunks)
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_feature_stats_start",
            **dict(progress_context or {}),
            chunk_count=raw_feature_chunk_count,
            total_pair_frame_count=int(sum(frame_counts)),
            feature_width=int(raw_feature_chunks[0].shape[1]),
            stats_accumulation_device="cpu",
            device=str(device),
        )
    feature_mean, feature_std = _feature_mean_std_from_chunks(raw_feature_chunks)
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_feature_stats_finish",
            **dict(progress_context or {}),
            chunk_count=raw_feature_chunk_count,
            total_pair_frame_count=int(sum(frame_counts)),
            feature_width=int(feature_mean.shape[1]),
            stats_accumulation_device="cpu",
            stats_tensor_dtype=str(feature_mean.dtype),
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_feature_normalize_start",
            **dict(progress_context or {}),
            chunk_count=raw_feature_chunk_count,
            total_pair_frame_count=int(sum(frame_counts)),
            normalized_tensor_storage_device="cpu",
            normalized_tensor_storage_dtype=str(cpu_storage_dtype),
            normalization_strategy="chunked_cpu_low_peak",
            device=str(device),
        )
    normalized_feature_chunks: list[torch.Tensor] = []
    for chunk_index in range(raw_feature_chunk_count):
        chunk = raw_feature_chunks[chunk_index]
        chunk_float = chunk.to(torch.float32)
        raw_feature_chunks[chunk_index] = torch.empty(0, dtype=chunk.dtype, device=chunk.device)
        del chunk
        chunk_float.sub_(feature_mean).div_(feature_std)
        normalized_feature_chunks.append(chunk_float.to(cpu_storage_dtype))
        del chunk_float
    raw_feature_chunks.clear()
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_feature_normalize_finish",
            **dict(progress_context or {}),
            chunk_count=len(normalized_feature_chunks),
            total_pair_frame_count=int(sum(frame_counts)),
            normalized_tensor_storage_device="cpu",
            normalized_tensor_storage_dtype=str(cpu_storage_dtype),
            normalization_strategy="chunked_cpu_low_peak",
            device=str(device),
        )
    features = torch.cat(normalized_feature_chunks, dim=0)
    del normalized_feature_chunks
    tensor_storage_device = torch.device("cpu")
    temporal_features = _temporal_position_features(
        int(features.shape[0]),
        int(config.time_feature_count),
        device=tensor_storage_device,
    )
    if int(temporal_features.shape[1]) > 0:
        model_features = torch.cat([features, temporal_features.to(features.dtype)], dim=1)
    else:
        model_features = features
    frame_count = int(model_features.shape[0])
    frequency_bins = int(frequency_bins or 0)
    active_indices = (
        torch.cat(active_index_chunks, dim=0)
        if active_index_chunks
        else torch.empty(0, dtype=torch.long, device=torch.device("cpu"))
    )
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_feature_stack_finish",
            **dict(progress_context or {}),
            frame_count=frame_count,
            frequency_bins=frequency_bins,
            model_feature_width=int(model_features.shape[1]),
            source_stft_frame_count=int(sum(frame_counts)),
            training_tensor_storage_device=str(model_features.device),
            training_tensor_storage_dtype=str(model_features.dtype),
            normalization_strategy="chunked_cpu_low_peak",
            device=str(device),
        )
    released_chunk_count = raw_feature_chunk_count + len(active_index_chunks)
    del temporal_features, active_index_chunks, frame_counts
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_feature_chunks_released",
            **dict(progress_context or {}),
            released_chunk_count=int(released_chunk_count),
            frame_count=frame_count,
            frequency_bins=frequency_bins,
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_training_tensor_stack_start",
            **dict(progress_context or {}),
            chunk_count=len(source_stft_chunks),
            frame_count=frame_count,
            frequency_bins=frequency_bins,
            device=str(device),
        )
    if needs_complex_training_tensors:
        source_stft_all = torch.cat(source_stft_chunks, dim=0)
        target_stft_all = torch.cat(target_stft_chunks, dim=0)
        complex_mask_all = torch.cat(complex_mask_chunks, dim=0)
    else:
        source_stft_all = None
        target_stft_all = None
        complex_mask_all = None
    source_mag_all = torch.cat(source_mag_chunks, dim=0).to(cpu_storage_dtype)
    target_mag_all = torch.cat(target_mag_chunks, dim=0).to(cpu_storage_dtype)
    oracle_mask_all = torch.cat(oracle_mask_chunks, dim=0).to(cpu_storage_dtype)
    weights_all = torch.cat(weight_chunks, dim=0).to(cpu_storage_dtype)
    preservation_scope_all = torch.cat(preservation_scope_chunks, dim=0).to(cpu_storage_dtype)
    source_mag_mean_scale = float(source_mag_all.float().mean().clamp_min(EPSILON))
    released_training_chunk_count = (
        len(source_stft_chunks)
        + len(target_stft_chunks)
        + len(source_mag_chunks)
        + len(target_mag_chunks)
        + len(oracle_mask_chunks)
        + len(complex_mask_chunks)
        + len(weight_chunks)
        + len(preservation_scope_chunks)
    )
    del (
        source_stft_chunks,
        target_stft_chunks,
        source_mag_chunks,
        target_mag_chunks,
        oracle_mask_chunks,
        complex_mask_chunks,
        weight_chunks,
        preservation_scope_chunks,
    )
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_training_tensor_stack_finish",
            **dict(progress_context or {}),
            frame_count=frame_count,
            frequency_bins=frequency_bins,
            source_stft_frame_count=int(source_stft_all.shape[0]) if source_stft_all is not None else 0,
            target_stft_frame_count=int(target_stft_all.shape[0]) if target_stft_all is not None else 0,
            training_tensor_storage_device=str(source_mag_all.device),
            training_tensor_storage_dtype=str(source_mag_all.dtype),
            complex_training_tensor_stored=needs_complex_training_tensors,
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_training_chunks_released",
            **dict(progress_context or {}),
            released_chunk_count=int(released_training_chunk_count),
            frame_count=frame_count,
            frequency_bins=frequency_bins,
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_training_indices_start",
            **dict(progress_context or {}),
            frame_count=frame_count,
            active_frame_count=int(active_indices.numel()),
            active_frame_oversample=int(config.active_frame_oversample),
            device=str(device),
        )
    base_indices = torch.arange(frame_count, device=torch.device("cpu"))
    if int(config.active_frame_oversample) > 1 and active_indices.numel() > 0:
        indices = torch.cat([base_indices, active_indices.repeat(int(config.active_frame_oversample) - 1)])
    else:
        indices = base_indices
    batch_size = max(1, min(int(config.batch_size), frame_count))
    batch_count = int(math.ceil(int(indices.numel()) / float(batch_size)))
    batch_progress_stride = max(1, batch_count // 10)
    epoch_count = int(config.epochs)
    last_loss = 0.0
    if progress_path is not None:
        progress(
            progress_path,
            "dataset_training_indices_finish",
            **dict(progress_context or {}),
            frame_count=frame_count,
            active_frame_count=int(active_indices.numel()),
            training_index_count=int(indices.numel()),
            active_frame_oversample=int(config.active_frame_oversample),
            batch_size=batch_size,
            batch_count=batch_count,
            batch_progress_stride=batch_progress_stride,
            device=str(device),
        )
        progress(
            progress_path,
            "dataset_train_loop_start",
            **dict(progress_context or {}),
            epoch_count=epoch_count,
            batch_count=batch_count,
            batch_size=batch_size,
            frame_count=frame_count,
            training_index_count=int(indices.numel()),
            batch_progress_stride=batch_progress_stride,
            device=str(device),
        )
    _synchronize_cuda_phase(
        progress_path,
        progress_context,
        event_prefix="dataset_train_pre_loop",
        device=device,
        frame_count=frame_count,
        training_index_count=int(indices.numel()),
        batch_count=batch_count,
        batch_size=batch_size,
    )
    for epoch_index in range(epoch_count):
        if progress_path is not None:
            progress(
                progress_path,
                "dataset_train_epoch_start",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                frame_count=frame_count,
                training_index_count=int(indices.numel()),
                batch_size=batch_size,
                batch_count=batch_count,
                device=str(device),
            )
        permutation_device = torch.device("cpu")
        if progress_path is not None:
            progress(
                progress_path,
                "dataset_train_permutation_start",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                training_index_count=int(indices.numel()),
                training_device=str(device),
                permutation_device=str(permutation_device),
            )
        permutation_order = torch.randperm(indices.numel(), device=permutation_device)
        if progress_path is not None:
            progress(
                progress_path,
                "dataset_train_permutation_finish",
                **dict(progress_context or {}),
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                training_index_count=int(indices.numel()),
                training_device=str(device),
                permutation_device=str(permutation_device),
            )
        for start in range(0, int(indices.numel()), batch_size):
            batch_index = (int(start) // int(batch_size)) + 1
            should_log_batch_progress = (
                batch_index == 1 or batch_index == batch_count or batch_index % batch_progress_stride == 0
            )
            if progress_path is not None and should_log_batch_progress:
                progress(
                    progress_path,
                    "dataset_train_batch_index_select_start",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    training_device=str(device),
                    permutation_device=str(permutation_device),
                    device=str(device),
                )
            batch_order = permutation_order[start : start + batch_size]
            batch_indices = indices[batch_order]
            if progress_path is not None and should_log_batch_progress:
                progress(
                    progress_path,
                    "dataset_train_batch_index_select_finish",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    batch_frame_count=int(batch_indices.numel()),
                    training_device=str(device),
                    permutation_device=str(permutation_device),
                    device=str(device),
                )
                progress(
                    progress_path,
                    "dataset_train_batch_start",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    batch_frame_count=int(batch_indices.numel()),
                    device=str(device),
                )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_feature_gather_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                training_tensor_storage_device=str(model_features.device),
                device=device,
            )
            batch_model_features_cpu = model_features[batch_indices]
            batch_weights_cpu = weights_all[batch_indices]
            batch_preservation_scope_cpu = preservation_scope_all[batch_indices]
            if mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
                batch_complex_mask_cpu = complex_mask_all[batch_indices]
                batch_source_stft_cpu = source_stft_all[batch_indices]
                batch_target_stft_cpu = target_stft_all[batch_indices]
            else:
                batch_oracle_mask_cpu = oracle_mask_all[batch_indices]
                batch_source_mag_cpu = source_mag_all[batch_indices]
                batch_target_mag_cpu = target_mag_all[batch_indices]
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_feature_gather_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                training_tensor_storage_device=str(model_features.device),
                gathered_tensor_storage_device=str(batch_model_features_cpu.device),
                device=device,
            )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_to_device_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                training_tensor_storage_device=str(model_features.device),
                training_device=str(device),
                device=device,
            )
            batch_model_features = batch_model_features_cpu.to(device=device, dtype=torch.float32)
            batch_weights = batch_weights_cpu.to(device=device, dtype=torch.float32)
            batch_preservation_scope = batch_preservation_scope_cpu.to(device=device, dtype=torch.float32)
            if mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
                batch_complex_mask = batch_complex_mask_cpu.to(device)
                batch_source_stft = batch_source_stft_cpu.to(device)
                batch_target_stft = batch_target_stft_cpu.to(device)
            else:
                batch_oracle_mask = batch_oracle_mask_cpu.to(device=device, dtype=torch.float32)
                batch_source_mag = batch_source_mag_cpu.to(device=device, dtype=torch.float32)
                batch_target_mag = batch_target_mag_cpu.to(device=device, dtype=torch.float32)
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_to_device_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                training_tensor_storage_device=str(model_features.device),
                batch_tensor_device=str(batch_model_features.device),
                training_device=str(device),
                device=device,
            )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_forward_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            prediction = model(batch_model_features)
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_forward_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_loss_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            if mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
                prediction_complex = _complex_prediction_frames(prediction, frequency_bins)
                mask_loss = _weighted_complex_mse(prediction_complex - batch_complex_mask, batch_weights)
                estimated_stft = prediction_complex * batch_source_stft
                stft_loss = _weighted_complex_mse(estimated_stft - batch_target_stft, batch_weights)
                identity_complex = torch.ones_like(prediction_complex)
                source_preservation_loss = _scoped_source_preservation_loss(
                    prediction_complex - identity_complex,
                    batch_weights,
                    batch_preservation_scope,
                )
                loss = (
                    mask_loss
                    + float(config.magnitude_loss_weight) * stft_loss / max(source_mag_mean_scale, EPSILON)
                    + max(0.0, float(config.source_preservation_weight)) * source_preservation_loss
                )
            else:
                mask_loss = ((prediction - batch_oracle_mask) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
                estimated_mag = prediction * batch_source_mag
                magnitude_loss = ((estimated_mag - batch_target_mag) ** 2 * batch_weights).sum() / batch_weights.sum().clamp_min(EPSILON)
                source_preservation_loss = _scoped_source_preservation_loss(
                    prediction - 1.0,
                    batch_weights,
                    batch_preservation_scope,
                )
                loss = (
                    mask_loss
                    + float(config.magnitude_loss_weight) * magnitude_loss / max(source_mag_mean_scale, EPSILON)
                    + max(0.0, float(config.source_preservation_weight)) * source_preservation_loss
                )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_loss_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_backward_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            loss.backward()
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_backward_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_optimizer_step_start",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            optimizer.step()
            _progress_train_batch_phase(
                progress_path if should_log_batch_progress else None,
                "dataset_train_batch_optimizer_step_finish",
                progress_context,
                epoch=epoch_index + 1,
                epoch_count=epoch_count,
                batch_index=batch_index,
                batch_count=batch_count,
                batch_start_index=int(start),
                batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                training_index_count=int(indices.numel()),
                batch_frame_count=int(batch_indices.numel()),
                device=device,
            )
            last_loss = float(loss.detach().cpu())
            if progress_path is not None and should_log_batch_progress:
                progress(
                    progress_path,
                    "dataset_train_batch_heartbeat",
                    **dict(progress_context or {}),
                    epoch=epoch_index + 1,
                    epoch_count=epoch_count,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_start_index=int(start),
                    batch_end_index=int(min(int(start) + int(batch_size), int(indices.numel()))),
                    training_index_count=int(indices.numel()),
                    current_loss=round(last_loss, 9),
                    device=str(device),
                )
        if progress_path is not None:
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
        "mask_output_mode": mask_output_mode,
        "complex_mask_ceiling": float(config.complex_mask_ceiling),
        "conditioning_audio_count": int(conditioning_audio_count),
        "conditioning_feature_schema": CONDITIONING_FEATURE_SCHEMA if conditioning_audio_count else "none",
        "feature_mean": feature_mean.detach().cpu().numpy().astype(np.float32),
        "feature_std": feature_std.detach().cpu().numpy().astype(np.float32),
    }
    source_feature_memory_stored_frame_count = 0
    source_feature_memory_projection_dim_used = 0
    if int(config.source_feature_memory_neighbors) > 0:
        memory_features = model_features.float()
        projection_matrix = _source_feature_memory_projection_matrix(
            feature_width=int(model_features.shape[1]),
            projection_dim=int(config.source_feature_memory_projection_dim),
            seed=int(config.seed),
            device=torch.device("cpu"),
        )
        if projection_matrix is not None:
            memory_features = _project_source_feature_memory_features(model_features, projection_matrix)
            source_feature_memory_projection_dim_used = int(projection_matrix.shape[1])
            metadata["source_feature_memory_projection_matrix"] = (
                projection_matrix.detach().cpu().numpy().astype(np.float16, copy=False)
            )
        selected_indices = _source_feature_memory_selected_indices(
            frame_count=int(memory_features.shape[0]),
            max_frames=int(config.source_feature_memory_max_frames),
            device=torch.device("cpu"),
        )
        if selected_indices is not None:
            memory_features = memory_features[selected_indices]
            memory_masks = oracle_mask_all[selected_indices].float()
            memory_complex_masks = complex_mask_all[selected_indices] if complex_mask_all is not None else None
            metadata["source_feature_memory_selected_indices"] = selected_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            memory_masks = oracle_mask_all.float()
            memory_complex_masks = complex_mask_all
        source_feature_memory_stored_frame_count = int(memory_features.shape[0])
        metadata["source_feature_memory_features"] = memory_features.detach().cpu().numpy().astype(np.float16, copy=False)
        metadata["source_feature_memory_masks"] = memory_masks.detach().cpu().numpy().astype(np.float16, copy=False)
        if mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
            if memory_complex_masks is None:
                raise RuntimeError("complex source-feature memory requested without complex mask tensors")
            metadata["source_feature_memory_complex_frame_mask_real"] = (
                memory_complex_masks.real.detach().cpu().numpy().astype(np.float16, copy=False)
            )
            metadata["source_feature_memory_complex_frame_mask_imag"] = (
                memory_complex_masks.imag.detach().cpu().numpy().astype(np.float16, copy=False)
            )
    diagnostics = {
        "training_loss": round(last_loss, 9),
        "training_example_count": float(len(training_pairs)),
        "unique_training_pair_count": float(len(training_pairs)),
        "frame_count": float(frame_count),
        "training_index_count": float(indices.numel()),
        "active_frame_count": float(active_indices.numel()),
        "frequency_bins": float(frequency_bins),
        "mask_output_mode": mask_output_mode,
        "complex_mask_ceiling": float(config.complex_mask_ceiling),
        "device_used": 1.0 if is_accelerator_device(device) else 0.0,
        "neural_training_device": str(device.type),
        "dsp_feature_device": str(dsp_device.type),
        "max_frames_per_pair": float(max(0, int(max_frames_per_pair))),
        "source_feature_memory_frame_count": float(source_feature_memory_stored_frame_count)
        if int(config.source_feature_memory_neighbors) > 0
        else 0.0,
        "source_feature_memory_original_frame_count": float(frame_count)
        if int(config.source_feature_memory_neighbors) > 0
        else 0.0,
        "source_feature_memory_projection_dim": float(source_feature_memory_projection_dim_used),
        "source_feature_memory_max_frames": float(int(config.source_feature_memory_max_frames)),
        "source_preservation_scope": source_preservation_scope,
        "identity_preservation_pair_count": float(sum(resolved_identity_pair_flags)),
        "conditioning_audio_count": float(conditioning_audio_count),
        "conditioning_feature_schema": CONDITIONING_FEATURE_SCHEMA if conditioning_audio_count else "none",
    }
    return model, metadata, diagnostics


def apply_teacher_mask(
    source_audio: np.ndarray,
    *,
    model: TeacherMaskNetwork,
    metadata: dict[str, Any],
    config: TeacherMaskTrainConfig,
    conditioning_audio: Sequence[np.ndarray] | None = None,
    progress_callback: ProgressCallback | None = None,
    max_source_feature_memory_distance_ops: int = DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = next(model.parameters()).device
    mask_output_mode = validate_mask_output_mode(config.mask_output_mode)
    if device.type == "mps" and mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
        raise RuntimeError("MPS complex-mask application is unsupported; use magnitude or magnitude_residual mode")
    expected_conditioning_count = int(metadata.get("conditioning_audio_count", 0))
    resolved_conditioning_audio = list(conditioning_audio or [])
    if len(resolved_conditioning_audio) != expected_conditioning_count:
        raise ValueError(
            "conditioning audio count mismatch: "
            f"checkpoint={expected_conditioning_count}, application={len(resolved_conditioning_audio)}"
        )
    expected_conditioning_schema = str(metadata.get("conditioning_feature_schema") or "none")
    if expected_conditioning_count > 0 and expected_conditioning_schema != CONDITIONING_FEATURE_SCHEMA:
        raise ValueError(f"unsupported conditioning feature schema: {expected_conditioning_schema}")
    dsp_device = torch.device("cpu") if device.type == "mps" else device
    emit_progress(
        progress_callback,
        "apply_feature_extraction_start",
        device=str(device.type),
        source_frames=int(source_audio.shape[0]),
        source_channels=int(source_audio.shape[1]) if source_audio.ndim == 2 else 0,
        conditioning_audio_count=expected_conditioning_count,
        n_fft=int(config.n_fft),
        hop_length=int(config.hop_length),
    )
    source_stft_mono = _stft_mono(
        mono(source_audio), n_fft=config.n_fft, hop_length=config.hop_length, device=dsp_device
    )
    source_mag = source_stft_mono.abs().transpose(0, 1).clamp_min(EPSILON)
    conditioning_magnitudes = [
        _stft_mono(
            mono(fit_audio_frames(audio, int(source_audio.shape[0]))),
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            device=dsp_device,
        )
        .abs()
        .transpose(0, 1)
        .clamp_min(EPSILON)
        for audio in resolved_conditioning_audio
    ]
    raw_features = _conditioned_local_context_features(
        source_mag,
        conditioning_magnitudes,
        int(config.context_frame_radius),
    )
    feature_mean = torch.as_tensor(metadata["feature_mean"], dtype=torch.float32, device=dsp_device)
    feature_std = torch.as_tensor(metadata["feature_std"], dtype=torch.float32, device=dsp_device).clamp_min(1.0e-4)
    features = (raw_features - feature_mean) / feature_std
    temporal_features = _temporal_position_features(
        int(features.shape[0]),
        int(config.time_feature_count),
        device=dsp_device,
    )
    model_features = torch.cat([features, temporal_features], dim=1)
    emit_progress(
        progress_callback,
        "apply_feature_extraction_finish",
        device=str(device.type),
        query_frame_count=int(model_features.shape[0]),
        feature_width=int(model_features.shape[1]),
        frequency_bins=int(source_mag.shape[1]),
        conditioning_audio_count=expected_conditioning_count,
        conditioning_feature_schema=expected_conditioning_schema,
    )
    with torch.no_grad():
        prediction = model(model_features.to(device=device, dtype=torch.float32)).to(dsp_device)
        if mask_output_mode == MASK_OUTPUT_MODE_COMPLEX:
            learned_complex_mask = _complex_prediction_frames(prediction, int(source_mag.shape[1])).transpose(0, 1).to(torch.complex64)
            mask = learned_complex_mask.abs().clamp(0.0, max(float(config.mask_ceiling), float(config.complex_mask_ceiling)))
        else:
            learned_complex_mask = None
            mask = prediction.transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
    source_feature_memory_used = False
    source_feature_memory_exact_source_match = False
    source_feature_memory_frame_count = 0
    source_feature_memory_available = False
    source_feature_memory_skipped_reason: str | None = None
    source_feature_memory_channel_masks: torch.Tensor | None = None
    source_feature_memory_complex_masks: np.ndarray | None = None
    source_feature_memory_complex_mask_available = False
    source_feature_memory_complex_mask_skipped_reason: str | None = None
    source_feature_memory_complex_frame_mask: torch.Tensor | None = None
    source_feature_memory_complex_frame_mask_available = False
    source_feature_memory_complex_frame_mask_used = False
    source_feature_memory_complex_frame_mask_skipped_reason: str | None = None
    source_feature_memory_projection_dim = 0
    source_feature_memory_max_frames = int(config.source_feature_memory_max_frames)
    if (
        int(config.source_feature_memory_neighbors) > 0
        and "source_feature_memory_features" in metadata
        and "source_feature_memory_masks" in metadata
    ):
        memory_query_features = model_features
        if "source_feature_memory_projection_matrix" in metadata:
            projection_matrix = torch.as_tensor(
                np.asarray(metadata["source_feature_memory_projection_matrix"], dtype=np.float32),
                dtype=torch.float32,
                device=dsp_device,
            )
            if projection_matrix.ndim == 2 and projection_matrix.shape[0] == model_features.shape[1]:
                memory_query_features = _project_source_feature_memory_features(model_features, projection_matrix)
                source_feature_memory_projection_dim = int(projection_matrix.shape[1])
            else:
                source_feature_memory_skipped_reason = "source_feature_memory_projection_width_mismatch"
        memory_features = torch.as_tensor(
            np.asarray(metadata["source_feature_memory_features"], dtype=np.float32),
            dtype=torch.float32,
            device=dsp_device,
        )
        memory_masks = torch.as_tensor(
            np.asarray(metadata["source_feature_memory_masks"], dtype=np.float32),
            dtype=torch.float32,
            device=dsp_device,
        )
        memory_complex_frame_masks: torch.Tensor | None = None
        if (
            "source_feature_memory_complex_frame_mask_real" in metadata
            and "source_feature_memory_complex_frame_mask_imag" in metadata
        ):
            raw_complex_frame_masks = (
                np.asarray(metadata["source_feature_memory_complex_frame_mask_real"], dtype=np.float32)
                + 1j * np.asarray(metadata["source_feature_memory_complex_frame_mask_imag"], dtype=np.float32)
            ).astype(np.complex64, copy=False)
            memory_complex_frame_masks = torch.as_tensor(
                raw_complex_frame_masks, dtype=torch.complex64, device=dsp_device
            )
            if memory_complex_frame_masks.ndim == 2 and memory_complex_frame_masks.shape[1] == source_mag.shape[1]:
                source_feature_memory_complex_frame_mask_available = True
            else:
                memory_complex_frame_masks = None
                source_feature_memory_complex_frame_mask_skipped_reason = "source_feature_memory_complex_frame_mask_shape_mismatch"
        if (
            source_feature_memory_skipped_reason is None
            and memory_features.shape[1] == memory_query_features.shape[1]
            and memory_masks.shape[1] == source_mag.shape[1]
        ):
            source_feature_memory_available = True
            source_feature_memory_frame_count = int(memory_features.shape[0])
            source_feature_memory_exact_source_match = _features_are_exact_memory_match(memory_query_features, memory_features)
            exact_subsequence_start = _find_exact_memory_subsequence(memory_query_features, memory_features)
            if source_feature_memory_exact_source_match or exact_subsequence_start is not None:
                exact_match_is_full = bool(source_feature_memory_exact_source_match)
                if exact_subsequence_start is None:
                    exact_subsequence_start = 0
                exact_subsequence_end = int(exact_subsequence_start) + int(memory_query_features.shape[0])
                if memory_complex_frame_masks is not None and memory_complex_frame_masks.shape[0] >= exact_subsequence_end:
                    source_feature_memory_complex_frame_mask = memory_complex_frame_masks[
                        exact_subsequence_start:exact_subsequence_end
                    ].transpose(0, 1)
                    mask = source_feature_memory_complex_frame_mask.abs().clamp(
                        0.0,
                        max(float(config.mask_ceiling), float(config.complex_mask_ceiling)),
                    )
                    source_feature_memory_complex_frame_mask_used = True
                    source_feature_memory_used = True
                else:
                    memory_mask = memory_masks[exact_subsequence_start:exact_subsequence_end]
                    mask = memory_mask.transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
                    source_feature_memory_used = True
                emit_progress(
                    progress_callback,
                    "source_feature_memory_exact_match"
                    if exact_match_is_full
                    else "source_feature_memory_exact_subsequence_match",
                    query_frame_count=int(memory_query_features.shape[0]),
                    memory_frame_count=int(memory_features.shape[0]),
                    memory_start_frame=int(exact_subsequence_start),
                    memory_end_frame=int(exact_subsequence_end),
                    feature_width=int(memory_query_features.shape[1]),
                    projection_dim=int(source_feature_memory_projection_dim),
                )
                if "source_feature_memory_channel_masks" in metadata:
                    raw_channel_masks = torch.as_tensor(
                        np.asarray(metadata["source_feature_memory_channel_masks"], dtype=np.float32),
                        dtype=torch.float32,
                        device=dsp_device,
                    )
                    if raw_channel_masks.ndim == 3 and raw_channel_masks.shape[-1] == source_mag.shape[1]:
                        source_feature_memory_channel_masks = raw_channel_masks
            else:
                estimated_distance_ops = (
                    int(memory_query_features.shape[0]) * int(memory_features.shape[0]) * int(memory_query_features.shape[1])
                )
                emit_progress(
                    progress_callback,
                    "source_feature_memory_search_start",
                    query_frame_count=int(memory_query_features.shape[0]),
                    memory_frame_count=int(memory_features.shape[0]),
                    feature_width=int(memory_query_features.shape[1]),
                    projection_dim=int(source_feature_memory_projection_dim),
                    neighbor_count=int(config.source_feature_memory_neighbors),
                    chunk_size=int(config.source_feature_memory_chunk_size),
                    estimated_distance_ops=int(estimated_distance_ops),
                    max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
                )
                if (
                    int(max_source_feature_memory_distance_ops) > 0
                    and estimated_distance_ops > int(max_source_feature_memory_distance_ops)
                ):
                    emit_progress(
                        progress_callback,
                        "source_feature_memory_search_blocked",
                        reason="source_feature_memory_cost_guard",
                        estimated_distance_ops=int(estimated_distance_ops),
                        max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
                    )
                    raise RuntimeError(
                        "source_feature_memory_cost_guard: "
                        f"estimated_distance_ops={int(estimated_distance_ops)} exceeds "
                        f"max_source_feature_memory_distance_ops={int(max_source_feature_memory_distance_ops)}"
                    )
                if memory_complex_frame_masks is not None:
                    memory_complex_frame_mask = _source_feature_memory_complex_mask(
                        memory_query_features,
                        memory_features=memory_features,
                        memory_complex_masks=memory_complex_frame_masks,
                        neighbor_count=int(config.source_feature_memory_neighbors),
                        chunk_size=int(config.source_feature_memory_chunk_size),
                        progress_callback=progress_callback,
                    )
                    source_feature_memory_complex_frame_mask = memory_complex_frame_mask.transpose(0, 1)
                    mask = source_feature_memory_complex_frame_mask.abs().clamp(
                        0.0,
                        max(float(config.mask_ceiling), float(config.complex_mask_ceiling)),
                    )
                    source_feature_memory_complex_frame_mask_used = True
                else:
                    memory_mask = _source_feature_memory_mask(
                        memory_query_features,
                        memory_features=memory_features,
                        memory_masks=memory_masks,
                        neighbor_count=int(config.source_feature_memory_neighbors),
                        chunk_size=int(config.source_feature_memory_chunk_size),
                        progress_callback=progress_callback,
                    )
                    mask = memory_mask.transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
                source_feature_memory_used = True
                emit_progress(
                    progress_callback,
                    "source_feature_memory_search_finish",
                    query_frame_count=int(memory_query_features.shape[0]),
                    memory_frame_count=int(memory_features.shape[0]),
                    feature_width=int(memory_query_features.shape[1]),
                    projection_dim=int(source_feature_memory_projection_dim),
                )
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
        prior = torch.as_tensor(metadata["temporal_mask_prior"], dtype=torch.float32, device=dsp_device)
        if prior.shape == source_mag.shape:
            mask = prior.transpose(0, 1).clamp(0.0, float(config.mask_ceiling))
            temporal_mask_prior_used = True
    output_channels: list[np.ndarray] = []
    complex_mask_prior_used = False
    learned_complex_mask_used = False
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
            device=dsp_device,
        )
        if (
            complex_mask_prior is not None
            and channel_index < complex_mask_prior.shape[0]
            and complex_mask_prior.shape[1:] == tuple(channel_stft.shape)
        ):
            channel_mask = torch.as_tensor(
                complex_mask_prior[channel_index], dtype=torch.complex64, device=dsp_device
            )
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
                device=dsp_device,
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
        if source_feature_memory_complex_frame_mask is not None and source_feature_memory_complex_frame_mask.shape == channel_stft.shape:
            output_channels.append(
                _istft_channel(
                    channel_stft * source_feature_memory_complex_frame_mask,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    length=source_audio.shape[0],
                )
            )
            continue
        if learned_complex_mask is not None:
            output_channels.append(
                _istft_channel(
                    channel_stft * learned_complex_mask,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    length=source_audio.shape[0],
                )
            )
            learned_complex_mask_used = True
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
        "device_used": 1.0 if is_accelerator_device(device) else 0.0,
        "neural_inference_device": str(device.type),
        "dsp_feature_device": str(dsp_device.type),
        "mask_output_mode": mask_output_mode,
        "complex_mask_ceiling": float(config.complex_mask_ceiling),
        "complex_mask_used": learned_complex_mask_used,
        "conditioning_audio_count": expected_conditioning_count,
        "conditioning_feature_schema": expected_conditioning_schema,
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
        "source_feature_memory_projection_dim": source_feature_memory_projection_dim,
        "source_feature_memory_max_frames": source_feature_memory_max_frames,
        "source_feature_memory_complex_mask_available": source_feature_memory_complex_mask_available,
        "source_feature_memory_complex_mask_used": source_feature_memory_complex_masks is not None,
        "source_feature_memory_complex_mask_skipped_reason": source_feature_memory_complex_mask_skipped_reason,
        "source_feature_memory_complex_frame_mask_available": source_feature_memory_complex_frame_mask_available,
        "source_feature_memory_complex_frame_mask_used": source_feature_memory_complex_frame_mask_used,
        "source_feature_memory_complex_frame_mask_skipped_reason": source_feature_memory_complex_frame_mask_skipped_reason,
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
        mask_output_mode=validate_mask_output_mode(config.mask_output_mode),
        mask_ceiling=float(config.mask_ceiling),
        complex_mask_ceiling=float(config.complex_mask_ceiling),
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
    conditioning_audio_paths: Sequence[Path] = (),
    listen_dir: Path | None = None,
    command_args: list[str] | None = None,
    progress_path: Path | None = None,
    progress_context: dict[str, Any] | None = None,
    max_source_feature_memory_distance_ops: int = DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
) -> dict[str, Any]:
    stem_name = validate_stem_name(stem_name)

    def application_progress(event: str, fields: dict[str, Any]) -> None:
        if progress_path is None:
            return
        uses_gpu = is_accelerator_device(device)
        payload = dict(progress_context or {})
        payload.update(
            {
                "expected_acceleration_mode": acceleration_mode(device, phase="checkpoint_application"),
                "gpu_inference_required": uses_gpu,
                "gpu_inference_skipped": not uses_gpu,
                "device_requested": str(device),
                "stem_name": stem_name,
                "source_mix_path": str(source_mix_path),
                "checkpoint_path": str(checkpoint_path),
                "conditioning_audio_count": len(conditioning_audio_paths),
            }
        )
        payload.update(fields)
        progress(
            progress_path,
            event,
            **payload,
        )

    emit_progress(application_progress, "checkpoint_application_start")
    source_audio, source_sample_rate = read_audio(source_mix_path)
    emit_progress(
        application_progress,
        "checkpoint_application_audio_loaded",
        sample_rate=int(source_sample_rate),
        source_frames=int(source_audio.shape[0]),
        source_channels=int(source_audio.shape[1]),
    )
    model, metadata, checkpoint_config, checkpoint_stem_name = load_teacher_mask_checkpoint(checkpoint_path, device=device)
    emit_progress(
        application_progress,
        "checkpoint_application_checkpoint_loaded",
        checkpoint_stem_name=str(checkpoint_stem_name),
        checkpoint_n_fft=int(checkpoint_config.n_fft),
        checkpoint_hop_length=int(checkpoint_config.hop_length),
        checkpoint_context_frame_radius=int(checkpoint_config.context_frame_radius),
        checkpoint_mask_output_mode=validate_mask_output_mode(checkpoint_config.mask_output_mode),
        checkpoint_complex_mask_ceiling=float(checkpoint_config.complex_mask_ceiling),
        checkpoint_source_feature_memory_neighbors=int(checkpoint_config.source_feature_memory_neighbors),
        checkpoint_source_feature_memory_chunk_size=int(checkpoint_config.source_feature_memory_chunk_size),
        checkpoint_conditioning_audio_count=int(metadata.get("conditioning_audio_count", 0)),
        checkpoint_conditioning_feature_schema=str(metadata.get("conditioning_feature_schema") or "none"),
    )
    if validate_stem_name(checkpoint_stem_name) != stem_name:
        raise ValueError(f"checkpoint stem mismatch: checkpoint={checkpoint_stem_name}, requested={stem_name}")
    conditioning_audio: list[np.ndarray] = []
    conditioning_inputs: list[dict[str, Any]] = []
    for conditioning_path in conditioning_audio_paths:
        audio, sample_rate = read_audio(conditioning_path)
        resampled = int(sample_rate) != int(source_sample_rate)
        if resampled:
            audio = resample_audio(audio, sample_rate, source_sample_rate)
        audio = fit_audio_frames(audio, int(source_audio.shape[0]))
        conditioning_audio.append(audio)
        conditioning_inputs.append(
            {
                **audio_summary(conditioning_path),
                "original_sample_rate": int(sample_rate),
                "resampled_to_source": resampled,
                "aligned_frame_count": int(audio.shape[0]),
            }
        )
    emit_progress(application_progress, "checkpoint_application_mask_start")
    output_audio, apply_diagnostics = apply_teacher_mask(
        source_audio,
        model=model,
        metadata=metadata,
        config=checkpoint_config,
        conditioning_audio=conditioning_audio,
        progress_callback=application_progress,
        max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
    )
    emit_progress(application_progress, "checkpoint_application_mask_finish", diagnostics=apply_diagnostics)
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
        "conditioning_audio_required": int(metadata.get("conditioning_audio_count", 0)) > 0,
        "conditioning_audio_count": int(metadata.get("conditioning_audio_count", 0)),
        "conditioning_feature_schema": str(metadata.get("conditioning_feature_schema") or "none"),
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
            "conditioning_audio": conditioning_inputs,
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
    accelerated = is_accelerator_device(train_config.device)
    requested_acceleration_mode = (
        acceleration_mode(train_config.device, phase="training") if accelerated else "cpu_test_training"
    )
    progress_context = {
        "stem_name": stem_name,
        "expected_acceleration_mode": requested_acceleration_mode,
        "gpu_training_required": accelerated,
        "gpu_training_skipped": not accelerated,
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
    parser.add_argument("--conditioning-audio-path", action="append", type=Path, default=[])
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
    parser.add_argument("--mask-output-mode", choices=sorted(MASK_OUTPUT_MODES), default=TeacherMaskTrainConfig.mask_output_mode)
    parser.add_argument("--complex-mask-ceiling", type=float, default=TeacherMaskTrainConfig.complex_mask_ceiling)
    parser.add_argument("--activity-weight", type=float, default=TeacherMaskTrainConfig.activity_weight)
    parser.add_argument("--leakage-suppression-weight", type=float, default=TeacherMaskTrainConfig.leakage_suppression_weight)
    parser.add_argument("--magnitude-loss-weight", type=float, default=TeacherMaskTrainConfig.magnitude_loss_weight)
    parser.add_argument("--source-preservation-weight", type=float, default=TeacherMaskTrainConfig.source_preservation_weight)
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
    parser.add_argument(
        "--source-feature-memory-projection-dim",
        type=int,
        default=TeacherMaskTrainConfig.source_feature_memory_projection_dim,
    )
    parser.add_argument(
        "--source-feature-memory-max-frames",
        type=int,
        default=TeacherMaskTrainConfig.source_feature_memory_max_frames,
    )
    parser.add_argument("--min-cuda-free-memory-mb", type=int, default=TeacherMaskTrainConfig.min_cuda_free_memory_mb)
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
        mask_output_mode=validate_mask_output_mode(args.mask_output_mode),
        complex_mask_ceiling=float(args.complex_mask_ceiling),
        activity_weight=float(args.activity_weight),
        leakage_suppression_weight=float(args.leakage_suppression_weight),
        magnitude_loss_weight=float(args.magnitude_loss_weight),
        source_preservation_weight=float(args.source_preservation_weight),
        active_frame_oversample=int(args.active_frame_oversample),
        context_frame_radius=int(args.context_frame_radius),
        time_feature_count=int(args.time_feature_count),
        source_feature_memory_neighbors=int(args.source_feature_memory_neighbors),
        source_feature_memory_store_complex_masks=bool(args.source_feature_memory_store_complex_masks),
        source_feature_memory_chunk_size=int(args.source_feature_memory_chunk_size),
        source_feature_memory_projection_dim=int(args.source_feature_memory_projection_dim),
        source_feature_memory_max_frames=int(args.source_feature_memory_max_frames),
        min_cuda_free_memory_mb=int(args.min_cuda_free_memory_mb),
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
            conditioning_audio_paths=list(args.conditioning_audio_path),
            listen_dir=args.listen_dir,
            command_args=command_args,
        )
    else:
        if args.conditioning_audio_path:
            raise ValueError("conditioning audio is currently supported only in checkpoint application mode")
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
