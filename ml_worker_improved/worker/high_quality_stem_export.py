from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as torch_functional
import torchaudio
from demucs.apply import apply_model
import librosa
import essentia.standard as essentia_std

from .device_policy import resolve_torch_device
from .instrument_melody_profile import (
    SEPARATION_SAMPLE_RATE,
    _ensure_stereo,
    _load_demucs_model,
)
from .stem_separator_defaults import (
    DEFAULT_STEM_EXPORT_INCLUDE_MELODY_STEM,
    DEFAULT_STEM_EXPORT_MODEL,
    DEFAULT_STEM_EXPORT_SHARPEN_BLEED,
)

HYBRID_STEM_EXPORT_MODEL = "hybrid_htdemucs_ft_6s"
HYBRID_BS_STEM_EXPORT_MODEL = "hybrid_htdemucs_ft_bs_roformer_6s"
HYBRID_STAGE1_MODEL = "htdemucs_ft"
HYBRID_STAGE2_MODEL = "htdemucs_6s"
FINAL_STEM_EXPORT_SUBTYPE = "PCM_24"
HYBRID_INTERMEDIATE_SUBTYPE = "FLOAT"
BS_ROFORMER_REPO = Path("/tmp/msst_zfturbo")
BS_ROFORMER_CONFIG = Path("/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml")
BS_ROFORMER_CKPT = Path("/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt")
DEFAULT_MELODY_REBALANCE_SOURCES = ("other", "guitar", "piano")
HYBRID_VOCAL_FUSION_LABEL = "spectral_split_bs_low_ft_high_v2"


def _build_no_reference_stem_export_contract(*, route_name: str) -> dict[str, Any]:
    return {
        "full_track": True,
        "benchmark_compatible": True,
        "target_reference_assisted": False,
        "target_reference_used": False,
        "target_reference_used_at_application": False,
        "application_reference_free": True,
        "delivery_allowed_without_reference": True,
        "registered_model_candidate": True,
        "fallback_used": False,
        "hidden_inference": False,
        "per_track_hardcode_allowed": False,
        "not_timestamp_hardcoded": True,
        "route_name": str(route_name),
    }


def _emit_progress(message: str) -> None:
    print(f"[hq_stem_export] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_hybrid_stage_models(candidate: dict[str, Any] | None = None) -> dict[str, str]:
    raw = candidate if isinstance(candidate, dict) else {}
    requested_device = str(raw.get("requestedDevice", "auto")).strip() or "auto"
    stage1_model = str(raw.get("stage1Model", HYBRID_STAGE1_MODEL)).strip() or HYBRID_STAGE1_MODEL
    stage2_model = str(raw.get("stage2Model", HYBRID_STAGE2_MODEL)).strip() or HYBRID_STAGE2_MODEL
    return {
        "requestedDevice": requested_device,
        "stage1Model": stage1_model,
        "stage2Model": stage2_model,
        "stage1RequestedDevice": str(raw.get("stage1RequestedDevice", requested_device)).strip() or requested_device,
        "stage2RequestedDevice": str(raw.get("stage2RequestedDevice", requested_device)).strip() or requested_device,
        "stage1Split": bool(raw.get("stage1Split", True)),
        "stage1Progress": bool(raw.get("stage1Progress", True)),
        "stage1Shifts": max(1, min(8, int(raw.get("stage1Shifts", 1)))),
        "stage1Overlap": max(0.0, min(0.99, float(raw.get("stage1Overlap", 0.25)))),
        "stage1Segment": max(0.0, min(600.0, float(raw.get("stage1Segment", 0.0)))),
        "stage2Split": bool(raw.get("stage2Split", True)),
        "stage2Progress": bool(raw.get("stage2Progress", True)),
        "stage2Shifts": max(1, min(8, int(raw.get("stage2Shifts", 1)))),
        "stage2Overlap": max(0.0, min(0.99, float(raw.get("stage2Overlap", 0.25)))),
        "stage2Segment": max(0.0, min(600.0, float(raw.get("stage2Segment", 0.0)))),
    }


def _normalize_bs_roformer_runtime(candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = candidate if isinstance(candidate, dict) else {}
    return {
        "requestedDevice": str(raw.get("requestedDevice", "auto")).strip() or "auto",
        "deviceIds": [
            str(value).strip()
            for value in (raw.get("deviceIds", ["0"]) if isinstance(raw.get("deviceIds", ["0"]), list) else ["0"])
            if str(value).strip()
        ],
        "forceCpu": bool(raw.get("forceCpu", False)),
        "disableDetailedPbar": bool(raw.get("disableDetailedPbar", False)),
        "repoPath": str(raw.get("repoPath", str(BS_ROFORMER_REPO))).strip() or str(BS_ROFORMER_REPO),
        "configPath": str(raw.get("configPath", str(BS_ROFORMER_CONFIG))).strip() or str(BS_ROFORMER_CONFIG),
        "checkpointPath": str(raw.get("checkpointPath", str(BS_ROFORMER_CKPT))).strip() or str(BS_ROFORMER_CKPT),
        "useTta": bool(raw.get("useTta", True)),
        "extractInstrumental": bool(raw.get("extractInstrumental", True)),
        "filenameTemplate": str(raw.get("filenameTemplate", "{instr}")).strip() or "{instr}",
        "configOverrideYaml": str(raw.get("configOverrideYaml", "")),
        "extraCliArgs": [
            str(value).strip()
            for value in (raw.get("extraCliArgs", []) if isinstance(raw.get("extraCliArgs", []), list) else [])
            if str(value).strip()
        ],
    }


def _effective_requested_device(explicit_device: str | None, override_device: str | None) -> str | None:
    if explicit_device is not None and str(explicit_device).strip():
        return str(explicit_device).strip()
    if override_device is None:
        return None
    normalized = str(override_device).strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _normalize_demucs_runtime(candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = candidate if isinstance(candidate, dict) else {}
    return {
        "split": bool(raw.get("split", True)),
        "progress": bool(raw.get("progress", True)),
        "shifts": max(1, min(8, int(raw.get("shifts", 1)))),
        "overlap": max(0.0, min(0.99, float(raw.get("overlap", 0.25)))),
        "segment": max(0.0, min(600.0, float(raw.get("segment", 0.0)))),
    }


def _extract_hybrid_stage_runtime(stage_models: dict[str, Any], stage_prefix: str) -> dict[str, Any]:
    stage_name = str(stage_prefix).strip().lower()
    if stage_name not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported hybrid stage runtime request: {stage_prefix}")

    return _normalize_demucs_runtime({
        "split": stage_models.get(f"{stage_name}Split", True),
        "progress": stage_models.get(f"{stage_name}Progress", True),
        "shifts": stage_models.get(f"{stage_name}Shifts", 1),
        "overlap": stage_models.get(f"{stage_name}Overlap", 0.25),
        "segment": stage_models.get(f"{stage_name}Segment", 0.0),
    })


_emit_progress("module_loaded high_quality_stem_export")


def _resolve_preview_frame_window(
    sample_rate: int,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[int, int | None]:
    start_frame = max(0, int(round(float(start_sec or 0.0) * sample_rate)))
    if end_sec is None:
        return start_frame, None
    end_frame = max(start_frame, int(round(float(end_sec) * sample_rate)))
    return start_frame, end_frame


def _read_audio_window(
    audio_path: str | Path,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(audio_path)) as handle:
        sample_rate = int(handle.samplerate)
        start_frame, end_frame = _resolve_preview_frame_window(
            sample_rate,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        handle.seek(start_frame)
        frames = -1 if end_frame is None else max(0, end_frame - start_frame)
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)
    return audio.astype(np.float32), sample_rate


def _resample_audio_to_sample_rate(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    resampled_channels = [
        librosa.resample(
            audio[:, channel_index].astype(np.float32, copy=False),
            orig_sr=int(source_sample_rate),
            target_sr=int(target_sample_rate),
        ).astype(np.float32, copy=False)
        for channel_index in range(audio.shape[1])
    ]
    frame_count = min(channel.shape[0] for channel in resampled_channels)
    return np.stack(
        [channel[:frame_count] for channel in resampled_channels],
        axis=1,
    ).astype(np.float32, copy=False)


def separate_track_stems_full_res(
    audio_path: str | Path,
    *,
    model_name: str = DEFAULT_STEM_EXPORT_MODEL,
    device: str | None = None,
    hybrid_stage_models: dict[str, Any] | None = None,
    bs_roformer_runtime: dict[str, Any] | None = None,
    demucs_runtime: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    _emit_progress(f"ENTER separate_track_stems_full_res model={model_name} audio_path={audio_path}")
    if model_name == HYBRID_STEM_EXPORT_MODEL:
        stems = _separate_track_stems_hybrid_vocals_first(
            audio_path,
            device=device,
            hybrid_stage_models=hybrid_stage_models,
            bs_roformer_runtime=bs_roformer_runtime,
        )
        _emit_progress(f"EXIT separate_track_stems_full_res model={model_name} stems={list(stems.keys())}")
        return stems
    if model_name == HYBRID_BS_STEM_EXPORT_MODEL:
        stems = _separate_track_stems_hybrid_bs_vocals_first(
            audio_path,
            device=device,
            hybrid_stage_models=hybrid_stage_models,
            bs_roformer_runtime=bs_roformer_runtime,
        )
        _emit_progress(f"EXIT separate_track_stems_full_res model={model_name} stems={list(stems.keys())}")
        return stems

    resolved_device = str(resolve_torch_device(task_name="hq_stem_export", requested=device, allow_cpu=True))
    runtime = _normalize_demucs_runtime(demucs_runtime)
    model = _load_demucs_model(model_name, resolved_device)
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = _ensure_stereo(waveform)
    if sample_rate != SEPARATION_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SEPARATION_SAMPLE_RATE)

    with torch.no_grad():
        apply_kwargs: dict[str, Any] = {
            "split": bool(runtime["split"]),
            "progress": bool(runtime["progress"]),
            "device": resolved_device,
            "shifts": int(runtime["shifts"]),
            "overlap": float(runtime["overlap"]),
        }
        if float(runtime["segment"]) > 0:
            apply_kwargs["segment"] = float(runtime["segment"])
        separated = apply_model(
            model,
            waveform.unsqueeze(0),
            **apply_kwargs,
        )

    if separated.ndim != 4 or separated.shape[0] != 1:
        raise ValueError(f"Unexpected Demucs output shape: {tuple(separated.shape)}")

    stems: dict[str, np.ndarray] = {}
    for index, stem_name in enumerate(model.sources):
        stem_audio = separated[0, index].detach().cpu().numpy().T.astype(np.float32)
        stems[str(stem_name)] = stem_audio

    if resolved_device == "cuda":
        torch.cuda.empty_cache()

    _emit_progress(
        f"EXIT separate_track_stems_full_res model={model_name} stems={list(stems.keys())} sample_rate={SEPARATION_SAMPLE_RATE}"
    )
    return stems


def _build_instrumental_mix_from_stage1_non_vocals(
    stage1_stems: dict[str, np.ndarray],
) -> np.ndarray:
    non_vocal_names = [stem_name for stem_name in stage1_stems.keys() if stem_name != "vocals"]
    if not non_vocal_names:
        raise ValueError("Stage1 separator did not return non-vocal stems")

    instrumental: np.ndarray | None = None
    for stem_name in non_vocal_names:
        stem_audio = np.asarray(stage1_stems[stem_name], dtype=np.float32)
        if instrumental is None:
            instrumental = np.zeros_like(stem_audio, dtype=np.float32)
        instrumental += stem_audio
    if instrumental is None:
        raise ValueError("Stage1 separator did not return non-vocal stems")
    peak = float(np.max(np.abs(instrumental))) if instrumental.size else 0.0
    if peak > 1.0:
        instrumental = instrumental / peak
    return instrumental.astype(np.float32, copy=False)


def _safe_audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return math.sqrt(float(np.mean(np.square(mono, dtype=np.float32))))


def _resolve_dsp_device(
    requested: str | torch.device | None = None,
) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if isinstance(requested, str) and requested.strip():
        return resolve_torch_device(task_name="hq_stem_export_dsp", requested=requested, allow_cpu=True)
    return resolve_torch_device(task_name="hq_stem_export_dsp", allow_cpu=True)


def _torch_hann_window(
    n_fft: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.hann_window(n_fft, device=device, dtype=torch.float32)


def _torch_audio_channels(
    audio: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    return torch.from_numpy(array.T.copy()).to(device=device, dtype=torch.float32)


def _torch_median_filter(
    magnitude: torch.Tensor,
    *,
    kernel_size: int,
    axis: str,
) -> torch.Tensor:
    if kernel_size <= 1:
        return magnitude
    pad = kernel_size // 2
    tensor = magnitude.unsqueeze(1)
    if axis == "time":
        padded = torch_functional.pad(tensor, (pad, pad, 0, 0), mode="reflect")
        unfolded = padded.unfold(3, kernel_size, 1)
    elif axis == "frequency":
        padded = torch_functional.pad(tensor, (0, 0, pad, pad), mode="reflect")
        unfolded = padded.unfold(2, kernel_size, 1)
    else:
        raise ValueError(f"Unsupported median filter axis: {axis}")
    return unfolded.median(dim=-1).values.squeeze(1)


def _split_harmonic_percussive_torch(
    audio: np.ndarray,
    *,
    harmonic_margin: float,
    percussive_margin: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    length = int(audio.shape[0])
    waveform = _torch_audio_channels(audio, device=device)
    window = _torch_hann_window(n_fft, device=device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs()
    harmonic_strength = _torch_median_filter(magnitude, kernel_size=31, axis="time")
    percussive_strength = _torch_median_filter(magnitude, kernel_size=31, axis="frequency")
    harmonic_mask = harmonic_strength / (harmonic_strength + percussive_strength * float(percussive_margin) + 1e-8)
    percussive_mask = percussive_strength / (percussive_strength + harmonic_strength * float(harmonic_margin) + 1e-8)
    harmonic_stft = stft * harmonic_mask
    percussive_stft = stft * percussive_mask
    harmonic = torch.istft(
        harmonic_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=length,
    ).transpose(0, 1)
    percussive = torch.istft(
        percussive_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=length,
    ).transpose(0, 1)
    return (
        harmonic.detach().cpu().numpy().astype(np.float32),
        percussive.detach().cpu().numpy().astype(np.float32),
    )


def _split_harmonic_percussive(
    audio: np.ndarray,
    *,
    harmonic_margin: float,
    percussive_margin: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    dsp_device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_device = _resolve_dsp_device(dsp_device)
    if target_device.type == "cuda":
        return _split_harmonic_percussive_torch(
            audio,
            harmonic_margin=harmonic_margin,
            percussive_margin=percussive_margin,
            n_fft=n_fft,
            hop_length=hop_length,
            device=target_device,
        )
    harmonic = np.zeros_like(audio, dtype=np.float32)
    percussive = np.zeros_like(audio, dtype=np.float32)
    for channel_index in range(audio.shape[1]):
        stft = librosa.stft(audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        harmonic_stft, percussive_stft = librosa.decompose.hpss(
            stft,
            margin=(harmonic_margin, percussive_margin),
        )
        harmonic[:, channel_index] = librosa.istft(
            harmonic_stft,
            hop_length=hop_length,
            length=audio.shape[0],
        ).astype(np.float32)
        percussive[:, channel_index] = librosa.istft(
            percussive_stft,
            hop_length=hop_length,
            length=audio.shape[0],
        ).astype(np.float32)
    return harmonic, percussive


def _component_transfer_envelope(
    primary_component: np.ndarray,
    secondary_component: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    activity_floor_ratio: float = 0.08,
    smoothing_frames: int = 11,
    dsp_device: str | torch.device | None = None,
) -> np.ndarray:
    primary_mono = primary_component.mean(axis=1).astype(np.float32)
    secondary_mono = secondary_component.mean(axis=1).astype(np.float32)
    target_device = _resolve_dsp_device(dsp_device)
    if target_device.type == "cuda":
        window = _torch_hann_window(n_fft, device=target_device)
        primary_tensor = torch.from_numpy(primary_mono.copy()).to(device=target_device, dtype=torch.float32)
        secondary_tensor = torch.from_numpy(secondary_mono.copy()).to(device=target_device, dtype=torch.float32)
        primary_stft = (
            torch.stft(primary_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            .abs()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        secondary_stft = (
            torch.stft(secondary_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            .abs()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    else:
        primary_stft = np.abs(librosa.stft(primary_mono, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
        secondary_stft = np.abs(librosa.stft(secondary_mono, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
    primary_frame_energy = np.mean(primary_stft, axis=0).astype(np.float32)
    secondary_frame_energy = np.mean(secondary_stft, axis=0).astype(np.float32)
    if primary_frame_energy.size == 0:
        return np.zeros_like(primary_component, dtype=np.float32)

    reference_energy = float(np.percentile(primary_frame_energy, 95)) if primary_frame_energy.size else 0.0
    abs_activity = np.clip(
        primary_frame_energy / max(reference_energy * float(activity_floor_ratio), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    dominance = (
        primary_frame_energy
        / np.maximum(primary_frame_energy + secondary_frame_energy, 1e-8)
    ).astype(np.float32)
    frame_activity = np.clip(abs_activity * dominance, 0.0, 1.0)
    if int(smoothing_frames) > 1 and frame_activity.size:
        kernel = np.ones(int(smoothing_frames), dtype=np.float32) / float(smoothing_frames)
        frame_activity = np.convolve(frame_activity, kernel, mode="same").astype(np.float32)
    frame_activity = np.clip(frame_activity, 0.0, 1.0)

    frame_times = (np.arange(frame_activity.size, dtype=np.float32) * float(hop_length)).astype(np.float32)
    sample_positions = np.arange(primary_component.shape[0], dtype=np.float32)
    sample_activity = np.interp(
        sample_positions,
        frame_times,
        frame_activity,
        left=float(frame_activity[0]),
        right=float(frame_activity[-1]),
    ).astype(np.float32)
    return np.repeat(sample_activity[:, np.newaxis], primary_component.shape[1], axis=1).astype(np.float32)


def _rebalance_hybrid_stage2_stems(
    stage2_stems: dict[str, np.ndarray],
    *,
    drum_harmonic_move: float = 0.82,
    harmonic_stem_percussive_move: float = 0.42,
    cleanup_drum_harmonic_move: float = 0.22,
    cleanup_harmonic_stem_percussive_move: float = 0.65,
    dsp_device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    if "drums" not in stage2_stems or "other" not in stage2_stems:
        return {
            stem_name: np.asarray(stem_audio, dtype=np.float32)
            for stem_name, stem_audio in stage2_stems.items()
        }

    rebalanced = {
        stem_name: np.asarray(stem_audio, dtype=np.float32)
        for stem_name, stem_audio in stage2_stems.items()
    }

    drums = rebalanced["drums"]
    other = rebalanced["other"]
    guitar = rebalanced.get("guitar", np.zeros_like(other, dtype=np.float32))
    piano = rebalanced.get("piano", np.zeros_like(other, dtype=np.float32))

    drums_harmonic, drums_percussive = _split_harmonic_percussive(
        drums,
        harmonic_margin=1.0,
        percussive_margin=4.0,
        dsp_device=dsp_device,
    )
    other_harmonic, other_percussive = _split_harmonic_percussive(
        other,
        harmonic_margin=2.0,
        percussive_margin=1.0,
        dsp_device=dsp_device,
    )
    guitar_harmonic, guitar_percussive = _split_harmonic_percussive(
        guitar,
        harmonic_margin=2.0,
        percussive_margin=1.0,
        dsp_device=dsp_device,
    )
    piano_harmonic, piano_percussive = _split_harmonic_percussive(
        piano,
        harmonic_margin=2.0,
        percussive_margin=1.0,
        dsp_device=dsp_device,
    )

    harmonic_energies = np.asarray(
        [
            _safe_audio_rms(other_harmonic) * 1.0 + 1e-6,
            _safe_audio_rms(guitar_harmonic) * 0.7 + 1e-6,
            _safe_audio_rms(piano_harmonic) * 0.5 + 1e-6,
        ],
        dtype=np.float32,
    )
    harmonic_weights = harmonic_energies / float(np.sum(harmonic_energies))

    drum_harmonic_envelope = _component_transfer_envelope(
        drums_harmonic,
        drums_percussive,
        activity_floor_ratio=0.06,
        smoothing_frames=13,
        dsp_device=dsp_device,
    )
    other_percussive_envelope = _component_transfer_envelope(
        other_percussive,
        other_harmonic,
        activity_floor_ratio=0.05,
        smoothing_frames=9,
        dsp_device=dsp_device,
    )
    guitar_percussive_envelope = _component_transfer_envelope(
        guitar_percussive,
        guitar_harmonic,
        activity_floor_ratio=0.05,
        smoothing_frames=9,
        dsp_device=dsp_device,
    )
    piano_percussive_envelope = _component_transfer_envelope(
        piano_percussive,
        piano_harmonic,
        activity_floor_ratio=0.05,
        smoothing_frames=9,
        dsp_device=dsp_device,
    )

    moved_drum_harmonic = drums_harmonic * float(drum_harmonic_move) * drum_harmonic_envelope
    moved_other_percussive = other_percussive * float(harmonic_stem_percussive_move) * other_percussive_envelope
    moved_guitar_percussive = guitar_percussive * float(harmonic_stem_percussive_move) * guitar_percussive_envelope
    moved_piano_percussive = piano_percussive * float(harmonic_stem_percussive_move) * piano_percussive_envelope

    rebalanced["drums"] = (
        drums
        - moved_drum_harmonic
        + moved_other_percussive
        + moved_guitar_percussive
        + moved_piano_percussive
    ).astype(np.float32)
    rebalanced["other"] = (
        other
        + moved_drum_harmonic * float(harmonic_weights[0])
        - moved_other_percussive
    ).astype(np.float32)
    if "guitar" in rebalanced:
        rebalanced["guitar"] = (
            guitar
            + moved_drum_harmonic * float(harmonic_weights[1])
            - moved_guitar_percussive
        ).astype(np.float32)
    if "piano" in rebalanced:
        rebalanced["piano"] = (
            piano
            + moved_drum_harmonic * float(harmonic_weights[2])
            - moved_piano_percussive
        ).astype(np.float32)

    final_drums = rebalanced["drums"]
    final_other = rebalanced["other"]
    final_guitar = rebalanced.get("guitar", np.zeros_like(final_other, dtype=np.float32))
    final_piano = rebalanced.get("piano", np.zeros_like(final_other, dtype=np.float32))

    final_drums_harmonic, _ = _split_harmonic_percussive(
        final_drums,
        harmonic_margin=1.0,
        percussive_margin=4.0,
    )
    final_other_harmonic, final_other_percussive = _split_harmonic_percussive(
        final_other,
        harmonic_margin=2.0,
        percussive_margin=1.0,
    )
    final_guitar_harmonic, final_guitar_percussive = _split_harmonic_percussive(
        final_guitar,
        harmonic_margin=2.0,
        percussive_margin=1.0,
    )
    final_piano_harmonic, final_piano_percussive = _split_harmonic_percussive(
        final_piano,
        harmonic_margin=2.0,
        percussive_margin=1.0,
    )

    final_harmonic_energies = np.asarray(
        [
            _safe_audio_rms(final_other_harmonic) * 1.0 + 1e-6,
            _safe_audio_rms(final_guitar_harmonic) * 0.7 + 1e-6,
            _safe_audio_rms(final_piano_harmonic) * 0.5 + 1e-6,
        ],
        dtype=np.float32,
    )
    final_harmonic_weights = final_harmonic_energies / float(np.sum(final_harmonic_energies))

    moved_cleanup_drums_harmonic = final_drums_harmonic * float(cleanup_drum_harmonic_move)
    moved_cleanup_other_percussive = final_other_percussive * float(cleanup_harmonic_stem_percussive_move)
    moved_cleanup_guitar_percussive = final_guitar_percussive * float(cleanup_harmonic_stem_percussive_move)
    moved_cleanup_piano_percussive = final_piano_percussive * float(cleanup_harmonic_stem_percussive_move)

    rebalanced["drums"] = (
        final_drums
        - moved_cleanup_drums_harmonic
        + moved_cleanup_other_percussive
        + moved_cleanup_guitar_percussive
        + moved_cleanup_piano_percussive
    ).astype(np.float32)
    rebalanced["other"] = (
        final_other
        + moved_cleanup_drums_harmonic * float(final_harmonic_weights[0])
        - moved_cleanup_other_percussive
    ).astype(np.float32)
    if "guitar" in rebalanced:
        rebalanced["guitar"] = (
            final_guitar
            + moved_cleanup_drums_harmonic * float(final_harmonic_weights[1])
            - moved_cleanup_guitar_percussive
        ).astype(np.float32)
    if "piano" in rebalanced:
        rebalanced["piano"] = (
            final_piano
            + moved_cleanup_drums_harmonic * float(final_harmonic_weights[2])
            - moved_cleanup_piano_percussive
        ).astype(np.float32)
    return rebalanced


def _blend_vocal_stems(
    stage1_vocals: np.ndarray,
    bs_vocals: np.ndarray,
    *,
    low_bs_weight: float = 0.75,
    high_bs_weight: float = 0.25,
    inactive_low_bs_weight: float = 0.97,
    inactive_high_bs_weight: float = 0.9,
    crossover_hz: float = 3000.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    activity_abs_floor_ratio: float = 0.08,
    activity_rel_floor: float = 0.35,
    activity_smoothing_frames: int = 11,
) -> np.ndarray:
    stage1_vocals = np.asarray(stage1_vocals, dtype=np.float32)
    bs_vocals = np.asarray(bs_vocals, dtype=np.float32)
    if stage1_vocals.ndim != 2 or bs_vocals.ndim != 2 or stage1_vocals.shape[1] != bs_vocals.shape[1]:
        raise ValueError(
            f"Vocal fusion shape mismatch: stage1={stage1_vocals.shape} bs={bs_vocals.shape}"
        )
    frame_mismatch = abs(int(stage1_vocals.shape[0]) - int(bs_vocals.shape[0]))
    max_allowed_frame_mismatch = max(int(0.25 * SEPARATION_SAMPLE_RATE), int(n_fft))
    if frame_mismatch > max_allowed_frame_mismatch:
        raise ValueError(
            f"Vocal fusion frame mismatch is too large: stage1={stage1_vocals.shape} bs={bs_vocals.shape}"
        )
    if frame_mismatch:
        frame_count = min(stage1_vocals.shape[0], bs_vocals.shape[0])
        stage1_vocals = stage1_vocals[:frame_count]
        bs_vocals = bs_vocals[:frame_count]

    fused = np.zeros_like(stage1_vocals, dtype=np.float32)
    freq_bins = librosa.fft_frequencies(sr=SEPARATION_SAMPLE_RATE, n_fft=n_fft).astype(np.float32)
    low_mask = (freq_bins < float(crossover_hz)).astype(np.float32)[:, np.newaxis]
    high_mask = 1.0 - low_mask

    for channel_index in range(stage1_vocals.shape[1]):
        stage1_stft = librosa.stft(stage1_vocals[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        bs_stft = librosa.stft(bs_vocals[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        stage1_frame_energy = np.mean(np.abs(stage1_stft), axis=0).astype(np.float32)
        bs_frame_energy = np.mean(np.abs(bs_stft), axis=0).astype(np.float32)
        bs_reference_energy = float(np.percentile(bs_frame_energy, 95)) if bs_frame_energy.size else 0.0
        bs_abs_activity = np.clip(
            bs_frame_energy / max(bs_reference_energy * float(activity_abs_floor_ratio), 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        bs_rel_activity = np.clip(
            bs_frame_energy / np.maximum(stage1_frame_energy * float(activity_rel_floor), 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        bs_activity = np.maximum(bs_abs_activity, bs_rel_activity)
        if int(activity_smoothing_frames) > 1 and bs_activity.size:
            smoothing_kernel = np.ones(int(activity_smoothing_frames), dtype=np.float32) / float(activity_smoothing_frames)
            bs_activity = np.convolve(bs_activity, smoothing_kernel, mode="same").astype(np.float32)
        bs_activity = np.clip(bs_activity, 0.0, 1.0)
        dynamic_low_bs_weight = (
            float(low_bs_weight)
            + (float(inactive_low_bs_weight) - float(low_bs_weight)) * (1.0 - bs_activity)
        ).astype(np.float32)[np.newaxis, :]
        dynamic_high_bs_weight = (
            float(high_bs_weight)
            + (float(inactive_high_bs_weight) - float(high_bs_weight)) * (1.0 - bs_activity)
        ).astype(np.float32)[np.newaxis, :]
        fused_stft = (
            (stage1_stft * (1.0 - dynamic_low_bs_weight) + bs_stft * dynamic_low_bs_weight) * low_mask
            + (stage1_stft * (1.0 - dynamic_high_bs_weight) + bs_stft * dynamic_high_bs_weight) * high_mask
        )
        fused[:, channel_index] = librosa.istft(
            fused_stft,
            hop_length=hop_length,
            length=stage1_vocals.shape[0],
        ).astype(np.float32)
    return fused


def _single_stem_activity_gate(
    audio: np.ndarray,
    *,
    floor_ratio: float = 0.08,
    smoothing_frames: int = 11,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    if audio.size == 0:
        return np.zeros_like(audio, dtype=np.float32)
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    frame_energy = np.mean(np.abs(stft), axis=0).astype(np.float32)
    reference_energy = float(np.percentile(frame_energy, 95)) if frame_energy.size else 0.0
    frame_activity = np.clip(
        frame_energy / max(reference_energy * float(floor_ratio), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    if int(smoothing_frames) > 1 and frame_activity.size:
        kernel = np.ones(int(smoothing_frames), dtype=np.float32) / float(smoothing_frames)
        frame_activity = np.convolve(frame_activity, kernel, mode="same").astype(np.float32)
    frame_activity = np.clip(frame_activity, 0.0, 1.0)
    frame_activity = np.square(frame_activity).astype(np.float32)
    frame_times = (np.arange(frame_activity.size, dtype=np.float32) * float(hop_length)).astype(np.float32)
    sample_times = np.arange(audio.shape[0], dtype=np.float32)
    sample_activity = np.interp(
        sample_times,
        frame_times,
        frame_activity,
        left=float(frame_activity[0]) if frame_activity.size else 0.0,
        right=float(frame_activity[-1]) if frame_activity.size else 0.0,
    )
    return np.repeat(sample_activity[:, np.newaxis], audio.shape[1], axis=1).astype(np.float32)


def _cleanup_single_vocal_stem(
    vocals_audio: np.ndarray,
    mixture_audio: np.ndarray,
    *,
    sharpen_alpha: float = 0.0,
    tail_gate_strength: float = 0.0,
    tail_gate_floor_ratio: float = 0.08,
    tail_gate_smoothing_frames: int = 11,
    subtractive_strength: float = 0.0,
    subtractive_power: float = 1.0,
    subtractive_residual_floor: float = 0.12,
    subtractive_active_restore: float = 0.0,
    subtractive_target_rms_ratio: float = 0.0,
    subtractive_max_gain: float = 1.35,
    hard_mask_threshold: float = 0.0,
    hard_mask_power: float = 3.0,
    hard_mask_residual_floor: float = 0.02,
    dsp_device: str | torch.device | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_vocals = np.asarray(vocals_audio, dtype=np.float32)
    source_vocal_rms = _safe_audio_rms(source_vocals)
    cleaned = source_vocals.copy()
    subtractive_mask_mean = 0.0
    subtractive_restore_mean = 0.0
    subtractive_gain_applied = 1.0
    hard_mask_mean = 0.0
    mixture = np.asarray(mixture_audio, dtype=np.float32)
    if mixture.shape != cleaned.shape:
        raise ValueError(f"Vocal cleanup shape mismatch: vocals={cleaned.shape} mixture={mixture.shape}")

    accompaniment = (mixture - cleaned).astype(np.float32)
    if float(subtractive_strength) > 1e-6 and float(hard_mask_threshold) > 1e-6:
        raise ValueError("Vocal cleanup received both subtractive and hard-mask cleanup settings")
    if float(sharpen_alpha) > 1e-6:
        sharpened = sharpen_stem_bleed(
            {
                "vocals": cleaned,
                "accompaniment": accompaniment,
            },
            alpha=float(sharpen_alpha),
            n_fft=n_fft,
            hop_length=hop_length,
        )
        cleaned = np.asarray(sharpened["vocals"], dtype=np.float32)

    self_activity_gate = _single_stem_activity_gate(
        cleaned,
        floor_ratio=float(tail_gate_floor_ratio),
        smoothing_frames=int(tail_gate_smoothing_frames),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    dominance_gate = _component_transfer_envelope(
        cleaned,
        accompaniment,
        n_fft=n_fft,
        hop_length=hop_length,
        activity_floor_ratio=float(tail_gate_floor_ratio),
        smoothing_frames=int(tail_gate_smoothing_frames),
    )
    activity_gate = np.square(np.clip(self_activity_gate * dominance_gate, 0.0, 1.0)).astype(np.float32)
    if float(tail_gate_strength) > 1e-6:
        gated_mix = ((1.0 - float(tail_gate_strength)) + float(tail_gate_strength) * activity_gate).astype(np.float32)
        cleaned = (cleaned * gated_mix).astype(np.float32)

    if float(subtractive_strength) > 1e-6:
        accompaniment = (mixture - cleaned).astype(np.float32)
        target_device = _resolve_dsp_device(dsp_device)
        if target_device.type == "cuda":
            length = int(cleaned.shape[0])
            vocals_waveform = _torch_audio_channels(cleaned, device=target_device)
            accompaniment_waveform = _torch_audio_channels(accompaniment, device=target_device)
            window = _torch_hann_window(n_fft, device=target_device)
            vocals_stft = torch.stft(
                vocals_waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            )
            accompaniment_stft = torch.stft(
                accompaniment_waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            )
            vocals_mag = vocals_stft.abs()
            accompaniment_mag = accompaniment_stft.abs()
            target_mag = torch.clamp(vocals_mag - float(subtractive_strength) * accompaniment_mag, min=0.0)
            mask = target_mag / (vocals_mag + 1e-8)
            mask = torch.clamp(mask, float(subtractive_residual_floor), 1.0)
            if float(subtractive_power) != 1.0:
                mask = torch.clamp(mask.pow(float(subtractive_power)), float(subtractive_residual_floor), 1.0)
            cleaned_stft = vocals_stft * mask
            cleaned = torch.istft(
                cleaned_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                length=length,
            ).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
            subtractive_mask_mean = float(mask.mean().detach().cpu().item())
            hard_mask_mean = 0.0
        else:
            cleaned_channels: list[np.ndarray] = []
            subtractive_mask_values: list[float] = []
            for channel_index in range(cleaned.shape[1]):
                vocals_stft = librosa.stft(cleaned[:, channel_index], n_fft=n_fft, hop_length=hop_length)
                accompaniment_stft = librosa.stft(accompaniment[:, channel_index], n_fft=n_fft, hop_length=hop_length)
                vocals_mag = np.abs(vocals_stft).astype(np.float32)
                accompaniment_mag = np.abs(accompaniment_stft).astype(np.float32)
                target_mag = np.clip(
                    vocals_mag - float(subtractive_strength) * accompaniment_mag,
                    0.0,
                    None,
                ).astype(np.float32)
                mask = (target_mag / (vocals_mag + 1e-8)).astype(np.float32)
                mask = np.clip(mask, float(subtractive_residual_floor), 1.0).astype(np.float32)
                if float(subtractive_power) != 1.0:
                    mask = np.clip(
                        np.power(mask, float(subtractive_power), dtype=np.float32),
                        float(subtractive_residual_floor),
                        1.0,
                    ).astype(np.float32)
                cleaned_channel = librosa.istft(
                    vocals_stft * mask,
                    hop_length=hop_length,
                    length=cleaned.shape[0],
                ).astype(np.float32)
                cleaned_channels.append(cleaned_channel)
                subtractive_mask_values.append(float(np.mean(mask)) if mask.size else 0.0)
            cleaned = np.stack(cleaned_channels, axis=1).astype(np.float32)
            subtractive_mask_mean = float(np.mean(subtractive_mask_values)) if subtractive_mask_values else 0.0
            hard_mask_mean = 0.0
        restore_mix = np.clip(activity_gate * float(subtractive_active_restore), 0.0, 1.0).astype(np.float32)
        if float(subtractive_active_restore) > 1e-6:
            cleaned = (cleaned * (1.0 - restore_mix) + source_vocals * restore_mix).astype(np.float32)
        subtractive_restore_mean = float(np.mean(restore_mix)) if restore_mix.size else 0.0
        subtractive_gain_applied = 1.0
        if float(subtractive_target_rms_ratio) > 1e-6 and source_vocal_rms > 1e-8:
            cleaned_rms = _safe_audio_rms(cleaned)
            target_rms = source_vocal_rms * float(subtractive_target_rms_ratio)
            if cleaned_rms > 1e-8 and target_rms > cleaned_rms:
                subtractive_gain_applied = min(float(subtractive_max_gain), target_rms / cleaned_rms)
                cleaned = (cleaned * subtractive_gain_applied).astype(np.float32)
    elif float(hard_mask_threshold) > 1e-6:
        accompaniment = (mixture - cleaned).astype(np.float32)
        target_device = _resolve_dsp_device(dsp_device)
        if target_device.type == "cuda":
            length = int(cleaned.shape[0])
            vocals_waveform = _torch_audio_channels(cleaned, device=target_device)
            accompaniment_waveform = _torch_audio_channels(accompaniment, device=target_device)
            window = _torch_hann_window(n_fft, device=target_device)
            vocals_stft = torch.stft(
                vocals_waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            )
            accompaniment_stft = torch.stft(
                accompaniment_waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            )
            vocals_mag = vocals_stft.abs()
            accompaniment_mag = accompaniment_stft.abs()
            dominance = vocals_mag / (vocals_mag + accompaniment_mag + 1e-8)
            mask = torch.clamp(
                (dominance - float(hard_mask_threshold)) / max(1e-6, 1.0 - float(hard_mask_threshold)),
                0.0,
                1.0,
            )
            if float(hard_mask_power) != 1.0:
                mask = mask.pow(float(hard_mask_power))
            final_mask = float(hard_mask_residual_floor) + (1.0 - float(hard_mask_residual_floor)) * mask
            cleaned_stft = vocals_stft * final_mask
            cleaned = torch.istft(
                cleaned_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                length=length,
            ).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
            hard_mask_mean = float(final_mask.mean().detach().cpu().item())
        else:
            cleaned_channels: list[np.ndarray] = []
            hard_mask_values: list[float] = []
            for channel_index in range(cleaned.shape[1]):
                vocals_stft = librosa.stft(cleaned[:, channel_index], n_fft=n_fft, hop_length=hop_length)
                accompaniment_stft = librosa.stft(accompaniment[:, channel_index], n_fft=n_fft, hop_length=hop_length)
                vocals_mag = np.abs(vocals_stft).astype(np.float32)
                accompaniment_mag = np.abs(accompaniment_stft).astype(np.float32)
                dominance = vocals_mag / (vocals_mag + accompaniment_mag + 1e-8)
                mask = np.clip(
                    (dominance - float(hard_mask_threshold)) / max(1e-6, 1.0 - float(hard_mask_threshold)),
                    0.0,
                    1.0,
                ).astype(np.float32)
                if float(hard_mask_power) != 1.0:
                    mask = np.power(mask, float(hard_mask_power), dtype=np.float32)
                final_mask = (float(hard_mask_residual_floor) + (1.0 - float(hard_mask_residual_floor)) * mask).astype(np.float32)
                cleaned_channel = librosa.istft(
                    vocals_stft * final_mask,
                    hop_length=hop_length,
                    length=cleaned.shape[0],
                ).astype(np.float32)
                cleaned_channels.append(cleaned_channel)
                hard_mask_values.append(float(np.mean(final_mask)) if final_mask.size else 0.0)
            cleaned = np.stack(cleaned_channels, axis=1).astype(np.float32)
            hard_mask_mean = float(np.mean(hard_mask_values)) if hard_mask_values else 0.0
        subtractive_mask_mean = 0.0

    cleaned = _limit_peak(cleaned)
    if float(subtractive_strength) > 1e-6:
        if float(subtractive_active_restore) > 1e-6 or float(subtractive_target_rms_ratio) > 1e-6:
            vocal_postprocess = "spectral_subtractive_vocal_denoise_v2"
        else:
            vocal_postprocess = "spectral_subtractive_vocal_denoise_v1"
    elif float(hard_mask_threshold) > 1e-6:
        vocal_postprocess = "hard_subtractive_vocal_mask_v1"
    else:
        vocal_postprocess = "single_stem_cleanup_v1"
    debug = {
        "vocal_postprocess": vocal_postprocess,
        "sharpen_alpha": round(float(sharpen_alpha), 6),
        "tail_gate_strength": round(float(tail_gate_strength), 6),
        "tail_gate_floor_ratio": round(float(tail_gate_floor_ratio), 6),
        "tail_gate_smoothing_frames": int(tail_gate_smoothing_frames),
        "subtractive_strength": round(float(subtractive_strength), 6),
        "subtractive_power": round(float(subtractive_power), 6),
        "subtractive_residual_floor": round(float(subtractive_residual_floor), 6),
        "subtractive_mask_mean": round(float(subtractive_mask_mean), 6),
        "subtractive_active_restore": round(float(subtractive_active_restore), 6),
        "subtractive_restore_mean": round(float(subtractive_restore_mean), 6),
        "subtractive_target_rms_ratio": round(float(subtractive_target_rms_ratio), 6),
        "subtractive_max_gain": round(float(subtractive_max_gain), 6),
        "subtractive_gain_applied": round(float(subtractive_gain_applied), 6),
        "hard_mask_threshold": round(float(hard_mask_threshold), 6),
        "hard_mask_power": round(float(hard_mask_power), 6),
        "hard_mask_residual_floor": round(float(hard_mask_residual_floor), 6),
        "hard_mask_mean": round(float(hard_mask_mean), 6),
        "self_activity_gate_mean": round(float(np.mean(self_activity_gate)) if self_activity_gate.size else 0.0, 6),
        "dominance_gate_mean": round(float(np.mean(dominance_gate)) if dominance_gate.size else 0.0, 6),
        "activity_gate_mean": round(float(np.mean(activity_gate)) if activity_gate.size else 0.0, 6),
        "cleaned_vocals_rms": round(_safe_audio_rms(cleaned), 6),
    }
    return cleaned, debug


def _separate_track_stems_hybrid_vocals_first(
    audio_path: str | Path,
    *,
    device: str | None = None,
    hybrid_stage_models: dict[str, Any] | None = None,
    bs_roformer_runtime: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    stage_models = _normalize_hybrid_stage_models(hybrid_stage_models)
    stage1_device = _effective_requested_device(device, str(stage_models["stage1RequestedDevice"]))
    stage2_device = _effective_requested_device(device, str(stage_models["stage2RequestedDevice"]))
    _emit_progress(
        f"ENTER hybrid_vocals_first stage1_model={stage_models['stage1Model']} stage2_model={stage_models['stage2Model']} audio_path={audio_path}"
    )
    stage1_stems = separate_track_stems_full_res(
        audio_path,
        model_name=str(stage_models["stage1Model"]),
        device=stage1_device,
        demucs_runtime=_extract_hybrid_stage_runtime(stage_models, "stage1"),
    )
    stage1_vocals = np.asarray(stage1_stems.get("vocals"), dtype=np.float32)
    if stage1_vocals.size == 0:
        raise ValueError("Stage1 separator did not return vocals stem")
    bs_vocals_other = _load_bs_roformer_vocals_other(
        audio_path,
        device=device,
        bs_roformer_runtime=bs_roformer_runtime,
    )
    vocals = _blend_vocal_stems(
        stage1_vocals,
        np.asarray(bs_vocals_other["vocals"], dtype=np.float32),
    )

    instrumental = _build_instrumental_mix_from_stage1_non_vocals(stage1_stems)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        _emit_progress(f"hybrid_vocals_first stage1_complete writing_intermediate={temp_path.name}")
        sf.write(temp_path, instrumental, SEPARATION_SAMPLE_RATE, subtype=HYBRID_INTERMEDIATE_SUBTYPE)
        stage2_stems = separate_track_stems_full_res(
            temp_path,
            model_name=str(stage_models["stage2Model"]),
            device=stage2_device,
            demucs_runtime=_extract_hybrid_stage_runtime(stage_models, "stage2"),
        )
    finally:
        temp_path.unlink(missing_ok=True)

    stage2_stems = _rebalance_hybrid_stage2_stems(stage2_stems)
    hybrid_stems = {
        stem_name: np.asarray(stem_audio, dtype=np.float32)
        for stem_name, stem_audio in stage2_stems.items()
        if stem_name != "vocals"
    }
    hybrid_stems["vocals"] = vocals
    ordered_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]
    return {
        stem_name: hybrid_stems[stem_name]
        for stem_name in ordered_names
        if stem_name in hybrid_stems
    }


def _load_bs_roformer_vocals_other(
    audio_path: str | Path,
    *,
    device: str | None = None,
    bs_roformer_runtime: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    runtime = _normalize_bs_roformer_runtime(bs_roformer_runtime)
    repo_path = Path(str(runtime["repoPath"])).resolve()
    config_path = Path(str(runtime["configPath"])).resolve()
    checkpoint_path = Path(str(runtime["checkpointPath"])).resolve()
    if not repo_path.exists() or not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"BS Roformer repo/config/checkpoint are missing: repo={repo_path} config={config_path} checkpoint={checkpoint_path}"
        )

    requested_device = "cpu" if bool(runtime["forceCpu"]) else _effective_requested_device(device, str(runtime["requestedDevice"]))
    resolved_device = str(resolve_torch_device(task_name="hq_stem_export_bs_roformer", requested=requested_device, allow_cpu=True))
    with tempfile.TemporaryDirectory(prefix="bsroformer_input_") as input_dir, tempfile.TemporaryDirectory(prefix="bsroformer_output_") as output_dir:
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        source_path = Path(audio_path)
        staged_input = input_dir_path / source_path.name
        os.symlink(source_path, staged_input)
        temp_override_config: Path | None = None
        config_path_for_run = config_path
        if str(runtime["configOverrideYaml"]).strip():
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
                temp_override_config = Path(temp_file.name)
            temp_override_config.write_text(str(runtime["configOverrideYaml"]), encoding="utf-8")
            config_path_for_run = temp_override_config
        command = [
            "python",
            str(repo_path / "inference.py"),
            "--model_type",
            "bs_roformer",
            "--config_path",
            str(config_path_for_run),
            "--start_check_point",
            str(checkpoint_path),
            "--input_folder",
            str(input_dir_path),
            "--store_dir",
            str(output_dir_path),
            "--flac_file",
            "--pcm_type",
            FINAL_STEM_EXPORT_SUBTYPE,
            "--filename_template",
            str(runtime["filenameTemplate"]),
        ]
        if bool(runtime["extractInstrumental"]):
            command.append("--extract_instrumental")
        if bool(runtime["useTta"]):
            command.append("--use_tta")
        if bool(runtime["disableDetailedPbar"]):
            command.append("--disable_detailed_pbar")
        command.extend(str(arg) for arg in runtime["extraCliArgs"])
        if not bool(runtime["forceCpu"]) and resolved_device == "cuda":
            device_ids = runtime["deviceIds"] if runtime["deviceIds"] else ["0"]
            command.extend(["--device_ids", *[str(device_id) for device_id in device_ids]])
        else:
            command.append("--force_cpu")
        _emit_progress(
            "ENTER bs_roformer_inference "
            f"audio_path={audio_path} repo_path={repo_path} config_path={config_path_for_run} checkpoint_path={checkpoint_path} "
            f"use_tta={bool(runtime['useTta'])} extract_instrumental={bool(runtime['extractInstrumental'])} "
            f"filename_template={runtime['filenameTemplate']} requested_device={runtime['requestedDevice']} "
            f"device_ids={runtime['deviceIds']} force_cpu={bool(runtime['forceCpu'])} "
            f"disable_detailed_pbar={bool(runtime['disableDetailedPbar'])} "
            f"config_override_yaml={'yes' if temp_override_config is not None else 'no'} extra_cli_args={runtime['extraCliArgs']}"
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
        finally:
            if temp_override_config is not None:
                temp_override_config.unlink(missing_ok=True)
        if completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
        def _resolve_output(stem_name: str) -> Path:
            matches = sorted(output_dir_path.glob(f"{stem_name}.*"))
            if not matches:
                raise FileNotFoundError(f"BS Roformer {stem_name} output missing in: {output_dir_path}")
            return matches[0]

        vocals_path = _resolve_output("vocals")
        instrumental_path = _resolve_output("instrumental")
        vocals, vocals_sample_rate = sf.read(str(vocals_path), always_2d=True)
        instrumental, instrumental_sample_rate = sf.read(str(instrumental_path), always_2d=True)
        if vocals_sample_rate != SEPARATION_SAMPLE_RATE or instrumental_sample_rate != SEPARATION_SAMPLE_RATE:
            raise ValueError(
                f"Unexpected BS Roformer sample rates: vocals={vocals_sample_rate} instrumental={instrumental_sample_rate}"
            )
        _emit_progress(
            f"EXIT bs_roformer_inference vocals_path={vocals_path.name} instrumental_path={instrumental_path.name}"
        )
        return {
            "vocals": vocals.astype(np.float32),
            "instrumental": instrumental.astype(np.float32),
        }


def export_vocal_preview_from_audio(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_mode: str = "hybrid_vocals",
    start_sec: float | None = None,
    end_sec: float | None = None,
    device: str | None = None,
    low_bs_weight: float = 0.75,
    high_bs_weight: float = 0.25,
    inactive_low_bs_weight: float = 0.97,
    inactive_high_bs_weight: float = 0.9,
    crossover_hz: float = 3000.0,
    activity_abs_floor_ratio: float = 0.08,
    activity_rel_floor: float = 0.35,
    activity_smoothing_frames: int = 11,
    vocal_sharpen_alpha: float = 0.0,
    tail_gate_strength: float = 0.0,
    tail_gate_floor_ratio: float = 0.08,
    tail_gate_smoothing_frames: int = 11,
    subtractive_strength: float = 0.0,
    subtractive_power: float = 1.0,
    subtractive_residual_floor: float = 0.12,
    subtractive_active_restore: float = 0.0,
    subtractive_target_rms_ratio: float = 0.0,
    subtractive_max_gain: float = 1.35,
    hard_mask_threshold: float = 0.0,
    hard_mask_power: float = 3.0,
    hard_mask_residual_floor: float = 0.02,
    hybrid_stage_models: dict[str, Any] | None = None,
    bs_roformer_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_models = _normalize_hybrid_stage_models(hybrid_stage_models)
    bs_runtime = _normalize_bs_roformer_runtime(bs_roformer_runtime)
    hybrid_device = _effective_requested_device(device, str(stage_models["requestedDevice"]))
    _emit_progress(
        f"ENTER export_vocal_preview_from_audio audio_path={audio_path} output_dir={output_dir} candidate_mode={candidate_mode} start_sec={start_sec} end_sec={end_sec}"
    )
    resolved_audio_path = Path(audio_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    preview_source_path = resolved_audio_path
    preview_sample_rate: int | None = None
    preview_audio: np.ndarray | None = None
    temporary_preview_path: Path | None = None
    if start_sec is not None or end_sec is not None:
        preview_audio, preview_sample_rate = _read_audio_window(
            resolved_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temporary_preview_path = Path(temp_file.name)
        sf.write(
            temporary_preview_path,
            preview_audio,
            int(preview_sample_rate),
            subtype=HYBRID_INTERMEDIATE_SUBTYPE,
        )
        preview_source_path = temporary_preview_path
    _emit_progress(
        f"preview_window_resolved start_sec={start_sec} end_sec={end_sec} preview_source={preview_source_path.name}"
    )

    try:
        vocals_audio: np.ndarray
        pipeline: dict[str, Any]
        if candidate_mode == "ft_vocals":
            _emit_progress("ENTER vocal_candidate ft_vocals")
            stage1_stems = separate_track_stems_full_res(
                preview_source_path,
                model_name=str(stage_models["stage1Model"]),
                device=hybrid_device,
                demucs_runtime=_extract_hybrid_stage_runtime(stage_models, "stage1"),
            )
            vocals_audio = np.asarray(stage1_stems["vocals"], dtype=np.float32)
            pipeline = {
                "mode": "vocal_preview_from_audio",
                "candidate_mode": "ft_vocals",
                "vocals_source": str(stage_models["stage1Model"]),
                "vocal_fusion": None,
                "hybrid_stage_models": stage_models,
            }
            _emit_progress("EXIT vocal_candidate ft_vocals")
        elif candidate_mode == "bs_vocals":
            _emit_progress("ENTER vocal_candidate bs_vocals")
            bs_vocals_other = _load_bs_roformer_vocals_other(
                preview_source_path,
                device=device,
                bs_roformer_runtime=bs_runtime,
            )
            vocals_audio = np.asarray(bs_vocals_other["vocals"], dtype=np.float32)
            pipeline = {
                "mode": "vocal_preview_from_audio",
                "candidate_mode": "bs_vocals",
                "vocals_source": "bs_roformer_vocals_other",
                "vocal_fusion": None,
                "bs_roformer_runtime": bs_runtime,
            }
            _emit_progress("EXIT vocal_candidate bs_vocals")
        elif candidate_mode == "hybrid_vocals":
            _emit_progress("ENTER vocal_candidate hybrid_vocals")
            stage1_stems = separate_track_stems_full_res(
                preview_source_path,
                model_name=str(stage_models["stage1Model"]),
                device=hybrid_device,
                demucs_runtime=_extract_hybrid_stage_runtime(stage_models, "stage1"),
            )
            stage1_vocals = np.asarray(stage1_stems["vocals"], dtype=np.float32)
            bs_vocals_other = _load_bs_roformer_vocals_other(
                preview_source_path,
                device=device,
                bs_roformer_runtime=bs_runtime,
            )
            bs_vocals = np.asarray(bs_vocals_other["vocals"], dtype=np.float32)
            vocals_audio = _blend_vocal_stems(
                stage1_vocals,
                bs_vocals,
                low_bs_weight=float(low_bs_weight),
                high_bs_weight=float(high_bs_weight),
                inactive_low_bs_weight=float(inactive_low_bs_weight),
                inactive_high_bs_weight=float(inactive_high_bs_weight),
                crossover_hz=float(crossover_hz),
                activity_abs_floor_ratio=float(activity_abs_floor_ratio),
                activity_rel_floor=float(activity_rel_floor),
                activity_smoothing_frames=int(activity_smoothing_frames),
            )
            pipeline = {
                "mode": "vocal_preview_from_audio",
                "candidate_mode": "hybrid_vocals",
                "vocals_source": f"{stage_models['stage1Model']}+bs_roformer_vocals_other",
                "vocal_fusion": HYBRID_VOCAL_FUSION_LABEL,
                "hybrid_stage_models": stage_models,
                "bs_roformer_runtime": bs_runtime,
            }
            _emit_progress("EXIT vocal_candidate hybrid_vocals")
        else:
            raise ValueError(f"Unsupported vocal candidate_mode: {candidate_mode}")

        if (
            float(vocal_sharpen_alpha) > 1e-6
            or float(tail_gate_strength) > 1e-6
            or float(subtractive_strength) > 1e-6
            or float(hard_mask_threshold) > 1e-6
        ):
            mixture_audio, mixture_sample_rate = _read_audio_window(preview_source_path)
            mixture_audio = _resample_audio_to_sample_rate(
                mixture_audio,
                mixture_sample_rate,
                SEPARATION_SAMPLE_RATE,
            )
            frame_count = min(vocals_audio.shape[0], mixture_audio.shape[0])
            vocals_audio = vocals_audio[:frame_count]
            mixture_audio = mixture_audio[:frame_count]
            vocals_audio, cleanup_debug = _cleanup_single_vocal_stem(
                vocals_audio,
                mixture_audio,
                sharpen_alpha=float(vocal_sharpen_alpha),
                tail_gate_strength=float(tail_gate_strength),
                tail_gate_floor_ratio=float(tail_gate_floor_ratio),
                tail_gate_smoothing_frames=int(tail_gate_smoothing_frames),
                subtractive_strength=float(subtractive_strength),
                subtractive_power=float(subtractive_power),
                subtractive_residual_floor=float(subtractive_residual_floor),
                subtractive_active_restore=float(subtractive_active_restore),
                subtractive_target_rms_ratio=float(subtractive_target_rms_ratio),
                subtractive_max_gain=float(subtractive_max_gain),
                hard_mask_threshold=float(hard_mask_threshold),
                hard_mask_power=float(hard_mask_power),
                hard_mask_residual_floor=float(hard_mask_residual_floor),
                dsp_device=str(device or "auto"),
            )
            pipeline["vocal_postprocess"] = cleanup_debug["vocal_postprocess"]
        else:
            cleanup_debug = {
                "vocal_postprocess": None,
                "sharpen_alpha": round(float(vocal_sharpen_alpha), 6),
                "tail_gate_strength": round(float(tail_gate_strength), 6),
                "tail_gate_floor_ratio": round(float(tail_gate_floor_ratio), 6),
                "tail_gate_smoothing_frames": int(tail_gate_smoothing_frames),
                "subtractive_strength": round(float(subtractive_strength), 6),
                "subtractive_power": round(float(subtractive_power), 6),
                "subtractive_residual_floor": round(float(subtractive_residual_floor), 6),
                "subtractive_active_restore": round(float(subtractive_active_restore), 6),
                "subtractive_target_rms_ratio": round(float(subtractive_target_rms_ratio), 6),
                "subtractive_max_gain": round(float(subtractive_max_gain), 6),
                "hard_mask_threshold": round(float(hard_mask_threshold), 6),
                "hard_mask_power": round(float(hard_mask_power), 6),
                "hard_mask_residual_floor": round(float(hard_mask_residual_floor), 6),
            }

        vocals_path = resolved_output_dir / "vocals.flac"
        _emit_progress(f"writing_vocal_preview path={vocals_path.name}")
        sf.write(vocals_path, vocals_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
        manifest: dict[str, Any] = {
            "audio_path": str(resolved_audio_path),
            "source_audio_sha256": _sha256_file(resolved_audio_path),
            "sample_rate": SEPARATION_SAMPLE_RATE,
            "export_subtype": FINAL_STEM_EXPORT_SUBTYPE,
            "preview_window_sec": {
                "start": None if start_sec is None else float(start_sec),
                "end": None if end_sec is None else float(end_sec),
            },
            "pipeline": pipeline,
            "tuning": {
                "low_bs_weight": float(low_bs_weight),
                "high_bs_weight": float(high_bs_weight),
                "inactive_low_bs_weight": float(inactive_low_bs_weight),
                "inactive_high_bs_weight": float(inactive_high_bs_weight),
                "crossover_hz": float(crossover_hz),
                "activity_abs_floor_ratio": float(activity_abs_floor_ratio),
                "activity_rel_floor": float(activity_rel_floor),
                "activity_smoothing_frames": int(activity_smoothing_frames),
                "vocal_sharpen_alpha": float(vocal_sharpen_alpha),
                "tail_gate_strength": float(tail_gate_strength),
                "tail_gate_floor_ratio": float(tail_gate_floor_ratio),
                "tail_gate_smoothing_frames": int(tail_gate_smoothing_frames),
                "subtractive_strength": float(subtractive_strength),
                "subtractive_power": float(subtractive_power),
                "subtractive_residual_floor": float(subtractive_residual_floor),
                "subtractive_active_restore": float(subtractive_active_restore),
                "subtractive_target_rms_ratio": float(subtractive_target_rms_ratio),
                "subtractive_max_gain": float(subtractive_max_gain),
                "hard_mask_threshold": float(hard_mask_threshold),
                "hard_mask_power": float(hard_mask_power),
                "hard_mask_residual_floor": float(hard_mask_residual_floor),
                "device": str(device or "auto"),
            },
            "stems": {
                "vocals": {
                    "path": str(vocals_path),
                    **_audio_summary(vocals_audio),
                    **cleanup_debug,
                }
            },
        }
        manifest_path = resolved_output_dir / "stem_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        _emit_progress(f"EXIT export_vocal_preview_from_audio manifest_path={manifest_path}")
        return manifest
    finally:
        if temporary_preview_path is not None:
            temporary_preview_path.unlink(missing_ok=True)


def _separate_track_stems_hybrid_bs_vocals_first(
    audio_path: str | Path,
    *,
    device: str | None = None,
    hybrid_stage_models: dict[str, Any] | None = None,
    bs_roformer_runtime: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    stage_models = _normalize_hybrid_stage_models(hybrid_stage_models)
    stage1_device = _effective_requested_device(device, str(stage_models["stage1RequestedDevice"]))
    stage2_device = _effective_requested_device(device, str(stage_models["stage2RequestedDevice"]))
    _emit_progress(
        f"ENTER hybrid_bs_vocals_first stage1_model={stage_models['stage1Model']} stage0_model=bs_roformer_vocals_other stage2_model={stage_models['stage2Model']} audio_path={audio_path}"
    )
    stage1_stems = separate_track_stems_full_res(
        audio_path,
        model_name=str(stage_models["stage1Model"]),
        device=stage1_device,
        demucs_runtime=_extract_hybrid_stage_runtime(stage_models, "stage1"),
    )
    stage1_vocals = np.asarray(stage1_stems.get("vocals"), dtype=np.float32)
    if stage1_vocals.size == 0:
        raise ValueError("Stage1 separator did not return vocals stem")

    bs_vocals_other = _load_bs_roformer_vocals_other(
        audio_path,
        device=device,
        bs_roformer_runtime=bs_roformer_runtime,
    )
    vocals = _blend_vocal_stems(
        stage1_vocals,
        np.asarray(bs_vocals_other["vocals"], dtype=np.float32),
    )
    instrumental = np.asarray(bs_vocals_other["instrumental"], dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        _emit_progress(f"hybrid_bs_vocals_first stage0_complete writing_intermediate={temp_path.name}")
        sf.write(temp_path, instrumental, SEPARATION_SAMPLE_RATE, subtype=HYBRID_INTERMEDIATE_SUBTYPE)
        stage2_stems = separate_track_stems_full_res(
            temp_path,
            model_name=str(stage_models["stage2Model"]),
            device=stage2_device,
            demucs_runtime=_extract_hybrid_stage_runtime(stage_models, "stage2"),
        )
    finally:
        temp_path.unlink(missing_ok=True)

    hybrid_stems = {
        stem_name: np.asarray(stem_audio, dtype=np.float32)
        for stem_name, stem_audio in stage2_stems.items()
        if stem_name != "vocals"
    }
    hybrid_stems["vocals"] = vocals
    ordered_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]
    return {
        stem_name: hybrid_stems[stem_name]
        for stem_name in ordered_names
        if stem_name in hybrid_stems
    }


def _audio_summary(audio: np.ndarray) -> dict[str, Any]:
    if audio.size == 0:
        return {
            "channels": 0,
            "frames": 0,
            "duration_sec": 0.0,
            "rms": 0.0,
            "peak": 0.0,
        }
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    rms = math.sqrt(float(np.mean(np.square(mono, dtype=np.float32))))
    peak = float(np.max(np.abs(mono)))
    return {
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "frames": int(audio.shape[0]),
        "duration_sec": round(float(audio.shape[0] / SEPARATION_SAMPLE_RATE), 3),
        "rms": round(float(rms), 6),
        "peak": round(float(peak), 6),
    }


def sharpen_stem_bleed(
    stems: dict[str, np.ndarray],
    *,
    alpha: float = 2.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, np.ndarray]:
    if not stems:
        return {}

    ordered_names = list(stems.keys())
    lengths = [audio.shape[0] for audio in stems.values()]
    channel_counts = [audio.shape[1] if audio.ndim == 2 else 1 for audio in stems.values()]
    if len(set(lengths)) != 1 or len(set(channel_counts)) != 1:
        raise ValueError("All stems must share the same shape for bleed sharpening")

    frame_count = lengths[0]
    channel_count = channel_counts[0]
    mixture = np.zeros((frame_count, channel_count), dtype=np.float32)
    for audio in stems.values():
        mixture += np.asarray(audio, dtype=np.float32)

    sharpened: dict[str, np.ndarray] = {
        stem_name: np.zeros((frame_count, channel_count), dtype=np.float32)
        for stem_name in ordered_names
    }

    for channel_index in range(channel_count):
        mixture_channel = mixture[:, channel_index]
        mixture_stft = librosa.stft(mixture_channel, n_fft=n_fft, hop_length=hop_length)
        magnitude_stack = []
        for stem_name in ordered_names:
            stem_channel = np.asarray(stems[stem_name][:, channel_index], dtype=np.float32)
            stem_stft = librosa.stft(stem_channel, n_fft=n_fft, hop_length=hop_length)
            magnitude_stack.append(np.abs(stem_stft).astype(np.float32))

        magnitudes = np.stack(magnitude_stack, axis=0)
        powered = np.power(np.maximum(magnitudes, 1e-8), float(alpha), dtype=np.float32)
        denominator = np.sum(powered, axis=0, keepdims=True) + 1e-8
        masks = powered / denominator

        for stem_index, stem_name in enumerate(ordered_names):
            masked_stft = mixture_stft * masks[stem_index]
            sharpened_channel = librosa.istft(masked_stft, hop_length=hop_length, length=frame_count)
            sharpened[stem_name][:, channel_index] = sharpened_channel.astype(np.float32)

    return sharpened


def _build_melodic_bus(stems: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    melodic_sources = ["other", "vocals", "guitar", "piano"]
    available = [stem for stem in melodic_sources if stem in stems]
    if not available:
        raise ValueError("No melodic stems available to build melody bus")

    mono_sources: dict[str, np.ndarray] = {}
    rms_by_stem: dict[str, float] = {}
    for stem_name in available:
        audio = np.asarray(stems[stem_name], dtype=np.float32)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        mono = mono.astype(np.float32, copy=False)
        rms = math.sqrt(float(np.mean(np.square(mono, dtype=np.float32)))) if mono.size else 0.0
        mono_sources[stem_name] = mono
        rms_by_stem[stem_name] = rms

    non_zero_rms = [value for value in rms_by_stem.values() if value > 1e-6]
    target_rms = float(np.median(non_zero_rms)) if non_zero_rms else 1.0
    weights: dict[str, float] = {}
    weighted = np.zeros_like(next(iter(mono_sources.values())), dtype=np.float32)
    for stem_name, mono in mono_sources.items():
        rms = rms_by_stem[stem_name]
        base_weight = 1.0
        if stem_name == "vocals":
            base_weight = 0.9
        elif stem_name in {"guitar", "piano"}:
            base_weight = 1.2
        scale = 1.0 if rms <= 1e-6 else float(np.clip(target_rms / rms, 0.4, 2.5))
        weight = base_weight * scale
        weighted += mono * weight
        weights[stem_name] = round(weight, 4)
    return weighted, weights


def _extract_predominant_pitch(audio: np.ndarray, *, sample_rate: int) -> dict[str, np.ndarray]:
    if audio.size < sample_rate:
        return {
            "times": np.zeros(0, dtype=np.float32),
            "frequencies": np.zeros(0, dtype=np.float32),
            "confidence": np.zeros(0, dtype=np.float32),
        }

    pyin_frequencies, _, pyin_confidence = librosa.pyin(
        np.asarray(audio, dtype=np.float32),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=2048,
        hop_length=512,
    )
    pyin_frequencies = np.nan_to_num(pyin_frequencies, nan=0.0).astype(np.float32)
    pyin_confidence = np.nan_to_num(pyin_confidence, nan=0.0).astype(np.float32)
    if int(np.count_nonzero(pyin_confidence >= 0.1)) >= 32:
        times = np.arange(len(pyin_frequencies), dtype=np.float32) * 512.0 / float(sample_rate)
        return {
            "times": times.astype(np.float32),
            "frequencies": pyin_frequencies,
            "confidence": pyin_confidence,
        }

    extractor = essentia_std.PredominantPitchMelodia(
        frameSize=2048,
        hopSize=128,
        sampleRate=sample_rate,
        minFrequency=80,
        maxFrequency=2200,
    )
    pitch, confidence = extractor(np.asarray(audio, dtype=np.float32))
    times = np.arange(len(pitch), dtype=np.float32) * 128.0 / float(sample_rate)
    return {
        "times": times.astype(np.float32),
        "frequencies": np.asarray(pitch, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
    }


def _interpolate_pitch_track(
    pitch_track: dict[str, np.ndarray],
    *,
    target_frame_count: int,
    hop_length: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target_frame_count <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if pitch_track["times"].size == 0:
        return np.zeros(target_frame_count, dtype=np.float32), np.zeros(target_frame_count, dtype=np.float32)

    target_times = (np.arange(target_frame_count, dtype=np.float32) * float(hop_length) / float(sample_rate)).astype(np.float32)
    source_times = pitch_track["times"].astype(np.float32)
    frequencies = np.interp(target_times, source_times, pitch_track["frequencies"], left=0.0, right=0.0).astype(np.float32)
    confidence = np.interp(target_times, source_times, pitch_track["confidence"], left=0.0, right=0.0).astype(np.float32)
    return frequencies, confidence


def _build_harmonic_mask(
    frequencies: np.ndarray,
    confidence: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    frame_count: int,
    harmonic_count: int = 5,
    confidence_floor: float = 0.1,
) -> np.ndarray:
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).astype(np.float32)
    mask = np.zeros((freq_bins.size, frame_count), dtype=np.float32)
    if frame_count == 0:
        return mask

    for frame_index in range(frame_count):
        f0 = float(frequencies[frame_index]) if frame_index < len(frequencies) else 0.0
        conf = float(confidence[frame_index]) if frame_index < len(confidence) else 0.0
        if f0 <= 0.0 or conf < confidence_floor:
            continue
        frame_mask = np.zeros(freq_bins.size, dtype=np.float32)
        for harmonic in range(1, harmonic_count + 1):
            harmonic_frequency = f0 * harmonic
            if harmonic_frequency >= (sample_rate / 2.0):
                break
            sigma_hz = max(20.0, harmonic_frequency * 0.025)
            distances = (freq_bins - harmonic_frequency) / sigma_hz
            weight = math.exp(-0.22 * (harmonic - 1))
            frame_mask = np.maximum(frame_mask, weight * np.exp(-0.5 * np.square(distances, dtype=np.float32)))
        frame_mask *= conf
        mask[:, frame_index] = np.clip(frame_mask, 0.0, 1.0)
    return mask


def _apply_time_frequency_mask(
    audio: np.ndarray,
    mask: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    dsp_device: str | torch.device | None = None,
) -> np.ndarray:
    target_device = _resolve_dsp_device(dsp_device)
    if target_device.type == "cuda":
        waveform = _torch_audio_channels(audio, device=target_device)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device=target_device, dtype=torch.float32)
        window = _torch_hann_window(n_fft, device=target_device)
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        masked_stft = stft * mask_tensor[:, : stft.shape[2]].unsqueeze(0)
        masked_audio = torch.istft(
            masked_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=audio.shape[0],
        ).transpose(0, 1)
        return masked_audio.detach().cpu().numpy().astype(np.float32)
    masked_audio = np.zeros_like(audio, dtype=np.float32)
    if audio.size == 0:
        return masked_audio

    for channel_index in range(audio.shape[1]):
        stft = librosa.stft(audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        masked_stft = stft * mask[:, : stft.shape[1]]
        masked_audio[:, channel_index] = librosa.istft(
            masked_stft,
            hop_length=hop_length,
            length=audio.shape[0],
        ).astype(np.float32)
    return masked_audio


def _extract_melody_components_from_stems(
    stems: dict[str, np.ndarray],
    *,
    melodic_sources: tuple[str, ...],
    analysis_sources: tuple[str, ...] | None = None,
    sample_rate: int = SEPARATION_SAMPLE_RATE,
    n_fft: int = 2048,
    hop_length: int = 512,
    harmonic_count: int = 5,
    confidence_floor: float = 0.1,
    dsp_device: str | torch.device | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    resolved_analysis_sources = analysis_sources or melodic_sources
    melodic_bus_mono, bus_weights = _build_melodic_bus(
        {stem_name: stems[stem_name] for stem_name in resolved_analysis_sources if stem_name in stems}
    )
    pitch_track = _extract_predominant_pitch(melodic_bus_mono, sample_rate=sample_rate)

    reference_stem = next(iter(stems.values()))
    frame_count = int(math.floor(reference_stem.shape[0] / hop_length)) + 1
    frequencies, confidence = _interpolate_pitch_track(
        pitch_track,
        target_frame_count=frame_count,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
    harmonic_mask = _build_harmonic_mask(
        frequencies,
        confidence,
        sample_rate=sample_rate,
        n_fft=n_fft,
        frame_count=frame_count,
        harmonic_count=harmonic_count,
        confidence_floor=confidence_floor,
    )

    source_components: dict[str, np.ndarray] = {}
    melody_stereo = np.zeros_like(reference_stem, dtype=np.float32)
    for stem_name in melodic_sources:
        if stem_name not in stems:
            continue
        source_audio = np.asarray(stems[stem_name], dtype=np.float32)
        source_melody = _apply_time_frequency_mask(
            source_audio,
            harmonic_mask,
            n_fft=n_fft,
            hop_length=hop_length,
            dsp_device=dsp_device,
        )
        source_components[stem_name] = source_melody.astype(np.float32)
        melody_stereo += source_components[stem_name]

    voiced_ratio = float(np.mean(confidence >= 0.1)) if confidence.size else 0.0
    debug = {
        "melody_bus_weights": bus_weights,
        "melody_sources": list(melodic_sources),
        "melody_analysis_sources": list(resolved_analysis_sources),
        "voiced_ratio": round(voiced_ratio, 4),
        "mean_confidence": round(float(np.mean(confidence)) if confidence.size else 0.0, 4),
        "median_frequency_hz": round(float(np.median(frequencies[frequencies > 0])) if np.any(frequencies > 0) else 0.0, 3),
        "active_frame_count": int(np.count_nonzero(confidence >= 0.1)),
    }
    return melody_stereo.astype(np.float32), source_components, debug, harmonic_mask


def extract_melody_stem_from_stems(
    stems: dict[str, np.ndarray],
    *,
    sample_rate: int = SEPARATION_SAMPLE_RATE,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    melody_stereo, _, debug, _ = _extract_melody_components_from_stems(
        stems,
        melodic_sources=("other", "vocals", "guitar", "piano"),
        analysis_sources=("other", "vocals", "guitar", "piano"),
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return melody_stereo, debug


def _redistribute_instrumental_melody_from_stems(
    stems: dict[str, np.ndarray],
    *,
    melody_sources: tuple[str, ...] = DEFAULT_MELODY_REBALANCE_SOURCES,
    other_melody_move: float = 0.78,
    guitar_melody_move: float = 0.55,
    piano_melody_move: float = 0.55,
    drum_melody_move: float = 0.85,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    if not stems:
        raise ValueError("No stems available for melody redistribution")

    rebalanced = {
        stem_name: np.asarray(stem_audio, dtype=np.float32).copy()
        for stem_name, stem_audio in stems.items()
    }
    _, _, debug, harmonic_mask = _extract_melody_components_from_stems(
        rebalanced,
        melodic_sources=tuple(stem_name for stem_name in melody_sources if stem_name in rebalanced),
        analysis_sources=tuple(
            stem_name
            for stem_name in ("other", "vocals", "guitar", "piano")
            if stem_name in rebalanced
        ),
        sample_rate=SEPARATION_SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    moved_components: dict[str, float] = {}
    reference_audio = next(iter(rebalanced.values()))
    melody_accumulator = np.zeros_like(reference_audio, dtype=np.float32)
    frame_activity = np.clip(np.max(harmonic_mask, axis=0) * 1.35, 0.0, 1.0).astype(np.float32) if harmonic_mask.size else np.zeros(0, dtype=np.float32)
    if frame_activity.size:
        frame_times = (np.arange(frame_activity.size, dtype=np.float32) * float(hop_length)).astype(np.float32)
        sample_positions = np.arange(reference_audio.shape[0], dtype=np.float32)
        sample_activity = np.interp(
            sample_positions,
            frame_times,
            frame_activity,
            left=float(frame_activity[0]),
            right=float(frame_activity[-1]),
        ).astype(np.float32)
    else:
        sample_activity = np.zeros(reference_audio.shape[0], dtype=np.float32)
    activity_envelope = np.repeat(sample_activity[:, np.newaxis], reference_audio.shape[1], axis=1).astype(np.float32)
    source_move_map = {
        "other": float(other_melody_move),
        "guitar": float(guitar_melody_move),
        "piano": float(piano_melody_move),
    }
    for stem_name, move_ratio in source_move_map.items():
        if stem_name not in rebalanced:
            continue
        source_harmonic, _ = _split_harmonic_percussive(
            rebalanced[stem_name],
            harmonic_margin=2.0,
            percussive_margin=1.0,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        moved_component = source_harmonic * move_ratio * activity_envelope
        rebalanced[stem_name] = (rebalanced[stem_name] - moved_component).astype(np.float32)
        melody_accumulator += moved_component
        moved_components[f"{stem_name}_moved_rms"] = round(_safe_audio_rms(moved_component), 6)

    if "drums" in rebalanced:
        drums_harmonic, _ = _split_harmonic_percussive(
            rebalanced["drums"],
            harmonic_margin=1.0,
            percussive_margin=4.0,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        moved_drum_melody = drums_harmonic * float(drum_melody_move) * activity_envelope
        rebalanced["drums"] = (rebalanced["drums"] - moved_drum_melody).astype(np.float32)
        melody_accumulator += moved_drum_melody
        moved_components["drums_moved_rms"] = round(_safe_audio_rms(moved_drum_melody), 6)

    debug.update(moved_components)
    debug["melody_strategy"] = "instrumental_mask_reassign_v1"
    return rebalanced, melody_accumulator.astype(np.float32), debug


def _limit_peak(audio: np.ndarray, *, peak_limit: float = 0.999) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= float(peak_limit) or peak <= 1e-8:
        return audio.astype(np.float32, copy=False)
    return (audio * (float(peak_limit) / peak)).astype(np.float32)


def _aggressive_melody_first_rebalance(
    stems: dict[str, np.ndarray],
    *,
    other_harmonic_move: float = 0.94,
    guitar_harmonic_move: float = 0.96,
    piano_harmonic_move: float = 0.96,
    drum_harmonic_move: float = 0.9,
    other_percussive_to_drums: float = 0.82,
    guitar_percussive_to_drums: float = 0.75,
    piano_percussive_to_drums: float = 0.75,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    if not stems:
        raise ValueError("No stems available for aggressive melody-first rebalance")

    rebalanced = {
        stem_name: np.asarray(stem_audio, dtype=np.float32).copy()
        for stem_name, stem_audio in stems.items()
    }
    reference_audio = next(iter(rebalanced.values()))
    zero_audio = np.zeros_like(reference_audio, dtype=np.float32)

    drums = rebalanced.get("drums", zero_audio)
    other = rebalanced.get("other", zero_audio)
    guitar = rebalanced.get("guitar", zero_audio)
    piano = rebalanced.get("piano", zero_audio)

    drums_harmonic, drums_percussive = _split_harmonic_percussive(
        drums,
        harmonic_margin=1.0,
        percussive_margin=4.0,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    other_harmonic, other_percussive = _split_harmonic_percussive(
        other,
        harmonic_margin=2.2,
        percussive_margin=1.0,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    guitar_harmonic, guitar_percussive = _split_harmonic_percussive(
        guitar,
        harmonic_margin=2.2,
        percussive_margin=1.0,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    piano_harmonic, piano_percussive = _split_harmonic_percussive(
        piano,
        harmonic_margin=2.2,
        percussive_margin=1.0,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    harmonic_bus = (
        other_harmonic
        + guitar_harmonic
        + piano_harmonic
        + drums_harmonic * 0.4
    ).astype(np.float32)
    percussive_bed = (
        drums_percussive
        + other_percussive * 0.5
        + guitar_percussive * 0.2
        + piano_percussive * 0.2
    ).astype(np.float32)
    activity_envelope = _component_transfer_envelope(
        harmonic_bus,
        percussive_bed,
        n_fft=n_fft,
        hop_length=hop_length,
        activity_floor_ratio=0.02,
        smoothing_frames=17,
    )
    activity_envelope = np.clip(activity_envelope * 1.45 + 0.18, 0.0, 1.0).astype(np.float32)

    moved_other_harmonic = other_harmonic * float(other_harmonic_move) * activity_envelope
    moved_guitar_harmonic = guitar_harmonic * float(guitar_harmonic_move) * activity_envelope
    moved_piano_harmonic = piano_harmonic * float(piano_harmonic_move) * activity_envelope
    moved_drums_harmonic = drums_harmonic * float(drum_harmonic_move) * activity_envelope

    moved_other_percussive = other_percussive * float(other_percussive_to_drums)
    moved_guitar_percussive = guitar_percussive * float(guitar_percussive_to_drums)
    moved_piano_percussive = piano_percussive * float(piano_percussive_to_drums)

    melody_audio = (
        moved_other_harmonic
        + moved_guitar_harmonic
        + moved_piano_harmonic
        + moved_drums_harmonic
    ).astype(np.float32)
    drums_audio = (
        drums
        - moved_drums_harmonic
        + moved_other_percussive
        + moved_guitar_percussive
        + moved_piano_percussive
    ).astype(np.float32)
    other_audio = (
        other
        - moved_other_harmonic
        - moved_other_percussive
    ).astype(np.float32)
    guitar_audio = (
        guitar
        - moved_guitar_harmonic
        - moved_guitar_percussive
    ).astype(np.float32)
    piano_audio = (
        piano
        - moved_piano_harmonic
        - moved_piano_percussive
    ).astype(np.float32)

    if "drums" in rebalanced:
        rebalanced["drums"] = _limit_peak(drums_audio)
    if "other" in rebalanced:
        rebalanced["other"] = _limit_peak(other_audio)
    if "guitar" in rebalanced:
        rebalanced["guitar"] = _limit_peak(guitar_audio)
    if "piano" in rebalanced:
        rebalanced["piano"] = _limit_peak(piano_audio)

    melody_audio = _limit_peak(melody_audio)
    debug = {
        "melody_strategy": "aggressive_harmonic_bus_reassign_v1",
        "activity_envelope_mean": round(float(np.mean(activity_envelope)) if activity_envelope.size else 0.0, 6),
        "other_harmonic_moved_rms": round(_safe_audio_rms(moved_other_harmonic), 6),
        "guitar_harmonic_moved_rms": round(_safe_audio_rms(moved_guitar_harmonic), 6),
        "piano_harmonic_moved_rms": round(_safe_audio_rms(moved_piano_harmonic), 6),
        "drums_harmonic_moved_rms": round(_safe_audio_rms(moved_drums_harmonic), 6),
        "other_percussive_to_drums_rms": round(_safe_audio_rms(moved_other_percussive), 6),
        "guitar_percussive_to_drums_rms": round(_safe_audio_rms(moved_guitar_percussive), 6),
        "piano_percussive_to_drums_rms": round(_safe_audio_rms(moved_piano_percussive), 6),
        "melody_rms": round(_safe_audio_rms(melody_audio), 6),
    }
    return rebalanced, melody_audio, debug


def _selective_melody_first_rebalance_from_stems(
    stems: dict[str, np.ndarray],
    *,
    other_move: float = 0.35,
    guitar_move: float = 1.0,
    piano_move: float = 1.0,
    drums_masked_move: float = 0.08,
    target_melody_rms: float = 0.024,
    max_melody_gain: float = 1.4,
    stage_drum_harmonic_move: float = 0.35,
    stage_harmonic_stem_percussive_move: float = 0.18,
    stage_cleanup_drum_harmonic_move: float = 0.03,
    stage_cleanup_harmonic_stem_percussive_move: float = 0.25,
    harmonic_count: int = 2,
    confidence_floor: float = 0.35,
    dsp_device: str | torch.device | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    if not stems:
        raise ValueError("No stems available for selective melody-first rebalance")

    rebalanced = {
        stem_name: np.asarray(stem_audio, dtype=np.float32).copy()
        for stem_name, stem_audio in stems.items()
    }
    if "drums" in rebalanced and "other" in rebalanced:
        rebalanced = _rebalance_hybrid_stage2_stems(
            rebalanced,
            drum_harmonic_move=float(stage_drum_harmonic_move),
            harmonic_stem_percussive_move=float(stage_harmonic_stem_percussive_move),
            cleanup_drum_harmonic_move=float(stage_cleanup_drum_harmonic_move),
            cleanup_harmonic_stem_percussive_move=float(stage_cleanup_harmonic_stem_percussive_move),
            dsp_device=dsp_device,
        )

    melody_audio, source_components, debug, harmonic_mask = _extract_melody_components_from_stems(
        rebalanced,
        melodic_sources=tuple(stem_name for stem_name in ("other", "guitar", "piano") if stem_name in rebalanced),
        analysis_sources=tuple(stem_name for stem_name in ("other", "guitar", "piano") if stem_name in rebalanced),
        sample_rate=SEPARATION_SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        harmonic_count=int(harmonic_count),
        confidence_floor=float(confidence_floor),
        dsp_device=dsp_device,
    )

    moved_components: dict[str, float] = {}
    source_move_map = {
        "other": float(other_move),
        "guitar": float(guitar_move),
        "piano": float(piano_move),
    }
    melody_accumulator = np.zeros_like(next(iter(rebalanced.values())), dtype=np.float32)
    for stem_name, component in source_components.items():
        moved_component = component * source_move_map.get(stem_name, 1.0)
        rebalanced[stem_name] = (rebalanced[stem_name] - moved_component).astype(np.float32)
        melody_accumulator += moved_component.astype(np.float32)
        moved_components[f"{stem_name}_masked_moved_rms"] = round(_safe_audio_rms(moved_component), 6)

    if "drums" in rebalanced:
        drums_harmonic, _ = _split_harmonic_percussive(
            rebalanced["drums"],
            harmonic_margin=1.0,
            percussive_margin=4.0,
            n_fft=n_fft,
            hop_length=hop_length,
            dsp_device=dsp_device,
        )
        drums_masked = _apply_time_frequency_mask(
            drums_harmonic,
            harmonic_mask,
            n_fft=n_fft,
            hop_length=hop_length,
            dsp_device=dsp_device,
        )
        moved_drums = drums_masked * float(drums_masked_move)
        rebalanced["drums"] = (rebalanced["drums"] - moved_drums).astype(np.float32)
        melody_accumulator += moved_drums.astype(np.float32)
        moved_components["drums_masked_moved_rms"] = round(_safe_audio_rms(moved_drums), 6)

    melody_rms = _safe_audio_rms(melody_accumulator)
    melody_gain = 1.0
    if melody_rms > 1e-8:
        melody_gain = float(np.clip(float(target_melody_rms) / melody_rms, 1.0, float(max_melody_gain)))
    melody_output = _limit_peak(melody_accumulator * melody_gain)

    debug.update(moved_components)
    debug["melody_strategy"] = "selective_pitch_mask_reassign_v3"
    debug["harmonic_count"] = int(harmonic_count)
    debug["confidence_floor"] = float(confidence_floor)
    debug["melody_output_gain"] = round(melody_gain, 6)
    debug["melody_rms"] = round(_safe_audio_rms(melody_output), 6)
    debug["mask_max_mean"] = round(float(np.mean(np.max(harmonic_mask, axis=0))) if harmonic_mask.size else 0.0, 6)
    debug["other_move"] = round(float(other_move), 6)
    debug["guitar_move"] = round(float(guitar_move), 6)
    debug["piano_move"] = round(float(piano_move), 6)
    debug["drums_masked_move"] = round(float(drums_masked_move), 6)
    debug["target_melody_rms"] = round(float(target_melody_rms), 6)
    debug["max_melody_gain"] = round(float(max_melody_gain), 6)
    debug["stage_drum_harmonic_move"] = round(float(stage_drum_harmonic_move), 6)
    debug["stage_harmonic_stem_percussive_move"] = round(float(stage_harmonic_stem_percussive_move), 6)
    debug["stage_cleanup_drum_harmonic_move"] = round(float(stage_cleanup_drum_harmonic_move), 6)
    debug["stage_cleanup_harmonic_stem_percussive_move"] = round(float(stage_cleanup_harmonic_stem_percussive_move), 6)
    debug["dsp_device"] = str(_resolve_dsp_device(dsp_device))
    return rebalanced, melody_output.astype(np.float32), debug


def export_track_stems_full_res(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = DEFAULT_STEM_EXPORT_MODEL,
    device: str | None = None,
    sharpen_bleed: bool = DEFAULT_STEM_EXPORT_SHARPEN_BLEED,
    include_melody_stem: bool = DEFAULT_STEM_EXPORT_INCLUDE_MELODY_STEM,
    hybrid_stage_models: dict[str, Any] | None = None,
    bs_roformer_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_models = _normalize_hybrid_stage_models(hybrid_stage_models)
    bs_runtime = _normalize_bs_roformer_runtime(bs_roformer_runtime)
    _emit_progress(
        f"ENTER export_track_stems_full_res model={model_name} output_dir={output_dir} include_melody_stem={include_melody_stem}"
    )
    resolved_audio_path = Path(audio_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    raw_stems = separate_track_stems_full_res(
        resolved_audio_path,
        model_name=model_name,
        device=device,
        hybrid_stage_models=stage_models,
        bs_roformer_runtime=bs_runtime,
    )
    stems = sharpen_stem_bleed(raw_stems) if sharpen_bleed else raw_stems
    melody_audio: np.ndarray | None = None
    melody_debug: dict[str, Any] | None = None
    if include_melody_stem:
        stems, melody_audio, melody_debug = _redistribute_instrumental_melody_from_stems(stems)

    manifest = {
        "audio_path": str(resolved_audio_path),
        "model_name": model_name,
        "sample_rate": SEPARATION_SAMPLE_RATE,
        "export_subtype": FINAL_STEM_EXPORT_SUBTYPE,
        "bleed_sharpening": bool(sharpen_bleed),
        "melody_stem_included": bool(include_melody_stem),
        "contract": _build_no_reference_stem_export_contract(route_name="full_res_stem_export"),
        "stems": {},
    }
    if model_name == HYBRID_STEM_EXPORT_MODEL:
        manifest["pipeline"] = {
            "mode": HYBRID_STEM_EXPORT_MODEL,
            "stage1_model": str(stage_models["stage1Model"]),
            "stage2_model": str(stage_models["stage2Model"]),
            "vocals_source": f"{stage_models['stage1Model']}+bs_roformer_vocals_other",
            "accompaniment_source": str(stage_models["stage2Model"]),
            "instrumental_strategy": "stage1_non_vocal_sum",
            "postprocess": "harmonic_percussive_rebalance_v1",
            "vocal_fusion": HYBRID_VOCAL_FUSION_LABEL,
            "melody_strategy": "instrumental_mask_reassign_v1" if include_melody_stem else None,
            "hybrid_stage_models": stage_models,
            "bs_roformer_runtime": bs_runtime,
        }
    if model_name == HYBRID_BS_STEM_EXPORT_MODEL:
        manifest["pipeline"] = {
            "mode": HYBRID_BS_STEM_EXPORT_MODEL,
            "stage0_model": "bs_roformer_vocals_other",
            "stage1_model": str(stage_models["stage1Model"]),
            "stage2_model": str(stage_models["stage2Model"]),
            "vocals_source": f"{stage_models['stage1Model']}+bs_roformer_vocals_other",
            "accompaniment_source": "bs_roformer_vocals_other",
            "instrumental_strategy": "bs_roformer_instrumental",
            "vocal_fusion": HYBRID_VOCAL_FUSION_LABEL,
            "melody_strategy": "instrumental_mask_reassign_v1" if include_melody_stem else None,
            "hybrid_stage_models": stage_models,
            "bs_roformer_runtime": bs_runtime,
        }

    for stem_name, stem_audio in stems.items():
        stem_path = resolved_output_dir / f"{stem_name}.flac"
        _emit_progress(f"writing_stem stem={stem_name} path={stem_path.name}")
        sf.write(stem_path, stem_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
        manifest["stems"][stem_name] = {
            "path": str(stem_path),
            **_audio_summary(stem_audio),
        }

    if include_melody_stem and melody_audio is not None and melody_debug is not None:
        melody_path = resolved_output_dir / "melody.flac"
        sf.write(melody_path, melody_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
        manifest["stems"]["melody"] = {
            "path": str(melody_path),
            **_audio_summary(melody_audio),
            **melody_debug,
        }

    manifest_path = resolved_output_dir / "stem_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    _emit_progress(f"EXIT export_track_stems_full_res manifest_path={manifest_path}")
    return manifest


def repair_exported_stem_directory(
    stem_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    include_melody_stem: bool = DEFAULT_STEM_EXPORT_INCLUDE_MELODY_STEM,
) -> dict[str, Any]:
    resolved_stem_dir = Path(stem_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_stem_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolved_stem_dir / "stem_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stems: dict[str, np.ndarray] = {}
    for stem_name in ("drums", "bass", "other", "vocals", "guitar", "piano"):
        stem_path = resolved_stem_dir / f"{stem_name}.flac"
        if not stem_path.exists():
            continue
        stem_audio, sample_rate = sf.read(str(stem_path), always_2d=True)
        if sample_rate != SEPARATION_SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate in {stem_path}: {sample_rate}")
        stems[stem_name] = stem_audio.astype(np.float32)

    if not stems:
        raise ValueError(f"No supported stem files found in {resolved_stem_dir}")

    melody_audio: np.ndarray | None = None
    melody_debug: dict[str, Any] | None = None
    if include_melody_stem:
        stems, melody_audio, melody_debug = _redistribute_instrumental_melody_from_stems(stems)

    manifest: dict[str, Any] = {
        "audio_path": existing_manifest.get("audio_path"),
        "model_name": existing_manifest.get("model_name"),
        "sample_rate": SEPARATION_SAMPLE_RATE,
        "export_subtype": existing_manifest.get("export_subtype", FINAL_STEM_EXPORT_SUBTYPE),
        "bleed_sharpening": existing_manifest.get("bleed_sharpening", False),
        "melody_stem_included": bool(include_melody_stem),
        "repair_source_manifest": str(manifest_path) if manifest_path.exists() else None,
        "pipeline": existing_manifest.get("pipeline", {}),
        "stems": {},
    }
    if include_melody_stem:
        manifest.setdefault("pipeline", {})
        manifest["pipeline"]["melody_strategy"] = "instrumental_mask_reassign_v1"

    for stem_name, stem_audio in stems.items():
        target_path = resolved_output_dir / f"{stem_name}.flac"
        sf.write(target_path, stem_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
        manifest["stems"][stem_name] = {
            "path": str(target_path),
            **_audio_summary(stem_audio),
        }

    if include_melody_stem and melody_audio is not None and melody_debug is not None:
        melody_path = resolved_output_dir / "melody.flac"
        sf.write(melody_path, melody_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
        manifest["stems"]["melody"] = {
            "path": str(melody_path),
            **_audio_summary(melody_audio),
            **melody_debug,
        }

    output_manifest_path = resolved_output_dir / "stem_manifest.json"
    output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(output_manifest_path)
    return manifest


def export_track_stems_aggressive_melody_first(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "htdemucs_6s",
    device: str | None = None,
    sharpen_bleed: bool = True,
    hybrid_stage_models: dict[str, Any] | None = None,
    bs_roformer_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_models = _normalize_hybrid_stage_models(hybrid_stage_models)
    bs_runtime = _normalize_bs_roformer_runtime(bs_roformer_runtime)
    _emit_progress(
        f"ENTER export_track_stems_aggressive_melody_first model={model_name} output_dir={output_dir}"
    )
    resolved_audio_path = Path(audio_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    raw_stems = separate_track_stems_full_res(
        resolved_audio_path,
        model_name=model_name,
        device=device,
        hybrid_stage_models=stage_models,
        bs_roformer_runtime=bs_runtime,
    )
    stems = sharpen_stem_bleed(raw_stems) if sharpen_bleed else raw_stems
    stems, melody_audio, melody_debug = _aggressive_melody_first_rebalance(stems)

    manifest: dict[str, Any] = {
        "audio_path": str(resolved_audio_path),
        "model_name": model_name,
        "sample_rate": SEPARATION_SAMPLE_RATE,
        "export_subtype": FINAL_STEM_EXPORT_SUBTYPE,
        "bleed_sharpening": bool(sharpen_bleed),
        "melody_stem_included": True,
        "contract": _build_no_reference_stem_export_contract(route_name="aggressive_melody_first"),
        "pipeline": {
            "mode": "aggressive_melody_first",
            "base_model": model_name,
            "instrumental_strategy": "raw_demucs_direct",
            "postprocess": "aggressive_harmonic_bus_reassign_v1",
            "melody_strategy": "aggressive_harmonic_bus_reassign_v1",
            "drums_strategy": "percussive_dominant_cleanup_v1",
            "hybrid_stage_models": (
                stage_models
                if model_name in {HYBRID_STEM_EXPORT_MODEL, HYBRID_BS_STEM_EXPORT_MODEL}
                else None
            ),
            "bs_roformer_runtime": (
                bs_runtime
                if model_name in {HYBRID_STEM_EXPORT_MODEL, HYBRID_BS_STEM_EXPORT_MODEL}
                else None
            ),
        },
        "stems": {},
    }

    for stem_name, stem_audio in stems.items():
        stem_path = resolved_output_dir / f"{stem_name}.flac"
        _emit_progress(f"writing_stem stem={stem_name} path={stem_path.name}")
        sf.write(stem_path, stem_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
        manifest["stems"][stem_name] = {
            "path": str(stem_path),
            **_audio_summary(stem_audio),
        }

    melody_path = resolved_output_dir / "melody.flac"
    sf.write(melody_path, melody_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
    manifest["stems"]["melody"] = {
        "path": str(melody_path),
        **_audio_summary(melody_audio),
        **melody_debug,
    }

    manifest_path = resolved_output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    _emit_progress(f"EXIT export_track_stems_aggressive_melody_first manifest_path={manifest_path}")
    return manifest


def export_aggressive_melody_first_from_stem_dir(
    stem_dir: str | Path,
    output_dir: str | Path,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    other_move: float = 0.35,
    guitar_move: float = 1.0,
    piano_move: float = 1.0,
    drums_masked_move: float = 0.08,
    target_melody_rms: float = 0.024,
    max_melody_gain: float = 1.4,
    stage_drum_harmonic_move: float = 0.35,
    stage_harmonic_stem_percussive_move: float = 0.18,
    stage_cleanup_drum_harmonic_move: float = 0.03,
    stage_cleanup_harmonic_stem_percussive_move: float = 0.25,
    harmonic_count: int = 2,
    confidence_floor: float = 0.35,
    dsp_device: str | None = None,
) -> dict[str, Any]:
    _emit_progress(
        f"ENTER export_aggressive_melody_first_from_stem_dir stem_dir={stem_dir} output_dir={output_dir} start_sec={start_sec} end_sec={end_sec}"
    )
    resolved_stem_dir = Path(stem_dir)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = resolved_stem_dir / "stem_manifest.json"
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    stems: dict[str, np.ndarray] = {}
    for stem_name in ("drums", "bass", "other", "vocals", "guitar", "piano"):
        stem_path = resolved_stem_dir / f"{stem_name}.flac"
        if not stem_path.exists():
            continue
        stem_audio, sample_rate = _read_audio_window(
            stem_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if sample_rate != SEPARATION_SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate in {stem_path}: {sample_rate}")
        stems[stem_name] = stem_audio.astype(np.float32)

    if not stems:
        _emit_progress(f"ERROR export_aggressive_melody_first_from_stem_dir no_supported_stem_files stem_dir={resolved_stem_dir}")
        raise ValueError(f"No supported stem files found in {resolved_stem_dir}")

    _emit_progress(
        f"preview_window_resolved start_sec={start_sec} end_sec={end_sec} loaded_stems={list(stems.keys())}"
    )
    _emit_progress("ENTER selective_melody_first_rebalance_from_stem_dir")
    rebalanced, melody_audio, melody_debug = _selective_melody_first_rebalance_from_stems(
        stems,
        other_move=other_move,
        guitar_move=guitar_move,
        piano_move=piano_move,
        drums_masked_move=drums_masked_move,
        target_melody_rms=target_melody_rms,
        max_melody_gain=max_melody_gain,
        stage_drum_harmonic_move=stage_drum_harmonic_move,
        stage_harmonic_stem_percussive_move=stage_harmonic_stem_percussive_move,
        stage_cleanup_drum_harmonic_move=stage_cleanup_drum_harmonic_move,
        stage_cleanup_harmonic_stem_percussive_move=stage_cleanup_harmonic_stem_percussive_move,
        harmonic_count=harmonic_count,
        confidence_floor=confidence_floor,
        dsp_device=dsp_device,
    )
    _emit_progress(
        f"EXIT selective_melody_first_rebalance_from_stem_dir melody_rms={melody_debug.get('melody_rms')} confidence_floor={melody_debug.get('confidence_floor')}"
    )
    manifest: dict[str, Any] = {
        "audio_path": source_manifest.get("audio_path"),
        "model_name": source_manifest.get("model_name"),
        "sample_rate": SEPARATION_SAMPLE_RATE,
        "export_subtype": source_manifest.get("export_subtype", FINAL_STEM_EXPORT_SUBTYPE),
        "bleed_sharpening": source_manifest.get("bleed_sharpening", False),
        "melody_stem_included": True,
        "source_stem_manifest": str(source_manifest_path) if source_manifest_path.exists() else None,
        "preview_window_sec": {
            "start": None if start_sec is None else float(start_sec),
            "end": None if end_sec is None else float(end_sec),
        },
        "tuning": {
            "other_move": float(other_move),
            "guitar_move": float(guitar_move),
            "piano_move": float(piano_move),
            "drums_masked_move": float(drums_masked_move),
            "target_melody_rms": float(target_melody_rms),
            "max_melody_gain": float(max_melody_gain),
            "stage_drum_harmonic_move": float(stage_drum_harmonic_move),
            "stage_harmonic_stem_percussive_move": float(stage_harmonic_stem_percussive_move),
            "stage_cleanup_drum_harmonic_move": float(stage_cleanup_drum_harmonic_move),
            "stage_cleanup_harmonic_stem_percussive_move": float(stage_cleanup_harmonic_stem_percussive_move),
            "harmonic_count": int(harmonic_count),
            "confidence_floor": float(confidence_floor),
            "dsp_device": str(_resolve_dsp_device(dsp_device)),
        },
        "pipeline": {
            "mode": "aggressive_melody_first_from_stem_dir",
            "base_model": source_manifest.get("model_name"),
            "source_stem_dir": str(resolved_stem_dir),
            "instrumental_strategy": "existing_stem_rewrite",
            "postprocess": "selective_pitch_mask_reassign_v3",
            "melody_strategy": "selective_pitch_mask_reassign_v3",
            "drums_strategy": "hpss_cleanup_plus_masked_drum_subtract_v2",
            "vocals_strategy": "preserve_source_stem_exact",
        },
        "stems": {},
    }

    for stem_name, stem_audio in rebalanced.items():
        target_path = resolved_output_dir / f"{stem_name}.flac"
        source_path = resolved_stem_dir / f"{stem_name}.flac"
        if stem_name in {"vocals", "bass"} and source_path.exists() and start_sec is None and end_sec is None:
            shutil.copyfile(source_path, target_path)
        else:
            sf.write(target_path, stem_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
        _emit_progress(f"writing_stem_dir_preview stem={stem_name} path={target_path.name}")
        manifest["stems"][stem_name] = {
            "path": str(target_path),
            **_audio_summary(stem_audio),
        }

    melody_path = resolved_output_dir / "melody.flac"
    sf.write(melody_path, melody_audio, SEPARATION_SAMPLE_RATE, subtype=FINAL_STEM_EXPORT_SUBTYPE)
    _emit_progress(f"writing_stem_dir_preview stem=melody path={melody_path.name}")
    manifest["stems"]["melody"] = {
        "path": str(melody_path),
        **_audio_summary(melody_audio),
        **melody_debug,
    }

    manifest_path = resolved_output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    _emit_progress(f"EXIT export_aggressive_melody_first_from_stem_dir manifest_path={manifest_path}")
    return manifest
