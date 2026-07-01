#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

OBJECTIVE_NAME = "melband_pitch_attribution_separator_objective_v1"
SCHEMA_NAME = "melband_pitch_attribution_adapter_training.v1"
START_TIME = time.monotonic()
EPSILON = 1.0e-12


@dataclass(frozen=True)
class PitchAttributionAdapterConfig:
    epochs: int = 160
    learning_rate: float = 0.04
    attribution_loss_weight: float = 1.0
    outside_loss_weight: float = 0.5
    mixture_loss_weight: float = 0.05
    max_probe_gain: float = 1.5
    adapter_mode: str = "global"
    conditioned_hidden_channels: int = 16
    conditioned_kernel_size: int = 17
    seed: int = 2026
    device: str = "cpu"


class PitchAttributionAdapter(nn.Module):
    """A small separator-adapter head that moves source-probe energy between lanes.

    The adapter is mixture-consistent by construction: the predicted delta is
    added to vocals and subtracted from instrumental. It is not a delivery
    route; it is a trainable objective probe for separator attribution.
    """

    def __init__(self, channels: int = 2, max_probe_gain: float = 1.5) -> None:
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.max_probe_gain = float(max_probe_gain)
        self.probe_gain = nn.Parameter(torch.zeros(1, self.channels, 1))

    def forward(
        self,
        *,
        source_probe_audio: torch.Tensor,
        region_mask: torch.Tensor,
        vocals_audio: torch.Tensor | None = None,
        instrumental_audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_probe_audio.ndim != 3:
            raise ValueError("source_probe_audio must have shape [batch, frames, channels]")
        if region_mask.ndim != 3:
            raise ValueError("region_mask must have shape [batch, frames, 1]")
        probe = source_probe_audio.transpose(1, 2)
        mask = region_mask.transpose(1, 2)
        gain = torch.clamp(self.probe_gain, -self.max_probe_gain, self.max_probe_gain)
        delta = probe * gain * mask
        return delta.transpose(1, 2)


class ConditionedPitchAttributionAdapter(nn.Module):
    """Frame-conditioned attribution adapter for non-uniform leak regions."""

    def __init__(
        self,
        channels: int = 2,
        hidden_channels: int = 16,
        max_delta: float = 1.5,
        kernel_size: int = 17,
        region_basis: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.max_delta = float(max_delta)
        if region_basis is not None:
            basis = torch.as_tensor(region_basis, dtype=torch.float32)
            if basis.ndim != 2:
                raise ValueError("region_basis must have shape [regions, frames]")
            if basis.shape[0] <= 0 or basis.shape[1] <= 0:
                raise ValueError("region_basis must not be empty")
            self.register_buffer("region_basis", basis.unsqueeze(0), persistent=True)
            self.region_gains = nn.Parameter(torch.zeros(1, self.channels, int(basis.shape[0])))
        else:
            self.register_buffer("region_basis", torch.empty(1, 0, 0), persistent=True)
            self.region_gains = None
        kernel = max(1, int(kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        input_channels = int(channels) * 7 + 1
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, int(hidden_channels), kernel_size=kernel, padding=padding),
            nn.Tanh(),
            nn.Conv1d(int(hidden_channels), int(hidden_channels), kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(int(hidden_channels), int(channels), kernel_size=1),
        )

    def forward(
        self,
        *,
        source_probe_audio: torch.Tensor,
        region_mask: torch.Tensor,
        vocals_audio: torch.Tensor | None = None,
        instrumental_audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_probe_audio.ndim != 3:
            raise ValueError("source_probe_audio must have shape [batch, frames, channels]")
        if region_mask.ndim != 3:
            raise ValueError("region_mask must have shape [batch, frames, 1]")
        if vocals_audio is None or instrumental_audio is None:
            raise ValueError("conditioned adapter requires vocals_audio and instrumental_audio")
        if vocals_audio.shape != source_probe_audio.shape or instrumental_audio.shape != source_probe_audio.shape:
            raise ValueError("conditioned adapter audio tensors must share shape")
        if self.region_gains is not None and int(self.region_basis.shape[-1]) == int(source_probe_audio.shape[1]):
            probe = source_probe_audio.transpose(1, 2)
            mask = region_mask.transpose(1, 2)
            gains = torch.clamp(self.region_gains, -self.max_delta, self.max_delta)
            gain_map = torch.sum(gains.unsqueeze(-1) * self.region_basis.unsqueeze(1), dim=2)
            return (probe * gain_map * mask).transpose(1, 2)
        features = torch.cat(
            [
                source_probe_audio.transpose(1, 2),
                vocals_audio.transpose(1, 2),
                instrumental_audio.transpose(1, 2),
                torch.abs(source_probe_audio).transpose(1, 2),
                torch.abs(vocals_audio).transpose(1, 2),
                torch.abs(instrumental_audio).transpose(1, 2),
                (source_probe_audio * instrumental_audio).transpose(1, 2),
                region_mask.transpose(1, 2),
            ],
            dim=1,
        )
        mask = region_mask.transpose(1, 2)
        delta = torch.tanh(self.net(features)) * self.max_delta * mask
        return delta.transpose(1, 2)


def validate_adapter_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in {"global", "conditioned"}:
        raise ValueError(f"unsupported pitch-attribution adapter_mode: {value}")
    return mode


def _region_basis_from_mask(region_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(region_mask, dtype=np.float32).reshape(-1) > 0.0
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if bool(active) and start is None:
            start = int(index)
        if (not bool(active) or index == len(mask) - 1) and start is not None:
            end = int(index + 1 if bool(active) and index == len(mask) - 1 else index)
            if end > start:
                regions.append((start, end))
            start = None
    if not regions:
        return np.zeros((0, int(mask.shape[0])), dtype=np.float32)
    basis = np.zeros((len(regions), int(mask.shape[0])), dtype=np.float32)
    for region_index, (start_i, end_i) in enumerate(regions):
        basis[region_index, start_i:end_i] = 1.0
    return basis


def build_pitch_attribution_adapter(
    *, config: PitchAttributionAdapterConfig, channels: int, region_mask: np.ndarray | None = None
) -> nn.Module:
    mode = validate_adapter_mode(str(config.adapter_mode))
    if mode == "conditioned":
        return ConditionedPitchAttributionAdapter(
            channels=int(channels),
            hidden_channels=int(config.conditioned_hidden_channels),
            max_delta=float(config.max_probe_gain),
            kernel_size=int(config.conditioned_kernel_size),
            region_basis=_region_basis_from_mask(region_mask) if region_mask is not None else None,
        )
    return PitchAttributionAdapter(channels=int(channels), max_probe_gain=float(config.max_probe_gain))


def resolve_pitch_attribution_device(requested_device: str) -> torch.device:
    requested = str(requested_device).lower()
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested for pitch-attribution training but is unavailable")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested for pitch-attribution training but is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported pitch-attribution training device: {requested_device}")


def validate_pitch_attribution_label_contract(*, label_type: str, is_ground_truth: bool) -> None:
    label = str(label_type)
    allowed = {
        "synthetic_verified",
        "reference_verified",
        "human_verified",
        "pseudo_teacher",
        "weak_proxy",
    }
    if label not in allowed:
        raise ValueError(f"unsupported pitch-attribution label_type: {label}")
    if bool(is_ground_truth) and label in {"pseudo_teacher", "weak_proxy"}:
        raise ValueError(f"{label} labels cannot be marked as ground truth")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"elapsed_sec": round(time.monotonic() - START_TIME, 6), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _stereo_array(name: str, audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [frames, channels]")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _mask_array(region_mask: np.ndarray, frames: int) -> np.ndarray:
    mask = np.asarray(region_mask, dtype=np.float32)
    if mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 1:
        raise ValueError("region_mask must have shape [frames] or [frames, 1]")
    if int(mask.shape[0]) != int(frames):
        raise ValueError(f"region_mask length mismatch: expected {frames}, got {mask.shape[0]}")
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def _aligned_inputs(
    *,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    region_mask: np.ndarray,
    target_delta_audio: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    mix = _stereo_array("mix_audio", mix_audio)
    vocals = _stereo_array("vocals_audio", vocals_audio)
    instrumental = _stereo_array("instrumental_audio", instrumental_audio)
    probe = _stereo_array("source_probe_audio", source_probe_audio)
    arrays = [mix, vocals, instrumental, probe]
    if target_delta_audio is not None:
        arrays.append(_stereo_array("target_delta_audio", target_delta_audio))
    frames = min(array.shape[0] for array in arrays)
    channels = min(array.shape[1] for array in arrays)
    if frames <= 0 or channels <= 0:
        raise ValueError("aligned pitch-attribution inputs are empty")
    payload = {
        "mix": mix[:frames, :channels],
        "vocals": vocals[:frames, :channels],
        "instrumental": instrumental[:frames, :channels],
        "probe": probe[:frames, :channels],
        "mask": _mask_array(region_mask, frames)[:frames],
    }
    if target_delta_audio is not None:
        payload["target_delta"] = _stereo_array("target_delta_audio", target_delta_audio)[:frames, :channels]
    return payload


def _to_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)


def _loss_for_delta(
    *,
    delta: torch.Tensor,
    target_delta: torch.Tensor,
    region_mask: torch.Tensor,
    vocals: torch.Tensor,
    instrumental: torch.Tensor,
    config: PitchAttributionAdapterConfig,
) -> torch.Tensor:
    mask = region_mask
    outside = 1.0 - mask
    region_den = torch.clamp(mask.sum() * float(delta.shape[-1]), min=1.0)
    outside_den = torch.clamp(outside.sum() * float(delta.shape[-1]), min=1.0)
    attribution_loss = torch.sum(torch.square(delta - target_delta) * mask) / region_den
    outside_loss = torch.sum(torch.square(delta) * outside) / outside_den
    repaired_vocals = vocals + delta
    repaired_instrumental = instrumental - delta
    mixture_loss = torch.mean(torch.square((repaired_vocals + repaired_instrumental) - (vocals + instrumental)))
    return (
        float(config.attribution_loss_weight) * attribution_loss
        + float(config.outside_loss_weight) * outside_loss
        + float(config.mixture_loss_weight) * mixture_loss
    )


def _state_max_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    max_delta = 0.0
    for name, before_value in before.items():
        after_value = after[name]
        max_delta = max(max_delta, float(torch.max(torch.abs(after_value.detach().cpu() - before_value)).item()))
    return max_delta


def estimate_region_projection_gains(
    *,
    delta_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    regions: list[tuple[int, int]],
) -> list[float]:
    delta = _stereo_array("delta_audio", delta_audio)
    probe = _stereo_array("source_probe_audio", source_probe_audio)
    frames = min(delta.shape[0], probe.shape[0])
    delta = delta[:frames]
    probe = probe[:frames]
    gains: list[float] = []
    for start, end in regions:
        start_i = max(0, min(frames, int(start)))
        end_i = max(start_i, min(frames, int(end)))
        if end_i <= start_i:
            gains.append(0.0)
            continue
        delta_seg = delta[start_i:end_i].astype(np.float64)
        probe_seg = probe[start_i:end_i].astype(np.float64)
        denom = float(np.sum(probe_seg * probe_seg, dtype=np.float64)) + EPSILON
        gains.append(float(np.sum(delta_seg * probe_seg, dtype=np.float64) / denom))
    return gains


def train_pitch_attribution_adapter_from_arrays(
    *,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    region_mask: np.ndarray,
    target_delta_audio: np.ndarray,
    config: PitchAttributionAdapterConfig,
    progress_path: Path | None = None,
) -> tuple[PitchAttributionAdapter, dict[str, Any]]:
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = resolve_pitch_attribution_device(str(config.device))
    payload = _aligned_inputs(
        mix_audio=mix_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_probe_audio,
        region_mask=region_mask,
        target_delta_audio=target_delta_audio,
    )
    adapter_mode = validate_adapter_mode(str(config.adapter_mode))
    model = build_pitch_attribution_adapter(
        config=config,
        channels=int(payload["vocals"].shape[1]),
        region_mask=payload["mask"] if adapter_mode == "conditioned" else None,
    ).to(device)
    before_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    vocals = _to_tensor(payload["vocals"], device=device)
    instrumental = _to_tensor(payload["instrumental"], device=device)
    probe = _to_tensor(payload["probe"], device=device)
    mask = _to_tensor(payload["mask"][:, None], device=device)
    target_delta = _to_tensor(payload["target_delta"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))

    with torch.no_grad():
        initial_delta = model(
            source_probe_audio=probe,
            region_mask=mask,
            vocals_audio=vocals,
            instrumental_audio=instrumental,
        )
        initial_loss = _loss_for_delta(
            delta=initial_delta,
            target_delta=target_delta,
            region_mask=mask,
            vocals=vocals,
            instrumental=instrumental,
            config=config,
        )

    loss_history: list[float] = []
    for epoch in range(1, int(config.epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = model(
            source_probe_audio=probe,
            region_mask=mask,
            vocals_audio=vocals,
            instrumental_audio=instrumental,
        )
        loss = _loss_for_delta(
            delta=delta,
            target_delta=target_delta,
            region_mask=mask,
            vocals=vocals,
            instrumental=instrumental,
            config=config,
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        loss_history.append(loss_value)
        if progress_path is not None and (epoch == 1 or epoch == int(config.epochs) or epoch % max(1, int(config.epochs) // 5) == 0):
            _append_progress(
                progress_path,
                {
                    "event": "epoch",
                    "epoch": int(epoch),
                    "epochs": int(config.epochs),
                    "loss": round(loss_value, 9),
                    "device": str(device),
                },
            )

    with torch.no_grad():
        final_delta = model(
            source_probe_audio=probe,
            region_mask=mask,
            vocals_audio=vocals,
            instrumental_audio=instrumental,
        )
        final_loss = _loss_for_delta(
            delta=final_delta,
            target_delta=target_delta,
            region_mask=mask,
            vocals=vocals,
            instrumental=instrumental,
            config=config,
        )
    after_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    max_weight_delta = _state_max_delta(before_state, after_state)
    diagnostics = {
        "resolved_device": str(device),
        "adapter_mode": adapter_mode,
        "initial_loss": float(initial_loss.detach().cpu().item()),
        "final_loss": float(final_loss.detach().cpu().item()),
        "loss_history": loss_history,
        "weights_changed": bool(max_weight_delta > 1.0e-9),
        "max_weight_delta": max_weight_delta,
        "training_frames": int(payload["vocals"].shape[0]),
        "training_channels": int(payload["vocals"].shape[1]),
        "active_region_fraction": float(np.mean(payload["mask"], dtype=np.float64)),
    }
    return model.cpu(), diagnostics


def apply_pitch_attribution_adapter(
    *,
    model: PitchAttributionAdapter,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    region_mask: np.ndarray,
    config: PitchAttributionAdapterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    device = resolve_pitch_attribution_device(str(config.device))
    payload = _aligned_inputs(
        mix_audio=mix_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_probe_audio,
        region_mask=region_mask,
    )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        probe = _to_tensor(payload["probe"], device=device)
        mask = _to_tensor(payload["mask"][:, None], device=device)
        vocals_tensor = _to_tensor(payload["vocals"], device=device)
        instrumental_tensor = _to_tensor(payload["instrumental"], device=device)
        delta = (
            model(
                source_probe_audio=probe,
                region_mask=mask,
                vocals_audio=vocals_tensor,
                instrumental_audio=instrumental_tensor,
            )
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )
    vocals = payload["vocals"].astype(np.float32, copy=False)
    instrumental = payload["instrumental"].astype(np.float32, copy=False)
    repaired_vocals = vocals + delta
    repaired_instrumental = instrumental - delta
    outside = (1.0 - payload["mask"])[:, None].astype(np.float32)
    mixture_error = (repaired_vocals + repaired_instrumental) - (vocals + instrumental)
    diagnostics = {
        "resolved_device": str(device),
        "adapter_mode": validate_adapter_mode(str(config.adapter_mode)),
        "delivery_allowed": False,
        "presentation_status": "training_diagnostic_only",
        "mixture_consistency_max_abs_error": float(np.max(np.abs(mixture_error))) if mixture_error.size else 0.0,
        "outside_change_max_abs": float(max(np.max(np.abs(delta * outside)), 0.0)) if delta.size else 0.0,
        "delta_rms": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))) if delta.size else 0.0,
    }
    return repaired_vocals, repaired_instrumental, delta, diagnostics


def run_pitch_attribution_adapter_training(
    *,
    output_dir: Path,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    region_mask: np.ndarray,
    target_delta_audio: np.ndarray,
    label_type: str,
    is_ground_truth: bool,
    config: PitchAttributionAdapterConfig,
) -> dict[str, Any]:
    validate_pitch_attribution_label_contract(label_type=label_type, is_ground_truth=bool(is_ground_truth))
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(
        progress_path,
        {
            "event": "start",
            "objective": OBJECTIVE_NAME,
            "device_requested": str(config.device),
            "delivery_allowed": False,
        },
    )
    model, train_diagnostics = train_pitch_attribution_adapter_from_arrays(
        mix_audio=mix_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_probe_audio,
        region_mask=region_mask,
        target_delta_audio=target_delta_audio,
        config=config,
        progress_path=progress_path,
    )
    repaired_vocals, repaired_instrumental, delta, apply_diagnostics = apply_pitch_attribution_adapter(
        model=model,
        mix_audio=mix_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_probe_audio,
        region_mask=region_mask,
        config=PitchAttributionAdapterConfig(**{**asdict(config), "device": "cpu"}),
    )
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "schema": SCHEMA_NAME,
            "objective": OBJECTIVE_NAME,
            "training_level": "separator_adapter",
            "adapter_mode": validate_adapter_mode(str(config.adapter_mode)),
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "channels": int(repaired_vocals.shape[1]),
            "region_basis_shape": list(getattr(model, "region_basis", torch.empty(1, 0, 0)).shape),
        },
        str(checkpoint_path),
    )
    manifest = {
        "schema": SCHEMA_NAME,
        "objective": OBJECTIVE_NAME,
        "status": "trained_separator_adapter_diagnostic"
        if bool(train_diagnostics["weights_changed"])
        else "blocked_no_weight_update",
        "training_level": "separator_adapter",
        "target_model_family": "melband_roformer_vocals_or_adapter",
        "must_train_weights": True,
        "weights_changed": bool(train_diagnostics["weights_changed"]),
        "label_type": str(label_type),
        "is_ground_truth": bool(is_ground_truth),
        "pseudo_teacher_targets_are_ground_truth": False,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "release_ready": False,
        "listen_pack_created": False,
        "presentation_status": "training_diagnostic_only",
        "checkpoint_path": str(checkpoint_path),
        "progress_jsonl": str(progress_path),
        "training": train_diagnostics,
        "application": apply_diagnostics,
        "audio_diagnostics": {
            "delta_rms": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))) if delta.size else 0.0,
            "repaired_vocals_rms": float(np.sqrt(np.mean(np.square(repaired_vocals), dtype=np.float64)))
            if repaired_vocals.size
            else 0.0,
            "repaired_instrumental_rms": float(np.sqrt(np.mean(np.square(repaired_instrumental), dtype=np.float64)))
            if repaired_instrumental.size
            else 0.0,
        },
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "cannot_override_gates": True,
            "mixture_consistent_by_construction": True,
            "pseudo_teacher_cannot_be_ground_truth": True,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        {
            "event": "finish",
            "status": manifest["status"],
            "weights_changed": bool(train_diagnostics["weights_changed"]),
            "delivery_allowed": False,
        },
    )
    return manifest
