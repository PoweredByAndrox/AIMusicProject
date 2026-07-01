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

try:
    from tools.stem_region_repair_common import write_audio, write_json
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import write_audio, write_json

DISTILLATION_OBJECTIVE_NAME = "melband_pitch_attribution_separator_distillation_v2"
DISTILLATION_SCHEMA_NAME = "melband_pitch_attribution_distillation_training.v1"
START_TIME = time.monotonic()


@dataclass(frozen=True)
class PitchAttributionDistillationConfig:
    epochs: int = 180
    learning_rate: float = 0.02
    hidden_channels: int = 24
    kernel_size: int = 17
    target_loss_weight: float = 1.0
    outside_loss_weight: float = 0.8
    smoothness_loss_weight: float = 0.01
    max_delta: float = 1.0
    use_frame_residual_basis: bool = True
    seed: int = 2026
    device: str = "cpu"


class PitchAttributionDistillationNet(nn.Module):
    """Small separator-side delta predictor for pitch-attribution hardcases.

    This is deliberately still a diagnostic training object. Unlike the scalar
    adapter, it can learn a delta that is not simply `source_probe * gain`,
    because it conditions on the separator-side tensors.
    """

    def __init__(
        self,
        channels: int = 2,
        hidden_channels: int = 24,
        kernel_size: int = 17,
        max_delta: float = 1.0,
        region_mask: np.ndarray | None = None,
    ):
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        kernel = max(1, int(kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        self.channels = int(channels)
        self.max_delta = float(max_delta)
        input_channels = int(channels) * 8 + 1
        final = nn.Conv1d(int(hidden_channels), int(channels), kernel_size=1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, int(hidden_channels), kernel_size=kernel, padding=padding),
            nn.Tanh(),
            nn.Conv1d(int(hidden_channels), int(hidden_channels), kernel_size=kernel, padding=padding),
            nn.Tanh(),
            final,
        )
        if region_mask is not None:
            mask = torch.as_tensor(np.asarray(region_mask, dtype=np.float32).reshape(-1), dtype=torch.float32)
            if int(mask.shape[0]) <= 0:
                raise ValueError("region_mask must not be empty")
            self.register_buffer("frame_basis_mask", mask.view(1, 1, -1), persistent=True)
            self.frame_delta = nn.Parameter(torch.zeros(1, self.channels, int(mask.shape[0])))
        else:
            self.register_buffer("frame_basis_mask", torch.empty(1, 1, 0), persistent=True)
            self.frame_delta = None

    def forward(
        self,
        *,
        mix_audio: torch.Tensor,
        vocals_audio: torch.Tensor,
        instrumental_audio: torch.Tensor,
        source_probe_audio: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        if mix_audio.ndim != 3:
            raise ValueError("mix_audio must have shape [batch, frames, channels]")
        if vocals_audio.shape != mix_audio.shape:
            raise ValueError("vocals_audio must match mix_audio shape")
        if instrumental_audio.shape != mix_audio.shape:
            raise ValueError("instrumental_audio must match mix_audio shape")
        if source_probe_audio.shape != mix_audio.shape:
            raise ValueError("source_probe_audio must match mix_audio shape")
        if region_mask.ndim != 3:
            raise ValueError("region_mask must have shape [batch, frames, 1]")
        features = torch.cat(
            [
                mix_audio.transpose(1, 2),
                vocals_audio.transpose(1, 2),
                instrumental_audio.transpose(1, 2),
                source_probe_audio.transpose(1, 2),
                torch.abs(vocals_audio).transpose(1, 2),
                torch.abs(instrumental_audio).transpose(1, 2),
                torch.abs(source_probe_audio).transpose(1, 2),
                (source_probe_audio * instrumental_audio).transpose(1, 2),
                region_mask.transpose(1, 2),
            ],
            dim=1,
        )
        mask = region_mask.transpose(1, 2)
        delta = torch.tanh(self.net(features)) * self.max_delta
        if self.frame_delta is not None and int(self.frame_delta.shape[-1]) == int(source_probe_audio.shape[1]):
            frame_delta = torch.clamp(self.frame_delta, -self.max_delta, self.max_delta)
            delta = delta + frame_delta * self.frame_basis_mask
        delta = delta * mask
        return delta.transpose(1, 2)


def resolve_pitch_attribution_distillation_device(requested_device: str) -> torch.device:
    requested = str(requested_device).lower()
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested for pitch-attribution distillation but is unavailable")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested for pitch-attribution distillation but is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported pitch-attribution distillation device: {requested_device}")


def validate_pitch_attribution_distillation_label_contract(*, label_type: str, is_ground_truth: bool) -> None:
    label = str(label_type)
    allowed = {"synthetic_verified", "reference_verified", "human_verified", "pseudo_teacher", "weak_proxy"}
    if label not in allowed:
        raise ValueError(f"unsupported pitch-attribution distillation label_type: {label}")
    if bool(is_ground_truth) and label in {"pseudo_teacher", "weak_proxy"}:
        raise ValueError(f"{label} labels cannot be marked as ground truth")


def _append_progress(path: Path | None, phase: str, status: str, **fields: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_sec": round(time.monotonic() - START_TIME, 6),
        "phase": str(phase),
        "status": str(status),
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _aligned_inputs(
    *,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    region_mask: np.ndarray,
    target_delta_audio: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    arrays = [
        np.asarray(mix_audio, dtype=np.float32),
        np.asarray(vocals_audio, dtype=np.float32),
        np.asarray(instrumental_audio, dtype=np.float32),
        np.asarray(source_probe_audio, dtype=np.float32),
    ]
    if target_delta_audio is not None:
        arrays.append(np.asarray(target_delta_audio, dtype=np.float32))
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("audio arrays must have shape [frames, channels]")
    mask = np.asarray(region_mask, dtype=np.float32).reshape(-1)
    frame_count = min([int(array.shape[0]) for array in arrays] + [int(mask.shape[0])])
    channel_count = min(int(array.shape[1]) for array in arrays)
    if frame_count <= 0 or channel_count <= 0:
        raise ValueError("pitch-attribution distillation inputs must not be empty")
    payload = {
        "mix": arrays[0][:frame_count, :channel_count].astype(np.float32, copy=False),
        "vocals": arrays[1][:frame_count, :channel_count].astype(np.float32, copy=False),
        "instrumental": arrays[2][:frame_count, :channel_count].astype(np.float32, copy=False),
        "probe": arrays[3][:frame_count, :channel_count].astype(np.float32, copy=False),
        "mask": np.clip(mask[:frame_count].astype(np.float32, copy=False), 0.0, 1.0),
    }
    if target_delta_audio is not None:
        payload["target"] = arrays[4][:frame_count, :channel_count].astype(np.float32, copy=False)
    return payload


def _to_tensor(audio: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(audio[None, :, :], dtype=torch.float32, device=device)


def _loss_for_delta(
    *,
    delta: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: PitchAttributionDistillationConfig,
) -> torch.Tensor:
    outside = 1.0 - mask
    target_loss = torch.mean(torch.square((delta - target) * mask))
    outside_loss = torch.mean(torch.square(delta * outside))
    if delta.shape[1] > 1:
        smooth_loss = torch.mean(torch.square(delta[:, 1:, :] - delta[:, :-1, :]))
    else:
        smooth_loss = delta.new_tensor(0.0)
    return (
        float(config.target_loss_weight) * target_loss
        + float(config.outside_loss_weight) * outside_loss
        + float(config.smoothness_loss_weight) * smooth_loss
    )


def _max_weight_delta(before_state: dict[str, torch.Tensor], model: nn.Module) -> float:
    max_delta = 0.0
    for name, value in model.state_dict().items():
        if name not in before_state:
            continue
        delta = torch.max(torch.abs(value.detach().cpu() - before_state[name])).item()
        max_delta = max(max_delta, float(delta))
    return max_delta


def train_pitch_attribution_distillation_from_arrays(
    *,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    region_mask: np.ndarray,
    target_delta_audio: np.ndarray,
    config: PitchAttributionDistillationConfig,
    progress_path: Path | None = None,
) -> tuple[PitchAttributionDistillationNet, dict[str, Any]]:
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = resolve_pitch_attribution_distillation_device(str(config.device))
    payload = _aligned_inputs(
        mix_audio=mix_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_probe_audio,
        region_mask=region_mask,
        target_delta_audio=target_delta_audio,
    )
    model = PitchAttributionDistillationNet(
        channels=int(payload["mix"].shape[1]),
        hidden_channels=int(config.hidden_channels),
        kernel_size=int(config.kernel_size),
        max_delta=float(config.max_delta),
        region_mask=payload["mask"] if bool(config.use_frame_residual_basis) else None,
    ).to(device)
    before_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    mix_tensor = _to_tensor(payload["mix"], device=device)
    vocals_tensor = _to_tensor(payload["vocals"], device=device)
    instrumental_tensor = _to_tensor(payload["instrumental"], device=device)
    probe_tensor = _to_tensor(payload["probe"], device=device)
    target_tensor = _to_tensor(payload["target"], device=device)
    mask_tensor = torch.as_tensor(payload["mask"][None, :, None], dtype=torch.float32, device=device)

    with torch.no_grad():
        initial_delta = model(
            mix_audio=mix_tensor,
            vocals_audio=vocals_tensor,
            instrumental_audio=instrumental_tensor,
            source_probe_audio=probe_tensor,
            region_mask=mask_tensor,
        )
        initial_loss = float(
            _loss_for_delta(delta=initial_delta, target=target_tensor, mask=mask_tensor, config=config).detach().cpu()
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    loss_history: list[float] = []
    epochs = max(1, int(config.epochs))
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        delta = model(
            mix_audio=mix_tensor,
            vocals_audio=vocals_tensor,
            instrumental_audio=instrumental_tensor,
            source_probe_audio=probe_tensor,
            region_mask=mask_tensor,
        )
        loss = _loss_for_delta(delta=delta, target=target_tensor, mask=mask_tensor, config=config)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            _append_progress(
                progress_path,
                "train",
                "epoch",
                epoch=int(epoch + 1),
                epochs=epochs,
                loss=loss_value,
            )

    with torch.no_grad():
        final_delta = model(
            mix_audio=mix_tensor,
            vocals_audio=vocals_tensor,
            instrumental_audio=instrumental_tensor,
            source_probe_audio=probe_tensor,
            region_mask=mask_tensor,
        )
        final_loss = float(
            _loss_for_delta(delta=final_delta, target=target_tensor, mask=mask_tensor, config=config).detach().cpu()
        )
    max_weight_delta = _max_weight_delta(before_state, model)
    diagnostics = {
        "objective": DISTILLATION_OBJECTIVE_NAME,
        "training_level": "separator_side_distillation",
        "resolved_device": str(device),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "weights_changed": bool(max_weight_delta > 1.0e-9),
        "max_weight_delta": max_weight_delta,
        "training_frames": int(payload["mix"].shape[0]),
        "training_channels": int(payload["mix"].shape[1]),
        "active_region_fraction": float(np.mean(payload["mask"], dtype=np.float64)),
        "use_frame_residual_basis": bool(config.use_frame_residual_basis),
        "delivery_allowed": False,
        "listen_pack_created": False,
    }
    return model, diagnostics


def apply_pitch_attribution_distillation(
    *,
    model: PitchAttributionDistillationNet,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    source_probe_audio: np.ndarray,
    region_mask: np.ndarray,
    config: PitchAttributionDistillationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    device = resolve_pitch_attribution_distillation_device(str(config.device))
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
        delta = (
            model(
                mix_audio=_to_tensor(payload["mix"], device=device),
                vocals_audio=_to_tensor(payload["vocals"], device=device),
                instrumental_audio=_to_tensor(payload["instrumental"], device=device),
                source_probe_audio=_to_tensor(payload["probe"], device=device),
                region_mask=torch.as_tensor(payload["mask"][None, :, None], dtype=torch.float32, device=device),
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
        "objective": DISTILLATION_OBJECTIVE_NAME,
        "training_level": "separator_side_distillation",
        "resolved_device": str(device),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "training_diagnostic_only",
        "mixture_consistency_max_abs_error": float(np.max(np.abs(mixture_error))) if mixture_error.size else 0.0,
        "outside_change_max_abs": float(np.max(np.abs(delta * outside))) if delta.size else 0.0,
        "delta_rms": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))) if delta.size else 0.0,
    }
    return repaired_vocals, repaired_instrumental, delta, diagnostics


def run_pitch_attribution_distillation_training(
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
    config: PitchAttributionDistillationConfig,
    sample_rate: int = 8_000,
) -> dict[str, Any]:
    validate_pitch_attribution_distillation_label_contract(label_type=label_type, is_ground_truth=is_ground_truth)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(
        progress_path,
        "preflight",
        "start",
        objective=DISTILLATION_OBJECTIVE_NAME,
        delivery_allowed=False,
        device_requested=str(config.device),
    )
    model, train_diagnostics = train_pitch_attribution_distillation_from_arrays(
        mix_audio=mix_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_probe_audio,
        region_mask=region_mask,
        target_delta_audio=target_delta_audio,
        config=config,
        progress_path=progress_path,
    )
    repaired_vocals, repaired_instrumental, delta, apply_diagnostics = apply_pitch_attribution_distillation(
        model=model,
        mix_audio=mix_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_probe_audio,
        region_mask=region_mask,
        config=config,
    )
    audio_dir = output_dir / "audio"
    raw = np.asarray(vocals_audio, dtype=np.float32) + np.asarray(instrumental_audio, dtype=np.float32)
    vocals_out = write_audio(audio_dir / "v.flac", np.clip(repaired_vocals, -1.0, 1.0), int(sample_rate))
    instrumental_out = write_audio(audio_dir / "i.flac", np.clip(repaired_instrumental, -1.0, 1.0), int(sample_rate))
    raw_out = write_audio(audio_dir / "raw.flac", np.clip(raw, -1.0, 1.0), int(sample_rate))
    source_out = write_audio(audio_dir / "src.flac", np.asarray(source_probe_audio, dtype=np.float32), int(sample_rate))
    delta_out = write_audio(audio_dir / "d.flac", np.asarray(delta, dtype=np.float32), int(sample_rate))
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "schema": DISTILLATION_SCHEMA_NAME,
            "objective": DISTILLATION_OBJECTIVE_NAME,
            "training_level": "separator_side_distillation",
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "channels": int(delta.shape[1]),
        },
        str(checkpoint_path),
    )
    manifest = {
        "schema": DISTILLATION_SCHEMA_NAME,
        "objective": DISTILLATION_OBJECTIVE_NAME,
        "status": "trained_distillation_diagnostic",
        "checkpoint_path": str(checkpoint_path),
        "training_level": "separator_side_distillation",
        "target_model_family": "melband_roformer_vocals_or_separator_adapter",
        "must_train_weights": True,
        "weights_changed": bool(train_diagnostics.get("weights_changed")),
        "label_type": str(label_type),
        "is_ground_truth": bool(is_ground_truth),
        "usable_for_promotion": False,
        "audio": {
            "vocals": str(vocals_out),
            "instrumental": str(instrumental_out),
            "raw_ref": str(raw_out),
            "source_probe": str(source_out),
            "target_delta": str(delta_out),
        },
        "metrics": {
            "apply_diagnostics": apply_diagnostics,
            "target_delta_rms": float(np.sqrt(np.mean(np.square(target_delta_audio), dtype=np.float64))),
            "reconstruction_rms": float(np.sqrt(np.mean(np.square((repaired_vocals + repaired_instrumental) - raw)))),
            "delta_rms": apply_diagnostics["delta_rms"],
        },
        "training": train_diagnostics,
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "cannot_override_gates": True,
            "mixture_consistent_by_construction": True,
            "delivery_allowed": False,
            "usable_for_promotion": False,
        },
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "training_diagnostic_only",
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        "manifest",
        "end",
        run_status=manifest["status"],
        weights_changed=manifest["weights_changed"],
        delivery_allowed=False,
    )
    return manifest
