"""Tiny trainable stereo patch repair model for stem-region defects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from tools.stem_region_repair_common import ISSUE_TYPES, ensure_stereo
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import ISSUE_TYPES, ensure_stereo


EPSILON = 1.0e-8


@dataclass(frozen=True)
class StemRegionRepairTrainConfig:
    hidden_channels: int = 32
    kernel_size: int = 31
    dilated_layers: int = 0
    epochs: int = 120
    learning_rate: float = 0.003
    region_loss_weight: float = 1.0
    outside_loss_weight: float = 0.2
    mix_conservation_loss_weight: float = 0.0
    sibling_preserve_loss_weight: float = 0.0
    stem_sum_loss_weight: float = 0.0
    stereo_preserve_loss_weight: float = 0.0
    target_delta_active_loss_weight: float = 0.0
    forensic_source_decorrelation_loss_weight: float = 0.0
    forensic_source_envelope_loss_weight: float = 0.0
    forensic_source_active_energy_loss_weight: float = 0.0
    forensic_source_target_loss_weight: float = 0.0
    forensic_envelope_window_ms: float = 20.0
    forensic_source_conditioning: bool = False
    allow_weak_teacher_region_loss: bool = False
    source_skip_init_from_target: bool = False
    curriculum_target_boost: float = 1.0
    curriculum_hard_fraction: float = 0.0
    curriculum_hard_loss_weight: float = 1.0
    max_delta: float = 0.5
    seed: int = 20260624
    device: str = "cuda"


def issue_index(issue_type: str) -> int:
    normalized = str(issue_type or "").strip().lower()
    if normalized not in ISSUE_TYPES:
        normalized = "unknown"
    return ISSUE_TYPES.index(normalized)


class StemRegionRepairNet(nn.Module):
    """Small stereo refiner that predicts a local delta for a bad stem."""

    def __init__(
        self,
        *,
        hidden_channels: int = 32,
        kernel_size: int = 31,
        dilated_layers: int = 0,
        max_delta: float = 0.5,
        input_channels: int = 6,
        issue_count: int = len(ISSUE_TYPES),
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.dilated_layers = int(dilated_layers)
        self.max_delta = float(max_delta)
        self.input_channels = int(input_channels)
        self.issue_count = int(issue_count)
        padding = kernel_size // 2
        output = nn.Conv1d(hidden_channels, 2, kernel_size, padding=padding)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers: list[nn.Module] = [
            nn.Conv1d(self.input_channels, hidden_channels, kernel_size, padding=padding),
            nn.GELU(),
        ]
        if self.dilated_layers <= 0:
            layers.extend(
                [
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
                    nn.GELU(),
                ]
            )
        else:
            for layer_idx in range(self.dilated_layers):
                dilation = 2 ** layer_idx
                layers.extend(
                    [
                        nn.Conv1d(
                            hidden_channels,
                            hidden_channels,
                            kernel_size,
                            padding=padding * dilation,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                    ]
                )
        layers.extend([output, nn.Tanh()])
        self.net = nn.Sequential(*layers)
        self.source_skip = nn.Conv1d(2, 2, 1, bias=False) if self.input_channels >= 8 else None
        if self.source_skip is not None:
            nn.init.zeros_(self.source_skip.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta = self.net(features) * self.max_delta
        if self.source_skip is not None:
            source_delta = torch.tanh(self.source_skip(features[:, -2:, :])) * self.max_delta
            delta = delta + source_delta
        return torch.clamp(delta, min=-self.max_delta, max=self.max_delta)


def _prepare_arrays(
    *,
    mix_audio: np.ndarray,
    bad_stem_audio: np.ndarray,
    region_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mix = ensure_stereo(mix_audio)
    bad = ensure_stereo(bad_stem_audio)
    if mix.shape != bad.shape:
        frames = min(mix.shape[0], bad.shape[0])
        mix = mix[:frames]
        bad = bad[:frames]
    mask = np.asarray(region_mask, dtype=np.float32).reshape(-1)
    if mask.shape[0] != bad.shape[0]:
        frames = min(mask.shape[0], bad.shape[0])
        mix = mix[:frames]
        bad = bad[:frames]
        mask = mask[:frames]
    if bad.shape[0] <= 0:
        raise ValueError("empty repair arrays")
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mix.astype(np.float32), bad.astype(np.float32), mask


def build_repair_features(
    *,
    mix_audio: np.ndarray,
    bad_stem_audio: np.ndarray,
    region_mask: np.ndarray,
    issue_type: str,
    device: str | torch.device,
    source_condition_audio: np.ndarray | None = None,
) -> torch.Tensor:
    mix, bad, mask = _prepare_arrays(mix_audio=mix_audio, bad_stem_audio=bad_stem_audio, region_mask=region_mask)
    source_condition = ensure_stereo(source_condition_audio) if source_condition_audio is not None else None
    if source_condition is not None:
        frames = min(mix.shape[0], bad.shape[0], mask.shape[0], source_condition.shape[0])
        mix = mix[:frames]
        bad = bad[:frames]
        mask = mask[:frames]
        source_condition = source_condition[:frames].astype(np.float32)
    issue_scalar = float(issue_index(issue_type)) / float(max(len(ISSUE_TYPES) - 1, 1))
    issue_channel = np.full_like(mask, issue_scalar, dtype=np.float32)
    channels = [
        bad[:, 0],
        bad[:, 1],
        mix[:, 0],
        mix[:, 1],
        mask,
        issue_channel,
    ]
    if source_condition is not None:
        channels.extend([source_condition[:, 0], source_condition[:, 1]])
    features = np.stack(channels, axis=0)
    return torch.from_numpy(features[None, ...]).to(device=device, dtype=torch.float32)


def train_region_repair_model_from_arrays(
    *,
    mix_audio: np.ndarray,
    bad_stem_audio: np.ndarray,
    target_stem_audio: np.ndarray,
    sibling_stem_audio: np.ndarray | None = None,
    target_mix_audio: np.ndarray | None = None,
    stem_sum_before_audio: np.ndarray | None = None,
    forensic_source_audio: np.ndarray | None = None,
    region_mask: np.ndarray,
    sample_rate: int,
    issue_type: str,
    config: StemRegionRepairTrainConfig | None = None,
) -> tuple[StemRegionRepairNet, dict[str, Any]]:
    config = config or StemRegionRepairTrainConfig()
    mix, bad, mask = _prepare_arrays(mix_audio=mix_audio, bad_stem_audio=bad_stem_audio, region_mask=region_mask)
    target = ensure_stereo(target_stem_audio)
    sibling = ensure_stereo(sibling_stem_audio) if sibling_stem_audio is not None else None
    target_mix = ensure_stereo(target_mix_audio) if target_mix_audio is not None else None
    stem_sum_before = ensure_stereo(stem_sum_before_audio) if stem_sum_before_audio is not None else None
    forensic_source = ensure_stereo(forensic_source_audio) if forensic_source_audio is not None else None
    frame_candidates = [target.shape[0], bad.shape[0]]
    if sibling is not None:
        frame_candidates.append(sibling.shape[0])
    if target_mix is not None:
        frame_candidates.append(target_mix.shape[0])
    if stem_sum_before is not None:
        frame_candidates.append(stem_sum_before.shape[0])
    if forensic_source is not None:
        frame_candidates.append(forensic_source.shape[0])
    frames = min(frame_candidates)
    mix = mix[:frames]
    bad = bad[:frames]
    target = target[:frames]
    if sibling is not None:
        sibling = sibling[:frames]
    if target_mix is None:
        target_mix = mix[:frames]
    else:
        target_mix = target_mix[:frames]
    if stem_sum_before is not None:
        stem_sum_before = stem_sum_before[:frames]
    if forensic_source is not None:
        forensic_source = forensic_source[:frames]
    mask = mask[:frames]
    if float(np.max(mask)) <= 0.0:
        raise ValueError("region_mask must include at least one active frame")
    mix_conservation_enabled = (
        sibling is not None
        and target_mix is not None
        and float(config.mix_conservation_loss_weight) > 0.0
    )
    stem_sum_enabled = (
        stem_sum_before is not None
        and target_mix is not None
        and float(config.stem_sum_loss_weight) > 0.0
    )
    stereo_preserve_enabled = float(config.stereo_preserve_loss_weight) > 0.0
    target_delta_active_enabled = float(config.target_delta_active_loss_weight) > 0.0
    forensic_source_decorrelation_enabled = (
        forensic_source is not None
        and float(config.forensic_source_decorrelation_loss_weight) > 0.0
    )
    forensic_source_envelope_enabled = (
        forensic_source is not None
        and float(config.forensic_source_envelope_loss_weight) > 0.0
    )
    forensic_source_active_energy_enabled = (
        forensic_source is not None
        and float(config.forensic_source_active_energy_loss_weight) > 0.0
    )
    forensic_source_target_enabled = (
        forensic_source is not None
        and float(config.forensic_source_target_loss_weight) > 0.0
    )
    source_condition = forensic_source if bool(config.forensic_source_conditioning) and forensic_source is not None else None

    def _masked_source_projection_loss(
        *,
        repaired_stem: torch.Tensor,
        source_stem: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        repaired_mono = torch.mean(repaired_stem, dim=1, keepdim=True)
        source_mono = torch.mean(source_stem, dim=1, keepdim=True)
        repaired_hp = repaired_mono[..., 1:] - repaired_mono[..., :-1]
        source_hp = source_mono[..., 1:] - source_mono[..., :-1]
        mask_hp = torch.minimum(mask_t[..., 1:], mask_t[..., :-1])
        repaired_hp = repaired_hp * mask_hp
        source_hp = source_hp * mask_hp
        source_energy = torch.sum(torch.square(source_hp)) + EPSILON
        projection_coeff = torch.sum(repaired_hp * source_hp) / source_energy
        projected_source = projection_coeff * source_hp
        active = torch.clamp(torch.sum(mask_hp), min=1.0)
        return torch.sum(torch.square(projected_source)) / active

    def _weighted_projection_energy(tested: torch.Tensor, source: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        active = torch.clamp(torch.sum(weights), min=1.0)
        tested_mean = torch.sum(tested * weights) / active
        source_mean = torch.sum(source * weights) / active
        tested_centered = (tested - tested_mean) * weights
        source_centered = (source - source_mean) * weights
        source_energy = torch.sum(torch.square(source_centered)) + EPSILON
        coeff = torch.sum(tested_centered * source_centered) / source_energy
        projected = coeff * source_centered
        return torch.sum(torch.square(projected)) / active

    def _masked_source_envelope_loss(
        *,
        repaired_stem: torch.Tensor,
        source_stem: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        repaired_mono = torch.mean(repaired_stem, dim=1, keepdim=True)
        source_mono = torch.mean(source_stem, dim=1, keepdim=True)
        repaired_hp = repaired_mono[..., 1:] - repaired_mono[..., :-1]
        source_hp = source_mono[..., 1:] - source_mono[..., :-1]
        mask_hp = torch.minimum(mask_t[..., 1:], mask_t[..., :-1])
        length = int(repaired_hp.shape[-1])
        if length <= 2:
            return torch.zeros((), device=repaired_stem.device)
        window = max(3, int(round((float(sample_rate) * float(config.forensic_envelope_window_ms)) / 1000.0)))
        if window % 2 == 0:
            window += 1
        window = min(window, length if length % 2 == 1 else length - 1)
        if window < 3:
            window = 3
        padding = window // 2
        repaired_env = F.avg_pool1d(torch.abs(repaired_hp) * mask_hp, kernel_size=window, stride=1, padding=padding)
        source_env = F.avg_pool1d(torch.abs(source_hp) * mask_hp, kernel_size=window, stride=1, padding=padding)
        env_weights = F.avg_pool1d(mask_hp, kernel_size=window, stride=1, padding=padding).clamp(0.0, 1.0)
        env_loss = _weighted_projection_energy(repaired_env, source_env, env_weights)
        repaired_flux = repaired_env[..., 1:] - repaired_env[..., :-1]
        source_flux = source_env[..., 1:] - source_env[..., :-1]
        flux_weights = torch.minimum(env_weights[..., 1:], env_weights[..., :-1])
        flux_loss = _weighted_projection_energy(repaired_flux, source_flux, flux_weights)
        return env_loss + flux_loss

    def _masked_source_active_energy_loss(
        *,
        repaired_stem: torch.Tensor,
        source_stem: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        repaired_mono = torch.mean(repaired_stem, dim=1, keepdim=True)
        source_mono = torch.mean(source_stem, dim=1, keepdim=True)
        repaired_hp = repaired_mono[..., 1:] - repaired_mono[..., :-1]
        source_hp = source_mono[..., 1:] - source_mono[..., :-1]
        mask_hp = torch.minimum(mask_t[..., 1:], mask_t[..., :-1])
        source_activity = torch.abs(source_hp)
        peak = torch.amax(source_activity).clamp_min(EPSILON)
        weights = torch.square(source_activity / peak) * mask_hp
        active = torch.clamp(torch.sum(weights), min=1.0)
        return torch.sum(torch.square(repaired_hp) * weights) / active

    def _masked_target_delta_active_loss(
        *,
        pred_delta: torch.Tensor,
        target_delta: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        activity = torch.mean(torch.abs(target_delta), dim=1, keepdim=True)
        peak = torch.amax(activity).clamp_min(EPSILON)
        weights = torch.square(activity / peak) * mask_t
        active = torch.clamp(torch.sum(weights) * float(pred_delta.shape[1]), min=1.0)
        return torch.sum(torch.square(pred_delta - target_delta) * weights) / active

    def _masked_source_target_delta_loss(
        *,
        pred_delta: torch.Tensor,
        target_delta: torch.Tensor,
        source_stem: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        source_mono = torch.mean(source_stem, dim=1, keepdim=True)
        source_activity = torch.abs(source_mono[..., 1:] - source_mono[..., :-1])
        left = F.pad(source_activity, (1, 0))
        right = F.pad(source_activity, (0, 1))
        frame_activity = torch.maximum(left, right)
        peak = torch.amax(frame_activity).clamp_min(EPSILON)
        weights = torch.square(frame_activity / peak) * mask_t
        active = torch.clamp(torch.sum(weights) * float(pred_delta.shape[1]), min=1.0)
        return torch.sum(torch.square(pred_delta - target_delta) * weights) / active

    def _train_once(attempt_config: StemRegionRepairTrainConfig) -> tuple[StemRegionRepairNet, list[float], float]:
        torch.manual_seed(int(attempt_config.seed))
        torch.set_num_threads(1)
        device = torch.device(attempt_config.device if attempt_config.device == "cpu" or torch.cuda.is_available() else "cpu")
        model = StemRegionRepairNet(
            hidden_channels=attempt_config.hidden_channels,
            kernel_size=attempt_config.kernel_size,
            dilated_layers=attempt_config.dilated_layers,
            max_delta=attempt_config.max_delta,
            input_channels=8 if source_condition is not None else 6,
        ).to(device)

        features = build_repair_features(
            mix_audio=mix,
            bad_stem_audio=bad,
            region_mask=mask,
            issue_type=issue_type,
            device=device,
            source_condition_audio=source_condition,
        )
        target_delta = torch.from_numpy((target - bad).T[None, ...].astype(np.float32)).to(device)
        target_t = torch.from_numpy(target.T[None, ...].astype(np.float32)).to(device)
        bad_t = torch.from_numpy(bad.T[None, ...].astype(np.float32)).to(device)
        sibling_t = (
            torch.from_numpy(sibling.T[None, ...].astype(np.float32)).to(device) if mix_conservation_enabled else None
        )
        target_mix_t = (
            torch.from_numpy(target_mix.T[None, ...].astype(np.float32)).to(device)
            if (mix_conservation_enabled or stem_sum_enabled)
            else None
        )
        stem_sum_before_t = (
            torch.from_numpy(stem_sum_before.T[None, ...].astype(np.float32)).to(device) if stem_sum_enabled else None
        )
        forensic_source_t = (
            torch.from_numpy(forensic_source.T[None, ...].astype(np.float32)).to(device)
            if (
                forensic_source_decorrelation_enabled
                or forensic_source_envelope_enabled
                or forensic_source_active_energy_enabled
                or forensic_source_target_enabled
            )
            else None
        )
        mask_t = torch.from_numpy(mask[None, None, :].astype(np.float32)).to(device)
        outside_t = 1.0 - mask_t
        if (
            bool(attempt_config.source_skip_init_from_target)
            and model.source_skip is not None
            and forensic_source_t is not None
        ):
            with torch.no_grad():
                source = forensic_source_t
                weights = mask_t.expand_as(source)
                denom = torch.sum(torch.square(source) * weights, dim=2).clamp_min(EPSILON)
                coeff = torch.sum(target_delta * source * weights, dim=2) / denom
                model.source_skip.weight.zero_()
                scale = max(float(attempt_config.max_delta), EPSILON)
                for channel_idx in range(min(2, int(coeff.shape[1]))):
                    model.source_skip.weight[channel_idx, channel_idx, 0] = coeff[0, channel_idx] / scale
        optimizer = torch.optim.Adam(model.parameters(), lr=float(attempt_config.learning_rate))

        history: list[float] = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs = max(1, int(attempt_config.epochs))
        curriculum_target_boost = max(1.0, float(attempt_config.curriculum_target_boost))
        curriculum_hard_fraction = max(0.0, min(1.0, float(attempt_config.curriculum_hard_fraction)))
        curriculum_hard_loss_weight = max(1.0, float(attempt_config.curriculum_hard_loss_weight))
        hard_epochs = int(round(float(epochs) * curriculum_hard_fraction))
        hard_epochs = max(1, hard_epochs) if curriculum_target_boost > 1.0 and curriculum_hard_fraction > 0.0 else 0
        for epoch_index in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred_delta = model(features)
            if hard_epochs > 0 and epoch_index < hard_epochs:
                hard_progress = float(epoch_index) / float(max(hard_epochs - 1, 1))
                target_scale = curriculum_target_boost - ((curriculum_target_boost - 1.0) * hard_progress)
                region_weight_scale = curriculum_hard_loss_weight - ((curriculum_hard_loss_weight - 1.0) * hard_progress)
            else:
                target_scale = 1.0
                region_weight_scale = 1.0
            active_target_delta = torch.clamp(
                target_delta * float(target_scale),
                min=-float(attempt_config.max_delta),
                max=float(attempt_config.max_delta),
            )
            region_loss = torch.mean(torch.square((pred_delta - active_target_delta) * mask_t))
            selection_region_loss = torch.mean(torch.square((pred_delta - target_delta) * mask_t))
            outside_loss = torch.mean(torch.square(pred_delta * outside_t))
            mix_conservation_loss = torch.zeros((), device=device)
            if sibling_t is not None and target_mix_t is not None:
                repaired_pair = bad_t + pred_delta + sibling_t
                mix_conservation_loss = torch.mean(torch.square((repaired_pair - target_mix_t) * mask_t))
            sibling_preserve_loss = torch.zeros((), device=device)
            if float(attempt_config.sibling_preserve_loss_weight) > 0.0:
                sibling_preserve_loss = torch.mean(torch.square(pred_delta * outside_t))
            stem_sum_loss = torch.zeros((), device=device)
            if stem_sum_before_t is not None and target_mix_t is not None:
                repaired_sum = stem_sum_before_t + pred_delta
                stem_sum_loss = torch.mean(torch.square((repaired_sum - target_mix_t) * mask_t))
            stereo_preserve_loss = torch.zeros((), device=device)
            if float(attempt_config.stereo_preserve_loss_weight) > 0.0:
                repaired_stem = bad_t + pred_delta
                repaired_side = (repaired_stem[:, 0:1, :] - repaired_stem[:, 1:2, :]) * 0.5
                target_side = (target_t[:, 0:1, :] - target_t[:, 1:2, :]) * 0.5
                stereo_preserve_loss = torch.mean(torch.square((repaired_side - target_side) * mask_t))
            forensic_source_decorrelation_loss = torch.zeros((), device=device)
            if forensic_source_t is not None:
                forensic_source_decorrelation_loss = _masked_source_projection_loss(
                    repaired_stem=bad_t + pred_delta,
                    source_stem=forensic_source_t,
                    mask_t=mask_t,
                )
            forensic_source_envelope_loss = torch.zeros((), device=device)
            if forensic_source_t is not None and float(attempt_config.forensic_source_envelope_loss_weight) > 0.0:
                forensic_source_envelope_loss = _masked_source_envelope_loss(
                    repaired_stem=bad_t + pred_delta,
                    source_stem=forensic_source_t,
                    mask_t=mask_t,
                )
            forensic_source_active_energy_loss = torch.zeros((), device=device)
            if forensic_source_t is not None and float(attempt_config.forensic_source_active_energy_loss_weight) > 0.0:
                forensic_source_active_energy_loss = _masked_source_active_energy_loss(
                    repaired_stem=bad_t + pred_delta,
                    source_stem=forensic_source_t,
                    mask_t=mask_t,
                )
            target_delta_active_loss = torch.zeros((), device=device)
            if float(attempt_config.target_delta_active_loss_weight) > 0.0:
                target_delta_active_loss = _masked_target_delta_active_loss(
                    pred_delta=pred_delta,
                    target_delta=target_delta,
                    mask_t=mask_t,
                )
            forensic_source_target_loss = torch.zeros((), device=device)
            if forensic_source_t is not None and float(attempt_config.forensic_source_target_loss_weight) > 0.0:
                forensic_source_target_loss = _masked_source_target_delta_loss(
                    pred_delta=pred_delta,
                    target_delta=target_delta,
                    source_stem=forensic_source_t,
                    mask_t=mask_t,
                )
            loss = (
                float(attempt_config.region_loss_weight) * float(region_weight_scale) * region_loss
                + float(attempt_config.outside_loss_weight) * outside_loss
            )
            loss = (
                loss
                + float(attempt_config.mix_conservation_loss_weight) * mix_conservation_loss
                + float(attempt_config.sibling_preserve_loss_weight) * sibling_preserve_loss
                + float(attempt_config.stem_sum_loss_weight) * stem_sum_loss
                + float(attempt_config.stereo_preserve_loss_weight) * stereo_preserve_loss
                + float(attempt_config.target_delta_active_loss_weight) * target_delta_active_loss
                + float(attempt_config.forensic_source_decorrelation_loss_weight) * forensic_source_decorrelation_loss
                + float(attempt_config.forensic_source_envelope_loss_weight) * forensic_source_envelope_loss
                + float(attempt_config.forensic_source_active_energy_loss_weight) * forensic_source_active_energy_loss
                + float(attempt_config.forensic_source_target_loss_weight) * forensic_source_target_loss
            )
            selection_loss_tensor = (
                float(attempt_config.region_loss_weight) * selection_region_loss
                + float(attempt_config.outside_loss_weight) * outside_loss
                + float(attempt_config.mix_conservation_loss_weight) * mix_conservation_loss
                + float(attempt_config.sibling_preserve_loss_weight) * sibling_preserve_loss
                + float(attempt_config.stem_sum_loss_weight) * stem_sum_loss
                + float(attempt_config.stereo_preserve_loss_weight) * stereo_preserve_loss
                + float(attempt_config.target_delta_active_loss_weight) * target_delta_active_loss
                + float(attempt_config.forensic_source_decorrelation_loss_weight) * forensic_source_decorrelation_loss
                + float(attempt_config.forensic_source_envelope_loss_weight) * forensic_source_envelope_loss
                + float(attempt_config.forensic_source_active_energy_loss_weight) * forensic_source_active_energy_loss
                + float(attempt_config.forensic_source_target_loss_weight) * forensic_source_target_loss
            )
            selection_loss = float(selection_loss_tensor.detach().cpu().item())
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            history.append(loss_value)

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model.cpu(), history, best_loss if history else 0.0

    attempts = [config]
    if float(config.learning_rate) > 0.003 or int(config.epochs) < 200:
        attempts.append(
            replace(
                config,
                learning_rate=min(float(config.learning_rate), 0.003),
                epochs=max(int(config.epochs), 200),
            )
        )
    attempt_results: list[tuple[StemRegionRepairTrainConfig, StemRegionRepairNet, list[float], float]] = []
    for attempt_config in attempts:
        attempt_model, attempt_history, attempt_best_loss = _train_once(attempt_config)
        attempt_results.append((attempt_config, attempt_model, attempt_history, attempt_best_loss))

    selected_config, model, history, best_loss = min(attempt_results, key=lambda item: item[3])
    return model.cpu(), {
        "schema": "stem_region_repair_training_history.v1",
        "sample_rate": int(sample_rate),
        "issue_type": str(issue_type),
        "epochs": int(selected_config.epochs),
        "initial_loss": float(history[0]) if history else 0.0,
        "final_loss": float(best_loss) if history else 0.0,
        "loss_history": [round(value, 8) for value in history],
        "config": asdict(selected_config),
        "loss_terms_enabled": {
            "region": float(selected_config.region_loss_weight) > 0.0,
            "outside": float(selected_config.outside_loss_weight) > 0.0,
            "mix_conservation": mix_conservation_enabled,
            "sibling_preserve": float(selected_config.sibling_preserve_loss_weight) > 0.0,
            "stem_sum": stem_sum_enabled,
            "stereo_preserve": stereo_preserve_enabled,
            "target_delta_active": target_delta_active_enabled,
            "forensic_source_decorrelation": forensic_source_decorrelation_enabled,
            "forensic_source_envelope": forensic_source_envelope_enabled,
            "forensic_source_active_energy": forensic_source_active_energy_enabled,
            "forensic_source_target": forensic_source_target_enabled,
            "forensic_source_conditioning": source_condition is not None,
            "source_skip_init_from_target": bool(selected_config.source_skip_init_from_target),
        },
        "attempts": [
            {
                "learning_rate": float(attempt_config.learning_rate),
                "epochs": int(attempt_config.epochs),
                "initial_loss": float(attempt_history[0]) if attempt_history else 0.0,
                "final_loss": float(attempt_history[-1]) if attempt_history else 0.0,
                "best_loss": float(attempt_best_loss),
                "selected": attempt_config == selected_config,
            }
            for attempt_config, _attempt_model, attempt_history, attempt_best_loss in attempt_results
        ],
    }


def apply_region_repair_model(
    *,
    model: StemRegionRepairNet,
    mix_audio: np.ndarray,
    bad_stem_audio: np.ndarray,
    region_mask: np.ndarray,
    sample_rate: int,
    issue_type: str,
    source_condition_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mix, bad, mask = _prepare_arrays(mix_audio=mix_audio, bad_stem_audio=bad_stem_audio, region_mask=region_mask)
    source_condition = ensure_stereo(source_condition_audio) if source_condition_audio is not None else None
    if source_condition is not None:
        frames = min(mix.shape[0], bad.shape[0], mask.shape[0], source_condition.shape[0])
        mix = mix[:frames]
        bad = bad[:frames]
        mask = mask[:frames]
        source_condition = source_condition[:frames].astype(np.float32)
    if int(getattr(model, "input_channels", 6)) > 6 and source_condition is None:
        raise ValueError("source-conditioned repair checkpoint requires source_condition_audio")
    model = model.cpu().eval()
    with torch.no_grad():
        features = build_repair_features(
            mix_audio=mix,
            bad_stem_audio=bad,
            region_mask=mask,
            issue_type=issue_type,
            device="cpu",
            source_condition_audio=source_condition,
        )
        delta = model(features).detach().cpu().numpy()[0].T.astype(np.float32)
    masked_delta = delta * mask[:, None]
    repaired = np.clip(bad + masked_delta, -1.0, 1.0).astype(np.float32)
    changed = repaired - bad
    region_energy = float(np.sqrt(np.mean(np.square(changed[mask > 1.0e-4])))) if np.any(mask > 1.0e-4) else 0.0
    outside_energy = float(np.sqrt(np.mean(np.square(changed[mask <= 1.0e-4])))) if np.any(mask <= 1.0e-4) else 0.0
    return repaired, {
        "schema": "stem_region_repair_apply_diagnostics.v1",
        "sample_rate": int(sample_rate),
        "issue_type": str(issue_type),
        "changed_region_ratio": region_energy,
        "changed_outside_ratio": outside_energy,
            "active_frame_count": int(np.sum(mask > 1.0e-4)),
            "source_conditioned": source_condition is not None,
        }


def save_region_repair_checkpoint(
    *,
    model: StemRegionRepairNet,
    path: Path,
    training_history: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "stem_region_repair_checkpoint.v1",
            "state_dict": model.state_dict(),
            "model_config": {
                "hidden_channels": model.hidden_channels,
                "kernel_size": model.kernel_size,
                "dilated_layers": model.dilated_layers,
                "max_delta": model.max_delta,
                "input_channels": model.input_channels,
                "issue_count": model.issue_count,
            },
            "training_history": training_history,
        },
        str(path),
    )
    return path


def load_region_repair_checkpoint(path: Path) -> tuple[StemRegionRepairNet, dict[str, Any]]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch builds
        payload = torch.load(str(path), map_location="cpu")
    config = payload.get("model_config") or {}
    model = StemRegionRepairNet(
        hidden_channels=int(config.get("hidden_channels", 32)),
        kernel_size=int(config.get("kernel_size", 31)),
        dilated_layers=int(config.get("dilated_layers", 0)),
        max_delta=float(config.get("max_delta", 0.5)),
        input_channels=int(config.get("input_channels", 6)),
        issue_count=int(config.get("issue_count", len(ISSUE_TYPES))),
    )
    missing_keys, unexpected_keys = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {"source_skip.weight"} if model.source_skip is not None else set()
    unexpected = [key for key in unexpected_keys]
    missing = [key for key in missing_keys if key not in allowed_missing]
    if unexpected or missing:
        raise RuntimeError(
            f"repair checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, dict(payload)


def write_training_history(path: Path, history: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    return path
