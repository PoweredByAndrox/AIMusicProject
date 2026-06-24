"""Tiny trainable stereo patch repair model for stem-region defects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from tools.stem_region_repair_common import ISSUE_TYPES, ensure_stereo
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import ISSUE_TYPES, ensure_stereo


EPSILON = 1.0e-8


@dataclass(frozen=True)
class StemRegionRepairTrainConfig:
    hidden_channels: int = 32
    kernel_size: int = 31
    epochs: int = 120
    learning_rate: float = 0.003
    region_loss_weight: float = 1.0
    outside_loss_weight: float = 0.2
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
        max_delta: float = 0.5,
        issue_count: int = len(ISSUE_TYPES),
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.max_delta = float(max_delta)
        self.issue_count = int(issue_count)
        padding = kernel_size // 2
        # channels: bad L/R, mix L/R, region mask, issue scalar
        output = nn.Conv1d(hidden_channels, 2, kernel_size, padding=padding)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.net = nn.Sequential(
            nn.Conv1d(6, hidden_channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.GELU(),
            output,
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features) * self.max_delta


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
) -> torch.Tensor:
    mix, bad, mask = _prepare_arrays(mix_audio=mix_audio, bad_stem_audio=bad_stem_audio, region_mask=region_mask)
    issue_scalar = float(issue_index(issue_type)) / float(max(len(ISSUE_TYPES) - 1, 1))
    issue_channel = np.full_like(mask, issue_scalar, dtype=np.float32)
    features = np.stack(
        [
            bad[:, 0],
            bad[:, 1],
            mix[:, 0],
            mix[:, 1],
            mask,
            issue_channel,
        ],
        axis=0,
    )
    return torch.from_numpy(features[None, ...]).to(device=device, dtype=torch.float32)


def train_region_repair_model_from_arrays(
    *,
    mix_audio: np.ndarray,
    bad_stem_audio: np.ndarray,
    target_stem_audio: np.ndarray,
    region_mask: np.ndarray,
    sample_rate: int,
    issue_type: str,
    config: StemRegionRepairTrainConfig | None = None,
) -> tuple[StemRegionRepairNet, dict[str, Any]]:
    config = config or StemRegionRepairTrainConfig()
    torch.manual_seed(int(config.seed))
    torch.set_num_threads(1)

    mix, bad, mask = _prepare_arrays(mix_audio=mix_audio, bad_stem_audio=bad_stem_audio, region_mask=region_mask)
    target = ensure_stereo(target_stem_audio)
    frames = min(target.shape[0], bad.shape[0])
    mix = mix[:frames]
    bad = bad[:frames]
    target = target[:frames]
    mask = mask[:frames]
    if float(np.max(mask)) <= 0.0:
        raise ValueError("region_mask must include at least one active frame")

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = StemRegionRepairNet(
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        max_delta=config.max_delta,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))

    features = build_repair_features(
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        issue_type=issue_type,
        device=device,
    )
    target_delta = torch.from_numpy((target - bad).T[None, ...].astype(np.float32)).to(device)
    mask_t = torch.from_numpy(mask[None, None, :].astype(np.float32)).to(device)
    outside_t = 1.0 - mask_t

    history: list[float] = []
    for _ in range(int(config.epochs)):
        optimizer.zero_grad(set_to_none=True)
        pred_delta = model(features)
        region_loss = torch.mean(torch.square((pred_delta - target_delta) * mask_t))
        outside_loss = torch.mean(torch.square(pred_delta * outside_t))
        loss = float(config.region_loss_weight) * region_loss + float(config.outside_loss_weight) * outside_loss
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))

    model.eval()
    return model.cpu(), {
        "schema": "stem_region_repair_training_history.v1",
        "sample_rate": int(sample_rate),
        "issue_type": str(issue_type),
        "epochs": int(config.epochs),
        "initial_loss": float(history[0]) if history else 0.0,
        "final_loss": float(history[-1]) if history else 0.0,
        "loss_history": [round(value, 8) for value in history],
        "config": asdict(config),
    }


def apply_region_repair_model(
    *,
    model: StemRegionRepairNet,
    mix_audio: np.ndarray,
    bad_stem_audio: np.ndarray,
    region_mask: np.ndarray,
    sample_rate: int,
    issue_type: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    mix, bad, mask = _prepare_arrays(mix_audio=mix_audio, bad_stem_audio=bad_stem_audio, region_mask=region_mask)
    model = model.cpu().eval()
    with torch.no_grad():
        features = build_repair_features(
            mix_audio=mix,
            bad_stem_audio=bad,
            region_mask=mask,
            issue_type=issue_type,
            device="cpu",
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
                "max_delta": model.max_delta,
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
        max_delta=float(config.get("max_delta", 0.5)),
        issue_count=int(config.get("issue_count", len(ISSUE_TYPES))),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload)


def write_training_history(path: Path, history: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    return path
