from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data"))
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "/app/models"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
LYRICS_DEVICE = os.getenv("LYRICS_DEVICE", "auto")
LYRICS_COMPUTE_TYPE = os.getenv("LYRICS_COMPUTE_TYPE", "auto")
LYRICS_BATCH_SIZE = int(os.getenv("LYRICS_BATCH_SIZE", "8"))
LYRICS_WHISPERX_MODEL = os.getenv("LYRICS_WHISPERX_MODEL", "large-v3")
LYRICS_ALIGN_MODEL = os.getenv("LYRICS_ALIGN_MODEL", "large-v3")

DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)


def choose_device() -> str:
    if LYRICS_DEVICE != "auto":
        return LYRICS_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def choose_compute_type(device: str) -> str:
    if LYRICS_COMPUTE_TYPE != "auto":
        return LYRICS_COMPUTE_TYPE
    return "float16" if device.startswith("cuda") else "int8"
