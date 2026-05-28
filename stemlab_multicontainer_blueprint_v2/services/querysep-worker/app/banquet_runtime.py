from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

import gc
import torch
import librosa
import numpy as np
import soundfile as sf
import torchaudio as ta
from torch import nn

from app.utils import (
    DATA_ROOT,
    QUERYSEP_BANQUET_BATCH_SIZE,
    QUERYSEP_BANQUET_CKPT_MD5,
    QUERYSEP_BANQUET_CKPT_PATH,
    QUERYSEP_BANQUET_CKPT_URL,
    QUERYSEP_BANQUET_CONFIG_ROOT,
    QUERYSEP_BANQUET_DOWNLOAD_ON_START,
    QUERYSEP_BANQUET_ENABLED,
    QUERYSEP_BANQUET_EXPT_CONFIG,
    QUERYSEP_BANQUET_QUERY_LENGTH_SECONDS,
    QUERYSEP_BANQUET_VENDOR_ROOT,
    QUERYSEP_CACHE_ROOT,
    QUERYSEP_IDLE_UNLOAD_SECONDS,
    QUERYSEP_RUNTIME_DEVICE,
)

MODEL_FS = 44100


def _normalize_device(raw_device: str | None) -> str:
    if raw_device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    token = raw_device.strip().lower()
    if not token or token == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return raw_device


@dataclass
class BanquetRuntime:
    ckpt_path: str
    config_path: str
    device: str
    system: Any
    loaded_at: float
    load_seconds: float
    last_used_at: float


class _NoopLoss(nn.Module):
    def forward(self, batch: Any) -> dict[str, float]:
        return {}


class _NoopMetrics:
    def get_mode(self, mode: Any) -> dict[str, Any]:
        return {}


_LOCK = RLock()
_RUNTIME: BanquetRuntime | None = None


def _cache_dir() -> Path:
    return QUERYSEP_CACHE_ROOT / "banquet"


def _ensure_vendor_on_path() -> None:
    vendor_root = str(QUERYSEP_BANQUET_VENDOR_ROOT)
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)
    os.environ.setdefault("CONFIG_ROOT", str(QUERYSEP_BANQUET_CONFIG_ROOT))
    os.environ.setdefault("DATA_ROOT", str(DATA_ROOT))


def banquet_ready_state() -> dict[str, object]:
    return {
        "enabled": QUERYSEP_BANQUET_ENABLED,
        "checkpoint_path": str(QUERYSEP_BANQUET_CKPT_PATH),
        "checkpoint_exists": QUERYSEP_BANQUET_CKPT_PATH.exists(),
        "checkpoint_size": QUERYSEP_BANQUET_CKPT_PATH.stat().st_size if QUERYSEP_BANQUET_CKPT_PATH.exists() else 0,
        "config_path": str(QUERYSEP_BANQUET_EXPT_CONFIG),
        "vendor_root": str(QUERYSEP_BANQUET_VENDOR_ROOT),
    }


def cache_snapshot() -> dict[str, object]:
    with _LOCK:
        snapshot = banquet_ready_state()
        _maybe_unload_idle_runtime_locked()
        if _RUNTIME is None:
            snapshot["loaded"] = False
            return snapshot
        snapshot.update(
            {
                "loaded": True,
                "device": _RUNTIME.device,
                "loaded_at": _RUNTIME.loaded_at,
                "load_seconds": round(_RUNTIME.load_seconds, 3),
                "last_used_at": _RUNTIME.last_used_at,
            }
        )
        return snapshot


def _download_checkpoint_if_needed() -> None:
    if QUERYSEP_BANQUET_CKPT_PATH.exists():
        return
    if not QUERYSEP_BANQUET_DOWNLOAD_ON_START:
        raise FileNotFoundError(
            f"Banquet checkpoint missing at {QUERYSEP_BANQUET_CKPT_PATH} and auto-download is disabled."
        )

    tmp_path = QUERYSEP_BANQUET_CKPT_PATH.with_suffix(f"{QUERYSEP_BANQUET_CKPT_PATH.suffix}.part")
    md5 = hashlib.md5()
    with urllib.request.urlopen(QUERYSEP_BANQUET_CKPT_URL) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            md5.update(chunk)

    digest = md5.hexdigest()
    if QUERYSEP_BANQUET_CKPT_MD5 and digest.lower() != QUERYSEP_BANQUET_CKPT_MD5.lower():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded Banquet checkpoint checksum mismatch: "
            f"expected {QUERYSEP_BANQUET_CKPT_MD5}, got {digest}"
        )

    tmp_path.replace(QUERYSEP_BANQUET_CKPT_PATH)


def _load_config():
    _ensure_vendor_on_path()
    from omegaconf import OmegaConf

    raw = OmegaConf.load(str(QUERYSEP_BANQUET_EXPT_CONFIG))
    merged: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value.endswith(".yml"):
            merged[key] = OmegaConf.load(os.path.expandvars(value))
        else:
            merged[key] = value
    config = OmegaConf.merge(merged)

    if getattr(config.data, "inference_kwargs", None) is not None:
        config.data.inference_kwargs.batch_size = QUERYSEP_BANQUET_BATCH_SIZE
    if getattr(config.model, "kwargs", None) is not None and config.model.kwargs.get("pretrain_encoder"):
        # Inference is fully restored from the checkpoint; local training-time paths are not valid here.
        config.model.kwargs.pretrain_encoder = None
    return config


def _build_system(config, device: str):
    _ensure_vendor_on_path()

    from core.models.ebase import EndToEndLightningSystem
    from core.models.e2e.bandit.bandit import Bandit, PasstFiLMConditionedBandit

    model_name = config.model.cls
    model_kwargs = dict(config.model.get("kwargs", {}))
    model_cls = {
        "Bandit": Bandit,
        "PasstFiLMConditionedBandit": PasstFiLMConditionedBandit,
    }.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unsupported Banquet model class: {model_name}")

    model = model_cls(**model_kwargs)
    optimization_bundle = SimpleNamespace(
        optimizer=SimpleNamespace(cls=torch.optim.Adam, kwargs={"lr": 1e-4}),
        scheduler=None,
    )
    system = EndToEndLightningSystem(
        model=model,
        loss_handler=_NoopLoss(),
        metrics=_NoopMetrics(),
        augmentation_handler=nn.Identity(),
        inference_handler=config.data.inference_kwargs,
        optimization_bundle=optimization_bundle,
        fast_run=False,
        batch_size=config.data.batch_size,
        effective_batch_size=config.data.get("effective_batch_size", None),
    )

    try:
        checkpoint = torch.load(str(QUERYSEP_BANQUET_CKPT_PATH), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(QUERYSEP_BANQUET_CKPT_PATH), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_result = system.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Banquet checkpoint failed strict load: "
            f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
        )

    system = system.to(torch.device(device))
    system.eval()
    return system


def _load_runtime() -> BanquetRuntime:
    if not QUERYSEP_BANQUET_ENABLED:
        raise RuntimeError("Banquet backend is disabled.")
    if not QUERYSEP_BANQUET_VENDOR_ROOT.exists():
        raise FileNotFoundError(f"Banquet vendor root not found: {QUERYSEP_BANQUET_VENDOR_ROOT}")
    if not QUERYSEP_BANQUET_EXPT_CONFIG.exists():
        raise FileNotFoundError(f"Banquet config not found: {QUERYSEP_BANQUET_EXPT_CONFIG}")

    _download_checkpoint_if_needed()
    device = _normalize_device(QUERYSEP_RUNTIME_DEVICE)
    config = _load_config()
    start = time.perf_counter()
    system = _build_system(config, device=device)
    return BanquetRuntime(
        ckpt_path=str(QUERYSEP_BANQUET_CKPT_PATH),
        config_path=str(QUERYSEP_BANQUET_EXPT_CONFIG),
        device=device,
        system=system,
        loaded_at=time.time(),
        load_seconds=time.perf_counter() - start,
        last_used_at=time.time(),
    )


def ensure_runtime() -> tuple[BanquetRuntime, bool]:
    global _RUNTIME
    with _LOCK:
        _maybe_unload_idle_runtime_locked()
        if _RUNTIME is not None:
            _RUNTIME.last_used_at = time.time()
            return _RUNTIME, True
        _RUNTIME = _load_runtime()
        return _RUNTIME, False


def _evict_runtime_locked() -> None:
    global _RUNTIME
    runtime = _RUNTIME
    _RUNTIME = None
    if runtime is None:
        return
    try:
        runtime.system.cpu()
    except Exception:
        pass
    try:
        del runtime.system
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _maybe_unload_idle_runtime_locked() -> bool:
    if _RUNTIME is None:
        return False
    if QUERYSEP_IDLE_UNLOAD_SECONDS <= 0:
        return False
    idle_seconds = time.time() - _RUNTIME.last_used_at
    if idle_seconds < QUERYSEP_IDLE_UNLOAD_SECONDS:
        return False
    _evict_runtime_locked()
    return True


def maybe_unload_idle_runtime(force: bool = False) -> bool:
    with _LOCK:
        if force:
            had_runtime = _RUNTIME is not None
            _evict_runtime_locked()
            return had_runtime
        return _maybe_unload_idle_runtime_locked()


def _ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]
    return audio.to(torch.float32)


def _load_audio_file(path: Path) -> tuple[torch.Tensor, int]:
    try:
        audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32).T.copy())
        return _ensure_stereo(audio), int(sr)
    except Exception:
        audio, sr = librosa.load(str(path), sr=None, mono=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        return _ensure_stereo(torch.from_numpy(audio)), int(sr)


def _fit_query_length(audio: torch.Tensor) -> torch.Tensor:
    target_samples = int(round(MODEL_FS * QUERYSEP_BANQUET_QUERY_LENGTH_SECONDS))
    if audio.shape[1] >= target_samples:
        return audio[:, :target_samples]
    repeats = (target_samples + audio.shape[1] - 1) // max(1, audio.shape[1])
    return audio.repeat(1, repeats)[:, :target_samples]


def run_banquet_query(
    mixture_path: Path,
    query_path: Path,
    output_path: Path,
    *,
    stem_name: str = "target",
) -> dict[str, object]:
    runtime, cache_hit = ensure_runtime()

    mixture, mixture_sr = _load_audio_file(mixture_path)
    query, query_sr = _load_audio_file(query_path)

    if mixture_sr != MODEL_FS:
        mixture = ta.functional.resample(mixture, orig_freq=mixture_sr, new_freq=MODEL_FS)
    if query_sr != MODEL_FS:
        query = ta.functional.resample(query, orig_freq=query_sr, new_freq=MODEL_FS)
    query = _fit_query_length(query)

    batch = {
        "mixture": {"audio": mixture.unsqueeze(0).to(runtime.device)},
        "query": {"audio": query.unsqueeze(0).to(runtime.device)},
        "metadata": {"stem": [stem_name]},
        "estimates": {},
    }
    with torch.inference_mode():
        out = runtime.system.chunked_inference(batch)
        inference_mode = "chunked"
        estimate_key = stem_name

    estimate = out["estimates"][estimate_key]["audio"].squeeze(0).detach().cpu()
    if mixture_sr != MODEL_FS:
        estimate = ta.functional.resample(estimate, orig_freq=MODEL_FS, new_freq=mixture_sr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        estimate.transpose(0, 1).contiguous().cpu().numpy().astype(np.float32, copy=False),
        mixture_sr,
        subtype="FLOAT",
    )
    return {
        "cache_hit": cache_hit,
        "device": runtime.device,
        "checkpoint_path": runtime.ckpt_path,
        "config_path": runtime.config_path,
        "model_fs": MODEL_FS,
        "inference_mode": inference_mode,
        "chunk_padding_mode": getattr(runtime.system, "_last_chunk_padding_mode", "unknown"),
    }


def prefetch_runtime() -> dict[str, object]:
    runtime, cache_hit = ensure_runtime()
    marker = {
        "checkpoint_path": runtime.ckpt_path,
        "config_path": runtime.config_path,
        "device": runtime.device,
        "cache_hit": cache_hit,
        "loaded_at": runtime.loaded_at,
        "load_seconds": round(runtime.load_seconds, 3),
        "last_used_at": runtime.last_used_at,
    }
    marker_path = _cache_dir() / "prefetch_state.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return marker
