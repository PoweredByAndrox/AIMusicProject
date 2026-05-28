from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel

from app.utils import QUERYSEP_CACHE_ROOT, QUERYSEP_IDLE_UNLOAD_SECONDS, QUERYSEP_MODEL_ID, QUERYSEP_RUNTIME_DEVICE


def _normalize_device(raw_device: str | None) -> str:
    if raw_device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    token = raw_device.strip().lower()
    if not token or token == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return raw_device


@dataclass
class QuerysepRuntime:
    model_id: str
    device: str
    processor: AutoProcessor
    model: ClapModel
    loaded_at: float
    load_seconds: float
    last_used_at: float


_LOCK = RLock()
_RUNTIME: QuerysepRuntime | None = None


def _cache_dir() -> Path:
    return QUERYSEP_CACHE_ROOT / "hf"


def _load_runtime() -> QuerysepRuntime:
    model_id = QUERYSEP_MODEL_ID
    device = _normalize_device(QUERYSEP_RUNTIME_DEVICE)
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=_cache_dir())
    model = ClapModel.from_pretrained(model_id, cache_dir=_cache_dir())
    model = model.to(torch.device(device))
    model.eval()
    loaded_at = time.time()
    return QuerysepRuntime(
        model_id=model_id,
        device=device,
        processor=processor,
        model=model,
        loaded_at=loaded_at,
        load_seconds=time.perf_counter() - start,
        last_used_at=loaded_at,
    )


def ensure_runtime() -> tuple[QuerysepRuntime, bool]:
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
        runtime.model.cpu()
    except Exception:
        pass
    try:
        del runtime.model
    except Exception:
        pass
    try:
        del runtime.processor
    except Exception:
        pass
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


def cache_snapshot() -> dict[str, object]:
    with _LOCK:
        _maybe_unload_idle_runtime_locked()
        if _RUNTIME is None:
            return {"loaded": False, "model_id": QUERYSEP_MODEL_ID}
        return {
            "loaded": True,
            "model_id": _RUNTIME.model_id,
            "device": _RUNTIME.device,
            "loaded_at": _RUNTIME.loaded_at,
            "load_seconds": round(_RUNTIME.load_seconds, 3),
            "last_used_at": _RUNTIME.last_used_at,
        }


def _move_inputs(inputs: dict[str, object], device: str) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def encode_text(text: str) -> tuple[np.ndarray, dict[str, object]]:
    runtime, cache_hit = ensure_runtime()
    inputs = runtime.processor(text=[text], return_tensors="pt", padding=True)
    moved = _move_inputs(dict(inputs), runtime.device)
    with torch.inference_mode():
        embeddings = runtime.model.get_text_features(**moved)
        embeddings = F.normalize(embeddings, dim=-1)
    vector = embeddings[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return vector, {
        "cache_hit": cache_hit,
        "model_id": runtime.model_id,
        "device": runtime.device,
    }


def encode_audio(audio: np.ndarray, sample_rate: int, runtime_meta: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
    runtime, cache_hit = ensure_runtime()
    inputs = runtime.processor(audios=[audio], sampling_rate=sample_rate, return_tensors="pt", padding=True)
    moved = _move_inputs(dict(inputs), runtime.device)
    with torch.inference_mode():
        embeddings = runtime.model.get_audio_features(**moved)
        embeddings = F.normalize(embeddings, dim=-1)
    vector = embeddings[0].detach().cpu().numpy().astype(np.float32, copy=False)
    meta = {
        "cache_hit": cache_hit,
        "model_id": runtime.model_id,
        "device": runtime.device,
    }
    if runtime_meta:
        meta = {**runtime_meta, **meta}
    return vector, meta


def prefetch_runtime() -> dict[str, object]:
    runtime, cache_hit = ensure_runtime()
    marker = {
        "model_id": runtime.model_id,
        "device": runtime.device,
        "cache_hit": cache_hit,
        "loaded_at": runtime.loaded_at,
        "load_seconds": round(runtime.load_seconds, 3),
        "last_used_at": runtime.last_used_at,
    }
    marker_path = QUERYSEP_CACHE_ROOT / "prefetch_state.json"
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return marker
