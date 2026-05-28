from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import importlib.util
import io
import os
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from bs_roformer.inference import SafeLoaderWithTuple, _select_device
from bs_roformer.utils import get_model_from_config
from ml_collections import ConfigDict

from app.utils import MODELS_ROOT

torch.backends.cudnn.benchmark = True


def _parse_device_ids(raw_value: str | None) -> tuple[int, ...]:
    if not raw_value:
        return ()

    values: list[int] = []
    for chunk in raw_value.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(int(token))
    return tuple(values)


def resolve_model_paths(model_slug: str) -> tuple[Path, Path]:
    model_dir = MODELS_ROOT / model_slug
    yaml_candidates = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
    ckpt_candidates = list(model_dir.glob("*.ckpt"))
    config_path = model_dir / "model.yaml"
    checkpoint_path = model_dir / "model.ckpt"
    if yaml_candidates:
        config_path = yaml_candidates[0]
    if ckpt_candidates:
        checkpoint_path = ckpt_candidates[0]
    return config_path, checkpoint_path


def _normalize_requested_device(requested_device: str | None) -> str | None:
    if requested_device is None:
        return None
    normalized = requested_device.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _runtime_device_ids(explicit_device_ids: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if explicit_device_ids is not None:
        return explicit_device_ids
    return _parse_device_ids(os.getenv("BSROFORMER_RUNTIME_DEVICE_IDS"))


def _runtime_device(explicit_device: str | None = None) -> str | None:
    if explicit_device is not None:
        return _normalize_requested_device(explicit_device)
    return _normalize_requested_device(os.getenv("BSROFORMER_RUNTIME_DEVICE"))


def _cache_limit() -> int:
    raw_value = os.getenv("BSROFORMER_MAX_CACHED_MODELS", "1").strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 1


def _idle_unload_seconds() -> int:
    raw_value = os.getenv("BSROFORMER_IDLE_UNLOAD_SECONDS", "900").strip()
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 900


def _tta_modes() -> tuple[str, ...]:
    """Resolve the active Test-Time Augmentation set.

    Modes (env BSR_TTA, default ``standard`` for the accuracy_max profile):

    * ``off`` — single forward pass (legacy behaviour, fastest).
    * ``standard`` — original mix + polarity-flipped mix, averaged. ~2x time,
      typically gains 0.3–0.5 dB SI-SDR on Roformer-class separators.
    * ``aggressive`` — original + polarity flip + L/R channel swap +
      polarity-flipped channel-swapped, averaged. ~4x time, marginal extra
      gain (~0.1–0.2 dB) on top of ``standard``. Recommended only for the
      final master pass on slow-but-accurate jobs (cf. R6 in the
      stem-tandem-improvement-roadmap).
    """
    raw = os.getenv("BSR_TTA", "standard").strip().lower()
    if raw in {"", "off", "none", "0", "false", "no"}:
        return ("identity",)
    if raw in {"standard", "polarity", "default", "on", "true", "yes"}:
        return ("identity", "polarity")
    if raw in {"aggressive", "max", "full"}:
        return ("identity", "polarity", "channel_swap", "polarity_channel_swap")
    return ("identity",)


def _augment_mixture(mode: str, mixture: torch.Tensor) -> torch.Tensor:
    if mode == "identity":
        return mixture
    if mode == "polarity":
        return -mixture
    if mode == "channel_swap":
        if mixture.shape[0] >= 2:
            return mixture.flip(0)
        return mixture
    if mode == "polarity_channel_swap":
        if mixture.shape[0] >= 2:
            return -mixture.flip(0)
        return -mixture
    return mixture


def _invert_augmentation(mode: str, predicted: np.ndarray) -> np.ndarray:
    """Undo the augmentation on the model's prediction so all variants are
    aligned in the original (mix-frame) coordinate system.

    Both the model output and the inverse follow the same shape convention as
    the demix_track contract: (channels, length) numpy arrays. We make a
    contiguous copy so the buffer is safe to accumulate into.
    """
    arr = np.ascontiguousarray(predicted).astype(np.float32, copy=False)
    if mode == "identity":
        return arr.copy()
    if mode == "polarity":
        return -arr
    if mode == "channel_swap":
        if arr.shape[0] >= 2:
            return arr[::-1, :].copy()
        return arr.copy()
    if mode == "polarity_channel_swap":
        if arr.shape[0] >= 2:
            return -arr[::-1, :].copy()
        return -arr
    return arr.copy()


def _demix_with_tta(
    demix_fn,
    config,
    model: nn.Module,
    mixture: torch.Tensor,
    device: torch.device,
    first_chunk_time: float | None,
    modes: tuple[str, ...] | None = None,
):
    """Run separation under one or more augmentations and average results.

    ``demix_fn`` is the vendor demix_track callable (its signature differs
    slightly between mel_band_roformer and bs_roformer vendors, but the
    shape of the returned dict[str, ndarray] is identical). The first call
    establishes ``first_chunk_time`` for ETA reporting; subsequent passes
    re-use it.
    """
    active_modes = modes or _tta_modes()
    accumulated: dict[str, np.ndarray] = {}
    last_first_chunk_time = first_chunk_time

    for index, mode in enumerate(active_modes):
        augmented = _augment_mixture(mode, mixture)
        result, last_first_chunk_time = demix_fn(
            config,
            model,
            augmented,
            device,
            last_first_chunk_time,
        )
        for stem_name, predicted in result.items():
            inverted = _invert_augmentation(mode, predicted)
            if stem_name not in accumulated:
                accumulated[stem_name] = inverted.astype(np.float32, copy=True)
            else:
                base = accumulated[stem_name]
                if base.shape != inverted.shape:
                    n = min(base.shape[-1], inverted.shape[-1])
                    base = base[..., :n]
                    inverted = inverted[..., :n]
                    accumulated[stem_name] = base
                accumulated[stem_name] = accumulated[stem_name] + inverted

    if len(active_modes) > 1:
        scale = 1.0 / float(len(active_modes))
        for stem_name in accumulated:
            accumulated[stem_name] = accumulated[stem_name] * scale

    return accumulated, last_first_chunk_time


@dataclass
class CachedRuntime:
    cache_key: str
    model_slug: str
    model_family: str
    config_path: Path
    checkpoint_path: Path
    config: ConfigDict
    model: nn.Module
    device: torch.device
    device_label: str
    device_ids: tuple[int, ...]
    instruments: tuple[str, ...]
    target_instrument: str | None
    load_seconds: float
    loaded_at: float
    last_used_at: float


_CACHE_LOCK = RLock()
_RUN_LOCK = RLock()
_MODEL_CACHE: "OrderedDict[str, CachedRuntime]" = OrderedDict()
_MELBAND_VENDOR_LOCK = RLock()
_MELBAND_VENDOR_MODULE = None


def _runtime_cache_key(model_slug: str, requested_device: str | None, device_ids: tuple[int, ...]) -> str:
    device_part = requested_device or "auto"
    device_ids_part = ",".join(str(item) for item in device_ids) if device_ids else "-"
    return f"{model_slug}|{device_part}|{device_ids_part}"


def _read_config(config_path: Path) -> ConfigDict:
    with config_path.open("r", encoding="utf-8") as handle:
        return ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))


def _melband_vendor_root() -> Path:
    return Path(os.getenv("MELBAND_VENDOR_ROOT", "/app/vendor/mel_band_roformer_vocal_model"))


def _ensure_sys_path(path: Path) -> None:
    normalized = str(path)
    if normalized not in sys.path:
        sys.path.insert(0, normalized)


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _melband_vendor_module():
    global _MELBAND_VENDOR_MODULE
    with _MELBAND_VENDOR_LOCK:
        if _MELBAND_VENDOR_MODULE is not None:
            return _MELBAND_VENDOR_MODULE

        vendor_root = _melband_vendor_root()
        if not vendor_root.exists():
            raise FileNotFoundError(f"melband vendor root not found: {vendor_root}")

        _ensure_sys_path(vendor_root)
        _MELBAND_VENDOR_MODULE = _load_module_from_path(
            "stemlab_melband_vendor_utils",
            vendor_root / "utils.py",
        )
        return _MELBAND_VENDOR_MODULE


def _detect_model_family(config_path: Path, config: ConfigDict) -> str:
    model_cfg = getattr(config, "model", None)
    if model_cfg is not None and getattr(model_cfg, "num_bands", None) is not None:
        return "mel_band_roformer"

    config_name = config_path.name.lower()
    if "mel_band" in config_name or "melband" in config_name:
        return "mel_band_roformer"

    # R10: SCNet medium support. Detect either by an explicit SCNet marker in
    # the config (the ZFTurbo MSST repo uses `model.dims` + `model.bandsplit`),
    # or by file-name convention `scnet*.yaml`. The actual model weights must
    # be placed manually under MODELS_ROOT (see docs/02_MODEL_STRATEGY_RU.md).
    if (
        model_cfg is not None
        and (
            getattr(model_cfg, "bandsplit", None) is not None
            or getattr(model_cfg, "bandsplit_ratios", None) is not None
        )
    ):
        return "scnet"
    if "scnet" in config_name:
        return "scnet"

    return "bs_roformer"


def _load_state_dict(checkpoint_path: Path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def _create_model(model_family: str, config: ConfigDict) -> nn.Module:
    if model_family == "mel_band_roformer":
        vendor_root = _melband_vendor_root()
        _ensure_sys_path(vendor_root)
        melband_module = importlib.import_module("models.mel_band_roformer")
        model_kwargs = dict(config.model)
        if model_kwargs.get("multi_stft_resolutions_window_sizes") is not None:
            model_kwargs["multi_stft_resolutions_window_sizes"] = tuple(model_kwargs["multi_stft_resolutions_window_sizes"])
        return melband_module.MelBandRoformer(**model_kwargs)
    if model_family == "scnet":
        return _create_scnet_model(config)
    return get_model_from_config("bs_roformer", config)


def _create_scnet_model(config: ConfigDict) -> nn.Module:
    """R10: SCNet medium loader.

    SCNet from the Music-Source-Separation-Training repo (ZFTurbo) is not
    bundled with the bs-roformer-infer wheel. We try a small set of import
    paths in priority order so admins can either pip-install the package or
    drop a vendored copy under ``vendor/scnet`` without changing this file.
    Failure is loud and includes the recommended next step.
    """
    last_error: Exception | None = None
    candidate_modules = (
        "scnet.scnet",
        "scnet",
        "models.scnet",
        "Music_Source_Separation_Training.models.scnet",
    )
    scnet_module = None
    for module_name in candidate_modules:
        try:
            scnet_module = importlib.import_module(module_name)
            break
        except ImportError as exc:
            last_error = exc

    if scnet_module is None:
        vendor_candidates = (
            Path(os.environ.get("SCNET_VENDOR_ROOT", "/app/vendor/scnet")),
            Path("/app/vendor/Music-Source-Separation-Training"),
        )
        for candidate in vendor_candidates:
            if candidate.exists():
                _ensure_sys_path(candidate)
                try:
                    scnet_module = importlib.import_module("scnet")
                    break
                except ImportError as exc:
                    last_error = exc
                    continue

    if scnet_module is None:
        raise ImportError(
            "SCNet model requested but no SCNet implementation could be imported. "
            "Either `pip install scnet` (when the package becomes available) or "
            "vendor the ZFTurbo Music-Source-Separation-Training repo to "
            "`/app/vendor/scnet` (see docs/02_MODEL_STRATEGY_RU.md). "
            f"Last import error: {last_error!r}"
        )

    model_kwargs = dict(config.model)
    if hasattr(scnet_module, "SCNet"):
        return scnet_module.SCNet(**model_kwargs)
    if hasattr(scnet_module, "Scnet"):
        return scnet_module.Scnet(**model_kwargs)
    raise ImportError(
        "SCNet module imported but no `SCNet` / `Scnet` class found inside. "
        "Pin the implementation you intend to use."
    )


def _hardened_instrumental_residual(
    mix_ch_first: np.ndarray, vocals_arr: np.ndarray
) -> np.ndarray:
    """Compute the instrumental residual with the R7b hardening applied.

    Length-aligns the inputs, subtracts in float32, and clamps the result by
    the per-sample mix magnitude (with 5% headroom) so the residual cannot
    exceed |mix| and create the "negative-vocal" pumping artefact described
    in the audit. Returns a (channels, length) numpy array.
    """
    if mix_ch_first.shape[1] != vocals_arr.shape[1]:
        n = min(mix_ch_first.shape[1], vocals_arr.shape[1])
        mix_ch_first = mix_ch_first[:, :n]
        vocals_arr = vocals_arr[:, :n]
    instrumental = (mix_ch_first - vocals_arr).astype(np.float32, copy=False)
    max_abs = np.abs(mix_ch_first) * 1.05 + 1e-6
    np.clip(instrumental, -max_abs, max_abs, out=instrumental)
    np.nan_to_num(instrumental, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return instrumental


def _resolve_instrumental_strategy(
    instruments: list[str], target_instrument: str | None
) -> tuple[str, str | None]:
    """Decide how to materialise the synthetic ``*_instrumental.wav`` file.

    Returns ``(mode, key)`` where ``mode`` is one of:

    * ``"residual"``: ``instrumental = mix - result[key]`` (key is always
      ``"vocals"``). Used when the model predicts the vocal stem and the
      instrumental must be reconstructed by subtraction.
    * ``"direct"``: ``instrumental = result[key]`` verbatim. Used when the
      checkpoint directly emits the instrumental side under a non-vocal key
      (e.g. ``target_instrument == "other"`` for the unwa
      ``bs_roformer_instrumental_resurrection`` model, where the ``other``
      head **is** the full instrumental — see ``other_fix`` comment in the
      training configs).
    * ``"sum"``: fallback for multi-stem models that contain neither a
      ``"vocals"`` stem nor an explicit target. ``instrumental`` is built as
      ``sum(result[i] for i in instruments)`` (caller responsibility).

    The previous implementation unconditionally used ``mix - result[instruments[0]]``,
    which produced the vocal side instead of the instrumental for both
    ``instrumental_resurrection`` (target=other) and ``coarse_6stem``
    (target=null, instruments[0]=='bass'). See R7c root-cause note.
    """
    if target_instrument is not None:
        if target_instrument == "vocals":
            return ("residual", "vocals")
        return ("direct", target_instrument)
    if "vocals" in instruments:
        return ("residual", "vocals")
    return ("sum", None)


def _run_bsroformer_folder(runtime: CachedRuntime, input_folder: Path, store_dir: Path) -> None:
    """In-house BS-Roformer folder runner with TTA + R7b-grade residual.

    Mirrors the vendor :func:`bs_roformer.inference.run_folder` but routes the
    inner ``demix_track`` call through :func:`_demix_with_tta` (R6) and uses
    :func:`_hardened_instrumental_residual` (R7b) when synthesising the
    ``instrumental`` stem from the residual.
    """
    from bs_roformer.utils import demix_track  # local import keeps the module load lazy

    input_files = sorted(input_folder.glob("*.wav"))
    store_dir.mkdir(parents=True, exist_ok=True)
    first_chunk_time: float | None = None

    instruments: list[str] = list(runtime.instruments)
    if runtime.target_instrument is not None:
        instruments = [runtime.target_instrument]

    for input_path in input_files:
        original_mix, sample_rate = sf.read(str(input_path))
        if original_mix.dtype != np.float32:
            original_mix = original_mix.astype(np.float32, copy=False)

        mix = original_mix
        original_mono = False
        if mix.ndim == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)
        result, first_chunk_time = _demix_with_tta(
            demix_track,
            runtime.config,
            runtime.model,
            mixture,
            runtime.device,
            first_chunk_time,
        )

        for instr in instruments:
            arr = np.ascontiguousarray(result[instr]).astype(np.float32, copy=False)
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            output = arr.T
            if original_mono:
                output = output[:, 0]
            output_path = store_dir / f"{input_path.stem}_{instr}.wav"
            sf.write(str(output_path), output, sample_rate, subtype="FLOAT")

        if instruments:
            mix_ch_first = np.ascontiguousarray(mix.T).astype(np.float32, copy=False)
            mode, key = _resolve_instrumental_strategy(
                list(runtime.instruments), runtime.target_instrument
            )
            if mode == "residual":
                pred = np.ascontiguousarray(result[key]).astype(np.float32, copy=False)
                np.nan_to_num(pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                instrumental_arr = _hardened_instrumental_residual(mix_ch_first, pred)
            elif mode == "direct":
                direct = np.ascontiguousarray(result[key]).astype(np.float32, copy=False)
                np.nan_to_num(direct, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                if direct.shape[1] != mix_ch_first.shape[1]:
                    n = min(direct.shape[1], mix_ch_first.shape[1])
                    direct = direct[:, :n]
                instrumental_arr = direct
            else:  # "sum"
                summed: np.ndarray | None = None
                for instr in instruments:
                    arr = np.ascontiguousarray(result[instr]).astype(np.float32, copy=False)
                    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    if summed is None:
                        summed = arr
                    else:
                        if arr.shape[1] != summed.shape[1]:
                            n = min(arr.shape[1], summed.shape[1])
                            summed = summed[:, :n]
                            arr = arr[:, :n]
                        summed = summed + arr
                instrumental_arr = summed if summed is not None else np.zeros_like(mix_ch_first)
            instrumental_output = instrumental_arr.T
            if original_mono:
                instrumental_output = instrumental_output[:, 0]
            instrumental_path = store_dir / f"{input_path.stem}_instrumental.wav"
            sf.write(str(instrumental_path), instrumental_output, sample_rate, subtype="FLOAT")


def _run_melband_folder(runtime: CachedRuntime, input_folder: Path, store_dir: Path) -> None:
    vendor_utils = _melband_vendor_module()
    demix_track = vendor_utils.demix_track

    input_files = sorted(input_folder.glob("*.wav"))
    store_dir.mkdir(parents=True, exist_ok=True)
    first_chunk_time = None
    instruments = [runtime.target_instrument] if runtime.target_instrument else list(runtime.instruments)
    primary_instrument = instruments[0]

    for input_path in input_files:
        original_mix, sample_rate = sf.read(str(input_path))
        # Force float32 up-front: demix_track returns float32 numpy arrays, and
        # subtracting a float32 prediction from a float64 source previously left
        # ~1e-7 phase residue in the instrumental on every sample (R7b).
        if original_mix.dtype != np.float32:
            original_mix = original_mix.astype(np.float32, copy=False)

        mix = original_mix
        original_mono = False
        if mix.ndim == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)
        result, first_chunk_time = _demix_with_tta(
            demix_track,
            runtime.config,
            runtime.model,
            mixture,
            runtime.device,
            first_chunk_time,
        )

        primary_arr = np.ascontiguousarray(result[primary_instrument]).astype(np.float32, copy=False)
        np.nan_to_num(primary_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        mix_ch_first = np.ascontiguousarray(mix.T).astype(np.float32, copy=False)

        mode, key = _resolve_instrumental_strategy(
            list(runtime.instruments), runtime.target_instrument
        )
        if mode == "residual":
            residual_pred = (
                primary_arr
                if key == primary_instrument
                else np.ascontiguousarray(result[key]).astype(np.float32, copy=False)
            )
            np.nan_to_num(residual_pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            instrumental_arr = _hardened_instrumental_residual(mix_ch_first, residual_pred)
        elif mode == "direct":
            direct = (
                primary_arr
                if key == primary_instrument
                else np.ascontiguousarray(result[key]).astype(np.float32, copy=False)
            )
            np.nan_to_num(direct, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if direct.shape[1] != mix_ch_first.shape[1]:
                n = min(direct.shape[1], mix_ch_first.shape[1])
                direct = direct[:, :n]
            instrumental_arr = direct
        else:  # "sum"
            summed: np.ndarray | None = None
            for instr in instruments:
                arr = np.ascontiguousarray(result[instr]).astype(np.float32, copy=False)
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                if summed is None:
                    summed = arr
                else:
                    if arr.shape[1] != summed.shape[1]:
                        n = min(arr.shape[1], summed.shape[1])
                        summed = summed[:, :n]
                        arr = arr[:, :n]
                    summed = summed + arr
            instrumental_arr = summed if summed is not None else np.zeros_like(mix_ch_first)

        primary_output = primary_arr.T
        instrumental_output = instrumental_arr.T
        if original_mono:
            primary_output = primary_output[:, 0]
            instrumental_output = instrumental_output[:, 0]

        primary_path = store_dir / f"{input_path.stem}_{primary_instrument}.wav"
        sf.write(str(primary_path), primary_output, sample_rate, subtype="FLOAT")

        instrumental_path = store_dir / f"{input_path.stem}_instrumental.wav"
        sf.write(str(instrumental_path), instrumental_output, sample_rate, subtype="FLOAT")


def _effective_device_ids(device: torch.device, device_ids: tuple[int, ...]) -> tuple[int, ...]:
    if device.type != "cuda":
        return ()
    return device_ids


def _load_runtime(model_slug: str, requested_device: str | None, device_ids: tuple[int, ...]) -> CachedRuntime:
    config_path, checkpoint_path = resolve_model_paths(model_slug)
    config = _read_config(config_path)
    model_family = _detect_model_family(config_path, config)

    parser_args = argparse.Namespace(
        device=requested_device,
        device_ids=list(device_ids) if device_ids else None,
    )
    device = _select_device(parser_args)
    effective_device_ids = _effective_device_ids(device, device_ids)

    start_time = time.perf_counter()
    model = _create_model(model_family, config)
    state_dict = _load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)

    if effective_device_ids:
        model = nn.DataParallel(model, device_ids=list(effective_device_ids)).to(device)
    else:
        model = model.to(device)

    model.eval()

    loaded_at = time.time()
    instruments = tuple(config.training.instruments)
    target_instrument = getattr(config.training, "target_instrument", None)
    cache_key = _runtime_cache_key(model_slug, requested_device, effective_device_ids)

    return CachedRuntime(
        cache_key=cache_key,
        model_slug=model_slug,
        model_family=model_family,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        config=config,
        model=model,
        device=device,
        device_label=str(device),
        device_ids=effective_device_ids,
        instruments=instruments,
        target_instrument=target_instrument,
        load_seconds=time.perf_counter() - start_time,
        loaded_at=loaded_at,
        last_used_at=loaded_at,
    )


def _evict_runtime(runtime: CachedRuntime) -> None:
    try:
        base_model = runtime.model.module if isinstance(runtime.model, nn.DataParallel) else runtime.model
        base_model.cpu()
    except Exception:
        pass

    try:
        del runtime.model
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evict_idle_runtimes() -> list[str]:
    ttl = _idle_unload_seconds()
    if ttl <= 0:
        return []

    now = time.time()
    stale_keys = [
        cache_key
        for cache_key, runtime in _MODEL_CACHE.items()
        if (now - runtime.last_used_at) >= ttl
    ]
    evicted: list[str] = []
    for cache_key in stale_keys:
        runtime = _MODEL_CACHE.pop(cache_key, None)
        if runtime is None:
            continue
        _evict_runtime(runtime)
        evicted.append(cache_key)
    return evicted


def _ensure_runtime(model_slug: str, requested_device: str | None, device_ids: tuple[int, ...]) -> tuple[CachedRuntime, bool]:
    normalized_device = _runtime_device(requested_device)
    normalized_device_ids = _runtime_device_ids(device_ids)
    cache_key = _runtime_cache_key(model_slug, normalized_device, normalized_device_ids)

    with _CACHE_LOCK:
        _evict_idle_runtimes()
        cached = _MODEL_CACHE.pop(cache_key, None)
        if cached is not None:
            cached.last_used_at = time.time()
            _MODEL_CACHE[cache_key] = cached
            return cached, True

    runtime = _load_runtime(model_slug, normalized_device, normalized_device_ids)

    with _CACHE_LOCK:
        _MODEL_CACHE[runtime.cache_key] = runtime
        while len(_MODEL_CACHE) > _cache_limit():
            oldest_key, oldest_runtime = _MODEL_CACHE.popitem(last=False)
            if oldest_key == runtime.cache_key:
                _MODEL_CACHE[oldest_key] = oldest_runtime
                break
            _evict_runtime(oldest_runtime)

    return runtime, False


def cache_snapshot() -> dict[str, object]:
    with _CACHE_LOCK:
        evicted_idle = _evict_idle_runtimes()
        entries = [
            {
                "cache_key": runtime.cache_key,
                "model_slug": runtime.model_slug,
                "model_family": runtime.model_family,
                "device": runtime.device_label,
                "device_ids": list(runtime.device_ids),
                "instruments": list(runtime.instruments),
                "target_instrument": runtime.target_instrument,
                "load_seconds": round(runtime.load_seconds, 3),
                "loaded_at": runtime.loaded_at,
                "last_used_at": runtime.last_used_at,
            }
            for runtime in _MODEL_CACHE.values()
        ]
    return {
        "max_cached_models": _cache_limit(),
        "idle_unload_seconds": _idle_unload_seconds(),
        "evicted_idle": evicted_idle,
        "loaded_models": entries,
    }


def run_inprocess_cached(
    model_slug: str,
    *,
    input_folder: Path,
    store_dir: Path,
    requested_device: str | None = None,
    device_ids: tuple[int, ...] | None = None,
) -> tuple[bool, str, str, dict[str, object]]:
    with _RUN_LOCK:
        runtime, cache_hit = _ensure_runtime(model_slug, requested_device, device_ids or ())
        args = argparse.Namespace(
            input_folder=Path(input_folder),
            store_dir=Path(store_dir),
            device=runtime.device_label,
            device_ids=list(runtime.device_ids) if runtime.device_ids else None,
        )

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        success = True
        run_started_at = time.perf_counter()

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                if runtime.model_family == "mel_band_roformer":
                    _run_melband_folder(runtime, args.input_folder, args.store_dir)
                else:
                    _run_bsroformer_folder(runtime, args.input_folder, args.store_dir)
            except Exception:
                success = False
                traceback.print_exc(file=stderr_buffer)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        runtime.last_used_at = time.time()
        metadata = {
            "runtime_mode": "inprocess_cached",
            "cache_hit": cache_hit,
            "model_slug": runtime.model_slug,
            "model_family": runtime.model_family,
            "config_path": str(runtime.config_path),
            "checkpoint_path": str(runtime.checkpoint_path),
            "resolved_device": runtime.device_label,
            "resolved_device_ids": list(runtime.device_ids),
            "target_instrument": runtime.target_instrument,
            "instruments": list(runtime.instruments),
            "model_load_seconds": round(runtime.load_seconds, 3),
            "inference_seconds": round(time.perf_counter() - run_started_at, 3),
            "cache_state": cache_snapshot(),
        }
        return success, stdout_buffer.getvalue(), stderr_buffer.getvalue(), metadata
