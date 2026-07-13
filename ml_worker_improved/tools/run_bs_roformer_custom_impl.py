from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml


def emit(phase: str, **fields: Any) -> None:
    print(json.dumps({"phase": phase, **fields}, ensure_ascii=False, sort_keys=True), flush=True)


def require_mps() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the custom BS-RoFormer adapter")


def config_uses_amp(config_path: Path) -> bool:
    payload = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must be a mapping: {config_path}")
    training = payload.get("training") or {}
    if not isinstance(training, dict):
        raise ValueError(f"Model training config must be a mapping: {config_path}")
    return bool(training.get("use_amp", True))


def install_mps_autocast_compatibility() -> dict[str, Any]:
    if not torch.amp.autocast_mode.is_autocast_available("mps"):
        raise RuntimeError("Native MPS autocast is unavailable")

    def mps_autocast(
        enabled: bool = True,
        dtype: torch.dtype | None = torch.float16,
        cache_enabled: bool = True,
    ):
        return torch.autocast(
            device_type="mps",
            dtype=dtype or torch.float16,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )

    # The vendored demixer calls the legacy CUDA autocast API even when its
    # selected device is MPS. This redirect is local to the one-job exporter process.
    torch.cuda.amp.autocast = mps_autocast
    return {
        "compatibility_installed": True,
        "source_api": "torch.cuda.amp.autocast",
        "target_api": "torch.autocast",
        "device_type": "mps",
        "dtype": "float16",
    }


def guard_mps_fft_operation(operation_name: str, operation, state: dict[str, int]):
    mps_call_key = f"{operation_name}_mps_calls"
    mps_failure_key = f"{operation_name}_mps_failures"
    blocked_retry_key = f"{operation_name}_cpu_retry_attempts"
    retry_state = threading.local()

    def guarded(input_tensor, *args, **kwargs):
        device_type = getattr(getattr(input_tensor, "device", None), "type", "")
        if device_type == "mps":
            state[mps_call_key] += 1
            retry_state.cpu_retry_pending = False
            try:
                return operation(input_tensor, *args, **kwargs)
            except Exception:
                state[mps_failure_key] += 1
                retry_state.cpu_retry_pending = True
                raise
        if device_type == "cpu" and getattr(retry_state, "cpu_retry_pending", False):
            state[blocked_retry_key] += 1
            retry_state.cpu_retry_pending = False
            raise RuntimeError(
                f"CPU {operation_name} retry after failed MPS execution is forbidden"
            )
        return operation(input_tensor, *args, **kwargs)

    return guarded


def install_mps_fft_cpu_retry_guard() -> dict[str, int]:
    state = {
        "stft_mps_calls": 0,
        "stft_mps_failures": 0,
        "stft_cpu_retry_attempts": 0,
        "istft_mps_calls": 0,
        "istft_mps_failures": 0,
        "istft_cpu_retry_attempts": 0,
    }
    torch.stft = guard_mps_fft_operation("stft", torch.stft, state)
    torch.istft = guard_mps_fft_operation("istft", torch.istft, state)
    return state


def install_bs_roformer_implementation(runtime_repo: Path, model_implementation_path: Path) -> str:
    runtime_root = str(runtime_repo.resolve())
    if runtime_root not in sys.path:
        sys.path.insert(0, runtime_root)
    package = importlib.import_module("models.bs_roformer")
    module_id = hashlib.sha256(str(model_implementation_path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"models.bs_roformer.custom_runtime_{module_id}"
    spec = importlib.util.spec_from_file_location(module_name, model_implementation_path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load BS-RoFormer implementation: {model_implementation_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    implementation = getattr(module, "BSRoformer", None)
    if implementation is None:
        raise RuntimeError(f"BSRoformer class is missing: {model_implementation_path}")
    package.BSRoformer = implementation
    return module_name


def build_proc_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_type": "bs_roformer",
        "config_path": str(args.config_path),
        "start_check_point": str(args.checkpoint_path),
        "input_folder": str(args.input_dir),
        "store_dir": str(args.output_dir),
        "device_ids": [0],
        "force_cpu": False,
        "disable_detailed_pbar": bool(args.disable_detailed_pbar),
        "filename_template": str(args.filename_template),
        "use_tta": bool(args.use_tta),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BS-RoFormer checkpoint with its exact isolated architecture.")
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--model-implementation-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-template", default="{instr}")
    parser.add_argument("--disable-detailed-pbar", action="store_true")
    parser.add_argument("--use-tta", action="store_true")
    return parser.parse_args(argv)


def run_custom_architecture(args: argparse.Namespace) -> int:
    for required_path in (
        args.runtime_repo,
        args.model_implementation_path,
        args.config_path,
        args.checkpoint_path,
        args.input_dir,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    if not (args.runtime_repo / "inference.py").is_file():
        raise FileNotFoundError(args.runtime_repo / "inference.py")

    original_cuda_autocast = torch.cuda.amp.autocast
    original_stft = torch.stft
    original_istft = torch.istft
    try:
        require_mps()
        autocast_contract = install_mps_autocast_compatibility()
        autocast_contract["enabled_by_config"] = config_uses_amp(args.config_path)
        fft_guard_state = install_mps_fft_cpu_retry_guard()
        module_name = install_bs_roformer_implementation(
            args.runtime_repo,
            args.model_implementation_path,
        )
        inference = importlib.import_module("inference")
        proc_folder = getattr(inference, "proc_folder", None)
        if proc_folder is None:
            raise RuntimeError("Vendor inference.proc_folder is unavailable")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        emit(
            "custom_architecture_ready",
            device="mps",
            cpu_fallback_allowed=False,
            strict_checkpoint_load=True,
            implementation_module=module_name,
            model_implementation_path=str(args.model_implementation_path.resolve()),
            precision_mode="mps_autocast_float16"
            if autocast_contract["enabled_by_config"]
            else "mps_float32",
            autocast_contract=autocast_contract,
            mps_fft_cpu_retry_guard=True,
        )
        proc_folder(build_proc_args(args))
        emit(
            "custom_architecture_finished",
            device="mps",
            output_dir=str(args.output_dir.resolve()),
            fft_guard_state=fft_guard_state,
        )
        return 0
    finally:
        torch.cuda.amp.autocast = original_cuda_autocast
        torch.stft = original_stft
        torch.istft = original_istft


def main(argv: Sequence[str] | None = None) -> int:
    return run_custom_architecture(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
