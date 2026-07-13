from __future__ import annotations

import importlib.util
import os
import tempfile
from contextlib import nullcontext
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "run_bs_roformer_custom_impl.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_bs_roformer_custom_impl", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_proc_args_is_strict_mps_inference_contract(tmp_path):
    module = _load_module()
    args = Namespace(
        config_path=tmp_path / "cfg.yaml",
        checkpoint_path=tmp_path / "model.ckpt",
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        disable_detailed_pbar=True,
        filename_template="{instr}",
        use_tta=False,
    )

    payload = module.build_proc_args(args)

    assert payload["model_type"] == "bs_roformer"
    assert payload["force_cpu"] is False
    assert payload["device_ids"] == [0]
    assert payload["start_check_point"] == str(args.checkpoint_path)
    assert payload["config_path"] == str(args.config_path)


def test_install_replaces_only_isolated_package_class(tmp_path):
    module = _load_module()
    runtime_repo = tmp_path / "runtime"
    runtime_repo.mkdir()
    implementation_path = tmp_path / "bs_roformer.py"
    implementation_path.write_text("class BSRoformer:\n    marker = 'custom'\n", encoding="utf-8")
    package = SimpleNamespace(BSRoformer=object)
    original_import_module = module.importlib.import_module
    module.importlib.import_module = lambda name: package
    try:
        module_name = module.install_bs_roformer_implementation(runtime_repo, implementation_path)
    finally:
        module.importlib.import_module = original_import_module

    assert module_name.startswith("models.bs_roformer.custom_runtime_")
    assert package.BSRoformer.marker == "custom"


def test_require_mps_rejects_enabled_fallback(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    with pytest.raises(RuntimeError, match="must be 0"):
        module.require_mps()

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)


def test_install_mps_autocast_compatibility_redirects_legacy_cuda_api(monkeypatch):
    module = _load_module()
    captured = {}
    original_cuda_autocast = module.torch.cuda.amp.autocast

    def fake_autocast(*, device_type, dtype, enabled, cache_enabled):
        captured.update(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )
        return nullcontext()

    monkeypatch.setattr(module.torch.amp.autocast_mode, "is_autocast_available", lambda device: device == "mps")
    monkeypatch.setattr(module.torch, "autocast", fake_autocast)
    try:
        contract = module.install_mps_autocast_compatibility()
        with module.torch.cuda.amp.autocast(enabled=True):
            pass
    finally:
        module.torch.cuda.amp.autocast = original_cuda_autocast

    assert captured["device_type"] == "mps"
    assert captured["dtype"] is module.torch.float16
    assert captured["enabled"] is True
    assert contract["compatibility_installed"] is True
    assert contract["device_type"] == "mps"


def test_install_mps_autocast_compatibility_fails_closed_when_unavailable(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.torch.amp.autocast_mode, "is_autocast_available", lambda _device: False)

    with pytest.raises(RuntimeError, match="MPS autocast is unavailable"):
        module.install_mps_autocast_compatibility()


def test_fft_guard_blocks_cpu_retry_after_failed_mps_attempt():
    module = _load_module()
    state = {
        "stft_mps_calls": 0,
        "stft_mps_failures": 0,
        "stft_cpu_retry_attempts": 0,
    }
    calls = []

    def operation(tensor):
        calls.append(tensor.device.type)
        if tensor.device.type == "mps":
            raise RuntimeError("synthetic MPS STFT failure")
        return "ok"

    guarded = module.guard_mps_fft_operation("stft", operation, state)
    assert guarded(SimpleNamespace(device=SimpleNamespace(type="cpu"))) == "ok"
    with pytest.raises(RuntimeError, match="synthetic MPS STFT failure"):
        guarded(SimpleNamespace(device=SimpleNamespace(type="mps")))

    with pytest.raises(RuntimeError, match="CPU stft retry after failed MPS execution is forbidden"):
        guarded(SimpleNamespace(device=SimpleNamespace(type="cpu")))

    assert calls == ["cpu", "mps"]
    assert state["stft_mps_calls"] == 1
    assert state["stft_mps_failures"] == 1
    assert state["stft_cpu_retry_attempts"] == 1


def test_fft_guard_allows_independent_cpu_call_after_successful_mps_call():
    module = _load_module()
    state = {
        "stft_mps_calls": 0,
        "stft_mps_failures": 0,
        "stft_cpu_retry_attempts": 0,
    }
    calls = []

    def operation(tensor):
        calls.append(tensor.device.type)
        return "ok"

    guarded = module.guard_mps_fft_operation("stft", operation, state)

    assert guarded(SimpleNamespace(device=SimpleNamespace(type="mps"))) == "ok"
    assert guarded(SimpleNamespace(device=SimpleNamespace(type="cpu"))) == "ok"
    assert calls == ["mps", "cpu"]
    assert state["stft_mps_failures"] == 0
    assert state["stft_cpu_retry_attempts"] == 0


def test_run_custom_architecture_restores_torch_hooks_between_invocations(tmp_path, monkeypatch):
    module = _load_module()
    runtime_repo = tmp_path / "runtime"
    runtime_repo.mkdir()
    (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
    implementation_path = tmp_path / "bs_roformer.py"
    implementation_path.write_text("class BSRoformer: pass\n", encoding="utf-8")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("training:\n  use_amp: true\n", encoding="utf-8")
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    def fake_stft(tensor, *args, **kwargs):
        return tensor

    def fake_istft(tensor, *args, **kwargs):
        return tensor

    monkeypatch.setattr(module.torch, "stft", fake_stft)
    monkeypatch.setattr(module.torch, "istft", fake_istft)
    original_cuda_autocast = module.torch.cuda.amp.autocast
    monkeypatch.setattr(module, "require_mps", lambda: None)
    monkeypatch.setattr(module.torch.amp.autocast_mode, "is_autocast_available", lambda device: device == "mps")
    monkeypatch.setattr(module.torch, "autocast", lambda **_kwargs: nullcontext())

    package = SimpleNamespace(BSRoformer=object)
    proc_calls = []

    def fake_proc_folder(proc_args):
        proc_calls.append(proc_args)
        assert module.torch.cuda.amp.autocast is not original_cuda_autocast
        with module.torch.cuda.amp.autocast(enabled=True):
            pass
        mps_tensor = SimpleNamespace(device=SimpleNamespace(type="mps"))
        assert module.torch.stft(mps_tensor) is mps_tensor
        assert module.torch.istft(mps_tensor) is mps_tensor

    inference = SimpleNamespace(proc_folder=fake_proc_folder)

    def fake_import_module(name):
        if name == "models.bs_roformer":
            return package
        if name == "inference":
            return inference
        raise AssertionError(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)
    args = Namespace(
        runtime_repo=runtime_repo,
        model_implementation_path=implementation_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        input_dir=input_dir,
        output_dir=output_dir,
        filename_template="{instr}",
        disable_detailed_pbar=True,
        use_tta=False,
    )

    assert module.run_custom_architecture(args) == 0
    assert module.torch.cuda.amp.autocast is original_cuda_autocast
    assert module.torch.stft is fake_stft
    assert module.torch.istft is fake_istft

    assert module.run_custom_architecture(args) == 0
    assert module.torch.cuda.amp.autocast is original_cuda_autocast
    assert module.torch.stft is fake_stft
    assert module.torch.istft is fake_istft
    assert len(proc_calls) == 2
