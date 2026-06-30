from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "export_scnet_full.py"


def _load_module():
    if str(TOOLS_ROOT) not in sys.path:
        sys.path.insert(0, str(TOOLS_ROOT))
    spec = importlib.util.spec_from_file_location("export_scnet_full", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scnet_full_exporter_writes_multistem_no_reference_manifest() -> None:
    module = _load_module()
    sample_rate = 44100
    source_audio = np.zeros((sample_rate, 2), dtype=np.float32)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        repo_path = root / "repo"
        config_path = root / "config.yaml"
        model_path = root / "model.ckpt"
        source_path = root / "source.flac"
        output_dir = root / "out"
        repo_path.mkdir()
        (repo_path / "inference.py").write_text("# fake", encoding="utf-8")
        config_path.write_text("inference:\n  num_overlap: 4\n", encoding="utf-8")
        model_path.write_bytes(b"fake checkpoint")
        sf.write(source_path, source_audio, sample_rate)

        executed_commands: list[list[str]] = []
        executed_timeouts: list[int] = []

        def fake_run(command, *, cwd, capture_output, text, check, timeout):
            executed_commands.append([str(part) for part in command])
            executed_timeouts.append(timeout)
            store_dir = Path(command[command.index("--store_dir") + 1])
            stem_dir = store_dir / "full_track"
            stem_dir.mkdir(parents=True)
            for index, stem_name in enumerate(("drums", "bass", "other", "vocals"), start=1):
                stem_audio = np.full((sample_rate, 2), 0.01 * index, dtype=np.float32)
                sf.write(stem_dir / f"{stem_name}.wav", stem_audio, sample_rate)

            class Completed:
                returncode = 0
                stdout = ""
                stderr = ""

            return Completed()

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            exit_code = module.main(
                [
                    str(source_path),
                    str(output_dir),
                    "--repo-path",
                    str(repo_path),
                    "--config-path",
                    str(config_path),
                    "--model-path",
                    str(model_path),
                    "--device-id",
                    "0",
                    "--num-overlap",
                    "2",
                    "--timeout-sec",
                    "11",
                ]
            )

        manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert executed_commands
    command = executed_commands[0]
    assert executed_timeouts == [11]
    assert "inference.py" in command
    assert command[command.index("--model_type") + 1] == "scnet"
    assert command[command.index("--filename_template") + 1] == "{file_name}/{instr}"
    assert manifest["model_id"] == "scnet_xl_ihf_musdb18"
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["per_track_hardcode_allowed"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    assert manifest["execution_contract"]["expected_acceleration_mode"] == "gpu_inference_then_cpu_dsp_scoring"
    assert manifest["execution_contract"]["gpu_inference_required"] is True
    assert manifest["execution_contract"]["gpu_inference_skipped"] is False
    assert set(manifest["stems"]) == {"drums", "bass", "other", "vocals"}
    assert manifest["stems"]["drums"]["role"] == "target"


def test_scnet_cuda_preflight_fails_closed_when_cuda_is_unavailable(monkeypatch) -> None:
    module = _load_module()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda _device_id: "missing",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    try:
        module.verify_cuda_inference_device(0)
    except RuntimeError as exc:
        assert "torch.cuda.is_available() is false" in str(exc)
    else:  # pragma: no cover - asserts fail if exception is not raised
        raise AssertionError("SCNet CUDA preflight must fail closed when CUDA is unavailable")


def test_scnet_preflight_accepts_explicit_native_macos_mps(monkeypatch) -> None:
    module = _load_module()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda _device_id: "missing",
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: True,
                is_built=lambda: True,
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC", "1")
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")

    preflight = module.verify_cuda_inference_device(0)

    assert preflight["requested_device"] == "mps"
    assert preflight["resolved_device"] == "mps"
    assert preflight["acceleration_mode"] == "native_macos_mps_diagnostic"
    assert preflight["cuda_available"] is False


def test_scnet_manifest_records_cuda_preflight(tmp_path: Path) -> None:
    module = _load_module()
    source_audio_path = tmp_path / "source.flac"
    raw_ref_path = tmp_path / "raw_ref.flac"
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    _write_audio = np.zeros((64, 2), dtype=np.float32)
    sf.write(source_audio_path, _write_audio, 44100)
    sf.write(raw_ref_path, _write_audio, 44100)
    stem_paths = {}
    for stem_name in ("drums", "bass", "other", "vocals"):
        stem_path = stems_dir / f"{stem_name}.flac"
        sf.write(stem_path, _write_audio, 44100)
        stem_paths[stem_name] = stem_path

    manifest = module.build_full_track_stem_manifest(
        source_audio_path=source_audio_path,
        stem_flac_paths=stem_paths,
        raw_ref_flac_path=raw_ref_path,
        repo_path=Path("/repo"),
        model_path=Path("/model.ckpt"),
        config_template_path=Path("/config.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=4,
        sample_rate=44100,
        frames=64,
        elapsed_sec=1.0,
        cuda_preflight={"device_name": "NVIDIA GeForce RTX 4080 Laptop GPU"},
    )

    assert manifest["execution_contract"]["cuda_preflight"]["device_name"] == "NVIDIA GeForce RTX 4080 Laptop GPU"


def test_scnet_subprocess_timeout_reports_explicit_blocker(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    repo_path = tmp_path / "repo"
    config_path = tmp_path / "config.yaml"
    model_path = tmp_path / "model.ckpt"
    source_path = tmp_path / "source.flac"
    output_dir = tmp_path / "out"
    repo_path.mkdir()
    config_path.write_text("inference:\n  num_overlap: 4\n", encoding="utf-8")
    model_path.write_bytes(b"fake checkpoint")
    sf.write(source_path, np.zeros((64, 2), dtype=np.float32), 44100)

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_: None)
    monkeypatch.setattr(
        module,
        "verify_cuda_inference_device",
        lambda device_id: {"device_id": device_id, "device_count": 1, "device_name": "test-gpu"},
    )
    monkeypatch.setattr(
        module,
        "save_audio_window_as_wav",
        lambda source, target, start_sec, end_sec: (sf.write(target, np.zeros((64, 2), dtype=np.float32), 44100), (44100, 64))[1],
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(command, timeout=5, output="partial stdout", stderr="partial stderr")
        ),
    )

    with pytest.raises(RuntimeError, match="scnet_subprocess_timeout"):
        module.main(
            [
                str(source_path),
                str(output_dir),
                "--repo-path",
                str(repo_path),
                "--config-path",
                str(config_path),
                "--model-path",
                str(model_path),
                "--timeout-sec",
                "5",
            ]
        )
