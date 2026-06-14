from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "export_mdx23c_vocal_full.py"


def _load_mdx23c_full_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("export_mdx23c_vocal_full", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_flac(path: Path, *, sample_rate: int = 44100, frames: int = 64) -> None:
    audio = np.zeros((frames, 2), dtype=np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")


def test_full_track_manifest_is_quality_gate_compatible(tmp_path: Path) -> None:
    module = _load_mdx23c_full_module()
    source_audio_path = tmp_path / "source.flac"
    output_dir = tmp_path / "candidate"
    vocals_path = output_dir / "vocals.flac"
    instrumental_path = output_dir / "instrumental.flac"
    raw_ref_path = output_dir / "raw_ref.flac"
    output_dir.mkdir()
    for path in (source_audio_path, vocals_path, instrumental_path, raw_ref_path):
        _write_tiny_flac(path)

    manifest = module.build_full_track_stem_manifest(
        source_audio_path=source_audio_path,
        vocals_flac_path=vocals_path,
        instrumental_flac_path=instrumental_path,
        raw_ref_flac_path=raw_ref_path,
        repo_path=Path("/app/vendor/music_source_separation_training"),
        model_path=Path("/app/vendor/mdx23c_vocals_official/model_vocals_mdx23c_sdr_10.17.ckpt"),
        config_template_path=Path("/app/vendor/mdx23c_vocals_official/config_vocals_mdx23c.yaml"),
        runtime_config_path=Path("/tmp/config_runtime.yaml"),
        device_id=0,
        num_overlap=4,
        sample_rate=44100,
        frames=441000,
        elapsed_sec=12.345,
        cuda_preflight={"device_name": "NVIDIA GeForce RTX 4080 Laptop GPU"},
    )

    assert "preview_window_sec" not in manifest
    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "mdx23c_vocals"
    assert manifest["model_name"] == "mdx23c_vocals"
    assert manifest["contract"]["full_track"] is True
    assert manifest["contract"]["benchmark_compatible"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["hidden_inference"] is False
    assert manifest["execution_contract"]["expected_acceleration_mode"] == "gpu_inference_then_cpu_dsp_scoring"
    assert manifest["execution_contract"]["gpu_inference_required"] is True
    assert manifest["execution_contract"]["gpu_inference_skipped"] is False
    assert manifest["execution_contract"]["cuda_preflight"]["device_name"] == "NVIDIA GeForce RTX 4080 Laptop GPU"
    assert manifest["stems"]["vocals"]["path"] == str(vocals_path)
    assert manifest["stems"]["instrumental"]["path"] == str(instrumental_path)
    assert manifest["artifacts"]["raw_ref_flac"] == str(raw_ref_path)
    assert manifest["audio"]["duration_sec"] == 10.0


def test_writes_full_track_manifest_to_expected_stem_manifest_path(tmp_path: Path) -> None:
    module = _load_mdx23c_full_module()
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    source_audio_path = tmp_path / "source.flac"
    vocals_path = output_dir / "vocals.flac"
    instrumental_path = output_dir / "instrumental.flac"
    raw_ref_path = output_dir / "raw_ref.flac"
    for path in (source_audio_path, vocals_path, instrumental_path, raw_ref_path):
        _write_tiny_flac(path, sample_rate=48000)

    manifest_path = module.write_full_track_stem_manifest(
        output_dir=output_dir,
        source_audio_path=source_audio_path,
        vocals_flac_path=vocals_path,
        instrumental_flac_path=instrumental_path,
        raw_ref_flac_path=raw_ref_path,
        repo_path=Path("/repo"),
        model_path=Path("/model.ckpt"),
        config_template_path=Path("/config.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=4,
        sample_rate=48000,
        frames=96000,
        elapsed_sec=3.0,
        cuda_preflight={"device_name": "NVIDIA GeForce RTX 4080 Laptop GPU"},
    )

    assert manifest_path == output_dir / "stem_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["stems"]["vocals"]["path"] == str(vocals_path)
    assert payload["stems"]["instrumental"]["path"] == str(instrumental_path)
    assert "preview_window_sec" not in payload
    assert payload["execution_contract"]["cuda_preflight"]["device_name"] == "NVIDIA GeForce RTX 4080 Laptop GPU"


def test_mdx23c_cuda_preflight_fails_closed_when_cuda_is_unavailable(monkeypatch) -> None:
    module = _load_mdx23c_full_module()

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
        raise AssertionError("MDX23C CUDA preflight must fail closed when CUDA is unavailable")
