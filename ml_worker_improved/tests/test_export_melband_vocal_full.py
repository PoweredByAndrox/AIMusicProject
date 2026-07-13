from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "export_melband_vocal_full.py"


def _load_melband_full_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("export_melband_vocal_full", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_flac(path: Path, *, sample_rate: int = 44100, frames: int = 64) -> None:
    audio = np.zeros((frames, 2), dtype=np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")


def _neutral_audio(audio: np.ndarray, *_args, **_kwargs) -> np.ndarray:
    return audio


def test_model_sample_rate_contract_rejects_missing_or_conflicting_config(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("audio:\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\n", encoding="utf-8")
    assert module.load_configured_model_sample_rate(config_path) == 44100

    config_path.write_text("audio:\n  sample_rate: 48000\nmodel:\n  sample_rate: 44100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting model sample_rate"):
        module.load_configured_model_sample_rate(config_path)

    config_path.write_text("audio:\n  chunk_size: 44100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing model sample_rate"):
        module.load_configured_model_sample_rate(config_path)


def test_resample_audio_to_layout_restores_exact_source_frame_count() -> None:
    module = _load_melband_full_module()
    source_rate = 48000
    model_rate = 44100
    frames = 24000
    time_axis = np.arange(frames, dtype=np.float32) / source_rate
    source = np.stack(
        [np.sin(2.0 * np.pi * 440.0 * time_axis), np.sin(2.0 * np.pi * 880.0 * time_axis)],
        axis=1,
    ).astype(np.float32)

    model_frames = round(frames * model_rate / source_rate)
    model_audio = module.resample_audio_to_layout(
        source,
        source_sample_rate=source_rate,
        target_sample_rate=model_rate,
        target_frames=model_frames,
    )
    restored = module.resample_audio_to_layout(
        model_audio,
        source_sample_rate=model_rate,
        target_sample_rate=source_rate,
        target_frames=frames,
    )

    assert model_audio.shape == (model_frames, 2)
    assert restored.shape == source.shape
    assert np.isfinite(restored).all()
    assert np.corrcoef(source[:, 0], restored[:, 0])[0, 1] > 0.9999


def test_audio_window_decode_uses_float_wav_without_pcm16_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_melband_full_module()
    captured: dict[str, object] = {}

    class FakeTorchaudio:
        @staticmethod
        def load(_path: str):
            return np.zeros((2, 8), dtype=np.float32), 48000

        @staticmethod
        def save(path: str, waveform: np.ndarray, sample_rate: int, **kwargs):
            captured.update(path=path, waveform=waveform, sample_rate=sample_rate, kwargs=kwargs)

    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio)
    sample_rate, frames = module.save_audio_window_as_wav(
        tmp_path / "source.flac",
        tmp_path / "target.wav",
        start_sec=0.0,
        end_sec=None,
    )

    assert sample_rate == 48000
    assert frames == 8
    assert captured["kwargs"] == {"encoding": "PCM_F", "bits_per_sample": 32}


def test_full_track_manifest_is_quality_gate_compatible(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        repo_path=Path("/app/vendor/mel_band_roformer_vocal_model"),
        model_path=Path("/app/vendor/mel_band_roformer_vocal_model_weights/MelBandRoformer.ckpt"),
        config_template_path=Path("/app/vendor/mel_band_roformer_vocal_model/configs/config_vocals_mel_band_roformer.yaml"),
        runtime_config_path=Path("/tmp/config_runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=441000,
        elapsed_sec=12.345,
        postprocess={"name": None},
        model_display_name="MelBand RoFormer vocal checkpoint",
    )

    assert "preview_window_sec" not in manifest
    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "melband_roformer_vocals"
    assert manifest["model_name"] == "MelBand RoFormer vocal checkpoint"
    assert manifest["model"]["name"] == "MelBand RoFormer vocal checkpoint"
    assert manifest["contract"]["full_track"] is True
    assert manifest["contract"]["benchmark_compatible"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["delivery_status"] == "not_evaluated"
    assert manifest["presentation_status"] == "blocked_until_reference_gate"
    assert manifest["stems"]["vocals"]["path"] == str(vocals_path)
    assert manifest["stems"]["instrumental"]["path"] == str(instrumental_path)
    assert manifest["artifacts"]["raw_ref_flac"] == str(raw_ref_path)
    assert manifest["audio"]["duration_sec"] == 10.0


def test_manifest_records_cuda_preflight_for_gpu_inference(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        repo_path=Path("/repo"),
        model_path=Path("/model.ckpt"),
        config_template_path=Path("/config.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        cuda_preflight={
            "cuda_available": True,
            "device_id": 0,
            "device_count": 1,
            "device_name": "Unit Test GPU",
            "acceleration_mode": "cuda_gpu_inference",
        },
    )

    assert manifest["execution_contract"]["expected_acceleration_mode"] == "gpu_inference_then_cpu_dsp_scoring"
    assert manifest["execution_contract"]["expected_inference_acceleration_mode"] == "cuda_gpu_inference"
    assert manifest["execution_contract"]["actual_acceleration_mode"] == "cuda_gpu_inference"
    assert manifest["execution_contract"]["preflight_acceleration_mode"] == "cuda_gpu_inference"
    assert manifest["execution_contract"]["gpu_inference_required"] is True
    assert manifest["execution_contract"]["gpu_inference_skipped"] is False
    assert manifest["execution_contract"]["cuda_preflight"]["device_name"] == "Unit Test GPU"


def test_manifest_records_vendor_runtime_patches(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        repo_path=Path("/repo"),
        model_path=Path("/model.ckpt"),
        config_template_path=Path("/config.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        cuda_preflight={"acceleration_mode": "native_macos_mps_diagnostic"},
        runtime_patches=[
            {
                "patch_id": "melband_vendor_mps_complex_scatter",
                "file": "/repo/models/mel_band_roformer/mel_band_roformer.py",
                "status": "applied",
                "sha256": "abc",
            }
        ],
    )

    assert manifest["execution_contract"]["vendor_runtime_patches_applied"] is True
    assert manifest["execution_contract"]["expected_acceleration_mode"] == "gpu_inference_then_cpu_dsp_scoring"
    assert manifest["execution_contract"]["expected_inference_acceleration_mode"] == "native_macos_mps_diagnostic"
    assert manifest["execution_contract"]["actual_acceleration_mode"] == "native_macos_mps_diagnostic"
    assert manifest["execution_contract"]["vendor_runtime_patches"][0]["patch_id"] == (
        "melband_vendor_mps_complex_scatter"
    )


def test_verify_cuda_inference_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_melband_full_module()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    with pytest.raises(RuntimeError, match="CUDA inference required"):
        module.verify_cuda_inference_device(0)


def test_verify_cuda_inference_device_returns_device_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_melband_full_module()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(device_id: int) -> str:
            return f"Unit Test GPU {device_id}"

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    preflight = module.verify_cuda_inference_device(1)

    assert preflight == {
        "cuda_available": True,
        "device_id": 1,
        "device_count": 2,
        "device_name": "Unit Test GPU 1",
        "requested_device": "cuda:1",
        "resolved_device": "cuda:1",
        "acceleration_mode": "cuda_gpu_inference",
    }


def test_vendor_cpu_fallback_blocks_manifest_delivery(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    source_audio_path = tmp_path / "source.flac"
    output_dir = tmp_path / "candidate"
    vocals_path = output_dir / "vocals.flac"
    instrumental_path = output_dir / "instrumental.flac"
    raw_ref_path = output_dir / "raw_ref.flac"
    output_dir.mkdir()
    for path in (source_audio_path, vocals_path, instrumental_path, raw_ref_path):
        _write_tiny_flac(path)

    runtime = module.inspect_vendor_inference_runtime(
        stdout="CUDA is not available. Run inference on CPU. It will be very slow...\n",
        stderr="",
        preflight={"acceleration_mode": "native_macos_mps_diagnostic"},
    )
    manifest = module.build_full_track_stem_manifest(
        source_audio_path=source_audio_path,
        vocals_flac_path=vocals_path,
        instrumental_flac_path=instrumental_path,
        raw_ref_flac_path=raw_ref_path,
        repo_path=Path("/repo"),
        model_path=Path("/model.ckpt"),
        config_template_path=Path("/config.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        cuda_preflight={"acceleration_mode": "native_macos_mps_diagnostic"},
        inference_runtime=runtime,
    )

    assert runtime["vendor_cpu_fallback_detected"] is True
    assert manifest["contract"]["fallback_used"] is True
    assert manifest["contract"]["hidden_inference"] is True
    assert manifest["execution_contract"]["gpu_inference_skipped"] is True
    assert manifest["execution_contract"]["actual_acceleration_mode"] == "cpu_inference_fallback"
    assert manifest["delivery_allowed"] is False
    assert manifest["delivery_status"] == "blocked_runtime_contract"


def test_vendor_mps_runtime_patch_skips_cuda_preflight(tmp_path: Path) -> None:
    module = _load_melband_full_module()

    assert module.apply_vendor_mps_runtime_patches(
        tmp_path / "missing_vendor",
        {"acceleration_mode": "cuda_gpu_inference"},
    ) == []


def test_vendor_mps_runtime_patch_materializes_device_and_complex_scatter(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    repo_path = tmp_path / "vendor"
    model_path = repo_path / "models" / "mel_band_roformer" / "mel_band_roformer.py"
    model_path.parent.mkdir(parents=True)
    inference_path = repo_path / "inference.py"
    inference_path.write_text(
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "import warnings",
                "warnings.filterwarnings(\"ignore\")",
                "",
                "def proc_folder(args):",
                "    model = object()",
                module.VENDOR_ORIGINAL_DEVICE_BLOCK.rstrip("\n"),
                "    run_folder(model, args, config, device, verbose=False)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    model_path.write_text(
        "\n".join(
            [
                "import torch",
                "",
                "def exists(val):",
                "    return val is not None",
                "",
                "def forward():",
                module.VENDOR_ORIGINAL_SCATTER_LINE.rstrip("\n"),
                "",
            ]
        ),
        encoding="utf-8",
    )

    patches = module.apply_vendor_mps_runtime_patches(
        repo_path,
        {"acceleration_mode": "native_macos_mps_diagnostic"},
    )

    assert [patch["patch_id"] for patch in patches] == [
        "melband_vendor_mps_device",
        "melband_vendor_mps_complex_scatter",
    ]
    assert [patch["status"] for patch in patches] == ["applied", "applied"]
    assert all(patch["patch_origin"] == "tracked_exporter_runtime_patch" for patch in patches)
    assert all(patch["pre_patch_sha256"] != patch["post_patch_sha256"] for patch in patches)
    assert all(patch["sha256"] == patch["post_patch_sha256"] for patch in patches)
    inference_source = inference_path.read_text(encoding="utf-8")
    model_source = model_path.read_text(encoding="utf-8")
    assert "def configure_inference_device" in inference_source
    assert "torch.device('mps')" in inference_source
    assert "Run inference on MPS" in inference_source
    assert module.VENDOR_CONFIGURE_DEVICE_CALL.strip() in inference_source
    assert "def scatter_add_complex_mps_safe" in model_source
    assert "src.device.type == 'mps'" in model_source
    assert "torch.complex(real, imag)" in model_source
    assert module.VENDOR_MPS_SCATTER_LINE.strip() in model_source

    repeat_patches = module.apply_vendor_mps_runtime_patches(
        repo_path,
        {"acceleration_mode": "native_macos_mps_diagnostic"},
    )
    assert [patch["status"] for patch in repeat_patches] == ["already_present", "already_present"]
    assert all(patch["patch_origin"] == "validated_exact_runtime_patch" for patch in repeat_patches)
    assert all(patch["pre_patch_sha256"] == patch["post_patch_sha256"] for patch in repeat_patches)


def test_vendor_mps_runtime_patch_rejects_divergent_device_helper(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    repo_path = tmp_path / "vendor"
    repo_path.mkdir()
    inference_path = repo_path / "inference.py"
    inference_path.write_text(
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "import warnings",
                "warnings.filterwarnings(\"ignore\")",
                "",
                "def configure_inference_device(model, device_ids):",
                "    return 'cpu', model",
                "",
                "def proc_folder(args):",
                "    model = object()",
                module.VENDOR_CONFIGURE_DEVICE_CALL.rstrip("\n"),
                "    run_folder(model, args, config, device, verbose=False)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="divergent MPS device helper"):
        module.patch_vendor_inference_for_mps(repo_path)


def test_vendor_mps_runtime_patch_rejects_divergent_complex_scatter_helper(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    repo_path = tmp_path / "vendor"
    model_path = repo_path / "models" / "mel_band_roformer" / "mel_band_roformer.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text(
        "\n".join(
            [
                "import torch",
                "",
                "def exists(val):",
                "    return val is not None",
                "",
                "def scatter_add_complex_mps_safe(target, dim, index, src):",
                "    return target",
                "",
                "def forward():",
                module.VENDOR_MPS_SCATTER_LINE.rstrip("\n"),
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="divergent MPS complex scatter helper"):
        module.patch_vendor_complex_scatter_for_mps(repo_path)


def test_vendor_target_aware_companion_patch_prevents_case_insensitive_role_collision(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    inference_path = repo_path / "inference.py"
    inference_path.write_text(
        "def run_folder():\n" + module.VENDOR_LEGACY_COMPANION_BLOCK,
        encoding="utf-8",
    )

    patch = module.patch_vendor_target_aware_companion(repo_path)
    patched = inference_path.read_text(encoding="utf-8")

    assert patch["patch_id"] == "melband_vendor_target_aware_companion"
    assert patch["status"] == "applied"
    assert module.VENDOR_LEGACY_COMPANION_BLOCK not in patched
    assert module.VENDOR_TARGET_AWARE_COMPANION_BLOCK in patched
    assert "if len(instruments) == 1:" in patched
    assert "companion_stem = 'vocals'" in patched
    assert "{}_instrumental.wav" not in patched

    repeat = module.patch_vendor_target_aware_companion(repo_path)
    assert repeat["status"] == "already_present"


def test_vendor_target_aware_companion_patch_rejects_ambiguous_vendor_source(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "inference.py").write_text("def run_folder():\n    pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing ambiguous stem roles"):
        module.patch_vendor_target_aware_companion(repo_path)


def test_melband_preflight_accepts_explicit_native_macos_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_melband_full_module()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def get_device_name(_device_id: int) -> str:
            return "missing"

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_built() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()
        backends = type("Backends", (), {"mps": FakeMps()})()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setenv("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC", "1")
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")

    preflight = module.verify_cuda_inference_device(0)

    assert preflight["requested_device"] == "mps"
    assert preflight["resolved_device"] == "mps"
    assert preflight["acceleration_mode"] == "native_macos_mps_diagnostic"
    assert preflight["cuda_available"] is False


def test_writes_full_track_manifest_to_expected_stem_manifest_path(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        num_overlap=2,
        sample_rate=48000,
        frames=96000,
        elapsed_sec=3.0,
        postprocess={"name": None},
    )

    assert manifest_path == output_dir / "stem_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["stems"]["vocals"]["path"] == str(vocals_path)
    assert payload["stems"]["instrumental"]["path"] == str(instrumental_path)
    assert "preview_window_sec" not in payload


def test_manifest_can_record_registered_checkpoint_identity_without_changing_exporter(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        repo_path=Path("/repo"),
        model_path=Path("/becruily.ckpt"),
        config_template_path=Path("/becruily.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        model_id="becruily_karaoke_vocals",
        model_display_name="Becruily MelBand RoFormer karaoke checkpoint",
    )

    assert manifest["model_id"] == "becruily_karaoke_vocals"
    assert manifest["model"]["model_id"] == "becruily_karaoke_vocals"
    assert manifest["model"]["name"] == "Becruily MelBand RoFormer karaoke checkpoint"


def test_model_source_metadata_verifies_checkpoint_and_config_sha256(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    model_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    metadata_path = tmp_path / "source.json"
    model_path.write_bytes(b"model-bytes")
    config_path.write_bytes(b"config-bytes")
    metadata_path.write_text(
        json.dumps(
            {
                "production_allowed": False,
                "files": [
                    {"file": model_path.name, "expected_sha256": module._sha256_file(model_path)},
                    {"file": config_path.name, "expected_sha256": module._sha256_file(config_path)},
                ],
            }
        ),
        encoding="utf-8",
    )

    metadata = module.load_model_source_metadata(metadata_path)
    verified = module.verify_model_source_files(metadata, [model_path, config_path])

    assert verified == {
        model_path.name: module._sha256_file(model_path),
        config_path.name: module._sha256_file(config_path),
    }

    model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Model source sha256 mismatch"):
        module.verify_model_source_files(metadata, [model_path, config_path])


def test_manifest_records_verified_model_source_and_blocks_unlicensed_production(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    source_audio_path = tmp_path / "source.flac"
    output_dir = tmp_path / "candidate"
    vocals_path = output_dir / "vocals.flac"
    instrumental_path = output_dir / "instrumental.flac"
    raw_ref_path = output_dir / "raw_ref.flac"
    runtime_config_path = tmp_path / "runtime.yaml"
    output_dir.mkdir()
    for path in (source_audio_path, vocals_path, instrumental_path, raw_ref_path):
        _write_tiny_flac(path)
    runtime_config_path.write_text("inference:\n  batch_size: 1\n", encoding="utf-8")
    verified_sha256 = {"model.ckpt": "abc", "config.yaml": "def"}

    manifest = module.build_full_track_stem_manifest(
        source_audio_path=source_audio_path,
        vocals_flac_path=vocals_path,
        instrumental_flac_path=instrumental_path,
        raw_ref_flac_path=raw_ref_path,
        repo_path=Path("/repo"),
        model_path=Path("/model.ckpt"),
        config_template_path=Path("/config.yaml"),
        runtime_config_path=runtime_config_path,
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        model_source={"status": "verified_downloaded_diagnostic_only", "production_allowed": False},
        verified_model_sha256=verified_sha256,
    )

    assert manifest["model"]["verified_sha256"] == verified_sha256
    assert manifest["model"]["runtime_config_sha256"] == module._sha256_file(runtime_config_path)
    assert manifest["model_source"]["status"] == "verified_downloaded_diagnostic_only"
    assert manifest["model_production_allowed"] is False
    assert manifest["delivery_allowed"] is False


def test_single_stem_instrumental_output_materializes_vocal_residual_with_provenance() -> None:
    module = _load_melband_full_module()
    raw_audio = np.array([[0.7, -0.7], [0.2, -0.2]], dtype=np.float32)
    instrumental_audio = np.array([[0.3, -0.4], [0.1, -0.1]], dtype=np.float32)

    resolved = module.materialize_missing_companion_stems(
        output_contract={
            "num_stems": 1,
            "target_instrument": "Instrumental",
            "expected_model_stems": ("instrumental",),
            "allow_residual_companion_materialization": True,
        },
        raw_audio=raw_audio,
        raw_sample_rate=44100,
        vocals_audio=None,
        vocals_sample_rate=None,
        instrumental_audio=instrumental_audio,
        instrumental_sample_rate=44100,
    )

    assert resolved["vocals_sample_rate"] == 44100
    assert resolved["instrumental_sample_rate"] == 44100
    np.testing.assert_allclose(resolved["vocals_audio"], raw_audio - instrumental_audio, atol=1e-7)
    np.testing.assert_allclose(resolved["instrumental_audio"], instrumental_audio, atol=1e-7)
    assert resolved["stem_provenance"]["instrumental"]["source"] == "model_output"
    assert resolved["stem_provenance"]["vocals"]["source"] == "residual_from_raw_minus_instrumental"
    assert resolved["residual_companion_materialized"] is True


def test_single_stem_vendor_companion_is_verified_and_not_claimed_as_model_output() -> None:
    module = _load_melband_full_module()
    raw_audio = np.array([[0.7, -0.7], [0.2, -0.2]], dtype=np.float32)
    instrumental_audio = np.array([[0.3, -0.4], [0.1, -0.1]], dtype=np.float32)
    vocals_audio = raw_audio - instrumental_audio

    resolved = module.materialize_missing_companion_stems(
        output_contract={
            "num_stems": 1,
            "target_instrument": "Instrumental",
            "expected_model_stems": ("instrumental",),
            "allow_residual_companion_materialization": True,
        },
        raw_audio=raw_audio,
        raw_sample_rate=44100,
        vocals_audio=vocals_audio,
        vocals_sample_rate=44100,
        instrumental_audio=instrumental_audio,
        instrumental_sample_rate=44100,
    )

    assert resolved["stem_provenance"]["instrumental"] == {"source": "model_output", "model_output": True}
    assert resolved["stem_provenance"]["vocals"] == {
        "source": "vendor_residual_from_raw_minus_instrumental",
        "model_output": False,
        "residual_companion": True,
        "residual_verified": True,
    }
    assert resolved["residual_companion_materialized"] is True


def test_two_stem_outputs_remain_model_outputs_without_residual_relabeling() -> None:
    module = _load_melband_full_module()
    vocals_audio = np.array([[0.4, -0.4], [0.1, -0.1]], dtype=np.float32)
    instrumental_audio = np.array([[0.3, -0.3], [0.1, -0.1]], dtype=np.float32)
    raw_audio = vocals_audio + instrumental_audio

    resolved = module.materialize_missing_companion_stems(
        output_contract={
            "num_stems": 2,
            "target_instrument": None,
            "expected_model_stems": ("vocals", "instrumental"),
            "allow_residual_companion_materialization": False,
        },
        raw_audio=raw_audio,
        raw_sample_rate=44100,
        vocals_audio=vocals_audio,
        vocals_sample_rate=44100,
        instrumental_audio=instrumental_audio,
        instrumental_sample_rate=44100,
    )

    assert resolved["stem_provenance"]["vocals"]["model_output"] is True
    assert resolved["stem_provenance"]["instrumental"]["model_output"] is True
    assert resolved["residual_companion_materialized"] is False


def test_two_stem_contract_rejects_missing_vocal_output_without_residual_materialization() -> None:
    module = _load_melband_full_module()
    raw_audio = np.zeros((2, 2), dtype=np.float32)
    instrumental_audio = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(FileNotFoundError, match="vocals output is missing"):
        module.materialize_missing_companion_stems(
            output_contract={
                "num_stems": 2,
                "target_instrument": None,
                "expected_model_stems": ("vocals", "instrumental"),
                "allow_residual_companion_materialization": False,
            },
            raw_audio=raw_audio,
            raw_sample_rate=44100,
            vocals_audio=None,
            vocals_sample_rate=None,
            instrumental_audio=instrumental_audio,
            instrumental_sample_rate=44100,
        )


def test_instrumental_single_stem_contract_rejects_vocal_only_output_as_runtime_mismatch() -> None:
    module = _load_melband_full_module()
    raw_audio = np.zeros((2, 2), dtype=np.float32)
    vocals_audio = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="expected instrumental output"):
        module.materialize_missing_companion_stems(
            output_contract={
                "num_stems": 1,
                "target_instrument": "Instrumental",
                "expected_model_stems": ("instrumental",),
                "allow_residual_companion_materialization": True,
            },
            raw_audio=raw_audio,
            raw_sample_rate=44100,
            vocals_audio=vocals_audio,
            vocals_sample_rate=44100,
            instrumental_audio=None,
            instrumental_sample_rate=None,
        )


def test_main_rejects_two_stem_config_when_model_omits_vocals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_melband_full_module()
    source_audio_path = tmp_path / "source.flac"
    output_dir = tmp_path / "candidate"
    repo_path = tmp_path / "repo"
    model_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    repo_path.mkdir()
    model_path.write_text("stub checkpoint", encoding="utf-8")
    _write_tiny_flac(source_audio_path, sample_rate=44100, frames=8)
    config_path.write_text(
        yaml.dump(
            {
                "audio": {"chunk_size": 44100, "sample_rate": 44100},
                "model": {"num_stems": 2},
                "training": {"instruments": ["Vocals", "Instrumental"]},
                "inference": {"num_overlap": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    raw_audio = np.array([[0.5, -0.5], [0.2, -0.2]], dtype=np.float32)
    instrumental_audio = np.array([[0.1, -0.1], [0.05, -0.05]], dtype=np.float32)

    def fake_save_audio_window_as_wav(_source: Path, target: Path, *_args, **_kwargs):
        sf.write(str(target), raw_audio, 44100, subtype="FLOAT")
        return 44100, int(raw_audio.shape[0])

    def fake_run(command: list[str], *_args, **_kwargs):
        store_dir = Path(command[command.index("--store_dir") + 1])
        store_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(store_dir / "full_track_instrumental.wav"), instrumental_audio, 44100, subtype="FLOAT")
        return module.subprocess.CompletedProcess(command, 0, stdout="fake inference\n", stderr="")

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(module, "verify_cuda_inference_device", lambda _device_id: {"device_id": 0, "device_count": 1, "device_name": "test"})
    monkeypatch.setattr(module, "save_audio_window_as_wav", fake_save_audio_window_as_wav)
    monkeypatch.setattr(module, "_run_vendor_command", fake_run)
    monkeypatch.setattr(module, "apply_light_debleed_presence", _neutral_audio)
    monkeypatch.setattr(module, "apply_peak_note_debleed", _neutral_audio)
    monkeypatch.setattr(module, "apply_windowed_note_notch", _neutral_audio)
    monkeypatch.setattr(module, "apply_surgical_notches", _neutral_audio)
    monkeypatch.setattr(module, "apply_windowed_band_subtract", _neutral_audio)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_melband_vocal_full.py",
            str(source_audio_path),
            str(output_dir),
            "--repo-path",
            str(repo_path),
            "--model-path",
            str(model_path),
            "--config-path",
            str(config_path),
        ],
    )

    with pytest.raises(FileNotFoundError, match="vocals output is missing"):
        module.main()


def test_main_uses_current_python_for_vendor_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_melband_full_module()
    source_audio_path = tmp_path / "source.flac"
    output_dir = tmp_path / "candidate"
    repo_path = tmp_path / "repo"
    model_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    repo_path.mkdir()
    model_path.write_text("stub checkpoint", encoding="utf-8")
    _write_tiny_flac(source_audio_path, sample_rate=44100, frames=8)
    config_path.write_text(
        yaml.dump(
            {
                "audio": {"chunk_size": 44100, "sample_rate": 44100},
                "model": {"num_stems": 2},
                "training": {"instruments": ["Vocals", "Instrumental"]},
                "inference": {"num_overlap": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    raw_audio = np.array([[0.5, -0.5], [0.2, -0.2]], dtype=np.float32)
    vocals_audio = np.array([[0.4, -0.4], [0.1, -0.1]], dtype=np.float32)
    instrumental_audio = raw_audio - vocals_audio
    captured: dict[str, list[str]] = {}

    def fake_save_audio_window_as_wav(_source: Path, target: Path, *_args, **_kwargs):
        sf.write(str(target), raw_audio, 44100, subtype="FLOAT")
        return 44100, int(raw_audio.shape[0])

    def fake_run(command: list[str], *_args, **_kwargs):
        captured["command"] = command
        store_dir = Path(command[command.index("--store_dir") + 1])
        store_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(store_dir / "full_track_vocals.wav"), vocals_audio, 44100, subtype="FLOAT")
        sf.write(str(store_dir / "full_track_instrumental.wav"), instrumental_audio, 44100, subtype="FLOAT")
        return module.subprocess.CompletedProcess(command, 0, stdout="fake inference\n", stderr="")

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(module, "verify_cuda_inference_device", lambda _device_id: {"device_id": 0, "device_count": 1, "device_name": "test"})
    monkeypatch.setattr(module, "save_audio_window_as_wav", fake_save_audio_window_as_wav)
    monkeypatch.setattr(module, "_run_vendor_command", fake_run)
    monkeypatch.setattr(module, "apply_light_debleed_presence", _neutral_audio)
    monkeypatch.setattr(module, "apply_peak_note_debleed", _neutral_audio)
    monkeypatch.setattr(module, "apply_windowed_note_notch", _neutral_audio)
    monkeypatch.setattr(module, "apply_surgical_notches", _neutral_audio)
    monkeypatch.setattr(module, "apply_windowed_band_subtract", _neutral_audio)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_melband_vocal_full.py",
            str(source_audio_path),
            str(output_dir),
            "--repo-path",
            str(repo_path),
            "--model-path",
            str(model_path),
            "--config-path",
            str(config_path),
        ],
    )

    module.main()

    assert captured["command"][0] == sys.executable


def test_main_prints_vendor_output_before_nonzero_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_melband_full_module()
    source_audio_path = tmp_path / "source.flac"
    output_dir = tmp_path / "candidate"
    repo_path = tmp_path / "repo"
    model_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    repo_path.mkdir()
    (repo_path / "inference.py").write_text(
        "def run_folder():\n" + module.VENDOR_LEGACY_COMPANION_BLOCK,
        encoding="utf-8",
    )
    model_path.write_text("stub checkpoint", encoding="utf-8")
    _write_tiny_flac(source_audio_path, sample_rate=44100, frames=8)
    config_path.write_text(
        yaml.dump(
            {
                "audio": {"chunk_size": 44100, "sample_rate": 44100},
                "model": {"num_stems": 2},
                "training": {"instruments": ["Vocals", "Instrumental"]},
                "inference": {"num_overlap": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    raw_audio = np.array([[0.5, -0.5], [0.2, -0.2]], dtype=np.float32)

    def fake_save_audio_window_as_wav(_source: Path, target: Path, *_args, **_kwargs):
        sf.write(str(target), raw_audio, 44100, subtype="FLOAT")
        return 44100, int(raw_audio.shape[0])

    def fake_run(command: list[str], *_args, **_kwargs):
        print("vendor stdout before failure")
        print("vendor stderr before failure")
        return module.subprocess.CompletedProcess(
            command,
            17,
            stdout="vendor stdout before failure\n",
            stderr="vendor stderr before failure\n",
        )

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "verify_cuda_inference_device",
        lambda _device_id: {
            "device_id": 0,
            "device_count": 1,
            "device_name": "test",
            "acceleration_mode": "cuda_gpu_inference",
        },
    )
    monkeypatch.setattr(module, "save_audio_window_as_wav", fake_save_audio_window_as_wav)
    monkeypatch.setattr(module, "_run_vendor_command", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_melband_vocal_full.py",
            str(source_audio_path),
            str(output_dir),
            "--repo-path",
            str(repo_path),
            "--model-path",
            str(model_path),
            "--config-path",
            str(config_path),
        ],
    )

    with pytest.raises(RuntimeError, match="returncode=17"):
        module.main()

    captured = capsys.readouterr()
    assert "vendor stdout before failure" in captured.out
    assert "vendor stderr before failure" in captured.out


def test_manifest_records_residual_companion_materialization(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        repo_path=Path("/repo"),
        model_path=Path("/becruily_instrumental.ckpt"),
        config_template_path=Path("/becruily_instrumental.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        model_id="becruily_instrumental_vocals",
        model_display_name="Becruily MelBand RoFormer instrumental checkpoint",
        stem_provenance={
            "vocals": {"source": "residual_from_raw_minus_instrumental"},
            "instrumental": {"source": "model_output"},
        },
        residual_companion_materialized=True,
    )

    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["residual_companion_materialized"] is True
    assert manifest["stems"]["vocals"]["source"] == "residual_from_raw_minus_instrumental"
    assert manifest["stems"]["instrumental"]["source"] == "model_output"


def test_manifest_marks_instrumental_model_output_as_target_stem(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        repo_path=Path("/repo"),
        model_path=Path("/becruily_instrumental.ckpt"),
        config_template_path=Path("/becruily_instrumental.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        model_id="becruily_instrumental_vocals",
        model_display_name="Becruily MelBand RoFormer instrumental checkpoint",
        stem_provenance={
            "vocals": {"source": "residual_from_raw_minus_instrumental", "model_output": False},
            "instrumental": {"source": "model_output", "model_output": True},
        },
        residual_companion_materialized=True,
    )

    assert manifest["target_stem"] == "instrumental"
    assert manifest["target_audio_path"] == str(instrumental_path)
    assert manifest["stems"]["instrumental"]["role"] == "target"
    assert manifest["stems"]["vocals"]["role"] == "companion"


def test_instrumental_target_manifest_keeps_minus_as_metadata_only_product_alias(tmp_path: Path) -> None:
    module = _load_melband_full_module()
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
        repo_path=Path("/repo"),
        model_path=Path("/becruily_instrumental.ckpt"),
        config_template_path=Path("/becruily_instrumental.yaml"),
        runtime_config_path=Path("/runtime.yaml"),
        device_id=0,
        num_overlap=2,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
        postprocess={"name": None},
        model_id="becruily_instrumental_vocals",
        model_display_name="Becruily MelBand RoFormer instrumental checkpoint",
        stem_provenance={
            "vocals": {"source": "residual_from_raw_minus_instrumental", "model_output": False},
            "instrumental": {"source": "model_output", "model_output": True},
        },
        residual_companion_materialized=True,
    )

    assert manifest["target_stem"] == "instrumental"
    assert manifest["target_product_stem"] == "minus"
    assert manifest["target_product_audio_path"] == str(instrumental_path)
    assert manifest["target_product_audio_materialized_in_run"] is False
    assert manifest["target_product_materialized_audio_path"] == ""
    assert manifest["role_safe_stem_contract"] == {
        "source_manifest_target_stem": "instrumental",
        "canonical_stem_type": "instrumental",
        "user_facing_stem_role": "minus",
        "manifest_stem_name": "instrumental",
        "product_stem_type": "minus",
        "model_output_stem_type": "instrumental",
        "naming_contract": "source_manifest_uses_instrumental_stem; judge_rows_use_minus_product_role",
    }
    assert manifest["target_stem_artifact"]["path"] == str(instrumental_path)
    assert manifest["target_stem_artifact"]["target_stem"] == "instrumental"
    assert manifest["target_stem_artifact"]["product_stem_type"] == "minus"
    assert manifest["target_stem_artifact"]["model_output_stem_type"] == "instrumental"
    assert manifest["target_stem_artifact"]["product_audio_path"] == str(instrumental_path)
    assert manifest["product_aliases"]["minus"] == {
        "source_stem": "instrumental",
        "source_audio_path": str(instrumental_path),
        "product_stem_type": "minus",
        "model_output_stem_type": "instrumental",
        "product_audio_path": str(instrumental_path),
        "model_output_audio_path": str(instrumental_path),
        "materialized_audio_path": "",
        "materialized_in_run": False,
        "alias_of": "instrumental",
        "role": "product_role_alias",
        "naming_contract": "source_manifest_uses_instrumental_stem; judge_rows_use_minus_product_role",
    }
    assert manifest["stems"]["instrumental"]["path"] == str(instrumental_path)
    assert manifest["stems"]["instrumental"]["role"] == "target"
    assert "minus" not in manifest["stems"]


def test_main_listen_copy_uses_instrumental_for_single_stem_instrumental_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_melband_full_module()
    source_audio_path = tmp_path / "source.flac"
    output_dir = tmp_path / "candidate"
    listen_dir = tmp_path / "listen"
    repo_path = tmp_path / "repo"
    model_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    repo_path.mkdir()
    model_path.write_text("stub checkpoint", encoding="utf-8")
    _write_tiny_flac(source_audio_path, sample_rate=44100, frames=8)
    config_path.write_text(
        yaml.dump(
            {
                "audio": {"chunk_size": 44100, "sample_rate": 44100},
                "model": {"num_stems": 1},
                "training": {"target_instrument": "Instrumental"},
                "inference": {"num_overlap": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    raw_audio = np.array([[0.7, -0.7], [0.2, -0.2]], dtype=np.float32)
    instrumental_audio = np.array([[0.3, -0.4], [0.1, -0.1]], dtype=np.float32)

    def fake_save_audio_window_as_wav(_source: Path, target: Path, *_args, **_kwargs):
        sf.write(str(target), raw_audio, 44100, subtype="FLOAT")
        return 44100, int(raw_audio.shape[0])

    def fake_run(command: list[str], *_args, **_kwargs):
        store_dir = Path(command[command.index("--store_dir") + 1])
        store_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(store_dir / "full_track_instrumental.wav"), instrumental_audio, 44100, subtype="FLOAT")
        return module.subprocess.CompletedProcess(command, 0, stdout="fake inference\n", stderr="")

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(module, "verify_cuda_inference_device", lambda _device_id: {"device_id": 0, "device_count": 1, "device_name": "test"})
    monkeypatch.setattr(module, "save_audio_window_as_wav", fake_save_audio_window_as_wav)
    monkeypatch.setattr(module, "_run_vendor_command", fake_run)
    monkeypatch.setattr(module, "apply_light_debleed_presence", _neutral_audio)
    monkeypatch.setattr(module, "apply_peak_note_debleed", _neutral_audio)
    monkeypatch.setattr(module, "apply_windowed_note_notch", _neutral_audio)
    monkeypatch.setattr(module, "apply_surgical_notches", _neutral_audio)
    monkeypatch.setattr(module, "apply_windowed_band_subtract", _neutral_audio)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_melband_vocal_full.py",
            str(source_audio_path),
            str(output_dir),
            "--repo-path",
            str(repo_path),
            "--model-path",
            str(model_path),
            "--config-path",
            str(config_path),
            "--listen-dir",
            str(listen_dir),
            "--listen-label",
            "candidate.flac",
            "--manifest-model-id",
            "becruily_instrumental_vocals",
        ],
    )

    module.main()

    manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    minus_alias_path = output_dir / "minus.flac"
    instrumental_path = output_dir / "instrumental.flac"
    assert not minus_alias_path.exists()
    assert manifest["target_product_audio_path"] == str(instrumental_path)
    assert manifest["target_model_output_audio_path"] == str(instrumental_path)
    assert manifest["target_stem_artifact"]["product_audio_path"] == str(instrumental_path)
    assert manifest["target_stem_artifact"]["model_output_audio_path"] == str(instrumental_path)
    assert manifest["target_product_audio_materialized_in_run"] is False
    assert manifest["target_product_materialized_audio_path"] == ""
    assert "minus" not in manifest["stems"]
    assert manifest["product_aliases"]["minus"]["source_audio_path"] == str(instrumental_path)

    listen_audio, _ = sf.read(str(listen_dir / "candidate.flac"), always_2d=True)
    assert float(np.max(np.abs(listen_audio - instrumental_audio))) < 1e-6
    candidate_map = json.loads((listen_dir / "candidate_map.json").read_text(encoding="utf-8"))
    assert candidate_map["candidate.flac"].endswith("instrumental.flac")


def test_resolves_melband_outputs_case_insensitively(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    model_output_dir = tmp_path / "model_output"
    model_output_dir.mkdir()
    vocals_path = model_output_dir / "full_track_Vocals.wav"
    vocals_path.write_bytes(b"stub")

    assert module.resolve_model_output_path(model_output_dir, "vocals") == vocals_path


def test_melband_config_normalization_adds_missing_inference_chunk_size(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    config_path = tmp_path / "config.yaml"
    runtime_config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "audio": {"chunk_size": 485100},
                "inference": {"dim_t": 1101, "num_overlap": 8},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    module.write_config_with_overlap(config_path, runtime_config_path, num_overlap=2)

    runtime_config = yaml.load(runtime_config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    assert runtime_config["inference"]["chunk_size"] == 485100
    assert runtime_config["inference"]["num_overlap"] == 2


def test_melband_runtime_config_applies_memory_safe_batch_without_shrinking_chunk(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    config_path = tmp_path / "config.yaml"
    runtime_config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "audio": {"chunk_size": 352800},
                "model": {"linear_transformer_depth": 0},
                "inference": {"batch_size": 4, "num_overlap": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    effective = module.write_runtime_config(
        config_path,
        runtime_config_path,
        num_overlap=2,
        inference_batch_size=1,
    )
    runtime_config = yaml.load(runtime_config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)

    assert effective == {"batch_size": 1, "chunk_size": 352800, "num_overlap": 2}
    assert runtime_config["audio"]["chunk_size"] == 352800
    assert runtime_config["inference"]["chunk_size"] == 352800
    assert runtime_config["inference"]["batch_size"] == 1
    assert "linear_transformer_depth" not in runtime_config["model"]


def test_melband_runtime_config_rejects_nonzero_unsupported_linear_transformer(tmp_path: Path) -> None:
    module = _load_melband_full_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "audio": {"chunk_size": 352800},
                "model": {"linear_transformer_depth": 1},
                "inference": {"batch_size": 1, "num_overlap": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-zero model.linear_transformer_depth"):
        module.write_runtime_config(
            config_path,
            tmp_path / "runtime.yaml",
            num_overlap=2,
            inference_batch_size=1,
        )
