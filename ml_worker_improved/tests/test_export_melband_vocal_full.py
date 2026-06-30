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
    )

    assert "preview_window_sec" not in manifest
    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "melband_roformer_vocals"
    assert manifest["model_name"] == "melband_roformer_vocals"
    assert manifest["contract"]["full_track"] is True
    assert manifest["contract"]["benchmark_compatible"] is True
    assert manifest["contract"]["fallback_used"] is False
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
        },
    )

    assert manifest["execution_contract"]["expected_acceleration_mode"] == "gpu_inference_then_cpu_dsp_scoring"
    assert manifest["execution_contract"]["gpu_inference_required"] is True
    assert manifest["execution_contract"]["gpu_inference_skipped"] is False
    assert manifest["execution_contract"]["cuda_preflight"]["device_name"] == "Unit Test GPU"


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
    }


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
                "audio": {"chunk_size": 44100},
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
    monkeypatch.setattr(module, "save_audio_window_as_wav", fake_save_audio_window_as_wav)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
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
                "audio": {"chunk_size": 44100},
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
    monkeypatch.setattr(module.subprocess, "run", fake_run)
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
