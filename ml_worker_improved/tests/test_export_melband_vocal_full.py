from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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
