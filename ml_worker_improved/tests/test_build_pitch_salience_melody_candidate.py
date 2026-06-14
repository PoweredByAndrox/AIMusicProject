from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_no_reference_stem_listen_preview_manifest import build_manifest as build_stem_listen_manifest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_pitch_salience_melody_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_pitch_salience_melody_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sine(sample_rate: int, duration_sec: float, frequency_hz: float, gain: float) -> np.ndarray:
    samples = np.arange(int(sample_rate * duration_sec), dtype=np.float32)
    tone = np.sin(2.0 * np.pi * frequency_hz * samples / sample_rate).astype(np.float32) * gain
    return np.repeat(tone[:, np.newaxis], 2, axis=1)


def _noise(sample_rate: int, duration_sec: float, gain: float) -> np.ndarray:
    rng = np.random.default_rng(1729)
    return rng.normal(0.0, gain, size=(int(sample_rate * duration_sec), 2)).astype(np.float32)


def _write_stem(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    sf.write(path, audio, sample_rate, subtype="PCM_24")
    return {"path": str(path)}


def _base_contract() -> dict[str, object]:
    return {
        "target_reference_assisted": False,
        "target_reference_used": False,
        "target_reference_used_at_application": False,
        "application_reference_free": True,
        "delivery_allowed_without_reference": True,
        "registered_model_candidate": True,
        "fallback_used": False,
        "hidden_inference": False,
        "per_track_hardcode_allowed": False,
        "not_timestamp_hardcoded": True,
    }


def _gpu_execution_contract() -> dict[str, object]:
    return {
        "docker_runtime_required": True,
        "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
        "gpu_inference_required": True,
        "gpu_inference_skipped": False,
        "cuda_preflight": {
            "cuda_available": True,
            "requested_device": "cuda",
            "resolved_device": "cuda",
            "device_count": 1,
            "device_name": "test gpu",
        },
    }


def test_pitch_salience_candidate_improves_pitch_confidence_without_reference() -> None:
    module = _load_module()
    sample_rate = 8000
    duration_sec = 2.0
    melody = _sine(sample_rate, duration_sec, 223.7, 0.012)
    vocals = _sine(sample_rate, duration_sec, 660.0, 0.05)
    drums = _noise(sample_rate, duration_sec, 0.12)
    bass = _sine(sample_rate, duration_sec, 90.0, 0.04)
    base_melody = melody + vocals * 0.12 + drums * 1.20 + bass * 0.10
    source_audio = melody + vocals + drums + bass

    before = module._pitch_confidence_like_semantic_judge(base_melody, sample_rate)
    candidate, diagnostics = module.build_pitch_salience_melody_candidate_audio(
        {
            "melody": base_melody,
            "vocals": vocals,
            "drums": drums,
            "bass": bass,
        },
        source_audio=source_audio,
        sample_rate=sample_rate,
        min_frequency_hz=80.0,
        max_frequency_hz=800.0,
        max_harmonic_hz=1600.0,
        bandwidth_ratio=0.04,
    )
    after = module._pitch_confidence_like_semantic_judge(candidate, sample_rate)

    assert diagnostics["route_name"] == "pitch_salience_melody_v1"
    assert diagnostics["pitch_confidence_before"]["pitch_confidence"] == before["pitch_confidence"]
    assert after["pitch_confidence"] > before["pitch_confidence"]
    assert after["pitch_confidence"] >= 0.35
    assert diagnostics["pitch_confidence_after"]["pitch_confidence"] == after["pitch_confidence"]
    assert "drums" in diagnostics["projection_debleed"]


def test_pitch_salience_default_ignores_sub_bass_fundamental() -> None:
    module = _load_module()
    sample_rate = 8000
    duration_sec = 2.0
    bass_like = _sine(sample_rate, duration_sec, 57.0, 0.16)
    melody = _sine(sample_rate, duration_sec, 220.0, 0.045)
    vocals = _sine(sample_rate, duration_sec, 660.0, 0.02)
    drums = _noise(sample_rate, duration_sec, 0.01)
    base_melody = bass_like + melody + drums
    source_audio = base_melody + vocals

    _candidate, diagnostics = module.build_pitch_salience_melody_candidate_audio(
        {
            "melody": base_melody,
            "vocals": vocals,
            "drums": drums,
            "bass": bass_like,
        },
        source_audio=source_audio,
        sample_rate=sample_rate,
    )

    assert diagnostics["parameters"]["min_frequency_hz"] == 80.0
    assert diagnostics["dominant_frequency_hz"] >= 80.0


def test_pitch_salience_manifest_inherits_gpu_provenance_and_passes_listen_gate(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    source_path = tmp_path / "source_mix.wav"
    melody = _sine(sample_rate, 1.0, 220.0, 0.08)
    vocals = _sine(sample_rate, 1.0, 660.0, 0.05)
    drums = _noise(sample_rate, 1.0, 0.035)
    bass = _sine(sample_rate, 1.0, 90.0, 0.04)
    sf.write(source_path, melody + vocals + drums + bass, sample_rate, subtype="PCM_16")

    stem_dir = tmp_path / "base"
    stem_dir.mkdir()
    base_manifest = {
        "audio_path": str(source_path),
        "audio_sha256": module.sha256_file(source_path),
        "model_name": "synthetic_pitch_salience_base",
        "sample_rate": sample_rate,
        "contract": _base_contract(),
        "execution_contract": _gpu_execution_contract(),
        "pipeline": {
            "target_reference_read": False,
            "reference_sum_mix_read": False,
        },
        "stems": {
            "melody": _write_stem(stem_dir / "melody.flac", melody + drums * 0.90, sample_rate),
            "vocals": _write_stem(stem_dir / "vocals.flac", vocals, sample_rate),
            "drums": _write_stem(stem_dir / "drums.flac", drums, sample_rate),
            "bass": _write_stem(stem_dir / "bass.flac", bass, sample_rate),
        },
    }
    base_manifest_path = stem_dir / "stem_manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    output_dir = tmp_path / "candidate"
    listen_dir = tmp_path / "listen"
    manifest = module.build_pitch_salience_melody_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        base_manifest_path=base_manifest_path,
        listen_dir=listen_dir,
        min_frequency_hz=80.0,
        max_frequency_hz=800.0,
        max_harmonic_hz=1600.0,
    )

    saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert (output_dir / "melody.flac").exists()
    assert (listen_dir / "01_pitch_salience_melody_v1.flac").exists()
    listen_source, listen_source_sample_rate = sf.read(listen_dir / "00_source_mix.flac", always_2d=True)
    assert sf.info(listen_dir / "00_source_mix.flac").format == "FLAC"
    assert listen_source_sample_rate == sample_rate
    assert listen_source.shape == melody.shape
    assert saved_manifest["contract"]["target_reference_used_at_application"] is False
    assert saved_manifest["contract"]["route_name"] == "pitch_salience_melody_v1"
    assert saved_manifest["execution_contract"]["gpu_inference_required"] is True
    assert saved_manifest["execution_contract"]["gpu_inference_skipped"] is False
    assert saved_manifest["execution_contract"]["cuda_preflight"]["resolved_device"] == "cuda"
    assert saved_manifest["target_stem_artifact"]["path"] == str(output_dir / "melody.flac")
    assert saved_manifest["pipeline"]["source_resampled"] is False
    listen_manifest = build_stem_listen_manifest(
        stem_manifest_json=output_dir / "stem_manifest.json",
        target_stem="melody",
        track_id="track-a",
        min_duration_sec=0.25,
    )
    assert listen_manifest["delivery_status"] == "accepted"
    assert listen_manifest["gpu_provenance_gate"]["status"] == "passed"
    assert manifest["manifest_path"] == str(output_dir / "stem_manifest.json")


def test_pitch_salience_route_name_override_is_manifested(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    source_path = tmp_path / "source_mix.flac"
    melody = _sine(sample_rate, 1.0, 220.0, 0.08)
    sf.write(source_path, melody, sample_rate, subtype="PCM_24")

    stem_dir = tmp_path / "base"
    stem_dir.mkdir()
    base_manifest = {
        "audio_path": str(source_path),
        "audio_sha256": module.sha256_file(source_path),
        "model_name": "synthetic_pitch_salience_base",
        "sample_rate": sample_rate,
        "contract": _base_contract(),
        "execution_contract": _gpu_execution_contract(),
        "pipeline": {
            "target_reference_read": False,
            "reference_sum_mix_read": False,
        },
        "stems": {
            "melody": _write_stem(stem_dir / "melody.flac", melody, sample_rate),
        },
    }
    base_manifest_path = stem_dir / "stem_manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    output_dir = tmp_path / "candidate"
    listen_dir = tmp_path / "listen"
    manifest = module.build_pitch_salience_melody_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        base_manifest_path=base_manifest_path,
        listen_dir=listen_dir,
        route_name="pitch_salience_melody_v2",
    )

    saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_name"] == "pitch_salience_melody_v2"
    assert saved_manifest["contract"]["route_name"] == "pitch_salience_melody_v2"
    assert saved_manifest["execution_contract"]["route_name"] == "pitch_salience_melody_v2"
    assert saved_manifest["pipeline"]["diagnostics"]["route_name"] == "pitch_salience_melody_v2"
    assert (listen_dir / "01_pitch_salience_melody_v2.flac").exists()
