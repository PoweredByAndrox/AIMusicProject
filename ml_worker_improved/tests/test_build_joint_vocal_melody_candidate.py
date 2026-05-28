from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_joint_vocal_melody_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_joint_vocal_melody_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sine(sample_rate: int, duration_sec: float, frequency_hz: float, gain: float) -> np.ndarray:
    samples = np.arange(int(sample_rate * duration_sec), dtype=np.float32)
    tone = np.sin(2.0 * np.pi * frequency_hz * samples / sample_rate).astype(np.float32) * gain
    return np.repeat(tone[:, np.newaxis], 2, axis=1)


def _projection_rms(audio: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(audio, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    gain = float(np.sum(candidate * ref) / (np.sum(ref * ref) + 1e-12))
    projected = ref * gain
    return float(np.sqrt(np.mean(np.square(projected)) + 1e-12))


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


def test_joint_candidate_separates_vocal_and_melody_cross_leakage() -> None:
    module = _load_module()
    sample_rate = 8000
    duration_sec = 2.0
    vocal = _sine(sample_rate, duration_sec, 660.0, 0.24)
    melody = _sine(sample_rate, duration_sec, 440.0, 0.28)
    raw_vocals = vocal + melody * 0.60
    raw_melody = melody + vocal * 0.70
    source_audio = vocal + melody

    outputs, diagnostics = module.build_joint_vocal_melody_candidate_audio(
        {
            "vocals": raw_vocals,
            "melody": raw_melody,
        },
        source_audio=source_audio,
        sample_rate=sample_rate,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
        source_tonal_n_fft=512,
        source_tonal_hop_length=128,
        vocal_from_melody_suppression=0.92,
        melody_from_vocal_suppression=0.94,
    )

    vocals = outputs["vocals"]
    melody_out = outputs["melody"]
    assert _projection_rms(melody_out, vocal) < _projection_rms(raw_melody, vocal) * 0.35
    assert _projection_rms(vocals, melody) < _projection_rms(raw_vocals, melody) * 0.40
    assert _projection_rms(melody_out, melody) > _projection_rms(raw_melody, melody) * 0.55
    assert _projection_rms(vocals, vocal) > _projection_rms(raw_vocals, vocal) * 0.55
    assert diagnostics["route_name"] == "joint_vocal_melody_v1"
    assert diagnostics["melody_builder"]["projection_removal"]["vocals"]["strength"] > 0.0
    assert "melody_from_vocals" in diagnostics["cross_debleed"]
    assert "vocals_from_melody" in diagnostics["cross_debleed"]


def test_joint_manifest_writes_separate_vocals_and_melody_without_target_reference(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    source_path = tmp_path / "source_mix.flac"
    vocal = _sine(sample_rate, 1.0, 660.0, 0.22)
    melody = _sine(sample_rate, 1.0, 440.0, 0.25)
    sf.write(source_path, vocal + melody, sample_rate, subtype="PCM_24")

    stem_dir = tmp_path / "base"
    stem_dir.mkdir()
    base_manifest = {
        "audio_path": str(source_path),
        "audio_sha256": module.sha256_file(source_path),
        "model_name": "synthetic_joint_base",
        "sample_rate": sample_rate,
        "contract": _base_contract(),
        "pipeline": {
            "target_reference_read": False,
            "reference_sum_mix_read": False,
        },
        "stems": {
            "vocals": _write_stem(stem_dir / "vocals.flac", vocal + melody * 0.35, sample_rate),
            "melody": _write_stem(stem_dir / "melody.flac", melody + vocal * 0.45, sample_rate),
        },
    }
    base_manifest_path = stem_dir / "stem_manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    output_dir = tmp_path / "candidate"
    manifest = module.build_joint_vocal_melody_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        base_manifest_path=base_manifest_path,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
        source_tonal_n_fft=512,
        source_tonal_hop_length=128,
    )

    saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert (output_dir / "vocals.flac").exists()
    assert (output_dir / "melody.flac").exists()
    assert saved_manifest["contract"]["target_reference_assisted"] is False
    assert saved_manifest["contract"]["target_reference_used"] is False
    assert saved_manifest["contract"]["target_reference_used_at_application"] is False
    assert saved_manifest["contract"]["application_reference_free"] is True
    assert saved_manifest["contract"]["fallback_used"] is False
    assert saved_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert saved_manifest["pipeline"]["mode"] == "joint_vocal_melody_v1"
    assert saved_manifest["pipeline"]["target_reference_read"] is False
    assert saved_manifest["pipeline"]["reference_sum_mix_read"] is False
    assert saved_manifest["pipeline"]["base_manifest_contract_checked"] is True
    assert saved_manifest["stems"]["vocals"]["path"] == str(output_dir / "vocals.flac")
    assert saved_manifest["stems"]["melody"]["path"] == str(output_dir / "melody.flac")
    assert manifest["manifest_path"] == str(output_dir / "stem_manifest.json")
