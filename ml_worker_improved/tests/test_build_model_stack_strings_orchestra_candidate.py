from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_model_stack_strings_orchestra_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_model_stack_strings_orchestra_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sine(sample_rate: int, duration_sec: float, frequency_hz: float, gain: float) -> np.ndarray:
    samples = np.arange(int(sample_rate * duration_sec), dtype=np.float32)
    tone = np.sin(2.0 * np.pi * frequency_hz * samples / sample_rate).astype(np.float32) * gain
    return np.repeat(tone[:, np.newaxis], 2, axis=1)


def _noise_burst(sample_rate: int, duration_sec: float, gain: float) -> np.ndarray:
    audio = np.zeros((int(sample_rate * duration_sec), 2), dtype=np.float32)
    start = int(0.65 * sample_rate)
    end = int(0.78 * sample_rate)
    rng = np.random.default_rng(20260518)
    audio[start:end] = rng.normal(0.0, gain, size=audio[start:end].shape).astype(np.float32)
    return audio


def _projection_rms(audio: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(audio, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    gain = float(np.sum(candidate * ref) / (np.sum(ref * ref) + 1.0e-12))
    projected = ref * gain
    return float(np.sqrt(np.mean(np.square(projected)) + 1.0e-12))


def _write_stem(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    sf.write(path, audio, sample_rate, subtype="PCM_24")
    return {"path": str(path)}


def test_strings_model_stack_candidate_debleeds_non_target_projections() -> None:
    module = _load_module()
    sample_rate = 8_000
    duration_sec = 2.0
    strings = _sine(sample_rate, duration_sec, 990.0, 0.12)
    melody = _sine(sample_rate, duration_sec, 440.0, 0.22)
    vocal = _sine(sample_rate, duration_sec, 660.0, 0.18)
    bass = _sine(sample_rate, duration_sec, 80.0, 0.28)
    drums = _noise_burst(sample_rate, duration_sec, 0.55)
    raw_other_bus = strings + melody * 0.45 + vocal * 0.55 + drums * 0.35 + bass * 0.25

    candidate, diagnostics = module.build_model_stack_strings_orchestra_candidate_audio(
        {
            "other": raw_other_bus,
            "melody": melody,
            "vocals": vocal,
            "drums": drums,
            "bass": bass,
        },
        source_audio=raw_other_bus + melody * 0.5,
        sample_rate=sample_rate,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
        source_tonal_n_fft=512,
        source_tonal_hop_length=128,
        source_tonal_weight=0.20,
        model_stack_weight=1.0,
        vocal_suppression=0.92,
        drum_suppression=0.80,
        bass_suppression=0.90,
        melody_suppression=0.65,
    )

    assert _projection_rms(candidate, vocal) < _projection_rms(raw_other_bus, vocal) * 0.40
    assert _projection_rms(candidate, drums) < _projection_rms(raw_other_bus, drums) * 0.45
    assert _projection_rms(candidate, bass) < _projection_rms(raw_other_bus, bass) * 0.45
    assert _projection_rms(candidate, melody) < _projection_rms(raw_other_bus, melody) * 0.75
    assert _projection_rms(candidate, strings) > _projection_rms(raw_other_bus, strings) * 0.35
    assert diagnostics["route_name"] == "model_stack_strings_orchestra_v1"
    assert diagnostics["projection_removal"]["vocals"]["gain"] > 0.0


def test_strings_model_stack_manifest_requires_no_reference_base_and_writes_strings(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    source_path = tmp_path / "source_mix.flac"
    source_audio = _sine(sample_rate, 1.0, 990.0, 0.12) + _sine(sample_rate, 1.0, 440.0, 0.18)
    sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

    stem_dir = tmp_path / "base"
    stem_dir.mkdir()
    strings = _sine(sample_rate, 1.0, 990.0, 0.12)
    melody = _sine(sample_rate, 1.0, 440.0, 0.18)
    vocals = _sine(sample_rate, 1.0, 660.0, 0.14)
    drums = _noise_burst(sample_rate, 1.0, 0.30)
    bass = _sine(sample_rate, 1.0, 80.0, 0.20)
    base_manifest = {
        "audio_path": str(source_path),
        "audio_sha256": module.sha256_file(source_path),
        "model_name": "synthetic_no_reference_base",
        "sample_rate": sample_rate,
        "contract": {
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
        },
        "pipeline": {
            "mode": "synthetic_base",
            "target_reference_read": False,
            "reference_sum_mix_read": False,
        },
        "stems": {
            "other": _write_stem(stem_dir / "other.flac", strings + melody * 0.25, sample_rate),
            "melody": _write_stem(stem_dir / "melody.flac", melody, sample_rate),
            "vocals": _write_stem(stem_dir / "vocals.flac", vocals, sample_rate),
            "drums": _write_stem(stem_dir / "drums.flac", drums, sample_rate),
            "bass": _write_stem(stem_dir / "bass.flac", bass, sample_rate),
        },
    }
    base_manifest_path = stem_dir / "stem_manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    output_dir = tmp_path / "candidate"
    manifest = module.build_model_stack_strings_orchestra_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        base_manifest_path=base_manifest_path,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
        source_tonal_n_fft=512,
        source_tonal_hop_length=128,
    )

    saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert (output_dir / "strings_orchestra.flac").exists()
    assert saved_manifest["contract"]["target_reference_assisted"] is False
    assert saved_manifest["contract"]["target_reference_used_at_application"] is False
    assert saved_manifest["contract"]["application_reference_free"] is True
    assert saved_manifest["contract"]["fallback_used"] is False
    assert saved_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert saved_manifest["pipeline"]["mode"] == "model_stack_strings_orchestra_v1"
    assert saved_manifest["pipeline"]["target_reference_read"] is False
    assert saved_manifest["pipeline"]["reference_sum_mix_read"] is False
    assert saved_manifest["stems"]["strings_orchestra"]["path"] == str(output_dir / "strings_orchestra.flac")
    assert manifest["manifest_path"] == str(output_dir / "stem_manifest.json")


def test_strings_model_stack_manifest_can_preserve_registered_route_name(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    source_path = tmp_path / "source_mix.flac"
    source_audio = _sine(sample_rate, 1.0, 990.0, 0.12)
    sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

    stem_dir = tmp_path / "base"
    stem_dir.mkdir()
    base_manifest = {
        "audio_path": str(source_path),
        "audio_sha256": module.sha256_file(source_path),
        "model_name": "synthetic_no_reference_base",
        "sample_rate": sample_rate,
        "contract": {
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
        },
        "pipeline": {"target_reference_read": False, "reference_sum_mix_read": False},
        "stems": {
            "other": _write_stem(stem_dir / "other.flac", _sine(sample_rate, 1.0, 990.0, 0.12), sample_rate),
            "drums": _write_stem(stem_dir / "drums.flac", _noise_burst(sample_rate, 1.0, 0.20), sample_rate),
        },
    }
    base_manifest_path = stem_dir / "stem_manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    output_dir = tmp_path / "candidate"
    module.build_model_stack_strings_orchestra_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        base_manifest_path=base_manifest_path,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
        source_tonal_n_fft=512,
        source_tonal_hop_length=128,
        route_name="model_stack_strings_orchestra_hybrid_bs_v1",
    )

    saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["model_id"] == "model_stack_strings_orchestra_hybrid_bs_v1"
    assert saved_manifest["contract"]["route_name"] == "model_stack_strings_orchestra_hybrid_bs_v1"
    assert saved_manifest["pipeline"]["mode"] == "model_stack_strings_orchestra_hybrid_bs_v1"
    assert saved_manifest["pipeline"]["diagnostics"]["base_algorithm_route_name"] == "model_stack_strings_orchestra_v1"
