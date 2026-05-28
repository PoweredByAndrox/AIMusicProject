from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_tonal_harmonic_melody_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_tonal_harmonic_melody_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sine(sample_rate: int, duration_sec: float, frequency_hz: float, gain: float) -> np.ndarray:
    samples = np.arange(int(sample_rate * duration_sec), dtype=np.float32)
    tone = np.sin(2.0 * np.pi * frequency_hz * samples / sample_rate).astype(np.float32) * gain
    return np.repeat(tone[:, np.newaxis], 2, axis=1)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float32))) + 1e-12))


def test_tonal_candidate_keeps_sustained_melody_and_reduces_drum_burst() -> None:
    module = _load_module()
    sample_rate = 8000
    melody = _sine(sample_rate, 2.0, 440.0, 0.25)
    drum = np.zeros_like(melody)
    burst_start = int(0.85 * sample_rate)
    burst_end = int(0.95 * sample_rate)
    rng = np.random.default_rng(123)
    drum[burst_start:burst_end] = rng.normal(0.0, 0.55, size=drum[burst_start:burst_end].shape).astype(np.float32)
    source = melody + drum

    candidate, diagnostics = module.build_tonal_harmonic_melody_candidate_audio(
        source,
        sample_rate=sample_rate,
        n_fft=512,
        hop_length=128,
        harmonic_margin=3.0,
        percussive_margin=6.0,
        percussive_suppression=0.82,
        highpass_hz=140.0,
        lowpass_hz=3500.0,
    )

    quiet_slice = slice(int(0.20 * sample_rate), int(0.70 * sample_rate))
    drum_slice = slice(burst_start, burst_end)
    assert _rms(candidate[quiet_slice]) > 0.12
    assert _rms(candidate[drum_slice]) < _rms(source[drum_slice]) * 0.55
    assert diagnostics["route_name"] == "tonal_harmonic_melody_v1"
    assert diagnostics["percussive_suppression"] == 0.82


def test_manifest_declares_no_reference_route_and_writes_melody(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    source_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "candidate"
    audio = _sine(sample_rate, 1.0, 330.0, 0.2)
    sf.write(source_path, audio, sample_rate, subtype="PCM_24")

    manifest = module.build_tonal_harmonic_melody_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        n_fft=512,
        hop_length=128,
        highpass_hz=120.0,
        lowpass_hz=3500.0,
    )

    manifest_path = output_dir / "stem_manifest.json"
    melody_path = output_dir / "melody.flac"
    assert manifest_path.exists()
    assert melody_path.exists()
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["contract"]["target_reference_assisted"] is False
    assert saved_manifest["contract"]["target_reference_used"] is False
    assert saved_manifest["contract"]["target_reference_used_at_application"] is False
    assert saved_manifest["contract"]["application_reference_free"] is True
    assert saved_manifest["contract"]["fallback_used"] is False
    assert saved_manifest["contract"]["hidden_inference"] is False
    assert saved_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert saved_manifest["contract"]["no_reference_production_claim"] is True
    assert saved_manifest["pipeline"]["source"] == "authoritative_full_mix_only"
    assert saved_manifest["pipeline"]["target_reference_read"] is False
    assert saved_manifest["pipeline"]["reference_sum_mix_read"] is False
    assert saved_manifest["pipeline"]["mode"] == "tonal_harmonic_melody_v1"
    assert saved_manifest["stems"]["melody"]["path"] == str(melody_path)
    assert manifest["manifest_path"] == str(manifest_path)
