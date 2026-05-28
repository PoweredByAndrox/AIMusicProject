from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import soundfile as sf

from tools import drum_tonal_leakage_repair as tonal_repair


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _drum_pulses(t: np.ndarray, beat_seconds: float) -> np.ndarray:
    pulses = np.zeros_like(t, dtype=np.float32)
    pulse_indices = (np.arange(0.1, float(t[-1]), beat_seconds) * (len(t) / float(t[-1]))).astype(int)
    pulse_indices = pulse_indices[pulse_indices < len(pulses)]
    pulses[pulse_indices] = 1.0
    return np.convolve(pulses, np.hanning(48).astype(np.float32), mode="same")


def _sine_projection_db(audio: np.ndarray, sample_rate: int, frequency_hz: float) -> float:
    mono = audio.mean(axis=1)
    t = np.arange(mono.shape[0], dtype=np.float32) / float(sample_rate)
    basis = np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float32)
    amplitude = float(2.0 * np.mean(mono * basis))
    return 20.0 * np.log10(abs(amplitude) + 1e-12)


def test_repair_reduces_tonal_leakage_while_preserving_drum_peaks(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    tonal_leakage = 0.16 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    input_audio = np.stack([drums + tonal_leakage, drums + tonal_leakage], axis=1)
    input_path = tmp_path / "dirty_drums.flac"
    _write_audio(input_path, input_audio, sample_rate)

    manifest = tonal_repair.repair_drum_tonal_leakage(
        input_path,
        tmp_path / "out",
        listen_dir=tmp_path / "listen",
        output_label="repaired.flac",
        validation_windows={"all": (0.0, duration_sec)},
        config=tonal_repair.TonalRepairConfig(
            n_fft=1024,
            hop_length=256,
            min_tonal_hz=180.0,
            max_tonal_hz=1200.0,
            suppression_strength=0.9,
            min_tonal_gain=0.12,
            harmonic_ratio_floor=0.42,
            transient_guard_strength=0.55,
        ),
    )

    output_audio, output_sample_rate = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    assert output_sample_rate == sample_rate
    input_tone_db = _sine_projection_db(input_audio, sample_rate, 440.0)
    output_tone_db = _sine_projection_db(output_audio, sample_rate, 440.0)
    assert output_tone_db < input_tone_db - 5.0
    assert float(np.max(np.abs(output_audio))) > float(np.max(np.abs(input_audio))) * 0.65
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    candidate_manifest = json.loads((tmp_path / "out" / "stem_manifest.json").read_text(encoding="utf-8"))
    assert candidate_manifest["contract"]["benchmark_compatible"] is True
    assert candidate_manifest["contract"]["target_reference_assisted"] is False
    assert candidate_manifest["contract"]["target_reference_used_at_application"] is False
    assert candidate_manifest["contract"]["application_reference_free"] is True
    assert candidate_manifest["contract"]["delivery_allowed_without_reference"] is True
    assert candidate_manifest["contract"]["fallback_used"] is False
    assert candidate_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert candidate_manifest["contract"]["not_timestamp_hardcoded"] is True
    assert candidate_manifest["stems"]["drums"]["path"].endswith("repaired.flac")
    assert manifest["metrics"]["all"]["delta"]["percussive_minus_harmonic_db"] > 0.0


def test_repair_keeps_clean_percussive_input_close_to_source(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    input_audio = np.stack([drums, drums], axis=1)
    input_path = tmp_path / "clean_drums.flac"
    _write_audio(input_path, input_audio, sample_rate)

    manifest = tonal_repair.repair_drum_tonal_leakage(
        input_path,
        tmp_path / "out",
        config=tonal_repair.TonalRepairConfig(
            n_fft=1024,
            hop_length=256,
            min_tonal_hz=180.0,
            max_tonal_hz=1200.0,
            suppression_strength=0.9,
            min_tonal_gain=0.12,
            harmonic_ratio_floor=0.42,
            transient_guard_strength=0.75,
        ),
    )

    output_audio, _ = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    diff = output_audio - input_audio[: output_audio.shape[0]]
    diff_rms = float(np.sqrt(np.mean(diff * diff)))
    input_rms = float(np.sqrt(np.mean(input_audio * input_audio)))
    assert diff_rms < input_rms * 0.25
    assert manifest["metrics"]["full"]["delta"]["rms_db"] > -2.0


def test_replay_manifest_rebuilds_same_audio_hash(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    tonal_leakage = 0.12 * np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32)
    input_audio = np.stack([drums + tonal_leakage, drums + tonal_leakage], axis=1)
    input_path = tmp_path / "dirty_drums.flac"
    _write_audio(input_path, input_audio, sample_rate)

    original_manifest = tonal_repair.repair_drum_tonal_leakage(
        input_path,
        tmp_path / "original",
        output_label="repaired_formula.flac",
        config=tonal_repair.TonalRepairConfig(
            n_fft=1024,
            hop_length=256,
            min_tonal_hz=180.0,
            max_tonal_hz=1200.0,
            suppression_strength=0.72,
            min_tonal_gain=0.34,
            harmonic_ratio_floor=0.5,
            transient_guard_strength=0.6,
        ),
    )

    replay_manifest = tonal_repair.replay_drum_tonal_leakage_repair(
        Path(tmp_path / "original" / "drum_tonal_leakage_repair_manifest.json"),
        tmp_path / "replayed",
    )

    assert replay_manifest["output"]["sha256"] == original_manifest["output"]["sha256"]
    assert replay_manifest["output"]["path"].endswith("repaired_formula.flac")
    assert replay_manifest["replay"]["source_manifest"].endswith("drum_tonal_leakage_repair_manifest.json")
    assert replay_manifest["replay"]["expected_output_sha256"] == original_manifest["output"]["sha256"]
    assert replay_manifest["replay"]["sha256_matched"] is True
