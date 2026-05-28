from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools import drum_micro_leakage_debleed as micro_debleed


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _tone_projection_db(audio: np.ndarray, sample_rate: int, frequency_hz: float, start_sec: float, end_sec: float) -> float:
    start = int(start_sec * sample_rate)
    end = int(end_sec * sample_rate)
    segment = audio[start:end].mean(axis=1)
    t = np.arange(segment.shape[0], dtype=np.float32) / float(sample_rate)
    basis = np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float32)
    amplitude = float(2.0 * np.mean(segment * basis))
    return 20.0 * np.log10(abs(amplitude) + 1e-12)


def _drum_pulses(t: np.ndarray, beat_seconds: float) -> np.ndarray:
    pulses = np.zeros_like(t, dtype=np.float32)
    for beat_time in np.arange(0.12, float(t[-1]), beat_seconds):
        index = int(beat_time * (len(t) / float(t[-1])))
        if index + 32 < len(pulses):
            pulses[index : index + 32] += np.hanning(32).astype(np.float32)
    return pulses


def test_micro_debleed_reduces_local_leakage_and_preserves_drums(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    leakage_gate = ((t >= 1.0) & (t < 1.8)).astype(np.float32)
    leakage_tone = 0.18 * leakage_gate * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + leakage_tone, drums + leakage_tone], axis=1)
    leakage_bus = np.stack([leakage_tone, leakage_tone], axis=1)

    dirty_path = tmp_path / "dirty_drums.flac"
    leakage_path = tmp_path / "leakage_bus.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, dirty_drums, sample_rate)
    _write_audio(leakage_path, leakage_bus, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[leakage_path],
        output_dir=output_dir,
        listen_dir=tmp_path / "listen",
        raw_ref_path=None,
        output_label="02_micro_debleed.flac",
        input_label="01_dirty_drums.flac",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=1024,
            hop_length=256,
            projection_strength=1.25,
            dynamic_strength=0.35,
            frame_floor=0.025,
            frame_range=0.12,
            max_alpha=0.95,
            max_reduction=0.82,
            transient_protection=0.15,
        ),
    )

    output_audio, output_sample_rate = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    assert output_sample_rate == sample_rate

    input_tone_db = _tone_projection_db(dirty_drums, sample_rate, 660.0, 1.0, 1.8)
    output_tone_db = _tone_projection_db(output_audio, sample_rate, 660.0, 1.0, 1.8)
    assert output_tone_db < input_tone_db - 5.0
    assert float(np.max(np.abs(output_audio))) > float(np.max(np.abs(dirty_drums))) * 0.70

    candidate_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert candidate_manifest["model_id"] == "drum_micro_leakage_debleed"
    assert candidate_manifest["contract"]["benchmark_compatible"] is True
    assert candidate_manifest["contract"]["target_reference_assisted"] is False
    assert candidate_manifest["contract"]["target_reference_used_at_application"] is False
    assert candidate_manifest["contract"]["application_reference_free"] is True
    assert candidate_manifest["contract"]["delivery_allowed_without_reference"] is True
    assert candidate_manifest["contract"]["delivery_allowed_without_quality_gate"] is False
    assert candidate_manifest["contract"]["fallback_used"] is False
    assert candidate_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert candidate_manifest["contract"]["not_timestamp_hardcoded"] is True
    assert candidate_manifest["contract"]["quality_gate_required"] is True
    assert candidate_manifest["stems"]["drums"]["role"] == "drum_micro_leakage_debleed_candidate"


def test_micro_debleed_can_emit_derived_drum_target_stem_name(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 1.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    leakage_tone = 0.03 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + leakage_tone, drums + leakage_tone], axis=1)
    leakage_bus = np.stack([leakage_tone, leakage_tone], axis=1)

    dirty_path = tmp_path / "dirty_drums_plus.flac"
    leakage_path = tmp_path / "leakage_bus.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, dirty_drums, sample_rate)
    _write_audio(leakage_path, leakage_bus, sample_rate)

    micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[leakage_path],
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_drums_plus_debleed.flac",
        input_label="01_dirty_drums_plus.flac",
        stem_name="drums_plus_percussive_fx_aux",
        config=micro_debleed.MicroDebleedConfig(n_fft=512, hop_length=128),
    )

    candidate_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert "drums_plus_percussive_fx_aux" in candidate_manifest["stems"]
    assert "drums" not in candidate_manifest["stems"]
    assert candidate_manifest["stems"]["drums_plus_percussive_fx_aux"]["role"] == "drum_micro_leakage_debleed_candidate"


def test_verified_reference_leakage_mode_marks_candidate_reference_assisted(tmp_path: Path) -> None:
    sample_rate = 8_000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    leakage_tone = 0.05 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + leakage_tone, drums + leakage_tone], axis=1)
    leakage_stem = np.stack([leakage_tone, leakage_tone], axis=1)

    drum_path = tmp_path / "dirty_drums.flac"
    leakage_path = tmp_path / "verified_melody.flac"
    output_dir = tmp_path / "out"
    _write_audio(drum_path, dirty_drums, sample_rate)
    _write_audio(leakage_path, leakage_stem, sample_rate)

    micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=drum_path,
        leakage_stem_paths=[leakage_path],
        source_mix_path=None,
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_reference_assisted.flac",
        input_label="01_input.flac",
        leakage_evidence_mode="verified_reference_stems",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=512,
            hop_length=128,
            projection_strength=0.4,
            frame_floor=0.0,
            frame_range=0.02,
        ),
    )

    diagnostic_manifest = json.loads((output_dir / "drum_micro_leakage_debleed_manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert diagnostic_manifest["inputs"]["leakage_evidence_mode"] == "verified_reference_stems"
    assert diagnostic_manifest["contract"]["application_reference_free"] is False
    assert candidate_manifest["model"]["family"] == "reference_assisted_drum_micro_debleed"
    assert candidate_manifest["inputs"]["leakage_evidence_mode"] == "verified_reference_stems"
    assert candidate_manifest["contract"]["target_reference_assisted"] is True
    assert candidate_manifest["contract"]["target_reference_used_at_application"] is True
    assert candidate_manifest["contract"]["application_reference_free"] is False
    assert candidate_manifest["contract"]["delivery_allowed_without_reference"] is False
    assert candidate_manifest["contract"]["no_reference_production_claim"] is False


def test_source_residual_proxy_requires_source_mix_path(tmp_path: Path) -> None:
    sample_rate = 8_000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    drums = np.stack([_drum_pulses(t, 0.5), _drum_pulses(t, 0.5)], axis=1)
    leakage = np.stack([0.02 * np.sin(2.0 * np.pi * 440.0 * t)] * 2, axis=1).astype(np.float32)
    drum_path = tmp_path / "drums.flac"
    leakage_path = tmp_path / "leakage.flac"
    _write_audio(drum_path, drums, sample_rate)
    _write_audio(leakage_path, leakage, sample_rate)

    with pytest.raises(ValueError, match="source_residual_proxy mode requires --source-mix-path"):
        micro_debleed.build_drum_micro_leakage_debleed(
            drum_stem_path=drum_path,
            leakage_stem_paths=[leakage_path],
            source_mix_path=None,
            output_dir=tmp_path / "out",
            listen_dir=None,
            raw_ref_path=None,
            output_label="out.flac",
            input_label="in.flac",
            leakage_evidence_mode="source_residual_proxy",
            config=micro_debleed.MicroDebleedConfig(n_fft=512, hop_length=128),
        )


def test_model_proxy_rejects_source_mix_path_to_prevent_wrong_provenance(tmp_path: Path) -> None:
    sample_rate = 8_000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    drums = np.stack([_drum_pulses(t, 0.5), _drum_pulses(t, 0.5)], axis=1)
    drum_path = tmp_path / "drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    _write_audio(drum_path, drums, sample_rate)
    _write_audio(source_mix_path, drums, sample_rate)

    with pytest.raises(ValueError, match="model_proxy mode must not use --source-mix-path"):
        micro_debleed.build_drum_micro_leakage_debleed(
            drum_stem_path=drum_path,
            leakage_stem_paths=[],
            source_mix_path=source_mix_path,
            output_dir=tmp_path / "out",
            listen_dir=None,
            raw_ref_path=None,
            output_label="out.flac",
            input_label="in.flac",
            config=micro_debleed.MicroDebleedConfig(n_fft=512, hop_length=128),
        )


def test_invalid_reduction_range_fails_fast(tmp_path: Path) -> None:
    sample_rate = 8_000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    drums = np.stack([_drum_pulses(t, 0.5), _drum_pulses(t, 0.5)], axis=1)
    leakage = np.stack([0.02 * np.sin(2.0 * np.pi * 440.0 * t)] * 2, axis=1).astype(np.float32)
    drum_path = tmp_path / "drums.flac"
    leakage_path = tmp_path / "leakage.flac"
    _write_audio(drum_path, drums, sample_rate)
    _write_audio(leakage_path, leakage, sample_rate)

    with pytest.raises(ValueError, match="max_reduction must be between 0.0 and 1.0"):
        micro_debleed.build_drum_micro_leakage_debleed(
            drum_stem_path=drum_path,
            leakage_stem_paths=[leakage_path],
            source_mix_path=None,
            output_dir=tmp_path / "out",
            listen_dir=None,
            raw_ref_path=None,
            output_label="out.flac",
            input_label="in.flac",
            config=micro_debleed.MicroDebleedConfig(n_fft=512, hop_length=128, max_reduction=1.2),
        )


def test_invalid_max_alpha_range_fails_fast(tmp_path: Path) -> None:
    sample_rate = 8_000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    drums = np.stack([_drum_pulses(t, 0.5), _drum_pulses(t, 0.5)], axis=1)
    leakage = np.stack([0.02 * np.sin(2.0 * np.pi * 440.0 * t)] * 2, axis=1).astype(np.float32)
    drum_path = tmp_path / "drums.flac"
    leakage_path = tmp_path / "leakage.flac"
    _write_audio(drum_path, drums, sample_rate)
    _write_audio(leakage_path, leakage, sample_rate)

    with pytest.raises(ValueError, match="max_alpha must be between 0.0 and 1.0"):
        micro_debleed.build_drum_micro_leakage_debleed(
            drum_stem_path=drum_path,
            leakage_stem_paths=[leakage_path],
            source_mix_path=None,
            output_dir=tmp_path / "out",
            listen_dir=None,
            raw_ref_path=None,
            output_label="out.flac",
            input_label="in.flac",
            config=micro_debleed.MicroDebleedConfig(n_fft=512, hop_length=128, max_alpha=1.5),
        )


def test_source_mix_residual_proxy_can_drive_debleed_without_reference_stems(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    leakage_gate = ((t >= 1.0) & (t < 1.8)).astype(np.float32)
    full_melody = 0.18 * leakage_gate * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
    leaked_melody = 0.08 * leakage_gate * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + leaked_melody, drums + leaked_melody], axis=1)
    source_mix = np.stack([drums + full_melody, drums + full_melody], axis=1)

    dirty_path = tmp_path / "dirty_drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, dirty_drums, sample_rate)
    _write_audio(source_mix_path, source_mix, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[],
        source_mix_path=source_mix_path,
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_source_residual_debleed.flac",
        input_label="01_dirty_drums.flac",
        leakage_evidence_mode="source_residual_proxy",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=1024,
            hop_length=256,
            projection_strength=1.1,
            dynamic_strength=0.25,
            frame_floor=0.02,
            frame_range=0.10,
            max_reduction=0.80,
            transient_protection=0.18,
        ),
    )

    output_audio, _ = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    input_tone_db = _tone_projection_db(dirty_drums, sample_rate, 660.0, 1.0, 1.8)
    output_tone_db = _tone_projection_db(output_audio, sample_rate, 660.0, 1.0, 1.8)
    assert output_tone_db < input_tone_db - 4.0
    candidate_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert candidate_manifest["inputs"]["source_mix_path"].endswith("source_mix.flac")
    assert candidate_manifest["inputs"]["leakage_evidence_mode"] == "source_residual_proxy"
    assert candidate_manifest["contract"]["target_reference_assisted"] is False
    assert candidate_manifest["contract"]["target_reference_used_at_application"] is False
    assert candidate_manifest["contract"]["application_reference_free"] is True
    assert candidate_manifest["contract"]["delivery_allowed_without_reference"] is True
    assert candidate_manifest["contract"]["no_reference_production_claim"] is True


def test_length_audit_records_trim_to_shortest_policy(tmp_path: Path) -> None:
    sample_rate = 8_000
    drum_t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    source_t = np.linspace(0.0, 1.5, int(sample_rate * 1.5), endpoint=False, dtype=np.float32)
    drums = np.stack([_drum_pulses(drum_t, 0.5), _drum_pulses(drum_t, 0.5)], axis=1)
    source_mix = np.stack([_drum_pulses(source_t, 0.5), _drum_pulses(source_t, 0.5)], axis=1)
    drum_path = tmp_path / "drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "out"
    _write_audio(drum_path, drums, sample_rate)
    _write_audio(source_mix_path, source_mix, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=drum_path,
        leakage_stem_paths=[],
        source_mix_path=source_mix_path,
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="out.flac",
        input_label="in.flac",
        leakage_evidence_mode="source_residual_proxy",
        config=micro_debleed.MicroDebleedConfig(n_fft=512, hop_length=128, projection_strength=0.1),
    )

    length_audit = manifest["diagnostics"]["length_audit"]
    assert length_audit["policy"] == "trim_to_shortest_input"
    assert length_audit["aligned_frames"] == source_mix.shape[0]
    assert length_audit["inputs"]["drums"]["trimmed_frames"] == drums.shape[0] - source_mix.shape[0]
    assert length_audit["inputs"]["source_mix"]["trimmed_frames"] == 0


def test_nonpercussive_source_mix_gate_suppresses_fx_only_regions(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    fx_only_gate = ((t >= 1.05) & (t < 1.55)).astype(np.float32)
    fx_tone = 0.16 * fx_only_gate * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + fx_tone, drums + fx_tone], axis=1)
    source_mix = np.stack([drums + fx_tone, drums + fx_tone], axis=1)

    dirty_path = tmp_path / "dirty_drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, dirty_drums, sample_rate)
    _write_audio(source_mix_path, source_mix, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[],
        source_mix_path=source_mix_path,
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_source_residual_gate.flac",
        input_label="01_dirty_drums.flac",
        leakage_evidence_mode="source_residual_proxy",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=1024,
            hop_length=256,
            projection_strength=0.2,
            dynamic_strength=0.0,
            frame_floor=0.02,
            frame_range=0.10,
            max_reduction=0.50,
            transient_protection=0.20,
            nonpercussive_gate_strength=1.0,
            nonpercussive_gate_min_gain=0.02,
            nonpercussive_gate_percussive_ceiling=0.34,
            nonpercussive_gate_harmonic_floor=0.35,
        ),
    )

    output_audio, _ = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    input_fx_db = _tone_projection_db(dirty_drums, sample_rate, 660.0, 1.05, 1.55)
    output_fx_db = _tone_projection_db(output_audio, sample_rate, 660.0, 1.05, 1.55)
    assert output_fx_db < input_fx_db - 12.0

    input_hit_peak = float(np.max(np.abs(dirty_drums[int(0.10 * sample_rate) : int(0.16 * sample_rate)])))
    output_hit_peak = float(np.max(np.abs(output_audio[int(0.10 * sample_rate) : int(0.16 * sample_rate)])))
    assert output_hit_peak > input_hit_peak * 0.72

    assert manifest["config"]["nonpercussive_gate_strength"] == 1.0
    assert manifest["diagnostics"]["channels"][0]["p95_nonpercussive_gate"] > 0.0


def test_low_activity_gate_suppresses_quiet_drop_residue_without_muting_hits(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    quiet_fx_gate = ((t >= 1.05) & (t < 1.55)).astype(np.float32)
    quiet_fx = 0.035 * quiet_fx_gate * np.sin(2.0 * np.pi * 880.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + quiet_fx, drums + quiet_fx], axis=1)
    source_mix = np.stack([drums + quiet_fx, drums + quiet_fx], axis=1)

    dirty_path = tmp_path / "dirty_drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, dirty_drums, sample_rate)
    _write_audio(source_mix_path, source_mix, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[],
        source_mix_path=source_mix_path,
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_low_activity_gate.flac",
        input_label="01_dirty_drums.flac",
        leakage_evidence_mode="source_residual_proxy",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=1024,
            hop_length=256,
            projection_strength=0.0,
            dynamic_strength=0.0,
            max_reduction=0.50,
            low_activity_gate_strength=1.0,
            low_activity_gate_min_gain=0.03,
            low_activity_gate_threshold_dbfs=-22.0,
            low_activity_gate_range_db=7.0,
        ),
    )

    output_audio, _ = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    input_fx_db = _tone_projection_db(dirty_drums, sample_rate, 880.0, 1.05, 1.55)
    output_fx_db = _tone_projection_db(output_audio, sample_rate, 880.0, 1.05, 1.55)
    assert output_fx_db < input_fx_db - 10.0

    input_hit_peak = float(np.max(np.abs(dirty_drums[int(0.10 * sample_rate) : int(0.16 * sample_rate)])))
    output_hit_peak = float(np.max(np.abs(output_audio[int(0.10 * sample_rate) : int(0.16 * sample_rate)])))
    assert output_hit_peak > input_hit_peak * 0.80
    assert manifest["diagnostics"]["channels"][0]["p95_low_activity_gate"] > 0.0


def test_parse_args_exposes_frequency_band_controls(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drum_micro_leakage_debleed.py",
            "input.flac",
            "out",
            "--min-freq-hz",
            "35",
            "--max-freq-hz",
            "520",
            "--min-freq-ramp-hz",
            "80",
            "--max-freq-ramp-hz",
            "300",
            "--leakage-evidence-mode",
            "source_residual_proxy",
            "--post-transient-tail-gate-strength",
            "0.45",
            "--post-transient-tail-onset-percentile",
            "92",
        ],
    )

    args = micro_debleed.parse_args()

    assert args.min_freq_hz == 35.0
    assert args.max_freq_hz == 520.0
    assert args.min_freq_ramp_hz == 80.0
    assert args.max_freq_ramp_hz == 300.0
    assert args.leakage_evidence_mode == "source_residual_proxy"
    assert args.post_transient_tail_gate_strength == 0.45
    assert args.post_transient_tail_onset_percentile == 92.0


def test_main_preserves_full_cli_command_args_in_manifest(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8000
    duration_sec = 1.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = np.stack([_drum_pulses(t, 0.5), _drum_pulses(t, 0.5)], axis=1)

    drum_path = tmp_path / "drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "out"
    _write_audio(drum_path, drums, sample_rate)
    _write_audio(source_mix_path, drums, sample_rate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drum_micro_leakage_debleed.py",
            str(drum_path),
            str(output_dir),
            "--source-mix-path",
            str(source_mix_path),
            "--leakage-evidence-mode",
            "source_residual_proxy",
            "--post-transient-tail-gate-strength",
            "0.45",
        ],
    )

    micro_debleed.main()

    manifest = json.loads((output_dir / "drum_micro_leakage_debleed_manifest.json").read_text(encoding="utf-8"))
    command_args = manifest["provenance"]["command_args"]
    assert "--source-mix-path" in command_args
    assert "--leakage-evidence-mode" in command_args
    assert "--post-transient-tail-gate-strength" in command_args
    assert "0.45" in command_args


def test_low_band_projection_reduces_kick_synchronous_bass_leakage(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    kick_window = ((t >= 1.0) & (t < 1.5)).astype(np.float32)
    bass_leakage = 0.14 * kick_window * np.sin(2.0 * np.pi * 95.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + bass_leakage, drums + bass_leakage], axis=1)
    bass_proxy = np.stack([bass_leakage, bass_leakage], axis=1)

    dirty_path = tmp_path / "dirty_drums.flac"
    bass_proxy_path = tmp_path / "bass_proxy.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, dirty_drums, sample_rate)
    _write_audio(bass_proxy_path, bass_proxy, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[bass_proxy_path],
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_low_band_projection.flac",
        input_label="01_dirty_drums.flac",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=1024,
            hop_length=256,
            min_freq_hz=35.0,
            max_freq_hz=520.0,
            min_freq_ramp_hz=80.0,
            max_freq_ramp_hz=300.0,
            projection_strength=0.0,
            dynamic_strength=0.0,
            low_band_projection_strength=3.0,
            low_band_projection_max_reduction=0.90,
            low_band_projection_frame_floor=0.0,
            low_band_projection_frame_range=0.02,
            transient_protection=0.20,
        ),
    )

    output_audio, _ = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    input_bass_db = _tone_projection_db(dirty_drums, sample_rate, 95.0, 1.0, 1.5)
    output_bass_db = _tone_projection_db(output_audio, sample_rate, 95.0, 1.0, 1.5)
    assert output_bass_db < input_bass_db - 6.0

    input_hit_peak = float(np.max(np.abs(dirty_drums[int(0.10 * sample_rate) : int(0.16 * sample_rate)])))
    output_hit_peak = float(np.max(np.abs(output_audio[int(0.10 * sample_rate) : int(0.16 * sample_rate)])))
    assert output_hit_peak > input_hit_peak * 0.78
    input_overlap_hit_peak = float(np.max(np.abs(dirty_drums[int(1.10 * sample_rate) : int(1.16 * sample_rate)])))
    output_overlap_hit_peak = float(np.max(np.abs(output_audio[int(1.10 * sample_rate) : int(1.16 * sample_rate)])))
    assert output_overlap_hit_peak > input_overlap_hit_peak * 0.55
    assert manifest["diagnostics"]["channels"][0]["p95_low_band_projection_gate"] > 0.0


def test_post_transient_tail_gate_reduces_tonal_tail_after_kick_attack(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 3.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 1.0)
    tail_gate = ((t >= 1.05) & (t < 1.30)).astype(np.float32)
    tonal_tail = 0.16 * tail_gate * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    dirty_drums = np.stack([drums + tonal_tail, drums + tonal_tail], axis=1)
    source_mix = np.stack([drums + 2.0 * tonal_tail, drums + 2.0 * tonal_tail], axis=1)

    dirty_path = tmp_path / "dirty_drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, dirty_drums, sample_rate)
    _write_audio(source_mix_path, source_mix, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[],
        source_mix_path=source_mix_path,
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_post_transient_tail_gate.flac",
        input_label="01_dirty_drums.flac",
        leakage_evidence_mode="source_residual_proxy",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=1024,
            hop_length=128,
            min_freq_hz=120.0,
            max_freq_hz=900.0,
            min_freq_ramp_hz=120.0,
            max_freq_ramp_hz=420.0,
            projection_strength=0.0,
            dynamic_strength=0.0,
            post_transient_tail_gate_strength=0.88,
            post_transient_tail_gate_min_gain=0.08,
            post_transient_tail_start_ms=35.0,
            post_transient_tail_end_ms=260.0,
            post_transient_tail_harmonic_floor=0.42,
            post_transient_tail_leakage_floor=0.0,
            post_transient_tail_leakage_range=0.02,
            post_transient_tail_projection_floor=0.0,
            post_transient_tail_projection_range=0.05,
        ),
    )

    output_audio, _ = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    input_tail_db = _tone_projection_db(dirty_drums, sample_rate, 220.0, 1.08, 1.28)
    output_tail_db = _tone_projection_db(output_audio, sample_rate, 220.0, 1.08, 1.28)
    assert output_tail_db < input_tail_db - 7.0

    input_attack_peak = float(np.max(np.abs(dirty_drums[int(1.10 * sample_rate) : int(1.15 * sample_rate)])))
    output_attack_peak = float(np.max(np.abs(output_audio[int(1.10 * sample_rate) : int(1.15 * sample_rate)])))
    assert output_attack_peak > input_attack_peak * 0.80
    assert manifest["diagnostics"]["channels"][0]["p95_post_transient_tail_gate"] > 0.0


def test_post_transient_tail_gate_requires_proxy_evidence(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 3.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 1.0)
    tail_gate = ((t >= 1.05) & (t < 1.30)).astype(np.float32)
    clean_drum_tail = 0.16 * tail_gate * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    clean_drums = np.stack([drums + clean_drum_tail, drums + clean_drum_tail], axis=1)

    dirty_path = tmp_path / "clean_harmonic_drums.flac"
    source_mix_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "out"
    _write_audio(dirty_path, clean_drums, sample_rate)
    _write_audio(source_mix_path, clean_drums, sample_rate)

    manifest = micro_debleed.build_drum_micro_leakage_debleed(
        drum_stem_path=dirty_path,
        leakage_stem_paths=[],
        source_mix_path=source_mix_path,
        output_dir=output_dir,
        listen_dir=None,
        raw_ref_path=None,
        output_label="02_post_transient_tail_gate.flac",
        input_label="01_clean_harmonic_drums.flac",
        leakage_evidence_mode="source_residual_proxy",
        config=micro_debleed.MicroDebleedConfig(
            n_fft=1024,
            hop_length=128,
            min_freq_hz=120.0,
            max_freq_hz=900.0,
            min_freq_ramp_hz=120.0,
            max_freq_ramp_hz=420.0,
            projection_strength=0.0,
            dynamic_strength=0.0,
            post_transient_tail_gate_strength=0.88,
            post_transient_tail_gate_min_gain=0.08,
            post_transient_tail_start_ms=35.0,
            post_transient_tail_end_ms=260.0,
            post_transient_tail_harmonic_floor=0.42,
        ),
    )

    output_audio, _ = sf.read(manifest["output"]["path"], always_2d=True, dtype="float32")
    input_tail_db = _tone_projection_db(clean_drums, sample_rate, 220.0, 1.08, 1.28)
    output_tail_db = _tone_projection_db(output_audio, sample_rate, 220.0, 1.08, 1.28)
    assert output_tail_db > input_tail_db - 1.0
    assert manifest["diagnostics"]["channels"][0]["p95_post_transient_tail_gate"] == 0.0
