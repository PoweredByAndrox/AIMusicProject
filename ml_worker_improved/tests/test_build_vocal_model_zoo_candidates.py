from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_vocal_model_zoo_candidates import (  # noqa: E402
    CandidateArtifact,
    apply_multimodel_support_gate,
    build_focused_detail_recovery_candidate,
    calculate_delta_metrics,
    write_listen_pack,
)
from export_dereverb_vocal_preview import validate_preview_window  # noqa: E402


def _band_rms(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    mono = audio.mean(axis=1)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / sample_rate)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.abs(spectrum[band])))))


def test_multimodel_support_gate_suppresses_unsupported_high_band(tmp_path: Path) -> None:
    sample_rate = 8000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal_tone = 0.22 * np.sin(2.0 * np.pi * 440.0 * times)
    unsupported_tone = 0.16 * np.sin(2.0 * np.pi * 3000.0 * times)

    primary = np.stack([vocal_tone + unsupported_tone, vocal_tone + unsupported_tone], axis=1).astype(np.float32)
    support = np.stack([vocal_tone, vocal_tone], axis=1).astype(np.float32)

    primary_path = tmp_path / "primary.flac"
    support_path = tmp_path / "support.flac"
    output_path = tmp_path / "cleaned.flac"
    sf.write(str(primary_path), primary, sample_rate, subtype="PCM_24")
    sf.write(str(support_path), support, sample_rate, subtype="PCM_24")

    debug = apply_multimodel_support_gate(
        primary_vocals_path=primary_path,
        support_vocals_paths=[support_path],
        output_vocals_path=output_path,
        support_mode="max",
        support_threshold=0.18,
        mask_power=1.4,
        inactive_floor=0.02,
        active_floor=0.25,
        vocal_core_floor=0.25,
        high_band_floor_scale=0.20,
        high_band_start_hz=1800.0,
        activity_floor_ratio=0.05,
        activity_smoothing_frames=3,
        tail_memory_sec=0.1,
        target_rms_ratio=1.0,
        max_gain=1.0,
    )

    cleaned, cleaned_sr = sf.read(str(output_path), always_2d=True, dtype="float32")
    assert cleaned_sr == sample_rate
    assert debug["support_mode"] == "max"
    assert _band_rms(cleaned, sample_rate, 2500.0, 3400.0) < _band_rms(primary, sample_rate, 2500.0, 3400.0)
    assert _band_rms(cleaned, sample_rate, 350.0, 550.0) > 0.0


def test_drop_safe_gate_preserves_vocal_core_when_support_dips(tmp_path: Path) -> None:
    sample_rate = 8000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal_tone = 0.24 * np.sin(2.0 * np.pi * 440.0 * times)
    unsupported_tone = 0.14 * np.sin(2.0 * np.pi * 3000.0 * times)
    support_tone = vocal_tone.copy()
    drop_slice = slice(int(0.35 * sample_rate), int(0.70 * sample_rate))
    support_tone[drop_slice] *= 0.18

    primary = np.stack([vocal_tone + unsupported_tone, vocal_tone + unsupported_tone], axis=1).astype(np.float32)
    support = np.stack([support_tone, support_tone], axis=1).astype(np.float32)

    primary_path = tmp_path / "primary.flac"
    support_path = tmp_path / "support_with_drop.flac"
    strict_output_path = tmp_path / "strict_gate.flac"
    drop_safe_output_path = tmp_path / "drop_safe_gate.flac"
    sf.write(str(primary_path), primary, sample_rate, subtype="PCM_24")
    sf.write(str(support_path), support, sample_rate, subtype="PCM_24")

    apply_multimodel_support_gate(
        primary_vocals_path=primary_path,
        support_vocals_paths=[support_path],
        output_vocals_path=strict_output_path,
        support_mode="max",
        support_threshold=0.18,
        mask_power=1.30,
        inactive_floor=0.08,
        active_floor=0.46,
        vocal_core_floor=0.58,
        high_band_floor_scale=0.48,
        high_band_start_hz=1800.0,
        activity_floor_ratio=0.050,
        activity_smoothing_frames=13,
        tail_memory_sec=0.36,
        target_rms_ratio=0.98,
        max_gain=1.06,
    )
    apply_multimodel_support_gate(
        primary_vocals_path=primary_path,
        support_vocals_paths=[support_path],
        output_vocals_path=drop_safe_output_path,
        support_mode="max",
        support_threshold=0.12,
        mask_power=0.98,
        inactive_floor=0.12,
        active_floor=0.64,
        vocal_core_floor=0.80,
        high_band_floor_scale=0.56,
        high_band_start_hz=1800.0,
        activity_floor_ratio=0.040,
        activity_smoothing_frames=19,
        tail_memory_sec=0.52,
        target_rms_ratio=1.04,
        max_gain=1.12,
    )

    strict_audio, strict_sr = sf.read(str(strict_output_path), always_2d=True, dtype="float32")
    drop_safe_audio, drop_safe_sr = sf.read(str(drop_safe_output_path), always_2d=True, dtype="float32")
    assert strict_sr == sample_rate
    assert drop_safe_sr == sample_rate

    strict_drop_core = _band_rms(strict_audio[drop_slice], sample_rate, 350.0, 550.0)
    drop_safe_core = _band_rms(drop_safe_audio[drop_slice], sample_rate, 350.0, 550.0)
    assert drop_safe_core > strict_drop_core * 1.10
    assert _band_rms(drop_safe_audio, sample_rate, 2500.0, 3400.0) < _band_rms(primary, sample_rate, 2500.0, 3400.0)


def test_focused_detail_recovery_preserves_air_without_lifting_unbounded_high_band(tmp_path: Path) -> None:
    sample_rate = 32000
    times = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
    vocal_body = 0.16 * np.sin(2.0 * np.pi * 440.0 * times)
    vocal_presence = 0.08 * np.sin(2.0 * np.pi * 2400.0 * times)
    vocal_air = 0.04 * np.sin(2.0 * np.pi * 6500.0 * times)
    non_vocal_high = 0.05 * np.sin(2.0 * np.pi * 12000.0 * times)
    focus_slice = slice(int(0.80 * sample_rate), int(1.40 * sample_rate))

    donor = vocal_body + vocal_presence + vocal_air + non_vocal_high
    base = donor.copy()
    base[focus_slice] = (
        0.48 * vocal_body[focus_slice]
        + 0.48 * vocal_presence[focus_slice]
        + 0.22 * vocal_air[focus_slice]
        + 0.18 * non_vocal_high[focus_slice]
    )

    base_audio = np.stack([base, base], axis=1).astype(np.float32)
    donor_audio = np.stack([donor, donor], axis=1).astype(np.float32)
    base_path = tmp_path / "base.flac"
    donor_path = tmp_path / "donor.flac"
    output_path = tmp_path / "focused_recovery.flac"
    sf.write(str(base_path), base_audio, sample_rate, subtype="PCM_24")
    sf.write(str(donor_path), donor_audio, sample_rate, subtype="PCM_24")

    build_focused_detail_recovery_candidate(
        base_vocals_path=base_path,
        donor_vocals_paths=[donor_path],
        output_vocals_path=output_path,
        focus_start_sec=0.80,
        focus_end_sec=1.40,
        focus_fade_sec=0.10,
        body_max_gain=1.58,
        body_blend=0.78,
        air_max_gain=2.35,
        air_blend=0.34,
    )

    recovered_audio, recovered_sr = sf.read(str(output_path), always_2d=True, dtype="float32")
    assert recovered_sr == sample_rate

    pre_slice = slice(0, int(0.40 * sample_rate))
    assert _band_rms(recovered_audio[focus_slice], sample_rate, 350.0, 550.0) > (
        _band_rms(base_audio[focus_slice], sample_rate, 350.0, 550.0) * 1.12
    )
    assert _band_rms(recovered_audio[focus_slice], sample_rate, 6000.0, 7200.0) > (
        _band_rms(base_audio[focus_slice], sample_rate, 6000.0, 7200.0) * 1.08
    )
    assert _band_rms(recovered_audio[focus_slice], sample_rate, 10500.0, 13500.0) <= (
        _band_rms(base_audio[focus_slice], sample_rate, 10500.0, 13500.0) * 1.05
    )
    assert float(np.sqrt(np.mean(np.square(recovered_audio[pre_slice] - base_audio[pre_slice])))) < 1e-4


def test_write_listen_pack_removes_stale_generated_files(tmp_path: Path) -> None:
    sample_rate = 8000
    audio = np.zeros((sample_rate // 10, 2), dtype=np.float32)
    raw_ref_path = tmp_path / "raw.flac"
    first_path = tmp_path / "first.flac"
    second_path = tmp_path / "second.flac"
    sf.write(str(raw_ref_path), audio, sample_rate, subtype="PCM_24")
    sf.write(str(first_path), audio, sample_rate, subtype="PCM_24")
    sf.write(str(second_path), audio + 0.05, sample_rate, subtype="PCM_24")

    listen_dir = tmp_path / "listen"
    listen_dir.mkdir()
    (listen_dir / "09_stale_candidate.flac").write_bytes(b"stale")
    (listen_dir / "model_zoo_delta_metrics.json").write_text("stale", encoding="utf-8")

    candidates = [
        CandidateArtifact("01_first.flac", first_path, "first", {}),
        CandidateArtifact("02_second.flac", second_path, "second", {}),
    ]
    delta_metrics = calculate_delta_metrics(candidates)
    write_listen_pack(
        listen_dir=listen_dir,
        raw_ref_path=raw_ref_path,
        candidates=candidates,
        metrics=[],
        delta_metrics=delta_metrics,
    )

    assert not (listen_dir / "09_stale_candidate.flac").exists()
    assert (listen_dir / "00_raw_ref.flac").exists()
    assert (listen_dir / "01_first.flac").exists()
    assert (listen_dir / "02_second.flac").exists()
    assert "delta_vs_baseline_rms" in (listen_dir / "model_zoo_delta_metrics.json").read_text(encoding="utf-8")


def test_validate_preview_window_rejects_empty_or_negative_windows() -> None:
    validate_preview_window(0.0, 1.0)
    validate_preview_window(0.0, None)

    for start_sec, end_sec in [(-0.1, 1.0), (5.0, 5.0), (6.0, 5.0)]:
        try:
            validate_preview_window(start_sec, end_sec)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid preview window: {start_sec}:{end_sec}")
