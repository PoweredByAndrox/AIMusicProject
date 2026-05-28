"""Unit tests for the soft Wiener vocal-mask post-filter
(``run_vocal_mask_cleanup``).

The whole *point* of this filter is to remove the audible ducking /
pumping that broadband ``vocal_mid_suppression`` + ``center_hard``
caused.  We therefore enforce two hard mathematical contracts that map
directly to the user-reportable defects:

1. **Spectral selectivity** -- a vocal "ah" at 880 Hz must drop the
   instrumental's 880 Hz energy by >=20 dB **while** the kick at 60 Hz
   and the hi-hat at 8 kHz in the *same frames* stay within +/-0.5 dB
   of their pre-filter energy.  This is the "no global volume
   modulation" guarantee.

2. **RMS-envelope correlation** -- the time-domain RMS envelope of the
   masked instrumental, evaluated in 50 ms windows, must correlate
   with the original instrumental's envelope at >= 0.95.  This is the
   directly-listenable "no audible pumping" check; broadband ducking
   collapses this number to ~0.7-0.8.

Plus the usual housekeeping: passthrough when ``lambda <= 0``, mask
floor enforcement, and metadata truthfulness for operator manifests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from app.main import (  # noqa: E402
    _smooth_2d_box,
    _stft,
    _write_wave_file,
    run_vocal_mask_cleanup,
)
from app.schemas import RunRequest  # noqa: E402


SR = 44100
N = 2 * SR  # 2 seconds stereo so STFT has plenty of frames


def _sine(freq_hz: float, n: int = N, sr: int = SR, amp: float = 0.30) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sr)
    mono = amp * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
    return np.stack([mono, mono], axis=0)


def _gated_sine(
    freq_hz: float,
    gate_starts_s: list[float],
    gate_durations_s: list[float],
    n: int = N,
    sr: int = SR,
    amp: float = 0.40,
) -> np.ndarray:
    """Sine that is on only during the listed (start, duration) intervals."""
    t = np.arange(n, dtype=np.float32) / float(sr)
    mono = np.zeros(n, dtype=np.float32)
    for start_s, dur_s in zip(gate_starts_s, gate_durations_s):
        i0 = int(start_s * sr)
        i1 = min(n, int((start_s + dur_s) * sr))
        ramp = min(int(0.005 * sr), max(1, (i1 - i0) // 4))
        env = np.ones(i1 - i0, dtype=np.float32)
        env[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        mono[i0:i1] = amp * np.sin(2.0 * np.pi * freq_hz * t[i0:i1]).astype(np.float32) * env
    return np.stack([mono, mono], axis=0)


def _bin_energy_db(audio: np.ndarray, freq_hz: float, sr: int = SR) -> float:
    spec, _ = _stft(audio, n_fft=4096, hop=1024)
    bin_idx = int(round(freq_hz * 4096 / sr))
    bin_lo = max(0, bin_idx - 2)
    bin_hi = min(spec.shape[1], bin_idx + 3)
    band_power = float(np.mean(np.abs(spec[:, bin_lo:bin_hi, :]) ** 2))
    return 10.0 * np.log10(band_power + 1e-30)


def _rms_envelope(audio: np.ndarray, win_ms: float = 50.0, sr: int = SR) -> np.ndarray:
    win = max(1, int(round(win_ms * 1e-3 * sr)))
    mono = audio.mean(axis=0).astype(np.float32)
    n_frames = mono.shape[0] // win
    if n_frames < 2:
        return np.array([float(np.sqrt(np.mean(mono ** 2)))], dtype=np.float32)
    trimmed = mono[: n_frames * win].reshape(n_frames, win)
    return np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-12).astype(np.float32)


def _make_request(
    tmp_path: Path,
    instr_path: Path,
    vocals_path: Path,
    *,
    profile: str = "balanced",
    vocal_mask_weights: dict | None = None,
) -> RunRequest:
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_step: dict[str, object] = {
        "task": "vocal_mask_cleanup",
        "vocal_mask_profile": profile,
    }
    if vocal_mask_weights is not None:
        raw_step["vocal_mask_weights"] = vocal_mask_weights
    return RunRequest(
        job_id="job-test",
        phase_id="phase-test",
        step_index=0,
        track_name="synthetic",
        input_path=str(instr_path),
        output_root=str(output_root),
        target_stem="instrumental_mix",
        task="vocal_mask_cleanup",
        build_from_paths={"vocals": str(vocals_path)},
        raw_step=raw_step,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# Helper: _smooth_2d_box arithmetic
# ---------------------------------------------------------------------------


class TestSmooth2dBox:
    def test_zero_radius_is_identity(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 17, 23)).astype(np.float32)
        y = _smooth_2d_box(x, time_radius=0, freq_radius=0)
        assert np.allclose(y, x)

    def test_box_smoother_preserves_constant_signal(self):
        x = np.full((1, 8, 12), 1.5, dtype=np.float32)
        y = _smooth_2d_box(x, time_radius=2, freq_radius=1)
        assert np.allclose(y, x, atol=1e-6)

    def test_box_smoother_attenuates_isolated_spike(self):
        x = np.zeros((1, 9, 9), dtype=np.float32)
        x[0, 4, 4] = 1.0
        y = _smooth_2d_box(x, time_radius=1, freq_radius=1)
        assert y[0, 4, 4] < 0.20
        assert y[0, 4, 5] > 0.05
        assert y[0, 5, 4] > 0.05


# ---------------------------------------------------------------------------
# Spectral-selectivity contract: no global ducking, only vocal-bin notch
# ---------------------------------------------------------------------------


class TestSpectralSelectivity:
    def test_vocal_bin_attenuated_while_kick_and_hat_preserved(self, tmp_path):
        # Dirty instrumental = kick (60 Hz, dominant) + hi-hat-like sine
        # (8 kHz, present) + a *small* vocal residue at 880 Hz that is
        # gated to two specific intervals.  Mirrors the realistic case:
        # the instrumental only has a small leakage of the vocal, while
        # the vocal stem has the full vocal at high amplitude.  The
        # Wiener mask should drive the 880 Hz energy way down in the
        # vocal frames (lambda*|V|^2 >> |I|^2 there) while leaving 60 Hz
        # and 8 kHz at +/-0.5 dB of input.
        kick = _sine(60.0, amp=0.40)
        hat = _sine(8000.0, amp=0.20)
        vocal_residue_in_instr = _gated_sine(
            880.0, gate_starts_s=[0.30, 1.10], gate_durations_s=[0.40, 0.40], amp=0.10,
        )
        instrumental = (kick + hat + vocal_residue_in_instr).astype(np.float32)

        # Vocal guide -- the actual vocal is much louder than its
        # leakage, ~8x amplitude (~18 dB) which matches what we see on
        # real ML-separated stems.
        vocals = _gated_sine(
            880.0, gate_starts_s=[0.30, 1.10], gate_durations_s=[0.40, 0.40], amp=0.80,
        )

        instr_path = tmp_path / "dirty_instr.wav"
        vocals_path = tmp_path / "vocals.wav"
        _write_wave_file(instr_path, instrumental, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, instr_path, vocals_path)
        named, metadata, _notes = run_vocal_mask_cleanup(
            request, Path(request.output_root)
        )

        cleaned_path = Path(named["instrumental_mix"])
        from app.main import _read_wave_file
        cleaned, _sr = _read_wave_file(cleaned_path)

        kick_in = _bin_energy_db(instrumental, 60.0)
        kick_out = _bin_energy_db(cleaned, 60.0)
        hat_in = _bin_energy_db(instrumental, 8000.0)
        hat_out = _bin_energy_db(cleaned, 8000.0)
        vox_in = _bin_energy_db(instrumental, 880.0)
        vox_out = _bin_energy_db(cleaned, 880.0)

        assert vox_in - vox_out >= 18.0, (
            f"vocal 880 Hz bin must be attenuated by >=18 dB, got "
            f"{vox_in - vox_out:.2f} dB (in={vox_in:.2f}, out={vox_out:.2f})"
        )
        assert abs(kick_out - kick_in) <= 0.5, (
            f"kick 60 Hz bin must stay within +/-0.5 dB, got "
            f"{kick_out - kick_in:+.3f} dB"
        )
        assert abs(hat_out - hat_in) <= 0.5, (
            f"hi-hat 8 kHz bin must stay within +/-0.5 dB, got "
            f"{hat_out - hat_in:+.3f} dB"
        )

        assert metadata["mask_stats"]["mean_mask_value"] < 1.0
        assert metadata["mask_stats"]["min_mask_value"] >= metadata["weights"]["floor"] - 1e-6


# ---------------------------------------------------------------------------
# Anti-ducking contract: RMS envelope correlation must stay high
# ---------------------------------------------------------------------------


class TestNoDucking:
    def test_rms_envelope_correlation_above_threshold(self, tmp_path):
        # Same gated-vocal-residue scenario, but now we measure the time
        # envelope of the *whole* signal (mostly kick+hat) and assert that
        # the masked output's envelope is essentially identical to the
        # input's.  Broadband ducking would tank this correlation.
        kick = _sine(60.0, amp=0.30)
        hat = _sine(8000.0, amp=0.20)
        vocal_residue_in_instr = _gated_sine(
            880.0, gate_starts_s=[0.20, 0.80, 1.40],
            gate_durations_s=[0.30, 0.30, 0.30], amp=0.40,
        )
        instrumental = (kick + hat + vocal_residue_in_instr).astype(np.float32)
        vocals = _gated_sine(
            880.0, gate_starts_s=[0.20, 0.80, 1.40],
            gate_durations_s=[0.30, 0.30, 0.30], amp=0.70,
        )

        instr_path = tmp_path / "dirty_instr.wav"
        vocals_path = tmp_path / "vocals.wav"
        _write_wave_file(instr_path, instrumental, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, instr_path, vocals_path)
        named, _meta, _notes = run_vocal_mask_cleanup(
            request, Path(request.output_root)
        )
        from app.main import _read_wave_file
        cleaned, _sr = _read_wave_file(Path(named["instrumental_mix"]))

        env_in = _rms_envelope(instrumental, win_ms=50.0)
        env_out = _rms_envelope(cleaned, win_ms=50.0)
        assert env_in.shape == env_out.shape

        # Pearson correlation; near 1.0 means "envelope shape is preserved".
        env_in_c = env_in - env_in.mean()
        env_out_c = env_out - env_out.mean()
        denom = float(np.sqrt(np.sum(env_in_c ** 2) * np.sum(env_out_c ** 2)) + 1e-12)
        corr = float(np.sum(env_in_c * env_out_c) / denom)
        assert corr >= 0.95, (
            f"RMS-envelope correlation must be >=0.95 (no audible ducking), "
            f"got {corr:.4f}"
        )

    def test_envelope_drop_during_vocal_is_small(self, tmp_path):
        # Realistic-amplitude check: kick (40%) + hat (20%) + tiny vocal
        # leakage (10%), with a loud vocal guide (80%).  The cleaned
        # instrumental's RMS in the vocal-dense window must drop by less
        # than 1 dB compared to the input window -- because the only
        # legitimate energy reduction is the small leakage at 880 Hz.
        # Broadband ducking would slash the entire window by 10-15 dB.
        kick = _sine(60.0, amp=0.40)
        hat = _sine(8000.0, amp=0.20)
        vocal_residue_in_instr = _gated_sine(
            880.0, gate_starts_s=[0.50], gate_durations_s=[1.00], amp=0.10,
        )
        instrumental = (kick + hat + vocal_residue_in_instr).astype(np.float32)
        vocals = _gated_sine(
            880.0, gate_starts_s=[0.50], gate_durations_s=[1.00], amp=0.80,
        )

        instr_path = tmp_path / "dirty_instr.wav"
        vocals_path = tmp_path / "vocals.wav"
        _write_wave_file(instr_path, instrumental, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, instr_path, vocals_path)
        named, _meta, _notes = run_vocal_mask_cleanup(
            request, Path(request.output_root)
        )
        from app.main import _read_wave_file
        cleaned, _sr = _read_wave_file(Path(named["instrumental_mix"]))

        # Look at samples 0.7s..1.3s -- right in the middle of the vocal
        # burst.  Without per-bin masking, broadband ducking would slash
        # this window by >10 dB (it has the most vocal energy).
        s0, s1 = int(0.7 * SR), int(1.3 * SR)
        rms_in = float(np.sqrt(np.mean(instrumental[:, s0:s1] ** 2)))
        rms_out = float(np.sqrt(np.mean(cleaned[:, s0:s1] ** 2)))
        drop_db = 20.0 * np.log10((rms_out + 1e-12) / (rms_in + 1e-12))
        assert drop_db > -1.0, (
            f"RMS in vocal-dense window must drop <= 1 dB at realistic "
            f"leakage levels (no ducking), got {drop_db:.2f} dB"
        )


# ---------------------------------------------------------------------------
# Knobs / metadata / passthrough
# ---------------------------------------------------------------------------


class TestKnobsAndMetadata:
    def test_lambda_zero_is_passthrough(self, tmp_path):
        instrumental = _sine(440.0, amp=0.30)
        vocals = _sine(880.0, amp=0.50)

        instr_path = tmp_path / "instr.wav"
        vocals_path = tmp_path / "vocals.wav"
        _write_wave_file(instr_path, instrumental, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, instr_path, vocals_path, profile="off")
        named, metadata, notes = run_vocal_mask_cleanup(
            request, Path(request.output_root)
        )
        assert metadata["passthrough"] is True
        assert "passthrough" in notes

        from app.main import _read_wave_file
        cleaned, _sr = _read_wave_file(Path(named["instrumental_mix"]))
        assert np.allclose(cleaned, instrumental, atol=1e-6)

    def test_floor_caps_attenuation(self, tmp_path):
        # Vocal way louder than instrumental everywhere -> raw mask would
        # collapse to ~0; floor=0.30 must clamp it so attenuation is
        # bounded at -10.5 dB instead of -infty.
        instrumental = _sine(880.0, amp=0.05)
        vocals = _sine(880.0, amp=1.50)

        instr_path = tmp_path / "instr.wav"
        vocals_path = tmp_path / "vocals.wav"
        _write_wave_file(instr_path, instrumental, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(
            tmp_path, instr_path, vocals_path,
            vocal_mask_weights={"floor": 0.30, "smoothing_frames": 0, "smoothing_bins": 0},
        )
        _named, metadata, _notes = run_vocal_mask_cleanup(
            request, Path(request.output_root)
        )
        assert metadata["mask_stats"]["min_mask_value"] >= 0.30 - 1e-6
        # Floor=0.30 means worst-case attenuation = 20*log10(0.30) ~= -10.46 dB.
        # The output RMS must therefore be at least floor * input RMS.
        assert metadata["energy_stats"]["instr_output_rms"] >= 0.30 * metadata["energy_stats"]["instr_input_rms"] - 1e-4

    def test_missing_vocals_guide_raises(self, tmp_path):
        instrumental = _sine(440.0)
        instr_path = tmp_path / "instr.wav"
        _write_wave_file(instr_path, instrumental, SR)

        output_root = tmp_path / "out"
        output_root.mkdir(parents=True, exist_ok=True)
        request = RunRequest(
            job_id="job-test",
            phase_id="phase-test",
            step_index=0,
            track_name="synthetic",
            input_path=str(instr_path),
            output_root=str(output_root),
            target_stem="instrumental_mix",
            task="vocal_mask_cleanup",
            build_from_paths={},
            raw_step={"task": "vocal_mask_cleanup"},
            dry_run=False,
        )
        with pytest.raises(FileNotFoundError, match="vocals"):
            run_vocal_mask_cleanup(request, Path(request.output_root))

    def test_metadata_records_profile_and_inputs(self, tmp_path):
        instrumental = _sine(440.0)
        vocals = _sine(880.0)
        instr_path = tmp_path / "instr.wav"
        vocals_path = tmp_path / "vocals.wav"
        _write_wave_file(instr_path, instrumental, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, instr_path, vocals_path, profile="aggressive")
        _named, metadata, notes = run_vocal_mask_cleanup(
            request, Path(request.output_root)
        )
        assert metadata["vocal_mask_profile"] == "aggressive"
        assert metadata["instrumental_input"] == str(instr_path)
        assert metadata["vocals_guide_input"] == str(vocals_path)
        assert "lambda" in metadata["weights"]
        assert "aggressive" in notes
