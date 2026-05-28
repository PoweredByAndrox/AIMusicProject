"""Regression tests for the bsroformer-worker instrumental materialisation logic.

Covers the R7c root-cause fix in :mod:`app.runtime`:

* ``_resolve_instrumental_strategy`` must dispatch to the correct mode for
  every combination of ``(instruments, target_instrument)`` we care about
  in production.
* ``_run_bsroformer_folder`` (target=other) must write the *predicted other*
  verbatim into ``*_instrumental.wav``, not ``mix - other`` (which yields the
  vocal residual). This was the bug behind ``bs_resurrected_instrumental_mix``
  having vocal_bleed of -0.56 dB.
* ``_run_bsroformer_folder`` (multi-stem, target=null) must locate the
  ``"vocals"`` stem by name and produce ``mix - vocals``, not
  ``mix - instruments[0]`` (which yielded ``mix - bass``).
* ``_run_melband_folder`` (target=vocals) keeps producing ``mix - vocals``
  (regression guard for the previously-correct path).

The folder-runner tests monkeypatch ``_demix_with_tta`` so we never touch the
real model weights or torch CUDA. They still require ``torch`` /
``soundfile`` / ``bs_roformer`` for module import — outside of the
bsroformer-worker container they are skipped via ``pytest.importorskip``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Pure unit tests below need no heavy deps; integration tests on the folder
# runners do. We attempt the import once, skip the heavy module if the env is
# bare (e.g. the developer's laptop without bs_roformer), and run the rest.
torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("ml_collections")
pytest.importorskip("bs_roformer")

# Make ``app.runtime`` importable when the test runs from
# ``services/bsroformer-worker/`` (docker WORKDIR /app, host workdir varies).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import runtime  # noqa: E402
from app.runtime import _resolve_instrumental_strategy  # noqa: E402


# ---------------------------------------------------------------------------
# Pure unit tests for the strategy resolver
# ---------------------------------------------------------------------------


class TestResolveInstrumentalStrategy:
    def test_target_vocals_returns_residual_vocals(self):
        assert _resolve_instrumental_strategy(["vocals", "other"], "vocals") == (
            "residual",
            "vocals",
        )

    def test_target_other_returns_direct_other(self):
        # The unwa instrumental_resurrection model: head 'other' IS the instrumental.
        assert _resolve_instrumental_strategy(["vocals", "other"], "other") == (
            "direct",
            "other",
        )

    def test_target_no_vocals_returns_direct_target(self):
        # Generic future-proofing: any non-vocal target means "model emits the
        # instrumental side directly under that key".
        assert _resolve_instrumental_strategy(["vocals", "instrumental"], "instrumental") == (
            "direct",
            "instrumental",
        )

    def test_no_target_with_vocals_returns_residual_vocals(self):
        # The 6-stem coarse model used to fall into the bug here:
        # primary=instruments[0]=='bass' → mix - bass. Now we look up 'vocals'.
        assert _resolve_instrumental_strategy(
            ["bass", "drums", "other", "vocals", "guitar", "piano"], None
        ) == ("residual", "vocals")

    def test_no_target_no_vocals_returns_sum(self):
        # Hypothetical multi-stem model without a vocals head — best we can do
        # is reconstruct the instrumental as the sum of predicted stems.
        assert _resolve_instrumental_strategy(["drums", "bass"], None) == ("sum", None)


# ---------------------------------------------------------------------------
# Folder-runner tests with monkeypatched _demix_with_tta
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Stand-in for :class:`app.runtime.CachedRuntime` used by the folder runners."""

    def __init__(self, instruments, target_instrument):
        self.instruments = tuple(instruments)
        self.target_instrument = target_instrument
        self.config = None
        self.model = None
        self.device = torch.device("cpu")


def _write_stereo_wav(path: Path, audio_t_2: "np.ndarray", sr: int = 44100) -> None:
    """Write ``audio_t_2`` ((length, 2) float32) to ``path`` as a 32-bit float WAV."""
    sf.write(str(path), audio_t_2.astype(np.float32, copy=False), sr, subtype="FLOAT")


def _read_wav_ch_first(path: Path) -> tuple["np.ndarray", int]:
    arr, sr = sf.read(str(path), always_2d=True)
    return np.ascontiguousarray(arr.T).astype(np.float32, copy=False), sr


def _make_two_band_mix(n: int = 44100, sr: int = 44100):
    """Build a synthetic ``mix = vocals + instrumental`` where ``|vocals| >> |instr|``.

    Using grossly asymmetric amplitudes means ``|mix| ~= |vocals|`` everywhere,
    so the residual ``mix - vocals = instr`` never bumps into the
    ``|residual| > |mix| * 1.05`` clamp inside ``_hardened_instrumental_residual``
    (which would legitimately fire on phase-cancellation troughs of two
    equal-amplitude sinusoids and add ~22 dB error to the test signal).
    """
    t = np.arange(n, dtype=np.float32) / float(sr)
    # "Vocals" dominate hard (0.70 amplitude), "instrumental" is tiny (0.01).
    # The 70:1 ratio guarantees ``min(|mix|) ~ 0.69`` and ``max(|instr|) = 0.01``
    # so the ``|residual| > |mix| * 1.05`` clamp inside
    # ``_hardened_instrumental_residual`` never engages on the test signal,
    # giving us deterministic ``-inf dB`` separation when the math is correct.
    vocals_mono = 0.70 * np.sin(2.0 * np.pi * 880.0 * t)
    instr_mono = 0.01 * np.sin(2.0 * np.pi * 220.0 * t)
    vocals = np.stack([vocals_mono, vocals_mono], axis=-1)
    instr = np.stack([instr_mono, instr_mono], axis=-1)
    mix = vocals + instr
    return mix.astype(np.float32), vocals.astype(np.float32), instr.astype(np.float32)


def _rmse_db(a: "np.ndarray", b: "np.ndarray") -> float:
    n = min(a.shape[-1], b.shape[-1])
    diff = a[..., :n] - b[..., :n]
    rmse = float(np.sqrt(np.mean(diff * diff)) + 1e-12)
    return 20.0 * float(np.log10(rmse))


def test_run_bsroformer_folder_target_other_writes_predicted_other_directly(
    tmp_path, monkeypatch
):
    """When target='other' the runner must write *_instrumental.wav = result['other'],
    NOT mix - result['other']. Pre-fix this test fails by ~30+ dB."""

    sr = 44100
    n = sr  # 1 second
    mix, vocals, instr = _make_two_band_mix(n, sr)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _write_stereo_wav(in_dir / "track.wav", mix, sr)

    def fake_demix(demix_track_fn, config, model, mixture, device, first_chunk_time):
        # mixture is torch.Tensor (channels, length); we ignore it and
        # synthesise a perfect separation in (channels, length) numpy.
        vocals_pred = vocals.T.astype(np.float32)
        other_pred = instr.T.astype(np.float32)
        return {"vocals": vocals_pred, "other": other_pred}, 0.0

    monkeypatch.setattr(runtime, "_demix_with_tta", fake_demix)

    fake_runtime = _FakeRuntime(["vocals", "other"], "other")
    runtime._run_bsroformer_folder(fake_runtime, in_dir, out_dir)

    out_instr_ch_first, _ = _read_wav_ch_first(out_dir / "track_instrumental.wav")
    out_other_ch_first, _ = _read_wav_ch_first(out_dir / "track_other.wav")
    expected_instr = instr.T

    err_to_instr_db = _rmse_db(out_instr_ch_first, expected_instr)
    err_to_vocals_db = _rmse_db(out_instr_ch_first, vocals.T)

    assert err_to_instr_db < -60.0, (
        f"track_instrumental.wav must equal predicted 'other' (error {err_to_instr_db:.1f} dB)."
    )
    assert err_to_vocals_db > err_to_instr_db + 20.0, (
        "Pre-fix regression: instrumental file must NOT match vocals signature "
        f"(err_vs_vocals={err_to_vocals_db:.1f} dB, err_vs_instr={err_to_instr_db:.1f} dB)."
    )
    other_match_db = _rmse_db(out_other_ch_first, instr.T)
    assert other_match_db < -60.0, (
        f"track_other.wav must equal predicted 'other' as well (error {other_match_db:.1f} dB)."
    )


def test_run_bsroformer_folder_no_target_picks_vocals_for_residual(
    tmp_path, monkeypatch
):
    """6-stem coarse model: target=null. Pre-fix the runner used instruments[0]=='bass'
    and produced ``mix - bass`` (still containing vocals). After fix it must locate
    'vocals' by name and produce ``mix - vocals``."""

    sr = 44100
    n = sr
    mix, vocals, instr = _make_two_band_mix(n, sr)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _write_stereo_wav(in_dir / "track.wav", mix, sr)

    def fake_demix(demix_track_fn, config, model, mixture, device, first_chunk_time):
        # Synthesise plausible 6-stem separation. Bass = part of instrumental,
        # vocals = perfect vocals. The pre-fix code subtracted bass (small
        # signal) and left ~all vocals in the file.
        vocals_pred = vocals.T.astype(np.float32)
        bass_pred = (instr.T * 0.1).astype(np.float32)
        drums_pred = (instr.T * 0.2).astype(np.float32)
        other_pred = (instr.T * 0.3).astype(np.float32)
        guitar_pred = (instr.T * 0.2).astype(np.float32)
        piano_pred = (instr.T * 0.2).astype(np.float32)
        return {
            "bass": bass_pred,
            "drums": drums_pred,
            "other": other_pred,
            "vocals": vocals_pred,
            "guitar": guitar_pred,
            "piano": piano_pred,
        }, 0.0

    monkeypatch.setattr(runtime, "_demix_with_tta", fake_demix)

    fake_runtime = _FakeRuntime(
        ["bass", "drums", "other", "vocals", "guitar", "piano"], None
    )
    runtime._run_bsroformer_folder(fake_runtime, in_dir, out_dir)

    out_instr_ch_first, _ = _read_wav_ch_first(out_dir / "track_instrumental.wav")
    err_vs_instr = _rmse_db(out_instr_ch_first, instr.T)
    err_vs_vocals = _rmse_db(out_instr_ch_first, vocals.T)
    err_vs_mix_minus_bass = _rmse_db(
        out_instr_ch_first, (mix.T - 0.1 * instr.T)
    )

    assert err_vs_instr < -60.0, (
        f"After fix the instrumental file must equal mix - vocals = real instrumental "
        f"(error {err_vs_instr:.1f} dB)."
    )
    assert err_vs_vocals > err_vs_instr + 20.0, (
        "Instrumental file must not match the vocal signal."
    )
    assert err_vs_mix_minus_bass > err_vs_instr + 20.0, (
        "Pre-fix regression: instrumental file must NOT be mix - bass anymore "
        f"(err_vs_mix_minus_bass={err_vs_mix_minus_bass:.1f} dB)."
    )


def test_run_melband_folder_target_vocals_keeps_residual_math(
    tmp_path, monkeypatch
):
    """Regression test for vocals_resurrection: target=vocals should still produce
    instrumental = mix - vocals (this branch was already correct pre-fix; we
    guard it against the new abstraction breaking it)."""

    sr = 44100
    n = sr
    mix, vocals, instr = _make_two_band_mix(n, sr)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _write_stereo_wav(in_dir / "track.wav", mix, sr)

    def fake_demix(demix_track_fn, config, model, mixture, device, first_chunk_time):
        return {"vocals": vocals.T.astype(np.float32)}, 0.0

    # Patch melband's _demix_with_tta and also stub the lazy vendor module
    # import so the function doesn't try to load the real mel-band package.
    monkeypatch.setattr(runtime, "_demix_with_tta", fake_demix)

    class _FakeVendor:
        demix_track = staticmethod(lambda *a, **kw: {})

    monkeypatch.setattr(runtime, "_melband_vendor_module", lambda: _FakeVendor())

    fake_runtime = _FakeRuntime(["vocals", "other"], "vocals")
    runtime._run_melband_folder(fake_runtime, in_dir, out_dir)

    out_instr_ch_first, _ = _read_wav_ch_first(out_dir / "track_instrumental.wav")
    out_vocals_ch_first, _ = _read_wav_ch_first(out_dir / "track_vocals.wav")

    err_instr = _rmse_db(out_instr_ch_first, instr.T)
    err_vocals = _rmse_db(out_vocals_ch_first, vocals.T)
    assert err_instr < -60.0, f"instrumental must equal mix-vocals (err {err_instr:.1f} dB)"
    assert err_vocals < -60.0, f"vocals stem must equal predicted vocals (err {err_vocals:.1f} dB)"
