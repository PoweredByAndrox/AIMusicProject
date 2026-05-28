"""Regression tests for the S2 ``bs_overlay_safety_net`` in
``run_tandem_instrumental_hybrid``.

The safety-net measures speech-band vocal contamination in both the Demucs
``base`` and the BS-Roformer ``bs_seed`` (and optional ``raw_bs``) and
zero-es the BS contribution if it is significantly *worse* than the base.
This guard is what prevents the historical failure mode where a vocal-heavy
``bs_resurrected_instrumental_mix`` was deterministically blended in and
dragged ``final.bleed_db_rel_total`` into the -10 dB range.

The pure-unit test for ``_vocal_contamination_db`` covers the signal-level
math; the two integration tests below exercise the full
``run_tandem_instrumental_hybrid`` entry point with synthetic stems so the
override / metadata path is end-to-end verified.
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
    _vocal_contamination_db,
    _write_wave_file,
    run_tandem_instrumental_hybrid,
)
from app.schemas import RunRequest  # noqa: E402


SR = 44100
N = SR  # 1 second stereo synthetic clip


def _make_synthetic_components():
    t = np.arange(N, dtype=np.float32) / float(SR)
    vocals_mono = 0.30 * np.sin(2.0 * np.pi * 880.0 * t)
    instr_mono = 0.30 * np.sin(2.0 * np.pi * 220.0 * t)
    # Stereo (channels, length) so it matches the layout used by main.py.
    vocals = np.stack([vocals_mono, vocals_mono], axis=0)
    instr = np.stack([instr_mono, instr_mono], axis=0)
    return instr.astype(np.float32), vocals.astype(np.float32)


def _make_request(
    tmp_path: Path, base_path: Path, bs_seed_path: Path, vocals_path: Path,
    *, raw_step_extra: dict | None = None, raw_bs_path: Path | None = None,
) -> RunRequest:
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_step = {"task": "tandem_instrumental_hybrid"}
    if raw_step_extra:
        raw_step.update(raw_step_extra)
    build_from_paths: dict[str, str] = {
        "bs_resurrected_instrumental_mix": str(bs_seed_path),
        "vocals": str(vocals_path),
    }
    if raw_bs_path is not None:
        build_from_paths["bs_instrumental_mix"] = str(raw_bs_path)
    return RunRequest(
        job_id="job-test",
        phase_id="phase-test",
        step_index=0,
        track_name="synthetic",
        input_path=str(base_path),
        output_root=str(output_root),
        target_stem="instrumental_hybrid",
        task="tandem_instrumental_hybrid",
        build_from_paths=build_from_paths,
        raw_step=raw_step,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# _vocal_contamination_db pure-unit
# ---------------------------------------------------------------------------


class TestVocalContaminationDb:
    def test_clean_instrumental_yields_low_bleed(self):
        instr, vocals = _make_synthetic_components()
        bleed_db = _vocal_contamination_db(instr, vocals, SR)
        # 880 Hz vs 220 Hz are spectrally orthogonal → projection energy ~0.
        assert bleed_db < -40.0, f"clean instrumental should have ≤-40 dB bleed, got {bleed_db:.2f}"

    def test_dirty_instrumental_yields_high_bleed(self):
        instr, vocals = _make_synthetic_components()
        dirty = instr + 0.5 * vocals
        bleed_db = _vocal_contamination_db(dirty, vocals, SR)
        # 0.5 vocal energy added → projection should land at most ~-10 dB.
        assert bleed_db > -15.0, f"dirty instrumental should have ≥-15 dB bleed, got {bleed_db:.2f}"

    def test_dirty_minus_clean_exceeds_safety_margin(self):
        instr, vocals = _make_synthetic_components()
        clean_db = _vocal_contamination_db(instr, vocals, SR)
        dirty_db = _vocal_contamination_db(instr + 0.5 * vocals, vocals, SR)
        assert dirty_db - clean_db > 20.0, (
            f"contamination delta must clearly exceed default safety margin (3 dB); "
            f"got delta={dirty_db - clean_db:.2f}"
        )


# ---------------------------------------------------------------------------
# Full run_tandem_instrumental_hybrid integration with safety-net
# ---------------------------------------------------------------------------


class TestRunTandemInstrumentalHybridSafetyNet:
    def test_safety_net_disables_overlay_when_bs_seed_is_dirty(self, tmp_path):
        instr, vocals = _make_synthetic_components()

        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed_dirty.wav"
        vocals_path = tmp_path / "vocals.wav"

        _write_wave_file(base_path, instr, SR)
        # bs_seed = base + 0.5 * vocals → significantly worse than base.
        _write_wave_file(bs_seed_path, instr + 0.5 * vocals, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, base_path, bs_seed_path, vocals_path)
        named_outputs, metadata, notes = run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        assert metadata["weights"]["bs_seed_weight"] == 0.0, (
            "safety-net must zero-out bs_seed_weight when seed is dirtier than base"
        )
        assert metadata["bs_overlay_disabled_reason"], (
            "metadata must record a non-empty bs_overlay_disabled_reason"
        )
        diag = metadata["bs_overlay_safety_net"]
        assert diag["base_bleed_db"] < diag["bs_seed_bleed_db"]
        assert diag["margin_db"] == pytest.approx(3.0)
        overrides = diag["overrides_applied"]
        assert "bs_seed_weight" in overrides
        assert overrides["bs_seed_weight"]["to"] == 0.0

        # Output file exists and is a real WAV.
        out_path = Path(named_outputs[request.target_stem])
        assert out_path.exists() and out_path.stat().st_size > 1024

        assert "safety_net[bs_seed_weight]" in notes

    def test_safety_net_keeps_overlay_when_bs_seed_is_clean(self, tmp_path):
        instr, vocals = _make_synthetic_components()

        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed_clean.wav"
        vocals_path = tmp_path / "vocals.wav"

        _write_wave_file(base_path, instr, SR)
        # bs_seed has a tiny gain offset but no vocal injection. Crucially we
        # do NOT mix in white noise — noise creates non-zero projection on
        # ``vocals`` band even when there is no vocal energy, which would
        # trip the safety-net relative to the perfectly-clean ``base``
        # whose projection is exactly zero.
        bs_seed = (instr * 1.02).astype(np.float32)
        _write_wave_file(bs_seed_path, bs_seed, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, base_path, bs_seed_path, vocals_path)
        named_outputs, metadata, _notes = run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        # Default profile is 'balanced' with seed_weight=0.25 — must stay applied.
        assert metadata["weights"]["bs_seed_weight"] == pytest.approx(0.25)
        assert metadata["bs_overlay_disabled_reason"] is None
        assert metadata["bs_overlay_safety_net"]["overrides_applied"] == {}

    def test_safety_net_can_be_disabled_with_zero_margin(self, tmp_path):
        instr, vocals = _make_synthetic_components()

        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed_dirty.wav"
        vocals_path = tmp_path / "vocals.wav"

        _write_wave_file(base_path, instr, SR)
        _write_wave_file(bs_seed_path, instr + 0.5 * vocals, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(
            tmp_path, base_path, bs_seed_path, vocals_path,
            raw_step_extra={"bs_overlay_safety_margin_db": 0.0},
        )
        _, metadata, _ = run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        assert metadata["weights"]["bs_seed_weight"] == pytest.approx(0.25)
        assert metadata["bs_overlay_disabled_reason"] is None

    def test_safety_net_disables_raw_bs_when_only_raw_is_dirty(self, tmp_path):
        # Symmetric red-path for the raw-bs slot: bs_seed is clean, raw_bs is
        # dirty. Asserts that ``weights.bs_raw_weight`` collapses to 0.0 while
        # ``weights.bs_seed_weight`` stays at the profile default, and that
        # diagnostics + ``bs_overlay_disabled_reason`` correctly attribute the
        # override to the raw-bs slot only (M3 from the S1/S2/S3/S5 review).
        instr, vocals = _make_synthetic_components()

        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed_clean.wav"
        raw_bs_path = tmp_path / "raw_bs_dirty.wav"
        vocals_path = tmp_path / "vocals.wav"

        _write_wave_file(base_path, instr, SR)
        _write_wave_file(bs_seed_path, (instr * 1.02).astype(np.float32), SR)
        _write_wave_file(raw_bs_path, instr + 0.5 * vocals, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(
            tmp_path, base_path, bs_seed_path, vocals_path,
            raw_bs_path=raw_bs_path,
        )
        _named, metadata, notes = run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        assert metadata["weights"]["bs_seed_weight"] == pytest.approx(0.25)
        assert metadata["weights"]["bs_raw_weight"] == 0.0

        diag = metadata["bs_overlay_safety_net"]
        assert "bs_raw_bleed_db" in diag, "raw-bs bleed must be recorded when raw_bs is wired"
        assert diag["base_bleed_db"] < diag["bs_raw_bleed_db"]
        overrides = diag["overrides_applied"]
        assert "bs_raw_weight" in overrides
        assert "bs_seed_weight" not in overrides, (
            "seed slot must not be touched when only raw_bs is contaminated"
        )
        assert overrides["bs_raw_weight"]["to"] == 0.0

        assert metadata["bs_overlay_disabled_reason"], (
            "metadata must surface bs_overlay_disabled_reason when raw_bs is suppressed"
        )
        assert "safety_net[bs_raw_weight]" in notes


class TestSafetyMarginParsing:
    """Defensive parsing for ``bs_overlay_safety_margin_db`` (M1)."""

    @pytest.mark.parametrize(
        "raw_value",
        [None, "off", "", "not-a-number", object()],
    )
    def test_invalid_safety_margin_falls_back_to_default(self, tmp_path, raw_value):
        instr, vocals = _make_synthetic_components()
        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed_dirty.wav"
        vocals_path = tmp_path / "vocals.wav"

        _write_wave_file(base_path, instr, SR)
        _write_wave_file(bs_seed_path, instr + 0.5 * vocals, SR)
        _write_wave_file(vocals_path, vocals, SR)

        request = _make_request(
            tmp_path, base_path, bs_seed_path, vocals_path,
            raw_step_extra={"bs_overlay_safety_margin_db": raw_value},
        )

        _, metadata, _ = run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        # Garbage in raw_step must not crash the worker and must not silently
        # disable the safety-net — the default 3.0 dB margin should kick in.
        assert metadata["bs_overlay_safety_net"]["margin_db"] == pytest.approx(3.0)
        assert metadata["weights"]["bs_seed_weight"] == 0.0
        assert metadata["bs_overlay_disabled_reason"]
