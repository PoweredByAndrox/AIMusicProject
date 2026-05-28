"""End-to-end integration tests for the S1+S2+S3+S5 fix-up.

This test does **not** spin up Docker; it runs the production
restoration-worker module directly so it can be executed inside the
``services/restoration-worker`` container (where ``pytest``, ``numpy``
and ``pydantic`` are already installed) or via
``pytest tests/integration`` from a host with the same packages.

What the test enforces is the *full-pipeline contract* from the
fix-up roadmap, with hard ``assert``s on the STRICT acceptance
thresholds the user signed off on:

1. **S2 safety-net green path** — when the BS overlay is *cleaner*
   than Demucs base, the safety-net keeps the seed weight intact and
   produces a hybrid whose ``vocal_bleed.bleed_db_rel_total`` stays
   below **-30 dB** on a synthetic, spectrally-orthogonal mix.
2. **S2 safety-net red path** — when the BS overlay is *worse* than
   Demucs base by more than ``bs_overlay_safety_margin_db`` (default
   3 dB), the safety-net zeros the contribution and the final hybrid
   never gets dirtier than the base.
3. **S5 metadata truthfulness** — ``metadata.weights`` reports the
   *applied* weights post-override, ``metadata.bs_overlay_disabled_reason``
   names which slot was disabled and why, and the human-readable
   ``notes`` string includes ``safety_net[...]`` so manifests are
   self-describing.
4. **Escape hatch** — ``bs_overlay_safety_margin_db: 0.0`` reproduces
   the legacy behaviour: no override, even on dirty seeds.

For the ``_resolve_instrumental_strategy`` half of the contract (S1)
see the dedicated unit suite at
``services/bsroformer-worker/tests/test_runtime_instrumental_strategy.py``,
which runs inside the bsroformer-worker container.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The test is designed to run in either of two contexts:
#   (a) Inside the ``restoration-worker`` container, where the worker
#       package lives at ``/app/app/`` and is already importable as
#       ``app.main`` via the ``WORKDIR /app`` declared in the Dockerfile.
#   (b) From the host repo root, where the worker package lives at
#       ``services/restoration-worker/app/`` and needs to be added to
#       ``sys.path`` explicitly.
# We try (a) first; if that fails, we fall back to (b). Doing it in this
# order means that tests run in-container don't need any path tweaks
# and host-side ``pytest tests/integration`` still works without
# manually setting ``PYTHONPATH``.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESTORATION_ROOT_HOST = REPO_ROOT / "services" / "restoration-worker"
RESTORATION_ROOT_CONTAINER = Path("/app")
for candidate in (RESTORATION_ROOT_CONTAINER, RESTORATION_ROOT_HOST):
    if (candidate / "app" / "main.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

import app.main as restoration_main  # type: ignore  # noqa: E402
from app.schemas import RunRequest as RestorationRunRequest  # type: ignore  # noqa: E402

SR = 44100
N = SR  # 1 second of stereo audio is enough for spectral metrics

# STRICT acceptance thresholds, mirrored from the user-signed-off plan:
#   - vocal_bleed.final_vocals.bleed_db_rel_total <= -30.0 dB
#   - safety-net must never make a track dirtier than the base
STRICT_BLEED_DB = -30.0
SAFETY_NET_BASE_TOLERANCE_DB = 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_orthogonal_components() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(instrumental, vocals)`` as ``(2, N)`` float32 stereo.

    220 Hz vs 880 Hz are spectrally orthogonal so the projection energy
    of one onto the other is essentially zero, which makes the
    ``_vocal_contamination_db`` metric independent of any model
    weights or hyperparameters.
    """
    t = np.arange(N, dtype=np.float32) / float(SR)
    instr_mono = 0.30 * np.sin(2.0 * np.pi * 220.0 * t)
    vocals_mono = 0.30 * np.sin(2.0 * np.pi * 880.0 * t)
    instr = np.stack([instr_mono, instr_mono], axis=0).astype(np.float32)
    vocals = np.stack([vocals_mono, vocals_mono], axis=0).astype(np.float32)
    return instr, vocals


def _make_request(
    tmp_path: Path,
    base_path: Path,
    bs_seed_path: Path,
    vocals_path: Path,
    *,
    raw_step_extra: dict | None = None,
) -> RestorationRunRequest:
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_step: dict = {"task": "tandem_instrumental_hybrid"}
    if raw_step_extra:
        raw_step.update(raw_step_extra)
    return RestorationRunRequest(
        job_id="integration-test",
        phase_id="phase-test",
        step_index=0,
        track_name="synthetic",
        input_path=str(base_path),
        output_root=str(output_root),
        target_stem="instrumental_hybrid",
        task="tandem_instrumental_hybrid",
        build_from_paths={
            "bs_resurrected_instrumental_mix": str(bs_seed_path),
            "vocals": str(vocals_path),
        },
        raw_step=raw_step,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# S2 + S5 + STRICT bleed assertion
# ---------------------------------------------------------------------------


class TestFullHybridPipelineAgainstReference:
    def test_clean_pipeline_meets_strict_bleed_threshold(self, tmp_path):
        instr, vocals = _make_orthogonal_components()

        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed.wav"
        vocals_path = tmp_path / "vocals.wav"

        restoration_main._write_wave_file(base_path, instr, SR)
        # Tiny gain offset on bs_seed mimics ML-cleanup output without
        # injecting any vocal-band energy. Crucially we do NOT add
        # white noise — noise has a non-zero projection on the vocal
        # band even when there is no vocal content, which would make
        # the safety-net trip relative to the perfectly-clean base.
        restoration_main._write_wave_file(bs_seed_path, (instr * 1.02).astype(np.float32), SR)
        restoration_main._write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, base_path, bs_seed_path, vocals_path)
        named, metadata, _notes = restoration_main.run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        # Safety-net green path: profile defaults are preserved.
        assert metadata["weights"]["bs_seed_weight"] == pytest.approx(0.25)
        assert metadata["bs_overlay_disabled_reason"] is None
        assert metadata["bs_overlay_safety_net"]["overrides_applied"] == {}

        # STRICT acceptance threshold — clean synthetic must clear -30 dB.
        hybrid_audio, _sr = restoration_main._read_wave_file(
            Path(named[request.target_stem])
        )
        bleed_db = restoration_main._vocal_contamination_db(hybrid_audio, vocals, SR)
        assert bleed_db < STRICT_BLEED_DB, (
            "STRICT acceptance threshold breached: clean synthetic hybrid "
            f"must keep bleed_db_rel_total < {STRICT_BLEED_DB:.1f} dB, "
            f"got {bleed_db:.2f} dB"
        )

    def test_dirty_seed_triggers_safety_net_and_reports_truthful_metadata(self, tmp_path):
        instr, vocals = _make_orthogonal_components()

        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed_dirty.wav"
        vocals_path = tmp_path / "vocals.wav"

        restoration_main._write_wave_file(base_path, instr, SR)
        # bs_seed = base + 0.5 * vocals → ~-6 dB vocal contamination,
        # 30+ dB worse than the clean base.
        restoration_main._write_wave_file(
            bs_seed_path, (instr + 0.5 * vocals).astype(np.float32), SR,
        )
        restoration_main._write_wave_file(vocals_path, vocals, SR)

        request = _make_request(tmp_path, base_path, bs_seed_path, vocals_path)
        named, metadata, notes = restoration_main.run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        # S5 contract: applied weight is what was multiplied into the
        # signal, not the desired profile weight.
        assert metadata["weights"]["bs_seed_weight"] == 0.0

        reason = metadata["bs_overlay_disabled_reason"]
        assert reason and "bs_seed_weight" in reason and "by >3.00dB" in reason

        diag = metadata["bs_overlay_safety_net"]
        assert diag["margin_db"] == pytest.approx(3.0)
        assert diag["base_bleed_db"] < diag["bs_seed_bleed_db"]

        overrides = diag["overrides_applied"]
        assert overrides["bs_seed_weight"]["from"] == pytest.approx(0.25)
        assert overrides["bs_seed_weight"]["to"] == 0.0

        # Notes must surface the override so manifests are self-describing.
        assert "safety_net[bs_seed_weight]" in notes

        # Final hybrid must not be dirtier than the base — the whole
        # point of the safety-net.
        hybrid_audio, _sr = restoration_main._read_wave_file(
            Path(named[request.target_stem])
        )
        hybrid_bleed = restoration_main._vocal_contamination_db(hybrid_audio, vocals, SR)
        base_bleed = restoration_main._vocal_contamination_db(instr, vocals, SR)
        assert hybrid_bleed <= base_bleed + SAFETY_NET_BASE_TOLERANCE_DB, (
            "Safety-net engaged but final hybrid still got dirtier than "
            f"base: hybrid={hybrid_bleed:.2f} dB, base={base_bleed:.2f} dB"
        )

    def test_zero_margin_escape_hatch_disables_safety_net(self, tmp_path):
        instr, vocals = _make_orthogonal_components()
        base_path = tmp_path / "base.wav"
        bs_seed_path = tmp_path / "bs_seed_dirty.wav"
        vocals_path = tmp_path / "vocals.wav"

        restoration_main._write_wave_file(base_path, instr, SR)
        restoration_main._write_wave_file(
            bs_seed_path, (instr + 0.5 * vocals).astype(np.float32), SR,
        )
        restoration_main._write_wave_file(vocals_path, vocals, SR)

        request = _make_request(
            tmp_path,
            base_path,
            bs_seed_path,
            vocals_path,
            raw_step_extra={"bs_overlay_safety_margin_db": 0.0},
        )
        _named, metadata, _notes = restoration_main.run_tandem_instrumental_hybrid(
            request, Path(request.output_root)
        )

        # Margin <= 0 is the documented escape hatch — the override
        # path must not fire even when the seed is obviously dirty.
        assert metadata["weights"]["bs_seed_weight"] == pytest.approx(0.25)
        assert metadata["bs_overlay_disabled_reason"] is None
        assert metadata["bs_overlay_safety_net"]["overrides_applied"] == {}
