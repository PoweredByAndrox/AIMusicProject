from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_vocal_guided_minus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_vocal_guided_minus", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def test_vocal_guided_minus_fills_vocal_active_base_dip(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 16000
    times = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
    instrumental = (
        0.16 * np.sin(2.0 * np.pi * 130.0 * times)
        + 0.08 * np.sin(2.0 * np.pi * 1800.0 * times)
    )
    vocal = 0.14 * np.sin(2.0 * np.pi * 440.0 * times)
    active_slice = slice(int(0.55 * sample_rate), int(1.20 * sample_rate))
    vocal_envelope = np.zeros_like(vocal)
    vocal_envelope[active_slice] = 1.0
    vocal *= vocal_envelope

    raw = np.stack([instrumental + vocal, instrumental + vocal], axis=1).astype(np.float32)
    selected_vocals = np.stack([vocal, vocal], axis=1).astype(np.float32)
    base_minus = np.stack([instrumental, instrumental], axis=1).astype(np.float32)
    base_minus[active_slice] *= 0.42

    raw_path = tmp_path / "raw.flac"
    vocals_path = tmp_path / "vocals.flac"
    base_path = tmp_path / "base.flac"
    output_dir = tmp_path / "out"
    listen_dir = tmp_path / "listen"
    sf.write(str(raw_path), raw, sample_rate, subtype="PCM_24")
    sf.write(str(vocals_path), selected_vocals, sample_rate, subtype="PCM_24")
    sf.write(str(base_path), base_minus, sample_rate, subtype="PCM_24")

    manifest = module.run_build(
        Namespace(
            raw_audio_path=str(raw_path),
            selected_vocals_path=str(vocals_path),
            base_minus_path=str(base_path),
            output_dir=str(output_dir),
            listen_dir=str(listen_dir),
            listen_minus_label="02_vocal_guided_minus.flac",
            vocal_subtract_gain=1.0,
            base_blend_active=1.0,
            base_blend_inactive=0.0,
            activity_floor_ratio=0.02,
            activity_smoothing_sec=0.04,
            max_peak=0.999,
        )
    )

    guided, guided_sr = sf.read(str(output_dir / "03_vocal_guided_minus.flac"), always_2d=True, dtype="float32")
    assert guided_sr == sample_rate
    assert (listen_dir / "01_selected_vocals.flac").exists()
    assert (listen_dir / "02_vocal_guided_minus.flac").exists()
    assert manifest["debug"]["strategy"] == "base_minus_blended_with_raw_minus_from_selected_vocal"

    expected = np.stack([instrumental, instrumental], axis=1)
    base_error = _rms(base_minus[active_slice] - expected[active_slice])
    guided_error = _rms(guided[active_slice] - expected[active_slice])
    assert guided_error < base_error * 0.35


def test_raw_minus_mode_subtracts_selected_vocal_without_base(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    instrumental = 0.12 * np.sin(2.0 * np.pi * 220.0 * times)
    vocal = 0.08 * np.sin(2.0 * np.pi * 660.0 * times)
    raw = np.stack([instrumental + vocal, instrumental + vocal], axis=1).astype(np.float32)
    selected_vocals = np.stack([vocal, vocal], axis=1).astype(np.float32)

    raw_path = tmp_path / "raw.flac"
    vocals_path = tmp_path / "vocals.flac"
    output_dir = tmp_path / "out"
    sf.write(str(raw_path), raw, sample_rate, subtype="PCM_24")
    sf.write(str(vocals_path), selected_vocals, sample_rate, subtype="PCM_24")

    manifest = module.run_build(
        Namespace(
            raw_audio_path=str(raw_path),
            selected_vocals_path=str(vocals_path),
            base_minus_path=None,
            output_dir=str(output_dir),
            listen_dir=None,
            listen_minus_label="02_vocal_guided_minus.flac",
            vocal_subtract_gain=1.0,
            base_blend_active=0.72,
            base_blend_inactive=0.18,
            activity_floor_ratio=0.035,
            activity_smoothing_sec=0.12,
            max_peak=0.999,
        )
    )

    guided, guided_sr = sf.read(str(output_dir / "03_vocal_guided_minus.flac"), always_2d=True, dtype="float32")
    assert guided_sr == sample_rate
    assert manifest["debug"]["strategy"] == "raw_minus_from_selected_vocal"
    expected = np.stack([instrumental, instrumental], axis=1)
    assert _rms(guided - expected) < 2e-4


def test_reconstruction_metric_ignores_peak_limiter_delta(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    instrumental = 0.72 * np.sin(2.0 * np.pi * 220.0 * times)
    vocal = 0.06 * np.sin(2.0 * np.pi * 660.0 * times)
    raw = np.stack([instrumental + vocal, instrumental + vocal], axis=1).astype(np.float32)
    selected_vocals = np.stack([vocal, vocal], axis=1).astype(np.float32)

    raw_path = tmp_path / "raw.flac"
    vocals_path = tmp_path / "vocals.flac"
    output_dir = tmp_path / "out"
    sf.write(str(raw_path), raw, sample_rate, subtype="PCM_24")
    sf.write(str(vocals_path), selected_vocals, sample_rate, subtype="PCM_24")

    manifest = module.run_build(
        Namespace(
            raw_audio_path=str(raw_path),
            selected_vocals_path=str(vocals_path),
            base_minus_path=None,
            output_dir=str(output_dir),
            listen_dir=None,
            listen_minus_label="02_vocal_guided_minus.flac",
            vocal_subtract_gain=1.0,
            base_blend_active=0.72,
            base_blend_inactive=0.18,
            activity_floor_ratio=0.035,
            activity_smoothing_sec=0.12,
            max_peak=0.5,
        )
    )

    assert manifest["debug"]["raw_minus_peak_scale"] < 1.0
    assert manifest["metrics"]["reconstruction_error_rms_dbfs"] < -120.0
    assert manifest["metrics"]["post_limiter_delta_rms_dbfs"] > -30.0
    assert manifest["metrics"]["post_limiter_reconstruction_error_rms_dbfs"] > -30.0
