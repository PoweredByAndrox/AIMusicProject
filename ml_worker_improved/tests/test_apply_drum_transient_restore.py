from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import apply_drum_transient_restore as transient_restore


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def test_transient_restore_writes_benchmark_compatible_no_reference_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    sample_rate = 8000
    duration_sec = 1.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    pulse = np.zeros_like(t)
    pulse[800:820] = np.hanning(20).astype(np.float32)
    clean = 0.6 * pulse
    base = clean + 0.05 * np.sin(2.0 * np.pi * 520.0 * t).astype(np.float32)

    base_path = tmp_path / "base.flac"
    clean_path = tmp_path / "clean.flac"
    output_dir = tmp_path / "out"
    _write_audio(base_path, np.stack([base, base], axis=1), sample_rate)
    _write_audio(clean_path, np.stack([clean, clean], axis=1), sample_rate)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_drum_transient_restore.py",
            str(base_path),
            str(clean_path),
            str(output_dir),
            "--dsp-device",
            "cpu",
        ],
    )

    assert transient_restore.main() == 0

    manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "drum_transient_restore"
    assert manifest["contract"]["benchmark_compatible"] is True
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["contract"]["delivery_allowed_without_quality_gate"] is False
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["per_track_hardcode_allowed"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    assert manifest["contract"]["quality_gate_required"] is True
    assert manifest["route"]["mode"] == "drum_transient_restore_v1"
    assert manifest["stems"]["drums"]["role"] == "drum_transient_restored_candidate"
    assert Path(manifest["stems"]["drums"]["path"]).name == "drums_transient_restored.flac"
