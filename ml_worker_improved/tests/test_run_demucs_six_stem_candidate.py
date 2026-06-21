from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_demucs_six_stem_candidate import (
    apply_residual_correction,
    collect_demucs_outputs,
    run_demucs_six_stem_candidate,
)
from tools.stem_contamination_common import STEMS


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def test_collect_demucs_outputs_copies_six_stems(tmp_path: Path) -> None:
    demucs_track_dir = tmp_path / "demucs" / "htdemucs_6s" / "mix"
    for stem in STEMS:
        write_wav(demucs_track_dir / f"{stem}.wav", sine(220.0))
    stems_dir = tmp_path / "stems"

    manifest = collect_demucs_outputs(
        demucs_output_root=tmp_path / "demucs",
        model_name="htdemucs_6s",
        track_name="mix",
        stems_dir=stems_dir,
    )

    assert manifest["collection_status"] == "completed"
    assert manifest["missing_stems"] == []
    assert set(manifest["stem_paths"]) == set(STEMS)
    for stem in STEMS:
        assert (stems_dir / f"{stem}.flac").exists()


def test_collect_demucs_outputs_matches_target_audio_shape(tmp_path: Path) -> None:
    demucs_track_dir = tmp_path / "demucs" / "htdemucs_6s" / "mix"
    for stem in STEMS:
        demucs_track_dir.mkdir(parents=True, exist_ok=True)
        sf.write(demucs_track_dir / f"{stem}.wav", sine(220.0, seconds=0.1), 8_000)
    stems_dir = tmp_path / "stems"

    collect_demucs_outputs(
        demucs_output_root=tmp_path / "demucs",
        model_name="htdemucs_6s",
        track_name="mix",
        stems_dir=stems_dir,
        target_sample_rate=16_000,
        target_sample_count=4_000,
    )

    info = sf.info(stems_dir / "vocals.flac")
    assert info.samplerate == 16_000
    assert info.frames == 4_000


def test_run_demucs_uses_integer_segment_when_possible(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "mix.flac", sine(220.0))
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

        class Completed:
            returncode = 1
            stdout = ""
            stderr = "forced"

        return Completed()

    monkeypatch.setattr("tools.run_demucs_six_stem_candidate.subprocess.run", fake_run)

    run_demucs_six_stem_candidate(
        input_audio=input_audio,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        segment_sec=6.0,
    )

    idx = captured["command"].index("--segment")
    assert captured["command"][idx + 1] == "6"
    device_idx = captured["command"].index("-d")
    assert captured["command"][device_idx + 1] == "cuda"


def test_residual_correction_restores_mix_conservation(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.flac", sine(220.0, amp=0.2))
    stems_dir = tmp_path / "stems"
    for stem in STEMS:
        write_wav(stems_dir / f"{stem}.flac", np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32))

    report = apply_residual_correction(input_audio=mix, stems_dir=stems_dir, target_stem="other")

    assert report["applied"] is True
    mix_audio, _ = sf.read(mix, dtype="float32", always_2d=True)
    summed = np.zeros_like(mix_audio)
    for stem in STEMS:
        audio, _ = sf.read(stems_dir / f"{stem}.flac", dtype="float32", always_2d=True)
        summed += audio
    np.testing.assert_allclose(summed, mix_audio, atol=2e-4)
