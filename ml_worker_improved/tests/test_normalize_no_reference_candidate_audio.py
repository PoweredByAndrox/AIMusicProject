from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.normalize_no_reference_candidate_audio import normalize_candidate_audio_to_input_shape


def sine(freq: float, sample_rate: int, seconds: float, channels: int = 1) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    audio = (0.2 * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    if channels == 1:
        return audio
    return np.stack([audio for _ in range(channels)], axis=1)


def write_audio(path: Path, *, sample_rate: int, seconds: float, channels: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, sine(220.0, sample_rate, seconds, channels), sample_rate)
    return path


def test_48k_input_and_44100_output_becomes_normalized_candidate(tmp_path: Path) -> None:
    input_audio = write_audio(tmp_path / "input_48k_stereo.wav", sample_rate=48_000, seconds=1.0, channels=2)
    raw_output = write_audio(tmp_path / "raw" / "vocals_44100_mono.wav", sample_rate=44_100, seconds=1.0, channels=1)
    normalized_output = tmp_path / "normalized" / "vocals_48k_stereo.wav"
    raw_bytes_before = raw_output.read_bytes()

    report = normalize_candidate_audio_to_input_shape(
        input_audio_path=input_audio,
        raw_output_path=raw_output,
        normalized_output_path=normalized_output,
    )

    normalized_info = sf.info(str(normalized_output))
    input_info = sf.info(str(input_audio))

    assert report["shape_normalization_applied"] is True
    assert "sample_rate_mismatch" in report["shape_normalization_reason"]
    assert "channel_mismatch" in report["shape_normalization_reason"]
    assert report["original_files_overwritten"] is False
    assert report["release_ready"] is False
    assert raw_output.read_bytes() == raw_bytes_before
    assert normalized_output.exists()
    assert normalized_output != raw_output
    assert normalized_info.samplerate == input_info.samplerate
    assert normalized_info.channels == input_info.channels
    assert normalized_info.frames == input_info.frames
