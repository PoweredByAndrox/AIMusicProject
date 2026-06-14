from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.apply_no_reference_vocal_silence_hallucination_repair import (
    build_vocal_silence_hallucination_repair,
)


SAMPLE_RATE = 8000


def sine(freq: float, *, seconds: float, amp: float = 0.2) -> np.ndarray:
    samples = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / float(SAMPLE_RATE)
    return (np.sin(2.0 * np.pi * freq * samples) * amp).astype(np.float32)


def stereo(mono: np.ndarray) -> np.ndarray:
    return np.column_stack([mono, mono]).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, stereo(audio), SAMPLE_RATE, subtype="FLOAT")
    return path


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + 1.0e-9)


def test_vocal_silence_hallucination_repair_suppresses_source_quiet_vocal_only(
    tmp_path: Path,
) -> None:
    quiet_mix = np.zeros(SAMPLE_RATE, dtype=np.float32)
    active_mix = sine(180.0, seconds=3.0, amp=0.35)
    mix = np.concatenate([quiet_mix, active_mix])
    hallucinated_vocal = np.concatenate(
        [
            sine(440.0, seconds=1.0, amp=0.18),
            sine(440.0, seconds=3.0, amp=0.18),
        ]
    )
    instrumental = mix - hallucinated_vocal

    mix_path = write_wav(tmp_path / "mix.wav", mix)
    vocal_path = write_wav(tmp_path / "vocal.wav", hallucinated_vocal)
    instrumental_path = write_wav(tmp_path / "minus.wav", instrumental)

    manifest = build_vocal_silence_hallucination_repair(
        mix_path=mix_path,
        vocal_path=vocal_path,
        instrumental_path=instrumental_path,
        output_dir=tmp_path / "out",
        track_id="track-a",
        base_candidate_id="vocal-a",
    )

    repaired, sample_rate = sf.read(manifest["rendered_vocal_candidate_path"], always_2d=True, dtype="float32")
    repaired_mono = repaired.mean(axis=1)
    original_mono = hallucinated_vocal

    assert sample_rate == SAMPLE_RATE
    assert manifest["repair_status"] == "rendered_source_quiet_vocal_silence_hallucination_repair"
    assert manifest["render_status"] == "rendered_vocal_source_quiet_silence_repair_candidate"
    assert manifest["requires_post_render_vocal_judge"] is True
    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "preview_only_not_for_listen_pack"
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["suppressed_window_count"] > 0
    assert rms(repaired_mono[:SAMPLE_RATE]) < rms(original_mono[:SAMPLE_RATE]) * 0.12
    assert rms(repaired_mono[SAMPLE_RATE:]) > rms(original_mono[SAMPLE_RATE:]) * 0.85

    candidate_row = json.loads(Path(manifest["candidate_row_jsonl"]).read_text(encoding="utf-8"))
    assert candidate_row["candidate_id"] == "source_quiet_vocal_silence_repair__vocal-a"
    assert candidate_row["vocal_path"] == manifest["rendered_vocal_candidate_path"]
    assert candidate_row["instrumental_path"] == str(instrumental_path)
    assert candidate_row["render_status"] == "rendered_vocal_source_quiet_silence_repair_candidate"
    assert candidate_row["requires_post_render_vocal_judge"] is True


def test_vocal_silence_hallucination_repair_blocks_when_no_source_quiet_vocal_windows(
    tmp_path: Path,
) -> None:
    mix = sine(180.0, seconds=2.0, amp=0.35)
    vocal = sine(440.0, seconds=2.0, amp=0.18)
    instrumental = mix - vocal

    manifest = build_vocal_silence_hallucination_repair(
        mix_path=write_wav(tmp_path / "mix.wav", mix),
        vocal_path=write_wav(tmp_path / "vocal.wav", vocal),
        instrumental_path=write_wav(tmp_path / "minus.wav", instrumental),
        output_dir=tmp_path / "out",
        track_id="track-a",
        base_candidate_id="vocal-a",
    )

    assert manifest["repair_status"] == "blocked_no_source_quiet_vocal_activity_windows"
    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["candidate_row_jsonl"] == ""
    assert manifest["rendered_vocal_candidate_path"] == ""
