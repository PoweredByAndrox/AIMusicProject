from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.mac_melband_pitch_guard import pitch_shift_resample_same_length
from tools.run_mac_melband_pitch_guard_pipeline import run_pipeline


SAMPLE_RATE = 16_000


def _voice_like(seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    phase = np.cumsum(2.0 * math.pi * (185.0 + 12.0 * np.sin(2.0 * math.pi * 4.0 * t)) / SAMPLE_RATE)
    audio = np.zeros_like(t)
    for harmonic, gain in ((1, 0.16), (2, 0.10), (3, 0.07), (5, 0.04), (7, 0.025)):
        audio += gain * np.sin(harmonic * phase + 0.07 * harmonic)
    return (audio * (0.55 + 0.45 * np.sin(2.0 * math.pi * 2.1 * t) ** 2)).astype(np.float32)


def _instrumental_like(seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (
        0.11 * np.sin(2.0 * math.pi * 73.42 * t)
        + 0.07 * np.sin(2.0 * math.pi * 293.66 * t)
        + 0.04 * np.sin(2.0 * math.pi * 587.33 * t)
    ).astype(np.float32)


def _write(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.stack([audio, audio], axis=1).astype(np.float32), SAMPLE_RATE, subtype="PCM_24")
    return path


def test_pipeline_uses_mac_api_then_pitch_guard_and_stays_diagnostic(tmp_path: Path) -> None:
    source_run = tmp_path / "api" / "run"
    vocals = _voice_like()
    instrumental = _instrumental_like() + pitch_shift_resample_same_length(vocals, semitones=-5.0) * 0.42
    raw = vocals + instrumental
    vocals_path = _write(source_run / "vocals.flac", vocals)
    instrumental_path = _write(source_run / "instrumental.flac", instrumental)
    raw_path = _write(source_run / "raw_ref.flac", raw)
    (source_run / "stem_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "melband_roformer_vocals",
                "source_audio_path": "/Users/andrejviluha/AIMusicProject_win/testing_stuff_music/source/full.flac",
                "stems": {
                    "vocals": {"path": str(vocals_path)},
                    "instrumental": {"path": str(instrumental_path)},
                },
                "artifacts": {"raw_ref_flac": str(raw_path)},
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_submit(*, api_url: str, source_audio_path: str, output_subdir: str) -> dict[str, object]:
        calls.append({"api_url": api_url, "source_audio_path": source_audio_path, "output_subdir": output_subdir})
        return {
            "job_id": "mfixture01",
            "route_id": "melband_roformer_vocals",
            "status": "queued",
            "run_dir": str(source_run),
        }

    def fake_poll(*, api_url: str, job_id: str, poll_sec: float, timeout_sec: float) -> dict[str, object]:
        return {
            "job_id": job_id,
            "route_id": "melband_roformer_vocals",
            "status": "ok",
            "run_dir": str(source_run),
            "delivery_allowed": False,
            "presentation_status": "diagnostic_mps_api_only",
        }

    manifest = run_pipeline(
        mac_api_url="http://mac.local:8765",
        source_audio_path="/Users/andrejviluha/AIMusicProject_win/testing_stuff_music/source/full.flac",
        output_dir=tmp_path / "mel_guard001",
        api_output_subdir="mac_api/mel_guard001/api",
        submit_job=fake_submit,
        poll_job=fake_poll,
    )

    assert calls == [
        {
            "api_url": "http://mac.local:8765",
            "source_audio_path": "/Users/andrejviluha/AIMusicProject_win/testing_stuff_music/source/full.flac",
            "output_subdir": "mac_api/mel_guard001/api",
        }
    ]
    assert manifest["profile"] == "melband_pitch_guard_v1"
    assert manifest["route_id"] == "melband_roformer_vocals"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_melband_pitch_guard_only"
    assert manifest["guard_manifest"]["score"]["verdict"] == "fail"
    assert (tmp_path / "mel_guard001" / "progress.jsonl").exists()
