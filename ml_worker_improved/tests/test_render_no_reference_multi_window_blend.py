from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_multi_window_blend import build_multi_window_preview, write_outputs


SAMPLE_RATE = 16_000


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def contract_row(tmp_path: Path, *, start_sec: float, duration_sec: float, render_status: str = "render_ready") -> dict:
    base = np.zeros((SAMPLE_RATE * 4, 2), dtype=np.float32)
    alt = np.ones((SAMPLE_RATE * 4, 2), dtype=np.float32) * 0.5
    vocal = np.zeros((SAMPLE_RATE * 4, 2), dtype=np.float32)
    return {
        "track_id": "track-a",
        "base_candidate_id": "base",
        "alternate_candidate_id": "alt",
        "base_minus_path": write_wav(tmp_path / "base_minus.wav", base),
        "alt_minus_path": write_wav(tmp_path / "alt_minus.wav", alt),
        "base_vocal_path": write_wav(tmp_path / "base_vocal.wav", vocal),
        "alt_vocal_path": write_wav(tmp_path / "alt_vocal.wav", vocal),
        "source_stem": "minus",
        "selected_target_stem": "minus",
        "window_start_sec": start_sec,
        "window_duration_sec": duration_sec,
        "crossfade_ms": 0,
        "render_status": render_status,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }


def test_multi_window_blend_renders_one_full_length_minus(tmp_path: Path) -> None:
    rows = [
        contract_row(tmp_path, start_sec=0.5, duration_sec=0.5),
        contract_row(tmp_path, start_sec=2.0, duration_sec=0.5),
    ]

    result = build_multi_window_preview(rows, tmp_path / "out")

    minus_path = Path(result["rendered_minus_path"])
    audio, sample_rate = sf.read(minus_path, always_2d=True, dtype="float32")

    assert sample_rate == SAMPLE_RATE
    assert audio.shape[0] == SAMPLE_RATE * 4
    assert np.max(audio[: SAMPLE_RATE // 2]) == 0.0
    assert np.isclose(np.mean(audio[SAMPLE_RATE // 2 : SAMPLE_RATE]), 0.5, atol=1.0e-4)
    assert np.isclose(np.mean(audio[SAMPLE_RATE * 2 : SAMPLE_RATE * 2 + SAMPLE_RATE // 2]), 0.5, atol=1.0e-4)
    assert result["applied_window_count"] == 2
    assert result["release_ready"] is False
    assert result["can_publish_as_result"] is False


def test_multi_window_blend_ignores_blocked_rows(tmp_path: Path) -> None:
    rows = [
        contract_row(tmp_path, start_sec=0.5, duration_sec=0.5, render_status="blocked_overlap_conflict"),
        contract_row(tmp_path, start_sec=2.0, duration_sec=0.5),
    ]

    result = build_multi_window_preview(rows, tmp_path / "out")

    assert result["applied_window_count"] == 1
    assert result["blocked_or_skipped_window_count"] == 1


def test_multi_window_blend_writes_deterministic_manifest(tmp_path: Path) -> None:
    result = build_multi_window_preview([contract_row(tmp_path, start_sec=0.5, duration_sec=0.5)], tmp_path / "out")
    write_outputs(result, tmp_path / "out" / "manifest.json")

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["rendered_minus_path"].endswith("/audio/minus.flac")
    assert manifest["release_ready"] is False
    assert manifest["can_mark_release_ready"] is False
