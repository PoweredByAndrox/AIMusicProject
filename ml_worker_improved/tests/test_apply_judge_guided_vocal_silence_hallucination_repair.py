from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from tools.apply_judge_guided_vocal_silence_hallucination_repair import (
    build_judge_guided_vocal_silence_hallucination_repair,
    build_manifest,
    select_target_windows,
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


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + 1.0e-9)


def test_selects_vocal_judge_silence_hallucination_windows() -> None:
    rows = [
        {
            "candidate_id": "vocal-a",
            "vocal_path": "/app/vocal-a.wav",
            "window_metrics": [
                {
                    "start_sec": 12.0,
                    "duration_sec": 6.0,
                    "silence_hallucination_score": 0.91,
                    "reasons": ["silence_hallucination"],
                },
                {
                    "start_sec": 24.0,
                    "duration_sec": 6.0,
                    "silence_hallucination_score": 0.11,
                    "reasons": ["no_reference_proxy_passed"],
                },
                {
                    "start_sec": 12.0,
                    "duration_sec": 6.0,
                    "silence_hallucination_score": 0.91,
                    "reasons": ["silence_hallucination"],
                },
            ],
        },
        {
            "candidate_id": "vocal-b",
            "vocal_path": "/app/vocal-b.wav",
            "window_metrics": [
                {
                    "start_sec": 0.0,
                    "duration_sec": 6.0,
                    "silence_hallucination_score": 0.99,
                    "reasons": ["silence_hallucination"],
                }
            ],
        },
    ]

    windows = select_target_windows(
        rows,
        candidate_id="vocal-a",
        vocal_path="/app/vocal-a.wav",
        hallucination_threshold=0.75,
    )

    assert len(windows) == 1
    assert windows[0]["start_sec"] == 12.0
    assert windows[0]["duration_sec"] == 6.0
    assert windows[0]["target_window_source"] == "no_reference_vocal_windowed_judge"
    assert windows[0]["silence_hallucination_score"] == 0.91


def test_manifest_is_candidate_only_and_requires_post_vocal_judge() -> None:
    manifest = build_manifest(
        track_id="track-a",
        candidate_id="vocal-a__judge_guided_silence_repair",
        raw_audio_path=Path("/app/raw.flac"),
        base_vocal_path=Path("/app/base_vocal.flac"),
        instrumental_path=Path("/app/minus.flac"),
        judge_jsonl_path=Path("/app/judge.jsonl"),
        output_path=Path("/app/out/vocal_judge_guided_silence_repair.wav"),
        candidate_row_jsonl=Path("/app/out/candidate_pair.jsonl"),
        sample_rate=44100,
        frames=44100,
        selected_windows=[{"start_sec": 0.0, "duration_sec": 1.0}],
        repair_metrics={"gate_active_ratio": 0.1},
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "preview_only_not_for_listen_pack"
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["render_status"] == "rendered_vocal_judge_guided_silence_repair_candidate"
    assert manifest["requires_post_render_vocal_judge"] is True
    assert manifest["candidate_row"]["render_status"] == "rendered_vocal_judge_guided_silence_repair_candidate"
    assert manifest["candidate_row"]["vocal_path"] == "/app/out/vocal_judge_guided_silence_repair.wav"
    assert manifest["candidate_row"]["instrumental_path"] == "/app/minus.flac"
    assert manifest["safety"]["thresholds_relaxed"] is False
    assert manifest["safety"]["training_started"] is False


def test_judge_guided_vocal_silence_repair_suppresses_selected_windows_only(tmp_path: Path) -> None:
    quiet_mix = np.zeros(SAMPLE_RATE, dtype=np.float32)
    active_mix = sine(180.0, seconds=3.0, amp=0.35)
    mix = np.concatenate([quiet_mix, active_mix])
    vocal = sine(440.0, seconds=4.0, amp=0.18)
    instrumental = mix - vocal

    mix_path = write_wav(tmp_path / "mix.wav", mix)
    vocal_path = write_wav(tmp_path / "vocal.wav", vocal)
    instrumental_path = write_wav(tmp_path / "minus.wav", instrumental)
    judge_jsonl = write_jsonl(
        tmp_path / "vocal_judge.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "vocal-a",
                "mix_path": str(mix_path),
                "vocal_path": str(vocal_path),
                "instrumental_path": str(instrumental_path),
                "score_mode": "vocal",
                "window_metrics": [
                    {
                        "start_sec": 0.0,
                        "duration_sec": 2.0,
                        "silence_hallucination_score": 0.96,
                        "reasons": ["silence_hallucination"],
                    }
                ],
            }
        ],
    )

    manifest = build_judge_guided_vocal_silence_hallucination_repair(
        SimpleNamespace(
            raw_audio_path=mix_path,
            base_vocal_path=vocal_path,
            instrumental_path=instrumental_path,
            windowed_judge_jsonl=judge_jsonl,
            output_dir=tmp_path / "out",
            candidate_id="vocal-a",
            track_id="track-a",
            hallucination_threshold=0.75,
            attenuation_gain=0.0,
            gate_smoothing_sec=0.01,
            window_margin_sec=0.0,
        )
    )

    repaired, sample_rate = sf.read(manifest["rendered_vocal_candidate_path"], always_2d=True, dtype="float32")
    repaired_mono = repaired.mean(axis=1)

    assert sample_rate == SAMPLE_RATE
    assert manifest["repair"]["selected_window_count"] == 1
    assert rms(repaired_mono[:SAMPLE_RATE]) < rms(vocal[:SAMPLE_RATE]) * 0.12
    assert rms(repaired_mono[SAMPLE_RATE : 2 * SAMPLE_RATE]) > rms(vocal[SAMPLE_RATE : 2 * SAMPLE_RATE]) * 0.85
    assert rms(repaired_mono[2 * SAMPLE_RATE :]) > rms(vocal[2 * SAMPLE_RATE :]) * 0.85

    candidate_row = json.loads(Path(manifest["candidate_row_jsonl"]).read_text(encoding="utf-8"))
    assert candidate_row["candidate_id"] == "vocal-a__judge_guided_silence_repair"
    assert candidate_row["vocal_path"] == manifest["rendered_vocal_candidate_path"]
    assert candidate_row["selected_minus_path"] == str(instrumental_path)
