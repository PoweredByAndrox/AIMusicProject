from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_segment_switch_candidate import (
    build_segment_switch_candidate,
    main,
)


def write_wav(path: Path, data: np.ndarray, sample_rate: int = 8000) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sample_rate)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def sine(freq: float, seconds: float, sample_rate: int = 8000, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    mono = amp * np.sin(2.0 * np.pi * freq * t)
    return np.column_stack([mono, mono])


def candidate_row(tmp_path: Path, candidate_id: str, minus: np.ndarray) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": str(tmp_path / "mix.wav"),
        "vocal_path": str(tmp_path / "vocal.wav"),
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", minus),
        "minus_path": str(tmp_path / candidate_id / "minus.wav"),
    }


def judge_row(candidate: dict, first_score: float, second_score: float, *, severe_second: bool = False) -> dict:
    second_reasons = ["severe_vocal_bleed_in_minus"] if severe_second else ["no_reference_proxy_passed"]
    return {
        "track_id": candidate["track_id"],
        "candidate_id": candidate["candidate_id"],
        "mix_path": candidate["mix_path"],
        "vocal_path": candidate["vocal_path"],
        "instrumental_path": candidate["instrumental_path"],
        "window_metrics": [
            {
                "window_kind": "window",
                "window_start_sec": 0.0,
                "window_duration_sec": 1.0,
                "minus_score": first_score,
                "window_fitness": first_score,
                "reasons": ["no_reference_proxy_passed"],
            },
            {
                "window_kind": "window",
                "window_start_sec": 1.0,
                "window_duration_sec": 1.0,
                "minus_score": second_score,
                "window_fitness": second_score,
                "reasons": second_reasons,
            },
        ],
    }


def test_segment_switch_uses_best_safe_minus_per_window(tmp_path: Path) -> None:
    mix = sine(110, 2.0, amp=0.4)
    vocal = sine(330, 2.0, amp=0.05)
    write_wav(tmp_path / "mix.wav", mix)
    write_wav(tmp_path / "vocal.wav", vocal)
    first_minus = np.concatenate([sine(110, 1.0, amp=0.2), sine(110, 1.0, amp=0.02)])
    second_minus = np.concatenate([sine(220, 1.0, amp=0.02), sine(220, 1.0, amp=0.25)])
    first = candidate_row(tmp_path, "first", first_minus)
    second = candidate_row(tmp_path, "second", second_minus)

    result = build_segment_switch_candidate(
        [first, second],
        [judge_row(first, 0.95, 0.2), judge_row(second, 0.1, 0.96)],
        output_dir=tmp_path / "out",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "rendered_preview_candidate"
    assert result["delivery_allowed"] is False
    assert result["can_mark_release_ready"] is False
    assert result["selected_window_count"] == 2
    assert [window["selected_candidate_id"] for window in result["selected_windows"]] == ["first", "second"]
    rendered, sample_rate = sf.read(result["rendered_minus_preview_path"], always_2d=True)
    assert sample_rate == 8000
    assert rendered.shape[0] == mix.shape[0]


def test_segment_switch_accepts_real_window_judge_start_duration_schema(tmp_path: Path) -> None:
    mix = sine(110, 2.0, amp=0.4)
    vocal = sine(330, 2.0, amp=0.05)
    write_wav(tmp_path / "mix.wav", mix)
    write_wav(tmp_path / "vocal.wav", vocal)
    first = candidate_row(tmp_path, "first", sine(110, 2.0, amp=0.2))
    second = candidate_row(tmp_path, "second", sine(220, 2.0, amp=0.2))

    first_judge = judge_row(first, 0.91, 0.1)
    second_judge = judge_row(second, 0.2, 0.92)
    for row in (first_judge, second_judge):
        for window in row["window_metrics"]:
            window["start_sec"] = window.pop("window_start_sec")
            window["duration_sec"] = window.pop("window_duration_sec")

    result = build_segment_switch_candidate(
        [first, second],
        [first_judge, second_judge],
        output_dir=tmp_path / "out",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "rendered_preview_candidate"
    assert [window["window_start_sec"] for window in result["selected_windows"]] == [0.0, 1.0]
    assert [window["selected_candidate_id"] for window in result["selected_windows"]] == ["first", "second"]


def test_segment_switch_blocks_window_with_only_severe_candidates(tmp_path: Path) -> None:
    mix = sine(110, 2.0, amp=0.4)
    vocal = sine(330, 2.0, amp=0.05)
    write_wav(tmp_path / "mix.wav", mix)
    write_wav(tmp_path / "vocal.wav", vocal)
    first = candidate_row(tmp_path, "first", sine(110, 2.0, amp=0.2))

    result = build_segment_switch_candidate(
        [first],
        [judge_row(first, 0.95, 0.95, severe_second=True)],
        output_dir=tmp_path / "out",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "blocked_incomplete_window_coverage"
    assert result["rendered_minus_preview_path"] == ""
    assert result["blocked_window_count"] == 1
    assert result["audio_exported"] is False


def test_segment_switch_cli_is_deterministic_and_preview_only(tmp_path: Path) -> None:
    mix = sine(110, 2.0, amp=0.4)
    vocal = sine(330, 2.0, amp=0.05)
    write_wav(tmp_path / "mix.wav", mix)
    write_wav(tmp_path / "vocal.wav", vocal)
    first = candidate_row(tmp_path, "first", sine(110, 2.0, amp=0.2))
    candidate_jsonl = write_jsonl(tmp_path / "candidates.jsonl", [first])
    judge_jsonl = write_jsonl(tmp_path / "judge.jsonl", [judge_row(first, 0.95, 0.96)])
    out_dir = tmp_path / "out"

    args = [
        "--candidate-pool-jsonl",
        str(candidate_jsonl),
        "--minus-windowed-judge-jsonl",
        str(judge_jsonl),
        "--output-dir",
        str(out_dir),
        "--manifest-json",
        str(out_dir / "manifest.json"),
        "--summary-md",
        str(out_dir / "summary.md"),
        "--crossfade-ms",
        "0",
    ]
    assert main(args) == 0
    first_manifest = (out_dir / "manifest.json").read_text(encoding="utf-8")
    assert main(args) == 0
    assert (out_dir / "manifest.json").read_text(encoding="utf-8") == first_manifest
    manifest = json.loads(first_manifest)
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["training_started"] is False
    assert manifest["models_changed"] is False
