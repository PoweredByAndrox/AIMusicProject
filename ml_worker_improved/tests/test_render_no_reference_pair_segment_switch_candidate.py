from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_pair_segment_switch_candidate import (
    build_pair_segment_switch_candidate,
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


def tone(freq: float, seconds: float, sample_rate: int = 8000, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    mono = amp * np.sin(2.0 * np.pi * freq * t)
    return np.column_stack([mono, mono])


def candidate_row(tmp_path: Path, candidate_id: str, vocal: np.ndarray, minus: np.ndarray) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": str(tmp_path / "mix.wav"),
        "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", vocal),
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", minus),
        "minus_path": str(tmp_path / candidate_id / "minus.wav"),
        "is_complete_pair": True,
    }


def judge_row(candidate: dict, first_score: float, second_score: float, *, severe_second: bool = False) -> dict:
    second_reasons = ["vocal_leakage_in_minus"] if severe_second else ["no_reference_proxy_passed"]
    return {
        "track_id": candidate["track_id"],
        "candidate_id": candidate["candidate_id"],
        "mix_path": candidate["mix_path"],
        "vocal_path": candidate["vocal_path"],
        "instrumental_path": candidate["instrumental_path"],
        "window_metrics": [
            {
                "start_sec": 0.0,
                "duration_sec": 1.0,
                "minus_score": first_score,
                "window_fitness": first_score,
                "reasons": ["no_reference_proxy_passed"],
            },
            {
                "start_sec": 1.0,
                "duration_sec": 1.0,
                "minus_score": second_score,
                "window_fitness": second_score,
                "reasons": second_reasons,
            },
        ],
    }


def vocal_judge_row(candidate: dict, first_score: float, second_score: float) -> dict:
    row = judge_row(candidate, 0.1, 0.1)
    for score, window in zip((first_score, second_score), row["window_metrics"]):
        window["vocal_score"] = score
        window["vocal_decision"] = "accept_candidate" if score >= 0.72 else "blocked_hard_case"
        window["reasons"] = ["no_reference_proxy_passed"] if score >= 0.72 else ["vocal_missing_on_active_region"]
    return row


def test_pair_segment_switch_renders_vocal_and_minus_from_same_window_candidates(tmp_path: Path) -> None:
    write_wav(tmp_path / "mix.wav", tone(110, 2.0, amp=0.4))
    first = candidate_row(
        tmp_path,
        "first",
        np.concatenate([tone(330, 1.0, amp=0.4), tone(330, 1.0, amp=0.02)]),
        np.concatenate([tone(110, 1.0, amp=0.4), tone(110, 1.0, amp=0.02)]),
    )
    second = candidate_row(
        tmp_path,
        "second",
        np.concatenate([tone(440, 1.0, amp=0.02), tone(440, 1.0, amp=0.4)]),
        np.concatenate([tone(220, 1.0, amp=0.02), tone(220, 1.0, amp=0.4)]),
    )

    result = build_pair_segment_switch_candidate(
        [first, second],
        [judge_row(first, 0.95, 0.2), judge_row(second, 0.1, 0.96)],
        output_dir=tmp_path / "out",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "rendered_preview_candidate"
    assert result["delivery_allowed"] is False
    assert result["can_publish_as_result"] is False
    assert result["can_mark_release_ready"] is False
    assert [row["selected_candidate_id"] for row in result["selected_windows"]] == ["first", "second"]
    vocal, vocal_rate = sf.read(result["rendered_vocal_preview_path"], always_2d=True)
    minus, minus_rate = sf.read(result["rendered_minus_preview_path"], always_2d=True)
    assert vocal_rate == minus_rate == 8000
    assert vocal.shape == minus.shape == (16000, 2)
    assert result["candidate_row"]["vocal_path"] == result["rendered_vocal_preview_path"]
    assert result["candidate_row"]["minus_path"] == result["rendered_minus_preview_path"]


def test_pair_segment_switch_can_select_by_vocal_score_and_decision(tmp_path: Path) -> None:
    write_wav(tmp_path / "mix.wav", tone(110, 2.0, amp=0.4))
    first = candidate_row(tmp_path, "first", tone(330, 2.0), tone(110, 2.0))
    second = candidate_row(tmp_path, "second", tone(440, 2.0), tone(220, 2.0))

    result = build_pair_segment_switch_candidate(
        [first, second],
        [vocal_judge_row(first, 0.93, 0.1), vocal_judge_row(second, 0.2, 0.94)],
        output_dir=tmp_path / "out",
        min_window_score=0.72,
        crossfade_ms=0,
        selection_score_field="vocal_score",
        selection_decision_field="vocal_decision",
    )

    assert result["render_status"] == "rendered_preview_candidate"
    assert result["selection_score_field"] == "vocal_score"
    assert result["selection_decision_field"] == "vocal_decision"
    assert [row["selected_candidate_id"] for row in result["selected_windows"]] == ["first", "second"]


def test_pair_segment_switch_can_select_low_activity_proxy_gap_when_explicitly_allowed(tmp_path: Path) -> None:
    write_wav(tmp_path / "mix.wav", tone(110, 2.0, amp=0.4))
    first = candidate_row(tmp_path, "first", tone(330, 2.0), tone(110, 2.0))
    row = vocal_judge_row(first, 0.93, 0.75)
    row["window_metrics"][1]["vocal_decision"] = "blocked_hard_case"
    row["window_metrics"][1]["vocal_blockers"] = ["vocal_missing_on_active_region"]
    row["window_metrics"][1]["reasons"] = ["no_reference_proxy_passed"]

    blocked = build_pair_segment_switch_candidate(
        [first],
        [row],
        output_dir=tmp_path / "blocked",
        min_window_score=0.72,
        crossfade_ms=0,
        selection_score_field="vocal_score",
        selection_decision_field="vocal_decision",
    )
    result = build_pair_segment_switch_candidate(
        [first],
        [row],
        output_dir=tmp_path / "out",
        min_window_score=0.72,
        crossfade_ms=0,
        selection_score_field="vocal_score",
        selection_decision_field="vocal_decision",
        allow_low_activity_proxy_gap=True,
    )

    assert blocked["render_status"] == "blocked_incomplete_window_coverage"
    assert result["render_status"] == "rendered_preview_candidate"
    assert result["delivery_allowed"] is False
    assert result["can_publish_as_result"] is False
    assert result["selected_windows"][1]["selection_note"] == "low_activity_proxy_gap"


def test_pair_segment_switch_blocks_incomplete_safe_window_coverage(tmp_path: Path) -> None:
    write_wav(tmp_path / "mix.wav", tone(110, 2.0, amp=0.4))
    first = candidate_row(tmp_path, "first", tone(330, 2.0), tone(110, 2.0))

    result = build_pair_segment_switch_candidate(
        [first],
        [judge_row(first, 0.95, 0.95, severe_second=True)],
        output_dir=tmp_path / "out",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "blocked_incomplete_window_coverage"
    assert result["audio_exported"] is False
    assert result["rendered_vocal_preview_path"] == ""
    assert result["rendered_minus_preview_path"] == ""


def test_pair_segment_switch_cli_is_deterministic_and_preview_only(tmp_path: Path) -> None:
    write_wav(tmp_path / "mix.wav", tone(110, 2.0, amp=0.4))
    first = candidate_row(tmp_path, "first", tone(330, 2.0), tone(110, 2.0))
    candidate_jsonl = write_jsonl(tmp_path / "candidates.jsonl", [first])
    judge_jsonl = write_jsonl(tmp_path / "judge.jsonl", [judge_row(first, 0.95, 0.96)])
    out_dir = tmp_path / "out"
    args = [
        "--candidate-pool-jsonl",
        str(candidate_jsonl),
        "--windowed-judge-jsonl",
        str(judge_jsonl),
        "--output-dir",
        str(out_dir),
        "--manifest-json",
        str(out_dir / "manifest.json"),
        "--summary-md",
        str(out_dir / "summary.md"),
        "--crossfade-ms",
        "0",
        "--selection-score-field",
        "minus_score",
    ]

    assert main(args) == 0
    first_manifest = (out_dir / "manifest.json").read_text(encoding="utf-8")
    assert main(args) == 0
    assert (out_dir / "manifest.json").read_text(encoding="utf-8") == first_manifest
    manifest = json.loads(first_manifest)
    assert manifest["delivery_allowed"] is False
    assert manifest["requires_post_render_judge"] is True
    assert manifest["safety_flags"]["training_started"] is False
