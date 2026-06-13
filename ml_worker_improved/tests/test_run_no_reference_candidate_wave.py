from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_no_reference_candidate_wave import main, run_candidate_wave


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_candidate_wave_creates_pool_from_mocked_separator_outputs(tmp_path: Path) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.5))
    source_vocal = write_wav(tmp_path / "mocked_separator" / "vocals.wav", sine(440, 1.5))
    source_minus = write_wav(tmp_path / "mocked_separator" / "minus.wav", sine(120, 1.5))
    source_jsonl = write_jsonl(
        tmp_path / "sources.jsonl",
        [
            {
                "candidate_config_id": "safe-vocal",
                "candidate_family": "vocal_focused",
                "source_model_id": "local-existing-vocal",
                "source_tool": "mocked_existing_separator",
                "target_mode": "vocal",
                "vocal_path": source_vocal,
            },
            {
                "candidate_config_id": "safe-minus",
                "candidate_family": "minus_focused",
                "source_model_id": "local-existing-minus",
                "source_tool": "mocked_existing_separator",
                "target_mode": "minus",
                "minus_path": source_minus,
            },
            {
                "candidate_config_id": "missing-minus",
                "candidate_family": "minus_focused",
                "source_tool": "mocked_existing_separator",
                "target_mode": "minus",
                "minus_path": str(tmp_path / "missing.wav"),
            },
        ],
    )
    output_dir = tmp_path / "candidate_wave"
    candidate_pool = tmp_path / "candidate_pool.jsonl"

    exit_code = main(
        [
            "--input-audio",
            input_audio,
            "--output-dir",
            str(output_dir),
            "--candidate-pool-jsonl",
            str(candidate_pool),
            "--profile",
            "safe",
            "--source-jsonl",
            str(source_jsonl),
            "--max-candidates",
            "6",
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in candidate_pool.read_text(encoding="utf-8").splitlines()]
    failures = [json.loads(line) for line in (output_dir / "candidate_wave_failures.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((output_dir / "candidate_wave_summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    row = rows[0]
    assert row["target_mode"] == "both"
    assert row["candidate_family"] == "vocal_focused+minus_focused"
    assert row["candidate_config_id"] == "safe-vocal+safe-minus"
    assert row["generation_profile"] == "safe"
    assert row["generation_status"] == "generated"
    assert row["can_use_for_vocal"] is True
    assert row["can_use_for_minus"] is True
    assert row["can_use_for_pair"] is True
    assert row["audio_rendered"] is False
    assert row["release_ready"] is False
    assert output_dir in Path(row["vocal_path"]).parents
    assert output_dir in Path(row["minus_path"]).parents
    assert Path(row["vocal_path"]).exists()
    assert Path(row["minus_path"]).exists()
    assert failures and failures[0]["skip_reason"] == "missing_minus_path"
    assert summary["candidates_generated"] == 1
    assert summary["complete_pairs"] == 1
    assert summary["failed_candidates"] == 1


def test_candidate_wave_is_deterministic_for_same_input_config_profile(tmp_path: Path) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    source_vocal = write_wav(tmp_path / "mocked_separator" / "vocals.wav", sine(440, 1.0))
    source_minus = write_wav(tmp_path / "mocked_separator" / "minus.wav", sine(120, 1.0))
    source_jsonl = write_jsonl(
        tmp_path / "sources.jsonl",
        [
            {"candidate_config_id": "v1", "candidate_family": "vocal_focused", "target_mode": "vocal", "vocal_path": source_vocal},
            {"candidate_config_id": "m1", "candidate_family": "minus_focused", "target_mode": "minus", "minus_path": source_minus},
        ],
    )

    first = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "run-a",
        candidate_pool_jsonl=tmp_path / "run-a" / "candidate_pool.jsonl",
        profile="safe",
        source_jsonl=source_jsonl,
        max_candidates=6,
    )
    second = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "run-b",
        candidate_pool_jsonl=tmp_path / "run-b" / "candidate_pool.jsonl",
        profile="safe",
        source_jsonl=source_jsonl,
        max_candidates=6,
    )

    assert [row["candidate_id"] for row in first["rows"]] == [row["candidate_id"] for row in second["rows"]]
    assert first["summary"]["release_ready_count"] == 0
    assert second["summary"]["release_ready_count"] == 0
