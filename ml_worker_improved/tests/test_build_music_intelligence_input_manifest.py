from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_music_intelligence_input_manifest import (
    build_music_intelligence_input_manifest,
    music_intelligence_run_section,
    write_music_intelligence_input_manifest,
)


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_accepted_vocal_stem_becomes_analysis_allowed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vocal_path = write_wav(run_dir / "accepted" / "vocal.wav", sine(440, 1.0))
    minus_path = write_wav(run_dir / "accepted" / "minus.wav", sine(120, 1.0))
    runtime_manifest = {
        "input_audio": str(run_dir / "mix.wav"),
        "output_dir": str(run_dir),
        "output_status": "preview_ready",
        "selected_vocal_candidate": "candidate-a",
        "selected_minus_candidate": "candidate-a",
        "safety_blockers": [],
        "release_ready": False,
    }
    write_jsonl(
        run_dir / "selected_candidates.jsonl",
        [
            {
                "track_id": "track-a",
                "selected_vocal_candidate_id": "candidate-a",
                "selected_minus_candidate_id": "candidate-a",
                "selected_vocal_path": vocal_path,
                "selected_minus_path": minus_path,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "pair_compatibility.jsonl",
        [
            {
                "track_id": "track-a",
                "selected_vocal_candidate_id": "candidate-a",
                "selected_minus_candidate_id": "candidate-a",
                "pair_decision": "pair_compatible",
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "candidate-a",
                "vocal_path": vocal_path,
                "minus_path": minus_path,
                "source_separator_config": "htdemucs_ft",
                "candidate_family": "local_registered",
                "release_ready": False,
            }
        ],
    )

    manifest = build_music_intelligence_input_manifest(run_dir=run_dir, runtime_manifest=runtime_manifest)
    vocal_rows = [row for row in manifest["stems"] if row["stem_type"] == "vocal"]

    assert len(vocal_rows) == 1
    assert vocal_rows[0]["analysis_allowed"] is True
    assert vocal_rows[0]["analysis_block_reason"] == ""
    assert vocal_rows[0]["quality_status"] == "accepted"
    assert vocal_rows[0]["sample_rate"] == SAMPLE_RATE
    assert vocal_rows[0]["channels"] == 1
    assert vocal_rows[0]["release_ready"] is False
    assert manifest["melody_transcription_ran"] is False
    assert manifest["recommendation_index_updated"] is False
    assert manifest["future_analysis_supported"] is False
    manifest_path = write_music_intelligence_input_manifest(
        run_dir=run_dir,
        output_path=run_dir / "music_intelligence_input_manifest.json",
        runtime_manifest=runtime_manifest,
    )
    assert music_intelligence_run_section(manifest, manifest_path) == {
        "manifest_written": True,
        "manifest_path": str(manifest_path),
        "analysis_ready_stems_count": 2,
        "blocked_stems_count": 0,
        "future_analysis_supported": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "release_ready": False,
    }


def test_blocked_minus_stem_still_appears_with_block_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vocal_path = write_wav(run_dir / "blocked" / "vocal.wav", sine(440, 1.0))
    minus_path = write_wav(run_dir / "blocked" / "minus.wav", sine(120, 1.0))
    runtime_manifest = {
        "input_audio": str(run_dir / "mix.wav"),
        "output_dir": str(run_dir),
        "output_status": "needs_more_candidates",
        "failure_reason": "minus_hard_case_route_exhausted",
        "safety_blockers": ["severe_phase_collapse", "severe_vocal_bleed_in_minus"],
        "release_ready": False,
    }
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-b",
                "candidate_id": "candidate-blocked",
                "vocal_path": vocal_path,
                "minus_path": minus_path,
                "source_separator_config": "htdemucs_ft",
                "source_candidate_family": "local_registered",
                "release_ready": False,
            }
        ],
    )
    write_jsonl(run_dir / "selected_candidates.jsonl", [])
    write_jsonl(run_dir / "pair_compatibility.jsonl", [])

    manifest = build_music_intelligence_input_manifest(run_dir=run_dir, runtime_manifest=runtime_manifest)
    minus_rows = [row for row in manifest["stems"] if row["stem_type"] == "minus"]

    assert len(minus_rows) == 1
    assert minus_rows[0]["analysis_allowed"] is False
    assert minus_rows[0]["quality_status"] == "blocked"
    assert "minus_hard_case_route_exhausted" in minus_rows[0]["analysis_block_reason"]
    assert "severe_phase_collapse" in minus_rows[0]["blockers"]
    assert minus_rows[0]["pair_compatibility_status"] == "not_evaluated"
    assert minus_rows[0]["release_ready"] is False
    assert manifest["analysis_ready_stems_count"] == 0
    assert manifest["blocked_stems_count"] == 2


def test_manifest_writer_forces_release_ready_false(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    audio_path = write_wav(run_dir / "candidate" / "vocal.wav", sine(440, 1.0))
    runtime_manifest = {
        "input_audio": str(run_dir / "mix.wav"),
        "output_dir": str(run_dir),
        "output_status": "preview_ready",
        "selected_vocal_candidate": "unsafe",
        "selected_minus_candidate": "",
        "release_ready": True,
    }
    write_jsonl(
        run_dir / "selected_candidates.jsonl",
        [
            {
                "track_id": "track-c",
                "selected_vocal_candidate_id": "unsafe",
                "selected_vocal_path": audio_path,
                "release_ready": True,
            }
        ],
    )
    write_jsonl(run_dir / "pair_compatibility.jsonl", [])

    output_path = write_music_intelligence_input_manifest(
        run_dir=run_dir,
        output_path=run_dir / "music_intelligence_input_manifest.json",
        runtime_manifest=runtime_manifest,
    )
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["release_ready"] is False
    assert all(row["release_ready"] is False for row in manifest["stems"])
    assert all(row["analysis_allowed"] is False for row in manifest["stems"])
