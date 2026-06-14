from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.backfill_no_reference_preview_contract import backfill_preview_contract

SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_legacy_pair_run(tmp_path: Path, *, compatible: bool = True, missing_audio: bool = False) -> Path:
    run_dir = tmp_path / "legacy-run"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = audio_dir / "vocal.wav"
    minus_path = audio_dir / "minus.wav"
    if not missing_audio:
        sf.write(vocal_path, sine(440.0, 1.0), SAMPLE_RATE)
        sf.write(minus_path, sine(220.0, 1.0), SAMPLE_RATE)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "legacy-run",
            "track_id": "track-a",
            "output_status": "accepted_pair_ready",
            "release_ready": False,
        },
    )
    write_jsonl(
        run_dir / "selected_candidates.jsonl",
        [
            {
                "track_id": "track-a",
                "selected_vocal_candidate_id": "candidate-v",
                "selected_minus_candidate_id": "candidate-m",
                "selected_vocal_path": str(vocal_path),
                "selected_minus_path": str(minus_path),
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "pair_compatibility.jsonl",
        [
            {
                "track_id": "track-a",
                "selected_vocal_candidate_id": "candidate-v",
                "selected_minus_candidate_id": "candidate-m",
                "pair_decision": "pair_compatible" if compatible else "pair_incompatible",
                "pair_blockers": [] if compatible else ["test_blocker"],
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(run_dir / "candidates" / "candidate_wave_jobs.jsonl", [{"job_id": "existing-job", "generation_status": "generated"}])
    return run_dir


def test_backfill_renders_preview_only_from_existing_selected_stems(tmp_path: Path) -> None:
    run_dir = make_legacy_pair_run(tmp_path)
    selected_before = (run_dir / "selected_candidates.jsonl").read_text(encoding="utf-8")
    jobs_before = (run_dir / "candidates" / "candidate_wave_jobs.jsonl").read_text(encoding="utf-8")

    result = backfill_preview_contract(run_dir=run_dir, preview_duration_sec=0.25, preview_render_mode="bounded")

    rendered_rows = [json.loads(line) for line in (run_dir / "rendered_preview_audio.jsonl").read_text(encoding="utf-8").splitlines()]
    contract = json.loads((run_dir / "preview_render_contract.json").read_text(encoding="utf-8"))
    assert result["backfill_status"] == "backfill_succeeded"
    assert result["separator_jobs_rerun"] is False
    assert result["selected_candidates_modified"] is False
    assert result["release_ready"] is False
    assert len([row for row in rendered_rows if row["render_status"] == "rendered"]) == 2
    assert all(Path(row["rendered_path"]).is_relative_to(run_dir) for row in rendered_rows if row.get("rendered_path"))
    assert (run_dir / "selected_candidates.jsonl").read_text(encoding="utf-8") == selected_before
    assert (run_dir / "candidates" / "candidate_wave_jobs.jsonl").read_text(encoding="utf-8") == jobs_before
    assert contract["backfill_status"] == "backfill_succeeded"
    assert contract["release_ready"] is False


def test_backfill_missing_audio_marks_failed_without_release_ready(tmp_path: Path) -> None:
    run_dir = make_legacy_pair_run(tmp_path, missing_audio=True)

    result = backfill_preview_contract(run_dir=run_dir, preview_duration_sec=0.25, preview_render_mode="bounded")

    manifest = json.loads((run_dir / "preview_manifest.json").read_text(encoding="utf-8"))
    assert result["backfill_status"] == "backfill_failed"
    assert manifest["preview_status"] == "preview_render_failed"
    assert "selected_vocal_audio_missing" in manifest["failure_reason"]
    assert manifest["release_ready"] is False


def test_backfill_incompatible_pair_marks_safety_blocked(tmp_path: Path) -> None:
    run_dir = make_legacy_pair_run(tmp_path, compatible=False)

    result = backfill_preview_contract(run_dir=run_dir, preview_duration_sec=0.25, preview_render_mode="bounded")

    manifest = json.loads((run_dir / "preview_manifest.json").read_text(encoding="utf-8"))
    assert result["backfill_status"] == "backfill_blocked_safety_contract"
    assert manifest["preview_status"] == "preview_blocked_safety_contract"
    assert manifest["release_ready"] is False
