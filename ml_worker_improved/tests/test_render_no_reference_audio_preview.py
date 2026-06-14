from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_audio_preview import render_audio_preview_contract


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_compatible_run(tmp_path: Path, *, missing_minus: bool = False) -> tuple[Path, str, str]:
    run_dir = tmp_path / "run"
    audio_dir = tmp_path / "source_audio"
    vocal_path = write_wav(audio_dir / "vocal.wav", sine(440, 2.0))
    minus_path = write_wav(audio_dir / "minus.wav", sine(120, 2.0))
    selected_minus_path = str(audio_dir / "missing_minus.wav") if missing_minus else minus_path
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run-preview-001",
            "track_id": "song-001",
            "input_audio": str(audio_dir / "mix.wav"),
            "output_dir": str(run_dir),
            "output_status": "accepted_pair_ready",
            "release_ready": False,
        },
    )
    write_jsonl(
        run_dir / "selected_candidates.jsonl",
        [
            {
                "track_id": "song-001",
                "selected_vocal_candidate_id": "candidate-v",
                "selected_minus_candidate_id": "candidate-m",
                "selected_vocal_path": vocal_path,
                "selected_minus_path": selected_minus_path,
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
                "track_id": "song-001",
                "selected_vocal_candidate_id": "candidate-v",
                "selected_minus_candidate_id": "candidate-m",
                "pair_decision": "pair_compatible",
                "pair_blockers": [],
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "song-001",
                "candidate_id": "candidate-v",
                "vocal_path": vocal_path,
                "minus_path": selected_minus_path,
                "is_complete_pair": True,
                "release_ready": False,
            }
        ],
    )
    return run_dir, vocal_path, minus_path


def test_compatible_selected_pair_renders_vocal_and_minus_preview_audio(tmp_path: Path) -> None:
    run_dir, vocal_path, minus_path = create_compatible_run(tmp_path)

    manifest = render_audio_preview_contract(
        run_dir=run_dir,
        preview_duration_sec=0.75,
        preview_start_sec=0.25,
        preview_render_mode="bounded",
        enable_audio_preview_render=True,
        require_preview_audio_for_preview_ready=True,
    )

    rows = read_jsonl(run_dir / "rendered_preview_audio.jsonl")
    assert manifest["preview_status"] == "audio_preview_ready"
    assert manifest["accepted_pair_ready"] is True
    assert manifest["manifest_preview_ready"] is True
    assert manifest["audio_preview_ready"] is True
    assert manifest["preview_render_attempted"] is True
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert {row["stem_type"] for row in rows} == {"vocal_preview", "minus_preview"}
    assert all(row["render_status"] == "rendered" for row in rows)
    assert all(row["release_ready"] is False for row in rows)
    assert all(Path(row["rendered_path"]).exists() for row in rows)
    assert all(Path(row["rendered_path"]).resolve().is_relative_to(run_dir.resolve()) for row in rows)
    assert all(Path(row["rendered_path"]).resolve() not in {Path(vocal_path).resolve(), Path(minus_path).resolve()} for row in rows)
    assert all(0.70 <= sf.info(row["rendered_path"]).duration <= 0.80 for row in rows)
    assert (run_dir / "preview_render_contract.json").exists()
    assert (run_dir / "preview_render_summary.md").exists()


def test_render_disabled_writes_skipped_contract(tmp_path: Path) -> None:
    run_dir, _, _ = create_compatible_run(tmp_path)

    manifest = render_audio_preview_contract(
        run_dir=run_dir,
        preview_duration_sec=0.5,
        preview_render_mode="bounded",
        enable_audio_preview_render=False,
        require_preview_audio_for_preview_ready=True,
    )

    rows = read_jsonl(run_dir / "rendered_preview_audio.jsonl")
    assert manifest["preview_status"] == "preview_render_skipped"
    assert manifest["preview_render_skipped"] is True
    assert manifest["preview_render_attempted"] is False
    assert manifest["skip_reason"] == "audio_preview_render_disabled"
    assert manifest["audio_preview_ready"] is False
    assert rows
    assert all(row["render_status"] == "skipped" for row in rows)
    assert all(not row["rendered_path"] for row in rows)
    assert manifest["release_ready"] is False


def test_missing_selected_audio_fails_contract_without_publish(tmp_path: Path) -> None:
    run_dir, _, _ = create_compatible_run(tmp_path, missing_minus=True)

    manifest = render_audio_preview_contract(
        run_dir=run_dir,
        preview_duration_sec=0.5,
        preview_render_mode="bounded",
        enable_audio_preview_render=True,
        require_preview_audio_for_preview_ready=True,
    )

    rows = read_jsonl(run_dir / "rendered_preview_audio.jsonl")
    assert manifest["preview_status"] == "preview_render_failed"
    assert manifest["preview_render_failed"] is True
    assert manifest["audio_preview_ready"] is False
    assert "selected_minus_audio_missing" in manifest["failure_reason"]
    assert any(row["render_status"] == "failed" for row in rows)
    assert manifest["can_publish_as_result"] is False
    assert manifest["release_ready"] is False
