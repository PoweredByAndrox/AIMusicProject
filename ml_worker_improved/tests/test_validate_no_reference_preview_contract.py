from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_audio_preview import render_audio_preview_contract
from tools.validate_no_reference_preview_contract import validate_preview_contract


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


def create_rendered_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    audio_dir = tmp_path / "source_audio"
    vocal_path = write_wav(audio_dir / "vocal.wav", sine(440, 2.0))
    minus_path = write_wav(audio_dir / "minus.wav", sine(120, 2.0))
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run-preview-validate",
            "track_id": "song-validate",
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
                "track_id": "song-validate",
                "selected_vocal_candidate_id": "candidate-v",
                "selected_minus_candidate_id": "candidate-m",
                "selected_vocal_path": vocal_path,
                "selected_minus_path": minus_path,
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
                "track_id": "song-validate",
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
    render_audio_preview_contract(run_dir=run_dir, preview_duration_sec=0.5, preview_render_mode="bounded")
    return run_dir


def test_validate_accepts_rendered_audio_preview_contract(tmp_path: Path) -> None:
    run_dir = create_rendered_run(tmp_path)

    result = validate_preview_contract(run_dir=run_dir)

    assert result["ok"] is True
    assert result["preview_status"] == "audio_preview_ready"
    assert result["rendered_preview_audio_count"] == 2
    assert result["release_ready"] is False
    assert result["can_publish_as_result"] is False


def test_validate_rejects_fake_audio_ready_without_rendered_rows(tmp_path: Path) -> None:
    run_dir = create_rendered_run(tmp_path)
    manifest_path = run_dir / "preview_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preview_status"] = "audio_preview_ready"
    manifest["audio_preview_ready"] = True
    manifest["rendered_preview_audio"] = []
    write_json(manifest_path, manifest)
    write_jsonl(run_dir / "rendered_preview_audio.jsonl", [])

    result = validate_preview_contract(run_dir=run_dir)

    assert result["ok"] is False
    assert "audio_ready_without_rendered_audio_rows" in result["blockers"]


def test_validate_rejects_release_ready_preview_contract(tmp_path: Path) -> None:
    run_dir = create_rendered_run(tmp_path)
    manifest_path = run_dir / "preview_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_ready"] = True
    write_json(manifest_path, manifest)

    result = validate_preview_contract(run_dir=run_dir)

    assert result["ok"] is False
    assert "release_ready_forbidden" in result["blockers"]
