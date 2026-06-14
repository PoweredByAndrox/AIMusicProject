from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_no_reference_delivery_batch import main


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
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def optimizer_row(tmp_path: Path, *, track_id: str, accepted: bool) -> dict:
    vocal = sine(440.0, 6.0)
    minus = sine(120.0, 6.0)
    mix = vocal + minus
    return {
        "track_id": track_id,
        "candidate_id": f"mode_specific_pair:{track_id}:candidate",
        "final_decision": "mode_specific_pair_selected" if accepted else "blocked_hard_case",
        "pair_compatibility_decision": "pair_compatible",
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "mix_path": write_wav(tmp_path / track_id / "mix.wav", mix),
        "selected_vocal_path": write_wav(tmp_path / track_id / "vocal.wav", vocal) if accepted else "",
        "selected_minus_path": write_wav(tmp_path / track_id / "minus.wav", minus) if accepted else "",
        "selected_vocal_candidate_id": "vocal-a",
        "selected_minus_candidate_id": "minus-b",
        "selected_vocal_score": 0.83,
        "selected_minus_score": 0.84,
        "pair_compatibility_score": 0.76,
    }


def test_batch_runner_reports_accepted_and_blocked_without_audio_export(tmp_path: Path) -> None:
    accepted = write_jsonl(tmp_path / "accepted_optimizer.jsonl", [optimizer_row(tmp_path, track_id="track-a", accepted=True)])
    blocked = write_jsonl(tmp_path / "blocked_optimizer.jsonl", [optimizer_row(tmp_path, track_id="track-b", accepted=False)])
    output_root = tmp_path / "batch"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(accepted),
            "--optimizer-result-jsonl",
            str(blocked),
            "--output-root",
            str(output_root),
            "--window-seconds",
            "1",
            "--window-step-seconds",
            "1",
            "--max-windows-per-candidate",
            "0",
            "--min-window-count",
            "5",
        ]
    ) == 0

    manifest_path = output_root / "no_reference_delivery_batch_manifest.json"
    summary_path = output_root / "no_reference_delivery_batch_summary.md"
    rows_path = output_root / "no_reference_delivery_batch_rows.jsonl"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert rows_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_inputs"] == 2
    assert manifest["accepted_for_listen_pack"] == 1
    assert manifest["blocked_from_listen_pack"] == 1
    assert manifest["audio_copied"] is False
    assert manifest["audio_exported"] is False
    assert manifest["safety"]["training_started"] is False
    assert manifest["safety"]["models_changed"] is False

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["final_status"] for row in rows] == ["accepted_for_listen_pack", "blocked_from_listen_pack"]
    assert not list(output_root.rglob("*.flac"))
    assert not list(output_root.rglob("*.wav"))


def test_batch_runner_splits_multi_track_optimizer_jsonl(tmp_path: Path) -> None:
    optimizer = write_jsonl(
        tmp_path / "multi_optimizer.jsonl",
        [
            optimizer_row(tmp_path, track_id="track-a", accepted=True),
            optimizer_row(tmp_path, track_id="track-b", accepted=False),
        ],
    )
    output_root = tmp_path / "split_batch"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(optimizer),
            "--output-root",
            str(output_root),
            "--window-seconds",
            "1",
            "--window-step-seconds",
            "1",
            "--max-windows-per-candidate",
            "0",
            "--min-window-count",
            "5",
        ]
    ) == 0

    manifest = json.loads((output_root / "no_reference_delivery_batch_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output_root / "no_reference_delivery_batch_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert manifest["total_inputs"] == 2
    assert manifest["accepted_for_listen_pack"] == 1
    assert manifest["blocked_from_listen_pack"] == 1
    assert [row["track_id"] for row in rows] == ["track-a", "track-b"]
    assert not list(output_root.rglob("*.flac"))
    assert not list(output_root.rglob("*.wav"))
