import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_audio_error_grounding_dataset import build_audio_error_grounding_dataset
from tools.validate_audio_error_grounding_dataset import validate_audio_error_grounding_dataset


SAMPLE_RATE = 8_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def stereo(audio: np.ndarray) -> np.ndarray:
    return np.column_stack([audio, audio * 0.8]).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_run(root: Path, *, missing_minus: bool = False) -> Path:
    run_dir = root / "run-a"
    vocal = sine(440, 1.5, amp=0.18)
    minus = sine(120, 1.5, amp=0.24)
    mix = vocal + minus
    mix_path = write_wav(run_dir / "audio" / "mix.wav", stereo(mix))
    vocal_path = write_wav(run_dir / "audio" / "vocal.wav", stereo(vocal))
    minus_path = str(run_dir / "audio" / "missing-minus.wav") if missing_minus else write_wav(
        run_dir / "audio" / "minus.wav",
        stereo(minus),
    )
    candidate = {
        "track_id": "track-a",
        "run_id": "run-a",
        "candidate_id": "cand-a",
        "route_id": "route-safe",
        "mix_path": mix_path,
        "input_audio_path": mix_path,
        "vocal_path": vocal_path,
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "is_complete_pair": True,
        "release_ready": False,
    }
    window = {
        "kind": "grid_0000",
        "start_sec": 0.0,
        "duration_sec": 0.5,
        "decision": "accept_candidate",
        "reasons": [],
        "mixture_consistency_score": 0.92,
        "phase_consistency_score": 0.83,
    }
    write_jsonl(run_dir / "candidate_pool.jsonl", [candidate])
    write_jsonl(run_dir / "vocal_windowed_judge.jsonl", [{**candidate, "score_mode": "vocal", "decision": "accept_candidate", "window_metrics": [window]}])
    write_jsonl(run_dir / "minus_windowed_judge.jsonl", [{**candidate, "score_mode": "minus", "decision": "accept_candidate", "window_metrics": [window]}])
    write_jsonl(run_dir / "pair_compatibility.jsonl", [{"track_id": "track-a", "candidate_id": "cand-a", "pair_decision": "pair_compatible", "release_ready": False}])
    write_json(run_dir / "preview_render_contract.json", {"release_ready": False, "audio_preview_ready": True})
    return run_dir


def test_validate_audio_error_grounding_dataset_accepts_builder_output(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path)
    dataset_dir = tmp_path / "dataset"
    build_audio_error_grounding_dataset(
        run_roots=[run_dir],
        output_dir=dataset_dir,
        window_sec=0.5,
        hop_sec=0.5,
        map_max_bins=16,
        map_max_frames=8,
        max_waveform_samples=64,
    )

    result = validate_audio_error_grounding_dataset(dataset_dir)

    assert result["ok"] is True
    assert result["rows_checked"] == 2
    assert result["safe_for_training_count"] == 2
    assert result["release_ready"] is False
    assert result["melody_transcription_ran"] is False
    assert result["recommendation_index_updated"] is False


def test_validate_rejects_safe_row_without_computable_maps(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path)
    dataset_dir = tmp_path / "dataset"
    build_audio_error_grounding_dataset(run_roots=[run_dir], output_dir=dataset_dir, window_sec=0.5, hop_sec=0.5)
    rows = read_jsonl(dataset_dir / "audio_error_window_dataset.jsonl")
    rows[0]["error_map_paths"] = {}
    (dataset_dir / "audio_error_window_dataset.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = validate_audio_error_grounding_dataset(dataset_dir)

    assert result["ok"] is False
    assert any("safe_row_missing_error_map" in error for error in result["errors"])


def test_validate_allows_missing_audio_only_when_row_is_unsafe(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path, missing_minus=True)
    dataset_dir = tmp_path / "dataset"
    build_audio_error_grounding_dataset(run_roots=[run_dir], output_dir=dataset_dir, window_sec=0.5, hop_sec=0.5)

    result = validate_audio_error_grounding_dataset(dataset_dir)

    assert result["ok"] is True
    assert result["safe_for_training_count"] == 0
    assert result["unsafe_count"] == 2


def test_validate_rejects_any_release_ready_flag(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path)
    dataset_dir = tmp_path / "dataset"
    build_audio_error_grounding_dataset(run_roots=[run_dir], output_dir=dataset_dir, window_sec=0.5, hop_sec=0.5)
    manifest_path = dataset_dir / "audio_error_grounding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_ready"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_audio_error_grounding_dataset(dataset_dir)

    assert result["ok"] is False
    assert any("release_ready_must_be_false" in error for error in result["errors"])
