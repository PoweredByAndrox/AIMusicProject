import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_wave(path: Path, *, frequency: float, seconds: float = 0.35, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for index in range(int(sample_rate * seconds)):
        value = 0.25 * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        values.append(int(value * 32767.0))
    data = np.asarray(values, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def split_row(tmp_path: Path, *, stem: str, label: int, split: str, candidate_id: str = "cand-a") -> dict:
    mix = tmp_path / "audio" / "mix.wav"
    vocal = tmp_path / "audio" / "vocal.wav"
    minus = tmp_path / "audio" / "minus.wav"
    if not mix.exists():
        write_wave(mix, frequency=220.0)
        write_wave(vocal, frequency=440.0)
        write_wave(minus, frequency=330.0)
    return {
        "row_uid": f"{candidate_id}-{stem}",
        "canonical_row_key": f"{candidate_id}-{stem}-key",
        "track_id": "track-a",
        "run_id": "run-a",
        "candidate_id": candidate_id,
        "stem_type": stem,
        "split": split,
        "label": label,
        "train_allowed": split == "train",
        "hardcase_holdout": False,
        "window_start_sec": 0.0,
        "window_end_sec": 0.25,
        "mix_path": str(mix),
        "candidate_audio_path": str(vocal if stem == "vocal" else minus),
        "route_id": "route-a",
        "candidate_family": "family-a",
        "source_model_family": "model-a",
        "blockers": ["severe_phase_collapse"] if label == 0 else [],
        "release_ready": False,
    }


def write_split_dir(tmp_path: Path, rows: list[dict], *, critical_leakage_count: int = 0) -> Path:
    split_dir = tmp_path / "split"
    write_jsonl(split_dir / "leak_safe_split_rows.jsonl", rows)
    write_json(
        split_dir / "leak_safe_split_manifest.json",
        {"schema": "leak_safe_audio_eval_harness.v1", "critical_leakage_count": critical_leakage_count, "release_ready": False},
    )
    write_json(split_dir / "leakage_report.json", {"critical_leakage_count": critical_leakage_count, "release_ready": False})
    return split_dir


def test_build_tensor_dataset_extracts_npz_maps_from_locked_pairs(tmp_path: Path) -> None:
    from tools.build_locked_tensor_audio_error_dataset import build_locked_tensor_audio_error_dataset

    rows = [
        split_row(tmp_path, stem="vocal", label=1, split="train"),
        split_row(tmp_path, stem="minus", label=0, split="train"),
        split_row(tmp_path, stem="vocal", label=1, split="test", candidate_id="cand-b"),
        split_row(tmp_path, stem="minus", label=1, split="test", candidate_id="cand-b"),
    ]
    split_dir = write_split_dir(tmp_path, rows)

    manifest = build_locked_tensor_audio_error_dataset(split_dir=split_dir, output_dir=tmp_path / "tensor", max_pairs=0)

    assert manifest["schema"] == "locked_tensor_audio_error_dataset.v1"
    assert manifest["tensor_maps_required"] is True
    assert manifest["scalar_fallback_used"] is False
    assert manifest["release_ready"] is False
    assert manifest["rows_total"] == 4
    assert manifest["pairs_extracted"] == 2
    tensor_rows = read_jsonl(tmp_path / "tensor" / "tensor_audio_error_dataset.jsonl")
    assert {row["split"] for row in tensor_rows} == {"train", "test"}
    assert all(Path(row["tensor_map_path"]).exists() for row in tensor_rows)
    with np.load(tensor_rows[0]["tensor_map_path"]) as payload:
        assert "stft_residual_magnitude" in payload.files
        assert "phase_difference_map" in payload.files


def test_build_tensor_dataset_refuses_critical_leakage(tmp_path: Path) -> None:
    from tools.build_locked_tensor_audio_error_dataset import build_locked_tensor_audio_error_dataset

    split_dir = write_split_dir(
        tmp_path,
        [split_row(tmp_path, stem="vocal", label=1, split="train"), split_row(tmp_path, stem="minus", label=0, split="train")],
        critical_leakage_count=1,
    )

    with pytest.raises(ValueError, match="critical_leakage_present"):
        build_locked_tensor_audio_error_dataset(split_dir=split_dir, output_dir=tmp_path / "tensor")
