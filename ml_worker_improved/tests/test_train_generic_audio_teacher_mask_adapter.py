from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.train_generic_audio_teacher_mask_adapter import fingerprint_rows, train_generic_audio_teacher_mask_adapter
from tools.train_teacher_mask_separator import TeacherMaskTrainConfig


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int = 8_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")
    return path


def test_adapter_fingerprint_changes_when_training_config_changes() -> None:
    rows = [
        {
            "id": "a",
            "track_id": "track-a",
            "source_audio_path": "/tmp/source.flac",
            "target_audio_path": "/tmp/target.flac",
        }
    ]
    base = TeacherMaskTrainConfig(n_fft=256, hop_length=64, hidden_size=64, device="cpu")
    memory = TeacherMaskTrainConfig(
        n_fft=256,
        hop_length=64,
        hidden_size=64,
        source_feature_memory_neighbors=3,
        device="cpu",
    )

    assert fingerprint_rows(rows, stem="vocal", config=base, max_frames_per_pair=64, max_examples_per_track=1) != fingerprint_rows(
        rows,
        stem="vocal",
        config=memory,
        max_frames_per_pair=64,
        max_examples_per_track=1,
    )
    assert fingerprint_rows(rows, stem="vocal", config=base, max_frames_per_pair=64, max_examples_per_track=1) != fingerprint_rows(
        rows,
        stem="vocal",
        config=base,
        max_frames_per_pair=128,
        max_examples_per_track=1,
    )


def test_trains_shared_adapter_and_writes_holdout_safe_run_rows(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rows = []
    for track_id, interference_hz in (("track-a", 120.0), ("track-b", 180.0), ("track-c", 150.0)):
        target = _stereo(0.26 * np.sin(2.0 * np.pi * 720.0 * time))
        source = target + _stereo(0.34 * np.sin(2.0 * np.pi * interference_hz * time))
        source_path = _write_audio(tmp_path / track_id / "source.flac", source, sample_rate)
        target_path = _write_audio(tmp_path / track_id / "target.flac", target, sample_rate)
        rows.append(
            {
                "id": f"{track_id}-vocal",
                "track_id": track_id,
                "stem_type": "vocal",
                "source_audio_path": str(source_path),
                "target_audio_path": str(target_path),
                "trainable": True,
                "split": "train",
            }
        )
    duplicate_target = _stereo(0.26 * np.sin(2.0 * np.pi * 720.0 * time))
    duplicate_source = duplicate_target + _stereo(0.34 * np.sin(2.0 * np.pi * 130.0 * time))
    duplicate_source_path = _write_audio(tmp_path / "track-a-duplicate" / "source.flac", duplicate_source, sample_rate)
    duplicate_target_path = _write_audio(tmp_path / "track-a-duplicate" / "target.flac", duplicate_target, sample_rate)
    rows.append(
        {
            "id": "track-a-duplicate-vocal",
            "track_id": "track-a",
            "stem_type": "vocal",
            "source_audio_path": str(duplicate_source_path),
            "target_audio_path": str(duplicate_target_path),
            "trainable": True,
            "split": "train",
        }
    )
    examples = _write_jsonl(tmp_path / "examples.jsonl", rows)

    manifest = train_generic_audio_teacher_mask_adapter(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "shared_adapter",
        stem="vocal",
        scope="vocal_shared_teacher_mask_adapter",
        exclude_holdout_track_ids={"track-c"},
        min_unique_tracks=2,
        max_examples_per_track=1,
        n_fft=256,
        hop_length=64,
        hidden_size=64,
        epochs=12,
        batch_size=64,
        learning_rate=0.01,
        activity_weight=10.0,
        source_feature_memory_neighbors=1,
        source_feature_memory_chunk_size=32,
        seed=2026,
        device="cpu",
    )

    assert manifest["selected_training_example_count"] == 2
    assert manifest["max_examples_per_track"] == 1
    assert manifest["unique_training_track_count"] == 2
    assert manifest["training_track_ids"] == ["track-a", "track-b"]
    assert manifest["excluded_holdout_track_ids"] == ["track-c"]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["contract"]["stores_learned_source_feature_memory"] is True
    assert manifest["training"]["diagnostics"]["source_feature_memory_frame_count"] > 0
    assert Path(manifest["checkpoint_path"]).exists()
    assert Path(manifest["training_preview_path"]).exists()

    run_rows = _read_jsonl(Path(manifest["run_rows_jsonl"]))
    assert len(run_rows) == 1
    assert run_rows[0]["scope"] == "vocal_shared_teacher_mask_adapter"
    assert run_rows[0]["status"] == "executed"
    assert run_rows[0]["training_track_ids"] == ["track-a", "track-b"]
    assert run_rows[0]["device_used"] == 0.0
    assert Path(run_rows[0]["checkpoint_path"]).exists()

    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert progress_rows[0]["event"] == "start"
    assert progress_rows[-1]["event"] == "finish"
    disk_manifest = _read_json(tmp_path / "shared_adapter" / "generic_audio_teacher_mask_adapter_manifest.json")
    assert disk_manifest["adapter_id"] == manifest["adapter_id"]
