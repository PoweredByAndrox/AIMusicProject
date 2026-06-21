from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.train_generic_audio_teacher_mask_adapter import (
    fingerprint_rows,
    select_training_rows,
    train_generic_audio_teacher_mask_adapter,
)
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
    bounded_memory = TeacherMaskTrainConfig(
        n_fft=256,
        hop_length=64,
        hidden_size=64,
        source_feature_memory_neighbors=3,
        source_feature_memory_projection_dim=8,
        source_feature_memory_max_frames=16,
        device="cpu",
    )
    complex_mask = TeacherMaskTrainConfig(
        n_fft=256,
        hop_length=64,
        hidden_size=64,
        mask_output_mode="complex",
        complex_mask_ceiling=2.5,
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
    assert fingerprint_rows(rows, stem="vocal", config=memory, max_frames_per_pair=64, max_examples_per_track=1) != fingerprint_rows(
        rows,
        stem="vocal",
        config=bounded_memory,
        max_frames_per_pair=64,
        max_examples_per_track=1,
    )
    assert fingerprint_rows(rows, stem="vocal", config=base, max_frames_per_pair=64, max_examples_per_track=1) != fingerprint_rows(
        rows,
        stem="vocal",
        config=complex_mask,
        max_frames_per_pair=64,
        max_examples_per_track=1,
    )


def test_select_training_rows_uses_canonical_group_and_family_filter(tmp_path: Path) -> None:
    source_a = _write_audio(tmp_path / "a" / "source.flac", _stereo(np.zeros(128, dtype=np.float32)))
    target_a = _write_audio(tmp_path / "a" / "target.flac", _stereo(np.zeros(128, dtype=np.float32)))
    source_a_alt = _write_audio(tmp_path / "a_alt" / "source.flac", _stereo(np.zeros(128, dtype=np.float32)))
    target_a_alt = _write_audio(tmp_path / "a_alt" / "target.flac", _stereo(np.zeros(128, dtype=np.float32)))
    source_b = _write_audio(tmp_path / "b" / "source.flac", _stereo(np.zeros(128, dtype=np.float32)))
    target_b = _write_audio(tmp_path / "b" / "target.flac", _stereo(np.zeros(128, dtype=np.float32)))
    rows = [
        {
            "id": "alias-a",
            "track_id": "alias-a",
            "source_plan_track_id": "canonical-a",
            "stem_type": "vocal",
            "source_candidate_family": "scnet",
            "source_audio_path": str(source_a),
            "target_audio_path": str(target_a),
            "trainable": True,
        },
        {
            "id": "alias-a-alt",
            "track_id": "alias-a-alt",
            "source_plan_track_id": "canonical-a",
            "stem_type": "vocal",
            "source_candidate_family": "scnet",
            "source_audio_path": str(source_a_alt),
            "target_audio_path": str(target_a_alt),
            "trainable": True,
        },
        {
            "id": "track-b",
            "track_id": "track-b",
            "stem_type": "vocal",
            "source_candidate_family": "melband",
            "source_audio_path": str(source_b),
            "target_audio_path": str(target_b),
            "trainable": True,
        },
    ]

    selected = select_training_rows(
        rows,
        stem="vocal",
        exclude_holdout_track_ids=set(),
        max_examples_per_track=1,
        source_candidate_family_filter="scnet",
    )
    excluded = select_training_rows(
        rows,
        stem="vocal",
        exclude_holdout_track_ids={"canonical-a"},
        max_examples_per_track=1,
        source_candidate_family_filter="scnet",
    )

    assert [row["id"] for row in selected] == ["alias-a"]
    assert excluded == []


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
        source_feature_memory_projection_dim=8,
        source_feature_memory_max_frames=12,
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
    assert manifest["training"]["config"]["source_feature_memory_projection_dim"] == 8
    assert manifest["training"]["config"]["source_feature_memory_max_frames"] == 12
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


def test_trains_with_explicit_pseudo_teacher_rows_without_claiming_ground_truth(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    target = _stereo(0.22 * np.sin(2.0 * np.pi * 660.0 * time))
    source = target + _stereo(0.18 * np.sin(2.0 * np.pi * 160.0 * time))
    verified_rows = [
        {
            "id": "verified-a",
            "track_id": "verified-a",
            "stem_type": "vocal",
            "source_audio_path": str(_write_audio(tmp_path / "verified-a" / "source.flac", source, sample_rate)),
            "target_audio_path": str(_write_audio(tmp_path / "verified-a" / "target.flac", target, sample_rate)),
            "trainable": True,
            "split": "train",
            "human_verified": True,
            "reference_verified": False,
        }
    ]
    pseudo_rows = [
        {
            "schema": "generic_finetuning_pseudo_teacher_audio_training_example.v1",
            "id": "pseudo-b",
            "track_id": "pseudo-b",
            "stem_type": "vocal",
            "source_audio_path": str(_write_audio(tmp_path / "pseudo-b" / "source.flac", source, sample_rate)),
            "target_audio_path": str(_write_audio(tmp_path / "pseudo-b" / "target.flac", target, sample_rate)),
            "trainable": True,
            "split": "train",
            "pseudo_teacher": True,
            "human_verified": False,
            "reference_verified": False,
            "release_ready": False,
            "delivery_allowed": False,
        }
    ]
    examples = _write_jsonl(tmp_path / "verified.jsonl", verified_rows)
    pseudo_examples = _write_jsonl(tmp_path / "pseudo.jsonl", pseudo_rows)

    manifest = train_generic_audio_teacher_mask_adapter(
        audio_training_examples_jsonl=examples,
        pseudo_teacher_examples_jsonl=pseudo_examples,
        output_dir=tmp_path / "shared_adapter",
        stem="vocal",
        scope="pseudo_teacher_adapter",
        min_unique_tracks=2,
        n_fft=256,
        hop_length=64,
        hidden_size=64,
        epochs=8,
        batch_size=64,
        learning_rate=0.01,
        seed=2026,
        device="cpu",
    )

    assert manifest["selected_training_example_count"] == 2
    assert manifest["verified_training_example_count"] == 1
    assert manifest["pseudo_teacher_training_example_count"] == 1
    assert manifest["pseudo_teacher_examples_jsonl"] == str(pseudo_examples)
    assert manifest["contract"]["pseudo_teacher_targets_used_at_training"] is True
    assert manifest["contract"]["dirty_stems_never_ground_truth"] is True
    assert manifest["contract"]["pseudo_teacher_targets_are_ground_truth"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False

    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert progress_rows[0]["pseudo_teacher_input_row_count"] == 1
    assert progress_rows[0]["pseudo_teacher_training_example_count"] == 1
