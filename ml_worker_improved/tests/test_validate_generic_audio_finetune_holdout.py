from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import validate_generic_audio_finetune_holdout as holdout_module
from tools.validate_generic_audio_finetune_holdout import select_holdout_examples, validate_generic_audio_finetune_holdout


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int = 8_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _stem_path(manifest: dict, stem: str = "vocal") -> Path:
    return Path(manifest["stems"][stem]["path"])


def test_holdout_progress_records_mps_as_gpu_inference(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.jsonl"

    holdout_module.progress(progress_path, "start", device="mps")

    row = _read_jsonl(progress_path)[0]
    assert row["expected_acceleration_mode"] == "mps_gpu_inference_holdout"
    assert row["gpu_inference_required"] is True
    assert row["gpu_inference_skipped"] is False


def test_select_holdout_examples_filters_source_candidate_family(tmp_path: Path) -> None:
    source_a = _write_audio(tmp_path / "scnet" / "source.flac", _stereo(np.zeros(128, dtype=np.float32)))
    target_a = _write_audio(tmp_path / "scnet" / "target.flac", _stereo(np.zeros(128, dtype=np.float32)))
    source_b = _write_audio(tmp_path / "melband" / "source.flac", _stereo(np.zeros(128, dtype=np.float32)))
    target_b = _write_audio(tmp_path / "melband" / "target.flac", _stereo(np.zeros(128, dtype=np.float32)))
    rows = [
        {
            "id": "holdout-scnet",
            "track_id": "track-a",
            "stem_type": "vocal",
            "source_candidate_family": "scnet",
            "source_audio_path": str(source_a),
            "target_audio_path": str(target_a),
            "trainable": True,
        },
        {
            "id": "holdout-melband",
            "track_id": "track-b",
            "stem_type": "vocal",
            "source_candidate_family": "melband",
            "source_audio_path": str(source_b),
            "target_audio_path": str(target_b),
            "trainable": True,
        },
    ]

    selected = select_holdout_examples(
        rows,
        stem="vocal",
        forbidden_track_ids=set(),
        max_holdouts=4,
        source_candidate_family_filter="scnet",
    )

    assert [row["id"] for row in selected] == ["holdout-scnet"]


def test_holdout_evaluates_only_cross_track_examples_and_blocks_delivery(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    holdout_target = _stereo(0.25 * np.sin(2.0 * np.pi * 440.0 * t))
    holdout_source = holdout_target + _stereo(0.35 * np.sin(2.0 * np.pi * 110.0 * t))
    same_track_source = _stereo(0.2 * np.sin(2.0 * np.pi * 220.0 * t))
    same_track_target = same_track_source * 0.5
    train_source = _stereo(0.15 * np.sin(2.0 * np.pi * 330.0 * t))
    train_target = train_source * 0.75

    train_source_path = _write_audio(tmp_path / "train" / "source.flac", train_source, sample_rate)
    train_target_path = _write_audio(tmp_path / "train" / "target.flac", train_target, sample_rate)
    same_track_source_path = _write_audio(tmp_path / "same" / "source.flac", same_track_source, sample_rate)
    same_track_target_path = _write_audio(tmp_path / "same" / "target.flac", same_track_target, sample_rate)
    holdout_source_path = _write_audio(tmp_path / "holdout" / "source.flac", holdout_source, sample_rate)
    holdout_target_path = _write_audio(tmp_path / "holdout" / "target.flac", holdout_target, sample_rate)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    run_rows = _write_jsonl(
        tmp_path / "run_rows.jsonl",
        [
            {
                "scope": "vocal_teacher_mask_adapter",
                "status": "skipped_existing_complete",
                "example_id": "train-example",
                "track_id": "track-a",
                "source_audio_path": str(train_source_path),
                "target_audio_path": str(train_target_path),
                "checkpoint_path": str(checkpoint_path),
                "device_used": 1.0,
                "progress_first_event": "start",
                "progress_last_event": "finish",
            }
        ],
    )
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            {
                "id": "same-track",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_audio_path": str(same_track_source_path),
                "target_audio_path": str(same_track_target_path),
                "trainable": True,
            },
            {
                "id": "holdout-example",
                "track_id": "track-b",
                "stem_type": "vocal",
                "source_audio_path": str(holdout_source_path),
                "target_audio_path": str(holdout_target_path),
                "trainable": True,
            },
        ],
    )

    def fake_apply(*, source_mix_path, checkpoint_path, output_dir, stem_name, device, command_args=None, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"01_teacher_mask_checkpoint_{stem_name}.flac"
        _write_audio(output_path, holdout_target, sample_rate)
        return {
            "application": {"diagnostics": {"device": "cuda", "device_used": 1.0}},
            "stems": {stem_name: {"path": str(output_path)}},
        }

    import tools.validate_generic_audio_finetune_holdout as module

    monkeypatch.setattr(module, "apply_teacher_mask_checkpoint_to_source", fake_apply)

    manifest = validate_generic_audio_finetune_holdout(
        run_rows_jsonl=run_rows,
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "holdout_eval",
        scope="vocal_teacher_mask_adapter",
        stem="vocal",
        device="cuda",
        min_sdr_db=12.0,
        min_improvement_db=1.0,
    )

    rows = _read_jsonl(Path(manifest["holdout_rows_jsonl"]))
    assert len(rows) == 1
    assert rows[0]["checkpoint_track_id"] == "track-a"
    assert rows[0]["holdout_track_id"] == "track-b"
    assert rows[0]["status"] == "passed"
    assert rows[0]["application_device"] == "cuda"
    assert rows[0]["application_device_used"] == 1.0
    assert manifest["holdout_validation_status"] == "passed"
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_holdout_fails_closed_when_checkpoint_does_not_improve_source_baseline(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    target = _stereo(0.22 * np.sin(2.0 * np.pi * 660.0 * t))
    source = target + _stereo(0.42 * np.sin(2.0 * np.pi * 120.0 * t))
    source_path = _write_audio(tmp_path / "source.flac", source, sample_rate)
    target_path = _write_audio(tmp_path / "target.flac", target, sample_rate)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    run_rows = _write_jsonl(
        tmp_path / "run_rows.jsonl",
        [
            {
                "scope": "vocal_teacher_mask_adapter",
                "status": "executed",
                "example_id": "train-example",
                "track_id": "track-a",
                "source_audio_path": str(tmp_path / "train_source.flac"),
                "target_audio_path": str(tmp_path / "train_target.flac"),
                "checkpoint_path": str(checkpoint_path),
                "device_used": 1.0,
                "progress_first_event": "start",
                "progress_last_event": "finish",
            }
        ],
    )
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            {
                "id": "holdout-example",
                "track_id": "track-b",
                "stem_type": "vocal",
                "source_audio_path": str(source_path),
                "target_audio_path": str(target_path),
                "trainable": True,
            }
        ],
    )

    def fake_apply(*, source_mix_path, checkpoint_path, output_dir, stem_name, device, command_args=None, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"01_teacher_mask_checkpoint_{stem_name}.flac"
        _write_audio(output_path, source, sample_rate)
        return {"stems": {stem_name: {"path": str(output_path)}}}

    import tools.validate_generic_audio_finetune_holdout as module

    monkeypatch.setattr(module, "apply_teacher_mask_checkpoint_to_source", fake_apply)

    manifest = validate_generic_audio_finetune_holdout(
        run_rows_jsonl=run_rows,
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "holdout_eval",
        scope="vocal_teacher_mask_adapter",
        stem="vocal",
        device="cuda",
        min_sdr_db=0.0,
        min_improvement_db=0.5,
    )

    rows = _read_jsonl(Path(manifest["holdout_rows_jsonl"]))
    assert rows[0]["status"] == "failed"
    assert "improvement_below_threshold" in rows[0]["failed_checks"]
    assert manifest["holdout_validation_status"] == "failed"
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_holdout_excludes_all_training_track_ids_for_shared_checkpoints(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    target_c = _stereo(0.2 * np.sin(2.0 * np.pi * 550.0 * t))
    source_c = target_c + _stereo(0.3 * np.sin(2.0 * np.pi * 150.0 * t))
    target_b = _stereo(0.2 * np.sin(2.0 * np.pi * 440.0 * t))
    source_b = target_b + _stereo(0.3 * np.sin(2.0 * np.pi * 120.0 * t))
    source_b_path = _write_audio(tmp_path / "b" / "source.flac", source_b, sample_rate)
    target_b_path = _write_audio(tmp_path / "b" / "target.flac", target_b, sample_rate)
    source_c_path = _write_audio(tmp_path / "c" / "source.flac", source_c, sample_rate)
    target_c_path = _write_audio(tmp_path / "c" / "target.flac", target_c, sample_rate)
    checkpoint_path = tmp_path / "shared.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    run_rows = _write_jsonl(
        tmp_path / "run_rows.jsonl",
        [
            {
                "scope": "vocal_shared_teacher_mask_adapter",
                "status": "executed",
                "example_id": "shared",
                "track_id": "shared-vocal-adapter",
                "training_track_ids": ["track-a", "track-b"],
                "checkpoint_path": str(checkpoint_path),
                "device_used": 1.0,
                "progress_first_event": "start",
                "progress_last_event": "finish",
            }
        ],
    )
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            {
                "id": "blocked-training-track",
                "track_id": "track-b",
                "stem_type": "vocal",
                "source_audio_path": str(source_b_path),
                "target_audio_path": str(target_b_path),
                "trainable": True,
            },
            {
                "id": "true-holdout",
                "track_id": "track-c",
                "stem_type": "vocal",
                "source_audio_path": str(source_c_path),
                "target_audio_path": str(target_c_path),
                "trainable": True,
            },
        ],
    )

    def fake_apply(*, source_mix_path, checkpoint_path, output_dir, stem_name, device, command_args=None, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"01_teacher_mask_checkpoint_{stem_name}.flac"
        _write_audio(output_path, target_c, sample_rate)
        return {
            "application": {"diagnostics": {"device": "cuda", "device_used": 1.0}},
            "stems": {stem_name: {"path": str(output_path)}},
        }

    import tools.validate_generic_audio_finetune_holdout as module

    monkeypatch.setattr(module, "apply_teacher_mask_checkpoint_to_source", fake_apply)

    manifest = validate_generic_audio_finetune_holdout(
        run_rows_jsonl=run_rows,
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "holdout_eval",
        scope="vocal_shared_teacher_mask_adapter",
        stem="vocal",
        device="cuda",
        min_sdr_db=12.0,
        min_improvement_db=1.0,
    )

    rows = _read_jsonl(Path(manifest["holdout_rows_jsonl"]))
    assert [row["holdout_track_id"] for row in rows] == ["track-c"]
    assert rows[0]["status"] == "passed"


def test_holdout_records_source_feature_memory_cost_guard(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = _stereo(0.26 * np.sin(2.0 * np.pi * 440.0 * t))
    target = _stereo(0.18 * np.sin(2.0 * np.pi * 660.0 * t))
    source_path = _write_audio(tmp_path / "source.flac", source, sample_rate)
    target_path = _write_audio(tmp_path / "target.flac", target, sample_rate)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    run_rows = _write_jsonl(
        tmp_path / "run_rows.jsonl",
        [
            {
                "scope": "vocal_teacher_mask_adapter",
                "status": "executed",
                "example_id": "train-example",
                "track_id": "track-a",
                "checkpoint_path": str(checkpoint_path),
                "device_used": 1.0,
                "progress_first_event": "start",
                "progress_last_event": "finish",
            }
        ],
    )
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            {
                "id": "holdout-example",
                "track_id": "track-b",
                "stem_type": "vocal",
                "source_audio_path": str(source_path),
                "target_audio_path": str(target_path),
                "trainable": True,
            }
        ],
    )

    import tools.validate_generic_audio_finetune_holdout as module

    def fake_apply(
        *,
        source_mix_path,
        checkpoint_path,
        output_dir,
        stem_name,
        device,
        command_args=None,
        progress_path=None,
        progress_context=None,
        max_source_feature_memory_distance_ops=None,
        **kwargs,
    ):
        assert progress_path is not None
        module.progress(
            progress_path,
            "source_feature_memory_search_blocked",
            device=device,
            **dict(progress_context or {}),
            reason="source_feature_memory_cost_guard",
            estimated_distance_ops=2,
            max_source_feature_memory_distance_ops=max_source_feature_memory_distance_ops,
        )
        raise RuntimeError("source_feature_memory_cost_guard: estimated_distance_ops=2 exceeds max=1")

    monkeypatch.setattr(module, "apply_teacher_mask_checkpoint_to_source", fake_apply)

    manifest = validate_generic_audio_finetune_holdout(
        run_rows_jsonl=run_rows,
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "holdout_eval",
        scope="vocal_teacher_mask_adapter",
        stem="vocal",
        device="cuda",
        max_source_feature_memory_distance_ops=1,
    )

    rows = _read_jsonl(Path(manifest["holdout_rows_jsonl"]))
    assert rows[0]["status"] == "failed"
    assert "application_or_metric_error" in rows[0]["failed_checks"]
    assert "source_feature_memory_cost_guard" in rows[0]["failed_checks"]
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["release_ready"] is False
    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert any(row["event"] == "source_feature_memory_search_blocked" for row in progress_rows)
    assert any(row["event"] == "holdout_application_failed" for row in progress_rows)
    assert progress_rows[-1]["event"] == "finish"
