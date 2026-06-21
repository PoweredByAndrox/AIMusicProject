from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_candidate_family_blend_matrix import run_candidate_family_blend_matrix


def _write_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int = 8_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _training_rows(tmp_path: Path) -> list[dict]:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rows: list[dict] = []
    for index, track_id in enumerate(("track-a", "track-b", "track-c"), start=1):
        target = _stereo(0.24 * np.sin(2.0 * np.pi * (220.0 + index * 55.0) * t))
        error = _stereo(0.12 * np.sin(2.0 * np.pi * (90.0 + index * 10.0) * t))
        scnet = target + error
        melband = target - error
        target_path = _write_audio(tmp_path / track_id / "target.flac", target, sample_rate)
        paths = {
            "scnet": _write_audio(tmp_path / track_id / "scnet.flac", scnet, sample_rate),
            "melband": _write_audio(tmp_path / track_id / "melband.flac", melband, sample_rate),
        }
        for family, source_path in paths.items():
            rows.append(
                {
                    "id": f"{track_id}-{family}",
                    "track_id": track_id,
                    "stem_type": "vocal",
                    "source_candidate_family": family,
                    "source_audio_path": str(source_path),
                    "target_audio_path": str(target_path),
                    "target_audio_sha256": f"{track_id}-target",
                    "trainable": True,
                    "reference_verified": True,
                    "human_verified": False,
                    "release_ready": False,
                }
            )
    return rows


def test_blend_matrix_trains_global_weights_and_passes_holdout_without_release(tmp_path: Path) -> None:
    examples = _write_jsonl(tmp_path / "examples.jsonl", _training_rows(tmp_path))
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "configs": [
                {
                    "id": "scnet_melband_blend",
                    "families": ["scnet", "melband"],
                    "weight_grid_step": 0.1,
                    "baseline_policy": "best_family",
                }
            ]
        },
    )

    manifest = run_candidate_family_blend_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=tmp_path / "blend_matrix",
        stem="vocal",
        scope="v32_blend",
        exclude_holdout_track_ids={"track-c"},
        min_sdr_db=3.0,
        min_improvement_db=0.5,
        max_holdouts_per_config=2,
    )

    assert manifest["matrix_validation_status"] == "passed"
    assert manifest["best_config_id"] == "scnet_melband_blend"
    assert manifest["passed_config_count"] == 1
    assert manifest["release_ready"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["contract"]["baseline_policy"] == "best_family"
    assert manifest["contract"]["holdout_uses_verified_examples_only"] is True

    rows = _read_jsonl(Path(manifest["matrix_rows_jsonl"]))
    assert rows[0]["status"] == "passed"
    assert rows[0]["metrics"]["min_improvement_db"] > 0.5
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["release_ready"] is False

    checkpoint = _read_json(Path(rows[0]["checkpoint_path"]))
    assert checkpoint["families"] == ["scnet", "melband"]
    assert checkpoint["weights"] == {"scnet": 0.5, "melband": 0.5}
    assert checkpoint["training_track_ids"] == ["track-a", "track-b"]
    assert "track-c" not in checkpoint["training_track_ids"]

    holdout_rows = _read_jsonl(Path(rows[0]["holdout_rows_jsonl"]))
    assert len(holdout_rows) == 1
    assert holdout_rows[0]["holdout_track_id"] == "track-c"
    assert Path(holdout_rows[0]["candidate_audio_path"]).exists()
    assert holdout_rows[0]["baseline_policy"] == "best_family"
    assert holdout_rows[0]["release_ready"] is False

    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert progress_rows[0]["expected_acceleration_mode"] == "external_only_cpu_dsp_training_and_eval"
    assert progress_rows[0]["gpu_inference_skipped"] is True
    assert progress_rows[-1]["event"] == "finish"


def test_blend_matrix_fails_closed_when_holdout_group_lacks_required_family(tmp_path: Path) -> None:
    rows = [row for row in _training_rows(tmp_path) if not (row["track_id"] == "track-c" and row["source_candidate_family"] == "melband")]
    examples = _write_jsonl(tmp_path / "examples.jsonl", rows)
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"configs": [{"id": "scnet_melband_blend", "families": ["scnet", "melband"], "weight_grid_step": 0.1}]},
    )

    manifest = run_candidate_family_blend_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=tmp_path / "blend_matrix",
        stem="vocal",
        scope="v32_blend",
        exclude_holdout_track_ids={"track-c"},
        min_sdr_db=3.0,
        min_improvement_db=0.5,
    )

    rows_out = _read_jsonl(Path(manifest["matrix_rows_jsonl"]))
    assert manifest["matrix_validation_status"] == "failed"
    assert manifest["best_config_id"] == ""
    assert rows_out[0]["status"] == "failed"
    assert "holdout_complete_groups_missing" in rows_out[0]["failed_checks"]
    assert rows_out[0]["delivery_allowed"] is False
    assert rows_out[0]["release_ready"] is False


def test_blend_matrix_does_not_leak_holdout_track_into_training(tmp_path: Path) -> None:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rows: list[dict] = []
    for track_id, train_optimal_scnet in (("track-a", False), ("track-b", False), ("track-c", True)):
        target = _stereo(0.22 * np.sin(2.0 * np.pi * 330.0 * t))
        error = _stereo(0.18 * np.sin(2.0 * np.pi * 110.0 * t))
        scnet = target if train_optimal_scnet else target + error
        melband = target + error if train_optimal_scnet else target
        target_path = _write_audio(tmp_path / track_id / "target.flac", target, sample_rate)
        for family, audio in {"scnet": scnet, "melband": melband}.items():
            rows.append(
                {
                    "id": f"{track_id}-{family}",
                    "track_id": track_id,
                    "stem_type": "vocal",
                    "source_candidate_family": family,
                    "source_audio_path": str(_write_audio(tmp_path / track_id / f"{family}.flac", audio, sample_rate)),
                    "target_audio_path": str(target_path),
                    "target_audio_sha256": f"{track_id}-target",
                    "trainable": True,
                    "reference_verified": True,
                    "release_ready": False,
                }
            )
    examples = _write_jsonl(tmp_path / "examples.jsonl", rows)
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"configs": [{"id": "leak_guard", "families": ["scnet", "melband"], "weight_grid_step": 1.0}]},
    )

    manifest = run_candidate_family_blend_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=tmp_path / "blend_matrix",
        stem="vocal",
        scope="v32_blend",
        exclude_holdout_track_ids={"track-c"},
        min_sdr_db=3.0,
        min_improvement_db=0.5,
    )

    row = _read_jsonl(Path(manifest["matrix_rows_jsonl"]))[0]
    checkpoint = _read_json(Path(row["checkpoint_path"]))
    assert checkpoint["weights"] == {"scnet": 0.0, "melband": 1.0}
    assert checkpoint["training_track_ids"] == ["track-a", "track-b"]
    assert row["status"] == "failed"
    assert "holdout_metric_below_threshold" in row["failed_checks"]
