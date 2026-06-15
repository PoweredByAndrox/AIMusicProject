import json
from pathlib import Path

import joblib
import numpy as np
import pytest


MAP_NAMES = (
    "stft_residual_magnitude",
    "phase_difference_map",
    "stereo_width_map",
    "vocal_bleed_proxy_map",
    "instrumental_bleed_proxy_map",
    "missing_detail_proxy_map",
    "transient_loss_proxy_map",
    "artifact_noise_proxy_map",
    "reconstruction_error_map",
)


class MeanFeatureProbabilityModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.mean(features, axis=1), 0.0, 1.0)
        return np.column_stack([1.0 - probabilities, probabilities])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_map(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.full((4, 6), value, dtype=np.float32) for name in MAP_NAMES}
    payload["waveform_residual"] = np.full((16, 1), value, dtype=np.float32)
    np.savez_compressed(path, **payload)


def tensor_row(tmp_path: Path, index: int, *, label: int, stem: str = "minus") -> dict:
    map_path = tmp_path / "maps" / f"row-{index}.npz"
    write_map(map_path, 0.2 + index * 0.1)
    return {
        "row_uid": f"row-{index}",
        "canonical_row_key": f"track-{index}|cand-{index}|{stem}",
        "track_id": f"track-{index}",
        "run_id": "run-1",
        "candidate_id": f"cand-{index}",
        "route_id": "route-a",
        "candidate_family": "family-a",
        "source_model_family": "model-a",
        "stem_type": stem,
        "split": "test",
        "label": label,
        "train_allowed": False,
        "hardcase": False,
        "hardcase_holdout": False,
        "blockers": [],
        "window_start_sec": float(index * 12),
        "window_end_sec": float(index * 12 + 12),
        "tensor_map_path": str(map_path),
        "tensor_map_names": list(MAP_NAMES),
        "scalar_features_used": False,
        "release_ready": False,
    }


def write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_dir = tmp_path / "dataset"
    write_jsonl(dataset_dir / "tensor_audio_error_dataset.jsonl", rows)
    write_json(
        dataset_dir / "locked_tensor_audio_error_dataset_manifest.json",
        {
            "schema": "locked_tensor_audio_error_dataset.v1",
            "rows_total": len(rows),
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
        },
    )
    return dataset_dir


def write_model_and_calibration(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    calibration_dir = tmp_path / "cal"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "critic_not_usable_reason": "",
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
            "shadow_only": True,
            "model_kind": "mean_feature_probability_test",
            "model_file": "tensor_audio_error_critic.joblib",
        },
    )
    joblib.dump(
        {"model": MeanFeatureProbabilityModel(), "required_map_names": list(MAP_NAMES)},
        model_dir / "tensor_audio_error_critic.joblib",
    )
    write_json(
        calibration_dir / "tensor_audio_error_thresholds.json",
        {
            "schema": "tensor_audio_error_critic_calibration.v1",
            "model_dir": str(model_dir),
            "calibration_status": "calibrated",
            "global": {"threshold": 0.5, "status": "usable", "false_accept_rate": 0.0, "accept_coverage": 1.0},
            "by_stem": {"minus": {"threshold": 0.45, "status": "usable"}, "vocal": {"threshold": 0.55, "status": "usable"}},
            "shadow_only": True,
            "scalar_fallback_used": False,
            "tensor_maps_required": True,
            "release_ready": False,
        },
    )
    return model_dir, calibration_dir


def test_score_tensor_critic_writes_shadow_only_scores(tmp_path: Path) -> None:
    from tools.score_tensor_audio_error_critic import score_tensor_audio_error_critic

    dataset_dir = write_dataset(
        tmp_path,
        [
            tensor_row(tmp_path, 0, label=0, stem="minus"),
            tensor_row(tmp_path, 1, label=1, stem="vocal"),
        ],
    )
    model_dir, calibration_dir = write_model_and_calibration(tmp_path)

    summary = score_tensor_audio_error_critic(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        calibration_dir=calibration_dir,
        output_dir=tmp_path / "scores",
    )

    rows = read_jsonl(tmp_path / "scores" / "tensor_audio_error_shadow_scores.jsonl")
    report = read_json(tmp_path / "scores" / "tensor_audio_error_shadow_score_summary.json")
    assert summary["rows_scored"] == 2
    assert report["schema"] == "tensor_audio_error_shadow_scores.v1"
    assert report["shadow_only"] is True
    assert report["release_ready"] is False
    assert rows
    assert {row["runtime_gate_decision_applied"] for row in rows} == {False}
    assert {row["release_ready"] for row in rows} == {False}
    assert {row["shadow_only"] for row in rows} == {True}
    assert all(row["calibration_status"] == "calibrated" for row in rows)
    assert all(row["shadow_decision"] in {"shadow_accept_candidate", "shadow_block_candidate"} for row in rows)


def test_score_tensor_critic_uses_global_threshold_when_stem_threshold_missing(tmp_path: Path) -> None:
    from tools.score_tensor_audio_error_critic import score_tensor_audio_error_critic

    dataset_dir = write_dataset(tmp_path, [tensor_row(tmp_path, 0, label=0, stem="other")])
    model_dir, calibration_dir = write_model_and_calibration(tmp_path)

    score_tensor_audio_error_critic(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        calibration_dir=calibration_dir,
        output_dir=tmp_path / "scores",
    )

    rows = read_jsonl(tmp_path / "scores" / "tensor_audio_error_shadow_scores.jsonl")
    assert rows[0]["threshold_scope"] == "global"
    assert rows[0]["calibrated_threshold"] == 0.5
    assert rows[0]["runtime_gate_decision_applied"] is False


def test_score_tensor_critic_rejects_unusable_calibration(tmp_path: Path) -> None:
    from tools.score_tensor_audio_error_critic import score_tensor_audio_error_critic

    dataset_dir = write_dataset(tmp_path, [tensor_row(tmp_path, 0, label=0)])
    model_dir, calibration_dir = write_model_and_calibration(tmp_path)
    write_json(
        calibration_dir / "tensor_audio_error_thresholds.json",
        {
            "schema": "tensor_audio_error_critic_calibration.v1",
            "model_dir": str(model_dir),
            "calibration_status": "not_usable",
            "shadow_only": True,
            "scalar_fallback_used": False,
            "tensor_maps_required": True,
            "release_ready": False,
        },
    )

    with pytest.raises(ValueError, match="tensor_critic_validation_failed"):
        score_tensor_audio_error_critic(
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            calibration_dir=calibration_dir,
            output_dir=tmp_path / "scores",
        )
