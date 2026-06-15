import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_calibrator_finds_useful_zero_false_accept_threshold(tmp_path: Path) -> None:
    from tools.calibrate_tensor_audio_error_critic import calibrate_tensor_audio_error_critic

    model_dir = tmp_path / "model"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "release_ready": False,
            "scalar_fallback_used": False,
            "tensor_maps_required": True,
        },
    )
    write_jsonl(
        model_dir / "tensor_audio_error_critic_eval.jsonl",
        [
            {"stem_type": "minus", "split": "test", "label": 0, "predicted_accept_probability": 0.1},
            {"stem_type": "minus", "split": "test", "label": 0, "predicted_accept_probability": 0.2},
            {"stem_type": "minus", "split": "test", "label": 1, "predicted_accept_probability": 0.8},
            {"stem_type": "minus", "split": "test", "label": 1, "predicted_accept_probability": 0.9},
        ],
    )

    result = calibrate_tensor_audio_error_critic(model_dir=model_dir, output_dir=tmp_path / "cal", max_false_accept_rate=0.0)

    assert result["calibration_status"] == "calibrated"
    assert result["global"]["false_accept_rate"] == 0.0
    assert result["global"]["accept_coverage"] > 0
    assert result["release_ready"] is False


def test_calibrator_marks_unusable_when_zero_false_accept_has_no_coverage(tmp_path: Path) -> None:
    from tools.calibrate_tensor_audio_error_critic import calibrate_tensor_audio_error_critic

    model_dir = tmp_path / "model"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "release_ready": False,
            "scalar_fallback_used": False,
            "tensor_maps_required": True,
        },
    )
    write_jsonl(
        model_dir / "tensor_audio_error_critic_eval.jsonl",
        [
            {"stem_type": "minus", "split": "test", "label": 0, "predicted_accept_probability": 0.9},
            {"stem_type": "minus", "split": "test", "label": 1, "predicted_accept_probability": 0.8},
        ],
    )

    result = calibrate_tensor_audio_error_critic(model_dir=model_dir, output_dir=tmp_path / "cal", max_false_accept_rate=0.0)

    assert result["calibration_status"] == "not_usable"
    assert result["global"]["false_accept_rate"] == 0.0
    assert result["global"]["accept_coverage"] == 0.0
    assert "no_positive_accept_coverage_at_false_accept_budget" in result["not_usable_reasons"]
