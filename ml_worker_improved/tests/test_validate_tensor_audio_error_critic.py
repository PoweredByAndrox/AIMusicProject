import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_validator_rejects_scalar_fallback_and_release_ready(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic import validate_tensor_audio_error_critic

    model_dir = tmp_path / "model"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "tensor_maps_required": True,
            "scalar_fallback_used": True,
            "critical_leakage_count": 0,
            "release_ready": True,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )

    result = validate_tensor_audio_error_critic(model_dir)

    assert result["ok"] is False
    assert "scalar_fallback_used" in result["errors"]
    assert "release_ready_true" in result["errors"]


def test_validator_accepts_shadow_tensor_critic(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic import validate_tensor_audio_error_critic

    model_dir = tmp_path / "model"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    (model_dir / "tensor_audio_error_critic.joblib").write_bytes(b"placeholder")

    result = validate_tensor_audio_error_critic(model_dir)

    assert result["ok"] is True
    assert result["errors"] == []


def test_validator_rejects_required_unusable_calibration(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic import validate_tensor_audio_error_critic

    model_dir = tmp_path / "model"
    calibration_dir = tmp_path / "cal"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    (model_dir / "tensor_audio_error_critic.joblib").write_bytes(b"placeholder")
    write_json(
        calibration_dir / "tensor_audio_error_thresholds.json",
        {
            "schema": "tensor_audio_error_critic_calibration.v1",
            "calibration_status": "not_usable",
            "release_ready": False,
            "scalar_fallback_used": False,
        },
    )

    result = validate_tensor_audio_error_critic(
        model_dir,
        calibration_dir=calibration_dir,
        require_trained=True,
        require_usable_calibration=True,
    )

    assert result["ok"] is False
    assert "calibration_not_usable" in result["errors"]


def test_validator_requires_calibration_dir_when_requested(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic import validate_tensor_audio_error_critic

    model_dir = tmp_path / "model"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    (model_dir / "tensor_audio_error_critic.joblib").write_bytes(b"placeholder")

    result = validate_tensor_audio_error_critic(
        model_dir,
        require_trained=True,
        require_usable_calibration=True,
    )

    assert result["ok"] is False
    assert "missing_required_calibration_dir" in result["errors"]


def test_validator_rejects_malformed_calibration_contract(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic import validate_tensor_audio_error_critic

    model_dir = tmp_path / "model"
    calibration_dir = tmp_path / "cal"
    write_json(
        model_dir / "tensor_audio_error_critic_manifest.json",
        {
            "schema": "tensor_audio_error_critic.v1",
            "training_completed": True,
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    (model_dir / "tensor_audio_error_critic.joblib").write_bytes(b"placeholder")
    write_json(
        calibration_dir / "tensor_audio_error_thresholds.json",
        {
            "schema": "wrong.v1",
            "calibration_status": "calibrated",
            "model_dir": str(tmp_path / "foreign_model"),
            "release_ready": False,
            "scalar_fallback_used": False,
            "tensor_maps_required": False,
            "shadow_only": False,
        },
    )

    result = validate_tensor_audio_error_critic(
        model_dir,
        calibration_dir=calibration_dir,
        require_trained=True,
        require_usable_calibration=True,
    )

    assert result["ok"] is False
    assert "invalid_calibration_schema" in result["errors"]
    assert "calibration_tensor_maps_not_required" in result["errors"]
    assert "calibration_not_shadow_only" in result["errors"]
    assert "calibration_model_dir_mismatch" in result["errors"]
