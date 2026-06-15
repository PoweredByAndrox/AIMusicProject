import json
from pathlib import Path

from tests.test_train_target_audio_error_critics import trainable_rows, write_views


def test_calibration_writes_explicit_skipped_reason_when_insufficient_data(tmp_path: Path) -> None:
    from tools.calibrate_target_audio_error_critics import calibrate_target_audio_error_critics
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    views_dir = tmp_path / "views"
    model_dir = tmp_path / "model"
    calibration_dir = tmp_path / "calibration"
    write_views(views_dir, trainable_rows(tmp_path, "minus_critic"), trainable_rows(tmp_path, "vocal_critic"))
    train_target_audio_error_critics(
        views_dir=views_dir,
        output_dir=model_dir,
        targets=("minus", "vocal"),
        model_kind="random_forest",
        group_split="candidate_id",
        shadow_only=True,
        min_safe_rows=4,
        min_positive=1,
        min_negative=1,
        min_negative_test_rows=20,
    )

    manifest = calibrate_target_audio_error_critics(
        model_dir=model_dir,
        views_dir=views_dir,
        output_dir=calibration_dir,
        method="auto",
    )

    assert manifest["targets"]["minus"]["calibration_status"] == "skipped"
    assert manifest["targets"]["minus"]["calibration_skipped_reason"]
    assert manifest["targets"]["vocal"]["calibration_skipped_reason"]
    assert (calibration_dir / "critic_calibration_manifest.json").exists()
    assert (calibration_dir / "critic_calibration_metrics.json").exists()
    assert json.loads((calibration_dir / "critic_calibration_metrics.json").read_text(encoding="utf-8"))["release_ready"] is False
