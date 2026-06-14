import json
from pathlib import Path

from tests.test_train_audio_error_critic import trainable_rows, write_json, write_repaired_dataset


def test_validator_accepts_shadow_only_critic_model(tmp_path: Path) -> None:
    from tools.evaluate_audio_error_critic import evaluate_audio_error_critic
    from tools.score_audio_error_windows_with_critic import score_audio_error_windows_with_critic
    from tools.train_audio_error_critic import train_audio_error_critic
    from tools.validate_audio_error_critic_model import validate_audio_error_critic_model

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
    score_dir = tmp_path / "scores"
    write_repaired_dataset(dataset_dir, trainable_rows(tmp_path))
    train_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        min_safe_rows=4,
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
    )
    evaluate_audio_error_critic(model_dir=model_dir, dataset_dir=dataset_dir, output_dir=eval_dir)
    score_audio_error_windows_with_critic(model_dir=model_dir, dataset_dir=dataset_dir, output_dir=score_dir)

    result = validate_audio_error_critic_model(model_dir=model_dir, eval_dir=eval_dir, scores_dir=score_dir)

    assert result["ok"] is True
    assert result["unsafe_rows_used"] is False
    assert result["release_ready"] is False
    assert result["separator_training_started"] is False


def test_validator_rejects_model_manifest_if_unsafe_rows_were_used(tmp_path: Path) -> None:
    from tools.validate_audio_error_critic_model import validate_audio_error_critic_model

    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
    score_dir = tmp_path / "scores"
    write_json(
        model_dir / "audio_error_critic_manifest.json",
        {
            "model_id": "bad",
            "feature_columns": ["residual_energy"],
            "target_columns": ["judge_accept_probability"],
            "unsafe_rows_used": True,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
            "critic_usable_for_shadow_scoring": False,
            "decision": "critic_training_skipped_insufficient_data",
            "critic_not_usable_reason": "test",
        },
    )

    result = validate_audio_error_critic_model(model_dir=model_dir, eval_dir=eval_dir, scores_dir=score_dir)

    assert result["ok"] is False
    assert "unsafe_rows_used_must_be_false:manifest" in result["errors"]


def test_validator_rejects_release_ready_and_missing_feature_schema(tmp_path: Path) -> None:
    from tools.validate_audio_error_critic_model import validate_audio_error_critic_model

    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
    score_dir = tmp_path / "scores"
    write_json(
        model_dir / "audio_error_critic_manifest.json",
        {
            "model_id": "bad",
            "target_columns": ["judge_accept_probability"],
            "unsafe_rows_used": False,
            "release_ready": True,
            "training_started": False,
            "separator_training_started": False,
            "critic_usable_for_shadow_scoring": False,
            "decision": "critic_training_skipped_insufficient_data",
            "critic_not_usable_reason": "test",
        },
    )

    result = validate_audio_error_critic_model(model_dir=model_dir, eval_dir=eval_dir, scores_dir=score_dir)

    assert result["ok"] is False
    assert "release_ready_must_be_false:manifest" in result["errors"]
    assert "missing_feature_schema:manifest" in result["errors"]


def test_validator_rejects_separator_training_started(tmp_path: Path) -> None:
    from tools.validate_audio_error_critic_model import validate_audio_error_critic_model

    model_dir = tmp_path / "model"
    write_json(
        model_dir / "audio_error_critic_manifest.json",
        {
            "model_id": "bad",
            "feature_columns": ["residual_energy"],
            "target_columns": ["judge_accept_probability"],
            "unsafe_rows_used": False,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": True,
            "critic_usable_for_shadow_scoring": False,
            "decision": "critic_training_skipped_insufficient_data",
            "critic_not_usable_reason": "test",
        },
    )

    result = validate_audio_error_critic_model(model_dir=model_dir, eval_dir=tmp_path / "eval", scores_dir=tmp_path / "scores")

    assert result["ok"] is False
    assert "separator_training_started_must_be_false:manifest" in result["errors"]
