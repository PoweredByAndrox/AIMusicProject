import json
from pathlib import Path

from tests.test_train_audio_error_critic import trainable_rows, write_repaired_dataset


def test_evaluator_reports_false_accept_rate_and_group_metrics(tmp_path: Path) -> None:
    from tools.evaluate_audio_error_critic import evaluate_audio_error_critic
    from tools.train_audio_error_critic import train_audio_error_critic

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
    write_repaired_dataset(dataset_dir, trainable_rows(tmp_path))
    train_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        min_safe_rows=4,
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
    )

    metrics = evaluate_audio_error_critic(model_dir=model_dir, dataset_dir=dataset_dir, output_dir=eval_dir)

    assert (eval_dir / "audio_error_critic_metrics.json").exists()
    assert (eval_dir / "audio_error_critic_eval.jsonl").exists()
    assert (eval_dir / "audio_error_critic_summary.md").exists()
    assert "false_accept_rate" in metrics
    assert "false_reject_rate" in metrics
    assert "per_track_metrics" in metrics
    assert "per_route_metrics" in metrics
    assert metrics["release_ready"] is False
    assert metrics["separator_training_started"] is False


def test_evaluator_keeps_skipped_model_diagnostic_without_fake_scores(tmp_path: Path) -> None:
    from tools.evaluate_audio_error_critic import evaluate_audio_error_critic
    from tools.train_audio_error_critic import train_audio_error_critic
    from tests.test_train_audio_error_critic import row

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
    write_repaired_dataset(dataset_dir, [row(tmp_path, index, accepted=True, track_id=f"track-{index}") for index in range(5)])
    train_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        min_safe_rows=4,
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
    )

    metrics = evaluate_audio_error_critic(model_dir=model_dir, dataset_dir=dataset_dir, output_dir=eval_dir)
    eval_rows = [
        json.loads(line)
        for line in (eval_dir / "audio_error_critic_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert metrics["critic_usable_for_shadow_scoring"] is False
    assert metrics["false_accept_rate"] is None
    assert eval_rows == []
    assert metrics["release_ready"] is False
