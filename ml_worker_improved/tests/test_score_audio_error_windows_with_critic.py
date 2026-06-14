import json
from pathlib import Path

from tests.test_train_audio_error_critic import trainable_rows, write_repaired_dataset


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_scorer_writes_per_window_critic_scores(tmp_path: Path) -> None:
    from tools.score_audio_error_windows_with_critic import score_audio_error_windows_with_critic
    from tools.train_audio_error_critic import train_audio_error_critic

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
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

    summary = score_audio_error_windows_with_critic(model_dir=model_dir, dataset_dir=dataset_dir, output_dir=score_dir)
    rows = read_jsonl(score_dir / "scored_audio_error_windows.jsonl")

    assert summary["rows_scored"] == 6
    assert rows
    assert all(row["critic_score_available"] is True for row in rows)
    for field in (
        "judge_accept_probability",
        "blocked_probability",
        "vocal_bleed_in_minus_score",
        "phase_collapse_score",
        "pair_compatibility_score",
        "overall_audio_error_score",
    ):
        assert field in rows[0]
    assert all(row["shadow_only"] is True for row in rows)
    assert all(row["release_ready"] is False for row in rows)


def test_scorer_does_not_fake_scores_when_model_was_skipped(tmp_path: Path) -> None:
    from tools.score_audio_error_windows_with_critic import score_audio_error_windows_with_critic
    from tools.train_audio_error_critic import train_audio_error_critic
    from tests.test_train_audio_error_critic import row

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    score_dir = tmp_path / "scores"
    write_repaired_dataset(dataset_dir, [row(tmp_path, index, accepted=True, track_id=f"track-{index}") for index in range(5)])
    train_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        min_safe_rows=4,
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
    )

    summary = score_audio_error_windows_with_critic(model_dir=model_dir, dataset_dir=dataset_dir, output_dir=score_dir)
    rows = read_jsonl(score_dir / "scored_audio_error_windows.jsonl")

    assert summary["rows_scored"] == 0
    assert all(row["critic_score_available"] is False for row in rows)
    assert all(row["critic_not_usable_reason"] == "missing_positive_or_negative_class" for row in rows)
    assert all("judge_accept_probability" not in row for row in rows)
