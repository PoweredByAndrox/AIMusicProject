from pathlib import Path

from tests.test_train_target_audio_error_critics import read_jsonl, trainable_rows, write_views


def test_scorer_writes_per_window_target_scores(tmp_path: Path) -> None:
    from tools.score_target_audio_error_views import score_target_audio_error_views
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    views_dir = tmp_path / "views"
    model_dir = tmp_path / "model"
    score_dir = tmp_path / "scores"
    write_views(views_dir, trainable_rows(tmp_path, "minus_critic"), trainable_rows(tmp_path, "vocal_critic"))
    train_target_audio_error_critics(
        views_dir=views_dir,
        output_dir=model_dir,
        targets=("minus", "vocal"),
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
        min_safe_rows=4,
        min_positive=1,
        min_negative=1,
    )

    summary = score_target_audio_error_views(model_dir=model_dir, views_dir=views_dir, output_dir=score_dir)

    minus_rows = read_jsonl(score_dir / "scored_minus_critic_windows.jsonl")
    vocal_rows = read_jsonl(score_dir / "scored_vocal_critic_windows.jsonl")
    assert summary["minus_rows_scored"] == 8
    assert summary["vocal_rows_scored"] == 8
    assert minus_rows and vocal_rows
    for row in minus_rows + vocal_rows:
        assert row["shadow_only"] is True
        assert 0.0 <= row["predicted_accept_probability"] <= 1.0
        assert 0.0 <= row["predicted_block_probability"] <= 1.0
        assert row["critic_decision"] in {"accept_candidate", "blocked_hard_case"}
        assert row["release_ready"] is False
