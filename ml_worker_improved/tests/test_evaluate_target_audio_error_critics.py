from pathlib import Path

from tests.test_train_target_audio_error_critics import read_json, read_jsonl, trainable_rows, write_views


def test_evaluator_writes_per_target_metrics(tmp_path: Path) -> None:
    from tools.evaluate_target_audio_error_critics import evaluate_target_audio_error_critics
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    views_dir = tmp_path / "views"
    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
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

    result = evaluate_target_audio_error_critics(model_dir=model_dir, views_dir=views_dir, output_dir=eval_dir)

    assert (eval_dir / "target_audio_error_critic_eval.jsonl").exists()
    assert result["targets"]["minus"]["false_accept_rate"] is not None
    assert result["targets"]["vocal"]["false_accept_rate"] is not None
    assert "per_track_metrics" in result["targets"]["minus"]
    assert "per_route_metrics" in result["targets"]["vocal"]
    rows = read_jsonl(eval_dir / "target_audio_error_critic_eval.jsonl")
    assert {row["target_name"] for row in rows} == {"minus_critic", "vocal_critic"}
    assert result["release_ready"] is False
    assert read_json(eval_dir / "minus_audio_error_critic_metrics.json")["target_name"] == "minus_critic"
