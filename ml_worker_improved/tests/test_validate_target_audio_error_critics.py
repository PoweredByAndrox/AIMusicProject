import json
from pathlib import Path

from tests.test_train_target_audio_error_critics import read_json, trainable_rows, write_json, write_views


def build_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from tools.evaluate_target_audio_error_critics import evaluate_target_audio_error_critics
    from tools.score_target_audio_error_views import score_target_audio_error_views
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    views_dir = tmp_path / "views"
    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
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
    evaluate_target_audio_error_critics(model_dir=model_dir, views_dir=views_dir, output_dir=eval_dir)
    score_target_audio_error_views(model_dir=model_dir, views_dir=views_dir, output_dir=score_dir)
    return views_dir, model_dir, eval_dir, score_dir


def test_validator_accepts_shadow_only_target_critics(tmp_path: Path) -> None:
    from tools.validate_target_audio_error_critics import validate_target_audio_error_critics

    _, model_dir, eval_dir, score_dir = build_artifacts(tmp_path)

    result = validate_target_audio_error_critics(model_dir=model_dir, eval_dir=eval_dir, scores_dir=score_dir)

    assert result["ok"] is True
    assert result["unsafe_rows_used"] is False
    assert result["pair_rows_used"] is False
    assert result["overall_rows_used"] is False
    assert result["shadow_only"] is True
    assert result["release_ready"] is False


def test_validator_rejects_unsafe_rows_shadow_false_and_release_ready(tmp_path: Path) -> None:
    from tools.validate_target_audio_error_critics import validate_target_audio_error_critics

    _, model_dir, eval_dir, score_dir = build_artifacts(tmp_path)
    manifest_path = model_dir / "target_audio_error_critics_manifest.json"
    manifest = read_json(manifest_path)
    manifest["unsafe_rows_used"] = True
    manifest["shadow_only"] = False
    manifest["release_ready"] = True
    write_json(manifest_path, manifest)

    result = validate_target_audio_error_critics(model_dir=model_dir, eval_dir=eval_dir, scores_dir=score_dir)

    assert result["ok"] is False
    assert "unsafe_rows_used_must_be_false:manifest" in result["errors"]
    assert "shadow_only_must_be_true:manifest" in result["errors"]
    assert "release_ready_must_be_false:manifest" in result["errors"]


def test_validator_rejects_missing_false_accept_rate_for_trained_target(tmp_path: Path) -> None:
    from tools.validate_target_audio_error_critics import validate_target_audio_error_critics

    _, model_dir, eval_dir, score_dir = build_artifacts(tmp_path)
    metrics_path = eval_dir / "minus_audio_error_critic_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.pop("false_accept_rate", None)
    write_json(metrics_path, metrics)

    result = validate_target_audio_error_critics(model_dir=model_dir, eval_dir=eval_dir, scores_dir=score_dir)

    assert result["ok"] is False
    assert "missing_false_accept_rate:minus" in result["errors"]


def test_validator_rejects_candidate_overlap_low_negatives_and_high_false_accept(tmp_path: Path) -> None:
    from tools.validate_target_audio_error_critics import validate_target_audio_error_critics

    _, model_dir, eval_dir, score_dir = build_artifacts(tmp_path)
    report_path = model_dir / "target_audio_error_critic_reliability_report.json"
    report = read_json(report_path)
    report["targets"]["minus"]["candidate_id_overlap_train_test"] = True
    report["targets"]["minus"]["test_negative_rows"] = 0
    report["targets"]["minus"]["false_accept_rate"] = 0.75
    write_json(report_path, report)

    result = validate_target_audio_error_critics(
        model_dir=model_dir,
        eval_dir=eval_dir,
        scores_dir=score_dir,
        max_false_accept_rate=0.10,
        min_negative_test_rows=20,
    )

    assert result["ok"] is False
    assert "candidate_id_overlap_train_test:minus" in result["errors"]
    assert "test_negative_rows_below_minimum:minus" in result["errors"]
    assert "false_accept_rate_above_max:minus" in result["errors"]


def test_validator_accepts_shadow_usable_when_reliability_gates_pass(tmp_path: Path) -> None:
    from tools.validate_target_audio_error_critics import validate_target_audio_error_critics

    _, model_dir, eval_dir, score_dir = build_artifacts(tmp_path)
    report_path = model_dir / "target_audio_error_critic_reliability_report.json"
    report = read_json(report_path)
    for target in ("minus", "vocal"):
        report["targets"][target]["candidate_id_overlap_train_test"] = False
        report["targets"][target]["test_negative_rows"] = 20
        report["targets"][target]["false_accept_rate"] = 0.0
        report["targets"][target]["calibration_status"] = "skipped"
        report["targets"][target]["calibration_skipped_reason"] = "not_requested"
        report["targets"][target]["per_route_false_accept_audit"] = {"route-a": {"false_accept_rate": 0.0}}
        report["targets"][target]["per_track_false_accept_audit"] = {"track-a": {"false_accept_rate": 0.0}}
    write_json(report_path, report)

    result = validate_target_audio_error_critics(
        model_dir=model_dir,
        eval_dir=eval_dir,
        scores_dir=score_dir,
        max_false_accept_rate=0.10,
        min_negative_test_rows=20,
    )

    assert result["ok"] is True
