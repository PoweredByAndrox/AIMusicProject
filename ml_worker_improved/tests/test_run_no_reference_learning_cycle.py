from __future__ import annotations

import json
from pathlib import Path

from tools.promote_no_reference_critic_model import main as promote_main
from tools.run_no_reference_learning_cycle import main as learning_main


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_learning_cycle_excludes_raw_predictions_and_does_not_train(tmp_path: Path) -> None:
    outcomes_root = tmp_path / "runs"
    run_dir = outcomes_root / "song-001"
    current_model_dir = tmp_path / "models_current"
    output_model_dir = tmp_path / "models_next"
    eval_output_dir = tmp_path / "eval_next"
    write_json(current_model_dir / "model_manifest.json", {"model_version": "current", "safety": {"release_ready": False}})
    write_jsonl(
        run_dir / "learning_outcome.jsonl",
        [
            {
                "track_id": "song-001",
                "candidate_id": "candidate-a",
                "mode": "minus",
                "label": 1,
                "verified_outcome": False,
                "raw_prediction_only": True,
                "features": {"overall_no_reference_score": 0.9},
            }
        ],
    )

    exit_code = learning_main(
        [
            "--outcomes-root",
            str(outcomes_root),
            "--current-model-dir",
            str(current_model_dir),
            "--output-model-dir",
            str(output_model_dir),
            "--eval-output-dir",
            str(eval_output_dir),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((eval_output_dir / "learning_cycle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_prediction_rows_excluded"] == 1
    assert manifest["verified_examples_total"] == 0
    assert manifest["training_started"] is False
    assert manifest["release_ready"] is False
    assert manifest["promotion_approved"] is False
    assert not (output_model_dir / "model_manifest.json").exists()


def test_model_promotion_rejects_worse_model_and_keeps_current_active(tmp_path: Path) -> None:
    comparison_json = tmp_path / "comparison.json"
    candidate_model_dir = tmp_path / "models_next"
    active_model_dir = tmp_path / "models_active"
    promotion_dir = tmp_path / "promotion"
    write_json(candidate_model_dir / "model_manifest.json", {"model_version": "next", "safety": {"release_ready": False}})
    write_json(active_model_dir / "model_manifest.json", {"model_version": "current", "safety": {"release_ready": False}})
    write_json(
        comparison_json,
        {
            "current": {
                "max_false_accept_rate": 0.0,
                "pairwise_ranking_accuracy": 0.91,
                "disagreement_outcome_accuracy": 0.80,
            },
            "next": {
                "max_false_accept_rate": 0.25,
                "pairwise_ranking_accuracy": 0.89,
                "disagreement_outcome_accuracy": 0.60,
            },
            "blocker_safety_regressed": True,
            "release_ready": False,
        },
    )

    exit_code = promote_main(
        [
            "--comparison-json",
            str(comparison_json),
            "--candidate-model-dir",
            str(candidate_model_dir),
            "--active-model-dir",
            str(active_model_dir),
            "--output-dir",
            str(promotion_dir),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((promotion_dir / "promotion_manifest.json").read_text(encoding="utf-8"))
    active_manifest = json.loads((active_model_dir / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["promotion_status"] == "rejected"
    assert manifest["safety_regressed"] is True
    assert manifest["false_accept_rate_regressed"] is True
    assert manifest["release_ready"] is False
    assert active_manifest["model_version"] == "current"
