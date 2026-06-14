from __future__ import annotations

import json
from pathlib import Path

from tools.build_no_reference_ml_dataset import write_jsonl
from tools.eval_no_reference_critic import evaluate_critics
from tools.run_no_reference_learned_critic import score_rows
from tools.train_no_reference_critic import train_critics


def example(
    track_id: str,
    candidate_id: str,
    mode: str,
    label: int,
    score: float,
    *,
    split: str,
    policy_action: str = "accept_candidate",
) -> dict:
    return {
        "example_id": f"{track_id}:{candidate_id}:{mode}",
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mode": mode,
        "label": label,
        "policy_action": policy_action,
        "split": split,
        "script_score": score,
        "features": {
            "overall_no_reference_score": score,
            "confidence": score,
            "vocal_score": score if mode != "minus" else 0.45,
            "minus_score": score if mode != "vocal" else 0.45,
            "pair_compatibility_score": score,
            "vocal_leakage_in_minus_score": 1.0 - score,
            "instrumental_leakage_in_vocal_score": 1.0 - score,
            "mixture_consistency_score": score,
            "phase_consistency_score": score,
            "window_fitness": score,
            "reason_count": 0 if label else 1,
            "forbidden_action_count": 1,
        },
        "source_artifact_refs": ["toy"],
        "strict_mode_label": True,
        "release_ready": False,
        "training_started": False,
        "separator_training_started": False,
    }


def write_examples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_train_saves_local_models_manifest_and_eval_summary(tmp_path: Path) -> None:
    examples_path = tmp_path / "dataset" / "no_reference_ml_examples.jsonl"
    pairwise_path = tmp_path / "dataset" / "no_reference_pairwise_examples.jsonl"
    model_dir = tmp_path / "models"
    eval_dir = tmp_path / "eval"
    rows = [
        example("track-a", "vocal-good", "vocal", 1, 0.91, split="train"),
        example("track-b", "vocal-bad", "vocal", 0, 0.21, split="dev"),
        example("track-c", "minus-good", "minus", 1, 0.88, split="train"),
        example("track-d", "minus-bad", "minus", 0, 0.24, split="test"),
        example("track-e", "pair-good", "both", 1, 0.86, split="train"),
        example("track-f", "pair-bad", "both", 0, 0.27, split="test"),
        example("track-g", "window-good", "window", 1, 0.84, split="train", policy_action="preserve_candidate"),
        example("track-h", "window-bad", "window", 0, 0.32, split="dev", policy_action="block_candidate"),
    ]
    pairwise_rows = [
        {
            "pair_id": "track-x:vocal",
            "track_id": "track-x",
            "mode": "vocal",
            "preferred_example_id": "good",
            "rejected_example_id": "bad",
            "preferred_score": 0.91,
            "rejected_score": 0.21,
            "split": "test",
            "source_artifact_refs": ["toy"],
        }
    ]
    write_examples(examples_path, rows)
    write_examples(pairwise_path, pairwise_rows)

    manifest = train_critics(examples_jsonl=examples_path, pairwise_jsonl=pairwise_path, model_dir=model_dir)

    for name in [
        "vocal_critic.joblib",
        "minus_critic.joblib",
        "pair_critic.joblib",
        "policy_controller.joblib",
        "model_manifest.json",
    ]:
        assert (model_dir / name).exists()
    assert manifest["safety"]["release_ready"] is False
    assert manifest["safety"]["separator_training_started"] is False
    assert manifest["safety"]["training_started"] is False
    assert manifest["training_data_hash"]
    assert manifest["feature_list"]
    assert set(manifest["models"]) >= {"vocal_critic", "minus_critic", "pair_critic", "policy_controller"}

    summary = evaluate_critics(
        examples_jsonl=examples_path,
        pairwise_jsonl=pairwise_path,
        model_dir=model_dir,
        output_dir=eval_dir,
    )
    assert (eval_dir / "evaluation_summary.md").exists()
    assert "false_accept_rate" in json.dumps(summary)
    assert "pairwise_ranking_accuracy" in summary


def test_learned_critic_scores_candidate_and_safety_blocks_accept(tmp_path: Path) -> None:
    examples_path = tmp_path / "dataset" / "examples.jsonl"
    pairwise_path = tmp_path / "dataset" / "pairs.jsonl"
    model_dir = tmp_path / "models"
    rows = [
        example("track-a", "minus-good", "minus", 1, 0.92, split="train"),
        example("track-b", "minus-bad", "minus", 0, 0.18, split="test"),
        example("track-c", "vocal-good", "vocal", 1, 0.9, split="train"),
        example("track-d", "vocal-bad", "vocal", 0, 0.2, split="test"),
        example("track-e", "pair-good", "both", 1, 0.89, split="train"),
        example("track-f", "pair-bad", "both", 0, 0.25, split="test"),
    ]
    write_examples(examples_path, rows)
    write_examples(pairwise_path, [])
    train_critics(examples_jsonl=examples_path, pairwise_jsonl=pairwise_path, model_dir=model_dir)

    candidate_rows = [
        {
            "track_id": "track-z",
            "candidate_id": "unsafe-minus",
            "score_mode": "minus",
            "overall_no_reference_score": 0.94,
            "minus_score": 0.94,
            "vocal_score": 0.94,
            "pair_compatibility_score": 0.94,
            "confidence": 0.94,
            "decision": "accept_candidate",
            "reasons": ["vocal_leakage_in_minus"],
            "forbidden_actions": ["do_not_release_consider_leakage_repair_or_new_model_route"],
        }
    ]

    scored = score_rows(candidate_rows, model_dir=model_dir)
    assert len(scored) == 1
    row = scored[0]
    for field in [
        "learned_vocal_score",
        "learned_minus_score",
        "learned_pair_score",
        "learned_policy_action",
        "learned_confidence",
        "script_score",
        "hybrid_score",
    ]:
        assert field in row
    assert row["hybrid_score"] == round(0.5 * row["script_score"] + 0.5 * row["learned_score"], 6)
    assert row["safety_blocked"] is True
    assert row["learned_policy_action"] != "accept_candidate"
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False


def test_write_jsonl_helper_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [{"b": 2, "a": 1}])
    assert path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'
