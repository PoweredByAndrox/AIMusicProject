import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_locked_eval(tmp_path: Path, hardcase_row_ids: list[str]) -> Path:
    locked_eval_dir = tmp_path / "locked_eval"
    write_json(
        locked_eval_dir / "leak_safe_split_manifest.json",
        {
            "schema": "leak_safe_audio_eval_harness.v1",
            "source_dataset_id": "locked-dataset-1",
            "critical_leakage_count": 0,
            "warning_leakage_count": 0,
            "hardcase_rows_count": len(hardcase_row_ids),
            "future_r1_r2_r3_r4_must_use_this_split": True,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    write_json(
        locked_eval_dir / "leakage_report.json",
        {
            "schema": "leakage_report.v1",
            "critical_leakage_count": 0,
            "checks": {"same_candidate_train_test": {"critical": False}},
            "release_ready": False,
        },
    )
    write_json(
        locked_eval_dir / "hardcase_holdout_manifest.json",
        {
            "schema": "hardcase_holdout_manifest.v1",
            "locked": True,
            "hardcase_rows_count": len(hardcase_row_ids),
            "hardcase_rows": [
                {"row_uid": row_uid, "split": "test", "train_allowed": False, "release_ready": False}
                for row_uid in hardcase_row_ids
            ],
            "release_ready": False,
        },
    )
    return locked_eval_dir


def write_model_dir(
    tmp_path: Path,
    name: str,
    *,
    auroc: float,
    accuracy: float,
    mining: dict | None = None,
) -> Path:
    model_dir = tmp_path / name
    manifest = {
        "schema": "tensor_audio_error_critic.v1",
        "source_dataset_dir": str(tmp_path / "tensor_dataset"),
        "training_completed": True,
        "train_rows": 10,
        "eval_rows": 4,
        "model_kind": "random_forest_balanced",
        "tensor_maps_required": True,
        "scalar_fallback_used": False,
        "shadow_only": True,
        "eval_metrics": {
            "rows": 4,
            "accuracy": accuracy,
            "auroc": auroc,
            "false_accept": 1,
            "false_accept_rate": 0.5,
            "false_reject": 1,
            "false_reject_rate": 0.5,
            "positive_rows": 2,
            "negative_rows": 2,
        },
        "threshold_metrics": {"accept_coverage": 0.5, "threshold_false_accept_rate": 0.0},
        "release_ready": False,
        "can_publish_as_result": False,
        "runtime_gates_changed": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "thresholds_relaxed": False,
    }
    manifest.update(mining or {})
    write_json(model_dir / "tensor_audio_error_critic_manifest.json", manifest)
    return model_dir


def write_analysis_dir(tmp_path: Path, name: str, *, hard_negative_count: int, false_reject_count: int) -> Path:
    analysis_dir = tmp_path / name
    write_json(
        analysis_dir / "tensor_audio_error_shadow_analysis_summary.json",
        {
            "schema": "tensor_audio_error_shadow_analysis.v1",
            "rows_analyzed": 4,
            "negative_rows": 2,
            "positive_rows": 2,
            "hard_negative_count": hard_negative_count,
            "false_reject_count": false_reject_count,
            "hard_negative_rate": hard_negative_count / 2,
            "false_reject_rate": false_reject_count / 2,
            "by_stem": {},
            "by_route": {},
            "calibration_status": "calibrated",
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    return analysis_dir


def score_row(row_uid: str, *, label: int, decision: str, stem: str = "minus", route: str = "route-a") -> dict:
    return {
        "row_uid": row_uid,
        "track_id": "track-a" if row_uid in {"hard-neg", "hard-pos"} else "track-b",
        "candidate_id": f"cand-{row_uid}",
        "route_id": route,
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs",
        "stem_type": stem,
        "split": "test" if row_uid.startswith("hard") else "dev",
        "true_label": label,
        "predicted_accept_probability": 0.9 if decision == "shadow_accept_candidate" else 0.1,
        "shadow_decision": decision,
        "calibration_status": "calibrated",
        "release_ready": False,
        "runtime_gates_changed": False,
        "runtime_gate_decision_applied": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "shadow_only": True,
    }


def write_scores_dir(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    scores_dir = tmp_path / name
    write_jsonl(scores_dir / "tensor_audio_error_shadow_scores.jsonl", rows)
    write_json(
        scores_dir / "tensor_audio_error_shadow_score_summary.json",
        {
            "schema": "tensor_audio_error_shadow_scores.v1",
            "rows_scored": len(rows),
            "calibration_status": "calibrated",
            "model_kind": "random_forest_balanced",
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
            "shadow_only": True,
        },
    )
    return scores_dir


def write_tensor_dataset_manifest(tmp_path: Path, locked_eval_dir: Path) -> None:
    write_json(
        tmp_path / "tensor_dataset" / "locked_tensor_audio_error_dataset_manifest.json",
        {
            "schema": "locked_tensor_audio_error_dataset.v1",
            "source_split_dir": str(locked_eval_dir),
            "source_split_manifest": str(locked_eval_dir / "leak_safe_split_manifest.json"),
            "source_split_dataset_id": "locked-dataset-1",
            "critical_leakage_count": 0,
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "release_ready": False,
        },
    )


def test_compare_writes_locked_decision_for_specialist_shadow_model(tmp_path: Path) -> None:
    from tools.compare_tensor_audio_error_critic_variants import compare_tensor_audio_error_critic_variants

    locked_eval_dir = write_locked_eval(tmp_path, ["hard-neg", "hard-pos"])
    write_tensor_dataset_manifest(tmp_path, locked_eval_dir)
    baseline_model = write_model_dir(tmp_path, "baseline_model", auroc=0.82, accuracy=0.75)
    candidate_model = write_model_dir(
        tmp_path,
        "candidate_model",
        auroc=0.80,
        accuracy=0.75,
        mining={
            "mining_bundle_used": True,
            "mining_candidate_rows_total": 4,
            "mining_train_rows_available": 2,
            "mining_train_rows_matched": 2,
            "mining_non_train_rows_ignored": 2,
        },
    )
    baseline_scores = write_scores_dir(
        tmp_path,
        "baseline_scores",
        [
            score_row("hard-neg", label=0, decision="shadow_accept_candidate"),
            score_row("hard-pos", label=1, decision="shadow_block_candidate", stem="vocal"),
            score_row("dev-neg", label=0, decision="shadow_block_candidate"),
            score_row("dev-pos", label=1, decision="shadow_accept_candidate", stem="vocal"),
        ],
    )
    candidate_scores = write_scores_dir(
        tmp_path,
        "candidate_scores",
        [
            score_row("hard-neg", label=0, decision="shadow_block_candidate"),
            score_row("hard-pos", label=1, decision="shadow_accept_candidate", stem="vocal"),
            score_row("dev-neg", label=0, decision="shadow_block_candidate"),
            score_row("dev-pos", label=1, decision="shadow_accept_candidate", stem="vocal"),
        ],
    )
    baseline_analysis = write_analysis_dir(tmp_path, "baseline_analysis", hard_negative_count=1, false_reject_count=1)
    candidate_analysis = write_analysis_dir(tmp_path, "candidate_analysis", hard_negative_count=0, false_reject_count=0)

    result = compare_tensor_audio_error_critic_variants(
        locked_eval_dir=locked_eval_dir,
        baseline_model_dir=baseline_model,
        baseline_scores_dir=baseline_scores,
        baseline_analysis_dir=baseline_analysis,
        candidate_model_dir=candidate_model,
        candidate_scores_dir=candidate_scores,
        candidate_analysis_dir=candidate_analysis,
        output_dir=tmp_path / "decision",
    )

    comparison = read_json(tmp_path / "decision" / "tensor_audio_error_critic_variant_comparison.json")
    decision = read_json(tmp_path / "decision" / "tensor_audio_error_critic_decision_manifest.json")
    comparison_rows = read_jsonl(tmp_path / "decision" / "tensor_audio_error_critic_variant_comparison.jsonl")
    assert result["decision"] == "keep_as_specialist_shadow_model"
    assert decision["decision"] == "keep_as_specialist_shadow_model"
    assert decision["promotion_allowed"] is False
    assert decision["runtime_gate_update_allowed"] is False
    assert decision["release_ready"] is False
    assert decision["contamination_status"]["non_locked_split_used"] is False
    assert decision["contamination_status"]["dev_test_mining_rows_used_for_training"] is False
    assert comparison["candidate"]["hardcase_holdout"]["hard_negative_count"] == 0
    assert comparison["baseline"]["hardcase_holdout"]["hard_negative_count"] == 1
    assert {row["variant"] for row in comparison_rows} == {"baseline", "candidate", "delta"}
    assert (tmp_path / "decision" / "tensor_audio_error_critic_decision_summary.md").exists()


def test_compare_ignores_train_rows_for_locked_eval_decision(tmp_path: Path) -> None:
    from tools.compare_tensor_audio_error_critic_variants import compare_tensor_audio_error_critic_variants

    locked_eval_dir = write_locked_eval(tmp_path, ["hard-neg"])
    write_tensor_dataset_manifest(tmp_path, locked_eval_dir)
    baseline_model = write_model_dir(tmp_path, "baseline_model", auroc=0.82, accuracy=0.75)
    candidate_model = write_model_dir(tmp_path, "candidate_model", auroc=0.82, accuracy=0.75)
    baseline_scores = write_scores_dir(
        tmp_path,
        "baseline_scores",
        [
            score_row("hard-neg", label=0, decision="shadow_block_candidate"),
            {**score_row("train-pos", label=1, decision="shadow_block_candidate", stem="vocal"), "split": "train"},
        ],
    )
    candidate_scores = write_scores_dir(
        tmp_path,
        "candidate_scores",
        [
            score_row("hard-neg", label=0, decision="shadow_block_candidate"),
            {**score_row("train-pos", label=1, decision="shadow_accept_candidate", stem="vocal"), "split": "train"},
        ],
    )
    baseline_analysis = write_analysis_dir(tmp_path, "baseline_analysis", hard_negative_count=0, false_reject_count=1)
    candidate_analysis = write_analysis_dir(tmp_path, "candidate_analysis", hard_negative_count=0, false_reject_count=0)

    result = compare_tensor_audio_error_critic_variants(
        locked_eval_dir=locked_eval_dir,
        baseline_model_dir=baseline_model,
        baseline_scores_dir=baseline_scores,
        baseline_analysis_dir=baseline_analysis,
        candidate_model_dir=candidate_model,
        candidate_scores_dir=candidate_scores,
        candidate_analysis_dir=candidate_analysis,
        output_dir=tmp_path / "decision",
    )

    comparison = read_json(tmp_path / "decision" / "tensor_audio_error_critic_variant_comparison.json")
    assert result["decision"] == "mine_next_dataset_slice"
    assert comparison["baseline"]["locked_eval_metrics"]["rows"] == 1
    assert comparison["candidate"]["locked_eval_metrics"]["rows"] == 1
    assert comparison["delta"]["false_reject_count"] == 0.0


def test_compare_rejects_dev_test_mining_contamination_from_either_variant(tmp_path: Path) -> None:
    from tools.compare_tensor_audio_error_critic_variants import compare_tensor_audio_error_critic_variants

    locked_eval_dir = write_locked_eval(tmp_path, ["hard-neg"])
    write_tensor_dataset_manifest(tmp_path, locked_eval_dir)
    baseline_model = write_model_dir(
        tmp_path,
        "baseline_model",
        auroc=0.82,
        accuracy=0.75,
        mining={"dev_test_mining_rows_used_for_training": True},
    )
    candidate_model = write_model_dir(
        tmp_path,
        "candidate_model",
        auroc=0.82,
        accuracy=0.75,
        mining={"mining_non_train_rows_used_for_training": 1},
    )
    scores = [score_row("hard-neg", label=0, decision="shadow_block_candidate")]
    baseline_scores = write_scores_dir(tmp_path, "baseline_scores", scores)
    candidate_scores = write_scores_dir(tmp_path, "candidate_scores", scores)
    baseline_analysis = write_analysis_dir(tmp_path, "baseline_analysis", hard_negative_count=0, false_reject_count=0)
    candidate_analysis = write_analysis_dir(tmp_path, "candidate_analysis", hard_negative_count=0, false_reject_count=0)

    result = compare_tensor_audio_error_critic_variants(
        locked_eval_dir=locked_eval_dir,
        baseline_model_dir=baseline_model,
        baseline_scores_dir=baseline_scores,
        baseline_analysis_dir=baseline_analysis,
        candidate_model_dir=candidate_model,
        candidate_scores_dir=candidate_scores,
        candidate_analysis_dir=candidate_analysis,
        output_dir=tmp_path / "decision",
    )

    assert result["decision"] == "candidate_rejected_runtime"
    assert result["contamination_status"]["dev_test_mining_rows_used_for_training"] is True
    assert "safety_flags_failed" in result["decision_reasons"]
