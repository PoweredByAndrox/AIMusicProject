import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_decision_payload(**overrides: dict) -> dict:
    payload = {
        "schema": "tensor_audio_error_critic_decision.v1",
        "decision": "keep_as_specialist_shadow_model",
        "decision_allowed_values": [
            "keep_as_specialist_shadow_model",
            "mine_next_dataset_slice",
            "candidate_rejected_runtime",
        ],
        "promotion_allowed": False,
        "runtime_gate_update_allowed": False,
        "runtime_gates_changed": False,
        "non_locked_split_used": False,
        "dev_test_mining_rows_used_for_training": False,
        "contamination_status": {
            "critical_leakage_count": 0,
            "non_locked_split_used": False,
            "dev_test_mining_rows_used_for_training": False,
        },
        "hardcase_holdout": {
            "baseline": {"rows": 2, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
            "candidate": {"rows": 2, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
            "delta": {"hard_negative_count": 0.0, "false_reject_count": 0.0, "accuracy": 0.0},
        },
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "thresholds_relaxed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }
    payload.update(overrides)
    return payload


def safe_comparison_payload(**overrides: dict) -> dict:
    payload = {
        "schema": "tensor_audio_error_critic_variant_comparison.v1",
        "locked_eval": {"critical_leakage_count": 0, "future_r1_r2_r3_r4_must_use_this_split": True},
        "baseline": {
            "hardcase_holdout": {"rows": 2, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
            "release_ready": False,
            "runtime_gates_changed": False,
        },
        "candidate": {
            "hardcase_holdout": {"rows": 2, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
            "release_ready": False,
            "runtime_gates_changed": False,
        },
        "delta": {"hardcase_holdout": {"hard_negative_count": 0.0, "false_reject_count": 0.0, "accuracy": 0.0}},
        "release_ready": False,
        "runtime_gates_changed": False,
    }
    payload.update(overrides)
    return payload


def write_decision_dir(tmp_path: Path, decision: dict | None = None, comparison: dict | None = None) -> Path:
    decision_dir = tmp_path / "decision"
    write_json(
        decision_dir / "tensor_audio_error_critic_decision_manifest.json",
        decision if decision is not None else safe_decision_payload(),
    )
    write_json(
        decision_dir / "tensor_audio_error_critic_variant_comparison.json",
        comparison if comparison is not None else safe_comparison_payload(),
    )
    return decision_dir


def test_validator_accepts_safe_shadow_decision(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    result = validate_tensor_audio_error_critic_decision(write_decision_dir(tmp_path))

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["release_ready"] is False


def test_validator_rejects_promote_and_release_ready(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        decision=safe_decision_payload(
            decision="promote",
            promotion_allowed=True,
            release_ready=True,
        ),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "promotion_decision_not_allowed" in result["errors"]
    assert "promotion_allowed_true" in result["errors"]
    assert "release_ready_true" in result["errors"]


def test_hard_negative_decision_never_promotes(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        decision=safe_decision_payload(decision="promote", promotion_allowed=True),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "promotion_decision_not_allowed" in result["errors"]


def test_hard_negative_decision_requires_hardcase_metrics(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        decision=safe_decision_payload(hardcase_holdout={}),
        comparison=safe_comparison_payload(delta={}),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "missing_decision_hardcase_holdout_metrics" in result["errors"]
    assert "missing_comparison_hardcase_holdout_metrics" in result["errors"]


def test_validator_rejects_non_locked_split_and_missing_hardcase(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        decision=safe_decision_payload(
            non_locked_split_used=True,
            contamination_status={"critical_leakage_count": 0, "non_locked_split_used": True},
            hardcase_holdout={},
        ),
        comparison=safe_comparison_payload(candidate={"release_ready": False}),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "non_locked_split_used" in result["errors"]
    assert "missing_decision_hardcase_holdout_metrics" in result["errors"]
    assert "missing_comparison_hardcase_holdout_metrics" in result["errors"]


def test_validator_rejects_dev_test_mining_rows_used_for_training(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        decision=safe_decision_payload(
            dev_test_mining_rows_used_for_training=True,
            contamination_status={
                "critical_leakage_count": 0,
                "non_locked_split_used": False,
                "dev_test_mining_rows_used_for_training": True,
            },
        ),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "dev_test_mining_rows_used_for_training" in result["errors"]


def test_validator_rejects_comparison_contamination_even_if_decision_omits_it(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        comparison=safe_comparison_payload(
            contamination_status={
                "critical_leakage_count": 0,
                "non_locked_split_used": True,
                "dev_test_mining_rows_used_for_training": True,
            }
        ),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "non_locked_split_used" in result["errors"]
    assert "dev_test_mining_rows_used_for_training" in result["errors"]


def test_validator_rejects_hardcase_rows_without_metrics(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        decision=safe_decision_payload(hardcase_holdout={"baseline": {"rows": 2}, "candidate": {"rows": 2}, "delta": {}}),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "missing_decision_hardcase_holdout_metrics" in result["errors"]


def test_validator_rejects_locked_eval_errors_and_variant_safety_errors(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_critic_decision import validate_tensor_audio_error_critic_decision

    decision_dir = write_decision_dir(
        tmp_path,
        comparison=safe_comparison_payload(
            locked_eval={
                "critical_leakage_count": 0,
                "future_r1_r2_r3_r4_must_use_this_split": True,
                "errors": ["hardcase_holdout_not_locked"],
            },
            baseline={
                "hardcase_holdout": {"rows": 2, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
                "release_ready": False,
                "runtime_gates_changed": False,
                "safety_errors": ["missing_locked_eval_score_rows"],
            },
            candidate={
                "hardcase_holdout": {"rows": 2, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
                "release_ready": False,
                "runtime_gates_changed": False,
                "safety_errors": ["score_or_analysis_forbidden_true_flag"],
            },
        ),
    )

    result = validate_tensor_audio_error_critic_decision(decision_dir)

    assert result["ok"] is False
    assert "locked_eval_errors_present" in result["errors"]
    assert "baseline_safety_errors_present" in result["errors"]
    assert "candidate_safety_errors_present" in result["errors"]
