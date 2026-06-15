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


def score_row(row_uid: str, *, label: int, split: str, decision: str) -> dict:
    return {
        "row_uid": row_uid,
        "canonical_row_key": f"track-a|cand-a|{row_uid}",
        "track_id": "track-a",
        "run_id": "run-a",
        "candidate_id": f"cand-{row_uid}",
        "route_id": "route-a",
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs",
        "stem_type": "minus",
        "split": split,
        "true_label": label,
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "predicted_accept_probability": 0.9 if decision == "shadow_accept_candidate" else 0.1,
        "calibrated_threshold": 0.5,
        "threshold_scope": "stem:minus",
        "shadow_decision": decision,
        "runtime_gate_decision_applied": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "shadow_only": True,
    }


def write_decision_fixture(tmp_path: Path) -> Path:
    decision_dir = tmp_path / "decision"
    baseline_scores_dir = tmp_path / "baseline_scores"
    candidate_scores_dir = tmp_path / "candidate_scores"
    baseline_rows = [
        score_row("regressed-positive", label=1, split="test", decision="shadow_accept_candidate"),
        score_row("persistent-positive", label=1, split="dev", decision="shadow_block_candidate"),
        score_row("regressed-negative", label=0, split="test", decision="shadow_block_candidate"),
        score_row("persistent-negative", label=0, split="dev", decision="shadow_accept_candidate"),
        score_row("train-only", label=1, split="train", decision="shadow_accept_candidate"),
    ]
    candidate_rows = [
        score_row("regressed-positive", label=1, split="test", decision="shadow_block_candidate"),
        score_row("persistent-positive", label=1, split="dev", decision="shadow_block_candidate"),
        score_row("regressed-negative", label=0, split="test", decision="shadow_accept_candidate"),
        score_row("persistent-negative", label=0, split="dev", decision="shadow_accept_candidate"),
        score_row("train-only", label=1, split="train", decision="shadow_block_candidate"),
    ]
    write_jsonl(baseline_scores_dir / "tensor_audio_error_shadow_scores.jsonl", baseline_rows)
    write_jsonl(candidate_scores_dir / "tensor_audio_error_shadow_scores.jsonl", candidate_rows)
    comparison = {
        "schema": "tensor_audio_error_critic_variant_comparison.v1",
        "locked_eval": {
            "locked_eval_dir": str(tmp_path / "locked_eval"),
            "critical_leakage_count": 0,
            "future_r1_r2_r3_r4_must_use_this_split": True,
            "hardcase_rows_count": 1,
        },
        "baseline": {"scores_dir": str(baseline_scores_dir), "release_ready": False, "runtime_gates_changed": False},
        "candidate": {"scores_dir": str(candidate_scores_dir), "release_ready": False, "runtime_gates_changed": False},
        "contamination_status": {"non_locked_split_used": False, "dev_test_mining_rows_used_for_training": False},
        "release_ready": False,
        "runtime_gates_changed": False,
    }
    write_json(decision_dir / "tensor_audio_error_critic_variant_comparison.json", comparison)
    write_json(
        decision_dir / "tensor_audio_error_critic_decision_manifest.json",
        {
            "schema": "tensor_audio_error_critic_decision.v1",
            "decision": "mine_next_dataset_slice",
            "promotion_allowed": False,
            "runtime_gate_update_allowed": False,
            "runtime_gates_changed": False,
            "non_locked_split_used": False,
            "dev_test_mining_rows_used_for_training": False,
            "hardcase_holdout": {
                "baseline": {"rows": 1, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
                "candidate": {"rows": 1, "hard_negative_count": 0, "false_reject_count": 0, "accuracy": 1.0},
                "delta": {"hard_negative_count": 0.0, "false_reject_count": 0.0, "accuracy": 0.0},
            },
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
            "shadow_only": True,
        },
    )
    return decision_dir


def test_build_locked_mining_slice_writes_dev_test_only_candidate_rows(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_locked_mining_slice import build_tensor_audio_error_locked_mining_slice

    decision_dir = write_decision_fixture(tmp_path)

    manifest = build_tensor_audio_error_locked_mining_slice(
        decision_dir=decision_dir,
        output_dir=tmp_path / "slice",
    )

    rows = read_jsonl(tmp_path / "slice" / "tensor_audio_error_locked_mining_slice.jsonl")
    persisted = read_json(tmp_path / "slice" / "tensor_audio_error_locked_mining_slice_manifest.json")
    assert manifest == persisted
    assert manifest["schema"] == "tensor_audio_error_locked_mining_slice.v1"
    assert manifest["slice_status"] == "candidate_only_locked_eval_review"
    assert manifest["candidate_rows_total"] == 4
    assert manifest["train_rows_ignored"] == 1
    assert manifest["candidate_regressed_false_reject_rows"] == 1
    assert manifest["persistent_false_reject_rows"] == 1
    assert manifest["candidate_regressed_hard_negative_rows"] == 1
    assert manifest["persistent_hard_negative_rows"] == 1
    assert manifest["slice_label_counts"] == {
        "candidate_regressed_false_reject": 1,
        "candidate_regressed_hard_negative": 1,
        "persistent_false_reject": 1,
        "persistent_hard_negative": 1,
    }
    assert {row["split"] for row in rows} == {"dev", "test"}
    assert {row["training_allowed"] for row in rows} == {False}
    assert {row["release_ready"] for row in rows} == {False}
    assert {row["runtime_gates_changed"] for row in rows} == {False}


def test_locked_mining_slice_rejects_promotion_decision(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_locked_mining_slice import build_tensor_audio_error_locked_mining_slice

    decision_dir = write_decision_fixture(tmp_path)
    manifest_path = decision_dir / "tensor_audio_error_critic_decision_manifest.json"
    payload = read_json(manifest_path)
    payload["decision"] = "promote"
    write_json(manifest_path, payload)

    try:
        build_tensor_audio_error_locked_mining_slice(decision_dir=decision_dir, output_dir=tmp_path / "slice")
    except ValueError as exc:
        assert "promotion_decision_not_allowed" in str(exc)
    else:
        raise AssertionError("promotion decision was not rejected")


def test_locked_mining_slice_rejects_runtime_rejected_decision(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_locked_mining_slice import build_tensor_audio_error_locked_mining_slice

    decision_dir = write_decision_fixture(tmp_path)
    manifest_path = decision_dir / "tensor_audio_error_critic_decision_manifest.json"
    payload = read_json(manifest_path)
    payload["decision"] = "candidate_rejected_runtime"
    write_json(manifest_path, payload)

    try:
        build_tensor_audio_error_locked_mining_slice(decision_dir=decision_dir, output_dir=tmp_path / "slice")
    except ValueError as exc:
        assert "source_decision_rejected_runtime_not_sliceable" in str(exc)
    else:
        raise AssertionError("runtime-rejected decision was not rejected")


def test_locked_mining_slice_rejects_locked_eval_errors(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_locked_mining_slice import build_tensor_audio_error_locked_mining_slice

    decision_dir = write_decision_fixture(tmp_path)
    comparison_path = decision_dir / "tensor_audio_error_critic_variant_comparison.json"
    payload = read_json(comparison_path)
    payload["locked_eval"]["errors"] = ["hardcase_holdout_not_locked"]
    write_json(comparison_path, payload)

    try:
        build_tensor_audio_error_locked_mining_slice(decision_dir=decision_dir, output_dir=tmp_path / "slice")
    except ValueError as exc:
        assert "locked_eval_errors_present" in str(exc)
    else:
        raise AssertionError("locked-eval errors were not rejected")


def test_locked_mining_slice_rejects_variant_safety_errors(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_locked_mining_slice import build_tensor_audio_error_locked_mining_slice

    decision_dir = write_decision_fixture(tmp_path)
    comparison_path = decision_dir / "tensor_audio_error_critic_variant_comparison.json"
    payload = read_json(comparison_path)
    payload["baseline"]["safety_errors"] = ["missing_locked_eval_score_rows"]
    payload["candidate"]["safety_errors"] = ["score_or_analysis_forbidden_true_flag"]
    write_json(comparison_path, payload)

    try:
        build_tensor_audio_error_locked_mining_slice(decision_dir=decision_dir, output_dir=tmp_path / "slice")
    except ValueError as exc:
        assert "baseline_safety_errors_present" in str(exc)
        assert "candidate_safety_errors_present" in str(exc)
    else:
        raise AssertionError("variant safety errors were not rejected")
