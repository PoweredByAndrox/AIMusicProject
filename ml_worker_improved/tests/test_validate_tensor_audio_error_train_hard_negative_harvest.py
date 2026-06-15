import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_harvest(tmp_path: Path, *, row_overrides: dict | None = None) -> Path:
    harvest_dir = tmp_path / "harvest"
    tensor_path = tmp_path / "maps" / "row.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"npz-placeholder")
    row = {
        "row_schema": "tensor_audio_error_train_hard_negative_candidate.v1",
        "harvest_candidate_id": "hard-neg-a",
        "row_uid": "row-a",
        "canonical_row_key": "key-a",
        "track_id": "track-a",
        "run_id": "run-a",
        "candidate_id": "cand-a",
        "route_id": "route-a",
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs",
        "stem_type": "minus",
        "split": "train",
        "train_allowed": True,
        "hardcase_holdout": False,
        "locked_eval_row": False,
        "split_overlap_checked": True,
        "blocked_overlap_keys": [],
        "tensor_map_path": str(tensor_path),
        "tensor_map_names": ["stft_residual_magnitude"],
        "true_label": 0,
        "source_mining_label": "hard_negative_false_accept",
        "label_source": "train_split_existing_judge_or_train_only_generation",
        "error_reason": "artifact_noise_proxy_map",
        "blockers": ["artifact_noise_proxy_map"],
        "top_error_channel": "artifact_noise_proxy_map",
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "runtime_gate_decision_applied": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "release_ready": False,
        "shadow_only": True,
    }
    row.update(row_overrides or {})
    write_jsonl(harvest_dir / "tensor_audio_error_train_hard_negative_candidates.jsonl", [row])
    write_jsonl(harvest_dir / "tensor_audio_error_train_hard_negative_skipped.jsonl", [])
    write_json(
        harvest_dir / "tensor_audio_error_train_hard_negative_harvest_manifest.json",
        {
            "schema": "tensor_audio_error_train_hard_negative_harvest.v1",
            "harvest_status": "harvest_ready",
            "source_mode": "existing_train_rows",
            "source_a_existing_train_rows_used": True,
            "source_b_candidate_generation_used": False,
            "candidate_rows_total": 1,
            "negative_rows": 1,
            "hard_negative_rows": 1,
            "positive_rows": 0,
            "split": "train",
            "train_only": True,
            "dev_test_rows_used": False,
            "locked_eval_rows_used": False,
            "hardcase_rows_used": False,
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_started": False,
            "critic_training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
            "shadow_only": True,
        },
    )
    return harvest_dir


def test_hard_negative_harvest_rejects_dev_test_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_hard_negative_harvest import validate_tensor_audio_error_train_hard_negative_harvest

    result = validate_tensor_audio_error_train_hard_negative_harvest(write_harvest(tmp_path, row_overrides={"split": "test"}))

    assert result["ok"] is False
    assert "row_split_not_train:0" in result["errors"]


def test_hard_negative_harvest_rejects_locked_eval_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_hard_negative_harvest import validate_tensor_audio_error_train_hard_negative_harvest

    result = validate_tensor_audio_error_train_hard_negative_harvest(write_harvest(tmp_path, row_overrides={"locked_eval_row": True}))

    assert result["ok"] is False
    assert "row_locked_eval_row_true:0" in result["errors"]


def test_hard_negative_harvest_requires_label_zero(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_hard_negative_harvest import validate_tensor_audio_error_train_hard_negative_harvest

    result = validate_tensor_audio_error_train_hard_negative_harvest(write_harvest(tmp_path, row_overrides={"true_label": 1}))

    assert result["ok"] is False
    assert "row_true_label_not_zero:0" in result["errors"]


def test_hard_negative_harvest_requires_tensor_maps(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_hard_negative_harvest import validate_tensor_audio_error_train_hard_negative_harvest

    result = validate_tensor_audio_error_train_hard_negative_harvest(
        write_harvest(tmp_path, row_overrides={"tensor_map_path": str(tmp_path / "missing.npz")})
    )

    assert result["ok"] is False
    assert "row_missing_tensor_map:0" in result["errors"]


def test_hard_negative_harvest_validator_allows_non_holdout_hardcase_train_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_hard_negative_harvest import validate_tensor_audio_error_train_hard_negative_harvest

    result = validate_tensor_audio_error_train_hard_negative_harvest(
        write_harvest(tmp_path, row_overrides={"hardcase": True, "hardcase_holdout": False})
    )

    assert result["ok"] is True


def test_hard_negative_harvest_validator_rejects_hardcase_holdout_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_hard_negative_harvest import validate_tensor_audio_error_train_hard_negative_harvest

    result = validate_tensor_audio_error_train_hard_negative_harvest(
        write_harvest(tmp_path, row_overrides={"hardcase": True, "hardcase_holdout": True})
    )

    assert result["ok"] is False
    assert "row_hardcase_holdout_true:0" in result["errors"]
