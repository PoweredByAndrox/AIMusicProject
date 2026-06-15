import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_request(tmp_path: Path, *, manifest_overrides: dict | None = None, row_overrides: dict | None = None) -> Path:
    request_dir = tmp_path / "request"
    row = {
        "requirement_schema": "tensor_audio_error_train_safe_mining_requirement.v1",
        "requirement_id": "req-a",
        "source_slice_label": "persistent_false_reject",
        "target_mining_label": "false_reject_positive",
        "target_training_label": 1,
        "stem_type": "minus",
        "top_error_channel_or_unknown": "unknown",
        "candidate_family_or_unknown": "htdemucs",
        "source_model_family_or_unknown": "htdemucs_ft",
        "route_family_or_unknown": "htdemucs_ft",
        "locked_eval_evidence_count": 2,
        "locked_eval_split_counts": {"dev": 1, "test": 1},
        "recommended_train_query": {"split": "train", "stem_type": "minus", "target_training_label": 1},
        "must_use_split": "train",
        "forbidden_splits": ["dev", "test", "hardcase", "holdout"],
        "training_allowed": False,
        "release_ready": False,
        "shadow_only": True,
    }
    row.update(row_overrides or {})
    write_jsonl(request_dir / "tensor_audio_error_train_safe_mining_requirements.jsonl", [row])
    manifest = {
        "schema": "tensor_audio_error_train_safe_mining_request.v1",
        "source_locked_slice_dir": str(tmp_path / "slice"),
        "source_locked_slice_manifest_path": str(tmp_path / "slice" / "tensor_audio_error_locked_mining_slice_manifest.json"),
        "locked_eval_only_source": True,
        "review_only_source": True,
        "training_rows_created": False,
        "training_allowed": False,
        "requirements_total": 1,
        "required_train_positive_rows_min": 10,
        "required_train_negative_rows_min": 10,
        "required_train_rows_min": 40,
        "min_per_label_rows": 10,
        "min_per_stem_rows": 10,
        "min_hard_negative_rows": 10,
        "min_false_reject_rows": 10,
        "release_ready": False,
        "runtime_gates_changed": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "shadow_only": True,
    }
    manifest.update(manifest_overrides or {})
    write_json(request_dir / "tensor_audio_error_train_safe_mining_request_manifest.json", manifest)
    return request_dir


def test_validate_train_safe_request_accepts_safe_request(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_safe_mining_request import validate_tensor_audio_error_train_safe_mining_request

    result = validate_tensor_audio_error_train_safe_mining_request(write_request(tmp_path))

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["release_ready"] is False


def test_validate_train_safe_request_rejects_training_allowed_and_dev_target(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_safe_mining_request import validate_tensor_audio_error_train_safe_mining_request

    request_dir = write_request(
        tmp_path,
        manifest_overrides={"training_allowed": True},
        row_overrides={"must_use_split": "dev", "training_allowed": True},
    )

    result = validate_tensor_audio_error_train_safe_mining_request(request_dir)

    assert result["ok"] is False
    assert "training_allowed_true" in result["errors"]
    assert "row_training_allowed_true:0" in result["errors"]
    assert "row_must_use_split_not_train:0" in result["errors"]
