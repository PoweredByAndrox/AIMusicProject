import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_bundle(tmp_path: Path, *, manifest_overrides: dict | None = None, row_overrides: dict | None = None) -> Path:
    bundle_dir = tmp_path / "bundle"
    tensor_path = tmp_path / "maps" / "row.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"npz-placeholder")
    row = {
        "training_candidate_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "bundle_row_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "training_candidate_id": "cand-a",
        "source_requirement_id": "req-a",
        "source_request_dir": str(tmp_path / "request"),
        "row_uid": "row-a",
        "canonical_row_key": "key-a",
        "track_id": "track-a",
        "run_id": "run-a",
        "candidate_id": "candidate-a",
        "route_id": "route-a",
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs_ft",
        "stem_type": "minus",
        "split": "train",
        "train_allowed": True,
        "hardcase_holdout": False,
        "locked_eval_row": False,
        "tensor_map_path": str(tensor_path),
        "tensor_map_names": ["stft_residual_magnitude"],
        "recommended_training_label": 1,
        "source_true_label": 1,
        "source_mining_label": "false_reject_positive",
        "label_source": "train_safe_analog_from_locked_eval_request",
        "training_candidate_status": "candidate_only",
        "eligible_for_future_training_review": True,
        "runtime_gate_decision_applied": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "release_ready": False,
        "shadow_only": True,
    }
    row.update(row_overrides or {})
    write_jsonl(bundle_dir / "tensor_audio_error_train_only_training_candidates.jsonl", [row])
    manifest = {
        "schema": "tensor_audio_error_train_only_mining_bundle.v1",
        "bundle_status": "blocked_insufficient_train_safe_rows",
        "candidate_rows_total": 1,
        "positive_rows": 1,
        "negative_rows": 0,
        "training_allowed": False,
        "tensor_maps_required": True,
        "scalar_features_used": False,
        "dev_test_mining_rows_used_for_training": False,
        "locked_eval_rows_used_for_training": False,
        "hardcase_rows_used_for_training": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "shadow_only": True,
    }
    manifest.update(manifest_overrides or {})
    write_json(bundle_dir / "tensor_audio_error_train_only_mining_bundle_manifest.json", manifest)
    return bundle_dir


def test_validate_train_only_bundle_rejects_dev_test_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_only_mining_bundle import validate_tensor_audio_error_train_only_mining_bundle

    result = validate_tensor_audio_error_train_only_mining_bundle(write_bundle(tmp_path, row_overrides={"split": "dev"}))

    assert result["ok"] is False
    assert "row_split_not_train:0" in result["errors"]


def test_validate_train_only_bundle_rejects_hardcase_holdout_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_only_mining_bundle import validate_tensor_audio_error_train_only_mining_bundle

    result = validate_tensor_audio_error_train_only_mining_bundle(write_bundle(tmp_path, row_overrides={"hardcase_holdout": True}))

    assert result["ok"] is False
    assert "row_hardcase_holdout_true:0" in result["errors"]


def test_validate_train_only_bundle_rejects_missing_tensor_maps(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_train_only_mining_bundle import validate_tensor_audio_error_train_only_mining_bundle

    result = validate_tensor_audio_error_train_only_mining_bundle(
        write_bundle(tmp_path, row_overrides={"tensor_map_path": str(tmp_path / "missing.npz")})
    )

    assert result["ok"] is False
    assert "row_missing_tensor_map:0" in result["errors"]
