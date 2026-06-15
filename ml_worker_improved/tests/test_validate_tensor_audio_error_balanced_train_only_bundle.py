import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_bundle(tmp_path: Path, *, row_overrides: dict | None = None, manifest_overrides: dict | None = None) -> Path:
    bundle_dir = tmp_path / "bundle"
    tensor_path = tmp_path / "maps" / "row.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"npz-placeholder")
    row = {
        "training_candidate_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "bundle_row_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "training_candidate_id": "row-a",
        "source_requirement_id": "req-a",
        "source_request_dir": str(tmp_path / "request"),
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
        "tensor_map_path": str(tensor_path),
        "tensor_map_names": ["stft_residual_magnitude"],
        "recommended_training_label": 0,
        "source_true_label": 0,
        "source_mining_label": "hard_negative_false_accept",
        "label_source": "train_split_existing_judge_or_train_only_generation",
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
        "bundle_status": "blocked_missing_positive_class",
        "candidate_rows_total": 1,
        "positive_rows": 0,
        "negative_rows": 1,
        "hard_negative_rows": 1,
        "min_train_rows_total": 40,
        "min_train_positive_rows": 10,
        "min_train_negative_rows": 10,
        "training_allowed": False,
        "tensor_maps_required": True,
        "scalar_features_used": False,
        "dev_test_mining_rows_used_for_training": False,
        "locked_eval_rows_used_for_training": False,
        "hardcase_rows_used_for_training": False,
        "release_ready": False,
        "runtime_gates_changed": False,
        "training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "shadow_only": True,
    }
    manifest.update(manifest_overrides or {})
    write_json(bundle_dir / "tensor_audio_error_train_only_mining_bundle_manifest.json", manifest)
    return bundle_dir


def test_balanced_bundle_validator_rejects_dev_test_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_balanced_train_only_bundle import validate_tensor_audio_error_balanced_train_only_bundle

    result = validate_tensor_audio_error_balanced_train_only_bundle(write_bundle(tmp_path, row_overrides={"split": "dev"}))

    assert result["ok"] is False
    assert "row_split_not_train:0" in result["errors"]


def test_balanced_bundle_validator_rejects_usable_without_negative_threshold(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_balanced_train_only_bundle import validate_tensor_audio_error_balanced_train_only_bundle

    result = validate_tensor_audio_error_balanced_train_only_bundle(
        write_bundle(
            tmp_path,
            manifest_overrides={
                "bundle_status": "train_only_candidate_only",
                "training_allowed": True,
                "positive_rows": 10,
                "negative_rows": 1,
                "candidate_rows_total": 11,
            },
        )
    )

    assert result["ok"] is False
    assert "usable_bundle_negative_rows_below_minimum" in result["errors"]
