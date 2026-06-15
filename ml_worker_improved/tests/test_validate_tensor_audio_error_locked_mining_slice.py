import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_slice(tmp_path: Path, *, manifest_overrides: dict | None = None, row_overrides: dict | None = None) -> Path:
    slice_dir = tmp_path / "slice"
    row = {
        "slice_row_schema": "tensor_audio_error_locked_mining_slice_row.v1",
        "slice_label": "candidate_regressed_false_reject",
        "recommended_training_label": 1,
        "training_allowed": False,
        "row_uid": "row-a",
        "split": "test",
        "true_label": 1,
        "baseline_shadow_decision": "shadow_accept_candidate",
        "candidate_shadow_decision": "shadow_block_candidate",
        "release_ready": False,
        "runtime_gates_changed": False,
        "training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "shadow_only": True,
    }
    row.update(row_overrides or {})
    write_jsonl(slice_dir / "tensor_audio_error_locked_mining_slice.jsonl", [row])
    manifest = {
        "schema": "tensor_audio_error_locked_mining_slice.v1",
        "slice_status": "candidate_only_locked_eval_review",
        "candidate_rows_total": 1,
        "locked_eval_only": True,
        "training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "shadow_only": True,
    }
    manifest.update(manifest_overrides or {})
    write_json(slice_dir / "tensor_audio_error_locked_mining_slice_manifest.json", manifest)
    return slice_dir


def test_validate_locked_mining_slice_accepts_safe_candidate_only_slice(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_locked_mining_slice import validate_tensor_audio_error_locked_mining_slice

    result = validate_tensor_audio_error_locked_mining_slice(write_slice(tmp_path))

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["release_ready"] is False


def test_validate_locked_mining_slice_rejects_training_and_train_rows(tmp_path: Path) -> None:
    from tools.validate_tensor_audio_error_locked_mining_slice import validate_tensor_audio_error_locked_mining_slice

    slice_dir = write_slice(
        tmp_path,
        manifest_overrides={"training_started": True},
        row_overrides={"split": "train", "training_allowed": True},
    )

    result = validate_tensor_audio_error_locked_mining_slice(slice_dir)

    assert result["ok"] is False
    assert "training_started_true" in result["errors"]
    assert "row_train_split_not_allowed:0" in result["errors"]
    assert "row_training_allowed_true:0" in result["errors"]
