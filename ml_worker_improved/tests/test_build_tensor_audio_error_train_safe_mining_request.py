import json
from pathlib import Path

import pytest


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


def write_locked_slice(tmp_path: Path, rows: list[dict]) -> Path:
    slice_dir = tmp_path / "locked_slice"
    write_jsonl(slice_dir / "tensor_audio_error_locked_mining_slice.jsonl", rows)
    write_json(
        slice_dir / "tensor_audio_error_locked_mining_slice_manifest.json",
        {
            "schema": "tensor_audio_error_locked_mining_slice.v1",
            "slice_status": "candidate_only_locked_eval_review",
            "locked_eval_only": True,
            "candidate_rows_total": len(rows),
            "training_allowed": False,
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
            "shadow_only": True,
        },
    )
    write_json(
        slice_dir / "tensor_audio_error_locked_mining_slice_validation.json",
        {
            "schema": "tensor_audio_error_locked_mining_slice_validation.v1",
            "ok": True,
            "errors": [],
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    return slice_dir


def locked_row(row_uid: str, *, slice_label: str, label: int, split: str = "test") -> dict:
    return {
        "slice_row_schema": "tensor_audio_error_locked_mining_slice_row.v1",
        "slice_label": slice_label,
        "recommended_training_label": label,
        "row_uid": row_uid,
        "canonical_row_key": f"key-{row_uid}",
        "track_id": f"track-{row_uid}",
        "run_id": "run-a",
        "candidate_id": f"cand-{row_uid}",
        "route_id": "htdemucs_ft__minus",
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs_ft",
        "stem_type": "minus",
        "split": split,
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "training_allowed": False,
        "release_ready": False,
        "runtime_gates_changed": False,
        "training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "shadow_only": True,
    }


def write_split_and_dataset(tmp_path: Path) -> tuple[Path, Path]:
    split_dir = tmp_path / "split"
    dataset_dir = tmp_path / "dataset"
    write_json(
        split_dir / "leak_safe_split_manifest.json",
        {
            "schema": "leak_safe_audio_eval_harness.v1",
            "critical_leakage_count": 0,
            "future_r1_r2_r3_r4_must_use_this_split": True,
            "release_ready": False,
        },
    )
    write_json(split_dir / "leakage_report.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(
        dataset_dir / "locked_tensor_audio_error_dataset_manifest.json",
        {
            "schema": "locked_tensor_audio_error_dataset.v1",
            "rows_total": 0,
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
        },
    )
    return split_dir, dataset_dir


def test_train_safe_request_maps_slice_labels_to_target_labels(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_train_safe_mining_request import build_tensor_audio_error_train_safe_mining_request

    split_dir, dataset_dir = write_split_and_dataset(tmp_path)
    slice_dir = write_locked_slice(
        tmp_path,
        [
            locked_row("a", slice_label="persistent_false_reject", label=1, split="dev"),
            locked_row("b", slice_label="candidate_regressed_hard_negative", label=0, split="test"),
        ],
    )

    manifest = build_tensor_audio_error_train_safe_mining_request(
        locked_slice_dir=slice_dir,
        split_dir=split_dir,
        tensor_dataset_dir=dataset_dir,
        output_dir=tmp_path / "request",
    )

    rows = read_jsonl(tmp_path / "request" / "tensor_audio_error_train_safe_mining_requirements.jsonl")
    assert manifest["schema"] == "tensor_audio_error_train_safe_mining_request.v1"
    assert manifest["requirements_total"] == 2
    assert {row["target_mining_label"] for row in rows} == {"false_reject_positive", "hard_negative_false_accept"}
    assert {row["target_training_label"] for row in rows} == {0, 1}
    assert {row["must_use_split"] for row in rows} == {"train"}
    assert {row["training_allowed"] for row in rows} == {False}


def test_train_safe_request_rejects_training_allowed_locked_slice(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_train_safe_mining_request import build_tensor_audio_error_train_safe_mining_request

    split_dir, dataset_dir = write_split_and_dataset(tmp_path)
    bad_row = locked_row("a", slice_label="persistent_false_reject", label=1)
    bad_row["training_allowed"] = True
    slice_dir = write_locked_slice(tmp_path, [bad_row])

    try:
        build_tensor_audio_error_train_safe_mining_request(
            locked_slice_dir=slice_dir,
            split_dir=split_dir,
            tensor_dataset_dir=dataset_dir,
            output_dir=tmp_path / "request",
        )
    except ValueError as exc:
        assert "source_locked_slice_has_training_allowed_row" in str(exc)
    else:
        raise AssertionError("training-allowed locked slice row was not rejected")


def test_train_safe_request_fails_closed_when_output_validation_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.build_tensor_audio_error_train_safe_mining_request as builder

    split_dir, dataset_dir = write_split_and_dataset(tmp_path)
    slice_dir = write_locked_slice(tmp_path, [locked_row("a", slice_label="persistent_false_reject", label=1)])

    monkeypatch.setattr(
        builder,
        "validate_tensor_audio_error_train_safe_mining_request",
        lambda _output_dir: {"ok": False, "errors": ["forced_validation_failure"]},
    )

    with pytest.raises(ValueError, match="train_safe_mining_request_validation_failed"):
        builder.build_tensor_audio_error_train_safe_mining_request(
            locked_slice_dir=slice_dir,
            split_dir=split_dir,
            tensor_dataset_dir=dataset_dir,
            output_dir=tmp_path / "request",
        )
