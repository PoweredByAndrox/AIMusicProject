import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def row(tmp_path: Path, index: int, *, split: str = "train", label: int = 1, train_allowed: bool = True) -> dict:
    tensor_path = tmp_path / "maps" / f"row-{index}.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"npz-placeholder")
    return {
        "row_uid": f"row-{index}",
        "canonical_row_key": f"key-{index}",
        "track_id": f"track-{index}",
        "run_id": "run-a",
        "candidate_id": f"cand-{index}",
        "route_id": "local_registered__htdemucs_ft__both__shared",
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs",
        "stem_type": "minus",
        "split": split,
        "label": label,
        "train_allowed": train_allowed,
        "hardcase": False,
        "hardcase_holdout": False,
        "release_ready": False,
        "scalar_features_used": False,
        "tensor_map_path": str(tensor_path),
        "tensor_map_names": ["stft_residual_magnitude"],
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
    }


def write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_dir = tmp_path / "dataset"
    write_jsonl(dataset_dir / "tensor_audio_error_dataset.jsonl", rows)
    write_json(
        dataset_dir / "locked_tensor_audio_error_dataset_manifest.json",
        {
            "schema": "locked_tensor_audio_error_dataset.v1",
            "rows_total": len(rows),
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critical_leakage_count": 0,
            "release_ready": False,
        },
    )
    return dataset_dir


def write_split(tmp_path: Path) -> Path:
    split_dir = tmp_path / "split"
    write_json(split_dir / "leak_safe_split_manifest.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(split_dir / "leakage_report.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(split_dir / "hardcase_holdout_manifest.json", {"locked": True, "hardcase_rows": [], "release_ready": False})
    write_jsonl(split_dir / "leak_safe_split_rows.jsonl", [])
    return split_dir


def write_request(tmp_path: Path) -> Path:
    request_dir = tmp_path / "request"
    write_json(
        request_dir / "tensor_audio_error_train_safe_mining_request_manifest.json",
        {
            "schema": "tensor_audio_error_train_safe_mining_request.v1",
            "requirements_total": 1,
            "release_ready": False,
            "training_allowed": False,
        },
    )
    write_jsonl(
        request_dir / "tensor_audio_error_train_safe_mining_requirements.jsonl",
        [
            {
                "requirement_schema": "tensor_audio_error_train_safe_mining_requirement.v1",
                "target_mining_label": "false_reject_positive",
                "target_training_label": 1,
                "stem_type": "minus",
                "must_use_split": "train",
                "training_allowed": False,
                "release_ready": False,
                "shadow_only": True,
            }
        ],
    )
    return request_dir


def write_r1i_bundle(tmp_path: Path, *, positive_rows: int, negative_rows: int) -> Path:
    bundle_dir = tmp_path / "r1i_bundle"
    write_json(
        bundle_dir / "tensor_audio_error_train_only_mining_bundle_manifest.json",
        {
            "schema": "tensor_audio_error_train_only_mining_bundle.v1",
            "bundle_status": "blocked_insufficient_train_safe_rows",
            "candidate_rows_total": positive_rows + negative_rows,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "hard_negative_rows": negative_rows,
            "false_reject_rows": positive_rows,
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_allowed": False,
        },
    )
    write_jsonl(bundle_dir / "tensor_audio_error_train_only_training_candidates.jsonl", [])
    return bundle_dir


def test_negative_gap_audit_detects_absent_train_negatives(tmp_path: Path) -> None:
    from tools.audit_tensor_audio_error_train_negative_gap import audit_tensor_audio_error_train_negative_gap

    manifest = audit_tensor_audio_error_train_negative_gap(
        tensor_dataset_dir=write_dataset(tmp_path, [row(tmp_path, 0, label=1), row(tmp_path, 1, label=1)]),
        split_dir=write_split(tmp_path),
        request_dir=write_request(tmp_path),
        bundle_dir=write_r1i_bundle(tmp_path, positive_rows=2, negative_rows=0),
        output_dir=tmp_path / "audit",
    )

    assert manifest["diagnostic_decision"] == "train_negatives_absent_in_tensor_dataset"
    assert manifest["train_negative_rows_total"] == 0
    assert manifest["release_ready"] is False


def test_negative_gap_audit_detects_filter_too_strict(tmp_path: Path) -> None:
    from tools.audit_tensor_audio_error_train_negative_gap import audit_tensor_audio_error_train_negative_gap

    manifest = audit_tensor_audio_error_train_negative_gap(
        tensor_dataset_dir=write_dataset(tmp_path, [row(tmp_path, 0, label=0), row(tmp_path, 1, label=1)]),
        split_dir=write_split(tmp_path),
        request_dir=write_request(tmp_path),
        bundle_dir=write_r1i_bundle(tmp_path, positive_rows=1, negative_rows=0),
        output_dir=tmp_path / "audit",
    )

    assert manifest["diagnostic_decision"] == "train_negatives_exist_but_r1i_filter_too_strict"
    assert manifest["train_negative_rows_total"] == 1
    assert manifest["train_safe_negative_rows_total"] == 1
