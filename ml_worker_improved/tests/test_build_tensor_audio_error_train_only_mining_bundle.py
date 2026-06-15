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


def write_request(tmp_path: Path, labels: list[int]) -> Path:
    request_dir = tmp_path / "request"
    rows = []
    for label in labels:
        mining_label = "hard_negative_false_accept" if label == 0 else "false_reject_positive"
        source_label = "persistent_hard_negative" if label == 0 else "persistent_false_reject"
        rows.append(
            {
                "requirement_schema": "tensor_audio_error_train_safe_mining_requirement.v1",
                "requirement_id": f"req-{label}",
                "source_slice_label": source_label,
                "target_mining_label": mining_label,
                "target_training_label": label,
                "stem_type": "minus",
                "top_error_channel_or_unknown": "unknown",
                "candidate_family_or_unknown": "htdemucs",
                "source_model_family_or_unknown": "htdemucs_ft",
                "route_family_or_unknown": "route-a",
                "locked_eval_evidence_count": 3,
                "locked_eval_split_counts": {"test": 3},
                "recommended_train_query": {"split": "train", "stem_type": "minus", "target_training_label": label},
                "must_use_split": "train",
                "forbidden_splits": ["dev", "test", "hardcase", "holdout"],
                "training_allowed": False,
                "release_ready": False,
                "shadow_only": True,
            }
        )
    write_jsonl(request_dir / "tensor_audio_error_train_safe_mining_requirements.jsonl", rows)
    write_json(
        request_dir / "tensor_audio_error_train_safe_mining_request_manifest.json",
        {
            "schema": "tensor_audio_error_train_safe_mining_request.v1",
            "requirements_total": len(rows),
            "required_train_rows_min": 4,
            "required_train_positive_rows_min": 2,
            "required_train_negative_rows_min": 2,
            "min_per_label_rows": 2,
            "min_per_stem_rows": 2,
            "min_hard_negative_rows": 2,
            "min_false_reject_rows": 2,
            "source_locked_slice_dir": str(tmp_path / "slice"),
            "locked_eval_only_source": True,
            "review_only_source": True,
            "training_rows_created": False,
            "training_allowed": False,
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_started": False,
            "critic_training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
            "shadow_only": True,
        },
    )
    return request_dir


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


def dataset_row(tmp_path: Path, index: int, *, label: int, split: str = "train", stem: str = "minus") -> dict:
    tensor_path = tmp_path / "maps" / f"row-{index}.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"npz-placeholder")
    return {
        "row_uid": f"row-{index}",
        "canonical_row_key": f"key-{index}",
        "track_id": f"track-{index}",
        "run_id": "run-a",
        "candidate_id": f"cand-{index}",
        "route_id": "route-a",
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs_ft",
        "stem_type": stem,
        "split": split,
        "label": label,
        "train_allowed": split == "train",
        "hardcase": False,
        "hardcase_holdout": False,
        "tensor_map_path": str(tensor_path),
        "tensor_map_names": ["stft_residual_magnitude"],
        "scalar_features_used": False,
        "release_ready": False,
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
    }


def write_split(tmp_path: Path) -> Path:
    split_dir = tmp_path / "split"
    write_json(split_dir / "leak_safe_split_manifest.json", {"schema": "leak_safe_audio_eval_harness.v1", "critical_leakage_count": 0, "release_ready": False})
    write_json(split_dir / "leakage_report.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(split_dir / "hardcase_holdout_manifest.json", {"locked": True, "hardcase_rows": [], "release_ready": False})
    write_jsonl(split_dir / "leak_safe_split_rows.jsonl", [])
    return split_dir


def test_train_only_bundle_builds_usable_two_class_candidates(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_train_only_mining_bundle import build_tensor_audio_error_train_only_mining_bundle

    rows = [dataset_row(tmp_path, i, label=i % 2) for i in range(6)]
    manifest = build_tensor_audio_error_train_only_mining_bundle(
        request_dir=write_request(tmp_path, [0, 1]),
        tensor_dataset_dir=write_dataset(tmp_path, rows),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "bundle",
    )

    candidates = read_jsonl(tmp_path / "bundle" / "tensor_audio_error_train_only_training_candidates.jsonl")
    assert manifest["bundle_status"] == "train_only_candidate_only"
    assert manifest["candidate_rows_total"] >= 4
    assert manifest["positive_rows"] >= 2
    assert manifest["negative_rows"] >= 2
    assert {row["split"] for row in candidates} == {"train"}
    assert {row["train_allowed"] for row in candidates} == {True}


def test_train_only_bundle_blocks_without_two_classes(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_train_only_mining_bundle import build_tensor_audio_error_train_only_mining_bundle

    rows = [dataset_row(tmp_path, i, label=1) for i in range(5)]
    manifest = build_tensor_audio_error_train_only_mining_bundle(
        request_dir=write_request(tmp_path, [1]),
        tensor_dataset_dir=write_dataset(tmp_path, rows),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "bundle",
    )

    assert manifest["bundle_status"] == "blocked_insufficient_train_safe_rows"
    assert manifest["training_allowed"] is False
    assert manifest["negative_rows"] == 0


def test_train_only_bundle_rejects_canonical_key_overlap(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_train_only_mining_bundle import build_tensor_audio_error_train_only_mining_bundle

    request_dir = write_request(tmp_path, [1])
    slice_dir = tmp_path / "slice"
    write_jsonl(
        slice_dir / "tensor_audio_error_locked_mining_slice.jsonl",
        [{"row_uid": "locked-row", "candidate_id": "locked-candidate", "track_id": "locked-track", "canonical_row_key": "blocked-key"}],
    )
    rows = [
        {**dataset_row(tmp_path, 0, label=1), "canonical_row_key": "blocked-key"},
        {**dataset_row(tmp_path, 1, label=1), "canonical_row_key": "allowed-key"},
    ]

    build_tensor_audio_error_train_only_mining_bundle(
        request_dir=request_dir,
        tensor_dataset_dir=write_dataset(tmp_path, rows),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "bundle",
    )

    candidates = read_jsonl(tmp_path / "bundle" / "tensor_audio_error_train_only_training_candidates.jsonl")
    assert {row["canonical_row_key"] for row in candidates} == {"allowed-key"}


def test_train_only_bundle_fails_closed_when_output_validation_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.build_tensor_audio_error_train_only_mining_bundle as builder

    rows = [dataset_row(tmp_path, i, label=i % 2) for i in range(6)]
    monkeypatch.setattr(
        builder,
        "validate_tensor_audio_error_train_only_mining_bundle",
        lambda _output_dir: {"ok": False, "errors": ["forced_validation_failure"]},
    )

    with pytest.raises(ValueError, match="train_only_mining_bundle_validation_failed"):
        builder.build_tensor_audio_error_train_only_mining_bundle(
            request_dir=write_request(tmp_path, [0, 1]),
            tensor_dataset_dir=write_dataset(tmp_path, rows),
            split_dir=write_split(tmp_path),
            output_dir=tmp_path / "bundle",
        )
