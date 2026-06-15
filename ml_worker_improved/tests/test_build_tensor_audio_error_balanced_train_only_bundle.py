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


def write_request(tmp_path: Path, *, min_pos: int = 2, min_neg: int = 2, min_total: int = 4) -> Path:
    request_dir = tmp_path / "request"
    write_json(
        request_dir / "tensor_audio_error_train_safe_mining_request_manifest.json",
        {
            "schema": "tensor_audio_error_train_safe_mining_request.v1",
            "requirements_total": 1,
            "required_train_rows_min": min_total,
            "required_train_positive_rows_min": min_pos,
            "required_train_negative_rows_min": min_neg,
            "release_ready": False,
            "training_allowed": False,
        },
    )
    return request_dir


def candidate(tmp_path: Path, index: int, *, label: int) -> dict:
    tensor_path = tmp_path / "maps" / f"row-{index}.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(b"npz-placeholder")
    return {
        "training_candidate_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "bundle_row_schema": "tensor_audio_error_train_only_training_candidate.v1",
        "training_candidate_id": f"pos-{index}",
        "source_requirement_id": "req-a",
        "source_request_dir": str(tmp_path / "request"),
        "row_uid": f"row-{index}",
        "canonical_row_key": f"key-{index}",
        "track_id": f"track-{index}",
        "run_id": "run-a",
        "candidate_id": f"cand-{index}",
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
        "recommended_training_label": label,
        "source_true_label": label,
        "source_mining_label": "false_reject_positive" if label == 1 else "hard_negative_false_accept",
        "label_source": "train_safe_analog_from_locked_eval_request" if label == 1 else "train_split_existing_judge_or_train_only_generation",
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


def write_positive_bundle(tmp_path: Path, rows: list[dict]) -> Path:
    bundle_dir = tmp_path / "positive_bundle"
    write_jsonl(bundle_dir / "tensor_audio_error_train_only_training_candidates.jsonl", rows)
    write_json(
        bundle_dir / "tensor_audio_error_train_only_mining_bundle_manifest.json",
        {
            "schema": "tensor_audio_error_train_only_mining_bundle.v1",
            "bundle_status": "blocked_insufficient_train_safe_rows",
            "candidate_rows_total": len(rows),
            "positive_rows": len(rows),
            "negative_rows": 0,
            "release_ready": False,
            "training_allowed": False,
        },
    )
    return bundle_dir


def harvest_row(tmp_path: Path, index: int) -> dict:
    source = candidate(tmp_path, index, label=0)
    return {
        "harvest_candidate_id": f"hard-neg-{index}",
        **{key: source[key] for key in ("row_uid", "canonical_row_key", "track_id", "run_id", "candidate_id", "route_id", "candidate_family", "source_model_family", "stem_type", "split", "train_allowed", "hardcase_holdout", "locked_eval_row", "tensor_map_path", "tensor_map_names", "window_start_sec", "window_end_sec") if key in source},
        "true_label": 0,
        "source_mining_label": "hard_negative_false_accept",
        "label_source": "train_split_existing_judge_or_train_only_generation",
        "error_reason": "artifact_noise_proxy_map",
        "blockers": ["artifact_noise_proxy_map"],
        "top_error_channel": "artifact_noise_proxy_map",
        "runtime_gate_decision_applied": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "release_ready": False,
        "shadow_only": True,
    }


def write_harvest(tmp_path: Path, rows: list[dict]) -> Path:
    harvest_dir = tmp_path / "harvest"
    write_jsonl(harvest_dir / "tensor_audio_error_train_hard_negative_candidates.jsonl", rows)
    write_jsonl(harvest_dir / "tensor_audio_error_train_hard_negative_skipped.jsonl", [])
    write_json(
        harvest_dir / "tensor_audio_error_train_hard_negative_harvest_manifest.json",
        {
            "schema": "tensor_audio_error_train_hard_negative_harvest.v1",
            "harvest_status": "harvest_ready" if rows else "blocked_no_train_negative_source",
            "candidate_rows_total": len(rows),
            "negative_rows": len(rows),
            "hard_negative_rows": len(rows),
            "release_ready": False,
            "runtime_gates_changed": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    return harvest_dir


def write_split_and_dataset(tmp_path: Path) -> tuple[Path, Path]:
    split_dir = tmp_path / "split"
    dataset_dir = tmp_path / "dataset"
    write_json(split_dir / "leak_safe_split_manifest.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(split_dir / "leakage_report.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(dataset_dir / "locked_tensor_audio_error_dataset_manifest.json", {"tensor_maps_required": True, "scalar_fallback_used": False, "critical_leakage_count": 0, "release_ready": False})
    return split_dir, dataset_dir


def test_balanced_bundle_blocks_without_negative_class(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_balanced_train_only_bundle import build_tensor_audio_error_balanced_train_only_bundle

    split_dir, dataset_dir = write_split_and_dataset(tmp_path)
    manifest = build_tensor_audio_error_balanced_train_only_bundle(
        request_dir=write_request(tmp_path),
        positive_bundle_dir=write_positive_bundle(tmp_path, [candidate(tmp_path, i, label=1) for i in range(4)]),
        hard_negative_harvest_dir=write_harvest(tmp_path, []),
        tensor_dataset_dir=dataset_dir,
        split_dir=split_dir,
        output_dir=tmp_path / "balanced",
    )

    assert manifest["bundle_status"] == "blocked_missing_negative_class"
    assert manifest["training_allowed"] is False
    assert manifest["negative_rows"] == 0


def test_balanced_bundle_accepts_train_positive_and_negative_rows(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_balanced_train_only_bundle import build_tensor_audio_error_balanced_train_only_bundle

    split_dir, dataset_dir = write_split_and_dataset(tmp_path)
    manifest = build_tensor_audio_error_balanced_train_only_bundle(
        request_dir=write_request(tmp_path),
        positive_bundle_dir=write_positive_bundle(tmp_path, [candidate(tmp_path, i, label=1) for i in range(2)]),
        hard_negative_harvest_dir=write_harvest(tmp_path, [harvest_row(tmp_path, 10), harvest_row(tmp_path, 11)]),
        tensor_dataset_dir=dataset_dir,
        split_dir=split_dir,
        output_dir=tmp_path / "balanced",
    )

    rows = read_jsonl(tmp_path / "balanced" / "tensor_audio_error_train_only_training_candidates.jsonl")
    persisted = read_json(tmp_path / "balanced" / "tensor_audio_error_train_only_mining_bundle_manifest.json")
    assert manifest == persisted
    assert manifest["bundle_status"] == "train_only_candidate_only"
    assert manifest["positive_rows"] == 2
    assert manifest["negative_rows"] == 2
    assert {row["split"] for row in rows} == {"train"}
