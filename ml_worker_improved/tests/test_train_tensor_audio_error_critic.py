import json
from pathlib import Path

import numpy as np


MAP_NAMES = (
    "stft_residual_magnitude",
    "phase_difference_map",
    "stereo_width_map",
    "vocal_bleed_proxy_map",
    "instrumental_bleed_proxy_map",
    "missing_detail_proxy_map",
    "transient_loss_proxy_map",
    "artifact_noise_proxy_map",
    "reconstruction_error_map",
)


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


def write_map(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.full((4, 6), value, dtype=np.float32) for name in MAP_NAMES}
    payload["waveform_residual"] = np.full((16, 1), value, dtype=np.float32)
    np.savez_compressed(path, **payload)


def row(tmp_path: Path, index: int, *, split: str, label: int, stem: str = "minus") -> dict:
    map_path = tmp_path / "maps" / f"row-{index}.npz"
    write_map(map_path, 0.1 + label * 0.7 + index * 0.01)
    return {
        "row_uid": f"row-{index}",
        "track_id": f"track-{index}",
        "candidate_id": f"cand-{index}",
        "stem_type": stem,
        "split": split,
        "label": label,
        "train_allowed": split == "train",
        "tensor_map_path": str(map_path),
        "tensor_map_names": list(MAP_NAMES),
        "release_ready": False,
    }


def write_tensor_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_dir = tmp_path / "tensor_dataset"
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


def mining_candidate(source_row: dict, *, split: str | None = None) -> dict:
    label = int(source_row["label"])
    return {
        "training_candidate_id": f"mine-{source_row['row_uid']}",
        "bundle_row_schema": "tensor_audio_error_training_candidate.v1",
        "source_mining_label": "hard_negative_false_accept" if label == 0 else "false_reject_positive",
        "recommended_training_label": label,
        "eligible_for_future_training_review": True,
        "training_candidate_status": "candidate_only",
        "row_uid": source_row["row_uid"],
        "canonical_row_key": source_row.get("canonical_row_key", ""),
        "track_id": source_row["track_id"],
        "run_id": source_row.get("run_id", ""),
        "candidate_id": source_row["candidate_id"],
        "route_id": source_row.get("route_id", ""),
        "candidate_family": source_row.get("candidate_family", ""),
        "source_model_family": source_row.get("source_model_family", ""),
        "stem_type": source_row["stem_type"],
        "split": split or source_row["split"],
        "window_start_sec": float(source_row.get("window_start_sec") or 0.0),
        "window_end_sec": float(source_row.get("window_end_sec") or 0.0),
        "runtime_gate_decision_applied": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "shadow_only": True,
    }


def write_mining_bundle(tmp_path: Path, rows: list[dict], *, manifest_overrides: dict | None = None) -> Path:
    bundle_dir = tmp_path / "mining_bundle"
    write_jsonl(bundle_dir / "tensor_audio_error_training_candidates.jsonl", rows)
    manifest = {
        "schema": "tensor_audio_error_mining_bundle.v1",
        "bundle_status": "candidate_only",
        "candidate_rows_total": len(rows),
        "hard_negative_rows": sum(1 for row_value in rows if row_value["source_mining_label"] == "hard_negative_false_accept"),
        "false_reject_rows": sum(1 for row_value in rows if row_value["source_mining_label"] == "false_reject_positive"),
        "training_candidates_path": str(bundle_dir / "tensor_audio_error_training_candidates.jsonl"),
        "eligible_for_future_training_review": True,
        "training_started": False,
        "training_completed": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "shadow_only": True,
    }
    manifest.update(manifest_overrides or {})
    write_json(bundle_dir / "tensor_audio_error_mining_bundle_manifest.json", manifest)
    return bundle_dir


def write_train_only_mining_bundle(tmp_path: Path, rows: list[dict], *, manifest_overrides: dict | None = None) -> Path:
    bundle_dir = tmp_path / "train_only_mining_bundle"
    train_only_rows = []
    for source_row in rows:
        candidate = mining_candidate(source_row)
        candidate["training_candidate_schema"] = "tensor_audio_error_train_only_training_candidate.v1"
        candidate["bundle_row_schema"] = "tensor_audio_error_train_only_training_candidate.v1"
        candidate["source_request_dir"] = str(tmp_path / "request")
        candidate["source_requirement_id"] = f"req-{source_row['row_uid']}"
        candidate["train_allowed"] = True
        candidate["hardcase_holdout"] = False
        candidate["locked_eval_row"] = False
        candidate["tensor_map_path"] = source_row["tensor_map_path"]
        candidate["tensor_map_names"] = source_row["tensor_map_names"]
        candidate["label_source"] = "train_safe_analog_from_locked_eval_request"
        candidate["source_true_label"] = source_row["label"]
        train_only_rows.append(candidate)
    write_jsonl(bundle_dir / "tensor_audio_error_train_only_training_candidates.jsonl", train_only_rows)
    manifest = {
        "schema": "tensor_audio_error_train_only_mining_bundle.v1",
        "bundle_status": "train_only_candidate_only",
        "candidate_rows_total": len(train_only_rows),
        "positive_rows": sum(1 for row_value in train_only_rows if row_value["recommended_training_label"] == 1),
        "negative_rows": sum(1 for row_value in train_only_rows if row_value["recommended_training_label"] == 0),
        "training_allowed": True,
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


def write_hard_negative_harvest_dir(tmp_path: Path) -> Path:
    harvest_dir = tmp_path / "hard_negative_harvest"
    write_json(
        harvest_dir / "tensor_audio_error_train_hard_negative_harvest_manifest.json",
        {
            "schema": "tensor_audio_error_train_hard_negative_harvest.v1",
            "harvest_status": "harvest_ready",
            "candidate_rows_total": 1,
            "negative_rows": 1,
            "hard_negative_rows": 1,
            "training_started": False,
            "release_ready": False,
        },
    )
    write_jsonl(harvest_dir / "tensor_audio_error_train_hard_negative_candidates.jsonl", [])
    return harvest_dir


def test_train_tensor_critic_uses_npz_maps_and_writes_localization(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
        row(tmp_path, 4, split="test", label=0),
        row(tmp_path, 5, split="test", label=1),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)

    manifest = train_tensor_audio_error_critic(dataset_dir=dataset_dir, output_dir=tmp_path / "model", min_train_rows=4)

    assert manifest["schema"] == "tensor_audio_error_critic.v1"
    assert manifest["training_completed"] is True
    assert manifest["tensor_maps_required"] is True
    assert manifest["scalar_fallback_used"] is False
    assert manifest["release_ready"] is False
    assert manifest["threshold_metrics"]["threshold_false_accept_rate"] == 0.0
    assert manifest["model_kind"] in {"logistic_regression_balanced", "random_forest_balanced"}
    assert set(manifest["candidate_model_metrics"]) == {"logistic_regression_balanced", "random_forest_balanced"}
    assert (tmp_path / "model" / "tensor_audio_error_critic.joblib").exists()
    assert read_jsonl(tmp_path / "model" / "tensor_audio_error_critic_eval.jsonl")
    localizations = read_jsonl(tmp_path / "model" / "tensor_audio_error_localization.jsonl")
    assert localizations
    assert localizations[0]["top_error_channel"] in MAP_NAMES


def test_model_selection_key_prioritizes_zero_false_accept_coverage() -> None:
    from tools.train_tensor_audio_error_critic import conservative_threshold_metrics, model_selection_key

    rows = [{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}]
    lower_coverage_threshold_metrics = conservative_threshold_metrics(rows, [0.1, 0.2, 0.05, 0.9])
    higher_coverage_threshold_metrics = conservative_threshold_metrics(rows, [0.4, 0.5, 0.6, 0.7])
    candidates = [
        ("high_auroc_low_coverage", object(), np.asarray([0.0]), {"auroc": 0.99, "accuracy": 0.99}, lower_coverage_threshold_metrics),
        ("low_auroc_high_coverage", object(), np.asarray([0.0]), {"auroc": 0.10, "accuracy": 0.10}, higher_coverage_threshold_metrics),
    ]

    selected = max(candidates, key=model_selection_key)

    assert lower_coverage_threshold_metrics["accept_coverage"] == 0.5
    assert higher_coverage_threshold_metrics["accept_coverage"] == 1.0
    assert selected[0] == "low_auroc_high_coverage"


def test_train_tensor_critic_skips_without_train_negative_class(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    dataset_dir = write_tensor_dataset(
        tmp_path,
        [
            row(tmp_path, 0, split="train", label=1),
            row(tmp_path, 1, split="train", label=1),
            row(tmp_path, 2, split="test", label=0),
        ],
    )

    manifest = train_tensor_audio_error_critic(dataset_dir=dataset_dir, output_dir=tmp_path / "model", min_train_rows=2)

    assert manifest["training_completed"] is False
    assert manifest["critic_not_usable_reason"] == "missing_train_positive_or_negative_class"
    assert manifest["scalar_fallback_used"] is False
    assert manifest["release_ready"] is False


def test_train_tensor_critic_skips_without_locked_eval_rows(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)

    manifest = train_tensor_audio_error_critic(dataset_dir=dataset_dir, output_dir=tmp_path / "model", min_train_rows=4)

    assert manifest["training_completed"] is False
    assert manifest["critic_not_usable_reason"] == "missing_locked_eval_rows"
    assert manifest["critic_training_started"] is False
    assert manifest["release_ready"] is False


def test_train_tensor_critic_applies_train_only_mining_weights(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
        row(tmp_path, 4, split="test", label=0),
        row(tmp_path, 5, split="test", label=1),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)
    mining_bundle_dir = write_mining_bundle(
        tmp_path,
        [
            mining_candidate(rows[0]),
            mining_candidate(rows[4]),
        ],
    )

    manifest = train_tensor_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "model",
        min_train_rows=4,
        mining_bundle_dir=mining_bundle_dir,
        mining_weight_multiplier=4.0,
    )

    assert manifest["training_completed"] is True
    assert manifest["mining_bundle_used"] is True
    assert manifest["mining_candidate_rows_total"] == 2
    assert manifest["mining_train_rows_available"] == 1
    assert manifest["mining_non_train_rows_ignored"] == 1
    assert manifest["mining_train_rows_matched"] == 1
    assert manifest["mining_sample_weight_applied"] is True
    assert manifest["mining_weight_multiplier"] == 4.0
    assert manifest["release_ready"] is False


def test_trainer_accepts_train_only_mining_bundle_schema(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
        row(tmp_path, 4, split="test", label=0),
        row(tmp_path, 5, split="test", label=1),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)
    mining_bundle_dir = write_train_only_mining_bundle(tmp_path, [rows[0], rows[1]])

    manifest = train_tensor_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "model",
        min_train_rows=4,
        mining_bundle_dir=mining_bundle_dir,
        mining_weight_multiplier=3.0,
    )

    assert manifest["training_completed"] is True
    assert manifest["mining_bundle_schema"] == "tensor_audio_error_train_only_mining_bundle.v1"
    assert manifest["mining_candidate_rows_total"] == 2
    assert manifest["mining_train_rows_available"] == 2
    assert manifest["mining_non_train_rows_ignored"] == 0
    assert manifest["dev_test_mining_rows_used_for_training"] is False
    assert manifest["locked_eval_rows_used_for_training"] is False
    assert manifest["hardcase_rows_used_for_training"] is False
    assert manifest["release_ready"] is False


def test_trainer_accepts_balanced_train_only_bundle(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
        row(tmp_path, 4, split="test", label=0),
        row(tmp_path, 5, split="test", label=1),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)
    mining_bundle_dir = write_train_only_mining_bundle(
        tmp_path,
        [rows[0], rows[1]],
        manifest_overrides={
            "source_hard_negative_harvest_dir": str(tmp_path / "harvest"),
            "hard_negative_rows": 1,
        },
    )

    manifest = train_tensor_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "model",
        min_train_rows=4,
        mining_bundle_dir=mining_bundle_dir,
        mining_weight_multiplier=3.0,
    )

    assert manifest["training_completed"] is True
    assert manifest["mining_bundle_schema"] == "tensor_audio_error_train_only_mining_bundle.v1"
    assert manifest["mining_non_train_rows_ignored"] == 0
    assert manifest["release_ready"] is False


def test_trainer_rejects_harvest_dir_directly(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)

    try:
        train_tensor_audio_error_critic(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "model",
            min_train_rows=4,
            mining_bundle_dir=write_hard_negative_harvest_dir(tmp_path),
        )
    except ValueError as exc:
        assert "hard_negative_harvest_not_training_bundle" in str(exc)
    else:
        raise AssertionError("hard-negative harvest dir was not rejected for training")


def test_trainer_rejects_locked_eval_mining_bundle_for_training(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)
    bundle_dir = tmp_path / "locked_slice"
    write_json(
        bundle_dir / "tensor_audio_error_locked_mining_slice_manifest.json",
        {"schema": "tensor_audio_error_locked_mining_slice.v1", "training_allowed": False, "release_ready": False},
    )

    try:
        train_tensor_audio_error_critic(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "model",
            min_train_rows=4,
            mining_bundle_dir=bundle_dir,
        )
    except ValueError as exc:
        assert "locked_eval_mining_slice_not_training_bundle" in str(exc)
    else:
        raise AssertionError("locked eval mining slice was not rejected for training")


def test_train_tensor_critic_rejects_unsafe_mining_bundle(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)
    mining_bundle_dir = write_mining_bundle(
        tmp_path,
        [mining_candidate(rows[0])],
        manifest_overrides={"release_ready": True},
    )

    try:
        train_tensor_audio_error_critic(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "model",
            min_train_rows=4,
            mining_bundle_dir=mining_bundle_dir,
        )
    except ValueError as exc:
        assert "unsafe_mining_bundle" in str(exc)
    else:
        raise AssertionError("unsafe mining bundle was not rejected")


def test_train_tensor_critic_rejects_invalid_mining_split(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)
    mining_bundle_dir = write_mining_bundle(tmp_path, [mining_candidate(rows[0], split="holdout")])

    try:
        train_tensor_audio_error_critic(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "model",
            min_train_rows=4,
            mining_bundle_dir=mining_bundle_dir,
        )
    except ValueError as exc:
        assert "invalid_mining_split" in str(exc)
    else:
        raise AssertionError("invalid mining split was not rejected")


def test_train_tensor_critic_rejects_matched_label_mismatch(tmp_path: Path) -> None:
    from tools.train_tensor_audio_error_critic import train_tensor_audio_error_critic

    rows = [
        row(tmp_path, 0, split="train", label=0),
        row(tmp_path, 1, split="train", label=1),
        row(tmp_path, 2, split="train", label=0, stem="vocal"),
        row(tmp_path, 3, split="train", label=1, stem="vocal"),
    ]
    dataset_dir = write_tensor_dataset(tmp_path, rows)
    tampered = mining_candidate(rows[1])
    tampered["source_mining_label"] = "hard_negative_false_accept"
    tampered["recommended_training_label"] = 0
    tampered["source_true_label"] = 0
    mining_bundle_dir = write_mining_bundle(tmp_path, [tampered])

    try:
        train_tensor_audio_error_critic(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "model",
            min_train_rows=4,
            mining_bundle_dir=mining_bundle_dir,
        )
    except ValueError as exc:
        assert "mining_label_mismatch_for_matched_row" in str(exc)
    else:
        raise AssertionError("matched label mismatch was not rejected")
