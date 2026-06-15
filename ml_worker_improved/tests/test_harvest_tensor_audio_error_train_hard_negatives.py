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


def write_map(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.full((4, 4), value, dtype=np.float32) for name in MAP_NAMES})


def dataset_row(
    tmp_path: Path,
    index: int,
    *,
    label: int = 0,
    split: str = "train",
    hardcase: bool = False,
    hardcase_holdout: bool = False,
) -> dict:
    tensor_path = tmp_path / "maps" / f"row-{index}.npz"
    write_map(tensor_path, 0.2 + index)
    return {
        "row_uid": f"row-{index}",
        "canonical_row_key": f"key-{index}",
        "track_id": f"track-{index}",
        "run_id": "run-a",
        "candidate_id": f"cand-{index}",
        "route_id": "route-a",
        "candidate_family": "htdemucs",
        "source_model_family": "htdemucs",
        "stem_type": "minus",
        "split": split,
        "label": label,
        "train_allowed": split == "train",
        "hardcase": hardcase,
        "hardcase_holdout": hardcase_holdout,
        "release_ready": False,
        "scalar_features_used": False,
        "tensor_map_path": str(tensor_path),
        "tensor_map_names": list(MAP_NAMES),
        "blockers": ["artifact_noise_proxy_map"],
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
    }


def write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_dir = tmp_path / "dataset"
    write_jsonl(dataset_dir / "tensor_audio_error_dataset.jsonl", rows)
    write_json(dataset_dir / "locked_tensor_audio_error_dataset_manifest.json", {"tensor_maps_required": True, "scalar_fallback_used": False, "release_ready": False})
    return dataset_dir


def write_split(tmp_path: Path) -> Path:
    split_dir = tmp_path / "split"
    write_json(split_dir / "leak_safe_split_manifest.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(split_dir / "leakage_report.json", {"critical_leakage_count": 0, "release_ready": False})
    write_json(split_dir / "hardcase_holdout_manifest.json", {"locked": True, "hardcase_rows": [], "release_ready": False})
    write_jsonl(split_dir / "leak_safe_split_rows.jsonl", [])
    return split_dir


def write_audit(tmp_path: Path, decision: str = "train_negatives_exist_but_r1i_filter_too_strict") -> Path:
    audit_dir = tmp_path / "audit"
    write_json(
        audit_dir / "tensor_audio_error_train_negative_gap_audit.json",
        {
            "schema": "tensor_audio_error_train_negative_gap_audit.v1",
            "diagnostic_decision": decision,
            "train_safe_negative_rows_total": 1,
            "release_ready": False,
        },
    )
    return audit_dir


def write_training_views(tmp_path: Path, rows: list[dict]) -> Path:
    views_dir = tmp_path / "training_views"
    write_jsonl(views_dir / "minus_critic_training_view.jsonl", rows)
    write_jsonl(views_dir / "vocal_critic_training_view.jsonl", [])
    write_json(
        views_dir / "audio_error_training_views_manifest.json",
        {
            "schema": "audio_error_training_views.v1",
            "strict_negative_rows_found": sum(1 for row in rows if row.get("label") == 0),
            "training_started": False,
            "separator_training_started": False,
            "release_ready": False,
        },
    )
    return views_dir


def source_b_view_row(
    tmp_path: Path,
    index: int,
    *,
    label: int = 0,
    safe: bool = True,
    with_map: bool = True,
    with_provenance: bool = True,
) -> dict:
    tensor_path = tmp_path / "source-b-maps" / f"row-{index}.npz"
    if with_map:
        write_map(tensor_path, 0.4 + index)
    return {
        "row_id": f"source-b-row-{index}",
        "track_id": f"source-b-track-{index}",
        "run_id": "source-b-run",
        "candidate_id": f"source-b-cand-{index}",
        "route_id": "source-b-route",
        "stem_type": "minus",
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "candidate_audio_path": str(tmp_path / "minus.wav"),
        "mix_path": str(tmp_path / "mix.wav"),
        "error_map_path": str(tensor_path) if with_map else "",
        "label": label,
        "label_source": "minus_windowed_judge.jsonl" if with_provenance else "",
        "blocker_labels": ["severe_vocal_bleed_in_minus"],
        "provenance_source_files": ["minus_windowed_judge.jsonl", "candidate_pool.jsonl"] if with_provenance else [],
        "safe_for_this_view": safe,
        "unsafe_reason_for_this_view": "" if safe else "missing_error_maps_or_scalar_features",
        "release_ready": False,
    }


def write_grounding_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    grounding_dir = tmp_path / "grounding_dataset"
    write_jsonl(grounding_dir / "audio_error_window_dataset.jsonl", rows)
    write_json(
        grounding_dir / "audio_error_grounding_manifest.json",
        {
            "schema": "audio_error_grounding_manifest.v1",
            "training_started": False,
            "separator_training_started": False,
            "release_ready": False,
        },
    )
    return grounding_dir


def source_b_grounding_row(
    tmp_path: Path,
    index: int,
    *,
    safe: bool = True,
    example_label: str = "negative_hard",
    split: str = "train",
    with_map: bool = True,
) -> dict:
    tensor_path = tmp_path / "source-b-grounding-maps" / f"row-{index}.npz"
    if with_map:
        write_map(tensor_path, 0.7 + index)
    error_map_paths = {"npz": str(tensor_path), **{name: f"{tensor_path}#{name}" for name in MAP_NAMES}} if with_map else {}
    return {
        "dataset_id": "source-b-grounding",
        "track_id": f"source-b-ground-track-{index}",
        "run_id": "source-b-ground-run",
        "window_id": f"source-b-window-{index}",
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "mix_path": str(tmp_path / "mix.wav"),
        "vocal_candidate_path": str(tmp_path / "vocal.wav"),
        "minus_candidate_path": str(tmp_path / "minus.wav"),
        "candidate_id": f"source-b-ground-cand-{index}",
        "route_id": "local_registered__htdemucs_ft__both__shared",
        "stem_type": "minus",
        "blockers": ["severe_vocal_bleed_in_minus"],
        "judge_decision": "blocked_hard_case",
        "pair_compatibility_status": "pair_compatible",
        "error_map_paths": error_map_paths,
        "train_split": split,
        "safe_for_training": safe,
        "unsafe_reason": "" if safe else "missing_route_id",
        "example_label": example_label,
        "source_candidate_family": "artifact_safe_default",
        "release_ready": False,
    }


def test_hard_negative_harvest_uses_existing_train_negative_rows(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path),
        tensor_dataset_dir=write_dataset(tmp_path, [dataset_row(tmp_path, 0, label=0), dataset_row(tmp_path, 1, label=1)]),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "harvest",
    )

    rows = [json.loads(line) for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_candidates.jsonl").read_text().splitlines()]
    assert manifest["harvest_status"] == "harvest_ready"
    assert manifest["negative_rows"] == 1
    assert manifest["source_a_existing_train_rows_used"] is True
    assert rows[0]["true_label"] == 0
    assert rows[0]["source_mining_label"] == "hard_negative_false_accept"


def test_hard_negative_harvest_blocks_without_safe_train_negative_source(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path, decision="train_negatives_absent_in_tensor_dataset"),
        tensor_dataset_dir=write_dataset(tmp_path, [dataset_row(tmp_path, 0, label=1)]),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "harvest",
    )

    assert manifest["harvest_status"] == "blocked_no_train_negative_source"
    assert manifest["negative_rows"] == 0
    assert manifest["training_started"] is False


def test_hard_negative_harvest_includes_non_holdout_hardcase_train_rows(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path),
        tensor_dataset_dir=write_dataset(
            tmp_path,
            [
                dataset_row(tmp_path, 0, label=0, hardcase=True, hardcase_holdout=False),
                dataset_row(tmp_path, 1, label=0, hardcase=True, hardcase_holdout=True),
                dataset_row(tmp_path, 2, label=1),
            ],
        ),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "harvest",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_candidates.jsonl").read_text().splitlines()
    ]
    skipped = [
        json.loads(line)
        for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_skipped.jsonl").read_text().splitlines()
    ]
    assert manifest["harvest_status"] == "harvest_ready"
    assert manifest["negative_rows"] == 1
    assert manifest["hardcase_train_rows_used"] is True
    assert manifest["hardcase_holdout_rows_used"] is False
    assert rows[0]["hardcase"] is True
    assert rows[0]["hardcase_holdout"] is False
    assert {row["skip_reason"] for row in skipped} == {"hardcase_holdout_true"}


def test_hard_negative_harvest_uses_source_b_training_view_negatives(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path, decision="train_negatives_absent_in_tensor_dataset"),
        tensor_dataset_dir=write_dataset(tmp_path, [dataset_row(tmp_path, 0, label=1)]),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "harvest",
        source_b_training_views_dir=write_training_views(tmp_path, [source_b_view_row(tmp_path, 0, label=0)]),
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_candidates.jsonl").read_text().splitlines()
    ]
    assert manifest["harvest_status"] == "harvest_ready"
    assert manifest["source_a_existing_train_rows_used"] is False
    assert manifest["source_b_candidate_generation_used"] is True
    assert manifest["source_b_training_view_rows_used"] == 1
    assert manifest["negative_rows"] == 1
    assert rows[0]["label_source"] == "source_b_training_view"
    assert rows[0]["true_label"] == 0
    assert rows[0]["hardcase"] is True
    assert rows[0]["source_mining_label"] == "hard_negative_false_accept"
    assert rows[0]["tensor_map_path"].endswith(".npz")


def test_hard_negative_harvest_source_b_skips_unsafe_or_blocked_rows(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    split_dir = write_split(tmp_path)
    write_jsonl(
        split_dir / "leak_safe_split_rows.jsonl",
        [
            {
                "split": "test",
                "track_id": "source-b-track-1",
                "candidate_id": "source-b-cand-1",
                "release_ready": False,
            }
        ],
    )
    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path, decision="train_negatives_absent_in_tensor_dataset"),
        tensor_dataset_dir=write_dataset(tmp_path, [dataset_row(tmp_path, 0, label=1)]),
        split_dir=split_dir,
        output_dir=tmp_path / "harvest",
        source_b_training_views_dir=write_training_views(
            tmp_path,
            [
                source_b_view_row(tmp_path, 0, label=0, safe=False),
                source_b_view_row(tmp_path, 1, label=0, safe=True),
                source_b_view_row(tmp_path, 2, label=0, safe=True, with_provenance=False),
            ],
        ),
    )

    skipped = [
        json.loads(line)
        for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_skipped.jsonl").read_text().splitlines()
    ]
    assert manifest["harvest_status"] == "blocked_no_train_negative_source"
    assert manifest["source_b_training_view_rows_used"] == 0
    assert manifest["source_b_candidate_generation_used"] is False
    assert {row["skip_reason"] for row in skipped if row.get("source") == "source_b_training_view"} == {
        "safe_for_this_view_not_true",
        "locked_eval_or_hardcase_key_overlap",
        "missing_judge_provenance",
    }


def test_hard_negative_harvest_source_b_view_missing_map_skips_without_crash(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path, decision="train_negatives_absent_in_tensor_dataset"),
        tensor_dataset_dir=write_dataset(tmp_path, [dataset_row(tmp_path, 0, label=1)]),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "harvest",
        source_b_training_views_dir=write_training_views(tmp_path, [source_b_view_row(tmp_path, 0, label=0, with_map=False)]),
    )

    skipped = [
        json.loads(line)
        for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_skipped.jsonl").read_text().splitlines()
    ]
    assert manifest["harvest_status"] == "blocked_no_train_negative_source"
    assert manifest["source_b_training_view_rows_used"] == 0
    assert {row["skip_reason"] for row in skipped if row.get("source") == "source_b_training_view"} == {"missing_tensor_map"}


def test_hard_negative_harvest_uses_source_b_grounding_dataset_negatives(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path, decision="train_negatives_absent_in_tensor_dataset"),
        tensor_dataset_dir=write_dataset(tmp_path, [dataset_row(tmp_path, 0, label=1)]),
        split_dir=write_split(tmp_path),
        output_dir=tmp_path / "harvest",
        source_b_grounding_dataset_dir=write_grounding_dataset(tmp_path, [source_b_grounding_row(tmp_path, 0)]),
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_candidates.jsonl").read_text().splitlines()
    ]
    assert manifest["harvest_status"] == "harvest_ready"
    assert manifest["source_a_existing_train_rows_used"] is False
    assert manifest["source_b_candidate_generation_used"] is True
    assert manifest["source_b_grounding_rows_used"] == 1
    assert manifest["negative_rows"] == 1
    assert rows[0]["label_source"] == "source_b_grounding_dataset"
    assert rows[0]["row_uid"] == "source-b-window-0"
    assert rows[0]["true_label"] == 0
    assert rows[0]["hardcase"] is True
    assert rows[0]["tensor_map_path"].endswith(".npz")


def test_hard_negative_harvest_source_b_grounding_skips_positive_unsafe_or_blocked_rows(tmp_path: Path) -> None:
    from tools.harvest_tensor_audio_error_train_hard_negatives import harvest_tensor_audio_error_train_hard_negatives

    split_dir = write_split(tmp_path)
    write_jsonl(
        split_dir / "leak_safe_split_rows.jsonl",
        [
            {
                "split": "test",
                "window_id": "source-b-window-2",
                "release_ready": False,
            }
        ],
    )
    manifest = harvest_tensor_audio_error_train_hard_negatives(
        negative_gap_audit_dir=write_audit(tmp_path, decision="train_negatives_absent_in_tensor_dataset"),
        tensor_dataset_dir=write_dataset(tmp_path, [dataset_row(tmp_path, 0, label=1)]),
        split_dir=split_dir,
        output_dir=tmp_path / "harvest",
        source_b_grounding_dataset_dir=write_grounding_dataset(
            tmp_path,
            [
                source_b_grounding_row(tmp_path, 0, safe=True, example_label="positive"),
                source_b_grounding_row(tmp_path, 1, safe=False, example_label="negative_hard"),
                source_b_grounding_row(tmp_path, 2, safe=True, example_label="negative_hard"),
                source_b_grounding_row(tmp_path, 3, safe=True, example_label="negative_hard", with_map=False),
            ],
        ),
    )

    skipped = [
        json.loads(line)
        for line in (tmp_path / "harvest" / "tensor_audio_error_train_hard_negative_skipped.jsonl").read_text().splitlines()
    ]
    assert manifest["harvest_status"] == "blocked_no_train_negative_source"
    assert manifest["source_b_grounding_rows_used"] == 0
    assert {row["skip_reason"] for row in skipped if row.get("source") == "source_b_grounding_dataset"} == {
        "example_label_not_negative_hard",
        "safe_for_training_not_true",
        "locked_eval_or_hardcase_key_overlap",
        "missing_tensor_map",
    }
