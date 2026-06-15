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
