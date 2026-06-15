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
