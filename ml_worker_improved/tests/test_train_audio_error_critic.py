import json
from pathlib import Path

import numpy as np


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_assets(tmp_path: Path, name: str) -> dict[str, str]:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mix_path": audio_dir / f"{name}_mix.wav",
        "vocal_candidate_path": audio_dir / f"{name}_vocal.wav",
        "minus_candidate_path": audio_dir / f"{name}_minus.wav",
        "npz": tmp_path / "maps" / f"{name}.npz",
    }
    for key in ("mix_path", "vocal_candidate_path", "minus_candidate_path"):
        paths[key].write_bytes(b"audio-placeholder")
    paths["npz"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["npz"],
        waveform_residual=np.zeros((8, 2), dtype=np.float32),
        stft_residual_magnitude=np.ones((6, 6), dtype=np.float32),
        phase_difference_map=np.full((6, 6), 0.2, dtype=np.float32),
        stereo_width_map=np.full((6, 6), 0.3, dtype=np.float32),
        vocal_bleed_proxy_map=np.full((6, 6), 0.1, dtype=np.float32),
        instrumental_bleed_proxy_map=np.full((6, 6), 0.4, dtype=np.float32),
        missing_detail_proxy_map=np.full((6, 6), 0.2, dtype=np.float32),
        transient_loss_proxy_map=np.full((6, 6), 0.05, dtype=np.float32),
        artifact_noise_proxy_map=np.full((6, 6), 0.07, dtype=np.float32),
        reconstruction_error_map=np.full((6, 6), 0.08, dtype=np.float32),
    )
    return {key: str(value) for key, value in paths.items()}


def row(tmp_path: Path, index: int, *, accepted: bool, track_id: str, candidate_id: str | None = None, **overrides: object) -> dict:
    assets = write_assets(tmp_path, f"row_{index}")
    blockers = [] if accepted else ["severe_vocal_bleed_in_minus"]
    payload = {
        "dataset_id": "dataset-test",
        "track_id": track_id,
        "run_id": f"run-{track_id}",
        "window_id": f"w-{index}",
        "window_start_sec": float(index),
        "window_end_sec": float(index + 1),
        "mix_path": assets["mix_path"],
        "vocal_candidate_path": assets["vocal_candidate_path"],
        "minus_candidate_path": assets["minus_candidate_path"],
        "candidate_id": candidate_id or f"cand-{track_id}",
        "route_id": f"route-{track_id}",
        "stem_type": "minus" if index % 2 else "vocal",
        "blockers": blockers,
        "judge_decision": "accept_candidate" if accepted else "blocked_hard_case",
        "pair_compatibility_status": "pair_compatible" if accepted else "pair_incompatible",
        "residual_energy": 0.01 + index * 0.001,
        "mixture_consistency_error": 0.02 + index * 0.001,
        "phase_collapse_score": 0.1 if accepted else 0.8,
        "stereo_width_score": 0.3,
        "vocal_bleed_proxy_score": 0.1 if accepted else 0.9,
        "instrumental_bleed_proxy_score": 0.2,
        "missing_detail_score": 0.1,
        "transient_damage_score": 0.05,
        "artifact_noise_score": 0.07,
        "error_map_paths": {"npz": assets["npz"]},
        "train_split": "train",
        "safe_for_training": True,
        "unsafe_reason": "",
        "release_ready": False,
        "training_started": False,
        "separator_training_started": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }
    payload.update(overrides)
    return payload


def write_repaired_dataset(dataset_dir: Path, rows: list[dict]) -> None:
    write_json(
        dataset_dir / "repaired_audio_error_grounding_manifest.json",
        {
            "dataset_id": "dataset-test",
            "rows_total": len(rows),
            "safe_for_training_count_after": sum(1 for item in rows if item.get("safe_for_training") is True),
            "unsafe_count_after": sum(1 for item in rows if item.get("safe_for_training") is not True),
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
        },
    )
    write_json(
        dataset_dir / "provenance_repair_report.json",
        {
            "rows_total": len(rows),
            "safe_for_training_count_after": sum(1 for item in rows if item.get("safe_for_training") is True),
            "unsafe_count_after": sum(1 for item in rows if item.get("safe_for_training") is not True),
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
        },
    )
    write_jsonl(dataset_dir / "repaired_audio_error_window_dataset.jsonl", rows)
    write_jsonl(dataset_dir / "repaired_audio_error_maps_index.jsonl", [])


def trainable_rows(tmp_path: Path) -> list[dict]:
    rows: list[dict] = []
    for index, track_id in enumerate(["track-a", "track-b", "track-c", "track-d", "track-e", "track-f"]):
        rows.append(row(tmp_path, index, accepted=index % 2 == 0, track_id=track_id))
    return rows


def test_trainer_excludes_unsafe_unknown_route_and_synthetic_rows(tmp_path: Path) -> None:
    from tools.train_audio_error_critic import train_audio_error_critic

    rows = trainable_rows(tmp_path)
    rows.extend(
        [
            row(tmp_path, 20, accepted=True, track_id="track-unsafe", safe_for_training=False, unsafe_reason="manual_unsafe"),
            row(tmp_path, 21, accepted=True, track_id="track-unknown", route_id="unknown"),
            row(tmp_path, 22, accepted=True, track_id="track-synth", candidate_id="synthetic_cand_001"),
            row(tmp_path, 23, accepted=True, track_id="track-missing-route", route_id=""),
        ]
    )
    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, rows)

    manifest = train_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "model",
        min_safe_rows=4,
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
    )

    assert manifest["unsafe_rows_used"] is False
    assert manifest["rows_total"] == 10
    assert manifest["rows_used_for_training"] == 6
    assert manifest["rows_excluded"] == 4
    assert manifest["excluded_reason_counts"]["safe_for_training_false"] == 1
    assert manifest["excluded_reason_counts"]["route_id_unknown"] == 1
    assert manifest["excluded_reason_counts"]["synthetic_candidate_id"] == 1
    assert manifest["excluded_reason_counts"]["missing_route_id"] == 1
    assert (tmp_path / "model" / "audio_error_critic_model.joblib").exists()
    assert manifest["release_ready"] is False
    assert manifest["separator_training_started"] is False


def test_trainer_uses_group_aware_split_and_marks_small_dataset_low_confidence(tmp_path: Path) -> None:
    from tools.train_audio_error_critic import train_audio_error_critic

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, trainable_rows(tmp_path))

    manifest = train_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "model",
        min_safe_rows=4,
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
    )

    eval_rows = read_jsonl(tmp_path / "model" / "audio_error_critic_eval.jsonl")
    split_by_candidate: dict[str, set[str]] = {}
    for item in eval_rows:
        split_by_candidate.setdefault(item["candidate_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in split_by_candidate.values())
    assert manifest["group_split_strategy"] == "track_or_candidate"
    assert manifest["train_rows"] > 0
    assert manifest["test_rows"] > 0
    assert manifest["model_quality_confidence"] == "low"
    assert manifest["critic_usable_for_shadow_scoring"] is True


def test_trainer_skips_classifier_when_negative_class_is_missing(tmp_path: Path) -> None:
    from tools.train_audio_error_critic import train_audio_error_critic

    dataset_dir = tmp_path / "dataset"
    rows = [row(tmp_path, index, accepted=True, track_id=f"track-{index}") for index in range(5)]
    write_repaired_dataset(dataset_dir, rows)

    manifest = train_audio_error_critic(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "model",
        min_safe_rows=4,
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
    )

    assert manifest["decision"] == "critic_training_skipped_insufficient_data"
    assert manifest["critic_usable_for_shadow_scoring"] is False
    assert manifest["critic_not_usable_reason"] == "missing_positive_or_negative_class"
    assert manifest["rows_used_for_training"] == 0
    assert not (tmp_path / "model" / "audio_error_critic_model.joblib").exists()
    assert manifest["unsafe_rows_used"] is False
