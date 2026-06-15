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
        waveform_residual=np.zeros((4, 2), dtype=np.float32),
        stft_residual_magnitude=np.ones((4, 4), dtype=np.float32),
        phase_difference_map=np.full((4, 4), 0.2, dtype=np.float32),
        stereo_width_map=np.full((4, 4), 0.3, dtype=np.float32),
        vocal_bleed_proxy_map=np.full((4, 4), 0.1, dtype=np.float32),
        instrumental_bleed_proxy_map=np.full((4, 4), 0.2, dtype=np.float32),
        missing_detail_proxy_map=np.full((4, 4), 0.1, dtype=np.float32),
        transient_loss_proxy_map=np.full((4, 4), 0.05, dtype=np.float32),
        artifact_noise_proxy_map=np.full((4, 4), 0.07, dtype=np.float32),
        reconstruction_error_map=np.full((4, 4), 0.08, dtype=np.float32),
    )
    return {key: str(value) for key, value in paths.items()}


def repaired_row(
    tmp_path: Path,
    index: int,
    *,
    stem_type: str = "minus",
    accepted: bool = True,
    candidate_id: str | None = None,
    route_id: str = "route-a",
    pair_status: str = "",
    source_judge_file: str | None = None,
    unsafe_reason: str = "missing_pair_compatibility_status",
    **overrides: object,
) -> dict:
    assets = write_assets(tmp_path, f"row_{index}")
    blockers = [] if accepted else ["severe_vocal_bleed_in_minus" if stem_type == "minus" else "severe_phase_comb_filtering"]
    decision = "accept_candidate" if accepted else "blocked_hard_case"
    payload = {
        "dataset_id": "dataset-test",
        "track_id": "track-a",
        "run_id": "run-a",
        "window_id": f"track-a__cand-{index}__{stem_type}__{index:04d}",
        "window_start_sec": float(index),
        "window_end_sec": float(index + 1),
        "mix_path": assets["mix_path"],
        "vocal_candidate_path": assets["vocal_candidate_path"],
        "minus_candidate_path": assets["minus_candidate_path"],
        "candidate_id": candidate_id or f"cand-{index}",
        "route_id": route_id,
        "stem_type": stem_type,
        "blockers": blockers,
        "judge_decision": decision,
        "pair_compatibility_status": pair_status,
        "residual_energy": 0.01,
        "mixture_consistency_error": 0.02,
        "phase_collapse_score": 0.1 if accepted else 0.8,
        "stereo_width_score": 0.3,
        "vocal_bleed_proxy_score": 0.1 if accepted else 0.9,
        "instrumental_bleed_proxy_score": 0.2,
        "missing_detail_score": 0.1,
        "transient_damage_score": 0.05,
        "artifact_noise_score": 0.07,
        "error_map_paths": {"npz": assets["npz"]},
        "train_split": "train",
        "safe_for_training": unsafe_reason == "",
        "unsafe_reason": unsafe_reason,
        "source_judge_file": source_judge_file or f"{stem_type}_windowed_judge.jsonl",
        "source_pair_file": "pair_compatibility.jsonl",
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
            "safe_for_training_count_after": sum(1 for row in rows if row.get("safe_for_training") is True),
            "unsafe_count_after": sum(1 for row in rows if row.get("safe_for_training") is not True),
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
        },
    )
    write_json(
        dataset_dir / "provenance_repair_report.json",
        {
            "rows_total": len(rows),
            "safe_for_training_count_after": sum(1 for row in rows if row.get("safe_for_training") is True),
            "unsafe_count_after": sum(1 for row in rows if row.get("safe_for_training") is not True),
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
        },
    )
    write_jsonl(dataset_dir / "repaired_audio_error_window_dataset.jsonl", rows)
    write_jsonl(dataset_dir / "repaired_audio_error_maps_index.jsonl", [])


def rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row["row_id"]): row for row in rows}


def test_minus_critic_view_does_not_require_pair_compatibility_status(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    source = repaired_row(tmp_path, 1, stem_type="minus", accepted=True, pair_status="")
    write_repaired_dataset(dataset_dir, [source])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl")

    row = rows_by_id(minus_rows)[source["window_id"]]
    assert row["safe_for_this_view"] is True
    assert row["label"] == 1
    assert row["label_source"] == "minus_windowed_judge.jsonl"
    assert "missing_pair_compatibility_status" not in row["unsafe_reason_for_this_view"]
    assert row["release_ready"] is False


def test_pair_critic_view_requires_pair_compatibility_status(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    source = repaired_row(tmp_path, 2, stem_type="minus", accepted=True, pair_status="")
    write_repaired_dataset(dataset_dir, [source])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    pair_rows = read_jsonl(tmp_path / "views" / "pair_critic_training_view.jsonl")

    row = rows_by_id(pair_rows)[source["window_id"]]
    assert row["safe_for_this_view"] is False
    assert "missing_pair_compatibility_status" in row["unsafe_reason_for_this_view"]


def test_minus_labels_come_only_from_exact_minus_windowed_judge_evidence(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    accepted = repaired_row(tmp_path, 3, stem_type="minus", accepted=True, pair_status="")
    blocked = repaired_row(tmp_path, 4, stem_type="minus", accepted=False, pair_status="")
    final_status_only = repaired_row(
        tmp_path,
        5,
        stem_type="minus",
        accepted=True,
        pair_status="",
        final_runtime_status="preview_ready",
    )
    final_status_only["source_judge_file"] = ""
    write_repaired_dataset(dataset_dir, [accepted, blocked, final_status_only])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = rows_by_id(read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl"))

    assert minus_rows[accepted["window_id"]]["label"] == 1
    assert minus_rows[accepted["window_id"]]["label_source"] == "minus_windowed_judge.jsonl"
    assert minus_rows[blocked["window_id"]]["label"] == 0
    assert minus_rows[blocked["window_id"]]["blocker_labels"] == ["severe_vocal_bleed_in_minus"]
    assert minus_rows[final_status_only["window_id"]]["safe_for_this_view"] is False
    assert "missing_exact_minus_window_judge_evidence" in minus_rows[final_status_only["window_id"]]["unsafe_reason_for_this_view"]


def test_unknown_route_synthetic_candidate_and_missing_audio_are_not_safe_for_views(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    rows = [
        repaired_row(tmp_path, 6, route_id="unknown"),
        repaired_row(tmp_path, 7, candidate_id="synthetic_candidate_001"),
        repaired_row(tmp_path, 8, minus_candidate_path=str(tmp_path / "missing_minus.wav"), unsafe_reason="missing_audio_file:minus_candidate_path"),
    ]
    write_repaired_dataset(dataset_dir, rows)

    manifest = build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl")

    unsafe_reasons = {row["row_id"]: row["unsafe_reason_for_this_view"] for row in minus_rows}
    assert "route_id_unknown" in unsafe_reasons[rows[0]["window_id"]]
    assert "synthetic_candidate_id" in unsafe_reasons[rows[1]["window_id"]]
    assert "missing_audio_file:minus_candidate_path" in unsafe_reasons[rows[2]["window_id"]]
    assert manifest["views"]["minus_critic"]["safe_rows"] == 0


def test_existing_source_judge_file_must_match_candidate_and_window(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "source_run"
    row = repaired_row(tmp_path, 9, stem_type="minus", accepted=True, pair_status="", source_run_dir=str(run_dir))
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": row["track_id"],
                "candidate_id": "different-candidate",
                "window_metrics": [
                    {
                        "start_sec": row["window_start_sec"],
                        "duration_sec": row["window_end_sec"] - row["window_start_sec"],
                        "minus_decision": "accept_candidate",
                        "minus_blockers": [],
                    }
                ],
            }
        ],
    )
    write_repaired_dataset(dataset_dir, [row])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = rows_by_id(read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl"))

    assert minus_rows[row["window_id"]]["safe_for_this_view"] is False
    assert "missing_exact_minus_window_judge_evidence" in minus_rows[row["window_id"]]["unsafe_reason_for_this_view"]


def test_runtime_run_roots_add_strict_window_view_rows(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, [])
    assets = write_assets(tmp_path, "runtime")
    run_dir = tmp_path / "runtime_run"
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-runtime",
                "run_id": "run-runtime",
                "candidate_id": "cand-runtime",
                "route_id": "route-runtime",
                "candidate_family": "runtime_candidate",
                "mix_path": assets["mix_path"],
                "vocal_path": assets["vocal_candidate_path"],
                "minus_path": assets["minus_candidate_path"],
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track-runtime",
                "candidate_id": "cand-runtime",
                "window_metrics": [
                    {"start_sec": 0.0, "duration_sec": 1.0, "minus_decision": "accept_candidate", "minus_blockers": []},
                    {
                        "start_sec": 1.0,
                        "duration_sec": 1.0,
                        "minus_decision": "blocked_hard_case",
                        "minus_blockers": ["severe_vocal_bleed_in_minus"],
                    },
                ],
            }
        ],
    )

    manifest = build_audio_error_training_views(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "views",
        run_roots=[run_dir],
    )
    minus_rows = read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl")

    safe_rows = [row for row in minus_rows if row["safe_for_this_view"] is True]
    assert manifest["source_run_rows_added"] == 2
    assert [row["label"] for row in safe_rows] == [1, 0]
    assert all(row["label_source"] == "minus_windowed_judge.jsonl" for row in safe_rows)
