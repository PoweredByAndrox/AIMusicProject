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


def add_window_scores(window: dict, *, accepted: bool) -> dict:
    window.update(
        {
            "residual_energy": 0.01,
            "mixture_consistency_error": 0.02,
            "phase_collapse_score": 0.1 if accepted else 0.8,
            "stereo_width_score": 0.3,
            "vocal_bleed_proxy_score": 0.1 if accepted else 0.9,
            "instrumental_bleed_proxy_score": 0.2,
            "missing_detail_score": 0.1 if accepted else 0.7,
            "transient_damage_score": 0.05,
            "artifact_noise_score": 0.07,
        }
    )
    return window


def add_no_reference_window_scores(window: dict, *, accepted: bool) -> dict:
    window.update(
        {
            "mixture_consistency_score": 0.98 if accepted else 0.62,
            "phase_consistency_score": 0.91 if accepted else 0.24,
            "instrumental_phase_consistency_score": 0.93 if accepted else 0.21,
            "vocal_leakage_in_minus_score": 0.02 if accepted else 0.74,
            "instrumental_leakage_in_vocal_score": 0.03 if accepted else 0.18,
            "robotic_artifact_score": 0.01 if accepted else 0.08,
            "silence_hallucination_score": 0.01 if accepted else 0.11,
            "thin_vocal_score": 0.02 if accepted else 0.23,
            "transient_damage_score": 0.03 if accepted else 0.15,
        }
    )
    return window


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


def test_zero_acoustic_feature_rows_are_not_safe_for_stem_views(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    row = repaired_row(tmp_path, 10, stem_type="minus", accepted=False, pair_status="")
    row["error_map_paths"] = {}
    for name in (
        "residual_energy",
        "mixture_consistency_error",
        "phase_collapse_score",
        "stereo_width_score",
        "vocal_bleed_proxy_score",
        "instrumental_bleed_proxy_score",
        "missing_detail_score",
        "transient_damage_score",
        "artifact_noise_score",
    ):
        row[name] = 0.0
    write_repaired_dataset(dataset_dir, [row])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = rows_by_id(read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl"))

    assert minus_rows[row["window_id"]]["safe_for_this_view"] is False
    assert "missing_acoustic_feature_evidence" in minus_rows[row["window_id"]]["unsafe_reason_for_this_view"]


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
                    add_window_scores(
                        {
                            "start_sec": 0.0,
                            "duration_sec": 1.0,
                            "minus_decision": "accept_candidate",
                            "minus_blockers": [],
                        },
                        accepted=True,
                    ),
                    add_window_scores(
                        {
                            "start_sec": 1.0,
                            "duration_sec": 1.0,
                            "minus_decision": "blocked_hard_case",
                            "minus_blockers": ["severe_vocal_bleed_in_minus"],
                        },
                        accepted=False,
                    ),
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


def test_runtime_run_roots_import_no_reference_scores_as_acoustic_evidence(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, [])
    assets = write_assets(tmp_path, "runtime_nr")
    run_dir = tmp_path / "runtime_nr_run"
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-runtime-nr",
                "run_id": "run-runtime-nr",
                "candidate_id": "cand-runtime-nr",
                "route_id": "route-runtime-nr",
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
                "track_id": "track-runtime-nr",
                "candidate_id": "cand-runtime-nr",
                "window_metrics": [
                    add_no_reference_window_scores(
                        {
                            "start_sec": 0.0,
                            "duration_sec": 1.0,
                            "minus_decision": "accept_candidate",
                            "minus_blockers": [],
                        },
                        accepted=True,
                    ),
                    add_no_reference_window_scores(
                        {
                            "start_sec": 1.0,
                            "duration_sec": 1.0,
                            "minus_decision": "blocked_hard_case",
                            "minus_blockers": ["minus_detail_gate_blocked"],
                        },
                        accepted=False,
                    ),
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
    assert all(row["feature_values"]["acoustic_feature_evidence_present"] == 1.0 for row in safe_rows)
    assert safe_rows[0]["feature_values"]["mixture_consistency_error"] > 0.0
    assert safe_rows[1]["feature_values"]["phase_collapse_score"] > 0.0
    assert "missing_acoustic_feature_evidence" not in ";".join(row["unsafe_reason_for_this_view"] for row in minus_rows)


def test_runtime_run_roots_normalize_mac_project_artifact_paths(tmp_path: Path, monkeypatch) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, [])
    artifact_mount = tmp_path / "host_artifacts"
    monkeypatch.setenv("AIMUSIC_ARTIFACTS_MOUNT", str(artifact_mount))
    for relative in ("legacy/mix.flac", "legacy/vocal.flac", "legacy/minus.flac"):
        path = artifact_mount / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio-placeholder")
    run_dir = tmp_path / "runtime_legacy_path_run"
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-legacy-path",
                "run_id": "run-legacy-path",
                "candidate_id": "cand-legacy-path",
                "route_id": "route-legacy-path",
                "candidate_family": "runtime_candidate",
                "mix_path": "/Users/andrejviluha/AIMusicProject_win/artifacts/legacy/mix.flac",
                "vocal_path": "/Users/andrejviluha/AIMusicProject_win/artifacts/legacy/vocal.flac",
                "minus_path": "/Users/andrejviluha/AIMusicProject_win/artifacts/legacy/minus.flac",
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track-legacy-path",
                "candidate_id": "cand-legacy-path",
                "window_metrics": [
                    add_no_reference_window_scores(
                        {
                            "start_sec": 0.0,
                            "duration_sec": 1.0,
                            "minus_decision": "accept_candidate",
                            "minus_blockers": [],
                        },
                        accepted=True,
                    ),
                ],
            }
        ],
    )

    build_audio_error_training_views(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "views",
        run_roots=[run_dir],
    )
    minus_rows = read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl")

    assert len(minus_rows) == 1
    row = minus_rows[0]
    assert row["safe_for_this_view"] is True
    assert row["mix_path"] == str(artifact_mount / "legacy/mix.flac")
    assert row["candidate_audio_path"] == str(artifact_mount / "legacy/minus.flac")


def test_repaired_rows_normalize_mac_project_artifact_paths(tmp_path: Path, monkeypatch) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    artifact_mount = tmp_path / "host_artifacts"
    monkeypatch.setenv("AIMUSIC_ARTIFACTS_MOUNT", str(artifact_mount))
    for relative in ("repaired/mix.flac", "repaired/vocal.flac", "repaired/minus.flac"):
        path = artifact_mount / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio-placeholder")
    row = repaired_row(
        tmp_path,
        11,
        stem_type="minus",
        accepted=True,
        pair_status="",
        unsafe_reason="",
        error_map_paths={},
        mix_path="/Users/andrejviluha/AIMusicProject_win/artifacts/repaired/mix.flac",
        vocal_candidate_path="/Users/andrejviluha/AIMusicProject_win/artifacts/repaired/vocal.flac",
        minus_candidate_path="/Users/andrejviluha/AIMusicProject_win/artifacts/repaired/minus.flac",
    )
    write_repaired_dataset(dataset_dir, [row])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl")
    pair_rows = read_jsonl(tmp_path / "views" / "pair_critic_training_view.jsonl")
    overall_rows = read_jsonl(tmp_path / "views" / "overall_accept_training_view.jsonl")

    assert len(minus_rows) == 1
    output = minus_rows[0]
    assert output["safe_for_this_view"] is True
    assert output["mix_path"] == str(artifact_mount / "repaired/mix.flac")
    assert output["candidate_audio_path"] == str(artifact_mount / "repaired/minus.flac")
    assert pair_rows[0]["candidate_audio_path"] == str(artifact_mount / "repaired/minus.flac")
    assert overall_rows[0]["candidate_audio_path"] == str(artifact_mount / "repaired/minus.flac")


def test_repaired_rows_normalize_mac_project_error_map_paths(tmp_path: Path, monkeypatch) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    artifact_mount = tmp_path / "host_artifacts"
    monkeypatch.setenv("AIMUSIC_ARTIFACTS_MOUNT", str(artifact_mount))
    map_path = artifact_mount / "maps" / "legacy_map.npz"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        map_path,
        stft_residual_magnitude=np.ones((4, 4), dtype=np.float32),
    )
    row = repaired_row(
        tmp_path,
        12,
        stem_type="minus",
        accepted=True,
        pair_status="",
        unsafe_reason="",
        error_map_paths={"npz": "/Users/andrejviluha/AIMusicProject_win/artifacts/maps/legacy_map.npz"},
    )
    for name in (
        "residual_energy",
        "mixture_consistency_error",
        "phase_collapse_score",
        "stereo_width_score",
        "vocal_bleed_proxy_score",
        "instrumental_bleed_proxy_score",
        "missing_detail_score",
        "transient_damage_score",
        "artifact_noise_score",
    ):
        row[name] = 0.0
    write_repaired_dataset(dataset_dir, [row])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl")

    assert len(minus_rows) == 1
    output = minus_rows[0]
    assert output["safe_for_this_view"] is True
    assert output["error_map_path"] == str(map_path)
    assert output["feature_values"]["error_map_evidence_present"] == 1.0
    assert output["feature_values"]["map_stft_residual_magnitude_mean"] == 1.0
