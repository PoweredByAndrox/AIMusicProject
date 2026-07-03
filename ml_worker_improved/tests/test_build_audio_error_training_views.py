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


def write_listen_preview_manifest(tmp_path: Path, *, accepted: bool = True) -> Path:
    assets = write_assets(tmp_path, "listen_preview")
    source_dir = tmp_path / "listen_preview_sources"
    candidate_id = "accepted-minus-candidate"
    self_judge_path = source_dir / "instrumental_self_judge_manifest.json"
    result_judge_path = source_dir / "no_reference_result_judge.jsonl"
    selector_path = source_dir / "no_reference_candidate_selection.jsonl"
    manifest_path = source_dir / "no_reference_minus_listen_preview_manifest.json"
    write_json(
        self_judge_path,
        {
            "segments": [
                {
                    "segment_id": "segment_0000",
                    "window_sec": [0.0, 1.0],
                    "issues": [],
                    "issue_severities": {"air_loss": 0.0, "presence_loss": 0.0, "low_rms_ratio": 0.0, "vocal_projection": 0.0},
                    "metrics": {
                        "source_candidate_corr_abs": 0.95,
                        "source_candidate_delta_rms_ratio": 0.12,
                        "vocal_stem_corr_abs": 0.04,
                    },
                },
                {
                    "segment_id": "segment_0001",
                    "window_sec": [1.0, 2.0],
                    "issues": [],
                    "issue_severities": {"air_loss": 0.0, "presence_loss": 0.0, "low_rms_ratio": 0.0, "vocal_projection": 0.0},
                    "metrics": {
                        "source_candidate_corr_abs": 0.91,
                        "source_candidate_delta_rms_ratio": 0.18,
                        "vocal_stem_corr_abs": 0.03,
                    },
                },
            ]
        },
    )
    write_jsonl(
        result_judge_path,
        [
            {
                "track_id": "track-listen-preview",
                "candidate_id": candidate_id,
                "minus_decision": "accept_candidate",
                "minus_blockers": [],
                "mixture_consistency_score": 0.76,
                "instrumental_phase_consistency_score": 0.88,
                "vocal_leakage_in_minus_score": 0.21,
                "instrumental_leakage_in_vocal_score": 0.02,
                "robotic_artifact_score": 0.01,
                "silence_hallucination_score": 0.0,
                "thin_vocal_score": 0.0,
                "transient_damage_score": 0.03,
            }
        ],
    )
    write_jsonl(
        selector_path,
        [
            {
                "track_id": "track-listen-preview",
                "selected_candidate_id": candidate_id,
                "decision": "accept_candidate" if accepted else "blocked_hard_case",
            }
        ],
    )
    write_json(
        manifest_path,
        {
            "manifest_type": "no_reference_minus_listen_preview",
            "delivery_status": "accepted" if accepted else "blocked",
            "delivery_allowed": accepted,
            "presentation_status": "accepted_for_listen_pack" if accepted else "blocked",
            "release_status": "listen_preview_only",
            "stem_type": "minus",
            "score_mode": "minus",
            "track_id": "track-listen-preview",
            "candidate_id": candidate_id,
            "mix_path": assets["mix_path"],
            "vocal_path": assets["vocal_candidate_path"],
            "selected_minus_path": assets["minus_candidate_path"],
            "judge_mode": "no_reference_minus_self_judge_and_selector",
            "accepted_listen_preview_rows": [
                {
                    "track_id": "track-listen-preview",
                    "candidate_id": candidate_id,
                    "stem_type": "minus",
                    "minus_decision": "accept_candidate",
                    "minus_blockers": [],
                    "minus_musical_detail_preservation_gate": {
                        "metrics": {
                            "instrumental_integrity": 0.74,
                            "mixture_consistency_score": 0.76,
                            "phase_stereo_stability": 0.88,
                            "spectral_balance": 0.96,
                            "transient_damage_score": 0.03,
                        }
                    },
                }
            ],
            "minus_musical_detail_preservation_gate": {
                "metrics": {
                    "instrumental_integrity": 0.74,
                    "mixture_consistency_score": 0.76,
                    "phase_stereo_stability": 0.88,
                    "spectral_balance": 0.96,
                    "transient_damage_score": 0.03,
                }
            },
            "source_self_judge_manifest": {"path": str(self_judge_path), "sha256": "test"},
            "source_result_judge_jsonl": {"path": str(result_judge_path), "sha256": "test"},
            "source_selector_jsonl": {"path": str(selector_path), "sha256": "test"},
        },
    )
    return manifest_path


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


def test_overall_accept_requires_exact_pair_evidence_even_with_selected_candidate(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    source = repaired_row(
        tmp_path,
        21,
        stem_type="minus",
        accepted=True,
        pair_status="",
        unsafe_reason="",
        selected_candidate_evidence=True,
        source_selected_file="selected_candidates.jsonl",
        source_pair_file="",
    )
    write_repaired_dataset(dataset_dir, [source])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    overall_rows = read_jsonl(tmp_path / "views" / "overall_accept_training_view.jsonl")

    row = rows_by_id(overall_rows)[source["window_id"]]
    assert row["safe_for_this_view"] is False
    assert row["label"] is None
    assert "missing_pair_compatibility_status" in row["unsafe_reason_for_this_view"]
    assert "missing_exact_pair_compatibility_evidence" in row["unsafe_reason_for_this_view"]


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
    assert all(row["runtime_source_row"] is True for row in safe_rows)
    assert all(row["source_run_dir"] == str(run_dir) for row in safe_rows)
    assert all(row["source_judge_file"] == "minus_windowed_judge.jsonl" for row in safe_rows)
    assert all(row["source_candidate_family"] == "runtime_candidate" for row in safe_rows)
    assert all(row["candidate_family"] == "runtime_candidate" for row in safe_rows)
    assert [row["hardcase"] for row in safe_rows] == [False, True]
    assert all(row["hardcase_holdout"] is False for row in safe_rows)
    assert safe_rows[0]["corrective_status"] == "not_required"
    assert safe_rows[0]["corrective_action_labels"] == []
    assert safe_rows[1]["issue_labels"] == ["vocal_bleed_in_minus"]
    assert safe_rows[1]["corrective_action_labels"] == [
        "reduce_vocal_bleed_in_minus",
        "try_stronger_dedicated_minus_route",
    ]
    assert safe_rows[1]["corrective_target_lanes"] == ["minus_model", "minus_router"]
    assert safe_rows[1]["corrective_status"] == "actionable_hardcase"
    minus_summary = manifest["views"]["minus_critic"]
    assert minus_summary["actionable_hardcase_rows"] == 1
    assert minus_summary["issue_counts"] == {"vocal_bleed_in_minus": 1}
    assert minus_summary["corrective_action_counts"] == {
        "reduce_vocal_bleed_in_minus": 1,
        "try_stronger_dedicated_minus_route": 1,
    }


def test_repaired_rows_preserve_source_run_without_runtime_flag(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    source_run_dir = tmp_path / "historic_run"
    source = repaired_row(
        tmp_path,
        13,
        accepted=True,
        pair_status="",
        unsafe_reason="",
        source_run_dir=str(source_run_dir),
        source_candidate_family="historic_family",
    )
    write_repaired_dataset(dataset_dir, [source])

    build_audio_error_training_views(dataset_dir=dataset_dir, output_dir=tmp_path / "views")
    minus_rows = read_jsonl(tmp_path / "views" / "minus_critic_training_view.jsonl")

    assert len(minus_rows) == 1
    row = minus_rows[0]
    assert row["runtime_source_row"] is False
    assert row["source_run_dir"] == str(source_run_dir)
    assert row["source_candidate_family"] == "historic_family"
    assert row["candidate_family"] == "historic_family"


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
                            "minus_blockers": ["minus_detail_gate_blocked", "phase_stereo_stability_below_detail_gate"],
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
    assert safe_rows[1]["issue_labels"] == ["minus_detail_loss", "phase_stereo_damage"]
    assert safe_rows[1]["corrective_action_labels"] == [
        "preserve_minus_instrumental_detail",
        "reject_detail_damaging_minus_route",
        "repair_phase_stereo_stability",
        "reject_phase_damaging_route",
    ]
    assert safe_rows[1]["corrective_target_lanes"] == ["minus_model", "minus_detail_guard", "pair_router"]
    assert "missing_acoustic_feature_evidence" not in ";".join(row["unsafe_reason_for_this_view"] for row in minus_rows)


def test_runtime_run_roots_import_self_contained_no_reference_judge_rows(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, [])
    assets = write_assets(tmp_path, "runtime_self_contained")
    run_dir = tmp_path / "runtime_self_contained_run"
    assert not (run_dir / "candidate_pool.jsonl").exists()
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track-self-contained",
                "run_id": "run-self-contained",
                "candidate_id": "cand-self-contained",
                "minus_route_id": "route-self-contained",
                "route_family": "self_contained_runtime_candidate",
                "mix_path": assets["mix_path"],
                "vocal_path": assets["vocal_candidate_path"],
                "minus_path": assets["minus_candidate_path"],
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
    assert all(row["candidate_audio_path"] == assets["minus_candidate_path"] for row in safe_rows)
    assert all(row["mix_path"] == assets["mix_path"] for row in safe_rows)
    assert all(row["route_id"] == "route-self-contained" for row in safe_rows)
    assert all(row["source_candidate_family"] == "self_contained_runtime_candidate" for row in safe_rows)
    assert all(row["candidate_family"] == "self_contained_runtime_candidate" for row in safe_rows)
    assert "missing_audio_file:candidate_audio_path" not in ";".join(
        row["unsafe_reason_for_this_view"] for row in minus_rows
    )


def test_accepted_listen_preview_manifests_add_minus_only_positive_rows(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, [])
    listen_manifest = write_listen_preview_manifest(tmp_path, accepted=True)
    output_dir = tmp_path / "views"

    manifest = build_audio_error_training_views(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        listen_preview_manifests=[listen_manifest],
    )
    minus_rows = read_jsonl(output_dir / "minus_critic_training_view.jsonl")
    pair_rows = read_jsonl(output_dir / "pair_critic_training_view.jsonl")
    overall_rows = read_jsonl(output_dir / "overall_accept_training_view.jsonl")

    safe_rows = [row for row in minus_rows if row["safe_for_this_view"] is True]
    assert manifest["accepted_listen_preview_rows_added"] == 2
    assert manifest["source_run_rows_added"] == 0
    assert [row["label"] for row in safe_rows] == [1, 1]
    assert pair_rows == []
    assert overall_rows == []
    assert all(row["source_candidate_family"] == "accepted_listen_preview_minus" for row in safe_rows)
    assert all(row["source_judge_file"] == "no_reference_minus_windowed_judge.jsonl" for row in safe_rows)
    assert all(row["feature_values"]["acoustic_feature_evidence_present"] == 1.0 for row in safe_rows)
    assert safe_rows[0]["feature_values"]["mixture_consistency_error"] > 0.0
    assert safe_rows[0]["feature_values"]["vocal_bleed_proxy_score"] > 0.0
    assert safe_rows[0]["release_ready"] is False
    evidence_file = Path(safe_rows[0]["source_run_dir"]) / safe_rows[0]["source_judge_file"]
    evidence_rows = read_jsonl(evidence_file)
    assert len(evidence_rows) == 1
    assert len(evidence_rows[0]["window_metrics"]) == 2
    provenance = set(safe_rows[0]["provenance_source_files"])
    assert str(listen_manifest) in provenance
    assert any(path.endswith("instrumental_self_judge_manifest.json") for path in provenance)
    assert any(path.endswith("no_reference_result_judge.jsonl") for path in provenance)
    assert any(path.endswith("no_reference_candidate_selection.jsonl") for path in provenance)


def test_blocked_listen_preview_manifest_does_not_add_positive_rows(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, [])
    listen_manifest = write_listen_preview_manifest(tmp_path, accepted=False)
    output_dir = tmp_path / "views"

    manifest = build_audio_error_training_views(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        listen_preview_manifests=[listen_manifest],
    )

    assert manifest["accepted_listen_preview_rows_added"] == 0
    assert read_jsonl(output_dir / "minus_critic_training_view.jsonl") == []
    assert read_jsonl(output_dir / "pair_critic_training_view.jsonl") == []
    assert read_jsonl(output_dir / "overall_accept_training_view.jsonl") == []
    assert not (output_dir / "accepted_preview_sources").exists()


def test_ralph_manifests_add_minus_score_only_hardcase_rows(tmp_path: Path) -> None:
    from tools.build_audio_error_training_views import build_audio_error_training_views, corrective_targets_for
    from tools.train_target_audio_error_critics import load_target_rows

    corrective_targets = corrective_targets_for(
        "minus",
        [
            "insufficient_vocal_residue_improvement",
            "no_meaningful_minus_delta",
            "air_dip_damage",
            "presence_dip_damage",
            "rms_dip_damage",
        ],
    )
    assert corrective_targets["unmapped_blocker_labels"] == []
    assert corrective_targets["issue_labels"] == [
        "vocal_bleed_in_minus",
        "minus_repair_no_effect",
        "minus_detail_loss",
    ]
    assert corrective_targets["corrective_action_labels"] == [
        "reduce_vocal_bleed_in_minus",
        "try_stronger_dedicated_minus_route",
        "route_to_alternative_minus_model_candidate",
        "reject_no_delta_repair_route",
        "preserve_minus_instrumental_detail",
        "reject_detail_damaging_minus_route",
    ]

    dataset_dir = tmp_path / "dataset"
    write_repaired_dataset(dataset_dir, [])
    assets = write_assets(tmp_path, "ralph")
    rejected_minus = tmp_path / "audio" / "ralph_focus_minus.wav"
    rejected_minus.write_bytes(b"audio-placeholder")
    manifest_path = tmp_path / "ralph" / "ralph_minus_improvement_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "ralph_minus_improvement_manifest.v1",
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "selected_candidate": "hybrid_detail_safe",
            "focus_evaluation_used_for_delivery": False,
            "focus_window_source": str(tmp_path / "focus.jsonl"),
            "inputs": {
                "raw_audio_path": assets["mix_path"],
                "primary_vocals_path": assets["vocal_candidate_path"],
                "normalized_inputs": {
                    "raw_audio_path": assets["mix_path"],
                    "primary_vocals_path": assets["vocal_candidate_path"],
                },
            },
            "candidates": [
                {
                    "candidate_name": "hybrid_detail_safe",
                    "artifact_path": assets["minus_candidate_path"],
                    "evaluation": {
                        "status": "accepted",
                        "blockers": [],
                        "average_vocal_improvement_db": 0.20,
                        "peak_corr_improvement": 0.002,
                        "summary": {
                            "rms_delta_db_vs_base": -47.0,
                            "worst_rms_dip_db": -0.01,
                            "worst_presence_dip_db": -0.02,
                            "worst_air_dip_db": -0.03,
                            "max_rms_boost_db": 0.0,
                            "max_presence_boost_db": 0.0,
                            "max_air_boost_db": 0.0,
                        },
                        "focus_evaluation": {
                            "windows": [
                                {
                                    "track_id": "track-ralph",
                                    "start_sec": 100.0,
                                    "end_sec": 105.0,
                                    "vocal_residue_corr": 0.08,
                                    "vocal_projection_ratio_db": -21.0,
                                    "rms_drift_db_vs_base": -0.01,
                                    "presence_band_ratio_drift_db": -0.02,
                                    "air_band_ratio_drift_db": -0.03,
                                    "critic_decision": "ralph_residual_focus",
                                }
                            ]
                        },
                    },
                },
                {
                    "candidate_name": "focus_window_guarded",
                    "artifact_path": str(rejected_minus),
                    "evaluation": {
                        "status": "rejected",
                        "blockers": [
                            {
                                "label": "insufficient_vocal_residue_improvement",
                                "reason": "candidate_does_not_reduce_vocal_proxy_enough",
                            }
                        ],
                        "average_vocal_improvement_db": 0.0,
                        "peak_corr_improvement": 0.0,
                        "summary": {
                            "rms_delta_db_vs_base": -70.0,
                            "worst_rms_dip_db": -0.40,
                            "worst_presence_dip_db": -0.45,
                            "worst_air_dip_db": -0.20,
                            "max_rms_boost_db": 0.0,
                            "max_presence_boost_db": 0.0,
                            "max_air_boost_db": 0.0,
                        },
                        "focus_evaluation": {
                            "windows": [
                                {
                                    "track_id": "track-ralph",
                                    "start_sec": 100.0,
                                    "end_sec": 105.0,
                                    "vocal_residue_corr": 0.09,
                                    "vocal_projection_ratio_db": -20.0,
                                    "rms_drift_db_vs_base": -0.05,
                                    "presence_band_ratio_drift_db": -0.04,
                                    "air_band_ratio_drift_db": -0.02,
                                    "critic_decision": "ralph_residual_focus",
                                    "predicted_blockers": ["ralph_top_vocal_residue_window"],
                                }
                            ]
                        },
                    },
                },
            ],
        },
    )

    output_dir = tmp_path / "views"
    manifest = build_audio_error_training_views(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        ralph_manifests=[manifest_path],
    )
    minus_rows = read_jsonl(output_dir / "minus_critic_training_view.jsonl")
    strict_rows = load_target_rows(output_dir, "minus")["usable_rows"]
    score_rows = load_target_rows(output_dir, "minus", include_score_only_rows=True)["usable_rows"]

    assert manifest["ralph_manifest_rows_added"] == 2
    assert manifest["can_train_minus_critic"] is False
    assert manifest["views"]["minus_critic"]["training_safe_rows"] == 0
    assert manifest["views"]["minus_critic"]["score_only_safe_rows"] == 2
    assert manifest["views"]["minus_critic"]["training_positive_labels"] == 0
    assert manifest["views"]["minus_critic"]["training_negative_labels"] == 0
    assert "safe_rows_below_minimum" in manifest["views"]["minus_critic"]["readiness_blockers"]
    assert [row["label"] for row in minus_rows] == [0, 1]
    assert strict_rows == []
    assert len(score_rows) == 2
    assert all(row["source_judge_file"] == "no_reference_minus_windowed_judge.jsonl" for row in minus_rows)
    assert all(row["source_candidate_family"] == "ralph_minus_improvement" for row in minus_rows)
    assert all(row["safe_for_this_view"] is True for row in minus_rows)
    assert all(row["feature_values"]["acoustic_feature_evidence_present"] == 1.0 for row in minus_rows)
    assert all(row["feature_values"]["vocal_bleed_proxy_score"] > 0.0 for row in minus_rows)
    rows_by_candidate = {row["candidate_id"]: row for row in minus_rows}
    accepted_features = rows_by_candidate["hybrid_detail_safe"]["feature_values"]
    rejected_features = rows_by_candidate["focus_window_guarded"]["feature_values"]
    assert accepted_features["residual_energy"] < 0.2
    assert rejected_features["residual_energy"] == 1.0
    assert rejected_features["missing_detail_score"] == 1.0
    assert rejected_features["vocal_bleed_proxy_score"] == 1.0
    assert minus_rows[0]["blocker_labels"] == ["insufficient_vocal_residue_improvement"]
    assert minus_rows[0]["unmapped_blocker_labels"] == []
    assert minus_rows[0]["issue_labels"] == ["vocal_bleed_in_minus"]
    assert minus_rows[0]["corrective_action_labels"] == [
        "reduce_vocal_bleed_in_minus",
        "try_stronger_dedicated_minus_route",
    ]
    assert minus_rows[0]["corrective_status"] == "actionable_hardcase"
    assert minus_rows[0]["source_evidence"]["source"] == "ralph_minus_improvement_manifest"
    assert minus_rows[0]["source_evidence"]["focus_evaluation_used_for_delivery"] is False
    evidence_file = Path(minus_rows[0]["source_run_dir"]) / minus_rows[0]["source_judge_file"]
    evidence_rows = read_jsonl(evidence_file)
    assert len(evidence_rows) == 2
    assert evidence_rows[0]["window_metrics"][0]["ralph_window_evidence"]["vocal_residue_corr"] == 0.08


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
