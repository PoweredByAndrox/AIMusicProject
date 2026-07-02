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


def test_builder_extracts_ralph_candidate_outcomes_without_delivery_claim(tmp_path: Path) -> None:
    from tools.build_minus_repair_outcome_training_view import build_minus_repair_outcome_training_view

    manifest_path = tmp_path / "ralph" / "ralph_minus_improvement_manifest.json"
    write_json(
        manifest_path,
        {
            "mode": "ralph_minus_improvement_loop",
            "focus_window_source": "/host_artifacts/loop/tcrit/scored_minus_critic_windows.jsonl",
            "focus_window_filters": {
                "track_id": "track-a",
                "candidate_id": "base-candidate",
            },
            "focus_windows": [{"start_sec": 0.0, "end_sec": 6.0}],
            "focus_evaluation_used_for_delivery": False,
            "candidates": [
                {
                    "candidate_name": "focus_window_guarded",
                    "artifact_path": "/host_artifacts/loop/it075/ralph/r006/candidates/focus/minus.flac",
                    "evaluation": {
                        "status": "rejected",
                        "score": -0.55654,
                        "average_vocal_improvement_db": 0.011457,
                        "peak_corr_improvement": -0.000359,
                        "blockers": [
                            {"label": "no_meaningful_minus_delta"},
                            {"label": "rms_dip_damage"},
                        ],
                        "summary": {
                            "worst_rms_dip_db": -0.194789,
                            "max_rms_boost_db": 0.0,
                            "worst_presence_dip_db": -0.083879,
                            "max_presence_boost_db": 0.0,
                            "worst_air_dip_db": -0.000848,
                            "max_air_boost_db": 0.0,
                        },
                        "focus_evaluation": {
                            "average_vocal_improvement_db": 0.11137,
                            "peak_corr_improvement": -0.000486,
                            "worst_rms_dip_db": -0.196925,
                        },
                    },
                }
            ],
        },
    )

    result = build_minus_repair_outcome_training_view(
        output_dir=tmp_path / "out",
        ralph_manifests=[manifest_path],
    )

    rows = read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert result["schema"] == "minus_repair_outcome_training_view.v1"
    assert result["delivery_allowed"] is False
    assert result["can_publish_as_result"] is False
    assert result["summary"]["rows_total"] == 1
    assert result["summary"]["rejected_rows"] == 1
    row = rows[0]
    assert row["source_kind"] == "ralph_manifest_candidate"
    assert row["track_id"] == "track-a"
    assert row["source_candidate_id"] == "base-candidate"
    assert row["repair_candidate_id"] == "focus_window_guarded"
    assert row["label"] == 0
    assert row["safe_for_repair_outcome_training"] is True
    assert row["has_detail_damage_blocker"] is True
    assert row["has_leakage_blocker"] is True
    assert row["feature_values"]["focus_average_vocal_improvement_db"] == 0.11137
    assert row["focus_evaluation_used_for_delivery"] is False
    assert row["delivery_allowed"] is False


def test_builder_extracts_matrix_rows_and_route_family(tmp_path: Path) -> None:
    from tools.build_minus_repair_outcome_training_view import build_minus_repair_outcome_training_view

    rows_path = tmp_path / "loop" / "it075" / "hnorm" / "r001" / "rows.jsonl"
    write_jsonl(
        rows_path,
        [
            {
                "path": "/host_artifacts/loop/it071/dmin/r002/run/instrumental.flac",
                "scale": 0.99999994,
                "status": "rejected",
                "score": -0.119375,
                "blockers": ["no_meaningful_minus_delta", "insufficient_vocal_residue_improvement"],
                "avg_improve": -0.001949,
                "peak_improve": -0.000461,
                "worst_rms": -0.036545,
                "max_rms": 0.000003,
                "worst_presence": -0.008829,
                "max_presence": 0.000021,
                "worst_air": -0.007573,
                "max_air": 0.000074,
                "focus_improve": -0.015268,
                "focus_peak": -0.000304,
                "focus_rms": -0.026383,
            }
        ],
    )

    build_minus_repair_outcome_training_view(output_dir=tmp_path / "out", matrix_rows=[rows_path])

    manifest = read_json(tmp_path / "out" / "manifest.json")
    rows = read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert manifest["summary"]["route_family_counts"] == {
        "level_normalized_historical_dedicated_minus_matrix": 1
    }
    row = rows[0]
    assert row["source_kind"] == "matrix_row"
    assert row["route_family"] == "level_normalized_historical_dedicated_minus_matrix"
    assert row["parameters"] == {"scale": 0.99999994}
    assert row["candidate_audio_path"].endswith("/instrumental.flac")
    assert row["feature_values"]["average_vocal_improvement_db"] == -0.001949
    assert row["has_leakage_blocker"] is True
    assert row["safe_for_repair_outcome_training"] is True


def test_builder_extracts_accepted_listen_preview_repair_outcome(tmp_path: Path) -> None:
    from tools.build_minus_repair_outcome_training_view import build_minus_repair_outcome_training_view

    manifest_path = tmp_path / "deliv" / "no_reference_minus_listen_preview_manifest.json"
    write_json(
        manifest_path,
        {
            "manifest_type": "no_reference_minus_listen_preview",
            "candidate_id": "base-minus-air-detail-r003",
            "delivery_allowed": True,
            "delivery_status": "accepted",
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "source_self_judge_manifest": {"path": "/host_artifacts/loop/it076/self/a003/manifest.json"},
            "source_result_judge_jsonl": {"path": "/host_artifacts/loop/it076/deliv/r001/no_reference_result_judge/no_reference_result_judge.jsonl"},
            "source_selector_jsonl": {"path": "/host_artifacts/loop/it076/deliv/r001/no_reference_candidate_selector/no_reference_candidate_selection.jsonl"},
            "accepted_listen_preview_rows": [
                {
                    "track_id": "raw_audio",
                    "candidate_id": "base-minus-air-detail-r003",
                    "minus_decision": "accept_candidate",
                    "minus_path": "/host_artifacts/loop/it076/adr/r003/m001.flac",
                    "minus_score": 0.848816,
                    "overall_no_reference_score": 0.843633,
                    "confidence": 0.827996,
                    "window_count": 3197,
                    "minus_blockers": [],
                    "minus_musical_detail_preservation_gate": {
                        "status": "proxy_pass",
                        "blockers": [],
                        "metrics": {
                            "instrumental_integrity": 0.747069,
                            "transient_preservation": 1.0,
                            "spectral_balance": 1.0,
                            "phase_stereo_stability": 0.901177,
                            "mixture_consistency_score": 0.747069,
                        },
                    },
                }
            ],
        },
    )

    build_minus_repair_outcome_training_view(
        output_dir=tmp_path / "out",
        listen_preview_manifests=[manifest_path],
    )

    manifest = read_json(tmp_path / "out" / "manifest.json")
    rows = read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert manifest["delivery_allowed"] is False
    assert manifest["audio_written"] is False
    assert manifest["listen_preview_manifests"] == [str(manifest_path)]
    assert manifest["summary"]["accepted_rows"] == 1
    assert manifest["summary"]["can_train_repair_outcome_model"] is False
    assert "insufficient_positive_repair_rows" in manifest["summary"]["readiness_blockers"]
    row = rows[0]
    assert row["source_kind"] == "listen_preview_manifest_row"
    assert row["route_family"] == "accepted_listen_preview_repair_chain"
    assert row["label"] == 1
    assert row["safe_for_repair_outcome_training"] is True
    assert row["accepted_for_repair_gate"] is True
    assert row["candidate_audio_path"].endswith("/adr/r003/m001.flac")
    assert row["listen_preview_delivery_allowed"] is True
    assert row["listen_preview_presentation_status"] == "accepted_for_listen_pack"
    assert row["feature_values"]["score"] == 0.848816
    assert row["feature_values"]["detail_gate_instrumental_integrity"] == 0.747069
    assert row["delivery_allowed"] is False


def test_blocked_listen_preview_manifest_is_not_positive_label(tmp_path: Path) -> None:
    from tools.build_minus_repair_outcome_training_view import build_minus_repair_outcome_training_view

    manifest_path = tmp_path / "deliv" / "blocked_preview.json"
    write_json(
        manifest_path,
        {
            "manifest_type": "no_reference_minus_listen_preview",
            "candidate_id": "candidate-a",
            "delivery_allowed": False,
            "delivery_status": "blocked",
            "presentation_status": "diagnostic_self_judge_not_for_listen_pack",
            "accepted_listen_preview_rows": [
                {
                    "track_id": "track-a",
                    "candidate_id": "candidate-a",
                    "minus_decision": "accept_candidate",
                    "minus_path": "/host_artifacts/loop/it076/adr/r003/m001.flac",
                    "minus_score": 0.7,
                    "minus_musical_detail_preservation_gate": {
                        "status": "proxy_pass",
                        "blockers": [],
                        "metrics": {"instrumental_integrity": 0.8},
                    },
                }
            ],
        },
    )

    build_minus_repair_outcome_training_view(
        output_dir=tmp_path / "out",
        listen_preview_manifests=[manifest_path],
    )

    manifest = read_json(tmp_path / "out" / "manifest.json")
    rows = read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert manifest["summary"]["accepted_rows"] == 0
    assert manifest["summary"]["rejected_rows"] == 1
    assert rows[0]["label"] == 0
    assert rows[0]["accepted_for_repair_gate"] is False
    assert rows[0]["listen_preview_delivery_allowed"] is False


def test_builder_maps_known_matrix_families(tmp_path: Path) -> None:
    from tools.build_minus_repair_outcome_training_view import route_family_from_matrix_path

    assert route_family_from_matrix_path(tmp_path / "loop" / "it075" / "fswp" / "r001" / "rows.jsonl") == "focus_strength_sweep"
    assert route_family_from_matrix_path(tmp_path / "loop" / "it075" / "gblend" / "r001" / "rows.jsonl") == "guide_blend_sweep"
    assert route_family_from_matrix_path(tmp_path / "loop" / "it075" / "hist" / "r002" / "rows.jsonl") == "historical_dedicated_minus_matrix"
    assert route_family_from_matrix_path(tmp_path / "loop" / "it075" / "hnorm" / "r001" / "rows.jsonl") == "level_normalized_historical_dedicated_minus_matrix"


def test_routing_only_status_is_not_positive_training_label(tmp_path: Path) -> None:
    from tools.build_minus_repair_outcome_training_view import build_minus_repair_outcome_training_view

    rows_path = tmp_path / "loop" / "it075" / "delta" / "r001" / "rows.jsonl"
    write_jsonl(
        rows_path,
        [
            {
                "id": "delta-pass",
                "status": "passed_for_candidate_routing",
                "score": 0.9,
                "blockers": [],
                "avg_improve": 0.5,
            }
        ],
    )

    build_minus_repair_outcome_training_view(output_dir=tmp_path / "out", matrix_rows=[rows_path])

    rows = read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert rows[0]["label"] is None
    assert rows[0]["safe_for_repair_outcome_training"] is False
    assert "routing_only_status_not_repair_outcome_label" in rows[0]["unsafe_reason"]


def test_builder_marks_rows_without_status_or_metrics_unsafe(tmp_path: Path) -> None:
    from tools.build_minus_repair_outcome_training_view import build_minus_repair_outcome_training_view

    rows_path = tmp_path / "loop" / "it075" / "fswp" / "r001" / "rows.jsonl"
    write_jsonl(rows_path, [{"id": "s0025", "blockers": []}])

    build_minus_repair_outcome_training_view(output_dir=tmp_path / "out", matrix_rows=[rows_path])

    manifest = read_json(tmp_path / "out" / "manifest.json")
    rows = read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert manifest["summary"]["safe_rows"] == 0
    assert rows[0]["label"] is None
    assert rows[0]["safe_for_repair_outcome_training"] is False
    assert "missing_repair_outcome_label" in rows[0]["unsafe_reason"]
    assert "missing_repair_metric_evidence" in rows[0]["unsafe_reason"]
