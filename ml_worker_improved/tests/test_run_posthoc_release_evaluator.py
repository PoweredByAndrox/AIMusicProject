from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools import run_posthoc_release_evaluator as evaluator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_vocal_manifest_rebuilds_release_gate_from_issue_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "vocal" / "adaptive_vocal_manifest.json"
    _write_json(
        manifest_path,
        {
            "mode": "adaptive_vocal_isolation_controller",
            "selected_profile": "candidate_16_bs_artifact_clean",
            "selected_score": 1.23,
            "selected_segment_issue_report": {
                "summary": {
                    "max_issue_score": 1.1,
                    "mean_top3_issue_score": 0.5,
                    "issue_window_ratio": 0.2,
                },
                "segments": [
                    {
                        "start_sec": 25.0,
                        "end_sec": 30.0,
                        "severity_score": 1.1,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "metrics": {"presence_delta_db": -8.0},
                    }
                ],
            },
            "detected_issue_action_contract": {
                "status": "pass",
                "segments": [
                    {
                        "start_sec": 25.0,
                        "end_sec": 30.0,
                        "severity_score": 1.1,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "actions": [
                            {
                                "label": "high_noise_or_robotic_artifact",
                                "status": "action_available",
                                "candidate_profiles": ["vocal_harshness_guard"],
                            }
                        ],
                        "rejections": [],
                        "uncovered_labels": [],
                    }
                ],
                "summary": {"uncovered_labels": []},
            },
            "candidates": [
                {
                    "profile_name": "candidate_16_bs_artifact_clean",
                    "metrics": {"segment_issue_max_score": 1.1},
                }
            ],
            "selected_artifacts": {
                "selected_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
            },
        },
    )

    item = evaluator.evaluate_manifest_path(manifest_path)

    assert item["kind"] == "vocal"
    assert item["release_status"] == "diagnostic_only"
    assert item["gate_status"] == "blocked_for_listening"
    assert item["rebuilt_fields"] == ["defect_effectiveness_gate", "issue_effectiveness_ledger"]
    assert item["worst_windows"][0]["labels"] == ["high_noise_or_robotic_artifact"]
    assert item["blocker_reasons"] == ["selected_candidate_still_has_severe_issue"]


def test_minus_manifest_rebuilds_release_gate_from_selected_candidate_metrics(tmp_path: Path) -> None:
    manifest_path = tmp_path / "minus" / "adaptive_manifest.json"
    _write_json(
        manifest_path,
        {
            "mode": "adaptive_minus_repair_controller",
            "selected_profile": "soft_delta",
            "selected_score": 0.44,
            "candidates": [
                {
                    "profile_name": "soft_delta",
                    "metrics": {
                        "vocal_residue_corr": 0.31,
                        "vocal_to_instrumental_rms_ratio_db": -12.0,
                        "presence_band_ratio_drift_db": -0.1,
                        "air_band_ratio_drift_db": -0.2,
                    },
                    "focus_metrics": {},
                }
            ],
            "selected_artifacts": {
                "selected_minus_flac": "/app/out/02_adaptive_selected_minus.flac",
            },
        },
    )

    item = evaluator.evaluate_manifest_path(manifest_path)

    assert item["kind"] == "minus"
    assert item["release_status"] == "diagnostic_only"
    assert item["gate_status"] == "blocked_for_listening"
    assert item["rebuilt_fields"] == ["selected_minus_release_gate"]
    assert item["blocker_labels"] == ["vocal_residue_in_minus"]


def test_minus_manifest_without_metrics_is_diagnostic_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "minus" / "adaptive_manifest.json"
    _write_json(
        manifest_path,
        {
            "mode": "adaptive_minus_repair_controller",
            "selected_profile": "soft_delta",
            "selected_score": 0.44,
            "candidates": [{"profile_name": "soft_delta"}],
            "selected_artifacts": {
                "selected_minus_flac": "/app/out/02_adaptive_selected_minus.flac",
            },
        },
    )

    item = evaluator.evaluate_manifest_path(manifest_path)

    assert item["release_status"] == "diagnostic_only"
    assert item["gate_status"] == "blocked_for_listening"
    assert item["blocker_reasons"] == ["missing_minus_release_metrics"]
    assert item["blocker_labels"] == ["missing_minus_release_metrics"]
    assert item["rebuilt_fields"] == ["selected_minus_release_gate"]


def test_minus_manifest_with_persisted_gate_still_blocks_missing_metrics(tmp_path: Path) -> None:
    manifest_path = tmp_path / "minus" / "adaptive_manifest.json"
    _write_json(
        manifest_path,
        {
            "mode": "adaptive_minus_repair_controller",
            "selected_profile": "pure_base",
            "candidates": [
                {
                    "profile_name": "pure_base",
                    "metrics": {
                        "presence_band_ratio_drift_db": 0.0,
                        "air_band_ratio_drift_db": 0.0,
                    },
                    "focus_metrics": {},
                }
            ],
            "selected_minus_release_gate": {
                "status": "release_ready",
                "release_status": "release_ready",
                "blockers": [],
            },
            "selected_artifacts": {
                "selected_minus_flac": "/app/out/02_adaptive_selected_minus.flac",
                "release_status": "release_ready",
            },
        },
    )

    item = evaluator.evaluate_manifest_path(manifest_path)

    assert item["release_status"] == "diagnostic_only"
    assert item["gate_status"] == "blocked_for_listening"
    assert item["blocker_reasons"] == ["missing_minus_release_metrics"]
    assert item["blocker_labels"] == ["missing_minus_release_metrics"]
    assert item["release_status_drift"] is True
    assert item["rebuilt_fields"] == ["selected_minus_release_gate"]


def test_minus_manifest_with_persisted_gate_blocks_non_finite_metrics(tmp_path: Path) -> None:
    manifest_path = tmp_path / "minus" / "adaptive_manifest.json"
    _write_json(
        manifest_path,
        {
            "mode": "adaptive_minus_repair_controller",
            "selected_profile": "pure_base",
            "candidates": [
                {
                    "profile_name": "pure_base",
                    "metrics": {
                        "vocal_residue_corr": float("inf"),
                        "vocal_to_instrumental_rms_ratio_db": -120.0,
                        "presence_band_ratio_drift_db": 0.0,
                        "air_band_ratio_drift_db": 0.0,
                    },
                    "focus_metrics": {},
                }
            ],
            "selected_minus_release_gate": {
                "status": "release_ready",
                "release_status": "release_ready",
                "blockers": [],
            },
            "selected_artifacts": {
                "selected_minus_flac": "/app/out/02_adaptive_selected_minus.flac",
                "release_status": "release_ready",
            },
        },
    )

    item = evaluator.evaluate_manifest_path(manifest_path)

    assert item["release_status"] == "diagnostic_only"
    assert item["gate_status"] == "blocked_for_listening"
    assert item["blocker_reasons"] == ["missing_minus_release_metrics"]
    assert item["blocker_labels"] == ["missing_minus_release_metrics"]
    assert item["release_status_drift"] is True


def test_posthoc_evaluation_scans_only_supported_manifests(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "mode": "adaptive_vocal_isolation_controller",
            "selected_profile": "clean",
            "selected_score": 0.8,
            "selected_segment_issue_report": {
                "summary": {
                    "max_issue_score": 0.0,
                    "mean_top3_issue_score": 0.0,
                    "issue_window_ratio": 0.0,
                },
                "segments": [],
            },
            "detected_issue_action_contract": {"status": "pass", "segments": [], "summary": {}},
            "candidates": [{"profile_name": "clean", "metrics": {}}],
            "selected_artifacts": {"selected_vocals_flac": "/app/vocal.flac"},
        },
    )
    _write_json(
        tmp_path / "other" / "adaptive_manifest.json",
        {"mode": "unrelated_controller", "selected_profile": "ignore_me"},
    )
    summary_path = tmp_path / "summary.json"

    summary = evaluator.run_posthoc_release_evaluation(
        artifact_root=tmp_path,
        summary_path=summary_path,
    )

    assert summary["evaluated_count"] == 1
    assert summary["release_status_counts"] == {"release_ready": 1}
    assert summary["items"][0]["selected_profile"] == "clean"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_posthoc_evaluation_records_corrupt_manifest_without_aborting(tmp_path: Path) -> None:
    corrupt_manifest = tmp_path / "bad" / "adaptive_manifest.json"
    corrupt_manifest.parent.mkdir(parents=True)
    corrupt_manifest.write_text("{not-json", encoding="utf-8")
    _write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "mode": "adaptive_vocal_isolation_controller",
            "selected_profile": "clean",
            "selected_score": 0.8,
            "selected_segment_issue_report": {
                "summary": {
                    "max_issue_score": 0.0,
                    "mean_top3_issue_score": 0.0,
                    "issue_window_ratio": 0.0,
                },
                "segments": [],
            },
            "detected_issue_action_contract": {"status": "pass", "segments": [], "summary": {}},
            "candidates": [{"profile_name": "clean", "metrics": {}}],
            "selected_artifacts": {"selected_vocals_flac": "/app/vocal.flac"},
        },
    )
    summary_path = tmp_path / "summary.json"

    summary = evaluator.run_posthoc_release_evaluation(
        artifact_root=tmp_path,
        summary_path=summary_path,
    )

    assert summary["evaluated_count"] == 1
    assert summary["error_count"] == 1
    assert summary["errors"][0]["manifest_path"] == str(corrupt_manifest)


def test_posthoc_evaluation_limit_zero_returns_no_items(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "mode": "adaptive_vocal_isolation_controller",
            "selected_profile": "clean",
            "selected_score": 0.8,
            "selected_segment_issue_report": {
                "summary": {
                    "max_issue_score": 0.0,
                    "mean_top3_issue_score": 0.0,
                    "issue_window_ratio": 0.0,
                },
                "segments": [],
            },
            "detected_issue_action_contract": {"status": "pass", "segments": [], "summary": {}},
            "candidates": [{"profile_name": "clean", "metrics": {}}],
            "selected_artifacts": {"selected_vocals_flac": "/app/vocal.flac"},
        },
    )
    summary_path = tmp_path / "summary.json"

    summary = evaluator.run_posthoc_release_evaluation(
        artifact_root=tmp_path,
        summary_path=summary_path,
        limit=0,
    )

    assert summary["evaluated_count"] == 0
    assert summary["release_status_counts"] == {}
    assert summary["items"] == []


def test_cli_script_path_runs_inside_docker_layout(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "mode": "adaptive_vocal_isolation_controller",
            "selected_profile": "clean",
            "selected_score": 0.8,
            "selected_segment_issue_report": {
                "summary": {
                    "max_issue_score": 0.0,
                    "mean_top3_issue_score": 0.0,
                    "issue_window_ratio": 0.0,
                },
                "segments": [],
            },
            "detected_issue_action_contract": {"status": "pass", "segments": [], "summary": {}},
            "candidates": [{"profile_name": "clean", "metrics": {}}],
            "selected_artifacts": {"selected_vocals_flac": "/app/vocal.flac"},
        },
    )
    summary_path = tmp_path / "summary.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(evaluator.__file__)),
            str(tmp_path),
            str(summary_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(summary_path.read_text(encoding="utf-8"))["evaluated_count"] == 1
