from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run-track002",
            "track_id": "track002",
            "output_status": "needs_more_candidates",
            "failure_reason": "conditional_hard_case_route_exhausted",
            "release_ready": False,
        },
    )
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "run_id": "run-track002",
                "track_id": "track002",
                "candidate_id": "cand-a",
                "candidate_family": "artifact_safe_default",
                "route_id": "route-a",
                "source_model_id": "model-a",
                "source_tool": "tool-a",
                "minus_path": "/tmp/normalized/a-minus.flac",
                "raw_minus_path": "/tmp/raw/a-minus.flac",
                "normalized_minus_path": "/tmp/normalized/a-minus.flac",
                "shape_normalization_applied": True,
                "shape_normalization_reason": "sample_rate_mismatch",
                "release_ready": False,
            }
        ],
    )
    return run_dir


def test_blocker_profiler_detects_severe_phase_and_vocal_bleed(tmp_path: Path) -> None:
    from tools.analyze_minus_window_blockers import build_window_blocker_profile, write_outputs

    run_dir = base_run_dir(tmp_path)
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "run_id": "run-track002",
                "track_id": "track002",
                "candidate_id": "cand-a",
                "route_id": "route-a",
                "minus_decision": "blocked_hard_case",
                "minus_blockers": ["severe_phase_collapse", "severe_vocal_bleed_in_minus"],
                "minus_score": 0.42,
                "window_metrics": [
                    {
                        "start_sec": 0.0,
                        "duration_sec": 12.0,
                        "minus_decision": "blocked_hard_case",
                        "minus_blockers": ["severe_phase_collapse"],
                        "minus_score": 0.2,
                        "minus_score_components": {"phase_stereo_stability": 0.1},
                    },
                    {
                        "start_sec": 12.0,
                        "duration_sec": 12.0,
                        "minus_decision": "blocked_hard_case",
                        "minus_blockers": ["severe_vocal_bleed_in_minus"],
                        "minus_score": 0.4,
                        "minus_score_components": {"vocal_leakage_absence": 0.2},
                    },
                ],
                "release_ready": False,
            }
        ],
    )

    result = build_window_blocker_profile(run_dir)
    write_outputs(result, tmp_path / "out")
    rows = read_jsonl(tmp_path / "out" / "minus_window_blocker_profile.jsonl")
    clusters = json.loads((tmp_path / "out" / "minus_window_blocker_clusters.json").read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert rows[0]["severe_phase_collapse"] is True
    assert rows[1]["severe_vocal_bleed_in_minus"] is True
    assert result["summary"]["severe_phase_collapse_total"] == 1
    assert result["summary"]["severe_vocal_bleed_in_minus_total"] == 1
    assert clusters["by_candidate"]["cand-a"]["window_count"] == 2
    assert clusters["by_window"]["0"]["blocker_signatures"] == ["severe_phase_collapse"]
    assert rows[0]["raw_minus_path"] == "/tmp/raw/a-minus.flac"
    assert rows[0]["normalized_minus_path"] == "/tmp/normalized/a-minus.flac"
    assert rows[0]["shape_normalization_applied"] is True
    assert "phase_stereo_stability" in rows[0]["score_components"]


def test_blocker_profiler_groups_clean_and_blocked_signatures_by_window(tmp_path: Path) -> None:
    from tools.analyze_minus_window_blockers import build_window_blocker_profile

    run_dir = base_run_dir(tmp_path)
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": "track002",
                "candidate_id": "cand-a",
                "route_id": "route-a",
                "minus_decision": "blocked_hard_case",
                "window_metrics": [
                    {"start_sec": 0, "duration_sec": 10, "minus_decision": "accept_candidate", "minus_blockers": []},
                    {"start_sec": 10, "duration_sec": 10, "minus_decision": "blocked_hard_case", "minus_blockers": ["severe_phase_comb_filtering"]},
                ],
                "release_ready": False,
            }
        ],
    )

    result = build_window_blocker_profile(run_dir)

    assert result["summary"]["localized_windows"] is True
    assert result["summary"]["global_all_windows"] is False
    assert result["summary"]["severe_phase_comb_filtering_total"] == 1
    assert result["clusters"]["by_candidate"]["cand-a"]["blocker_signatures"] == [
        "clean",
        "severe_phase_comb_filtering",
    ]


def test_blocker_profiler_reads_minus_route_expansion_artifact_names(tmp_path: Path) -> None:
    from tools.analyze_minus_window_blockers import build_window_blocker_profile

    run_dir = tmp_path / "route_expansion"
    write_json(run_dir / "run_manifest.json", {"run_id": "stem2", "track_id": "track002", "release_ready": False})
    write_jsonl(
        run_dir / "minus_route_candidate_pool.jsonl",
        [
            {
                "track_id": "track002",
                "candidate_id": "route-cand",
                "route_id": "local_registered__htdemucs_6s__both__shared",
                "candidate_family": "artifact_safe_default",
                "source_model_id": "htdemucs_6s",
                "source_tool": "unit-tool",
                "minus_path": "/tmp/norm/minus.flac",
                "raw_minus_path": "/tmp/raw/minus.flac",
                "normalized_minus_path": "/tmp/norm/minus.flac",
                "shape_normalization_applied": True,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "minus_route_judge_minus.jsonl",
        [
            {
                "track_id": "track002",
                "candidate_id": "route-cand",
                "minus_decision": "blocked_hard_case",
                "minus_blockers": ["severe_phase_collapse"],
                "window_metrics": [
                    {"start_sec": 0, "duration_sec": 10, "minus_decision": "blocked_hard_case", "minus_blockers": ["severe_phase_collapse"]}
                ],
                "release_ready": False,
            }
        ],
    )

    result = build_window_blocker_profile(run_dir)

    assert result["summary"]["candidates_analyzed"] == 1
    assert result["rows"][0]["route_id"] == "local_registered__htdemucs_6s__both__shared"
    assert result["summary"]["input_artifacts"]["minus_route_candidate_pool.jsonl"]["row_count"] == 1
    assert result["summary"]["input_artifacts"]["minus_route_judge_minus.jsonl"]["row_count"] == 1
