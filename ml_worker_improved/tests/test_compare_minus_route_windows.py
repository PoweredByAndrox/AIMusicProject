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


def create_run(run_dir: Path, judge_rows: list[dict]) -> None:
    write_json(run_dir / "run_manifest.json", {"run_id": "run-a", "track_id": "track002", "release_ready": False})
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track002",
                "candidate_id": row["candidate_id"],
                "route_id": row["route_id"],
                "candidate_family": "artifact_safe_default",
                "source_model_id": row.get("source_model_id", row["route_id"]),
                "source_tool": "unit-tool",
                "minus_path": f"/tmp/{row['candidate_id']}.flac",
                "raw_minus_path": f"/tmp/raw/{row['candidate_id']}.flac",
                "normalized_minus_path": f"/tmp/norm/{row['candidate_id']}.flac",
                "shape_normalization_applied": row.get("shape_normalization_applied", False),
                "release_ready": False,
            }
            for row in judge_rows
        ],
    )
    write_jsonl(run_dir / "minus_windowed_judge.jsonl", judge_rows)


def blocked_row(candidate_id: str, route_id: str, blockers: list[str]) -> dict:
    return {
        "track_id": "track002",
        "candidate_id": candidate_id,
        "route_id": route_id,
        "minus_decision": "blocked_hard_case",
        "minus_blockers": blockers,
        "minus_score": 0.2,
        "window_metrics": [
            {
                "start_sec": 0,
                "duration_sec": 10,
                "minus_decision": "blocked_hard_case",
                "minus_blockers": blockers,
                "minus_score": 0.2,
            },
            {
                "start_sec": 10,
                "duration_sec": 10,
                "minus_decision": "blocked_hard_case",
                "minus_blockers": blockers,
                "minus_score": 0.25,
            },
        ],
        "release_ready": False,
    }


def test_comparison_detects_all_routes_same_blocker_signature_and_exhaustion(tmp_path: Path) -> None:
    from tools.compare_minus_route_windows import compare_minus_routes, write_outputs

    run_dir = tmp_path / "run"
    create_run(
        run_dir,
        [
            blocked_row("cand-a", "route-a", ["severe_phase_collapse", "severe_vocal_bleed_in_minus"]),
            blocked_row("cand-b", "route-b", ["severe_vocal_bleed_in_minus", "severe_phase_collapse"]),
        ],
    )

    result = compare_minus_routes(run_dir)
    write_outputs(result, tmp_path / "out")
    rows = read_jsonl(tmp_path / "out" / "minus_route_window_comparison.jsonl")
    summary = json.loads((tmp_path / "out" / "minus_route_window_comparison_summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert summary["all_routes_same_blocker_signature"] is True
    assert summary["any_route_has_clean_window"] is False
    assert summary["segment_blend_possible"] is False
    assert summary["route_space_exhausted"] is True
    assert rows[0]["windows_blocked"] == 2
    assert rows[0]["severe_phase_collapse_count"] == 2
    assert rows[0]["severe_vocal_bleed_count"] == 2


def test_comparison_detects_clean_windows_and_segment_blend(tmp_path: Path) -> None:
    from tools.compare_minus_route_windows import compare_minus_routes

    run_dir = tmp_path / "run"
    create_run(
        run_dir,
        [
            {
                "track_id": "track002",
                "candidate_id": "cand-a",
                "route_id": "route-a",
                "minus_decision": "blocked_hard_case",
                "minus_blockers": ["severe_phase_collapse"],
                "minus_score": 0.5,
                "window_metrics": [
                    {"start_sec": 0, "duration_sec": 10, "minus_decision": "accept_candidate", "minus_blockers": [], "minus_score": 0.9},
                    {"start_sec": 10, "duration_sec": 10, "minus_decision": "blocked_hard_case", "minus_blockers": ["severe_phase_collapse"], "minus_score": 0.2},
                ],
                "release_ready": False,
            },
            {
                "track_id": "track002",
                "candidate_id": "cand-b",
                "route_id": "route-b",
                "minus_decision": "blocked_hard_case",
                "minus_blockers": ["severe_vocal_bleed_in_minus"],
                "minus_score": 0.55,
                "window_metrics": [
                    {"start_sec": 0, "duration_sec": 10, "minus_decision": "blocked_hard_case", "minus_blockers": ["severe_vocal_bleed_in_minus"], "minus_score": 0.3},
                    {"start_sec": 10, "duration_sec": 10, "minus_decision": "accept_candidate", "minus_blockers": [], "minus_score": 0.91},
                ],
                "release_ready": False,
            },
        ],
    )

    result = compare_minus_routes(run_dir)

    assert result["summary"]["any_route_has_clean_window"] is True
    assert result["summary"]["any_route_reduces_phase_collapse"] is True
    assert result["summary"]["any_route_reduces_vocal_bleed"] is True
    assert result["summary"]["segment_blend_possible"] is True
    assert result["summary"]["route_space_exhausted"] is False
    assert sorted(result["summary"]["segment_blend_clean_window_coverage"].keys()) == ["0", "1"]
