from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_validate_route_expansion_aggregates_safe_hard_case_counts(tmp_path: Path) -> None:
    from tools.validate_minus_route_expansion import validate_minus_route_expansion

    discovery = write_json(
        tmp_path / "minus_route_discovery.json",
        {
            "summary": {
                "discovered_routes_total": 2,
                "available_routes_total": 1,
                "unavailable_routes_total": 1,
                "route_space_limited": True,
                "release_ready": False,
            },
            "routes": [
                {"route_id": "route-a", "available": True, "forbidden_reason": "", "release_ready": False},
                {"route_id": "route-b", "available": False, "forbidden_reason": "", "release_ready": False},
            ],
        },
    )
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "minus_route_expansion_manifest.json",
        {
            "summary": {
                "jobs_attempted": 1,
                "jobs_generated": 1,
                "jobs_failed": 0,
                "minus_candidates_generated": 1,
                "accepted_minus_candidates": 0,
                "severe_phase_collapse_count": 1,
                "severe_vocal_bleed_in_minus_count": 1,
                "phase_comb_filtering_count": 0,
                "pair_compatible_count": 0,
                "release_ready_count": 0,
                "quality_improved_route_ids": [],
                "rejected_route_ids": ["route-a"],
                "route_space_exhausted": True,
                "release_ready": False,
            }
        },
    )
    write_jsonl(
        run_dir / "minus_route_candidate_pool.jsonl",
        [
            {
                "route_id": "route-a",
                "accepted_minus": False,
                "minus_blockers": ["severe_phase_collapse", "severe_vocal_bleed_in_minus"],
                "release_ready": False,
            }
        ],
    )

    result = validate_minus_route_expansion(discovery_json=discovery, expansion_dirs=[run_dir])

    assert result["ok"] is True
    assert result["discovered_routes_total"] == 2
    assert result["available_routes_total"] == 1
    assert result["accepted_minus_candidates"] == 0
    assert result["severe_phase_collapse_count"] == 1
    assert result["severe_vocal_bleed_in_minus_count"] == 1
    assert result["route_space_exhausted"] is True
    assert result["release_ready_count"] == 0
    assert result["training_started"] is False
    assert result["thresholds_relaxed"] is False
    assert result["cleanup_routes_added"] is False


def test_validate_route_expansion_blocks_release_ready_violation(tmp_path: Path) -> None:
    from tools.validate_minus_route_expansion import validate_minus_route_expansion

    discovery = write_json(
        tmp_path / "minus_route_discovery.json",
        {"summary": {"release_ready": False}, "routes": [{"route_id": "route-a", "release_ready": False}]},
    )
    run_dir = tmp_path / "run"
    write_json(run_dir / "minus_route_expansion_manifest.json", {"summary": {"release_ready_count": 1}})
    write_jsonl(run_dir / "minus_route_candidate_pool.jsonl", [{"route_id": "route-a", "release_ready": True}])

    result = validate_minus_route_expansion(discovery_json=discovery, expansion_dirs=[run_dir])

    assert result["ok"] is False
    assert "release_ready_violation" in result["blockers"]


def test_validate_route_expansion_scans_jobs_and_pair_artifacts_for_safety_violations(tmp_path: Path) -> None:
    from tools.validate_minus_route_expansion import validate_minus_route_expansion

    discovery = write_json(
        tmp_path / "minus_route_discovery.json",
        {"summary": {"release_ready": False}, "routes": [{"route_id": "route-a", "release_ready": False}]},
    )
    run_dir = tmp_path / "run"
    write_json(run_dir / "minus_route_expansion_manifest.json", {"summary": {"release_ready_count": 0}})
    write_jsonl(run_dir / "minus_route_candidate_pool.jsonl", [])
    write_jsonl(run_dir / "minus_route_jobs.jsonl", [{"route_id": "route-a", "release_ready": True}])
    write_jsonl(run_dir / "minus_route_pair_compatibility.jsonl", [{"route_id": "route-a", "forbidden_reason": "debleed"}])

    result = validate_minus_route_expansion(discovery_json=discovery, expansion_dirs=[run_dir])

    assert result["ok"] is False
    assert "release_ready_violation" in result["blockers"]
    assert "cleanup_route_violation" in result["blockers"]


def test_validate_route_expansion_blocks_forbidden_cleanup_route(tmp_path: Path) -> None:
    from tools.validate_minus_route_expansion import validate_minus_route_expansion

    discovery = write_json(
        tmp_path / "minus_route_discovery.json",
        {
            "summary": {"release_ready": False},
            "routes": [{"route_id": "cleanup", "forbidden_reason": "cleanup", "release_ready": False}],
        },
    )
    run_dir = tmp_path / "run"
    write_json(run_dir / "minus_route_expansion_manifest.json", {"summary": {"release_ready_count": 0}})
    write_jsonl(run_dir / "minus_route_candidate_pool.jsonl", [])

    result = validate_minus_route_expansion(discovery_json=discovery, expansion_dirs=[run_dir])

    assert result["ok"] is False
    assert "forbidden_route:cleanup" in result["blockers"]


def test_validate_route_expansion_allows_unsupported_cli_unavailable_route(tmp_path: Path) -> None:
    from tools.validate_minus_route_expansion import validate_minus_route_expansion

    discovery = write_json(
        tmp_path / "minus_route_discovery.json",
        {
            "summary": {"release_ready": False, "discovered_routes_total": 1, "available_routes_total": 0},
            "routes": [
                {
                    "route_id": "unsupported-overlap",
                    "available": False,
                    "forbidden_reason": "unsupported_cli_parameter:overlap",
                    "release_ready": False,
                }
            ],
        },
    )
    run_dir = tmp_path / "run"
    write_json(run_dir / "minus_route_expansion_manifest.json", {"summary": {"release_ready_count": 0}})
    write_jsonl(run_dir / "minus_route_candidate_pool.jsonl", [])

    result = validate_minus_route_expansion(discovery_json=discovery, expansion_dirs=[run_dir])

    assert result["ok"] is True
    assert result["blockers"] == []
