from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def base_forensics_dir(tmp_path: Path, comparison_summary: dict) -> Path:
    forensics_dir = tmp_path / "forensics"
    write_json(
        forensics_dir / "minus_window_blocker_clusters.json",
        {
            "summary": {
                "candidates_analyzed": 2,
                "routes_analyzed": 2,
                "windows_analyzed": 4,
                "severe_phase_collapse_total": 2,
                "severe_vocal_bleed_in_minus_total": 2,
                "blocker_signatures": ["severe_phase_collapse", "severe_vocal_bleed_in_minus"],
                "normalization_correlated": False,
                "release_ready": False,
            },
            "by_candidate": {},
            "by_route": {},
            "by_window": {},
        },
    )
    write_json(forensics_dir / "minus_route_window_comparison_summary.json", comparison_summary)
    write_jsonl(
        forensics_dir / "minus_route_window_comparison.jsonl",
        [
            {"route_id": "route-a", "windows_total": 2, "windows_accepted": 0, "windows_blocked": 2},
            {"route_id": "route-b", "windows_total": 2, "windows_accepted": 0, "windows_blocked": 2},
        ],
    )
    return forensics_dir


def test_failure_report_blocks_threshold_cleanup_and_release_ready(tmp_path: Path) -> None:
    from tools.build_track002_minus_failure_report import build_failure_report, write_outputs

    forensics_dir = base_forensics_dir(
        tmp_path,
        {
            "all_routes_same_blocker_signature": True,
            "any_route_has_clean_window": False,
            "any_route_reduces_phase_collapse": False,
            "any_route_reduces_vocal_bleed": False,
            "segment_blend_possible": False,
            "route_space_exhausted": True,
            "release_ready": False,
        },
    )

    report = build_failure_report(forensics_dir)
    write_outputs(report, tmp_path / "out")
    decision = json.loads((tmp_path / "out" / "track002_next_route_decision.json").read_text(encoding="utf-8"))

    assert report["decision"]["route_space_exhausted"] is True
    assert report["decision"]["unsupported_new_model_needed"] is True
    assert report["decision"]["unsupported_new_model_needed_execution"] is False
    assert report["decision"]["recommend_threshold_relaxation"] is False
    assert report["decision"]["recommend_cleanup_debleed_denoise"] is False
    assert report["decision"]["new_model_download_started"] is False
    assert report["release_ready"] is False
    assert report["melody_transcription_ran"] is False
    assert report["recommendation_index_updated"] is False
    assert decision["release_ready"] is False
    assert decision["recommended_next_step"] == "blocked_no_safe_local_route_available"


def test_failure_report_marks_segment_blend_candidate_without_publish(tmp_path: Path) -> None:
    from tools.build_track002_minus_failure_report import build_failure_report

    forensics_dir = base_forensics_dir(
        tmp_path,
        {
            "all_routes_same_blocker_signature": False,
            "any_route_has_clean_window": True,
            "any_route_reduces_phase_collapse": True,
            "any_route_reduces_vocal_bleed": True,
            "segment_blend_possible": True,
            "route_space_exhausted": False,
            "normalization_issue_suspected": False,
            "release_ready": False,
        },
    )

    report = build_failure_report(forensics_dir)

    assert report["decision"]["segment_blend_candidate"] is True
    assert report["decision"]["local_route_candidate_needed"] is False
    assert report["decision"]["route_space_exhausted"] is False
    assert report["decision"]["unsupported_new_model_needed"] is False
    assert report["can_publish_as_result"] is False
    assert report["release_ready"] is False
    assert report["recommended_next_step"] == "plan_segment_level_minus_repair"
