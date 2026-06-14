from __future__ import annotations

import json
from pathlib import Path

from tools.generate_no_reference_failure_reflections import generate_failure_reflections


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_reflection_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    write_json(
        bundle_dir / "replay_bundle_manifest.json",
        {"run_count": 1, "replay_complete_runs": 1, "replay_incomplete_runs": 0, "release_ready_count": 0},
    )
    write_jsonl(
        bundle_dir / "replay_bundle_events.jsonl",
        [
            {
                "event_type": "run_summary",
                "track_id": "track-002",
                "run_root": "/runs/track-002",
                "output_status": "needs_more_candidates",
                "failure_reason": "minus_hard_case_route_exhausted",
                "severe_phase_collapse_count": 12,
                "severe_vocal_bleed_in_minus_count": 12,
                "release_ready": False,
            },
            {
                "event_type": "minus_hard_case_router",
                "track_id": "track-002",
                "payload": {
                    "blocker_type": "severe_phase_collapse",
                    "recommended_route": "generate_phase_safe_variants",
                    "available_route_configs": ["local_registered__htdemucs_6s__minus"],
                    "can_run_without_new_models": True,
                    "release_ready": False,
                },
            },
            {
                "event_type": "candidate_wave_failures",
                "track_id": "track-002",
                "payload": {"candidate_id": "bad-minus", "skip_reason": "shape_mismatch", "release_ready": False},
            },
        ],
    )
    return bundle_dir


def test_failure_reflections_generated_from_replay_bundle(tmp_path: Path) -> None:
    bundle_dir = make_reflection_bundle(tmp_path)
    output_dir = tmp_path / "reflections"

    result = generate_failure_reflections(replay_bundle_dir=bundle_dir, output_dir=output_dir)

    rows = [json.loads(line) for line in (output_dir / "failure_reflections.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["reflection_count"] >= 2
    assert any(row["blocker_type"] == "severe_phase_collapse" for row in rows)
    assert any(row["blocker_type"] == "severe_vocal_bleed_in_minus" for row in rows)
    assert all(row["target_module"] for row in rows)
    assert all(row["target_stage_graph_policy"] for row in rows)
    assert all(isinstance(row["candidate_mutation_patch"], dict) for row in rows)
    assert all(row["forbidden_change_check"]["thresholds_relaxed"] is False for row in rows)
    assert all(row["forbidden_change_check"]["cleanup_debleed_denoise"] is False for row in rows)
    assert all(row["release_ready"] is False for row in rows)
    assert "route_space_exhausted" in (output_dir / "failure_reflections_summary.md").read_text(encoding="utf-8")
