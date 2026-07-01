from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def readiness_summary(path: Path) -> Path:
    return write_json(
        path,
        {
            "schema": "mac_candidate_readiness_summary.v1",
            "manifest": "/host_artifacts/mac_api/altm002/manifest.json",
            "source_schema": "mac_alt_minus_route_eval.v1",
            "source_status": "blocked",
            "source_audio_path": "/mac/source/t001.mp3",
            "source_track_identity": {"track_id": "t001"},
            "status": "blocked_human_listen_feedback",
            "delivery_allowed": False,
            "listen_result_allowed": False,
            "blocker_reasons": [
                "human_feedback_residual_vocal_leak",
                "human_feedback_detail_loss",
                "pitch_guard_failed",
            ],
            "next_actions": [
                "run_judge_guided_minus_vocal_leakage_repair",
                "run_judge_guided_minus_air_detail_repair",
            ],
            "objective_summary": {
                "before_fail_regions": 16,
                "after_fail_regions": 16,
                "region_delta": 0,
                "pitch_guard_verdict": "fail",
            },
            "promising_blocked_handoff": {
                "status": "promising_but_blocked",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_only_handoff",
                "issue_categories": ["detail_loss", "residual_vocal_leak"],
                "issue_tags": ["detail_loss", "residual_vocal_middle"],
                "recommended_routes": [
                    {
                        "issue_category": "detail_loss",
                        "route": "judge_guided_minus_air_detail_repair",
                        "route_status": "documented_route",
                        "tool": "tools/apply_judge_guided_minus_air_detail_repair.py",
                        "next_action": "run_judge_guided_minus_air_detail_repair",
                    },
                    {
                        "issue_category": "residual_vocal_leak",
                        "route": "judge_guided_minus_vocal_leakage_repair",
                        "route_status": "documented_route",
                        "tool": "tools/apply_judge_guided_minus_vocal_leakage_repair.py",
                        "next_action": "run_judge_guided_minus_vocal_leakage_repair",
                        "expected_acceleration_mode": "external_only_cpu_dsp_repair",
                        "requires_post_judge": True,
                    },
                ],
                "contract": {
                    "diagnostic_only": True,
                    "does_not_create_listen_result": True,
                    "requires_objective_gate_before_delivery": True,
                },
            },
        },
    )


def test_direct_script_builds_full_pack_loop_queue_from_promising_handoff(tmp_path: Path) -> None:
    summary = readiness_summary(tmp_path / "rd" / "summary.json")
    output_dir = tmp_path / "handoff"

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_mac_promising_blocked_handoff.py",
            "--mac-readiness-summary-json",
            str(summary),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "queue.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "short_listen_repair_queue_manifest.v1"
    assert manifest["queue_status"] == "ready_for_full_pack_loop"
    assert manifest["queue_source"] == "mac_promising_blocked_handoff"
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["queue_item_count"] == 1
    assert manifest["route_status_counts"] == {
        "documented_route": 2,
    }
    row = rows(output_dir / "rows.jsonl")[0]
    assert row["schema"] == "short_listen_repair_queue_item.v1"
    assert row["route_intent"] == "mac_promising_blocked_handoff"
    assert row["stem_type"] == "minus"
    assert row["issue_categories"] == ["detail_loss", "residual_vocal_leak"]
    assert row["objective_summary"]["after_fail_regions"] == 16
    assert row["recommended_routes"][0]["route"] == "judge_guided_minus_air_detail_repair"
    assert row["recommended_routes"][1]["route"] == "judge_guided_minus_vocal_leakage_repair"
    assert row["recommended_routes"][1]["route_status"] == "documented_route"
    assert row["required_actions"] == [
        "run_judge_guided_minus_air_detail_repair",
        "run_judge_guided_minus_vocal_leakage_repair",
        "minus_audit_gate",
        "holdout_eval_gate",
        "train_ready_decision_gate",
        "full_pack_gate_rebuild",
    ]
    assert row["release_ready"] is False
    assert row["can_publish_as_result"] is False
    assert all(len(path.name) <= 32 for path in output_dir.iterdir() if path.is_file())
