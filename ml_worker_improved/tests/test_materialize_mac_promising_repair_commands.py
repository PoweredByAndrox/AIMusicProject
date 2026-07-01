from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fixture_plan(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "mac_api" / "altm002"
    for name in ("instrumental.flac", "vocals.flac", "raw_ref.flac"):
        path = artifact_root / "bci" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    write_jsonl(
        artifact_root / "guard" / "score" / "regions.jsonl",
        [
            {
                "start": 10.0,
                "end": 10.75,
                "score": 0.81,
                "projection_rel_db": -18.8,
                "fail_mode": "standard_score",
            }
        ],
    )
    source_manifest = write_json(
        artifact_root / "manifest.json",
        {
            "schema": "mac_alt_minus_route_eval.v1",
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_only_not_for_listen_pack",
            "source_audio_path_mac": "/mac/source/t001.mp3",
            "windows_artifact_root": str(artifact_root),
            "candidate": {
                "instrumental_path": "bci/instrumental.flac",
                "vocals_path": "bci/vocals.flac",
                "raw_ref_path": "bci/raw_ref.flac",
            },
            "guard": {
                "score_path": "guard/score/pitch.json",
                "score_region_count": 1,
                "score_verdict": "fail",
            },
        },
    )
    readiness = write_json(
        tmp_path / "rd" / "summary.json",
        {
            "schema": "mac_candidate_readiness_summary.v1",
            "status": "blocked_human_listen_feedback",
            "manifest": str(source_manifest),
            "delivery_allowed": False,
            "listen_result_allowed": False,
        },
    )
    plan_rows = write_jsonl(
        tmp_path / "plan" / "rows.jsonl",
        [
            {
                "schema": "full_pack_loop_iteration_action.v1",
                "queue_item_id": "mph001",
                "route_intent": "mac_promising_blocked_handoff",
                "action_status": "ready_for_documented_mac_promising_repair_routes",
                "readiness_summary_json": str(readiness),
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
                        "requires_post_judge": True,
                    },
                ],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    return write_json(
        tmp_path / "plan" / "plan.json",
        {
            "schema": "full_pack_loop_iteration_plan.v1",
            "plan_status": "ready_for_documented_mac_promising_repair_routes",
            "rows_jsonl": str(plan_rows),
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
        },
    )


def test_materializer_builds_leakage_command_and_blocks_missing_detail_judge(tmp_path: Path) -> None:
    plan = fixture_plan(tmp_path)
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/materialize_mac_promising_repair_commands.py",
            "--plan-json",
            str(plan),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2, result.stdout
    manifest = json.loads((output_dir / "cmds.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "mac_promising_repair_command_plan.v1"
    assert manifest["command_plan_status"] == "partial_ready_missing_inputs"
    assert manifest["delivery_allowed"] is False
    assert manifest["execution_started"] is False
    assert manifest["ready_command_count"] == 1
    assert manifest["blocked_command_count"] == 1
    assert manifest["failed_checks"] == ["detail_route_windowed_judge_missing"]
    command_rows = rows(output_dir / "rows.jsonl")
    by_route = {row["route"]: row for row in command_rows}
    assert by_route["judge_guided_minus_air_detail_repair"]["command_status"] == "blocked_missing_windowed_detail_judge"
    assert by_route["judge_guided_minus_vocal_leakage_repair"]["command_status"] == "ready"
    leakage = by_route["judge_guided_minus_vocal_leakage_repair"]
    assert leakage["generated_judge_jsonl"].endswith("leak/rows.jsonl")
    assert leakage["docker_command"][:2] == ["python", "tools/apply_judge_guided_minus_vocal_leakage_repair.py"]
    assert "--short-output-names" in leakage["docker_command"]
    generated = rows(output_dir / "leak" / "rows.jsonl")[0]
    assert generated["candidate_id"] == "mph001"
    assert generated["window_metrics"][0]["issues"] == ["vocal_leakage_in_minus"]
    assert generated["window_metrics"][0]["vocal_leakage_in_minus_score"] == 0.81
    assert all(len(path.name) <= 32 for path in output_dir.rglob("*") if path.is_file())
