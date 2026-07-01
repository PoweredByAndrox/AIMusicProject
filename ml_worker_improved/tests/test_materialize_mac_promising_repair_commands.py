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


def fixture_plan(tmp_path: Path) -> tuple[Path, Path]:
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
    plan = write_json(
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
    return plan, artifact_root


def test_materializer_builds_leakage_command_and_blocks_missing_detail_judge(tmp_path: Path) -> None:
    plan, _artifact_root = fixture_plan(tmp_path)
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


def test_materializer_builds_air_detail_command_from_self_judge_manifest(tmp_path: Path) -> None:
    plan, artifact_root = fixture_plan(tmp_path)
    repaired_minus = tmp_path / "loop" / "it016" / "lkg" / "m001.flac"
    repaired_minus.parent.mkdir(parents=True, exist_ok=True)
    repaired_minus.write_bytes(b"")
    rows_jsonl = write_jsonl(
        tmp_path / "self" / "rows.jsonl",
        [
            {
                "segment_id": "segment_0001",
                "window_sec": [5.0, 10.0],
                "candidate_label": "it016_lkg",
                "issues": ["low_rms_muffling_proxy", "presence_loss_proxy"],
                "issue_severities": {"low_rms_ratio": 0.71, "presence_loss": 0.31},
                "blocker": True,
                "severity_score": 0.71,
                "self_judge_contract": {"proxy_only": True, "not_delivery_result": True},
            },
            {
                "segment_id": "segment_0002",
                "window_sec": [10.0, 15.0],
                "candidate_label": "it016_lkg",
                "issues": ["air_loss_proxy", "vocal_projection_proxy"],
                "issue_severities": {"air_loss": 0.42, "vocal_projection": 0.83},
                "blocker": True,
                "severity_score": 0.83,
                "self_judge_contract": {"proxy_only": True, "not_delivery_result": True},
            },
        ],
    )
    self_manifest = write_json(
        tmp_path / "self" / "manifest.json",
        {
            "schema_version": 1,
            "mode": "instrumental_self_judge_gate",
            "source_audio_path": str(artifact_root / "bci" / "raw_ref.flac"),
            "segments_jsonl": str(rows_jsonl),
            "overall": {"status": "proxy_blocked", "blocker_count": 2, "segment_count": 2},
            "candidate": {
                "label": "it016_lkg",
                "instrumental_path": str(repaired_minus),
                "vocal_path": str(artifact_root / "bci" / "vocals.flac"),
            },
            "delivery_contract": {
                "delivery_allowed": False,
                "presentation_status": "diagnostic_self_judge_not_for_listen_pack",
            },
        },
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/materialize_mac_promising_repair_commands.py",
            "--plan-json",
            str(plan),
            "--output-dir",
            str(output_dir),
            "--minus-self-judge-manifest",
            str(self_manifest),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output_dir / "cmds.json").read_text(encoding="utf-8"))
    assert manifest["command_plan_status"] == "ready"
    assert manifest["delivery_allowed"] is False
    assert manifest["ready_command_count"] == 2
    assert manifest["blocked_command_count"] == 0
    assert manifest["failed_checks"] == []
    command_rows = rows(output_dir / "rows.jsonl")
    detail = {row["route"]: row for row in command_rows}["judge_guided_minus_air_detail_repair"]
    assert detail["command_status"] == "ready"
    assert detail["safe_detail_window_count"] == 1
    assert detail["blocked_unsafe_detail_window_count"] == 1
    assert detail["generated_detail_jsonl"].endswith("dtl/rows.jsonl")
    assert detail["base_minus_path"] == str(repaired_minus)
    assert detail["docker_command"][:2] == ["python", "tools/apply_judge_guided_minus_air_detail_repair.py"]
    assert detail["docker_command"][3] == str(repaired_minus)
    assert detail["docker_command"][4] == detail["generated_detail_jsonl"]
    assert "--short-output-names" in detail["docker_command"]
    assert "--vocal-guide-path" in detail["docker_command"]
    generated = rows(output_dir / "dtl" / "rows.jsonl")
    assert len(generated) == 1
    assert generated[0]["candidate_label"] == "it016_lkg"
    assert generated[0]["window_sec"] == [5.0, 10.0]
    assert generated[0]["issues"] == ["low_rms_muffling_proxy", "presence_loss_proxy"]
    assert generated[0]["judge_source"] == "instrumental_self_judge_gate"
    assert generated[0]["detail_materializer_source"] == "minus_self_judge_manifest"
    assert generated[0]["source_audio_path"] == str(artifact_root / "bci" / "raw_ref.flac")
    assert generated[0]["minus_path"] == str(repaired_minus)
    assert generated[0]["instrumental_path"] == str(repaired_minus)
    assert all(len(path.name) <= 32 for path in output_dir.rglob("*") if path.is_file())


def test_materializer_blocks_ambiguous_detail_self_judge_manifests(tmp_path: Path) -> None:
    plan, artifact_root = fixture_plan(tmp_path)
    rows_jsonl = write_jsonl(
        tmp_path / "self" / "rows.jsonl",
        [
            {
                "segment_id": "segment_0001",
                "window_sec": [5.0, 10.0],
                "candidate_label": "candidate_a",
                "issues": ["low_rms_muffling_proxy"],
                "issue_severities": {"low_rms_ratio": 0.71},
                "blocker": True,
                "severity_score": 0.71,
            }
        ],
    )
    manifest_paths = []
    for label in ("candidate_a", "candidate_b"):
        minus_path = tmp_path / "minus" / f"{label}.flac"
        minus_path.parent.mkdir(parents=True, exist_ok=True)
        minus_path.write_bytes(b"")
        manifest_paths.append(
            write_json(
                tmp_path / "self" / f"{label}.json",
                {
                    "schema_version": 1,
                    "mode": "instrumental_self_judge_gate",
                    "source_audio_path": str(artifact_root / "bci" / "raw_ref.flac"),
                    "segments_jsonl": str(rows_jsonl),
                    "overall": {"status": "proxy_blocked", "blocker_count": 1, "segment_count": 1},
                    "candidate": {
                        "label": label,
                        "instrumental_path": str(minus_path),
                        "vocal_path": str(artifact_root / "bci" / "vocals.flac"),
                    },
                    "delivery_contract": {"delivery_allowed": False},
                },
            )
        )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/materialize_mac_promising_repair_commands.py",
            "--plan-json",
            str(plan),
            "--output-dir",
            str(output_dir),
            "--minus-self-judge-manifest",
            str(manifest_paths[0]),
            "--minus-self-judge-manifest",
            str(manifest_paths[1]),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    manifest = json.loads((output_dir / "cmds.json").read_text(encoding="utf-8"))
    assert manifest["command_plan_status"] == "partial_ready_missing_inputs"
    assert manifest["ready_command_count"] == 1
    assert manifest["blocked_command_count"] == 1
    assert manifest["failed_checks"] == ["detail_route_self_judge_manifest_ambiguous"]
    command_rows = rows(output_dir / "rows.jsonl")
    detail = {row["route"]: row for row in command_rows}["judge_guided_minus_air_detail_repair"]
    assert detail["command_status"] == "blocked_ambiguous_windowed_detail_judge"
    assert detail["missing_inputs"] == ["unambiguous_self_judge_manifest"]
    assert detail["candidate_self_judge_manifest_count"] == 2
    assert detail["docker_command"] == []
    assert detail["delivery_allowed"] is False


def test_materializer_blocks_detail_self_judge_manifest_without_source_audio_path(tmp_path: Path) -> None:
    plan, artifact_root = fixture_plan(tmp_path)
    minus_path = tmp_path / "minus" / "candidate.flac"
    minus_path.parent.mkdir(parents=True, exist_ok=True)
    minus_path.write_bytes(b"")
    rows_jsonl = write_jsonl(
        tmp_path / "self" / "rows.jsonl",
        [
            {
                "segment_id": "segment_0001",
                "window_sec": [5.0, 10.0],
                "candidate_label": "candidate",
                "issues": ["low_rms_muffling_proxy"],
                "issue_severities": {"low_rms_ratio": 0.71},
                "blocker": True,
                "severity_score": 0.71,
            }
        ],
    )
    self_manifest = write_json(
        tmp_path / "self" / "manifest.json",
        {
            "schema_version": 1,
            "mode": "instrumental_self_judge_gate",
            "segments_jsonl": str(rows_jsonl),
            "overall": {"status": "proxy_blocked", "blocker_count": 1, "segment_count": 1},
            "candidate": {
                "label": "candidate",
                "instrumental_path": str(minus_path),
                "vocal_path": str(artifact_root / "bci" / "vocals.flac"),
            },
            "delivery_contract": {"delivery_allowed": False},
        },
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/materialize_mac_promising_repair_commands.py",
            "--plan-json",
            str(plan),
            "--output-dir",
            str(output_dir),
            "--minus-self-judge-manifest",
            str(self_manifest),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    manifest = json.loads((output_dir / "cmds.json").read_text(encoding="utf-8"))
    assert manifest["command_plan_status"] == "partial_ready_missing_inputs"
    assert manifest["failed_checks"] == ["detail_route_windowed_judge_missing"]
    command_rows = rows(output_dir / "rows.jsonl")
    detail = {row["route"]: row for row in command_rows}["judge_guided_minus_air_detail_repair"]
    assert detail["command_status"] == "blocked_missing_windowed_detail_judge"
    assert detail["missing_inputs"] == ["self_judge_source_audio_path"]
    assert detail["docker_command"] == []
    assert detail["delivery_allowed"] is False


def test_materializer_keeps_delta_model_candidate_router_out_of_mac_handoff_scope(tmp_path: Path) -> None:
    plan, _artifact_root = fixture_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    rows_path = Path(plan_payload["rows_jsonl"])
    row = rows(rows_path)[0]
    row["recommended_routes"] = [
        {
            "issue_category": "residual_vocal_leak",
            "route": "no_reference_minus_candidate_model_router",
            "route_status": "delta_router_route",
            "tool": "tools/materialize_minus_delta_model_candidate_route.py",
            "next_action": "route_to_alternative_model_candidate",
        }
    ]
    write_jsonl(rows_path, [row])
    output_dir = tmp_path / "out"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.materialize_mac_promising_repair_commands import materialize_mac_promising_repair_commands

    manifest = materialize_mac_promising_repair_commands(plan_json=plan, output_dir=output_dir)
    assert manifest["command_plan_status"] == "blocked_missing_inputs"
    assert manifest["ready_command_count"] == 0
    assert manifest["blocked_command_count"] == 1
    assert manifest["failed_checks"] == ["unsupported_route:no_reference_minus_candidate_model_router"]
    command_rows = rows(output_dir / "rows.jsonl")
    assert len(command_rows) == 1
    row = command_rows[0]
    assert row["route"] == "no_reference_minus_candidate_model_router"
    assert row["command_status"] == "blocked_unsupported_route"
    assert row["missing_inputs"] == ["supported_route_mapping"]
    assert row["docker_command"] == []
    assert row["delivery_allowed"] is False
