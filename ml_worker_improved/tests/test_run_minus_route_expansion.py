from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

from tools.no_reference_candidate_job_registry import CandidateJobConfig


def write_wav(path: Path, *, frequency: float = 220.0, seconds: float = 1.0, sample_rate: int = 8000) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(12000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(bytes(frames))
    return str(path)


def both_config(config_id: str = "local-both") -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id=config_id,
        candidate_family="balanced_pair_default",
        target_mode="both",
        source_model_id="htdemucs_ft",
        source_tool="tools/run_registered_model_candidate.py",
        entrypoint_type="python_script",
        command_template=("python", "/app/tools/run_registered_model_candidate.py"),
        expected_outputs={"vocal_path": "{output_dir}/vocals.wav", "minus_path": "{output_dir}/minus.wav"},
        gpu_required=True,
        timeout_sec=100,
        max_retries=0,
        profile_allowed=("safe", "balanced", "exploration"),
        safety_notes=("uses_existing_model_registry_entry", "no_model_download"),
        supported_parameters=("target_stem",),
        shared_execution_group_id="local-both-group",
        target_stem="minus",
        produces_vocal=True,
        produces_minus=True,
        materialized_candidate_modes=("vocal", "minus", "both"),
        reuse_outputs_across_modes=True,
    )


def clone_config(config: CandidateJobConfig, **updates: Any) -> CandidateJobConfig:
    payload = config.to_manifest()
    payload.update(updates)
    payload["command_template"] = tuple(payload["command_template"])
    payload["profile_allowed"] = tuple(payload["profile_allowed"])
    payload["safety_notes"] = tuple(payload["safety_notes"])
    payload["supported_parameters"] = tuple(payload["supported_parameters"])
    payload["materialized_candidate_modes"] = tuple(payload["materialized_candidate_modes"])
    payload["hard_case_routes"] = tuple(payload["hard_case_routes"])
    payload["expected_outputs"] = dict(payload["expected_outputs"])
    return CandidateJobConfig(**payload)


def discovery_json(path: Path, *, route_id: str = "local-both", candidate_config_id: str = "local-both") -> Path:
    payload = {
        "summary": {
            "discovered_routes_total": 1,
            "available_routes_total": 1,
            "unavailable_routes_total": 0,
            "route_space_limited": True,
            "release_ready": False,
        },
        "routes": [
            {
                "route_id": route_id,
                "candidate_config_id": candidate_config_id,
                "source_model_id": "htdemucs_ft",
                "source_tool": "tools/run_registered_model_candidate.py",
                "target_mode": "both",
                "route_family": "htdemucs_ft",
                "available": True,
                "supports_minus": True,
                "supports_both": True,
                "forbidden_reason": "",
                "can_run_without_download": True,
                "can_run_without_registry_edit": True,
                "release_ready": False,
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def fake_job_factory(*, blocker: str = ""):
    def fake_job(**kwargs: Any) -> dict[str, Any]:
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal = write_wav(output_dir / "vocals.wav", frequency=440.0)
        minus = write_wav(output_dir / "minus.wav", frequency=110.0)
        manifest = {
            "candidate_config_id": kwargs["candidate_config_id"],
            "candidate_family": "balanced_pair_default",
            "target_mode": "both",
            "source_model_id": "htdemucs_ft",
            "source_tool": "tools/run_registered_model_candidate.py",
            "source_config_hash": both_config(kwargs["candidate_config_id"]).source_config_hash,
            "generation_status": "generated",
            "skip_reason": "",
            "output_paths": {"vocal_path": vocal, "minus_path": minus},
            "raw_vocal_path": vocal,
            "raw_minus_path": minus,
            "normalized_vocal_path": vocal,
            "normalized_minus_path": minus,
            "shape_normalization_applied": False,
            "shape_normalization_reason": "shape_already_valid",
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }
        kwargs["job_manifest_json"].write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        kwargs["job_progress_jsonl"].write_text(json.dumps({"phase": "complete"}) + "\n", encoding="utf-8")
        return manifest

    return fake_job


def fake_judge_factory(*, blockers: list[str] | None = None):
    blockers = blockers or []

    def fake_judge(rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        judged = []
        for row in rows:
            judged.append(
                {
                    "track_id": row["track_id"],
                    "candidate_id": row["candidate_id"],
                    "minus_path": row["minus_path"],
                    "vocal_path": row["vocal_path"],
                    "minus_decision": "blocked_hard_case" if blockers else "accept_candidate",
                    "decision": "blocked_hard_case" if blockers else "accept_candidate",
                    "minus_blockers": blockers,
                    "reasons": blockers or ["no_reference_proxy_passed"],
                    "vocal_leakage_in_minus_score": 0.91 if blockers else 0.04,
                    "phase_consistency_score": 0.12 if "severe_phase_collapse" in blockers else 0.88,
                    "release_ready": False,
                }
            )
        output_jsonl = kwargs["output_jsonl"]
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in judged),
            encoding="utf-8",
        )
        kwargs["summary_md"].write_text("# judge\n", encoding="utf-8")
        kwargs["cache_jsonl"].write_text("", encoding="utf-8")
        return {"rows": judged, "summary": {"candidates_judged": len(judged), "release_ready": False}}

    return fake_judge


def test_route_expansion_writes_candidate_pool_and_preserves_release_ready_false(tmp_path: Path) -> None:
    from tools.run_minus_route_expansion import run_minus_route_expansion

    input_audio = Path(write_wav(tmp_path / "song.wav"))
    result = run_minus_route_expansion(
        input_audio=input_audio,
        output_dir=tmp_path / "out",
        route_discovery_json=discovery_json(tmp_path / "routes.json"),
        profile="hard_case",
        max_routes=1,
        max_runtime_sec=60,
        registry_configs=[both_config()],
        job_runner=fake_job_factory(),
        judge_writer=fake_judge_factory(),
    )

    assert result["summary"]["jobs_attempted"] == 1
    assert result["summary"]["jobs_generated"] == 1
    assert result["summary"]["minus_candidates_generated"] == 1
    assert result["summary"]["accepted_minus_candidates"] == 1
    assert result["summary"]["release_ready_count"] == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "minus_route_candidate_pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["accepted_minus"] is True
    assert rows[0]["release_ready"] is False
    assert Path(result["manifest_path"]).exists()
    progress_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "minus_route_expansion_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in progress_rows] == [
        "start",
        "route_start",
        "route_finish",
        "judge_start",
        "judge_finish",
        "finish",
    ]
    assert progress_rows[0]["expected_acceleration_mode"] == "gpu_registered_model_inference_route_expansion"
    assert progress_rows[0]["gpu_inference_required"] is True
    assert progress_rows[3]["expected_acceleration_mode"] == "external_only_cpu_dsp_scoring"
    assert progress_rows[3]["gpu_inference_skipped"] is True
    assert progress_rows[3]["gpu_inference_skip_reason"] == "minus_route_judge_reads_existing_audio_only"
    assert progress_rows[4]["expected_acceleration_mode"] == "external_only_cpu_dsp_scoring"
    assert progress_rows[4]["gpu_inference_skipped"] is True
    assert progress_rows[-1]["release_ready"] is False


def test_severe_vocal_bleed_in_minus_still_blocks_unsafe_minus(tmp_path: Path) -> None:
    from tools.run_minus_route_expansion import run_minus_route_expansion

    input_audio = Path(write_wav(tmp_path / "song.wav"))
    result = run_minus_route_expansion(
        input_audio=input_audio,
        output_dir=tmp_path / "out",
        route_discovery_json=discovery_json(tmp_path / "routes.json"),
        profile="hard_case",
        max_routes=1,
        max_runtime_sec=60,
        registry_configs=[both_config()],
        job_runner=fake_job_factory(),
        judge_writer=fake_judge_factory(blockers=["severe_vocal_bleed_in_minus"]),
    )

    assert result["summary"]["accepted_minus_candidates"] == 0
    assert result["summary"]["severe_vocal_bleed_in_minus_count"] == 1
    assert result["summary"]["route_space_exhausted"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "minus_route_candidate_pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["accepted_minus"] is False
    assert rows[0]["minus_blockers"] == ["severe_vocal_bleed_in_minus"]


def test_route_expansion_rejects_forbidden_cleanup_route(tmp_path: Path) -> None:
    from tools.run_minus_route_expansion import run_minus_route_expansion

    discovery = {
        "summary": {"release_ready": False},
        "routes": [
            {
                "route_id": "cleanup-route",
                "candidate_config_id": "local-both",
                "target_mode": "both",
                "supports_minus": True,
                "available": True,
                "forbidden_reason": "cleanup",
                "release_ready": False,
            }
        ],
    }
    route_json = tmp_path / "routes.json"
    route_json.write_text(json.dumps(discovery), encoding="utf-8")

    input_audio = Path(write_wav(tmp_path / "song.wav"))
    result = run_minus_route_expansion(
        input_audio=input_audio,
        output_dir=tmp_path / "out",
        route_discovery_json=route_json,
        profile="hard_case",
        max_routes=1,
        max_runtime_sec=60,
        registry_configs=[both_config()],
        job_runner=fake_job_factory(),
        judge_writer=fake_judge_factory(),
    )

    assert result["summary"]["jobs_attempted"] == 0
    assert result["summary"]["cleanup_routes_added"] is False
    assert result["summary"]["route_space_exhausted"] is True


def test_route_expansion_revalidates_external_discovery_against_current_registry(tmp_path: Path) -> None:
    from tools.run_minus_route_expansion import run_minus_route_expansion

    stale_discovery = discovery_json(tmp_path / "routes.json")
    unsafe_config = clone_config(
        both_config(),
        available=False,
        unavailable_reason="missing_path:models/htdemucs_ft.th",
        can_run_without_download=False,
    )

    input_audio = Path(write_wav(tmp_path / "song.wav"))
    result = run_minus_route_expansion(
        input_audio=input_audio,
        output_dir=tmp_path / "out",
        route_discovery_json=stale_discovery,
        profile="hard_case",
        max_routes=1,
        max_runtime_sec=60,
        registry_configs=[unsafe_config],
        job_runner=fake_job_factory(),
        judge_writer=fake_judge_factory(),
    )

    assert result["summary"]["jobs_attempted"] == 0
    assert result["summary"]["jobs_skipped"] == 1
    assert result["failures"][0]["skip_reason"].startswith("registry_route_unavailable")


def test_route_expansion_outputs_are_deterministic(tmp_path: Path) -> None:
    from tools.run_minus_route_expansion import run_minus_route_expansion

    input_audio = Path(write_wav(tmp_path / "song.wav"))
    kwargs = {
        "input_audio": input_audio,
        "route_discovery_json": discovery_json(tmp_path / "routes.json"),
        "profile": "hard_case",
        "max_routes": 1,
        "max_runtime_sec": 60,
        "registry_configs": [both_config()],
        "job_runner": fake_job_factory(),
        "judge_writer": fake_judge_factory(),
    }
    first = run_minus_route_expansion(output_dir=tmp_path / "a", **kwargs)
    second = run_minus_route_expansion(output_dir=tmp_path / "b", **kwargs)

    assert first["summary"]["selected_route_ids"] == second["summary"]["selected_route_ids"]
    assert first["summary"]["quality_improved_route_ids"] == second["summary"]["quality_improved_route_ids"]


def test_route_expansion_dedupes_shared_execution_groups() -> None:
    from tools.run_minus_route_expansion import selected_routes

    routes = [
        {
            "route_id": "both",
            "candidate_config_id": "both",
            "shared_execution_group_id": "shared-a",
            "supports_minus": True,
            "supports_both": True,
            "available": True,
            "forbidden_reason": "",
            "release_ready": False,
        },
        {
            "route_id": "minus",
            "candidate_config_id": "minus",
            "shared_execution_group_id": "shared-a",
            "supports_minus": True,
            "supports_both": False,
            "available": True,
            "forbidden_reason": "",
            "release_ready": False,
        },
    ]

    selected = selected_routes(routes, max_routes=4)

    assert [row["route_id"] for row in selected] == ["both"]
