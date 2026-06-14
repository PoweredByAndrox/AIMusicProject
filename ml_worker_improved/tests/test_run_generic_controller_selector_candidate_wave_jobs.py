from __future__ import annotations

import json
from pathlib import Path

from tools.plan_generic_controller_selector_candidate_wave import (
    CANDIDATE_WAVE_JOBS_NAME,
    CANDIDATE_WAVE_MANIFEST_NAME,
)
from tools.run_generic_controller_selector_candidate_wave_jobs import (
    CANDIDATE_WAVE_EXECUTION_JOB_MATRIX_NAME,
    CANDIDATE_WAVE_EXECUTION_MANIFEST_NAME,
    CANDIDATE_WAVE_EXECUTION_PROGRESS_NAME,
    run_candidate_wave_jobs,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_candidate_wave_plan(tmp_path: Path) -> Path:
    source = tmp_path / "song.wav"
    source.write_bytes(b"fake wav")
    output_dir = tmp_path / "planned_candidate"
    expected_manifest = output_dir / "stem_manifest.json"
    jobs_jsonl = write_jsonl(
        tmp_path / "candidate_wave" / CANDIDATE_WAVE_JOBS_NAME,
        [
            {
                "schema": "generic_controller_selector_candidate_wave_job.v1",
                "job_id": "job-a",
                "job_index": 1,
                "job_status": "planned",
                "execution_ready": True,
                "blockers": [],
                "wave_intent": "generate_new_candidate_wave",
                "stem_type": "minus",
                "target_stem": "minus",
                "route_id": "htdemucs_6s",
                "model_id": "htdemucs_6s",
                "execution_backend": "registered_model_candidate",
                "route_contract": "registered_model_candidate_manifest_required",
                "source_audio_path": str(source),
                "source_audio_exists": True,
                "track_id": "track-a",
                "output_dir": str(output_dir),
                "expected_candidate_manifest": str(expected_manifest),
                "command": [
                    "python",
                    "/app/tools/run_registered_model_candidate.py",
                    str(source),
                    str(output_dir),
                    "--model-id",
                    "htdemucs_6s",
                    "--device",
                    "cuda",
                    "--target-stem",
                    "minus",
                ],
                "gpu_inference_required": True,
                "expected_acceleration_mode": "gpu_inference_candidate_generation",
                "required_downstream_gates": ["candidate_execution_preflight"],
                "release_ready": False,
            }
        ],
    )
    return write_json(
        tmp_path / "candidate_wave" / CANDIDATE_WAVE_MANIFEST_NAME,
        {
            "schema": "generic_controller_selector_candidate_wave_manifest.v1",
            "manifest_type": "generic_controller_selector_candidate_wave_plan",
            "fingerprint": "candidate-wave-fingerprint",
            "summary": {"planned_job_count": 1, "blocked_job_count": 0},
            "promotion": {"status": "candidate_wave_plan_ready"},
            "outputs": {"jobs_jsonl": str(jobs_jsonl)},
            "release_ready": False,
        },
    )


def test_run_candidate_wave_jobs_reuses_existing_execution_contract(tmp_path: Path) -> None:
    plan_manifest = build_candidate_wave_plan(tmp_path)
    executed_commands: list[list[str]] = []

    def fake_runner(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        executed_commands.append(command)
        output_dir = Path(command[3])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "minus.flac").write_text("stub minus", encoding="utf-8")
        (output_dir / "stem_manifest.json").write_text(
            json.dumps(
                {
                    "target_stem": "minus",
                    "target_stem_artifact": {
                        "target_stem": "minus",
                        "path": str(output_dir / "minus.flac"),
                        "source": "test_fake_runner",
                        "source_stems": ["instrumental"],
                    },
                    "stems": {"minus": {"path": str(output_dir / "minus.flac")}},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    result = run_candidate_wave_jobs(
        candidate_wave_manifest_json=plan_manifest,
        output_root=tmp_path / "execution",
        max_jobs=1,
        command_runner=fake_runner,
    )

    assert result["summary"]["selected_jobs"] == 1
    assert result["summary"]["completed_jobs"] == 1
    assert result["summary"]["audio_inference_started"] is True
    assert result["manifest"]["execution_status"] == "completed"
    assert result["manifest"]["training_started"] is False
    assert result["manifest"]["release_ready"] is False
    assert result["rows"][0]["gpu_inference_skipped"] is False
    assert executed_commands[0][:2] == ["python", "/app/tools/run_registered_model_candidate.py"]

    normalized_rows = read_jsonl(tmp_path / "execution" / CANDIDATE_WAVE_EXECUTION_JOB_MATRIX_NAME)
    assert normalized_rows[0]["job_status"] == "queued"
    assert normalized_rows[0]["candidate_wave_job_status"] == "planned"
    assert normalized_rows[0]["gpu_inference_skipped"] is False

    disk_manifest = read_json(tmp_path / "execution" / CANDIDATE_WAVE_EXECUTION_MANIFEST_NAME)
    assert disk_manifest["manifest"]["execution_status"] == "completed"
    assert [row["event"] for row in read_jsonl(tmp_path / "execution" / CANDIDATE_WAVE_EXECUTION_PROGRESS_NAME)] == [
        "start",
        "normalize_jobs",
        "execute_jobs",
        "write_outputs",
        "finish",
    ]
