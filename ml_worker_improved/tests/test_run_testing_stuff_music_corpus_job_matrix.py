from __future__ import annotations

import json
from pathlib import Path

import tools.run_testing_stuff_music_corpus_job_matrix as executor
from worker.model_registry import ModelRegistry, ModelRegistryEntry


LONG_TRACK_ID = "baby_me_here_comes_the_hotstepper_evian_version_radio_record_baby_me_here_comes_the_hotstepper_e"


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def job_row(
    tmp_path: Path,
    *,
    track_id: str = "song_one",
    target_stem: str = "vocal",
    model_id: str = "demo_model",
    job_status: str = "queued",
) -> dict:
    source_audio = tmp_path / f"{track_id}.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    output_dir = tmp_path / "candidate_outputs" / track_id / target_stem / model_id
    return {
        "schema": "testing_stuff_music_corpus_job_row.v1",
        "track_index": 1,
        "track_id": track_id,
        "relative_path": f"{track_id}.mp3",
        "source_audio_path": str(source_audio),
        "target_stem": target_stem,
        "phase": "candidate_generation",
        "route_name": model_id,
        "route_family": "test",
        "route_contract": "registered_model_candidate_manifest_required",
        "model_id": model_id,
        "model_display_name": model_id,
        "job_status": job_status,
        "execution_ready": job_status == "queued",
        "planned_only": True,
        "execution_started": False,
        "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
        "gpu_inference_required": True,
        "gpu_inference_skipped": False,
        "output_dir": str(output_dir),
        "listen_dir": str(output_dir / "listen_diagnostic"),
        "command": [
            "python",
            "/app/tools/run_registered_model_candidate.py",
            str(source_audio),
            str(output_dir),
            "--model-id",
            model_id,
            "--device",
            "cuda",
            "--listen-dir",
            str(output_dir / "listen_diagnostic"),
        ],
        "expected_candidate_manifest": str(output_dir / "stem_manifest.json"),
        "execution_manifest_path": str(output_dir / "registry_candidate_execution_manifest.json"),
        "required_downstream_gates": ["candidate_manifest_contract", "mode_specific_score_matrix"],
        "training_allowed": False,
        "fine_tune_allowed": False,
        "can_start_training_now": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "delivery_allowed": False,
        "release_ready": False,
    }


def demo_registry(tmp_path: Path, *model_ids: str) -> ModelRegistry:
    entries = [
        ModelRegistryEntry(
            model_id=model_id,
            display_name=model_id,
            family="test",
            target_stems=("vocals", "instrumental", "melody", "drums", "bass", "other"),
            output_stems=("vocals", "instrumental", "melody", "drums", "bass", "other"),
            command_template=("python", "unused_direct_command.py"),
            expected_manifest="{output_dir}/stem_manifest.json",
        )
        for model_id in model_ids
    ]
    return ModelRegistry(entries=tuple(entries), runtime_root=tmp_path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dry_run_records_bounded_selection_without_executing_or_releasing(tmp_path: Path) -> None:
    jobs_path = write_jsonl(
        tmp_path / "jobs.jsonl",
        [
            job_row(tmp_path, track_id="song_one", target_stem="vocal", model_id="vocal_a"),
            job_row(tmp_path, track_id="song_two", target_stem="minus", model_id="minus_a"),
        ],
    )
    calls = []

    def reject_execution(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        calls.append(command)
        raise AssertionError("dry-run must not execute commands")

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        stems=["vocal"],
        dry_run=True,
        command_runner=reject_execution,
        registry=demo_registry(tmp_path, "vocal_a"),
    )

    assert calls == []
    assert result["summary"]["selected_jobs"] == 1
    assert result["summary"]["dry_run_jobs"] == 1
    assert result["summary"]["completed_jobs"] == 0
    assert result["manifest"]["execution_started"] is False
    assert result["manifest"]["execution_status"] == "dry_run"
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["release_ready"] is False
    assert result["rows"][0]["execution_status"] == "dry_run"
    assert result["rows"][0]["candidate_manifest"]["exists"] is False


def test_bounded_selection_covers_requested_stems_before_extra_models(tmp_path: Path) -> None:
    rows: list[dict] = []
    model_ids: list[str] = []
    for stem in ("bass", "drums", "melody"):
        for suffix in ("a", "b"):
            model_id = f"{stem}_{suffix}"
            model_ids.append(model_id)
            rows.append(job_row(tmp_path, target_stem=stem, model_id=model_id))
    for stem in ("vocal", "minus", "other"):
        model_id = f"{stem}_a"
        model_ids.append(model_id)
        rows.append(job_row(tmp_path, target_stem=stem, model_id=model_id))
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", rows)

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=6,
        stems=["vocal", "minus", "melody", "drums", "bass", "other"],
        dry_run=True,
        registry=demo_registry(tmp_path, *model_ids),
    )

    assert result["summary"]["selected_jobs"] == 6
    assert result["summary"]["target_stems"] == ["bass", "drums", "melody", "minus", "other", "vocal"]
    assert [row["target_stem"] for row in result["rows"]] == ["vocal", "minus", "melody", "drums", "bass", "other"]
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["release_ready"] is False


def test_executes_ready_job_writes_logs_progress_and_keeps_downstream_gate_closed(tmp_path: Path) -> None:
    row = job_row(tmp_path, track_id=LONG_TRACK_ID, target_stem="vocal", model_id="vocal_a")
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", [row])
    calls = []

    def fake_runner(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        calls.append(command)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("stdout ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        Path(row["expected_candidate_manifest"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["expected_candidate_manifest"]).write_text('{"status":"diagnostic_candidate"}\n', encoding="utf-8")
        return 0

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        dry_run=False,
        command_runner=fake_runner,
        registry=demo_registry(tmp_path, "vocal_a"),
    )

    assert calls == [row["command"]]
    assert result["summary"]["selected_jobs"] == 1
    assert result["summary"]["completed_jobs"] == 1
    assert result["manifest"]["execution_started"] is True
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["presentation_status"] == "diagnostic_execution_not_for_listen_pack"
    assert result["rows"][0]["execution_status"] == "completed"
    assert result["rows"][0]["execution_job_key"] == "j001"
    assert result["rows"][0]["track_id"] == LONG_TRACK_ID
    assert result["rows"][0]["candidate_manifest"]["exists"] is True
    stdout_path = Path(result["rows"][0]["logs"]["stdout_path"])
    assert stdout_path.read_text(encoding="utf-8") == "stdout ok\n"
    assert stdout_path.parent.name == "j001"
    assert len(stdout_path.parent.name) <= 32
    assert LONG_TRACK_ID not in str(stdout_path)

    top_level_files = {path.name for path in (tmp_path / "execution").iterdir() if path.is_file()}
    assert top_level_files == {
        executor.EXECUTION_RESULTS_NAME,
        executor.EXECUTION_MANIFEST_NAME,
        executor.EXECUTION_SUMMARY_NAME,
        executor.EXECUTION_PROGRESS_NAME,
    }
    assert all(len(name) <= 32 for name in top_level_files)
    assert (tmp_path / "execution" / executor.EXECUTION_RESULTS_NAME).exists()
    assert (tmp_path / "execution" / executor.EXECUTION_MANIFEST_NAME).exists()
    assert (tmp_path / "execution" / executor.EXECUTION_PROGRESS_NAME).exists()


def test_blocks_real_gpu_job_when_cuda_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    row = job_row(tmp_path, target_stem="vocal", model_id="vocal_a")
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", [row])
    monkeypatch.setattr(executor, "cuda_available", lambda: False)

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        dry_run=False,
        registry=demo_registry(tmp_path, "vocal_a"),
    )

    assert result["summary"]["selected_jobs"] == 1
    assert result["summary"]["blocked_preflight_jobs"] == 1
    assert result["summary"]["attempted_jobs"] == 0
    assert result["manifest"]["execution_status"] == "blocked_preflight"
    assert result["manifest"]["execution_started"] is False
    assert result["manifest"]["release_ready"] is False
    assert result["rows"][0]["execution_status"] == "blocked_unavailable_at_execution"
    assert result["rows"][0]["execution_started"] is False
    assert "cuda_unavailable" in result["rows"][0]["preflight"]["reasons"]


def test_completed_registered_job_rejects_stem_derived_minus_manifest(tmp_path: Path) -> None:
    row = job_row(tmp_path, target_stem="minus", model_id="minus_a")
    row["command"].extend(["--target-stem", "minus"])
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", [row])
    target_audio_path = Path(row["output_dir"]) / "minus.flac"

    def fake_runner(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("stdout ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        Path(row["expected_candidate_manifest"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["expected_candidate_manifest"]).write_text(
            json.dumps(
                {
                    "target_stem": "minus",
                    "target_stem_artifact": {
                        "path": str(target_audio_path),
                        "source": "sum_non_vocal_stems",
                        "source_stems": ["drums", "bass", "other"],
                    },
                    "stems": {"minus": {"path": str(target_audio_path)}},
                    "delivery_allowed": False,
                    "release_ready": False,
                }
            ),
            encoding="utf-8",
        )
        Path(row["execution_manifest_path"]).write_text(
            json.dumps(
                {
                    "status": "completed",
                    "candidate_manifest": {
                        "path": row["expected_candidate_manifest"],
                        "exists": True,
                        "target_stem": "minus",
                        "target_audio_path": str(target_audio_path),
                        "target_source": "sum_non_vocal_stems",
                        "target_source_stems": ["drums", "bass", "other"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        command_runner=fake_runner,
        registry=demo_registry(tmp_path, "minus_a"),
    )

    candidate_manifest = result["rows"][0]["candidate_manifest"]
    assert result["rows"][0]["execution_status"] == "failed_policy_rejected"
    assert result["summary"]["failed_jobs"] == 1
    assert result["manifest"]["execution_status"] == "failed"
    assert candidate_manifest["exists"] is True
    assert candidate_manifest["target_stem"] == "minus"
    assert candidate_manifest["target_audio_path"] == str(target_audio_path)
    assert candidate_manifest["target_source"] == "sum_non_vocal_stems"
    assert candidate_manifest["target_source_stems"] == ["drums", "bass", "other"]
    assert result["rows"][0]["error"]["type"] == "PolicyRejected"


def test_executes_minus_phase_repair_job_contract(tmp_path: Path) -> None:
    source_audio = tmp_path / "mix.wav"
    vocal_audio = tmp_path / "vocal.wav"
    minus_audio = tmp_path / "minus.wav"
    source_audio.write_bytes(b"audio-placeholder")
    vocal_audio.write_bytes(b"audio-placeholder")
    minus_audio.write_bytes(b"audio-placeholder")
    output_dir = tmp_path / "phase_repair"
    row = {
        **job_row(tmp_path, target_stem="minus", model_id="source_minus"),
        "source_audio_path": str(source_audio),
        "phase": "source_pool_phase_repair_candidate",
        "route_contract": "no_reference_minus_phase_repair_candidate",
        "route_family": "cpu_dsp_minus_phase_repair",
        "expected_acceleration_mode": "cpu_dsp_candidate_generation",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "output_dir": str(output_dir),
        "command": [
            "python",
            "/app/tools/run_no_reference_minus_phase_repair.py",
            "--mix-path",
            str(source_audio),
            "--vocal-path",
            str(vocal_audio),
            "--minus-path",
            str(minus_audio),
            "--output-dir",
            str(output_dir),
            "--track-id",
            "song_one",
            "--candidate-id",
            "candidate-a",
            "--minus-route-id",
            "source_minus",
        ],
        "expected_candidate_manifest": str(output_dir / "minus_phase_repair_manifest.json"),
        "execution_manifest_path": str(output_dir / "minus_phase_repair_manifest.json"),
    }
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", [row])

    def fake_runner(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("stdout ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(row["expected_candidate_manifest"]).write_text(
            json.dumps(
                {
                    "candidate_pair_jsonl": str(output_dir / "candidate_pair.jsonl"),
                    "delivery_allowed": False,
                    "release_ready": False,
                    "target_stem": "minus",
                    "target_stem_artifact": {
                        "path": str(output_dir / "minus_side_attenuate_25.flac"),
                        "source": "minus_phase_repair_variant_pool",
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        command_runner=fake_runner,
        registry=demo_registry(tmp_path, "source_minus"),
    )

    assert result["summary"]["completed_jobs"] == 1
    assert result["summary"]["expected_acceleration_mode"] == "cpu_dsp_candidate_generation"
    assert result["summary"]["expected_acceleration_modes"] == ["cpu_dsp_candidate_generation"]
    assert result["summary"]["gpu_inference_required"] is False
    assert result["rows"][0]["execution_status"] == "completed"
    assert result["rows"][0]["candidate_manifest"]["exists"] is True
    assert result["rows"][0]["candidate_manifest"]["target_stem"] == "minus"
    assert result["rows"][0]["candidate_manifest"]["target_audio_path"].endswith("minus_side_attenuate_25.flac")
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["release_ready"] is False


def test_skips_blocked_jobs_fail_closed_without_running_commands(tmp_path: Path) -> None:
    jobs_path = write_jsonl(
        tmp_path / "jobs.jsonl",
        [job_row(tmp_path, target_stem="bass", model_id="bass_a", job_status="blocked_model_unavailable")],
    )
    calls = []

    def reject_execution(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        calls.append(command)
        raise AssertionError("blocked jobs must not execute")

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        command_runner=reject_execution,
        registry=demo_registry(tmp_path, "vocal_a"),
    )

    assert calls == []
    assert result["summary"]["selected_jobs"] == 0
    assert result["summary"]["skipped_blocked_jobs"] == 1
    assert result["manifest"]["execution_started"] is False
    assert result["manifest"]["release_ready"] is False
    assert result["skipped_rows"][0]["skip_reason"] == "job_not_execution_ready"


def test_blocks_tampered_command_before_subprocess(tmp_path: Path) -> None:
    row = job_row(tmp_path, target_stem="vocal", model_id="vocal_a")
    row["command"] = ["python", "/app/tools/build_vocal_guided_minus.py", row["source_audio_path"], row["output_dir"]]
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", [row])
    calls = []

    def reject_execution(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        calls.append(command)
        raise AssertionError("tampered commands must not execute")

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        command_runner=reject_execution,
        registry=demo_registry(tmp_path, "vocal_a"),
    )

    assert calls == []
    assert result["summary"]["selected_jobs"] == 1
    assert result["summary"]["blocked_preflight_jobs"] == 1
    assert result["summary"]["completed_jobs"] == 0
    assert result["rows"][0]["execution_status"] == "blocked_preflight_contract"
    assert result["rows"][0]["preflight"]["status"] == "blocked"
    assert "unexpected_command_entrypoint" in result["rows"][0]["preflight"]["reasons"]
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["release_ready"] is False


def test_fail_fast_records_unattempted_selected_jobs(tmp_path: Path) -> None:
    first = job_row(tmp_path, track_id="song_one", target_stem="vocal", model_id="vocal_a")
    second = job_row(tmp_path, track_id="song_two", target_stem="vocal", model_id="vocal_b")
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", [first, second])

    def failing_runner(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("failed\n", encoding="utf-8")
        return 9

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=2,
        command_runner=failing_runner,
        fail_fast=True,
        registry=demo_registry(tmp_path, "vocal_a", "vocal_b"),
    )

    assert result["summary"]["selected_jobs"] == 2
    assert result["summary"]["attempted_jobs"] == 1
    assert result["summary"]["not_run_fail_fast_jobs"] == 1
    assert [row["execution_status"] for row in result["rows"]] == ["failed", "not_run_fail_fast"]
    assert result["rows"][1]["skip_reason"] == "fail_fast_after_previous_failure"
    assert result["manifest"]["release_ready"] is False


def test_blocks_direct_minus_when_runtime_availability_is_stale(tmp_path: Path, monkeypatch) -> None:
    row = job_row(tmp_path, target_stem="minus", model_id="instrumental_resurrection_minus")
    row["route_contract"] = "documented_clean_minus_baseline"
    row["command"] = ["python", "/app/tools/export_instrumental_resurrection_minus.py", row["source_audio_path"], row["output_dir"]]
    jobs_path = write_jsonl(tmp_path / "jobs.jsonl", [row])
    calls = []

    monkeypatch.setattr(
        executor,
        "minus_baseline_availability",
        lambda: {
            "status": "unavailable",
            "missing_paths": ["/app/vendor/missing.ckpt"],
            "missing_python_modules": [],
        },
    )

    def reject_execution(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        calls.append(command)
        raise AssertionError("unavailable minus route must not execute")

    result = executor.execute_job_matrix(
        job_matrix_jsonl=jobs_path,
        output_root=tmp_path / "execution",
        max_jobs=1,
        command_runner=reject_execution,
    )

    assert calls == []
    assert result["summary"]["blocked_preflight_jobs"] == 1
    assert result["rows"][0]["execution_status"] == "blocked_unavailable_at_execution"
    assert "minus_baseline_unavailable" in result["rows"][0]["preflight"]["reasons"]
    assert result["rows"][0]["preflight"]["availability"]["missing_paths"] == ["/app/vendor/missing.ckpt"]


def test_main_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(executor, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(executor.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        executor.main(["--job-matrix-jsonl", "/tmp/jobs.jsonl", "--output-root", "/tmp/out", "--max-jobs", "1"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "run_testing_stuff_music_corpus_job_matrix.py",
                "entrypoint_path": "/app/tools/run_testing_stuff_music_corpus_job_matrix.py",
            },
        )
    ]
