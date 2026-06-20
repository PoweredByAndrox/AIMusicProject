from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_aris_top_route_work_order import run_aris_top_route_work_order


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_bytes(path: Path, payload: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _work_order(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "aris_top_route_work_order.v1",
            "status": "ready_for_gpu_candidate_generation",
            "selected_route_ids": ["holdout_error_map_segment_router"],
            "work_items": [
                {
                    "id": "w001",
                    "route_id": "holdout_error_map_segment_router",
                    "candidate_matrix": {
                        "families": ["scnet", "bs_roformer"],
                        "stems": ["vocal", "minus"],
                        "pair_required": True,
                    },
                    "required_gates": ["reference_alignment_gate", "no_reference_generalization_gate"],
                    "delivery_allowed": False,
                }
            ],
            "execution_contract": {
                "docker_only": True,
                "gpu_candidate_generation_required": True,
                "paired_vocal_minus_required": True,
                "rejected_artifacts_not_deliverable": True,
            },
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
        },
    )


def _fake_runner(command: list[str], *, output_dir: Path, job: dict) -> int:
    assert command[:2] == ["/usr/bin/python", "/app/tools/run_registered_model_candidate.py"]
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cuda"
    assert "--target-stem" in command
    assert command[command.index("--target-stem") + 1] == job["target_stem"]
    target_audio_path = output_dir / f"{job['target_stem']}.flac"
    _write_bytes(target_audio_path, b"stem")
    manifest_path = output_dir / "registry_candidate_execution_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "registered_model_candidate_execution_manifest.v1",
            "status": "completed",
            "model_id": job["model_id"],
            "candidate_manifest": {
                "target_stem": job["target_stem"],
                "target_audio_path": str(target_audio_path),
            },
            "delivery_allowed": False,
            "release_ready": False,
        },
    )
    return 0


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    root = tmp_path / "artifact_rag"
    order = _work_order(root / "work" / "r001" / "order.json")
    corpus = tmp_path / "corpus"
    _write_bytes(corpus / "song.mp3")

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        run_aris_top_route_work_order(
            artifact_rag_root=root,
            work_order_json=order,
            corpus_dir=corpus,
            output_dir=tmp_path / "outside",
            max_jobs=1,
            command_runner=_fake_runner,
        )


def test_runs_locked_work_order_jobs_without_delivery(tmp_path: Path) -> None:
    root = tmp_path / "artifact_rag"
    order = _work_order(root / "work" / "r001" / "order.json")
    corpus = tmp_path / "corpus"
    _write_bytes(corpus / "raw title.mp3", b"source")

    manifest = run_aris_top_route_work_order(
        artifact_rag_root=root,
        work_order_json=order,
        corpus_dir=corpus,
        output_dir=root / "exec" / "r001",
        max_jobs=2,
        command_runner=_fake_runner,
        command_args=["--max-jobs", "2"],
    )

    assert manifest["status"] == "completed"
    assert manifest["job_count"] == 2
    assert manifest["completed_count"] == 2
    assert manifest["failed_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["matrix_gate_required_before_delivery"] is True

    jobs = _read_jsonl(Path(manifest["jobs_jsonl"]))
    assert [job["target_stem"] for job in jobs] == ["vocal", "minus"]
    assert [job["model_id"] for job in jobs] == ["scnet_xl_ihf_musdb18", "scnet_xl_ihf_musdb18"]
    assert all(job["delivery_allowed"] is False for job in jobs)
    assert all(job["candidate_status"] == "generated_pending_gate" for job in jobs)
    assert all(job["binary_audio_available"] is True for job in jobs)
    assert all(job["metadata_only"] is False for job in jobs)
    assert all(len(part) <= 32 for job in jobs for part in Path(job["relative_output_dir"]).parts)

    progress = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert progress[0]["event"] == "run_start"
    assert any(row["event"] == "job_start" for row in progress)
    assert progress[-1]["event"] == "run_finish"
    assert all(row["expected_acceleration_mode"] == "gpu_candidate_generation" for row in progress)
    assert all(row["gpu_inference_required"] is True for row in progress)


def test_reuse_existing_execution_manifest_without_rerunning_command(tmp_path: Path) -> None:
    root = tmp_path / "artifact_rag"
    order = _work_order(root / "work" / "r001" / "order.json")
    corpus = tmp_path / "corpus"
    _write_bytes(corpus / "raw title.mp3", b"source")
    existing_output = root / "exec" / "r001" / "jobs" / "t001" / "w001" / "f001" / "vocal"
    target_audio_path = _write_bytes(existing_output / "vocal.flac", b"stem")
    _write_json(
        existing_output / "registry_candidate_execution_manifest.json",
        {
            "schema": "registered_model_candidate_execution_manifest.v1",
            "status": "completed",
            "candidate_manifest": {
                "target_stem": "vocal",
                "target_audio_path": str(target_audio_path),
            },
        },
    )

    def forbidden_runner(command: list[str], *, output_dir: Path, job: dict) -> int:
        raise AssertionError("runner should not be called when reuse_existing is enabled")

    manifest = run_aris_top_route_work_order(
        artifact_rag_root=root,
        work_order_json=order,
        corpus_dir=corpus,
        output_dir=root / "exec" / "r001",
        max_jobs=1,
        reuse_existing=True,
        command_runner=forbidden_runner,
    )

    jobs = _read_jsonl(Path(manifest["jobs_jsonl"]))
    assert manifest["status"] == "completed"
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["binary_audio_available"] is True
    progress = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert any(row["event"] == "job_reuse" for row in progress)
