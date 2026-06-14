from __future__ import annotations

import json
from pathlib import Path

from tools.plan_generic_audio_finetune_jobs import main, plan_generic_audio_finetune_jobs


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audio_row(tmp_path: Path, row_id: str, track_id: str, stem_type: str = "vocal") -> dict:
    source_path = tmp_path / "source" / f"{track_id}.flac"
    target_path = tmp_path / "target" / f"{row_id}.flac"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"target")
    return {
        "id": row_id,
        "track_id": track_id,
        "stem_type": stem_type,
        "source_audio_path": str(source_path),
        "target_audio_path": str(target_path),
        "target_audio_sha256": f"sha-{row_id}",
        "permission_status": "project_owned",
        "trainable": True,
    }


def test_blocks_audio_finetune_when_unique_tracks_are_insufficient(tmp_path: Path) -> None:
    rows = write_jsonl(
        tmp_path / "audio.jsonl",
        [audio_row(tmp_path, "v1", "track-a", "vocal"), audio_row(tmp_path, "m1", "track-a", "minus")],
    )

    manifest = plan_generic_audio_finetune_jobs(
        audio_training_examples_jsonl=rows,
        output_dir=tmp_path / "out",
        min_unique_tracks=3,
    )

    assert manifest["summary"]["trainable_audio_examples"] == 2
    assert manifest["summary"]["ready_job_count"] == 0
    assert manifest["summary"]["blocked_job_count"] == 2
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["release_ready"] is False
    assert {job["status"] for job in manifest["jobs"]} == {"blocked_insufficient_unique_tracks"}


def test_plans_gpu_teacher_mask_jobs_when_enough_tracks_exist(tmp_path: Path) -> None:
    rows = write_jsonl(
        tmp_path / "audio.jsonl",
        [audio_row(tmp_path, f"v{idx}", f"track-{idx}", "vocal") for idx in range(3)],
    )

    manifest = plan_generic_audio_finetune_jobs(
        audio_training_examples_jsonl=rows,
        output_dir=tmp_path / "out",
        min_unique_tracks=3,
    )

    assert manifest["summary"]["ready_job_count"] == 1
    job = manifest["jobs"][0]
    assert job["status"] == "ready_for_gpu_training"
    assert job["accelerator_mode"] == "gpu_training_required"
    assert job["device"] == "cuda"
    assert all("--device" in example["command"] for example in job["example_jobs"])
    assert all("cuda" in example["command"] for example in job["example_jobs"])
    command = job["example_jobs"][0]["command"]
    assert "--source-mix" in command
    assert "--teacher-target" in command
    assert "--stem" in command
    assert "--source-mix-path" not in command
    assert "--teacher-target-path" not in command
    assert "--stem-name" not in command
    assert manifest["training_started"] is False
    assert manifest["model_registry_edited"] is False


def test_cli_writes_audio_finetune_plan(tmp_path: Path) -> None:
    rows = write_jsonl(tmp_path / "audio.jsonl", [audio_row(tmp_path, "v1", "track-a", "vocal")])

    assert main(["--audio-training-examples-jsonl", str(rows), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / "generic_audio_finetune_job_plan.json")
    assert manifest["summary"]["trainable_audio_examples"] == 1
    assert manifest["outputs"]["progress_jsonl"].endswith("generic_audio_finetune_job_plan_progress.jsonl")


def test_missing_audio_paths_are_not_trainable(tmp_path: Path) -> None:
    row = audio_row(tmp_path, "v1", "track-a", "vocal")
    Path(row["target_audio_path"]).unlink()
    rows = write_jsonl(tmp_path / "audio.jsonl", [row])

    manifest = plan_generic_audio_finetune_jobs(
        audio_training_examples_jsonl=rows,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    assert manifest["summary"]["trainable_audio_examples"] == 0
    assert manifest["summary"]["job_count"] == 0
