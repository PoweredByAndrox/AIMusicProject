from __future__ import annotations

import json
from pathlib import Path

from tools.run_registered_candidate_family_wave import (
    CandidateFamily,
    run_registered_candidate_family_wave,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audio(path: Path, payload: bytes = b"audio") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def _row(tmp_path: Path, row_id: str, stem: str = "vocal") -> dict:
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem,
        "source_audio_path": _audio(tmp_path / row_id / "mix.flac"),
        "target_audio_path": _audio(tmp_path / row_id / "target.flac"),
        "permission_status": "project_owned",
        "trainable": True,
        "release_ready": False,
        "delivery_allowed": False,
    }


def test_runs_one_registered_job_per_family_and_example(tmp_path: Path) -> None:
    examples_jsonl = _write_jsonl(tmp_path / "examples.jsonl", [_row(tmp_path, "a"), _row(tmp_path, "b")])
    calls: list[list[str]] = []

    def fake_runner(command: list[str], *, output_dir: Path, heartbeat) -> int:
        calls.append(command)
        heartbeat(0.25, None)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "registry_candidate_execution_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "candidate_manifest": {
                        "path": str(output_dir / "stem_manifest.json"),
                        "target_audio_path": str(output_dir / "vocals.flac"),
                    },
                    "release_ready": False,
                }
            ),
            encoding="utf-8",
        )
        return 0

    manifest = run_registered_candidate_family_wave(
        audio_training_examples_jsonl=examples_jsonl,
        families=[CandidateFamily("scnet", "scnet_xl_ihf_musdb18", tmp_path / "scnet")],
        output_dir=tmp_path / "wave",
        stem="vocal",
        include_example_ids={"a", "b"},
        run_command=fake_runner,
    )

    assert manifest["status"] == "passed"
    assert manifest["job_count"] == 2
    assert manifest["completed_count"] == 2
    assert manifest["failed_count"] == 0
    assert manifest["release_ready"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["safety"]["fallback_used"] is False
    assert manifest["safety"]["candidate_sources_are_not_ground_truth"] is True
    assert all("--model-id" in command for command in calls)
    assert all("--target-stem" in command and "vocal" in command for command in calls)

    jobs = _read_jsonl(Path(manifest["jobs_jsonl"]))
    assert [row["status"] for row in jobs] == ["completed", "completed"]
    assert {row["source_candidate_family"] for row in jobs} == {"scnet"}
    progress_events = [row["event"] for row in _read_jsonl(Path(manifest["progress_jsonl"]))]
    assert "job_heartbeat" in progress_events
    assert progress_events[-1] == "finish"


def test_records_failed_job_without_fallback_or_release(tmp_path: Path) -> None:
    examples_jsonl = _write_jsonl(tmp_path / "examples.jsonl", [_row(tmp_path, "a")])

    def fake_runner(command: list[str], *, output_dir: Path, heartbeat) -> int:
        heartbeat(0.1, None)
        return 7

    manifest = run_registered_candidate_family_wave(
        audio_training_examples_jsonl=examples_jsonl,
        families=[CandidateFamily("melband", "melband_roformer_vocals", tmp_path / "melband")],
        output_dir=tmp_path / "wave",
        stem="vocal",
        include_example_ids={"a"},
        continue_on_failure=True,
        run_command=fake_runner,
    )

    assert manifest["status"] == "failed"
    assert manifest["completed_count"] == 0
    assert manifest["failed_count"] == 1
    assert manifest["release_ready"] is False
    assert manifest["delivery_allowed"] is False
    jobs = _read_jsonl(Path(manifest["jobs_jsonl"]))
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["fallback_used"] is False
    assert jobs[0]["release_ready"] is False
