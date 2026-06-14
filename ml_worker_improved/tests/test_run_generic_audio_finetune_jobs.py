from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_generic_audio_finetune_jobs import run_generic_audio_finetune_jobs


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_runs_only_ready_gpu_jobs_and_records_validated_outputs(tmp_path: Path) -> None:
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text(
        "\n".join(
            [
                "import argparse, json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--output-dir', required=True)",
                "args = parser.parse_args()",
                "out = Path(args.output_dir)",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'teacher_mask_separator_progress.jsonl').write_text("
                "json.dumps({'event':'start'}) + '\\n' + json.dumps({'event':'finish'}) + '\\n', encoding='utf-8')",
                "manifest = {",
                "  'training': {'checkpoint_path': str(out / 'model.pt'), 'checkpoint_sha256': 'sha', 'diagnostics': {'device_used': 1.0}},",
                "  'presentation': {'result_delivery_allowed': False},",
                "  'stems': {'vocal': {'path': str(out / '01_teacher_mask_vocal.flac')}}",
                "}",
                "(out / 'model.pt').write_bytes(b'checkpoint')",
                "(out / '01_teacher_mask_vocal.flac').write_bytes(b'audio')",
                "(out / 'stem_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    ready_out = tmp_path / "ready_out"
    blocked_out = tmp_path / "blocked_out"
    plan = write_json(
        tmp_path / "plan.json",
        {
            "jobs": [
                {
                    "scope": "blocked_scope",
                    "status": "blocked_insufficient_unique_tracks",
                    "example_jobs": [
                        {
                            "example_id": "blocked",
                            "track_id": "blocked-track",
                            "command": ["python", str(fake_trainer), "--output-dir", str(blocked_out)],
                        }
                    ],
                },
                {
                    "scope": "vocal_teacher_mask_adapter",
                    "status": "ready_for_gpu_training",
                    "example_jobs": [
                        {
                            "example_id": "ready",
                            "track_id": "ready-track",
                            "command": ["python", str(fake_trainer), "--output-dir", str(ready_out)],
                        }
                    ],
                },
            ]
        },
    )

    manifest = run_generic_audio_finetune_jobs(
        job_plan_json=plan,
        output_dir=tmp_path / "run",
        scope="vocal_teacher_mask_adapter",
    )

    assert manifest["run_example_count"] == 1
    assert manifest["failed_example_count"] == 0
    assert manifest["all_examples_used_gpu"] is True
    assert manifest["all_progress_complete"] is True
    assert manifest["training_started"] is True
    assert not blocked_out.exists()
    rows = [
        json.loads(line)
        for line in Path(manifest["run_rows_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["example_id"] == "ready"
    assert rows[0]["status"] == "executed"


def test_skip_existing_complete_validates_without_reexecuting_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir(parents=True)
    (output_dir / "teacher_mask_separator_progress.jsonl").write_text(
        json.dumps({"event": "start"}) + "\n" + json.dumps({"event": "finish"}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "model.pt").write_bytes(b"checkpoint")
    (output_dir / "01_teacher_mask_vocal.flac").write_bytes(b"audio")
    write_json(
        output_dir / "stem_manifest.json",
        {
            "training": {
                "checkpoint_path": str(output_dir / "model.pt"),
                "checkpoint_sha256": "sha",
                "diagnostics": {"device_used": 1.0},
            },
            "presentation": {"result_delivery_allowed": False},
            "stems": {"vocal": {"path": str(output_dir / "01_teacher_mask_vocal.flac")}},
        },
    )
    missing_command = tmp_path / "must_not_run.py"
    plan = write_json(
        tmp_path / "plan.json",
        {
            "jobs": [
                {
                    "scope": "vocal_teacher_mask_adapter",
                    "status": "ready_for_gpu_training",
                    "example_jobs": [
                        {
                            "example_id": "ready",
                            "track_id": "ready-track",
                            "command": ["python", str(missing_command), "--output-dir", str(output_dir)],
                        }
                    ],
                }
            ]
        },
    )

    manifest = run_generic_audio_finetune_jobs(
        job_plan_json=plan,
        output_dir=tmp_path / "run",
        scope="vocal_teacher_mask_adapter",
        skip_existing_complete=True,
    )

    assert manifest["run_example_count"] == 1
    assert manifest["executed_example_count"] == 0
    assert manifest["validated_existing_example_count"] == 1
    assert manifest["training_started"] is False
    rows = [
        json.loads(line)
        for line in Path(manifest["run_rows_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["status"] == "skipped_existing_complete"


def test_skip_existing_complete_rejects_missing_checkpoint_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir(parents=True)
    (output_dir / "teacher_mask_separator_progress.jsonl").write_text(
        json.dumps({"event": "start"}) + "\n" + json.dumps({"event": "finish"}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "01_teacher_mask_vocal.flac").write_bytes(b"audio")
    write_json(
        output_dir / "stem_manifest.json",
        {
            "training": {
                "checkpoint_path": str(output_dir / "missing_model.pt"),
                "checkpoint_sha256": "sha",
                "diagnostics": {"device_used": 1.0},
            },
            "presentation": {"result_delivery_allowed": False},
            "stems": {"vocal": {"path": str(output_dir / "01_teacher_mask_vocal.flac")}},
        },
    )
    plan = write_json(
        tmp_path / "plan.json",
        {
            "jobs": [
                {
                    "scope": "vocal_teacher_mask_adapter",
                    "status": "ready_for_gpu_training",
                    "example_jobs": [
                        {
                            "example_id": "ready",
                            "track_id": "ready-track",
                            "command": ["python", str(tmp_path / "must_not_run.py"), "--output-dir", str(output_dir)],
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="generic_audio_finetune_failed_examples:1"):
        run_generic_audio_finetune_jobs(
            job_plan_json=plan,
            output_dir=tmp_path / "run",
            scope="vocal_teacher_mask_adapter",
            skip_existing_complete=True,
        )


def test_allow_non_gpu_marks_progress_as_diagnostic_cpu(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir(parents=True)
    (output_dir / "teacher_mask_separator_progress.jsonl").write_text(
        json.dumps({"event": "start"}) + "\n" + json.dumps({"event": "finish"}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "model.pt").write_bytes(b"checkpoint")
    (output_dir / "01_teacher_mask_vocal.flac").write_bytes(b"audio")
    write_json(
        output_dir / "stem_manifest.json",
        {
            "training": {
                "checkpoint_path": str(output_dir / "model.pt"),
                "checkpoint_sha256": "sha",
                "diagnostics": {"device_used": 0.0},
            },
            "presentation": {"result_delivery_allowed": False},
            "stems": {"vocal": {"path": str(output_dir / "01_teacher_mask_vocal.flac")}},
        },
    )
    plan = write_json(
        tmp_path / "plan.json",
        {
            "jobs": [
                {
                    "scope": "vocal_teacher_mask_adapter",
                    "status": "ready_for_gpu_training",
                    "example_jobs": [
                        {
                            "example_id": "ready",
                            "track_id": "ready-track",
                            "command": ["python", str(tmp_path / "must_not_run.py"), "--output-dir", str(output_dir)],
                        }
                    ],
                }
            ]
        },
    )

    manifest = run_generic_audio_finetune_jobs(
        job_plan_json=plan,
        output_dir=tmp_path / "run",
        scope="vocal_teacher_mask_adapter",
        skip_existing_complete=True,
        require_gpu=False,
    )

    progress_rows = [
        json.loads(line)
        for line in Path(manifest["progress_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert progress_rows[0]["expected_acceleration_mode"] == "cpu_diagnostic_training"
    assert progress_rows[0]["gpu_training_required"] is False
    assert progress_rows[0]["gpu_training_skipped"] is True
