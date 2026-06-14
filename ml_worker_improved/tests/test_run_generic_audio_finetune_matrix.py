from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_generic_audio_finetune_matrix import run_generic_audio_finetune_matrix


def _write_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_direct_script_help_runs_inside_docker() -> None:
    if not Path("/.dockerenv").exists():
        pytest.skip("direct project script execution is Docker-only")
    script = Path(__file__).resolve().parents[1] / "tools" / "run_generic_audio_finetune_matrix.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert "--config-matrix-json" in result.stdout


def test_matrix_trains_all_configs_validates_holdout_and_selects_best_without_release(tmp_path: Path, monkeypatch) -> None:
    examples = _write_json(tmp_path / "examples.jsonl", [])
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "configs": [
                {"id": "baseline", "epochs": 2, "activity_weight": 4.0},
                {"id": "knn_memory", "epochs": 2, "activity_weight": 8.0, "source_feature_memory_neighbors": 3},
            ]
        },
    )
    train_calls: list[dict] = []
    holdout_calls: list[dict] = []

    def fake_train(**kwargs):
        train_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "adapter.pt"
        checkpoint.write_bytes(b"checkpoint")
        run_rows = output_dir / "generic_audio_finetune_run_rows.jsonl"
        run_rows.write_text(
            json.dumps(
                {
                    "scope": kwargs["scope"],
                    "status": "executed",
                    "example_id": f"{kwargs['scope']}-adapter",
                    "track_id": f"{kwargs['scope']}-adapter",
                    "training_track_ids": ["track-a", "track-b"],
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": kwargs["scope"],
                    "device_used": 1.0,
                    "progress_first_event": "start",
                    "progress_last_event": "finish",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path = output_dir / "generic_audio_teacher_mask_adapter_manifest.json"
        manifest = {
            "adapter_id": f"{kwargs['scope']}-adapter",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": kwargs["scope"],
            "run_rows_jsonl": str(run_rows),
            "delivery_allowed": False,
            "release_ready": False,
        }
        _write_json(manifest_path, manifest)
        return manifest

    def fake_holdout(**kwargs):
        holdout_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        is_best = "knn_memory" in str(kwargs["scope"])
        rows_path = output_dir / "generic_audio_finetune_holdout_rows.jsonl"
        rows_path.write_text(
            json.dumps(
                {
                    "status": "passed" if is_best else "failed",
                    "failed_checks": [] if is_best else ["sdr_below_threshold"],
                        "metrics": {
                            "candidate_sdr_db": 4.5 if is_best else 1.0,
                            "improvement_db": 2.0 if is_best else 0.4,
                        },
                        "application_device_used": 1.0,
                        "delivery_allowed": False,
                        "release_ready": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "holdout_validation_status": "passed" if is_best else "failed",
            "holdout_eval_count": 1,
            "passed_eval_count": 1 if is_best else 0,
            "failed_eval_count": 0 if is_best else 1,
            "holdout_rows_jsonl": str(rows_path),
            "delivery_allowed": False,
            "release_ready": False,
        }
        _write_json(output_dir / "generic_audio_finetune_holdout_manifest.json", manifest)
        return manifest

    import tools.run_generic_audio_finetune_matrix as module

    monkeypatch.setattr(module, "train_generic_audio_teacher_mask_adapter", fake_train)
    monkeypatch.setattr(module, "validate_generic_audio_finetune_holdout", fake_holdout)

    manifest = run_generic_audio_finetune_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=tmp_path / "matrix_run",
        stem="vocal",
        scope="vocal_matrix",
        exclude_holdout_track_ids={"track-c"},
        min_unique_tracks=2,
        device="cuda",
        require_gpu=True,
        min_sdr_db=3.0,
        min_improvement_db=0.5,
        min_checkpoint_pass_rate=1.0,
        max_holdouts_per_checkpoint=2,
    )

    assert [call["scope"] for call in train_calls] == ["vocal_matrix__baseline", "vocal_matrix__knn_memory"]
    assert [call["source_feature_memory_neighbors"] for call in train_calls] == [0, 3]
    assert all(call["device"] == "cuda" for call in train_calls)
    assert train_calls[0]["command_args"][:4] == ["--matrix-config-id", "baseline", "--scope", "vocal_matrix__baseline"]
    assert holdout_calls[0]["command_args"][:4] == ["--matrix-config-id", "baseline", "--scope", "vocal_matrix__baseline"]
    assert all(call["require_gpu"] is True for call in holdout_calls)
    assert manifest["matrix_validation_status"] == "passed"
    assert manifest["best_config_id"] == "knn_memory"
    assert manifest["passed_config_count"] == 1
    assert manifest["failed_config_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["presentation_status"] == "training_matrix_artifact_not_for_listen_pack"

    rows = _read_jsonl(Path(manifest["matrix_rows_jsonl"]))
    assert [row["config_id"] for row in rows] == ["baseline", "knn_memory"]
    assert rows[0]["status"] == "failed"
    assert rows[1]["status"] == "passed"
    assert rows[1]["metrics"]["min_candidate_sdr_db"] == 4.5

    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == [
        "start",
        "config_start",
        "config_finish",
        "config_start",
        "config_finish",
        "finish",
    ]
    assert progress_rows[0]["expected_acceleration_mode"] == "gpu_training_and_holdout"
    assert progress_rows[0]["gpu_training_required"] is True


def test_matrix_fails_closed_when_no_config_passes_and_never_promotes(tmp_path: Path, monkeypatch) -> None:
    examples = _write_json(tmp_path / "examples.jsonl", [])
    matrix = _write_json(tmp_path / "matrix.json", [{"id": "bad"}])

    def fake_train(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "adapter.pt"
        checkpoint.write_bytes(b"checkpoint")
        run_rows = output_dir / "generic_audio_finetune_run_rows.jsonl"
        run_rows.write_text(
            json.dumps(
                {
                    "scope": kwargs["scope"],
                    "status": "executed",
                    "example_id": "bad-adapter",
                    "track_id": "bad-adapter",
                    "checkpoint_path": str(checkpoint),
                    "device_used": 1.0,
                    "progress_first_event": "start",
                    "progress_last_event": "finish",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"adapter_id": "bad-adapter", "checkpoint_path": str(checkpoint), "run_rows_jsonl": str(run_rows)}

    def fake_holdout(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        rows_path = output_dir / "generic_audio_finetune_holdout_rows.jsonl"
        rows_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_checks": ["sdr_below_threshold"],
                    "metrics": {"candidate_sdr_db": -1.0, "improvement_db": 0.1},
                    "delivery_allowed": False,
                    "release_ready": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "holdout_validation_status": "failed",
            "holdout_eval_count": 1,
            "passed_eval_count": 0,
            "failed_eval_count": 1,
            "holdout_rows_jsonl": str(rows_path),
            "delivery_allowed": False,
            "release_ready": False,
        }

    import tools.run_generic_audio_finetune_matrix as module

    monkeypatch.setattr(module, "train_generic_audio_teacher_mask_adapter", fake_train)
    monkeypatch.setattr(module, "validate_generic_audio_finetune_holdout", fake_holdout)

    manifest = run_generic_audio_finetune_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=tmp_path / "matrix_run",
        stem="vocal",
        scope="vocal_matrix",
    )

    assert manifest["matrix_validation_status"] == "failed"
    assert manifest["best_config_id"] == ""
    assert manifest["passed_config_count"] == 0
    assert manifest["failed_config_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["release_status"] == "blocked_until_holdout_and_promotion_gates"


def test_matrix_does_not_trust_unsafe_or_empty_holdout_pass_manifest(tmp_path: Path, monkeypatch) -> None:
    examples = _write_json(tmp_path / "examples.jsonl", [])
    matrix = _write_json(tmp_path / "matrix.json", [{"id": "unsafe"}])

    def fake_train(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "adapter.pt"
        checkpoint.write_bytes(b"checkpoint")
        run_rows = output_dir / "generic_audio_finetune_run_rows.jsonl"
        run_rows.write_text(
            json.dumps(
                {
                    "scope": kwargs["scope"],
                    "status": "executed",
                    "example_id": "unsafe-adapter",
                    "track_id": "unsafe-adapter",
                    "checkpoint_path": str(checkpoint),
                    "device_used": 1.0,
                    "progress_first_event": "start",
                    "progress_last_event": "finish",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "adapter_id": "unsafe-adapter",
            "checkpoint_path": str(checkpoint),
            "run_rows_jsonl": str(run_rows),
            "delivery_allowed": True,
            "release_ready": True,
        }

    def fake_holdout(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        rows_path = output_dir / "generic_audio_finetune_holdout_rows.jsonl"
        rows_path.write_text("", encoding="utf-8")
        return {
            "holdout_validation_status": "passed",
            "holdout_eval_count": 0,
            "passed_eval_count": 0,
            "failed_eval_count": 0,
            "holdout_rows_jsonl": str(rows_path),
            "delivery_allowed": True,
            "release_ready": True,
        }

    import tools.run_generic_audio_finetune_matrix as module

    monkeypatch.setattr(module, "train_generic_audio_teacher_mask_adapter", fake_train)
    monkeypatch.setattr(module, "validate_generic_audio_finetune_holdout", fake_holdout)

    manifest = run_generic_audio_finetune_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=tmp_path / "matrix_run",
        stem="vocal",
        scope="vocal_matrix",
    )

    rows = _read_jsonl(Path(manifest["matrix_rows_jsonl"]))
    assert manifest["matrix_validation_status"] == "failed"
    assert rows[0]["status"] == "failed"
    assert "unsafe_training_delivery_contract" in rows[0]["failed_checks"]
    assert "holdout_rows_missing" in rows[0]["failed_checks"]
    assert "unsafe_holdout_delivery_contract" in rows[0]["failed_checks"]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_matrix_progress_log_is_reset_on_fresh_run(tmp_path: Path, monkeypatch) -> None:
    examples = _write_json(tmp_path / "examples.jsonl", [])
    matrix = _write_json(tmp_path / "matrix.json", [{"id": "fresh"}])
    output_dir = tmp_path / "matrix_run"
    output_dir.mkdir(parents=True)
    progress_path = output_dir / "generic_audio_finetune_matrix_progress.jsonl"
    progress_path.write_text(json.dumps({"event": "stale_previous_run"}) + "\n", encoding="utf-8")

    def fake_train(**kwargs):
        train_dir = Path(kwargs["output_dir"])
        train_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = train_dir / "adapter.pt"
        checkpoint.write_bytes(b"checkpoint")
        run_rows = train_dir / "generic_audio_finetune_run_rows.jsonl"
        run_rows.write_text(
            json.dumps(
                {
                    "scope": kwargs["scope"],
                    "status": "executed",
                    "example_id": "fresh-adapter",
                    "track_id": "fresh-adapter",
                    "checkpoint_path": str(checkpoint),
                    "device_used": 1.0,
                    "progress_first_event": "start",
                    "progress_last_event": "finish",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"adapter_id": "fresh-adapter", "checkpoint_path": str(checkpoint), "run_rows_jsonl": str(run_rows)}

    def fake_holdout(**kwargs):
        holdout_dir = Path(kwargs["output_dir"])
        holdout_dir.mkdir(parents=True, exist_ok=True)
        rows_path = holdout_dir / "generic_audio_finetune_holdout_rows.jsonl"
        rows_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "failed_checks": [],
                    "metrics": {"candidate_sdr_db": 4.0, "improvement_db": 1.0},
                    "application_device_used": 1.0,
                    "delivery_allowed": False,
                    "release_ready": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "holdout_validation_status": "passed",
            "holdout_eval_count": 1,
            "passed_eval_count": 1,
            "failed_eval_count": 0,
            "holdout_rows_jsonl": str(rows_path),
            "delivery_allowed": False,
            "release_ready": False,
        }

    import tools.run_generic_audio_finetune_matrix as module

    monkeypatch.setattr(module, "train_generic_audio_teacher_mask_adapter", fake_train)
    monkeypatch.setattr(module, "validate_generic_audio_finetune_holdout", fake_holdout)

    manifest = run_generic_audio_finetune_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=output_dir,
        stem="vocal",
        scope="vocal_matrix",
    )

    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert progress_rows[0]["event"] == "start"
    assert all(row["event"] != "stale_previous_run" for row in progress_rows)


def test_matrix_recomputes_holdout_row_contract_before_passing_config(tmp_path: Path, monkeypatch) -> None:
    examples = _write_json(tmp_path / "examples.jsonl", [])
    matrix = _write_json(tmp_path / "matrix.json", [{"id": "forged"}])

    def fake_train(**kwargs):
        train_dir = Path(kwargs["output_dir"])
        train_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = train_dir / "adapter.pt"
        checkpoint.write_bytes(b"checkpoint")
        run_rows = train_dir / "generic_audio_finetune_run_rows.jsonl"
        run_rows.write_text(
            json.dumps(
                {
                    "scope": kwargs["scope"],
                    "status": "executed",
                    "example_id": "forged-adapter",
                    "track_id": "forged-adapter",
                    "checkpoint_path": str(checkpoint),
                    "device_used": 1.0,
                    "progress_first_event": "start",
                    "progress_last_event": "finish",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"adapter_id": "forged-adapter", "checkpoint_path": str(checkpoint), "run_rows_jsonl": str(run_rows)}

    def fake_holdout(**kwargs):
        holdout_dir = Path(kwargs["output_dir"])
        holdout_dir.mkdir(parents=True, exist_ok=True)
        rows_path = holdout_dir / "generic_audio_finetune_holdout_rows.jsonl"
        rows_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "status": "passed",
                            "failed_checks": [],
                            "metrics": {"candidate_sdr_db": 4.0},
                            "application_device_used": 1.0,
                            "delivery_allowed": False,
                            "release_ready": False,
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "status": "passed",
                            "failed_checks": [],
                            "metrics": {"candidate_sdr_db": 2.0, "improvement_db": 0.1},
                            "application_device_used": 0.0,
                            "delivery_allowed": False,
                            "release_ready": False,
                        },
                        sort_keys=True,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "holdout_validation_status": "passed",
            "holdout_eval_count": 2,
            "passed_eval_count": 2,
            "failed_eval_count": 0,
            "holdout_rows_jsonl": str(rows_path),
            "delivery_allowed": False,
            "release_ready": False,
        }

    import tools.run_generic_audio_finetune_matrix as module

    monkeypatch.setattr(module, "train_generic_audio_teacher_mask_adapter", fake_train)
    monkeypatch.setattr(module, "validate_generic_audio_finetune_holdout", fake_holdout)

    manifest = run_generic_audio_finetune_matrix(
        audio_training_examples_jsonl=examples,
        config_matrix_json=matrix,
        output_dir=tmp_path / "matrix_run",
        stem="vocal",
        scope="vocal_matrix",
        require_gpu=True,
        min_sdr_db=3.0,
        min_improvement_db=0.5,
    )

    rows = _read_jsonl(Path(manifest["matrix_rows_jsonl"]))
    assert manifest["matrix_validation_status"] == "failed"
    assert rows[0]["status"] == "failed"
    assert "holdout_metrics_missing" in rows[0]["failed_checks"]
    assert "holdout_metric_below_threshold" in rows[0]["failed_checks"]
    assert "holdout_application_not_gpu" in rows[0]["failed_checks"]


def test_matrix_rejects_duplicate_config_ids(tmp_path: Path) -> None:
    examples = _write_json(tmp_path / "examples.jsonl", [])
    matrix = _write_json(tmp_path / "matrix.json", [{"id": "same"}, {"id": "same"}])

    try:
        run_generic_audio_finetune_matrix(
            audio_training_examples_jsonl=examples,
            config_matrix_json=matrix,
            output_dir=tmp_path / "matrix_run",
            stem="vocal",
            scope="vocal_matrix",
        )
    except ValueError as exc:
        assert "duplicate matrix config id" in str(exc)
    else:
        raise AssertionError("duplicate config ids should be rejected")
