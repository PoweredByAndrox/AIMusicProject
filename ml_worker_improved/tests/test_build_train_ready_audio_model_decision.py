from __future__ import annotations

import json
from pathlib import Path

from tools.build_train_ready_audio_model_decision import build_train_ready_audio_model_decision


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(*, passed: int, total: int, status: str = "passed", failed_checks: list[str] | None = None) -> dict:
    return {
        "schema": "generic_audio_finetune_matrix_row.v1",
        "config_id": "best",
        "scope": "vocal_matrix__best",
        "status": status,
        "failed_checks": failed_checks or [],
        "holdout_validation_status": "passed" if status == "passed" else "failed",
        "holdout_eval_count": total,
        "passed_eval_count": passed,
        "failed_eval_count": max(total - passed, 0),
        "metrics": {"min_candidate_sdr_db": 3.4, "min_improvement_db": 0.8},
        "delivery_allowed": False,
        "release_ready": False,
    }


def _matrix(tmp_path: Path, rows: list[dict], *, release_ready: bool = False) -> Path:
    rows_path = _write_jsonl(tmp_path / "matrix_rows.jsonl", rows)
    return _write_json(
        tmp_path / "matrix_manifest.json",
        {
            "schema": "generic_audio_finetune_matrix_manifest.v1",
            "matrix_validation_status": "passed" if rows and rows[0]["status"] == "passed" else "failed",
            "best_config_id": "best" if rows and rows[0]["status"] == "passed" else "",
            "passed_config_count": 1 if rows and rows[0]["status"] == "passed" else 0,
            "failed_config_count": 0 if rows and rows[0]["status"] == "passed" else 1,
            "training_started": True,
            "fine_tune_started": True,
            "models_changed": True,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
            "thresholds_relaxed": False,
            "delivery_allowed": False,
            "release_ready": release_ready,
            "matrix_rows_jsonl": str(rows_path),
        },
    )


def _holdout_rows(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "holdout_rows.jsonl",
        [
            {
                "checkpoint_track_id": "track-a",
                "checkpoint_training_track_ids": ["track-a", "track-b"],
                "holdout_track_id": "track-c",
                "status": "passed",
                "release_ready": False,
                "delivery_allowed": False,
                "application_device_used": 1.0,
            }
        ],
    )


def test_decision_passes_only_when_success_rate_is_strictly_above_ninety(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, [_row(passed=10, total=10)])
    holdout_rows = _holdout_rows(tmp_path)

    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=matrix,
        output_dir=tmp_path / "decision",
        holdout_rows_jsonls=[holdout_rows],
        min_success_rate=0.90,
    )

    assert manifest["decision"] == "train_result_verified"
    assert manifest["train_ready"] is True
    assert manifest["success_rate"] == 1.0
    assert manifest["success_rate_rule"] == "> 0.900000"
    assert manifest["release_ready"] is False
    assert manifest["promotion_allowed"] is False


def test_exactly_ninety_percent_is_not_train_ready(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, [_row(passed=9, total=10)])

    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=matrix,
        output_dir=tmp_path / "decision",
        min_success_rate=0.90,
    )

    assert manifest["decision"] == "train_result_not_ready"
    assert manifest["train_ready"] is False
    assert manifest["success_rate"] == 0.9
    assert "success_rate_not_strictly_above_threshold" in manifest["failed_checks"]


def test_failed_matrix_still_reports_failed_holdout_eval_counts(tmp_path: Path) -> None:
    first = _row(passed=0, total=1, status="failed", failed_checks=["sdr_below_threshold"])
    first["config_id"] = "failed_a"
    second = _row(passed=0, total=3, status="failed", failed_checks=["sdr_below_threshold"])
    second["config_id"] = "failed_b"
    matrix = _matrix(tmp_path, [first, second])

    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=matrix,
        output_dir=tmp_path / "decision",
        min_success_rate=0.90,
    )

    assert manifest["decision"] == "train_result_not_ready"
    assert manifest["train_ready"] is False
    assert manifest["holdout_eval_count"] == 4
    assert manifest["passed_eval_count"] == 0
    assert manifest["failed_eval_count"] == 4
    assert manifest["success_rate"] == 0.0
    assert "matrix_validation_not_passed" in manifest["failed_checks"]
    assert "best_matrix_row_missing" in manifest["failed_checks"]
    assert "holdout_eval_count_missing" not in manifest["failed_checks"]


def test_release_ready_or_gate_change_fails_safety_contract(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, [_row(passed=10, total=10)], release_ready=True)

    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=matrix,
        output_dir=tmp_path / "decision",
        min_success_rate=0.90,
    )

    assert manifest["decision"] == "train_result_not_ready"
    assert manifest["train_ready"] is False
    assert "matrix_release_ready_true" in manifest["failed_checks"]
    assert manifest["release_ready"] is False


def test_critical_leakage_blocks_train_ready_decision(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, [_row(passed=10, total=10)])
    leakage = _write_json(tmp_path / "leakage.json", {"critical_leakage_count": 1})

    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=matrix,
        output_dir=tmp_path / "decision",
        leak_safe_split_manifest_json=leakage,
        min_success_rate=0.90,
    )

    assert manifest["decision"] == "train_result_not_ready"
    assert "critical_leakage_detected" in manifest["failed_checks"]


def test_holdout_rows_block_same_track_contamination(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, [_row(passed=10, total=10)])
    contaminated = _write_jsonl(
        tmp_path / "holdout_rows.jsonl",
        [
            {
                "checkpoint_training_track_ids": ["track-a"],
                "holdout_track_id": "track-a",
                "status": "passed",
                "release_ready": False,
            }
        ],
    )

    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=matrix,
        output_dir=tmp_path / "decision",
        holdout_rows_jsonls=[contaminated],
        min_success_rate=0.90,
    )

    assert manifest["decision"] == "train_result_not_ready"
    assert "holdout_track_leakage" in manifest["failed_checks"]


def test_holdout_rows_report_failed_status_and_row_failed_checks(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, [_row(passed=0, total=1, status="failed", failed_checks=["holdout_failed"])])
    failed_holdout_rows = _write_jsonl(
        tmp_path / "holdout_rows.jsonl",
        [
            {
                "checkpoint_training_track_ids": ["track-a"],
                "holdout_track_id": "track-b",
                "status": "failed",
                "failed_checks": ["improvement_below_threshold"],
                "release_ready": False,
                "delivery_allowed": False,
                "application_device_used": 1.0,
            }
        ],
    )

    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=matrix,
        output_dir=tmp_path / "decision",
        holdout_rows_jsonls=[failed_holdout_rows],
        min_success_rate=0.90,
    )

    assert manifest["decision"] == "train_result_not_ready"
    assert manifest["holdout_rows_checked"] == 1
    assert "holdout_row_failed" in manifest["failed_checks"]
    assert "holdout_row_failed_check:improvement_below_threshold" in manifest["failed_checks"]
