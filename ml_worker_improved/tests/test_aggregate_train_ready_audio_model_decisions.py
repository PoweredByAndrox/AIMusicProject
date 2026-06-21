from __future__ import annotations

import json
from pathlib import Path

from tools.aggregate_train_ready_audio_model_decisions import aggregate_train_ready_audio_model_decisions


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _decision(tmp_path: Path, name: str, *, train_ready: bool, passed: int = 1, total: int = 1, release_ready: bool = False) -> Path:
    return _write_json(
        tmp_path / name / "train_ready_audio_model_decision_manifest.json",
        {
            "schema": "train_ready_audio_model_decision_manifest.v1",
            "decision": "train_result_verified" if train_ready else "train_result_not_ready",
            "train_ready": train_ready,
            "failed_checks": [] if train_ready else ["holdout_failed"],
            "success_rate": passed / total if total else 0.0,
            "passed_eval_count": passed,
            "holdout_eval_count": total,
            "release_ready": release_ready,
            "delivery_allowed": False,
            "promotion_allowed": False,
        },
    )


def test_aggregate_requires_success_rate_strictly_above_threshold(tmp_path: Path) -> None:
    manifest = aggregate_train_ready_audio_model_decisions(
        decision_manifest_jsons=[
            _decision(tmp_path, "a", train_ready=True),
            _decision(tmp_path, "b", train_ready=True),
            _decision(tmp_path, "c", train_ready=True),
        ],
        output_dir=tmp_path / "out",
        min_success_rate=0.90,
        min_decision_count=3,
    )

    assert manifest["aggregate_decision"] == "train_result_verified"
    assert manifest["train_ready"] is True
    assert manifest["aggregate_success_rate"] == 1.0
    assert manifest["release_ready"] is False


def test_aggregate_fails_when_any_fold_keeps_success_at_ninety_or_below(tmp_path: Path) -> None:
    manifest = aggregate_train_ready_audio_model_decisions(
        decision_manifest_jsons=[
            _decision(tmp_path, "a", train_ready=True, passed=9, total=10),
            _decision(tmp_path, "b", train_ready=False, passed=0, total=1),
        ],
        output_dir=tmp_path / "out",
        min_success_rate=0.90,
        min_decision_count=2,
    )

    assert manifest["aggregate_decision"] == "train_result_not_ready"
    assert manifest["train_ready"] is False
    assert "aggregate_success_rate_not_strictly_above_threshold" in manifest["failed_checks"]


def test_aggregate_blocks_release_ready_child_manifest(tmp_path: Path) -> None:
    manifest = aggregate_train_ready_audio_model_decisions(
        decision_manifest_jsons=[_decision(tmp_path, "unsafe", train_ready=True, release_ready=True)],
        output_dir=tmp_path / "out",
        min_success_rate=0.90,
        min_decision_count=1,
    )

    assert manifest["aggregate_decision"] == "train_result_not_ready"
    assert "child_release_ready_true" in manifest["failed_checks"]
    assert manifest["release_ready"] is False
