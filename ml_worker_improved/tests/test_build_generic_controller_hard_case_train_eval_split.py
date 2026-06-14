from __future__ import annotations

import json
from pathlib import Path

from tools.build_generic_controller_hard_case_train_eval_split import (
    EVAL_EXAMPLES_NAME,
    MANIFEST_NAME,
    SKIPPED_NAME,
    TRAIN_EXAMPLES_NAME,
    build_hard_case_train_eval_split,
    main,
)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def controller_row(row_id: str, *, model_id: str = "accepted-route") -> dict:
    return {
        "schema": "generic_finetuning_controller_decision_example.v1",
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": "minus",
        "input": {
            "manifest_path": f"/app/lab_artifacts/base/{row_id}/manifest.json",
            "model_id": model_id,
            "delivery_status": "accepted",
            "failure_labels": [],
        },
        "target": {"action": "reuse_route_candidate", "chosen_model": model_id},
        "result": {"accepted": True},
        "trainable": True,
        "training_target": "llm_controller",
    }


def hard_case(row_id: str, *, model_id: str, split: str = "hard_eval") -> dict:
    return {
        "schema": "generic_finetuning_hard_eval_case.v1",
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": "minus",
        "manifest_path": f"/app/lab_artifacts/hard/{row_id}/manifest.json",
        "model_id": model_id,
        "dominant_failure_reason": "vocal_bleed",
        "failure_labels": ["vocal_bleed"],
        "human_verified": False,
        "human_rating": "",
        "split": split,
        "training_target": "regression_eval",
    }


def test_builds_disjoint_train_and_holdout_controller_examples(tmp_path: Path) -> None:
    base = write_jsonl(tmp_path / "controller.jsonl", [controller_row("base1")])
    hard_eval = write_jsonl(
        tmp_path / "hard_eval_cases.jsonl",
        [
            hard_case("route-a-1", model_id="bad-route-a"),
            hard_case("route-a-2", model_id="bad-route-a"),
            hard_case("route-b-singleton", model_id="bad-route-b"),
            hard_case("missing-route", model_id=""),
            hard_case("train-split", model_id="bad-train", split="train"),
        ],
    )

    manifest = build_hard_case_train_eval_split(
        base_controller_examples_jsonl=base,
        hard_eval_jsonl=hard_eval,
        output_dir=tmp_path / "out",
        min_route_examples_for_holdout=2,
    )

    assert manifest["summary"]["base_controller_examples"] == 1
    assert manifest["summary"]["hard_case_train_examples"] == 2
    assert manifest["summary"]["holdout_eval_examples"] == 1
    assert manifest["summary"]["skipped_no_route"] == 1
    assert manifest["summary"]["skipped_split"] == 1
    assert manifest["summary"]["train_eval_manifest_overlap_count"] == 0
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["release_ready"] is False

    train_rows = read_jsonl(tmp_path / "out" / TRAIN_EXAMPLES_NAME)
    eval_rows = read_jsonl(tmp_path / "out" / EVAL_EXAMPLES_NAME)
    skipped_rows = read_jsonl(tmp_path / "out" / SKIPPED_NAME)
    train_paths = {row["input"]["manifest_path"] for row in train_rows}
    eval_paths = {row["input"]["manifest_path"] for row in eval_rows}

    assert len(train_rows) == 3
    assert len(eval_rows) == 1
    assert not train_paths & eval_paths
    assert {row["input"]["model_id"] for row in eval_rows} <= {row["input"]["model_id"] for row in train_rows}
    assert {row["skip_reason"] for row in skipped_rows} == {"missing_model_id", "split_not_enabled"}
    assert all(row["result"]["accepted"] is False for row in train_rows if row["id"] != "base1")
    assert all(row["training_target"] == "llm_controller_hard_case_train" for row in train_rows if row["id"] != "base1")


def test_cli_writes_hard_case_split_manifest(tmp_path: Path) -> None:
    base = write_jsonl(tmp_path / "controller.jsonl", [controller_row("base1")])
    hard_eval = write_jsonl(
        tmp_path / "hard_eval_cases.jsonl",
        [hard_case("route-a-1", model_id="bad-route-a"), hard_case("route-a-2", model_id="bad-route-a")],
    )

    assert main(["--base-controller-examples-jsonl", str(base), "--hard-eval-jsonl", str(hard_eval), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / MANIFEST_NAME)
    assert manifest["summary"]["holdout_eval_examples"] == 1
    assert Path(manifest["outputs"]["augmented_controller_examples_jsonl"]).name == TRAIN_EXAMPLES_NAME
    assert Path(manifest["outputs"]["holdout_eval_examples_jsonl"]).name == EVAL_EXAMPLES_NAME
