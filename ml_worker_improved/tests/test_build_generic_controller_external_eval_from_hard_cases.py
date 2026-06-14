from __future__ import annotations

import json
from pathlib import Path

from tools.build_generic_controller_external_eval_from_hard_cases import (
    EXAMPLES_NAME,
    MANIFEST_NAME,
    SKIPPED_NAME,
    build_external_eval_set,
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


def hard_case(
    row_id: str,
    *,
    model_id: str,
    split: str = "hard_eval",
    stem_type: str = "minus",
    failure_labels: list[str] | None = None,
) -> dict:
    return {
        "schema": "generic_finetuning_hard_eval_case.v1",
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem_type,
        "manifest_path": f"/app/lab_artifacts/cases/{row_id}/manifest.json",
        "model_id": model_id,
        "dominant_failure_reason": (failure_labels or ["leakage"])[0],
        "failure_labels": failure_labels or ["leakage"],
        "human_verified": row_id.endswith("human"),
        "human_rating": "reject" if row_id.endswith("human") else "",
        "split": split,
        "training_target": "regression_eval",
    }


def test_builds_external_eval_examples_and_skips_unusable_rows(tmp_path: Path) -> None:
    hard_eval = write_jsonl(
        tmp_path / "hard_eval_cases.jsonl",
        [
            hard_case("route-case", model_id="minus_route_a", failure_labels=["vocal_bleed", "phase_artifact"]),
            hard_case("missing-route", model_id=""),
            hard_case("train-split", model_id="minus_route_train", split="train"),
        ],
    )

    manifest = build_external_eval_set(hard_eval_jsonl=hard_eval, output_dir=tmp_path / "out")

    assert manifest["summary"]["source_rows"] == 3
    assert manifest["summary"]["eval_examples"] == 1
    assert manifest["summary"]["skipped_rows"] == 2
    assert manifest["summary"]["skipped_no_route"] == 1
    assert manifest["summary"]["skipped_split"] == 1
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["model_registry_edited"] is False
    assert manifest["release_ready"] is False
    assert manifest["safety"]["metadata_only"] is True

    rows = read_jsonl(tmp_path / "out" / EXAMPLES_NAME)
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "generic_finetuning_controller_decision_example.v1"
    assert row["track_id"] == "track-route-case"
    assert row["stem_type"] == "minus"
    assert row["input"]["model_id"] == "minus_route_a"
    assert row["input"]["source_hard_eval_case_id"] == "route-case"
    assert row["input"]["source_split"] == "hard_eval"
    assert row["input"]["failure_labels"] == ["vocal_bleed", "phase_artifact"]
    assert row["target"]["action"] == "avoid_route_and_generate_alternative"
    assert row["target"]["chosen_model"] == "minus_route_a"
    assert "external_eval_replay" in row["target"]["acceptance_tests"]
    assert row["result"]["accepted"] is False
    assert row["trainable"] is True
    assert row["training_target"] == "llm_controller_external_eval"

    skipped = read_jsonl(tmp_path / "out" / SKIPPED_NAME)
    assert {row["skip_reason"] for row in skipped} == {"missing_model_id", "split_not_enabled"}


def test_can_include_unknown_route_explicitly(tmp_path: Path) -> None:
    hard_eval = write_jsonl(tmp_path / "hard_eval_cases.jsonl", [hard_case("missing-route", model_id="")])

    manifest = build_external_eval_set(
        hard_eval_jsonl=hard_eval,
        output_dir=tmp_path / "out",
        include_unknown_route=True,
    )

    assert manifest["summary"]["eval_examples"] == 1
    assert manifest["summary"]["unknown_route_examples"] == 1
    rows = read_jsonl(tmp_path / "out" / EXAMPLES_NAME)
    assert rows[0]["input"]["model_id"] == "unknown_route"
    assert rows[0]["target"]["chosen_model"] == "unknown_route"


def test_cli_writes_manifest_progress_and_examples(tmp_path: Path) -> None:
    hard_eval = write_jsonl(tmp_path / "hard_eval_cases.jsonl", [hard_case("route-case", model_id="minus_route_a")])

    assert main(["--hard-eval-jsonl", str(hard_eval), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / MANIFEST_NAME)
    assert manifest["summary"]["eval_examples"] == 1
    assert Path(manifest["outputs"]["eval_examples_jsonl"]).name == EXAMPLES_NAME
    assert Path(manifest["outputs"]["progress_jsonl"]).exists()
