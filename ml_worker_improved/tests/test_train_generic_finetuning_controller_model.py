from __future__ import annotations

import json
from pathlib import Path

from tools.train_generic_finetuning_controller_model import train_controller_model, main


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def controller_row(row_id: str, stem: str, model: str, accepted: bool, failure: str = "") -> dict:
    return {
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem,
        "input": {
            "model_id": model,
            "failure_labels": [failure] if failure else [],
            "delivery_status": "accepted" if accepted else "blocked",
        },
        "target": {
            "action": "reuse_route_candidate" if accepted else "avoid_route_and_generate_alternative",
            "chosen_model": model,
        },
        "result": {"accepted": accepted, "human_verified": accepted},
        "trainable": True,
    }


def test_trains_smoothed_controller_route_model(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("a1", "vocal", "model_good", True),
            controller_row("a2", "vocal", "model_good", True),
            controller_row("b1", "vocal", "model_bad", False, "severe_vocal_bleed"),
            controller_row("b2", "vocal", "model_bad", False, "severe_phase_collapse"),
            controller_row("m1", "minus", "minus_good", True),
        ],
    )

    manifest = train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_examples_per_route=1,
    )

    assert manifest["training_started"] is True
    assert manifest["fine_tune_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["release_status"] == "training_artifact_only"
    rows = manifest["model"]["route_priors"]
    vocal_rows = [row for row in rows if row["stem_type"] == "vocal"]
    assert vocal_rows[0]["route_id"] == "model_good"
    assert vocal_rows[0]["posterior_accept_rate"] > vocal_rows[1]["posterior_accept_rate"]
    assert manifest["promotion"]["status"] == "blocked"
    assert manifest["promotion"]["can_promote"] is False
    assert manifest["model"]["release_ready"] is False
    assert manifest["model"]["delivery_allowed"] is False
    assert (tmp_path / "out" / "generic_finetuning_controller_model.json").exists()


def test_training_fingerprint_is_stable(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("a1", "vocal", "model_good", True),
            controller_row("b1", "vocal", "model_bad", False, "severe_vocal_bleed"),
        ],
    )

    first = train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "out1",
        min_examples_per_route=1,
    )
    second = train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "out2",
        min_examples_per_route=1,
    )

    assert first["fingerprint"] == second["fingerprint"]


def test_external_eval_candidate_still_cannot_promote(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [controller_row(f"a{idx}", "vocal", "model_good", True) for idx in range(20)],
    )

    manifest = train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_examples_per_route=1,
    )

    assert manifest["eval_report"]["promotion_candidate"] is True
    assert manifest["promotion"]["status"] == "candidate_ready_for_external_eval"
    assert manifest["promotion"]["reason"] == "external_eval_required_before_promotion"
    assert manifest["promotion"]["can_promote"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["model_registry_edited"] is False
    assert manifest["release_ready"] is False


def test_cli_writes_model_manifest(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("a1", "vocal", "model_good", True),
            controller_row("b1", "vocal", "model_bad", False, "severe_vocal_bleed"),
        ],
    )

    assert main(["--controller-examples-jsonl", str(examples), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / "generic_finetuning_controller_training_manifest.json")
    assert manifest["summary"]["input_examples"] == 2
    assert manifest["training_started"] is True
    assert manifest["promotion"]["can_promote"] is False
    assert manifest["model_registry_edited"] is False
