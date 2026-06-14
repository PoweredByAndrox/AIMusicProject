from __future__ import annotations

import json
from pathlib import Path

from tools.eval_generic_finetuning_controller_model import eval_controller_model, main
from tools.train_generic_finetuning_controller_model import MANIFEST_NAME as TRAINING_MANIFEST_NAME
from tools.train_generic_finetuning_controller_model import train_controller_model


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def controller_row(row_id: str, stem: str, model: str, accepted: bool, manifest_path: str | None = None) -> dict:
    return {
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem,
        "input": {
            "model_id": model,
            "delivery_status": "accepted" if accepted else "blocked",
            "manifest_path": manifest_path or f"/app/lab_artifacts/controller/{row_id}/manifest.json",
        },
        "target": {
            "action": "reuse_route_candidate" if accepted else "avoid_route_and_generate_alternative",
            "chosen_model": model,
        },
        "result": {"accepted": accepted},
        "trainable": True,
    }


def test_external_eval_blocks_training_data_reuse(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("a1", "minus", "good-minus", True),
            controller_row("a2", "minus", "good-minus", True),
            controller_row("b1", "minus", "bad-minus", False),
            controller_row("b2", "minus", "bad-minus", False),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    manifest = eval_controller_model(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        eval_examples_jsonl=examples,
        output_dir=tmp_path / "eval",
        training_manifest_json=tmp_path / "train" / TRAINING_MANIFEST_NAME,
        min_eval_examples=1,
    )

    assert manifest["promotion"]["status"] == "blocked_training_data_eval_only"
    assert manifest["promotion"]["can_promote_controller_model"] is False
    assert manifest["safety"]["external_eval_required"] is True

    copied_examples = write_jsonl(
        tmp_path / "copied_controller.jsonl",
        [
            controller_row("a1", "minus", "good-minus", True),
            controller_row("a2", "minus", "good-minus", True),
            controller_row("b1", "minus", "bad-minus", False),
            controller_row("b2", "minus", "bad-minus", False),
        ],
    )
    copied_manifest = eval_controller_model(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        eval_examples_jsonl=copied_examples,
        output_dir=tmp_path / "eval_copy",
        training_manifest_json=tmp_path / "train" / TRAINING_MANIFEST_NAME,
        min_eval_examples=1,
    )

    assert copied_manifest["promotion"]["status"] == "blocked_training_data_eval_only"


def test_external_eval_passes_on_separate_eval_examples(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "train_controller.jsonl",
        [
            controller_row("a1", "minus", "good-minus", True),
            controller_row("a2", "minus", "good-minus", True),
            controller_row("b1", "minus", "bad-minus", False),
            controller_row("b2", "minus", "bad-minus", False),
        ],
    )
    eval_examples = write_jsonl(
        tmp_path / "external_eval_controller.jsonl",
        [
            controller_row("a3", "minus", "good-minus", True),
            controller_row("b3", "minus", "bad-minus", False),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    manifest = eval_controller_model(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        eval_examples_jsonl=eval_examples,
        output_dir=tmp_path / "eval",
        training_manifest_json=tmp_path / "train" / TRAINING_MANIFEST_NAME,
        min_eval_examples=2,
        min_accuracy=0.7,
    )

    assert manifest["eval_report"]["accuracy"] == 1.0
    assert manifest["promotion"]["status"] == "external_eval_passed"
    assert manifest["promotion"]["can_promote_controller_model"] is True
    assert manifest["model_registry_edited"] is False
    assert manifest["release_ready"] is False


def test_external_eval_blocks_low_accuracy(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "train_controller.jsonl",
        [
            controller_row("a1", "minus", "good-minus", True),
            controller_row("a2", "minus", "good-minus", True),
        ],
    )
    eval_examples = write_jsonl(
        tmp_path / "external_eval_controller.jsonl",
        [
            controller_row("a3", "minus", "good-minus", False),
            controller_row("a4", "minus", "good-minus", False),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    manifest = eval_controller_model(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        eval_examples_jsonl=eval_examples,
        output_dir=tmp_path / "eval",
        training_manifest_json=tmp_path / "train" / TRAINING_MANIFEST_NAME,
        min_eval_examples=2,
        min_accuracy=0.7,
    )

    assert manifest["eval_report"]["accuracy"] == 0.0
    assert manifest["promotion"]["status"] == "blocked_accuracy_below_threshold"
    assert manifest["promotion"]["can_promote_controller_model"] is False


def test_external_eval_blocks_manifest_overlap_with_training_examples(tmp_path: Path) -> None:
    shared_manifest = "/app/lab_artifacts/shared_case/no_reference_listen_preview_manifest.json"
    train_examples = write_jsonl(
        tmp_path / "train_controller.jsonl",
        [
            controller_row("train1", "minus", "bad-minus", False, manifest_path=shared_manifest),
            controller_row("train2", "minus", "bad-minus", False),
        ],
    )
    eval_examples = write_jsonl(
        tmp_path / "external_eval_controller.jsonl",
        [
            controller_row("eval1", "minus", "bad-minus", False, manifest_path=shared_manifest),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    manifest = eval_controller_model(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        eval_examples_jsonl=eval_examples,
        output_dir=tmp_path / "eval",
        training_manifest_json=tmp_path / "train" / TRAINING_MANIFEST_NAME,
        min_eval_examples=1,
        min_accuracy=0.7,
    )

    assert manifest["eval_report"]["training_overlap"]["overlap_count"] == 1
    assert manifest["promotion"]["status"] == "blocked_training_overlap_eval_only"
    assert manifest["promotion"]["can_promote_controller_model"] is False


def test_cli_writes_external_eval_manifest(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "train_controller.jsonl",
        [
            controller_row("a1", "minus", "good-minus", True),
            controller_row("a2", "minus", "good-minus", True),
            controller_row("b1", "minus", "bad-minus", False),
            controller_row("b2", "minus", "bad-minus", False),
        ],
    )
    eval_examples = write_jsonl(
        tmp_path / "external_eval_controller.jsonl",
        [
            controller_row("a3", "minus", "good-minus", True),
            controller_row("b3", "minus", "bad-minus", False),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    assert main(
        [
            "--controller-model-json",
            str(Path(training_manifest["outputs"]["model_json"])),
            "--eval-examples-jsonl",
            str(eval_examples),
            "--training-manifest-json",
            str(tmp_path / "train" / TRAINING_MANIFEST_NAME),
            "--output-dir",
            str(tmp_path / "eval"),
            "--min-eval-examples",
            "2",
        ]
    ) == 0

    manifest = read_json(tmp_path / "eval" / "generic_finetuning_controller_external_eval_manifest.json")
    assert manifest["promotion"]["status"] == "external_eval_passed"
    assert manifest["training_started"] is False
