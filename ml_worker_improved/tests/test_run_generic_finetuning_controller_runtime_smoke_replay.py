from __future__ import annotations

import json
from pathlib import Path

from tools.run_generic_finetuning_controller_runtime_smoke_replay import MANIFEST_NAME, main, run_runtime_smoke_replay
from tools.train_generic_finetuning_controller_model import train_controller_model


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def controller_row(row_id: str, stem: str, model: str, accepted: bool) -> dict:
    return {
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem,
        "input": {
            "manifest_path": f"/app/lab_artifacts/runtime/{row_id}/manifest.json",
            "model_id": model,
            "delivery_status": "accepted" if accepted else "blocked",
            "failure_labels": [] if accepted else ["leakage"],
        },
        "target": {
            "action": "reuse_route_candidate" if accepted else "avoid_route_and_generate_alternative",
            "chosen_model": model,
        },
        "result": {"accepted": accepted},
        "trainable": True,
    }


def test_runtime_smoke_replay_passes_controller_actions(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "train.jsonl",
        [
            controller_row("good1", "minus", "good-route", True),
            controller_row("good2", "minus", "good-route", True),
            controller_row("bad1", "minus", "bad-route", False),
            controller_row("bad2", "minus", "bad-route", False),
        ],
    )
    replay_examples = write_jsonl(
        tmp_path / "replay.jsonl",
        [
            controller_row("good3", "minus", "good-route", True),
            controller_row("bad3", "minus", "bad-route", False),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    manifest = run_runtime_smoke_replay(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        replay_examples_jsonl=replay_examples,
        output_dir=tmp_path / "smoke",
        min_replay_examples=2,
        min_action_accuracy=1.0,
    )

    assert manifest["runtime_smoke"]["action_accuracy"] == 1.0
    assert manifest["promotion"]["status"] == "runtime_smoke_replay_passed"
    assert manifest["promotion"]["can_edit_model_registry"] is False
    assert manifest["safety"]["metadata_only"] is True
    assert manifest["release_ready"] is False


def test_runtime_smoke_replay_blocks_missing_routes(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "train.jsonl",
        [controller_row("good1", "minus", "good-route", True), controller_row("good2", "minus", "good-route", True)],
    )
    replay_examples = write_jsonl(tmp_path / "replay.jsonl", [controller_row("unknown1", "minus", "unknown-route", False)])
    training_manifest = train_controller_model(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    manifest = run_runtime_smoke_replay(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        replay_examples_jsonl=replay_examples,
        output_dir=tmp_path / "smoke",
        min_replay_examples=1,
    )

    assert manifest["runtime_smoke"]["missing_route_examples"] == 1
    assert manifest["promotion"]["status"] == "blocked_missing_routes"
    assert manifest["promotion"]["can_stage_controller_candidate"] is False


def test_cli_writes_runtime_smoke_manifest(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "train.jsonl",
        [controller_row("bad1", "minus", "bad-route", False), controller_row("bad2", "minus", "bad-route", False)],
    )
    replay_examples = write_jsonl(tmp_path / "replay.jsonl", [controller_row("bad3", "minus", "bad-route", False)])
    training_manifest = train_controller_model(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "train",
        min_examples_per_route=1,
    )

    assert main(
        [
            "--controller-model-json",
            str(Path(training_manifest["outputs"]["model_json"])),
            "--replay-examples-jsonl",
            str(replay_examples),
            "--output-dir",
            str(tmp_path / "smoke"),
            "--min-replay-examples",
            "1",
        ]
    ) == 0

    manifest = read_json(tmp_path / "smoke" / MANIFEST_NAME)
    assert manifest["promotion"]["status"] == "runtime_smoke_replay_passed"
