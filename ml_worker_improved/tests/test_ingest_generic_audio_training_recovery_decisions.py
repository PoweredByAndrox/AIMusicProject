from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ingest_generic_audio_training_recovery_decisions import (
    RecoveryDecisionError,
    ingest_generic_audio_training_recovery_decisions,
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


def recovery_row(tmp_path: Path, row_id: str, *, blockers: list[str] | None = None) -> dict:
    source = tmp_path / f"{row_id}_source.flac"
    target = tmp_path / f"{row_id}_target.flac"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    return {
        "id": row_id,
        "source_type": "human_feedback_source_target",
        "source_file": str(tmp_path / "feedback.jsonl"),
        "track_id": f"track-{row_id}",
        "stem_type": "vocal",
        "source_audio_path": str(source),
        "target_audio_path": str(target),
        "rating": "good",
        "human_verified": True,
        "reference_verified": False,
        "accepted_machine_gate": False,
        "permission_status": "",
        "blockers": blockers if blockers is not None else ["missing_permission_status"],
        "ready_for_audio_training": not blockers,
    }


def gated_recovery_row(tmp_path: Path, row_id: str, *, blockers: list[str] | None = None) -> dict:
    row = recovery_row(tmp_path, row_id, blockers=blockers)
    row["accepted_machine_gate"] = True
    return row


def test_approved_permission_decision_creates_audio_training_example(tmp_path: Path) -> None:
    queue = write_jsonl(tmp_path / "queue.jsonl", [gated_recovery_row(tmp_path, "row-a")])
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [{"recovery_row_id": "row-a", "decision": "approve_for_training", "permission_status": "project_owned"}],
    )

    manifest = ingest_generic_audio_training_recovery_decisions(
        recovery_queue_jsonl=queue,
        decisions_jsonl=decisions,
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["approved_audio_training_examples"] == 1
    rows = read_jsonl(tmp_path / "out" / "approved_audio_training_examples.jsonl")
    assert rows[0]["permission_status"] == "project_owned"
    assert rows[0]["trainable"] is True
    assert rows[0]["human_verified"] is True
    assert rows[0]["source_gate"]["accepted_machine_gate"] is True
    assert rows[0]["training_target"] == "audio_separator_or_stem_adapter"


def test_approval_without_machine_gate_or_reference_is_rejected(tmp_path: Path) -> None:
    queue = write_jsonl(tmp_path / "queue.jsonl", [recovery_row(tmp_path, "row-a", blockers=[])])
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [{"recovery_row_id": "row-a", "decision": "approve_for_training", "permission_status": "project_owned"}],
    )

    manifest = ingest_generic_audio_training_recovery_decisions(
        recovery_queue_jsonl=queue,
        decisions_jsonl=decisions,
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["approved_audio_training_examples"] == 0
    rejected = read_jsonl(tmp_path / "out" / "rejected_audio_training_recovery_rows.jsonl")
    assert rejected[0]["remaining_blockers"] == ["missing_accepted_machine_gate_provenance"]


def test_approval_with_remaining_blocker_is_rejected(tmp_path: Path) -> None:
    row = gated_recovery_row(tmp_path, "row-b", blockers=["target_audio_not_found"])
    Path(row["target_audio_path"]).unlink()
    queue = write_jsonl(tmp_path / "queue.jsonl", [row])
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [{"recovery_row_id": "row-b", "decision": "approve_for_training", "permission_status": "project_owned"}],
    )

    manifest = ingest_generic_audio_training_recovery_decisions(
        recovery_queue_jsonl=queue,
        decisions_jsonl=decisions,
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["approved_audio_training_examples"] == 0
    rejected = read_jsonl(tmp_path / "out" / "rejected_audio_training_recovery_rows.jsonl")
    assert rejected[0]["remaining_blockers"] == ["target_audio_not_found"]


def test_unknown_recovery_row_id_is_rejected(tmp_path: Path) -> None:
    queue = write_jsonl(tmp_path / "queue.jsonl", [recovery_row(tmp_path, "row-a")])
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [{"recovery_row_id": "missing", "decision": "approve_for_training", "permission_status": "project_owned"}],
    )

    with pytest.raises(RecoveryDecisionError, match="unknown_recovery_row_id"):
        ingest_generic_audio_training_recovery_decisions(
            recovery_queue_jsonl=queue,
            decisions_jsonl=decisions,
            output_dir=tmp_path / "out",
        )


def test_invalid_decision_is_rejected(tmp_path: Path) -> None:
    queue = write_jsonl(tmp_path / "queue.jsonl", [recovery_row(tmp_path, "row-a")])
    decisions = write_jsonl(tmp_path / "decisions.jsonl", [{"recovery_row_id": "row-a", "decision": "ship_it"}])

    with pytest.raises(RecoveryDecisionError, match="invalid_recovery_decision"):
        ingest_generic_audio_training_recovery_decisions(
            recovery_queue_jsonl=queue,
            decisions_jsonl=decisions,
            output_dir=tmp_path / "out",
        )


def test_recovery_decision_ingest_can_write_short_output_names(tmp_path: Path) -> None:
    queue = write_jsonl(tmp_path / "queue.jsonl", [gated_recovery_row(tmp_path, "row-a")])
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [{"recovery_row_id": "row-a", "decision": "approve_for_training", "permission_status": "project_owned"}],
    )

    manifest = ingest_generic_audio_training_recovery_decisions(
        recovery_queue_jsonl=queue,
        decisions_jsonl=decisions,
        output_dir=tmp_path / "out",
        short_output_names=True,
    )

    assert manifest["short_output_names"] is True
    assert Path(manifest["outputs"]["approved_audio_training_examples_jsonl"]).name == "audio.jsonl"
    assert Path(manifest["outputs"]["rejected_rows_jsonl"]).name == "rej.jsonl"
    assert Path(manifest["outputs"]["manifest_json"]).name == "ingest.json"
    assert Path(manifest["outputs"]["progress_jsonl"]).name == "progress.jsonl"
    assert all(len(Path(path).name) <= 32 for path in manifest["outputs"].values())
    assert (tmp_path / "out" / "audio.jsonl").exists()
    assert not (tmp_path / "out" / "approved_audio_training_examples.jsonl").exists()


def test_cli_writes_decision_manifest(tmp_path: Path) -> None:
    queue = write_jsonl(tmp_path / "queue.jsonl", [gated_recovery_row(tmp_path, "row-a", blockers=[])])

    assert main(["--recovery-queue-jsonl", str(queue), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / "generic_audio_training_recovery_decision_manifest.json")
    assert manifest["summary"]["approved_audio_training_examples"] == 1
    assert manifest["decisions_jsonl"] == ""
    assert manifest["training_started"] is False


def test_cli_can_write_short_decision_ingest_outputs(tmp_path: Path) -> None:
    queue = write_jsonl(tmp_path / "queue.jsonl", [gated_recovery_row(tmp_path, "row-a", blockers=[])])

    assert (
        main(
            [
                "--recovery-queue-jsonl",
                str(queue),
                "--output-dir",
                str(tmp_path / "out"),
                "--short-output-names",
            ]
        )
        == 0
    )

    manifest = read_json(tmp_path / "out" / "ingest.json")
    assert manifest["short_output_names"] is True
    assert Path(manifest["outputs"]["approved_audio_training_examples_jsonl"]).name == "audio.jsonl"
