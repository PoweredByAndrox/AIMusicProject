from __future__ import annotations

import json
from pathlib import Path

from tools.validate_verified_audio_training_corpus import validate_verified_audio_training_corpus


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audio(path: Path, payload: bytes = b"audio") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def _row(tmp_path: Path, row_id: str, track_id: str, *, split: str = "train", **overrides: object) -> dict:
    row = {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": row_id,
        "track_id": track_id,
        "stem_type": "vocal",
        "source_audio_path": _audio(tmp_path / row_id / "source.flac", b"mix"),
        "target_audio_path": _audio(tmp_path / row_id / "target.flac", b"target"),
        "target_audio_sha256": row_id,
        "permission_status": "project_owned",
        "split": split,
        "trainable": True,
        "human_verified": True,
        "reference_verified": False,
        "source_gate": {"accepted_machine_gate": True, "reference_verified": False},
        "release_ready": False,
    }
    row.update(overrides)
    return row


def test_verified_corpus_passes_only_verified_existing_audio(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            _row(tmp_path, "a", "track-a", human_verified=True, reference_verified=False),
            _row(
                tmp_path,
                "b",
                "track-b",
                human_verified=False,
                reference_verified=True,
                source_gate={
                    "accepted_machine_gate": False,
                    "reference_verified": True,
                    "trainable_ground_truth": True,
                    "full_mix_reconstruction_status": "accepted",
                    "source_mix_1_to_1_claim_allowed": True,
                },
            ),
        ],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=2,
    )

    assert manifest["corpus_validation_status"] == "passed"
    assert manifest["verified_trainable_examples"] == 2
    assert manifest["unique_track_count"] == 2
    assert manifest["release_ready"] is False
    rows = _read_jsonl(Path(manifest["row_report_jsonl"]))
    assert all(row["safe_for_training"] is True for row in rows)


def test_reference_verified_row_without_trainable_ground_truth_gate_fails_closed(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            _row(
                tmp_path,
                "badref",
                "track-bad",
                human_verified=False,
                reference_verified=True,
                source_gate={
                    "accepted_machine_gate": False,
                    "reference_verified": True,
                    "trainable_ground_truth": False,
                    "full_mix_reconstruction_status": "rejected",
                    "source_mix_1_to_1_claim_allowed": False,
                },
            ),
        ],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    rows = _read_jsonl(Path(manifest["row_report_jsonl"]))
    assert manifest["corpus_validation_status"] == "failed"
    assert "missing_trainable_ground_truth_provenance" in rows[0]["unsafe_reasons"]
    assert "rejected_full_mix_reconstruction" in rows[0]["unsafe_reasons"]


def test_human_verified_row_uses_machine_gate_even_when_reference_flag_is_present(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            _row(
                tmp_path,
                "humanref",
                "track-humanref",
                human_verified=True,
                reference_verified=True,
                source_gate={
                    "accepted_machine_gate": True,
                    "reference_verified": True,
                },
            ),
        ],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    rows = _read_jsonl(Path(manifest["row_report_jsonl"]))
    assert manifest["corpus_validation_status"] == "passed"
    assert rows[0]["safe_for_training"] is True


def test_reference_verified_row_with_exact_reference_audit_gate_passes(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            _row(
                tmp_path,
                "exactref",
                "track-exactref",
                human_verified=False,
                reference_verified=True,
                source_gate={
                    "accepted_machine_gate": False,
                    "reference_verified": True,
                    "exact_reference_waveform_audit_accepted": True,
                },
            ),
        ],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    rows = _read_jsonl(Path(manifest["row_report_jsonl"]))
    assert manifest["corpus_validation_status"] == "passed"
    assert rows[0]["safe_for_training"] is True


def test_dirty_or_unverified_rows_fail_closed(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            _row(tmp_path, "dirty", "track-a", human_verified=False, reference_verified=False),
            _row(tmp_path, "missing", "track-b", target_audio_path=str(tmp_path / "missing.flac")),
        ],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    assert manifest["corpus_validation_status"] == "failed"
    rows = _read_jsonl(Path(manifest["row_report_jsonl"]))
    by_id = {row["id"]: row for row in rows}
    assert "not_human_or_reference_verified" in by_id["dirty"]["unsafe_reasons"]
    assert "target_audio_not_found" in by_id["missing"]["unsafe_reasons"]
    assert by_id["dirty"]["safe_for_training"] is False
    assert by_id["missing"]["safe_for_training"] is False


def test_human_verified_row_without_machine_gate_or_reference_fails_closed(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [_row(tmp_path, "ungated", "track-a", source_gate={"accepted_machine_gate": False, "reference_verified": False})],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    assert manifest["corpus_validation_status"] == "failed"
    rows = _read_jsonl(Path(manifest["row_report_jsonl"]))
    assert "missing_accepted_machine_gate_provenance" in rows[0]["unsafe_reasons"]
    assert rows[0]["safe_for_training"] is False


def test_track_split_leakage_blocks_verified_corpus(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            _row(tmp_path, "train", "same-track", split="train"),
            _row(tmp_path, "test", "same-track", split="test"),
        ],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    assert manifest["corpus_validation_status"] == "failed"
    assert "track_split_leakage" in manifest["failed_checks"]
    assert manifest["track_split_leakage_count"] == 1


def test_release_ready_input_row_is_never_training_safe(tmp_path: Path) -> None:
    examples = _write_jsonl(tmp_path / "examples.jsonl", [_row(tmp_path, "unsafe", "track-a", release_ready=True)])

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
    )

    assert manifest["corpus_validation_status"] == "failed"
    rows = _read_jsonl(Path(manifest["row_report_jsonl"]))
    assert "input_row_release_ready_true" in rows[0]["unsafe_reasons"]
    assert manifest["release_ready"] is False


def test_short_output_names_are_available_for_new_artifacts(tmp_path: Path) -> None:
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            _row(tmp_path, "a", "track-a", human_verified=True, reference_verified=False),
            _row(tmp_path, "b", "track-b", human_verified=False, reference_verified=True),
        ],
    )

    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "out",
        min_unique_tracks=2,
        short_output_names=True,
    )

    assert manifest["short_output_names"] is True
    assert Path(manifest["manifest_json"]).name == "val.json"
    assert Path(manifest["row_report_jsonl"]).name == "rows.jsonl"
    assert Path(manifest["progress_jsonl"]).name == "progress.jsonl"
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())
