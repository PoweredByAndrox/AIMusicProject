from __future__ import annotations

import json
from pathlib import Path

from tools.build_multi_candidate_conditioned_audio_training_view import (
    build_multi_candidate_conditioned_audio_training_view,
)
from tools.pseudo_teacher_contract import PSEUDO_TEACHER_SCHEMA


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audio(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def _base_row(tmp_path: Path, row_id: str, track_id: str) -> dict:
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": row_id,
        "track_id": track_id,
        "stem_type": "vocal",
        "source_audio_path": _audio(tmp_path / row_id / "mix.flac", b"mix"),
        "target_audio_path": _audio(tmp_path / row_id / "verified_vocal.flac", b"target"),
        "permission_status": "project_owned",
        "split": "train",
        "trainable": True,
        "human_verified": True,
        "reference_verified": False,
        "release_ready": False,
        "delivery_allowed": False,
    }


def _candidate_manifest(tmp_path: Path, root_name: str, row_id: str, model_id: str, payload: bytes) -> Path:
    candidate_audio = _audio(tmp_path / "candidate_audio" / root_name / row_id / "vocals.flac", payload)
    return _write_json(
        tmp_path / root_name / row_id / "registry_candidate_execution_manifest.json",
        {
            "status": "completed",
            "model": {"model_id": model_id},
            "candidate_manifest": {
                "target_stem": "vocal",
                "target_audio_path": candidate_audio,
                "path": str(tmp_path / root_name / row_id / "stem_manifest.json"),
            },
            "release_ready": False,
        },
    )


def test_builds_one_training_row_per_candidate_family_without_changing_target(tmp_path: Path) -> None:
    base_row = _base_row(tmp_path, "row-a", "track-a")
    base_jsonl = _write_jsonl(tmp_path / "base.jsonl", [base_row])
    htd6_manifest = _candidate_manifest(tmp_path, "htd6", "row-a", "htdemucs_6s", b"candidate-a")
    ft_manifest = _candidate_manifest(tmp_path, "ft", "row-a", "htdemucs_ft", b"candidate-b")

    manifest = build_multi_candidate_conditioned_audio_training_view(
        audio_training_examples_jsonl=base_jsonl,
        candidate_roots=[("htd6", tmp_path / "htd6"), ("ft", tmp_path / "ft")],
        output_dir=tmp_path / "out",
        stem="vocal",
        scope="multi_candidate_conditioned_test",
    )

    assert manifest["status"] == "passed"
    assert manifest["conditioned_example_count"] == 2
    assert manifest["blocked_example_count"] == 0
    assert manifest["release_ready"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["safety"]["candidate_source_is_not_ground_truth"] is True
    assert manifest["family_counts"] == {"ft": 1, "htd6": 1}

    rows = _read_jsonl(Path(manifest["candidate_conditioned_examples_jsonl"]))
    assert [row["id"] for row in rows] == [
        "candidate_conditioned__htd6__row-a",
        "candidate_conditioned__ft__row-a",
    ]
    assert rows[0]["target_audio_path"] == base_row["target_audio_path"]
    assert rows[1]["target_audio_path"] == base_row["target_audio_path"]
    assert rows[0]["source_candidate_family"] == "htd6"
    assert rows[1]["source_candidate_family"] == "ft"
    assert rows[0]["source_candidate_model_id"] == "htdemucs_6s"
    assert rows[1]["source_candidate_model_id"] == "htdemucs_ft"
    assert rows[0]["candidate_execution_manifest_json"] == str(htd6_manifest)
    assert rows[1]["candidate_execution_manifest_json"] == str(ft_manifest)
    assert rows[0]["source_candidate_is_ground_truth"] is False
    assert rows[1]["source_candidate_is_ground_truth"] is False


def test_builds_candidate_conditioned_pseudo_teacher_rows_for_train_only(tmp_path: Path) -> None:
    base_row = _base_row(tmp_path, "accepted-a", "track-a")
    base_row["accepted_no_reference_teacher"] = True
    base_row["target_audio_is_ground_truth"] = False
    base_jsonl = _write_jsonl(tmp_path / "base.jsonl", [base_row])
    _candidate_manifest(tmp_path, "scnet", "accepted-a", "scnet_xl_ihf_musdb18", b"candidate-a")

    manifest = build_multi_candidate_conditioned_audio_training_view(
        audio_training_examples_jsonl=base_jsonl,
        candidate_roots=[("scnet", tmp_path / "scnet")],
        output_dir=tmp_path / "out",
        stem="vocal",
        scope="candidate_conditioned_pseudo_test",
        output_pseudo_teacher_rows=True,
    )

    rows = _read_jsonl(Path(manifest["candidate_conditioned_examples_jsonl"]))
    assert manifest["status"] == "passed"
    assert manifest["output_pseudo_teacher_rows"] is True
    assert manifest["safety"]["pseudo_teacher_rows_not_ground_truth"] is True
    assert rows[0]["schema"] == PSEUDO_TEACHER_SCHEMA
    assert rows[0]["id"] == "pseudo__candidate_conditioned__scnet__accepted-a"
    assert rows[0]["pseudo_teacher"] is True
    assert rows[0]["pseudo_teacher_targets_are_ground_truth"] is False
    assert rows[0]["human_verified"] is False
    assert rows[0]["reference_verified"] is False
    assert rows[0]["source_candidate_family"] == "scnet"
    assert rows[0]["source_candidate_is_ground_truth"] is False
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["release_ready"] is False
    assert rows[0]["safety"]["fallback_used"] is False


def test_blocks_missing_family_candidate_instead_of_falling_back(tmp_path: Path) -> None:
    base_row = _base_row(tmp_path, "row-a", "track-a")
    base_jsonl = _write_jsonl(tmp_path / "base.jsonl", [base_row])
    _candidate_manifest(tmp_path, "htd6", "row-a", "htdemucs_6s", b"candidate-a")

    manifest = build_multi_candidate_conditioned_audio_training_view(
        audio_training_examples_jsonl=base_jsonl,
        candidate_roots=[("htd6", tmp_path / "htd6"), ("missing", tmp_path / "missing")],
        output_dir=tmp_path / "out",
        stem="vocal",
        scope="multi_candidate_conditioned_test",
    )

    assert manifest["status"] == "passed"
    assert manifest["conditioned_example_count"] == 1
    assert manifest["blocked_example_count"] == 1
    blocked = _read_jsonl(Path(manifest["blocked_examples_jsonl"]))
    assert blocked[0]["id"] == "row-a"
    assert blocked[0]["source_candidate_family"] == "missing"
    assert "candidate_execution_manifest_missing" in blocked[0]["blocked_reasons"]
    assert blocked[0]["fallback_used"] is False
    assert blocked[0]["release_ready"] is False
