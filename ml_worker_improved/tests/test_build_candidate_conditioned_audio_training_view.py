from __future__ import annotations

import json
from pathlib import Path

from tools.build_candidate_conditioned_audio_training_view import build_candidate_conditioned_audio_training_view


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


def test_builds_candidate_conditioned_view_without_changing_verified_target(tmp_path: Path) -> None:
    base_row = _base_row(tmp_path, "row-a", "track-a")
    base_jsonl = _write_jsonl(tmp_path / "base.jsonl", [base_row])
    candidate_audio = _audio(tmp_path / "candidates" / "row-a" / "vocals.flac", b"candidate")
    candidate_manifest = _write_json(
        tmp_path / "candidate_root" / "row-a" / "registry_candidate_execution_manifest.json",
        {
            "status": "completed",
            "model": {"model_id": "local_registered__htdemucs_ft__both__shared"},
            "candidate_manifest": {
                "target_stem": "vocal",
                "target_audio_path": candidate_audio,
                "path": str(tmp_path / "candidate_root" / "row-a" / "stem_manifest.json"),
            },
        },
    )

    manifest = build_candidate_conditioned_audio_training_view(
        audio_training_examples_jsonl=base_jsonl,
        candidate_root=tmp_path / "candidate_root",
        output_dir=tmp_path / "out",
        stem="vocal",
        scope="candidate_conditioned_test",
    )

    assert manifest["status"] == "passed"
    assert manifest["conditioned_example_count"] == 1
    assert manifest["blocked_example_count"] == 0
    assert manifest["release_ready"] is False
    assert manifest["safety"]["candidate_source_is_not_ground_truth"] is True

    rows = _read_jsonl(Path(manifest["candidate_conditioned_examples_jsonl"]))
    assert rows[0]["schema"] == "generic_finetuning_audio_training_example.v1"
    assert rows[0]["id"] == "candidate_conditioned__row-a"
    assert rows[0]["source_audio_path"] == candidate_audio
    assert rows[0]["target_audio_path"] == base_row["target_audio_path"]
    assert rows[0]["original_source_audio_path"] == base_row["source_audio_path"]
    assert rows[0]["source_candidate_conditioned"] is True
    assert rows[0]["source_candidate_is_ground_truth"] is False
    assert rows[0]["human_verified"] is True
    assert rows[0]["reference_verified"] is False
    assert rows[0]["candidate_execution_manifest_json"] == str(candidate_manifest)
    assert rows[0]["release_ready"] is False
    assert rows[0]["delivery_allowed"] is False


def test_blocks_missing_candidate_instead_of_falling_back_to_raw_mix(tmp_path: Path) -> None:
    base_row = _base_row(tmp_path, "row-a", "track-a")
    base_jsonl = _write_jsonl(tmp_path / "base.jsonl", [base_row])

    manifest = build_candidate_conditioned_audio_training_view(
        audio_training_examples_jsonl=base_jsonl,
        candidate_root=tmp_path / "candidate_root",
        output_dir=tmp_path / "out",
        stem="vocal",
        scope="candidate_conditioned_test",
    )

    assert manifest["status"] == "failed"
    assert manifest["conditioned_example_count"] == 0
    blocked = _read_jsonl(Path(manifest["blocked_examples_jsonl"]))
    assert blocked[0]["id"] == "row-a"
    assert "candidate_execution_manifest_missing" in blocked[0]["blocked_reasons"]
    assert blocked[0]["fallback_used"] is False
