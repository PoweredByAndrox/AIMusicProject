from __future__ import annotations

import json
from pathlib import Path

from tools.build_candidate_family_blend_locked_split import build_candidate_family_blend_locked_split


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rows() -> list[dict]:
    rows: list[dict] = []
    for track_id in ("track-a", "track-b", "track-c"):
        for family in ("scnet", "melband"):
            rows.append(
                {
                    "id": f"{track_id}-{family}",
                    "track_id": track_id,
                    "stem_type": "vocal",
                    "source_candidate_family": family,
                    "source_audio_path": f"/tmp/{track_id}/{family}.flac",
                    "target_audio_path": f"/tmp/{track_id}/target.flac",
                    "target_audio_sha256": f"{track_id}-target",
                    "trainable": True,
                    "reference_verified": True,
                    "release_ready": False,
                }
            )
    return rows


def test_locked_split_manifest_records_zero_leakage_and_release_false(tmp_path: Path) -> None:
    examples = _write_jsonl(tmp_path / "examples.jsonl", _rows())

    manifest = build_candidate_family_blend_locked_split(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "split",
        stem="vocal",
        families=["scnet", "melband"],
        holdout_track_ids={"track-b", "track-c"},
    )

    assert manifest["locked"] is True
    assert manifest["critical_leakage_count"] == 0
    assert manifest["train_track_count"] == 1
    assert manifest["holdout_track_count"] == 2
    assert manifest["release_ready"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["thresholds_relaxed"] is False
    assert manifest["runtime_gates_changed"] is False

    rows = _read_jsonl(Path(manifest["split_rows_jsonl"]))
    assert {row["split"] for row in rows} == {"train", "holdout"}
    assert {row["track_id"] for row in rows if row["split"] == "train"} == {"track-a"}
    assert {row["track_id"] for row in rows if row["split"] == "holdout"} == {"track-b", "track-c"}
    assert all(row["release_ready"] is False for row in rows)


def test_locked_split_fails_when_holdout_track_also_enters_training(tmp_path: Path) -> None:
    examples = _write_jsonl(tmp_path / "examples.jsonl", _rows())

    manifest = build_candidate_family_blend_locked_split(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "split",
        stem="vocal",
        families=["scnet", "melband"],
        holdout_track_ids={"track-a"},
        force_train_track_ids={"track-a"},
    )

    assert manifest["locked"] is False
    assert manifest["critical_leakage_count"] == 1
    assert "track_split_overlap" in manifest["failed_checks"]


def test_locked_split_blocks_release_ready_rows(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["release_ready"] = True
    examples = _write_jsonl(tmp_path / "examples.jsonl", rows)

    manifest = build_candidate_family_blend_locked_split(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "split",
        stem="vocal",
        families=["scnet", "melband"],
        holdout_track_ids={"track-b", "track-c"},
    )

    assert manifest["locked"] is False
    assert manifest["critical_leakage_count"] == 1
    assert "release_ready_input_row" in manifest["failed_checks"]


def test_locked_split_requires_forced_train_track_to_be_complete(tmp_path: Path) -> None:
    examples = _write_jsonl(tmp_path / "examples.jsonl", _rows())

    manifest = build_candidate_family_blend_locked_split(
        audio_training_examples_jsonl=examples,
        output_dir=tmp_path / "split",
        stem="vocal",
        families=["scnet", "melband"],
        holdout_track_ids={"track-a", "track-b", "track-c"},
        force_train_track_ids={"missing-track"},
    )

    assert manifest["locked"] is False
    assert manifest["critical_leakage_count"] == 1
    assert "forced_train_track_missing_complete_group" in manifest["failed_checks"]
    assert "train_track_missing" in manifest["failed_checks"]
