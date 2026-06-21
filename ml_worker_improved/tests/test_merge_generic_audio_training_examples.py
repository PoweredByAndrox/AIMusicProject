from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tools.merge_generic_audio_training_examples import main, merge_generic_audio_training_examples


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def audio_row(
    tmp_path: Path,
    row_id: str,
    track_id: str,
    stem_type: str,
    *,
    target_payload: bytes | None = None,
    permission_status: str = "project_owned",
    trainable: bool = True,
    target_audio_sha256: str | None = None,
    target_exists: bool = True,
) -> dict:
    source_path = tmp_path / "source" / f"{track_id}.flac"
    target_path = tmp_path / "target" / f"{row_id}.flac"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(f"source-{track_id}".encode("utf-8"))
    payload = target_payload if target_payload is not None else f"target-{row_id}".encode("utf-8")
    if target_exists:
        target_path.write_bytes(payload)
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": row_id,
        "track_id": track_id,
        "stem_type": stem_type,
        "manifest_path": str(tmp_path / "manifest.json"),
        "source_audio_path": str(source_path),
        "target_audio_path": str(target_path),
        "target_audio_sha256": target_audio_sha256 if target_audio_sha256 is not None else sha256_bytes(payload),
        "model_id": "test",
        "rating": "reference_verified",
        "human_score": None,
        "human_verified": False,
        "reference_verified": True,
        "permission_status": permission_status,
        "split": "train",
        "notes": "",
        "trainable": trainable,
        "training_target": "audio_separator_or_stem_adapter",
    }


def test_merges_base_and_approved_examples_with_dedupe_and_computed_sha(tmp_path: Path) -> None:
    base_payload = b"base-target"
    base_row = audio_row(tmp_path, "base-vocal", "track-a", "vocal", target_payload=base_payload)
    duplicate = dict(base_row)
    duplicate["id"] = "approved-duplicate"
    duplicate["model_id"] = "approved_recovery::feedback"
    approved_new = audio_row(
        tmp_path,
        "approved-minus",
        "track-b",
        "minus",
        target_payload=b"approved-target",
        target_audio_sha256="",
    )
    base = write_jsonl(tmp_path / "base.jsonl", [base_row])
    approved = write_jsonl(tmp_path / "approved.jsonl", [duplicate, approved_new])

    manifest = merge_generic_audio_training_examples(
        base_audio_training_examples_jsonls=[base],
        approved_audio_training_examples_jsonls=[approved],
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["input_rows"] == 3
    assert manifest["summary"]["merged_examples"] == 2
    assert manifest["summary"]["duplicate_examples"] == 1
    assert manifest["summary"]["rejected_rows"] == 0
    assert manifest["summary"]["unique_track_count"] == 2
    assert manifest["summary"]["stem_counts"] == {"minus": 1, "vocal": 1}
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["release_ready"] is False

    rows = read_jsonl(tmp_path / "out" / "merged_audio_training_examples.jsonl")
    assert [row["track_id"] for row in rows] == ["track-a", "track-b"]
    assert rows[0]["merge_source"] == "base"
    assert rows[1]["merge_source"] == "approved_recovery"
    assert rows[1]["target_audio_sha256"] == sha256_bytes(b"approved-target")


def test_rejects_untrainable_and_missing_audio_rows_fail_closed(tmp_path: Path) -> None:
    untrainable = audio_row(tmp_path, "bad-permission", "track-a", "vocal", permission_status="requires_review")
    missing_target = audio_row(tmp_path, "missing-target", "track-b", "minus", target_exists=False)
    base = write_jsonl(tmp_path / "base.jsonl", [untrainable, missing_target])

    manifest = merge_generic_audio_training_examples(
        base_audio_training_examples_jsonls=[base],
        approved_audio_training_examples_jsonls=[],
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["merged_examples"] == 0
    assert manifest["summary"]["rejected_rows"] == 2
    rejected = read_jsonl(tmp_path / "out" / "rejected_audio_training_examples.jsonl")
    assert {row["id"] for row in rejected} == {"bad-permission", "missing-target"}
    assert any("permission_not_trainable" in row["rejection_reasons"] for row in rejected)
    assert any("target_audio_not_found" in row["rejection_reasons"] for row in rejected)


def test_cli_writes_merged_manifest(tmp_path: Path) -> None:
    base = write_jsonl(tmp_path / "base.jsonl", [audio_row(tmp_path, "v1", "track-a", "vocal")])
    approved = write_jsonl(tmp_path / "approved.jsonl", [audio_row(tmp_path, "m1", "track-b", "minus")])

    assert (
        main(
            [
                "--base-audio-training-examples-jsonl",
                str(base),
                "--approved-audio-training-examples-jsonl",
                str(approved),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )

    manifest = read_json(tmp_path / "out" / "merged_audio_training_examples_manifest.json")
    assert manifest["summary"]["merged_examples"] == 2
    assert manifest["outputs"]["merged_audio_training_examples_jsonl"].endswith("merged_audio_training_examples.jsonl")


def test_short_output_names_are_available_for_new_artifacts(tmp_path: Path) -> None:
    base = write_jsonl(tmp_path / "base.jsonl", [audio_row(tmp_path, "v1", "track-a", "vocal")])

    manifest = merge_generic_audio_training_examples(
        base_audio_training_examples_jsonls=[base],
        approved_audio_training_examples_jsonls=[],
        output_dir=tmp_path / "out",
        short_output_names=True,
    )

    assert manifest["short_output_names"] is True
    assert Path(manifest["outputs"]["manifest_json"]).name == "merge.json"
    assert Path(manifest["outputs"]["merged_audio_training_examples_jsonl"]).name == "rows.jsonl"
    assert Path(manifest["outputs"]["rejected_audio_training_examples_jsonl"]).name == "rej.jsonl"
    assert Path(manifest["outputs"]["duplicate_audio_training_examples_jsonl"]).name == "dup.jsonl"
    assert Path(manifest["outputs"]["progress_jsonl"]).name == "progress.jsonl"
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_direct_script_entrypoint_writes_merged_manifest(tmp_path: Path) -> None:
    base = write_jsonl(tmp_path / "base.jsonl", [audio_row(tmp_path, "v1", "track-a", "vocal")])
    script = Path(__file__).resolve().parents[1] / "tools" / "merge_generic_audio_training_examples.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-audio-training-examples-jsonl",
            str(base),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = read_json(tmp_path / "out" / "merged_audio_training_examples_manifest.json")
    assert manifest["summary"]["merged_examples"] == 1
