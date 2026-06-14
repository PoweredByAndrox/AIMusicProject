from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.build_generic_audio_training_approval_pack import build_generic_audio_training_approval_pack, main


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recovery_row(
    tmp_path: Path,
    row_id: str,
    *,
    priority: str,
    source_domain: str,
    blockers: list[str] | None = None,
    track_id: str | None = None,
    stem_type: str = "vocal",
) -> dict:
    source = tmp_path / "source" / f"{row_id}.flac"
    target = tmp_path / "target" / f"{row_id}.flac"
    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    blockers = blockers if blockers is not None else ["missing_permission_status"]
    return {
        "schema": "generic_audio_training_recovery_queue_row.v1",
        "id": row_id,
        "source_type": "feedback_jsonl",
        "source_file": str(tmp_path / "feedback.jsonl"),
        "source_row_index": 1,
        "track_id": track_id or f"track-{row_id}",
        "stem_type": stem_type,
        "source_audio_path": str(source),
        "target_audio_path": str(target),
        "source_audio_exists": True,
        "target_audio_exists": True,
        "rating": "good",
        "human_verified": True,
        "reference_verified": False,
        "permission_status": "",
        "blockers": blockers,
        "ready_for_audio_training": False,
        "source_domain": source_domain,
        "recovery_priority": priority,
        "recommended_actions": ["attach_trainable_permission_status_or_exclude"],
    }


def test_builds_permission_only_p0_p1_approval_pack(tmp_path: Path) -> None:
    queue = write_jsonl(
        tmp_path / "queue.jsonl",
        [
            recovery_row(
                tmp_path,
                "p1",
                priority="P1_permission_only_verified_music_gt",
                source_domain="user_supplied_flstudio_music",
                track_id="track-b",
                stem_type="minus",
            ),
            recovery_row(
                tmp_path,
                "p0",
                priority="P0_permission_only_verified_feedback_pair",
                source_domain="human_feedback_audio_pair",
                track_id="track-a",
                stem_type="vocal",
            ),
            recovery_row(
                tmp_path,
                "p4",
                priority="P4_synthetic_fixture_not_production_training",
                source_domain="synthetic_fixture",
            ),
            recovery_row(
                tmp_path,
                "p3",
                priority="P3_recover_audio_paths",
                source_domain="human_feedback_audio_pair",
                blockers=["missing_permission_status", "target_audio_not_found"],
            ),
        ],
    )

    manifest = build_generic_audio_training_approval_pack(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["candidate_rows"] == 2
    assert manifest["summary"]["non_candidate_rows"] == 2
    assert manifest["summary"]["unique_track_count"] == 2
    assert manifest["summary"]["recovery_priority_counts"] == {
        "P0_permission_only_verified_feedback_pair": 1,
        "P1_permission_only_verified_music_gt": 1,
    }
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["release_ready"] is False

    candidates = read_jsonl(tmp_path / "out" / "generic_audio_training_approval_candidates.jsonl")
    assert [row["recovery_row_id"] for row in candidates] == ["p0", "p1"]
    assert candidates[0]["approval_candidate_status"] == "needs_permission_review"
    assert candidates[0]["allowed_permission_statuses"] == ["owned", "licensed", "cleared", "project_owned"]

    draft = read_jsonl(tmp_path / "out" / "generic_audio_training_draft_decisions.jsonl")
    assert [row["recovery_row_id"] for row in draft] == ["p0", "p1"]
    assert {row["decision"] for row in draft} == {"needs_more_evidence"}
    assert {row["permission_status"] for row in draft} == {""}


def test_builds_approval_pack_for_permission_not_trainable_only_row(tmp_path: Path) -> None:
    queue = write_jsonl(
        tmp_path / "queue.jsonl",
        [
            recovery_row(
                tmp_path,
                "requires-review",
                priority="P0_permission_only_verified_feedback_pair",
                source_domain="human_feedback_audio_pair",
                blockers=["permission_not_trainable"],
            )
        ],
    )

    manifest = build_generic_audio_training_approval_pack(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["candidate_rows"] == 1
    candidates = read_jsonl(tmp_path / "out" / "generic_audio_training_approval_candidates.jsonl")
    assert candidates[0]["current_blockers"] == ["permission_not_trainable"]
    draft = read_jsonl(tmp_path / "out" / "generic_audio_training_draft_decisions.jsonl")
    assert draft[0]["recovery_row_id"] == "requires-review"


def test_cli_writes_approval_pack_manifest(tmp_path: Path) -> None:
    queue = write_jsonl(
        tmp_path / "queue.jsonl",
        [
            recovery_row(
                tmp_path,
                "p0",
                priority="P0_permission_only_verified_feedback_pair",
                source_domain="human_feedback_audio_pair",
            )
        ],
    )

    assert main(["--recovery-queue-jsonl", str(queue), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / "generic_audio_training_approval_pack_manifest.json")
    assert manifest["summary"]["candidate_rows"] == 1
    assert manifest["outputs"]["draft_decisions_jsonl"].endswith("generic_audio_training_draft_decisions.jsonl")


def test_direct_script_entrypoint_writes_approval_pack(tmp_path: Path) -> None:
    queue = write_jsonl(
        tmp_path / "queue.jsonl",
        [
            recovery_row(
                tmp_path,
                "p0",
                priority="P0_permission_only_verified_feedback_pair",
                source_domain="human_feedback_audio_pair",
            )
        ],
    )
    script = Path(__file__).resolve().parents[1] / "tools" / "build_generic_audio_training_approval_pack.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--recovery-queue-jsonl",
            str(queue),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = read_json(tmp_path / "out" / "generic_audio_training_approval_pack_manifest.json")
    assert manifest["summary"]["candidate_rows"] == 1
