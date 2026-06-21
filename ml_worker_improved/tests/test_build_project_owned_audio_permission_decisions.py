from __future__ import annotations

import json
from pathlib import Path

from tools.build_project_owned_audio_permission_decisions import build_project_owned_audio_permission_decisions, main


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _audio(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return str(path)


def _recovery_row(tmp_path: Path, row_id: str, **overrides: object) -> dict:
    row = {
        "schema": "generic_audio_training_recovery_queue_row.v1",
        "id": row_id,
        "source_type": "ground_truth_manifest",
        "source_domain": "user_supplied_flstudio_music",
        "source_file": str(tmp_path / "golden_audio" / row_id / "ground_truth_manifest.json"),
        "track_id": f"track-{row_id}",
        "stem_type": "vocal",
        "source_audio_path": _audio(tmp_path / row_id / "source.flac"),
        "target_audio_path": _audio(tmp_path / row_id / "target.flac"),
        "source_audio_exists": True,
        "target_audio_exists": True,
        "reference_verified": True,
        "human_verified": False,
        "permission_status": "",
        "blockers": ["missing_permission_status"],
        "ready_for_audio_training": False,
        "recovery_priority": "P1_permission_only_verified_music_gt",
    }
    row.update(overrides)
    return row


def test_attests_only_verified_project_owned_ground_truth_rows(tmp_path: Path) -> None:
    queue = _write_jsonl(
        tmp_path / "queue.jsonl",
        [
            _recovery_row(tmp_path, "ok"),
            _recovery_row(tmp_path, "feedback", source_domain="human_feedback_audio_pair", source_type="human_feedback_source_target"),
        ],
    )

    manifest = build_project_owned_audio_permission_decisions(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "out",
        allowed_source_roots=[tmp_path / "golden_audio"],
    )

    assert manifest["decision_rows"] == 1
    assert manifest["rejected_rows"] == 1
    assert manifest["release_ready"] is False
    decisions = _read_jsonl(Path(manifest["decisions_jsonl"]))
    assert decisions == [
        {
            "decision": "approve_for_training",
            "notes": "project_owned_attestation:verified_user_supplied_flstudio_ground_truth",
            "permission_status": "project_owned",
            "recovery_row_id": "ok",
            "split": "train",
        }
    ]


def test_human_feedback_pairs_require_explicit_opt_in(tmp_path: Path) -> None:
    queue = _write_jsonl(
        tmp_path / "queue.jsonl",
        [
            _recovery_row(
                tmp_path,
                "feedback",
                source_domain="human_feedback_audio_pair",
                source_type="human_feedback_source_target",
                human_verified=True,
                reference_verified=False,
                rating="ideal",
                permission_status="requires_review",
                blockers=["permission_not_trainable"],
            )
        ],
    )

    blocked = build_project_owned_audio_permission_decisions(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "blocked",
        allowed_source_roots=[tmp_path],
    )
    assert blocked["decision_rows"] == 0

    allowed = build_project_owned_audio_permission_decisions(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "allowed",
        allowed_source_roots=[tmp_path],
        allow_human_feedback_pairs=True,
    )

    assert allowed["decision_rows"] == 1
    decisions = _read_jsonl(Path(allowed["decisions_jsonl"]))
    assert decisions[0]["recovery_row_id"] == "feedback"
    assert decisions[0]["notes"] == "project_owned_attestation:human_verified_project_audio_pair"


def test_attestation_rejects_unverified_or_dirty_rows(tmp_path: Path) -> None:
    queue = _write_jsonl(
        tmp_path / "queue.jsonl",
        [
            _recovery_row(tmp_path, "unverified", reference_verified=False),
            _recovery_row(tmp_path, "dirty", blockers=["missing_permission_status", "target_audio_not_found"]),
            _recovery_row(tmp_path, "synthetic", source_domain="synthetic_fixture"),
        ],
    )

    manifest = build_project_owned_audio_permission_decisions(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "out",
        allowed_source_roots=[tmp_path / "golden_audio"],
    )

    assert manifest["decision_rows"] == 0
    rejected = _read_jsonl(Path(manifest["rejected_jsonl"]))
    by_id = {row["recovery_row_id"]: row for row in rejected}
    assert "not_reference_verified" in by_id["unverified"]["rejection_reasons"]
    assert "not_permission_only_blocked" in by_id["dirty"]["rejection_reasons"]
    assert "source_domain_not_allowed" in by_id["synthetic"]["rejection_reasons"]


def test_attestation_requires_allowed_source_root(tmp_path: Path) -> None:
    queue = _write_jsonl(tmp_path / "queue.jsonl", [_recovery_row(tmp_path, "outside")])

    manifest = build_project_owned_audio_permission_decisions(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "out",
        allowed_source_roots=[tmp_path / "different_root"],
    )

    assert manifest["decision_rows"] == 0
    rejected = _read_jsonl(Path(manifest["rejected_jsonl"]))
    assert "source_file_not_under_allowed_root" in rejected[0]["rejection_reasons"]


def test_permission_decisions_can_write_short_output_names(tmp_path: Path) -> None:
    queue = _write_jsonl(tmp_path / "queue.jsonl", [_recovery_row(tmp_path, "ok")])

    manifest = build_project_owned_audio_permission_decisions(
        recovery_queue_jsonl=queue,
        output_dir=tmp_path / "out",
        allowed_source_roots=[tmp_path / "golden_audio"],
        short_output_names=True,
    )

    output_paths = [
        manifest["decisions_jsonl"],
        manifest["rejected_jsonl"],
        manifest["manifest_json"],
        manifest["progress_jsonl"],
    ]
    assert manifest["short_output_names"] is True
    assert Path(manifest["decisions_jsonl"]).name == "dec.jsonl"
    assert Path(manifest["rejected_jsonl"]).name == "rej.jsonl"
    assert Path(manifest["manifest_json"]).name == "perm.json"
    assert Path(manifest["progress_jsonl"]).name == "progress.jsonl"
    assert all(len(Path(path).name) <= 32 for path in output_paths)
    assert (tmp_path / "out" / "dec.jsonl").exists()
    assert not (tmp_path / "out" / "project_owned_audio_permission_decisions.jsonl").exists()


def test_permission_decision_cli_can_write_short_outputs(tmp_path: Path) -> None:
    queue = _write_jsonl(tmp_path / "queue.jsonl", [_recovery_row(tmp_path, "ok")])

    assert (
        main(
            [
                "--recovery-queue-jsonl",
                str(queue),
                "--output-dir",
                str(tmp_path / "out"),
                "--allowed-source-root",
                str(tmp_path / "golden_audio"),
                "--short-output-names",
            ]
        )
        == 0
    )

    manifest = _read_json(tmp_path / "out" / "perm.json")
    assert manifest["short_output_names"] is True
    assert Path(manifest["decisions_jsonl"]).name == "dec.jsonl"
