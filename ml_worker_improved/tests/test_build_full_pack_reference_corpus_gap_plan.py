from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_reference_corpus_gap_plan import build_full_pack_reference_corpus_gap_plan


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _blocked_full_pack_plan(tmp_path: Path, *, delivery_allowed: bool = False) -> Path:
    rows = _write_jsonl(
        tmp_path / "fullp" / "rows.jsonl",
        [
            {
                "schema": "full_pack_loop_iteration_action.v1",
                "queue_item_id": "q002",
                "stem_type": "drums",
                "route_intent": "stem_training_coverage_gap",
                "action_status": "blocked_insufficient_unique_tracks",
                "unique_track_count": 1,
                "min_unique_tracks": 3,
                "required_next_actions": ["verified_reference_corpus_expand", "holdout_eval_gate"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "full_pack_loop_iteration_action.v1",
                "queue_item_id": "q005",
                "stem_type": "guitar",
                "route_intent": "stem_training_coverage_gap",
                "action_status": "blocked_missing_training_job",
                "unique_track_count": None,
                "min_unique_tracks": None,
                "required_next_actions": ["verified_reference_corpus_expand", "rerun_generic_audio_finetune_job_plan"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "fullp" / "plan.json",
        {
            "schema": "full_pack_loop_iteration_plan.v1",
            "plan_status": "blocked_waiting_for_verified_reference_corpus",
            "action_count": 2,
            "rows_jsonl": str(rows),
            "failed_checks": [],
            "delivery_allowed": delivery_allowed,
            "release_ready": False,
            "training_started": False,
        },
    )


def _training_plan(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "tr" / "generic_audio_finetune_job_plan.json",
        {
            "schema": "generic_audio_finetune_job_plan.v1",
            "summary": {
                "trainable_audio_examples": 6,
                "job_count": 1,
                "ready_job_count": 0,
                "blocked_job_count": 1,
                "min_unique_tracks": 3,
            },
            "jobs": [
                {
                    "schema": "generic_audio_finetune_job.v1",
                    "scope": "stem_teacher_mask_adapter",
                    "status": "blocked_insufficient_unique_tracks",
                    "stem_types": ["drums"],
                    "example_count": 6,
                    "unique_track_count": 1,
                    "min_unique_tracks": 3,
                    "training_started": False,
                    "release_ready": False,
                    "delivery_allowed": False,
                }
            ],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )


def _corpus_validation(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "val" / "verified_audio_training_corpus_manifest.json",
        {
            "schema": "verified_audio_training_corpus_manifest.v1",
            "corpus_validation_status": "failed",
            "failed_checks": ["insufficient_unique_tracks"],
            "row_count": 6,
            "verified_trainable_examples": 6,
            "unique_track_count": 1,
            "min_unique_tracks": 3,
            "unsafe_row_count": 0,
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def test_builds_reference_corpus_gap_plan_from_full_pack_blockers(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_corpus_gap_plan(
        full_pack_loop_plan_json=_blocked_full_pack_plan(tmp_path),
        training_plan_json=_training_plan(tmp_path),
        verified_corpus_manifest_json=_corpus_validation(tmp_path),
        output_dir=tmp_path / "gap",
    )

    assert manifest["schema"] == "full_pack_reference_corpus_gap_plan.v1"
    assert manifest["gap_status"] == "blocked_waiting_for_verified_reference_corpus"
    assert manifest["gap_count"] == 2
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "gap" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_stem = {row["stem_type"]: row for row in rows}
    assert by_stem["drums"]["missing_unique_track_count"] == 2
    assert by_stem["drums"]["required_next_actions"] == ["collect_verified_reference_stem", "rerun_corpus_validation", "rerun_training_plan"]
    assert by_stem["guitar"]["missing_training_job"] is True
    assert by_stem["guitar"]["missing_unique_track_count"] == 3
    assert "collect_verified_reference_stem" in by_stem["guitar"]["required_next_actions"]
    assert all(len(path.name) <= 32 for path in (tmp_path / "gap").iterdir() if path.is_file())


def test_blocks_unsafe_full_pack_plan(tmp_path: Path) -> None:
    manifest = build_full_pack_reference_corpus_gap_plan(
        full_pack_loop_plan_json=_blocked_full_pack_plan(tmp_path, delivery_allowed=True),
        training_plan_json=_training_plan(tmp_path),
        verified_corpus_manifest_json=_corpus_validation(tmp_path),
        output_dir=tmp_path / "gap",
    )

    assert manifest["gap_status"] == "blocked"
    assert manifest["gap_count"] == 0
    assert "full_pack_loop_plan_delivery_allowed_true" in manifest["failed_checks"]
