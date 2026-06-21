from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_loop_iteration_plan import build_full_pack_loop_iteration_plan


FULL_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def full_pack_loop_queue(tmp_path: Path) -> Path:
    rows_path = tmp_path / "queue" / "rows.jsonl"
    rows = [
        {
            "schema": "short_listen_repair_queue_item.v1",
            "queue_item_id": "q001",
            "audio_path": "audio/stm/t001/minus.flac",
            "route_intent": "minus_repair",
            "stem_type": "minus",
            "issue_tags": [
                "minus_gate_blocked",
                "train_ready_model_missing",
                "training_job_blocked",
                "insufficient_unique_tracks",
            ],
            "required_actions": [
                "window_rejudge",
                "candidate_mining",
                "segment_blend_or_rerender",
                "verified_reference_corpus_expand",
                "train_ready_decision_gate",
            ],
            "source_row": {"machine_gate_json": "/app/gate/gate.json"},
            "release_ready": False,
            "can_publish_as_result": False,
        },
        *[
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": f"q{index:03d}",
                "audio_path": f"audio/stm/t001/{stem}.flac",
                "route_intent": "stem_training_coverage_gap",
                "stem_type": stem,
                "issue_tags": ["train_ready_model_missing", "training_job_blocked", "insufficient_unique_tracks"],
                "required_actions": [
                    "verified_reference_corpus_expand",
                    "gpu_teacher_mask_train",
                    "holdout_eval_gate",
                    "train_ready_decision_gate",
                    "full_pack_gate_rebuild",
                ],
                "release_ready": False,
                "can_publish_as_result": False,
            }
            for index, stem in enumerate(["drums", "bass", "other", "piano"], start=2)
        ],
        {
            "schema": "short_listen_repair_queue_item.v1",
            "queue_item_id": "q006",
            "audio_path": "audio/stm/t001/guitar.flac",
            "route_intent": "stem_training_coverage_gap",
            "stem_type": "guitar",
            "issue_tags": ["train_ready_model_missing", "training_job_missing"],
            "required_actions": ["verified_reference_corpus_expand", "train_ready_decision_gate", "full_pack_gate_rebuild"],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    ]
    write_jsonl(rows_path, rows)
    return write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_full_pack_loop",
            "queue_source": "machine_full_pack_gate",
            "queue_item_count": len(rows),
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "delivery_allowed": False,
            "safety": {
                "release_ready": False,
                "training_started": False,
                "thresholds_relaxed": False,
                "model_registry_edited": False,
            },
        },
    )


def training_plan(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "training" / "plan.json",
        {
            "schema": "generic_audio_finetune_job_plan.v1",
            "summary": {"ready_job_count": 0, "blocked_job_count": 2, "min_unique_tracks": 3},
            "training_started": False,
            "fine_tune_started": False,
            "release_ready": False,
            "delivery_allowed": False,
            "jobs": [
                {
                    "scope": "minus_teacher_mask_adapter",
                    "status": "blocked_insufficient_unique_tracks",
                    "stem_types": ["minus"],
                    "unique_track_count": 1,
                    "min_unique_tracks": 3,
                    "blocker_reason": "audio fine-tune requires more unique verified tracks",
                },
                {
                    "scope": "stem_teacher_mask_adapter",
                    "status": "blocked_insufficient_unique_tracks",
                    "stem_types": ["drums", "bass", "other", "piano"],
                    "unique_track_count": 1,
                    "min_unique_tracks": 3,
                    "blocker_reason": "audio fine-tune requires more unique verified tracks",
                },
            ],
        },
    )


def ready_training_plan(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "training" / "ready_plan.json",
        {
            "schema": "generic_audio_finetune_job_plan.v1",
            "summary": {"ready_job_count": 1, "blocked_job_count": 0, "min_unique_tracks": 3},
            "training_started": False,
            "fine_tune_started": False,
            "release_ready": False,
            "delivery_allowed": False,
            "jobs": [
                {
                    "scope": "stem_teacher_mask_adapter",
                    "status": "ready_for_gpu_training",
                    "stem_types": ["drums"],
                    "unique_track_count": 3,
                    "min_unique_tracks": 3,
                    "accelerator_mode": "gpu_training_required",
                    "device": "cuda",
                    "example_jobs": [
                        {
                            "example_id": "d001",
                            "command": [
                                "python",
                                "/app/tools/train_teacher_mask_separator.py",
                                "--source-mix",
                                "/app/audio/mix.flac",
                                "--teacher-target",
                                "/app/audio/drums.flac",
                                "--stem",
                                "drums",
                                "--device",
                                "cuda",
                            ],
                        }
                    ],
                }
            ],
        },
    )


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_pack_loop_iteration_plan_blocks_missing_training_coverage(tmp_path: Path) -> None:
    manifest = build_full_pack_loop_iteration_plan(
        repair_queue_manifest_json=full_pack_loop_queue(tmp_path),
        training_plan_json=training_plan(tmp_path),
        output_dir=tmp_path / "plan",
    )

    assert manifest["plan_status"] == "blocked_waiting_for_verified_reference_corpus_and_minus_repair"
    assert manifest["execution_started"] is False
    assert manifest["training_started"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["action_count"] == 6
    assert manifest["ready_action_count"] == 0
    assert manifest["blocked_action_count"] == 6
    assert manifest["route_intent_counts"] == {"minus_repair": 1, "stem_training_coverage_gap": 5}
    plan_rows = rows(tmp_path / "plan" / "rows.jsonl")
    by_stem = {row["stem_type"]: row for row in plan_rows}
    assert by_stem["minus"]["action_status"] == "blocked_minus_repair_and_training_coverage"
    assert by_stem["minus"]["training_job_statuses"] == ["blocked_insufficient_unique_tracks"]
    assert "run_minus_repair_route" in by_stem["minus"]["required_next_actions"]
    assert by_stem["drums"]["action_status"] == "blocked_insufficient_unique_tracks"
    assert by_stem["drums"]["unique_track_count"] == 1
    assert by_stem["guitar"]["action_status"] == "blocked_missing_training_job"
    assert "rerun_generic_audio_finetune_job_plan" in by_stem["guitar"]["required_next_actions"]
    assert (tmp_path / "plan" / "plan.json").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "plan").iterdir() if path.is_file())


def test_full_pack_loop_iteration_plan_marks_ready_gpu_training_actions(tmp_path: Path) -> None:
    queue_manifest = full_pack_loop_queue(tmp_path)
    queue = json.loads(queue_manifest.read_text(encoding="utf-8"))
    queue["rows_jsonl"] = str(write_jsonl(tmp_path / "queue" / "rows.jsonl", [
        {
            "schema": "short_listen_repair_queue_item.v1",
            "queue_item_id": "q001",
            "audio_path": "audio/stm/t001/drums.flac",
            "route_intent": "stem_training_coverage_gap",
            "stem_type": "drums",
            "issue_tags": ["train_ready_model_missing", "training_job_blocked"],
            "required_actions": ["gpu_teacher_mask_train", "train_ready_decision_gate", "full_pack_gate_rebuild"],
            "release_ready": False,
            "can_publish_as_result": False,
        }
    ]))
    queue["queue_item_count"] = 1
    queue_manifest.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")

    manifest = build_full_pack_loop_iteration_plan(
        repair_queue_manifest_json=queue_manifest,
        training_plan_json=ready_training_plan(tmp_path),
        output_dir=tmp_path / "plan",
    )

    assert manifest["plan_status"] == "ready_for_gpu_training_actions"
    assert manifest["ready_action_count"] == 1
    assert manifest["blocked_action_count"] == 0
    assert manifest["delivery_allowed"] is False
    plan_rows = rows(tmp_path / "plan" / "rows.jsonl")
    assert plan_rows[0]["action_status"] == "ready_for_gpu_training"
    assert plan_rows[0]["expected_acceleration_mode"] == "gpu_training"
    assert plan_rows[0]["docker_commands"][0][-1] == "cuda"

