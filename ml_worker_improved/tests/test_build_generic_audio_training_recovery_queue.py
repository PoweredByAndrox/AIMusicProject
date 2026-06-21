from __future__ import annotations

import json
from pathlib import Path

from tools.build_generic_audio_training_recovery_queue import build_generic_audio_training_recovery_queue, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_recovery_queue_marks_verified_ground_truth_as_ready(tmp_path: Path) -> None:
    root = tmp_path / "lab" / "gt"
    (root / "stems").mkdir(parents=True)
    (root / "source_mix.flac").write_bytes(b"mix")
    (root / "stems" / "vocals.flac").write_bytes(b"vocal")
    write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": "gt-track",
            "ground_truth_contract": {"verified": True},
            "mix": {"path": "source_mix.flac"},
            "stems": {"vocals": {"path": "stems/vocals.flac"}},
            "report_alignment_contract": {"rights_status": "user_owned"},
        },
    )

    manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
    )

    assert manifest["summary"]["ready_for_audio_training"] == 1
    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["ready_for_audio_training"] is True
    assert rows[0]["permission_status"] == "project_owned"
    assert rows[0]["reference_verified"] is True


def test_recovery_queue_prioritizes_user_supplied_music_ground_truth_permission_gap(tmp_path: Path) -> None:
    root = tmp_path / "lab" / "gt_music"
    (root / "stems").mkdir(parents=True)
    (root / "source_mix.flac").write_bytes(b"mix")
    (root / "stems" / "vocals.flac").write_bytes(b"vocal")
    write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": "music-track",
            "ground_truth_contract": {
                "verified": True,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {"vocals": {"path": "stems/vocals.flac"}},
        },
    )

    build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
    )

    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    template_rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_decision_template.jsonl")
    assert rows[0]["source_domain"] == "user_supplied_flstudio_music"
    assert rows[0]["recovery_priority"] == "P1_permission_only_verified_music_gt"
    assert template_rows[0]["source_domain"] == "user_supplied_flstudio_music"
    assert template_rows[0]["recovery_priority"] == "P1_permission_only_verified_music_gt"


def test_recovery_queue_marks_synthetic_ground_truth_as_fixture(tmp_path: Path) -> None:
    root = tmp_path / "lab" / "synthetic" / "ground_truth"
    (root / "stems").mkdir(parents=True)
    (root / "source_mix.flac").write_bytes(b"mix")
    (root / "stems" / "vocals.flac").write_bytes(b"vocal")
    write_json(
        root / "ground_truth_manifest.json",
        {
            "ground_truth_contract": {
                "verified": True,
                "source": "ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
                "verification_method": "synthetic_deterministic_signal_generation",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {"vocals": {"path": "stems/vocals.flac"}},
        },
    )

    build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
    )

    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["source_domain"] == "synthetic_fixture"
    assert rows[0]["recovery_priority"] == "P4_synthetic_fixture_not_production_training"
    assert "mark_synthetic_training_allowed_or_keep_as_fixture" in rows[0]["recommended_actions"]


def test_recovery_queue_blocks_feedback_without_permission(tmp_path: Path) -> None:
    source = tmp_path / "source.flac"
    target = tmp_path / "target.flac"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "source_audio_path": str(source),
                "candidate_audio_path": str(target),
                "target_stem": "instrumental",
                "rating": "good",
                "human_verified": True,
            }
        ],
    )

    manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path,
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
    )

    assert manifest["summary"]["blocked_rows"] == 1
    assert manifest["summary"]["missing_permission_status"] == 1
    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["stem_type"] == "minus"
    assert rows[0]["blockers"] == ["missing_permission_status"]
    assert rows[0]["recommended_actions"] == ["attach_trainable_permission_status_or_exclude"]
    template_rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_decision_template.jsonl")
    assert template_rows == [
        {
            "schema": "generic_audio_training_recovery_decision_template_row.v1",
            "recovery_row_id": rows[0]["id"],
            "decision": "needs_more_evidence",
            "recommended_permission_status": "",
            "allowed_permission_statuses": ["owned", "licensed", "cleared", "project_owned"],
            "track_id": rows[0]["track_id"],
            "stem_type": "minus",
            "source_domain": "human_feedback_audio_pair",
            "recovery_priority": "P0_permission_only_verified_feedback_pair",
            "blockers": ["missing_permission_status"],
            "recommended_actions": ["attach_trainable_permission_status_or_exclude"],
        }
    ]


def test_recovery_queue_links_full_pack_loop_plan_stem_gaps(tmp_path: Path) -> None:
    root = tmp_path / "lab" / "gt_music"
    (root / "stems").mkdir(parents=True)
    (root / "source_mix.flac").write_bytes(b"mix")
    (root / "stems" / "drums.flac").write_bytes(b"drums")
    write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": "music-track",
            "ground_truth_contract": {
                "verified": True,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {"drums": {"path": "stems/drums.flac"}},
            "report_alignment_contract": {"rights_status": "user_owned"},
        },
    )
    plan_rows = write_jsonl(
        tmp_path / "plan" / "rows.jsonl",
        [
            {
                "schema": "full_pack_loop_iteration_action.v1",
                "queue_item_id": "q002",
                "stem_type": "drums",
                "route_intent": "stem_training_coverage_gap",
                "action_status": "blocked_insufficient_unique_tracks",
                "unique_track_count": 1,
                "min_unique_tracks": 3,
                "required_next_actions": ["verified_reference_corpus_expand"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "full_pack_loop_iteration_action.v1",
                "queue_item_id": "q005",
                "stem_type": "guitar",
                "route_intent": "stem_training_coverage_gap",
                "action_status": "blocked_missing_training_job",
                "required_next_actions": ["verified_reference_corpus_expand", "rerun_generic_audio_finetune_job_plan"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
    )
    plan = write_json(
        tmp_path / "plan" / "plan.json",
        {
            "schema": "full_pack_loop_iteration_plan.v1",
            "plan_status": "blocked_waiting_for_verified_reference_corpus",
            "action_count": 2,
            "rows_jsonl": str(plan_rows),
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "failed_checks": [],
        },
    )

    manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        full_pack_loop_plan_json=plan,
    )

    assert manifest["summary"]["full_pack_required_stems"] == ["drums", "guitar"]
    assert manifest["summary"]["full_pack_missing_candidate_stems"] == ["guitar"]
    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    by_stem = {row["stem_type"]: row for row in rows}
    assert by_stem["drums"]["full_pack_required_stem"] is True
    assert by_stem["drums"]["full_pack_action_status"] == "blocked_insufficient_unique_tracks"
    assert by_stem["drums"]["ready_for_audio_training"] is True
    assert by_stem["guitar"]["source_type"] == "full_pack_loop_plan_missing_stem"
    assert by_stem["guitar"]["ready_for_audio_training"] is False
    assert by_stem["guitar"]["blockers"] == ["missing_verified_reference_candidate"]
    assert "collect_verified_reference_stem" in by_stem["guitar"]["recommended_actions"]
    template_rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_decision_template.jsonl")
    assert any(row["stem_type"] == "guitar" for row in template_rows)


def test_recovery_queue_recovers_feedback_audio_paths_from_listen_manifest(tmp_path: Path) -> None:
    source = tmp_path / "mix.mp3"
    target = tmp_path / "vocals.flac"
    source.write_bytes(b"mix")
    target.write_bytes(b"vocal")
    manifest = write_json(
        tmp_path / "listen_manifest.json",
        {
            "delivery_allowed": True,
            "delivery_status": "accepted",
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "stem_type": "vocal",
            "track_id": "manifest-track",
            "mix_path": str(source),
            "selected_vocal_path": str(target),
        },
    )
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "manifest_path": str(manifest),
                "target_stem": "vocal",
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
            }
        ],
    )

    recovery_manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path,
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
    )

    assert recovery_manifest["summary"]["ready_for_audio_training"] == 1
    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["track_id"] == "manifest-track"
    assert rows[0]["source_audio_path"] == str(source)
    assert rows[0]["target_audio_path"] == str(target)
    assert rows[0]["ready_for_audio_training"] is True
    assert rows[0]["blockers"] == []
    assert rows[0]["source_manifest_path"] == str(manifest)


def test_recovery_queue_recovers_feedback_source_from_candidate_map_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source_mix.flac"
    source.write_bytes(b"mix")
    listen_dir = tmp_path / "listen_pack"
    listen_target = listen_dir / "candidate_16_bs_artifact_clean.flac"
    listen_target.parent.mkdir(parents=True)
    listen_target.write_bytes(b"selected vocal")
    source_sha = "abc123"

    candidate_dir = tmp_path / "adaptive_run" / "candidates" / "bs_artifact_clean"
    candidate_vocal = candidate_dir / "vocals.flac"
    candidate_vocal.parent.mkdir(parents=True)
    candidate_vocal.write_bytes(b"selected vocal")
    write_json(
        candidate_dir / "stem_manifest.json",
        {
            "audio_path": str(source),
            "source_audio_sha256": source_sha,
            "stems": {"vocals": {"path": str(candidate_vocal)}},
        },
    )
    write_json(
        listen_dir / "candidate_map.json",
        {"candidate_16_bs_artifact_clean.flac": str(candidate_vocal)},
    )
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "source_artifact": str(listen_target),
                "source_audio_sha256": source_sha,
                "rating": "better",
                "listener_verified": True,
                "permission_status": "owned",
            }
        ],
    )

    recovery_manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path,
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
    )

    assert recovery_manifest["summary"]["ready_for_audio_training"] == 1
    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["source_audio_path"] == str(source)
    assert rows[0]["target_audio_path"] == str(listen_target)
    assert rows[0]["stem_type"] == "vocal"
    assert rows[0]["source_recovery_method"] == "candidate_map_stem_manifest"
    assert rows[0]["source_recovery_mapped_target_path"] == str(candidate_vocal)
    assert rows[0]["ready_for_audio_training"] is True
    assert rows[0]["blockers"] == []


def test_recovery_queue_recovers_candidate_map_source_from_trainer_manifest_shape(tmp_path: Path) -> None:
    source = tmp_path / "source_mix.flac"
    source.write_bytes(b"mix")
    listen_dir = tmp_path / "listen_pack"
    listen_target = listen_dir / "candidate_01_teacher_mask.flac"
    listen_target.parent.mkdir(parents=True)
    listen_target.write_bytes(b"selected vocal")

    trainer_dir = tmp_path / "trainer_run"
    trainer_vocal = trainer_dir / "01_teacher_mask_vocal.flac"
    trainer_vocal.parent.mkdir(parents=True)
    trainer_vocal.write_bytes(b"selected vocal")
    write_json(
        trainer_dir / "stem_manifest.json",
        {
            "source_audio_sha256": "source-sha",
            "inputs": {"source_mix": {"path": str(source)}},
            "stems": {"vocal": {"path": str(trainer_vocal)}},
        },
    )
    write_json(listen_dir / "candidate_map.json", {listen_target.name: str(trainer_vocal)})
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "source_artifact": str(listen_target),
                "source_audio_sha256": "source-sha",
                "rating": "better",
                "listener_verified": True,
                "permission_status": "owned",
            }
        ],
    )

    manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path,
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
    )

    assert manifest["summary"]["ready_for_audio_training"] == 1
    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["source_audio_path"] == str(source)
    assert rows[0]["stem_type"] == "vocal"
    assert rows[0]["source_recovery_method"] == "candidate_map_stem_manifest"


def test_recovery_queue_treats_requires_review_feedback_pair_as_permission_only(tmp_path: Path) -> None:
    source = tmp_path / "mix.mp3"
    target = tmp_path / "vocals.flac"
    source.write_bytes(b"mix")
    target.write_bytes(b"vocal")
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "source_audio_path": str(source),
                "candidate_audio_path": str(target),
                "target_stem": "vocal",
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "requires_review",
            }
        ],
    )

    build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path,
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
    )

    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["blockers"] == ["permission_not_trainable"]
    assert rows[0]["recovery_priority"] == "P0_permission_only_verified_feedback_pair"
    template_rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_decision_template.jsonl")
    assert template_rows[0]["recovery_row_id"] == rows[0]["id"]


def test_decision_template_is_sorted_by_recovery_priority(tmp_path: Path) -> None:
    gt_root = tmp_path / "lab" / "gt_music"
    (gt_root / "stems").mkdir(parents=True)
    (gt_root / "source_mix.flac").write_bytes(b"mix")
    (gt_root / "stems" / "vocals.flac").write_bytes(b"vocal")
    write_json(
        gt_root / "ground_truth_manifest.json",
        {
            "track_name": "music-track",
            "ground_truth_contract": {
                "verified": True,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {"vocals": {"path": "stems/vocals.flac"}},
        },
    )
    source = tmp_path / "source.flac"
    target = tmp_path / "target.flac"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "source_audio_path": str(source),
                "candidate_audio_path": str(target),
                "target_stem": "vocal",
                "rating": "good",
                "human_verified": True,
            }
        ],
    )

    build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
    )

    template_rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_decision_template.jsonl")
    assert [row["recovery_priority"] for row in template_rows] == [
        "P0_permission_only_verified_feedback_pair",
        "P1_permission_only_verified_music_gt",
    ]


def test_recovery_queue_blocks_missing_target_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.flac"
    source.write_bytes(b"source")
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "source_audio_path": str(source),
                "candidate_audio_path": str(tmp_path / "missing.flac"),
                "target_stem": "vocal",
                "rating": "better",
                "human_verified": True,
                "permission_status": "owned",
            }
        ],
    )

    manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path,
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
    )

    assert manifest["summary"]["target_audio_not_found"] == 1
    rows = read_jsonl(tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl")
    assert rows[0]["ready_for_audio_training"] is False
    assert rows[0]["blockers"] == ["target_audio_not_found"]


def test_recovery_queue_can_write_short_output_names(tmp_path: Path) -> None:
    source = tmp_path / "source.flac"
    target = tmp_path / "target.flac"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "source_audio_path": str(source),
                "candidate_audio_path": str(target),
                "target_stem": "vocal",
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
            }
        ],
    )

    manifest = build_generic_audio_training_recovery_queue(
        artifact_root=tmp_path,
        output_dir=tmp_path / "out",
        feedback_jsonls=[feedback],
        short_output_names=True,
    )

    assert manifest["short_output_names"] is True
    assert Path(manifest["outputs"]["queue_jsonl"]).name == "queue.jsonl"
    assert Path(manifest["outputs"]["decision_template_jsonl"]).name == "tpl.jsonl"
    assert Path(manifest["outputs"]["manifest_json"]).name == "rec.json"
    assert Path(manifest["outputs"]["progress_jsonl"]).name == "progress.jsonl"
    assert all(len(Path(path).name) <= 32 for path in manifest["outputs"].values())
    assert (tmp_path / "out" / "queue.jsonl").exists()
    assert not (tmp_path / "out" / "generic_audio_training_recovery_queue.jsonl").exists()


def test_cli_writes_recovery_manifest(tmp_path: Path) -> None:
    feedback = write_jsonl(tmp_path / "feedback.jsonl", [{"rating": "accept", "human_verified": True}])

    assert (
        main(
            [
                "--artifact-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--feedback-jsonl",
                str(feedback),
            ]
        )
        == 0
    )

    manifest = read_json(tmp_path / "out" / "generic_audio_training_recovery_manifest.json")
    assert manifest["summary"]["recovery_rows"] == 1
    assert manifest["training_started"] is False


def test_cli_can_write_short_recovery_outputs(tmp_path: Path) -> None:
    feedback = write_jsonl(tmp_path / "feedback.jsonl", [{"rating": "accept", "human_verified": True}])

    assert (
        main(
            [
                "--artifact-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--feedback-jsonl",
                str(feedback),
                "--short-output-names",
            ]
        )
        == 0
    )

    manifest = read_json(tmp_path / "out" / "rec.json")
    assert manifest["short_output_names"] is True
    assert Path(manifest["outputs"]["queue_jsonl"]).name == "queue.jsonl"
