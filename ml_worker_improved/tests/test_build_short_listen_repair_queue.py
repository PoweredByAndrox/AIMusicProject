from __future__ import annotations

import json
from pathlib import Path

from tools.build_short_listen_repair_queue import build_short_listen_repair_queue


FULL_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _feedback(tmp_path: Path, *, overall: str = "rejected", failed: list[str] | None = None, release_ready: bool = False) -> Path:
    items = [
        {
            "schema": "short_listen_feedback_item.v1",
            "audio_path": "audio/voc/v002.flac",
            "verdict": "rejected",
            "issue_tags": ["bad_vocal_quality", "user_reported_unacceptable"],
            "note": "Rejected vocal.",
            "source_alias": "voc",
            "stem_type": "vocal",
            "track_alias": "voc",
            "source_manifest": "/app/source/voc/pack.json",
            "source_audio_path": "/app/source/voc/v002.flac",
            "source_row": {"track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)"},
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_listen_feedback_item.v1",
            "audio_path": "audio/noe/minus.flac",
            "verdict": "rejected",
            "issue_tags": ["muffled_minus", "residual_vocal_middle", "user_reported_unacceptable"],
            "note": "Middle has vocal and muffling.",
            "source_alias": "noe",
            "stem_type": "minus_preview",
            "track_alias": "noe",
            "source_manifest": "/app/source/noe/listen_check_manifest.json",
            "source_audio_path": "/app/source/noe/minus.flac",
            "source_row": {
                "judge_evidence": {
                    "applied_window_count": 2,
                    "applied_windows": [
                        {"alternate_candidate_id": "c005", "start_sec": 48.0, "duration_sec": 12.0},
                        {"alternate_candidate_id": "c005", "start_sec": 60.0, "duration_sec": 12.0},
                    ],
                    "full_result_vocal_leakage_in_minus_score": 0.073294,
                    "max_vocal_leakage_in_minus_score": 0.148468,
                    "min_mixture_consistency_score": 0.715377,
                }
            },
            "release_ready": False,
            "can_publish_as_result": False,
        },
    ]
    if overall != "rejected":
        items = []
    return _write_json(
        tmp_path / "listen_feedback.json",
        {
            "schema": "short_listen_feedback_manifest.v1",
            "pack_id": "lp_test",
            "pack_manifest_json": "/app/pack.json",
            "overall_listen_verdict": overall,
            "review_coverage_status": "partial",
            "reviewed_item_count": len(items),
            "pack_item_count": 29,
            "feedback_item_count": len(items),
            "items": items,
            "rejected_audio_paths": [item["audio_path"] for item in items],
            "failed_checks": failed or [],
            "release_ready": release_ready,
            "can_publish_as_result": False,
        },
    )


def _minus_audit(tmp_path: Path, *, release_ready: bool = False) -> Path:
    rows_path = tmp_path / "audit" / "rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "schema": "short_minus_quality_audit_row.v1",
            "short_audio_path": "audio/sub/g002.flac",
            "source_alias": "mfs",
            "track_alias": "no_escape",
            "stem_type": "minus",
            "audit_decision": "passed",
            "audit_blockers": [],
            "quality_warning_reasons": ["residual_vocal_watch"],
            "windowed_vocal_leakage_in_minus_score": 0.186911,
            "source_manifest": "/app/src/minus_pack.json",
            "source_audio_path": "/app/src/audio/minus.flac",
            "source_row": {"render_manifest": "render.json", "track_id": "no_escape"},
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_minus_quality_audit_row.v1",
            "short_audio_path": "audio/hh6/t001/minus.flac",
            "source_alias": "hh6",
            "track_alias": "t001",
            "stem_type": "minus",
            "audit_decision": "blocked",
            "audit_blockers": ["severe_vocal_bleed_in_minus"],
            "quality_warning_reasons": ["residual_vocal_watch"],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    ]
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return _write_json(
        tmp_path / "audit" / "audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed_with_blockers",
            "rows_jsonl": str(rows_path),
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "delivery_allowed": False,
            "failed_checks": ["minus_audit_blocked:audio/hh6/t001/minus.flac:severe_vocal_bleed_in_minus"],
        },
    )


def _full_pack(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "pack" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "items": [
                {
                    "schema": "aggregate_short_listen_pack_audio.v1",
                    "audio_path": f"audio/stm/t001/{stem}.flac",
                    "stem_type": stem,
                    "source_manifest": str(tmp_path / "source" / stem / "pack.json"),
                    "source_audio_path": str(tmp_path / "source" / stem / f"{stem}.flac"),
                    "source_row": {
                        "track_id": "trk_full",
                        "stem_type": stem,
                        "source_manifest": str(tmp_path / "source" / stem / "pack.json"),
                    },
                    "semantic_quality_verified": True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
                for stem in FULL_STEMS
            ],
        },
    )


def _machine_gate(tmp_path: Path, *, accepted: bool = False, release_ready: bool = False) -> Path:
    failed_checks = []
    blocked_training_job_stems = {}
    if not accepted:
        failed_checks = [
            "minus_audit_blocked_minus_present",
            "minus_audit_failed_checks_present",
            "required_stem_train_ready_missing:minus",
            "required_stem_train_ready_missing:drums",
            "required_stem_train_ready_missing:bass",
            "required_stem_train_ready_missing:other",
            "required_stem_train_ready_missing:guitar",
            "required_stem_train_ready_missing:piano",
            "training_job_blocked:minus:blocked_insufficient_unique_tracks",
            "training_job_blocked:drums:blocked_insufficient_unique_tracks",
            "training_job_blocked:bass:blocked_insufficient_unique_tracks",
            "training_job_blocked:other:blocked_insufficient_unique_tracks",
            "training_job_blocked:piano:blocked_insufficient_unique_tracks",
        ]
        blocked_training_job_stems = {
            "minus": ["blocked_insufficient_unique_tracks"],
            "drums": ["blocked_insufficient_unique_tracks"],
            "bass": ["blocked_insufficient_unique_tracks"],
            "other": ["blocked_insufficient_unique_tracks"],
            "piano": ["blocked_insufficient_unique_tracks"],
            "vocal": ["blocked_insufficient_unique_tracks"],
        }
    return _write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "accepted" if accepted else "blocked",
            "final_result_allowed": accepted,
            "delivery_allowed": False,
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "pack_manifest_json": str(_full_pack(tmp_path)),
            "required_stems": FULL_STEMS,
            "train_ready_stems": ["vocal"] if not accepted else FULL_STEMS,
            "blocked_training_job_stems": blocked_training_job_stems,
            "failed_checks": failed_checks,
        },
    )


def test_build_short_listen_repair_queue_routes_rejected_items(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=_feedback(tmp_path),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "ready_for_repair_cycle"
    assert manifest["queue_item_count"] == 2
    assert manifest["route_intent_counts"] == {
        "minus_repair": 1,
        "vocal_candidate_replacement": 1,
    }
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    rows = [json.loads(line) for line in (tmp_path / "queue" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["queue_item_id"] == "q001"
    assert rows[0]["route_intent"] == "vocal_candidate_replacement"
    assert rows[1]["queue_item_id"] == "q002"
    assert rows[1]["route_intent"] == "minus_repair"
    assert rows[1]["diagnostic_context"]["judge_evidence"]["max_vocal_leakage_in_minus_score"] == 0.148468
    assert rows[1]["diagnostic_context"]["target_windows"][0]["start_sec"] == 48.0
    assert "localized_bleed_diagnostic" in rows[1]["required_actions"]
    assert (tmp_path / "queue" / "queue.json").exists()
    assert (tmp_path / "queue" / "summary.md").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "queue").iterdir() if path.is_file())


def test_build_short_listen_repair_queue_routes_minus_audit_warnings(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=None,
        minus_audit_json=_minus_audit(tmp_path),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "ready_for_machine_warning_cycle"
    assert manifest["queue_item_count"] == 2
    assert manifest["queue_source"] == "minus_audit_warning"
    assert manifest["route_intent_counts"] == {"minus_repair": 2}
    assert manifest["release_ready"] is False
    rows = [json.loads(line) for line in (tmp_path / "queue" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["audio_path"] == "audio/sub/g002.flac"
    assert rows[0]["verdict"] == "machine_warning"
    assert rows[0]["issue_tags"] == ["residual_vocal"]
    assert rows[0]["route_intent"] == "minus_repair"
    assert rows[0]["source_manifest"] == "/app/src/minus_pack.json"
    assert rows[0]["source_audio_path"] == "/app/src/audio/minus.flac"
    assert rows[0]["source_row"]["render_manifest"] == "render.json"
    assert rows[0]["diagnostic_context"]["minus_audit"]["windowed_vocal_leakage_in_minus_score"] == 0.186911
    assert "localized_bleed_diagnostic" in rows[0]["required_actions"]
    assert rows[1]["audio_path"] == "audio/hh6/t001/minus.flac"
    assert rows[1]["verdict"] == "machine_blocked"
    assert rows[1]["issue_tags"] == ["residual_vocal", "severe_vocal_bleed_in_minus"]
    assert rows[1]["priority"] == "high"
    assert rows[1]["route_intent"] == "minus_repair"
    assert rows[1]["diagnostic_context"]["minus_audit"]["audit_blockers"] == ["severe_vocal_bleed_in_minus"]
    assert "localized_bleed_diagnostic" in rows[1]["required_actions"]


def test_build_short_listen_repair_queue_routes_full_pack_machine_gate(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        machine_gate_json=_machine_gate(tmp_path),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "ready_for_full_pack_loop"
    assert manifest["queue_source"] == "machine_full_pack_gate"
    assert manifest["queue_item_count"] == 6
    assert manifest["route_intent_counts"] == {
        "minus_repair": 1,
        "stem_training_coverage_gap": 5,
    }
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    rows = [json.loads(line) for line in (tmp_path / "queue" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    by_stem = {row["stem_type"]: row for row in rows}
    assert "vocal" not in by_stem
    assert by_stem["minus"]["audio_path"] == "audio/stm/t001/minus.flac"
    assert by_stem["minus"]["route_intent"] == "minus_repair"
    assert by_stem["minus"]["source_manifest"].endswith("/source/minus/pack.json")
    assert by_stem["minus"]["source_audio_path"].endswith("/source/minus/minus.flac")
    assert by_stem["minus"]["source_row"]["source_manifest"].endswith("/source/minus/pack.json")
    assert "minus_gate_blocked" in by_stem["minus"]["issue_tags"]
    assert "train_ready_model_missing" in by_stem["minus"]["issue_tags"]
    assert "verified_reference_corpus_expand" in by_stem["minus"]["required_actions"]
    assert by_stem["bass"]["route_intent"] == "stem_training_coverage_gap"
    assert "insufficient_unique_tracks" in by_stem["bass"]["issue_tags"]
    assert "gpu_teacher_mask_train" in by_stem["bass"]["required_actions"]
    assert by_stem["guitar"]["issue_tags"] == ["train_ready_model_missing", "training_job_missing"]
    assert "gpu_teacher_mask_train" not in by_stem["guitar"]["required_actions"]
    assert by_stem["guitar"]["diagnostic_context"]["machine_gate"]["train_ready_stems"] == ["vocal"]


def test_build_short_listen_repair_queue_prioritizes_explicit_vocal_stem(tmp_path: Path) -> None:
    feedback_json = _feedback(tmp_path)
    feedback = json.loads(feedback_json.read_text(encoding="utf-8"))
    feedback["items"] = [
        {
            **feedback["items"][0],
            "issue_tags": ["vocal_bleed", "user_reported_unacceptable"],
            "stem_type": "vocal",
        }
    ]
    feedback["rejected_audio_paths"] = ["audio/voc/v002.flac"]
    feedback["reviewed_item_count"] = 1
    feedback["feedback_item_count"] = 1
    feedback_json.write_text(json.dumps(feedback, sort_keys=True), encoding="utf-8")

    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=feedback_json,
        output_dir=tmp_path / "queue",
    )

    rows = [json.loads(line) for line in (tmp_path / "queue" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["route_intent_counts"] == {"vocal_candidate_replacement": 1}
    assert rows[0]["route_intent"] == "vocal_candidate_replacement"


def test_build_short_listen_repair_queue_blocks_non_rejected_feedback(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=_feedback(tmp_path, overall="accepted"),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert "feedback_overall_not_rejected" in manifest["failed_checks"]
    assert manifest["queue_item_count"] == 0


def test_build_short_listen_repair_queue_blocks_failed_feedback_manifest(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=_feedback(tmp_path, failed=["feedback_audio_path_not_in_pack:missing"]),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert "feedback_failed_checks_not_empty" in manifest["failed_checks"]


def test_build_short_listen_repair_queue_blocks_release_ready_feedback(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=_feedback(tmp_path, release_ready=True),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert "feedback_release_ready_true" in manifest["failed_checks"]


def test_build_short_listen_repair_queue_blocks_release_ready_minus_audit(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=None,
        minus_audit_json=_minus_audit(tmp_path, release_ready=True),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert "minus_audit_release_ready_true" in manifest["failed_checks"]


def test_build_short_listen_repair_queue_blocks_accepted_machine_gate(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        machine_gate_json=_machine_gate(tmp_path, accepted=True),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert manifest["queue_item_count"] == 0
    assert "machine_gate_not_blocked" in manifest["failed_checks"]
    assert "machine_gate_failed_checks_missing" in manifest["failed_checks"]


def test_build_short_listen_repair_queue_blocks_release_ready_machine_gate(tmp_path: Path) -> None:
    manifest = build_short_listen_repair_queue(
        machine_gate_json=_machine_gate(tmp_path, release_ready=True),
        output_dir=tmp_path / "queue",
    )

    assert manifest["queue_status"] == "blocked"
    assert manifest["queue_item_count"] == 0
    assert "machine_gate_release_ready_true" in manifest["failed_checks"]
