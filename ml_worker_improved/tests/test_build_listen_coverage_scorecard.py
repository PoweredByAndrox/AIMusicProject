from __future__ import annotations

import json
from pathlib import Path

from tools.build_listen_coverage_scorecard import build_listen_coverage_scorecard


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _pack(tmp_path: Path, *, release_ready: bool = False) -> Path:
    items = []
    for group in ("trk1", "trk2", "trk3"):
        for stem in ("vocal", "minus", "drums", "bass", "other", "guitar", "piano"):
            items.append(
                {
                    "source_alias": group,
                    "track_alias": group,
                    "audio_path": f"audio/{group}/t001/{stem}.flac",
                    "stem_type": stem,
                    "duration_sec": 72.0,
                    "semantic_quality_verified": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            )
    items.append(
        {
            "source_alias": "mfs",
            "track_alias": "mfs",
            "audio_path": "audio/mfs/min/m001.wav",
            "stem_type": "minus",
            "duration_sec": 150.0,
            "semantic_quality_verified": False,
            "release_ready": False,
            "can_publish_as_result": False,
        }
    )
    return _write_json(
        tmp_path / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "item_count": len(items),
            "items": items,
            "failed_checks": [],
        },
    )


def _coverage(tmp_path: Path, *, release_ready: bool = False) -> Path:
    return _write_json(
        tmp_path / "coverage.json",
        {
            "schema": "aggregate_short_listen_pack_coverage.v1",
            "pack_id": "lp_test",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "failed_checks": [],
            "stem_groups": [
                {
                    "group_alias": group,
                    "item_count": 7,
                    "semantic_quality_verified": False,
                    "stem_types": ["bass", "drums", "guitar", "minus", "other", "piano", "vocal"],
                }
                for group in ("trk1", "trk2", "trk3")
            ],
        },
    )


def _goal_audit(tmp_path: Path, *, release_ready: bool = False) -> Path:
    return _write_json(
        tmp_path / "goal" / "goal_status_audit.json",
        {
            "schema": "short_listen_goal_status_audit.v1",
            "goal_status": "machine_pass_human_listen_pending",
            "goal_complete": False,
            "remaining_blockers": ["human_listen_verdict_missing"],
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "checks": [
                {"id": "short_windows_safe_names", "status": "proved", "blockers": []},
                {
                    "id": "six_stem_coverage",
                    "status": "proved",
                    "blockers": [],
                    "evidence": {"complete_group_count": 3},
                },
                {"id": "human_listen_verdict", "status": "blocked", "blockers": ["human_listen_verdict_missing"]},
            ],
        },
    )


def _idea_funnel(
    tmp_path: Path,
    *,
    release_ready: bool = False,
    status: str = "ready",
    delivery_allowed: bool = False,
    presentation_status: str = "idea_funnel_not_listen_result",
) -> Path:
    return _write_json(
        tmp_path / "idea" / "funnel.json",
        {
            "schema": "short_listen_idea_funnel.v1",
            "funnel_status": status,
            "idea_count": 4 if status == "ready" else 0,
            "failed_checks": [],
            "delivery_allowed": delivery_allowed,
            "presentation_status": presentation_status,
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "safety": {
                "training_started": False,
                "separator_training_started": False,
                "models_changed": False,
                "model_registry_edited": False,
                "thresholds_relaxed": False,
                "cleanup_routes_added": False,
                "release_ready": False,
            },
        },
    )


def _semantic_review(tmp_path: Path, *, status: str = "pending") -> Path:
    return _write_json(
        tmp_path / "sem" / "review.json",
        {
            "schema": "semantic_stem_quality_review.v1",
            "review_status": status,
            "reviewer_kind": "human_or_machine_pending",
            "semantic_quality_verified_item_count": 0,
            "pending_item_count": 21 if status == "pending" else 0,
            "total_item_count": 21,
            "reviewed_group_aliases": ["hh6", "noe6", "t026"],
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "delivery_allowed": False,
            "presentation_status": "semantic_review_artifact_not_listen_result",
        },
    )


def _minus_audit(tmp_path: Path, *, release_ready: bool = False) -> Path:
    rows_path = tmp_path / "mina" / "rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        json.dumps(
            {
                "schema": "short_minus_quality_audit_row.v1",
                "short_audio_path": "audio/trk1/t001/minus.flac",
                "audit_decision": "blocked",
                "audit_blockers": ["minus_detail_gate_blocked", "windowed_minus_not_accepted"],
                "quality_warning_reasons": ["instrumental_integrity_margin_watch"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return _write_json(
        tmp_path / "mina" / "audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed_with_blockers",
            "accepted_minus_count": 3,
            "blocked_minus_count": 1,
            "evidence_missing_count": 0,
            "warning_count": 1,
            "failed_checks": [],
            "rows_jsonl": str(rows_path),
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "delivery_allowed": False,
        },
    )


def test_build_listen_coverage_scorecard_requires_ready_idea_funnel(tmp_path: Path) -> None:
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        goal_audit_json=_goal_audit(tmp_path),
        idea_funnel_json=_idea_funnel(tmp_path),
        semantic_review_json=_semantic_review(tmp_path, status="completed"),
    )

    assert manifest["schema"] == "listen_coverage_scorecard.v1"
    assert manifest["scorecard_status"] == "machine_pass_human_listen_pending"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["idea_funnel"]["required"] is True
    assert manifest["idea_funnel"]["status"] == "ready"
    assert manifest["idea_funnel"]["idea_count"] == 4
    assert manifest["semantic_review"]["status"] == "completed"
    assert manifest["complete_6stem_group_count"] == 3
    assert manifest["total_item_count"] == 22
    assert manifest["stem_type_counts"]["vocal"] == 3
    assert manifest["stem_type_counts"]["minus"] == 4
    assert manifest["full_length_minus_count"] == 4
    assert manifest["human_listen_pending"] is True
    assert manifest["semantic_verified_group_count"] == 0
    assert "human_listen_verdict_missing" in manifest["remaining_blockers"]
    assert "semantic_quality_not_ground_truth" in manifest["not_claimed"]

    groups = [
        json.loads(line)
        for line in (tmp_path / "score" / "groups.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["group_alias"] for row in groups} == {"trk1", "trk2", "trk3"}
    assert all(row["full_6stem_coverage"] is True for row in groups)
    assert all(row["next_action"] == "semantic_stem_review" for row in groups)
    assert all(len(path.name) <= 32 for path in (tmp_path / "score").iterdir() if path.is_file())


def test_build_listen_coverage_scorecard_blocks_minus_audit_blockers(tmp_path: Path) -> None:
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        goal_audit_json=_goal_audit(tmp_path),
        idea_funnel_json=_idea_funnel(tmp_path),
        semantic_review_json=_semantic_review(tmp_path, status="completed"),
        minus_audit_json=_minus_audit(tmp_path),
    )

    assert manifest["scorecard_status"] == "blocked"
    assert manifest["complete_6stem_group_count"] == 3
    assert manifest["minus_audit"]["status"] == "completed_with_blockers"
    assert manifest["minus_audit"]["blocked_minus_count"] == 1
    assert manifest["minus_audit"]["blocked_audio_paths"] == ["audio/trk1/t001/minus.flac"]
    assert manifest["minus_audit"]["blocker_reasons"] == ["minus_detail_gate_blocked", "windowed_minus_not_accepted"]
    assert "minus_audit_blocked_minus_count:1" in manifest["failed_checks"]
    assert "minus_audit_blocked_minus_count:1" in manifest["remaining_blockers"]
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_build_listen_coverage_scorecard_maps_semantic_review_track_aliases(tmp_path: Path) -> None:
    semantic_pack = _write_json(
        tmp_path / "sem" / "pack.json",
        {
            "schema": "music_intelligence_stem_listen_pack_manifest.v1",
            "release_ready": False,
            "can_publish_as_result": False,
            "track_aliases": [
                {"source_track_id": "trk1", "track_alias": "t001"},
                {"source_track_id": "trk2", "track_alias": "t002"},
                {"source_track_id": "trk3", "track_alias": "t003"},
            ],
        },
    )
    semantic_review = _write_json(
        tmp_path / "sem" / "review.json",
        {
            "schema": "semantic_stem_quality_review.v1",
            "review_status": "completed",
            "reviewer_kind": "machine_proxy_semantic_batch",
            "semantic_quality_verified_item_count": 21,
            "pending_item_count": 0,
            "total_item_count": 21,
            "reviewed_group_aliases": ["t001", "t002", "t003"],
            "input_pack_jsons": [str(semantic_pack)],
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "delivery_allowed": False,
            "presentation_status": "semantic_review_artifact_not_listen_result",
        },
    )

    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        goal_audit_json=_goal_audit(tmp_path),
        idea_funnel_json=_idea_funnel(tmp_path),
        semantic_review_json=semantic_review,
    )

    groups = [
        json.loads(line)
        for line in (tmp_path / "score" / "groups.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["semantic_verified_group_count"] == 3
    assert manifest["semantic_unverified_group_count"] == 0
    assert all(row["semantic_quality_verified"] is True for row in groups)
    assert all(row["next_action"] == "human_listen_verdict" for row in groups)


def test_build_listen_coverage_scorecard_blocks_release_ready_sources(tmp_path: Path) -> None:
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path, release_ready=True),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        goal_audit_json=_goal_audit(tmp_path),
        idea_funnel_json=_idea_funnel(tmp_path),
    )

    assert manifest["scorecard_status"] == "blocked"
    assert "pack_release_ready_true" in manifest["failed_checks"]
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_build_listen_coverage_scorecard_blocks_missing_idea_funnel(tmp_path: Path) -> None:
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        goal_audit_json=_goal_audit(tmp_path),
    )

    assert manifest["scorecard_status"] == "blocked"
    assert "idea_funnel_missing" in manifest["failed_checks"]
    assert manifest["idea_funnel"]["required"] is True
    assert manifest["idea_funnel"]["status"] == "missing"


def test_build_listen_coverage_scorecard_blocks_missing_goal_audit(tmp_path: Path) -> None:
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        idea_funnel_json=_idea_funnel(tmp_path),
    )

    assert manifest["scorecard_status"] == "blocked"
    assert "goal_audit_missing" in manifest["failed_checks"]
    assert manifest["human_listen_pending"] is False


def test_build_listen_coverage_scorecard_blocks_deliverable_idea_funnel(tmp_path: Path) -> None:
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        goal_audit_json=_goal_audit(tmp_path),
        idea_funnel_json=_idea_funnel(
            tmp_path,
            delivery_allowed=True,
            presentation_status="accepted_for_listen_pack",
        ),
    )

    assert manifest["scorecard_status"] == "blocked"
    assert "idea_funnel_delivery_allowed_not_false" in manifest["failed_checks"]
    assert "idea_funnel_presentation_status_unexpected" in manifest["failed_checks"]
    assert manifest["idea_funnel"]["delivery_allowed"] is False
    assert manifest["idea_funnel"]["presentation_status"] == "idea_funnel_not_listen_result"


def test_build_listen_coverage_scorecard_blocks_pending_semantic_stem_review(tmp_path: Path) -> None:
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=_pack(tmp_path),
        coverage_json=_coverage(tmp_path),
        output_dir=tmp_path / "score",
        goal_audit_json=_goal_audit(tmp_path),
        idea_funnel_json=_idea_funnel(tmp_path),
        semantic_review_json=_semantic_review(tmp_path, status="pending"),
    )

    assert manifest["scorecard_status"] == "blocked"
    assert "semantic_stem_review_pending" in manifest["failed_checks"]
    assert manifest["semantic_review"]["required"] is True
    assert manifest["semantic_review"]["status"] == "pending"
    assert manifest["semantic_review"]["pending_item_count"] == 21
