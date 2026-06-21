from __future__ import annotations

import json
from pathlib import Path

from tools.build_listen_idea_funnel import build_listen_idea_funnel


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _goal_audit(tmp_path: Path, *, release_ready: bool = False) -> Path:
    return _write_json(
        tmp_path / "goal" / "goal_status_audit.json",
        {
            "goal_status": "machine_pass_human_listen_pending",
            "goal_complete": False,
            "goal_status_reason": "needs_human_listen_verdict",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "checks": [
                {"id": "short_windows_safe_names", "status": "proved", "blockers": []},
                {
                    "id": "six_stem_coverage",
                    "status": "proved",
                    "blockers": [],
                    "evidence": {
                        "complete_groups": [
                            {
                                "group_alias": "noe6",
                                "stem_types": ["bass", "drums", "guitar", "minus", "other", "piano", "vocal"],
                            }
                        ]
                    },
                },
                {"id": "human_listen_verdict", "status": "blocked", "blockers": ["human_listen_verdict_missing"]},
            ],
            "not_claimed": ["release_ready", "publishable_result"],
        },
    )


def _pack(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "item_count": 2,
            "items": [
                {"audio_path": "audio/mfs/min/m001.wav", "stem_type": "minus", "source_alias": "mfs"},
                {"audio_path": "audio/noe6/t001/vocal.flac", "stem_type": "vocal", "source_alias": "noe6"},
            ],
        },
    )


def _cmp(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "cmp" / "cmp.json",
        {
            "schema": "minus_middle_comparison_report.v1",
            "comparison_status": "completed",
            "row_count": 9,
            "improved_count": 6,
            "regressed_count": 3,
            "inconclusive_count": 0,
            "release_ready": False,
            "can_publish_as_result": False,
            "delivery_allowed": False,
            "failed_checks": [],
        },
    )


def _targeted_cmp(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "tcmp" / "manifest.json",
        {
            "schema": "targeted_minus_window_cmp.v1",
            "comparison_status": "completed",
            "row_count": 3,
            "improved_or_equal_count": 3,
            "regressed_count": 0,
            "release_ready": False,
            "can_publish_as_result": False,
            "delivery_allowed": False,
        },
    )


def _minus_audit(tmp_path: Path) -> Path:
    rows_jsonl = tmp_path / "mina" / "rows.jsonl"
    rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "short_audio_path": "audio/m01/min/m001.flac",
                        "audit_decision": "passed",
                        "quality_warning_reasons": ["residual_vocal_watch"],
                        "release_ready": False,
                        "can_publish_as_result": False,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "short_audio_path": "audio/m02/min/m001.flac",
                        "audit_decision": "passed",
                        "quality_warning_reasons": ["residual_vocal_watch"],
                        "release_ready": False,
                        "can_publish_as_result": False,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "short_audio_path": "audio/m03/min/m001.flac",
                        "audit_decision": "blocked",
                        "audit_blockers": ["minus_detail_gate_blocked", "transient_preservation_below_detail_gate"],
                        "quality_warning_reasons": [],
                        "release_ready": False,
                        "can_publish_as_result": False,
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return _write_json(
        tmp_path / "mina" / "audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed_with_blockers",
            "accepted_minus_count": 2,
            "blocked_minus_count": 1,
            "evidence_missing_count": 0,
            "warning_count": 2,
            "failed_checks": [],
            "rows_jsonl": str(rows_jsonl),
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )


def test_build_listen_idea_funnel_writes_ranked_evidence_backed_ideas(tmp_path: Path) -> None:
    manifest = build_listen_idea_funnel(
        goal_audit_json=_goal_audit(tmp_path),
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "funnel",
        minus_cmp_json=_cmp(tmp_path),
        targeted_cmp_json=_targeted_cmp(tmp_path),
        minus_audit_json=_minus_audit(tmp_path),
    )

    assert manifest["schema"] == "short_listen_idea_funnel.v1"
    assert manifest["funnel_status"] == "ready"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["failed_checks"] == []
    assert manifest["idea_count"] >= 3

    ideas = [json.loads(line) for line in (tmp_path / "funnel" / "ideas.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [idea["idea_id"] for idea in ideas]
    assert ids[0] == "i001_run_short_minus_quality_audit"
    assert ids[1] == "i002_build_minus_audit_gated_subset"
    assert ids[2] == "i003_build_minus_warning_repair_queue"
    assert any(idea_id.endswith("mine_current_minus_warning_rows") for idea_id in ids)
    assert any(idea_id.endswith("collect_human_listen_verdict") for idea_id in ids)
    assert any(idea_id.endswith("keep_segment_switch_for_minus") for idea_id in ids)
    assert any(idea_id.endswith("expand_multi_track_short_pack_cycle") for idea_id in ids)
    assert any(idea_id.endswith("mine_current_minus_blocker_rows") for idea_id in ids)
    minus_audit_idea = ideas[0]
    assert minus_audit_idea["manual_required"] is False
    assert "tools/build_short_minus_quality_audit.py" in minus_audit_idea["docker_command"]
    assert "--pack-manifest-json" in minus_audit_idea["docker_command"]
    assert minus_audit_idea["expected_output"] == "minus_quality_audit_dir/audit.json"
    gated_subset_idea = ideas[1]
    assert "tools/build_short_listen_subset_pack.py" in gated_subset_idea["docker_command"]
    assert "--minus-audit-json <minus_quality_audit_dir>/audit.json" in gated_subset_idea["docker_command"]
    assert gated_subset_idea["expected_output"] == "audit_gated_subset_dir/pack.json"
    machine_warning_idea = ideas[2]
    assert machine_warning_idea["manual_required"] is False
    assert "tools/build_short_listen_repair_queue.py" in machine_warning_idea["docker_command"]
    assert "--minus-audit-json <minus_quality_audit_dir>/audit.json" in machine_warning_idea["docker_command"]
    assert machine_warning_idea["expected_output"] == "minus_warning_queue_dir/queue.json"
    concrete_warning_idea = next(idea for idea in ideas if idea["idea_id"].endswith("mine_current_minus_warning_rows"))
    assert concrete_warning_idea["evidence"]["minus_audit_json"].endswith("audit.json")
    assert concrete_warning_idea["evidence"]["warning_count"] == 2
    assert concrete_warning_idea["evidence"]["warning_reasons"] == ["residual_vocal_watch"]
    assert "audio/m01/min/m001.flac" in concrete_warning_idea["evidence"]["warning_audio_paths"]
    sweep_idea = next(idea for idea in ideas if idea["idea_id"].endswith("run_minus_replacement_sweep"))
    assert sweep_idea["manual_required"] is False
    assert sweep_idea["expected_output"] == "minus_replacement_sweep_dir/sweep.json"
    assert "tools/build_short_minus_replacement_sweep.py" in sweep_idea["docker_command"]
    assert "--baseline-audit-json" in sweep_idea["docker_command"]
    assert sweep_idea["evidence"]["warning_count"] == 2
    concrete_blocker_idea = next(idea for idea in ideas if idea["idea_id"].endswith("mine_current_minus_blocker_rows"))
    assert concrete_blocker_idea["evidence"]["blocked_minus_count"] == 1
    assert concrete_blocker_idea["evidence"]["blocker_reasons"] == [
        "minus_detail_gate_blocked",
        "transient_preservation_below_detail_gate",
    ]
    assert concrete_blocker_idea["evidence"]["blocked_audio_paths"] == ["audio/m03/min/m001.flac"]
    assert "tools/build_short_listen_repair_queue.py" in concrete_blocker_idea["docker_command"]
    assert "tools/build_short_listen_repair_queue.py" in concrete_warning_idea["docker_command"]
    assert ideas[3]["manual_required"] is False
    assert any(idea["manual_required"] is True for idea in ideas)
    assert any(idea["docker_command"].startswith("docker compose run --rm --no-deps ml-worker") for idea in ideas)
    semantic_review_idea = next(idea for idea in ideas if idea["idea_id"].endswith("add_semantic_stem_quality_review"))
    semantic_command = semantic_review_idea["docker_command"]
    assert "--input-pack-json" in semantic_command
    assert "--source-alias noe6" in semantic_command
    assert "--required-stem-type vocal" in semantic_command
    assert "--required-stem-type piano" in semantic_command
    semantic_batch_idea = next(idea for idea in ideas if idea["idea_id"].endswith("run_semantic_stem_batch"))
    semantic_batch_command = semantic_batch_idea["docker_command"]
    assert "tools/run_no_reference_stem_semantic_batch.py" in semantic_batch_command
    assert "--input-pack-json <music_stem_review_dir>/pack.json" in semantic_batch_command
    assert "--output-root <semantic_stem_batch_dir>" in semantic_batch_command
    assert semantic_batch_idea["expected_output"] == "semantic_stem_batch_dir/review/review.json"
    semantic_gap_idea = next(idea for idea in ideas if idea["idea_id"].endswith("plan_semantic_route_gaps"))
    semantic_gap_command = semantic_gap_idea["docker_command"]
    assert "tools/build_semantic_stem_route_gap_plan.py" in semantic_gap_command
    assert "--semantic-review-json <semantic_stem_batch_dir>/review/review.json" in semantic_gap_command
    assert semantic_gap_idea["expected_output"] == "semantic_route_gap_dir/gap.json"
    assert all(idea["release_ready"] is False for idea in ideas)
    assert all(idea["can_publish_as_result"] is False for idea in ideas)
    assert all(len(path.name) <= 32 for path in (tmp_path / "funnel").iterdir() if path.is_file())


def test_build_listen_idea_funnel_blocks_release_ready_goal_audit(tmp_path: Path) -> None:
    manifest = build_listen_idea_funnel(
        goal_audit_json=_goal_audit(tmp_path, release_ready=True),
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "funnel",
        targeted_cmp_json=_targeted_cmp(tmp_path),
    )

    assert manifest["funnel_status"] == "blocked"
    assert "goal_audit_release_ready_true" in manifest["failed_checks"]
    assert manifest["idea_count"] == 0
