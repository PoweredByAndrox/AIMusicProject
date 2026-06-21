from __future__ import annotations

import json
from pathlib import Path

from tools.build_semantic_stem_route_gap_plan import build_semantic_stem_route_gap_plan


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _review(tmp_path: Path, *, release_ready: bool = False) -> Path:
    rows_jsonl = _write_jsonl(
        tmp_path / "review" / "rows.jsonl",
        [
            {
                "schema": "semantic_stem_quality_review_row.v1",
                "group_alias": "t001",
                "stem_type": "bass",
                "semantic_quality_verified": True,
                "review_status": "accepted",
                "blockers": [],
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "semantic_stem_quality_review_row.v1",
                "group_alias": "t001",
                "stem_type": "vocal",
                "semantic_quality_verified": False,
                "review_status": "blocked",
                "blockers": ["unsupported_stem_type"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "semantic_stem_quality_review_row.v1",
                "group_alias": "t002",
                "stem_type": "minus",
                "semantic_quality_verified": False,
                "review_status": "blocked",
                "blockers": ["unsupported_stem_type"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "review" / "review.json",
        {
            "schema": "semantic_stem_quality_review.v1",
            "review_status": "completed",
            "rows_jsonl": str(rows_jsonl),
            "total_item_count": 3,
            "semantic_quality_verified_item_count": 1,
            "pending_item_count": 0,
            "failed_checks": [
                "stem_semantic_blocked:t001:vocal:unsupported_stem_type",
                "stem_semantic_blocked:t002:minus:unsupported_stem_type",
            ],
            "release_ready": release_ready,
            "can_publish_as_result": False,
        },
    )


def test_build_semantic_stem_route_gap_plan_groups_unsupported_stems(tmp_path: Path) -> None:
    manifest = build_semantic_stem_route_gap_plan(
        semantic_review_json=_review(tmp_path),
        output_dir=tmp_path / "gap",
    )

    assert manifest["schema"] == "semantic_stem_route_gap_plan.v1"
    assert manifest["gap_status"] == "route_gaps_ready"
    assert manifest["gap_count"] == 2
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    rows = [json.loads(line) for line in (tmp_path / "gap" / "gaps.jsonl").read_text(encoding="utf-8").splitlines()]
    by_stem = {row["stem_type"]: row for row in rows}
    assert by_stem["vocal"]["recommended_action"] == "add_or_run_vocal_semantic_quality_gate"
    assert by_stem["minus"]["recommended_action"] == "add_or_run_minus_bleed_phase_quality_gate"
    assert by_stem["vocal"]["blocked_group_aliases"] == ["t001"]
    assert by_stem["minus"]["blocked_group_aliases"] == ["t002"]
    assert all(len(path.name) <= 32 for path in (tmp_path / "gap").iterdir() if path.is_file())


def test_build_semantic_stem_route_gap_plan_blocks_release_ready_review(tmp_path: Path) -> None:
    manifest = build_semantic_stem_route_gap_plan(
        semantic_review_json=_review(tmp_path, release_ready=True),
        output_dir=tmp_path / "gap",
    )

    assert manifest["gap_status"] == "blocked"
    assert "semantic_review_release_ready_true" in manifest["failed_checks"]
    assert manifest["gap_count"] == 0
