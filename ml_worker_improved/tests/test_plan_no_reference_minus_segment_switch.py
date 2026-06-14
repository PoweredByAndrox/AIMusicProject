from __future__ import annotations

import json
from pathlib import Path

from tools.plan_no_reference_minus_segment_switch import build_plan, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_audio_placeholder(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return str(path)


def self_judge_manifest(
    tmp_path: Path,
    candidate_id: str,
    *,
    blocker_segments: set[int],
    severity: float = 0.4,
) -> dict:
    segments = []
    for index in range(4):
        blocker = index in blocker_segments
        segments.append(
            {
                "segment_id": f"segment_{index:04d}",
                "window_sec": [float(index), float(index + 1)],
                "blocker": blocker,
                "issues": ["vocal_projection_proxy"] if blocker else [],
                "severity_score": severity if blocker else 0.0,
            }
        )
    return {
        "mode": "instrumental_self_judge_gate",
        "source_audio_path": write_audio_placeholder(tmp_path / "mix.wav"),
        "candidate": {
            "label": candidate_id,
            "instrumental_path": write_audio_placeholder(tmp_path / candidate_id / "minus.wav"),
            "vocal_path": write_audio_placeholder(tmp_path / candidate_id / "vocal.wav"),
        },
        "overall": {
            "status": "proxy_blocked" if blocker_segments else "proxy_pass",
            "blocker_count": len(blocker_segments),
            "segment_count": 4,
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "presentation_status": "diagnostic_self_judge_not_for_listen_pack",
        },
        "segments": segments,
    }


def score_matrix_for(candidate_id: str) -> dict:
    return {
        "manifest_type": "no_reference_minus_candidate_score_matrix",
        "rows": [
            {
                "rank": 1,
                "selection_priority": 0.95,
                "canonical_candidate_id": candidate_id,
                "next_action": "run_segment_router_for_self_blockers",
            }
        ],
    }


def test_plan_selects_matrix_base_and_safe_alternates(tmp_path: Path) -> None:
    base_path = write_json(
        tmp_path / "base" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "base", blocker_segments={1, 2}),
    )
    alt_a_path = write_json(
        tmp_path / "alt-a" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "alt-a", blocker_segments={2}),
    )
    alt_b_path = write_json(
        tmp_path / "alt-b" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "alt-b", blocker_segments={1}),
    )
    matrix_path = write_json(tmp_path / "minus_score_matrix.json", score_matrix_for("base"))

    plan = build_plan(
        self_judge_manifest_paths=[base_path, alt_a_path, alt_b_path],
        candidate_score_matrix_json=matrix_path,
        explicit_base_candidate_id=None,
        render_output_dir=tmp_path / "render",
    )

    assert plan["plan_status"] == "ready_for_render"
    assert plan["render_allowed"] is True
    assert plan["base_candidate_id"] == "base"
    assert plan["base_selection_source"] == "candidate_score_matrix"
    assert plan["selected_segment_count"] == 2
    assert plan["unresolved_segment_count"] == 0
    assert [row["selected_candidate_id"] for row in plan["selected_segments"]] == ["alt-a", "alt-b"]
    assert "--base-candidate-id" in plan["recommended_render_args"]
    assert str(base_path) in plan["recommended_render_args"]
    assert plan["delivery_allowed"] is False
    assert plan["release_ready"] is False


def test_plan_blocks_unresolved_base_blocker_segment(tmp_path: Path) -> None:
    base_path = write_json(
        tmp_path / "base" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "base", blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "alt", blocker_segments={1}),
    )

    plan = build_plan(
        self_judge_manifest_paths=[base_path, alt_path],
        candidate_score_matrix_json=None,
        explicit_base_candidate_id="base",
        render_output_dir=tmp_path / "render",
    )

    assert plan["plan_status"] == "blocked_unresolved_segments"
    assert plan["render_allowed"] is False
    assert plan["selected_segment_count"] == 0
    assert plan["unresolved_segment_count"] == 1
    assert plan["unresolved_segments"][0]["segment_id"] == "segment_0001"
    assert plan["unresolved_segments"][0]["blocked_reason"] == "no_safe_non_blocking_alternate_segment"
    assert plan["recommended_render_args"] == []
    assert plan["delivery_allowed"] is False


def test_cli_writes_deterministic_minus_segment_switch_plan(tmp_path: Path) -> None:
    base_path = write_json(
        tmp_path / "base" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "base", blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "alt", blocker_segments=set()),
    )
    output_json = tmp_path / "plan.json"
    summary_md = tmp_path / "plan.md"
    args = [
        "--self-judge-manifest",
        str(base_path),
        "--self-judge-manifest",
        str(alt_path),
        "--base-candidate-id",
        "base",
        "--render-output-dir",
        str(tmp_path / "render"),
        "--output-json",
        str(output_json),
        "--summary-md",
        str(summary_md),
    ]

    assert main(args) == 0
    first = output_json.read_text(encoding="utf-8")
    assert main(args) == 0
    assert output_json.read_text(encoding="utf-8") == first
    payload = json.loads(first)
    assert payload["plan_status"] == "ready_for_render"
    assert "plan_status: ready_for_render" in summary_md.read_text(encoding="utf-8")
