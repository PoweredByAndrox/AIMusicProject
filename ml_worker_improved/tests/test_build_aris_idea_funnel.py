from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_aris_idea_funnel import build_aris_idea_funnel, parse_args


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rag_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "artifact_rag"
    _write_json(
        root / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "la001",
            "status": "thin_rag_binary_pruned",
            "current_retention_mode": "thin_rag_only",
            "binary_audio_model_bytes_available": False,
            "idx": "la001/idx.json",
        },
    )
    top_plan = _write_json(
        root / "analysis" / "r001" / "plan.json",
        {
            "schema": "aris_top1_top2_rag_plan.v1",
            "route_id": "failure_conditioned_refinement",
            "status": "planning_only",
            "candidate_families": ["demucs", "htdemucs", "scnet", "bs_roformer", "melband", "mdx23c", "bs_mamba2"],
            "priority_failures": [
                {"failure": "holdout_row_failed", "count": 40},
                {"failure": "minus_detail_gate_blocked", "count": 20},
                {"failure": "transient_preservation_below_detail_gate", "count": 10},
            ],
            "explicitly_cut_directions": {
                "qwen_controller": "cut_for_now_more_control_logic_same_bad_stems",
                "diffusion_generative_repair": "cut_for_now_faithfulness_risk",
                "repeat_teacher_mask_adapter": "cut_as_primary_previous_holdout_failures",
            },
        },
    )
    rejected = _write_jsonl(
        root / "runs" / "r001" / "rejected.jsonl",
        [
            {
                "track_alias": "t001",
                "failed_checks": ["binary_audio_unavailable", "minus_candidate_missing", "objective_gate_missing"],
                "presentation_status": "blocked_from_listen_pack",
            },
            {
                "track_alias": "t002",
                "failed_checks": ["minus_detail_gate_blocked", "pair_consistency_gate_failed"],
                "presentation_status": "blocked_from_listen_pack",
            },
        ],
    )
    matrix_manifest = _write_json(
        root / "runs" / "r001" / "manifest.json",
        {
            "schema": "aris_refinement_matrix_manifest.v1",
            "status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "rejected_jsonl": str(rejected),
            "accepted_count": 0,
            "rejected_count": 2,
        },
    )
    return root, top_plan, matrix_manifest


def test_builds_non_old_idea_funnel_from_rag_plan_and_matrix_evidence(tmp_path: Path) -> None:
    root, top_plan, matrix_manifest = _rag_inputs(tmp_path)

    manifest = build_aris_idea_funnel(
        artifact_rag_root=root,
        top_plan_json=top_plan,
        matrix_manifest_json=matrix_manifest,
        output_dir=root / "analysis" / "r002",
        command_args=["--artifact-rag-root", str(root)],
    )

    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    funnel = json.loads(Path(manifest["funnel_json"]).read_text(encoding="utf-8"))
    ideas = _read_jsonl(Path(manifest["ideas_jsonl"]))
    assert funnel["schema"] == "aris_strict_idea_funnel.v1"
    assert funnel["confirmed_problem"]["not_a_threshold_strictness_problem"] is True
    assert funnel["top_routes"][0]["route_id"] == "paired_vocal_minus_model_zoo_ladder"
    assert "binary_audio_unavailable" in funnel["failure_histogram"]
    assert all(idea["delivery_allowed"] is False for idea in ideas)
    assert all("qwen_controller" not in json.dumps(idea, sort_keys=True) for idea in ideas)
    assert all("diffusion_generative_repair" not in idea["route_id"] for idea in ideas)
    assert funnel["blocked_old_directions"]["repeat_teacher_mask_adapter"] == "cut_as_primary_previous_holdout_failures"


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    root, top_plan, matrix_manifest = _rag_inputs(tmp_path)

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        build_aris_idea_funnel(
            artifact_rag_root=root,
            top_plan_json=top_plan,
            matrix_manifest_json=matrix_manifest,
            output_dir=tmp_path / "outside",
        )


def test_parse_args_accepts_matrix_manifest(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--artifact-rag-root",
            str(tmp_path / "artifact_rag"),
            "--top-plan-json",
            str(tmp_path / "artifact_rag" / "analysis" / "r001" / "plan.json"),
            "--matrix-manifest-json",
            str(tmp_path / "artifact_rag" / "runs" / "r001" / "manifest.json"),
            "--output-dir",
            str(tmp_path / "artifact_rag" / "analysis" / "r002"),
        ]
    )

    assert args.matrix_manifest_json.name == "manifest.json"
