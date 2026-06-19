from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_aris_refinement_matrix import parse_args, run_aris_refinement_matrix


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _plan(root: Path, *, families: list[str] | None = None, route_id: str = "failure_conditioned_refinement") -> Path:
    return _write_json(
        root / "analysis" / "top1_top2_plan.json",
        {
            "schema": "aris_top1_top2_rag_plan.v1",
            "route_id": route_id,
            "candidate_families": families or ["demucs", "scnet", "bs_roformer"],
            "priority_failures": [
                {"label": "vocal_bleed", "count": 1197},
                {"label": "minus_detail_gate_blocked", "count": 305},
            ],
        },
    )


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(tmp_path / "pool.jsonl", [])

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        run_aris_refinement_matrix(
            artifact_rag_root=rag_root,
            planning_json=plan_json,
            candidate_pool_jsonl=pool,
            output_dir=tmp_path / "loose_output",
        )


def test_rejects_output_dir_with_unsafe_segments_inside_artifact_rag(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(tmp_path / "pool.jsonl", [])

    with pytest.raises(ValueError, match="output_dir must use short Windows-safe relative path inside artifact_rag_root"):
        run_aris_refinement_matrix(
            artifact_rag_root=rag_root,
            planning_json=plan_json,
            candidate_pool_jsonl=pool,
            output_dir=rag_root / "runs" / "Track Name With Spaces And A Very Long Raw Title",
        )


def test_rejects_old_primary_directions_from_plan(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root, families=["demucs", "qwen_controller"])
    pool = _write_jsonl(tmp_path / "pool.jsonl", [])

    with pytest.raises(ValueError, match="forbidden primary direction"):
        run_aris_refinement_matrix(
            artifact_rag_root=rag_root,
            planning_json=plan_json,
            candidate_pool_jsonl=pool,
            output_dir=rag_root / "runs" / "r001",
        )


def test_rejected_candidates_are_blocked_from_listen_pack(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "track_alias": "t001",
                "stem": "vocal",
                "family": "demucs",
                "candidate_alias": "c001",
                "candidate_status": "failed",
                "failed_checks": ["vocal_bleed"],
                "short_audio_path": "audio/t001/vocal.flac",
            },
            {
                "track_alias": "t001",
                "stem": "minus",
                "family": "demucs",
                "candidate_alias": "c002",
                "candidate_status": "failed",
                "failed_checks": ["minus_detail_gate_blocked"],
                "short_audio_path": "audio/t001/minus.flac",
            },
        ],
    )

    manifest = run_aris_refinement_matrix(
        artifact_rag_root=rag_root,
        planning_json=plan_json,
        candidate_pool_jsonl=pool,
        output_dir=rag_root / "runs" / "r001",
    )

    assert manifest["accepted_count"] == 0
    assert manifest["rejected_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    accepted_rows = _read_jsonl(Path(manifest["accepted_jsonl"]))
    rejected_rows = _read_jsonl(Path(manifest["rejected_jsonl"]))
    assert accepted_rows == []
    assert rejected_rows[0]["delivery_status"] == "rejected"
    assert rejected_rows[0]["delivery_allowed"] is False
    assert rejected_rows[0]["presentation_status"] == "blocked_from_listen_pack"
    assert rejected_rows[0]["fallback_used"] is False
    assert set(rejected_rows[0]["failed_checks"]) == {"vocal_bleed", "minus_detail_gate_blocked"}
    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert {row["event"] for row in progress_rows} >= {"run_start", "track_start", "gate_finish", "run_finish"}
    assert all("expected_acceleration_mode" in row for row in progress_rows)
    assert all("gpu_inference_skipped" in row for row in progress_rows)


def test_accepts_only_vocal_minus_pair_with_required_delivery_contract(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "track_alias": "t001",
                "stem": "vocal",
                "family": "scnet",
                "candidate_alias": "v001",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "short_audio_path": "audio/t001/vocal.flac",
            },
            {
                "track_alias": "t001",
                "stem": "minus",
                "family": "bs_roformer",
                "candidate_alias": "m001",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "short_audio_path": "audio/t001/minus.flac",
            },
        ],
    )

    manifest = run_aris_refinement_matrix(
        artifact_rag_root=rag_root,
        planning_json=plan_json,
        candidate_pool_jsonl=pool,
        output_dir=rag_root / "runs" / "r001",
    )

    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 0
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    row = _read_jsonl(Path(manifest["accepted_jsonl"]))[0]
    assert row["delivery_status"] == "accepted"
    assert row["delivery_allowed"] is True
    assert row["presentation_status"] == "accepted_for_listen_pack"
    assert row["release_ready"] is False
    assert row["accepted_stems"] == ["vocal", "minus"]
    assert row["objective_evidence"] == {
        "reference_gate": "passed",
        "no_reference_gate": "not_required",
        "pair_consistency_gate": "passed",
    }


def test_rejects_passed_candidates_without_objective_gate_evidence(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "track_alias": "t001",
                "stem": "vocal",
                "family": "scnet",
                "candidate_alias": "v001",
                "candidate_status": "passed",
                "pair_consistency_gate": "passed",
                "short_audio_path": "audio/t001/vocal.flac",
            },
            {
                "track_alias": "t001",
                "stem": "minus",
                "family": "bs_roformer",
                "candidate_alias": "m001",
                "candidate_status": "passed",
                "pair_consistency_gate": "passed",
                "short_audio_path": "audio/t001/minus.flac",
            },
        ],
    )

    manifest = run_aris_refinement_matrix(
        artifact_rag_root=rag_root,
        planning_json=plan_json,
        candidate_pool_jsonl=pool,
        output_dir=rag_root / "runs" / "r001",
    )

    assert manifest["accepted_count"] == 0
    assert manifest["delivery_allowed"] is False
    rejected = _read_jsonl(Path(manifest["rejected_jsonl"]))
    assert rejected[0]["presentation_status"] == "blocked_from_listen_pack"
    assert "objective_gate_missing" in rejected[0]["failed_checks"]


def test_rejects_fallback_provenance_as_success(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "track_alias": "t001",
                "stem": "vocal",
                "family": "scnet",
                "candidate_alias": "v001",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "fallback_used": True,
                "short_audio_path": "audio/t001/vocal.flac",
            },
            {
                "track_alias": "t001",
                "stem": "minus",
                "family": "bs_roformer",
                "candidate_alias": "m001",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "fallback_used": False,
                "short_audio_path": "audio/t001/minus.flac",
            },
        ],
    )

    manifest = run_aris_refinement_matrix(
        artifact_rag_root=rag_root,
        planning_json=plan_json,
        candidate_pool_jsonl=pool,
        output_dir=rag_root / "runs" / "r001",
    )

    assert manifest["accepted_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert _read_jsonl(Path(manifest["accepted_jsonl"])) == []
    rejected = _read_jsonl(Path(manifest["rejected_jsonl"]))
    assert rejected[0]["presentation_status"] == "blocked_from_listen_pack"
    assert "fallback_used_not_allowed" in rejected[0]["failed_checks"]


def test_rejects_fallback_as_success_provenance(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "track_alias": "t001",
                "stem": "vocal",
                "family": "scnet",
                "candidate_alias": "v001",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "fallback_used": False,
                "fallback_as_success": True,
                "short_audio_path": "audio/t001/vocal.flac",
            },
            {
                "track_alias": "t001",
                "stem": "minus",
                "family": "bs_roformer",
                "candidate_alias": "m001",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "fallback_used": False,
                "short_audio_path": "audio/t001/minus.flac",
            },
        ],
    )

    manifest = run_aris_refinement_matrix(
        artifact_rag_root=rag_root,
        planning_json=plan_json,
        candidate_pool_jsonl=pool,
        output_dir=rag_root / "runs" / "r001",
    )

    assert manifest["accepted_count"] == 0
    rejected = _read_jsonl(Path(manifest["rejected_jsonl"]))
    assert "fallback_used_not_allowed" in rejected[0]["failed_checks"]


def test_rejects_unsafe_or_raw_track_alias_paths(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _plan(rag_root)
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "track_alias": "Track Name With Spaces And A Very Long Raw Title",
                "stem": "vocal",
                "family": "demucs",
                "candidate_alias": "c001",
                "candidate_status": "passed",
                "short_audio_path": "audio/Track Name With Spaces And A Very Long Raw Title/vocal.flac",
            }
        ],
    )

    with pytest.raises(ValueError, match="unsafe artifact alias"):
        run_aris_refinement_matrix(
            artifact_rag_root=rag_root,
            planning_json=plan_json,
            candidate_pool_jsonl=pool,
            output_dir=rag_root / "runs" / "r001",
        )


def test_parse_args_does_not_duplicate_default_target_stems(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--artifact-rag-root",
            str(tmp_path / "artifact_rag"),
            "--planning-json",
            str(tmp_path / "artifact_rag" / "analysis" / "top1_top2_plan.json"),
            "--candidate-pool-jsonl",
            str(tmp_path / "pool.jsonl"),
            "--output-dir",
            str(tmp_path / "artifact_rag" / "runs" / "r001"),
            "--target-stem",
            "vocal",
            "--target-stem",
            "minus",
        ]
    )

    assert args.target_stem == ["vocal", "minus"]
