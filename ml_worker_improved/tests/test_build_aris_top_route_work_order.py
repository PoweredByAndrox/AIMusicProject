from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_aris_top_route_work_order import build_aris_top_route_work_order


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rag_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "artifact_rag"
    funnel = _write_json(
        root / "analysis" / "r005" / "funnel.json",
        {
            "schema": "aris_strict_idea_funnel.v1",
            "top_routes": [
                {
                    "route_id": "holdout_error_map_segment_router",
                    "rank_score": 21508,
                    "candidate_families": ["scnet", "bs_roformer", "mdx23c", "bs_mamba2"],
                    "required_gates": [
                        "reference_alignment_gate",
                        "segment_error_map_coverage_gate",
                        "holdout_no_track_overlap_gate",
                        "no_reference_generalization_gate",
                    ],
                    "matched_failures": {
                        "holdout_row_failed": 152,
                        "instrumental_integrity_below_detail_gate": 31,
                    },
                    "delivery_allowed": False,
                    "presentation_status": "blocked_from_listen_pack",
                },
                {
                    "route_id": "cross_family_pair_consensus_selector",
                    "rank_score": 8465,
                    "candidate_families": ["scnet", "bs_roformer", "melband", "mdx23c", "bs_mamba2", "demucs"],
                    "required_gates": [
                        "pair_complementarity_gate",
                        "mixture_closure_gate",
                        "cross_family_disagreement_gate",
                        "no_best_of_bad_delivery_gate",
                    ],
                    "matched_failures": {"best_matrix_row_missing": 137},
                    "delivery_allowed": False,
                    "presentation_status": "blocked_from_listen_pack",
                },
            ],
            "blocked_old_directions": {
                "qwen_controller": "cut_for_now_more_control_logic_same_bad_stems",
                "diffusion_generative_repair": "cut_for_now_faithfulness_risk",
                "repeat_teacher_mask_adapter": "cut_as_primary_previous_holdout_failures",
            },
        },
    )
    matrix = _write_json(
        root / "runs" / "r006" / "manifest.json",
        {
            "schema": "aris_refinement_matrix_manifest.v1",
            "status": "blocked",
            "accepted_count": 0,
            "rejected_count": 1,
            "delivery_allowed": False,
            "delivery_status": "blocked",
            "presentation_status": "blocked_from_listen_pack",
            "target_stems": ["vocal", "minus"],
        },
    )
    return root, funnel, matrix


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    root, funnel, matrix = _rag_inputs(tmp_path)

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        build_aris_top_route_work_order(
            artifact_rag_root=root,
            funnel_json=funnel,
            matrix_manifest_json=matrix,
            output_dir=tmp_path / "outside",
        )


def test_materializes_top_routes_as_locked_non_delivery_work_order(tmp_path: Path) -> None:
    root, funnel, matrix = _rag_inputs(tmp_path)

    manifest = build_aris_top_route_work_order(
        artifact_rag_root=root,
        funnel_json=funnel,
        matrix_manifest_json=matrix,
        output_dir=root / "work" / "r001",
        command_args=["--artifact-rag-root", str(root)],
    )

    assert manifest["status"] == "ready_for_gpu_candidate_generation"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["selected_route_ids"] == [
        "holdout_error_map_segment_router",
        "cross_family_pair_consensus_selector",
    ]

    work_order = _read_json(Path(manifest["work_order_json"]))
    assert work_order["schema"] == "aris_top_route_work_order.v1"
    assert work_order["source_matrix"]["status"] == "blocked"
    assert work_order["source_matrix"]["accepted_count"] == 0
    assert work_order["source_matrix"]["rejected_count"] == 1
    assert work_order["execution_contract"]["docker_only"] is True
    assert work_order["execution_contract"]["gpu_candidate_generation_required"] is True
    assert work_order["execution_contract"]["manual_listening_required"] is False
    assert work_order["execution_contract"]["rejected_artifacts_not_deliverable"] is True
    assert work_order["execution_contract"]["qwen_controller_forbidden"] is True
    assert work_order["execution_contract"]["diffusion_repair_forbidden"] is True
    assert work_order["execution_contract"]["repeat_teacher_mask_forbidden"] is True
    assert work_order["delivery_allowed"] is False
    assert all(item["delivery_allowed"] is False for item in work_order["work_items"])
    assert work_order["work_items"][0]["route_id"] == "holdout_error_map_segment_router"
    assert work_order["work_items"][0]["candidate_matrix"]["stems"] == ["vocal", "minus"]
    assert work_order["work_items"][0]["output_root"] == "audio/t001"
    assert "qwen_controller" not in json.dumps(work_order["work_items"], sort_keys=True)
    assert "diffusion_generative_repair" not in json.dumps(work_order["work_items"], sort_keys=True)

    progress = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert [row["event"] for row in progress] == ["run_start", "routes_locked", "run_finish"]
    assert all(row["gpu_inference_skipped"] is True for row in progress)
    assert all(row["expected_acceleration_mode"] == "external_only_cpu_manifest_build" for row in progress)
