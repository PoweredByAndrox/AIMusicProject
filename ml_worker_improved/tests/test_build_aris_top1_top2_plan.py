from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_aris_top1_top2_plan import build_aris_top1_top2_plan, parse_args


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


def _rag_root(tmp_path: Path) -> Path:
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
            "summary": "la001/sum.md",
        },
    )
    _write_jsonl(
        root / "la001" / "manifest_refs.jsonl",
        [
            {
                "relative_path": "run/a.json",
                "excerpt": "failed_checks vocal_bleed vocal_bleed minus_detail_gate_blocked",
                "status_fields": {"stem": "vocal", "failed_checks": ["vocal_bleed", "holdout_failed"]},
            },
            {
                "relative_path": "run/b.json",
                "excerpt": "transient_preservation_below_detail_gate severe_music_damage",
                "status_fields": {"stem": "minus", "blockers": ["minus_detail_gate_blocked"]},
            },
        ],
    )
    return root


def test_builds_runner_compatible_plan_from_rag_manifest_refs(tmp_path: Path) -> None:
    root = _rag_root(tmp_path)

    manifest = build_aris_top1_top2_plan(
        artifact_rag_root=root,
        output_dir=root / "analysis" / "r001",
        command_args=["--artifact-rag-root", str(root)],
    )

    plan = json.loads(Path(manifest["plan_json"]).read_text(encoding="utf-8"))
    assert manifest["delivery_allowed"] is False
    assert manifest["binary_audio_model_bytes_available"] is False
    assert plan["schema"] == "aris_top1_top2_rag_plan.v1"
    assert plan["route_id"] == "failure_conditioned_refinement"
    assert plan["candidate_families"] == ["demucs", "htdemucs", "scnet", "bs_roformer", "melband", "mdx23c", "bs_mamba2"]
    assert "qwen_controller" not in plan["candidate_families"]
    assert plan["explicitly_cut_directions"]["qwen_controller"] == "cut_for_now_more_control_logic_same_bad_stems"
    assert plan["priority_failures"][0]["failure"] == "vocal_bleed"
    assert plan["stem_priorities"]["vocal"][0]["failure"] == "vocal_bleed"
    assert plan["top1_fixed_inference_refinement_matrix"]["vocal"]["grammar"][-1] == "exact_reference_or_no_reference_gate"

    progress_rows = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == ["run_start", "refs_scanned", "run_finish"]
    assert all(row["gpu_inference_skipped"] is True for row in progress_rows)
    assert all(row["expected_acceleration_mode"] == "external_only_cpu_manifest_build" for row in progress_rows)


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    root = _rag_root(tmp_path)

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        build_aris_top1_top2_plan(
            artifact_rag_root=root,
            output_dir=tmp_path / "outside",
        )


def test_rejects_output_dir_with_unsafe_segments_inside_artifact_rag(tmp_path: Path) -> None:
    root = _rag_root(tmp_path)

    with pytest.raises(ValueError, match="output_dir must use short Windows-safe relative path inside artifact_rag_root"):
        build_aris_top1_top2_plan(
            artifact_rag_root=root,
            output_dir=root / "analysis" / "Track Name With Spaces And A Very Long Raw Title",
        )


def test_parse_args_defaults_to_current_json(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--artifact-rag-root",
            str(tmp_path / "artifact_rag"),
            "--output-dir",
            str(tmp_path / "artifact_rag" / "analysis" / "r001"),
        ]
    )

    assert args.current_json is None
