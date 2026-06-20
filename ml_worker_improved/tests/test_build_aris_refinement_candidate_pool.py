from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_aris_refinement_candidate_pool import build_aris_refinement_candidate_pool, parse_args
from tools.run_aris_refinement_matrix import run_aris_refinement_matrix


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


def _wave_manifest(root_path: Path, jobs: list[dict]) -> Path:
    jobs_jsonl = _write_jsonl(root_path / "wave" / "registered_candidate_family_wave_jobs.jsonl", jobs)
    return _write_json(
        root_path / "wave" / "registered_candidate_family_wave_manifest.json",
        {
            "schema": "registered_candidate_family_wave_manifest.v1",
            "manifest_type": "registered_candidate_family_wave",
            "status": "passed",
            "stem": "vocal",
            "jobs_jsonl": str(jobs_jsonl),
            "delivery_allowed": False,
            "release_ready": False,
        },
    )


def _execution_manifest(path: Path, *, status: str = "completed", target_audio_path: str = "audio/t001/vocal.flac") -> Path:
    return _write_json(
        path,
        {
            "schema": "registered_model_candidate_execution_manifest.v1",
            "status": status,
            "candidate_manifest": {
                "path": str(path.parent / "stem_manifest.json"),
                "target_stem": "vocal",
                "target_audio_path": target_audio_path,
            },
            "delivery_allowed": False,
            "release_ready": False,
        },
    )


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    wave_manifest = _wave_manifest(rag_root, [])

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        build_aris_refinement_candidate_pool(
            artifact_rag_root=rag_root,
            wave_manifest_jsons=[wave_manifest],
            output_dir=tmp_path / "loose",
        )


def test_rejects_output_dir_with_unsafe_segments_inside_artifact_rag(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    wave_manifest = _wave_manifest(rag_root, [])

    with pytest.raises(ValueError, match="output_dir must use short Windows-safe relative path inside artifact_rag_root"):
        build_aris_refinement_candidate_pool(
            artifact_rag_root=rag_root,
            wave_manifest_jsons=[wave_manifest],
            output_dir=rag_root / "runs" / "Track Name With Spaces And A Very Long Raw Title",
        )


def test_rejects_wave_manifest_outside_artifact_rag(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    wave_manifest = _wave_manifest(tmp_path / "outside", [])

    with pytest.raises(ValueError, match="wave_manifest_json must be inside artifact_rag_root"):
        build_aris_refinement_candidate_pool(
            artifact_rag_root=rag_root,
            wave_manifest_jsons=[wave_manifest],
            output_dir=rag_root / "runs" / "r001" / "pool",
        )


def test_rejects_jobs_jsonl_outside_artifact_rag(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    jobs_jsonl = _write_jsonl(tmp_path / "outside" / "registered_candidate_family_wave_jobs.jsonl", [])
    wave_manifest = _write_json(
        rag_root / "wave" / "registered_candidate_family_wave_manifest.json",
        {
            "schema": "registered_candidate_family_wave_manifest.v1",
            "manifest_type": "registered_candidate_family_wave",
            "status": "passed",
            "stem": "vocal",
            "jobs_jsonl": str(jobs_jsonl),
            "delivery_allowed": False,
            "release_ready": False,
        },
    )

    with pytest.raises(ValueError, match="jobs_jsonl must be inside artifact_rag_root"):
        build_aris_refinement_candidate_pool(
            artifact_rag_root=rag_root,
            wave_manifest_jsons=[wave_manifest],
            output_dir=rag_root / "runs" / "r001" / "pool",
        )


def test_rejects_execution_manifest_outside_artifact_rag(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    exec_manifest = _execution_manifest(
        tmp_path / "outside" / "registry_candidate_execution_manifest.json",
    )
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "id": "example-a",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "demucs",
                "source_candidate_model_id": "htdemucs",
                "execution_manifest_path": str(exec_manifest),
                "status": "completed",
            }
        ],
    )

    with pytest.raises(ValueError, match="execution_manifest_path must be inside artifact_rag_root"):
        build_aris_refinement_candidate_pool(
            artifact_rag_root=rag_root,
            wave_manifest_jsons=[wave_manifest],
            output_dir=rag_root / "runs" / "r001" / "pool",
        )


def test_converts_registered_wave_jobs_without_granting_delivery(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    exec_manifest = _execution_manifest(rag_root / "wave" / "scnet" / "a" / "registry_candidate_execution_manifest.json")
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "schema": "registered_candidate_family_wave_job.v1",
                "id": "example-a",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "scnet",
                "source_candidate_model_id": "scnet_xl_ihf_musdb18",
                "execution_manifest_path": str(exec_manifest),
                "status": "completed",
                "fallback_used": False,
                "delivery_allowed": False,
                "release_ready": False,
            }
        ],
    )

    manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=rag_root,
        wave_manifest_jsons=[wave_manifest],
        output_dir=rag_root / "runs" / "r001" / "pool",
    )

    assert manifest["candidate_count"] == 1
    assert manifest["skipped_count"] == 0
    assert manifest["delivery_allowed"] is False
    rows = _read_jsonl(Path(manifest["candidate_pool_jsonl"]))
    assert rows[0]["schema"] == "aris_refinement_candidate_row.v1"
    assert rows[0]["track_alias"] == "t001"
    assert rows[0]["track_id"] == "track-a"
    assert rows[0]["stem"] == "vocal"
    assert rows[0]["family"] == "scnet"
    assert rows[0]["candidate_status"] == "generated"
    assert rows[0]["short_audio_path"] == "audio/t001/vocal.flac"
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["release_ready"] is False
    assert rows[0]["fallback_used"] is False
    assert "reference_gate" not in rows[0]
    assert "no_reference_gate" not in rows[0]


def test_preserves_safe_executor_track_alias_for_corpus_batches(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    target_audio = rag_root / "wave" / "scnet" / "b" / "vocal.flac"
    exec_manifest = _execution_manifest(
        rag_root / "wave" / "scnet" / "b" / "registry_candidate_execution_manifest.json",
        target_audio_path=str(target_audio),
    )
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "schema": "registered_candidate_family_wave_job.v1",
                "id": "example-b",
                "track_id": "track-b",
                "track_alias": "t002",
                "stem_type": "vocal",
                "source_candidate_family": "scnet",
                "source_candidate_model_id": "scnet_xl_ihf_musdb18",
                "execution_manifest_path": str(exec_manifest),
                "status": "completed",
                "fallback_used": False,
                "delivery_allowed": False,
                "release_ready": False,
            }
        ],
    )

    manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=rag_root,
        wave_manifest_jsons=[wave_manifest],
        output_dir=rag_root / "runs" / "r001" / "pool",
    )

    rows = _read_jsonl(Path(manifest["candidate_pool_jsonl"]))
    assert rows[0]["track_alias"] == "t002"
    assert rows[0]["short_audio_path"] == "audio/t002/vocal.flac"


def test_passes_through_existing_objective_gate_evidence_without_inventing_it(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    exec_manifest = _execution_manifest(rag_root / "wave" / "bs" / "a" / "registry_candidate_execution_manifest.json")
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "id": "example-a",
                "track_id": "track-a",
                "stem_type": "minus",
                "source_candidate_family": "bs_roformer",
                "source_candidate_model_id": "bs_roformer_viperx_1297",
                "execution_manifest_path": str(exec_manifest),
                "status": "completed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "candidate_status": "passed",
            }
        ],
    )

    manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=rag_root,
        wave_manifest_jsons=[wave_manifest],
        output_dir=rag_root / "runs" / "r001" / "pool",
    )

    row = _read_jsonl(Path(manifest["candidate_pool_jsonl"]))[0]
    assert row["candidate_status"] == "passed"
    assert row["reference_gate"] == "passed"
    assert row["no_reference_gate"] == "not_required"
    assert row["pair_consistency_gate"] == "passed"


def test_preserves_fallback_provenance_without_granting_delivery(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    exec_manifest = _execution_manifest(rag_root / "wave" / "scnet" / "a" / "registry_candidate_execution_manifest.json")
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "id": "example-a",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "scnet",
                "source_candidate_model_id": "scnet_xl_ihf_musdb18",
                "execution_manifest_path": str(exec_manifest),
                "status": "completed",
                "fallback_used": True,
                "fallback_as_success": True,
            }
        ],
    )

    manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=rag_root,
        wave_manifest_jsons=[wave_manifest],
        output_dir=rag_root / "runs" / "r001" / "pool",
    )

    row = _read_jsonl(Path(manifest["candidate_pool_jsonl"]))[0]
    assert row["fallback_used"] is True
    assert row["fallback_as_success"] is True
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False


def test_builder_to_runner_blocks_fallback_provenance_from_acceptance(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    plan_json = _write_json(
        rag_root / "analysis" / "top1_top2_plan.json",
        {
            "schema": "aris_top1_top2_rag_plan.v1",
            "route_id": "failure_conditioned_refinement",
            "candidate_families": ["scnet", "bs_roformer"],
        },
    )
    vocal_exec = _execution_manifest(
        rag_root / "wave" / "scnet" / "v" / "registry_candidate_execution_manifest.json",
        target_audio_path="audio/t001/vocal.flac",
    )
    minus_exec = _execution_manifest(
        rag_root / "wave" / "bs" / "m" / "registry_candidate_execution_manifest.json",
        target_audio_path="audio/t001/minus.flac",
    )
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "id": "example-v",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "scnet",
                "source_candidate_model_id": "scnet_xl_ihf_musdb18",
                "execution_manifest_path": str(vocal_exec),
                "status": "completed",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "fallback_used": True,
            },
            {
                "id": "example-m",
                "track_id": "track-a",
                "stem_type": "minus",
                "source_candidate_family": "bs_roformer",
                "source_candidate_model_id": "bs_roformer_viperx_1297",
                "execution_manifest_path": str(minus_exec),
                "status": "completed",
                "candidate_status": "passed",
                "reference_gate": "passed",
                "no_reference_gate": "not_required",
                "pair_consistency_gate": "passed",
                "fallback_used": False,
            },
        ],
    )
    pool_manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=rag_root,
        wave_manifest_jsons=[wave_manifest],
        output_dir=rag_root / "runs" / "r001" / "pool",
    )

    matrix_manifest = run_aris_refinement_matrix(
        artifact_rag_root=rag_root,
        planning_json=plan_json,
        candidate_pool_jsonl=Path(pool_manifest["candidate_pool_jsonl"]),
        output_dir=rag_root / "runs" / "r002",
    )

    assert matrix_manifest["accepted_count"] == 0
    assert matrix_manifest["delivery_allowed"] is False
    assert _read_jsonl(Path(matrix_manifest["accepted_jsonl"])) == []
    rejected = _read_jsonl(Path(matrix_manifest["rejected_jsonl"]))
    assert "fallback_used_not_allowed" in rejected[0]["failed_checks"]


def test_records_missing_execution_manifest_as_skipped_not_accepted(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    missing = rag_root / "wave" / "missing" / "registry_candidate_execution_manifest.json"
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "id": "example-a",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "demucs",
                "source_candidate_model_id": "htdemucs",
                "execution_manifest_path": str(missing),
                "status": "completed",
            }
        ],
    )

    manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=rag_root,
        wave_manifest_jsons=[wave_manifest],
        output_dir=rag_root / "runs" / "r001" / "pool",
    )

    assert manifest["candidate_count"] == 0
    assert manifest["skipped_count"] == 1
    skipped = _read_jsonl(Path(manifest["skipped_jsonl"]))
    assert skipped[0]["skip_reason"] == "execution_manifest_missing"
    assert skipped[0]["delivery_allowed"] is False


def test_rejects_forbidden_family_and_unsafe_target_path(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    exec_manifest = _execution_manifest(
        rag_root / "wave" / "qwen" / "a" / "registry_candidate_execution_manifest.json",
        target_audio_path="audio/Track Name With Spaces And A Very Long Raw Title/vocal.flac",
    )
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "id": "example-a",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "qwen_controller",
                "source_candidate_model_id": "qwen",
                "execution_manifest_path": str(exec_manifest),
                "status": "completed",
            }
        ],
    )

    with pytest.raises(ValueError, match="forbidden primary direction"):
        build_aris_refinement_candidate_pool(
            artifact_rag_root=rag_root,
            wave_manifest_jsons=[wave_manifest],
            output_dir=rag_root / "runs" / "r001" / "pool",
        )


def test_rejects_unsafe_audio_relative_path_from_execution_manifest(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    exec_manifest = _execution_manifest(
        rag_root / "wave" / "demucs" / "a" / "registry_candidate_execution_manifest.json",
        target_audio_path="audio/Track Name With Spaces And A Very Long Raw Title/vocal.flac",
    )
    wave_manifest = _wave_manifest(
        rag_root,
        [
            {
                "id": "example-a",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "demucs",
                "source_candidate_model_id": "htdemucs",
                "execution_manifest_path": str(exec_manifest),
                "status": "completed",
            }
        ],
    )

    with pytest.raises(ValueError):
        build_aris_refinement_candidate_pool(
            artifact_rag_root=rag_root,
            wave_manifest_jsons=[wave_manifest],
            output_dir=rag_root / "runs" / "r001" / "pool",
        )


def test_parse_args_accepts_multiple_wave_manifests(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--artifact-rag-root",
            str(tmp_path / "artifact_rag"),
            "--wave-manifest-json",
            str(tmp_path / "wave-a.json"),
            "--wave-manifest-json",
            str(tmp_path / "wave-b.json"),
            "--output-dir",
            str(tmp_path / "artifact_rag" / "runs" / "r001" / "pool"),
        ]
    )

    assert [path.name for path in args.wave_manifest_json] == ["wave-a.json", "wave-b.json"]
