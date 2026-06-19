from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from tools.build_aris_refinement_candidate_pool import build_aris_refinement_candidate_pool
from tools.materialize_aris_rag_wave_metadata import materialize_aris_rag_wave_metadata
from tools.run_aris_refinement_matrix import run_aris_refinement_matrix


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_rag_pack(rag_root: Path) -> Path:
    pack = rag_root / "la001"
    pack.mkdir(parents=True, exist_ok=True)
    _write_json(
        rag_root / "current.json",
        {
            "rag_id": "la001",
            "idx": "la001/idx.json",
            "status": "thin_rag_binary_pruned",
            "current_retention_mode": "thin_rag_only",
            "binary_audio_model_bytes_available": False,
        },
    )
    _write_json(
        pack / "idx.json",
        {
            "rag_id": "la001",
            "status": "thin_rag_binary_pruned",
            "current_retention_mode": "thin_rag_only",
            "binary_audio_model_bytes_available": False,
        },
    )
    (pack / "roots.jsonl").write_text(
        json.dumps({"root_alias": "r001", "root": "/app/lab_artifacts", "files": 4}) + "\n",
        encoding="utf-8",
    )
    (pack / "chunks.jsonl").write_text("", encoding="utf-8")

    run_dir = "gt90_cycle_001/test_wave"
    manifest_member = f"r001/{run_dir}/registered_candidate_family_wave_manifest.json"
    jobs_member = f"r001/{run_dir}/registered_candidate_family_wave_jobs.jsonl"
    vocal_exec_member = "r001/gt90_cycle_001/src_v/v001/registry_candidate_execution_manifest.json"
    minus_exec_member = "r001/gt90_cycle_001/src_m/m001/registry_candidate_execution_manifest.json"
    old_jobs_path = f"/app/lab_artifacts/generic/{run_dir}/registered_candidate_family_wave_jobs.jsonl"
    old_vocal_exec = "/app/lab_artifacts/generic/gt90_cycle_001/src_v/v001/registry_candidate_execution_manifest.json"
    old_minus_exec = "/app/lab_artifacts/generic/gt90_cycle_001/src_m/m001/registry_candidate_execution_manifest.json"
    manifest_payload = {
        "schema": "registered_candidate_family_wave_manifest.v1",
        "manifest_type": "registered_candidate_family_wave",
        "status": "passed",
        "job_count": 2,
        "jobs_jsonl": old_jobs_path,
        "delivery_allowed": False,
        "release_ready": False,
        "presentation_status": "registered_candidate_family_wave_not_for_listen_pack",
        "release_status": "candidate_generation_only_requires_training_eval",
    }
    jobs = [
        {
            "schema": "registered_candidate_family_wave_job.v1",
            "id": "raw-vocal-teacher-id",
            "track_id": "track-a",
            "stem_type": "vocal",
            "source_candidate_family": "scnet",
            "source_candidate_model_id": "scnet_xl_ihf_musdb18",
            "execution_manifest_path": old_vocal_exec,
            "status": "completed",
            "candidate_status": "passed",
            "reference_gate": "passed",
            "no_reference_gate": "not_required",
            "pair_consistency_gate": "passed",
            "delivery_allowed": False,
            "release_ready": False,
        },
        {
            "schema": "registered_candidate_family_wave_job.v1",
            "id": "raw-minus-teacher-id",
            "track_id": "track-a",
            "stem_type": "minus",
            "source_candidate_family": "bs_roformer",
            "source_candidate_model_id": "bs_roformer_viperx_1297",
            "execution_manifest_path": old_minus_exec,
            "status": "completed",
            "candidate_status": "passed",
            "reference_gate": "passed",
            "no_reference_gate": "not_required",
            "pair_consistency_gate": "passed",
            "delivery_allowed": False,
            "release_ready": False,
        },
    ]
    vocal_exec_payload = {
        "schema": "registered_model_candidate_execution_manifest.v1",
        "status": "completed",
        "model_id": "scnet_xl_ihf_musdb18",
        "candidate_manifest": {
            "path": "/app/lab_artifacts/generic/gt90_cycle_001/src_v/v001/stem_manifest.json",
            "target_stem": "vocal",
            "target_audio_path": "/app/lab_artifacts/generic/gt90_cycle_001/src_v/v001/vocals.flac",
        },
        "delivery_allowed": False,
        "release_ready": False,
    }
    minus_exec_payload = {
        "schema": "registered_model_candidate_execution_manifest.v1",
        "status": "completed",
        "model_id": "bs_roformer_viperx_1297",
        "candidate_manifest": {
            "path": "/app/lab_artifacts/generic/gt90_cycle_001/src_m/m001/stem_manifest.json",
            "target_stem": "minus",
            "target_audio_path": "/app/lab_artifacts/generic/gt90_cycle_001/src_m/m001/minus.flac",
        },
        "delivery_allowed": False,
        "release_ready": False,
    }
    manifest_bytes = _json_bytes(manifest_payload)
    jobs_bytes = _jsonl_bytes(jobs)
    with zipfile.ZipFile(pack / "text_refs.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(manifest_member, manifest_bytes)
        archive.writestr(jobs_member, jobs_bytes)
        archive.writestr(vocal_exec_member, _json_bytes(vocal_exec_payload))
        archive.writestr(minus_exec_member, _json_bytes(minus_exec_payload))
    manifest_refs = [
        {
            "root_alias": "r001",
            "relative_path": f"{run_dir}/registered_candidate_family_wave_manifest.json",
            "zip_member": manifest_member,
            "bytes": len(manifest_bytes),
            "sha256": _sha256(manifest_bytes),
            "status_fields": {
                "schema": "registered_candidate_family_wave_manifest.v1",
                "manifest_type": "registered_candidate_family_wave",
                "delivery_allowed": False,
                "release_ready": False,
                "presentation_status": "registered_candidate_family_wave_not_for_listen_pack",
            },
        }
    ]
    (pack / "manifest_refs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_refs),
        encoding="utf-8",
    )
    return pack


def test_materializes_wave_metadata_from_text_refs_without_audio_delivery(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    _build_rag_pack(rag_root)

    manifest = materialize_aris_rag_wave_metadata(
        artifact_rag_root=rag_root,
        output_dir=rag_root / "mat" / "r001",
        include_substrings=["test_wave"],
    )

    assert manifest["status"] == "metadata_only_blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["binary_audio_available"] is False
    assert manifest["materialized_manifest_count"] == 1
    assert manifest["materialized_job_count"] == 2
    wave_path = Path(manifest["wave_manifest_jsons"][0])
    wave_path.resolve().relative_to(rag_root.resolve())
    wave = json.loads(wave_path.read_text(encoding="utf-8"))
    jobs_path = Path(wave["jobs_jsonl"])
    jobs_path.resolve().relative_to(rag_root.resolve())
    assert wave["binary_audio_available"] is False
    jobs = _read_jsonl(jobs_path)
    assert {job["stem_type"] for job in jobs} == {"vocal", "minus"}
    assert all(job["binary_audio_available"] is False for job in jobs)
    assert all(job["audio_bytes_available"] is False for job in jobs)
    assert all(job["metadata_only"] is True for job in jobs)
    for job in jobs:
        exec_path = Path(job["execution_manifest_path"])
        exec_path.resolve().relative_to(rag_root.resolve())
        exec_manifest = json.loads(exec_path.read_text(encoding="utf-8"))
        assert exec_manifest["binary_audio_available"] is False
        assert exec_manifest["metadata_only"] is True


def test_materialized_metadata_only_pool_is_blocked_by_matrix_runner(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    _build_rag_pack(rag_root)
    materialized = materialize_aris_rag_wave_metadata(
        artifact_rag_root=rag_root,
        output_dir=rag_root / "mat" / "r001",
        include_substrings=["test_wave"],
    )
    pool_manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=rag_root,
        wave_manifest_jsons=[Path(path) for path in materialized["wave_manifest_jsons"]],
        output_dir=rag_root / "runs" / "r001" / "pool",
    )
    plan = _write_json(
        rag_root / "analysis" / "top1_top2_plan.json",
        {
            "schema": "aris_top1_top2_rag_plan.v1",
            "route_id": "failure_conditioned_refinement",
            "candidate_families": ["scnet", "bs_roformer"],
        },
    )

    matrix = run_aris_refinement_matrix(
        artifact_rag_root=rag_root,
        planning_json=plan,
        candidate_pool_jsonl=Path(pool_manifest["candidate_pool_jsonl"]),
        output_dir=rag_root / "runs" / "r002",
    )

    assert matrix["accepted_count"] == 0
    assert matrix["delivery_allowed"] is False
    assert _read_jsonl(Path(matrix["accepted_jsonl"])) == []
    rejected = _read_jsonl(Path(matrix["rejected_jsonl"]))
    assert "binary_audio_unavailable" in rejected[0]["failed_checks"]
