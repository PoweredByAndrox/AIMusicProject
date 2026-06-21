from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools.query_artifact_rag import query_artifact_rag


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_base_rag(root: Path, *, declared_root_count: int = 1) -> Path:
    rag_root = root / "artifact_rag"
    pack_dir = rag_root / "ar01"
    write_json(
        rag_root / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar01",
            "idx": "ar01/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )
    write_json(
        pack_dir / "idx.json",
        {
            "schema": "artifact_rag_pack.v1",
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar01",
            "summary": {
                "root_count": declared_root_count,
                "file_count": 1,
                "manifest_ref_count": 1,
                "chunk_count": 1,
                "source_bytes": 20,
                "source_gib": 0.0,
                "text_snapshot_files": 1,
                "text_snapshot_source_bytes": 20,
            },
            "outputs": {
                "idx": "idx.json",
                "summary": "sum.md",
                "roots": "roots.jsonl",
                "files": "files.jsonl",
                "manifest_refs": "manifest_refs.jsonl",
                "chunks": "chunks.jsonl",
                "text_refs_zip": "text_refs.zip",
                "cleanup": "cleanup.json",
                "path_map": "path_map.jsonl",
                "progress": "progress.jsonl",
            },
            "retention_contract": {"llm_retrieval_complete_for_textual_evidence": True, "delivery_allowed": False, "release_ready": False},
            "query_hint": "Use artifact_rag/current.json -> idx.json, then chunks.jsonl/manifest_refs.jsonl.",
        },
    )
    write_jsonl(pack_dir / "roots.jsonl", [{"root": "base", "root_alias": "r001", "requested": "base", "original_path": str(root / "base"), "bytes": 20, "gib": 0.0, "files": 1, "text_snapshot_files": 1}])
    write_jsonl(pack_dir / "files.jsonl", [{"root": "base", "root_alias": "r001", "relative_path": "manifest.json", "bytes": 10, "extension": ".json", "text_snapshot": True, "text_sha256": "base-sha", "text_zip_member": "r001/manifest.json"}])
    write_jsonl(
        pack_dir / "manifest_refs.jsonl",
        [
            {
                "root": "base",
                "root_alias": "r001",
                "relative_path": "manifest.json",
                "sha256": "base-sha",
                "zip_member": "r001/manifest.json",
                "status_fields": {"delivery_status": "accepted", "delivery_allowed": True, "presentation_status": "accepted_for_listen_pack"},
                "excerpt": "delivery_status accepted presentation_status accepted_for_listen_pack",
            }
        ],
    )
    write_jsonl(pack_dir / "chunks.jsonl", [{"root": "base", "root_alias": "r001", "relative_path": "manifest.json", "chunk_id": "r001_000001", "chunk_index": 1, "status_fields": {"delivery_status": "accepted"}, "text": "delivery_status accepted"}])
    write_jsonl(pack_dir / "path_map.jsonl", [])
    write_json(pack_dir / "cleanup.json", {"schema": "artifact_rag_cleanup.v1", "rag_id": "ar01", "cleanup_summary": {"cleanup_mode": "none"}})
    (pack_dir / "sum.md").write_text("# base\n", encoding="utf-8")
    (pack_dir / "progress.jsonl").write_text("", encoding="utf-8")
    with zipfile.ZipFile(pack_dir / "text_refs.zip", "w") as archive:
        archive.writestr("r001/manifest.json", "{\"delivery_status\":\"accepted\"}")
    return rag_root


def make_metadata_root(rag_root: Path) -> Path:
    root = rag_root / "aqp" / "p001"
    write_json(
        root / "manifest.json",
        {
            "schema": "audio_quality_promotion_candidate_discovery.v1",
            "discovery_status": "blocked",
            "candidate_count": 0,
            "blocked_ref_count": 4,
            "promotion_allowed": False,
            "runtime_authority_enabled": False,
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    write_jsonl(
        root / "blocked.jsonl",
        [
            {
                "schema": "audio_quality_promotion_candidate_blocked_row.v1",
                "blocked_reasons": ["promotion_candidate_not_found"],
                "ready_for_promotion_decision": False,
                "delivery_allowed": False,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    (root / "summary.md").write_text("Status: blocked\n", encoding="utf-8")
    return root


def test_metadata_overlay_preserves_current_rag_and_adds_queryable_evidence(tmp_path: Path) -> None:
    from tools.build_artifact_rag_metadata_overlay import build_artifact_rag_metadata_overlay

    rag_root = make_base_rag(tmp_path)
    metadata_root = make_metadata_root(rag_root)

    manifest = build_artifact_rag_metadata_overlay(
        rag_current_json=rag_root / "current.json",
        output_dir=rag_root / "ar02",
        include_roots=[metadata_root],
    )

    current = read_json(rag_root / "current.json")
    roots = read_jsonl(rag_root / "ar02" / "roots.jsonl")
    manifest_refs = read_jsonl(rag_root / "ar02" / "manifest_refs.jsonl")
    chunks = read_jsonl(rag_root / "ar02" / "chunks.jsonl")

    assert current["idx"] == "ar02/idx.json"
    assert manifest["base_rag_id"] == "ar01"
    assert manifest["summary"]["base_root_count"] == 1
    assert manifest["summary"]["root_count"] == 2
    assert manifest["summary"]["file_count"] > manifest["summary"]["base_file_count"]
    assert manifest["summary"]["manifest_ref_count"] > manifest["summary"]["base_manifest_ref_count"]
    assert manifest["summary"]["chunk_count"] > manifest["summary"]["base_chunk_count"]
    assert manifest["retention_contract"]["delivery_allowed"] is False
    assert manifest["retention_contract"]["release_ready"] is False
    assert manifest["overlay_safety"]["base_coverage_preserved"] is True
    assert manifest["overlay_safety"]["metadata_roots_delivery_allowed"] is False
    assert {Path(path).name for path in manifest["outputs"].values()} >= {"idx.json", "sum.md", "roots.jsonl", "files.jsonl", "manifest_refs.jsonl", "chunks.jsonl", "text_refs.zip", "cleanup.json", "path_map.jsonl", "progress.jsonl"}
    assert roots[-1]["root"] == "aqp/p001"
    assert roots[-1]["root_alias"] == "r002"
    overlay_chunk_ids = [row["chunk_id"] for row in chunks if row["root_alias"] == "r002"]
    assert len(overlay_chunk_ids) == len(set(overlay_chunk_ids))
    assert all(len(part) <= 32 for part in ["ar02", "idx.json", "sum.md", "r002"])
    assert any(row["relative_path"] == "manifest.json" and row["status_fields"].get("schema") == "audio_quality_promotion_candidate_discovery.v1" for row in manifest_refs)

    accepted = query_artifact_rag(rag_root=rag_root, query="accepted_for_listen_pack", limit=5)
    discovery = query_artifact_rag(rag_root=rag_root, query="audio_quality_promotion_candidate_discovery", limit=5)
    assert accepted["result_count"] >= 1
    assert discovery["result_count"] >= 1
    assert discovery["matches"][0]["status_fields"]["delivery_allowed"] is False


def test_metadata_overlay_refuses_inconsistent_base_counts(tmp_path: Path) -> None:
    from tools.build_artifact_rag_metadata_overlay import build_artifact_rag_metadata_overlay

    rag_root = make_base_rag(tmp_path, declared_root_count=2)
    metadata_root = make_metadata_root(rag_root)

    try:
        build_artifact_rag_metadata_overlay(
            rag_current_json=rag_root / "current.json",
            output_dir=rag_root / "ar02",
            include_roots=[metadata_root],
        )
    except ValueError as exc:
        assert "base_root_count_mismatch" in str(exc)
    else:
        raise AssertionError("expected base_root_count_mismatch")


def test_metadata_overlay_can_use_explicit_base_rag_id_without_stacking_overlays(tmp_path: Path) -> None:
    from tools.build_artifact_rag_metadata_overlay import build_artifact_rag_metadata_overlay

    rag_root = make_base_rag(tmp_path)
    metadata_root = make_metadata_root(rag_root)
    build_artifact_rag_metadata_overlay(
        rag_current_json=rag_root / "current.json",
        output_dir=rag_root / "ar02",
        include_roots=[metadata_root],
    )

    manifest = build_artifact_rag_metadata_overlay(
        rag_current_json=rag_root / "current.json",
        output_dir=rag_root / "ar03",
        include_roots=[metadata_root],
        base_rag_id="ar01",
    )

    current = read_json(rag_root / "current.json")
    roots = read_jsonl(rag_root / "ar03" / "roots.jsonl")

    assert current["idx"] == "ar03/idx.json"
    assert manifest["base_rag_id"] == "ar01"
    assert manifest["summary"]["base_root_count"] == 1
    assert manifest["summary"]["overlay_root_count"] == 1
    assert manifest["summary"]["root_count"] == 2
    assert [row["root"] for row in roots].count("aqp/p001") == 1
