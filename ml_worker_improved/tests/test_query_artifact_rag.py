from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.query_artifact_rag import query_artifact_rag


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_rag_pack(root: Path) -> Path:
    rag_root = root / "artifact_rag"
    pack_dir = rag_root / "la001"
    write_json(
        rag_root / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "la001",
            "idx": "la001/idx.json",
            "status": "thin_rag_binary_pruned",
            "current_retention_mode": "thin_rag_only",
            "binary_audio_model_bytes_available": False,
            "query_hint": "Use chunks and manifest refs.",
        },
    )
    write_json(
        pack_dir / "idx.json",
        {
            "schema": "artifact_rag_index.v2",
            "rag_id": "la001",
            "status": "thin_rag_binary_pruned",
            "summary": {"root_count": 1, "file_count": 2},
            "retention_contract": {
                "llm_retrieval_complete_for_textual_evidence": True,
                "binary_audio_model_bytes_available": False,
            },
        },
    )
    write_jsonl(
        pack_dir / "roots.jsonl",
        [{"root": "old_run", "root_alias": "r001", "gib": 42.0, "files": 2, "text_snapshot_files": 2}],
    )
    write_jsonl(
        pack_dir / "manifest_refs.jsonl",
        [
            {
                "root": "old_run",
                "root_alias": "r001",
                "relative_path": "manifest.json",
                "status_fields": {
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                },
                "excerpt": "delivery_status accepted presentation_status accepted_for_listen_pack vocal",
            }
        ],
    )
    write_jsonl(
        pack_dir / "chunks.jsonl",
        [
            {
                "root": "old_run",
                "root_alias": "r001",
                "relative_path": "progress.jsonl",
                "status_fields": {"dominant_failure_reason": "vocal_bleed"},
                "text": "route rejected because vocal_bleed remained in minus candidate",
            }
        ],
    )
    return rag_root


def test_query_artifact_rag_returns_llm_context_from_current_pack(tmp_path: Path) -> None:
    rag_root = make_rag_pack(tmp_path)

    result = query_artifact_rag(rag_root=rag_root, query="accepted vocal", limit=5)

    assert result["schema"] == "artifact_rag_query_result.v1"
    assert result["current"]["rag_id"] == "la001"
    assert result["current"]["current_retention_mode"] == "thin_rag_only"
    assert result["current"]["binary_audio_model_bytes_available"] is False
    assert result["retention_contract"]["llm_retrieval_complete_for_textual_evidence"] is True
    assert result["matches"][0]["source"] == "manifest_refs.jsonl"
    assert result["matches"][0]["status_fields"]["delivery_status"] == "accepted"


def test_query_artifact_rag_finds_failure_evidence_in_chunks(tmp_path: Path) -> None:
    rag_root = make_rag_pack(tmp_path)

    result = query_artifact_rag(rag_root=rag_root, query="vocal_bleed minus", limit=5)

    assert result["matches"][0]["source"] == "chunks.jsonl"
    assert result["matches"][0]["status_fields"]["dominant_failure_reason"] == "vocal_bleed"
    assert "minus candidate" in result["matches"][0]["excerpt"]


def test_query_artifact_rag_rejects_empty_query(tmp_path: Path) -> None:
    rag_root = make_rag_pack(tmp_path)

    with pytest.raises(ValueError, match="searchable term"):
        query_artifact_rag(rag_root=rag_root, query=" ")


def test_query_artifact_rag_rejects_current_idx_escape(tmp_path: Path) -> None:
    rag_root = make_rag_pack(tmp_path)
    write_json(
        rag_root / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "la001",
            "idx": "../outside/idx.json",
        },
    )

    with pytest.raises(ValueError, match="current.idx escapes artifact rag root"):
        query_artifact_rag(rag_root=rag_root, query="accepted")


def test_query_artifact_rag_rejects_rag_id_escape(tmp_path: Path) -> None:
    rag_root = make_rag_pack(tmp_path)

    with pytest.raises(ValueError, match="rag_id escapes artifact rag root"):
        query_artifact_rag(rag_root=rag_root, rag_id="../outside", query="accepted")


def test_query_artifact_rag_rejects_incomplete_pack(tmp_path: Path) -> None:
    rag_root = make_rag_pack(tmp_path)
    (rag_root / "la001" / "chunks.jsonl").unlink()

    with pytest.raises(ValueError, match="missing required files: chunks.jsonl"):
        query_artifact_rag(rag_root=rag_root, query="accepted")
