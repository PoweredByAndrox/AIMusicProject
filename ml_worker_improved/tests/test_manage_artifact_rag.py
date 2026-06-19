from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tools.manage_artifact_rag import build_artifact_rag_pack, prune_existing_vault_pack, validate_output_mount_contract


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_root(lab_root: Path, name: str = "old_run") -> Path:
    root = lab_root / name
    (root / "audio").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / "audio" / "vocal.flac").write_bytes(b"fake-audio")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "test_manifest.v1",
                "track_id": "track-a",
                "stem_type": "vocal",
                "candidate_id": "candidate-a",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "release_ready": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "logs" / "progress.jsonl").write_text(
        json.dumps({"event": "start", "delivery_allowed": False}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_build_artifact_rag_pack_writes_queryable_files_and_current_pointer(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    make_root(lab_root)
    output_dir = tmp_path / "artifact_rag" / "la003"

    manifest = build_artifact_rag_pack(
        lab_root=lab_root,
        output_dir=output_dir,
        include_roots=["old_run"],
    )

    assert manifest["summary"]["root_count"] == 1
    assert manifest["summary"]["file_count"] == 3
    assert manifest["summary"]["text_snapshot_files"] == 2
    assert (output_dir / "idx.json").exists()
    assert (output_dir / "sum.md").exists()
    assert (output_dir / "roots.jsonl").exists()
    assert (output_dir / "files.jsonl").exists()
    assert (output_dir / "manifest_refs.jsonl").exists()
    assert (output_dir / "chunks.jsonl").exists()
    assert (output_dir / "text_refs.zip").exists()
    assert (output_dir / "cleanup.json").exists()
    assert (output_dir / "path_map.jsonl").exists()
    assert (output_dir / "progress.jsonl").exists()
    assert read_json(tmp_path / "artifact_rag" / "current.json")["idx"] == "la003/idx.json"

    roots = read_jsonl(output_dir / "roots.jsonl")
    assert roots[0]["root_alias"] == "r001"
    assert roots[0]["root"] == "old_run"
    assert all(len(part) <= 32 for part in ["idx.json", "sum.md", "roots.jsonl", "files.jsonl", "r001"])

    refs = read_jsonl(output_dir / "manifest_refs.jsonl")
    manifest_ref = next(row for row in refs if row["relative_path"] == "manifest.json")
    assert manifest_ref["root_alias"] == "r001"
    assert manifest_ref["status_fields"]["delivery_status"] == "accepted"
    chunks = read_jsonl(output_dir / "chunks.jsonl")
    assert any("accepted_for_listen_pack" in row["text"] for row in chunks)
    with zipfile.ZipFile(output_dir / "text_refs.zip") as archive:
        assert archive.read("r001/manifest.json") == (lab_root / "old_run" / "manifest.json").read_bytes()


def test_metadata_only_delete_requires_explicit_allow_flag(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    make_root(lab_root)

    with pytest.raises(ValueError, match="allow-metadata-only-delete"):
        build_artifact_rag_pack(
            lab_root=lab_root,
            output_dir=tmp_path / "artifact_rag" / "la003",
            include_roots=["old_run"],
            cleanup_mode="metadata_only_delete",
            execute_cleanup=True,
        )


def test_metadata_only_delete_removes_covered_disposable_root(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    make_root(lab_root)
    output_dir = tmp_path / "artifact_rag" / "la003"

    manifest = build_artifact_rag_pack(
        lab_root=lab_root,
        output_dir=output_dir,
        include_roots=["old_run"],
        cleanup_mode="metadata_only_delete",
        execute_cleanup=True,
        allow_metadata_only_delete=True,
    )

    assert manifest["summary"]["deleted_roots"] == 1
    assert not (lab_root / "old_run").exists()
    cleanup = read_json(output_dir / "cleanup.json")
    assert cleanup["cleanup_summary"]["deleted_roots"] == 1
    with zipfile.ZipFile(output_dir / "text_refs.zip") as archive:
        assert "r001/manifest.json" in archive.namelist()


def test_thin_rag_delete_requires_explicit_allow_flag(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    make_root(lab_root)

    with pytest.raises(ValueError, match="allow-thin-rag-delete"):
        build_artifact_rag_pack(
            lab_root=lab_root,
            output_dir=tmp_path / "artifact_rag" / "la003",
            include_roots=["old_run"],
            cleanup_mode="thin_rag_delete",
            execute_cleanup=True,
        )


def test_thin_rag_delete_keeps_llm_evidence_and_removes_bulky_root(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    make_root(lab_root, "old_run")
    output_dir = tmp_path / "artifact_rag" / "la003"

    manifest = build_artifact_rag_pack(
        lab_root=lab_root,
        output_dir=output_dir,
        include_roots=["old_run"],
        cleanup_mode="thin_rag_delete",
        execute_cleanup=True,
        allow_thin_rag_delete=True,
    )

    assert manifest["summary"]["thin_deleted_roots"] == 1
    assert not (lab_root / "old_run").exists()
    assert manifest["retention_contract"]["llm_retrieval_complete_for_textual_evidence"] is True
    assert manifest["retention_contract"]["binary_audio_model_bytes_available"] is False
    cleanup = read_json(output_dir / "cleanup.json")
    assert cleanup["cleanup_summary"]["thin_deleted_roots"] == 1
    chunks = read_jsonl(output_dir / "chunks.jsonl")
    assert any(row["relative_path"] == "manifest.json" for row in chunks)
    with zipfile.ZipFile(output_dir / "text_refs.zip") as archive:
        assert archive.read("r001/manifest.json") == b'{"candidate_id": "candidate-a", "delivery_allowed": true, "delivery_status": "accepted", "presentation_status": "accepted_for_listen_pack", "release_ready": false, "schema": "test_manifest.v1", "stem_type": "vocal", "track_id": "track-a"}'


def test_move_to_vault_preserves_exact_bytes_and_removes_lab_root(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    source_root = make_root(lab_root, "golden_audio")
    original_audio = (source_root / "audio" / "vocal.flac").read_bytes()
    output_dir = tmp_path / "artifact_rag" / "la003"

    manifest = build_artifact_rag_pack(
        lab_root=lab_root,
        output_dir=output_dir,
        include_roots=["golden_audio"],
        cleanup_mode="move_to_vault",
        execute_cleanup=True,
    )

    assert manifest["summary"]["moved_roots"] == 1
    assert not source_root.exists()
    vault_audio = output_dir / "vault" / "r001" / "audio" / "vocal.flac"
    assert vault_audio.read_bytes() == original_audio
    path_map = read_jsonl(output_dir / "path_map.jsonl")
    assert path_map[0]["retention_mode"] == "exact_byte_vault"
    assert path_map[0]["root_alias"] == "r001"
    assert path_map[0]["move_strategy"] in {"rename", "copy_then_delete"}


def test_refuses_include_root_path_traversal(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab_artifacts"
    make_root(lab_root)

    with pytest.raises(ValueError, match="traversal"):
        build_artifact_rag_pack(
            lab_root=lab_root,
            output_dir=tmp_path / "artifact_rag" / "la003",
            include_roots=["../outside"],
        )


def test_cleanup_under_artifact_rag_mount_requires_active_mount(tmp_path: Path) -> None:
    mount_root = tmp_path / "artifact_rag"
    output_dir = mount_root / "la003"

    with pytest.raises(RuntimeError, match="not an active Docker mount"):
        validate_output_mount_contract(
            output_dir=output_dir,
            cleanup_mode="move_to_vault",
            execute_cleanup=True,
            artifact_rag_mount_root=mount_root,
            mount_checker=lambda _path: False,
        )

    validate_output_mount_contract(
        output_dir=output_dir,
        cleanup_mode="move_to_vault",
        execute_cleanup=True,
        artifact_rag_mount_root=mount_root,
        mount_checker=lambda _path: True,
    )

    validate_output_mount_contract(
        output_dir=output_dir,
        cleanup_mode="none",
        execute_cleanup=True,
        artifact_rag_mount_root=mount_root,
        mount_checker=lambda _path: False,
    )


def test_prune_existing_vault_marks_pack_thin_and_removes_binary_bytes(tmp_path: Path) -> None:
    pack_dir = tmp_path / "artifact_rag" / "la004"
    vault_audio = pack_dir / "vault" / "r001" / "audio" / "vocal.flac"
    vault_audio.parent.mkdir(parents=True)
    vault_audio.write_bytes(b"fake-audio")
    (pack_dir / "idx.json").write_text(
        json.dumps(
            {
                "schema": "artifact_rag_index.v2",
                "rag_id": "la004",
                "summary": {},
                "outputs": {"idx_json": "idx.json"},
                "retention_contract": {"binary_audio_model_bytes_available": True},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (pack_dir / "cleanup.json").write_text(
        json.dumps({"schema": "artifact_rag_cleanup.v1", "cleanup_summary": {}}, sort_keys=True),
        encoding="utf-8",
    )
    write_rows = [
        {
            "root": "old_run",
            "root_alias": "r001",
            "vault_path": str(pack_dir / "vault" / "r001"),
            "retention_mode": "exact_byte_vault",
        }
    ]
    (pack_dir / "path_map.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in write_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allow-prune-vault"):
        prune_existing_vault_pack(pack_dir=pack_dir)

    manifest = prune_existing_vault_pack(pack_dir=pack_dir, allow_prune_vault=True)

    assert manifest["status"] == "vault_pruned"
    assert manifest["binary_audio_model_bytes_available"] is False
    assert manifest["llm_retrieval_complete_for_textual_evidence"] is True
    assert not (pack_dir / "vault").exists()
    idx = read_json(pack_dir / "idx.json")
    assert idx["current_retention_mode"] == "thin_rag_only"
    assert idx["retention_contract"]["exact_byte_vault_available"] is False
    cleanup = read_json(pack_dir / "cleanup.json")
    assert cleanup["path_map_status"] == "historical_only_vault_pruned"
    path_map = read_jsonl(pack_dir / "path_map.jsonl")
    assert path_map[0]["vault_path_available"] is False
    assert path_map[0]["retention_mode"] == "thin_rag_only_after_vault_prune"
    assert (pack_dir / "prune.json").exists()
