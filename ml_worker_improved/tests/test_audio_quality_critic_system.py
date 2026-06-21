from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tensor_file(tmp_path: Path, name: str, value: float) -> str:
    path = tmp_path / "tensors" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    mix = np.full((2, 16), value, dtype=np.float32)
    vocal = np.full((2, 16), value * 0.25, dtype=np.float32)
    minus = np.full((2, 16), value * 0.70, dtype=np.float32)
    np.savez_compressed(path, mix=mix, vocal=vocal, minus=minus)
    return str(path)


def rag_current(tmp_path: Path) -> Path:
    idx = tmp_path / "artifact_rag" / "ar99" / "idx.json"
    write_json(
        idx,
        {
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar99",
            "outputs": {"chunks": "chunks.jsonl", "manifest_refs": "manifest_refs.jsonl", "files": "files.jsonl"},
            "retention_contract": {"llm_retrieval_complete_for_textual_evidence": True, "release_ready": False},
        },
    )
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def rag_current_with_manifest_ref(tmp_path: Path, manifest_path: Path) -> Path:
    rag_root = tmp_path / "artifact_rag" / "ar99"
    rag_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        rag_root / "manifest_refs.jsonl",
        [
            {
                "root": "verified",
                "root_alias": "r001",
                "relative_path": "verified/manifest.json",
                "sha256": "verified-manifest-sha",
                "chunk_id": "verified_manifest_chunk",
                "zip_member": "r001/verified/manifest.json",
                "original_path": str(manifest_path),
                "status_fields": {
                    "schema": "verified_audio_training_corpus_manifest.v1",
                    "delivery_allowed": False,
                    "release_ready": False,
                },
            }
        ],
    )
    write_json(
        rag_root / "idx.json",
        {
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar99",
            "outputs": {"chunks": "chunks.jsonl", "manifest_refs": "manifest_refs.jsonl", "files": "files.jsonl"},
        },
    )
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def rag_current_with_chunk_manifest_ref(tmp_path: Path, manifest_path: Path) -> Path:
    rag_root = tmp_path / "artifact_rag" / "ar99"
    rag_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        rag_root / "roots.jsonl",
        [
            {
                "root": "verified",
                "root_alias": "r001",
                "requested": "verified/r001",
                "original_path": str(manifest_path.parent.parent),
            }
        ],
    )
    write_jsonl(rag_root / "manifest_refs.jsonl", [])
    write_jsonl(
        rag_root / "chunks.jsonl",
        [
            {
                "root": "verified",
                "root_alias": "r001",
                "relative_path": f"{manifest_path.parent.name}/{manifest_path.name}",
                "chunk_id": "chunk_only_manifest",
                "status_fields": {"schema": "verified_audio_training_corpus_manifest.v1"},
                "text": '"schema": "verified_audio_training_corpus_manifest.v1"',
            }
        ],
    )
    write_jsonl(
        rag_root / "files.jsonl",
        [
            {
                "root": "verified",
                "root_alias": "r001",
                "relative_path": f"{manifest_path.parent.name}/{manifest_path.name}",
                "original_path": str(manifest_path),
                "text_sha256": "chunk-only-sha",
                "text_zip_member": "r001/verified/manifest.json",
            }
        ],
    )
    write_json(
        rag_root / "idx.json",
        {
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar99",
            "outputs": {"chunks": "chunks.jsonl", "manifest_refs": "manifest_refs.jsonl", "files": "files.jsonl", "roots": "roots.jsonl"},
        },
    )
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def rag_current_with_delivery_state(tmp_path: Path) -> Path:
    rag_root = tmp_path / "artifact_rag" / "ar99"
    rag_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        rag_root / "manifest_refs.jsonl",
        [
            {
                "relative_path": "progress.jsonl",
                "root": "r011",
                "root_alias": "r001",
                "status_fields": {"sample_rows": [{"delivery_status": "accepted"}]},
            },
            {
                "relative_path": "manifest.json",
                "root": "r010",
                "root_alias": "r003",
                "status_fields": {"delivery_status": "blocked", "delivery_allowed": False, "release_ready": False},
            },
            {
                "relative_path": "validation.json",
                "root": "r011",
                "root_alias": "r002",
                "status_fields": {"schema": "aggregate_short_listen_pack_validation.v1", "delivery_allowed": False, "release_ready": False},
            },
        ],
    )
    write_json(
        rag_root / "idx.json",
        {
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar99",
            "outputs": {"manifest_refs": "manifest_refs.jsonl"},
        },
    )
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def rag_current_with_gate(tmp_path: Path, gate_dir: Path, *, root_alias: str = "r001") -> Path:
    rag_root = tmp_path / "artifact_rag" / "ar99"
    rag_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        rag_root / "roots.jsonl",
        [
            {
                "root": "r001",
                "root_alias": root_alias,
                "requested": "gate/r001",
                "original_path": str(gate_dir),
                "text_snapshot_files": 3,
            }
        ],
    )
    write_jsonl(
        rag_root / "manifest_refs.jsonl",
        [
            {
                "root": "r001",
                "root_alias": root_alias,
                "relative_path": "manifest.json",
                "sha256": "manifest-sha",
                "status_fields": {
                    "schema": "aris_top_route_no_reference_gate_manifest.v1",
                    "manifest_type": "aris_top_route_no_reference_gate",
                    "delivery_allowed": False,
                    "release_ready": False,
                },
                "chunk_id": f"{root_alias}_manifest",
                "zip_member": f"{root_alias}/manifest.json",
            },
            {
                "root": "r001",
                "root_alias": root_alias,
                "relative_path": "judges.jsonl",
                "sha256": "judges-sha",
                "status_fields": {"sample_rows": [{"accepted": True}]},
                "chunk_id": f"{root_alias}_judges",
                "zip_member": f"{root_alias}/judges.jsonl",
            },
        ],
    )
    write_json(rag_root / "idx.json", {"index_schema": "artifact_rag_index.v2", "rag_id": "ar99", "outputs": {"roots": "roots.jsonl", "manifest_refs": "manifest_refs.jsonl"}})
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def dataset_row(tmp_path: Path, row_id: str, *, label_type: str, label: int, split: str = "train") -> dict:
    return {
        "track_id": f"track-{row_id}",
        "candidate_id": f"cand-{row_id}",
        "stem_type": "minus",
        "window_start_sec": 0.0,
        "window_end_sec": 6.0,
        "mix_path": str(tmp_path / "audio" / f"mix-{row_id}.flac"),
        "vocal_path": str(tmp_path / "audio" / f"vocal-{row_id}.flac"),
        "minus_path": str(tmp_path / "audio" / f"minus-{row_id}.flac"),
        "tensor_path": tensor_file(tmp_path, row_id, 0.2 + label),
        "route_id": f"route-{row_id}",
        "label_type": label_type,
        "label": label,
        "label_confidence": 1.0,
        "split": split,
        "source_manifest_refs": [f"/app/artifact_rag/runs/{row_id}/manifest.json"],
        "rag_refs": [{"rag_id": "ar99", "chunk_id": f"c-{row_id}"}],
        "issue_tags": [] if label else ["vocal_bleed_in_minus"],
        "blocker_reasons": [] if label else ["vocal_bleed_in_minus"],
    }


def gate_judge_row(tmp_path: Path, row_id: str, *, accepted: bool, pair_gate: str | None = None) -> dict:
    failed_checks = [] if accepted else ["phase_stereo_stability_below_detail_gate"]
    decision = "accept_candidate" if accepted else "blocked_hard_case"
    pair_consistency_gate = pair_gate or ("passed" if accepted else "failed")
    return {
        "accepted": accepted,
        "candidate_id": f"{row_id}-top",
        "failed_checks": failed_checks,
        "no_reference_gate": "passed" if accepted else "failed",
        "pair_consistency_gate": pair_consistency_gate,
        "reference_gate": "not_available",
        "schema": "aris_top_route_no_reference_pair_gate.v1",
        "track_id": f"track-{row_id}",
        "minus_judge": {
            "candidate_id": f"{row_id}-top",
            "confidence": 0.9,
            "decision": decision,
            "score_mode": "minus",
            "track_id": f"track-{row_id}",
            "mix_path": str(tmp_path / "audio" / f"mix-{row_id}.flac"),
            "vocal_path": str(tmp_path / "audio" / f"vocal-{row_id}.flac"),
            "instrumental_path": str(tmp_path / "audio" / f"minus-{row_id}.flac"),
            "minus_score": 0.9 if accepted else 0.2,
            "vocal_leakage_in_minus_score": 0.1 if accepted else 0.8,
            "instrumental_leakage_in_vocal_score": 0.0,
            "robotic_artifact_score": 0.0,
            "thin_vocal_score": 0.0,
            "silence_hallucination_score": 0.0,
            "transient_damage_score": 0.0,
            "phase_consistency_score": 0.8 if accepted else 0.3,
            "minus_blockers": failed_checks,
        },
        "vocal_judge": {
            "candidate_id": f"{row_id}-top",
            "confidence": 0.9,
            "decision": "accept_candidate",
            "score_mode": "vocal",
            "track_id": f"track-{row_id}",
            "mix_path": str(tmp_path / "audio" / f"mix-{row_id}.flac"),
            "vocal_path": str(tmp_path / "audio" / f"vocal-{row_id}.flac"),
            "instrumental_path": str(tmp_path / "audio" / f"minus-{row_id}.flac"),
            "vocal_score": 0.9,
            "vocal_blockers": [],
        },
        "pair_judge": {
            "candidate_id": f"{row_id}-top",
            "confidence": 0.9,
            "decision": "accept_candidate" if accepted and pair_consistency_gate == "passed" else "blocked_hard_case",
            "pair_decision": "pair_compatible" if accepted and pair_consistency_gate == "passed" else "blocked_pair_incompatible",
            "pair_compatibility_score": 0.9 if accepted else 0.2,
            "score_mode": "both",
            "track_id": f"track-{row_id}",
            "mix_path": str(tmp_path / "audio" / f"mix-{row_id}.flac"),
            "vocal_path": str(tmp_path / "audio" / f"vocal-{row_id}.flac"),
            "instrumental_path": str(tmp_path / "audio" / f"minus-{row_id}.flac"),
            "pair_blockers": failed_checks,
        },
    }


def write_gate_root(tmp_path: Path) -> Path:
    gate_dir = tmp_path / "artifact_rag" / "gate" / "r001"
    write_json(
        gate_dir / "manifest.json",
        {
            "schema": "aris_top_route_no_reference_gate_manifest.v1",
            "manifest_type": "aris_top_route_no_reference_gate",
            "status": "accepted",
            "delivery_allowed": False,
            "release_ready": False,
            "judges_jsonl": str(gate_dir / "judges.jsonl"),
        },
    )
    write_jsonl(gate_dir / "judges.jsonl", [gate_judge_row(tmp_path, "accepted", accepted=True), gate_judge_row(tmp_path, "blocked", accepted=False)])
    return gate_dir


def test_dataset_hard_fails_when_pseudo_teacher_claims_ground_truth(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset

    rows = write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                **dataset_row(tmp_path, "pseudo", label_type="pseudo_teacher", label=1),
                "pseudo_teacher_targets_are_ground_truth": True,
            }
        ],
    )

    try:
        build_audio_quality_critic_dataset(source_jsonls=[rows], output_dir=tmp_path / "dataset", rag_current_json=rag_current(tmp_path))
    except ValueError as exc:
        assert "pseudo_teacher_claims_ground_truth" in str(exc)
    else:  # pragma: no cover - explicit fail keeps pytest output direct.
        raise AssertionError("pseudo-teacher ground-truth claim must fail closed")


def test_rag_bridge_builds_weak_rows_from_no_reference_gate_without_ground_truth(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
    from tools.build_audio_quality_rows_from_rag import build_audio_quality_rows_from_rag

    gate_dir = write_gate_root(tmp_path)
    current = rag_current_with_gate(tmp_path, gate_dir)

    manifest = build_audio_quality_rows_from_rag(
        rag_current_json=current,
        output_dir=tmp_path / "aqc" / "r001",
        make_tensors=False,
    )
    dataset_manifest = build_audio_quality_critic_dataset(
        source_jsonls=[tmp_path / "aqc" / "r001" / "rows.jsonl"],
        output_dir=tmp_path / "aqc" / "r001" / "ds",
        rag_current_json=current,
    )
    rows = read_jsonl(tmp_path / "aqc" / "r001" / "rows.jsonl")

    assert manifest["schema"] == "audio_quality_rag_rows_manifest.v1"
    assert manifest["row_count"] == 2
    assert manifest["tensor_row_count"] == 0
    assert {row["label_type"] for row in rows} == {"weak_proxy", "rejected_blocked"}
    assert all(row["generated_stem_claims_ground_truth"] is False for row in rows)
    assert all(row["pseudo_teacher_targets_are_ground_truth"] is False for row in rows)
    assert all(row["source_manifest_refs"] for row in rows)
    assert all(row["rag_refs"] for row in rows)
    assert all(any(ref.get("zip_member") for ref in row["rag_refs"]) for row in rows)
    assert all(any(ref.get("sha256") for ref in row["rag_refs"]) for row in rows)
    assert dataset_manifest["safe_training_row_count"] == 2
    assert dataset_manifest["promotion_eligible_row_count"] == 0
    assert dataset_manifest["delivery_allowed"] is False


def test_rag_bridge_writes_short_tensor_aliases_when_audio_is_available(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    from tools.build_audio_quality_rows_from_rag import build_audio_quality_rows_from_rag

    gate_dir = write_gate_root(tmp_path)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.25, 0.25, 8000, dtype=np.float32)
    for name in ("mix-accepted", "vocal-accepted", "minus-accepted", "mix-blocked", "vocal-blocked", "minus-blocked"):
        soundfile.write(audio_dir / f"{name}.flac", samples, 8000)
    current = rag_current_with_gate(tmp_path, gate_dir)

    build_audio_quality_rows_from_rag(
        rag_current_json=current,
        output_dir=tmp_path / "aqc" / "r001",
        make_tensors=True,
        window_sec=0.25,
        tensor_bins=8,
    )
    rows = read_jsonl(tmp_path / "aqc" / "r001" / "rows.jsonl")

    assert all(Path(row["tensor_path"]).name in {"q001.npz", "q002.npz"} for row in rows)
    assert all("track-accepted" not in row["tensor_path"] and "track-blocked" not in row["tensor_path"] for row in rows)
    assert all(Path(row["tensor_path"]).exists() for row in rows)


def test_rag_bridge_pair_gate_failure_cannot_be_positive_weak_label(tmp_path: Path) -> None:
    from tools.build_audio_quality_rows_from_rag import build_audio_quality_rows_from_rag

    gate_dir = tmp_path / "artifact_rag" / "gate" / "r001"
    write_json(
        gate_dir / "manifest.json",
        {
            "schema": "aris_top_route_no_reference_gate_manifest.v1",
            "manifest_type": "aris_top_route_no_reference_gate",
            "status": "accepted",
            "delivery_allowed": False,
            "release_ready": False,
            "judges_jsonl": str(gate_dir / "judges.jsonl"),
        },
    )
    write_jsonl(gate_dir / "judges.jsonl", [gate_judge_row(tmp_path, "pairfail", accepted=True, pair_gate="failed")])
    current = rag_current_with_gate(tmp_path, gate_dir)

    build_audio_quality_rows_from_rag(
        rag_current_json=current,
        output_dir=tmp_path / "aqc" / "r001",
        make_tensors=False,
    )
    rows = read_jsonl(tmp_path / "aqc" / "r001" / "rows.jsonl")

    assert rows[0]["label_type"] == "rejected_blocked"
    assert rows[0]["label"] == 0
    assert "pair_consistency_gate_failed" in rows[0]["blocker_reasons"]


def test_source_lock_writes_short_rag_checkpoint_pack(tmp_path: Path) -> None:
    from tools.build_audio_quality_source_lock import build_audio_quality_source_lock

    pack = build_audio_quality_source_lock(
        rag_current_json=rag_current_with_delivery_state(tmp_path),
        output_dir=tmp_path / "artifact_rag" / "plan" / "p001",
        code_index_status="daemon_offline",
        code_index_fallback="CodeGraph + direct source reads",
        docker_service="ml-worker",
        gpu_status="not_checked_unit_test",
    )

    assert pack["schema"] == "audio_quality_source_lock.v1"
    assert pack["current_rag_id"] == "ar99"
    assert pack["code_index_status"] == "daemon_offline"
    assert pack["code_index_fallback"] == "CodeGraph + direct source reads"
    assert pack["delivery_allowed"] is False
    assert pack["release_ready"] is False
    assert pack["rag_delivery_state"]["delivery_status_counts"] == {"accepted": 1, "blocked": 1}
    assert pack["rag_delivery_state"]["aggregate_validation_ref_count"] == 1
    assert (tmp_path / "artifact_rag" / "plan" / "p001" / "pack.json").exists()
    assert (tmp_path / "artifact_rag" / "plan" / "p001" / "summary.md").exists()
    assert (tmp_path / "artifact_rag" / "plan" / "p001" / "progress.jsonl").exists()


def test_trainer_does_not_train_when_train_split_is_missing(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
    from tools.train_audio_quality_critic import train_audio_quality_critic

    rows = [
        dataset_row(tmp_path, "dev-a", label_type="weak_proxy", label=1, split="dev"),
        dataset_row(tmp_path, "test-b", label_type="weak_proxy", label=0, split="test"),
    ]
    build_audio_quality_critic_dataset(
        source_jsonls=[write_jsonl(tmp_path / "rows.jsonl", rows)],
        output_dir=tmp_path / "dataset",
        rag_current_json=rag_current(tmp_path),
    )

    manifest = train_audio_quality_critic(dataset_dir=tmp_path / "dataset", output_dir=tmp_path / "model", device="cpu", epochs=1, min_train_rows=1)

    assert manifest["training_status"] == "skipped"
    assert manifest["critic_not_usable_reason"] == "missing_train_split_rows"
    assert manifest["training_completed"] is False
    assert manifest["train_row_count"] == 0


def test_source_lock_pack_ref_propagates_to_dataset_model_and_score(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
    from tools.score_audio_quality_windows import score_audio_quality_windows
    from tools.train_audio_quality_critic import train_audio_quality_critic

    source_lock = write_json(tmp_path / "artifact_rag" / "plan" / "p001" / "pack.json", {"schema": "audio_quality_source_lock.v1"})
    rows = [
        {**dataset_row(tmp_path, "lock-a", label_type="weak_proxy", label=1), "source_lock_pack_json": str(source_lock)},
        {**dataset_row(tmp_path, "lock-b", label_type="weak_proxy", label=0), "source_lock_pack_json": str(source_lock)},
    ]
    dataset_manifest = build_audio_quality_critic_dataset(
        source_jsonls=[write_jsonl(tmp_path / "rows.jsonl", rows)],
        output_dir=tmp_path / "dataset",
        rag_current_json=rag_current(tmp_path),
    )
    model_manifest = train_audio_quality_critic(dataset_dir=tmp_path / "dataset", output_dir=tmp_path / "model", device="cpu", epochs=1)
    score_manifest = score_audio_quality_windows(
        candidate_jsonl=tmp_path / "dataset" / "dataset.jsonl",
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "score",
    )

    assert dataset_manifest["source_lock_pack_json"] == str(source_lock)
    assert model_manifest["source_lock_pack_json"] == str(source_lock)
    assert score_manifest["source_lock_pack_json"] == str(source_lock)


def test_dataset_preserves_rag_provenance_and_excludes_unsafe_rows(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset

    rows = [
        dataset_row(tmp_path, "verified-a", label_type="reference_verified", label=1),
        {**dataset_row(tmp_path, "unsafe", label_type="weak_proxy", label=0), "rag_refs": []},
    ]
    manifest = build_audio_quality_critic_dataset(
        source_jsonls=[write_jsonl(tmp_path / "rows.jsonl", rows)],
        output_dir=tmp_path / "dataset",
        rag_current_json=rag_current(tmp_path),
    )

    assert manifest["schema"] == "audio_quality_critic_dataset_manifest.v1"
    assert manifest["current_rag_id"] == "ar99"
    assert manifest["safe_training_row_count"] == 1
    assert manifest["unsafe_row_count"] == 1
    assert manifest["label_type_counts"]["reference_verified"] == 1
    assert manifest["label_type_counts"]["weak_proxy"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert {Path(path).name for path in manifest["outputs"].values()} == {"dataset.jsonl", "blocked.jsonl", "manifest.json", "progress.jsonl"}


def test_torch_critic_trains_shadow_model_but_weak_labels_cannot_promote(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
    from tools.train_audio_quality_critic import train_audio_quality_critic
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    rows = [
        dataset_row(tmp_path, "weak-a", label_type="weak_proxy", label=1),
        dataset_row(tmp_path, "weak-b", label_type="weak_proxy", label=0),
    ]
    build_audio_quality_critic_dataset(
        source_jsonls=[write_jsonl(tmp_path / "rows.jsonl", rows)],
        output_dir=tmp_path / "dataset",
        rag_current_json=rag_current(tmp_path),
    )

    manifest = train_audio_quality_critic(dataset_dir=tmp_path / "dataset", output_dir=tmp_path / "model", device="cpu", epochs=2)
    validation = validate_audio_quality_critic(tmp_path / "model", require_promotion_ready=True)

    assert manifest["schema"] == "audio_quality_critic_model_manifest.v1"
    assert manifest["model_family"] == "torch_audio_quality_critic"
    assert manifest["sklearn_used"] is False
    assert manifest["shadow_only"] is True
    assert manifest["promotion_allowed"] is False
    assert manifest["verified_training_row_count"] == 0
    assert validation["ok"] is False
    assert "verified_training_rows_missing" in validation["errors"]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def write_promotion_model_fixture(tmp_path: Path, *, device: str = "cuda", verified_rows: int = 2) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    source_lock = write_json(tmp_path / "artifact_rag" / "plan" / "p001" / "pack.json", {"schema": "audio_quality_source_lock.v1"})
    train_rows = [
        {**dataset_row(tmp_path, "train-pos", label_type="reference_verified", label=1, split="train"), "source_lock_pack_json": str(source_lock)},
        {**dataset_row(tmp_path, "train-neg", label_type="reference_verified", label=0, split="train"), "source_lock_pack_json": str(source_lock)},
    ]
    write_jsonl(dataset_dir / "dataset.jsonl", train_rows)
    write_json(dataset_dir / "manifest.json", {"schema": "audio_quality_critic_dataset_manifest.v1", "source_lock_pack_json": str(source_lock)})
    write_json(
        model_dir / "manifest.json",
        {
            "schema": "audio_quality_critic_model_manifest.v1",
            "manifest_type": "audio_quality_critic_model",
            "model_family": "torch_audio_quality_critic",
            "model_file": str(model_dir / "model.pt"),
            "dataset_manifest_json": str(dataset_dir / "manifest.json"),
            "source_lock_pack_json": str(source_lock),
            "training_completed": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "sklearn_used": False,
            "proxy_only_model": False,
            "input_tensor_required": True,
            "critical_leakage_count": 0,
            "rag_refs_complete": True,
            "false_accept_rate": None,
            "device": device,
            "verified_training_row_count": verified_rows,
            "label_type_counts": {"reference_verified": verified_rows},
            "delivery_allowed": False,
            "release_ready": False,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
            "thresholds_relaxed": False,
        },
    )
    (model_dir / "model.pt").write_bytes(b"fixture")
    return model_dir


def holdout_score_row(tmp_path: Path, row_id: str, *, label: int, critic_accepts: bool, label_type: str = "locked_holdout") -> dict:
    source_lock = tmp_path / "artifact_rag" / "plan" / "p001" / "pack.json"
    return {
        **dataset_row(tmp_path, row_id, label_type=label_type, label=label, split="locked_holdout"),
        "row_id": f"holdout-{row_id}",
        "source_lock_pack_json": str(source_lock),
        "critic_accepts": critic_accepts,
        "accept_probability": 0.99 if critic_accepts else 0.01,
        "selected_action": "shadow_accept_observation" if critic_accepts else "shadow_block_observation",
    }


def write_experiment_audit_pass_fixture(tmp_path: Path) -> Path:
    packet_manifest_json = write_json(
        tmp_path / "audit_packet" / "manifest.json",
        {
            "schema": "audio_quality_experiment_audit_packet.v1",
            "packet_status": "ready_for_independent_review",
            "experiment_audit_pass": False,
            "delivery_allowed": False,
            "release_ready": False,
            "path_count": 3,
        },
    )
    raw_audit_json = write_json(tmp_path / "raw_experiment_audit_pass.json", {"overall_verdict": "PASS"})
    return write_json(
        tmp_path / "experiment_audit_pass.json",
        {
            "overall_verdict": "PASS",
            "integrity_status": "pass",
            "auditor": "gpt-5.4-xhigh",
            "reviewer_trace_ref": "trace://aqc-test-pass",
            "checks": {
                "gt_provenance": {"status": "pass", "evidence": [str(tmp_path / "model" / "manifest.json")]},
                "score_normalization": {"status": "pass", "evidence": [str(tmp_path / "holdout_scores.jsonl")]},
                "result_existence": {"status": "pass", "evidence": [str(tmp_path / "model" / "model.pt")]},
                "dead_code": {"status": "pass", "evidence": [str(tmp_path / "model" / "manifest.json")]},
                "scope": {"status": "pass", "evidence": [str(tmp_path / "holdout_scores.jsonl")]},
                "eval_type": {"status": "pass", "classification": "real_gt", "evidence": [str(tmp_path / "holdout_scores.jsonl")]},
            },
            "source_packet_manifest_json": str(packet_manifest_json),
            "source_audit_json": str(raw_audit_json),
            "source_packet_path_count": 3,
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )


def rag_current_with_promotion_evidence(
    tmp_path: Path,
    *,
    model_manifest_json: Path,
    validation_json: Path,
    holdout_scores_jsonl: Path,
    experiment_audit_json: Path,
    validation_status_fields: dict | None = None,
) -> Path:
    rag_root = tmp_path / "artifact_rag" / "ar99"
    rag_root.mkdir(parents=True, exist_ok=True)
    validation_status = {
        "schema": "audio_quality_critic_validation.v1",
        "ok": True,
        "promotion_allowed": True,
        "delivery_allowed": False,
        "release_ready": False,
    }
    if validation_status_fields:
        validation_status.update(validation_status_fields)
    write_jsonl(
        rag_root / "manifest_refs.jsonl",
        [
            {
                "root": "aqc",
                "root_alias": "p001",
                "relative_path": "val/validation.json",
                "original_path": str(validation_json),
                "sha256": "validation-sha",
                "chunk_id": "validation-chunk",
                "zip_member": "p001/val/validation.json",
                "status_fields": validation_status,
            },
            {
                "root": "aqc",
                "root_alias": "p001",
                "relative_path": "model/manifest.json",
                "original_path": str(model_manifest_json),
                "sha256": "model-sha",
                "chunk_id": "model-chunk",
                "zip_member": "p001/model/manifest.json",
                "status_fields": {"schema": "audio_quality_critic_model_manifest.v1"},
            },
            {
                "root": "aqc",
                "root_alias": "p001",
                "relative_path": "audit/experiment_audit.json",
                "original_path": str(experiment_audit_json),
                "sha256": "audit-sha",
                "chunk_id": "audit-chunk",
                "zip_member": "p001/audit/experiment_audit.json",
                "status_fields": {"overall_verdict": "PASS", "integrity_status": "pass"},
            },
        ],
    )
    write_jsonl(
        rag_root / "files.jsonl",
        [
            {
                "root": "aqc",
                "root_alias": "p001",
                "relative_path": "hold/holdout_scores.jsonl",
                "original_path": str(holdout_scores_jsonl),
                "sha256": "holdout-sha",
                "chunk_id": "holdout-chunk",
                "text_zip_member": "p001/hold/holdout_scores.jsonl",
            }
        ],
    )
    write_jsonl(rag_root / "chunks.jsonl", [])
    write_json(
        rag_root / "idx.json",
        {
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar99",
            "outputs": {"manifest_refs": "manifest_refs.jsonl", "files": "files.jsonl", "chunks": "chunks.jsonl"},
        },
    )
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def test_promotion_validation_passes_only_locked_verified_holdout_without_release(tmp_path: Path) -> None:
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )

    result = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=write_experiment_audit_pass_fixture(tmp_path),
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )

    assert result["ok"] is True
    assert result["promotion_allowed"] is True
    assert result["delivery_allowed"] is False
    assert result["release_ready"] is False
    assert result["holdout_metrics"]["false_accept"] == 0
    assert result["holdout_metrics"]["split_leakage_count"] == 0


def test_promotion_validation_requires_experiment_audit_pass(tmp_path: Path) -> None:
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )

    missing_audit = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    assert missing_audit["ok"] is False
    assert missing_audit["promotion_allowed"] is False
    assert "missing_experiment_audit_json" in missing_audit["errors"]

    fail_audit_json = write_json(
        tmp_path / "experiment_audit_fail.json",
        {
            "overall_verdict": "FAIL",
            "integrity_status": "fail",
            "auditor": "gpt-5.4-xhigh",
            "checks": {"gt_provenance": {"status": "fail"}},
        },
    )
    fail_audit = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=fail_audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    assert fail_audit["ok"] is False
    assert fail_audit["promotion_allowed"] is False
    assert "experiment_audit_not_pass" in fail_audit["errors"]

    weak_pass_audit_json = write_json(
        tmp_path / "experiment_audit_weak_pass.json",
        {
            "overall_verdict": "PASS",
            "integrity_status": "pass",
            "auditor": "gpt-5.4-xhigh",
            "checks": {
                "gt_provenance": {"status": "pass"},
                "score_normalization": {"status": "pass"},
                "result_existence": {"status": "pass"},
            },
        },
    )
    weak_pass_audit = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=weak_pass_audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    assert weak_pass_audit["ok"] is False
    assert weak_pass_audit["promotion_allowed"] is False
    assert "reviewer_trace_ref_missing" in weak_pass_audit["errors"]
    assert "required_audit_check_missing" in weak_pass_audit["errors"]
    assert "audit_check_evidence_missing" in weak_pass_audit["errors"]

    pass_audit_json = write_experiment_audit_pass_fixture(tmp_path)
    pass_audit = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=pass_audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    assert pass_audit["ok"] is True
    assert pass_audit["promotion_allowed"] is True
    assert pass_audit["experiment_audit_status"] == "pass"
    assert pass_audit["delivery_allowed"] is False
    assert pass_audit["release_ready"] is False


def test_promotion_validation_rejects_raw_strict_audit_without_ingest_provenance(tmp_path: Path) -> None:
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )
    raw_strict_audit = write_json(
        tmp_path / "raw_strict_audit.json",
        {
            "overall_verdict": "PASS",
            "integrity_status": "pass",
            "auditor": "gpt-5.4-xhigh-independent",
            "reviewer_trace_ref": "trace://raw-direct-pass",
            "checks": {
                "gt_provenance": {"status": "pass", "evidence": [str(model_dir / "manifest.json")]},
                "score_normalization": {"status": "pass", "evidence": [str(holdout)]},
                "result_existence": {"status": "pass", "evidence": [str(model_dir / "model.pt")]},
                "dead_code": {"status": "pass", "evidence": [str(model_dir / "manifest.json")]},
                "scope": {"status": "pass", "evidence": [str(holdout)]},
                "eval_type": {"status": "pass", "classification": "real_gt", "evidence": [str(holdout)]},
            },
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )

    result = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=raw_strict_audit,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )

    assert result["ok"] is False
    assert result["promotion_allowed"] is False
    assert "experiment_audit_ingest_provenance_missing" in result["errors"]
    assert result["delivery_allowed"] is False
    assert result["release_ready"] is False


def test_promotion_validation_fails_closed_on_malformed_audit_ingest_provenance(tmp_path: Path) -> None:
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )
    packet_manifest_json = write_json(
        tmp_path / "audit_packet" / "manifest.json",
        {
            "schema": "audio_quality_experiment_audit_packet.v1",
            "packet_status": "ready_for_independent_review",
            "experiment_audit_pass": False,
            "delivery_allowed": False,
            "release_ready": False,
            "path_count": 3,
        },
    )
    raw_audit_json = write_json(tmp_path / "raw_experiment_audit.json", {"overall_verdict": "PASS"})
    malformed_audit = read_json(write_experiment_audit_pass_fixture(tmp_path))
    malformed_audit["source_packet_manifest_json"] = str(packet_manifest_json)
    malformed_audit["source_audit_json"] = str(raw_audit_json)
    malformed_audit["source_packet_path_count"] = "not-an-int"
    malformed_audit_json = write_json(tmp_path / "malformed_experiment_audit.json", malformed_audit)

    result = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=malformed_audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )

    assert result["ok"] is False
    assert result["promotion_allowed"] is False
    assert "experiment_audit_packet_path_count_invalid" in result["errors"]
    assert result["delivery_allowed"] is False
    assert result["release_ready"] is False


def test_promotion_decision_accepts_validated_locked_holdout_without_runtime_or_delivery(tmp_path: Path) -> None:
    from tools.build_audio_quality_promotion_decision import build_audio_quality_promotion_decision
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )
    audit_json = write_experiment_audit_pass_fixture(tmp_path)
    validation = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    validation_json = write_json(tmp_path / "validation.json", validation)

    manifest = build_audio_quality_promotion_decision(
        model_manifest_json=model_dir / "manifest.json",
        validation_json=validation_json,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
        output_dir=tmp_path / "promo" / "p001",
    )
    blocked = read_jsonl(tmp_path / "promo" / "p001" / "blocked.jsonl")

    assert manifest["schema"] == "audio_quality_promotion_decision.v1"
    assert manifest["decision_status"] == "accepted"
    assert manifest["promotion_decision"] == "accepted"
    assert manifest["promotion_allowed"] is True
    assert manifest["runtime_authority_enabled"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["model_manifest_json"] == str(model_dir / "manifest.json")
    assert manifest["validation_json"] == str(validation_json)
    assert manifest["holdout_scores_jsonl"] == str(holdout)
    assert manifest["experiment_audit_json"] == str(audit_json)
    assert manifest["validation_errors"] == []
    assert blocked == []
    assert Path(manifest["outputs"]["summary_md"]).name == "summary.md"


def test_promotion_decision_blocks_failed_or_delivery_contaminated_validation(tmp_path: Path) -> None:
    from tools.build_audio_quality_promotion_decision import build_audio_quality_promotion_decision
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=True),
        ],
    )
    audit_json = write_experiment_audit_pass_fixture(tmp_path)
    validation = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    validation["delivery_allowed"] = True
    validation_json = write_json(tmp_path / "validation.json", validation)

    manifest = build_audio_quality_promotion_decision(
        model_manifest_json=model_dir / "manifest.json",
        validation_json=validation_json,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
        output_dir=tmp_path / "promo" / "p001",
    )
    blocked = read_jsonl(tmp_path / "promo" / "p001" / "blocked.jsonl")
    blocked_reasons = [reason for row in blocked for reason in row["blocked_reasons"]]

    assert manifest["decision_status"] == "blocked"
    assert manifest["promotion_decision"] == "blocked"
    assert manifest["promotion_allowed"] is False
    assert manifest["runtime_authority_enabled"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert "validation_not_ok" in blocked_reasons
    assert "validation_delivery_allowed_true" in blocked_reasons


def test_promotion_candidate_discovery_finds_ready_current_rag_trio(tmp_path: Path) -> None:
    from tools.discover_audio_quality_promotion_candidates import discover_audio_quality_promotion_candidates
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )
    audit_json = write_experiment_audit_pass_fixture(tmp_path)
    validation = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    validation_json = write_json(tmp_path / "validation.json", validation)
    current = rag_current_with_promotion_evidence(
        tmp_path,
        model_manifest_json=model_dir / "manifest.json",
        validation_json=validation_json,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
    )

    manifest = discover_audio_quality_promotion_candidates(
        rag_current_json=current,
        output_dir=tmp_path / "artifact_rag" / "aqp" / "p001",
    )
    candidates = read_jsonl(tmp_path / "artifact_rag" / "aqp" / "p001" / "candidates.jsonl")
    blocked = read_jsonl(tmp_path / "artifact_rag" / "aqp" / "p001" / "blocked.jsonl")

    assert manifest["schema"] == "audio_quality_promotion_candidate_discovery.v1"
    assert manifest["discovery_status"] == "ready"
    assert manifest["candidate_count"] == 1
    assert manifest["blocked_ref_count"] == 0
    assert manifest["promotion_allowed"] is False
    assert manifest["runtime_authority_enabled"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert {Path(path).name for path in manifest["outputs"].values()} == {"candidates.jsonl", "blocked.jsonl", "manifest.json", "summary.md", "progress.jsonl"}
    assert blocked == []
    assert candidates[0]["ready_for_promotion_decision"] is True
    assert candidates[0]["selected_action"] == "build_audio_quality_promotion_decision"
    assert candidates[0]["model_manifest_json"] == str(model_dir / "manifest.json")
    assert candidates[0]["validation_json"] == str(validation_json)
    assert candidates[0]["holdout_scores_jsonl"] == str(holdout)
    assert candidates[0]["experiment_audit_json"] == str(audit_json)
    assert candidates[0]["delivery_allowed"] is False
    assert candidates[0]["release_ready"] is False
    assert len(candidates[0]["rag_refs"]) == 4


def test_promotion_candidate_discovery_blocks_failed_current_rag_without_fabricating_pass(tmp_path: Path) -> None:
    from tools.discover_audio_quality_promotion_candidates import discover_audio_quality_promotion_candidates
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=True),
        ],
    )
    audit_json = write_experiment_audit_pass_fixture(tmp_path)
    validation = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )
    validation_json = write_json(tmp_path / "validation.json", validation)
    current = rag_current_with_promotion_evidence(
        tmp_path,
        model_manifest_json=model_dir / "manifest.json",
        validation_json=validation_json,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=audit_json,
        validation_status_fields={"ok": False, "promotion_allowed": False},
    )

    manifest = discover_audio_quality_promotion_candidates(
        rag_current_json=current,
        output_dir=tmp_path / "artifact_rag" / "aqp" / "p002",
    )
    candidates = read_jsonl(tmp_path / "artifact_rag" / "aqp" / "p002" / "candidates.jsonl")
    blocked = read_jsonl(tmp_path / "artifact_rag" / "aqp" / "p002" / "blocked.jsonl")
    blocked_reasons = [reason for row in blocked for reason in row["blocked_reasons"]]

    assert manifest["discovery_status"] == "blocked"
    assert manifest["candidate_count"] == 0
    assert manifest["promotion_allowed"] is False
    assert manifest["runtime_authority_enabled"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert "promotion_candidate_not_found" in manifest["errors"]
    assert candidates == []
    assert "validation_not_ok" in blocked_reasons
    assert "validation_promotion_allowed_not_true" in blocked_reasons
    assert all(row["ready_for_promotion_decision"] is False for row in blocked)


def test_promotion_candidate_discovery_reports_missing_inputs_without_directory_reads(tmp_path: Path) -> None:
    from tools.discover_audio_quality_promotion_candidates import discover_audio_quality_promotion_candidates

    validation_json = write_json(
        tmp_path / "validation.json",
        {
            "schema": "audio_quality_critic_validation.v1",
            "ok": False,
            "promotion_allowed": False,
            "require_promotion_ready": True,
            "delivery_allowed": False,
            "release_ready": False,
            "errors": ["missing_holdout_scores_jsonl", "missing_experiment_audit_json"],
        },
    )
    rag_root = tmp_path / "artifact_rag" / "ar99"
    write_jsonl(
        rag_root / "manifest_refs.jsonl",
        [
            {
                "root": "aqc",
                "root_alias": "p001",
                "relative_path": "val/validation.json",
                "original_path": str(validation_json),
                "sha256": "validation-sha",
                "chunk_id": "validation-chunk",
                "zip_member": "p001/val/validation.json",
                "status_fields": {"schema": "audio_quality_critic_validation.v1"},
            }
        ],
    )
    write_jsonl(rag_root / "files.jsonl", [])
    write_jsonl(rag_root / "chunks.jsonl", [])
    write_json(
        rag_root / "idx.json",
        {"index_schema": "artifact_rag_index.v2", "rag_id": "ar99", "outputs": {"manifest_refs": "manifest_refs.jsonl", "files": "files.jsonl", "chunks": "chunks.jsonl"}},
    )
    current = write_json(
        tmp_path / "artifact_rag" / "current.json",
        {"schema": "artifact_rag_current.v1", "rag_id": "ar99", "idx": "ar99/idx.json"},
    )

    discover_audio_quality_promotion_candidates(
        rag_current_json=current,
        output_dir=tmp_path / "artifact_rag" / "aqp" / "p003",
    )
    blocked = read_jsonl(tmp_path / "artifact_rag" / "aqp" / "p003" / "blocked.jsonl")
    blocked_reasons = [reason for row in blocked for reason in row["blocked_reasons"]]

    assert "model_manifest_json_missing" in blocked_reasons
    assert "holdout_scores_jsonl_missing" in blocked_reasons
    assert "experiment_audit_json_missing" in blocked_reasons
    assert not any("IsADirectoryError" in reason for reason in blocked_reasons)


def test_experiment_audit_packet_collects_paths_without_claiming_pass(tmp_path: Path) -> None:
    from tools.build_audio_quality_experiment_audit_packet import build_audio_quality_experiment_audit_packet

    model_dir = write_promotion_model_fixture(tmp_path)
    model_manifest = model_dir / "manifest.json"
    validation_json = write_json(
        tmp_path / "validation.json",
        {
            "schema": "audio_quality_critic_validation.v1",
            "ok": False,
            "errors": ["missing_experiment_audit_json"],
            "model_dir": str(model_dir),
            "promotion_allowed": False,
            "delivery_allowed": False,
            "release_ready": False,
        },
    )
    holdout_scores = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )

    manifest = build_audio_quality_experiment_audit_packet(
        output_dir=tmp_path / "audit_packet",
        model_manifest_json=model_manifest,
        validation_json=validation_json,
        holdout_scores_jsonl=holdout_scores,
        extra_paths=[tmp_path / "dataset" / "manifest.json"],
    )
    paths = read_jsonl(tmp_path / "audit_packet" / "paths.jsonl")
    prompt = (tmp_path / "audit_packet" / "reviewer_prompt.md").read_text(encoding="utf-8")

    assert manifest["schema"] == "audio_quality_experiment_audit_packet.v1"
    assert manifest["packet_status"] == "ready_for_independent_review"
    assert manifest["experiment_audit_pass"] is False
    assert manifest["promotion_allowed"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["reviewer_required"] is True
    assert {row["category"] for row in paths} >= {"model_manifest", "validation_result", "holdout_scores", "extra_evidence"}
    assert all(Path(row["path"]).exists() for row in paths)
    assert "Read ALL files listed below" in prompt
    assert "Ground Truth Provenance" in prompt
    assert "Score Normalization" in prompt
    assert not (tmp_path / "audit_packet" / "EXPERIMENT_AUDIT.json").exists()


def test_experiment_audit_packet_blocks_missing_required_evidence(tmp_path: Path) -> None:
    from tools.build_audio_quality_experiment_audit_packet import build_audio_quality_experiment_audit_packet

    manifest = build_audio_quality_experiment_audit_packet(
        output_dir=tmp_path / "audit_packet",
        model_manifest_json=tmp_path / "missing_model_manifest.json",
        validation_json=tmp_path / "missing_validation.json",
    )
    blocked = read_jsonl(tmp_path / "audit_packet" / "blocked.jsonl")

    assert manifest["packet_status"] == "blocked"
    assert manifest["experiment_audit_pass"] is False
    assert manifest["promotion_allowed"] is False
    assert "model_manifest_missing" in manifest["errors"]
    assert "validation_json_missing" in manifest["errors"]
    assert blocked[0]["blocked_reasons"] == ["model_manifest_missing", "validation_json_missing"]
    assert read_jsonl(tmp_path / "audit_packet" / "paths.jsonl") == []


def test_experiment_audit_ingest_accepts_independent_pass_report(tmp_path: Path) -> None:
    from tools.build_audio_quality_experiment_audit_packet import build_audio_quality_experiment_audit_packet
    from tools.ingest_audio_quality_experiment_audit import ingest_audio_quality_experiment_audit
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "holdout-pos", label=1, critic_accepts=True),
            holdout_score_row(tmp_path, "holdout-neg", label=0, critic_accepts=False),
        ],
    )
    validation_json = write_json(
        tmp_path / "validation.json",
        {
            "schema": "audio_quality_critic_validation.v1",
            "ok": False,
            "errors": ["missing_experiment_audit_json"],
            "model_dir": str(model_dir),
            "promotion_allowed": False,
            "delivery_allowed": False,
            "release_ready": False,
        },
    )
    packet = build_audio_quality_experiment_audit_packet(
        output_dir=tmp_path / "audit_packet",
        model_manifest_json=model_dir / "manifest.json",
        validation_json=validation_json,
        holdout_scores_jsonl=holdout,
    )
    raw_audit = write_json(
        tmp_path / "raw_audit.json",
        {
            "overall_verdict": "PASS",
            "integrity_status": "pass",
            "auditor": "gpt-5.4-xhigh-independent",
            "auditor_model": "gpt-5.4",
            "reviewer_trace_ref": "trace://aqc-audit-pass",
            "checks": {
                "gt_provenance": {"status": "pass", "evidence": [str(model_dir / "manifest.json")]},
                "score_normalization": {"status": "pass", "evidence": [str(holdout)]},
                "result_existence": {"status": "pass", "evidence": [str(validation_json)]},
                "dead_code": {"status": "pass", "evidence": [str(model_dir / "manifest.json")]},
                "scope": {"status": "pass", "evidence": [str(holdout)]},
                "eval_type": {"status": "pass", "classification": "real_gt", "evidence": [str(holdout)]},
            },
            "claims": [{"id": "aqc-promotion", "impact": "supported"}],
        },
    )

    manifest = ingest_audio_quality_experiment_audit(
        packet_manifest_json=Path(packet["outputs"]["manifest_json"]),
        audit_json=raw_audit,
        output_dir=tmp_path / "audit_ingest",
    )
    normalized = json.loads((tmp_path / "audit_ingest" / "EXPERIMENT_AUDIT.json").read_text(encoding="utf-8"))
    validation = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        experiment_audit_json=tmp_path / "audit_ingest" / "EXPERIMENT_AUDIT.json",
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )

    assert manifest["ingest_status"] == "accepted"
    assert manifest["experiment_audit_pass"] is True
    assert manifest["promotion_allowed"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert normalized["overall_verdict"] == "PASS"
    assert normalized["integrity_status"] == "pass"
    assert normalized["source_packet_manifest_json"] == str(packet["outputs"]["manifest_json"])
    assert validation["ok"] is True
    assert validation["promotion_allowed"] is True


def test_experiment_audit_ingest_blocks_packet_or_weak_report_as_pass(tmp_path: Path) -> None:
    from tools.build_audio_quality_experiment_audit_packet import build_audio_quality_experiment_audit_packet
    from tools.ingest_audio_quality_experiment_audit import ingest_audio_quality_experiment_audit

    model_dir = write_promotion_model_fixture(tmp_path)
    validation_json = write_json(
        tmp_path / "validation.json",
        {
            "schema": "audio_quality_critic_validation.v1",
            "ok": False,
            "errors": ["missing_experiment_audit_json"],
            "model_dir": str(model_dir),
            "promotion_allowed": False,
            "delivery_allowed": False,
            "release_ready": False,
        },
    )
    packet = build_audio_quality_experiment_audit_packet(
        output_dir=tmp_path / "audit_packet",
        model_manifest_json=model_dir / "manifest.json",
        validation_json=validation_json,
    )

    packet_as_audit = ingest_audio_quality_experiment_audit(
        packet_manifest_json=Path(packet["outputs"]["manifest_json"]),
        audit_json=Path(packet["outputs"]["manifest_json"]),
        output_dir=tmp_path / "packet_as_audit",
    )
    assert packet_as_audit["ingest_status"] == "blocked"
    assert packet_as_audit["experiment_audit_pass"] is False
    assert "audit_json_is_packet_manifest" in packet_as_audit["errors"]
    assert not (tmp_path / "packet_as_audit" / "EXPERIMENT_AUDIT.json").exists()

    weak_audit = write_json(
        tmp_path / "weak_audit.json",
        {
            "overall_verdict": "PASS",
            "integrity_status": "pass",
            "auditor": "same-agent",
            "checks": {"gt_provenance": {"status": "pass"}},
        },
    )
    weak_result = ingest_audio_quality_experiment_audit(
        packet_manifest_json=Path(packet["outputs"]["manifest_json"]),
        audit_json=weak_audit,
        output_dir=tmp_path / "weak_ingest",
    )
    assert weak_result["ingest_status"] == "blocked"
    assert weak_result["experiment_audit_pass"] is False
    assert "reviewer_trace_ref_missing" in weak_result["errors"]
    assert "required_audit_check_missing" in weak_result["errors"]


def test_promotion_validation_fails_on_false_accept_leakage_and_weak_holdout(tmp_path: Path) -> None:
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    holdout = write_jsonl(
        tmp_path / "holdout_scores.jsonl",
        [
            holdout_score_row(tmp_path, "train-neg", label=0, critic_accepts=True),
            holdout_score_row(tmp_path, "weak", label=1, critic_accepts=True, label_type="weak_proxy"),
        ],
    )

    result = validate_audio_quality_critic(
        model_dir,
        require_promotion_ready=True,
        holdout_scores_jsonl=holdout,
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )

    assert result["ok"] is False
    assert result["promotion_allowed"] is False
    assert "false_accept_rate_above_budget" in result["errors"]
    assert "split_leakage_detected" in result["errors"]
    assert "non_promotion_label_in_holdout" in result["errors"]
    assert result["delivery_allowed"] is False
    assert result["release_ready"] is False


def test_promotion_validation_requires_locked_holdout_even_if_manifest_claims_promotion(tmp_path: Path) -> None:
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    model_dir = write_promotion_model_fixture(tmp_path)
    manifest_path = model_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["shadow_only"] = False
    manifest["promotion_allowed"] = True
    write_json(manifest_path, manifest)

    result = validate_audio_quality_critic(model_dir, require_promotion_ready=True, max_false_accept_rate=0.0)

    assert result["ok"] is False
    assert result["promotion_allowed"] is False
    assert "missing_holdout_scores_jsonl" in result["errors"]
    assert result["delivery_allowed"] is False
    assert result["release_ready"] is False


def write_locked_holdout_dataset_fixture(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_dir = tmp_path / "dataset"
    source_lock = write_json(tmp_path / "artifact_rag" / "plan" / "p001" / "pack.json", {"schema": "audio_quality_source_lock.v1"})
    normalized_rows = [
        {
            **row,
            "row_id": str(row.get("row_id") or f"row-{row.get('candidate_id')}"),
            "safe_for_training": True,
            "promotion_eligible": row.get("label_type") in {"reference_verified", "human_verified", "locked_holdout"},
            "source_lock_pack_json": str(row.get("source_lock_pack_json") or source_lock),
            "delivery_allowed": False,
            "release_ready": False,
        }
        for row in rows
    ]
    write_jsonl(dataset_dir / "dataset.jsonl", normalized_rows)
    write_json(
        dataset_dir / "manifest.json",
        {
            "schema": "audio_quality_critic_dataset_manifest.v1",
            "outputs": {
                "dataset_jsonl": str(dataset_dir / "dataset.jsonl"),
                "manifest_json": str(dataset_dir / "manifest.json"),
            },
            "source_lock_pack_json": str(source_lock),
            "delivery_allowed": False,
            "release_ready": False,
        },
    )
    return dataset_dir


def write_score_manifest_fixture(tmp_path: Path, rows: list[dict]) -> Path:
    score_dir = tmp_path / "scores"
    source_lock = tmp_path / "artifact_rag" / "plan" / "p001" / "pack.json"
    scores = [
        {
            **row,
            "row_id": str(row.get("row_id") or f"row-{row.get('candidate_id')}"),
            "critic_accepts": bool(row.get("critic_accepts")),
            "accept_probability": 0.99 if row.get("critic_accepts") else 0.01,
            "delivery_allowed": False,
            "release_ready": False,
            "source_lock_pack_json": str(row.get("source_lock_pack_json") or source_lock),
        }
        for row in rows
    ]
    scores_jsonl = write_jsonl(score_dir / "scores.jsonl", scores)
    return write_json(
        score_dir / "manifest.json",
        {
            "schema": "audio_quality_critic_score_manifest.v1",
            "scores_jsonl": str(scores_jsonl),
            "source_lock_pack_json": str(source_lock),
            "delivery_allowed": False,
            "release_ready": False,
        },
    )


def test_locked_holdout_pack_uses_only_verified_holdout_scores_for_promotion(tmp_path: Path) -> None:
    from tools.build_audio_quality_locked_holdout_pack import build_audio_quality_locked_holdout_pack
    from tools.validate_audio_quality_critic import validate_audio_quality_critic

    train_pos = dataset_row(tmp_path, "train-a", label_type="reference_verified", label=1, split="train")
    train_neg = dataset_row(tmp_path, "train-b", label_type="reference_verified", label=0, split="train")
    holdout_pos = dataset_row(tmp_path, "holdout-a", label_type="reference_verified", label=1, split="locked_holdout")
    holdout_neg = dataset_row(tmp_path, "holdout-b", label_type="human_verified", label=0, split="locked_holdout")
    weak_holdout = dataset_row(tmp_path, "weak-holdout", label_type="weak_proxy", label=1, split="locked_holdout")
    dataset_dir = write_locked_holdout_dataset_fixture(tmp_path, [train_pos, train_neg, holdout_pos, holdout_neg, weak_holdout])
    score_manifest = write_score_manifest_fixture(
        tmp_path,
        [
            {**holdout_pos, "critic_accepts": True},
            {**holdout_neg, "critic_accepts": False},
            {**weak_holdout, "critic_accepts": True},
        ],
    )

    manifest = build_audio_quality_locked_holdout_pack(
        dataset_dir=dataset_dir,
        score_manifest_json=score_manifest,
        output_dir=tmp_path / "holdout",
        min_holdout_rows=2,
    )
    holdout_rows = read_jsonl(tmp_path / "holdout" / "holdout_scores.jsonl")
    validation = validate_audio_quality_critic(
        write_promotion_model_fixture(tmp_path),
        require_promotion_ready=True,
        holdout_scores_jsonl=tmp_path / "holdout" / "holdout_scores.jsonl",
        experiment_audit_json=write_experiment_audit_pass_fixture(tmp_path),
        max_false_accept_rate=0.0,
        min_holdout_rows=2,
    )

    assert manifest["promotion_evidence_ready"] is True
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert [row["label_type"] for row in holdout_rows] == ["reference_verified", "human_verified"]
    assert all(row["split"] == "locked_holdout" for row in holdout_rows)
    assert all(row["delivery_allowed"] is False and row["release_ready"] is False for row in holdout_rows)
    assert read_jsonl(tmp_path / "holdout" / "blocked.jsonl")[0]["blocked_reasons"] == ["non_promotion_label_type"]
    assert validation["ok"] is True
    assert validation["promotion_allowed"] is True


def test_locked_holdout_pack_blocks_leakage_weak_labels_and_missing_hard_case(tmp_path: Path) -> None:
    from tools.build_audio_quality_locked_holdout_pack import build_audio_quality_locked_holdout_pack

    train_row = dataset_row(tmp_path, "same-track", label_type="reference_verified", label=1, split="train")
    leaky_holdout = {**dataset_row(tmp_path, "same-track", label_type="reference_verified", label=1, split="locked_holdout"), "row_id": "leaky"}
    weak_holdout = dataset_row(tmp_path, "weak-only", label_type="weak_proxy", label=1, split="locked_holdout")
    dataset_dir = write_locked_holdout_dataset_fixture(tmp_path, [train_row, leaky_holdout, weak_holdout])
    score_manifest = write_score_manifest_fixture(
        tmp_path,
        [
            {**leaky_holdout, "critic_accepts": True},
            {**weak_holdout, "critic_accepts": True},
        ],
    )

    manifest = build_audio_quality_locked_holdout_pack(
        dataset_dir=dataset_dir,
        score_manifest_json=score_manifest,
        output_dir=tmp_path / "holdout",
        min_holdout_rows=2,
    )

    assert manifest["promotion_evidence_ready"] is False
    assert "split_leakage_detected" in manifest["errors"]
    assert "negative_holdout_missing" in manifest["errors"]
    assert "hard_case_holdout_missing" in manifest["errors"]
    assert read_jsonl(tmp_path / "holdout" / "holdout_scores.jsonl") == []
    blocked_reasons = [reason for row in read_jsonl(tmp_path / "holdout" / "blocked.jsonl") for reason in row["blocked_reasons"]]
    assert "split_leakage_detected" in blocked_reasons
    assert "non_promotion_label_type" in blocked_reasons
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_locked_holdout_pack_rejects_delivery_or_release_contaminated_manifests(tmp_path: Path) -> None:
    from tools.build_audio_quality_locked_holdout_pack import build_audio_quality_locked_holdout_pack

    holdout_pos = dataset_row(tmp_path, "holdout-clean-pos", label_type="reference_verified", label=1, split="locked_holdout")
    holdout_neg = dataset_row(tmp_path, "holdout-clean-neg", label_type="human_verified", label=0, split="locked_holdout")
    dataset_dir = write_locked_holdout_dataset_fixture(tmp_path, [holdout_pos, holdout_neg])
    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = read_json(dataset_manifest_path)
    dataset_manifest["delivery_allowed"] = True
    write_json(dataset_manifest_path, dataset_manifest)
    score_manifest = write_score_manifest_fixture(
        tmp_path,
        [
            {**holdout_pos, "critic_accepts": True},
            {**holdout_neg, "critic_accepts": False},
        ],
    )
    score_manifest_payload = read_json(score_manifest)
    score_manifest_payload["release_ready"] = True
    write_json(score_manifest, score_manifest_payload)

    manifest = build_audio_quality_locked_holdout_pack(
        dataset_dir=dataset_dir,
        score_manifest_json=score_manifest,
        output_dir=tmp_path / "holdout",
        min_holdout_rows=2,
    )

    assert manifest["promotion_evidence_ready"] is False
    assert "dataset_delivery_allowed_true" in manifest["errors"]
    assert "score_release_ready_true" in manifest["errors"]
    assert read_jsonl(tmp_path / "holdout" / "holdout_scores.jsonl") == []
    blocked_reasons = [reason for row in read_jsonl(tmp_path / "holdout" / "blocked.jsonl") for reason in row["blocked_reasons"]]
    assert "dataset_delivery_allowed_true" in blocked_reasons
    assert "score_release_ready_true" in blocked_reasons
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_locked_holdout_pack_requires_score_manifest_source_lock(tmp_path: Path) -> None:
    from tools.build_audio_quality_locked_holdout_pack import build_audio_quality_locked_holdout_pack

    holdout_pos = dataset_row(tmp_path, "holdout-lock-pos", label_type="reference_verified", label=1, split="locked_holdout")
    holdout_neg = dataset_row(tmp_path, "holdout-lock-neg", label_type="human_verified", label=0, split="locked_holdout")
    dataset_dir = write_locked_holdout_dataset_fixture(tmp_path, [holdout_pos, holdout_neg])
    score_manifest = write_score_manifest_fixture(
        tmp_path,
        [
            {**holdout_pos, "critic_accepts": True},
            {**holdout_neg, "critic_accepts": False},
        ],
    )
    score_manifest_payload = read_json(score_manifest)
    score_manifest_payload.pop("source_lock_pack_json")
    write_json(score_manifest, score_manifest_payload)

    manifest = build_audio_quality_locked_holdout_pack(
        dataset_dir=dataset_dir,
        score_manifest_json=score_manifest,
        output_dir=tmp_path / "holdout",
        min_holdout_rows=2,
    )

    assert manifest["promotion_evidence_ready"] is False
    assert "score_manifest_source_lock_missing" in manifest["errors"]
    assert read_jsonl(tmp_path / "holdout" / "holdout_scores.jsonl") == []
    blocked_reasons = [reason for row in read_jsonl(tmp_path / "holdout" / "blocked.jsonl") for reason in row["blocked_reasons"]]
    assert "score_manifest_source_lock_missing" in blocked_reasons
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def _write_audio_file(path: Path, value: float = 0.1) -> str:
    soundfile = pytest.importorskip("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full((1600, 2), value, dtype=np.float32)
    soundfile.write(path, samples, 8000)
    return str(path)


def _verified_example_row(tmp_path: Path, row_id: str, *, split: str, stem_type: str, reference_verified: bool) -> dict:
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem_type,
        "manifest_path": str(tmp_path / "refs" / f"{row_id}.json"),
        "source_audio_path": _write_audio_file(tmp_path / "audio" / row_id / "mix.flac", 0.12),
        "target_audio_path": _write_audio_file(tmp_path / "audio" / row_id / "target.flac", 0.04),
        "target_audio_sha256": f"sha-{row_id}",
        "permission_status": "project_owned",
        "split": split,
        "trainable": True,
        "human_verified": not reference_verified,
        "reference_verified": reference_verified,
        "release_ready": False,
        "delivery_allowed": False,
    }


def _verified_report_row(row: dict, *, safe: bool = True) -> dict:
    return {
        "schema": "verified_audio_training_corpus_row.v1",
        "id": row["id"],
        "track_id": row["track_id"],
        "stem_type": row["stem_type"],
        "split": row["split"],
        "source_audio_path": row["source_audio_path"],
        "target_audio_path": row["target_audio_path"],
        "human_verified": row["human_verified"],
        "reference_verified": row["reference_verified"],
        "permission_status": row["permission_status"],
        "safe_for_training": safe,
        "unsafe_reasons": [] if safe else ["fixture_unsafe"],
        "release_ready": False,
    }


def _verified_corpus_manifest(tmp_path: Path, rows: list[dict], reports: list[dict], *, status: str = "passed") -> Path:
    examples_jsonl = write_jsonl(tmp_path / "verified" / "examples.jsonl", rows)
    report_jsonl = write_jsonl(tmp_path / "verified" / "rows.jsonl", reports)
    return write_json(
        tmp_path / "verified" / "manifest.json",
        {
            "schema": "verified_audio_training_corpus_manifest.v1",
            "manifest_type": "verified_audio_training_corpus",
            "audio_training_examples_jsonl": str(examples_jsonl),
            "row_report_jsonl": str(report_jsonl),
            "corpus_validation_status": status,
            "failed_checks": [] if status == "passed" else ["fixture_failed"],
            "delivery_allowed": False,
            "release_ready": False,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
            "thresholds_relaxed": False,
        },
    )


def _aqc_source_lock(
    tmp_path: Path,
    *,
    current_rag_id: str = "ar99",
    schema: str = "audio_quality_source_lock.v1",
    rag_idx_exists: bool = True,
) -> Path:
    return write_json(
        tmp_path / "artifact_rag" / "plan" / "p001" / "pack.json",
        {
            "schema": schema,
            "manifest_type": "audio_quality_source_lock",
            "current_rag_id": current_rag_id,
            "rag_idx": f"{current_rag_id}/idx.json",
            "rag_idx_exists": rag_idx_exists,
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
            "thresholds_relaxed": False,
        },
    )


def _rag_current_with_unverified_raw_reference(tmp_path: Path) -> Path:
    raw_reference_path = _write_audio_file(
        tmp_path / "artifact_rag" / "exec" / "r001" / "jobs" / "t001" / "w001" / "f001" / "vocal" / "00_raw_ref.flac",
        0.05,
    )
    stem_manifest = {
        "schema_version": 1,
        "source_audio_path": "/host_testing_stuff_music/source.mp3",
        "model_id": "scnet_xl_ihf_musdb18",
        "model": {"model_id": "scnet_xl_ihf_musdb18", "output_stems": ["vocals"]},
        "target_stem": "vocal",
        "raw_reference": {
            "path": raw_reference_path,
            "sha256": "raw-ref-sha",
            "source": "existing_manifest_stem",
            "source_stems": ["vocals"],
        },
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_not_for_listen_pack",
        "release_ready": False,
    }
    rag_root = tmp_path / "artifact_rag" / "ar99"
    rag_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        rag_root / "manifest_refs.jsonl",
        [
            {
                "root": "packsrc/r001",
                "root_alias": "r001",
                "relative_path": "manifest.json",
                "sha256": "packsrc-sha",
                "chunk_id": "packsrc_manifest_chunk",
                "status_fields": {
                    "schema": "aris_accepted_full_pack_source_manifest_manifest.v1",
                    "manifest_type": "aris_accepted_full_pack_source_manifest",
                    "delivery_allowed": False,
                    "release_ready": False,
                },
            }
        ],
    )
    write_jsonl(
        rag_root / "chunks.jsonl",
        [
            {
                "root": "exec/r001",
                "root_alias": "r002",
                "relative_path": "jobs/t001/w001/f001/vocal/stem_manifest.json",
                "chunk_id": "raw_reference_chunk",
                "text": json.dumps(stem_manifest, sort_keys=True),
                "status_fields": {
                    "delivery_allowed": False,
                    "presentation_status": "diagnostic_candidate_not_for_listen_pack",
                    "release_ready": False,
                },
            }
        ],
    )
    write_jsonl(rag_root / "files.jsonl", [])
    write_json(
        rag_root / "idx.json",
        {
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar99",
            "outputs": {"chunks": "chunks.jsonl", "manifest_refs": "manifest_refs.jsonl", "files": "files.jsonl"},
        },
    )
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def _rag_current_with_recovery_plan_schema_mention(tmp_path: Path) -> Path:
    recovery_plan = {
        "schema": "audio_quality_verified_corpus_recovery_plan.v1",
        "manifest_type": "audio_quality_verified_corpus_recovery_plan",
        "recovery_status": "blocked_waiting_for_verified_corpus_material",
        "required_artifacts": ["verified_audio_training_corpus_manifest.v1"],
        "delivery_allowed": False,
        "release_ready": False,
    }
    rag_root = tmp_path / "artifact_rag" / "ar99"
    rag_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        rag_root / "manifest_refs.jsonl",
        [
            {
                "root": "aqr/q001",
                "root_alias": "r001",
                "relative_path": "manifest.json",
                "chunk_id": "recovery_manifest_ref",
                "text": json.dumps(recovery_plan, sort_keys=True),
                "status_fields": {
                    "schema": "audio_quality_verified_corpus_recovery_plan.v1",
                    "manifest_type": "audio_quality_verified_corpus_recovery_plan",
                    "delivery_allowed": False,
                    "release_ready": False,
                },
            }
        ],
    )
    write_jsonl(
        rag_root / "chunks.jsonl",
        [
            {
                "root": "aqr/q001",
                "root_alias": "r001",
                "relative_path": "summary.md",
                "chunk_id": "recovery_summary_chunk",
                "text": "required artifact: verified_audio_training_corpus_manifest.v1",
                "status_fields": {
                    "schema": "audio_quality_verified_corpus_recovery_plan.v1",
                    "manifest_type": "audio_quality_verified_corpus_recovery_plan",
                    "delivery_allowed": False,
                    "release_ready": False,
                },
            },
            {
                "root": "aqvreq/v001",
                "root_alias": "r002",
                "relative_path": "requests.jsonl",
                "chunk_id": "request_row_chunk",
                "text": json.dumps(
                    {
                        "schema": "audio_quality_verified_material_request_row.v1",
                        "raw_reference_path": "/tmp/generated/00_raw_ref.flac",
                        "delivery_allowed": False,
                        "release_ready": False,
                    },
                    sort_keys=True,
                ),
                "status_fields": {
                    "sample_rows": [
                        {
                            "schema": "audio_quality_verified_material_request_row.v1",
                            "delivery_allowed": False,
                            "release_ready": False,
                        }
                    ]
                },
            },
            {
                "root": "r001",
                "root_alias": "r003",
                "relative_path": "judges.jsonl",
                "chunk_id": "pathless_raw_reference_text",
                "text": "raw_reference_not_verified_ground_truth",
                "status_fields": {
                    "sample_rows": [
                        {
                            "delivery_allowed": False,
                            "release_ready": False,
                        }
                    ]
                },
            }
        ],
    )
    write_jsonl(rag_root / "files.jsonl", [])
    write_json(
        rag_root / "idx.json",
        {
            "index_schema": "artifact_rag_index.v2",
            "rag_id": "ar99",
            "outputs": {"chunks": "chunks.jsonl", "manifest_refs": "manifest_refs.jsonl", "files": "files.jsonl"},
        },
    )
    return write_json(
        tmp_path / "artifact_rag" / "current.json",
        {
            "schema": "artifact_rag_current.v1",
            "rag_id": "ar99",
            "idx": "ar99/idx.json",
            "query_hint": "Use artifact_rag/current.json -> idx.json",
        },
    )


def test_verified_corpus_recovery_plan_blocks_unverified_raw_reference_material(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_corpus_recovery_plan import build_audio_quality_verified_corpus_recovery_plan

    current = _rag_current_with_unverified_raw_reference(tmp_path)

    manifest = build_audio_quality_verified_corpus_recovery_plan(
        rag_current_json=current,
        output_dir=tmp_path / "artifact_rag" / "aqr" / "q001",
    )
    rows = read_jsonl(tmp_path / "artifact_rag" / "aqr" / "q001" / "rows.jsonl")

    assert manifest["schema"] == "audio_quality_verified_corpus_recovery_plan.v1"
    assert manifest["recovery_status"] == "blocked_waiting_for_verified_corpus_material"
    assert manifest["verified_corpus_ref_count"] == 0
    assert manifest["raw_reference_manifest_count"] == 1
    assert manifest["raw_reference_existing_audio_count"] == 1
    assert manifest["raw_reference_unverified_count"] == 1
    assert "verified_audio_training_corpus_manifest.v1" in manifest["required_artifacts"]
    assert "full_pack_reference_corpus_gap_plan.v1" in manifest["required_artifacts"]
    assert rows[0]["raw_reference_verified"] is False
    assert "raw_reference_not_verified_ground_truth" in rows[0]["blockers"]
    assert "raw_reference_from_separator_model_output" in rows[0]["blockers"]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_verified_corpus_scans_ignore_recovery_plan_schema_mentions(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_corpus_recovery_plan import build_audio_quality_verified_corpus_recovery_plan
    from tools.discover_audio_quality_verified_corpora import discover_audio_quality_verified_corpora

    current = _rag_current_with_recovery_plan_schema_mention(tmp_path)

    discovery = discover_audio_quality_verified_corpora(
        rag_current_json=current,
        source_lock_pack_json=_aqc_source_lock(tmp_path),
        output_dir=tmp_path / "aqc" / "vc001",
    )
    recovery = build_audio_quality_verified_corpus_recovery_plan(
        rag_current_json=current,
        output_dir=tmp_path / "artifact_rag" / "aqr" / "q002",
    )

    assert discovery["build_status"] == "blocked"
    assert discovery["verified_corpus_ref_count"] == 0
    assert "verified_corpus_manifest_not_found_in_current_rag" in discovery["errors"]
    assert recovery["recovery_status"] == "blocked_no_verified_or_raw_reference_material"
    assert recovery["verified_corpus_ref_count"] == 0
    assert recovery["raw_reference_manifest_count"] == 0
    assert recovery["delivery_allowed"] is False
    assert recovery["release_ready"] is False


def test_verified_material_request_packet_keeps_raw_reference_out_of_training(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_corpus_recovery_plan import build_audio_quality_verified_corpus_recovery_plan
    from tools.build_audio_quality_verified_material_request_packet import build_audio_quality_verified_material_request_packet

    current = _rag_current_with_unverified_raw_reference(tmp_path)
    recovery_manifest = build_audio_quality_verified_corpus_recovery_plan(
        rag_current_json=current,
        output_dir=tmp_path / "artifact_rag" / "aqr" / "q001",
    )

    manifest = build_audio_quality_verified_material_request_packet(
        recovery_manifest_json=Path(recovery_manifest["outputs"]["manifest_json"]),
        output_dir=tmp_path / "artifact_rag" / "aqvreq" / "v001",
    )
    requests = read_jsonl(tmp_path / "artifact_rag" / "aqvreq" / "v001" / "requests.jsonl")
    decisions = read_jsonl(tmp_path / "artifact_rag" / "aqvreq" / "v001" / "decisions.jsonl")

    assert manifest["schema"] == "audio_quality_verified_material_request_packet.v1"
    assert manifest["request_status"] == "verification_required"
    assert manifest["request_count"] == 1
    assert manifest["training_started"] is False
    assert manifest["ground_truth_manifest_written"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert requests[0]["schema"] == "audio_quality_verified_material_request_row.v1"
    assert requests[0]["safe_for_training"] is False
    assert requests[0]["verified_outcome"] is False
    assert requests[0]["raw_prediction_only"] is True
    assert requests[0]["must_not_mark_generated_stem_ground_truth"] is True
    assert "raw_reference_not_verified_ground_truth" in requests[0]["source_blockers"]
    assert decisions[0]["schema"] == "audio_quality_verified_material_decision_template.v1"
    assert decisions[0]["request_id"] == requests[0]["request_id"]
    assert decisions[0]["allowed_outcome_types"] == ["human_verified", "reference_verified", "reject_quality", "exclude"]


def test_verified_material_request_packet_blocks_missing_recovery_rows(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_material_request_packet import build_audio_quality_verified_material_request_packet

    recovery_manifest_json = write_json(
        tmp_path / "artifact_rag" / "aqr" / "q001" / "manifest.json",
        {
            "schema": "audio_quality_verified_corpus_recovery_plan.v1",
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "outputs": {},
        },
    )

    manifest = build_audio_quality_verified_material_request_packet(
        recovery_manifest_json=recovery_manifest_json,
        output_dir=tmp_path / "artifact_rag" / "aqvreq" / "v001",
    )
    blocked = read_jsonl(tmp_path / "artifact_rag" / "aqvreq" / "v001" / "blocked.jsonl")

    assert manifest["request_status"] == "blocked"
    assert manifest["request_count"] == 0
    assert manifest["blocked_row_count"] == 1
    assert manifest["errors"] == ["recovery_rows_jsonl_missing"]
    assert blocked[0]["blocked_reasons"] == ["recovery_rows_jsonl_missing"]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["ground_truth_manifest_written"] is False


def test_verified_corpus_builds_tensor_backed_aqc_rows_for_training_and_holdout(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    human_row = _verified_example_row(tmp_path, "human-b", split="holdout", stem_type="instrumental", reference_verified=False)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row, human_row], [_verified_report_row(ref_row), _verified_report_row(human_row)])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )
    rows = read_jsonl(tmp_path / "aqc" / "vr001" / "rows.jsonl")
    dataset_manifest = build_audio_quality_critic_dataset(
        source_jsonls=[tmp_path / "aqc" / "vr001" / "rows.jsonl"],
        output_dir=tmp_path / "aqc" / "vr001" / "ds",
        rag_current_json=current,
    )

    assert manifest["schema"] == "audio_quality_verified_rows_manifest.v1"
    assert manifest["row_count"] == 2
    assert manifest["tensor_row_count"] == 2
    assert manifest["feature_only_row_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert {row["label_type"] for row in rows} == {"reference_verified", "human_verified"}
    assert {row["split"] for row in rows} == {"train", "locked_holdout"}
    assert {row["stem_type"] for row in rows} == {"vocal", "minus"}
    assert all(Path(row["tensor_path"]).name in {"q001.npz", "q002.npz"} for row in rows)
    assert all("track-" not in row["tensor_path"] for row in rows)
    assert all(row["source_manifest_refs"] for row in rows)
    assert all(row["rag_refs"] and row["rag_refs"][0]["rag_id"] == "ar99" for row in rows)
    assert all(row["rag_refs"][0]["chunk_id"] == "verified_manifest_chunk" for row in rows)
    assert all(row["rag_refs"][0]["sha256"] == "verified-manifest-sha" for row in rows)
    assert all(row["source_lock_pack_json"] == str(source_lock) for row in rows)
    assert all(row["generated_stem_claims_ground_truth"] is False for row in rows)
    assert all(row["pseudo_teacher_targets_are_ground_truth"] is False for row in rows)
    assert dataset_manifest["safe_training_row_count"] == 2
    assert dataset_manifest["promotion_eligible_row_count"] == 2
    assert dataset_manifest["delivery_allowed"] is False


def test_verified_corpus_row_intake_blocks_failed_validation_without_rows(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)], status="failed")
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )

    assert manifest["build_status"] == "blocked"
    assert "corpus_validation_not_passed" in manifest["errors"]
    assert manifest["row_count"] == 0
    assert read_jsonl(tmp_path / "aqc" / "vr001" / "rows.jsonl") == []
    assert read_jsonl(tmp_path / "aqc" / "vr001" / "blocked.jsonl")[0]["blocked_reasons"] == ["corpus_validation_not_passed"]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_verified_corpus_row_intake_hard_fails_generated_ground_truth_claims(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = {
        **_verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True),
        "generated_stem_claims_ground_truth": True,
    }
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    with pytest.raises(ValueError, match="generated_stem_claims_ground_truth"):
        build_audio_quality_verified_rows(
            verified_corpus_manifest_json=manifest_json,
            output_dir=tmp_path / "aqc" / "vr001",
            rag_current_json=current,
            source_lock_pack_json=source_lock,
            window_sec=0.2,
            tensor_bins=8,
        )


def test_verified_corpus_row_intake_blocks_missing_audio_paths_before_tensor_extraction(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = {
        **_verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True),
        "source_audio_path": "",
    }
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )

    assert manifest["build_status"] == "blocked"
    blocked = read_jsonl(tmp_path / "aqc" / "vr001" / "blocked.jsonl")
    assert "source_audio_not_found" in blocked[0]["blocked_reasons"]
    assert all(not reason.startswith("tensor_extract_failed") for reason in blocked[0]["blocked_reasons"])


def test_verified_corpus_row_intake_blocks_mismatched_safe_row_report(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    mismatched_report = {**_verified_report_row(ref_row), "target_audio_path": str(tmp_path / "other.flac")}
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [mismatched_report])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )

    assert manifest["build_status"] == "blocked"
    blocked = read_jsonl(tmp_path / "aqc" / "vr001" / "blocked.jsonl")
    assert "verified_row_report_target_audio_path_mismatch" in blocked[0]["blocked_reasons"]
    assert read_jsonl(tmp_path / "aqc" / "vr001" / "rows.jsonl") == []


def test_verified_corpus_row_intake_requires_source_lock_to_match_current_rag(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path, current_rag_id="old-rag")
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )

    assert manifest["build_status"] == "blocked"
    assert "source_lock_current_rag_mismatch" in manifest["errors"]
    assert read_jsonl(tmp_path / "aqc" / "vr001" / "rows.jsonl") == []
    assert "source_lock_current_rag_mismatch" in read_jsonl(tmp_path / "aqc" / "vr001" / "blocked.jsonl")[0]["blocked_reasons"]


def test_verified_corpus_row_intake_rejects_wrong_source_lock_schema(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path, schema="not_source_lock.v1")
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )

    assert manifest["build_status"] == "blocked"
    assert "source_lock_schema_unexpected" in manifest["errors"]
    assert read_jsonl(tmp_path / "aqc" / "vr001" / "rows.jsonl") == []


def test_verified_corpus_row_intake_requires_chunk_backed_rag_reference(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=rag_current(tmp_path),
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )

    assert manifest["build_status"] == "blocked"
    assert "verified_corpus_rag_ref_missing" in manifest["errors"]
    assert read_jsonl(tmp_path / "aqc" / "vr001" / "rows.jsonl") == []
    assert "verified_corpus_rag_ref_missing" in read_jsonl(tmp_path / "aqc" / "vr001" / "blocked.jsonl")[0]["blocked_reasons"]


def test_verified_corpus_row_intake_rejects_source_lock_without_existing_idx(tmp_path: Path) -> None:
    from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows

    source_lock = _aqc_source_lock(tmp_path, rag_idx_exists=False)
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=manifest_json,
        output_dir=tmp_path / "aqc" / "vr001",
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        window_sec=0.2,
        tensor_bins=8,
    )

    assert manifest["build_status"] == "blocked"
    assert "source_lock_rag_idx_exists_not_true" in manifest["errors"]
    assert read_jsonl(tmp_path / "aqc" / "vr001" / "rows.jsonl") == []


def test_verified_corpus_discovery_finds_ready_current_rag_manifest(tmp_path: Path) -> None:
    from tools.discover_audio_quality_verified_corpora import discover_audio_quality_verified_corpora

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = discover_audio_quality_verified_corpora(
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        output_dir=tmp_path / "aqc" / "vc001",
    )
    rows = read_jsonl(tmp_path / "aqc" / "vc001" / "rows.jsonl")

    assert manifest["schema"] == "audio_quality_verified_corpus_discovery_manifest.v1"
    assert manifest["build_status"] == "passed"
    assert manifest["ready_count"] == 1
    assert manifest["blocked_ref_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert rows[0]["ready_for_verified_row_intake"] is True
    assert rows[0]["verified_corpus_manifest_json"] == str(manifest_json)
    assert rows[0]["source_lock_pack_json"] == str(source_lock)
    assert rows[0]["rag_refs"][0]["chunk_id"] == "verified_manifest_chunk"
    assert rows[0]["rag_refs"][0]["sha256"] == "verified-manifest-sha"
    assert rows[0]["selected_action"] == "run_build_audio_quality_verified_rows"
    assert rows[0]["delivery_allowed"] is False


def test_verified_corpus_discovery_finds_chunk_only_current_rag_manifest(tmp_path: Path) -> None:
    from tools.discover_audio_quality_verified_corpora import discover_audio_quality_verified_corpora

    source_lock = _aqc_source_lock(tmp_path)
    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    manifest_json = _verified_corpus_manifest(tmp_path, [ref_row], [_verified_report_row(ref_row)])
    current = rag_current_with_chunk_manifest_ref(tmp_path, manifest_json)

    manifest = discover_audio_quality_verified_corpora(
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        output_dir=tmp_path / "aqc" / "vc001",
    )
    rows = read_jsonl(tmp_path / "aqc" / "vc001" / "rows.jsonl")
    progress = [row["event"] for row in read_jsonl(tmp_path / "aqc" / "vc001" / "progress.jsonl")]

    assert manifest["build_status"] == "passed"
    assert manifest["verified_corpus_ref_count"] == 1
    assert manifest["ready_count"] == 1
    assert manifest["scanned_manifest_ref_count"] == 0
    assert manifest["scanned_chunk_count"] == 1
    assert manifest["scanned_file_count"] == 1
    assert manifest["rag_search_order"] == ["current", "idx", "manifest_refs", "chunks", "files"]
    assert "scan_rag_tables" in progress
    assert rows[0]["rag_refs"][0]["chunk_id"] == "chunk_only_manifest"
    assert rows[0]["rag_refs"][0]["sha256"] == "chunk-only-sha"
    assert rows[0]["rag_refs"][0]["source_tables"] == ["chunks", "files"]


def test_verified_corpus_discovery_blocks_when_current_rag_has_no_verified_manifest(tmp_path: Path) -> None:
    from tools.discover_audio_quality_verified_corpora import discover_audio_quality_verified_corpora

    source_lock = _aqc_source_lock(tmp_path)

    manifest = discover_audio_quality_verified_corpora(
        rag_current_json=rag_current(tmp_path),
        source_lock_pack_json=source_lock,
        output_dir=tmp_path / "aqc" / "vc001",
    )

    assert manifest["build_status"] == "blocked"
    assert manifest["ready_count"] == 0
    assert manifest["verified_corpus_ref_count"] == 0
    assert "verified_corpus_manifest_not_found_in_current_rag" in manifest["errors"]
    assert read_jsonl(tmp_path / "aqc" / "vc001" / "rows.jsonl") == []
    assert read_jsonl(tmp_path / "aqc" / "vc001" / "blocked.jsonl")[0]["blocked_reasons"] == [
        "verified_corpus_manifest_not_found_in_current_rag"
    ]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_verified_corpus_discovery_blocks_failed_or_unavailable_corpus_refs(tmp_path: Path) -> None:
    from tools.discover_audio_quality_verified_corpora import discover_audio_quality_verified_corpora

    source_lock = _aqc_source_lock(tmp_path)
    failed_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    failed_manifest = _verified_corpus_manifest(tmp_path, [failed_row], [_verified_report_row(failed_row)], status="failed")
    current = rag_current_with_manifest_ref(tmp_path, failed_manifest)

    manifest = discover_audio_quality_verified_corpora(
        rag_current_json=current,
        source_lock_pack_json=source_lock,
        output_dir=tmp_path / "aqc" / "vc001",
    )
    blocked = read_jsonl(tmp_path / "aqc" / "vc001" / "blocked.jsonl")

    assert manifest["build_status"] == "blocked"
    assert manifest["ready_count"] == 0
    assert manifest["blocked_ref_count"] == 1
    assert "no_ready_verified_corpus_refs" in manifest["errors"]
    assert "corpus_validation_not_passed" in blocked[0]["blocked_reasons"]
    assert read_jsonl(tmp_path / "aqc" / "vc001" / "rows.jsonl") == []


def test_scoring_blocks_proxy_critic_disagreement_from_delivery(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
    from tools.score_audio_quality_windows import score_audio_quality_windows
    from tools.train_audio_quality_critic import train_audio_quality_critic

    rows = [
        dataset_row(tmp_path, "verified-pos", label_type="reference_verified", label=1),
        dataset_row(tmp_path, "verified-neg", label_type="reference_verified", label=0),
    ]
    build_audio_quality_critic_dataset(
        source_jsonls=[write_jsonl(tmp_path / "train.jsonl", rows)],
        output_dir=tmp_path / "dataset",
        rag_current_json=rag_current(tmp_path),
    )
    train_audio_quality_critic(dataset_dir=tmp_path / "dataset", output_dir=tmp_path / "model", device="cpu", epochs=2)
    candidates = write_jsonl(
        tmp_path / "candidates.jsonl",
        [{**dataset_row(tmp_path, "runtime-neg", label_type="rejected_blocked", label=0), "proxy_decision": "accept_candidate"}],
    )

    manifest = score_audio_quality_windows(
        candidate_jsonl=candidates,
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "scores",
        accept_threshold=0.99,
    )
    scored = read_jsonl(tmp_path / "scores" / "scores.jsonl")

    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "audio_quality_critic_shadow_scores_not_for_listen_pack"
    assert manifest["disagreement_count"] == 1
    assert scored[0]["proxy_critic_disagreement"] is True
    assert scored[0]["selected_action"] == "block_for_review"
    assert scored[0]["delivery_allowed"] is False


def test_learning_cycle_uses_only_verified_outcomes_and_writes_review_queue(tmp_path: Path) -> None:
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    outcomes_root = tmp_path / "runs"
    run_dir = outcomes_root / "r001"
    write_jsonl(
        run_dir / "audio_quality_learning_outcome.jsonl",
        [
            {**dataset_row(tmp_path, "verified", label_type="human_verified", label=1), "verified_outcome": True},
            {**dataset_row(tmp_path, "raw", label_type="weak_proxy", label=1), "raw_prediction_only": True},
        ],
    )

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=outcomes_root,
        output_dir=tmp_path / "cycle",
        rag_current_json=rag_current(tmp_path),
    )

    assert manifest["verified_examples_total"] == 1
    assert manifest["raw_prediction_rows_excluded"] == 1
    assert manifest["promotion_approved"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    review_rows = read_jsonl(tmp_path / "cycle" / "review.jsonl")
    assert review_rows[0]["review_reason"] == "raw_prediction_only"


def test_learning_cycle_writes_promotion_discovery_and_retains_it_in_current_rag(tmp_path: Path) -> None:
    from tools.query_artifact_rag import query_artifact_rag
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    outcomes_root = tmp_path / "runs"
    run_dir = outcomes_root / "r001"
    write_jsonl(
        run_dir / "audio_quality_learning_outcome.jsonl",
        [{**dataset_row(tmp_path, "verified", label_type="human_verified", label=1), "verified_outcome": True}],
    )
    current = rag_current(tmp_path)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=outcomes_root,
        output_dir=tmp_path / "cycle",
        rag_current_json=current,
        promotion_discovery_output_dir=tmp_path / "artifact_rag" / "aqp" / "p001",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar02",
        metadata_overlay_base_rag_id="ar99",
    )

    discovery_manifest = read_json(tmp_path / "artifact_rag" / "aqp" / "p001" / "manifest.json")
    overlay_manifest = read_json(tmp_path / "artifact_rag" / "ar02" / "idx.json")
    current_after = read_json(current)
    query = query_artifact_rag(
        rag_root=tmp_path / "artifact_rag",
        query="audio_quality_promotion_candidate_discovery",
        limit=5,
    )

    assert manifest["promotion_discovery_manifest_json"] == str(tmp_path / "artifact_rag" / "aqp" / "p001" / "manifest.json")
    assert manifest["promotion_discovery_status"] == "blocked"
    assert manifest["promotion_candidate_count"] == 0
    assert manifest["promotion_discovery_retained_in_rag"] is True
    assert manifest["metadata_overlay_manifest_json"] == str(tmp_path / "artifact_rag" / "ar02" / "idx.json")
    assert manifest["metadata_overlay_rag_id"] == "ar02"
    assert manifest["promotion_approved"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert discovery_manifest["promotion_allowed"] is False
    assert discovery_manifest["runtime_authority_enabled"] is False
    assert discovery_manifest["delivery_allowed"] is False
    assert overlay_manifest["overlay_safety"]["metadata_roots_delivery_allowed"] is False
    assert overlay_manifest["summary"]["overlay_root_count"] == 1
    assert current_after["idx"] == "ar02/idx.json"
    assert query["result_count"] >= 1
    assert query["matches"][0]["status_fields"]["delivery_allowed"] is False


def test_learning_cycle_retains_cycle_manifest_when_output_is_artifact_rag_root(tmp_path: Path) -> None:
    from tools.query_artifact_rag import query_artifact_rag
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    outcomes_root = tmp_path / "runs"
    run_dir = outcomes_root / "r001"
    write_jsonl(
        run_dir / "audio_quality_learning_outcome.jsonl",
        [{**dataset_row(tmp_path, "verified", label_type="human_verified", label=1), "verified_outcome": True}],
    )
    current = rag_current(tmp_path)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=outcomes_root,
        output_dir=tmp_path / "artifact_rag" / "aqc" / "r012",
        rag_current_json=current,
        promotion_discovery_output_dir=tmp_path / "artifact_rag" / "aqp" / "p002",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar02",
    )
    overlay_manifest = read_json(tmp_path / "artifact_rag" / "ar02" / "idx.json")
    overlay_chunks = read_jsonl(tmp_path / "artifact_rag" / "ar02" / "chunks.jsonl")
    progress_rows = read_jsonl(tmp_path / "artifact_rag" / "aqc" / "r012" / "progress.jsonl")
    cycle_query = query_artifact_rag(
        rag_root=tmp_path / "artifact_rag",
        query="audio_quality_learning_cycle",
        limit=5,
    )
    discovery_query = query_artifact_rag(
        rag_root=tmp_path / "artifact_rag",
        query="audio_quality_promotion_candidate_discovery",
        limit=5,
    )

    assert manifest["learning_cycle_retained_in_rag"] is True
    assert manifest["metadata_overlay_base_rag_id"] == "ar99"
    assert overlay_manifest["summary"]["overlay_root_count"] == 2
    assert any(
        row.get("root") == "aqc/r012" and '"metadata_overlay_base_rag_id": "ar99"' in str(row.get("text") or "")
        for row in overlay_chunks
    )
    assert any(
        row.get("event") == "retain_learning_cycle_metadata_in_rag"
        and row.get("metadata_overlay_base_rag_id") == "ar99"
        for row in progress_rows
    )
    assert cycle_query["result_count"] >= 1
    assert discovery_query["result_count"] >= 1
    assert all(match["status_fields"].get("delivery_allowed") is not True for match in cycle_query["matches"])


def test_learning_cycle_writes_training_validation_audit_handoff_without_running_it(tmp_path: Path) -> None:
    from tools.query_artifact_rag import query_artifact_rag
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    outcomes_root = tmp_path / "runs"
    run_dir = outcomes_root / "r001"
    write_jsonl(
        run_dir / "audio_quality_learning_outcome.jsonl",
        [
            {**dataset_row(tmp_path, "verified-pos", label_type="human_verified", label=1), "verified_outcome": True},
            {**dataset_row(tmp_path, "verified-neg", label_type="human_verified", label=0), "verified_outcome": True},
        ],
    )
    current = rag_current(tmp_path)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=outcomes_root,
        output_dir=tmp_path / "artifact_rag" / "aqc" / "r020",
        rag_current_json=current,
        training_handoff_output_dir=tmp_path / "artifact_rag" / "aqh" / "h001",
        training_model_output_dir=tmp_path / "artifact_rag" / "aqm" / "m001",
        validation_output_json=tmp_path / "artifact_rag" / "aqv" / "v001" / "validation.json",
        audit_packet_output_dir=tmp_path / "artifact_rag" / "aqa" / "a001",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar02",
    )
    handoff = read_json(tmp_path / "artifact_rag" / "aqh" / "h001" / "manifest.json")
    progress_rows = read_jsonl(tmp_path / "artifact_rag" / "aqc" / "r020" / "progress.jsonl")
    current_after = read_json(current)
    query = query_artifact_rag(
        rag_root=tmp_path / "artifact_rag",
        query="audio_quality_learning_handoff",
        limit=5,
    )

    assert manifest["training_handoff_manifest_json"] == str(tmp_path / "artifact_rag" / "aqh" / "h001" / "manifest.json")
    assert manifest["training_handoff_status"] == "ready_for_training"
    assert manifest["training_handoff_retained_in_rag"] is True
    assert manifest["promotion_discovery_retained_in_rag"] is False
    assert manifest["training_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["promotion_approved"] is False
    assert handoff["schema"] == "audio_quality_learning_handoff.v1"
    assert handoff["handoff_status"] == "ready_for_training"
    assert handoff["ready_for_training"] is True
    assert handoff["training_started"] is False
    assert handoff["models_changed"] is False
    assert handoff["delivery_allowed"] is False
    assert handoff["release_ready"] is False
    assert handoff["train_command"][:2] == ["python", "/app/tools/train_audio_quality_critic.py"]
    assert "--device" in handoff["train_command"]
    assert "cuda" in handoff["train_command"]
    assert "/app/tools/validate_audio_quality_critic.py" in handoff["validation_command"]
    assert "/app/tools/build_audio_quality_experiment_audit_packet.py" in handoff["audit_packet_command"]
    assert handoff["ready_for_audit_packet"] is False
    assert "model_manifest_pending" in handoff["waiting_reasons"]
    assert progress_rows[-1]["promotion_discovery_retained_in_rag"] is False
    assert current_after["idx"] == "ar02/idx.json"
    assert query["result_count"] >= 1
    assert query["matches"][0]["status_fields"]["delivery_allowed"] is False


def test_learning_cycle_builds_source_lock_and_blocked_verified_discovery(tmp_path: Path) -> None:
    from tools.query_artifact_rag import query_artifact_rag
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    current = rag_current(tmp_path)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=tmp_path / "runs",
        output_dir=tmp_path / "artifact_rag" / "aqc" / "r021",
        rag_current_json=current,
        source_lock_output_dir=tmp_path / "artifact_rag" / "plan" / "p002",
        verified_discovery_output_dir=tmp_path / "artifact_rag" / "aqd" / "d001",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar03",
        code_index_status="indexed_ready",
        code_index_fallback="CodeGraph unavailable in unit test",
        docker_service="ml-worker",
        gpu_status="unit_test_gpu_visible",
    )
    source_lock = read_json(tmp_path / "artifact_rag" / "plan" / "p002" / "pack.json")
    discovery = read_json(tmp_path / "artifact_rag" / "aqd" / "d001" / "manifest.json")
    current_after = read_json(current)
    query = query_artifact_rag(
        rag_root=tmp_path / "artifact_rag",
        query="audio_quality_verified_corpus_discovery verified_corpus_manifest_not_found_in_current_rag",
        limit=5,
    )

    assert source_lock["schema"] == "audio_quality_source_lock.v1"
    assert source_lock["current_rag_id"] == "ar99"
    assert discovery["build_status"] == "blocked"
    assert discovery["ready_count"] == 0
    assert "verified_corpus_manifest_not_found_in_current_rag" in discovery["errors"]
    assert manifest["source_lock_pack_json"] == str(tmp_path / "artifact_rag" / "plan" / "p002" / "pack.json")
    assert manifest["source_lock_retained_in_rag"] is True
    assert manifest["verified_discovery_status"] == "blocked"
    assert manifest["verified_discovery_ready_count"] == 0
    assert manifest["verified_discovery_retained_in_rag"] is True
    assert manifest["dataset_manifest_json"] == ""
    assert manifest["training_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert current_after["idx"] == "ar03/idx.json"
    assert query["result_count"] >= 1
    assert query["matches"][0]["status_fields"]["delivery_allowed"] is False


def test_learning_cycle_retains_verified_corpus_recovery_plan_when_discovery_blocks(tmp_path: Path) -> None:
    from tools.query_artifact_rag import query_artifact_rag
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    current = _rag_current_with_unverified_raw_reference(tmp_path)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=tmp_path / "runs",
        output_dir=tmp_path / "artifact_rag" / "aqc" / "r024",
        rag_current_json=current,
        source_lock_output_dir=tmp_path / "artifact_rag" / "plan" / "p005",
        verified_discovery_output_dir=tmp_path / "artifact_rag" / "aqd" / "d005",
        verified_recovery_output_dir=tmp_path / "artifact_rag" / "aqr" / "q002",
        training_handoff_output_dir=tmp_path / "artifact_rag" / "aqh" / "h006",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar05",
    )
    recovery = read_json(tmp_path / "artifact_rag" / "aqr" / "q002" / "manifest.json")
    current_after = read_json(current)
    query = query_artifact_rag(
        rag_root=tmp_path / "artifact_rag",
        query="audio_quality_verified_corpus_recovery_plan raw_reference_not_verified_ground_truth",
        limit=5,
    )

    assert manifest["verified_discovery_status"] == "blocked"
    assert manifest["verified_recovery_status"] == "blocked_waiting_for_verified_corpus_material"
    assert manifest["verified_recovery_raw_reference_manifest_count"] == 1
    assert manifest["verified_recovery_retained_in_rag"] is True
    assert recovery["raw_reference_unverified_count"] == 1
    assert recovery["training_started"] is False
    assert recovery["delivery_allowed"] is False
    assert recovery["release_ready"] is False
    assert manifest["dataset_manifest_json"] == ""
    assert manifest["training_handoff_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert current_after["idx"] == "ar05/idx.json"
    assert query["result_count"] >= 1
    assert query["matches"][0]["status_fields"]["delivery_allowed"] is False


def test_learning_cycle_retains_verified_material_request_packet_when_recovery_blocks(tmp_path: Path) -> None:
    from tools.query_artifact_rag import query_artifact_rag
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    current = _rag_current_with_unverified_raw_reference(tmp_path)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=tmp_path / "runs",
        output_dir=tmp_path / "artifact_rag" / "aqc" / "r025",
        rag_current_json=current,
        source_lock_output_dir=tmp_path / "artifact_rag" / "plan" / "p006",
        verified_discovery_output_dir=tmp_path / "artifact_rag" / "aqd" / "d006",
        verified_recovery_output_dir=tmp_path / "artifact_rag" / "aqr" / "q003",
        verified_material_request_output_dir=tmp_path / "artifact_rag" / "aqvreq" / "v002",
        training_handoff_output_dir=tmp_path / "artifact_rag" / "aqh" / "h007",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar06",
    )
    request_manifest = read_json(tmp_path / "artifact_rag" / "aqvreq" / "v002" / "manifest.json")
    request_rows = read_jsonl(tmp_path / "artifact_rag" / "aqvreq" / "v002" / "requests.jsonl")
    current_after = read_json(current)
    query = query_artifact_rag(
        rag_root=tmp_path / "artifact_rag",
        query="audio_quality_verified_material_request_packet raw_reference_not_verified_ground_truth",
        limit=5,
    )

    assert manifest["verified_material_request_status"] == "verification_required"
    assert manifest["verified_material_request_count"] == 1
    assert manifest["verified_material_request_retained_in_rag"] is True
    assert request_manifest["request_status"] == "verification_required"
    assert request_manifest["training_started"] is False
    assert request_manifest["delivery_allowed"] is False
    assert request_manifest["release_ready"] is False
    assert request_rows[0]["safe_for_training"] is False
    assert manifest["dataset_manifest_json"] == ""
    assert manifest["training_handoff_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert current_after["idx"] == "ar06/idx.json"
    assert query["result_count"] >= 1
    assert query["matches"][0]["status_fields"]["delivery_allowed"] is False


def test_learning_cycle_builds_dataset_from_ready_verified_corpus_discovery(tmp_path: Path) -> None:
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    ref_row = _verified_example_row(tmp_path, "ref-a", split="train", stem_type="vocal", reference_verified=True)
    human_row = _verified_example_row(tmp_path, "human-b", split="train", stem_type="instrumental", reference_verified=False)
    manifest_json = _verified_corpus_manifest(
        tmp_path,
        [ref_row, human_row],
        [_verified_report_row(ref_row), _verified_report_row(human_row)],
    )
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=tmp_path / "runs",
        output_dir=tmp_path / "artifact_rag" / "aqc" / "r022",
        rag_current_json=current,
        source_lock_output_dir=tmp_path / "artifact_rag" / "plan" / "p003",
        verified_discovery_output_dir=tmp_path / "artifact_rag" / "aqd" / "d003",
        training_handoff_output_dir=tmp_path / "artifact_rag" / "aqh" / "h003",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar03",
    )
    dataset_manifest = read_json(tmp_path / "artifact_rag" / "aqc" / "r022" / "dataset" / "manifest.json")
    handoff = read_json(tmp_path / "artifact_rag" / "aqh" / "h003" / "manifest.json")
    verified_rows_manifest = read_json(tmp_path / "artifact_rag" / "aqc" / "r022" / "vrows" / "vr001" / "manifest.json")
    dataset_rows = read_jsonl(tmp_path / "artifact_rag" / "aqc" / "r022" / "dataset" / "dataset.jsonl")

    assert manifest["verified_discovery_status"] == "passed"
    assert manifest["verified_discovery_ready_count"] == 1
    assert manifest["verified_row_intake_status"] == "passed"
    assert manifest["verified_row_intake_row_count"] == 2
    assert manifest["verified_row_intake_training_row_count"] == 2
    assert manifest["verified_row_intake_blocked_row_count"] == 0
    assert manifest["verified_row_intake_retained_in_rag"] is True
    assert manifest["dataset_manifest_json"] == str(tmp_path / "artifact_rag" / "aqc" / "r022" / "dataset" / "manifest.json")
    assert dataset_manifest["row_count"] == 2
    assert dataset_manifest["safe_training_row_count"] == 2
    assert dataset_manifest["promotion_eligible_row_count"] == 2
    assert dataset_manifest["source_jsonls"] == [str(tmp_path / "artifact_rag" / "aqc" / "r022" / "vrows" / "vr001" / "rows.jsonl")]
    assert verified_rows_manifest["build_status"] == "passed"
    assert all(row["label_type"] in {"reference_verified", "human_verified"} for row in dataset_rows)
    assert all(row["generated_stem_claims_ground_truth"] is False for row in read_jsonl(tmp_path / "artifact_rag" / "aqc" / "r022" / "vrows" / "vr001" / "rows.jsonl"))
    assert handoff["handoff_status"] == "ready_for_training"
    assert manifest["training_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_learning_cycle_excludes_partial_blocked_verified_row_intake_from_training_total(tmp_path: Path) -> None:
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    safe_row = _verified_example_row(tmp_path, "safe-a", split="train", stem_type="vocal", reference_verified=True)
    unsafe_row = _verified_example_row(tmp_path, "unsafe-b", split="train", stem_type="instrumental", reference_verified=False)
    manifest_json = _verified_corpus_manifest(
        tmp_path,
        [safe_row, unsafe_row],
        [_verified_report_row(safe_row), _verified_report_row(unsafe_row, safe=False)],
    )
    current = rag_current_with_manifest_ref(tmp_path, manifest_json)

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=tmp_path / "runs",
        output_dir=tmp_path / "artifact_rag" / "aqc" / "r023",
        rag_current_json=current,
        source_lock_output_dir=tmp_path / "artifact_rag" / "plan" / "p004",
        verified_discovery_output_dir=tmp_path / "artifact_rag" / "aqd" / "d004",
        training_handoff_output_dir=tmp_path / "artifact_rag" / "aqh" / "h004",
        metadata_overlay_output_dir=tmp_path / "artifact_rag" / "ar04",
    )
    intake = read_json(tmp_path / "artifact_rag" / "aqc" / "r023" / "vrows" / "manifest.json")
    handoff = read_json(tmp_path / "artifact_rag" / "aqh" / "h004" / "manifest.json")

    assert manifest["verified_discovery_status"] == "passed"
    assert manifest["verified_row_intake_status"] == "blocked"
    assert manifest["verified_row_intake_row_count"] == 1
    assert manifest["verified_row_intake_training_row_count"] == 0
    assert manifest["verified_row_intake_blocked_row_count"] == 1
    assert manifest["verified_row_intake_source_jsonls"] == []
    assert manifest["verified_training_rows_total"] == 0
    assert manifest["dataset_manifest_json"] == ""
    assert intake["build_status"] == "blocked"
    assert intake["source_jsonls"] == []
    assert handoff["handoff_status"] == "blocked"
    assert "dataset_manifest_missing" in handoff["blocked_reasons"]
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_learning_cycle_blocks_training_handoff_without_verified_dataset(tmp_path: Path) -> None:
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=tmp_path / "runs",
        output_dir=tmp_path / "cycle",
        rag_current_json=rag_current(tmp_path),
        training_handoff_output_dir=tmp_path / "artifact_rag" / "aqh" / "h002",
    )
    handoff = read_json(tmp_path / "artifact_rag" / "aqh" / "h002" / "manifest.json")

    assert manifest["training_handoff_status"] == "blocked"
    assert manifest["training_handoff_retained_in_rag"] is False
    assert manifest["training_started"] is False
    assert manifest["models_changed"] is False
    assert handoff["handoff_status"] == "blocked"
    assert handoff["ready_for_training"] is False
    assert "dataset_manifest_missing" in handoff["blocked_reasons"]
    assert handoff["train_command"] == []
    assert handoff["delivery_allowed"] is False
    assert handoff["release_ready"] is False


def test_learning_cycle_ingests_verified_review_decisions_before_dataset(tmp_path: Path) -> None:
    from tools.run_audio_quality_learning_cycle import run_audio_quality_learning_cycle

    source_lock = _aqc_source_lock(tmp_path)
    review_jsonl = write_jsonl(
        tmp_path / "review" / "review.jsonl",
        [
            {
                **dataset_row(tmp_path, "cycle-decision", label_type="weak_proxy", label=1),
                "schema": "audio_quality_review_queue_row.v1",
                "queue_id": "q001",
                "row_ref": "row-ref-cycle-decision",
                "raw_prediction_only": True,
                "verified_outcome": False,
                "safe_for_training": False,
                "source_lock_pack_json": str(source_lock),
            }
        ],
    )
    review_manifest = write_json(
        tmp_path / "review" / "manifest.json",
        {
            "schema": "audio_quality_review_queue_manifest.v1",
            "manifest_type": "audio_quality_review_queue",
            "queue_status": "review_required",
            "review_queue_count": 1,
            "source_lock_pack_json": str(source_lock),
            "outputs": {"review_jsonl": str(review_jsonl)},
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "thresholds_relaxed": False,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
        },
    )
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [
            {
                "schema": "audio_quality_review_decision.v1",
                "queue_id": "q001",
                "decision": "accept_quality",
                "outcome_type": "human_verified",
                "reviewer_id": "human-a",
                "split": "train",
            }
        ],
    )

    manifest = run_audio_quality_learning_cycle(
        outcomes_root=tmp_path / "runs",
        output_dir=tmp_path / "cycle",
        rag_current_json=rag_current(tmp_path),
        review_queue_manifest_json=review_manifest,
        review_decisions_jsonl=decisions,
    )

    outcomes = read_jsonl(tmp_path / "cycle" / "decision_ingest" / "audio_quality_learning_outcome.jsonl")
    dataset_manifest = json.loads((tmp_path / "cycle" / "dataset" / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["verified_examples_total"] == 1
    assert manifest["review_decision_verified_outcome_count"] == 1
    assert manifest["review_decision_blocked_count"] == 0
    assert manifest["review_decision_ingest_manifest_json"] == str(tmp_path / "cycle" / "decision_ingest" / "manifest.json")
    assert dataset_manifest["safe_training_row_count"] == 1
    assert dataset_manifest["promotion_eligible_row_count"] == 1
    assert outcomes[0]["label_type"] == "human_verified"
    assert outcomes[0]["verified_outcome"] is True
    assert outcomes[0]["raw_prediction_only"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_review_queue_collects_shadow_disagreement_and_uncertainty_without_training_claims(tmp_path: Path) -> None:
    from tools.build_audio_quality_review_queue import build_audio_quality_review_queue

    source_lock = _aqc_source_lock(tmp_path)
    scores_jsonl = write_jsonl(
        tmp_path / "score" / "scores.jsonl",
        [
            {
                **dataset_row(tmp_path, "disagree", label_type="weak_proxy", label=1),
                "row_id": "row-disagree",
                "accept_probability": 0.2,
                "uncertainty": 0.2,
                "proxy_decision": "accept_candidate",
                "proxy_critic_disagreement": True,
                "selected_action": "block_for_review",
                "source_lock_pack_json": str(source_lock),
            },
            {
                **dataset_row(tmp_path, "uncertain", label_type="rejected_blocked", label=0),
                "row_id": "row-uncertain",
                "accept_probability": 0.49,
                "uncertainty": 0.91,
                "proxy_decision": "blocked_hard_case",
                "proxy_critic_disagreement": False,
                "selected_action": "shadow_block_observation",
                "source_lock_pack_json": str(source_lock),
            },
            {
                **dataset_row(tmp_path, "clean", label_type="weak_proxy", label=1),
                "row_id": "row-clean",
                "accept_probability": 0.9,
                "uncertainty": 0.05,
                "proxy_decision": "accept_candidate",
                "proxy_critic_disagreement": False,
                "selected_action": "shadow_accept_observation",
                "source_lock_pack_json": str(source_lock),
            },
        ],
    )
    score_manifest = write_json(
        tmp_path / "score" / "manifest.json",
        {
            "schema": "audio_quality_critic_score_manifest.v1",
            "manifest_type": "audio_quality_critic_score",
            "scores_jsonl": str(scores_jsonl),
            "source_lock_pack_json": str(source_lock),
            "model_id": "aqc_fixture",
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "thresholds_relaxed": False,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
        },
    )

    manifest = build_audio_quality_review_queue(
        score_manifest_json=score_manifest,
        output_dir=tmp_path / "review",
        uncertainty_threshold=0.8,
    )
    review_rows = read_jsonl(tmp_path / "review" / "review.jsonl")

    assert manifest["schema"] == "audio_quality_review_queue_manifest.v1"
    assert manifest["queue_status"] == "review_required"
    assert manifest["review_queue_count"] == 2
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert [row["queue_id"] for row in review_rows] == ["q001", "q002"]
    assert review_rows[0]["review_reasons"] == ["proxy_critic_disagreement"]
    assert review_rows[1]["review_reasons"] == ["critic_uncertainty_high"]
    assert review_rows[0]["tensor_path"]
    assert review_rows[0]["route_id"] == "route-disagree"
    assert review_rows[0]["window_start_sec"] == 0.0
    assert review_rows[0]["window_end_sec"] == 6.0
    assert isinstance(review_rows[0]["targets"], dict)
    assert all(row["raw_prediction_only"] is True for row in review_rows)
    assert all(row["verified_outcome"] is False for row in review_rows)
    assert all(row["requested_outcome_types"] == ["human_verified", "reference_verified"] for row in review_rows)
    assert all(row["delivery_allowed"] is False and row["release_ready"] is False for row in review_rows)


def test_review_queue_blocks_contaminated_score_manifest(tmp_path: Path) -> None:
    from tools.build_audio_quality_review_queue import build_audio_quality_review_queue

    scores_jsonl = write_jsonl(tmp_path / "score" / "scores.jsonl", [{**dataset_row(tmp_path, "unsafe", label_type="weak_proxy", label=1)}])
    score_manifest = write_json(
        tmp_path / "score" / "manifest.json",
        {
            "schema": "audio_quality_critic_score_manifest.v1",
            "scores_jsonl": str(scores_jsonl),
            "source_lock_pack_json": "",
            "delivery_allowed": True,
            "release_ready": False,
        },
    )

    manifest = build_audio_quality_review_queue(
        score_manifest_json=score_manifest,
        output_dir=tmp_path / "review",
    )

    assert manifest["queue_status"] == "blocked"
    assert "score_manifest_delivery_allowed_true" in manifest["errors"]
    assert "score_manifest_source_lock_missing" in manifest["errors"]
    assert read_jsonl(tmp_path / "review" / "review.jsonl") == []
    assert read_jsonl(tmp_path / "review" / "blocked.jsonl")[0]["blocked_reasons"] == sorted(manifest["errors"])


def test_review_decision_ingest_writes_verified_learning_outcomes(tmp_path: Path) -> None:
    from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
    from tools.ingest_audio_quality_review_decisions import ingest_audio_quality_review_decisions

    source_lock = _aqc_source_lock(tmp_path)
    review_jsonl = write_jsonl(
        tmp_path / "review" / "review.jsonl",
        [
            {
                **dataset_row(tmp_path, "accepted", label_type="weak_proxy", label=1),
                "schema": "audio_quality_review_queue_row.v1",
                "queue_id": "q001",
                "row_ref": "row-ref-accepted",
                "raw_prediction_only": True,
                "verified_outcome": False,
                "safe_for_training": False,
                "source_lock_pack_json": str(source_lock),
            },
            {
                **dataset_row(tmp_path, "rejected", label_type="rejected_blocked", label=0),
                "schema": "audio_quality_review_queue_row.v1",
                "queue_id": "q002",
                "row_ref": "row-ref-rejected",
                "raw_prediction_only": True,
                "verified_outcome": False,
                "safe_for_training": False,
                "source_lock_pack_json": str(source_lock),
            },
            {
                **dataset_row(tmp_path, "pending", label_type="weak_proxy", label=1),
                "schema": "audio_quality_review_queue_row.v1",
                "queue_id": "q003",
                "row_ref": "row-ref-pending",
                "raw_prediction_only": True,
                "verified_outcome": False,
                "safe_for_training": False,
                "source_lock_pack_json": str(source_lock),
            },
        ],
    )
    review_manifest = write_json(
        tmp_path / "review" / "manifest.json",
        {
            "schema": "audio_quality_review_queue_manifest.v1",
            "manifest_type": "audio_quality_review_queue",
            "queue_status": "review_required",
            "review_queue_count": 3,
            "source_lock_pack_json": str(source_lock),
            "outputs": {"review_jsonl": str(review_jsonl)},
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "thresholds_relaxed": False,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
        },
    )
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [
            {
                "schema": "audio_quality_review_decision.v1",
                "queue_id": "q001",
                "decision": "accept_quality",
                "outcome_type": "human_verified",
                "reviewer_id": "human-a",
                "review_notes": "clean enough",
            },
            {
                "schema": "audio_quality_review_decision.v1",
                "queue_id": "q002",
                "decision": "reject_quality",
                "outcome_type": "reference_verified",
                "reference_evidence_ref": "ref-pack-001",
                "review_notes": "leakage confirmed",
            },
            {
                "schema": "audio_quality_review_decision.v1",
                "queue_id": "q003",
                "decision": "needs_more_evidence",
                "review_notes": "needs reference",
            },
        ],
    )

    manifest = ingest_audio_quality_review_decisions(
        review_queue_manifest_json=review_manifest,
        decisions_jsonl=decisions,
        output_dir=tmp_path / "ingest",
    )
    outcomes = read_jsonl(tmp_path / "ingest" / "audio_quality_learning_outcome.jsonl")
    blocked = read_jsonl(tmp_path / "ingest" / "blocked.jsonl")
    dataset_manifest = build_audio_quality_critic_dataset(
        source_jsonls=[tmp_path / "ingest" / "audio_quality_learning_outcome.jsonl"],
        output_dir=tmp_path / "ingest" / "dataset",
        rag_current_json=rag_current(tmp_path),
    )

    assert manifest["schema"] == "audio_quality_review_decision_ingest_manifest.v1"
    assert manifest["ingest_status"] == "passed"
    assert manifest["verified_outcome_count"] == 2
    assert manifest["blocked_decision_count"] == 1
    assert {row["label_type"] for row in outcomes} == {"human_verified", "reference_verified"}
    assert {row["label"] for row in outcomes} == {0, 1}
    assert all(row["verified_outcome"] is True for row in outcomes)
    assert all(row["raw_prediction_only"] is False for row in outcomes)
    assert all(row["source_review_queue_manifest_json"] == str(review_manifest) for row in outcomes)
    assert all(row["source_review_decisions_jsonl"] == str(decisions) for row in outcomes)
    assert blocked[0]["blocked_reasons"] == ["needs_more_evidence"]
    assert dataset_manifest["safe_training_row_count"] == 2
    assert dataset_manifest["promotion_eligible_row_count"] == 2
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_review_decision_ingest_blocks_contaminated_manifest_and_unknown_decision(tmp_path: Path) -> None:
    from tools.ingest_audio_quality_review_decisions import ingest_audio_quality_review_decisions

    review_jsonl = write_jsonl(
        tmp_path / "review" / "review.jsonl",
        [
            {
                **dataset_row(tmp_path, "unsafe", label_type="weak_proxy", label=1),
                "schema": "audio_quality_review_queue_row.v1",
                "queue_id": "q001",
            }
        ],
    )
    review_manifest = write_json(
        tmp_path / "review" / "manifest.json",
        {
            "schema": "audio_quality_review_queue_manifest.v1",
            "outputs": {"review_jsonl": str(review_jsonl)},
            "source_lock_pack_json": "",
            "delivery_allowed": True,
            "release_ready": False,
        },
    )
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [{"schema": "audio_quality_review_decision.v1", "queue_id": "unknown", "decision": "accept_quality", "outcome_type": "human_verified"}],
    )

    manifest = ingest_audio_quality_review_decisions(
        review_queue_manifest_json=review_manifest,
        decisions_jsonl=decisions,
        output_dir=tmp_path / "ingest",
    )

    assert manifest["ingest_status"] == "blocked"
    assert "review_queue_manifest_delivery_allowed_true" in manifest["errors"]
    assert "review_queue_manifest_source_lock_missing" in manifest["errors"]
    assert read_jsonl(tmp_path / "ingest" / "audio_quality_learning_outcome.jsonl") == []
    assert any("unknown_queue_id" in row["blocked_reasons"] for row in read_jsonl(tmp_path / "ingest" / "blocked.jsonl"))


def test_review_decision_ingest_records_pending_rows_when_decisions_are_missing(tmp_path: Path) -> None:
    from tools.ingest_audio_quality_review_decisions import ingest_audio_quality_review_decisions

    source_lock = _aqc_source_lock(tmp_path)
    review_jsonl = write_jsonl(
        tmp_path / "review" / "review.jsonl",
        [
            {
                **dataset_row(tmp_path, "pending", label_type="weak_proxy", label=1),
                "schema": "audio_quality_review_queue_row.v1",
                "queue_id": "q001",
                "raw_prediction_only": True,
                "verified_outcome": False,
                "source_lock_pack_json": str(source_lock),
            }
        ],
    )
    review_manifest = write_json(
        tmp_path / "review" / "manifest.json",
        {
            "schema": "audio_quality_review_queue_manifest.v1",
            "outputs": {"review_jsonl": str(review_jsonl)},
            "source_lock_pack_json": str(source_lock),
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )

    manifest = ingest_audio_quality_review_decisions(
        review_queue_manifest_json=review_manifest,
        decisions_jsonl=None,
        output_dir=tmp_path / "ingest",
    )

    assert manifest["ingest_status"] == "passed"
    assert manifest["verified_outcome_count"] == 0
    assert manifest["blocked_decision_count"] == 1
    assert read_jsonl(tmp_path / "ingest" / "audio_quality_learning_outcome.jsonl") == []
    assert read_jsonl(tmp_path / "ingest" / "blocked.jsonl")[0]["blocked_reasons"] == ["missing_review_decision"]
