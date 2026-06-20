from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.apply_aris_top_route_no_reference_gate import apply_aris_top_route_no_reference_gate


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pool_inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "artifact_rag"
    mix = root / "audio" / "t001" / "mix.flac"
    vocal = root / "audio" / "t001" / "vocal.flac"
    minus = root / "audio" / "t001" / "minus.flac"
    for path in (mix, vocal, minus):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")
    vocal_manifest = _write_json(root / "meta" / "t001" / "vstem.json", {"raw_reference": {"path": str(mix)}})
    minus_manifest = _write_json(root / "meta" / "t001" / "mstem.json", {"raw_reference": {"path": str(mix)}})
    pool = _write_jsonl(
        root / "runs" / "r001" / "candidate_pool.jsonl",
        [
            {
                "schema": "aris_refinement_candidate_row.v1",
                "track_alias": "t001",
                "track_id": "track-1",
                "stem": "vocal",
                "family": "scnet",
                "candidate_alias": "v001",
                "candidate_status": "generated_pending_gate",
                "target_audio_path": str(vocal),
                "candidate_manifest_path": str(vocal_manifest),
                "binary_audio_available": True,
                "audio_bytes_available": True,
                "metadata_only": False,
                "delivery_allowed": False,
            },
            {
                "schema": "aris_refinement_candidate_row.v1",
                "track_alias": "t001",
                "track_id": "track-1",
                "stem": "minus",
                "family": "scnet",
                "candidate_alias": "m001",
                "candidate_status": "generated_pending_gate",
                "target_audio_path": str(minus),
                "candidate_manifest_path": str(minus_manifest),
                "binary_audio_available": True,
                "audio_bytes_available": True,
                "metadata_only": False,
                "delivery_allowed": False,
            },
        ],
    )
    return root, pool


def _accepting_judge(row: dict, *, score_mode: str) -> dict:
    base = {"track_id": row["track_id"], "candidate_id": row["candidate_id"], "failed_checks": [], "reasons": []}
    if score_mode == "vocal":
        return {**base, "vocal_decision": "accept_candidate", "vocal_blockers": []}
    if score_mode == "minus":
        return {
            **base,
            "minus_decision": "accept_candidate",
            "minus_blockers": [],
            "minus_musical_detail_preservation_gate": {"status": "proxy_pass", "blockers": []},
        }
    if score_mode == "both":
        return {**base, "pair_decision": "pair_compatible", "pair_blockers": []}
    raise AssertionError(score_mode)


def test_applies_accepting_no_reference_gate_without_delivery(tmp_path: Path) -> None:
    root, pool = _pool_inputs(tmp_path)

    manifest = apply_aris_top_route_no_reference_gate(
        artifact_rag_root=root,
        candidate_pool_jsonl=pool,
        output_dir=root / "gate" / "r001",
        judge_runner=_accepting_judge,
    )

    assert manifest["status"] == "accepted"
    assert manifest["accepted_pair_count"] == 1
    assert manifest["delivery_allowed"] is False
    rows = _read_jsonl(Path(manifest["gated_candidate_pool_jsonl"]))
    assert [row["candidate_status"] for row in rows] == ["passed", "passed"]
    assert all(row["no_reference_gate"] == "passed" for row in rows)
    assert all(row["pair_consistency_gate"] == "passed" for row in rows)
    assert all(row["delivery_allowed"] is False for row in rows)
    progress = _read_jsonl(Path(manifest["progress_jsonl"]))
    assert [row["event"] for row in progress] == ["run_start", "track_gate_start", "track_gate_finish", "run_finish"]
    assert all(row["gpu_inference_skipped"] is True for row in progress)
    assert all(row["expected_acceleration_mode"] == "cpu_dsp_scoring" for row in progress)


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    root, pool = _pool_inputs(tmp_path)

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        apply_aris_top_route_no_reference_gate(
            artifact_rag_root=root,
            candidate_pool_jsonl=pool,
            output_dir=tmp_path / "outside",
            judge_runner=_accepting_judge,
        )
