from __future__ import annotations

import json
from pathlib import Path

from tools.artifact_status_contract import status_view_from_manifest
from tools.build_candidate_family_blend_listen_pack import build_candidate_family_blend_listen_pack


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(path: Path, *, train_ready: bool = True, leakage: int = 0) -> Path:
    return _write_json(
        path,
        {
            "schema": "train_ready_audio_model_decision_manifest.v1",
            "decision": "train_result_verified" if train_ready else "train_result_not_ready",
            "train_ready": train_ready,
            "success_rate": 1.0 if train_ready else 0.0,
            "failed_checks": [] if train_ready else ["not_ready"],
            "critical_leakage_count": leakage,
            "release_ready": False,
            "delivery_allowed": False,
            "thresholds_relaxed": False,
            "runtime_gates_changed": False,
            "model_registry_edited": False,
        },
    )


def _holdout_rows(path: Path, audio_path: Path) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-flac-bytes")
    return _write_jsonl(
        path,
        [
            {
                "schema": "candidate_family_blend_holdout_row.v1",
                "status": "passed",
                "failed_checks": [],
                "config_id": "v32_scnet_melband_blend_step005_with_a_needlessly_long_identifier",
                "stem": "vocal",
                "holdout_track_id": "Square a Saw - Oxygen / Tevlo Vocals Resynthesized Long Name",
                "holdout_group_id": "group-a",
                "checkpoint_training_track_ids": ["track-b"],
                "families": ["scnet", "melband"],
                "weights": {"scnet": 0.4, "melband": 0.6},
                "baseline_policy": "best_family",
                "baseline_family": "melband",
                "metrics": {"candidate_sdr_db": 17.0, "baseline_sdr_db": 16.0, "improvement_db": 1.0},
                "candidate_audio_path": str(audio_path),
                "release_ready": False,
                "delivery_allowed": False,
            }
        ],
    )


def test_listen_pack_accepts_train_ready_holdout_audio(tmp_path: Path) -> None:
    decision = _decision(tmp_path / "decision.json")
    holdout = _holdout_rows(tmp_path / "holdout.jsonl", tmp_path / "source" / "candidate.flac")

    manifest = build_candidate_family_blend_listen_pack(
        train_ready_decision_manifest_json=decision,
        holdout_rows_jsonls=[holdout],
        output_dir=tmp_path / "listen_pack",
        stem="vocal",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["listen_audio_count"] == 1
    listen_path = Path(manifest["listen_audio_files"][0]["listen_audio_path"])
    assert listen_path.exists()
    assert listen_path.parent.name == "audio"
    assert listen_path.name == "v001.flac"
    assert len(listen_path.name) <= 16
    assert "Square" not in listen_path.name
    assert "v32_scnet" not in listen_path.name
    assert manifest["listen_audio_files"][0]["artifact_alias"] == "v001"
    assert manifest["listen_audio_files"][0]["source_holdout_track_id"] == "Square a Saw - Oxygen / Tevlo Vocals Resynthesized Long Name"

    manifest_path = Path(manifest["output_dir"]) / "pack.json"
    view = status_view_from_manifest(manifest_path)
    assert view.listenable is True
    assert view.deliverable is False


def test_listen_pack_blocks_non_train_ready_decision(tmp_path: Path) -> None:
    decision = _decision(tmp_path / "decision.json", train_ready=False)
    holdout = _holdout_rows(tmp_path / "holdout.jsonl", tmp_path / "source" / "candidate.flac")

    manifest = build_candidate_family_blend_listen_pack(
        train_ready_decision_manifest_json=decision,
        holdout_rows_jsonls=[holdout],
        output_dir=tmp_path / "listen_pack",
        stem="vocal",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "train_ready_decision_not_verified" in manifest["failed_checks"]
    assert manifest["listen_audio_count"] == 0
    assert not (tmp_path / "listen_pack" / "listen_audio").exists()


def test_listen_pack_blocks_holdout_track_leakage(tmp_path: Path) -> None:
    decision = _decision(tmp_path / "decision.json")
    holdout = _holdout_rows(tmp_path / "holdout.jsonl", tmp_path / "source" / "candidate.flac")
    row = json.loads(holdout.read_text(encoding="utf-8").splitlines()[0])
    row["checkpoint_training_track_ids"] = ["Square a Saw - Oxygen / Tevlo Vocals Resynthesized Long Name"]
    _write_jsonl(holdout, [row])

    manifest = build_candidate_family_blend_listen_pack(
        train_ready_decision_manifest_json=decision,
        holdout_rows_jsonls=[holdout],
        output_dir=tmp_path / "listen_pack",
        stem="vocal",
    )

    assert manifest["delivery_status"] == "blocked"
    assert "holdout_track_leakage" in manifest["failed_checks"]
    assert manifest["delivery_allowed"] is False
