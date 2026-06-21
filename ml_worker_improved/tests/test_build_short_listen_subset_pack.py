from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_short_listen_subset_pack import build_short_listen_subset_pack


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _audio(path: Path, value: float) -> Path:
    sample_rate = 8000
    data = np.full((sample_rate // 20, 2), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)
    return path


def _source_pack(tmp_path: Path, *, release_ready: bool = False) -> Path:
    vocal = _audio(tmp_path / "src" / "v001.flac", 0.2)
    minus = _audio(tmp_path / "src" / "m001.flac", 0.3)
    return _write_json(
        tmp_path / "pack.json",
        {
            "schema": "source_pack.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "listen_audio_files": [
                {
                    "schema": "source_audio.v1",
                    "artifact_relative_path": "audio/voc/v001.flac",
                    "listen_audio_path": str(vocal),
                    "stem_type": "vocal",
                    "route_intent": "vocal_candidate_replacement",
                    "source_candidate_id": "candidate_conditioned__long_raw_id_should_not_be_in_path",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "schema": "source_audio.v1",
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(minus),
                    "stem_type": "minus",
                    "route_intent": "minus_repair",
                    "source_candidate_id": "q002_c009",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )


def _minus_audit(tmp_path: Path, *, release_ready: bool = False) -> Path:
    rows_path = tmp_path / "audit" / "rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "schema": "short_minus_quality_audit_row.v1",
            "short_audio_path": "audio/min/m001.flac",
            "audit_decision": "passed",
            "audit_blockers": [],
            "quality_warning_reasons": ["residual_vocal_watch"],
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_minus_quality_audit_row.v1",
            "short_audio_path": "audio/min/m002.flac",
            "audit_decision": "blocked",
            "audit_blockers": ["severe_vocal_bleed_in_minus"],
            "quality_warning_reasons": ["residual_vocal_watch"],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    ]
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return _write_json(
        tmp_path / "audit" / "audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed_with_blockers",
            "rows_jsonl": str(rows_path),
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "delivery_allowed": False,
        },
    )


def _source_pack_with_two_minus(tmp_path: Path) -> Path:
    vocal = _audio(tmp_path / "src" / "v001.flac", 0.2)
    minus_passed = _audio(tmp_path / "src" / "m001.flac", 0.3)
    minus_blocked = _audio(tmp_path / "src" / "m002.flac", 0.4)
    return _write_json(
        tmp_path / "pack.json",
        {
            "schema": "source_pack.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "listen_audio_files": [
                {
                    "schema": "source_audio.v1",
                    "artifact_relative_path": "audio/voc/v001.flac",
                    "listen_audio_path": str(vocal),
                    "stem_type": "vocal",
                    "route_intent": "vocal_candidate_replacement",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "schema": "source_audio.v1",
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(minus_passed),
                    "stem_type": "minus",
                    "route_intent": "minus_repair",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "schema": "source_audio.v1",
                    "artifact_relative_path": "audio/min/m002.flac",
                    "listen_audio_path": str(minus_blocked),
                    "stem_type": "minus",
                    "route_intent": "minus_repair",
                    "source_candidate_id": "long_raw_blocked_candidate_id_must_stay_in_manifest_only",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )


def test_build_short_listen_subset_pack_filters_by_stem_and_route(tmp_path: Path) -> None:
    manifest = build_short_listen_subset_pack(
        source_manifest_json=_source_pack(tmp_path),
        output_dir=tmp_path / "out",
        include_stems=["vocal"],
        include_route_intents=["vocal_candidate_replacement"],
        alias_prefix="v",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["selected_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "out" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["artifact_relative_path"] == "audio/sub/v001.flac"
    assert rows[0]["stem_type"] == "vocal"
    assert rows[0]["source_candidate_id"] == "candidate_conditioned__long_raw_id_should_not_be_in_path"
    assert "candidate_conditioned" not in rows[0]["artifact_relative_path"]


def test_build_short_listen_subset_pack_gates_minus_rows_by_audit(tmp_path: Path) -> None:
    manifest = build_short_listen_subset_pack(
        source_manifest_json=_source_pack_with_two_minus(tmp_path),
        output_dir=tmp_path / "out",
        alias_prefix="g",
        minus_audit_json=_minus_audit(tmp_path),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["selected_count"] == 2
    assert manifest["stem_type_counts"] == {"minus": 1, "vocal": 1}
    assert manifest["source_stem_type_counts"][""]["minus"] == 1
    assert manifest["minus_audit_gate"]["accepted_minus_count"] == 1
    assert manifest["minus_audit_gate"]["excluded_minus_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "out" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["stem_type"] for row in rows] == ["vocal", "minus"]
    assert rows[1]["source_artifact_relative_path"] == "audio/min/m001.flac"
    assert rows[1]["minus_audit_gate"]["audit_decision"] == "passed"
    assert rows[1]["quality_warning_reasons"] == ["residual_vocal_watch"]
    assert all("long_raw_blocked_candidate_id" not in row["artifact_relative_path"] for row in rows)

    excluded = [
        json.loads(line)
        for line in (tmp_path / "out" / "excluded.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert excluded[0]["source_artifact_relative_path"] == "audio/min/m002.flac"
    assert excluded[0]["audit_decision"] == "blocked"
    assert excluded[0]["audit_blockers"] == ["severe_vocal_bleed_in_minus"]
    assert excluded[0]["release_ready"] is False


def test_build_short_listen_subset_pack_blocks_release_ready_minus_audit(tmp_path: Path) -> None:
    manifest = build_short_listen_subset_pack(
        source_manifest_json=_source_pack_with_two_minus(tmp_path),
        output_dir=tmp_path / "out",
        alias_prefix="g",
        minus_audit_json=_minus_audit(tmp_path, release_ready=True),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "minus_audit_release_ready_true" in manifest["failed_checks"]
    assert manifest["selected_count"] == 0
    assert not (tmp_path / "out" / "audio").exists()


def test_build_short_listen_subset_pack_blocks_release_ready_source(tmp_path: Path) -> None:
    manifest = build_short_listen_subset_pack(
        source_manifest_json=_source_pack(tmp_path, release_ready=True),
        output_dir=tmp_path / "out",
        include_stems=["vocal"],
        alias_prefix="v",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "source_release_ready_true" in manifest["failed_checks"]
    assert manifest["selected_count"] == 0
