from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_short_listen_repair_selection_pack import build_short_listen_repair_selection_pack


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _audio(path: Path, value: float) -> Path:
    sample_rate = 8000
    data = np.full((sample_rate // 20, 2), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)
    return path


def _candidate_pack(tmp_path: Path, *, release_ready: bool = False) -> Path:
    root = tmp_path / "repair_candidates"
    vocal_good = _audio(root / "audio" / "q001" / "v001.flac", 0.2)
    vocal_poor = _audio(root / "audio" / "q001" / "v002.flac", 0.4)
    vocal_blocked = _audio(root / "audio" / "q001" / "v003.flac", 0.9)
    minus_1 = _audio(root / "audio" / "q002" / "m001.flac", 0.1)
    minus_2 = _audio(root / "audio" / "q002" / "m002.flac", 0.2)
    minus_blocked = _audio(root / "audio" / "q002" / "m003.flac", 0.3)
    rows = [
        {
            "schema": "short_listen_repair_candidate.v1",
            "queue_item_id": "q001",
            "route_intent": "vocal_candidate_replacement",
            "stem_type": "vocal",
            "artifact_relative_path": "audio/q001/v001.flac",
            "listen_audio_path": str(vocal_good),
            "listen_audio_sha256": "v-good",
            "source_track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)",
            "reference_metrics": {"reference_metric_status": "computed", "reference_sdr_db": 8.4},
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_listen_repair_candidate.v1",
            "queue_item_id": "q001",
            "route_intent": "vocal_candidate_replacement",
            "stem_type": "vocal",
            "artifact_relative_path": "audio/q001/v003.flac",
            "listen_audio_path": str(vocal_blocked),
            "listen_audio_sha256": "v-blocked",
            "source_track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)",
            "reference_metrics": {"reference_metric_status": "computed", "reference_sdr_db": 99.0},
            "delivery_status": "blocked",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_listen_repair_candidate.v1",
            "queue_item_id": "q001",
            "route_intent": "vocal_candidate_replacement",
            "stem_type": "vocal",
            "artifact_relative_path": "audio/q001/v002.flac",
            "listen_audio_path": str(vocal_poor),
            "listen_audio_sha256": "v-poor",
            "source_track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)",
            "reference_metrics": {"reference_metric_status": "computed", "reference_sdr_db": 4.1},
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_listen_repair_candidate.v1",
            "queue_item_id": "q002",
            "route_intent": "minus_repair",
            "stem_type": "minus",
            "artifact_relative_path": "audio/q002/m001.flac",
            "listen_audio_path": str(minus_1),
            "listen_audio_sha256": "m1",
            "candidate_id": "q002_c001",
            "minus_candidate_id": "c001",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_listen_repair_candidate.v1",
            "queue_item_id": "q002",
            "route_intent": "minus_repair",
            "stem_type": "minus",
            "artifact_relative_path": "audio/q002/m002.flac",
            "listen_audio_path": str(minus_2),
            "listen_audio_sha256": "m2",
            "candidate_id": "q002_c002",
            "minus_candidate_id": "c002",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_listen_repair_candidate.v1",
            "queue_item_id": "q002",
            "route_intent": "minus_repair",
            "stem_type": "minus",
            "artifact_relative_path": "audio/q002/m003.flac",
            "listen_audio_path": str(minus_blocked),
            "listen_audio_sha256": "m3",
            "candidate_id": "q002_c003",
            "minus_candidate_id": "c003",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    ]
    selected_jsonl = _write_jsonl(root / "selected.jsonl", rows)
    return _write_json(
        root / "pack.json",
        {
            "schema": "short_listen_repair_candidate_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "candidate_count": len(rows),
            "failed_checks": [],
            "selected_jsonl": str(selected_jsonl),
        },
    )


def _minus_judges(tmp_path: Path) -> tuple[Path, Path]:
    result_rows = [
        {"candidate_id": "q002_c001", "decision": "accept_candidate", "minus_decision": "accept_candidate"},
        {"candidate_id": "q002_c002", "decision": "accept_candidate", "minus_decision": "accept_candidate"},
        {"candidate_id": "q002_c003", "decision": "blocked_hard_case", "minus_decision": "blocked_hard_case"},
    ]
    windowed_rows = [
        {
            "candidate_id": "q002_c001",
            "decision": "accept_candidate",
            "minus_decision": "accept_candidate",
            "minus_score": 0.82,
            "vocal_leakage_in_minus_score": 0.12,
            "mixture_consistency_score": 0.91,
            "reasons": [],
        },
        {
            "candidate_id": "q002_c002",
            "decision": "accept_candidate",
            "minus_decision": "accept_candidate",
            "minus_score": 0.86,
            "vocal_leakage_in_minus_score": 0.25,
            "mixture_consistency_score": 0.88,
            "reasons": ["phase_or_stereo_inconsistency"],
        },
        {
            "candidate_id": "q002_c003",
            "decision": "blocked_hard_case",
            "minus_decision": "blocked_hard_case",
            "minus_score": 0.9,
            "vocal_leakage_in_minus_score": 0.05,
            "mixture_consistency_score": 0.44,
            "reasons": ["minus_detail_gate_blocked"],
        },
    ]
    return (
        _write_jsonl(tmp_path / "minus_result_judge.jsonl", result_rows),
        _write_jsonl(tmp_path / "minus_windowed_judge.jsonl", windowed_rows),
    )


def test_build_short_listen_repair_selection_pack_selects_safe_ranked_candidates(tmp_path: Path) -> None:
    pack = _candidate_pack(tmp_path)
    result_judge, windowed_judge = _minus_judges(tmp_path)

    manifest = build_short_listen_repair_selection_pack(
        repair_candidate_pack_json=pack,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "selection",
        max_vocal_candidates=1,
        max_minus_candidates=2,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["selected_count"] == 3
    assert len(manifest["listen_audio_files"]) == 3
    assert manifest["quality_warning_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "selection" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["artifact_relative_path"] for row in rows] == [
        "audio/voc/v001.flac",
        "audio/min/m001.flac",
        "audio/min/m002.flac",
    ]
    assert "Square" not in "\n".join(row["artifact_relative_path"] for row in rows)
    assert rows[0]["source_artifact_relative_path"] == "audio/q001/v001.flac"
    assert rows[0]["reference_metrics"]["reference_sdr_db"] == 8.4
    assert rows[1]["source_candidate_id"] == "q002_c002"
    assert rows[2]["source_candidate_id"] == "q002_c001"
    assert rows[1]["quality_warning_reasons"] == ["phase_or_stereo_inconsistency"]
    assert not any(row.get("source_candidate_id") == "q002_c003" for row in rows)


def test_build_short_listen_repair_selection_pack_allows_minus_only_selection(tmp_path: Path) -> None:
    pack = _candidate_pack(tmp_path)
    result_judge, windowed_judge = _minus_judges(tmp_path)

    manifest = build_short_listen_repair_selection_pack(
        repair_candidate_pack_json=pack,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "selection",
        max_vocal_candidates=0,
        max_minus_candidates=1,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["failed_checks"] == []
    assert manifest["selected_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "selection" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["artifact_relative_path"] for row in rows] == ["audio/min/m001.flac"]
    assert rows[0]["route_intent"] == "minus_repair"
    assert rows[0]["source_candidate_id"] == "q002_c002"


def test_build_short_listen_repair_selection_pack_can_dedupe_minus_by_track_and_exclude_reasons(tmp_path: Path) -> None:
    pack = _candidate_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(payload["selected_jsonl"]).read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("route_intent") == "minus_repair":
            row["track_id"] = "trk001"
    _write_jsonl(Path(payload["selected_jsonl"]), rows)
    result_judge, windowed_judge = _minus_judges(tmp_path)

    manifest = build_short_listen_repair_selection_pack(
        repair_candidate_pack_json=pack,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "selection",
        max_vocal_candidates=0,
        max_minus_candidates=2,
        dedupe_minus_by_track=True,
        exclude_minus_reasons=["phase_or_stereo_inconsistency"],
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["failed_checks"] == []
    assert manifest["selected_count"] == 1
    assert manifest["minus_selection_policy"]["dedupe_minus_by_track"] is True
    assert manifest["minus_selection_policy"]["exclude_minus_reasons"] == ["phase_or_stereo_inconsistency"]
    rows = [json.loads(line) for line in (tmp_path / "selection" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_candidate_id"] == "q002_c001"
    assert rows[0]["minus_source_key"] == "track:trk001:minus"
    assert rows[0]["quality_warning_reasons"] == []


def test_build_short_listen_repair_selection_pack_can_include_specific_minus_source_key(tmp_path: Path) -> None:
    pack = _candidate_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(payload["selected_jsonl"]).read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("candidate_id") == "q002_c001":
            row["track_id"] = "trk-a"
        elif row.get("candidate_id") == "q002_c002":
            row["track_id"] = "trk-b"
    _write_jsonl(Path(payload["selected_jsonl"]), rows)
    result_judge, windowed_judge = _minus_judges(tmp_path)

    manifest = build_short_listen_repair_selection_pack(
        repair_candidate_pack_json=pack,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "selection",
        max_vocal_candidates=0,
        max_minus_candidates=2,
        include_minus_source_keys=["track:trk-a:minus"],
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["failed_checks"] == []
    assert manifest["selected_count"] == 1
    assert manifest["minus_selection_policy"]["include_minus_source_keys"] == ["track:trk-a:minus"]
    selected = [json.loads(line) for line in (tmp_path / "selection" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert selected[0]["source_candidate_id"] == "q002_c001"
    assert selected[0]["minus_source_key"] == "track:trk-a:minus"


def test_build_short_listen_repair_selection_pack_blocks_unsafe_source_pack(tmp_path: Path) -> None:
    pack = _candidate_pack(tmp_path, release_ready=True)
    result_judge, windowed_judge = _minus_judges(tmp_path)

    manifest = build_short_listen_repair_selection_pack(
        repair_candidate_pack_json=pack,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "selection",
    )

    assert manifest["delivery_status"] == "blocked"
    assert "repair_candidate_pack_release_ready_true" in manifest["failed_checks"]
    assert manifest["selected_count"] == 0
    assert manifest["delivery_allowed"] is False
