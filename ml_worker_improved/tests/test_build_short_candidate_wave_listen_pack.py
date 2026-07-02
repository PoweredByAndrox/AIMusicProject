from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_short_candidate_wave_listen_pack import build_short_candidate_wave_listen_pack


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _audio(path: Path, value: float) -> Path:
    sample_rate = 8000
    data = np.full((sample_rate // 20, 2), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)
    return path


def _candidate_pool(
    tmp_path: Path,
    *,
    release_ready: bool = False,
    missing_first_audio: bool = False,
    missing_second_audio: bool = False,
) -> Path:
    root = tmp_path / "cw"
    minus_1 = root / "jobs" / "long_route_one" / "minus.flac"
    if not missing_first_audio:
        _audio(minus_1, 0.2)
    minus_2 = root / "jobs" / "long_route_two" / "minus.flac"
    if not missing_second_audio:
        _audio(minus_2, 0.3)
    rows = [
        {
            "track_id": "very_long_raw_track_title_must_stay_in_manifest_only",
            "candidate_id": "both_long_candidate_hash_11111111111111111111111111111111",
            "candidate_family": "artifact_safe_default",
            "route_id": "local_registered__htdemucs_6s__both__shared",
            "mix_path": "/app/input/song.flac",
            "vocal_path": str(root / "jobs" / "long_route_one" / "vocals.flac"),
            "minus_path": str(minus_1),
            "instrumental_path": str(minus_1),
            "is_complete_pair": True,
            "release_ready": release_ready,
            "can_publish_as_result": False,
        },
        {
            "track_id": "very_long_raw_track_title_must_stay_in_manifest_only",
            "candidate_id": "pair_long_candidate_hash_22222222222222222222222222222222",
            "candidate_family": "balanced_pair_default",
            "route_id": "local_registered__htdemucs_ft__both__shared",
            "mix_path": "/app/input/song.flac",
            "vocal_path": str(root / "jobs" / "long_route_two" / "vocals.flac"),
            "minus_path": str(minus_2),
            "instrumental_path": str(minus_2),
            "is_complete_pair": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    ]
    return _write_jsonl(root / "candidate_pool.jsonl", rows)


def _judges(tmp_path: Path) -> tuple[Path, Path]:
    result_rows = [
        {
            "candidate_id": "both_long_candidate_hash_11111111111111111111111111111111",
            "decision": "accept_candidate",
            "minus_decision": "accept_candidate",
            "minus_score": 0.91,
            "vocal_leakage_in_minus_score": 0.16,
        },
        {
            "candidate_id": "pair_long_candidate_hash_22222222222222222222222222222222",
            "decision": "accept_candidate",
            "minus_decision": "accept_candidate",
            "minus_score": 0.86,
            "vocal_leakage_in_minus_score": 0.24,
        },
    ]
    windowed_rows = [
        {
            "candidate_id": "both_long_candidate_hash_11111111111111111111111111111111",
            "decision": "accept_candidate",
            "minus_decision": "accept_candidate",
            "minus_score": 0.82,
            "vocal_leakage_in_minus_score": 0.49,
            "mixture_consistency_score": 0.92,
            "reasons": ["no_reference_proxy_passed"],
        },
        {
            "candidate_id": "pair_long_candidate_hash_22222222222222222222222222222222",
            "decision": "blocked_hard_case",
            "minus_decision": "blocked_hard_case",
            "minus_score": 0.89,
            "vocal_leakage_in_minus_score": 0.30,
            "mixture_consistency_score": 0.55,
            "reasons": ["severe_vocal_bleed_in_minus"],
        },
    ]
    return (
        _write_jsonl(tmp_path / "rj.jsonl", result_rows),
        _write_jsonl(tmp_path / "wj.jsonl", windowed_rows),
    )


def _warning_comparison(tmp_path: Path, *, improved_count: int, regressed_count: int = 0) -> Path:
    path = tmp_path / "cmp" / "cmp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "minus_middle_comparison_report.v1",
                "comparison_status": "completed",
                "improved_count": improved_count,
                "regressed_count": regressed_count,
                "failed_checks": [],
                "release_ready": False,
                "can_publish_as_result": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_build_short_candidate_wave_listen_pack_selects_judge_accepted_minus(tmp_path: Path) -> None:
    pool = _candidate_pool(tmp_path)
    result_judge, windowed_judge = _judges(tmp_path)

    manifest = build_short_candidate_wave_listen_pack(
        candidate_pool_jsonl=pool,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "out",
        max_minus_candidates=4,
    )

    assert manifest["schema"] == "short_candidate_wave_listen_pack_manifest.v1"
    assert manifest["selection_status"] == "selected_for_diagnostic_ab"
    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_only_not_for_listen_pack"
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["listenable"] is False
    assert "requires_staged_aggregate_machine_gate_for_listen_delivery" in manifest["delivery_blockers"]
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["selected_count"] == 1
    assert manifest["blocked_candidate_count"] == 1
    row = manifest["listen_audio_files"][0]
    assert row["artifact_relative_path"] == "audio/min/m001.flac"
    assert row["stem_type"] == "minus"
    assert row["selection_status"] == "selected_for_diagnostic_ab"
    assert row["delivery_status"] == "blocked"
    assert row["delivery_allowed"] is False
    assert row["presentation_status"] == "diagnostic_only_not_for_listen_pack"
    assert row["release_status"] == "not_release_ready"
    assert row["listenable"] is False
    assert row["source_candidate_id"] == "both_long_candidate_hash_11111111111111111111111111111111"
    assert "long_candidate_hash" not in row["artifact_relative_path"]
    assert "very_long_raw_track" in json.dumps(row["source_row"], sort_keys=True)
    assert "accepted_for_listen_pack" not in json.dumps(manifest, sort_keys=True)
    assert (tmp_path / "out" / "audio" / "min" / "m001.flac").exists()


def test_build_short_candidate_wave_listen_pack_blocks_warning_audit_without_improvement(tmp_path: Path) -> None:
    pool = _candidate_pool(tmp_path)
    result_judge, windowed_judge = _judges(tmp_path)

    manifest = build_short_candidate_wave_listen_pack(
        candidate_pool_jsonl=pool,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        minus_warning_comparison_json=_warning_comparison(tmp_path, improved_count=0, regressed_count=1),
        output_dir=tmp_path / "out",
        max_minus_candidates=4,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["minus_warning_comparison_json"].endswith("cmp.json")
    assert "minus_warning_comparison_no_improved_windows" in manifest["failed_checks"]
    assert "minus_warning_comparison_regressed_windows" in manifest["failed_checks"]
    assert manifest["selected_count"] == 0
    assert not (tmp_path / "out" / "audio").exists()


def test_build_short_candidate_wave_listen_pack_blocks_unsafe_pool_row(tmp_path: Path) -> None:
    pool = _candidate_pool(tmp_path, release_ready=True)
    result_judge, windowed_judge = _judges(tmp_path)

    manifest = build_short_candidate_wave_listen_pack(
        candidate_pool_jsonl=pool,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "out",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "candidate_pool_row_release_ready_true:both_long_candidate_hash_11111111111111111111111111111111" in manifest["failed_checks"]
    assert manifest["selected_count"] == 0
    assert not (tmp_path / "out" / "audio").exists()


def test_build_short_candidate_wave_listen_pack_fails_closed_on_missing_selected_audio(tmp_path: Path) -> None:
    pool = _candidate_pool(tmp_path, missing_first_audio=True)
    result_judge, windowed_judge = _judges(tmp_path)

    manifest = build_short_candidate_wave_listen_pack(
        candidate_pool_jsonl=pool,
        minus_result_judge_jsonl=result_judge,
        minus_windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "out",
        max_minus_candidates=4,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "selected_audio_missing:both_long_candidate_hash_11111111111111111111111111111111" in manifest["failed_checks"]
    assert manifest["selected_count"] == 0
    assert not (tmp_path / "out" / "audio").exists()
