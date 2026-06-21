from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_fts_listen_pack import build_fts_listen_pack


def _audio(path: Path) -> Path:
    sample_rate = 8_000
    data = np.zeros((sample_rate, 2), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _fixtures(tmp_path: Path, *, regressed_count: int = 0) -> tuple[Path, Path, Path, Path]:
    audio = _audio(tmp_path / "fts" / "fts_min.wav")
    switch_manifest = _write_json(
        tmp_path / "fts" / "manifest.json",
        {
            "manifest_type": "no_reference_full_track_segment_switch_candidate",
            "render_status": "rendered_full_track_candidate",
            "rendered_minus_candidate_path": str(audio),
            "duration_preserved": True,
            "sample_rate_preserved": True,
            "channels_preserved": True,
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "selected_window_count": 3,
            "overlay_window_count": 2,
            "blocked_window_count": 0,
            "track_id": "t001",
        },
    )
    result_judge = _write_jsonl(
        tmp_path / "rj" / "no_reference_result_judge.jsonl",
        [{"candidate_id": "full_track_segment_switch_old", "decision": "accept_candidate", "minus_decision": "accept_candidate"}],
    )
    windowed_judge = _write_jsonl(
        tmp_path / "wj" / "wj.jsonl",
        [{"candidate_id": "full_track_segment_switch_old", "decision": "accept_candidate", "minus_decision": "accept_candidate"}],
    )
    targeted_cmp = _write_json(
        tmp_path / "tcmp" / "manifest.json",
        {
            "schema": "targeted_minus_window_cmp.v1",
            "comparison_status": "completed",
            "row_count": 3,
            "regressed_count": regressed_count,
            "release_ready": False,
            "can_publish_as_result": False,
            "delivery_allowed": False,
        },
    )
    return switch_manifest, result_judge, windowed_judge, targeted_cmp


def test_build_fts_listen_pack_accepts_gated_full_switch(tmp_path: Path) -> None:
    switch_manifest, result_judge, windowed_judge, targeted_cmp = _fixtures(tmp_path)

    manifest = build_fts_listen_pack(
        switch_manifest_json=switch_manifest,
        result_judge_jsonl=result_judge,
        windowed_judge_jsonl=windowed_judge,
        targeted_cmp_json=targeted_cmp,
        output_dir=tmp_path / "pack",
    )

    assert manifest["schema"] == "full_track_switch_listen_pack.v1"
    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["selected_count"] == 1
    row = manifest["listen_audio_files"][0]
    assert row["artifact_relative_path"] == "audio/min/m001.wav"
    assert Path(row["listen_audio_path"]).name == "m001.wav"
    assert row["stem_type"] == "minus"
    assert all(len(path.name) <= 32 for path in (tmp_path / "pack").glob("**/*") if path.is_file())


def test_build_fts_listen_pack_blocks_targeted_regression(tmp_path: Path) -> None:
    switch_manifest, result_judge, windowed_judge, targeted_cmp = _fixtures(tmp_path, regressed_count=1)

    manifest = build_fts_listen_pack(
        switch_manifest_json=switch_manifest,
        result_judge_jsonl=result_judge,
        windowed_judge_jsonl=windowed_judge,
        targeted_cmp_json=targeted_cmp,
        output_dir=tmp_path / "pack",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "targeted_cmp_regressed_windows_present" in manifest["failed_checks"]
