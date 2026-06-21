from __future__ import annotations

import json
from pathlib import Path

from tools.build_minus_cmp_switch_inputs import build_minus_cmp_switch_inputs


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


def _cmp_report(tmp_path: Path, *, release_ready: bool = False) -> Path:
    root = tmp_path / "cmp"
    rows_path = root / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "window_id": "w001",
                "window_start_sec": 10.0,
                "window_duration_sec": 1.0,
                "baseline_metric_audio_path": str(tmp_path / "old_min.flac"),
                "baseline_minus_score": 0.80,
                "baseline_vocal_leakage": 0.30,
                "baseline_mixture_consistency": 0.70,
                "candidate_audio_path": "audio/min/m001.flac",
                "candidate_id": "q001",
                "candidate_minus_score": 0.88,
                "candidate_vocal_leakage": 0.10,
                "candidate_mixture_consistency": 0.82,
                "comparison_decision": "candidate_proxy_improved",
            },
            {
                "window_id": "w002",
                "window_start_sec": 20.0,
                "window_duration_sec": 1.0,
                "baseline_metric_audio_path": str(tmp_path / "old_min.flac"),
                "baseline_minus_score": 0.86,
                "baseline_vocal_leakage": 0.08,
                "baseline_mixture_consistency": 0.76,
                "candidate_audio_path": "audio/min/m001.flac",
                "candidate_id": "q001",
                "candidate_minus_score": 0.83,
                "candidate_vocal_leakage": 0.42,
                "candidate_mixture_consistency": 0.82,
                "comparison_decision": "candidate_proxy_regressed",
            },
        ],
    )
    return _write_json(
        root / "cmp.json",
        {
            "schema": "minus_middle_comparison_report.v1",
            "comparison_status": "completed",
            "row_count": 2,
            "failed_checks": [],
            "rows_jsonl": str(rows_path),
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "delivery_allowed": False,
        },
    )


def _selection_pack(tmp_path: Path, *, schema: str = "short_listen_repair_selection_pack_manifest.v1") -> Path:
    return _write_json(
        tmp_path / "rs" / "pack.json",
        {
            "schema": schema,
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "listen_audio_files": [
                {
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(tmp_path / "new_min.flac"),
                    "source_candidate_id": "q001",
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )


def test_build_minus_cmp_switch_inputs_blocks_regressed_windows(tmp_path: Path) -> None:
    manifest = build_minus_cmp_switch_inputs(
        cmp_json=_cmp_report(tmp_path),
        selection_pack_json=_selection_pack(tmp_path),
        output_dir=tmp_path / "out",
        track_id="t001",
        mix_path=tmp_path / "mix.flac",
        vocal_path=tmp_path / "voc.flac",
        base_candidate_id="old",
    )

    assert manifest["schema"] == "minus_cmp_switch_inputs.v1"
    assert manifest["input_status"] == "ready"
    assert manifest["failed_checks"] == []
    assert manifest["base_candidate_id"] == "old"
    assert manifest["alternate_candidate_count"] == 1
    assert manifest["improved_window_count"] == 1
    assert manifest["blocked_alternate_window_count"] == 1
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False

    pool_rows = [json.loads(line) for line in (tmp_path / "out" / "pool.jsonl").read_text(encoding="utf-8").splitlines()]
    judge_rows = [json.loads(line) for line in (tmp_path / "out" / "wj.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["candidate_id"] for row in pool_rows] == ["old", "q001"]
    assert pool_rows[0]["source_minus_path"].endswith("old_min.flac")
    assert pool_rows[0]["base_minus_path"].endswith("old_min.flac")
    assert pool_rows[1]["source_minus_path"].endswith("old_min.flac")
    assert pool_rows[1]["base_minus_path"].endswith("old_min.flac")
    by_id = {row["candidate_id"]: row for row in judge_rows}
    assert by_id["old"]["window_metrics"][1]["cmp_selection_decision"] == "accept_candidate"
    assert by_id["q001"]["window_metrics"][0]["cmp_selection_decision"] == "accept_candidate"
    assert by_id["q001"]["window_metrics"][1]["cmp_selection_decision"] == "blocked_hard_case"
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_build_minus_cmp_switch_inputs_accepts_candidate_wave_pack_schema(tmp_path: Path) -> None:
    manifest = build_minus_cmp_switch_inputs(
        cmp_json=_cmp_report(tmp_path),
        selection_pack_json=_selection_pack(tmp_path, schema="short_candidate_wave_listen_pack_manifest.v1"),
        output_dir=tmp_path / "out",
        track_id="t001",
        mix_path=tmp_path / "mix.flac",
        vocal_path=tmp_path / "voc.flac",
        base_candidate_id="old",
    )

    assert manifest["input_status"] == "ready"
    assert manifest["failed_checks"] == []
    assert manifest["alternate_candidate_count"] == 1
    assert manifest["improved_window_count"] == 1


def test_build_minus_cmp_switch_inputs_blocks_release_ready_cmp(tmp_path: Path) -> None:
    manifest = build_minus_cmp_switch_inputs(
        cmp_json=_cmp_report(tmp_path, release_ready=True),
        selection_pack_json=_selection_pack(tmp_path),
        output_dir=tmp_path / "out",
        track_id="t001",
        mix_path=tmp_path / "mix.flac",
        vocal_path=tmp_path / "voc.flac",
        base_candidate_id="old",
    )

    assert manifest["input_status"] == "blocked"
    assert "cmp_release_ready_true" in manifest["failed_checks"]
