from __future__ import annotations

import json
from pathlib import Path

from tools.audit_minus_shape_normalization_contract import (
    RAW_CANDIDATE_ID_MAX_LEN,
    build_shape_normalization_audit,
    raw_candidate_id,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_builds_raw_rejudge_rows_for_shape_normalized_minus_candidates(tmp_path: Path) -> None:
    route_dir = tmp_path / "route"
    raw_minus = tmp_path / "raw" / "minus.flac"
    raw_vocal = tmp_path / "raw" / "vocals.flac"
    norm_minus = tmp_path / "norm" / "minus.flac"
    for path in (raw_minus, raw_vocal, norm_minus):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")

    write_jsonl(
        route_dir / "minus_route_candidate_pool.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "cand-a",
                "route_id": "route-a",
                "candidate_family": "family-a",
                "mix_path": str(tmp_path / "mix.flac"),
                "vocal_path": str(route_dir / "norm" / "vocals.flac"),
                "minus_path": str(norm_minus),
                "instrumental_path": str(norm_minus),
                "raw_vocal_path": str(raw_vocal),
                "raw_minus_path": str(raw_minus),
                "normalized_minus_path": str(norm_minus),
                "shape_normalization_applied": True,
                "shape_normalization_reason": "sample_rate_mismatch+duration_mismatch",
                "release_ready": False,
            }
        ],
    )
    write_json(
        tmp_path / "forensics" / "minus_window_blocker_clusters.json",
        {
            "summary": {
                "normalization_correlated": True,
                "normalization_evidence": {
                    "shape_normalized_avg_blocker_count": 3.5,
                    "non_normalized_avg_blocker_count": 1.0,
                },
                "release_ready": False,
            }
        },
    )

    result = build_shape_normalization_audit(
        route_dir=route_dir,
        forensics_dir=tmp_path / "forensics",
    )

    assert result["summary"]["normalization_issue_suspected"] is True
    assert result["summary"]["raw_rejudge_candidates"] == 1
    assert result["summary"]["recommended_next_step"] == "run_raw_shape_minus_judge"
    assert result["summary"]["delivery_allowed"] is False
    assert result["summary"]["release_ready"] is False
    row = result["raw_rejudge_rows"][0]
    assert row["candidate_id"] == "cand-a__raw_shape"
    assert row["minus_path"] == str(raw_minus)
    assert row["instrumental_path"] == str(raw_minus)
    assert row["vocal_path"] == str(raw_vocal)
    assert row["candidate_origin"] == "raw_shape_normalization_contract_audit"
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False


def test_raw_candidate_id_is_short_ascii_stable() -> None:
    raw_id = raw_candidate_id("very/long/candidate/id/with unicode вокал and unsafe chars " * 3)

    assert raw_id.endswith("__raw_shape")
    assert len(raw_id) <= RAW_CANDIDATE_ID_MAX_LEN
    assert raw_id.isascii()
    assert raw_id == raw_candidate_id("very/long/candidate/id/with unicode вокал and unsafe chars " * 3)
