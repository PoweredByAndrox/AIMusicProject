from __future__ import annotations

import json
from pathlib import Path

from tools.build_hard_case_corpus import build_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_blocked_vocal_creates_hard_case_row_with_required_fields(tmp_path: Path) -> None:
    write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "track_id": "song-v",
            "stem_type": "vocal",
            "delivery_status": "rejected",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "failure_labels": ["thin_or_overcleaned"],
            "issue_windows": [{"start_sec": 10.0, "end_sec": 15.0}],
        },
    )
    rows = build_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["source_manifest"]
    assert row["failure_labels"]
    assert row["stem_type"] == "vocal"
    assert row["delivery_status"] == "rejected"
    assert row["window_start_sec"] == 10.0
    assert row["window_end_sec"] == 15.0


def test_blocked_minus_creates_minus_hard_case(tmp_path: Path) -> None:
    write_json(
        tmp_path / "minus" / "stem_manifest.json",
        {
            "track_id": "song-m",
            "stem_type": "minus",
            "delivery_status": "rejected",
            "delivery_allowed": False,
        },
    )
    rows = build_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["stem_type"] == "minus"


def test_accepted_minus_is_not_vocal_hard_failure(tmp_path: Path) -> None:
    write_json(
        tmp_path / "minus" / "stem_manifest.json",
        {
            "track_id": "song-m",
            "stem_type": "minus",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    assert build_rows(tmp_path) == []


def test_missing_failure_labels_become_unknown_reason_not_blank(tmp_path: Path) -> None:
    write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "track_id": "song-v",
            "stem_type": "vocal",
            "delivery_status": "rejected",
            "delivery_allowed": False,
        },
    )
    rows = build_rows(tmp_path)
    assert rows[0]["dominant_failure_reason"]
    assert rows[0]["failure_labels"]


def test_inconsistent_accepted_looking_blocked_manifest_is_hard_case(tmp_path: Path) -> None:
    write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "track_id": "song-v",
            "stem_type": "vocal",
            "delivery_status": "accepted",
            "delivery_allowed": False,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    rows = build_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["dominant_failure_reason"] == "unknown_blocked_reason"
    assert rows[0]["failure_labels"] == ["unknown_blocked_reason"]


def test_same_track_hash_has_same_split(tmp_path: Path) -> None:
    write_json(
        tmp_path / "a" / "adaptive_vocal_manifest.json",
        {"track_hash": "same", "stem_type": "vocal", "delivery_status": "rejected", "delivery_allowed": False},
    )
    write_json(
        tmp_path / "b" / "stem_manifest.json",
        {"track_hash": "same", "stem_type": "minus", "delivery_status": "rejected", "delivery_allowed": False},
    )
    rows = build_rows(tmp_path)
    assert len({row["split"] for row in rows}) == 1
