from __future__ import annotations

import json
from pathlib import Path

from tools.build_golden_corpus import build_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_accepted_minus_manifest_becomes_protected_golden_row(tmp_path: Path) -> None:
    write_json(
        tmp_path / "minus" / "stem_manifest.json",
        {
            "track_id": "song-a",
            "stem_type": "minus",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    rows = build_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["stem_type"] == "minus"
    assert rows[0]["split"] == "holdout"
    assert rows[0]["training_allowed"] is False
    assert rows[0]["threshold_tuning_allowed"] is False
    assert rows[0]["golden_role"] == "protected_regression_holdout"


def test_blocked_vocal_manifest_is_not_golden(tmp_path: Path) -> None:
    write_json(
        tmp_path / "vocal" / "adaptive_vocal_manifest.json",
        {
            "track_id": "song-b",
            "stem_type": "vocal",
            "delivery_status": "rejected",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
        },
    )
    assert build_rows(tmp_path) == []


def test_inconsistent_accepted_looking_blocked_manifest_is_not_golden(tmp_path: Path) -> None:
    write_json(
        tmp_path / "inconsistent" / "stem_manifest.json",
        {
            "track_id": "song-c",
            "stem_type": "minus",
            "delivery_status": "accepted",
            "delivery_allowed": False,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    assert build_rows(tmp_path) == []


def test_output_ordering_is_deterministic(tmp_path: Path) -> None:
    write_json(
        tmp_path / "b" / "stem_manifest.json",
        {
            "track_id": "b",
            "stem_type": "minus",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    write_json(
        tmp_path / "a" / "stem_manifest.json",
        {
            "track_id": "a",
            "stem_type": "minus",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    assert build_rows(tmp_path) == build_rows(tmp_path)
