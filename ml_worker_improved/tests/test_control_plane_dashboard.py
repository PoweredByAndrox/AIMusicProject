from __future__ import annotations

import json
from pathlib import Path

from tools.build_control_plane_dashboard import build_dashboard


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def test_dashboard_counts_control_plane_rows(tmp_path: Path) -> None:
    golden_jsonl = write_jsonl(
        tmp_path / "golden.jsonl",
        [
            {
                "case_id": "g1",
                "stem_type": "vocal",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
            },
            {
                "case_id": "g2",
                "stem_type": "minus",
                "delivery_allowed": True,
                "release_status": "release_ready",
            },
        ],
    )
    hard_jsonl = write_jsonl(
        tmp_path / "hard.jsonl",
        [
            {
                "hard_case_id": "h1",
                "stem_type": "vocal",
                "delivery_status": "no_accepted_candidate",
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
                "dominant_failure_reason": "candidate_space_exhausted",
                "failure_labels": ["candidate_space_exhausted", "no_accepted_candidate"],
            },
            {
                "hard_case_id": "h2",
                "stem_type": "minus",
                "delivery_status": "diagnostic_only",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_only",
                "dominant_failure_reason": "unknown_blocked_reason",
                "failure_labels": ["unknown_blocked_reason"],
                "audio_path": "/tmp/diagnostic.flac",
            },
        ],
    )

    dashboard = build_dashboard(golden_jsonl, hard_jsonl)

    summary = dashboard["summary"]
    assert summary["golden_rows"] == 2
    assert summary["hard_case_rows"] == 2
    assert summary["blocked_vocal"] == 1
    assert summary["blocked_minus"] == 1
    assert summary["unknown_reason"] == 1
    assert summary["unsafe_diagnostic"] == 1
    assert summary["no_accepted_candidate"] == 1
    assert summary["candidate_space_exhausted"] == 1
