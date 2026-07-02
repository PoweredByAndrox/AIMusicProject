from __future__ import annotations

import json
from pathlib import Path

from tools.build_ralph_residual_focus_windows import build_ralph_residual_focus_windows


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_builds_focus_rows_from_selected_ralph_residual_windows(tmp_path: Path) -> None:
    manifest_path = write_json(
        tmp_path / "ralph_manifest.json",
        {
            "mode": "ralph_minus_improvement_loop",
            "selected_candidate": "hybrid_detail_safe",
            "focus_window_filters": {"track_id": "raw_audio"},
            "candidates": [
                {
                    "candidate_name": "hybrid_detail_safe",
                    "evaluation": {
                        "status": "accepted",
                        "summary": {
                            "top_vocal_residue_windows": [
                                {
                                    "start_sec": 100.0,
                                    "end_sec": 105.0,
                                    "vocal_residue_corr": 0.086,
                                    "vocal_projection_ratio_db": -21.2,
                                },
                                {
                                    "start_sec": 50.0,
                                    "end_sec": 55.0,
                                    "vocal_residue_corr": 0.035,
                                    "vocal_projection_ratio_db": -29.0,
                                },
                                {
                                    "start_sec": 120.0,
                                    "end_sec": 125.0,
                                    "vocal_residue_corr": 0.039,
                                    "vocal_projection_ratio_db": -28.0,
                                },
                            ]
                        },
                    },
                },
                {
                    "candidate_name": "rejected_candidate",
                    "evaluation": {
                        "status": "rejected",
                        "summary": {
                            "top_vocal_residue_windows": [
                                {
                                    "start_sec": 10.0,
                                    "end_sec": 15.0,
                                    "vocal_residue_corr": 0.99,
                                    "vocal_projection_ratio_db": -5.0,
                                }
                            ]
                        },
                    },
                },
            ],
        },
    )

    result = build_ralph_residual_focus_windows(
        ralph_manifest=manifest_path,
        output_dir=tmp_path / "out",
        max_windows=2,
    )

    rows_path = Path(result["focus_windows_jsonl"])
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert [row["window_start_sec"] for row in rows] == [100.0, 120.0]
    assert {row["candidate_id"] for row in rows} == {"hybrid_detail_safe"}
    assert all(row["true_label"] == 0 for row in rows)
    assert all(row["predicted_blockers"] == ["ralph_top_vocal_residue_window"] for row in rows)
    assert result["delivery_allowed"] is False
    assert result["summary"]["focus_window_count"] == 2
