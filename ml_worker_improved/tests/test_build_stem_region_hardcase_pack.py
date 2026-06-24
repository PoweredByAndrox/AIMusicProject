from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_stem_region_hardcase_pack import build_stem_region_hardcase_pack


SAMPLE_RATE = 8_000


def _write_audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SAMPLE_RATE * 0.2), dtype=np.float32) / float(SAMPLE_RATE)
    mono = 0.1 * np.sin(2.0 * math.pi * 120.0 * t)
    sf.write(path, np.stack([mono, mono], axis=1).astype(np.float32), SAMPLE_RATE)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_hardcase_pack_aliases_bad_audio_and_records_data_gap(tmp_path: Path) -> None:
    bad_bass = _write_audio(tmp_path / "runs" / "w000" / "c371" / "stems" / "bass.flac")
    score = _write_json(tmp_path / "runs" / "w000" / "c371" / "score.json", {"verdict": "blocked_damage"})
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "row_id": "rc000013",
                "root_cause": "repair_gap",
                "issue_type": "stereo_collapse",
                "affected_stem": "bass",
                "foreign_source": "",
                "repeat_count": 5,
                "bad_stem_path": str(bad_bass),
                "score_path": str(score),
                "blocker_reason": "stereo_width_collapse:bass",
                "source_run_json": str(tmp_path / "runs" / "run.json"),
                "is_ground_truth": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_stem_region_hardcase_pack(
        queue_jsonl=queue,
        output_dir=tmp_path / "pack",
        issue_types=["stereo_collapse"],
        max_rows=4,
    )

    rows = [
        json.loads(line)
        for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["row_count"] == 1
    assert manifest["data_gap_count"] == 1
    assert rows[0]["training_state"] == "data_gap"
    assert rows[0]["target_stem_path"] == ""
    assert rows[0]["bad_stem_path"].endswith("audio/b000.flac")
    assert rows[0]["score_path"].endswith("score/s000.json")
    assert "c371" not in rows[0]["bad_stem_path"]
    assert Path(rows[0]["bad_stem_path"]).exists()
    assert Path(rows[0]["score_path"]).exists()
    assert manifest["delivery_allowed"] is False


def test_hardcase_pack_skips_rows_without_bad_audio(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "row_id": "rc000031",
                "root_cause": "repair_gap",
                "issue_type": "vocal_leak",
                "affected_stem": "drums",
                "foreign_source": "vocals",
                "repeat_count": 5,
                "bad_stem_path": str(tmp_path / "missing.flac"),
                "score_path": "",
                "blocker_reason": "foreign_source_recovered:drums:vocals",
                "is_ground_truth": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_stem_region_hardcase_pack(
        queue_jsonl=queue,
        output_dir=tmp_path / "pack",
        issue_types=["vocal_leak"],
        max_rows=4,
    )

    assert manifest["row_count"] == 0
    assert manifest["skipped_count"] == 1
    assert manifest["skipped"][0]["reason"] == "missing_bad_stem"
