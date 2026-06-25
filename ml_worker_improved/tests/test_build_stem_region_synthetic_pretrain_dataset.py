from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_stem_region_synthetic_pretrain_dataset import build_stem_region_synthetic_pretrain_dataset
from tools.stem_region_repair_common import LABEL_TYPE_SYNTHETIC_CLEAN


SAMPLE_RATE = 8_000


def _write_hardcase_audio(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    left = 0.1 * np.sin(2.0 * math.pi * 120.0 * t)
    right = 0.08 * np.sin(2.0 * math.pi * 123.0 * t)
    sf.write(path, np.stack([left, right], axis=1).astype(np.float32), SAMPLE_RATE)
    return path


def _write_hardcase_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_synthetic_pretrain_dataset_does_not_self_label_hardcase_audio(tmp_path: Path) -> None:
    source_bad = _write_hardcase_audio(tmp_path / "hardcase" / "audio" / "b000.flac")
    rows_jsonl = _write_hardcase_rows(
        tmp_path / "hardcase" / "rows.jsonl",
        [
            {
                "row_id": "hc000000",
                "training_state": "data_gap",
                "issue_type": "stereo_collapse",
                "affected_stem": "bass",
                "foreign_source": "",
                "bad_stem_path": str(source_bad),
                "blocker_reason": "stereo_width_collapse:bass",
                "is_ground_truth": False,
            }
        ],
    )

    manifest = build_stem_region_synthetic_pretrain_dataset(
        hardcase_rows_jsonl=rows_jsonl,
        output_dir=tmp_path / "pretrain",
        max_rows=1,
        duration_sec=0.4,
    )
    rows = [
        json.loads(line)
        for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["row_count"] == 1
    assert manifest["synthetic_ground_truth_count"] == 1
    assert rows[0]["label_type"] == LABEL_TYPE_SYNTHETIC_CLEAN
    assert rows[0]["is_ground_truth"] is True
    assert rows[0]["usable_for_promotion"] is False
    assert rows[0]["source_hardcase_bad_stem_path"] == str(source_bad)
    assert rows[0]["target_stem_path"] != str(source_bad)
    assert rows[0]["bad_stem_path"] != str(source_bad)
    assert rows[0]["target_stem_path"].endswith("audio/t000.flac")
    assert rows[0]["bad_stem_path"].endswith("audio/b000.flac")
    assert rows[0]["mix_path"].endswith("audio/m000.flac")
    assert Path(rows[0]["target_stem_path"]).exists()
    assert Path(rows[0]["bad_stem_path"]).exists()


def test_synthetic_pretrain_dataset_supports_vocal_leak_rows(tmp_path: Path) -> None:
    source_bad = _write_hardcase_audio(tmp_path / "hardcase" / "audio" / "b001.flac")
    rows_jsonl = _write_hardcase_rows(
        tmp_path / "hardcase" / "rows.jsonl",
        [
            {
                "row_id": "hc000001",
                "training_state": "data_gap",
                "issue_type": "vocal_leak",
                "affected_stem": "drums",
                "foreign_source": "vocals",
                "bad_stem_path": str(source_bad),
                "blocker_reason": "foreign_source_recovered:drums:vocals",
                "is_ground_truth": False,
            }
        ],
    )

    manifest = build_stem_region_synthetic_pretrain_dataset(
        hardcase_rows_jsonl=rows_jsonl,
        output_dir=tmp_path / "pretrain",
        max_rows=1,
        duration_sec=0.4,
    )

    assert manifest["issue_type_counts"] == {"vocal_leak": 1}
    assert manifest["stem_counts"] == {"drums": 1}


def test_synthetic_pretrain_dataset_can_filter_issue_type(tmp_path: Path) -> None:
    stereo_bad = _write_hardcase_audio(tmp_path / "hardcase" / "audio" / "b000.flac")
    leak_bad = _write_hardcase_audio(tmp_path / "hardcase" / "audio" / "b001.flac")
    rows_jsonl = _write_hardcase_rows(
        tmp_path / "hardcase" / "rows.jsonl",
        [
            {
                "row_id": "hc000000",
                "training_state": "data_gap",
                "issue_type": "stereo_collapse",
                "affected_stem": "bass",
                "bad_stem_path": str(stereo_bad),
                "is_ground_truth": False,
            },
            {
                "row_id": "hc000001",
                "training_state": "data_gap",
                "issue_type": "vocal_leak",
                "affected_stem": "bass",
                "foreign_source": "vocals",
                "bad_stem_path": str(leak_bad),
                "is_ground_truth": False,
            },
        ],
    )

    manifest = build_stem_region_synthetic_pretrain_dataset(
        hardcase_rows_jsonl=rows_jsonl,
        output_dir=tmp_path / "pretrain",
        max_rows=1,
        duration_sec=0.4,
        issue_type_filter="vocal_leak",
    )
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert manifest["issue_type_counts"] == {"vocal_leak": 1}
    assert row["source_hardcase_row_id"] == "hc000001"


def test_synthetic_pretrain_dataset_supports_instrument_leak_rows(tmp_path: Path) -> None:
    source_bad = _write_hardcase_audio(tmp_path / "hardcase" / "audio" / "b002.flac")
    rows_jsonl = _write_hardcase_rows(
        tmp_path / "hardcase" / "rows.jsonl",
        [
            {
                "row_id": "hc000002",
                "training_state": "data_gap",
                "issue_type": "instrument_leak",
                "affected_stem": "other",
                "foreign_source": "guitar",
                "bad_stem_path": str(source_bad),
                "blocker_reason": "foreign_source_recovered:other:guitar",
                "is_ground_truth": False,
            }
        ],
    )

    manifest = build_stem_region_synthetic_pretrain_dataset(
        hardcase_rows_jsonl=rows_jsonl,
        output_dir=tmp_path / "pretrain",
        max_rows=1,
        duration_sec=0.4,
    )
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert manifest["issue_type_counts"] == {"instrument_leak": 1}
    assert manifest["stem_counts"] == {"other": 1}
    assert row["source_hardcase_row_id"] == "hc000002"
    assert row["corruption"]["corruption"] == "instrument_leak_injection"


def test_synthetic_pretrain_dataset_can_filter_target_stem(tmp_path: Path) -> None:
    bass_bad = _write_hardcase_audio(tmp_path / "hardcase" / "audio" / "b003.flac")
    other_bad = _write_hardcase_audio(tmp_path / "hardcase" / "audio" / "b004.flac")
    rows_jsonl = _write_hardcase_rows(
        tmp_path / "hardcase" / "rows.jsonl",
        [
            {
                "row_id": "hc000003",
                "training_state": "data_gap",
                "issue_type": "vocal_leak",
                "affected_stem": "bass",
                "foreign_source": "vocals",
                "bad_stem_path": str(bass_bad),
                "is_ground_truth": False,
            },
            {
                "row_id": "hc000004",
                "training_state": "data_gap",
                "issue_type": "vocal_leak",
                "affected_stem": "other",
                "foreign_source": "vocals",
                "bad_stem_path": str(other_bad),
                "is_ground_truth": False,
            },
        ],
    )

    manifest = build_stem_region_synthetic_pretrain_dataset(
        hardcase_rows_jsonl=rows_jsonl,
        output_dir=tmp_path / "pretrain",
        max_rows=1,
        duration_sec=0.4,
        issue_type_filter="vocal_leak",
        stem_filter="other",
    )
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert manifest["stem_filter"] == "other"
    assert manifest["stem_counts"] == {"other": 1}
    assert row["source_hardcase_row_id"] == "hc000004"
