from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.apply_stem_region_repair_pack import apply_stem_region_repair_pack
from tools.stem_region_repair_common import ISSUE_STEREO_COLLAPSE, ISSUE_VOCAL_LEAK, make_region_mask, write_audio
from tools.stem_region_repair_model import (
    StemRegionRepairTrainConfig,
    save_region_repair_checkpoint,
    train_region_repair_model_from_arrays,
)


SAMPLE_RATE = 8_000


def _wide_bass(seconds: float = 0.4) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    mid = 0.16 * np.sin(2.0 * math.pi * 80.0 * t)
    side = 0.045 * np.sin(2.0 * math.pi * 113.0 * t)
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def _collapse(audio: np.ndarray) -> np.ndarray:
    mid = np.mean(audio, axis=1, keepdims=True)
    return np.repeat(mid, 2, axis=1).astype(np.float32)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _checkpoint_for_stereo_repair(tmp_path: Path, target: np.ndarray, bad: np.ndarray, mix: np.ndarray) -> Path:
    mask = make_region_mask(frame_count=bad.shape[0], sample_rate=SAMPLE_RATE, start_sec=0.0, end_sec=bad.shape[0] / SAMPLE_RATE)
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=target,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_STEREO_COLLAPSE,
        config=StemRegionRepairTrainConfig(epochs=70, hidden_channels=16, learning_rate=0.01, device="cpu"),
    )
    return save_region_repair_checkpoint(model=model, path=tmp_path / "model" / "model.pt", training_history=history)


def test_repair_pack_applies_checkpoint_and_records_stereo_width_improvement(tmp_path: Path) -> None:
    target = _wide_bass()
    bad = _collapse(target)
    mix = np.clip(target + 0.02, -1.0, 1.0).astype(np.float32)
    run_dir = tmp_path / "run" / "w000"
    bad_path = write_audio(run_dir / "c001" / "stems" / "bass.flac", bad, SAMPLE_RATE)
    write_audio(run_dir / "mix.flac", mix, SAMPLE_RATE)
    checkpoint = _checkpoint_for_stereo_repair(tmp_path, target, bad, mix)
    rows_jsonl = _write_jsonl(
        tmp_path / "hardcase" / "rows.jsonl",
        [
            {
                "row_id": "hc000000",
                "training_state": "data_gap",
                "issue_type": ISSUE_STEREO_COLLAPSE,
                "affected_stem": "bass",
                "bad_stem_path": str(tmp_path / "hardcase" / "audio" / "b000.flac"),
                "source_bad_stem_path": str(bad_path),
                "blocker_reason": "stereo_width_collapse:bass",
                "delivery_allowed": False,
            }
        ],
    )
    write_audio(tmp_path / "hardcase" / "audio" / "b000.flac", bad, SAMPLE_RATE)

    manifest = apply_stem_region_repair_pack(
        checkpoint_path=checkpoint,
        hardcase_rows_jsonl=rows_jsonl,
        output_dir=tmp_path / "apply",
        max_rows=4,
    )
    rows = [
        json.loads(line)
        for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["row_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["diagnostics"]["stereo_width_after"] > rows[0]["diagnostics"]["stereo_width_before"]
    assert rows[0]["diagnostics"]["target_issue_improved"] is True
    assert rows[0]["repaired_stem_path"].endswith("audio/r000.flac")
    assert "c001" not in rows[0]["repaired_stem_path"]
    assert Path(rows[0]["repaired_stem_path"]).exists()


def test_repair_pack_skips_rows_when_checkpoint_issue_mismatches(tmp_path: Path) -> None:
    target = _wide_bass()
    bad = _collapse(target)
    mix = target.copy()
    checkpoint = _checkpoint_for_stereo_repair(tmp_path, target, bad, mix)
    bad_path = write_audio(tmp_path / "run" / "w000" / "c001" / "stems" / "drums.flac", bad, SAMPLE_RATE)
    write_audio(tmp_path / "run" / "w000" / "mix.flac", mix, SAMPLE_RATE)
    rows_jsonl = _write_jsonl(
        tmp_path / "hardcase" / "rows.jsonl",
        [
            {
                "row_id": "hc000001",
                "training_state": "data_gap",
                "issue_type": ISSUE_VOCAL_LEAK,
                "affected_stem": "drums",
                "bad_stem_path": str(bad_path),
                "source_bad_stem_path": str(bad_path),
                "blocker_reason": "foreign_source_recovered:drums:vocals",
            }
        ],
    )

    manifest = apply_stem_region_repair_pack(
        checkpoint_path=checkpoint,
        hardcase_rows_jsonl=rows_jsonl,
        output_dir=tmp_path / "apply",
        max_rows=4,
    )

    assert manifest["row_count"] == 0
    assert manifest["skipped_count"] == 1
    assert manifest["skipped"][0]["reason"] == "issue_mismatch"
