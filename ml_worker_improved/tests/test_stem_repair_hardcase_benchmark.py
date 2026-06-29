from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tools.build_stem_repair_hardcase_benchmark import build_stem_repair_hardcase_benchmark
from tools.stem_region_repair_common import ISSUE_BAD_RECONSTRUCTION, ISSUE_VOCAL_LEAK, write_audio, write_jsonl
from tools.stem_region_repair_model import StemRegionRepairTrainConfig
from tools.train_stem_region_repair_model import train_stem_region_repair_model
from tools.validate_stem_repair_checkpoint import validate_stem_repair_checkpoint


SAMPLE_RATE = 8_000


def _sine(freq: float, seconds: float = 0.5, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    mono = amp * np.sin(2.0 * math.pi * freq * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def test_hardcase_benchmark_is_training_only_and_not_promotion_gt(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    source_run = tmp_path / "run.json"
    source_run.write_text(json.dumps({"schema": "run.v1", "candidates": []}), encoding="utf-8")
    write_jsonl(
        queue_jsonl,
        [
            {
                "schema": "training_first_root_cause_sample.v1",
                "row_id": "rc000001",
                "root_cause": "routing_gap",
                "issue_type": ISSUE_BAD_RECONSTRUCTION,
                "affected_stem": "drums",
                "route_signature": "stem_lane:hybrid_htdemucs_ft_bs_roformer_6s",
                "source_run_json": str(source_run),
                "bad_stem_path": str(tmp_path / "bad.flac"),
                "label_type": "blocked_candidate_mining",
                "is_ground_truth": False,
                "usable_for_promotion": False,
            }
        ],
    )

    manifest = build_stem_repair_hardcase_benchmark(
        queue_jsonl=queue_jsonl,
        source_run_json=source_run,
        output_dir=tmp_path / "bench",
    )
    rows = [
        json.loads(line)
        for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits = json.loads(Path(manifest["splits_json"]).read_text(encoding="utf-8"))

    assert manifest["contract"]["delivery_allowed"] is False
    assert manifest["unsafe_promotion_row_count"] == 0
    assert rows[0]["is_ground_truth"] is False
    assert rows[0]["usable_for_promotion"] is False
    assert rows[0]["split"] in {"train_mining", "locked_holdout"}
    assert rows[0]["source_queue_row_id"] == "rc000001"
    assert "locked_holdout" in splits["split_counts"]
    assert Path(manifest["before_after_jsonl"]).exists()


def test_hardcase_benchmark_rejects_fake_ground_truth_from_blocked_rows(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    source_run = tmp_path / "run.json"
    source_run.write_text(json.dumps({"schema": "run.v1", "candidates": []}), encoding="utf-8")
    write_jsonl(
        queue_jsonl,
        [
            {
                "row_id": "rc000666",
                "root_cause": "repair_gap",
                "issue_type": "vocal_leak",
                "affected_stem": "other",
                "label_type": "blocked_candidate_mining",
                "is_ground_truth": True,
                "usable_for_promotion": False,
            }
        ],
    )

    try:
        build_stem_repair_hardcase_benchmark(
            queue_jsonl=queue_jsonl,
            source_run_json=source_run,
            output_dir=tmp_path / "bench",
        )
    except ValueError as exc:
        assert "blocked candidate cannot be ground truth" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("benchmark builder accepted fake ground truth")


def test_checkpoint_validator_writes_before_after_improvement(tmp_path: Path) -> None:
    target = _sine(180.0, amp=0.12)
    leak = _sine(520.0, amp=0.04)
    sibling = _sine(90.0, amp=0.05)
    mask = np.zeros(target.shape[0], dtype=np.float32)
    mask[900:2600] = 1.0
    bad = target + leak * mask[:, None]
    mix = target + sibling
    mix_path = write_audio(tmp_path / "audio" / "mix.flac", mix, SAMPLE_RATE)
    bad_path = write_audio(tmp_path / "audio" / "bad.flac", bad, SAMPLE_RATE)
    target_path = write_audio(tmp_path / "audio" / "target.flac", target, SAMPLE_RATE)
    train_rows = tmp_path / "train_rows.jsonl"
    write_jsonl(
        train_rows,
        [
            {
                "row_id": "rr000001",
                "label_type": "synthetic_clean",
                "is_ground_truth": True,
                "usable_for_promotion": False,
                "issue_type": ISSUE_VOCAL_LEAK,
                "target_stem": "drums",
                "mix_path": str(mix_path),
                "bad_stem_path": str(bad_path),
                "target_stem_path": str(target_path),
                "region": {"start_sec": 0.1125, "end_sec": 0.325},
            }
        ],
    )
    model_manifest = train_stem_region_repair_model(
        rows_jsonl=train_rows,
        output_dir=tmp_path / "model",
        config=StemRegionRepairTrainConfig(epochs=70, hidden_channels=20, learning_rate=0.01, device="cpu"),
    )

    validation = validate_stem_repair_checkpoint(
        checkpoint_path=Path(model_manifest["checkpoint"]),
        benchmark_rows_jsonl=train_rows,
        output_dir=tmp_path / "val",
    )
    result = json.loads(Path(validation["before_after_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert validation["delivery_allowed"] is False
    assert validation["promotion_allowed"] is False
    assert result["evaluated"] is True
    assert result["improved"] is True
    assert result["after_error"] < result["before_error"]
