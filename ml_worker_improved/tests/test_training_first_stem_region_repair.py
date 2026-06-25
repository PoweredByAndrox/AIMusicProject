from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.build_training_first_root_cause_queue import build_root_cause_queue
from tools.build_stem_region_repair_dataset import build_stem_region_repair_dataset
from tools.train_stem_region_repair_model import train_stem_region_repair_model
from tools.apply_stem_region_repair import apply_stem_region_repair
from tools.stem_region_repair_common import (
    LABEL_TYPE_GENERATED_SEPARATOR,
    LABEL_TYPE_SYNTHETIC_CLEAN,
    ISSUE_VOCAL_LEAK,
    make_region_mask,
    read_audio_stereo,
    validate_region_repair_row,
    write_audio,
)
from tools.stem_region_repair_model import (
    StemRegionRepairTrainConfig,
    apply_region_repair_model,
    train_region_repair_model_from_arrays,
)


SAMPLE_RATE = 8_000


def _sine(freq: float, seconds: float = 0.5, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    mono = amp * np.sin(2.0 * math.pi * freq * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _stereo_widen(audio: np.ndarray, side_amp: float = 0.035) -> np.ndarray:
    t = np.arange(audio.shape[0], dtype=np.float32) / float(SAMPLE_RATE)
    side = side_amp * np.sin(2.0 * math.pi * 97.0 * t)
    widened = audio.copy()
    widened[:, 0] += side
    widened[:, 1] -= side
    return widened.astype(np.float32)


def _mse_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    weights = mask[:, None].astype(np.float32)
    denom = max(float(np.sum(weights)) * float(a.shape[1]), 1.0)
    return float(np.sum(np.square((a - b) * weights)) / denom)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    frames = min(a.shape[0], b.shape[0])
    return float(np.mean(np.square(a[:frames] - b[:frames])))


def _write_run_json(path: Path) -> Path:
    payload = {
        "schema": "fully_automatic_stem_run.v1",
        "candidates": [
            {
                "candidate_id": "c001",
                "window_id": "w000",
                "verdict": "blocked_leak",
                "next_action": {"reason": "foreign_source_recovered:drums:vocals"},
                "target_specific_composition": {
                    "routes": [
                        {"lane": "stem_lane", "route_id": "hybrid_htdemucs_ft_bs_roformer_6s"},
                        {"lane": "vocal_lane", "route_id": "mdx23c_vocals"},
                        {"lane": "minus_lane", "route_id": "mdx23c_vocals:instrumental"},
                    ]
                },
                "stem_paths": {"drums": "/tmp/drums_bad.flac"},
                "score_path": "/tmp/score.json",
            },
            {
                "candidate_id": "c002",
                "window_id": "w000",
                "verdict": "blocked_leak",
                "next_action": {"reason": "foreign_source_recovered:drums:vocals"},
                "target_specific_composition": {
                    "routes": [
                        {"lane": "stem_lane", "route_id": "hybrid_htdemucs_ft_bs_roformer_6s"},
                        {"lane": "vocal_lane", "route_id": "mdx23c_vocals"},
                        {"lane": "minus_lane", "route_id": "mdx23c_vocals:instrumental"},
                    ]
                },
                "stem_paths": {"drums": "/tmp/drums_bad_2.flac"},
                "score_path": "/tmp/score2.json",
            },
            {
                "candidate_id": "c003",
                "window_id": "w000",
                "verdict": "blocked_unknown",
                "execution_status": "failed_runtime_error",
                "error": {"type": "CalledProcessError", "message": "bs_mamba2_vocals unavailable"},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_repeated_blocker_becomes_single_training_queue_sample(tmp_path: Path) -> None:
    manifest = build_root_cause_queue(
        run_json_path=_write_run_json(tmp_path / "run.json"),
        output_dir=tmp_path / "queue",
        min_repeats=2,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "queue" / "queue.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["root_cause_counts"] == {"repair_gap": 1, "infra_gap": 1}
    assert len(rows) == 2
    repair_row = next(row for row in rows if row["root_cause"] == "repair_gap")
    assert repair_row["training_state"] == "data_gap"
    assert repair_row["issue_type"] == ISSUE_VOCAL_LEAK
    assert repair_row["affected_stem"] == "drums"
    assert repair_row["foreign_source"] == "vocals"
    assert repair_row["repeat_count"] == 2
    assert repair_row["training_action"] == "mine_region_repair_example"


def test_auto_accepted_candidates_are_not_training_failures(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "schema": "fully_automatic_stem_run.v1",
                "candidates": [
                    {"candidate_id": "c001", "window_id": "w000", "verdict": "auto_accept"},
                    {"candidate_id": "c001b", "window_id": "w000", "verdict": "auto_accept"},
                    {
                        "candidate_id": "c002",
                        "window_id": "w001",
                        "verdict": "blocked_leak",
                        "next_action": {"reason": "foreign_source_recovered:other:vocals"},
                    },
                    {
                        "candidate_id": "c003",
                        "window_id": "w001",
                        "verdict": "blocked_leak",
                        "next_action": {"reason": "foreign_source_recovered:other:vocals"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_root_cause_queue(run_json_path=run_json, output_dir=tmp_path / "queue", min_repeats=2)
    rows = [
        json.loads(line)
        for line in Path(manifest["queue_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["queue_row_count"] == 1
    assert rows[0]["candidate_ids"] == ["c002", "c003"]


def test_root_cause_queue_splits_multiple_foreign_source_blockers(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    reason = "foreign_source_recovered:bass:vocals,foreign_source_recovered:other:vocals"
    run_json.write_text(
        json.dumps(
            {
                "schema": "fully_automatic_stem_run.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "window_id": "w000",
                        "verdict": "blocked_leak",
                        "next_action": {"reason": reason},
                        "stem_paths": {"bass": "/tmp/bass.flac", "other": "/tmp/other.flac"},
                    },
                    {
                        "candidate_id": "c002",
                        "window_id": "w000",
                        "verdict": "blocked_leak",
                        "next_action": {"reason": reason},
                        "stem_paths": {"bass": "/tmp/bass2.flac", "other": "/tmp/other2.flac"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_root_cause_queue(run_json_path=run_json, output_dir=tmp_path / "queue", min_repeats=2)
    rows = [
        json.loads(line)
        for line in Path(manifest["queue_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["queue_row_count"] == 2
    assert {(row["affected_stem"], row["foreign_source"], row["blocker_reason"]) for row in rows} == {
        ("bass", "vocals", "foreign_source_recovered:bass:vocals"),
        ("other", "vocals", "foreign_source_recovered:other:vocals"),
    }
    assert all(row["training_state"] == "data_gap" for row in rows)


def test_generated_separator_output_cannot_be_region_repair_ground_truth() -> None:
    with pytest.raises(ValueError, match="generated separator"):
        validate_region_repair_row(
            {
                "row_id": "rr000001",
                "label_type": LABEL_TYPE_GENERATED_SEPARATOR,
                "is_ground_truth": True,
                "issue_type": ISSUE_VOCAL_LEAK,
                "target_stem": "drums",
            }
        )


def test_blocked_candidate_mining_rows_cannot_drive_promotion() -> None:
    with pytest.raises(ValueError, match="cannot drive promotion"):
        validate_region_repair_row(
            {
                "row_id": "rr000003",
                "label_type": "blocked_candidate_mining",
                "is_ground_truth": False,
                "usable_for_promotion": True,
                "issue_type": ISSUE_VOCAL_LEAK,
                "target_stem": "other",
            }
        )


def test_tiny_repair_model_overfits_vocal_leak_region() -> None:
    clean = _stereo_widen(_sine(110.0, amp=0.18))
    vocal_leak = _sine(660.0, amp=0.05)
    mask = make_region_mask(frame_count=clean.shape[0], sample_rate=SAMPLE_RATE, start_sec=0.12, end_sec=0.36)
    bad = clean + vocal_leak * mask[:, None]
    mix = bad + _sine(220.0, amp=0.08)

    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=clean,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_VOCAL_LEAK,
        config=StemRegionRepairTrainConfig(epochs=90, hidden_channels=24, learning_rate=0.01, device="cpu"),
    )
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_VOCAL_LEAK,
    )

    before = _mse_region(bad, clean, mask)
    after = _mse_region(repaired, clean, mask)
    assert history["final_loss"] < history["initial_loss"]
    assert after < before * 0.35
    assert diagnostics["changed_region_ratio"] > 0.0


def test_repair_model_does_not_modify_clean_regions_outside_mask() -> None:
    clean = _stereo_widen(_sine(130.0, amp=0.16))
    vocal_leak = _sine(710.0, amp=0.04)
    mask = make_region_mask(frame_count=clean.shape[0], sample_rate=SAMPLE_RATE, start_sec=0.18, end_sec=0.32)
    bad = clean + vocal_leak * mask[:, None]
    mix = bad + _sine(250.0, amp=0.08)

    model, _ = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=clean,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_VOCAL_LEAK,
        config=StemRegionRepairTrainConfig(epochs=60, hidden_channels=16, learning_rate=0.01, device="cpu"),
    )
    repaired, _ = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_VOCAL_LEAK,
    )

    outside = mask < 1.0e-4
    np.testing.assert_allclose(repaired[outside], bad[outside], atol=1.0e-6)


def test_region_repair_row_accepts_synthetic_ground_truth() -> None:
    validate_region_repair_row(
        {
            "row_id": "rr000002",
            "label_type": LABEL_TYPE_SYNTHETIC_CLEAN,
            "is_ground_truth": True,
            "issue_type": ISSUE_VOCAL_LEAK,
            "target_stem": "drums",
            "region": {"start_sec": 0.1, "end_sec": 0.2},
        }
    )


def test_dataset_train_and_apply_pipeline_writes_diagnostic_repair(tmp_path: Path) -> None:
    clean = _stereo_widen(_sine(150.0, seconds=0.6, amp=0.17))
    mix = clean + _sine(330.0, seconds=0.6, amp=0.05)
    clean_path = write_audio(tmp_path / "src" / "clean.flac", clean, SAMPLE_RATE)
    mix_path = write_audio(tmp_path / "src" / "mix.flac", mix, SAMPLE_RATE)

    dataset_manifest = build_stem_region_repair_dataset(
        clean_stem_path=clean_path,
        mix_path=mix_path,
        output_dir=tmp_path / "ds",
        stem="drums",
        issue_type=ISSUE_VOCAL_LEAK,
        start_sec=0.14,
        end_sec=0.42,
    )
    model_manifest = train_stem_region_repair_model(
        rows_jsonl=Path(dataset_manifest["rows_jsonl"]),
        output_dir=tmp_path / "model",
        config=StemRegionRepairTrainConfig(epochs=80, hidden_channels=20, learning_rate=0.01, device="cpu"),
    )
    apply_manifest = apply_stem_region_repair(
        checkpoint_path=Path(model_manifest["checkpoint"]),
        mix_path=Path(dataset_manifest["rows_jsonl"]).parent / "audio" / "mix.flac",
        bad_stem_path=Path(dataset_manifest["rows_jsonl"]).parent / "audio" / "bad.flac",
        output_dir=tmp_path / "apply",
        stem="drums",
        issue_type=ISSUE_VOCAL_LEAK,
        start_sec=0.14,
        end_sec=0.42,
    )

    repaired, _ = read_audio_stereo(apply_manifest["repaired_stem_path"])
    bad, _ = read_audio_stereo(Path(dataset_manifest["rows_jsonl"]).parent / "audio" / "bad.flac")
    target, _ = read_audio_stereo(Path(dataset_manifest["rows_jsonl"]).parent / "audio" / "target.flac")
    assert apply_manifest["delivery_allowed"] is False
    assert apply_manifest["diagnostics"]["changed_outside_ratio"] == pytest.approx(0.0, abs=1.0e-7)
    assert _mse(repaired, target) < _mse(bad, target)
