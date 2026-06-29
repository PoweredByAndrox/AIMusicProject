from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import tools.build_training_first_root_cause_queue as root_cause_queue
from tools.build_training_first_root_cause_queue import build_root_cause_queue
from tools.build_stem_region_repair_dataset import build_stem_region_repair_dataset
from tools.train_stem_region_repair_model import train_stem_region_repair_model
from tools.apply_stem_region_repair import apply_stem_region_repair
from tools.stem_region_repair_common import (
    LABEL_TYPE_GENERATED_SEPARATOR,
    LABEL_TYPE_SYNTHETIC_CLEAN,
    ISSUE_BAD_RECONSTRUCTION,
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


def _side_channel(audio: np.ndarray) -> np.ndarray:
    return ((audio[:, 0] - audio[:, 1]) * 0.5).astype(np.float32)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    frames = min(a.shape[0], b.shape[0])
    return float(np.mean(np.square(a[:frames] - b[:frames])))


def _source_projection_energy(audio: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    frames = min(audio.shape[0], source.shape[0], mask.shape[0])
    tested = np.mean(audio[:frames], axis=1).astype(np.float64)
    src = np.mean(source[:frames], axis=1).astype(np.float64)
    tested_hp = np.diff(tested)
    src_hp = np.diff(src)
    mask_hp = np.minimum(mask[:frames][1:], mask[:frames][:-1]).astype(np.float64)
    tested_hp *= mask_hp
    src_hp *= mask_hp
    denom = float(np.sum(np.square(src_hp)) + 1.0e-8)
    coeff = float(np.sum(tested_hp * src_hp) / denom)
    projected = coeff * src_hp
    return float(np.sum(np.square(projected)) / max(float(np.sum(mask_hp)), 1.0))


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


def test_failed_stem_region_repair_is_training_evidence_without_repeat_requirement(tmp_path: Path) -> None:
    failed_checkpoint = "/repo/artifacts/stem_eval/t337_recon_synth/model_vocals/model.pt"
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "schema": "fully_automatic_stem_run.v1",
                "candidates": [
                    {
                        "candidate_id": "c103",
                        "window_id": "w000",
                        "verdict": "blocked_bad_reconstruction",
                        "next_action": {"reason": "mix_conservation_residual_too_high"},
                        "target_specific_composition": {
                            "routes": [
                                {"lane": "stem_lane", "route_id": "htdemucs_6s"},
                                {"lane": "vocal_lane", "route_id": "mdx23c_vocals"},
                                {"lane": "minus_lane", "route_id": "mdx23c_vocals:instrumental"},
                            ],
                            "repair": {
                                "repair_id": "srseq",
                                "stem_region_repair": {
                                    "applied": True,
                                    "repair_owner": "runner_stem_region_repair_sequence_v1",
                                    "repairs": [
                                        {
                                            "applied": True,
                                            "stem": "vocals",
                                            "issue_type": ISSUE_BAD_RECONSTRUCTION,
                                            "checkpoint": failed_checkpoint,
                                            "diagnostics": {
                                                "target_issue_improved": False,
                                                "delivery_pair_residual_before": 0.005,
                                                "delivery_pair_residual_after": 0.026,
                                            },
                                        }
                                    ],
                                },
                            },
                        },
                        "stem_paths": {"vocals": "/tmp/vocals_bad.flac"},
                    }
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
    assert rows[0]["root_cause"] == "repair_gap"
    assert rows[0]["training_state"] == "data_gap"
    assert rows[0]["issue_type"] == ISSUE_BAD_RECONSTRUCTION
    assert rows[0]["affected_stem"] == "vocals"
    assert rows[0]["training_action"] == "retrain_failed_repair_checkpoint"
    assert rows[0]["blocker_reason"] == "stem_region_repair_not_improved:srseq:vocals:bad_reconstruction"
    assert rows[0]["failed_repair_id"] == "srseq"
    assert rows[0]["failed_repair_checkpoint"] == failed_checkpoint
    assert rows[0]["repeat_count"] == 1


def test_failed_repair_stem_is_not_overwritten_by_residual_localization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        root_cause_queue,
        "localize_candidate_reconstruction_residual",
        lambda _candidate: {
            "verdict": "localized",
            "affected_stem": "vocals",
            "residual_region": {"start_sec": 0.0, "end_sec": 1.0},
        },
    )
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "schema": "fully_automatic_stem_run.v1",
                "candidates": [
                    {
                        "candidate_id": "c103",
                        "window_id": "w000",
                        "verdict": "blocked_bad_reconstruction",
                        "next_action": {"reason": "mix_conservation_residual_too_high"},
                        "target_specific_composition": {
                            "repair": {
                                "repair_id": "srseq",
                                "stem_region_repair": {
                                    "repairs": [
                                        {
                                            "applied": True,
                                            "stem": "other",
                                            "issue_type": ISSUE_BAD_RECONSTRUCTION,
                                            "checkpoint": "/repo/artifacts/model_other.pt",
                                            "diagnostics": {"target_issue_improved": False},
                                        }
                                    ]
                                },
                            }
                        },
                    }
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

    assert rows[0]["affected_stem"] == "other"
    assert rows[0]["training_action"] == "retrain_failed_repair_checkpoint"
    assert rows[0]["residual_localization"]["affected_stem"] == "vocals"


def test_root_cause_queue_ingests_standalone_repair_probe_manifest(tmp_path: Path) -> None:
    checkpoint = "/repo/artifacts/stem_eval/t337_recon_synth/model_vocals/model.pt"
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"schema": "fully_automatic_stem_run.v1", "candidates": []}), encoding="utf-8")
    probe_manifest = tmp_path / "probe" / "manifest.json"
    probe_manifest.parent.mkdir(parents=True)
    probe_manifest.write_text(
        json.dumps(
            {
                "schema": "stem_region_repair_probe.v1",
                "source_candidate": "/repo/artifacts/stem_eval/t342_recon_comp_fix/w000/c102",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_probe_only",
                "repair": {
                    "applied": True,
                    "repair_owner": "runner_stem_region_repair_sequence_v1",
                    "repairs": [
                        {
                            "applied": True,
                            "stem": "vocals",
                            "issue_type": ISSUE_BAD_RECONSTRUCTION,
                            "checkpoint": checkpoint,
                            "diagnostics": {
                                "target_issue_improved": False,
                                "delivery_pair_residual_before": 0.005,
                                "delivery_pair_residual_after": 0.026,
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_root_cause_queue(
        run_json_path=run_json,
        output_dir=tmp_path / "queue",
        min_repeats=2,
        probe_manifest_paths=[probe_manifest],
    )
    rows = [
        json.loads(line)
        for line in Path(manifest["queue_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["probe_manifest_count"] == 1
    assert manifest["queue_row_count"] == 1
    assert rows[0]["candidate_ids"] == ["probe_manifest"]
    assert rows[0]["root_cause"] == "repair_gap"
    assert rows[0]["training_action"] == "retrain_failed_repair_checkpoint"
    assert rows[0]["failed_repair_checkpoint"] == checkpoint
    assert rows[0]["source_probe_manifest"] == str(probe_manifest)
    assert rows[0]["source_candidate"] == "/repo/artifacts/stem_eval/t342_recon_comp_fix/w000/c102"


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


def test_reconstruction_aware_repair_reduces_pair_residual() -> None:
    clean = _stereo_widen(_sine(120.0, amp=0.14))
    body = _sine(180.0, amp=0.045)
    sibling = _sine(260.0, amp=0.09)
    mask = make_region_mask(frame_count=clean.shape[0], sample_rate=SAMPLE_RATE, start_sec=0.10, end_sec=0.40)
    target = clean + body * mask[:, None]
    bad = clean
    mix = target + sibling

    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=target,
        sibling_stem_audio=sibling,
        target_mix_audio=mix,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type="bad_reconstruction",
        config=StemRegionRepairTrainConfig(
            epochs=100,
            hidden_channels=24,
            learning_rate=0.01,
            mix_conservation_loss_weight=1.0,
            device="cpu",
        ),
    )
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type="bad_reconstruction",
    )

    before_target = _mse_region(bad, target, mask)
    after_target = _mse_region(repaired, target, mask)
    before_pair = _mse_region(bad + sibling, mix, mask)
    after_pair = _mse_region(repaired + sibling, mix, mask)
    assert history["loss_terms_enabled"]["mix_conservation"] is True
    assert after_target < before_target * 0.45
    assert after_pair < before_pair * 0.45
    assert diagnostics["changed_outside_ratio"] == pytest.approx(0.0, abs=1.0e-7)


def test_reconstruction_aware_repair_preserves_stereo_side_channel() -> None:
    clean_side = _stereo_widen(_sine(125.0, amp=0.13), side_amp=0.06)
    mono_detail = _sine(430.0, amp=0.035)
    sibling = _sine(260.0, amp=0.08)
    mask = make_region_mask(frame_count=clean_side.shape[0], sample_rate=SAMPLE_RATE, start_sec=0.10, end_sec=0.40)
    target = clean_side + mono_detail * mask[:, None]
    bad = clean_side.copy()
    mix = target + sibling

    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=target,
        sibling_stem_audio=sibling,
        target_mix_audio=mix,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_BAD_RECONSTRUCTION,
        config=StemRegionRepairTrainConfig(
            epochs=120,
            hidden_channels=24,
            learning_rate=0.01,
            mix_conservation_loss_weight=1.0,
            stereo_preserve_loss_weight=8.0,
            device="cpu",
        ),
    )
    repaired, _ = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_BAD_RECONSTRUCTION,
    )

    before_pair = _mse_region(bad + sibling, mix, mask)
    after_pair = _mse_region(repaired + sibling, mix, mask)
    side_change = _mse_region(_side_channel(repaired)[:, None], _side_channel(bad)[:, None], mask)

    assert after_pair < before_pair * 0.45
    assert side_change < 1.0e-5
    assert history["loss_terms_enabled"]["stereo_preserve"] is True


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


def test_forensic_source_decorrelation_loss_reduces_aligned_drum_projection() -> None:
    seconds = 0.5
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    clean_vocal = np.stack(
        [
            0.055 * np.sin(2.0 * math.pi * 430.0 * t),
            0.05 * np.sin(2.0 * math.pi * 435.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    drum_source = np.zeros_like(clean_vocal)
    burst = ((t >= 0.16) & (t <= 0.34)).astype(np.float32)
    drum_source[:, 0] = 0.08 * burst * np.sin(2.0 * math.pi * 2600.0 * t)
    drum_source[:, 1] = 0.08 * burst * np.sin(2.0 * math.pi * 2640.0 * t)
    mask = make_region_mask(frame_count=clean_vocal.shape[0], sample_rate=SAMPLE_RATE, start_sec=0.12, end_sec=0.38)
    bad_vocal = clean_vocal + (0.45 * drum_source * mask[:, None])
    mix = bad_vocal + 0.2 * drum_source

    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad_vocal,
        target_stem_audio=bad_vocal,
        forensic_source_audio=drum_source,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type="instrument_leak",
        config=StemRegionRepairTrainConfig(
            epochs=160,
            hidden_channels=24,
            dilated_layers=2,
            learning_rate=0.006,
            region_loss_weight=0.0,
            outside_loss_weight=0.05,
            forensic_source_decorrelation_loss_weight=250.0,
            forensic_source_envelope_loss_weight=80.0,
            forensic_source_active_energy_loss_weight=40.0,
            forensic_envelope_window_ms=12.0,
            forensic_source_conditioning=True,
            max_delta=0.12,
            device="cpu",
        ),
    )
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad_vocal,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type="instrument_leak",
        source_condition_audio=drum_source,
    )

    before_projection = _source_projection_energy(bad_vocal, drum_source, mask)
    after_projection = _source_projection_energy(repaired, drum_source, mask)
    assert history["loss_terms_enabled"]["forensic_source_decorrelation"] is True
    assert history["loss_terms_enabled"]["forensic_source_envelope"] is True
    assert history["loss_terms_enabled"]["forensic_source_active_energy"] is True
    assert history["loss_terms_enabled"]["forensic_source_conditioning"] is True
    assert history["final_loss"] < history["initial_loss"]
    assert after_projection < before_projection * 0.55
    assert diagnostics["changed_outside_ratio"] == pytest.approx(0.0, abs=1.0e-7)
    assert diagnostics["source_conditioned"] is True


def test_train_stem_region_repair_model_loads_forensic_source_path(tmp_path: Path) -> None:
    seconds = 0.35
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    clean = _stereo_widen(_sine(430.0, seconds=seconds, amp=0.05))
    source = np.zeros_like(clean)
    burst = ((t >= 0.10) & (t <= 0.24)).astype(np.float32)
    source[:, 0] = 0.08 * burst * np.sin(2.0 * math.pi * 2600.0 * t)
    source[:, 1] = 0.08 * burst * np.sin(2.0 * math.pi * 2640.0 * t)
    bad = clean + 0.4 * source
    mix = bad + 0.2 * source
    mix_path = write_audio(tmp_path / "mix.flac", mix, SAMPLE_RATE)
    bad_path = write_audio(tmp_path / "bad.flac", bad, SAMPLE_RATE)
    target_path = write_audio(tmp_path / "target.flac", bad, SAMPLE_RATE)
    source_path = write_audio(tmp_path / "drums.flac", source, SAMPLE_RATE)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "row_id": "rr_forensic",
                "label_type": LABEL_TYPE_SYNTHETIC_CLEAN,
                "is_ground_truth": True,
                "issue_type": "instrument_leak",
                "target_stem": "vocals",
                "mix_path": str(mix_path),
                "bad_stem_path": str(bad_path),
                "target_stem_path": str(target_path),
                "forensic_source_path": str(source_path),
                "region": {"start_sec": 0.08, "end_sec": 0.28},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = train_stem_region_repair_model(
        rows_jsonl=rows_path,
        output_dir=tmp_path / "model",
        config=StemRegionRepairTrainConfig(
            epochs=20,
            hidden_channels=12,
            learning_rate=0.006,
            region_loss_weight=0.0,
            outside_loss_weight=0.05,
            forensic_source_decorrelation_loss_weight=10.0,
            forensic_source_envelope_loss_weight=5.0,
            forensic_source_conditioning=True,
            device="cpu",
        ),
    )
    history = json.loads(Path(manifest["history_json"]).read_text(encoding="utf-8"))

    assert history["loss_terms_enabled"]["forensic_source_decorrelation"] is True
    assert history["loss_terms_enabled"]["forensic_source_envelope"] is True
    assert history["loss_terms_enabled"]["forensic_source_conditioning"] is True


def test_forensic_only_training_allows_non_ground_truth_when_target_loss_disabled(tmp_path: Path) -> None:
    seconds = 0.3
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    bad = _stereo_widen(_sine(430.0, seconds=seconds, amp=0.05))
    source = np.zeros_like(bad)
    burst = ((t >= 0.08) & (t <= 0.22)).astype(np.float32)
    source[:, 0] = 0.08 * burst * np.sin(2.0 * math.pi * 2600.0 * t)
    source[:, 1] = 0.08 * burst * np.sin(2.0 * math.pi * 2640.0 * t)
    bad = bad + 0.4 * source
    mix_path = write_audio(tmp_path / "mix.flac", bad + 0.2 * source, SAMPLE_RATE)
    bad_path = write_audio(tmp_path / "bad.flac", bad, SAMPLE_RATE)
    source_path = write_audio(tmp_path / "drums.flac", source, SAMPLE_RATE)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "row_id": "rr_weak_forensic",
                "label_type": "weak_forensic_teacher",
                "is_ground_truth": False,
                "usable_for_promotion": False,
                "issue_type": "instrument_leak",
                "target_stem": "vocals",
                "mix_path": str(mix_path),
                "bad_stem_path": str(bad_path),
                "target_stem_path": str(bad_path),
                "forensic_source_path": str(source_path),
                "region": {"start_sec": 0.06, "end_sec": 0.24},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = train_stem_region_repair_model(
        rows_jsonl=rows_path,
        output_dir=tmp_path / "model",
        config=StemRegionRepairTrainConfig(
            epochs=8,
            hidden_channels=10,
            learning_rate=0.006,
            region_loss_weight=0.0,
            outside_loss_weight=0.05,
            forensic_source_decorrelation_loss_weight=10.0,
            forensic_source_envelope_loss_weight=5.0,
            forensic_source_conditioning=True,
            device="cpu",
        ),
    )

    history = json.loads(Path(manifest["history_json"]).read_text(encoding="utf-8"))
    assert manifest["contract"]["forensic_only_non_ground_truth_training"] is True
    assert history["loss_terms_enabled"]["forensic_source_decorrelation"] is True


def test_weak_teacher_region_loss_requires_explicit_non_promotion_contract(tmp_path: Path) -> None:
    seconds = 0.3
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    bad = _stereo_widen(_sine(430.0, seconds=seconds, amp=0.05))
    source = np.zeros_like(bad)
    burst = ((t >= 0.08) & (t <= 0.22)).astype(np.float32)
    source[:, 0] = 0.08 * burst * np.sin(2.0 * math.pi * 2600.0 * t)
    source[:, 1] = 0.08 * burst * np.sin(2.0 * math.pi * 2640.0 * t)
    bad = bad + 0.4 * source
    target = bad.copy()
    target -= 0.35 * source * burst[:, None]
    mix_path = write_audio(tmp_path / "mix.flac", bad + 0.2 * source, SAMPLE_RATE)
    bad_path = write_audio(tmp_path / "bad.flac", bad, SAMPLE_RATE)
    target_path = write_audio(tmp_path / "target.flac", target, SAMPLE_RATE)
    source_path = write_audio(tmp_path / "drums.flac", source, SAMPLE_RATE)
    teacher_manifest_path = tmp_path / "teacher_manifest.json"
    teacher_manifest_path.write_text(
        json.dumps(
            {
                "schema": "vocal_depercussive_teacher_target_manifest.v1",
                "contract": {
                    "label_type": "weak_forensic_teacher",
                    "is_ground_truth": False,
                    "usable_as_ground_truth": False,
                    "usable_for_training": True,
                    "usable_for_promotion": False,
                    "delivery_allowed": False,
                },
                "outputs": {"target_vocals_path": str(target_path)},
            }
        ),
        encoding="utf-8",
    )
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "row_id": "rr_weak_teacher_region",
                "label_type": "weak_forensic_teacher",
                "is_ground_truth": False,
                "usable_for_promotion": False,
                "issue_type": "instrument_leak",
                "target_stem": "vocals",
                "mix_path": str(mix_path),
                "bad_stem_path": str(bad_path),
                "target_stem_path": str(target_path),
                "forensic_source_path": str(source_path),
                "weak_teacher_manifest_path": str(teacher_manifest_path),
                "region": {"start_sec": 0.06, "end_sec": 0.24},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not ground truth"):
        train_stem_region_repair_model(
            rows_jsonl=rows_path,
            output_dir=tmp_path / "blocked_model",
            config=StemRegionRepairTrainConfig(
                epochs=4,
                hidden_channels=8,
                learning_rate=0.006,
                region_loss_weight=1.0,
                outside_loss_weight=0.05,
                forensic_source_decorrelation_loss_weight=1.0,
                forensic_source_conditioning=True,
                device="cpu",
            ),
        )

    manifest = train_stem_region_repair_model(
        rows_jsonl=rows_path,
        output_dir=tmp_path / "model",
        config=StemRegionRepairTrainConfig(
            epochs=8,
            hidden_channels=10,
            learning_rate=0.006,
            region_loss_weight=1.0,
            outside_loss_weight=0.05,
            forensic_source_decorrelation_loss_weight=1.0,
            forensic_source_conditioning=True,
            allow_weak_teacher_region_loss=True,
            device="cpu",
        ),
    )

    assert manifest["contract"]["weak_teacher_region_loss_training"] is True
    assert manifest["contract"]["forensic_only_non_ground_truth_training"] is False
    assert manifest["shadow_only"] is True
    assert manifest["delivery_allowed"] is False


def test_forensic_source_target_loss_focuses_repair_on_source_active_frames() -> None:
    seconds = 0.25
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    bad = _stereo_widen(_sine(430.0, seconds=seconds, amp=0.04))
    source = np.zeros_like(bad)
    burst = ((t >= 0.10) & (t <= 0.14)).astype(np.float32)
    source[:, 0] = 0.09 * burst * np.sin(2.0 * math.pi * 2600.0 * t)
    source[:, 1] = 0.09 * burst * np.sin(2.0 * math.pi * 2640.0 * t)
    bad = bad + 0.55 * source
    target = bad - 0.50 * source
    mask = make_region_mask(
        frame_count=bad.shape[0],
        sample_rate=SAMPLE_RATE,
        start_sec=0.05,
        end_sec=0.18,
    )
    active_mask = (np.max(np.abs(source), axis=1) > 1.0e-5).astype(np.float32)

    model, history = train_region_repair_model_from_arrays(
        mix_audio=bad,
        bad_stem_audio=bad,
        target_stem_audio=target,
        forensic_source_audio=source,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type="instrument_leak",
        config=StemRegionRepairTrainConfig(
            epochs=1,
            hidden_channels=24,
            learning_rate=0.0,
            region_loss_weight=0.0,
            outside_loss_weight=0.0,
            target_delta_active_loss_weight=40.0,
            forensic_source_target_loss_weight=80.0,
            forensic_source_conditioning=True,
            source_skip_init_from_target=True,
            max_delta=1.0,
            device="cpu",
        ),
    )
    repaired, _diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=bad,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type="instrument_leak",
        source_condition_audio=source,
    )

    before_error = _mse_region(bad, target, active_mask)
    after_error = _mse_region(repaired, target, active_mask)
    assert history["loss_terms_enabled"]["target_delta_active"] is True
    assert history["loss_terms_enabled"]["forensic_source_target"] is True
    assert after_error < before_error * 0.65


def test_apply_stem_region_repair_accepts_forensic_source_for_conditioned_checkpoint(tmp_path: Path) -> None:
    seconds = 0.3
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    bad = _stereo_widen(_sine(430.0, seconds=seconds, amp=0.05))
    source = np.zeros_like(bad)
    burst = ((t >= 0.08) & (t <= 0.22)).astype(np.float32)
    source[:, 0] = 0.08 * burst * np.sin(2.0 * math.pi * 2600.0 * t)
    source[:, 1] = 0.08 * burst * np.sin(2.0 * math.pi * 2640.0 * t)
    bad = bad + 0.4 * source
    mix_path = write_audio(tmp_path / "mix.flac", bad + 0.2 * source, SAMPLE_RATE)
    bad_path = write_audio(tmp_path / "bad.flac", bad, SAMPLE_RATE)
    source_path = write_audio(tmp_path / "drums.flac", source, SAMPLE_RATE)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "row_id": "rr_apply_forensic",
                "label_type": "weak_forensic_teacher",
                "is_ground_truth": False,
                "usable_for_promotion": False,
                "issue_type": "instrument_leak",
                "target_stem": "vocals",
                "mix_path": str(mix_path),
                "bad_stem_path": str(bad_path),
                "target_stem_path": str(bad_path),
                "forensic_source_path": str(source_path),
                "region": {"start_sec": 0.06, "end_sec": 0.24},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trained = train_stem_region_repair_model(
        rows_jsonl=rows_path,
        output_dir=tmp_path / "model",
        config=StemRegionRepairTrainConfig(
            epochs=8,
            hidden_channels=10,
            learning_rate=0.006,
            region_loss_weight=0.0,
            outside_loss_weight=0.05,
            forensic_source_decorrelation_loss_weight=10.0,
            forensic_source_conditioning=True,
            device="cpu",
        ),
    )

    manifest = apply_stem_region_repair(
        checkpoint_path=Path(trained["checkpoint"]),
        mix_path=mix_path,
        bad_stem_path=bad_path,
        output_dir=tmp_path / "applied",
        stem="vocals",
        issue_type="instrument_leak",
        start_sec=0.06,
        end_sec=0.24,
        forensic_source_path=source_path,
    )

    assert manifest["diagnostics"]["source_conditioned"] is True
    assert Path(manifest["repaired_stem_path"]).exists()


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


def test_dataset_row_records_reconstruction_context(tmp_path: Path) -> None:
    clean = _stereo_widen(_sine(170.0, seconds=0.5, amp=0.12))
    sibling = _sine(300.0, seconds=0.5, amp=0.05)
    mix = clean + sibling
    clean_path = write_audio(tmp_path / "src" / "clean.flac", clean, SAMPLE_RATE)
    sibling_path = write_audio(tmp_path / "src" / "minus.flac", sibling, SAMPLE_RATE)
    mix_path = write_audio(tmp_path / "src" / "mix.flac", mix, SAMPLE_RATE)

    manifest = build_stem_region_repair_dataset(
        clean_stem_path=clean_path,
        mix_path=mix_path,
        output_dir=tmp_path / "ds",
        stem="vocals",
        issue_type=ISSUE_VOCAL_LEAK,
        start_sec=0.10,
        end_sec=0.30,
        sibling_stem_paths={"minus": sibling_path},
        root_cause_row_id="rc000042",
    )
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert row["root_cause_row_id"] == "rc000042"
    assert row["sibling_stem_paths"] == {"minus": str(sibling_path)}
    assert row["reconstruction_surface"] == "vocals_plus_minus"
    assert row["target_surface"] == "mix"


def test_bad_reconstruction_dataset_trains_reconstruction_aware_checkpoint(tmp_path: Path) -> None:
    base = _stereo_widen(_sine(155.0, seconds=0.6, amp=0.12))
    detail = _sine(420.0, seconds=0.6, amp=0.035)
    clean = base + detail
    sibling = _sine(90.0, seconds=0.6, amp=0.06)
    mix = clean + sibling
    clean_path = write_audio(tmp_path / "src" / "clean.flac", clean, SAMPLE_RATE)
    detail_path = write_audio(tmp_path / "src" / "detail.flac", detail, SAMPLE_RATE)
    sibling_path = write_audio(tmp_path / "src" / "minus.flac", sibling, SAMPLE_RATE)
    mix_path = write_audio(tmp_path / "src" / "mix.flac", mix, SAMPLE_RATE)

    dataset_manifest = build_stem_region_repair_dataset(
        clean_stem_path=clean_path,
        mix_path=mix_path,
        output_dir=tmp_path / "ds",
        stem="vocals",
        issue_type=ISSUE_BAD_RECONSTRUCTION,
        start_sec=0.12,
        end_sec=0.44,
        foreign_source_path=detail_path,
        sibling_stem_paths={"minus": sibling_path},
        root_cause_row_id="rc000057",
    )
    model_manifest = train_stem_region_repair_model(
        rows_jsonl=Path(dataset_manifest["rows_jsonl"]),
        output_dir=tmp_path / "model",
        config=StemRegionRepairTrainConfig(
            epochs=90,
            hidden_channels=20,
            learning_rate=0.01,
            mix_conservation_loss_weight=1.0,
            stereo_preserve_loss_weight=4.0,
            device="cpu",
        ),
    )

    row = json.loads(Path(dataset_manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])
    history = json.loads(Path(model_manifest["history_json"]).read_text(encoding="utf-8"))
    assert row["issue_type"] == ISSUE_BAD_RECONSTRUCTION
    assert row["corruption"]["corruption"] == "region_missing_reconstruction_detail"
    assert model_manifest["reconstruction_aware"] is True
    assert history["loss_terms_enabled"]["mix_conservation"] is True
    assert history["loss_terms_enabled"]["stereo_preserve"] is True
    assert history["region"] == {"start_sec": 0.12, "end_sec": 0.44}
    assert history["training_row"]["root_cause_row_id"] == "rc000057"
    assert history["training_row"]["target_stem"] == "vocals"
