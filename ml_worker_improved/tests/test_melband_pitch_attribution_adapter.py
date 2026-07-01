from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.melband_pitch_attribution_adapter import (
    PitchAttributionAdapterConfig,
    apply_pitch_attribution_adapter,
    estimate_region_projection_gains,
    run_pitch_attribution_adapter_training,
    train_pitch_attribution_adapter_from_arrays,
    validate_pitch_attribution_label_contract,
)
from tools import train_melband_pitch_attribution_adapter as train_guard_adapter
from tools.train_melband_pitch_attribution_adapter import build_regionwise_target_delta


def _stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1).astype(np.float32)


def _projection_energy(audio: np.ndarray, probe: np.ndarray, mask: np.ndarray) -> float:
    masked_audio = audio * mask[:, None]
    masked_probe = probe * mask[:, None]
    numerator = float(np.sum(masked_audio * masked_probe, dtype=np.float64))
    denominator = float(np.sum(masked_probe * masked_probe, dtype=np.float64)) + 1.0e-12
    coeff = numerator / denominator
    return float(coeff * coeff * denominator)


def _synthetic_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_rate = 8_000
    frames = sample_rate // 2
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    clean_vocal = _stereo(0.08 * np.sin(2.0 * np.pi * 330.0 * time))
    clean_instrumental = _stereo(0.16 * np.sin(2.0 * np.pi * 110.0 * time))
    leak = _stereo(0.11 * np.sin(2.0 * np.pi * 185.0 * time))
    mask = np.zeros(frames, dtype=np.float32)
    mask[frames // 4 : frames // 2] = 1.0
    leak_region = leak * mask[:, None]
    bad_vocals = clean_vocal.copy()
    bad_instrumental = clean_instrumental + leak_region
    mix = bad_vocals + bad_instrumental
    return mix, bad_vocals, bad_instrumental, leak, mask, leak_region


def _two_region_variable_leak_case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
]:
    sample_rate = 8_000
    frames = sample_rate // 2
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    clean_vocal = _stereo(0.04 * np.sin(2.0 * np.pi * 330.0 * time))
    clean_instrumental = _stereo(0.08 * np.sin(2.0 * np.pi * 110.0 * time))
    probe = _stereo(np.sin(2.0 * np.pi * 185.0 * time))
    regions = [(frames // 8, frames // 4), (frames // 2, frames * 3 // 4)]
    mask = np.zeros(frames, dtype=np.float32)
    target_delta = np.zeros_like(clean_vocal)
    gains = [0.035, 0.16]
    for (start, end), gain in zip(regions, gains):
        mask[start:end] = 1.0
        target_delta[start:end] = gain * probe[start:end]
    instrumental = clean_instrumental + target_delta
    mix = clean_vocal + instrumental
    return mix, clean_vocal, instrumental, probe, mask, target_delta.astype(np.float32), regions


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int = 8_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")
    return path


def test_adapter_status_uses_uncapped_guard_failure_count() -> None:
    before_score = {
        "verdict": "fail",
        "regions": [{"score": 0.80}],
        "total_fail_regions": 5,
        "best_probe": {"score": 0.80},
    }
    after_score = {
        "verdict": "fail",
        "regions": [{"score": 0.70}],
        "total_fail_regions": 8,
        "best_probe": {"score": 0.70},
    }

    status, root_cause, next_action, improved = train_guard_adapter._status_for_scores(before_score, after_score)

    assert status == "trained_adapter_model_gap"
    assert root_cause == "model_gap"
    assert next_action == "change_separator_objective_or_architecture"
    assert improved is False


def test_adapter_status_derives_best_score_from_region_only_guard_score() -> None:
    before_score = {
        "verdict": "fail",
        "regions": [{"score": 0.80}, {"score": 0.72}],
        "config": {"window_sec": 0.25, "hop_sec": 0.125},
    }
    after_score = {
        "verdict": "fail",
        "regions": [{"score": 0.62}],
        "config": {"window_sec": 0.25, "hop_sec": 0.125},
    }

    status, root_cause, next_action, improved = train_guard_adapter._status_for_scores(before_score, after_score)

    assert status == "trained_adapter_improved_still_blocked"
    assert root_cause == "adapter_gap"
    assert next_action == "continue_separator_adapter_training"
    assert improved is True


def test_adapter_status_does_not_treat_score_drop_only_as_improvement() -> None:
    before_score = {
        "verdict": "fail",
        "regions": [{"score": 0.80}],
        "config": {"window_sec": 0.25, "hop_sec": 0.125},
    }
    after_score = {
        "verdict": "fail",
        "regions": [{"score": 0.62}],
        "config": {"window_sec": 0.25, "hop_sec": 0.125},
    }

    status, root_cause, next_action, improved = train_guard_adapter._status_for_scores(before_score, after_score)

    assert status == "trained_adapter_model_gap"
    assert root_cause == "model_gap"
    assert next_action == "change_separator_objective_or_architecture"
    assert improved is False


def test_adapter_updates_weights_and_moves_pitch_source_from_instrumental_to_vocals() -> None:
    mix, vocals, instrumental, probe, mask, target_delta = _synthetic_case()
    config = PitchAttributionAdapterConfig(
        epochs=120,
        learning_rate=0.08,
        attribution_loss_weight=1.0,
        outside_loss_weight=0.4,
        device="cpu",
        seed=17,
    )

    model, diagnostics = train_pitch_attribution_adapter_from_arrays(
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        target_delta_audio=target_delta,
        config=config,
    )
    repaired_vocals, repaired_instrumental, delta, apply_diagnostics = apply_pitch_attribution_adapter(
        model=model,
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        config=config,
    )

    before_energy = _projection_energy(instrumental, probe, mask)
    after_energy = _projection_energy(repaired_instrumental, probe, mask)
    vocal_gain_energy = _projection_energy(repaired_vocals - vocals, probe, mask)

    assert diagnostics["weights_changed"] is True
    assert diagnostics["final_loss"] < diagnostics["initial_loss"] * 0.2
    assert after_energy < before_energy * 0.2
    assert vocal_gain_energy > before_energy * 0.5
    assert float(np.mean(np.square((delta - target_delta) * mask[:, None]))) < 2.5e-4
    assert apply_diagnostics["delivery_allowed"] is False


def test_adapter_preserves_vocal_plus_instrumental_mixture_by_construction() -> None:
    mix, vocals, instrumental, probe, mask, target_delta = _synthetic_case()
    config = PitchAttributionAdapterConfig(epochs=80, learning_rate=0.08, device="cpu", seed=23)
    model, _diagnostics = train_pitch_attribution_adapter_from_arrays(
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        target_delta_audio=target_delta,
        config=config,
    )

    repaired_vocals, repaired_instrumental, _delta, diagnostics = apply_pitch_attribution_adapter(
        model=model,
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        config=config,
    )

    assert np.max(np.abs((repaired_vocals + repaired_instrumental) - (vocals + instrumental))) < 1.0e-7
    assert diagnostics["mixture_consistency_max_abs_error"] < 1.0e-7


def test_adapter_keeps_clean_outside_region_unchanged() -> None:
    mix, vocals, instrumental, probe, mask, target_delta = _synthetic_case()
    config = PitchAttributionAdapterConfig(epochs=80, learning_rate=0.08, outside_loss_weight=1.0, device="cpu")
    model, _diagnostics = train_pitch_attribution_adapter_from_arrays(
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        target_delta_audio=target_delta,
        config=config,
    )
    repaired_vocals, repaired_instrumental, _delta, diagnostics = apply_pitch_attribution_adapter(
        model=model,
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        config=config,
    )

    outside = (1.0 - mask)[:, None]
    assert float(np.max(np.abs((repaired_vocals - vocals) * outside))) < 1.0e-6
    assert float(np.max(np.abs((repaired_instrumental - instrumental) * outside))) < 1.0e-6
    assert diagnostics["outside_change_max_abs"] < 1.0e-6


def test_training_manifest_is_diagnostic_and_never_delivery(tmp_path: Path) -> None:
    mix, vocals, instrumental, probe, mask, target_delta = _synthetic_case()
    manifest = run_pitch_attribution_adapter_training(
        output_dir=tmp_path / "out",
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        target_delta_audio=target_delta,
        label_type="synthetic_verified",
        is_ground_truth=True,
        config=PitchAttributionAdapterConfig(epochs=60, learning_rate=0.08, device="cpu"),
    )

    assert manifest["objective"] == "melband_pitch_attribution_separator_objective_v1"
    assert manifest["training_level"] == "separator_adapter"
    assert manifest["must_train_weights"] is True
    assert manifest["weights_changed"] is True
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "training_diagnostic_only"
    assert Path(manifest["checkpoint_path"]).exists()
    disk_manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["delivery_allowed"] is False


def test_pseudo_teacher_label_cannot_be_marked_ground_truth() -> None:
    with pytest.raises(ValueError, match="pseudo_teacher.*ground truth"):
        validate_pitch_attribution_label_contract(label_type="pseudo_teacher", is_ground_truth=True)


def test_conditioned_adapter_learns_different_gains_per_region() -> None:
    mix, vocals, instrumental, probe, mask, target_delta, regions = _two_region_variable_leak_case()
    config = PitchAttributionAdapterConfig(
        adapter_mode="conditioned",
        epochs=180,
        learning_rate=0.018,
        conditioned_hidden_channels=12,
        attribution_loss_weight=1.0,
        outside_loss_weight=0.6,
        device="cpu",
        seed=91,
    )

    model, diagnostics = train_pitch_attribution_adapter_from_arrays(
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        target_delta_audio=target_delta,
        config=config,
    )
    repaired_vocals, repaired_instrumental, delta, apply_diagnostics = apply_pitch_attribution_adapter(
        model=model,
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=probe,
        region_mask=mask,
        config=config,
    )
    estimated_gains = estimate_region_projection_gains(delta_audio=delta, source_probe_audio=probe, regions=regions)

    assert diagnostics["adapter_mode"] == "conditioned"
    assert diagnostics["weights_changed"] is True
    assert diagnostics["final_loss"] < diagnostics["initial_loss"] * 0.2
    assert estimated_gains[0] == pytest.approx(0.035, abs=0.018)
    assert estimated_gains[1] == pytest.approx(0.16, abs=0.025)
    assert estimated_gains[1] > estimated_gains[0] * 2.5
    assert np.max(np.abs((repaired_vocals + repaired_instrumental) - (vocals + instrumental))) < 1.0e-7
    assert apply_diagnostics["delivery_allowed"] is False


def test_guard_trainer_writes_audio_manifest_and_blocks_delivery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mix, vocals, instrumental, _probe, _mask, _target_delta = _synthetic_case()
    guard = tmp_path / "guard"
    audio_dir = guard / "audio"
    v_path = _write_audio(audio_dir / "v.flac", vocals)
    i_path = _write_audio(audio_dir / "i.flac", instrumental)
    raw_path = _write_audio(audio_dir / "raw.flac", mix)
    guard_score = {
        "verdict": "fail",
        "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
        "regions": [{"start": 0.125, "end": 0.25, "semitones": -12.0, "score": 0.75}],
        "best_probe": {"score": 0.75, "semitones": -12.0},
        "config": {"window_sec": 0.25, "hop_sec": 0.125, "probe_semitones": [-12.0]},
    }
    (guard / "manifest.json").write_text(
        json.dumps(
            {
                "audio": {
                    "vocals": str(v_path),
                    "instrumental": str(i_path),
                    "raw_ref": str(raw_path),
                },
                "score": guard_score,
                "delivery_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        return {
            "verdict": "pass",
            "blocker_reasons": [],
            "regions": [],
            "best_probe": {"score": 0.05, "semitones": -12.0},
            "config": dict(config or {}),
        }

    monkeypatch.setattr(train_guard_adapter, "score_pitch_shifted_vocal_leak", fake_score)
    manifest = train_guard_adapter.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        epochs=60,
        require_mps=False,
        device="cpu",
        adapter_mode="conditioned",
        probe_semitones=[-12.0],
    )

    assert manifest["objective"] == "melband_pitch_attribution_separator_objective_v1"
    assert manifest["training_level"] == "separator_adapter"
    assert manifest["status"] == "trained_adapter_passed_pitch_guard"
    assert manifest["adapter_mode"] == "conditioned"
    assert manifest["training"]["adapter_mode"] == "conditioned"
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["audio"]["vocals"].endswith("audio/v.flac")
    assert manifest["audio"]["instrumental"].endswith("audio/i.flac")
    assert Path(manifest["audio"]["vocals"]).exists()
    assert Path(manifest["audio"]["instrumental"]).exists()
    assert Path(manifest["checkpoint"]).exists()


def test_regionwise_target_delta_uses_local_projection_coefficients() -> None:
    _mix, _vocals, instrumental, probe, mask, _target_delta, regions = _two_region_variable_leak_case()

    target_delta, coeffs = build_regionwise_target_delta(
        instrumental_audio=instrumental,
        source_condition=probe,
        region_mask=mask,
        regions=[{"start_frame": start, "end_frame": end} for start, end in regions],
        target_gain=1.0,
    )
    estimated = estimate_region_projection_gains(
        delta_audio=target_delta,
        source_probe_audio=probe,
        regions=regions,
    )

    assert len(coeffs) == 2
    assert estimated[0] == pytest.approx(0.035, abs=0.018)
    assert estimated[1] == pytest.approx(0.16, abs=0.025)
    assert estimated[1] > estimated[0] * 2.5


def test_regionwise_target_delta_accepts_second_based_guard_regions() -> None:
    _mix, _vocals, instrumental, probe, mask, _target_delta, regions = _two_region_variable_leak_case()
    sample_rate = 8_000

    target_delta, coeffs = build_regionwise_target_delta(
        instrumental_audio=instrumental,
        source_condition=probe,
        region_mask=mask,
        regions=[
            {"start": float(start) / sample_rate, "end": float(end) / sample_rate}
            for start, end in regions
        ],
        target_gain=1.0,
        sample_rate=sample_rate,
    )
    estimated = estimate_region_projection_gains(
        delta_audio=target_delta,
        source_probe_audio=probe,
        regions=regions,
    )

    assert len(coeffs) == 2
    assert estimated[0] == pytest.approx(0.035, abs=0.018)
    assert estimated[1] == pytest.approx(0.16, abs=0.025)
