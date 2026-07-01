from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools import train_melband_pitch_attribution_distillation as train_guard_distill
from tools.melband_pitch_attribution_distillation import (
    DISTILLATION_OBJECTIVE_NAME,
    PitchAttributionDistillationConfig,
    apply_pitch_attribution_distillation,
    run_pitch_attribution_distillation_training,
    train_pitch_attribution_distillation_from_arrays,
    validate_pitch_attribution_distillation_label_contract,
)


def _stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1).astype(np.float32)


def _non_probe_proportional_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_rate = 8_000
    frames = sample_rate // 2
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    vocals = _stereo(0.035 * np.sin(2.0 * np.pi * 330.0 * time))
    clean_instrumental = _stereo(0.055 * np.sin(2.0 * np.pi * 110.0 * time))
    source_probe = _stereo(np.sin(2.0 * np.pi * 185.0 * time))
    harmonic_probe = _stereo(np.sin(2.0 * np.pi * 370.0 * time + 0.4))
    mask = np.zeros(frames, dtype=np.float32)
    mask[frames // 5 : frames * 4 // 5] = 1.0
    target_delta = (0.045 * source_probe + 0.035 * harmonic_probe) * mask[:, None]
    instrumental = clean_instrumental + target_delta
    mix = vocals + instrumental
    return mix, vocals, instrumental.astype(np.float32), source_probe, mask, target_delta.astype(np.float32)


def _scalar_projection_mse(target_delta: np.ndarray, source_probe: np.ndarray, mask: np.ndarray) -> float:
    masked_target = target_delta * mask[:, None]
    masked_probe = source_probe * mask[:, None]
    numerator = float(np.sum(masked_target * masked_probe, dtype=np.float64))
    denominator = float(np.sum(masked_probe * masked_probe, dtype=np.float64)) + 1.0e-12
    projected = (numerator / denominator) * masked_probe
    return float(np.mean(np.square((projected - target_delta) * mask[:, None]), dtype=np.float64))


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int = 8_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")
    return path


def test_distillation_status_uses_uncapped_guard_failure_count() -> None:
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

    status, root_cause, next_action, improved = train_guard_distill._status_for_scores(before_score, after_score)

    assert status == "trained_distillation_model_gap"
    assert root_cause == "model_gap"
    assert next_action == "change_separator_objective_or_finetune_separator"
    assert improved is False


def test_distillation_status_derives_best_score_from_region_only_guard_score() -> None:
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

    status, root_cause, next_action, improved = train_guard_distill._status_for_scores(before_score, after_score)

    assert status == "trained_distillation_improved_still_blocked"
    assert root_cause == "distillation_gap"
    assert next_action == "continue_distillation_training"
    assert improved is True


def test_distillation_status_does_not_treat_score_drop_only_as_improvement() -> None:
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

    status, root_cause, next_action, improved = train_guard_distill._status_for_scores(before_score, after_score)

    assert status == "trained_distillation_model_gap"
    assert root_cause == "model_gap"
    assert next_action == "change_separator_objective_or_finetune_separator"
    assert improved is False


def _write_guard(root: Path) -> Path:
    mix, vocals, instrumental, _source_probe, _mask, _target_delta = _non_probe_proportional_case()
    audio_dir = root / "audio"
    v_path = _write_audio(audio_dir / "v.flac", vocals)
    i_path = _write_audio(audio_dir / "i.flac", instrumental)
    raw_path = _write_audio(audio_dir / "raw.flac", mix)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "audio": {"vocals": str(v_path), "instrumental": str(i_path), "raw_ref": str(raw_path)},
                "score": {
                    "verdict": "fail",
                    "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
                    "regions": [
                        {
                            "start": 0.12,
                            "end": 0.36,
                            "semitones": -12.0,
                            "score": 0.8,
                            "fail_mode": "standard_score",
                        }
                    ],
                    "best_probe": {"score": 0.8},
                    "config": {"window_sec": 0.25, "hop_sec": 0.125},
                },
                "delivery_allowed": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def test_distillation_learns_non_probe_proportional_delta() -> None:
    mix, vocals, instrumental, source_probe, mask, target_delta = _non_probe_proportional_case()
    scalar_mse = _scalar_projection_mse(target_delta, source_probe, mask)
    config = PitchAttributionDistillationConfig(
        epochs=180,
        learning_rate=0.02,
        hidden_channels=24,
        kernel_size=17,
        device="cpu",
        seed=123,
    )

    model, diagnostics = train_pitch_attribution_distillation_from_arrays(
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=source_probe,
        region_mask=mask,
        target_delta_audio=target_delta,
        config=config,
    )
    repaired_vocals, repaired_instrumental, delta, apply_diag = apply_pitch_attribution_distillation(
        model=model,
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=source_probe,
        region_mask=mask,
        config=config,
    )
    distill_mse = float(np.mean(np.square((delta - target_delta) * mask[:, None]), dtype=np.float64))

    assert diagnostics["objective"] == DISTILLATION_OBJECTIVE_NAME
    assert diagnostics["training_level"] == "separator_side_distillation"
    assert diagnostics["weights_changed"] is True
    assert scalar_mse > 2.0e-4
    assert distill_mse < scalar_mse * 0.35
    assert np.max(np.abs((repaired_vocals + repaired_instrumental) - (vocals + instrumental))) < 1.0e-7
    assert apply_diag["delivery_allowed"] is False
    assert apply_diag["listen_pack_created"] is False


def test_distillation_training_manifest_is_diagnostic_only(tmp_path: Path) -> None:
    mix, vocals, instrumental, source_probe, mask, target_delta = _non_probe_proportional_case()

    manifest = run_pitch_attribution_distillation_training(
        output_dir=tmp_path / "dist",
        mix_audio=mix,
        vocals_audio=vocals,
        instrumental_audio=instrumental,
        source_probe_audio=source_probe,
        region_mask=mask,
        target_delta_audio=target_delta,
        label_type="pseudo_teacher",
        is_ground_truth=False,
        config=PitchAttributionDistillationConfig(epochs=80, learning_rate=0.02, device="cpu", seed=77),
    )

    assert manifest["objective"] == DISTILLATION_OBJECTIVE_NAME
    assert manifest["training_level"] == "separator_side_distillation"
    assert manifest["must_train_weights"] is True
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["usable_for_promotion"] is False
    assert manifest["contract"]["cannot_override_gates"] is True
    assert manifest["audio"]["vocals"].endswith("audio/v.flac")
    assert manifest["audio"]["instrumental"].endswith("audio/i.flac")
    assert Path(manifest["checkpoint_path"]).exists()

    disk_manifest = json.loads((tmp_path / "dist" / "manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["delivery_allowed"] is False
    assert disk_manifest["presentation_status"] == "training_diagnostic_only"


def test_distillation_rejects_pseudo_teacher_ground_truth() -> None:
    with pytest.raises(ValueError, match="pseudo_teacher.*ground truth"):
        validate_pitch_attribution_distillation_label_contract(label_type="pseudo_teacher", is_ground_truth=True)


def test_guard_distillation_trainer_writes_diagnostic_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "pass" if is_after else "fail",
            "blocker_reasons": [] if is_after else ["pitch_shifted_vocal_in_instrumental"],
            "regions": []
            if is_after
            else [{"start": 0.12, "end": 0.36, "semitones": -12.0, "score": 0.8, "fail_mode": "standard_score"}],
            "best_probe": {"score": 0.1 if is_after else 0.8},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    monkeypatch.setattr(train_guard_distill, "score_pitch_shifted_vocal_leak", fake_score)

    manifest = train_guard_distill.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        epochs=30,
        require_mps=False,
        device="cpu",
        target_gain=1.0,
        separator_residual_mix=0.25,
    )

    assert manifest["objective"] == DISTILLATION_OBJECTIVE_NAME
    assert manifest["training_level"] == "separator_side_distillation"
    assert manifest["status"] == "trained_distillation_passed_pitch_guard"
    assert manifest["weights_changed"] is True
    assert manifest["label_type"] == "pseudo_teacher"
    assert manifest["is_ground_truth"] is False
    assert manifest["usable_for_promotion"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["audio"]["vocals"].endswith("audio/v.flac")
    assert manifest["audio"]["instrumental"].endswith("audio/i.flac")
    assert Path(manifest["checkpoint"]).exists()
