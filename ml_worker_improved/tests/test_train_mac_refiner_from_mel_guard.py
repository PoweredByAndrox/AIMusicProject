from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tools import train_mac_refiner_from_mel_guard as train_guard


SAMPLE_RATE = 16_000


def _write_audio(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def _write_guard(root: Path) -> Path:
    frames = SAMPLE_RATE // 2
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    vocals = np.stack([0.05 * np.sin(2.0 * np.pi * 220.0 * t)] * 2, axis=1)
    instrumental = np.stack([0.03 * np.sin(2.0 * np.pi * 330.0 * t)] * 2, axis=1)
    raw = vocals + instrumental
    audio_dir = root / "audio"
    v = _write_audio(audio_dir / "v.flac", vocals)
    i = _write_audio(audio_dir / "i.flac", instrumental)
    raw_path = _write_audio(audio_dir / "raw.flac", raw)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "audio": {"vocals": str(v), "instrumental": str(i), "raw_ref": str(raw_path)},
                "score": {"config": {"window_sec": 0.25, "hop_sec": 0.125}},
                "delivery_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_mel_guard_training_blocks_without_required_mps(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("cpu"))

    manifest = train_guard.run_training(mel_guard_dir=tmp_path / "missing", output_dir=tmp_path / "out", require_mps=True)

    assert manifest["status"] == "blocked_mps_unavailable"
    assert manifest["delivery_allowed"] is False
    assert manifest["resolved_device"] == "cpu"


def test_mel_guard_training_remains_pseudo_teacher_and_diagnostic(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "pass" if is_after else "fail",
            "blocker_reasons": [] if is_after else ["pitch_shifted_vocal_in_instrumental"],
            "regions": []
            if is_after
            else [{"start": 0.05, "end": 0.30, "semitones": -5.0, "score": 0.8, "fail_mode": "standard_score"}],
            "best_probe": {"score": 0.1 if is_after else 0.8},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        return FakeModel(), {"resolved_device": "mps", "final_loss": 0.001, "loss_history": [0.01, 0.001]}

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(mel_guard_dir=guard, output_dir=tmp_path / "out", require_mps=True)

    assert manifest["label_type"] == "pseudo_teacher"
    assert manifest["is_ground_truth"] is False
    assert manifest["usable_for_promotion"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["contract"]["pseudo_teacher_only"] is True


def test_mel_guard_training_records_multi_probe_source_objective(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    captured: dict[str, object] = {}

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "pass" if is_after else "fail",
            "blocker_reasons": [] if is_after else ["pitch_shifted_vocal_in_instrumental"],
            "regions": []
            if is_after
            else [{"start": 0.05, "end": 0.30, "semitones": -12.0, "score": 0.8, "fail_mode": "standard_score"}],
            "best_probe": {"score": 0.1 if is_after else 0.8},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        captured["forensic_source_audio"] = kwargs["forensic_source_audio"]
        captured["config"] = kwargs["config"]
        return FakeModel(), {"resolved_device": "mps", "final_loss": 0.001, "loss_history": [0.01, 0.001]}

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        probe_semitones=[-12.0, -7.0, 0.0],
        objective_profile="multi_probe_anti_vocal",
    )

    config = captured["config"]
    source = captured["forensic_source_audio"]
    assert manifest["source_condition_mode"] == "multi_probe"
    assert manifest["probe_semitones"] == [-12.0, -7.0, 0.0]
    assert source is not None
    assert config.forensic_source_target_loss_weight > 0.0
    assert config.forensic_source_active_energy_loss_weight > 0.0


def test_mel_guard_training_derives_best_score_from_region_only_guard_score(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    guard_manifest_path = guard / "manifest.json"
    guard_manifest = json.loads(guard_manifest_path.read_text(encoding="utf-8"))
    guard_manifest["score"] = {
        "config": {"window_sec": 0.25, "hop_sec": 0.125},
        "regions": [
            {"start": 0.05, "end": 0.30, "semitones": -5.0, "score": 0.72, "fail_mode": "standard_score"},
            {"start": 0.32, "end": 0.45, "semitones": 0.0, "score": 0.81, "fail_mode": "standard_score"},
        ],
    }
    guard_manifest_path.write_text(json.dumps(guard_manifest), encoding="utf-8")

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        return {
            "verdict": "fail",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
            "regions": [{"start": 0.05, "end": 0.30, "semitones": -5.0, "score": 0.62, "fail_mode": "standard_score"}],
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        return FakeModel(), {"resolved_device": "mps", "final_loss": 0.001, "loss_history": [0.01, 0.001]}

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(mel_guard_dir=guard, output_dir=tmp_path / "out", require_mps=True)

    assert manifest["metrics"]["before_best_score"] == pytest.approx(0.81)
    assert manifest["metrics"]["after_best_score"] == pytest.approx(0.62)


def test_multi_probe_source_condition_preserves_region_semitone_identity(monkeypatch) -> None:
    frames = SAMPLE_RATE
    vocals = np.zeros((frames, 2), dtype=np.float32)
    probe_values = {-12.0: 0.12, 0.0: 0.50, 3.0: 0.90}

    def fake_pitch_shift(_audio: np.ndarray, semitones: float) -> np.ndarray:
        return np.full_like(vocals, probe_values[float(semitones)], dtype=np.float32)

    monkeypatch.setattr(train_guard, "pitch_shift_resample_same_length", fake_pitch_shift)

    region_mask, source_condition, applied_regions, mode, normalized_probes = train_guard._build_training_source_condition(
        vocals_audio=vocals,
        sample_rate=SAMPLE_RATE,
        regions=[
            {"start": 0.20, "end": 0.30, "semitones": -12.0, "score": 0.80},
            {"start": 0.60, "end": 0.70, "semitones": 0.0, "score": 0.75},
        ],
        pad_sec=0.0,
        fade_sec=0.0,
        probe_semitones=[-12.0, 0.0, 3.0],
    )

    first_mid = int(0.25 * SAMPLE_RATE)
    second_mid = int(0.65 * SAMPLE_RATE)
    assert mode == "multi_probe"
    assert normalized_probes == [-12.0, 0.0, 3.0]
    assert region_mask[first_mid] == 1.0
    assert region_mask[second_mid] == 1.0
    assert source_condition[first_mid, 0] == pytest.approx(0.12, abs=1.0e-6)
    assert source_condition[second_mid, 0] == pytest.approx(0.50, abs=1.0e-6)
    assert applied_regions[0]["probe_semitones"] == [-12.0]
    assert applied_regions[1]["probe_semitones"] == [0.0]


def test_pitch_region_count_objective_marks_no_region_reduction_as_model_gap(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    captured: dict[str, object] = {}

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "fail",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
            "regions": [
                {"start": 0.05, "end": 0.30, "semitones": -12.0, "score": 0.80, "fail_mode": "standard_score"},
                {"start": 0.32, "end": 0.45, "semitones": 0.0, "score": 0.75, "fail_mode": "standard_score"},
            ],
            "best_probe": {"score": 0.60 if is_after else 0.80},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        captured["config"] = kwargs["config"]
        captured["forensic_source_audio"] = kwargs["forensic_source_audio"]
        captured["training_frames"] = int(kwargs["region_mask"].shape[0])
        kwargs["progress_callback"](
            {
                "event": "epoch",
                "attempt_index": 0,
                "epoch": 1,
                "epochs": 1,
                "loss": 0.001,
                "best_loss": 0.001,
            }
        )
        return FakeModel(), {"resolved_device": "mps", "final_loss": 0.001, "loss_history": [0.01, 0.001]}

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        max_regions=16,
        probe_semitones=[-12.0, -9.0, -7.0, -5.0, 0.0, 3.0],
        objective_profile="pitch_region_count_attack_v2",
    )

    config = captured["config"]
    assert manifest["status"] == "trained_repair_model_gap"
    assert manifest["root_cause"] == "model_gap"
    assert manifest["next_training_action"] == "change_training_objective_or_separator"
    assert manifest["metrics"]["target_issue_improved"] is False
    assert manifest["metrics"]["region_count_reduced"] is False
    assert manifest["contract"]["requires_region_count_reduction"] is True
    assert manifest["source_condition_mode"] == "multi_probe"
    assert captured["forensic_source_audio"] is not None
    assert config.hidden_channels >= 48
    assert config.dilated_layers >= 2
    assert config.forensic_source_decorrelation_loss_weight >= 1.0
    progress_events = [
        json.loads(line)
        for line in (tmp_path / "out" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    crop_event = next(event for event in progress_events if event["phase"] == "train_crop" and event["status"] == "end")
    assert crop_event["training_frames"] <= crop_event["original_frames"]
    assert captured["training_frames"] == crop_event["training_frames"]
    assert any(event["phase"] == "train" and event.get("event") == "epoch" for event in progress_events)


def test_pitch_region_count_objective_uses_uncapped_failure_count(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "fail",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
            "regions": [{"start": 0.05, "end": 0.30, "semitones": -12.0, "score": 0.80}],
            "total_fail_regions": 12 if is_after else 16,
            "selected_region_count": 1,
            "regions_capped": True,
            "failure_taxonomy": {
                "total_fail_regions": 12 if is_after else 16,
                "selected_region_count": 1,
                "by_fail_mode": {"standard_score": 12 if is_after else 16},
                "by_probe": {"-12.0": 12 if is_after else 16},
            },
            "best_probe": {"score": 0.60 if is_after else 0.80},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        return FakeModel(), {"resolved_device": "mps", "final_loss": 0.001, "loss_history": [0.01, 0.001]}

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        max_regions=1,
        probe_semitones=[-12.0],
        objective_profile="pitch_region_count_attack_v2",
    )

    assert manifest["metrics"]["before_region_count"] == 16
    assert manifest["metrics"]["after_region_count"] == 12
    assert manifest["metrics"]["before_selected_region_count"] == 1
    assert manifest["metrics"]["after_selected_region_count"] == 1
    assert manifest["metrics"]["region_count_reduced"] is True
    assert manifest["metrics"]["target_issue_improved"] is True
    assert manifest["status"] == "trained_repair_still_blocked"


def test_pitch_detector_objective_profile_enables_normalized_forensic_losses(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    captured: dict[str, object] = {}

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "fail",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
            "regions": [
                {"start": 0.05, "end": 0.30, "semitones": -12.0, "score": 0.80, "fail_mode": "standard_score"},
                {"start": 0.32, "end": 0.45, "semitones": 0.0, "score": 0.75, "fail_mode": "standard_score"},
            ],
            "best_probe": {"score": 0.58 if is_after else 0.80},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        captured["config"] = kwargs["config"]
        kwargs["progress_callback"](
            {
                "event": "epoch",
                "attempt_index": 0,
                "epoch": 1,
                "epochs": 1,
                "loss": 0.001,
                "best_loss": 0.001,
            }
        )
        return FakeModel(), {"resolved_device": "mps", "final_loss": 0.001, "loss_history": [0.01, 0.001]}

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        max_regions=16,
        probe_semitones=[-12.0, -7.0, -3.0, 0.0, 3.0],
        objective_profile="pitch_region_detector_objective_v3",
    )

    config = captured["config"]
    assert manifest["status"] == "trained_repair_model_gap"
    assert manifest["contract"]["requires_region_count_reduction"] is True
    assert config.forensic_source_correlation_loss_weight >= 1.0
    assert config.forensic_source_projection_margin_loss_weight >= 1.0
    assert config.forensic_source_hard_region_loss_weight >= 1.0
    assert 0.0 < config.forensic_source_projection_margin <= 0.03
    assert config.max_delta <= 0.75
    assert config.source_skip_init_from_target is False


def test_damage_guarded_pitch_profile_uses_conservative_delta_budget(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    captured: dict[str, object] = {}

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "fail",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
            "regions": [{"start": 0.05, "end": 0.30, "semitones": 0.0, "score": 0.80, "fail_mode": "strong_projection_stft"}],
            "total_fail_regions": 0 if is_after else 4,
            "selected_region_count": 0 if is_after else 1,
            "best_probe": {"score": 0.50 if is_after else 0.80},
            "config": { "window_sec": 0.25, "hop_sec": 0.125 },
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        captured["config"] = kwargs["config"]
        return FakeModel(), {"resolved_device": "mps", "final_loss": 0.001, "loss_history": [0.01, 0.001]}

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        max_regions=1,
        probe_semitones=[0.0],
        objective_profile="pitch_region_detector_damage_guarded_v1",
    )

    config = captured["config"]
    assert manifest["objective_profile"] == "pitch_region_detector_damage_guarded_v1"
    assert manifest["contract"]["requires_region_count_reduction"] is True
    assert config.max_delta <= 0.42
    assert config.outside_loss_weight > 1.0
    assert config.region_loss_weight < 0.12
    assert config.target_delta_active_loss_weight < 0.10
    assert config.source_skip_init_from_target is False


def test_damage_aware_pitch_profile_wires_train_time_damage_proxy(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    captured: dict[str, object] = {}

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "pass" if is_after else "fail",
            "blocker_reasons": [] if is_after else ["pitch_shifted_vocal_in_instrumental"],
            "regions": []
            if is_after
            else [{"start": 0.05, "end": 0.30, "semitones": 0.0, "score": 0.80, "fail_mode": "strong_projection_stft"}],
            "total_fail_regions": 0 if is_after else 4,
            "selected_region_count": 0 if is_after else 1,
            "best_probe": {"score": 0.20 if is_after else 0.80},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        captured["config"] = kwargs["config"]
        return FakeModel(), {
            "resolved_device": "mps",
            "final_loss": 0.001,
            "loss_history": [0.01, 0.001],
            "loss_terms_enabled": {"instrumental_region_delta_ratio": True},
        }

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        max_regions=1,
        probe_semitones=[0.0],
        objective_profile="pitch_region_detector_damage_aware_v2",
    )

    config = captured["config"]
    assert manifest["objective_profile"] == "pitch_region_detector_damage_aware_v2"
    assert manifest["contract"]["requires_region_count_reduction"] is True
    assert manifest["delivery_allowed"] is False
    assert config.instrumental_region_delta_ratio_loss_weight >= 1.0
    assert config.instrumental_region_delta_ratio_limit <= 0.42
    assert config.forensic_source_correlation_loss_weight >= 1.0
    assert config.max_delta <= 0.55


def test_damage_aware_training_passes_original_regions_as_constraint_masks(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    captured: dict[str, object] = {}

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "pass" if is_after else "fail",
            "blocker_reasons": [] if is_after else ["pitch_shifted_vocal_in_instrumental"],
            "regions": []
            if is_after
            else [
                {"start": 0.05, "end": 0.20, "semitones": 0.0, "score": 0.80, "fail_mode": "strong_projection_stft"},
                {"start": 0.12, "end": 0.30, "semitones": 0.0, "score": 0.77, "fail_mode": "strong_projection_stft"},
            ],
            "total_fail_regions": 0 if is_after else 2,
            "selected_region_count": 0 if is_after else 2,
            "best_probe": {"score": 0.20 if is_after else 0.80},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        captured["training_frames"] = int(kwargs["region_mask"].shape[0])
        captured["region_constraint_masks"] = kwargs["region_constraint_masks"]
        captured["config"] = kwargs["config"]
        return FakeModel(), {
            "resolved_device": "mps",
            "final_loss": 0.001,
            "loss_history": [0.01, 0.001],
            "loss_terms_enabled": {"instrumental_region_delta_ratio": True},
        }

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        max_regions=2,
        probe_semitones=[0.0],
        objective_profile="pitch_region_detector_damage_aware_v2",
    )

    masks = captured["region_constraint_masks"]
    assert manifest["objective_profile"] == "pitch_region_detector_damage_aware_v2"
    assert len(masks) == 2
    assert all(mask.shape[0] == captured["training_frames"] for mask in masks)
    assert all(float(np.max(mask)) > 0.0 for mask in masks)


def test_damage_margin_pitch_profile_uses_stricter_proxy_margin(monkeypatch, tmp_path: Path) -> None:
    guard = _write_guard(tmp_path / "guard")
    captured: dict[str, object] = {}

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, **_kwargs):
        is_after = output_dir.name == "after"
        return {
            "verdict": "pass" if is_after else "fail",
            "blocker_reasons": [] if is_after else ["pitch_shifted_vocal_in_instrumental"],
            "regions": []
            if is_after
            else [{"start": 0.05, "end": 0.30, "semitones": 0.0, "score": 0.80, "fail_mode": "strong_projection_stft"}],
            "total_fail_regions": 0 if is_after else 4,
            "selected_region_count": 0 if is_after else 1,
            "best_probe": {"score": 0.20 if is_after else 0.80},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }

    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        captured["config"] = kwargs["config"]
        return FakeModel(), {
            "resolved_device": "mps",
            "final_loss": 0.001,
            "loss_history": [0.01, 0.001],
            "loss_terms_enabled": {"instrumental_region_delta_ratio": True},
        }

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0}

    monkeypatch.setattr(train_guard, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(train_guard, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(train_guard, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(train_guard, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(train_guard, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(train_guard, "write_training_history", lambda path, history: train_guard.write_json(path, history))

    manifest = train_guard.run_training(
        mel_guard_dir=guard,
        output_dir=tmp_path / "out",
        require_mps=True,
        max_regions=1,
        probe_semitones=[0.0],
        objective_profile="pitch_region_detector_damage_margin_v3",
    )

    config = captured["config"]
    assert manifest["objective_profile"] == "pitch_region_detector_damage_margin_v3"
    assert manifest["contract"]["requires_region_count_reduction"] is True
    assert config.instrumental_region_delta_ratio_loss_weight >= 24.0
    assert config.instrumental_region_delta_ratio_limit <= 0.38
    assert config.max_delta <= 0.55


def test_training_crop_reduces_sparse_full_track_regions() -> None:
    frames = SAMPLE_RATE * 8
    raw = np.zeros((frames, 2), dtype=np.float32)
    bad = raw.copy()
    target = raw.copy()
    source = raw.copy()
    mask = np.zeros(frames, dtype=np.float32)
    mask[int(1.0 * SAMPLE_RATE) : int(1.1 * SAMPLE_RATE)] = 1.0
    mask[int(6.0 * SAMPLE_RATE) : int(6.1 * SAMPLE_RATE)] = 1.0

    *_cropped, info = train_guard._build_training_crop(
        raw_audio=raw,
        instrumental_audio=bad,
        pseudo_target=target,
        source_condition=source,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        context_sec=0.25,
    )

    assert info["enabled"] is True
    assert info["training_frames"] < info["original_frames"] // 2
    assert len(info["ranges"]) == 2
