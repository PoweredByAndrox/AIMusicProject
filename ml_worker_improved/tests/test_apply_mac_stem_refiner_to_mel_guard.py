from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tools import apply_mac_stem_refiner_to_mel_guard as replay


SAMPLE_RATE = 16_000


def _write_audio(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def _write_guard_fixture(root: Path) -> Path:
    frames = SAMPLE_RATE // 2
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    vocals = np.stack([0.04 * np.sin(2.0 * np.pi * 220.0 * t), 0.04 * np.sin(2.0 * np.pi * 220.0 * t)], axis=1)
    leak = np.stack([0.02 * np.sin(2.0 * np.pi * 110.0 * t), 0.02 * np.sin(2.0 * np.pi * 110.0 * t)], axis=1)
    instrumental = 0.03 * np.stack([np.sin(2.0 * np.pi * 330.0 * t), np.sin(2.0 * np.pi * 330.0 * t)], axis=1) + leak
    raw = vocals + instrumental
    audio_dir = root / "audio"
    vocals_path = _write_audio(audio_dir / "v.flac", vocals)
    instrumental_path = _write_audio(audio_dir / "i.flac", instrumental)
    raw_path = _write_audio(audio_dir / "raw.flac", raw)
    manifest = {
        "schema": "mac_melband_pitch_guard_manifest.v1",
        "audio": {
            "vocals": str(vocals_path),
            "instrumental": str(instrumental_path),
            "raw_ref": str(raw_path),
        },
        "score": {"config": {"window_sec": 0.25, "hop_sec": 0.125}},
        "delivery_allowed": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_refiner_replay_blocks_when_mps_required_but_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(replay, "resolve_region_repair_training_device", lambda requested: torch.device("cpu"))

    manifest = replay.run_refiner_replay(
        mel_guard_dir=tmp_path / "missing",
        checkpoint_path=tmp_path / "model.pt",
        output_dir=tmp_path / "out",
        require_mps=True,
    )

    assert manifest["status"] == "blocked_mps_unavailable"
    assert manifest["delivery_allowed"] is False
    assert manifest["resolved_device"] == "cpu"
    assert (tmp_path / "out" / "manifest.json").exists()


def test_target_issue_does_not_improve_when_new_leak_regions_appear() -> None:
    assert replay._target_issue_improved(
        before_region_count=4,
        after_region_count=8,
        before_best_score=0.64,
        after_best_score=0.58,
        after_verdict="fail",
    ) is False


def test_resolve_audio_path_falls_back_when_original_mount_is_broken(monkeypatch, tmp_path: Path) -> None:
    guard_dir = tmp_path / "guard"
    local_audio = guard_dir / "audio" / "v.flac"
    local_audio.parent.mkdir(parents=True)
    local_audio.write_bytes(b"audio")

    def fake_path_exists(path: Path) -> bool:
        if str(path).startswith("/broken/share"):
            raise OSError("socket is not connected")
        return path.exists()

    monkeypatch.setattr(replay, "_path_exists", fake_path_exists)

    resolved = replay._resolve_audio_path("/broken/share/v.flac", guard_dir=guard_dir)

    assert resolved == local_audio


def test_refiner_replay_keeps_delivery_disabled_and_preserves_mixture(monkeypatch, tmp_path: Path) -> None:
    guard_dir = _write_guard_fixture(tmp_path / "guard")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None):
        is_after = output_dir.name == "after"
        result = {
            "schema": "mac_melband_pitch_guard_score.v1",
            "verdict": "pass" if is_after else "fail",
            "blocker_reasons": [] if is_after else ["pitch_shifted_vocal_in_instrumental"],
            "regions": []
            if is_after
            else [{"start": 0.05, "end": 0.30, "semitones": -5.0, "score": 0.8, "fail_mode": "standard_score"}],
            "best_probe": {"score": 0.2 if is_after else 0.8},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pitch.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    def fake_apply_region_repair_model(**kwargs):
        bad = kwargs["bad_stem_audio"]
        source = kwargs["source_condition_audio"]
        mask = kwargs["region_mask"]
        repaired = bad - 0.1 * source * mask[:, None]
        return repaired.astype(np.float32), {"resolved_device": "mps", "changed_region_ratio": 0.1, "changed_outside_ratio": 0.0}

    monkeypatch.setattr(replay, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(replay, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(replay, "load_region_repair_checkpoint", lambda path: (object(), {"training_history": {"resolved_device": "mps"}}))
    monkeypatch.setattr(replay, "apply_region_repair_model", fake_apply_region_repair_model)

    manifest = replay.run_refiner_replay(
        mel_guard_dir=guard_dir,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "out",
        require_mps=True,
    )

    assert manifest["status"] == "repaired_passed_pitch_guard"
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["contract"]["mixture_consistent_delta_transfer"] is True
    assert manifest["metrics"]["target_issue_improved"] is True
    assert manifest["metrics"]["reconstruction_rms"] < 5.0e-5
    assert Path(manifest["audio"]["instrumental"]).name == "i.flac"
    assert Path(manifest["audio"]["vocals"]).name == "v.flac"


def test_refiner_replay_uses_region_only_best_score_without_score_only_improvement(monkeypatch, tmp_path: Path) -> None:
    guard_dir = _write_guard_fixture(tmp_path / "guard")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None):
        is_after = output_dir.name == "after"
        result = {
            "schema": "mac_melband_pitch_guard_score.v1",
            "verdict": "fail",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
            "regions": [
                {
                    "start": 0.05,
                    "end": 0.30,
                    "semitones": -5.0,
                    "score": 0.62 if is_after else 0.80,
                    "fail_mode": "standard_score",
                }
            ],
            "total_fail_regions": 1,
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pitch.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    def fake_apply_region_repair_model(**kwargs):
        return kwargs["bad_stem_audio"], {"resolved_device": "mps", "changed_region_ratio": 0.0, "changed_outside_ratio": 0.0}

    monkeypatch.setattr(replay, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(replay, "score_pitch_shifted_vocal_leak", fake_score)
    monkeypatch.setattr(replay, "load_region_repair_checkpoint", lambda path: (object(), {"training_history": {"resolved_device": "mps"}}))
    monkeypatch.setattr(replay, "apply_region_repair_model", fake_apply_region_repair_model)

    manifest = replay.run_refiner_replay(
        mel_guard_dir=guard_dir,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "out",
        require_mps=True,
    )

    assert manifest["status"] == "repaired_but_still_blocked"
    assert manifest["metrics"]["before_best_score"] == 0.80
    assert manifest["metrics"]["after_best_score"] == 0.62
    assert manifest["metrics"]["target_issue_improved"] is False
