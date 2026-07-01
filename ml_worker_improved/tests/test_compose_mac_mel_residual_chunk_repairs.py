from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import compose_mac_mel_residual_chunk_repairs as compose


SAMPLE_RATE = 1_000


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_audio(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
    return path


def _stereo(value: float, frames: int = 100) -> np.ndarray:
    return np.full((frames, 2), float(value), dtype=np.float32)


def _failed_stage(root: Path) -> Path:
    replay_dir = root / "replay"
    audio_dir = replay_dir / "audio"
    vocals = _write_audio(audio_dir / "v.flac", _stereo(0.0))
    instrumental = _write_audio(audio_dir / "i.flac", _stereo(0.0))
    raw = _write_audio(audio_dir / "raw.flac", _stereo(0.0))
    _write_json(
        replay_dir / "manifest.json",
        {
            "schema": "mac_melband_refiner_replay.v1",
            "status": "repaired_but_still_blocked",
            "audio": {"vocals": str(vocals), "instrumental": str(instrumental), "raw_ref": str(raw)},
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    _write_json(
        replay_dir / "score" / "after" / "pitch.json",
        {
            "schema": "mac_melband_pitch_guard_score.v1",
            "verdict": "fail",
            "total_fail_regions": 4,
            "selected_region_count": 4,
            "regions": [{"start": 0.01, "end": 0.04, "semitones": 0.0, "score": 0.7}],
            "best_probe": {"score": 0.7},
            "config": {"window_sec": 0.25, "hop_sec": 0.125},
        },
    )
    _write_json(
        root / "manifest.json",
        {
            "schema": "mac_mel_refiner_fulltrack_stage.v1",
            "status": "diagnostic_blocked_pitch_guard",
            "replay_manifest": str(replay_dir / "manifest.json"),
            "after_pitch_verdict": "fail",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "fulltrack_diagnostic_only",
        },
    )
    return root


def _residual(root: Path, *, work_id: str, delta: float, start: float, end: float, failed_stage_dir: Path) -> Path:
    run_dir = root / work_id
    train_dir = run_dir / "train"
    audio_dir = train_dir / "audio"
    _write_audio(audio_dir / "v.flac", _stereo(-delta))
    _write_audio(audio_dir / "i.flac", _stereo(delta))
    _write_audio(audio_dir / "raw.flac", _stereo(0.0))
    _write_audio(audio_dir / "target.flac", _stereo(delta))
    _write_json(
        train_dir / "manifest.json",
        {
            "schema": "mac_melband_refiner_training.v1",
            "status": "trained_repair_still_blocked",
            "audio": {
                "vocals": str(audio_dir / "v.flac"),
                "instrumental": str(audio_dir / "i.flac"),
                "raw_ref": str(audio_dir / "raw.flac"),
                "pseudo_target": str(audio_dir / "target.flac"),
            },
            "applied_regions": [{"start": start, "end": end, "semitones": 0.0, "source_score": 0.6}],
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "schema": "mac_mel_residual_refiner_training.v1",
            "status": "residual_trained_still_blocked",
            "source_failed_stage_dir": str(failed_stage_dir),
            "training_manifest": str(train_dir / "manifest.json"),
            "audio": {
                "vocals": str(audio_dir / "v.flac"),
                "instrumental": str(audio_dir / "i.flac"),
                "raw_ref": str(audio_dir / "raw.flac"),
                "pseudo_target": str(audio_dir / "target.flac"),
            },
            "taxonomy_batch": {"axis": "probe", "value": "0.0", "batch_id": work_id},
            "metrics": {"before_region_count": 4, "after_region_count": 2, "target_issue_improved": True},
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "residual_training_diagnostic_only",
        },
    )
    return run_dir / "manifest.json"


def test_chunk_composition_applies_masked_deltas_and_never_delivers(monkeypatch, tmp_path: Path) -> None:
    failed_stage = _failed_stage(tmp_path / "failed")
    residuals = [
        _residual(tmp_path / "residuals", work_id="r001", delta=0.20, start=0.01, end=0.04, failed_stage_dir=failed_stage),
        _residual(tmp_path / "residuals", work_id="r002", delta=0.50, start=0.06, end=0.08, failed_stage_dir=failed_stage),
    ]

    def fake_score(*, vocals_path: Path, instrumental_path: Path, output_dir: Path, config=None, progress_callback=None):
        if progress_callback is not None:
            progress_callback({"event": "probe_progress", "probe_index": 0, "windows": 1, "window_index": 1})
        result = {
            "schema": "mac_melband_pitch_guard_score.v1",
            "verdict": "fail",
            "total_fail_regions": 2,
            "selected_region_count": 2,
            "regions": [{"start": 0.06, "end": 0.08, "semitones": 0.0, "score": 0.6}],
            "best_probe": {"score": 0.6},
            "config": config or {},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pitch.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(compose, "score_pitch_shifted_vocal_leak", fake_score)

    manifest = compose.run_chunk_composition(
        failed_stage_dir=failed_stage,
        residual_manifest_paths=residuals,
        output_dir=tmp_path / "out",
        fade_sec=0.0,
    )

    instrumental, rate = compose.read_audio_stereo(Path(manifest["audio"]["instrumental"]))
    vocals, _ = compose.read_audio_stereo(Path(manifest["audio"]["vocals"]))
    raw, _ = compose.read_audio_stereo(Path(manifest["audio"]["raw_ref"]))

    assert rate == SAMPLE_RATE
    np.testing.assert_allclose(instrumental[:10], 0.0, atol=2.0e-4)
    np.testing.assert_allclose(instrumental[10:40], 0.20, atol=2.0e-4)
    np.testing.assert_allclose(instrumental[40:60], 0.0, atol=2.0e-4)
    np.testing.assert_allclose(instrumental[60:80], 0.50, atol=2.0e-4)
    np.testing.assert_allclose(instrumental[80:], 0.0, atol=2.0e-4)
    np.testing.assert_allclose(vocals + instrumental, raw, atol=2.0e-4)
    assert Path(manifest["audio"]["instrumental"]).name == "i.flac"
    assert Path(manifest["audio"]["vocals"]).name == "v.flac"
    assert manifest["status"] == "composition_blocked_pitch_guard"
    assert manifest["metrics"]["before_region_count"] == 4
    assert manifest["metrics"]["after_region_count"] == 2
    assert manifest["metrics"]["target_issue_improved"] is True
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["presentation_status"] == "chunk_composition_diagnostic_only"
    assert manifest["contract"]["no_listen_delivery"] is True
    assert manifest["contract"]["chunk_delta_composition"] is True
    progress = [
        json.loads(line)
        for line in (tmp_path / "out" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert progress[0]["gpu_inference_skipped"] is True
    assert progress[0]["acceleration_mode"] == "cpu_dsp_scoring"
    assert any(row["phase"] == "score_after" and row["status"] == "probe_progress" for row in progress)


def test_chunk_composition_blocks_delivery_contract_violations(tmp_path: Path) -> None:
    failed_stage = _failed_stage(tmp_path / "failed")
    residual = _residual(tmp_path / "residuals", work_id="r001", delta=0.20, start=0.01, end=0.04, failed_stage_dir=failed_stage)
    payload = json.loads(residual.read_text(encoding="utf-8"))
    payload["delivery_allowed"] = True
    residual.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compose.run_chunk_composition(
        failed_stage_dir=failed_stage,
        residual_manifest_paths=[residual],
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "blocked_invalid_residual_contract"
    assert manifest["failed_checks"] == ["residual_delivery_contract_invalid"]
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False


def test_chunk_composition_blocks_residuals_from_different_failed_stage(tmp_path: Path) -> None:
    failed_stage = _failed_stage(tmp_path / "failed")
    residual = _residual(tmp_path / "residuals", work_id="r001", delta=0.20, start=0.01, end=0.04, failed_stage_dir=failed_stage)
    payload = json.loads(residual.read_text(encoding="utf-8"))
    payload["source_failed_stage_dir"] = str(tmp_path / "other_failed_stage")
    residual.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compose.run_chunk_composition(
        failed_stage_dir=failed_stage,
        residual_manifest_paths=[residual],
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "blocked_residual_source_mismatch"
    assert manifest["failed_checks"] == ["residual_source_failed_stage_mismatch"]
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False


def test_chunk_composition_blocks_residuals_without_failed_stage_provenance(tmp_path: Path) -> None:
    failed_stage = _failed_stage(tmp_path / "failed")
    residual = _residual(tmp_path / "residuals", work_id="r001", delta=0.20, start=0.01, end=0.04, failed_stage_dir=failed_stage)
    payload = json.loads(residual.read_text(encoding="utf-8"))
    payload.pop("source_failed_stage_dir")
    residual.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compose.run_chunk_composition(
        failed_stage_dir=failed_stage,
        residual_manifest_paths=[residual],
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "blocked_residual_source_mismatch"
    assert manifest["failed_checks"] == ["residual_source_failed_stage_mismatch"]
    assert manifest["delivery_allowed"] is False
