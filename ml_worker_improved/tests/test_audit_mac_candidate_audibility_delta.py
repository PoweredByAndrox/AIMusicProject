from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from tools import audit_mac_candidate_audibility_delta as audit


SAMPLE_RATE = 1_000


def _write_audio(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
    return path


def _stereo(audio: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(audio, dtype=np.float32)[:, None], 2, axis=1)


def test_audibility_audit_blocks_near_identical_candidate(tmp_path: Path) -> None:
    t = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
    source = _stereo(0.2 * np.sin(2 * np.pi * 7 * t))
    candidate = source + np.float32(1.0e-5)

    manifest = audit.run_audibility_audit(
        source_audio_path=_write_audio(tmp_path / "src.flac", source),
        candidate_audio_path=_write_audio(tmp_path / "cand.flac", candidate),
        output_dir=tmp_path / "out",
        min_fulltrack_rel_db=-45.0,
        min_active_window_ratio=0.05,
        min_active_changed_sec=0.5,
    )

    assert manifest["schema"] == "mac_candidate_audibility_delta.v1"
    assert manifest["status"] == "audibility_delta_blocked"
    assert manifest["verdict"] == "fail"
    assert "audibility_delta_below_gate" in manifest["blocker_reasons"]
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["presentation_status"] == "audibility_delta_diagnostic_only"
    assert manifest["metrics"]["active_changed_sec"] == 0.0
    assert Path(manifest["manifest_path"]).is_file()


def test_audibility_audit_passes_localized_candidate_change(tmp_path: Path) -> None:
    t = np.arange(SAMPLE_RATE * 4, dtype=np.float32) / SAMPLE_RATE
    source = _stereo(0.2 * np.sin(2 * np.pi * 5 * t))
    candidate = source.copy()
    candidate[SAMPLE_RATE : SAMPLE_RATE * 2] += 0.18

    manifest = audit.run_audibility_audit(
        source_audio_path=_write_audio(tmp_path / "src.flac", source),
        candidate_audio_path=_write_audio(tmp_path / "cand.flac", candidate),
        output_dir=tmp_path / "out",
        min_fulltrack_rel_db=-45.0,
        min_active_window_ratio=0.05,
        min_active_changed_sec=0.5,
    )

    assert manifest["status"] == "audibility_delta_passed"
    assert manifest["verdict"] == "pass"
    assert manifest["blocker_reasons"] == []
    assert manifest["metrics"]["active_changed_sec"] >= 0.5
    assert manifest["metrics"]["max_window_diff_ratio"] >= 0.05
    assert manifest["top_windows"][0]["start"] >= 0.75
    assert manifest["top_windows"][0]["end"] <= 2.25


def test_audibility_audit_writes_blocked_manifest_for_sample_rate_mismatch(tmp_path: Path) -> None:
    source = np.zeros((100, 2), dtype=np.float32)
    candidate = np.zeros((100, 2), dtype=np.float32)
    source_path = tmp_path / "src.flac"
    candidate_path = tmp_path / "cand.flac"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(source_path, source, 1_000)
    sf.write(candidate_path, candidate, 2_000)

    manifest = audit.run_audibility_audit(
        source_audio_path=source_path,
        candidate_audio_path=candidate_path,
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "audibility_delta_blocked"
    assert manifest["verdict"] == "fail"
    assert manifest["blocker_reasons"] == ["audibility_sample_rate_mismatch"]
    assert manifest["metrics"]["source_sample_rate"] == 1_000
    assert manifest["metrics"]["candidate_sample_rate"] == 2_000
    assert Path(manifest["manifest_path"]).is_file()
