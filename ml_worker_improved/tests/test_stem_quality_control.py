from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import stem_quality_control


def test_quality_control_reduces_peak_to_target_without_silently_failing(tmp_path: Path) -> None:
    sample_rate = 44100
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    audio[100, 0] = 0.99
    audio[101, 1] = -0.99
    input_path = tmp_path / "drums.flac"
    sf.write(input_path, audio, sample_rate, subtype="PCM_24")

    manifest = stem_quality_control.write_quality_controlled_stem(
        input_path,
        tmp_path / "qc",
        stem_name="drums",
        target_peak_dbfs=-1.0,
        min_duration_sec=0.5,
        min_rms_dbfs=-120.0,
    )

    controlled_audio, controlled_sample_rate = sf.read(manifest["output_path"], always_2d=True, dtype="float32")
    assert controlled_sample_rate == sample_rate
    assert manifest["headroom_applied"] is True
    assert manifest["headroom_gain"] < 1.0
    assert manifest["delivery_status"] == "diagnostic_only"
    assert manifest["release_status"] == "diagnostic_only"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "stem_quality_control_diagnostic_not_for_listen_pack"
    assert manifest["metrics"]["after"]["peak_dbfs"] <= -0.99
    assert float(np.max(np.abs(controlled_audio))) <= 10 ** (-1.0 / 20.0) + 1e-5


def test_quality_control_does_not_boost_stem_below_target_peak(tmp_path: Path) -> None:
    sample_rate = 44100
    audio = np.full((sample_rate, 2), 0.1, dtype=np.float32)
    input_path = tmp_path / "quiet_drums.flac"
    sf.write(input_path, audio, sample_rate, subtype="PCM_24")

    manifest = stem_quality_control.write_quality_controlled_stem(
        input_path,
        tmp_path / "qc",
        stem_name="drums",
        target_peak_dbfs=-1.0,
        min_duration_sec=0.5,
        min_rms_dbfs=-40.0,
    )

    assert manifest["headroom_applied"] is False
    assert manifest["headroom_gain"] == 1.0
    assert manifest["metrics"]["after"]["peak"] == pytest.approx(manifest["metrics"]["before"]["peak"])


def test_quality_control_fails_fast_for_invalid_audio(tmp_path: Path) -> None:
    sample_rate = 44100
    audio = np.zeros((100, 2), dtype=np.float32)
    input_path = tmp_path / "invalid.flac"
    sf.write(input_path, audio, sample_rate, subtype="PCM_24")

    with pytest.raises(ValueError) as exc_info:
        stem_quality_control.write_quality_controlled_stem(
            input_path,
            tmp_path / "qc",
            stem_name="drums",
            target_peak_dbfs=-1.0,
            min_duration_sec=1.0,
            min_rms_dbfs=-80.0,
        )

    assert "audio_too_short" in str(exc_info.value)
    assert "audio_too_quiet" in str(exc_info.value)


def test_listen_pack_writes_candidate_map(tmp_path: Path) -> None:
    sample_rate = 44100
    audio = np.full((sample_rate, 2), 0.1, dtype=np.float32)
    input_path = tmp_path / "drums.flac"
    raw_ref_path = tmp_path / "raw_ref.flac"
    sf.write(input_path, audio, sample_rate, subtype="PCM_24")
    sf.write(raw_ref_path, audio, sample_rate, subtype="PCM_24")

    manifest = stem_quality_control.write_quality_controlled_stem(
        input_path,
        tmp_path / "qc",
        stem_name="drums",
        target_peak_dbfs=-1.0,
        min_duration_sec=0.5,
        min_rms_dbfs=-40.0,
    )
    listen_map = stem_quality_control.write_listen_pack(
        manifest=manifest,
        listen_dir=tmp_path / "listen",
        raw_ref_path=raw_ref_path,
        source_label="01_original.flac",
        output_label="02_qc.flac",
    )

    candidate_map_path = tmp_path / "listen" / "candidate_map.json"
    saved_map = json.loads(candidate_map_path.read_text(encoding="utf-8"))
    listen_manifest = json.loads((tmp_path / "listen" / "listen_pack_manifest.json").read_text(encoding="utf-8"))
    assert listen_map == saved_map
    assert sorted(saved_map) == ["00_raw_ref.flac", "01_original.flac", "02_qc.flac"]
    assert listen_manifest["delivery_allowed"] is False
    assert listen_manifest["delivery_status"] == "diagnostic_only"
    assert listen_manifest["presentation_status"] == "stem_quality_control_diagnostic_not_for_listen_pack"
    assert listen_manifest["accepted_listen_pack_member"] is False
