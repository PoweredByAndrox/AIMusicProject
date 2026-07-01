from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.mac_melband_pitch_guard import (
    apply_pitch_guard_repair,
    pitch_shift_resample_same_length,
    run_pitch_guard_on_existing_melband_run,
    score_best_value,
    score_pitch_shifted_vocal_leak,
    score_target_issue_improved,
)


SAMPLE_RATE = 16_000


def _voice_like(seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    f0 = 185.0 + 18.0 * np.sin(2.0 * math.pi * 4.5 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(SAMPLE_RATE)).astype(np.float32)
    audio = np.zeros_like(t)
    for harmonic, gain in ((1, 0.16), (2, 0.10), (3, 0.07), (5, 0.045), (7, 0.025), (9, 0.016)):
        audio += gain * np.sin(float(harmonic) * phase + 0.13 * harmonic)
    envelope = 0.50 + 0.50 * np.sin(2.0 * math.pi * 2.4 * t) ** 2
    return (audio * envelope).astype(np.float32)


def _instrumental_like(seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    piano = 0.08 * np.sin(2.0 * math.pi * 293.66 * t) + 0.05 * np.sin(2.0 * math.pi * 587.33 * t)
    bass = 0.12 * np.sin(2.0 * math.pi * 73.42 * t)
    pad = 0.035 * np.sin(2.0 * math.pi * 440.0 * t + 0.4 * np.sin(2.0 * math.pi * 0.6 * t))
    return (piano + bass + pad).astype(np.float32)


def _write(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.stack([audio, audio], axis=1).astype(np.float32)
    sf.write(path, stereo, SAMPLE_RATE, subtype="PCM_24")
    return path


def _read(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def test_pitch_shifted_vocal_in_instrumental_fails(tmp_path: Path) -> None:
    vocals = _voice_like()
    leak = pitch_shift_resample_same_length(vocals, semitones=-5.0)
    instrumental = _instrumental_like() + leak * 0.42

    result = score_pitch_shifted_vocal_leak(
        vocals_path=_write(tmp_path / "v.flac", vocals),
        instrumental_path=_write(tmp_path / "i.flac", instrumental),
        output_dir=tmp_path / "score",
        config={"window_sec": 0.5, "hop_sec": 0.25, "score_threshold": 0.48},
    )

    assert result["verdict"] == "fail"
    assert "pitch_shifted_vocal_in_instrumental" in result["blocker_reasons"]
    assert result["regions"]
    assert result["best_probe"]["semitones"] == pytest.approx(-5.0)
    assert (tmp_path / "score" / "pitch.json").exists()


def test_pitch_guard_reports_uncapped_failure_taxonomy(tmp_path: Path) -> None:
    vocals = _voice_like(seconds=2.5)
    leak = pitch_shift_resample_same_length(vocals, semitones=-5.0)
    instrumental = _instrumental_like(seconds=2.5) + leak * 0.46

    result = score_pitch_shifted_vocal_leak(
        vocals_path=_write(tmp_path / "v.flac", vocals),
        instrumental_path=_write(tmp_path / "i.flac", instrumental),
        output_dir=tmp_path / "score",
        config={
            "probe_semitones": [-5.0],
            "window_sec": 0.25,
            "hop_sec": 0.125,
            "score_threshold": 0.42,
            "max_regions": 1,
        },
    )

    assert result["verdict"] == "fail"
    assert len(result["regions"]) == 1
    assert result["selected_region_count"] == 1
    assert result["total_fail_regions"] > result["selected_region_count"]
    assert result["regions_capped"] is True
    taxonomy = result["failure_taxonomy"]
    assert taxonomy["total_fail_regions"] == result["total_fail_regions"]
    assert taxonomy["selected_region_count"] == result["selected_region_count"]
    assert taxonomy["by_probe"]["-5.0"] == result["total_fail_regions"]
    assert sum(taxonomy["by_fail_mode"].values()) == result["total_fail_regions"]
    regions_path = Path(result["all_regions_path"])
    assert regions_path.name == "regions.jsonl"
    assert regions_path.exists()
    all_regions = [json.loads(line) for line in regions_path.read_text(encoding="utf-8").splitlines()]
    assert len(all_regions) == result["total_fail_regions"]
    assert len(all_regions) > len(result["regions"])
    assert "raw_track_title" not in regions_path.as_posix()


def test_score_best_value_falls_back_to_region_scores_without_best_probe() -> None:
    score = {
        "regions": [
            {"score": 0.41},
            {"score": 0.73},
            {"score": 0.62},
        ],
        "config": {"window_sec": 0.25},
    }

    assert score_best_value(score) == pytest.approx(0.73)


def test_score_target_issue_improved_requires_region_count_reduction() -> None:
    before_score = {
        "verdict": "fail",
        "regions": [{"score": 0.80}],
        "total_fail_regions": 4,
    }
    score_drop_only = {
        "verdict": "fail",
        "regions": [{"score": 0.52}],
        "total_fail_regions": 4,
    }
    count_reduced = {
        "verdict": "fail",
        "regions": [{"score": 0.76}],
        "total_fail_regions": 3,
    }

    assert score_target_issue_improved(before_score=before_score, after_score=score_drop_only) is False
    assert score_target_issue_improved(before_score=before_score, after_score=count_reduced) is True
    assert score_target_issue_improved(before_score=before_score, after_score={"verdict": "pass", "regions": []}) is True


def test_unrelated_harmonic_instrumental_passes(tmp_path: Path) -> None:
    result = score_pitch_shifted_vocal_leak(
        vocals_path=_write(tmp_path / "v.flac", _voice_like()),
        instrumental_path=_write(tmp_path / "i.flac", _instrumental_like()),
        output_dir=tmp_path / "score",
        config={"window_sec": 0.5, "hop_sec": 0.25, "score_threshold": 0.55},
    )

    assert result["verdict"] == "pass"
    assert result["blocker_reasons"] == []
    assert result["regions"] == []


def test_low_direct_correlation_but_transformed_match_fails(tmp_path: Path) -> None:
    vocals = _voice_like()
    leak = pitch_shift_resample_same_length(vocals, semitones=-7.0)
    instrumental = _instrumental_like() + leak * 0.35

    direct_corr = float(np.corrcoef(vocals, instrumental)[0, 1])
    assert abs(direct_corr) < 0.35

    result = score_pitch_shifted_vocal_leak(
        vocals_path=_write(tmp_path / "v.flac", vocals),
        instrumental_path=_write(tmp_path / "i.flac", instrumental),
        output_dir=tmp_path / "score",
        config={"probe_semitones": [-12.0, -7.0, -5.0, 0.0, 5.0], "score_threshold": 0.45},
    )

    assert result["verdict"] == "fail"
    assert result["best_probe"]["semitones"] == pytest.approx(-7.0)
    assert result["best_probe"]["score"] > 0.45


def test_pitch_guard_repair_reduces_leak_and_preserves_reconstruction(tmp_path: Path) -> None:
    vocals = _voice_like()
    leak = pitch_shift_resample_same_length(vocals, semitones=-5.0)
    clean_instrumental = _instrumental_like()
    dirty_instrumental = clean_instrumental + leak * 0.42
    raw = vocals + dirty_instrumental

    vocals_path = _write(tmp_path / "audio" / "v.flac", vocals)
    instrumental_path = _write(tmp_path / "audio" / "i.flac", dirty_instrumental)
    raw_path = _write(tmp_path / "audio" / "raw.flac", raw)

    before = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        output_dir=tmp_path / "before",
        config={"score_threshold": 0.48},
    )
    repair = apply_pitch_guard_repair(
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        raw_ref_path=raw_path,
        score=before,
        output_dir=tmp_path / "repair",
    )
    after = score_pitch_shifted_vocal_leak(
        vocals_path=Path(repair["repaired_vocals_path"]),
        instrumental_path=Path(repair["repaired_instrumental_path"]),
        output_dir=tmp_path / "after",
        config={"score_threshold": 0.48},
    )

    assert repair["schema"] == "mac_melband_pitch_guard_repair.v1"
    assert after["best_probe"]["score"] < before["best_probe"]["score"]
    assert after["verdict"] == "pass"
    repaired_sum = _read(Path(repair["repaired_vocals_path"])) + _read(Path(repair["repaired_instrumental_path"]))
    assert _rms(repaired_sum - _read(raw_path)) < 5.0e-4


def test_existing_melband_run_stays_diagnostic_and_uses_short_paths(tmp_path: Path) -> None:
    source = tmp_path / "api_mel" / "run"
    vocals = _voice_like()
    leak = pitch_shift_resample_same_length(vocals, semitones=-5.0)
    instrumental = _instrumental_like() + leak * 0.42
    raw = vocals + instrumental
    v_path = _write(source / "vocals.flac", vocals)
    i_path = _write(source / "instrumental.flac", instrumental)
    raw_path = _write(source / "raw_ref.flac", raw)
    (source / "stem_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "melband_roformer_vocals",
                "source_audio_path": str(tmp_path / "raw_track_title_should_not_be_path.mp3"),
                "stems": {
                    "vocals": {"path": "/Users/andrejviluha/AIMusicProject_win/artifacts/mac_api/api_mel002/run/vocals.flac"},
                    "instrumental": {"path": "/Users/andrejviluha/AIMusicProject_win/artifacts/mac_api/api_mel002/run/instrumental.flac"},
                },
                "artifacts": {"raw_ref_flac": "/Users/andrejviluha/AIMusicProject_win/artifacts/mac_api/api_mel002/run/raw_ref.flac"},
            }
        ),
        encoding="utf-8",
    )

    manifest = run_pitch_guard_on_existing_melband_run(
        melband_run_dir=source,
        output_dir=tmp_path / "mel_guard001",
        config={"score_threshold": 0.48},
    )

    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["presentation_status"] == "diagnostic_melband_pitch_guard_only"
    assert manifest["score"]["verdict"] == "fail"
    assert "pitch_shifted_vocal_in_instrumental" in manifest["blocker_reasons"]
    expected_short_paths = [
        tmp_path / "mel_guard001" / "audio" / "v.flac",
        tmp_path / "mel_guard001" / "audio" / "i.flac",
        tmp_path / "mel_guard001" / "audio" / "raw.flac",
        tmp_path / "mel_guard001" / "score" / "pitch.json",
        tmp_path / "mel_guard001" / "repair" / "v.flac",
        tmp_path / "mel_guard001" / "repair" / "i.flac",
    ]
    for path in expected_short_paths:
        assert path.exists()
        assert "raw_track_title" not in path.as_posix()


def test_api_mel002_user_reported_tail_region_is_flagged_when_fixture_exists(tmp_path: Path) -> None:
    fixture = Path("/repo/artifacts/mac_api/api_mel002/run")
    if not fixture.exists():
        fixture = Path("D:/AIMusicProject/artifacts/mac_api/api_mel002/run")
    if not (fixture / "stem_manifest.json").exists():
        pytest.skip("api_mel002 regression fixture is not mounted")

    manifest = run_pitch_guard_on_existing_melband_run(
        melband_run_dir=fixture,
        output_dir=tmp_path / "mel_guard",
    )

    assert manifest["delivery_allowed"] is False
    assert manifest["score"]["verdict"] == "fail"
    assert any(
        float(region["start"]) < 12.0 and float(region["end"]) > 10.5
        for region in manifest["score"]["regions"]
    )
