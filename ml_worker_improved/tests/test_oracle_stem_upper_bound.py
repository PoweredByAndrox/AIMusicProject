import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "oracle_stem_upper_bound.py"
    spec = importlib.util.spec_from_file_location("oracle_stem_upper_bound", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def test_ideal_ratio_oracle_recovers_reference_when_sources_are_spectrally_separate(tmp_path):
    module = _load_module()
    sample_rate = 8000
    t = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
    drums = 0.45 * np.sin(2.0 * np.pi * 90.0 * t)
    melody = 0.24 * np.sin(2.0 * np.pi * 880.0 * t)
    source_mix = _stereo(drums + melody)
    target_reference = _stereo(drums)
    melody_reference = _stereo(melody)

    report = module.evaluate_oracle_upper_bound(
        source_mix_audio=source_mix,
        target_reference_audio=target_reference,
        reference_stems={"melody": melody_reference},
        sample_rate=sample_rate,
        stem_name="drums",
        n_fft=512,
        hop_size=128,
        reference_likeness_threshold=95.0,
    )

    oracle_metrics = report["oracle_candidates"]["ideal_ratio_mask"]["metrics"]
    self_source_metrics = report["source_mix_as_candidate"]["metrics"]

    assert oracle_metrics["reference_likeness_percent"] >= 98.0
    assert oracle_metrics["sdr_db"] > self_source_metrics["sdr_db"] + 10.0
    assert report["upper_bound_decision"]["status"] == "oracle_passed"
    assert report["contract"]["delivery_allowed_without_reference"] is False
    assert report["contract"]["no_reference_production_claim"] is False


def test_phrase_level_drop_melody_leakage_detects_long_non_drum_phrase(tmp_path):
    module = _load_module()
    sample_rate = 8000
    t = np.arange(sample_rate * 6, dtype=np.float64) / sample_rate
    drums = np.zeros_like(t)
    for hit_sec in [0.0, 1.5, 3.0, 4.5]:
        start = int(hit_sec * sample_rate)
        stop = min(start + int(0.08 * sample_rate), drums.size)
        envelope = np.linspace(1.0, 0.0, stop - start)
        drums[start:stop] += 0.85 * envelope * np.sin(2.0 * np.pi * 95.0 * t[: stop - start])
    melody = np.zeros_like(t)
    phrase_start = int(1.0 * sample_rate)
    phrase_stop = int(5.0 * sample_rate)
    melody[phrase_start:phrase_stop] = 0.18 * np.sin(2.0 * np.pi * 660.0 * t[: phrase_stop - phrase_start])

    target_reference = _stereo(drums)
    candidate = _stereo(drums + melody)
    melody_reference = _stereo(melody)

    windows = module.detect_phrase_level_leakage(
        candidate_audio=candidate,
        target_reference_audio=target_reference,
        reference_stems={"melody": melody_reference},
        sample_rate=sample_rate,
        window_seconds=2.0,
        hop_seconds=0.5,
        min_window_seconds=1.5,
        min_abs_correlation=0.55,
        min_residual_rms_dbfs=-55.0,
    )

    assert windows
    strongest = windows[0]
    assert strongest["label"] == "bar_spanning_drop_melody_leakage"
    assert strongest["source"] == "melody"
    assert strongest["start_seconds"] <= 1.5
    assert strongest["end_seconds"] >= 4.5
    assert strongest["duration_seconds"] >= 3.0


def test_phrase_level_drop_melody_leakage_flags_unattributed_harmonic_residual(tmp_path):
    module = _load_module()
    sample_rate = 8000
    t = np.arange(sample_rate * 6, dtype=np.float64) / sample_rate
    drums = np.zeros_like(t)
    for hit_sec in [0.0, 1.5, 3.0, 4.5]:
        start = int(hit_sec * sample_rate)
        stop = min(start + int(0.05 * sample_rate), drums.size)
        drums[start:stop] += np.linspace(1.0, 0.0, stop - start)
    sustained_drop = np.zeros_like(t)
    phrase_start = int(1.0 * sample_rate)
    phrase_stop = int(5.0 * sample_rate)
    sustained_drop[phrase_start:phrase_stop] = 0.14 * np.sin(
        2.0 * np.pi * 523.25 * t[: phrase_stop - phrase_start]
    )

    windows = module.detect_phrase_level_leakage(
        candidate_audio=_stereo(drums + sustained_drop),
        target_reference_audio=_stereo(drums),
        reference_stems={},
        sample_rate=sample_rate,
        window_seconds=2.0,
        hop_seconds=0.5,
        min_window_seconds=1.5,
        min_abs_correlation=0.55,
        min_residual_rms_dbfs=-55.0,
    )

    assert windows
    strongest = windows[0]
    assert strongest["label"] == "bar_spanning_drop_melody_leakage"
    assert strongest["source"] == "unattributed_harmonic_residual"
    assert strongest["start_seconds"] <= 1.5
    assert strongest["end_seconds"] >= 4.5


def test_cli_writes_oracle_audio_and_report_with_phrase_leakage(tmp_path):
    module = _load_module()
    sample_rate = 8000
    t = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
    drums = 0.35 * np.sin(2.0 * np.pi * 90.0 * t)
    melody = np.zeros_like(t)
    melody[int(0.4 * sample_rate) : int(1.8 * sample_rate)] = 0.18 * np.sin(
        2.0 * np.pi * 700.0 * t[: int(1.4 * sample_rate)]
    )

    source_path = tmp_path / "source.flac"
    target_path = tmp_path / "drums.flac"
    candidate_path = tmp_path / "candidate.flac"
    melody_path = tmp_path / "melody.flac"
    output_dir = tmp_path / "oracle"
    _write_audio(source_path, _stereo(drums + melody), sample_rate)
    _write_audio(target_path, _stereo(drums), sample_rate)
    _write_audio(candidate_path, _stereo(drums + melody), sample_rate)
    _write_audio(melody_path, _stereo(melody), sample_rate)

    exit_code = module.main(
        [
            "--source-mix",
            str(source_path),
            "--target-reference",
            str(target_path),
            "--candidate",
            str(candidate_path),
            "--reference-stem",
            f"melody={melody_path}",
            "--stem-name",
            "drums",
            "--output-dir",
            str(output_dir),
            "--n-fft",
            "512",
            "--hop-size",
            "128",
            "--phrase-window-seconds",
            "1.0",
            "--phrase-hop-seconds",
            "0.25",
            "--phrase-min-window-seconds",
            "0.75",
        ]
    )

    assert exit_code == 0
    report_path = output_dir / "oracle_upper_bound_report.json"
    oracle_path = output_dir / "01_ideal_ratio_mask_oracle.flac"
    assert report_path.exists()
    assert oracle_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["candidate_diagnostics"]["phrase_level_leakage_windows"]
    assert report["contract"]["target_reference_assisted"] is True
    assert report["contract"]["application_reference_free"] is False


def test_oracle_rejects_channel_mismatch_before_stft_indexing(tmp_path):
    module = _load_module()
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    stereo_source = _stereo(np.sin(2.0 * np.pi * 90.0 * t))
    stereo_target = _stereo(np.sin(2.0 * np.pi * 90.0 * t))
    mono_melody = np.sin(2.0 * np.pi * 440.0 * t)[:, None].astype(np.float32)

    try:
        module.evaluate_oracle_upper_bound(
            source_mix_audio=stereo_source,
            target_reference_audio=stereo_target,
            reference_stems={"melody": mono_melody},
            sample_rate=sample_rate,
            stem_name="drums",
            n_fft=256,
            hop_size=64,
        )
    except ValueError as exc:
        assert "channel count mismatch" in str(exc)
    else:
        raise AssertionError("expected channel count mismatch")


def test_oracle_rejects_target_stem_in_non_target_reference_stems(tmp_path):
    module = _load_module()
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    drums = _stereo(np.sin(2.0 * np.pi * 90.0 * t))
    melody = _stereo(np.sin(2.0 * np.pi * 440.0 * t))

    try:
        module.evaluate_oracle_upper_bound(
            source_mix_audio=drums + melody,
            target_reference_audio=drums,
            reference_stems={"drums": drums, "melody": melody},
            sample_rate=sample_rate,
            stem_name="drums",
            n_fft=256,
            hop_size=64,
        )
    except ValueError as exc:
        assert "target stem must not be listed as a non-target reference" in str(exc)
    else:
        raise AssertionError("expected target stem rejection")
