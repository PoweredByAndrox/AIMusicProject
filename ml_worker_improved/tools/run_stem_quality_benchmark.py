from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


P1_LABELS = {
    "drum_tonal_leakage",
    "harmonic_leakage_in_drums",
    "vocal_dropout",
    "vocal_tail_loss",
    "vocal_harshness",
    "vocal_residue_in_minus",
    "possible_other_proxy_leakage",
    "proxy_copy_risk",
    "insufficient_presence",
    "near_silent_output",
    "source_conservation_error",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    sample_rate: int = 16000
    duration_seconds: float = 4.0
    drum_tonal_narrowband_threshold: float = 0.06
    drum_tonal_chroma_threshold: float = 0.145
    vocal_tail_loss_db: float = -8.0
    vocal_dropout_frame_ratio_threshold: float = 0.18
    vocal_harshness_ratio_threshold: float = 0.22
    vocal_residue_corr_threshold: float = 0.18
    proxy_copy_corr_threshold: float = 0.85
    proxy_copy_min_rms_ratio_db: float = -12.0
    min_presence_rms_dbfs: float = -48.0
    min_active_window_rms_dbfs: float = -44.0
    source_conservation_error_threshold_db: float = -28.0


@dataclass(frozen=True)
class BenchmarkFixture:
    name: str
    source_audio: np.ndarray
    stems: dict[str, np.ndarray]
    expected_labels: frozenset[str]
    description: str
    allowed_extra_labels: frozenset[str] = frozenset()
    verified_ground_truth: bool = True
    proxy_references: dict[str, np.ndarray] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    value = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)
    return 20.0 * math.log10(value)


def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    frames = min(left.shape[0], right.shape[0])
    if frames < 128:
        return 0.0
    left_mono = left[:frames].reshape(frames, -1).mean(axis=1)
    right_mono = right[:frames].reshape(frames, -1).mean(axis=1)
    left_mono = left_mono - float(np.mean(left_mono))
    right_mono = right_mono - float(np.mean(right_mono))
    denominator = float(np.sqrt(np.sum(left_mono * left_mono) * np.sum(right_mono * right_mono)) + 1e-12)
    return float(np.sum(left_mono * right_mono) / denominator)


def _db_delta(candidate: np.ndarray, reference: np.ndarray) -> float:
    return _rms_dbfs(candidate) - _rms_dbfs(reference)


def _stereo(mono: np.ndarray, right_gain: float = 0.94) -> np.ndarray:
    return np.stack([mono, mono * right_gain], axis=1).astype(np.float32)


def _normalize_stems(stems: dict[str, np.ndarray], peak: float = 0.82) -> dict[str, np.ndarray]:
    max_peak = max(float(np.max(np.abs(audio))) for audio in stems.values())
    if max_peak <= 0.0:
        return stems
    gain = peak / max_peak
    return {name: (audio * gain).astype(np.float32) for name, audio in stems.items()}


def _drum_pulses(time: np.ndarray, sample_rate: int) -> np.ndarray:
    kick = np.zeros_like(time, dtype=np.float32)
    snare = np.zeros_like(time, dtype=np.float32)
    kick_indices = (np.arange(0.15, float(time[-1]), 0.5) * sample_rate).astype(int)
    snare_indices = (np.arange(0.4, float(time[-1]), 1.0) * sample_rate).astype(int)
    kick[kick_indices[kick_indices < time.shape[0]]] = 1.0
    snare[snare_indices[snare_indices < time.shape[0]]] = 0.65
    kick = np.convolve(kick, np.hanning(128).astype(np.float32), mode="same")
    snare = np.convolve(snare, np.hanning(64).astype(np.float32), mode="same")
    hat = 0.025 * np.sign(np.sin(2.0 * np.pi * 12.0 * time)).astype(np.float32)
    return (kick + snare + hat).astype(np.float32)


def _base_components(config: BenchmarkConfig) -> dict[str, np.ndarray]:
    sample_count = int(round(config.sample_rate * config.duration_seconds))
    time = np.linspace(0.0, config.duration_seconds, sample_count, endpoint=False, dtype=np.float32)
    phrase_envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.75 * time)).astype(np.float32)
    phrase_envelope *= ((time > 0.25) & (time < config.duration_seconds - 0.25)).astype(np.float32)
    vocals = phrase_envelope * (
        0.23 * np.sin(2.0 * np.pi * (220.0 + 4.0 * np.sin(2.0 * np.pi * 3.0 * time)) * time)
        + 0.08 * np.sin(2.0 * np.pi * 440.0 * time)
    )
    bass = 0.25 * np.sin(2.0 * np.pi * 55.0 * time) * (0.65 + 0.35 * np.sign(np.sin(2.0 * np.pi * 2.0 * time)))
    other = (
        0.12 * np.sin(2.0 * np.pi * 330.0 * time)
        + 0.09 * np.sin(2.0 * np.pi * 495.0 * time)
        + 0.06 * np.sin(2.0 * np.pi * 660.0 * time)
    )
    stems = {
        "vocals": _stereo(vocals.astype(np.float32), 0.97),
        "drums": _stereo(_drum_pulses(time, config.sample_rate), 0.9),
        "bass": _stereo(bass.astype(np.float32), 0.98),
        "other": _stereo(other.astype(np.float32), 0.92),
    }
    return _normalize_stems(stems)


def _mix(stems: dict[str, np.ndarray]) -> np.ndarray:
    mix = np.zeros_like(next(iter(stems.values())))
    for audio in stems.values():
        mix = mix + audio
    return mix.astype(np.float32)


def _scale_stem_bundle_to_source_headroom(stems: dict[str, np.ndarray], peak: float = 0.96) -> tuple[dict[str, np.ndarray], np.ndarray]:
    source = _mix(stems)
    source_peak = float(np.max(np.abs(source)))
    if source_peak <= peak:
        return stems, source
    gain = peak / source_peak
    scaled_stems = {name: (audio * gain).astype(np.float32) for name, audio in stems.items()}
    return scaled_stems, _mix(scaled_stems)


def build_benchmark_fixtures(config: BenchmarkConfig) -> list[BenchmarkFixture]:
    clean_stems, clean_source = _scale_stem_bundle_to_source_headroom(_base_components(config))
    fixtures: list[BenchmarkFixture] = [
        BenchmarkFixture(
            name="clean_stem_bundle",
            source_audio=clean_source,
            stems=clean_stems,
            expected_labels=frozenset(),
            description="Clean verified synthetic stem bundle; protects against P1 false positives.",
        )
    ]

    drums_with_tone = dict(clean_stems)
    tonal_leak = 0.42 * clean_stems["other"]
    drums_with_tone["drums"] = (clean_stems["drums"] + tonal_leak).astype(np.float32)
    drums_with_tone["other"] = (clean_stems["other"] - tonal_leak).astype(np.float32)
    fixtures.append(
        BenchmarkFixture(
            name="drums_with_tonal_melody_leak",
            source_audio=clean_source,
            stems=drums_with_tone,
            expected_labels=frozenset({"drum_tonal_leakage", "harmonic_leakage_in_drums"}),
            description="Drum stem contains stable harmonic/melodic leakage.",
        )
    )

    vocal_dropout = dict(clean_stems)
    dropout_vocal = clean_stems["vocals"].copy()
    dropout_vocal[int(dropout_vocal.shape[0] * 0.48) :] *= 0.08
    vocal_dropout["vocals"] = dropout_vocal
    vocal_dropout["other"] = (clean_stems["other"] + clean_stems["vocals"] - dropout_vocal).astype(np.float32)
    fixtures.append(
        BenchmarkFixture(
            name="vocal_dropout_tail_loss",
            source_audio=clean_source,
            stems=vocal_dropout,
            expected_labels=frozenset({"vocal_dropout", "vocal_tail_loss"}),
            description="Vocal body is present but phrase endings are strongly attenuated.",
        )
    )

    vocal_harsh = dict(clean_stems)
    rng = np.random.default_rng(7305)
    time = np.arange(clean_stems["vocals"].shape[0], dtype=np.float32) / float(config.sample_rate)
    harsh_tone = 0.09 * np.sin(2.0 * np.pi * 0.43 * config.sample_rate * time).astype(np.float32)
    high_noise = _stereo(harsh_tone, 0.97)
    high_noise = high_noise + rng.normal(0.0, 0.012, size=clean_stems["vocals"].shape).astype(np.float32)
    vocal_harsh["vocals"] = (clean_stems["vocals"] + high_noise.astype(np.float32)).astype(np.float32)
    vocal_harsh["other"] = (clean_stems["other"] - high_noise.astype(np.float32)).astype(np.float32)
    fixtures.append(
        BenchmarkFixture(
            name="vocal_harsh_noise",
            source_audio=clean_source,
            stems=vocal_harsh,
            expected_labels=frozenset({"vocal_harshness"}),
            description="Vocal stem has high-band harshness/noise.",
        )
    )

    minus_with_residue = {
        "instrumental": (clean_stems["drums"] + clean_stems["bass"] + clean_stems["other"] + 0.5 * clean_stems["vocals"]).astype(np.float32),
        "vocals_reference": clean_stems["vocals"],
    }
    fixtures.append(
        BenchmarkFixture(
            name="minus_with_vocal_residue",
            source_audio=clean_source,
            stems=minus_with_residue,
            expected_labels=frozenset({"vocal_residue_in_minus"}),
            description="Instrumental/minus candidate contains vocal residue.",
            allowed_extra_labels=frozenset({"source_conservation_error"}),
        )
    )

    fixtures.append(
        BenchmarkFixture(
            name="long_tail_proxy_copy",
            source_audio=clean_source,
            stems={"strings": clean_stems["other"].copy()},
            expected_labels=frozenset({"possible_other_proxy_leakage", "proxy_copy_risk"}),
            description="Long-tail stem is copied from the other/proxy bus.",
            allowed_extra_labels=frozenset({"source_conservation_error"}),
            proxy_references={"other": clean_stems["other"]},
        )
    )

    fixtures.append(
        BenchmarkFixture(
            name="near_silent_sparse_stem",
            source_audio=clean_source,
            stems={"strings": (clean_stems["other"] * 0.003).astype(np.float32)},
            expected_labels=frozenset({"insufficient_presence", "near_silent_output"}),
            description="Stem file exists but has insufficient active presence.",
            allowed_extra_labels=frozenset({"source_conservation_error"}),
        )
    )

    bad_conservation = {
        "vocals": clean_stems["vocals"],
        "drums": clean_stems["drums"],
        "bass": clean_stems["bass"] * 0.2,
        "other": clean_stems["other"],
    }
    fixtures.append(
        BenchmarkFixture(
            name="source_conservation_violation",
            source_audio=clean_source,
            stems=bad_conservation,
            expected_labels=frozenset({"source_conservation_error"}),
            description="Candidate stems do not reconstruct the source mixture.",
        )
    )
    return fixtures


def _spectral_band_ratio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    mono = audio.mean(axis=1)
    stft = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)) + 1e-9
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    return float(np.sum(stft[band]) / np.sum(stft))


def _tonal_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    mono = audio.mean(axis=1)
    stft = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)) + 1e-9
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    tonal_band = (frequencies >= 160.0) & (frequencies <= 4200.0)
    tonal_mag = stft[tonal_band]
    narrowband_peak_share = 0.0
    if tonal_mag.size:
        narrowband_peak_share = float(np.mean(np.max(tonal_mag, axis=0) / (np.sum(tonal_mag, axis=0) + 1e-9)))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_peak_share = float(np.mean(np.max(chroma, axis=0) / (np.sum(chroma, axis=0) + 1e-9)))
    harmonic, percussive = librosa.effects.hpss(mono)
    return {
        "narrowband_peak_share": round(narrowband_peak_share, 8),
        "chroma_peak_share": round(chroma_peak_share, 8),
        "percussive_minus_harmonic_db": round(_rms_dbfs(percussive.reshape(-1, 1)) - _rms_dbfs(harmonic.reshape(-1, 1)), 6),
    }


def _frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    mono = audio.mean(axis=1)
    if mono.shape[0] < frame_length:
        return np.array([float(np.sqrt(np.mean(mono * mono) + 1e-12))], dtype=np.float32)
    frames = librosa.util.frame(mono, frame_length=frame_length, hop_length=hop_length)
    return np.sqrt(np.mean(frames * frames, axis=0) + 1e-12).astype(np.float32)


def _detect_drum_tonal_leakage(stems: dict[str, np.ndarray], sample_rate: int, config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    labels: set[str] = set()
    drums = stems.get("drums")
    if drums is None:
        return labels, {}
    features = _tonal_features(drums, sample_rate)
    if (
        features["narrowband_peak_share"] >= config.drum_tonal_narrowband_threshold
        and features["chroma_peak_share"] >= config.drum_tonal_chroma_threshold
    ):
        labels.add("drum_tonal_leakage")
    if "drum_tonal_leakage" in labels and (
        features["percussive_minus_harmonic_db"] < 7.0
        or features["narrowband_peak_share"] >= config.drum_tonal_narrowband_threshold * 1.5
    ):
        labels.add("harmonic_leakage_in_drums")
    return labels, features


def _detect_vocal_dropout(stems: dict[str, np.ndarray], clean_reference: np.ndarray, sample_rate: int, config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    labels: set[str] = set()
    vocals = stems.get("vocals")
    if vocals is None:
        return labels, {}
    frame_length = max(256, int(round(sample_rate * 0.08)))
    hop_length = max(128, int(round(sample_rate * 0.04)))
    candidate_env = _frame_rms(vocals, frame_length, hop_length)
    reference_env = _frame_rms(clean_reference, frame_length, hop_length)
    frame_count = min(candidate_env.shape[0], reference_env.shape[0])
    candidate_env = candidate_env[:frame_count]
    reference_env = reference_env[:frame_count]
    active = reference_env > np.percentile(reference_env, 55)
    ratio_db = 20.0 * np.log10((candidate_env + 1e-9) / (reference_env + 1e-9))
    dropout_ratio = float(np.mean((ratio_db < config.vocal_tail_loss_db) & active)) if np.any(active) else 0.0
    tail_start = int(frame_count * 0.62)
    tail_loss_db = float(np.mean(ratio_db[tail_start:][active[tail_start:]])) if np.any(active[tail_start:]) else 0.0
    if dropout_ratio >= config.vocal_dropout_frame_ratio_threshold:
        labels.add("vocal_dropout")
    if tail_loss_db <= config.vocal_tail_loss_db:
        labels.add("vocal_tail_loss")
    return labels, {
        "dropout_frame_ratio": round(dropout_ratio, 8),
        "tail_loss_db": round(tail_loss_db, 6),
    }


def _detect_vocal_harshness(stems: dict[str, np.ndarray], sample_rate: int, config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    labels: set[str] = set()
    vocals = stems.get("vocals")
    if vocals is None:
        return labels, {}
    high_band_ratio = _spectral_band_ratio(vocals, sample_rate, sample_rate * 0.32, sample_rate * 0.49)
    if high_band_ratio >= config.vocal_harshness_ratio_threshold:
        labels.add("vocal_harshness")
    return labels, {"high_band_ratio": round(high_band_ratio, 8)}


def _detect_vocal_residue(stems: dict[str, np.ndarray], config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    labels: set[str] = set()
    instrumental = stems.get("instrumental")
    vocal_reference = stems.get("vocals_reference")
    if instrumental is None or vocal_reference is None:
        return labels, {}
    correlation = abs(_normalized_correlation(instrumental, vocal_reference))
    vocal_ratio_db = _db_delta(vocal_reference, instrumental)
    if correlation >= config.vocal_residue_corr_threshold and vocal_ratio_db > -18.0:
        labels.add("vocal_residue_in_minus")
    return labels, {
        "vocal_residue_corr": round(correlation, 8),
        "vocal_to_instrumental_rms_ratio_db": round(vocal_ratio_db, 6),
    }


def _detect_proxy_copy(fixture: BenchmarkFixture, config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    labels: set[str] = set()
    if not fixture.proxy_references:
        return labels, {}
    metrics: dict[str, Any] = {}
    for stem_name, stem_audio in fixture.stems.items():
        for reference_name, reference_audio in fixture.proxy_references.items():
            correlation = abs(_normalized_correlation(stem_audio, reference_audio))
            rms_ratio_db = _db_delta(stem_audio, reference_audio)
            metrics[f"{stem_name}_vs_{reference_name}"] = {
                "correlation": round(correlation, 8),
                "rms_ratio_db": round(rms_ratio_db, 6),
            }
            if correlation >= config.proxy_copy_corr_threshold and rms_ratio_db >= config.proxy_copy_min_rms_ratio_db:
                labels.add("possible_other_proxy_leakage")
                labels.add("proxy_copy_risk")
    return labels, metrics


def _detect_presence(stems: dict[str, np.ndarray], sample_rate: int, config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    labels: set[str] = set()
    metrics: dict[str, Any] = {}
    frame_length = max(512, int(round(sample_rate * 0.5)))
    hop_length = max(256, int(round(sample_rate * 0.25)))
    for stem_name, audio in stems.items():
        if stem_name.endswith("_reference"):
            continue
        rms_dbfs = _rms_dbfs(audio)
        frame_rms = _frame_rms(audio, frame_length, hop_length)
        active_windows = int(np.sum(20.0 * np.log10(frame_rms + 1e-12) >= config.min_active_window_rms_dbfs))
        metrics[stem_name] = {
            "rms_dbfs": round(rms_dbfs, 6),
            "active_window_count": active_windows,
        }
        if rms_dbfs < config.min_presence_rms_dbfs:
            labels.add("near_silent_output")
        if active_windows == 0:
            labels.add("insufficient_presence")
    return labels, metrics


def _detect_source_conservation(fixture: BenchmarkFixture, config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    labels: set[str] = set()
    summable_stems = [audio for name, audio in fixture.stems.items() if not name.endswith("_reference")]
    if not summable_stems:
        return labels, {}
    stem_sum = np.zeros_like(fixture.source_audio)
    for audio in summable_stems:
        frames = min(stem_sum.shape[0], audio.shape[0])
        stem_sum[:frames] += audio[:frames]
    error = fixture.source_audio - stem_sum
    error_dbfs = _rms_dbfs(error)
    source_dbfs = _rms_dbfs(fixture.source_audio)
    relative_error_db = error_dbfs - source_dbfs
    if relative_error_db > config.source_conservation_error_threshold_db:
        labels.add("source_conservation_error")
    return labels, {
        "source_rms_dbfs": round(source_dbfs, 6),
        "error_rms_dbfs": round(error_dbfs, 6),
        "relative_error_db": round(relative_error_db, 6),
    }


def detect_fixture_labels(fixture: BenchmarkFixture, clean_reference_stems: dict[str, np.ndarray], config: BenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    detected: set[str] = set()
    metrics: dict[str, Any] = {}
    detectors = [
        ("drum_tonal", _detect_drum_tonal_leakage(fixture.stems, config.sample_rate, config)),
        ("vocal_dropout", _detect_vocal_dropout(fixture.stems, clean_reference_stems["vocals"], config.sample_rate, config)),
        ("vocal_harshness", _detect_vocal_harshness(fixture.stems, config.sample_rate, config)),
        ("vocal_residue", _detect_vocal_residue(fixture.stems, config)),
        ("proxy_copy", _detect_proxy_copy(fixture, config)),
        ("presence", _detect_presence(fixture.stems, config.sample_rate, config)),
        ("source_conservation", _detect_source_conservation(fixture, config)),
    ]
    for detector_name, (labels, detector_metrics) in detectors:
        detected.update(labels)
        metrics[detector_name] = detector_metrics
    return detected, metrics


def _write_fixture_files(fixture: BenchmarkFixture, fixture_dir: Path, sample_rate: int) -> dict[str, Any]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    source_path = fixture_dir / "source.flac"
    _write_audio(source_path, fixture.source_audio, sample_rate)
    stem_entries: dict[str, dict[str, Any]] = {}
    stem_dir = fixture_dir / "stems"
    for stem_name, audio in sorted(fixture.stems.items()):
        stem_path = stem_dir / f"{stem_name}.flac"
        _write_audio(stem_path, audio, sample_rate)
        stem_entries[stem_name] = {
            "path": str(stem_path),
            "sha256": _sha256(stem_path),
            "rms_dbfs": round(_rms_dbfs(audio), 6),
        }
    return {
        "name": fixture.name,
        "description": fixture.description,
        "verified_ground_truth": fixture.verified_ground_truth,
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "rms_dbfs": round(_rms_dbfs(fixture.source_audio), 6),
        },
        "stems": stem_entries,
        "expected_labels": sorted(fixture.expected_labels),
    }


def run_benchmark(
    output_dir: Path,
    *,
    suite: str = "smoke",
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    if suite != "smoke":
        raise ValueError("suite must be 'smoke'")
    config = config or BenchmarkConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = build_benchmark_fixtures(config)
    clean_reference_stems = fixtures[0].stems
    fixture_reports: list[dict[str, Any]] = []
    missing_expected_total = 0
    unexpected_p1_total = 0

    for fixture in fixtures:
        fixture_dir = output_dir / "fixtures" / fixture.name
        file_manifest = _write_fixture_files(fixture, fixture_dir, config.sample_rate)
        detected_labels, metrics = detect_fixture_labels(fixture, clean_reference_stems, config)
        missing_expected = sorted(fixture.expected_labels - detected_labels)
        unexpected_p1 = sorted((detected_labels - fixture.expected_labels - fixture.allowed_extra_labels) & P1_LABELS)
        missing_expected_total += len(missing_expected)
        unexpected_p1_total += len(unexpected_p1)
        status = "passed" if not missing_expected and not unexpected_p1 else "failed"
        fixture_reports.append(
            {
                **file_manifest,
                "detected_labels": sorted(detected_labels),
                "missing_expected_labels": missing_expected,
                "unexpected_p1_labels": unexpected_p1,
                "allowed_extra_labels": sorted(fixture.allowed_extra_labels),
                "metrics": metrics,
                "status": status,
            }
        )

    overall_status = "passed" if missing_expected_total == 0 and unexpected_p1_total == 0 else "failed"
    manifest = {
        "schema_version": 1,
        "contract": {
            "benchmark": "stem_quality_benchmark_v1",
            "verified_synthetic_truth": True,
            "fallback_used": False,
            "no_real_track_claims": True,
            "interpretation": "Synthetic regression truth, not a substitute for human listening on real tracks.",
        },
        "suite": suite,
        "config": {
            "sample_rate": config.sample_rate,
            "duration_seconds": config.duration_seconds,
            "drum_tonal_narrowband_threshold": config.drum_tonal_narrowband_threshold,
            "drum_tonal_chroma_threshold": config.drum_tonal_chroma_threshold,
            "vocal_tail_loss_db": config.vocal_tail_loss_db,
            "vocal_dropout_frame_ratio_threshold": config.vocal_dropout_frame_ratio_threshold,
            "vocal_harshness_ratio_threshold": config.vocal_harshness_ratio_threshold,
            "vocal_residue_corr_threshold": config.vocal_residue_corr_threshold,
            "proxy_copy_corr_threshold": config.proxy_copy_corr_threshold,
            "proxy_copy_min_rms_ratio_db": config.proxy_copy_min_rms_ratio_db,
            "min_presence_rms_dbfs": config.min_presence_rms_dbfs,
            "min_active_window_rms_dbfs": config.min_active_window_rms_dbfs,
            "source_conservation_error_threshold_db": config.source_conservation_error_threshold_db,
        },
        "fixtures": fixture_reports,
        "summary": {
            "status": overall_status,
            "fixture_count": len(fixture_reports),
            "known_defect_fixture_count": sum(1 for fixture in fixtures if fixture.expected_labels),
            "known_defects_detected": sum(
                len(set(report["detected_labels"]) & set(report["expected_labels"]))
                for report in fixture_reports
                if report["expected_labels"]
            ),
            "missing_expected_label_count": missing_expected_total,
            "unexpected_p1_label_count": unexpected_p1_total,
        },
    }
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic stem quality benchmark fixtures.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--suite", choices=("smoke",), default="smoke")
    parser.add_argument("--sample-rate", type=int, default=BenchmarkConfig.sample_rate)
    parser.add_argument("--duration-seconds", type=float, default=BenchmarkConfig.duration_seconds)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main() -> int:
    require_docker_runtime(
        tool_name="run_stem_quality_benchmark.py",
        entrypoint_path="/app/tools/run_stem_quality_benchmark.py",
    )
    args = _build_arg_parser().parse_args()
    manifest = run_benchmark(
        args.output_dir,
        suite=str(args.suite),
        config=BenchmarkConfig(
            sample_rate=int(args.sample_rate),
            duration_seconds=float(args.duration_seconds),
        ),
    )
    print(json.dumps({"summary": manifest["summary"], "manifest_path": str(args.output_dir / "benchmark_manifest.json")}, indent=2))
    if args.fail_on_regression and manifest["summary"]["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
