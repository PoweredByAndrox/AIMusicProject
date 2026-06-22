"""Score forbidden source contamination inside a generated stem."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stem_contamination_model import SOURCE_BANDS_HZ, band_limited_audio, predict_no_reference_probability, source_band_limited_audio


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return path


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _dbfs(value: float) -> float:
    if value <= 1e-12:
        return -240.0
    return 20.0 * math.log10(value)


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _projection_leak(stem_audio: np.ndarray, foreign_audio: np.ndarray) -> np.ndarray:
    min_len = min(len(stem_audio), len(foreign_audio))
    if min_len <= 0:
        return np.zeros((0, stem_audio.shape[1] if stem_audio.ndim == 2 else 1), dtype=np.float32)
    stem = stem_audio[:min_len].astype(np.float64)
    foreign = foreign_audio[:min_len].astype(np.float64)
    denom = float(np.sum(foreign * foreign))
    if denom <= 1e-12:
        return np.zeros_like(stem, dtype=np.float32)
    coeff = float(np.sum(stem * foreign) / denom)
    return (foreign * coeff).astype(np.float32)


def _band_limited_audio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    return band_limited_audio(audio, sample_rate, low_hz, high_hz)


def _no_reference_vocal_in_minus_score(
    *,
    stem_audio: np.ndarray,
    sample_rate: int,
    detection_floor_db: float,
) -> tuple[str, np.ndarray, float]:
    leak_audio = _band_limited_audio(stem_audio, sample_rate, 700.0, 1200.0)
    leak_dbfs = _dbfs(_rms(leak_audio))
    verdict = "fail" if leak_dbfs > detection_floor_db else "pass"
    return verdict, leak_audio, leak_dbfs


def _bass_in_drums_acoustic_evidence(leak_audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    mono = np.asarray(leak_audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1).astype(np.float32)
    if mono.size <= 1:
        return {
            "bass_like_sustained": False,
            "bass_low_active_ratio": 0.0,
            "bass_low_peak_share": 0.0,
            "bass_low_transient_ratio": 0.0,
            "bass_low_envelope_cv": 0.0,
            "bass_low_crest_db": 0.0,
        }
    frame = max(128, int(sample_rate * 0.05))
    hop = max(1, frame // 2)
    if mono.size < frame:
        envelope = np.asarray([_rms(mono)], dtype=np.float32)
    else:
        envelope = np.asarray(
            [_rms(mono[start : start + frame]) for start in range(0, len(mono) - frame + 1, hop)],
            dtype=np.float32,
        )
    envelope_mean = float(np.mean(envelope) + 1.0e-12)
    envelope_max = float(np.max(envelope) + 1.0e-12)
    envelope_cv = float(np.std(envelope) / envelope_mean)
    transient_ratio = float(np.mean(np.abs(np.diff(envelope))) / envelope_mean) if len(envelope) > 1 else 0.0
    active_ratio = float(np.mean(envelope > 0.3 * envelope_max))
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))).astype(np.float64)
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    band = spectrum[(freqs >= 40.0) & (freqs <= 300.0)]
    peak_share = float(np.max(band) / (np.sum(band) + 1.0e-12)) if band.size else 0.0
    crest_db = _dbfs(float(np.max(np.abs(mono)))) - _dbfs(_rms(mono))
    bass_like_sustained = active_ratio >= 0.80 and peak_share >= 0.08 and transient_ratio <= 0.40
    return {
        "bass_like_sustained": bool(bass_like_sustained),
        "bass_low_active_ratio": round(active_ratio, 6),
        "bass_low_peak_share": round(peak_share, 6),
        "bass_low_transient_ratio": round(transient_ratio, 6),
        "bass_low_envelope_cv": round(envelope_cv, 6),
        "bass_low_crest_db": round(crest_db, 6),
    }


def _sustained_melodic_in_drums_acoustic_evidence(leak_audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    mono = np.asarray(leak_audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1).astype(np.float32)
    if mono.size <= 1:
        return {
            "melodic_like_sustained": False,
            "melodic_active_ratio": 0.0,
            "melodic_peak_share": 0.0,
            "melodic_transient_ratio": 0.0,
            "melodic_envelope_cv": 0.0,
            "melodic_crest_db": 0.0,
        }
    frame = max(128, int(sample_rate * 0.05))
    hop = max(1, frame // 2)
    if mono.size < frame:
        envelope = np.asarray([_rms(mono)], dtype=np.float32)
    else:
        envelope = np.asarray(
            [_rms(mono[start : start + frame]) for start in range(0, len(mono) - frame + 1, hop)],
            dtype=np.float32,
        )
    envelope_mean = float(np.mean(envelope) + 1.0e-12)
    envelope_max = float(np.max(envelope) + 1.0e-12)
    envelope_cv = float(np.std(envelope) / envelope_mean)
    transient_ratio = float(np.mean(np.abs(np.diff(envelope))) / envelope_mean) if len(envelope) > 1 else 0.0
    active_ratio = float(np.mean(envelope > 0.3 * envelope_max))
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))).astype(np.float64)
    peak_share = float(np.max(spectrum) / (np.sum(spectrum) + 1.0e-12)) if spectrum.size else 0.0
    crest_db = _dbfs(float(np.max(np.abs(mono)))) - _dbfs(_rms(mono))
    melodic_like_sustained = active_ratio >= 0.70 and peak_share >= 0.025 and transient_ratio <= 0.35
    return {
        "melodic_like_sustained": bool(melodic_like_sustained),
        "melodic_active_ratio": round(active_ratio, 6),
        "melodic_peak_share": round(peak_share, 6),
        "melodic_transient_ratio": round(transient_ratio, 6),
        "melodic_envelope_cv": round(envelope_cv, 6),
        "melodic_crest_db": round(crest_db, 6),
    }


def _foreign_in_vocals_acoustic_evidence(
    stem_audio: np.ndarray,
    leak_audio: np.ndarray,
    sample_rate: int,
    foreign_source: str,
) -> dict[str, Any]:
    mono = np.asarray(stem_audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1).astype(np.float32)
    leak_mono = np.asarray(leak_audio, dtype=np.float32)
    if leak_mono.ndim == 2:
        leak_mono = np.mean(leak_mono, axis=1).astype(np.float32)
    if mono.size <= 1 or leak_mono.size <= 1:
        return {
            "foreign_like_in_vocal": False,
            "vocal_foreign_peak_share": 0.0,
            "vocal_transient_ratio": 0.0,
            "vocal_spectral_centroid_hz": 0.0,
        }
    frame = max(128, int(sample_rate * 0.05))
    hop = max(1, frame // 2)
    if mono.size < frame:
        envelope = np.asarray([_rms(mono)], dtype=np.float32)
    else:
        envelope = np.asarray(
            [_rms(mono[start : start + frame]) for start in range(0, len(mono) - frame + 1, hop)],
            dtype=np.float32,
        )
    envelope_mean = float(np.mean(envelope) + 1.0e-12)
    transient_ratio = float(np.mean(np.abs(np.diff(envelope))) / envelope_mean) if len(envelope) > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))).astype(np.float64)
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1.0e-12)) if spectrum.size else 0.0
    leak_spectrum = np.abs(np.fft.rfft(leak_mono * np.hanning(len(leak_mono)))).astype(np.float64)
    leak_freqs = np.fft.rfftfreq(len(leak_mono), d=1.0 / float(sample_rate))
    peak_share = float(np.max(leak_spectrum) / (np.sum(leak_spectrum) + 1.0e-12)) if leak_spectrum.size else 0.0
    high_band = leak_spectrum[(leak_freqs >= 1200.0) & (leak_freqs <= 6000.0)]
    high_peak_share = float(np.max(high_band) / (np.sum(high_band) + 1.0e-12)) if high_band.size else 0.0
    if foreign_source == "bass":
        foreign_like = peak_share >= 0.09 and transient_ratio <= 0.12
    elif foreign_source in {"guitar", "piano", "other"}:
        foreign_like = peak_share >= 0.04 and transient_ratio <= 0.12
    elif foreign_source == "drums":
        foreign_like = high_peak_share >= 0.02 and (transient_ratio >= 0.20 or centroid >= 2500.0)
    else:
        foreign_like = True
    return {
        "foreign_like_in_vocal": bool(foreign_like),
        "vocal_foreign_peak_share": round(peak_share, 6),
        "vocal_high_peak_share": round(high_peak_share, 6),
        "vocal_transient_ratio": round(transient_ratio, 6),
        "vocal_spectral_centroid_hz": round(centroid, 6),
    }


def _foreign_in_bass_acoustic_evidence(
    stem_audio: np.ndarray,
    leak_audio: np.ndarray,
    sample_rate: int,
    foreign_source: str,
) -> dict[str, Any]:
    low_audio = _band_limited_audio(stem_audio, sample_rate, 40.0, 300.0)
    low_dbfs = _dbfs(_rms(low_audio))
    foreign_dbfs = _dbfs(_rms(leak_audio))
    foreign_to_low_db = foreign_dbfs - low_dbfs
    mono = np.asarray(stem_audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))).astype(np.float64) if mono.size else np.asarray([])
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate)) if mono.size else np.asarray([])
    centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1.0e-12)) if spectrum.size else 0.0
    foreign_like = foreign_source in {"guitar", "piano", "other"} and foreign_to_low_db >= -6.0
    return {
        "foreign_like_in_bass": bool(foreign_like),
        "bass_low_dbfs": round(low_dbfs, 6),
        "bass_foreign_dbfs": round(foreign_dbfs, 6),
        "bass_foreign_to_low_db": round(foreign_to_low_db, 6),
        "bass_spectral_centroid_hz": round(centroid, 6),
    }


def _source_band_peak_share(audio: np.ndarray, sample_rate: int, foreign_source: str) -> float:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1).astype(np.float32)
    if mono.size <= 1:
        return 0.0
    low_hz, high_hz = SOURCE_BANDS_HZ.get(foreign_source, (700.0, 1200.0))
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))).astype(np.float64)
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    band = spectrum[(freqs >= low_hz) & (freqs <= high_hz)]
    return float(np.max(band) / (np.sum(band) + 1.0e-12)) if band.size else 0.0


def _foreign_in_guitar_acoustic_evidence(
    stem_audio: np.ndarray,
    leak_audio: np.ndarray,
    sample_rate: int,
    foreign_source: str,
) -> dict[str, Any]:
    total_dbfs = _dbfs(_rms(stem_audio))
    foreign_dbfs = _dbfs(_rms(leak_audio))
    foreign_to_total_db = foreign_dbfs - total_dbfs
    peak_share = _source_band_peak_share(stem_audio, sample_rate, foreign_source)
    if foreign_source in {"piano", "other"}:
        foreign_like = foreign_to_total_db >= -2.0 and peak_share >= 0.25
    elif foreign_source == "vocals":
        foreign_like = foreign_to_total_db >= -10.0 and peak_share >= 0.35
    elif foreign_source == "drums":
        foreign_like = foreign_to_total_db >= -12.0 and peak_share >= 0.08
    elif foreign_source == "bass":
        foreign_like = foreign_to_total_db >= -1.0
    else:
        foreign_like = True
    return {
        "foreign_like_in_guitar": bool(foreign_like),
        "guitar_foreign_dbfs": round(foreign_dbfs, 6),
        "guitar_total_dbfs": round(total_dbfs, 6),
        "guitar_foreign_to_total_db": round(foreign_to_total_db, 6),
        "guitar_foreign_peak_share": round(peak_share, 6),
    }


def _no_reference_acoustic_leak_failed(
    *,
    tested_stem: str,
    foreign_source: str,
    stem_audio: np.ndarray,
    leak_audio: np.ndarray,
    leak_dbfs: float,
    sample_rate: int,
    detection_floor_db: float,
) -> tuple[bool, dict[str, Any]]:
    if leak_dbfs <= detection_floor_db:
        return False, {}
    if tested_stem == "drums" and foreign_source == "bass":
        evidence = _bass_in_drums_acoustic_evidence(leak_audio, sample_rate)
        return bool(evidence["bass_like_sustained"]), evidence
    if tested_stem == "drums" and foreign_source in {"guitar", "piano", "other"}:
        evidence = _sustained_melodic_in_drums_acoustic_evidence(leak_audio, sample_rate)
        return bool(evidence["melodic_like_sustained"]), evidence
    if tested_stem == "vocals":
        evidence = _foreign_in_vocals_acoustic_evidence(stem_audio, leak_audio, sample_rate, foreign_source)
        return bool(evidence["foreign_like_in_vocal"]), evidence
    if tested_stem == "bass" and foreign_source in {"guitar", "piano", "other"}:
        evidence = _foreign_in_bass_acoustic_evidence(stem_audio, leak_audio, sample_rate, foreign_source)
        return bool(evidence["foreign_like_in_bass"]), evidence
    if tested_stem == "guitar":
        evidence = _foreign_in_guitar_acoustic_evidence(stem_audio, leak_audio, sample_rate, foreign_source)
        return bool(evidence["foreign_like_in_guitar"]), evidence
    return True, {}


def _blocked_unknown_result(
    *,
    tested_stem: str,
    foreign_source: str,
    result_path: Path,
    reason: str,
) -> dict[str, Any]:
    result = {
        "tested_stem": tested_stem,
        "foreign_source": foreign_source,
        "verdict": "blocked_unknown",
        "leak_audio": "",
        "regions": [],
        "reason": reason,
        "detector_mode": "none",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _write_no_reference_result(
    *,
    tested_stem: str,
    foreign_source: str,
    output_dir: Path,
    result_path: Path,
    stem_rate: int,
    verdict: str,
    leak_audio: np.ndarray,
    leak_dbfs: float,
    detection_floor_db: float,
    detector_mode: str,
    spectral_band_hz: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    leak_path = ""
    regions: list[dict[str, float]] = []
    if verdict == "fail":
        audio_path = _write_audio(output_dir / "leaks" / f"{tested_stem}_{foreign_source}.flac", leak_audio, stem_rate)
        leak_path = _relative_path(audio_path, output_dir)
        duration = float(len(leak_audio) / stem_rate)
        regions = [{"start": 0.0, "end": round(duration, 6)}]
    result = {
        "tested_stem": tested_stem,
        "foreign_source": foreign_source,
        "verdict": verdict,
        "leak_audio": leak_path,
        "regions": regions,
        "leak_dbfs": round(leak_dbfs, 4),
        "detection_floor_db": detection_floor_db,
        "detector_mode": detector_mode,
        "spectral_band_hz": spectral_band_hz or [700.0, 1200.0],
    }
    if extra:
        result.update(extra)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def score_stem_contamination(
    *,
    tested_stem_path: Path,
    tested_stem: str,
    foreign_source: str,
    output_dir: Path,
    foreign_reference_path: Path | None = None,
    model_manifest_path: Path | None = None,
    detection_floor_db: float = -60.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{tested_stem}_{foreign_source}.json"

    if foreign_reference_path is None:
        stem_audio, stem_rate = _read_audio(tested_stem_path)
        if model_manifest_path is not None:
            try:
                prediction = predict_no_reference_probability(
                    model_manifest_path=model_manifest_path,
                    audio=stem_audio,
                    sample_rate=stem_rate,
                    tested_stem=tested_stem,
                    foreign_source=foreign_source,
                )
            except Exception as exc:
                return _blocked_unknown_result(
                    tested_stem=tested_stem,
                    foreign_source=foreign_source,
                    result_path=result_path,
                    reason=f"no_reference_model_unavailable:{type(exc).__name__}:{exc}",
                )
            leak_audio = source_band_limited_audio(stem_audio, stem_rate, foreign_source)
            leak_dbfs = _dbfs(_rms(leak_audio))
            probability = float(prediction["probability"])
            threshold = float(prediction["threshold"])
            probability_failed = probability >= threshold
            acoustic_leak_failed, acoustic_evidence = _no_reference_acoustic_leak_failed(
                tested_stem=tested_stem,
                foreign_source=foreign_source,
                stem_audio=stem_audio,
                leak_audio=leak_audio,
                leak_dbfs=leak_dbfs,
                sample_rate=stem_rate,
                detection_floor_db=detection_floor_db,
            )
            verdict = "fail" if probability_failed and acoustic_leak_failed else "pass"
            return _write_no_reference_result(
                tested_stem=tested_stem,
                foreign_source=foreign_source,
                output_dir=output_dir,
                result_path=result_path,
                stem_rate=stem_rate,
                verdict=verdict,
                leak_audio=leak_audio,
                leak_dbfs=leak_dbfs,
                detection_floor_db=detection_floor_db,
                detector_mode="no_reference_neural_v1",
                spectral_band_hz=list(SOURCE_BANDS_HZ.get(foreign_source, (700.0, 1200.0))),
                extra={
                    "contamination_probability": round(probability, 6),
                    "probability_threshold": threshold,
                    "probability_failed": probability_failed,
                    "acoustic_leak_failed": acoustic_leak_failed,
                    "acoustic_evidence": acoustic_evidence,
                    "model_id": prediction["model_id"],
                    "model_manifest": prediction["model_manifest"],
                    "features": prediction["features"],
                    "feature_names": prediction["feature_names"],
                    "shadow_only": True,
                },
            )

        if tested_stem != "minus" or foreign_source != "vocals":
            return _blocked_unknown_result(
                tested_stem=tested_stem,
                foreign_source=foreign_source,
                result_path=result_path,
                reason="no_reference_detector_not_available",
            )

        verdict, leak_audio, leak_dbfs = _no_reference_vocal_in_minus_score(
            stem_audio=stem_audio,
            sample_rate=stem_rate,
            detection_floor_db=detection_floor_db,
        )
        return _write_no_reference_result(
            tested_stem=tested_stem,
            foreign_source=foreign_source,
            output_dir=output_dir,
            result_path=result_path,
            stem_rate=stem_rate,
            verdict=verdict,
            leak_audio=leak_audio,
            leak_dbfs=leak_dbfs,
            detection_floor_db=detection_floor_db,
            detector_mode="no_reference_spectral_v1",
        )

    stem_audio, stem_rate = _read_audio(tested_stem_path)
    foreign_audio, foreign_rate = _read_audio(foreign_reference_path)
    if stem_rate != foreign_rate:
        raise ValueError(f"sample rate mismatch: {stem_rate} != {foreign_rate}")

    leak_audio = _projection_leak(stem_audio, foreign_audio)
    leak_dbfs = _dbfs(_rms(leak_audio))
    verdict = "fail" if leak_dbfs > detection_floor_db else "pass"
    leak_path = ""
    regions: list[dict[str, float]] = []
    if verdict == "fail":
        audio_path = _write_audio(output_dir / "leaks" / f"{tested_stem}_{foreign_source}.flac", leak_audio, stem_rate)
        leak_path = _relative_path(audio_path, output_dir)
        duration = float(len(leak_audio) / stem_rate)
        regions = [{"start": 0.0, "end": round(duration, 6)}]

    result = {
        "tested_stem": tested_stem,
        "foreign_source": foreign_source,
        "verdict": verdict,
        "leak_audio": leak_path,
        "regions": regions,
        "leak_dbfs": round(leak_dbfs, 4),
        "detection_floor_db": detection_floor_db,
        "detector_mode": "reference_projection_v1",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--tested-stem", required=True)
    parser.add_argument("--foreign-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--foreign-reference")
    parser.add_argument("--model-manifest")
    parser.add_argument("--detection-floor-db", type=float, default=-60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_stem_contamination(
        tested_stem_path=Path(args.stem),
        tested_stem=args.tested_stem,
        foreign_source=args.foreign_source,
        output_dir=Path(args.out),
        foreign_reference_path=Path(args.foreign_reference) if args.foreign_reference else None,
        model_manifest_path=Path(args.model_manifest) if args.model_manifest else None,
        detection_floor_db=args.detection_floor_db,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
