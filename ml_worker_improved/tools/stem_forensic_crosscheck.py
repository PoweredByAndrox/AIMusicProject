"""Cross-stem forensic checks before stem delivery."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stem_contamination_common import FORBIDDEN_SOURCES


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "probe_gains": [0.5, 1.0, 2.0, 4.0],
    "high_band_hz": [1200.0, 10000.0],
    "frame_sec": 0.02,
    "hop_sec": 0.01,
    "region_sec": 0.5,
    "region_hop_sec": 0.25,
    "peak_percentile": 82.0,
    "peak_tolerance_sec": 0.05,
    "min_source_hf_dbfs": -70.0,
    "min_tested_hf_relative_db": -36.0,
    "vocal_drum_score_threshold": 0.55,
    "vocal_drum_flux_corr_threshold": 0.34,
    "vocal_drum_peak_overlap_threshold": 0.50,
    "vocal_drum_projection_rel_db_threshold": -28.0,
    "vocal_drum_projection_corr_threshold": 0.04,
    "generic_env_corr_threshold": 0.84,
    "generic_time_corr_threshold": 0.86,
    "generic_stft_cosine_threshold": 0.84,
}


def _config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if isinstance(config, Mapping):
        merged.update(config)
    return merged


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate, subtype="PCM_24")
    return path


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _dbfs(value: float) -> float:
    if value <= 1.0e-12:
        return -240.0
    return 20.0 * math.log10(value)


def _mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        return np.mean(array, axis=1).astype(np.float32)
    return array.astype(np.float32)


def _fit_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = min(len(left), len(right))
    return left[:frames].astype(np.float32), right[:frames].astype(np.float32)


def _normcorr(left: np.ndarray, right: np.ndarray) -> float:
    left, right = _fit_pair(left, right)
    if left.size <= 1 or right.size <= 1:
        return 0.0
    left64 = left.astype(np.float64) - float(np.mean(left))
    right64 = right.astype(np.float64) - float(np.mean(right))
    denom = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(left64, right64) / denom)


def _band_limited_mono(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    mono = _mono(audio)
    if mono.size <= 1:
        return mono
    spectrum = np.fft.rfft(mono.astype(np.float64))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    high = min(float(high_hz), float(sample_rate) / 2.0 - 1.0)
    mask = (freqs >= float(low_hz)) & (freqs <= high)
    return np.fft.irfft(spectrum * mask, n=len(mono)).astype(np.float32)


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_sec: float, hop_sec: float) -> tuple[np.ndarray, float]:
    frame = max(16, int(round(float(frame_sec) * sample_rate)))
    hop = max(1, int(round(float(hop_sec) * sample_rate)))
    if len(audio) < frame:
        return np.asarray([_rms(audio)], dtype=np.float32), float(hop) / float(sample_rate)
    values = [_rms(audio[start : start + frame]) for start in range(0, len(audio) - frame + 1, hop)]
    return np.asarray(values, dtype=np.float32), float(hop) / float(sample_rate)


def _positive_flux(envelope: np.ndarray) -> np.ndarray:
    if envelope.size <= 1:
        return np.zeros_like(envelope, dtype=np.float32)
    flux = np.maximum(np.diff(envelope, prepend=envelope[0]), 0.0).astype(np.float32)
    max_value = float(np.max(flux))
    if max_value > 1.0e-12:
        flux = flux / max_value
    return flux


def _peak_times(values: np.ndarray, dt_sec: float, percentile: float) -> list[float]:
    if values.size < 3:
        return []
    threshold = max(float(np.percentile(values, percentile)), 0.05)
    times: list[float] = []
    for index in range(1, len(values) - 1):
        if values[index] >= threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            times.append(float(index) * dt_sec)
    return times


def _peak_overlap(left_times: list[float], right_times: list[float], tolerance_sec: float) -> dict[str, Any]:
    if not left_times or not right_times:
        return {"tested_peak_count": len(left_times), "source_peak_count": len(right_times), "tested_near_source": 0.0, "source_near_tested": 0.0}
    left = np.asarray(left_times, dtype=np.float64)
    right = np.asarray(right_times, dtype=np.float64)
    tested_near_source = sum(float(np.min(np.abs(right - value))) <= tolerance_sec for value in left) / len(left)
    source_near_tested = sum(float(np.min(np.abs(left - value))) <= tolerance_sec for value in right) / len(right)
    return {
        "tested_peak_count": int(len(left_times)),
        "source_peak_count": int(len(right_times)),
        "tested_near_source": round(float(tested_near_source), 6),
        "source_near_tested": round(float(source_near_tested), 6),
    }


def _stft_mag_cosine(left: np.ndarray, right: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    left, right = _fit_pair(left, right)
    n_fft = min(1024, max(128, 1 << (max(1, min(len(left), len(right))).bit_length() - 1)))
    hop = max(1, n_fft // 4)
    if len(left) < n_fft or len(right) < n_fft:
        return 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0 - 1.0))
    if not np.any(mask):
        return 0.0
    window = np.hanning(n_fft).astype(np.float64)
    scores: list[float] = []
    for start in range(0, min(len(left), len(right)) - n_fft + 1, hop):
        left_mag = np.abs(np.fft.rfft(left[start : start + n_fft].astype(np.float64) * window))[mask]
        right_mag = np.abs(np.fft.rfft(right[start : start + n_fft].astype(np.float64) * window))[mask]
        denom = float(np.linalg.norm(left_mag) * np.linalg.norm(right_mag))
        if denom > 1.0e-12:
            scores.append(float(np.dot(left_mag, right_mag) / denom))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _local_region_scores(
    *,
    tested_hf: np.ndarray,
    source_hf: np.ndarray,
    sample_rate: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    low_hz, high_hz = [float(value) for value in config["high_band_hz"]]
    region_frames = max(1, int(round(float(config["region_sec"]) * sample_rate)))
    hop_frames = max(1, int(round(float(config["region_hop_sec"]) * sample_rate)))
    frame_sec = float(config["frame_sec"])
    hop_sec = float(config["hop_sec"])
    percentile = float(config["peak_percentile"])
    tolerance = float(config["peak_tolerance_sec"])
    regions: list[dict[str, Any]] = []
    for start in range(0, max(1, len(tested_hf) - region_frames + 1), hop_frames):
        end = min(len(tested_hf), start + region_frames)
        if end - start <= max(64, int(0.08 * sample_rate)):
            continue
        tested_seg = tested_hf[start:end]
        source_seg = source_hf[start:end]
        tested_env, dt_sec = _frame_rms(tested_seg, sample_rate, frame_sec, hop_sec)
        source_env, _ = _frame_rms(source_seg, sample_rate, frame_sec, hop_sec)
        tested_flux = _positive_flux(tested_env)
        source_flux = _positive_flux(source_env)
        tested_peaks = _peak_times(tested_flux, dt_sec, percentile)
        source_peaks = _peak_times(source_flux, dt_sec, percentile)
        overlap = _peak_overlap(tested_peaks, source_peaks, tolerance)
        flux_corr = max(0.0, _normcorr(tested_flux, source_flux))
        env_corr = max(0.0, _normcorr(tested_env, source_env))
        cosine = _stft_mag_cosine(tested_seg, source_seg, sample_rate, low_hz, high_hz)
        overlap_score = max(float(overlap["tested_near_source"]), float(overlap["source_near_tested"]))
        score = 0.38 * flux_corr + 0.25 * env_corr + 0.22 * overlap_score + 0.15 * max(0.0, cosine)
        regions.append(
            {
                "start": round(float(start) / float(sample_rate), 6),
                "end": round(float(end) / float(sample_rate), 6),
                "score": round(float(score), 6),
                "hf_flux_corr": round(float(flux_corr), 6),
                "hf_env_corr": round(float(env_corr), 6),
                "stft_mag_cosine": round(float(cosine), 6),
                "peak_overlap": overlap,
                "tested_hf_dbfs": round(_dbfs(_rms(tested_seg)), 6),
                "source_hf_dbfs": round(_dbfs(_rms(source_seg)), 6),
            }
        )
    return sorted(regions, key=lambda item: float(item["score"]), reverse=True)


def _write_proxy(
    *,
    tested_audio: np.ndarray,
    sample_rate: int,
    regions: list[Mapping[str, Any]],
    output_dir: Path,
    tested_stem: str,
    source_stem: str,
    config: Mapping[str, Any],
) -> str:
    if not regions:
        return ""
    low_hz, high_hz = [float(value) for value in config["high_band_hz"]]
    proxy = np.zeros(len(_mono(tested_audio)), dtype=np.float32)
    tested_hf = _band_limited_mono(tested_audio, sample_rate, low_hz, high_hz)
    for region in regions[:5]:
        start = max(0, int(round(float(region["start"]) * sample_rate)))
        end = min(len(proxy), int(round(float(region["end"]) * sample_rate)))
        proxy[start:end] += tested_hf[start:end]
    peak = float(np.max(np.abs(proxy))) if proxy.size else 0.0
    if peak > 1.0e-12:
        proxy = (proxy / max(1.0, peak / 0.8)).astype(np.float32)
    path = _write_audio(output_dir / "proxy" / f"{tested_stem}_{source_stem}.flac", np.stack([proxy, proxy], axis=1), sample_rate)
    return path.relative_to(output_dir).as_posix()


def _score_vocals_drums(
    *,
    tested_audio: np.ndarray,
    source_audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    tested_stem: str,
    source_stem: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    low_hz, high_hz = [float(value) for value in config["high_band_hz"]]
    tested_hf = _band_limited_mono(tested_audio, sample_rate, low_hz, high_hz)
    full_rms = max(_rms(_mono(tested_audio)), 1.0e-12)
    tested_hf_relative_db = _dbfs(_rms(tested_hf) / full_rms)
    best: dict[str, Any] | None = None
    for gain in [float(value) for value in config["probe_gains"]]:
        source_hf = _band_limited_mono(source_audio * gain, sample_rate, low_hz, high_hz)
        regions = _local_region_scores(tested_hf=tested_hf, source_hf=source_hf, sample_rate=sample_rate, config=config)
        top_region = regions[0] if regions else {}
        source_hf_dbfs = _dbfs(_rms(source_hf))
        denom = float(np.dot(source_hf.astype(np.float64), source_hf.astype(np.float64)) + 1.0e-12)
        projection_coeff = float(np.dot(tested_hf.astype(np.float64), source_hf.astype(np.float64)) / denom)
        projected_source = (projection_coeff * source_hf).astype(np.float32)
        projection_rel_db = _dbfs(_rms(projected_source) / max(_rms(tested_hf), 1.0e-12))
        projection_corr = abs(_normcorr(tested_hf, source_hf))
        projection_score = max(0.0, min(1.0, (projection_rel_db + 36.0) / 24.0))
        score = max(float(top_region.get("score") or 0.0), projection_score)
        if best is None or score >= float(best.get("score") or 0.0):
            best = {
                "probe_gain": gain,
                "score": score,
                "source_hf_dbfs": source_hf_dbfs,
                "tested_hf_relative_db": tested_hf_relative_db,
                "projection_coeff": projection_coeff,
                "projection_corr": projection_corr,
                "projection_rel_db": projection_rel_db,
                "regions": regions[:8],
            }
    best = best or {"probe_gain": 1.0, "score": 0.0, "source_hf_dbfs": -240.0, "tested_hf_relative_db": -240.0, "regions": []}
    top = best["regions"][0] if best["regions"] else {}
    projection_fail = (
        float(best.get("projection_rel_db") or -240.0) >= float(config["vocal_drum_projection_rel_db_threshold"])
        and float(best.get("projection_corr") or 0.0) >= float(config["vocal_drum_projection_corr_threshold"])
        and float(best["source_hf_dbfs"]) >= float(config["min_source_hf_dbfs"])
        and float(best["tested_hf_relative_db"]) >= float(config["min_tested_hf_relative_db"])
    )
    region_fail = (
        float(best["score"]) >= float(config["vocal_drum_score_threshold"])
        and float(best["source_hf_dbfs"]) >= float(config["min_source_hf_dbfs"])
        and float(best["tested_hf_relative_db"]) >= float(config["min_tested_hf_relative_db"])
        and (
            float(top.get("hf_flux_corr") or 0.0) >= float(config["vocal_drum_flux_corr_threshold"])
            or float((top.get("peak_overlap") or {}).get("tested_near_source") or 0.0)
            >= float(config["vocal_drum_peak_overlap_threshold"])
        )
    )
    fail = projection_fail or region_fail
    proxy = ""
    blockers: list[str] = []
    if fail:
        blockers.append("vocal_contains_drum_transient_imprint")
        proxy = _write_proxy(
            tested_audio=tested_audio,
            sample_rate=sample_rate,
            regions=best["regions"],
            output_dir=output_dir,
            tested_stem=tested_stem,
            source_stem=source_stem,
            config=config,
        )
    return {
        "verdict": "fail" if fail else "pass",
        "blocker_reasons": blockers,
        "imprint_score": round(float(best["score"]), 6),
        "best_probe_gain": float(best["probe_gain"]),
        "source_hf_dbfs": round(float(best["source_hf_dbfs"]), 6),
        "tested_hf_relative_db": round(float(best["tested_hf_relative_db"]), 6),
        "projection_coeff": round(float(best.get("projection_coeff") or 0.0), 8),
        "projection_corr": round(float(best.get("projection_corr") or 0.0), 6),
        "projection_rel_db": round(float(best.get("projection_rel_db") or -240.0), 6),
        "projection_fail": bool(projection_fail),
        "region_fail": bool(region_fail),
        "regions": best["regions"],
        "forensic_proxy_audio": proxy,
    }


def _score_generic_pair(
    *,
    tested_audio: np.ndarray,
    source_audio: np.ndarray,
    sample_rate: int,
    tested_stem: str,
    source_stem: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    low_hz, high_hz = [float(value) for value in config["high_band_hz"]]
    tested, source = _fit_pair(_mono(tested_audio), _mono(source_audio))
    tested_env, _ = _frame_rms(tested, sample_rate, float(config["frame_sec"]), float(config["hop_sec"]))
    source_env, _ = _frame_rms(source, sample_rate, float(config["frame_sec"]), float(config["hop_sec"]))
    time_corr = abs(_normcorr(tested, source))
    env_corr = max(0.0, _normcorr(tested_env, source_env))
    cosine = _stft_mag_cosine(tested, source, sample_rate, low_hz, high_hz)
    fail = time_corr >= float(config["generic_time_corr_threshold"]) or (
        env_corr >= float(config["generic_env_corr_threshold"]) and cosine >= float(config["generic_stft_cosine_threshold"])
    )
    blockers = [f"stem_cross_source_imprint:{tested_stem}:{source_stem}"] if fail else []
    return {
        "verdict": "fail" if fail else "pass",
        "blocker_reasons": blockers,
        "imprint_score": round(max(time_corr, env_corr * cosine), 6),
        "time_corr": round(float(time_corr), 6),
        "env_corr": round(float(env_corr), 6),
        "stft_mag_cosine": round(float(cosine), 6),
        "regions": [],
        "forensic_proxy_audio": "",
        "best_probe_gain": 1.0,
    }


def score_pair_forensic(
    *,
    tested_stem: str,
    source_stem: str,
    tested_path: Path,
    source_path: Path,
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config(config)
    result_path = output_dir / f"{tested_stem}_{source_stem}.json"
    try:
        tested_audio, tested_rate = _read_audio(tested_path)
        source_audio, source_rate = _read_audio(source_path)
    except Exception as exc:
        result = {
            "schema": "stem_pair_forensic.v1",
            "tested_stem": tested_stem,
            "source_stem": source_stem,
            "tested_path": str(tested_path),
            "source_path": str(source_path),
            "verdict": "blocked_unknown",
            "blocker_reasons": [f"forensic_read_failed:{type(exc).__name__}"],
            "regions": [],
            "forensic_proxy_audio": "",
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    if tested_rate != source_rate:
        result = {
            "schema": "stem_pair_forensic.v1",
            "tested_stem": tested_stem,
            "source_stem": source_stem,
            "tested_path": str(tested_path),
            "source_path": str(source_path),
            "verdict": "blocked_unknown",
            "blocker_reasons": [f"forensic_sample_rate_mismatch:{tested_stem}:{source_stem}"],
            "regions": [],
            "forensic_proxy_audio": "",
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    frames = min(len(tested_audio), len(source_audio))
    tested_audio = tested_audio[:frames]
    source_audio = source_audio[:frames]
    if tested_stem == "vocals" and source_stem == "drums":
        score = _score_vocals_drums(
            tested_audio=tested_audio,
            source_audio=source_audio,
            sample_rate=tested_rate,
            output_dir=output_dir,
            tested_stem=tested_stem,
            source_stem=source_stem,
            config=cfg,
        )
    else:
        score = _score_generic_pair(
            tested_audio=tested_audio,
            source_audio=source_audio,
            sample_rate=tested_rate,
            tested_stem=tested_stem,
            source_stem=source_stem,
            config=cfg,
        )
    result = {
        "schema": "stem_pair_forensic.v1",
        "tested_stem": tested_stem,
        "source_stem": source_stem,
        "tested_path": str(tested_path),
        "source_path": str(source_path),
        "detector_mode": "cross_stem_forensic_v1",
        **score,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def score_cross_stem_forensic_matrix(
    stem_paths: Mapping[str, str],
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config(config)
    if cfg.get("enabled") is False:
        result = {
            "schema": "cross_stem_forensic_matrix.v1",
            "verdict": "blocked_unknown",
            "blocker_reasons": ["cross_stem_forensic_disabled"],
            "pair_results": [],
            "config": cfg,
        }
        (output_dir / "matrix.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    blockers: list[str] = []
    pair_results: list[dict[str, Any]] = []
    for tested_stem, forbidden_sources in FORBIDDEN_SOURCES.items():
        tested_path = stem_paths.get(tested_stem)
        if not tested_path:
            blockers.append(f"missing_forensic_tested_stem:{tested_stem}")
            continue
        for source_stem in forbidden_sources:
            source_path = stem_paths.get(source_stem)
            if not source_path:
                blockers.append(f"missing_forensic_source_stem:{tested_stem}:{source_stem}")
                continue
            pair = score_pair_forensic(
                tested_stem=tested_stem,
                source_stem=source_stem,
                tested_path=Path(str(tested_path)),
                source_path=Path(str(source_path)),
                output_dir=output_dir / "pairs",
                config=cfg,
            )
            pair_results.append(pair)
            if pair["verdict"] == "fail":
                blockers.extend(str(reason) for reason in pair.get("blocker_reasons") or [])
            elif pair["verdict"] == "blocked_unknown":
                blockers.append(f"forensic_unknown:{tested_stem}:{source_stem}")
    if any(reason.startswith("missing_forensic") or reason.startswith("forensic_unknown") for reason in blockers):
        verdict = "blocked_unknown"
    elif blockers:
        verdict = "fail"
    else:
        verdict = "auto_accept"
    result = {
        "schema": "cross_stem_forensic_matrix.v1",
        "verdict": verdict,
        "blocker_reasons": sorted(set(blockers)),
        "pair_results": pair_results,
        "config": cfg,
    }
    (output_dir / "matrix.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def validate_listen_pack_forensic(
    listen_audio_paths: Mapping[str, str],
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stem_paths = {stem: path for stem, path in listen_audio_paths.items() if stem != "mix"}
    return score_cross_stem_forensic_matrix(stem_paths=stem_paths, output_dir=output_dir, config=config)
