"""Pitch-robust vocal leak guard for Mac MelBand diagnostic outputs.

This module is deliberately diagnostic-only. It can find and repair vocal-like
material that leaked into MelBand instrumental output after pitch/formant-like
transforms, but it cannot publish listen/result artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import soundfile as sf


DEFAULT_CONFIG: dict[str, Any] = {
    "probe_semitones": [-12.0, -9.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0],
    "window_sec": 0.75,
    "hop_sec": 0.25,
    "vocal_band_hz": [90.0, 4200.0],
    "score_threshold": 0.56,
    "projection_rel_db_threshold": -31.0,
    "strong_projection_rel_db_threshold": -23.5,
    "strong_projection_stft_cosine_threshold": 0.50,
    "min_positive_projection_coeff": 0.02,
    "min_probe_dbfs": -66.0,
    "min_instrumental_dbfs": -66.0,
    "max_regions": 16,
    "repair_region_pad_sec": 0.08,
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
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_24")
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, ensure_ascii=False) + "\n")
        handle.flush()


def _mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        return np.mean(array, axis=1).astype(np.float32)
    return array.astype(np.float32)


def _fit_audio(*arrays: np.ndarray) -> list[np.ndarray]:
    frames = min(len(array) for array in arrays)
    return [np.asarray(array[:frames], dtype=np.float32) for array in arrays]


def _rms(audio: np.ndarray) -> float:
    if np.asarray(audio).size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


def _dbfs(value: float) -> float:
    if value <= 1.0e-12:
        return -240.0
    return 20.0 * math.log10(float(value))


def _normcorr(left: np.ndarray, right: np.ndarray) -> float:
    left, right = _fit_audio(_mono(left), _mono(right))
    if len(left) <= 1 or len(right) <= 1:
        return 0.0
    left64 = left.astype(np.float64) - float(np.mean(left))
    right64 = right.astype(np.float64) - float(np.mean(right))
    denom = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(left64, right64) / denom)


def _band_limited(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    mono = _mono(audio)
    if mono.size <= 1:
        return mono
    spectrum = np.fft.rfft(mono.astype(np.float64))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    high = min(float(high_hz), float(sample_rate) / 2.0 - 1.0)
    mask = (freqs >= float(low_hz)) & (freqs <= high)
    return np.fft.irfft(spectrum * mask, n=len(mono)).astype(np.float32)


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_sec: float = 0.04, hop_sec: float = 0.02) -> np.ndarray:
    frame = max(32, int(round(float(frame_sec) * sample_rate)))
    hop = max(1, int(round(float(hop_sec) * sample_rate)))
    if len(audio) < frame:
        return np.asarray([_rms(audio)], dtype=np.float32)
    return np.asarray([_rms(audio[start : start + frame]) for start in range(0, len(audio) - frame + 1, hop)], dtype=np.float32)


def _stft_mag_cosine(left: np.ndarray, right: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    left, right = _fit_audio(_mono(left), _mono(right))
    n_fft = min(1024, max(128, 1 << (max(1, min(len(left), len(right))).bit_length() - 1)))
    if len(left) < n_fft or len(right) < n_fft:
        return 0.0
    hop = max(1, n_fft // 4)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0 - 1.0))
    if not np.any(mask):
        return 0.0
    window = np.hanning(n_fft).astype(np.float64)
    values: list[float] = []
    for start in range(0, min(len(left), len(right)) - n_fft + 1, hop):
        left_mag = np.abs(np.fft.rfft(left[start : start + n_fft].astype(np.float64) * window))[mask]
        right_mag = np.abs(np.fft.rfft(right[start : start + n_fft].astype(np.float64) * window))[mask]
        denom = float(np.linalg.norm(left_mag) * np.linalg.norm(right_mag))
        if denom > 1.0e-12:
            values.append(float(np.dot(left_mag, right_mag) / denom))
    return float(np.mean(values)) if values else 0.0


def pitch_shift_resample_same_length(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Frequency-bin pitch transform that keeps duration unchanged.

    It is not a high-quality audio effect. It is a deterministic probe transform
    for leak detection and tests where pitch-shifted vocal content must still
    count as vocal content.
    """

    array = np.asarray(audio, dtype=np.float32)
    if abs(float(semitones)) <= 1.0e-9 or array.size == 0:
        return array.copy()
    if array.ndim == 2:
        shifted_channels = [pitch_shift_resample_same_length(array[:, channel], semitones) for channel in range(array.shape[1])]
        return np.stack(shifted_channels, axis=1).astype(np.float32)

    factor = 2.0 ** (float(semitones) / 12.0)
    spectrum = np.fft.rfft(array.astype(np.float64))
    shifted = np.zeros_like(spectrum)
    for source_bin, value in enumerate(spectrum):
        target_bin = int(round(float(source_bin) * factor))
        if 0 <= target_bin < len(shifted):
            shifted[target_bin] += value
    output = np.fft.irfft(shifted, n=len(array)).astype(np.float32)
    in_rms = _rms(array)
    out_rms = _rms(output)
    if in_rms > 1.0e-12 and out_rms > 1.0e-12:
        output = (output * min(4.0, in_rms / out_rms)).astype(np.float32)
    return output


def _score_window(
    *,
    instrumental_seg: np.ndarray,
    probe_seg: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> dict[str, float]:
    inst_band = _band_limited(instrumental_seg, sample_rate, low_hz, high_hz)
    probe_band = _band_limited(probe_seg, sample_rate, low_hz, high_hz)
    inst_rms = _rms(inst_band)
    probe_rms = _rms(probe_band)
    denom = float(np.dot(probe_band.astype(np.float64), probe_band.astype(np.float64)) + 1.0e-12)
    projection_coeff = float(np.dot(inst_band.astype(np.float64), probe_band.astype(np.float64)) / denom)
    projected = (projection_coeff * probe_band).astype(np.float32)
    projection_rel_db = _dbfs(_rms(projected) / max(inst_rms, 1.0e-12))
    time_corr = abs(_normcorr(inst_band, probe_band))
    inst_env = _frame_rms(inst_band, sample_rate)
    probe_env = _frame_rms(probe_band, sample_rate)
    env_corr = max(0.0, _normcorr(inst_env, probe_env))
    cosine = _stft_mag_cosine(inst_band, probe_band, sample_rate, low_hz, high_hz)
    projection_score = max(0.0, min(1.0, (projection_rel_db + 42.0) / 24.0))
    score = max(time_corr, 0.48 * env_corr + 0.32 * cosine + 0.20 * projection_score)
    return {
        "score": float(score),
        "time_corr": float(time_corr),
        "env_corr": float(env_corr),
        "stft_mag_cosine": float(cosine),
        "projection_coeff": float(projection_coeff),
        "projection_rel_db": float(projection_rel_db),
        "instrumental_dbfs": float(_dbfs(inst_rms)),
        "probe_dbfs": float(_dbfs(probe_rms)),
    }


def _failure_taxonomy(*, fail_regions: list[dict[str, Any]], selected_regions: list[dict[str, Any]], max_regions: int) -> dict[str, Any]:
    by_fail_mode: dict[str, int] = {}
    by_probe: dict[str, int] = {}
    for region in fail_regions:
        fail_mode = str(region.get("fail_mode") or "unknown")
        by_fail_mode[fail_mode] = by_fail_mode.get(fail_mode, 0) + 1
        probe_key = f"{float(region.get('semitones') or 0.0):.1f}"
        by_probe[probe_key] = by_probe.get(probe_key, 0) + 1
    total_fail_regions = len(fail_regions)
    selected_region_count = len(selected_regions)
    return {
        "total_fail_regions": int(total_fail_regions),
        "selected_region_count": int(selected_region_count),
        "max_regions": int(max_regions),
        "regions_capped": bool(total_fail_regions > selected_region_count),
        "by_fail_mode": dict(sorted(by_fail_mode.items())),
        "by_probe": dict(sorted(by_probe.items())),
    }


def score_region_count(score: Mapping[str, Any] | None) -> int:
    if not isinstance(score, Mapping):
        return 0
    total_fail_regions = score.get("total_fail_regions")
    if isinstance(total_fail_regions, (int, float)):
        return max(0, int(total_fail_regions))
    taxonomy = score.get("failure_taxonomy")
    if isinstance(taxonomy, Mapping):
        taxonomy_total = taxonomy.get("total_fail_regions")
        if isinstance(taxonomy_total, (int, float)):
            return max(0, int(taxonomy_total))
    return len(score.get("regions") or [])


def score_best_value(score: Mapping[str, Any] | None) -> float:
    if not isinstance(score, Mapping):
        return 0.0
    best_probe = score.get("best_probe")
    if isinstance(best_probe, Mapping):
        best_probe_score = best_probe.get("score")
        if isinstance(best_probe_score, (int, float)):
            return float(best_probe_score)
    region_scores = [
        float(region.get("score"))
        for region in list(score.get("regions") or [])
        if isinstance(region, Mapping) and isinstance(region.get("score"), (int, float))
    ]
    return max(region_scores, default=0.0)


def score_target_issue_improved_from_values(
    *,
    before_region_count: int,
    after_region_count: int,
    before_best_score: float,
    after_best_score: float,
    after_verdict: str,
    allow_score_only: bool = False,
) -> bool:
    if str(after_verdict) == "pass":
        return True
    if int(after_region_count) < int(before_region_count):
        return True
    if bool(allow_score_only):
        return int(after_region_count) <= int(before_region_count) and float(after_best_score) < float(before_best_score)
    return False


def score_target_issue_improved(
    *,
    before_score: Mapping[str, Any] | None,
    after_score: Mapping[str, Any] | None,
    allow_score_only: bool = False,
) -> bool:
    after_verdict = str(after_score.get("verdict")) if isinstance(after_score, Mapping) else ""
    return score_target_issue_improved_from_values(
        before_region_count=score_region_count(before_score),
        after_region_count=score_region_count(after_score),
        before_best_score=score_best_value(before_score),
        after_best_score=score_best_value(after_score),
        after_verdict=after_verdict,
        allow_score_only=bool(allow_score_only),
    )


def score_pitch_shifted_vocal_leak(
    *,
    vocals_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config(config)
    vocals_audio, vocals_rate = _read_audio(Path(vocals_path))
    instrumental_audio, instrumental_rate = _read_audio(Path(instrumental_path))
    if vocals_rate != instrumental_rate:
        result = {
            "schema": "mac_melband_pitch_guard_score.v1",
            "detector_mode": "pitch_robust_cross_transform_v1",
            "verdict": "blocked_unknown",
            "blocker_reasons": ["pitch_guard_sample_rate_mismatch"],
            "regions": [],
            "total_fail_regions": 0,
            "selected_region_count": 0,
            "regions_capped": False,
            "failure_taxonomy": _failure_taxonomy(fail_regions=[], selected_regions=[], max_regions=int(cfg["max_regions"])),
            "best_probe": {"score": 0.0},
            "config": cfg,
        }
        _write_json(output_dir / "pitch.json", result)
        return result

    vocals_audio, instrumental_audio = _fit_audio(vocals_audio, instrumental_audio)
    low_hz, high_hz = [float(value) for value in cfg["vocal_band_hz"]]
    window_frames = max(1, int(round(float(cfg["window_sec"]) * vocals_rate)))
    hop_frames = max(1, int(round(float(cfg["hop_sec"]) * vocals_rate)))
    score_threshold = float(cfg["score_threshold"])
    projection_threshold = float(cfg["projection_rel_db_threshold"])
    strong_projection_threshold = float(cfg["strong_projection_rel_db_threshold"])
    strong_projection_stft_threshold = float(cfg["strong_projection_stft_cosine_threshold"])
    min_positive_projection_coeff = float(cfg["min_positive_projection_coeff"])
    min_probe_dbfs = float(cfg["min_probe_dbfs"])
    min_instrumental_dbfs = float(cfg["min_instrumental_dbfs"])
    best_probe: dict[str, Any] = {"score": 0.0, "semitones": 0.0}
    fail_regions: list[dict[str, Any]] = []
    scanned = 0

    probe_values = [float(value) for value in cfg["probe_semitones"]]
    total_windows_per_probe = max(1, len(range(0, max(1, len(vocals_audio) - window_frames + 1), hop_frames)))
    for probe_index, semitones in enumerate(probe_values):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "probe_start",
                    "probe_index": int(probe_index),
                    "probe_count": int(len(probe_values)),
                    "semitones": float(semitones),
                    "windows": int(total_windows_per_probe),
                }
            )
        probe_audio = pitch_shift_resample_same_length(vocals_audio, semitones)
        for window_index, start in enumerate(range(0, max(1, len(vocals_audio) - window_frames + 1), hop_frames)):
            end = min(len(vocals_audio), start + window_frames)
            if end - start <= max(64, int(round(0.08 * vocals_rate))):
                continue
            scanned += 1
            scores = _score_window(
                instrumental_seg=instrumental_audio[start:end],
                probe_seg=probe_audio[start:end],
                sample_rate=vocals_rate,
                low_hz=low_hz,
                high_hz=high_hz,
            )
            candidate = {
                "start": round(float(start) / float(vocals_rate), 6),
                "end": round(float(end) / float(vocals_rate), 6),
                "semitones": semitones,
                **{key: round(float(value), 6) for key, value in scores.items()},
            }
            eligible_for_best_probe = (
                float(candidate["probe_dbfs"]) >= min_probe_dbfs
                and float(candidate["instrumental_dbfs"]) >= min_instrumental_dbfs
                and float(candidate["projection_coeff"]) >= min_positive_projection_coeff
            )
            if eligible_for_best_probe and float(candidate["score"]) > float(best_probe.get("score") or 0.0):
                best_probe = dict(candidate)
            standard_fail = (
                float(candidate["score"]) >= score_threshold
                and float(candidate["projection_rel_db"]) >= projection_threshold
                and float(candidate["projection_coeff"]) >= min_positive_projection_coeff
                and float(candidate["probe_dbfs"]) >= min_probe_dbfs
                and float(candidate["instrumental_dbfs"]) >= min_instrumental_dbfs
            )
            strong_projection_fail = (
                float(candidate["projection_rel_db"]) >= strong_projection_threshold
                and float(candidate["stft_mag_cosine"]) >= strong_projection_stft_threshold
                and float(candidate["projection_coeff"]) >= min_positive_projection_coeff
                and float(candidate["probe_dbfs"]) >= min_probe_dbfs
                and float(candidate["instrumental_dbfs"]) >= min_instrumental_dbfs
            )
            is_fail = standard_fail or strong_projection_fail
            if is_fail:
                candidate["fail_mode"] = "standard_score" if standard_fail else "strong_projection_stft"
                fail_regions.append(candidate)
            if progress_callback is not None and (
                window_index == 0
                or window_index == total_windows_per_probe - 1
                or (window_index + 1) % max(1, total_windows_per_probe // 4) == 0
            ):
                progress_callback(
                    {
                        "event": "probe_progress",
                        "probe_index": int(probe_index),
                        "probe_count": int(len(probe_values)),
                        "semitones": float(semitones),
                        "window_index": int(window_index + 1),
                        "windows": int(total_windows_per_probe),
                        "fail_regions_seen": int(len(fail_regions)),
                        "best_score": float(best_probe.get("score") or 0.0),
                    }
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "probe_end",
                    "probe_index": int(probe_index),
                    "probe_count": int(len(probe_values)),
                    "semitones": float(semitones),
                    "scanned_windows": int(scanned),
                    "fail_regions_seen": int(len(fail_regions)),
                }
            )

    fail_regions = sorted(fail_regions, key=lambda item: float(item["score"]), reverse=True)
    selected_regions = fail_regions[: int(cfg["max_regions"])]
    all_regions_path = output_dir / "regions.jsonl"
    _write_jsonl(all_regions_path, fail_regions)
    taxonomy = _failure_taxonomy(
        fail_regions=fail_regions,
        selected_regions=selected_regions,
        max_regions=int(cfg["max_regions"]),
    )
    verdict = "fail" if fail_regions else "pass"
    result = {
        "schema": "mac_melband_pitch_guard_score.v1",
        "detector_mode": "pitch_robust_cross_transform_v1",
        "tested_stem": "instrumental",
        "source_stem": "vocals",
        "verdict": verdict,
        "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"] if fail_regions else [],
        "regions": selected_regions,
        "all_regions_path": str(all_regions_path),
        "all_regions_format": "jsonl",
        "all_region_count": len(fail_regions),
        "total_fail_regions": taxonomy["total_fail_regions"],
        "selected_region_count": taxonomy["selected_region_count"],
        "regions_capped": taxonomy["regions_capped"],
        "failure_taxonomy": taxonomy,
        "best_probe": {key: (round(float(value), 6) if isinstance(value, float) else value) for key, value in best_probe.items()},
        "scanned_windows": int(scanned),
        "config": cfg,
    }
    _write_json(output_dir / "pitch.json", result)
    return result


def _fade_mask(length: int) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype=np.float32)
    edge = min(length // 4, max(8, length // 16))
    mask = np.ones(length, dtype=np.float32)
    ramp = np.sin(np.linspace(0.0, math.pi / 2.0, edge, dtype=np.float32)) ** 2
    mask[:edge] = ramp
    mask[-edge:] = ramp[::-1]
    return mask


def apply_pitch_guard_repair(
    *,
    vocals_path: Path,
    instrumental_path: Path,
    raw_ref_path: Path,
    score: Mapping[str, Any],
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config(config if config is not None else score.get("config") if isinstance(score.get("config"), Mapping) else None)
    vocals_audio, vocals_rate = _read_audio(Path(vocals_path))
    instrumental_audio, instrumental_rate = _read_audio(Path(instrumental_path))
    raw_audio, raw_rate = _read_audio(Path(raw_ref_path))
    if len({vocals_rate, instrumental_rate, raw_rate}) != 1:
        raise ValueError("pitch_guard_repair_sample_rate_mismatch")
    vocals_audio, instrumental_audio, raw_audio = _fit_audio(vocals_audio, instrumental_audio, raw_audio)
    if not score.get("regions"):
        vocals_out = output_dir / "v.flac"
        instrumental_out = output_dir / "i.flac"
        _write_audio(vocals_out, vocals_audio, vocals_rate)
        _write_audio(instrumental_out, instrumental_audio, vocals_rate)
        manifest = {
            "schema": "mac_melband_pitch_guard_repair.v1",
            "status": "skipped_no_regions",
            "repaired_vocals_path": str(vocals_out),
            "repaired_instrumental_path": str(instrumental_out),
            "reconstruction_rms": _rms(vocals_audio + instrumental_audio - raw_audio),
            "delivery_allowed": False,
        }
        _write_json(output_dir / "manifest.json", manifest)
        return manifest

    repaired_vocals = vocals_audio.copy()
    repaired_instrumental = instrumental_audio.copy()
    applied_regions: list[dict[str, Any]] = []
    low_hz, high_hz = [float(value) for value in cfg["vocal_band_hz"]]
    pad_frames = max(0, int(round(float(cfg["repair_region_pad_sec"]) * vocals_rate)))
    for region in sorted(score.get("regions") or [], key=lambda item: float(item.get("score") or 0.0), reverse=True):
        semitones = float(region.get("semitones") or (score.get("best_probe") or {}).get("semitones") or 0.0)
        probe_audio = pitch_shift_resample_same_length(vocals_audio, semitones)
        start = max(0, int(round(float(region["start"]) * vocals_rate)) - pad_frames)
        end = min(len(repaired_vocals), int(round(float(region["end"]) * vocals_rate)) + pad_frames)
        if end <= start:
            continue
        probe_seg = probe_audio[start:end]
        inst_seg = repaired_instrumental[start:end]
        probe_band = _band_limited(probe_seg, vocals_rate, low_hz, high_hz)
        inst_band = _band_limited(inst_seg, vocals_rate, low_hz, high_hz)
        denom = float(np.dot(probe_band.astype(np.float64), probe_band.astype(np.float64)) + 1.0e-12)
        coeff = float(np.dot(inst_band.astype(np.float64), probe_band.astype(np.float64)) / denom)
        coeff = max(-2.5, min(2.5, coeff))
        mask = _fade_mask(end - start)
        if probe_seg.ndim == 1:
            patch = (coeff * probe_seg * mask).astype(np.float32)
        else:
            patch = (coeff * probe_seg * mask[:, None]).astype(np.float32)
        repaired_instrumental[start:end] -= patch
        repaired_vocals[start:end] += patch
        applied_regions.append(
            {
                "start": round(float(start) / float(vocals_rate), 6),
                "end": round(float(end) / float(vocals_rate), 6),
                "semitones": semitones,
                "projection_coeff": round(coeff, 8),
                "source_score": region.get("score"),
            }
        )

    vocals_out = _write_audio(output_dir / "v.flac", repaired_vocals, vocals_rate)
    instrumental_out = _write_audio(output_dir / "i.flac", repaired_instrumental, vocals_rate)
    reconstruction_rms = _rms(repaired_vocals + repaired_instrumental - raw_audio)
    manifest = {
        "schema": "mac_melband_pitch_guard_repair.v1",
        "status": "repaired",
        "repaired_vocals_path": str(vocals_out),
        "repaired_instrumental_path": str(instrumental_out),
        "applied_regions": applied_regions,
        "reconstruction_rms": round(float(reconstruction_rms), 10),
        "delivery_allowed": False,
        "presentation_status": "diagnostic_melband_pitch_guard_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _copy_audio(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _resolve_existing_audio_path(path_value: str, melband_run_dir: Path) -> Path:
    path = Path(str(path_value or ""))
    if path.exists():
        return path
    candidate = Path(melband_run_dir) / path.name
    if path.name and candidate.exists():
        return candidate
    raise FileNotFoundError(f"missing audio path: {path}")


def _resolve_manifest_audio_paths(manifest: Mapping[str, Any], melband_run_dir: Path) -> tuple[Path, Path, Path]:
    stems = dict(manifest.get("stems") or {})
    artifacts = dict(manifest.get("artifacts") or {})
    vocals_path = _resolve_existing_audio_path(
        str((stems.get("vocals") or {}).get("path") or artifacts.get("vocals_flac") or ""),
        melband_run_dir,
    )
    instrumental_path = _resolve_existing_audio_path(
        str((stems.get("instrumental") or {}).get("path") or artifacts.get("instrumental_flac") or ""),
        melband_run_dir,
    )
    raw_ref_path = _resolve_existing_audio_path(str(artifacts.get("raw_ref_flac") or ""), melband_run_dir)
    return vocals_path, instrumental_path, raw_ref_path


def run_pitch_guard_on_existing_melband_run(
    *,
    melband_run_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "load_manifest", "start")
    source_manifest_path = Path(melband_run_dir) / "stem_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    vocals_src, instrumental_src, raw_src = _resolve_manifest_audio_paths(source_manifest, Path(melband_run_dir))
    _append_progress(progress_path, "copy_short_audio", "start")
    audio_dir = output_dir / "audio"
    vocals_path = _copy_audio(vocals_src, audio_dir / "v.flac")
    instrumental_path = _copy_audio(instrumental_src, audio_dir / "i.flac")
    raw_ref_path = _copy_audio(raw_src, audio_dir / "raw.flac")
    _append_progress(progress_path, "pitch_scan", "start")
    score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        output_dir=output_dir / "score",
        config=config,
    )
    repair_manifest: dict[str, Any] | None = None
    after_score: dict[str, Any] | None = None
    if score["verdict"] == "fail":
        _append_progress(progress_path, "repair", "start", blocker_reasons=score["blocker_reasons"])
        repair_manifest = apply_pitch_guard_repair(
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            raw_ref_path=raw_ref_path,
            score=score,
            output_dir=output_dir / "repair",
        )
        after_score = score_pitch_shifted_vocal_leak(
            vocals_path=Path(str(repair_manifest["repaired_vocals_path"])),
            instrumental_path=Path(str(repair_manifest["repaired_instrumental_path"])),
            output_dir=output_dir / "repair_score",
            config=config,
        )
        _append_progress(progress_path, "repair", "end", after_verdict=after_score["verdict"])
    blocker_reasons = list(score.get("blocker_reasons") or [])
    if after_score is not None and after_score.get("verdict") == "fail":
        blocker_reasons.extend(str(reason) for reason in after_score.get("blocker_reasons") or [])
    delivery_status = "blocked" if blocker_reasons else "diagnostic_passed"
    manifest = {
        "schema": "mac_melband_pitch_guard_run.v1",
        "profile": "melband_pitch_guard_v1",
        "source_run_dir": str(Path(melband_run_dir)),
        "source_manifest": str(source_manifest_path),
        "source_model_id": source_manifest.get("model_id") or source_manifest.get("model_name"),
        "source_audio_path": source_manifest.get("source_audio_path"),
        "audio": {
            "vocals": str(vocals_path),
            "instrumental": str(instrumental_path),
            "raw_ref": str(raw_ref_path),
        },
        "score": score,
        "repair": repair_manifest,
        "after_repair_score": after_score,
        "blocker_reasons": sorted(set(blocker_reasons)),
        "delivery_status": delivery_status,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "diagnostic_melband_pitch_guard_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", delivery_status=delivery_status)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pitch-robust guard on an existing Mac MelBand run.")
    parser.add_argument("--melband-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config_json) if args.config_json else None
    manifest = run_pitch_guard_on_existing_melband_run(
        melband_run_dir=args.melband_run_dir,
        output_dir=args.out,
        config=config,
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "delivery_allowed": manifest["delivery_allowed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
