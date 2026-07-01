#!/usr/bin/env python3
"""Audit instrumental damage introduced by Mac diagnostic repair candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG = {
    "pad_sec": 0.08,
    "min_region_rms_ratio": 0.72,
    "min_highband_ratio": 0.58,
    "min_centroid_ratio": 0.62,
    "max_region_delta_ratio": 0.42,
    "highband_low_hz": 3000.0,
    "highband_high_hz": 12000.0,
}


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def _fit_audio(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = min(left.shape[0], right.shape[0])
    return np.asarray(left[:frames], dtype=np.float32), np.asarray(right[:frames], dtype=np.float32)


def _mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        return np.mean(array, axis=1).astype(np.float32)
    return array.astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array))))


def _band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    mono = _mono(audio)
    if mono.size <= 2:
        return 0.0
    window = np.hanning(len(mono)).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(mono.astype(np.float64) * window))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    high = min(float(high_hz), float(sample_rate) / 2.0 - 1.0)
    mask = (freqs >= float(low_hz)) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(np.square(spectrum[mask]))))


def _spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    mono = _mono(audio)
    if mono.size <= 2:
        return 0.0
    window = np.hanning(len(mono)).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(mono.astype(np.float64) * window))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    denom = float(np.sum(spectrum) + 1.0e-12)
    return float(np.sum(freqs * spectrum) / denom)


def _stereo_width(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        return 0.0
    mid = array[:, 0] + array[:, 1]
    side = array[:, 0] - array[:, 1]
    return float(_rms(side) / max(_rms(mid), 1.0e-9))


def _region_bounds(region: Mapping[str, Any], *, sample_rate: int, frame_count: int, pad_sec: float) -> tuple[int, int]:
    start = max(0, int(round((float(region.get("start", 0.0)) - float(pad_sec)) * sample_rate)))
    end = min(frame_count, int(round((float(region.get("end", 0.0)) + float(pad_sec)) * sample_rate)))
    return start, max(start, end)


def score_instrumental_damage_from_arrays(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    regions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    reference, candidate = _fit_audio(reference_audio, candidate_audio)
    failed_regions: list[dict[str, Any]] = []
    region_scores: list[dict[str, Any]] = []
    for region in regions:
        start, end = _region_bounds(region, sample_rate=int(sample_rate), frame_count=reference.shape[0], pad_sec=float(cfg["pad_sec"]))
        if end <= start:
            continue
        ref = reference[start:end]
        cand = candidate[start:end]
        ref_rms = _rms(ref)
        cand_rms = _rms(cand)
        rms_ratio = cand_rms / max(ref_rms, 1.0e-9)
        ref_high = _band_energy(ref, int(sample_rate), float(cfg["highband_low_hz"]), float(cfg["highband_high_hz"]))
        cand_high = _band_energy(cand, int(sample_rate), float(cfg["highband_low_hz"]), float(cfg["highband_high_hz"]))
        high_ratio = cand_high / max(ref_high, 1.0e-9)
        ref_centroid = _spectral_centroid(ref, int(sample_rate))
        cand_centroid = _spectral_centroid(cand, int(sample_rate))
        centroid_ratio = cand_centroid / max(ref_centroid, 1.0e-9)
        delta_ratio = _rms(cand - ref) / max(ref_rms, 1.0e-9)
        width_ratio = _stereo_width(cand) / max(_stereo_width(ref), 1.0e-9)
        reasons: list[str] = []
        if rms_ratio < float(cfg["min_region_rms_ratio"]):
            reasons.append("instrumental_region_hole")
        if high_ratio < float(cfg["min_highband_ratio"]) or centroid_ratio < float(cfg["min_centroid_ratio"]):
            reasons.append("instrumental_region_detail_loss")
        if delta_ratio > float(cfg["max_region_delta_ratio"]):
            reasons.append("instrumental_region_overprocessed")
        score = {
            "start": round(float(start) / float(sample_rate), 6),
            "end": round(float(end) / float(sample_rate), 6),
            "rms_ratio": round(float(rms_ratio), 6),
            "highband_ratio": round(float(high_ratio), 6),
            "centroid_ratio": round(float(centroid_ratio), 6),
            "delta_ratio": round(float(delta_ratio), 6),
            "stereo_width_ratio": round(float(width_ratio), 6) if math.isfinite(width_ratio) else 0.0,
            "blocker_reasons": reasons,
        }
        region_scores.append(score)
        if reasons:
            failed_regions.append(score)
    blocker_reasons = sorted({reason for region in failed_regions for reason in region["blocker_reasons"]})
    return {
        "schema": "mac_instrumental_damage_audit.v1",
        "verdict": "fail" if failed_regions else "pass",
        "blocker_reasons": blocker_reasons,
        "failed_region_count": len(failed_regions),
        "regions": region_scores,
        "config": cfg,
    }


def run_damage_audit(
    *,
    reference_instrumental_path: Path,
    candidate_instrumental_path: Path,
    score_json_path: Path,
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start")
    reference, reference_rate = _read_audio(reference_instrumental_path)
    candidate, candidate_rate = _read_audio(candidate_instrumental_path)
    if reference_rate != candidate_rate:
        result = {
            "schema": "mac_instrumental_damage_audit.v1",
            "verdict": "blocked_unknown",
            "blocker_reasons": ["damage_audit_sample_rate_mismatch"],
            "delivery_allowed": False,
        }
        _write_json(output_dir / "damage.json", result)
        return result
    score_payload = json.loads(Path(score_json_path).read_text(encoding="utf-8"))
    regions = list(score_payload.get("regions") or [])
    result = score_instrumental_damage_from_arrays(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=reference_rate,
        regions=regions,
        config=config,
    )
    result.update(
        {
            "reference_instrumental": str(reference_instrumental_path),
            "candidate_instrumental": str(candidate_instrumental_path),
            "score_json": str(score_json_path),
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_audit_diagnostic_only",
        }
    )
    _write_json(output_dir / "damage.json", result)
    _append_progress(progress_path, "manifest", "end", verdict=result["verdict"], failed_region_count=result["failed_region_count"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-instrumental", type=Path, required=True)
    parser.add_argument("--candidate-instrumental", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_damage_audit(
        reference_instrumental_path=args.reference_instrumental,
        candidate_instrumental_path=args.candidate_instrumental,
        score_json_path=args.score_json,
        output_dir=args.out,
    )
    print(json.dumps({"manifest": str(args.out / "damage.json"), "verdict": result["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
