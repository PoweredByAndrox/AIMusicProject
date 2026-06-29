"""Build a weak teacher target that removes drum-aligned transients from vocals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stem_forensic_crosscheck import DEFAULT_CONFIG, score_pair_forensic


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate, subtype="PCM_24")
    return path


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _dbfs(value: float) -> float:
    if value <= 1.0e-12:
        return -240.0
    return 20.0 * math.log10(value)


def _match_audio(audio: np.ndarray, frames: int, channels: int) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32)
    if len(output) > frames:
        output = output[:frames]
    elif len(output) < frames:
        output = np.pad(output, ((0, frames - len(output)), (0, 0)), mode="constant")
    if output.shape[1] == channels:
        return output.astype(np.float32)
    if output.shape[1] == 1 and channels == 2:
        return np.repeat(output, 2, axis=1).astype(np.float32)
    return output[:, :channels].astype(np.float32)


def _band_limited_audio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    if len(audio) <= 1:
        return audio.astype(np.float32)
    spectrum = np.fft.rfft(audio.astype(np.float64), axis=0)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / float(sample_rate))
    high = min(float(high_hz), float(sample_rate) / 2.0 - 1.0)
    mask = ((freqs >= float(low_hz)) & (freqs <= high))[:, None]
    return np.fft.irfft(spectrum * mask, n=len(audio), axis=0).astype(np.float32)


def _region_window(frames: int) -> np.ndarray:
    if frames <= 1:
        return np.ones((max(0, frames), 1), dtype=np.float32)
    fade = min(frames // 4, max(1, frames // 12))
    window = np.ones(frames, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    window[:fade] = ramp
    window[-fade:] = ramp[::-1]
    return window[:, None]


def _projection_coeff(target: np.ndarray, basis: np.ndarray, *, max_projection_gain: float) -> float:
    denom = float(np.sum(np.square(basis.astype(np.float64))) + 1.0e-12)
    coeff = float(np.sum(target.astype(np.float64) * basis.astype(np.float64)) / denom)
    return float(np.clip(coeff, -float(max_projection_gain), float(max_projection_gain)))


def _repair_regions(
    *,
    vocals: np.ndarray,
    drums: np.ndarray,
    sample_rate: int,
    regions: list[Mapping[str, Any]],
    strength: float,
    self_hf_attenuation: float,
    region_hf_attenuation: float,
    hard_region_hf_suppression: bool,
    max_projection_gain: float,
    high_band_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    repaired = vocals.astype(np.float32).copy()
    removed = np.zeros_like(repaired, dtype=np.float32)
    drum_hf = _band_limited_audio(drums, sample_rate, high_band_hz[0], high_band_hz[1])
    vocal_hf = _band_limited_audio(vocals, sample_rate, high_band_hz[0], high_band_hz[1])
    applied: list[dict[str, Any]] = []
    for region in regions:
        start = max(0, int(round(float(region.get("start", 0.0)) * sample_rate)))
        end = min(len(repaired), int(round(float(region.get("end", 0.0)) * sample_rate)))
        if end <= start:
            continue
        coeff = _projection_coeff(vocal_hf[start:end], drum_hf[start:end], max_projection_gain=max_projection_gain)
        drum_segment = drum_hf[start:end]
        source_env = np.max(np.abs(drum_segment.astype(np.float64)), axis=1)
        env_peak = float(np.max(source_env)) if source_env.size else 0.0
        if env_peak > 1.0e-12:
            source_mask = np.square(source_env / env_peak).astype(np.float32)[:, None]
        else:
            source_mask = np.zeros((end - start, 1), dtype=np.float32)
        projection_delta = (float(strength) * coeff * drum_segment).astype(np.float32)
        self_delta = (float(self_hf_attenuation) * source_mask * vocal_hf[start:end]).astype(np.float32)
        region_delta = (float(region_hf_attenuation) * vocal_hf[start:end]).astype(np.float32)
        window = _region_window(end - start)
        soft_delta = ((projection_delta + self_delta + region_delta) * window).astype(np.float32)
        if hard_region_hf_suppression:
            delta = (vocal_hf[start:end] + soft_delta).astype(np.float32)
        else:
            delta = soft_delta
        repaired[start:end] -= delta
        removed[start:end] += delta
        applied.append(
            {
                "start": round(float(start) / float(sample_rate), 6),
                "end": round(float(end) / float(sample_rate), 6),
                "source_region_score": float(region.get("score") or 0.0),
                "projection_coeff": round(float(coeff), 8),
                "strength": float(strength),
                "self_hf_attenuation": float(self_hf_attenuation),
                "region_hf_attenuation": float(region_hf_attenuation),
                "hard_region_hf_suppression": bool(hard_region_hf_suppression),
                "removed_rms_dbfs": round(_dbfs(_rms(delta)), 6),
            }
        )
    peak = float(np.max(np.abs(repaired))) if repaired.size else 0.0
    if peak > 0.999:
        repaired = (repaired / peak * 0.999).astype(np.float32)
    return repaired.astype(np.float32), removed.astype(np.float32), applied


def build_vocal_depercussive_teacher_target(
    *,
    vocals_path: Path,
    drums_path: Path,
    output_dir: Path,
    strength: float = 1.0,
    self_hf_attenuation: float = 0.35,
    region_hf_attenuation: float = 0.0,
    hard_region_hf_suppression: bool = False,
    max_projection_gain: float = 2.0,
    min_region_score: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    threshold = float(min_region_score if min_region_score is not None else cfg["vocal_drum_score_threshold"])
    vocals, vocal_rate = _read_audio(Path(vocals_path))
    drums, drum_rate = _read_audio(Path(drums_path))
    if vocal_rate != drum_rate:
        raise ValueError(f"sample rate mismatch: vocals={vocal_rate}, drums={drum_rate}")
    frames = min(len(vocals), len(drums))
    channels = max(vocals.shape[1], drums.shape[1])
    vocals = _match_audio(vocals, frames, channels)
    drums = _match_audio(drums, frames, channels)
    before = score_pair_forensic(
        tested_stem="vocals",
        source_stem="drums",
        tested_path=Path(vocals_path),
        source_path=Path(drums_path),
        output_dir=output_dir / "score" / "before",
        config=cfg,
    )
    candidate_regions = [
        region
        for region in before.get("regions", [])
        if float(region.get("score") or 0.0) >= threshold
    ]
    if not candidate_regions and before.get("regions"):
        candidate_regions = list(before.get("regions", [])[:3])
    high_band = tuple(float(value) for value in cfg["high_band_hz"])
    repaired, removed, applied_regions = _repair_regions(
        vocals=vocals,
        drums=drums,
        sample_rate=vocal_rate,
        regions=candidate_regions,
        strength=float(strength),
        self_hf_attenuation=float(self_hf_attenuation),
        region_hf_attenuation=float(region_hf_attenuation),
        hard_region_hf_suppression=bool(hard_region_hf_suppression),
        max_projection_gain=float(max_projection_gain),
        high_band_hz=(high_band[0], high_band[1]),
    )
    target_path = _write_audio(output_dir / "audio" / "vocals.flac", repaired, vocal_rate)
    removed_path = _write_audio(output_dir / "audio" / "removed.flac", removed, vocal_rate)
    after = score_pair_forensic(
        tested_stem="vocals",
        source_stem="drums",
        tested_path=target_path,
        source_path=Path(drums_path),
        output_dir=output_dir / "score" / "after",
        config=cfg,
    )
    manifest = {
        "schema": "vocal_depercussive_teacher_target.v1",
        "inputs": {
            "vocals_path": str(vocals_path),
            "vocals_sha256": _sha256_file(Path(vocals_path)),
            "drums_path": str(drums_path),
            "drums_sha256": _sha256_file(Path(drums_path)),
        },
        "outputs": {
            "target_vocals_path": str(target_path),
            "removed_proxy_path": str(removed_path),
        },
        "parameters": {
            "strength": float(strength),
            "self_hf_attenuation": float(self_hf_attenuation),
            "region_hf_attenuation": float(region_hf_attenuation),
            "hard_region_hf_suppression": bool(hard_region_hf_suppression),
            "max_projection_gain": float(max_projection_gain),
            "min_region_score": threshold,
            "high_band_hz": [float(high_band[0]), float(high_band[1])],
        },
        "forensic": {
            "before": before,
            "after": after,
            "applied_regions": applied_regions,
        },
        "contract": {
            "label_type": "weak_forensic_teacher",
            "is_ground_truth": False,
            "usable_as_ground_truth": False,
            "usable_for_training": True,
            "usable_for_promotion": False,
            "delivery_allowed": False,
            "presentation_status": "diagnostic_teacher_target_only",
            "target_reference_assisted": True,
            "per_track_hardcode_allowed": False,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocals", required=True, type=Path)
    parser.add_argument("--drums", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--self-hf-attenuation", type=float, default=0.35)
    parser.add_argument("--region-hf-attenuation", type=float, default=0.0)
    parser.add_argument("--hard-region-hf-suppression", action="store_true")
    parser.add_argument("--max-projection-gain", type=float, default=2.0)
    parser.add_argument("--min-region-score", type=float, default=None)
    args = parser.parse_args(argv)
    build_vocal_depercussive_teacher_target(
        vocals_path=args.vocals,
        drums_path=args.drums,
        output_dir=args.out,
        strength=float(args.strength),
        self_hf_attenuation=float(args.self_hf_attenuation),
        region_hf_attenuation=float(args.region_hf_attenuation),
        hard_region_hf_suppression=bool(args.hard_region_hf_suppression),
        max_projection_gain=float(args.max_projection_gain),
        min_region_score=args.min_region_score,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
