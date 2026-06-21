"""Build procedural clean stem pairs for full-matrix contamination training."""

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

from tools.stem_contamination_common import FORBIDDEN_SOURCES, STEMS, validate_forbidden_source_pair


STEM_BASE_FREQUENCIES = {
    "vocals": (900.0, 1080.0),
    "drums": (2200.0, 3300.0),
    "bass": (90.0, 140.0),
    "guitar": (430.0, 560.0),
    "piano": (1450.0, 1900.0),
    "other": (650.0, 2800.0),
}
VARIANT_PROFILES = (
    "steady_tone",
    "offset_tone",
    "speech_like_harmonic",
    "plucked_decay",
    "transient_dense",
    "wide_band_modulated",
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return path


def _fade(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_len = min(len(audio) // 4, max(1, int(sample_rate * 0.01)))
    if fade_len <= 1:
        return audio
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    output = audio.copy()
    output[:fade_len] *= ramp
    output[-fade_len:] *= ramp[::-1]
    return output


def _adsr_envelope(frames: int, sample_rate: int, *, attack_ms: float, decay_ms: float, sustain: float) -> np.ndarray:
    attack = min(frames, max(1, int(sample_rate * attack_ms / 1000.0)))
    decay = min(max(1, frames - attack), max(1, int(sample_rate * decay_ms / 1000.0)))
    envelope = np.full(frames, float(sustain), dtype=np.float32)
    envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    end_decay = min(frames, attack + decay)
    envelope[attack:end_decay] = np.linspace(1.0, float(sustain), max(1, end_decay - attack), dtype=np.float32)
    return envelope


def _speech_like_vocal(t: np.ndarray, sample_rate: int, variant: int) -> np.ndarray:
    f0 = 165.0 + 18.0 * float(variant % 5)
    vibrato = 1.0 + 0.035 * np.sin(2.0 * math.pi * (4.2 + 0.15 * variant) * t)
    phase = np.cumsum(2.0 * math.pi * f0 * vibrato / float(sample_rate)).astype(np.float32)
    audio = np.zeros_like(t, dtype=np.float32)
    for harmonic, gain in ((1, 0.13), (2, 0.09), (3, 0.07), (4, 0.052), (5, 0.038), (7, 0.026), (9, 0.018)):
        audio += gain * np.sin(float(harmonic) * phase + 0.19 * float(variant + harmonic)).astype(np.float32)
    syllables = 0.48 + 0.52 * np.sin(2.0 * math.pi * (1.9 + 0.13 * variant) * t) ** 2
    return (audio * syllables).astype(np.float32)


def _plucked_tone(t: np.ndarray, frequencies: tuple[float, ...], variant: int) -> np.ndarray:
    audio = np.zeros_like(t, dtype=np.float32)
    decay = np.exp(-t * (3.0 + 0.35 * variant)).astype(np.float32)
    for idx, frequency in enumerate(frequencies, start=1):
        audio += (0.11 / float(idx)) * np.sin(2.0 * math.pi * frequency * t + 0.23 * variant * idx).astype(np.float32)
    return (audio * decay).astype(np.float32)


def _drum_like_audio(frames: int, sample_rate: int, variant: int) -> np.ndarray:
    audio = np.zeros(frames, dtype=np.float32)
    rng = np.random.default_rng(1000 + int(variant))
    pulse_period = max(1, int(sample_rate * (0.075 + 0.004 * (variant % 3))))
    pulse = np.zeros(frames, dtype=np.float32)
    pulse[::pulse_period] = 0.45
    kernel = np.exp(-np.arange(max(8, int(sample_rate * 0.018)), dtype=np.float32) / float(sample_rate * 0.004))
    audio += np.convolve(pulse, kernel, mode="same").astype(np.float32)
    noise = rng.normal(0.0, 0.045, frames).astype(np.float32)
    burst = np.zeros(frames, dtype=np.float32)
    burst[:: max(1, pulse_period // 2)] = 1.0
    burst_kernel = np.exp(-np.arange(max(8, int(sample_rate * 0.01)), dtype=np.float32) / float(sample_rate * 0.0025))
    audio += np.convolve(burst, burst_kernel, mode="same").astype(np.float32) * noise
    return audio.astype(np.float32)


def _procedural_stem_audio(stem: str, *, sample_rate: int, duration_sec: float, variant: int) -> np.ndarray:
    if stem not in STEMS:
        raise ValueError(f"unsupported stem: {stem}")
    frames = max(1, int(round(float(sample_rate) * float(duration_sec))))
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    frequencies = STEM_BASE_FREQUENCIES[stem]
    profile = VARIANT_PROFILES[int(variant) % len(VARIANT_PROFILES)]
    variant_shift = 1.0 + 0.015 * float(variant)
    if stem == "vocals" and profile in {"speech_like_harmonic", "wide_band_modulated"}:
        audio = _speech_like_vocal(t, int(sample_rate), int(variant))
    elif stem in {"guitar", "piano", "bass"} and profile in {"plucked_decay", "wide_band_modulated"}:
        audio = _plucked_tone(t, tuple(frequency * variant_shift for frequency in frequencies), int(variant))
    elif stem == "drums":
        audio = _drum_like_audio(frames, int(sample_rate), int(variant))
    else:
        audio = np.zeros(frames, dtype=np.float32)
        for idx, frequency in enumerate(frequencies, start=1):
            phase = 0.37 * float(variant + idx)
            audio += (0.07 / float(idx)) * np.sin(2.0 * math.pi * frequency * variant_shift * t + phase).astype(np.float32)
    modulation = 0.75 + 0.25 * np.sin(2.0 * math.pi * (2.0 + 0.1 * variant) * t).astype(np.float32)
    if profile == "transient_dense" and stem != "drums":
        modulation *= _adsr_envelope(frames, int(sample_rate), attack_ms=8.0, decay_ms=120.0, sustain=0.45)
    audio = _fade(audio * modulation, sample_rate)
    stereo = np.stack([audio, audio * (0.96 + 0.01 * (variant % 3))], axis=1)
    return stereo.astype(np.float32)


def build_procedural_stem_clean_pairs(
    *,
    output_dir: Path,
    sample_rate: int = 16_000,
    duration_sec: float = 1.0,
    examples_per_pair: int = 2,
) -> Path:
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    if float(duration_sec) <= 0:
        raise ValueError("duration_sec must be positive")
    if int(examples_per_pair) <= 0:
        raise ValueError("examples_per_pair must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"

    sources: dict[str, dict[str, Any]] = {}
    variant_sources: dict[int, dict[str, Path]] = {}
    for variant in range(int(examples_per_pair)):
        variant_sources[variant] = {}
        source_audio: dict[str, np.ndarray] = {}
        for stem in STEMS:
            audio = _procedural_stem_audio(stem, sample_rate=int(sample_rate), duration_sec=float(duration_sec), variant=variant)
            source_audio[stem] = audio
            source_path = _write_audio(audio_dir / f"s{variant:02d}_{stem}.flac", audio, int(sample_rate))
            variant_sources[variant][stem] = source_path
            if variant == 0:
                sources[stem] = {
                    "path": str(source_path),
                    "stem": stem,
                    "source_kind": "procedural_synthetic_clean",
                    "sample_rate": int(sample_rate),
                    "duration_sec": float(duration_sec),
                    "variant_profile": VARIANT_PROFILES[variant % len(VARIANT_PROFILES)],
                }
        minus_audio = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
        variant_sources[variant]["minus"] = _write_audio(audio_dir / f"s{variant:02d}_minus.flac", minus_audio, int(sample_rate))

    pairs: list[dict[str, Any]] = []
    for variant in range(int(examples_per_pair)):
        for tested_stem, foreign_sources in FORBIDDEN_SOURCES.items():
            for foreign_source in foreign_sources:
                validate_forbidden_source_pair(tested_stem, foreign_source)
                pairs.append(
                    {
                        "pair_id": f"p{len(pairs) + 1:05d}",
                        "variant_id": f"s{variant:02d}",
                        "base_stem": tested_stem,
                        "base_path": str(variant_sources[variant][tested_stem]),
                        "foreign_source": foreign_source,
                        "foreign_path": str(variant_sources[variant][foreign_source]),
                        "source_kind": "procedural_synthetic_clean",
                        "variant_profile": VARIANT_PROFILES[variant % len(VARIANT_PROFILES)],
                        "is_ground_truth": True,
                    }
                )

    manifest = {
        "schema": "stem_contamination_procedural_clean_pairs.v1",
        "source_kind": "procedural_synthetic_clean",
        "generated_separator_output_used": False,
        "dirty_reference_used": False,
        "sample_rate": int(sample_rate),
        "duration_sec": float(duration_sec),
        "examples_per_pair": int(examples_per_pair),
        "variant_profiles": list(VARIANT_PROFILES),
        "pair_count": len(pairs),
        "sources": sources,
        "pairs": pairs,
    }
    return _write_json(output_dir / "manifest.json", manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration-sec", type=float, default=1.0)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = build_procedural_stem_clean_pairs(
        output_dir=Path(args.out),
        sample_rate=int(args.sample_rate),
        duration_sec=float(args.duration_sec),
        examples_per_pair=int(args.examples_per_pair),
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
