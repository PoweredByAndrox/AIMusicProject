#!/usr/bin/env python3
"""Time-domain envelope-following gain reduction for vocal leak removal.

Instead of per-bin STFT masking (which creates phase inconsistency artifacts),
apply a slow gain envelope derived from the vocal RMS, heavily smoothed to
avoid audible holes/pumping. The gain follows the vocal energy but with
200ms+ time constant, so it sounds natural.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal


EPS = 1e-12


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_24")


def _rms_envelope(audio: np.ndarray, window: int) -> np.ndarray:
    """Compute RMS envelope with sliding window."""
    squared = np.asarray(audio, dtype=np.float64) ** 2
    env = np.zeros_like(squared)
    cumsum = np.cumsum(np.pad(squared, (window // 2, window // 2), mode="edge"))
    n = len(squared)
    for i in range(n):
        start = i
        end = i + window
        env[i] = np.sqrt(cumsum[end] - cumsum[start]) / math.sqrt(window)
    return env.astype(np.float32)


def apply_envelope_gain(
    instrumental: np.ndarray,
    vocals: np.ndarray,
    sample_rate: int,
    strength: float = 0.5,
    envelope_ms: float = 5.0,
    smooth_ms: float = 200.0,
    max_attenuation_db: float = -6.0,
    min_gain: float = 0.7,
) -> np.ndarray:
    """Apply time-domain envelope gain to remove vocal leak.
    
    Args:
        instrumental: stereo instrumental array (N, 2)
        vocals: stereo vocals array (N, 2)
        sample_rate: sample rate in Hz
        strength: how aggressively to follow vocal envelope (0-1)
        envelope_ms: RMS window in ms for envelope extraction
        smooth_ms: smoothing time constant in ms
        max_attenuation_db: maximum gain reduction in dB
        min_gain: absolute minimum gain (clamp)
    
    Returns:
        Processed instrumental (N, 2)
    """
    n = min(instrumental.shape[0], vocals.shape[0])
    instr = instrumental[:n].copy()
    voc = vocals[:n].copy()
    
    window = max(1, int(round(envelope_ms * sample_rate / 1000.0)))
    smooth_samples = int(round(smooth_ms * sample_rate / 1000.0))
    
    # Mono mix for envelope (use vocals + instrumental)
    voc_mono = np.mean(voc, axis=1).astype(np.float64)
    instr_mono = np.mean(instr, axis=1).astype(np.float64)
    
    # Vocal RMS envelope
    voc_env = _rms_envelope(voc_mono, window)
    
    # Normalize envelope to [0, 1] using max vocal RMS
    voc_max = float(np.percentile(voc_env, 99.9))
    if voc_max > EPS:
        voc_env = np.clip(voc_env / voc_max, 0.0, 1.0)
    else:
        return instr  # no vocal energy, return as-is
    
    # Compute target gain: 1 - strength * envelope, then clamp
    max_atten_linear = 10.0 ** (max_attenuation_db / 20.0)
    target_gain = np.clip(1.0 - strength * voc_env, max_atten_linear, 1.0)
    
    # Heavy smoothing
    # Use multiple passes of simple IIR for computational efficiency
    gain = target_gain.astype(np.float64)
    alpha = 1.0 / (smooth_samples / 2.0 + 1.0)  # approximate time constant
    for _ in range(2):
        # Forward
        for i in range(1, len(gain)):
            gain[i] = gain[i-1] + alpha * (gain[i] - gain[i-1])
        # Backward (zero-phase)
        for i in range(len(gain) - 2, -1, -1):
            gain[i] = gain[i+1] + alpha * (gain[i] - gain[i+1])
    
    # Clamp to min_gain
    gain = np.clip(gain, min_gain if min_gain < 1.0 else max_atten_linear, 1.0).astype(np.float32)
    
    # Apply to each channel
    result = instr.copy()
    for ch in range(result.shape[1]):
        result[:, ch] = (result[:, ch].astype(np.float64) * gain).astype(np.float32)
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-domain envelope following gain for vocal leak")
    parser.add_argument("--track-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=0.5, help="Envelope following strength (0-1)")
    parser.add_argument("--smooth-ms", type=float, default=200.0, help="Smoothing time constant in ms")
    parser.add_argument("--max-atten-db", type=float, default=-8.0, help="Max attenuation in dB")
    parser.add_argument("--min-gain", type=float, default=0.7, help="Absolute minimum gain")
    args = parser.parse_args()

    track_dir = Path(args.track_dir)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    instr_path = track_dir / "instrumental.flac"
    vocals_path = track_dir / "vocals.flac"
    if not instr_path.exists() or not vocals_path.exists():
        print(f"Missing files in {track_dir}")
        sys.exit(1)

    t0 = time.time()
    instr, sr = _read_audio(instr_path)
    vocals, sr2 = _read_audio(vocals_path)
    assert sr == sr2, f"Sample rate mismatch: {sr} vs {sr2}"
    print(f"Loaded: {instr.shape[0]} frames, {sr}Hz")

    refined = apply_envelope_gain(
        instr, vocals, sr,
        strength=args.strength,
        smooth_ms=args.smooth_ms,
        max_attenuation_db=args.max_atten_db,
        min_gain=args.min_gain,
    )
    _write_audio(out_dir / "m001.flac", refined, sr)

    elapsed = time.time() - t0
    delta_rms = float(np.sqrt(np.mean((refined - instr[:refined.shape[0]]) ** 2)))
    orig_rms = float(np.sqrt(np.mean(instr ** 2)))
    delta_db = 20.0 * math.log10(delta_rms / (orig_rms + EPS)) if orig_rms > EPS else -240.0

    result = {
        "strength": args.strength,
        "smooth_ms": args.smooth_ms,
        "max_atten_db": args.max_atten_db,
        "min_gain": args.min_gain,
        "delta_db": float(f"{delta_db:.2f}"),
        "elapsed_sec": float(f"{elapsed:.1f}"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Done: delta_db={delta_db:.1f}dB, elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
