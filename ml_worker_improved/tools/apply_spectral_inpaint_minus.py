#!/usr/bin/env python3
"""Spectrally inpaint vocal leak regions in instrumental using frequency-axis interpolation.

Instead of subtracting or attenuating vocal-dominated STFT bins, this replaces
their magnitude with interpolated values from neighboring clean bins, preserving
original phase. This avoids creating spectral holes.

Usage:
  python apply_spectral_inpaint_minus.py --track-dir <track_dir> --output-dir <out_dir> [--threshold-low 0.3] [--threshold-high 0.6]
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
from scipy import signal


EPS = 1e-12
N_FFT = 4096
HOP = 1024


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_24")


def _fit_audio(instr: np.ndarray, vocals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(instr.shape[0], vocals.shape[0])
    return instr[:n].copy(), vocals[:n].copy()


def _interp_freq(mag: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate magnitude along frequency axis for masked bins per time frame."""
    result = mag.copy().astype(np.float64)
    n_freq = mag.shape[0]
    freq_idx = np.arange(n_freq, dtype=np.float64)

    for t in range(mag.shape[1]):
        col = result[:, t]
        m = mask[:, t]
        if not m.any():
            continue
        if m.all():
            col[:] = np.mean(col)
            continue
        clean = ~m
        clean_idx = freq_idx[clean]
        clean_vals = col[clean]
        masked_idx = freq_idx[m]
        # linear interpolation
        col_interp = np.interp(masked_idx, clean_idx, clean_vals)
        col[m] = col_interp
        result[:, t] = col
    return result.astype(np.float32)


def spectral_inpaint_minus(
    instrumental: np.ndarray,
    vocals: np.ndarray,
    sample_rate: int,
    threshold_low: float = 0.3,
    threshold_high: float = 0.6,
) -> np.ndarray:
    """
    Apply spectral inpainting to remove vocal leak without creating holes.
    
    For each STFT bin, compute vocal dominance ratio. Bins below threshold_low
    are kept as-is. Bins above threshold_high are fully inpainted (interpolated
    from clean neighbors). Bins in between are smoothly blended.
    
    Returns inpainted instrumental (same shape as input).
    """
    instrumental, vocals = _fit_audio(instrumental, vocals)
    n = instrumental.shape[0]
    
    refined_channels = []
    for ch in range(instrumental.shape[1]):
        im = instrumental[:, ch]
        vm = vocals[:, min(ch, vocals.shape[1] - 1)]
        
        # STFT
        freqs, _, i_stft = signal.stft(im, fs=sample_rate, window="hann",
                                        nperseg=N_FFT, noverlap=N_FFT - HOP,
                                        boundary="zeros")
        _, _, v_stft = signal.stft(vm, fs=sample_rate, window="hann",
                                    nperseg=N_FFT, noverlap=N_FFT - HOP,
                                    boundary="zeros")
        
        i_mag = np.abs(i_stft).astype(np.float64)
        v_mag = np.abs(v_stft).astype(np.float64)
        i_phase = np.angle(i_stft)
        
        # Vocal dominance ratio
        ratio = v_mag / (i_mag + v_mag + EPS)
        
        # Smooth mask: 0=keep original, 1=full inpaint
        alpha = np.clip((ratio - threshold_low) / (threshold_high - threshold_low + EPS), 0.0, 1.0).astype(np.float32)
        
        # Mask for interpolation: bins with alpha > 0.5
        interp_mask = alpha > 0.5
        
        # Interpolate magnitude along frequency axis for masked bins
        interp_mag = _interp_freq(i_mag, interp_mask)
        
        # Blend: (1-alpha) * original + alpha * interpolated
        alpha_64 = alpha.astype(np.float64)
        blended_mag = (1.0 - alpha_64) * i_mag + alpha_64 * interp_mag
        
        # ISTFT with original phase
        _, refined = signal.istft(blended_mag * np.exp(1j * i_phase),
                                  fs=sample_rate, window="hann",
                                  nperseg=N_FFT, noverlap=N_FFT - HOP,
                                  input_onesided=True)
        refined_channels.append(refined[:n].astype(np.float32))
    
    return np.column_stack(refined_channels).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral inpainting for vocal leak removal")
    parser.add_argument("--track-dir", required=True, type=Path, help="Track directory with instrumental.flac and vocals.flac")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--threshold-low", type=float, default=0.3, help="Vocal ratio threshold for blending start (default: 0.3)")
    parser.add_argument("--threshold-high", type=float, default=0.6, help="Vocal ratio threshold for full inpaint (default: 0.6)")
    parser.add_argument("--alias", default=None, help="Track alias (optional, for naming)")
    args = parser.parse_args()

    track_dir = Path(args.track_dir)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    instr_path = track_dir / "instrumental.flac"
    vocals_path = track_dir / "vocals.flac"
    if not instr_path.exists() or not vocals_path.exists():
        print(f"Missing instrumental.flac or vocals.flac in {track_dir}")
        sys.exit(1)

    t0 = time.time()
    instr, sr = _read_audio(instr_path)
    vocals, sr2 = _read_audio(vocals_path)
    assert sr == sr2, f"Sample rate mismatch: {sr} vs {sr2}"
    print(f"Loaded: {instr.shape[0]} frames, {sr}Hz")

    refined = spectral_inpaint_minus(instr, vocals, sr,
                                     threshold_low=args.threshold_low,
                                     threshold_high=args.threshold_high)
    _write_audio(out_dir / "m001.flac", refined, sr)

    elapsed = time.time() - t0
    delta_rms = float(np.sqrt(np.mean((refined - instr[:refined.shape[0]]) ** 2)))
    orig_rms = float(np.sqrt(np.mean(instr ** 2)))
    delta_db = 20.0 * math.log10(delta_rms / (orig_rms + EPS)) if orig_rms > EPS else -240.0

    result = {
        "threshold_low": args.threshold_low,
        "threshold_high": args.threshold_high,
        "delta_db": float(f"{delta_db:.2f}"),
        "elapsed_sec": float(f"{elapsed:.1f}"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Done: delta_db={delta_db:.1f}dB, elapsed={elapsed:.0f}s")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
