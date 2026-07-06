#!/usr/bin/env python3
"""Coherence-based leak mask: detect correlated components between vocal and 
instrumental STFTs. Bins with high coherence are likely vocal leak. Apply 
gain reduction only to high-coherence bins.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal


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


def coherence_mask(
    i_stft: np.ndarray,
    v_stft: np.ndarray,
    smooth_frames: int = 7,
) -> np.ndarray:
    """Compute magnitude-squared coherence between instrumental and vocal STFTs.
    
    Uses temporal smoothing (moving average) to estimate expectations.
    
    Returns: coherence [0, 1] per TF bin.
    """
    i_c = np.asarray(i_stft, dtype=np.complex128)
    v_c = np.asarray(v_stft, dtype=np.complex128)
    
    # Cross-spectral density: E[I * conj(V)]
    cross = i_c * np.conj(v_c)
    
    # Auto-spectral densities
    i_psd = np.abs(i_c) ** 2
    v_psd = np.abs(v_c) ** 2
    
    # Smooth along time axis (moving average)
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=np.float64) / smooth_frames
        cross = scipy_signal.convolve(cross, kernel[None, :], mode="same", method="auto")
        i_psd = scipy_signal.convolve(i_psd, kernel[None, :], mode="same", method="auto")
        v_psd = scipy_signal.convolve(v_psd, kernel[None, :], mode="same", method="auto")
    
    # Coherence = |E[I*V*]|^2 / (E[|I|^2] * E[|V|^2])
    coh = np.abs(cross) ** 2 / (i_psd * v_psd + EPS)
    return np.clip(np.asarray(coh, dtype=np.float32), 0.0, 1.0)


def apply_coherence_mask(
    instrumental: np.ndarray,
    vocals: np.ndarray,
    sample_rate: int,
    coherence_threshold: float = 0.3,
    max_attenuation: float = 0.5,
    smooth_frames: int = 7,
    min_gain_voice: float = 0.78,
) -> np.ndarray:
    """Apply gain reduction based on coherence between instrumental and vocals."""
    instrumental, vocals = instrumental.copy(), vocals.copy()
    n = min(instrumental.shape[0], vocals.shape[0])
    instr = instrumental[:n]
    voc = vocals[:n]
    
    refined_channels = []
    for ch in range(instr.shape[1]):
        im = instr[:, ch]
        vm = voc[:, min(ch, voc.shape[1] - 1)]
        
        freqs, t, i_stft = scipy_signal.stft(im, fs=sample_rate, window="hann",
                                              nperseg=N_FFT, noverlap=N_FFT - HOP,
                                              boundary="zeros")
        _, _, v_stft = scipy_signal.stft(vm, fs=sample_rate, window="hann",
                                          nperseg=N_FFT, noverlap=N_FFT - HOP,
                                          boundary="zeros")
        
        # Compute coherence
        coh = coherence_mask(i_stft, v_stft, smooth_frames=smooth_frames)
        
        # Hard gate: bins below threshold are not modified
        hard_mask = coh < coherence_threshold
        # Gain: 1 - alpha * coherence^power, clamped to min_gain  
        gain = np.clip(1.0 - (1.0 - max_attenuation) * coh, min_gain_voice, 1.0)
        gain[hard_mask] = 1.0  # no change for low-coherence bins
        
        # Frequency-dependent min gain
        freqs_hz = freqs * sample_rate / N_FFT
        voice_mask = (freqs_hz >= 160) & (freqs_hz <= 5200)
        gain[~voice_mask] = np.maximum(gain[~voice_mask], 0.95)
        gain[voice_mask] = np.maximum(gain[voice_mask], min_gain_voice)
        
        _, refined = scipy_signal.istft(
            np.asarray(i_stft, dtype=np.complex128) * gain.astype(np.float64),
            fs=sample_rate, window="hann",
            nperseg=N_FFT, noverlap=N_FFT - HOP,
            input_onesided=True
        )
        refined_channels.append(refined[:n].astype(np.float32))
    
    return np.column_stack(refined_channels).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coherence-based vocal leak mask")
    parser.add_argument("--track-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coherence-threshold", type=float, default=0.3)
    parser.add_argument("--max-attenuation", type=float, default=0.5,
                       help="Max gain reduction linear (0.5 = -6dB)")
    parser.add_argument("--smooth-frames", type=int, default=7, help="Temporal smoothing frames")
    parser.add_argument("--min-gain-voice", type=float, default=0.78)
    args = parser.parse_args()

    track_dir = Path(args.track_dir)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    t0 = time.time()
    instr, sr = _read_audio(track_dir / "instrumental.flac")
    vocals, sr2 = _read_audio(track_dir / "vocals.flac")
    assert sr == sr2
    print(f"Loaded: {instr.shape[0]} frames, {sr}Hz")

    refined = apply_coherence_mask(instr, vocals, sr,
                                   coherence_threshold=args.coherence_threshold,
                                   max_attenuation=args.max_attenuation,
                                   smooth_frames=args.smooth_frames,
                                   min_gain_voice=args.min_gain_voice)
    _write_audio(out_dir / "m001.flac", refined, sr)

    dr = 20 * math.log10(
        np.sqrt(np.mean((refined - instr[:refined.shape[0]]) ** 2)) /
        (np.sqrt(np.mean(instr ** 2)) + EPS)
    )
    result = {
        "coherence_threshold": args.coherence_threshold,
        "max_attenuation": args.max_attenuation,
        "smooth_frames": args.smooth_frames,
        "min_gain_voice": args.min_gain_voice,
        "delta_db": float(f"{dr:.2f}"),
        "elapsed_sec": float(f"{time.time()-t0:.1f}"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Done: delta_db={dr:.1f}dB")


if __name__ == "__main__":
    main()
