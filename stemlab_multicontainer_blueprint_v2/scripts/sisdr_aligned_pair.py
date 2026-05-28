"""Phase-recheck: properly aligned SI-SDR/SDR for the (full_mix vs reference_minus) pair.

Why this script exists
----------------------
``scripts/sisdr_compare.py`` aligns the two signals only by ``min(len)`` cropping
(see ``_align_lengths``). The system's output is offset from the user-provided
reference instrumental by ~0.85 s (the analyzer's ``alignment.lag_sec``). At
44.1 kHz that is ~37,485 samples. SI-SDR on a 37k-sample-shifted reference is
always catastrophically negative even for *identical* tracks, because every
sinusoid is rotated by tens of full cycles.

Likewise the analyzer's ``reference_match_metrics`` lag is found at 50 ms hop
resolution (``estimate_alignment_lag.hop_sec=0.05``) which is plenty for envelope
correlation but useless for raw-waveform cosine similarity.

This script does proper sample-level alignment via FFT cross-correlation between
mid-band-filtered signals, then computes SI-SDR and plain SDR on the aligned
overlap. Output is a small JSON next to the existing diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


def _parse_wav(path: Path) -> tuple[np.ndarray, int]:
    raw_format_code = channels = sr = sample_width = None
    data_bytes = b""
    with path.open("rb") as h:
        if h.read(4) != b"RIFF":
            raise ValueError(f"not RIFF: {path}")
        h.read(4)
        if h.read(4) != b"WAVE":
            raise ValueError(f"not WAVE: {path}")
        while True:
            header = h.read(8)
            if len(header) < 8:
                break
            chunk_id = header[0:4]
            chunk_size = struct.unpack("<I", header[4:8])[0]
            chunk_data = h.read(chunk_size)
            if chunk_size % 2 == 1:
                h.read(1)
            if chunk_id == b"fmt ":
                raw_format_code = struct.unpack("<H", chunk_data[0:2])[0]
                channels = struct.unpack("<H", chunk_data[2:4])[0]
                sr = struct.unpack("<I", chunk_data[4:8])[0]
                sample_width = struct.unpack("<H", chunk_data[14:16])[0] // 8
                if raw_format_code == 0xFFFE and len(chunk_data) >= 26:
                    raw_format_code = struct.unpack("<H", chunk_data[24:26])[0]
            elif chunk_id == b"data":
                data_bytes = chunk_data

    if raw_format_code == 3:
        if sample_width == 4:
            audio = np.frombuffer(data_bytes, dtype="<f4").astype(np.float64)
        else:
            audio = np.frombuffer(data_bytes, dtype="<f8").astype(np.float64)
    elif raw_format_code == 1:
        if sample_width == 2:
            audio = np.frombuffer(data_bytes, dtype="<i2").astype(np.float64) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(data_bytes, dtype="<i4").astype(np.float64) / float((1 << 31) - 1)
        else:
            raise ValueError(f"unsupported PCM width {sample_width} in {path}")
    else:
        raise ValueError(f"unsupported wFormatTag {raw_format_code} in {path}")

    if channels and channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float64, copy=False), int(sr)


def _bandlimit_for_alignment(x: np.ndarray, sr: int, low_hz: float = 80.0, high_hz: float = 4000.0) -> np.ndarray:
    n = x.shape[0]
    fft_n = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(x, fft_n)
    freqs = np.fft.rfftfreq(fft_n, 1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec_filtered = np.zeros_like(spec)
    spec_filtered[mask] = spec[mask]
    out = np.fft.irfft(spec_filtered, fft_n)[:n]
    return out


def _find_integer_lag(reference: np.ndarray, target: np.ndarray, sr: int, max_lag_sec: float = 5.0) -> int:
    """Sample-accurate integer lag via FFT cross-correlation on a mid-band signal.

    Positive returned lag means: ``reference`` leads ``target`` by ``lag`` samples,
    i.e. align by ``reference[lag:]`` <-> ``target[:len-lag]`` (or vice versa).
    """
    ref_b = _bandlimit_for_alignment(reference, sr)
    tgt_b = _bandlimit_for_alignment(target, sr)

    n = min(ref_b.shape[0], tgt_b.shape[0])
    ref_b = ref_b[:n]
    tgt_b = tgt_b[:n]

    fft_n = 1 << (2 * n - 1).bit_length()
    R = np.fft.rfft(ref_b, fft_n)
    T = np.fft.rfft(tgt_b, fft_n)
    xcorr = np.fft.irfft(R * np.conj(T), fft_n)

    max_lag = int(round(max_lag_sec * sr))
    head = xcorr[: max_lag + 1]
    tail = xcorr[-max_lag:]
    candidates = np.concatenate([tail, head])
    lags = np.concatenate([np.arange(-max_lag, 0), np.arange(0, max_lag + 1)])
    best_idx = int(np.argmax(candidates))
    return int(lags[best_idx])


def _apply_integer_lag(reference: np.ndarray, target: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        ref_aligned = reference[lag:]
        tgt_aligned = target[: ref_aligned.shape[0]]
    else:
        tgt_aligned = target[-lag:]
        ref_aligned = reference[: tgt_aligned.shape[0]]
    n = min(ref_aligned.shape[0], tgt_aligned.shape[0])
    return ref_aligned[:n], tgt_aligned[:n]


def _si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    ref_energy = float(np.dot(reference, reference))
    if ref_energy <= eps:
        return float("nan")
    alpha = float(np.dot(estimate, reference)) / max(ref_energy, eps)
    s_target = alpha * reference
    e_noise = estimate - s_target
    target_e = float(np.dot(s_target, s_target))
    noise_e = float(np.dot(e_noise, e_noise))
    if noise_e <= eps:
        return float("inf")
    if target_e <= eps:
        return float("-inf")
    return 10.0 * math.log10(target_e / noise_e)


def _sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    ref_energy = float(np.dot(reference, reference))
    if ref_energy <= eps:
        return float("nan")
    diff = estimate - reference
    diff_e = float(np.dot(diff, diff))
    if diff_e <= eps:
        return float("inf")
    return 10.0 * math.log10(ref_energy / diff_e)


def _gain_match(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    ref_energy = float(np.dot(reference, reference))
    if ref_energy <= eps:
        return 1.0
    return float(np.dot(estimate, reference)) / max(ref_energy, eps)


def _pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)) + eps)
    return float(np.dot(a, b) / denom)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="ground-truth instrumental WAV")
    p.add_argument("--estimate", required=True, help="system-produced instrumental WAV")
    p.add_argument("--out", default=None, help="optional JSON report path")
    p.add_argument("--max-lag-sec", type=float, default=5.0)
    args = p.parse_args()

    ref_path = Path(args.reference)
    est_path = Path(args.estimate)
    ref, ref_sr = _parse_wav(ref_path)
    est, est_sr = _parse_wav(est_path)
    if ref_sr != est_sr:
        print(f"[ERROR] sample-rate mismatch ref={ref_sr} est={est_sr}", file=sys.stderr)
        return 2
    sr = ref_sr

    lag = _find_integer_lag(ref, est, sr, max_lag_sec=args.max_lag_sec)
    ref_aligned, est_aligned = _apply_integer_lag(ref, est, lag)

    naive_ref, naive_est = ref[: min(ref.shape[0], est.shape[0])], est[: min(ref.shape[0], est.shape[0])]
    naive = {
        "si_sdr_db": _si_sdr(naive_ref, naive_est),
        "sdr_db": _sdr(naive_ref, naive_est),
        "pearson": _pearson(naive_ref, naive_est),
    }

    gain = _gain_match(ref_aligned, est_aligned)
    matched_ref = gain * ref_aligned
    aligned = {
        "lag_samples": int(lag),
        "lag_sec": float(lag) / float(sr),
        "overlap_sec": float(ref_aligned.shape[0]) / float(sr),
        "gain_match_db": 20.0 * math.log10(abs(gain) + 1e-12),
        "pearson": _pearson(ref_aligned, est_aligned),
        "si_sdr_db": _si_sdr(ref_aligned, est_aligned),
        "sdr_db_gain_matched": _sdr(matched_ref, est_aligned),
    }

    report = {
        "reference": str(ref_path),
        "estimate": str(est_path),
        "sample_rate": sr,
        "naive_min_crop": naive,
        "aligned_integer_lag": aligned,
        "interpretation": {
            "naive_si_sdr_meaningful": False,
            "naive_si_sdr_reason": (
                "_align_lengths only crops to min(len); a 0.85 s lag is "
                "37 485 samples at 44.1 kHz which fully decorrelates phase. "
                "Use aligned_integer_lag instead."
            ),
            "good_si_sdr_for_instrumental_separation_db": 8.0,
            "marginal_si_sdr_db": 3.0,
        },
    }

    out_text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"[OK] report written to {args.out}")
    print(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
