"""Phase-1 probe: decode the two MP3s and check temporal alignment.

Goal: confirm or reject the user's claim "это 2 одинаковых трека (один полноценный
другой минус но именно что оригинальный)" before running an expensive separation
job. Computes:

- duration / sample-rate / channels
- mono RMS for both
- cross-correlation of the first 60 s of each (downmixed mono, downsampled to
  8 kHz for speed) to estimate global time lag and peak correlation
- Pearson correlation after lag-alignment

If peak xcorr > ~0.5 -> same master (vocal+instrumental pair).
If peak xcorr < ~0.2 -> different master / different remix arrangement.
The orchestrator's analyzer uses corr >= 0.62 as the "reference is usable" gate
(see analyze_tandem_job.py:1078), so this probe uses the same scale.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


WORK_ROOT = Path(__file__).resolve().parents[2]


def docker_decode(path_in_workspace: str, target_sr: int, mono: bool) -> tuple[np.ndarray, int]:
    """Decode any audio file via dockerised ffmpeg into a temp wav, then read it."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.wav"
        host_in = (WORK_ROOT / path_in_workspace).resolve()
        if not host_in.exists():
            raise FileNotFoundError(host_in)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{WORK_ROOT.as_posix()}:/work:ro",
            "-v", f"{Path(tmp).resolve().as_posix()}:/out",
            "--entrypoint", "ffmpeg",
            "linuxserver/ffmpeg",
            "-y", "-hide_banner", "-loglevel", "error",
            "-i", f"/work/{path_in_workspace}",
            "-vn",
            "-ac", "1" if mono else "2",
            "-ar", str(target_sr),
            "-acodec", "pcm_s16le",
            "/out/out.wav",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with wave.open(str(out_path), "rb") as handle:
            sr = handle.getframerate()
            channels = handle.getnchannels()
            n_frames = handle.getnframes()
            raw = handle.readframes(n_frames)
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).T
        return audio, sr


def _to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=0)


def main() -> int:
    full_track = "ytmp3free.cc_styles-of-beyond-nine-thou-need-for-speed-most-wanted-2005-youtubemp3free.org.mp3"
    reference = "Nine Thou (Grant Mohrman Superstars Remix) (320).mp3"

    target_sr = 8000
    print(f"[probe] decoding both tracks to mono / {target_sr} Hz ...")
    a_audio, a_sr = docker_decode(full_track, target_sr=target_sr, mono=True)
    b_audio, b_sr = docker_decode(reference, target_sr=target_sr, mono=True)

    a_mono = _to_mono(a_audio)
    b_mono = _to_mono(b_audio)
    a_dur = a_mono.shape[0] / a_sr
    b_dur = b_mono.shape[0] / b_sr
    print(f"  full track   : {a_dur:7.2f} s, sr={a_sr}, rms={np.sqrt(np.mean(a_mono**2)):.4f}")
    print(f"  reference    : {b_dur:7.2f} s, sr={b_sr}, rms={np.sqrt(np.mean(b_mono**2)):.4f}")

    window_sec = 60.0
    n_window = int(window_sec * target_sr)
    a_seg = a_mono[:n_window].astype(np.float64)
    b_seg = b_mono[:n_window].astype(np.float64)
    a_seg -= a_seg.mean(); b_seg -= b_seg.mean()
    a_norm = a_seg / (np.sqrt(np.dot(a_seg, a_seg)) + 1e-12)
    b_norm = b_seg / (np.sqrt(np.dot(b_seg, b_seg)) + 1e-12)

    print(f"[probe] cross-correlating first {window_sec:.0f} s ...")
    n = len(a_norm) + len(b_norm) - 1
    fft_size = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a_norm, fft_size)
    B = np.fft.rfft(b_norm, fft_size)
    xcorr = np.fft.irfft(A * np.conj(B), fft_size)
    max_lag_s = 8.0
    max_lag_samples = int(max_lag_s * target_sr)
    head = xcorr[:max_lag_samples]
    tail = xcorr[-max_lag_samples:]
    candidates = np.concatenate([tail, head])
    lags = np.concatenate([np.arange(-max_lag_samples, 0), np.arange(0, max_lag_samples)])
    best = int(np.argmax(candidates))
    best_corr = float(candidates[best])
    best_lag_samples = int(lags[best])
    best_lag_sec = best_lag_samples / target_sr
    print(f"  best lag    : {best_lag_sec:+.3f} s  (b leads a if positive)")
    print(f"  peak xcorr  : {best_corr:+.4f}    (1.0 = identical)")

    if best_lag_samples >= 0:
        a_aligned = a_seg[best_lag_samples:]
        b_aligned = b_seg[: len(a_aligned)]
    else:
        b_aligned = b_seg[-best_lag_samples:]
        a_aligned = a_seg[: len(b_aligned)]
    common = min(len(a_aligned), len(b_aligned))
    if common > 100:
        a_aligned = a_aligned[:common]
        b_aligned = b_aligned[:common]
        a_aligned = a_aligned - a_aligned.mean()
        b_aligned = b_aligned - b_aligned.mean()
        denom = (np.sqrt(np.dot(a_aligned, a_aligned)) * np.sqrt(np.dot(b_aligned, b_aligned)) + 1e-12)
        pearson = float(np.dot(a_aligned, b_aligned) / denom)
        print(f"  pearson r   : {pearson:+.4f}    (after lag alignment)")

    threshold_usable = 0.62
    threshold_remix = 0.20
    verdict_corr = abs(best_corr)
    if verdict_corr >= threshold_usable:
        verdict = "SAME MASTER (system will accept reference as usable)"
    elif verdict_corr >= threshold_remix:
        verdict = "WEAK MATCH (likely same song but different mix/master/remix)"
    else:
        verdict = "DIFFERENT MASTERS (almost certainly a remix or unrelated take)"
    print(f"[probe] verdict: {verdict}")
    print(f"  threshold for analyzer.usable: {threshold_usable}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
