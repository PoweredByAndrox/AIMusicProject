"""Compare instrumental quality: legacy vs HARD vs WIENER.

Reports for each instrumental candidate:
  * vocal_bleed_db_rel_total -- speech-band correlation vs the matching
    vocals stem (more negative = less vocal residue).
  * total RMS dB.
  * RMS-envelope correlation against the BEFORE-Wiener (intermediate)
    instrumental.  Values >= 0.95 prove no audible ducking; broadband
    suppression collapses this to ~0.7-0.8.
  * RMS-envelope correlation against the original full mix.

This is the host-side replacement for ``analyze_tandem_job.py`` for the
S6 (Wiener anti-ducking) validation.
"""
from __future__ import annotations

import sys
import struct
import math
from pathlib import Path

import numpy as np


def parse_wav(path: Path) -> tuple[np.ndarray, int]:
    """Tiny RIFF/WAVE parser tolerating IEEE_FLOAT (format=3)."""
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")
    pos = 12
    fmt_code = ch = sr = bits = None
    data = b""
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt_code, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
            if fmt_code == 0xFFFE and len(body) >= 24:
                fmt_code = struct.unpack("<H", body[24:26])[0]
        elif chunk_id == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt_code is None or not data:
        raise ValueError(f"Missing fmt/data: {path}")
    if fmt_code == 1 and bits == 16:
        arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif fmt_code == 1 and bits == 24:
        b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        as_int = (b[:, 0].astype(np.int32)
                  | (b[:, 1].astype(np.int32) << 8)
                  | (b[:, 2].astype(np.int32) << 16))
        sign_bit = 1 << 23
        as_int = np.where(as_int & sign_bit, as_int | ~0xFFFFFF, as_int)
        arr = as_int.astype(np.float32) / float(1 << 23)
    elif fmt_code == 3 and bits == 32:
        arr = np.frombuffer(data, dtype="<f4").copy()
    else:
        raise ValueError(f"Unsupported wav: code={fmt_code} bits={bits} ({path})")
    if ch >= 1:
        arr = arr.reshape(-1, ch).T
    return arr, sr


def to_stereo(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return np.stack([a, a], axis=0)
    if a.shape[0] == 1:
        return np.repeat(a, 2, axis=0)
    return a[:2]


def rms_db(a: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12))


def rms_envelope(a: np.ndarray, sr: int, win_ms: float = 50.0) -> np.ndarray:
    win = max(1, int(round(win_ms * 1e-3 * sr)))
    mono = a.mean(axis=0).astype(np.float32)
    n = mono.shape[0] // win
    if n < 2:
        return np.array([float(np.sqrt(np.mean(mono * mono)))], dtype=np.float32)
    return np.sqrt(np.mean(mono[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12).astype(np.float32)


def envelope_corr(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    ea = rms_envelope(a, sr)
    eb = rms_envelope(b, sr)
    n = min(len(ea), len(eb))
    if n < 2:
        return float("nan")
    ea = ea[:n] - ea[:n].mean()
    eb = eb[:n] - eb[:n].mean()
    denom = float(math.sqrt(float(np.sum(ea * ea) * np.sum(eb * eb)) + 1e-30))
    return float(np.sum(ea * eb) / denom)


def speech_band_bleed_db(instr: np.ndarray, vocals: np.ndarray, sr: int) -> float:
    """Return the speech-band cross-correlation power of instr vs vocals
    in dB relative to total instr power. Lower (more negative) = cleaner."""
    n_fft = 4096
    hop = 1024
    half = n_fft // 2
    win = np.hanning(n_fft).astype(np.float32)

    def stft(x):
        x = x.mean(axis=0).astype(np.float32)
        x = np.pad(x, (half, half), mode="reflect")
        n_frames = (len(x) - n_fft) // hop + 1
        out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
        for f in range(n_frames):
            out[:, f] = np.fft.rfft(x[f * hop:f * hop + n_fft] * win)
        return out

    Si = stft(instr)
    Sv = stft(vocals)
    n_min = min(Si.shape[1], Sv.shape[1])
    Si = Si[:, :n_min]
    Sv = Sv[:, :n_min]

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= 200.0) & (freqs <= 4000.0)

    cross = np.real(np.conj(Si[band]) * Sv[band])
    cross_power = float(np.mean(np.maximum(cross, 0.0)))
    instr_power_full = float(np.mean(np.abs(Si) ** 2)) + 1e-30
    return 10.0 * math.log10((cross_power + 1e-30) / instr_power_full)


def align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[1], b.shape[1])
    return a[:, :n], b[:, :n]


def report(name: str, instr_path: Path, vocals_path: Path, *,
           reference_for_corr: tuple[np.ndarray, int] | None = None) -> None:
    instr, sr_i = parse_wav(instr_path)
    vox, sr_v = parse_wav(vocals_path)
    instr = to_stereo(instr)
    vox = to_stereo(vox)
    if sr_i != sr_v:
        # Resample by simple linear interpolation -- precise enough for
        # bleed/RMS metrics; we never round-trip audio through this.
        target_sr = sr_i
        old_n = vox.shape[1]
        new_n = int(round(old_n * target_sr / sr_v))
        idx = np.linspace(0.0, old_n - 1.0, num=new_n, dtype=np.float32)
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, old_n - 1)
        frac = (idx - i0).astype(np.float32)
        vox = ((1.0 - frac) * vox[:, i0] + frac * vox[:, i1]).astype(np.float32)
        sr_v = target_sr
    instr, vox = align_lengths(instr, vox)
    bleed = speech_band_bleed_db(instr, vox, sr_i)
    rms = rms_db(instr)
    print(f"\n=== {name} ===")
    print(f"  instr_path:  {instr_path.name}")
    print(f"  vocals_path: {vocals_path.name}")
    print(f"  duration_s:  {instr.shape[1] / sr_i:.2f}")
    print(f"  total_rms_dB:                {rms:+7.2f}")
    print(f"  vocal_bleed_speechband_dB:   {bleed:+7.2f}    (lower = cleaner)")
    if reference_for_corr is not None:
        ref_audio, ref_sr = reference_for_corr
        if ref_sr != sr_i:
            print(f"  envelope_corr_vs_ref:        sr-mismatch ({ref_sr}!={sr_i}), skipped")
        else:
            ref_clip, instr_clip = align_lengths(ref_audio, instr)
            corr = envelope_corr(ref_clip, instr_clip, sr_i)
            print(f"  envelope_corr_vs_intermediate: {corr:+7.4f}    (>=0.95 = no ducking)")


def main():
    lab = Path(r"D:\AIMusicProject\stemlab_multicontainer_blueprint_v2\data\listening_lab\nine_thou_S6_wiener")
    inter_path = lab / "05a_INTERMEDIATE_polished_no_wiener__instrumental.wav"
    intermediate, inter_sr = parse_wav(inter_path)
    intermediate = to_stereo(intermediate)
    inter_ref = (intermediate, inter_sr)

    print("=" * 70)
    print(" S6 Wiener vs HARD vs Legacy: instrumental quality on Nine Thou")
    print("=" * 70)

    report(
        "WIENER (S6, new default candidate)",
        lab / "05_AFTER_WIENER__instrumental.wav",
        lab / "06_AFTER_WIENER__vocals.wav",
        reference_for_corr=inter_ref,
    )
    report(
        "INTERMEDIATE (after presence_lift, before Wiener)",
        inter_path,
        lab / "06_AFTER_WIENER__vocals.wav",
        reference_for_corr=inter_ref,
    )
    report(
        "PREV HARD (S3 default before today)",
        lab / "07_PREV_HARD__instrumental.wav",
        lab / "08_PREV_HARD__vocals.wav",
    )
    report(
        "LEGACY tandem (no center suppression)",
        lab / "03_BEFORE_legacy__instrumental.wav",
        lab / "04_BEFORE_legacy__vocals.wav",
    )

    print("\n" + "=" * 70)
    print(" Direct anti-ducking check: WIENER instrumental vs INTERMEDIATE")
    print("=" * 70)
    after_w, sr_w = parse_wav(lab / "05_AFTER_WIENER__instrumental.wav")
    after_w = to_stereo(after_w)
    aw, im = align_lengths(after_w, intermediate)
    corr = envelope_corr(im, aw, sr_w)
    rms_drop = rms_db(aw) - rms_db(im)
    print(f"\n  RMS envelope correlation:   {corr:+7.4f}  (1.0 = identical envelope, >=0.95 = no ducking)")
    print(f"  Global RMS drop applied:    {rms_drop:+7.3f} dB  (negative & small = filter only removed vocal residue)")
    if corr >= 0.95 and abs(rms_drop) <= 1.0:
        print("\n  VERDICT: anti-ducking contract satisfied -- per-bin filter "
              "did not modulate global volume envelope.")
    else:
        print("\n  WARNING: anti-ducking contract not met. Investigate.")


if __name__ == "__main__":
    main()
