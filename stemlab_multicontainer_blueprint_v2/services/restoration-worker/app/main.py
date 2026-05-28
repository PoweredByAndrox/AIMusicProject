from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI

from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(title="restoration-worker", version="0.3.0")

RECIPE_FILTERS = {
    # _passthrough: highpass + soft limiter only — no EQ. Use this when a stem
    # is already clean (e.g. ML-cleanup output) and any spectral lift would only
    # amplify model artefacts. Preferred default for stems that are post-ML.
    "passthrough": "highpass=f=20,alimiter=limit=0.98",
    # Softer, BS-preserving variants of the *_unmuffle recipes. Per the
    # stem-tandem-improvement-roadmap audit (R3), the previous +3.2dB / +4.1dB
    # peak shelves were "fake clarity" that mostly highlighted residual Demucs
    # artefacts and added WhisperX fatigue. Cap per-band shelves at ~1.5 dB.
    "generic_cleanup": "highpass=f=35,equalizer=f=2800:t=o:w=1.4:g=0.8,equalizer=f=7000:t=o:w=1.2:g=1.5,alimiter=limit=0.98",
    "music_unmuffle": "highpass=f=32,equalizer=f=2800:t=o:w=1.5:g=1.0,equalizer=f=6500:t=o:w=1.2:g=1.5,equalizer=f=11000:t=o:w=1.0:g=1.2,alimiter=limit=0.98",
    "music_unmuffle_plus": "highpass=f=34,equalizer=f=320:t=o:w=1.0:g=-1.0,equalizer=f=2900:t=o:w=1.5:g=1.5,equalizer=f=6800:t=o:w=1.2:g=2.0,equalizer=f=11800:t=o:w=1.0:g=1.8,alimiter=limit=0.97",
    "drums_unmuffle": "highpass=f=35,equalizer=f=4200:t=o:w=1.3:g=1.4,equalizer=f=8500:t=o:w=1.0:g=1.5,alimiter=limit=0.98",
    "drums_unmuffle_plus": "highpass=f=36,equalizer=f=4300:t=o:w=1.3:g=1.8,equalizer=f=9000:t=o:w=1.0:g=2.0,alimiter=limit=0.97",
    "bass_unmuffle": "highpass=f=26,equalizer=f=950:t=o:w=1.4:g=1.0,equalizer=f=2800:t=o:w=1.2:g=0.8,alimiter=limit=0.98",
    "bass_unmuffle_plus": "highpass=f=26,equalizer=f=1100:t=o:w=1.3:g=1.4,equalizer=f=3200:t=o:w=1.1:g=1.2,alimiter=limit=0.98",
    "harmonic_unmuffle": "highpass=f=40,equalizer=f=3200:t=o:w=1.3:g=1.4,equalizer=f=7600:t=o:w=1.1:g=1.5,alimiter=limit=0.98",
    "harmonic_unmuffle_plus": "highpass=f=42,equalizer=f=350:t=o:w=1.0:g=-0.8,equalizer=f=3400:t=o:w=1.3:g=1.8,equalizer=f=8000:t=o:w=1.1:g=2.0,equalizer=f=11800:t=o:w=1.0:g=1.5,alimiter=limit=0.97",
    "vocals_open": "highpass=f=70,equalizer=f=3200:t=o:w=1.3:g=1.2,equalizer=f=8000:t=o:w=1.1:g=1.5,alimiter=limit=0.98",
    "vocals_air_open": "highpass=f=75,equalizer=f=2600:t=o:w=1.2:g=0.9,equalizer=f=4200:t=o:w=1.2:g=1.4,equalizer=f=9000:t=o:w=1.1:g=1.8,equalizer=f=13000:t=o:w=0.9:g=1.5,alimiter=limit=0.97",
    "vocals_clean": "highpass=f=80,equalizer=f=3200:t=o:w=1.2:g=1.2,equalizer=f=7200:t=o:w=1.0:g=1.5,alimiter=limit=0.98",
    "strings_clean": "highpass=f=70,equalizer=f=3000:t=o:w=1.3:g=1.0,equalizer=f=9000:t=o:w=1.1:g=1.5,alimiter=limit=0.98",
    "synth_clean": "highpass=f=45,equalizer=f=3500:t=o:w=1.2:g=1.4,equalizer=f=9000:t=o:w=1.0:g=1.8,alimiter=limit=0.98",
    # `instrumental_polish` is now an alias of preserve_bs — anything that needs
    # the legacy aggressive curve must request `instrumental_polish_legacy`
    # explicitly. See resolve_recipe() for the routing.
    "instrumental_polish": "highpass=f=28,equalizer=f=2400:t=o:w=1.3:g=0.5,equalizer=f=6200:t=o:w=1.0:g=0.9,equalizer=f=10200:t=o:w=0.9:g=0.5,alimiter=limit=0.98",
    "instrumental_polish_preserve_bs": "highpass=f=28,equalizer=f=2400:t=o:w=1.3:g=0.5,equalizer=f=6200:t=o:w=1.0:g=0.9,equalizer=f=10200:t=o:w=0.9:g=0.5,alimiter=limit=0.98",
    "instrumental_polish_legacy": "highpass=f=30,equalizer=f=2500:t=o:w=1.6:g=1.2,equalizer=f=6200:t=o:w=1.2:g=2.6,equalizer=f=10500:t=o:w=1.0:g=1.8,alimiter=limit=0.97",
    "instrumental_polish_plus": "highpass=f=32,equalizer=f=320:t=o:w=1.0:g=-1.0,equalizer=f=2700:t=o:w=1.5:g=1.5,equalizer=f=6800:t=o:w=1.2:g=2.0,equalizer=f=11800:t=o:w=1.0:g=1.8,alimiter=limit=0.97",
}

TANDEM_HYBRID_PROFILES = {
    "balanced": {
        "seed_weight": 0.25,
        "raw_weight": 0.75,
        "hi_mid": 0.36,
        "hi_side": 1.25,
        "body_mid": 0.10,
        "body_side": 0.34,
        "vocal_mid_suppression": 0.80,
        "vocal_side_suppression": 0.24,
    },
    "bs_stronger": {
        "seed_weight": 0.18,
        "raw_weight": 0.82,
        "hi_mid": 0.48,
        "hi_side": 1.55,
        "body_mid": 0.12,
        "body_side": 0.44,
        "vocal_mid_suppression": 0.76,
        "vocal_side_suppression": 0.16,
    },
    "bs_aggressive": {
        "seed_weight": 0.10,
        "raw_weight": 0.90,
        "hi_mid": 0.60,
        "hi_side": 1.85,
        "body_mid": 0.15,
        "body_side": 0.54,
        "vocal_mid_suppression": 0.70,
        "vocal_side_suppression": 0.10,
    },
}

OTHER_REBUILD_PROFILES = {
    "balanced": {
        "soft_guitar_subtract": 0.82,
        "soft_piano_subtract": 0.68,
        "strict_body_blend": 0.38,
        "strict_high_blend": 0.58,
        "soft_body_blend": 0.92,
        "soft_high_blend": 0.76,
        "legacy_other_blend": 0.18,
        "center_keep": 0.72,
        "side_keep": 1.10,
        "vocal_subtract": 0.08,
        "target_rms_ratio": 0.080,
        "max_makeup_gain": 5.0,
    },
    "wide_open": {
        "soft_guitar_subtract": 0.74,
        "soft_piano_subtract": 0.58,
        "strict_body_blend": 0.34,
        "strict_high_blend": 0.66,
        "soft_body_blend": 1.00,
        "soft_high_blend": 0.92,
        "legacy_other_blend": 0.22,
        "center_keep": 0.66,
        "side_keep": 1.18,
        "vocal_subtract": 0.10,
        "target_rms_ratio": 0.105,
        "max_makeup_gain": 6.5,
    },
}

OTHER_GUIDE_PROFILES = {
    "off": {
        "hi_mid": 0.0,
        "hi_side": 0.0,
        "body_mid": 0.0,
        "body_side": 0.0,
        "mid_vocal_suppression": 0.0,
        "side_vocal_suppression": 0.0,
        "activity_floor": 0.0,
    },
    "residual_hint": {
        "hi_mid": 0.05,
        "hi_side": 0.26,
        "body_mid": 0.02,
        "body_side": 0.09,
        "mid_vocal_suppression": 0.82,
        "side_vocal_suppression": 0.24,
        "activity_floor": 0.18,
    },
    "residual_push": {
        "hi_mid": 0.09,
        "hi_side": 0.40,
        "body_mid": 0.04,
        "body_side": 0.14,
        "mid_vocal_suppression": 0.78,
        "side_vocal_suppression": 0.18,
        "activity_floor": 0.20,
    },
}

CENTER_SUPPRESSION_PROFILES = {
    "off": {
        "guide_final_weight": 1.0,
        "guide_bs_weight": 0.0,
        "body_mid_suppress": 0.0,
        "presence_mid_suppress": 0.0,
        "air_mid_suppress": 0.0,
        "presence_side_suppress": 0.0,
        "guide_subtract": 0.0,
        "activity_floor": 0.0,
    },
    "gentle": {
        "guide_final_weight": 0.78,
        "guide_bs_weight": 0.22,
        "body_mid_suppress": 0.08,
        "presence_mid_suppress": 0.18,
        "air_mid_suppress": 0.05,
        "presence_side_suppress": 0.03,
        "guide_subtract": 0.05,
        "activity_floor": 0.12,
    },
    "focused": {
        "guide_final_weight": 0.72,
        "guide_bs_weight": 0.28,
        "body_mid_suppress": 0.12,
        "presence_mid_suppress": 0.28,
        "air_mid_suppress": 0.08,
        "presence_side_suppress": 0.05,
        "guide_subtract": 0.08,
        "activity_floor": 0.14,
    },
    "hard": {
        "guide_final_weight": 0.66,
        "guide_bs_weight": 0.34,
        "body_mid_suppress": 0.18,
        "presence_mid_suppress": 0.40,
        "air_mid_suppress": 0.12,
        "presence_side_suppress": 0.08,
        "guide_subtract": 0.12,
        "activity_floor": 0.16,
    },
}


def _parse_wav_header(path: Path) -> tuple[int, int, int, int, bytes]:
    """Manual RIFF/WAVE parser that tolerates IEEE_FLOAT (format=3).

    Mirrors demucs-worker._parse_wav_header. Python's stdlib
    ``wave.open(rb)`` rejects any wFormatTag other than 1
    (``wave.Error: unknown format: 3``), so wavs written by demucs-worker's
    ``_write_float_wav`` (or by other tools that emit IEEE_FLOAT) cannot be
    consumed through ``wave``. We parse RIFF chunks manually so the read
    path matches the write path symmetrically.

    Supports format codes 1 (PCM) and 3 (IEEE_FLOAT); also unwraps
    WAVE_FORMAT_EXTENSIBLE (0xFFFE) to its inner SubFormat code. Returns
    ``(format_code, channels, samplerate, sample_width_bytes, raw_data_bytes)``.
    """
    raw_format_code: int | None = None
    channels: int | None = None
    samplerate: int | None = None
    sample_width: int | None = None
    data_bytes: bytes | None = None

    with path.open("rb") as handle:
        riff = handle.read(12)
        if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")

        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_id = header[0:4]
            chunk_size = struct.unpack("<I", header[4:8])[0]
            chunk_data = handle.read(chunk_size)
            if chunk_size % 2 == 1:
                handle.read(1)
            if chunk_id == b"fmt ":
                if len(chunk_data) < 16:
                    raise ValueError(f"fmt chunk too short in {path}")
                raw_format_code = struct.unpack("<H", chunk_data[0:2])[0]
                channels = struct.unpack("<H", chunk_data[2:4])[0]
                samplerate = struct.unpack("<I", chunk_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", chunk_data[14:16])[0]
                sample_width = bits_per_sample // 8
                if raw_format_code == 0xFFFE and len(chunk_data) >= 40:
                    sub_format = chunk_data[24:40]
                    inner_code = struct.unpack("<H", sub_format[0:2])[0]
                    raw_format_code = inner_code
            elif chunk_id == b"data":
                data_bytes = chunk_data

    if (
        raw_format_code is None
        or channels is None
        or samplerate is None
        or sample_width is None
        or data_bytes is None
    ):
        raise ValueError(f"Incomplete WAV header in {path}")

    return raw_format_code, int(channels), int(samplerate), int(sample_width), data_bytes


def _read_wave_file(path: Path) -> tuple[np.ndarray, int]:
    format_code, channels, samplerate, sample_width, raw = _parse_wav_header(path)

    if format_code == 3:
        if sample_width == 4:
            audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif sample_width == 8:
            audio = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported IEEE_FLOAT sample width: {sample_width} bytes ({path})"
            )
    elif format_code == 1:
        if sample_width == 2:
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        elif sample_width == 3:
            triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            ints = (
                triplets[:, 0].astype(np.int32)
                | (triplets[:, 1].astype(np.int32) << 8)
                | (triplets[:, 2].astype(np.int32) << 16)
            )
            sign_bit = 1 << 23
            ints = np.where(ints & sign_bit, ints - (1 << 24), ints)
            audio = ints.astype(np.float32) / float((1 << 23) - 1)
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(
                (1 << 31) - 1
            )
        else:
            raise ValueError(
                f"Unsupported PCM sample width: {sample_width} bytes ({path})"
            )
    else:
        raise ValueError(
            f"Unsupported WAV format code {format_code} (only 1=PCM, 3=IEEE_FLOAT) for {path}"
        )

    audio = audio.reshape(-1, channels).T
    return audio, samplerate


def _write_wave_file(path: Path, audio: np.ndarray, samplerate: int) -> None:
    """Write a 32-bit float WAV (IEEE_FLOAT, format code 3).

    Per R-PCM in the stem-tandem-improvement-roadmap, intermediate stems are
    kept in 32-bit float so cumulative ML refinements do not accrue 16-bit
    quantisation error. The Python `wave` module only writes PCM-tagged WAVs,
    so we emit the RIFF / fmt / fact / data chunks directly.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    channels = int(audio.shape[0])
    frames = np.ascontiguousarray(audio.T)
    raw = frames.astype("<f4", copy=False).tobytes()

    bits_per_sample = 32
    bytes_per_sample = 4
    block_align = channels * bytes_per_sample
    byte_rate = int(samplerate) * block_align

    fmt_chunk_size = 16
    fact_chunk_size = 4
    num_samples = int(frames.shape[0])
    data_size = len(raw)
    riff_size = (
        4
        + 8 + fmt_chunk_size
        + 8 + fact_chunk_size
        + 8 + data_size
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", riff_size))
        handle.write(b"WAVE")

        handle.write(b"fmt ")
        handle.write(struct.pack("<I", fmt_chunk_size))
        handle.write(struct.pack("<H", 3))
        handle.write(struct.pack("<H", channels))
        handle.write(struct.pack("<I", int(samplerate)))
        handle.write(struct.pack("<I", byte_rate))
        handle.write(struct.pack("<H", block_align))
        handle.write(struct.pack("<H", bits_per_sample))

        handle.write(b"fact")
        handle.write(struct.pack("<I", fact_chunk_size))
        handle.write(struct.pack("<I", num_samples))

        handle.write(b"data")
        handle.write(struct.pack("<I", data_size))
        handle.write(raw)


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]
    return audio.astype(np.float32, copy=False)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[1] / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[1], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    out = np.vstack([np.interp(dst_x, src_x, channel).astype(np.float32) for channel in audio])
    return out


def _match_audio(audio: np.ndarray, src_sr: int, dst_sr: int, target_len: int) -> np.ndarray:
    audio = _ensure_stereo(audio)
    if src_sr != dst_sr:
        audio = _resample_linear(audio, src_sr, dst_sr)
    if audio.shape[1] < target_len:
        padded = np.zeros((audio.shape[0], target_len), dtype=np.float32)
        padded[:, : audio.shape[1]] = audio
        return padded
    return audio[:, :target_len].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# R8 - lightweight Wiener post-filter helpers.
#
# We deliberately avoid pulling in scipy/torch here so that the existing
# restoration-worker image stays slim. The STFT / iSTFT below use numpy.fft
# only; with a Hann window and ``hop = n_fft // 4`` the COLA condition is
# satisfied, which is enough for a soft-mask Wiener post-filter.
# ---------------------------------------------------------------------------


def _hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    return np.hanning(n).astype(np.float32)


def _stft(
    audio: np.ndarray, n_fft: int = 4096, hop: int = 1024
) -> tuple[np.ndarray, int]:
    """Real-input centered STFT for a (channels, samples) array.

    Mirrors the framing used by ``scipy.signal.stft(..., padded=True)`` /
    ``librosa.stft(..., center=True)``: the input is reflected by ``n_fft // 2``
    samples on both sides so every original sample is covered by an integer
    number of overlapping Hann windows. Returns ``(spectrogram, original_len)``;
    the spectrogram has shape ``(channels, n_fft // 2 + 1, n_frames)``.
    """
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio = audio.astype(np.float32, copy=False)
    original_len = audio.shape[1]

    window = _hann_window(n_fft)
    half = n_fft // 2
    padded = np.pad(audio, ((0, 0), (half, half)), mode="reflect")

    tail_pad = (-(padded.shape[1] - n_fft)) % hop
    if tail_pad:
        padded = np.pad(padded, ((0, 0), (0, tail_pad)), mode="constant")
    n_frames = (padded.shape[1] - n_fft) // hop + 1

    spec = np.empty(
        (audio.shape[0], n_fft // 2 + 1, n_frames), dtype=np.complex64
    )
    for f in range(n_frames):
        start = f * hop
        frame = padded[:, start : start + n_fft] * window
        spec[:, :, f] = np.fft.rfft(frame, axis=1).astype(np.complex64)
    return spec, original_len


def _istft(
    spec: np.ndarray,
    n_fft: int = 4096,
    hop: int = 1024,
    target_len: int | None = None,
) -> np.ndarray:
    """Inverse of :func:`_stft`. Drops the centered-padding before returning."""
    window = _hann_window(n_fft)
    n_frames = spec.shape[2]
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros((spec.shape[0], out_len), dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    win_sq = (window * window).astype(np.float32)

    for f in range(n_frames):
        frame = np.fft.irfft(spec[:, :, f], n=n_fft, axis=1).astype(np.float32)
        start = f * hop
        out[:, start : start + n_fft] += frame * window
        norm[start : start + n_fft] += win_sq

    norm = np.maximum(norm, 1e-9)
    out = out / norm[np.newaxis, :]

    half = n_fft // 2
    out = out[:, half : out.shape[1] - half]

    if target_len is not None:
        if out.shape[1] >= target_len:
            out = out[:, :target_len]
        else:
            padded = np.zeros((out.shape[0], target_len), dtype=np.float32)
            padded[:, : out.shape[1]] = out
            out = padded
    return out.astype(np.float32, copy=False)


STEM_ENSEMBLE_METHODS = ("median_fft", "min_fft", "max_fft", "avg_fft", "min_mag")

STEM_ENSEMBLE_PROFILES = {
    # Per-bin median magnitude with mix-phase. Robust default - drops outliers
    # from one model without overweighting the others. Recommended starting
    # point per the audio-separator project (see R9 in the roadmap).
    "median_fft": {
        "method": "median_fft",
        "n_fft": 4096,
        "hop": 1024,
        "use_mix_phase": True,
        "mix_blend": 0.0,
    },
    # Per-bin minimum magnitude. Aggressive: takes the most conservative
    # estimate per bin. Best for vocals where false-positive content is
    # worse than missing a faint detail.
    "min_fft": {
        "method": "min_fft",
        "n_fft": 4096,
        "hop": 1024,
        "use_mix_phase": True,
        "mix_blend": 0.0,
    },
    # Per-bin maximum magnitude. Inclusive: keeps any content that any model
    # claims is in this stem. Useful when the candidates systematically
    # under-estimate (e.g. faint reverb tails, breath noise).
    "max_fft": {
        "method": "max_fft",
        "n_fft": 4096,
        "hop": 1024,
        "use_mix_phase": True,
        "mix_blend": 0.0,
    },
    # Per-bin average magnitude. Smoothest sounding result, but tends to
    # smear leakage rather than reject it. Use as A/B reference, not default.
    "avg_fft": {
        "method": "avg_fft",
        "n_fft": 4096,
        "hop": 1024,
        "use_mix_phase": True,
        "mix_blend": 0.0,
    },
    # Per-bin min magnitude with the candidate's own phase (rather than mix
    # phase). Closest to the audio-separator package's `min_mag` algorithm.
    "min_mag": {
        "method": "min_mag",
        "n_fft": 4096,
        "hop": 1024,
        "use_mix_phase": False,
        "mix_blend": 0.0,
    },
}


WIENER_PROFILES = {
    "off": {
        "alpha": 0.0,
        "n_fft": 4096,
        "hop": 1024,
        "spectral_floor": 0.0,
        "mix_blend": 0.0,
    },
    "soft": {
        # Magnitude-ratio mask, gentle on transients. Recommended default.
        "alpha": 1.0,
        "n_fft": 4096,
        "hop": 1024,
        "spectral_floor": 1e-6,
        "mix_blend": 0.30,
    },
    "classical": {
        # Power-ratio Wiener mask (alpha=2). Stronger separation at the
        # cost of slightly more "musical noise" if the mask floor is low.
        "alpha": 2.0,
        "n_fft": 4096,
        "hop": 1024,
        "spectral_floor": 1e-6,
        "mix_blend": 0.20,
    },
    "transparent": {
        # Very gentle mask - mostly used when stems are already very clean
        # (e.g. after BS-Roformer with TTA) and we just want to suppress
        # cross-talk without colouring the stem.
        "alpha": 1.0,
        "n_fft": 4096,
        "hop": 1024,
        "spectral_floor": 1e-5,
        "mix_blend": 0.50,
    },
}


def _fft_band(audio: np.ndarray, sr: int, low_hz: float | None, high_hz: float | None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    n = audio.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = np.fft.rfft(audio, axis=1)
    mask = np.ones_like(freqs, dtype=np.float32)
    if low_hz is not None:
        mask *= (freqs >= low_hz).astype(np.float32)
    if high_hz is not None:
        mask *= (freqs <= high_hz).astype(np.float32)
    filtered = np.fft.irfft(spectrum * mask[np.newaxis, :], n=n, axis=1)
    return filtered.astype(np.float32, copy=False)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal, kernel, mode="same").astype(np.float32, copy=False)


def _activity_mask(audio: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    """Adaptive activity envelope mask (R-ACT in the tandem-improvement audit).

    Previously the floor/ceiling were hard-coded at the 35th / 97th
    percentiles. That was too aggressive on heavily-compressed material
    (background "leaks" through because the 35th percentile is well above
    the actual silence floor) and too conservative on dynamic material
    (lots of legitimate activity gets pushed below the floor). The new
    formulation:

    1. Estimates the dynamic range from log-domain percentiles.
    2. Adapts both the floor and ceiling percentiles based on the dynamic
       range — compressed material gets a higher floor / lower ceiling so
       the noise plateau is suppressed, dynamic material gets a wider
       window so quiet detail survives.
    3. Applies a Hermite smoothstep (``3x^2 - 2x^3``) instead of a linear
       clip so the activation transition is gentle. This visibly reduces
       on/off flicker that the linear mask used to print on transient
       attacks.

    Inputs are expected as ``(channels, samples)`` numpy arrays. The output
    is a 1-D float32 mask aligned to ``target_len``.
    """
    mono = np.mean(np.abs(audio), axis=0)
    env = _moving_average(mono, max(1, int(sr * 0.08)))
    if env.size == 0:
        return np.zeros(target_len, dtype=np.float32)

    eps = 1e-9
    p5 = float(np.percentile(env, 5))
    p99 = float(np.percentile(env, 99))
    p_low = max(p5, eps)
    p_high = max(p99, p_low + eps)
    dynamic_range_db = 20.0 * np.log10(p_high / p_low)

    if dynamic_range_db <= 12.0:
        # Heavily-compressed material: tighten the window so the masked
        # plateau collapses to silence in the long tails between notes.
        floor_pct, ceiling_pct = 50.0, 96.0
    elif dynamic_range_db >= 30.0:
        # Highly dynamic material: widen the window so quiet detail is kept.
        floor_pct, ceiling_pct = 25.0, 99.0
    else:
        # Lerp between the two regimes.
        t = (dynamic_range_db - 12.0) / 18.0
        floor_pct = 50.0 + (25.0 - 50.0) * t
        ceiling_pct = 96.0 + (99.0 - 96.0) * t

    floor = float(np.percentile(env, floor_pct))
    ceiling = float(np.percentile(env, ceiling_pct))
    if ceiling <= floor + 1e-8:
        return np.zeros(target_len, dtype=np.float32)

    normalised = np.clip((env - floor) / (ceiling - floor), 0.0, 1.0)
    smoothstep = (3.0 * normalised * normalised) - (2.0 * normalised * normalised * normalised)
    mask = smoothstep.astype(np.float32, copy=False)

    if mask.shape[0] < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[: mask.shape[0]] = mask
        return padded
    return mask[:target_len]


def _vocal_activity_mask(vocals: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    return _activity_mask(vocals, sr, target_len)


def _mid_side(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = audio[0]
    right = audio[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return mid.astype(np.float32, copy=False), side.astype(np.float32, copy=False)


def _from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    left = mid + side
    right = mid - side
    return np.vstack([left, right]).astype(np.float32, copy=False)


def _peak_normalize(audio: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= peak_limit or peak <= 1e-9:
        return audio.astype(np.float32, copy=False)
    return (audio * (peak_limit / peak)).astype(np.float32, copy=False)


def _resolve_existing_build_from(request: RunRequest, keys: tuple[str, ...]) -> tuple[str | None, Path | None]:
    for key in keys:
        candidate = request.build_from_paths.get(key)
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return key, path
    return None, None


def _load_matched_build_from_audio(
    request: RunRequest,
    keys: tuple[str, ...],
    dst_sr: int,
    target_len: int,
) -> tuple[str | None, Path | None, np.ndarray | None]:
    key, path = _resolve_existing_build_from(request, keys)
    if path is None:
        return None, None, None
    audio, sr = _read_wave_file(path)
    return key, path, _match_audio(audio, sr, dst_sr, target_len)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _vocal_contamination_db(
    audio: np.ndarray, vocals_guide: np.ndarray, sr: int
) -> float:
    """Speech-band vocal-bleed estimate for a candidate instrumental track.

    Projects the speech-band content of ``audio`` (200..3500 Hz) onto the
    speech-band content of ``vocals_guide`` per channel and returns the
    energy of that projection in dB relative to the total RMS of ``audio``.

    Higher (less negative) values mean more vocal energy is sitting inside
    ``audio``. ``-30 dB`` and below is "clean enough" for hybrid blending
    purposes; ``-5 dB`` and above is essentially "this file is mostly vocals".

    The returned value is *relative*, so a small constant gain difference
    between ``audio`` and ``vocals_guide`` doesn't dominate the metric.
    """
    if audio is None or vocals_guide is None or audio.size == 0 or vocals_guide.size == 0:
        return float("nan")
    high_hz = min(3500.0, float(sr) * 0.48)
    if high_hz <= 200.0:
        return float("nan")
    audio_band = _fft_band(audio, sr, 200.0, high_hz)
    vocals_band = _fft_band(vocals_guide, sr, 200.0, high_hz)

    n = min(audio_band.shape[-1], vocals_band.shape[-1])
    if n <= 0:
        return float("nan")
    audio_band = audio_band[..., :n]
    vocals_band = vocals_band[..., :n]
    if audio_band.ndim == 1:
        audio_band = audio_band[np.newaxis, :]
    if vocals_band.ndim == 1:
        vocals_band = vocals_band[np.newaxis, :]
    channels = min(audio_band.shape[0], vocals_band.shape[0])
    proj_energy = 0.0
    for ch in range(channels):
        a = audio_band[ch]
        v = vocals_band[ch]
        v_energy = float(np.dot(v, v))
        if v_energy <= 1e-12:
            continue
        alpha = float(np.dot(a, v)) / v_energy
        proj = alpha * v
        proj_energy += float(np.dot(proj, proj))
    if channels <= 0 or proj_energy <= 0.0:
        return float("-inf")
    proj_rms = float(np.sqrt(proj_energy / float(channels)))
    audio_rms = _rms(audio)
    if audio_rms <= 1e-12:
        return float("nan")
    return 20.0 * float(np.log10((proj_rms + 1e-12) / (audio_rms + 1e-12)))


def _resolve_tandem_hybrid_profile(request: RunRequest) -> tuple[str, dict[str, float]]:
    raw_profile = str(request.raw_step.get("hybrid_profile", "balanced") or "balanced").strip().lower()
    if raw_profile not in TANDEM_HYBRID_PROFILES:
        raw_profile = "balanced"

    resolved = dict(TANDEM_HYBRID_PROFILES[raw_profile])
    overrides = request.raw_step.get("hybrid_weights")
    if isinstance(overrides, dict):
        for key in resolved:
            if key in overrides:
                try:
                    resolved[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue

    return raw_profile, resolved


def _resolve_other_rebuild_profile(request: RunRequest) -> tuple[str, dict[str, float]]:
    raw_profile = str(request.raw_step.get("rebuild_profile", "balanced") or "balanced").strip().lower()
    if raw_profile not in OTHER_REBUILD_PROFILES:
        raw_profile = "balanced"

    resolved = dict(OTHER_REBUILD_PROFILES[raw_profile])
    overrides = request.raw_step.get("rebuild_weights")
    if isinstance(overrides, dict):
        for key in resolved:
            if key in overrides:
                try:
                    resolved[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
    return raw_profile, resolved


_OTHER_REBUILD_MODES = {"direct", "residual", "auto"}


def _resolve_other_rebuild_mode(request: RunRequest) -> str:
    """R7: select between direct ML other (preferred) and legacy residual subtraction.

    The raw_step payload may carry ``mode: direct|residual|auto``. ``direct`` is
    the new default after the audit because the residual path accumulates
    drums/bass/guitar/piano errors into ``other`` (see F3 in the roadmap).
    ``residual`` is kept as an explicit fallback for legacy presets and A/B
    work; ``auto`` picks ``direct`` whenever a direct ML ``other`` artifact is
    actually wired in.
    """
    raw_mode = str(request.raw_step.get("mode", "direct") or "direct").strip().lower()
    if raw_mode not in _OTHER_REBUILD_MODES:
        raw_mode = "direct"
    if raw_mode == "auto":
        key, _ = _resolve_existing_build_from(
            request, ("demucs_other", "other", "bs_other")
        )
        return "direct" if key is not None else "residual"
    return raw_mode


def _resolve_other_guide_profile(request: RunRequest) -> tuple[str, dict[str, float]]:
    raw_profile = str(request.raw_step.get("other_detail_profile", "off") or "off").strip().lower()
    if raw_profile not in OTHER_GUIDE_PROFILES:
        raw_profile = "off"

    resolved = dict(OTHER_GUIDE_PROFILES[raw_profile])
    overrides = request.raw_step.get("other_detail_weights")
    if isinstance(overrides, dict):
        for key in resolved:
            if key in overrides:
                try:
                    resolved[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
    return raw_profile, resolved


def _resolve_center_suppression_profile(request: RunRequest) -> tuple[str, dict[str, float]]:
    raw_profile = str(request.raw_step.get("center_suppression_profile", "off") or "off").strip().lower()
    if raw_profile not in CENTER_SUPPRESSION_PROFILES:
        raw_profile = "off"

    resolved = dict(CENTER_SUPPRESSION_PROFILES[raw_profile])
    overrides = request.raw_step.get("center_suppression_weights")
    if isinstance(overrides, dict):
        for key in resolved:
            if key in overrides:
                try:
                    resolved[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
    return raw_profile, resolved


def _blend_audio_guides(*weighted_audios: tuple[np.ndarray | None, float]) -> np.ndarray | None:
    acc: np.ndarray | None = None
    total = 0.0
    for audio, weight in weighted_audios:
        if audio is None or weight <= 0.0:
            continue
        if acc is None:
            acc = np.zeros_like(audio, dtype=np.float32)
        acc = acc + (audio * float(weight))
        total += float(weight)
    if acc is None or total <= 1e-9:
        return None
    return (acc / total).astype(np.float32, copy=False)


def _apply_center_vocal_suppression(
    audio: np.ndarray,
    guide_audio: np.ndarray | None,
    activity_mask: np.ndarray,
    sr: int,
    weights: dict[str, float],
) -> np.ndarray:
    if guide_audio is None:
        return audio.astype(np.float32, copy=False)

    low_band = _fft_band(audio, sr, None, 180.0)
    body_band = _fft_band(audio, sr, 180.0, 1400.0)
    presence_band = _fft_band(audio, sr, 1400.0, 5200.0)
    air_band = _fft_band(audio, sr, 5200.0, min(14000.0, sr * 0.48))

    guide_body = _fft_band(guide_audio, sr, 180.0, 1400.0)
    guide_presence = _fft_band(guide_audio, sr, 1400.0, 5200.0)
    guide_air = _fft_band(guide_audio, sr, 5200.0, min(14000.0, sr * 0.48))

    body_mid, body_side = _mid_side(body_band)
    presence_mid, presence_side = _mid_side(presence_band)
    air_mid, air_side = _mid_side(air_band)

    guide_body_mid, _ = _mid_side(guide_body)
    guide_presence_mid, guide_presence_side = _mid_side(guide_presence)
    guide_air_mid, _ = _mid_side(guide_air)

    mask = weights["activity_floor"] + ((1.0 - weights["activity_floor"]) * activity_mask)

    body_mid = (body_mid * (1.0 - (weights["body_mid_suppress"] * mask))) - (
        guide_body_mid * (weights["guide_subtract"] * 0.55 * mask)
    )
    presence_mid = (presence_mid * (1.0 - (weights["presence_mid_suppress"] * mask))) - (
        guide_presence_mid * (weights["guide_subtract"] * mask)
    )
    air_mid = (air_mid * (1.0 - (weights["air_mid_suppress"] * mask))) - (
        guide_air_mid * (weights["guide_subtract"] * 0.45 * mask)
    )
    presence_side = (presence_side * (1.0 - (weights["presence_side_suppress"] * mask))) - (
        guide_presence_side * (weights["guide_subtract"] * 0.35 * mask)
    )

    return _peak_normalize(low_band + _from_mid_side(body_mid, body_side) + _from_mid_side(presence_mid, presence_side) + _from_mid_side(air_mid, air_side))


def _run_other_direct_polish(
    request: RunRequest,
    output_root: Path,
    base_audio: np.ndarray,
    base_sr: int,
    target_len: int,
    weights: dict[str, float],
    profile_name: str,
) -> tuple[dict[str, str], dict[str, object], str]:
    """R7 'direct' mode: take the ML model's `other` directly and polish it.

    This avoids the F3 error-accumulation chain (where residual = base - drums
    - bass - guitar - piano lifts every component error into `other`).
    The polish step optionally subtracts vocal bleed using the adaptive
    activity mask (R-ACT), then applies M/S keep weights, RMS makeup gain,
    and peak normalisation, mirroring the cosmetics of the residual path so
    downstream loudness stays consistent.
    """
    direct_key, direct_path, direct_audio = _load_matched_build_from_audio(
        request,
        ("demucs_other", "other", "bs_other"),
        base_sr,
        target_len,
    )
    if direct_audio is None or direct_path is None or direct_key is None:
        raise FileNotFoundError(
            "R7 direct mode requires a direct ML `other` artifact "
            "(demucs_other / other / bs_other). None of these were wired in."
        )

    rebuilt = direct_audio.astype(np.float32, copy=True)

    vocals_key, vocals_path, vocals_audio = _load_matched_build_from_audio(
        request,
        ("vocals", "demucs_vocals", "bs_vocals"),
        base_sr,
        target_len,
    )
    vocal_activity = np.zeros(target_len, dtype=np.float32)
    if (
        vocals_audio is not None
        and vocals_path is not None
        and vocals_key is not None
        and weights["vocal_subtract"] > 0.0
    ):
        vocal_activity = _activity_mask(vocals_audio, base_sr, target_len)
        vocal_band = _fft_band(vocals_audio, base_sr, 180.0, min(6500.0, base_sr * 0.48))
        rebuilt = rebuilt - (
            vocal_band * (weights["vocal_subtract"] * vocal_activity[np.newaxis, :])
        )

    rebuilt_mid, rebuilt_side = _mid_side(rebuilt)
    rebuilt = _from_mid_side(
        rebuilt_mid * weights["center_keep"], rebuilt_side * weights["side_keep"]
    )

    base_rms = _rms(base_audio)
    rebuilt_rms = _rms(rebuilt)
    rms_ratio = rebuilt_rms / (base_rms + 1e-12)
    applied_makeup_gain = 1.0
    if rms_ratio < weights["target_rms_ratio"]:
        applied_makeup_gain = min(
            weights["max_makeup_gain"],
            weights["target_rms_ratio"] / (rms_ratio + 1e-12),
        )
        rebuilt = rebuilt * applied_makeup_gain

    rebuilt = _peak_normalize(rebuilt)

    output_path = output_path_for(request, output_root)
    _write_wave_file(output_path, rebuilt, base_sr)
    named_outputs = {request.target_stem or request.task: str(output_path)}
    metadata = {
        "mode": "other_direct_ml",
        "rebuild_profile": profile_name,
        "direct_other_input": str(direct_path),
        "direct_other_key": direct_key,
        "vocals_input": str(vocals_path) if vocals_path else None,
        "vocals_key": vocals_key,
        "sample_rate": base_sr,
        "target_len_samples": target_len,
        "weights": {key: round(float(value), 6) for key, value in weights.items()},
        "rms": {
            "base": round(base_rms, 6),
            "rebuilt_pre_gain": round(rebuilt_rms, 6),
            "ratio_pre_gain": round(rms_ratio, 6),
            "makeup_gain": round(float(applied_makeup_gain), 6),
            "rebuilt_post_gain": round(_rms(rebuilt), 6),
        },
        "vocal_activity": {
            "mean": round(float(np.mean(vocal_activity)), 6),
            "max": round(float(np.max(vocal_activity)), 6),
        },
    }
    notes = (
        "Direct ML `other` from the separator (htdemucs_ft / htdemucs_6s / bs), "
        "polished with optional vocal-bleed subtraction and M/S shaping. "
        "This is the R7 default after the tandem-improvement audit; "
        "the legacy residual subtraction path is available via mode: residual."
    )
    return named_outputs, metadata, notes


def run_other_residual_rebuild(request: RunRequest, output_root: Path) -> tuple[dict[str, str], dict[str, object], str]:
    base_path = Path(request.input_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base instrumental input is missing: {base_path}")

    base_audio, base_sr = _read_wave_file(base_path)
    base_audio = _ensure_stereo(base_audio)
    target_len = base_audio.shape[1]

    profile_name, weights = _resolve_other_rebuild_profile(request)
    mode = _resolve_other_rebuild_mode(request)

    if mode == "direct":
        return _run_other_direct_polish(
            request, output_root, base_audio, base_sr, target_len, weights, profile_name
        )

    component_specs = [
        ("drums", ("drums", "demucs_drums", "bs_drums"), 1.0),
        ("bass", ("bass", "demucs_bass", "bs_bass"), 1.0),
        ("guitar", ("guitar", "demucs_guitar", "bs_guitar"), weights["soft_guitar_subtract"]),
        ("piano", ("piano", "demucs_piano", "bs_piano"), weights["soft_piano_subtract"]),
    ]

    strict_residual = base_audio.copy()
    soft_residual = base_audio.copy()
    used_components: dict[str, dict[str, object]] = {}

    for family, keys, soft_weight in component_specs:
        resolved_key, resolved_path, component_audio = _load_matched_build_from_audio(request, keys, base_sr, target_len)
        if component_audio is None or resolved_path is None or resolved_key is None:
            continue
        strict_residual = strict_residual - component_audio
        soft_residual = soft_residual - (component_audio * soft_weight)
        used_components[family] = {
            "artifact_key": resolved_key,
            "path": str(resolved_path),
            "soft_subtract_weight": round(float(soft_weight), 6),
        }

    if not used_components:
        raise FileNotFoundError("No accompaniment stems were available to rebuild the residual other stem.")

    strict_body = _fft_band(strict_residual, base_sr, 180.0, 2400.0)
    strict_high = _fft_band(strict_residual, base_sr, 2400.0, min(14000.0, base_sr * 0.48))
    soft_body = _fft_band(soft_residual, base_sr, 140.0, 2600.0)
    soft_high = _fft_band(soft_residual, base_sr, 2600.0, min(14000.0, base_sr * 0.48))

    rebuilt = (
        strict_body * weights["strict_body_blend"]
        + strict_high * weights["strict_high_blend"]
        + soft_body * weights["soft_body_blend"]
        + soft_high * weights["soft_high_blend"]
    )

    legacy_key, legacy_path, legacy_other = _load_matched_build_from_audio(
        request,
        ("demucs_other", "other", "bs_other"),
        base_sr,
        target_len,
    )
    if legacy_other is not None and legacy_path is not None and legacy_key is not None:
        rebuilt = rebuilt + (legacy_other * weights["legacy_other_blend"])

    vocals_key, vocals_path, vocals_audio = _load_matched_build_from_audio(
        request,
        ("vocals", "demucs_vocals", "bs_vocals"),
        base_sr,
        target_len,
    )
    vocal_activity = np.zeros(target_len, dtype=np.float32)
    if vocals_audio is not None and vocals_path is not None and vocals_key is not None and weights["vocal_subtract"] > 0.0:
        vocal_activity = _activity_mask(vocals_audio, base_sr, target_len)
        vocal_band = _fft_band(vocals_audio, base_sr, 180.0, min(6500.0, base_sr * 0.48))
        rebuilt = rebuilt - (vocal_band * (weights["vocal_subtract"] * vocal_activity[np.newaxis, :]))

    rebuilt_mid, rebuilt_side = _mid_side(rebuilt)
    rebuilt = _from_mid_side(rebuilt_mid * weights["center_keep"], rebuilt_side * weights["side_keep"])

    base_rms = _rms(base_audio)
    rebuilt_rms = _rms(rebuilt)
    rms_ratio = rebuilt_rms / (base_rms + 1e-12)
    applied_makeup_gain = 1.0
    if rms_ratio < weights["target_rms_ratio"]:
        applied_makeup_gain = min(weights["max_makeup_gain"], weights["target_rms_ratio"] / (rms_ratio + 1e-12))
        rebuilt = rebuilt * applied_makeup_gain

    rebuilt = _peak_normalize(rebuilt)

    output_path = output_path_for(request, output_root)
    _write_wave_file(output_path, rebuilt, base_sr)
    named_outputs = {request.target_stem or request.task: str(output_path)}
    metadata = {
        "mode": "instrumental_residual_other_rebuild",
        "mode_requested": mode,
        "rebuild_profile": profile_name,
        "base_input": str(base_path),
        "sample_rate": base_sr,
        "target_len_samples": target_len,
        "used_components": used_components,
        "legacy_other_input": str(legacy_path) if legacy_path else None,
        "legacy_other_key": legacy_key,
        "vocals_input": str(vocals_path) if vocals_path else None,
        "vocals_key": vocals_key,
        "weights": {key: round(float(value), 6) for key, value in weights.items()},
        "rms": {
            "base": round(base_rms, 6),
            "rebuilt_pre_gain": round(rebuilt_rms, 6),
            "ratio_pre_gain": round(rms_ratio, 6),
            "makeup_gain": round(float(applied_makeup_gain), 6),
            "rebuilt_post_gain": round(_rms(rebuilt), 6),
        },
        "vocal_activity": {
            "mean": round(float(np.mean(vocal_activity)), 6),
            "max": round(float(np.max(vocal_activity)), 6),
        },
    }
    notes = (
        "Legacy residual rebuild (mode: residual). Other = base - drums - bass - "
        "soft*guitar - soft*piano + FFT-band blending. This path accumulates errors "
        "from each component into `other` (audit finding F3); R7 makes the direct ML "
        "`other` the default. Switch to `mode: direct` to use the ML output directly."
    )
    return named_outputs, metadata, notes


def run_tandem_instrumental_hybrid(request: RunRequest, output_root: Path) -> tuple[dict[str, str], dict[str, object], str]:
    base_path = Path(request.input_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base instrumental input is missing: {base_path}")

    _, bs_seed_path = _resolve_existing_build_from(request, ("bs_resurrected_instrumental_mix", "bs_instrumental_mix"))
    if bs_seed_path is None:
        raise FileNotFoundError("No BS instrumental seed was available for tandem hybrid assembly.")

    _, vocals_path = _resolve_existing_build_from(request, ("vocals", "bs_vocals"))

    base_audio, base_sr = _read_wave_file(base_path)
    base_audio = _ensure_stereo(base_audio)
    target_len = base_audio.shape[1]

    bs_seed_audio, bs_seed_sr = _read_wave_file(bs_seed_path)
    bs_seed_audio = _match_audio(bs_seed_audio, bs_seed_sr, base_sr, target_len)

    raw_bs_audio = None
    raw_bs_path_str = request.build_from_paths.get("bs_instrumental_mix")
    if raw_bs_path_str:
        raw_bs_path = Path(raw_bs_path_str)
        if raw_bs_path.exists() and raw_bs_path != bs_seed_path:
            raw_bs_audio, raw_bs_sr = _read_wave_file(raw_bs_path)
            raw_bs_audio = _match_audio(raw_bs_audio, raw_bs_sr, base_sr, target_len)

    vocals_audio = None
    vocal_activity = np.zeros(target_len, dtype=np.float32)
    if vocals_path is not None:
        vocals_audio, vocals_sr = _read_wave_file(vocals_path)
        vocals_audio = _match_audio(vocals_audio, vocals_sr, base_sr, target_len)
        vocal_activity = _vocal_activity_mask(vocals_audio, base_sr, target_len)

    _, final_vocals_path, final_vocals_audio = _load_matched_build_from_audio(request, ("vocals",), base_sr, target_len)
    _, bs_vocals_path, bs_vocals_audio = _load_matched_build_from_audio(request, ("bs_vocals",), base_sr, target_len)

    profile_name, weights = _resolve_tandem_hybrid_profile(request)
    other_profile_name, other_weights = _resolve_other_guide_profile(request)
    center_profile_name, center_weights = _resolve_center_suppression_profile(request)

    # ---- S2 safety-net ----------------------------------------------------
    # Before blending, check that the BS-Roformer side is actually cleaner of
    # vocals than the Demucs base. If it is *worse* by more than
    # ``bs_overlay_safety_margin_db`` (default 3.0 dB), the seed (or raw) BS
    # contribution is zero-ed out for this track. This is a hysteresis-style
    # guard that prevents the historical "overlay disabled" failure mode where
    # a contaminated BS instrumental was deterministically blended in and
    # dragged ``final.bleed_db_rel_total`` from -30 dB toward -10 dB.
    raw_safety_margin = request.raw_step.get("bs_overlay_safety_margin_db", 3.0)
    try:
        safety_margin_db = (
            float(raw_safety_margin) if raw_safety_margin is not None else 3.0
        )
    except (TypeError, ValueError):
        safety_margin_db = 3.0
    bleed_diagnostics: dict[str, object] = {
        "margin_db": safety_margin_db,
        "vocals_guide_input": str(vocals_path) if vocals_path else None,
    }
    overrides_applied: dict[str, object] = {}
    base_bleed_db: float | None = None
    seed_bleed_db: float | None = None
    raw_bs_bleed_db: float | None = None
    if vocals_audio is not None and safety_margin_db > 0.0:
        base_bleed_db = _vocal_contamination_db(base_audio, vocals_audio, base_sr)
        seed_bleed_db = _vocal_contamination_db(bs_seed_audio, vocals_audio, base_sr)
        bleed_diagnostics["base_bleed_db"] = round(base_bleed_db, 3)
        bleed_diagnostics["bs_seed_bleed_db"] = round(seed_bleed_db, 3)
        if (
            np.isfinite(base_bleed_db)
            and np.isfinite(seed_bleed_db)
            and (seed_bleed_db - base_bleed_db) > safety_margin_db
        ):
            overrides_applied["bs_seed_weight"] = {
                "from": weights["seed_weight"],
                "to": 0.0,
                "reason": (
                    f"bs_seed_bleed={seed_bleed_db:.2f}dB worse than "
                    f"base_bleed={base_bleed_db:.2f}dB by >{safety_margin_db:.2f}dB"
                ),
            }
            weights["seed_weight"] = 0.0
        if raw_bs_audio is not None:
            raw_bs_bleed_db = _vocal_contamination_db(raw_bs_audio, vocals_audio, base_sr)
            bleed_diagnostics["bs_raw_bleed_db"] = round(raw_bs_bleed_db, 3)
            if (
                np.isfinite(base_bleed_db)
                and np.isfinite(raw_bs_bleed_db)
                and (raw_bs_bleed_db - base_bleed_db) > safety_margin_db
            ):
                overrides_applied["bs_raw_weight"] = {
                    "from": weights["raw_weight"],
                    "to": 0.0,
                    "reason": (
                        f"bs_raw_bleed={raw_bs_bleed_db:.2f}dB worse than "
                        f"base_bleed={base_bleed_db:.2f}dB by >{safety_margin_db:.2f}dB"
                    ),
                }
                weights["raw_weight"] = 0.0
    bleed_diagnostics["overrides_applied"] = overrides_applied
    # -----------------------------------------------------------------------

    detail_delta = weights["seed_weight"] * (bs_seed_audio - base_audio)
    if raw_bs_audio is not None:
        detail_delta = detail_delta + (weights["raw_weight"] * (raw_bs_audio - base_audio))

    high_detail = _fft_band(detail_delta, base_sr, 2200.0, min(14000.0, base_sr * 0.48))
    body_detail = _fft_band(detail_delta, base_sr, 420.0, 2200.0)

    hi_mid, hi_side = _mid_side(high_detail)
    body_mid, body_side = _mid_side(body_detail)

    suppress_mid = 1.0 - (weights["vocal_mid_suppression"] * vocal_activity)
    suppress_side = 1.0 - (weights["vocal_side_suppression"] * vocal_activity)

    overlay = _from_mid_side(
        (hi_mid * weights["hi_mid"] + body_mid * weights["body_mid"]) * suppress_mid,
        (hi_side * weights["hi_side"] + body_side * weights["body_side"]) * suppress_side,
    )

    other_key, other_path, other_audio = _load_matched_build_from_audio(
        request,
        ("other", "other_residual", "demucs_other", "bs_other"),
        base_sr,
        target_len,
    )
    other_activity = np.zeros(target_len, dtype=np.float32)
    if other_audio is not None and other_path is not None and other_key is not None and other_profile_name != "off":
        other_activity = _activity_mask(other_audio, base_sr, target_len)
        other_high = _fft_band(other_audio, base_sr, 1800.0, min(14000.0, base_sr * 0.48))
        other_body = _fft_band(other_audio, base_sr, 320.0, 2200.0)
        other_hi_mid, other_hi_side = _mid_side(other_high)
        other_body_mid, other_body_side = _mid_side(other_body)

        other_mask = other_weights["activity_floor"] + ((1.0 - other_weights["activity_floor"]) * other_activity)
        other_mid_mask = other_mask * (1.0 - (other_weights["mid_vocal_suppression"] * vocal_activity))
        other_side_mask = other_mask * (1.0 - (other_weights["side_vocal_suppression"] * vocal_activity))
        overlay = overlay + _from_mid_side(
            (other_hi_mid * other_weights["hi_mid"] + other_body_mid * other_weights["body_mid"]) * other_mid_mask,
            (other_hi_side * other_weights["hi_side"] + other_body_side * other_weights["body_side"]) * other_side_mask,
        )

    hybrid = base_audio + overlay

    center_guide = _blend_audio_guides(
        (final_vocals_audio, center_weights["guide_final_weight"]),
        (bs_vocals_audio, center_weights["guide_bs_weight"]),
    )
    if center_profile_name != "off":
        hybrid = _apply_center_vocal_suppression(hybrid, center_guide, vocal_activity, base_sr, center_weights)

    hybrid = _peak_normalize(hybrid)

    output_path = output_path_for(request, output_root)
    _write_wave_file(output_path, hybrid, base_sr)
    named_outputs = {request.target_stem or request.task: str(output_path)}
    metadata = {
        "mode": "demucs_base_plus_bs_guided_overlay",
        "hybrid_profile": profile_name,
        "base_input": str(base_path),
        "bs_seed_input": str(bs_seed_path),
        "raw_bs_input": str(Path(raw_bs_path_str)) if raw_bs_path_str else None,
        "vocal_mask_input": str(vocals_path) if vocals_path else None,
        "sample_rate": base_sr,
        "target_len_samples": target_len,
        "weights": {
            "hi_mid": weights["hi_mid"],
            "hi_side": weights["hi_side"],
            "body_mid": weights["body_mid"],
            "body_side": weights["body_side"],
            # The applied weights here reflect S2 safety-net overrides.
            # ``bs_seed_weight`` is whatever was actually multiplied into
            # ``bs_seed - base`` (0.0 if the safety-net disabled the seed).
            # ``bs_raw_weight`` is 0.0 when no ``raw_bs`` artifact was wired
            # in, otherwise the actually applied weight (post-override).
            "bs_seed_weight": weights["seed_weight"],
            "bs_raw_weight": weights["raw_weight"] if raw_bs_audio is not None else 0.0,
            "vocal_mid_suppression": weights["vocal_mid_suppression"],
            "vocal_side_suppression": weights["vocal_side_suppression"],
        },
        "bs_overlay_safety_net": bleed_diagnostics,
        "bs_overlay_disabled_reason": (
            "; ".join(
                f"{slot}: {info['reason']}"
                for slot, info in overrides_applied.items()
                if isinstance(info, dict) and "reason" in info
            )
            if overrides_applied
            else None
        ),
        "other_detail": {
            "profile": other_profile_name,
            "input": str(other_path) if other_path else None,
            "artifact_key": other_key,
            "weights": {
                "hi_mid": other_weights["hi_mid"],
                "hi_side": other_weights["hi_side"],
                "body_mid": other_weights["body_mid"],
                "body_side": other_weights["body_side"],
                "mid_vocal_suppression": other_weights["mid_vocal_suppression"],
                "side_vocal_suppression": other_weights["side_vocal_suppression"],
                "activity_floor": other_weights["activity_floor"],
            },
            "activity": {
                "mean": round(float(np.mean(other_activity)), 6),
                "max": round(float(np.max(other_activity)), 6),
            },
        },
        "center_suppression": {
            "profile": center_profile_name,
            "guide_final_input": str(final_vocals_path) if final_vocals_path else None,
            "guide_bs_input": str(bs_vocals_path) if bs_vocals_path else None,
            "guide_used": center_guide is not None,
            "weights": {
                "guide_final_weight": center_weights["guide_final_weight"],
                "guide_bs_weight": center_weights["guide_bs_weight"],
                "body_mid_suppress": center_weights["body_mid_suppress"],
                "presence_mid_suppress": center_weights["presence_mid_suppress"],
                "air_mid_suppress": center_weights["air_mid_suppress"],
                "presence_side_suppress": center_weights["presence_side_suppress"],
                "guide_subtract": center_weights["guide_subtract"],
                "activity_floor": center_weights["activity_floor"],
            },
        },
        "vocal_activity": {
            "mean": round(float(np.mean(vocal_activity)), 6),
            "max": round(float(np.max(vocal_activity)), 6),
        },
    }
    notes_parts = [
        "Built instrumental hybrid with Demucs as the stable base and BS as a vocal-aware side/detail enhancer.",
        "This is intentionally not a full BS replacement path.",
    ]
    if overrides_applied:
        for slot, info in overrides_applied.items():
            if isinstance(info, dict) and "reason" in info:
                notes_parts.append(f"safety_net[{slot}]: {info['reason']}")
    notes = " ".join(notes_parts)
    return named_outputs, metadata, notes


# ---------------------------------------------------------------------------
# Vocal-mask cleanup (soft Wiener filter, post C from Phase-1 of S6 plan).
# ---------------------------------------------------------------------------
# Profiles for ``run_vocal_mask_cleanup``. The mask is computed per
# time-frequency bin as
#
#   M(t, f) = clip( I(t,f)^2 / (I(t,f)^2 + lambda * V(t,f)^2), floor, 1.0 )
#
# where ``I`` is the dirty instrumental and ``V`` is the vocal guide.  Larger
# ``lambda`` cuts vocal energy more aggressively, ``floor`` is the minimum
# attenuation we ever apply (prevents audible spectral holes).  Optional
# Hann-style smoothing across time and frequency suppresses musical-noise
# artefacts at the cost of a slightly softer notch.
VOCAL_MASK_PROFILES: dict[str, dict[str, float]] = {
    "off": {
        "lambda": 0.0,
        "floor": 1.0,
        "smoothing_frames": 0.0,
        "smoothing_bins": 0.0,
        "n_fft": 4096.0,
        "hop": 1024.0,
    },
    # Conservative: ~12 dB max attenuation per bin, gentle smoothing. Use this
    # when the vocal stem is noisy or the instrumental has a lot of legitimate
    # energy in the speech band (sax, lead synth around 800-2000 Hz).
    "gentle": {
        "lambda": 0.5,
        "floor": 0.20,
        "smoothing_frames": 3.0,
        "smoothing_bins": 2.0,
        "n_fft": 4096.0,
        "hop": 1024.0,
    },
    # Default: ~20 dB max attenuation per bin (floor 0.10), 3-frame / 2-bin
    # smoothing.  Balances vocal removal against musical-noise risk on the
    # validation tracks (Nine Thou et al.).
    "balanced": {
        "lambda": 1.0,
        "floor": 0.10,
        "smoothing_frames": 3.0,
        "smoothing_bins": 2.0,
        "n_fft": 4096.0,
        "hop": 1024.0,
    },
    # Aggressive: deeper notch (~30 dB max), wider smoothing.  For tracks
    # with very clean vocal guides where every bit of bleed is annoying.
    "aggressive": {
        "lambda": 2.0,
        "floor": 0.03,
        "smoothing_frames": 5.0,
        "smoothing_bins": 3.0,
        "n_fft": 4096.0,
        "hop": 1024.0,
    },
}


def _resolve_vocal_mask_profile(request: RunRequest) -> tuple[str, dict[str, float]]:
    raw_profile = str(
        request.raw_step.get("vocal_mask_profile", "balanced") or "balanced"
    ).strip().lower()
    if raw_profile not in VOCAL_MASK_PROFILES:
        raw_profile = "balanced"
    resolved = dict(VOCAL_MASK_PROFILES[raw_profile])
    overrides = request.raw_step.get("vocal_mask_weights")
    if isinstance(overrides, dict):
        for key in resolved:
            if key in overrides:
                try:
                    resolved[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
    return raw_profile, resolved


def _smooth_2d_box(
    matrix: np.ndarray, time_radius: int, freq_radius: int
) -> np.ndarray:
    """Tiny separable box-smoother along the last two axes.

    Operates on shape ``(channels, freq, time)`` (the layout produced by
    :func:`_stft`) and returns the same shape.  Radius ``0`` short-circuits
    to a no-op.  Implemented via cumulative sums so it is O(N) and avoids a
    scipy dependency at runtime.
    """
    if matrix.ndim != 3:
        raise ValueError(f"_smooth_2d_box expects (C, F, T); got shape {matrix.shape}")
    out = matrix
    if freq_radius > 0:
        f_axis = 1
        cs = np.cumsum(out, axis=f_axis, dtype=np.float64)
        zero_pad = np.zeros(
            (out.shape[0], 1, out.shape[2]), dtype=cs.dtype
        )
        cs = np.concatenate([zero_pad, cs], axis=f_axis)
        n_freq = out.shape[f_axis]
        idx = np.arange(n_freq)
        lo = np.maximum(idx - freq_radius, 0)
        hi = np.minimum(idx + freq_radius + 1, n_freq)
        sums = cs[:, hi, :] - cs[:, lo, :]
        counts = (hi - lo).astype(np.float64).reshape(1, -1, 1)
        out = (sums / np.maximum(counts, 1.0)).astype(matrix.dtype, copy=False)
    if time_radius > 0:
        t_axis = 2
        cs = np.cumsum(out, axis=t_axis, dtype=np.float64)
        zero_pad = np.zeros(
            (out.shape[0], out.shape[1], 1), dtype=cs.dtype
        )
        cs = np.concatenate([zero_pad, cs], axis=t_axis)
        n_time = out.shape[t_axis]
        idx = np.arange(n_time)
        lo = np.maximum(idx - time_radius, 0)
        hi = np.minimum(idx + time_radius + 1, n_time)
        sums = cs[:, :, hi] - cs[:, :, lo]
        counts = (hi - lo).astype(np.float64).reshape(1, 1, -1)
        out = (sums / np.maximum(counts, 1.0)).astype(matrix.dtype, copy=False)
    return out


def run_vocal_mask_cleanup(
    request: RunRequest, output_root: Path
) -> tuple[dict[str, str], dict[str, object], str]:
    """Soft Wiener vocal-mask post-filter for the dirty instrumental.

    Reads:
      - ``request.input_path`` -- the (post-tandem, post-presence-lift) dirty
        ``instrumental_mix.wav`` we want to clean.
      - ``request.build_from_paths["vocals"]`` -- the matching
        ``vocals.wav`` produced by the same pipeline; it is treated as the
        per-bin vocal-energy *guide*, not subtracted in the time domain.

    For each STFT bin ``(t, f)`` we compute
    ``M = clip( |I|^2 / (|I|^2 + lambda * |V|^2), floor, 1.0 )``
    and apply ``I_clean = I * M``.  The mask is per-bin, so a vocal "ah"
    around 800 Hz only attenuates 800 Hz; a 60 Hz kick or an 8 kHz hi-hat
    in the same frame is **untouched**.  This is what makes the filter
    audibly different from the broadband ``vocal_mid_suppression`` /
    ``center_hard`` ducking it replaces: there is no global volume
    modulation, so no audible "pumping" on vocal moments.

    The ``floor`` parameter caps the deepest notch (default 0.10 -> -20 dB)
    so the cleaned instrumental never has true spectral holes.  Box-style
    smoothing across time and frequency optionally suppresses musical-noise
    artefacts at the cost of a slightly softer notch.
    """
    base_path = Path(request.input_path)
    if not base_path.exists():
        raise FileNotFoundError(
            f"Vocal-mask cleanup instrumental input is missing: {base_path}"
        )

    vocals_path_str = (
        request.build_from_paths.get("vocals")
        or request.build_from_paths.get("vocal_mask")
    )
    if not vocals_path_str:
        raise FileNotFoundError(
            "Vocal-mask cleanup requires build_from_paths['vocals'] (or 'vocal_mask') "
            "pointing to the vocal stem to use as a per-bin energy guide."
        )
    vocals_path = Path(vocals_path_str)
    if not vocals_path.exists():
        raise FileNotFoundError(
            f"Vocal-mask cleanup vocal guide is missing: {vocals_path}"
        )

    instr_audio, base_sr = _read_wave_file(base_path)
    instr_audio = _ensure_stereo(instr_audio)
    target_len = instr_audio.shape[1]

    vocals_audio, vocals_sr = _read_wave_file(vocals_path)
    vocals_audio = _match_audio(vocals_audio, vocals_sr, base_sr, target_len)
    vocals_audio = _ensure_stereo(vocals_audio)

    profile_name, weights = _resolve_vocal_mask_profile(request)
    lam = float(weights["lambda"])
    floor = float(np.clip(weights["floor"], 0.0, 1.0))
    n_fft = int(weights["n_fft"])
    hop = int(weights["hop"])
    smoothing_frames = int(max(0, round(weights["smoothing_frames"])))
    smoothing_bins = int(max(0, round(weights["smoothing_bins"])))

    target_stem = request.target_stem or "instrumental_mix"

    if lam <= 0.0:
        target_path = output_path_for_stem(request, output_root, target_stem)
        _write_wave_file(target_path, instr_audio, base_sr)
        metadata = {
            "mode": "vocal_mask_cleanup",
            "vocal_mask_profile": profile_name,
            "weights": dict(weights),
            "instrumental_input": str(base_path),
            "vocals_guide_input": str(vocals_path),
            "sample_rate": base_sr,
            "target_len_samples": target_len,
            "passthrough": True,
        }
        notes = (
            f"Vocal-mask cleanup profile '{profile_name}' is a passthrough "
            "(lambda <= 0); instrumental copied through unchanged."
        )
        return {target_stem: str(target_path)}, metadata, notes

    instr_spec, _ = _stft(instr_audio, n_fft=n_fft, hop=hop)
    vocals_spec, _ = _stft(vocals_audio, n_fft=n_fft, hop=hop)

    instr_power = (np.abs(instr_spec).astype(np.float32)) ** 2
    vocals_power = (np.abs(vocals_spec).astype(np.float32)) ** 2

    if smoothing_frames > 0 or smoothing_bins > 0:
        instr_power = _smooth_2d_box(
            instr_power, time_radius=smoothing_frames, freq_radius=smoothing_bins
        )
        vocals_power = _smooth_2d_box(
            vocals_power, time_radius=smoothing_frames, freq_radius=smoothing_bins
        )

    denom = instr_power + lam * vocals_power
    raw_mask = instr_power / np.maximum(denom, 1e-12)
    mask = np.clip(raw_mask, floor, 1.0).astype(np.float32, copy=False)

    cleaned_spec = instr_spec * mask.astype(np.complex64, copy=False)
    cleaned_audio = _istft(
        cleaned_spec, n_fft=n_fft, hop=hop, target_len=target_len
    )
    cleaned_audio = _ensure_stereo(cleaned_audio)

    input_rms = _rms(instr_audio)
    output_rms = _rms(cleaned_audio)

    target_path = output_path_for_stem(request, output_root, target_stem)
    _write_wave_file(target_path, cleaned_audio, base_sr)

    instr_energy = float(np.mean(instr_power))
    vocals_energy = float(np.mean(vocals_power))
    bins_at_floor_ratio = float(np.mean(mask <= floor + 1e-6))
    metadata = {
        "mode": "vocal_mask_cleanup",
        "vocal_mask_profile": profile_name,
        "weights": dict(weights),
        "instrumental_input": str(base_path),
        "vocals_guide_input": str(vocals_path),
        "sample_rate": base_sr,
        "target_len_samples": target_len,
        "mask_stats": {
            "mean_mask_value": round(float(np.mean(mask)), 6),
            "median_mask_value": round(float(np.median(mask)), 6),
            "min_mask_value": round(float(np.min(mask)), 6),
            "bins_at_floor_ratio": round(bins_at_floor_ratio, 6),
        },
        "energy_stats": {
            "instr_input_rms": round(input_rms, 6),
            "instr_output_rms": round(output_rms, 6),
            "instr_input_mean_power": round(instr_energy, 6),
            "vocals_guide_mean_power": round(vocals_energy, 6),
        },
    }
    notes = (
        f"Vocal-mask cleanup '{profile_name}' (lambda={lam:.2f}, floor={floor:.2f}, "
        f"smooth t={smoothing_frames} f={smoothing_bins}, n_fft={n_fft}, hop={hop}) "
        f"applied per-bin Wiener mask using vocals guide. mean_mask={metadata['mask_stats']['mean_mask_value']:.3f}, "
        f"bins_at_floor={bins_at_floor_ratio:.3f}. Phase-1 of the S6 anti-ducking roadmap."
    )
    return {target_stem: str(target_path)}, metadata, notes


def _resolve_wiener_profile(request: RunRequest) -> tuple[str, dict[str, float]]:
    raw_profile = str(request.raw_step.get("wiener_profile", "soft") or "soft").strip().lower()
    if raw_profile not in WIENER_PROFILES:
        raw_profile = "soft"
    resolved = dict(WIENER_PROFILES[raw_profile])
    overrides = request.raw_step.get("wiener_weights")
    if isinstance(overrides, dict):
        for key in resolved:
            if key in overrides:
                try:
                    if key in {"n_fft", "hop"}:
                        resolved[key] = int(overrides[key])
                    else:
                        resolved[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
    return raw_profile, resolved


def run_wiener_postfilter(
    request: RunRequest, output_root: Path
) -> tuple[dict[str, str], dict[str, object], str]:
    """R8: soft-mask multichannel Wiener post-filter for ML separator outputs.

    For each candidate stem in ``build_from_paths`` we compute the per-channel
    STFT magnitude raised to ``alpha``. The Wiener mask for stem ``i`` is then
    ``M_i = (|S_i|^a + floor) / (sum_j |S_j|^a + floor * N)`` and applied to
    the original mixture STFT. This redistributes the leakage between stems
    rather than letting each estimate keep its own hard-decision artefacts.

    Optionally the result is cross-faded with the original ML output via
    ``mix_blend`` so transients are not over-smoothed; ``mix_blend = 0`` means
    fully Wiener-filtered, ``mix_blend = 1`` is a passthrough (debug only).
    """
    base_path = Path(request.input_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Wiener post-filter mixture is missing: {base_path}")

    mix, base_sr = _read_wave_file(base_path)
    mix = _ensure_stereo(mix)
    target_len = mix.shape[1]

    profile_name, weights = _resolve_wiener_profile(request)
    n_fft = int(weights["n_fft"])
    hop = int(weights["hop"])
    alpha = float(weights["alpha"])
    spectral_floor = float(weights["spectral_floor"])
    mix_blend = float(np.clip(weights["mix_blend"], 0.0, 1.0))

    declared_stems = list(request.build_from_paths.keys())
    if not declared_stems:
        raise FileNotFoundError(
            "Wiener post-filter requires at least one stem in build_from_paths."
        )

    raw_stems: dict[str, np.ndarray] = {}
    stem_paths: dict[str, str] = {}
    for stem_name in declared_stems:
        raw_path = request.build_from_paths.get(stem_name)
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        audio, sr = _read_wave_file(path)
        raw_stems[stem_name] = _match_audio(audio, sr, base_sr, target_len)
        stem_paths[stem_name] = str(path)

    if not raw_stems:
        raise FileNotFoundError(
            "Wiener post-filter requires at least one resolvable stem in build_from_paths."
        )

    # If alpha == 0 (i.e. profile == "off") we just copy each input through;
    # this keeps the wiener_postfilter task usable as a no-op for A/B work.
    if alpha <= 0.0:
        named_outputs = {}
        for stem_name, audio in raw_stems.items():
            target_path = output_path_for_stem(request, output_root, stem_name)
            _write_wave_file(target_path, audio, base_sr)
            named_outputs[stem_name] = str(target_path)
        metadata = {
            "mode": "wiener_postfilter",
            "wiener_profile": profile_name,
            "weights": {key: weights[key] for key in weights},
            "stem_inputs": stem_paths,
            "sample_rate": base_sr,
            "target_len_samples": target_len,
            "passthrough": True,
        }
        notes = "Wiener profile 'off' - inputs copied through without filtering."
        return named_outputs, metadata, notes

    mix_spec, _ = _stft(mix, n_fft=n_fft, hop=hop)

    stem_specs: dict[str, np.ndarray] = {}
    stem_powers: dict[str, np.ndarray] = {}
    for stem_name, audio in raw_stems.items():
        spec, _ = _stft(audio, n_fft=n_fft, hop=hop)
        stem_specs[stem_name] = spec
        magnitude = np.abs(spec).astype(np.float32, copy=False)
        stem_powers[stem_name] = np.power(magnitude, alpha, dtype=np.float32)

    n_stems = len(stem_powers)
    denom = np.zeros_like(next(iter(stem_powers.values())))
    for power in stem_powers.values():
        denom = denom + power
    denom = denom + (spectral_floor * n_stems)

    named_outputs: dict[str, str] = {}
    stem_metadata: dict[str, dict[str, float]] = {}

    for stem_name in raw_stems:
        numerator = stem_powers[stem_name] + spectral_floor
        mask = numerator / np.maximum(denom, 1e-12)

        wiener_spec = mix_spec * mask.astype(np.complex64, copy=False)
        wiener_audio = _istft(wiener_spec, n_fft=n_fft, hop=hop, target_len=target_len)
        wiener_audio = _ensure_stereo(wiener_audio)

        if mix_blend > 0.0:
            blended = (1.0 - mix_blend) * wiener_audio + mix_blend * raw_stems[stem_name]
        else:
            blended = wiener_audio

        blended = _peak_normalize(blended)
        target_path = output_path_for_stem(request, output_root, stem_name)
        _write_wave_file(target_path, blended, base_sr)
        named_outputs[stem_name] = str(target_path)

        stem_metadata[stem_name] = {
            "input_path": stem_paths[stem_name],
            "mean_mask_value": round(float(np.mean(mask)), 6),
            "median_mask_value": round(float(np.median(mask)), 6),
            "input_rms": round(_rms(raw_stems[stem_name]), 6),
            "output_rms": round(_rms(blended), 6),
        }

    metadata = {
        "mode": "wiener_postfilter",
        "wiener_profile": profile_name,
        "weights": {key: weights[key] for key in weights},
        "sample_rate": base_sr,
        "target_len_samples": target_len,
        "n_stems": n_stems,
        "stems": stem_metadata,
    }
    notes = (
        f"Wiener post-filter (alpha={alpha}, n_fft={n_fft}, hop={hop}, "
        f"floor={spectral_floor}, mix_blend={mix_blend}) applied across {n_stems} stems. "
        "Soft mask redistributes spectral leakage between estimates instead of letting each "
        "stem keep hard-decision artefacts. R8 of the tandem-improvement roadmap."
    )
    return named_outputs, metadata, notes


def _resolve_stem_ensemble_profile(request: RunRequest) -> tuple[str, dict[str, object]]:
    raw_profile = str(request.raw_step.get("ensemble_profile", "median_fft") or "median_fft").strip().lower()
    if raw_profile not in STEM_ENSEMBLE_PROFILES:
        raw_profile = "median_fft"
    resolved: dict[str, object] = dict(STEM_ENSEMBLE_PROFILES[raw_profile])

    explicit_method = request.raw_step.get("ensemble_method")
    if isinstance(explicit_method, str) and explicit_method.strip().lower() in STEM_ENSEMBLE_METHODS:
        resolved["method"] = explicit_method.strip().lower()
        resolved["use_mix_phase"] = resolved["method"] != "min_mag"

    overrides = request.raw_step.get("ensemble_weights")
    if isinstance(overrides, dict):
        for key in list(resolved.keys()):
            if key in overrides:
                try:
                    if key in {"n_fft", "hop"}:
                        resolved[key] = int(overrides[key])
                    elif key == "use_mix_phase":
                        resolved[key] = bool(overrides[key])
                    elif key == "method":
                        cand = str(overrides[key]).strip().lower()
                        if cand in STEM_ENSEMBLE_METHODS:
                            resolved[key] = cand
                    else:
                        resolved[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
    return raw_profile, resolved


def run_stem_ensemble(
    request: RunRequest, output_root: Path
) -> tuple[dict[str, str], dict[str, object], str]:
    """R9: per-stem ensemble across multiple separator candidates.

    Each entry in ``build_from_paths`` is a candidate estimate of the SAME
    target stem (e.g. ``{"bs_vocals": ..., "demucs_vocals": ..., "mdx_vocals": ...}``).
    We compute the STFT magnitude of each candidate and combine them per
    bin via the configured method:

      * ``median_fft`` - per-bin median magnitude (robust default)
      * ``min_fft``    - per-bin minimum magnitude (conservative)
      * ``max_fft``    - per-bin maximum magnitude (inclusive)
      * ``avg_fft``    - per-bin mean magnitude (smooth, A/B reference)
      * ``min_mag``    - per-bin min magnitude with the candidate's own phase

    For ``*_fft`` the phase comes from the original mixture (``input_path``)
    by default, which keeps transients aligned. ``min_mag`` keeps the
    selected candidate's phase per bin to mirror what the audio-separator
    package does.

    This is a focused implementation that does not require the heavy
    `audio-separator` PyPI package; that package itself is tracked
    separately as R12 in the roadmap.
    """
    base_path = Path(request.input_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Stem ensemble mixture is missing: {base_path}")

    mix, base_sr = _read_wave_file(base_path)
    mix = _ensure_stereo(mix)
    target_len = mix.shape[1]

    profile_name, settings = _resolve_stem_ensemble_profile(request)
    method = str(settings["method"])
    n_fft = int(settings["n_fft"])
    hop = int(settings["hop"])
    use_mix_phase = bool(settings["use_mix_phase"])
    mix_blend = float(np.clip(settings["mix_blend"], 0.0, 1.0))

    candidates: dict[str, np.ndarray] = {}
    candidate_paths: dict[str, str] = {}
    for label in request.build_from_paths.keys():
        raw_path = request.build_from_paths.get(label)
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        audio, sr = _read_wave_file(path)
        candidates[label] = _match_audio(audio, sr, base_sr, target_len)
        candidate_paths[label] = str(path)

    if not candidates:
        raise FileNotFoundError(
            "Stem ensemble requires at least one resolvable candidate in build_from_paths."
        )

    if len(candidates) == 1:
        only_label, only_audio = next(iter(candidates.items()))
        target_path = output_path_for(request, output_root)
        _write_wave_file(target_path, only_audio, base_sr)
        named_outputs = {request.target_stem or request.task: str(target_path)}
        metadata = {
            "mode": "stem_ensemble",
            "ensemble_profile": profile_name,
            "method": method,
            "candidate_count": 1,
            "candidates": candidate_paths,
            "passthrough_reason": "single_candidate",
            "sample_rate": base_sr,
            "target_len_samples": target_len,
        }
        notes = (
            f"Only one candidate ({only_label}) was wired in; ensemble degenerates to "
            "passthrough. Provide >= 2 candidates in build_from_paths to actually ensemble."
        )
        return named_outputs, metadata, notes

    mix_spec, _ = _stft(mix, n_fft=n_fft, hop=hop)

    cand_specs: list[np.ndarray] = []
    cand_labels: list[str] = []
    for label, audio in candidates.items():
        spec, _ = _stft(audio, n_fft=n_fft, hop=hop)
        cand_specs.append(spec)
        cand_labels.append(label)

    stack = np.stack(cand_specs, axis=0)
    magnitudes = np.abs(stack).astype(np.float32, copy=False)

    if method == "median_fft":
        combined_mag = np.median(magnitudes, axis=0)
    elif method == "min_fft":
        combined_mag = np.min(magnitudes, axis=0)
    elif method == "max_fft":
        combined_mag = np.max(magnitudes, axis=0)
    elif method == "avg_fft":
        combined_mag = np.mean(magnitudes, axis=0)
    elif method == "min_mag":
        idx = np.argmin(magnitudes, axis=0)
        idx_exp = idx[np.newaxis, ...]
        selected_complex = np.take_along_axis(stack, idx_exp, axis=0)[0]
        combined_spec = selected_complex.astype(np.complex64, copy=False)
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")

    if method != "min_mag":
        if use_mix_phase:
            phase = np.exp(1j * np.angle(mix_spec)).astype(np.complex64, copy=False)
        else:
            phase = np.exp(1j * np.angle(stack[0])).astype(np.complex64, copy=False)
        combined_spec = (combined_mag.astype(np.complex64, copy=False) * phase)

    combined_audio = _istft(combined_spec, n_fft=n_fft, hop=hop, target_len=target_len)
    combined_audio = _ensure_stereo(combined_audio)

    if mix_blend > 0.0:
        avg_candidate = np.mean(np.stack(list(candidates.values()), axis=0), axis=0)
        combined_audio = (1.0 - mix_blend) * combined_audio + mix_blend * avg_candidate

    combined_audio = _peak_normalize(combined_audio)
    target_path = output_path_for(request, output_root)
    _write_wave_file(target_path, combined_audio, base_sr)
    named_outputs = {request.target_stem or request.task: str(target_path)}

    candidate_metadata = {
        label: {
            "path": candidate_paths[label],
            "rms": round(_rms(candidates[label]), 6),
        }
        for label in cand_labels
    }
    metadata = {
        "mode": "stem_ensemble",
        "ensemble_profile": profile_name,
        "method": method,
        "n_fft": n_fft,
        "hop": hop,
        "use_mix_phase": use_mix_phase,
        "mix_blend": mix_blend,
        "candidate_count": len(candidates),
        "candidates": candidate_metadata,
        "sample_rate": base_sr,
        "target_len_samples": target_len,
        "output_rms": round(_rms(combined_audio), 6),
    }
    notes = (
        f"Per-stem ensemble across {len(candidates)} candidates via method={method} "
        f"(n_fft={n_fft}, hop={hop}, mix_phase={use_mix_phase}). R9 of the "
        "tandem-improvement roadmap; replaces the BS-only / Demucs-only single-source "
        "path with a real multi-model fusion."
    )
    return named_outputs, metadata, notes


def output_root_for(request: RunRequest) -> Path:
    target = request.target_stem or request.task
    root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def output_path_for(request: RunRequest, output_root: Path) -> Path:
    target = request.target_stem or request.task
    return output_root / f"{target}.wav"


def output_path_for_stem(request: RunRequest, output_root: Path, stem_name: str) -> Path:
    """Multi-stem output helper for fan-out tasks (R8 wiener_postfilter, R9 ensemble).

    Some restoration tasks produce more than one output per invocation. This
    helper just gives each stem its own file inside the per-step output_root,
    distinct from the single-target path returned by ``output_path_for``.
    """
    safe_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem_name) or "stem"
    return output_root / f"{safe_name}.wav"


def resolve_recipe(request: RunRequest) -> str:
    raw = request.raw_step.get("recipe", "") or request.preset or ""
    recipe = str(raw).strip()
    if not recipe:
        # Safer default than the historical `generic_cleanup`: a no-EQ pass that
        # only normalises DC and applies a soft limiter. Aligns with R3 of the
        # tandem-improvement roadmap (avoid amplifying ML residue by accident).
        return "passthrough"
    if recipe not in RECIPE_FILTERS:
        return "passthrough"
    return recipe


def build_command(request: RunRequest, output_root: Path) -> tuple[list[str], str, Path]:
    recipe = resolve_recipe(request)
    output_path = output_path_for(request, output_root)
    filter_chain = RECIPE_FILTERS[recipe]
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        request.input_path,
        "-vn",
        "-af",
        filter_chain,
        # Intermediate stems are kept in 32-bit float to avoid the cumulative
        # 16-bit quantisation error that previously crept in at every stage of
        # the tandem pipeline (restoration -> mix -> ML refinement -> mix again).
        # Cf. R-PCM in the stem-tandem-improvement-roadmap.
        "-c:a",
        "pcm_f32le",
        str(output_path),
    ]
    return command, recipe, output_path


def planned_outputs(request: RunRequest, output_root: Path) -> dict[str, str]:
    target = request.target_stem or request.task
    return {target: str(output_path_for(request, output_root))}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "restoration-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "restoration-worker",
        "tasks": [
            "dereverb_cleanup",
            "harmonic_smoothing",
            "artifact_repair",
            "presence_lift",
            "other_residual_rebuild",
            "tandem_instrumental_hybrid",
            "wiener_postfilter",
            "stem_ensemble",
            "vocal_mask_cleanup",
        ],
        "recipes": sorted(RECIPE_FILTERS.keys()),
        "wiener_profiles": sorted(WIENER_PROFILES.keys()),
        "ensemble_profiles": sorted(STEM_ENSEMBLE_PROFILES.keys()),
        "vocal_mask_profiles": sorted(VOCAL_MASK_PROFILES.keys()),
        "state": "real ffmpeg-based cleanup and presence-lift processing",
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    output_root = output_root_for(request)
    named_outputs = planned_outputs(request, output_root)

    if request.task in {
        "tandem_instrumental_hybrid",
        "other_residual_rebuild",
        "wiener_postfilter",
        "stem_ensemble",
        "vocal_mask_cleanup",
    }:
        command = ["python", request.task, request.input_path]
        metadata = {"output_root": str(output_root), "build_from_paths": request.build_from_paths}
        status = "dry_run_completed"
        notes = f"Dry-run {request.task} assembled."
        if not request.dry_run:
            try:
                if request.task == "tandem_instrumental_hybrid":
                    named_outputs, task_metadata, notes = run_tandem_instrumental_hybrid(request, output_root)
                elif request.task == "wiener_postfilter":
                    named_outputs, task_metadata, notes = run_wiener_postfilter(request, output_root)
                elif request.task == "stem_ensemble":
                    named_outputs, task_metadata, notes = run_stem_ensemble(request, output_root)
                elif request.task == "vocal_mask_cleanup":
                    named_outputs, task_metadata, notes = run_vocal_mask_cleanup(request, output_root)
                else:
                    named_outputs, task_metadata, notes = run_other_residual_rebuild(request, output_root)
                metadata.update(task_metadata)
                status = "completed"
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                metadata["error"] = str(exc)
                notes = f"{request.task} failed."
    else:
        command, recipe, output_path = build_command(request, output_root)
        metadata = {"recipe": recipe, "filter_chain": RECIPE_FILTERS[recipe], "output_root": str(output_root)}
        status = "dry_run_completed"
        notes = "Dry-run ffmpeg restoration command assembled."

        if not request.dry_run:
            completed = subprocess.run(command, capture_output=True, text=True)
            metadata["stdout"] = completed.stdout[-4000:]
            metadata["stderr"] = completed.stderr[-4000:]
            status = "completed" if completed.returncode == 0 and output_path.exists() else "failed"
            notes = "Real ffmpeg restoration completed. This worker is intended to reduce muffling and add mild presence without changing stem routing."

    result = RunResult(
        service="restoration-worker",
        status=status,
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=list(named_outputs.values()),
        named_outputs=named_outputs if status == "completed" else {},
        metadata=metadata,
        notes=notes,
    )
    return result.model_dump()
