"""Local smoke test for the patched _read_wave_file in restoration-worker.

Generates an IEEE_FLOAT WAV (the same byte layout demucs-worker emits) and a
16-bit PCM WAV, runs them through ``_read_wave_file``, and asserts both
roundtrips succeed without raising ``wave.Error: unknown format: 3``.
"""
from __future__ import annotations

import struct
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))
from app.main import _read_wave_file  # noqa: E402


def write_float_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    raw = audio.astype("<f4", copy=False).tobytes()
    bits = 32
    bytes_per = 4
    block_align = audio.shape[1] * bytes_per
    byte_rate = sr * block_align
    fmt_size = 16
    fact_size = 4
    n_samples = audio.shape[0]
    data_size = len(raw)
    riff_size = 4 + 8 + fmt_size + 8 + fact_size + 8 + data_size
    with path.open("wb") as h:
        h.write(b"RIFF"); h.write(struct.pack("<I", riff_size)); h.write(b"WAVE")
        h.write(b"fmt "); h.write(struct.pack("<I", fmt_size))
        h.write(struct.pack("<H", 3)); h.write(struct.pack("<H", audio.shape[1]))
        h.write(struct.pack("<I", sr)); h.write(struct.pack("<I", byte_rate))
        h.write(struct.pack("<H", block_align)); h.write(struct.pack("<H", bits))
        h.write(b"fact"); h.write(struct.pack("<I", fact_size))
        h.write(struct.pack("<I", n_samples))
        h.write(b"data"); h.write(struct.pack("<I", data_size)); h.write(raw)


def write_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    ints = (audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.ascontiguousarray(ints).tobytes())


def main() -> int:
    sr = 44100
    dur = 0.25
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    right = 0.4 * np.sin(2 * np.pi * 660 * t)
    interleaved = np.vstack([left, right]).T.astype(np.float32)
    expected_shape = (2, interleaved.shape[0])

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        f = tmp_dir / "ieee_float.wav"
        write_float_wav(f, interleaved, sr)
        try:
            audio, samplerate = _read_wave_file(f)
        except Exception as exc:
            failures.append(f"IEEE_FLOAT read raised: {exc!r}")
        else:
            if samplerate != sr:
                failures.append(f"IEEE_FLOAT sr {samplerate} != {sr}")
            if audio.shape != expected_shape:
                failures.append(f"IEEE_FLOAT shape {audio.shape} != {expected_shape}")
            err = float(np.max(np.abs(audio.T.astype(np.float64) - interleaved.astype(np.float64))))
            if err > 1e-5:
                failures.append(f"IEEE_FLOAT roundtrip max err {err:.2e}")
            print(f"[smoke] IEEE_FLOAT ok shape={audio.shape} max_err={err:.2e}")

        p = tmp_dir / "pcm16.wav"
        write_pcm16(p, interleaved, sr)
        try:
            audio, samplerate = _read_wave_file(p)
        except Exception as exc:
            failures.append(f"PCM16 read raised: {exc!r}")
        else:
            if samplerate != sr:
                failures.append(f"PCM16 sr {samplerate} != {sr}")
            if audio.shape != expected_shape:
                failures.append(f"PCM16 shape {audio.shape} != {expected_shape}")
            err = float(np.max(np.abs(audio.T.astype(np.float64) - interleaved.astype(np.float64))))
            if err > 2 / 32767.0:
                failures.append(f"PCM16 roundtrip max err {err:.2e}")
            print(f"[smoke] PCM16     ok shape={audio.shape} max_err={err:.2e}")

    if failures:
        print("[smoke] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[smoke] all WAV reader cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
