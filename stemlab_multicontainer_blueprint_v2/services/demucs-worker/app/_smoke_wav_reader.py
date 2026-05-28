"""Local smoke test for _parse_wav_header / _read_wave_file.

Re-runs the read path against a freshly written IEEE_FLOAT wav (format=3)
and a 16-bit PCM wav (format=1), proving the demucs-worker can now read
both without raising `wave.Error: unknown format: 3`. Run from the
repo root with `py -m stemlab_multicontainer_blueprint_v2.services.demucs-worker.app._smoke_wav_reader`
or simply `py services/demucs-worker/app/_smoke_wav_reader.py`.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))
from app.main import _read_wave_file, _write_float_wav  # noqa: E402


def write_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    ints = (audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.ascontiguousarray(ints.T).tobytes())


def make_signal(sr: int, dur_sec: float = 0.25) -> np.ndarray:
    t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    right = 0.4 * np.sin(2 * np.pi * 660 * t)
    return np.vstack([left, right]).astype(np.float32)


def main() -> int:
    sr = 44100
    signal = make_signal(sr)

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        float_path = tmp_dir / "ieee_float.wav"
        _write_float_wav(float_path, np.ascontiguousarray(signal.T), 2, sr)
        with float_path.open("rb") as h:
            h.read(20)
            fmt_code = struct.unpack("<H", h.read(2))[0]
        if fmt_code != 3:
            failures.append(f"float file fmt code {fmt_code} (expected 3)")
        try:
            audio, samplerate = _read_wave_file(float_path)
        except Exception as exc:
            failures.append(f"IEEE_FLOAT read raised: {exc!r}")
        else:
            if samplerate != sr:
                failures.append(f"IEEE_FLOAT sr mismatch {samplerate} != {sr}")
            if audio.shape != signal.shape:
                failures.append(f"IEEE_FLOAT shape mismatch {audio.shape} != {signal.shape}")
            err = float(np.max(np.abs(audio.astype(np.float64) - signal.astype(np.float64))))
            if err > 1e-5:
                failures.append(f"IEEE_FLOAT roundtrip max err {err:.2e}")
            print(f"[smoke] IEEE_FLOAT roundtrip OK  shape={audio.shape} max_err={err:.2e}")

        pcm_path = tmp_dir / "pcm16.wav"
        write_pcm16(pcm_path, signal, sr)
        try:
            audio, samplerate = _read_wave_file(pcm_path)
        except Exception as exc:
            failures.append(f"PCM16 read raised: {exc!r}")
        else:
            if samplerate != sr:
                failures.append(f"PCM16 sr mismatch {samplerate} != {sr}")
            if audio.shape != signal.shape:
                failures.append(f"PCM16 shape mismatch {audio.shape} != {signal.shape}")
            quant = 1.0 / 32767.0
            err = float(np.max(np.abs(audio.astype(np.float64) - signal.astype(np.float64))))
            if err > 2 * quant:
                failures.append(f"PCM16 roundtrip max err {err:.2e}")
            print(f"[smoke] PCM16 roundtrip OK     shape={audio.shape} max_err={err:.2e}")

    if failures:
        print("[smoke] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[smoke] all WAV reader cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
