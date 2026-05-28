"""Compute a deterministic baseline snapshot for a job's stem outputs.

Per-WAV metrics: sha256(byte content), duration_sec, sample_rate, channels,
sample_width, peak_abs, peak_position_sec, rms, spectral_centroid_mean_hz,
spectral_centroid_std_hz, spectral_flatness_mean.

Usage:
    py scripts/baseline_snapshot.py --job-id <id> [--root <path>] \\
        [--out <baseline.json>] [--label <name>]

If --out is not given, the snapshot is written to
    <root>/data/outputs/<job_id>/baseline_snapshot.json

Designed for accuracy-roadmap regression checks (R13 in
stem-tandem-improvement-roadmap). Pair with baseline_compare.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

AUDIO_EXTS = {".wav"}
FFT_WINDOW = 2048
FFT_HOP = 1024


def read_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            triplets[:, 0].astype(np.int32)
            | (triplets[:, 1].astype(np.int32) << 8)
            | (triplets[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        ints = np.where(ints & sign_bit, ints - (1 << 24), ints)
        data = ints.astype(np.float64) / float((1 << 23) - 1)
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    else:
        raise ValueError(f"unsupported wav sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return data, sr, channels, sample_width


def sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _spectral_features(mono: np.ndarray, sr: int) -> tuple[float, float, float]:
    if mono.size < FFT_WINDOW:
        return 0.0, 0.0, 0.0

    window = np.hanning(FFT_WINDOW)
    centroids: list[float] = []
    flatness_vals: list[float] = []
    freqs = np.fft.rfftfreq(FFT_WINDOW, d=1.0 / sr)
    eps = 1e-12
    for start in range(0, mono.size - FFT_WINDOW + 1, FFT_HOP):
        frame = mono[start : start + FFT_WINDOW] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum_sum = float(spectrum.sum())
        if spectrum_sum <= eps:
            continue
        centroid = float((freqs * spectrum).sum() / spectrum_sum)
        centroids.append(centroid)
        positive = spectrum + eps
        log_mean = float(np.exp(np.mean(np.log(positive))))
        arith_mean = float(positive.mean())
        flatness_vals.append(log_mean / arith_mean if arith_mean > 0 else 0.0)

    if not centroids:
        return 0.0, 0.0, 0.0

    centroid_arr = np.asarray(centroids, dtype=np.float64)
    flat_arr = np.asarray(flatness_vals, dtype=np.float64)
    return (
        float(centroid_arr.mean()),
        float(centroid_arr.std()),
        float(flat_arr.mean()),
    )


def analyze_wav(path: Path) -> dict:
    data, sr, channels, sample_width = read_wav(path)
    mono = data.mean(axis=1)
    abs_mono = np.abs(mono)
    peak = float(abs_mono.max()) if abs_mono.size else 0.0
    peak_idx = int(abs_mono.argmax()) if abs_mono.size else 0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    centroid_mean, centroid_std, flatness_mean = _spectral_features(mono, sr)
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "sample_rate": sr,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": int(data.shape[0]),
        "duration_sec": round(data.shape[0] / sr, 6) if sr else 0.0,
        "peak_abs": round(peak, 8),
        "peak_position_sec": round(peak_idx / sr, 6) if sr else 0.0,
        "rms": round(rms, 8),
        "spectral_centroid_mean_hz": round(centroid_mean, 4),
        "spectral_centroid_std_hz": round(centroid_std, 4),
        "spectral_flatness_mean": round(flatness_mean, 8),
    }


def iter_wavs(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            yield path


def build_snapshot(job_outputs_root: Path, *, label: str | None) -> dict:
    entries: dict[str, dict] = {}
    failures: list[dict] = []
    for wav in iter_wavs(job_outputs_root):
        rel = str(wav.relative_to(job_outputs_root)).replace("\\", "/")
        try:
            entries[rel] = analyze_wav(wav)
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": rel, "error": repr(exc)})

    return {
        "schema": "stemlab.baseline.v1",
        "label": label or "",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outputs_root": str(job_outputs_root),
        "stems": entries,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="project root (defaults to the blueprint directory)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="destination JSON path (defaults to data/outputs/<job-id>/baseline_snapshot.json)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="optional human-readable label (e.g. 'pre-R1' or 'post-R1-overlap4')",
    )
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    outputs_root = project_root / "data" / "outputs" / args.job_id
    if not outputs_root.exists():
        print(f"outputs root not found: {outputs_root}")
        return 2

    snapshot = build_snapshot(outputs_root, label=args.label)

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = outputs_root / "baseline_snapshot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    stem_count = len(snapshot["stems"])
    fail_count = len(snapshot["failures"])
    print(f"snapshot written: {out_path}")
    print(f"stems: {stem_count}; failures: {fail_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
