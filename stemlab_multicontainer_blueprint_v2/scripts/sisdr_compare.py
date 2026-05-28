"""R14 - SI-SDR scorer for stems against a ground-truth reference set.

Given a directory of estimated stems (the output of one of the
``lyrics_sync_from_track_tandem*`` presets) and a directory of ground-truth
stems with the same file names (typically a MUSDB18-HQ track), computes
SI-SDR per stem in dB. Optionally aggregates across multiple tracks when a
``--manifest`` JSON is provided that lists ``(track_id, estimates_dir,
ground_truth_dir)`` triples.

This is the *real* quality metric used by every recent MSS leaderboard
(MDX, MoisesDB, MUSDB) - much more meaningful than the per-stem
fingerprints used by ``baseline_compare.py`` (R13). Use ``baseline_compare``
to detect *unintended* drift on the same input; use this script to detect
*real* improvement against a known reference.

Definition (Le Roux et al. 2019, "SDR - half-baked or well done?"):

    alpha    = <s_hat, s> / <s, s>
    s_target = alpha * s
    e_noise  = s_hat - s_target
    SI-SDR   = 10 * log10( ||s_target||^2 / ||e_noise||^2 )

Usage:
    py scripts/sisdr_compare.py \\
        --estimates path/to/preset_output \\
        --reference path/to/musdb_track \\
        [--out report.json] [--stems vocals,drums,bass,other]

    py scripts/sisdr_compare.py \\
        --manifest manifest.json --out aggregate.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Iterable

import numpy as np


def _read_wave(path: Path) -> tuple[np.ndarray, int]:
    """Lightweight WAV reader. Supports PCM (16/24/32-bit) and IEEE_FLOAT (32/64-bit).

    Bypasses Python's ``wave`` module, which raises
    ``wave.Error: unknown format: 3`` for IEEE_FLOAT files (the format that
    every worker writes via ``_write_float_wav``). Parses RIFF/WAVE chunks
    directly so PCM and IEEE_FLOAT are both readable.
    """
    import struct as _struct

    if not path.exists():
        raise FileNotFoundError(path)

    raw_format_code: int | None = None
    channels: int | None = None
    sr: int | None = None
    sample_width: int | None = None
    raw: bytes | None = None

    with path.open("rb") as handle:
        riff = handle.read(12)
        if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_id = header[0:4]
            chunk_size = _struct.unpack("<I", header[4:8])[0]
            chunk_data = handle.read(chunk_size)
            if chunk_size % 2 == 1:
                handle.read(1)
            if chunk_id == b"fmt ":
                if len(chunk_data) < 16:
                    raise ValueError(f"fmt chunk too short in {path}")
                raw_format_code = _struct.unpack("<H", chunk_data[0:2])[0]
                channels = _struct.unpack("<H", chunk_data[2:4])[0]
                sr = _struct.unpack("<I", chunk_data[4:8])[0]
                bits_per_sample = _struct.unpack("<H", chunk_data[14:16])[0]
                sample_width = bits_per_sample // 8
                if raw_format_code == 0xFFFE and len(chunk_data) >= 40:
                    raw_format_code = _struct.unpack("<H", chunk_data[24:26])[0]
            elif chunk_id == b"data":
                raw = chunk_data

    if (
        raw_format_code is None or channels is None or sr is None
        or sample_width is None or raw is None
    ):
        raise ValueError(f"Incomplete WAV header in {path}")

    if raw_format_code == 3:
        if sample_width == 4:
            audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif sample_width == 8:
            audio = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"Unsupported IEEE_FLOAT sample width: {sample_width} bytes ({path})")
    elif raw_format_code == 1:
        if sample_width == 2:
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
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
            audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float((1 << 31) - 1)
        else:
            raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes ({path})")
    else:
        raise ValueError(
            f"Unsupported WAV format code {raw_format_code} (only 1=PCM, 3=IEEE_FLOAT) for {path}"
        )

    audio = audio.reshape(-1, channels).T
    return audio, sr


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return np.mean(audio, axis=0).astype(np.float32, copy=False)


def _align_lengths(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(reference.shape[0], estimate.shape[0])
    return reference[:n], estimate[:n]


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-9) -> float:
    """Scale-invariant SDR in dB. Returns ``+inf`` for a perfect match."""
    reference = reference.astype(np.float64, copy=False)
    estimate = estimate.astype(np.float64, copy=False)
    reference, estimate = _align_lengths(reference, estimate)

    ref_energy = float(np.dot(reference, reference))
    if ref_energy <= eps:
        return float("nan")

    alpha = float(np.dot(estimate, reference)) / max(ref_energy, eps)
    s_target = alpha * reference
    e_noise = estimate - s_target

    target_energy = float(np.dot(s_target, s_target))
    noise_energy = float(np.dot(e_noise, e_noise))
    if noise_energy <= eps:
        return float("inf")
    if target_energy <= eps:
        return float("-inf")

    return 10.0 * math.log10(target_energy / noise_energy)


def sdr_db(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-9) -> float:
    """Plain SDR (no scale invariance) for cross-checking."""
    reference, estimate = _align_lengths(
        reference.astype(np.float64, copy=False),
        estimate.astype(np.float64, copy=False),
    )
    ref_energy = float(np.dot(reference, reference))
    if ref_energy <= eps:
        return float("nan")
    diff = estimate - reference
    diff_energy = float(np.dot(diff, diff))
    if diff_energy <= eps:
        return float("inf")
    return 10.0 * math.log10(ref_energy / diff_energy)


def score_pair(reference_path: Path, estimate_path: Path) -> dict[str, float]:
    reference, ref_sr = _read_wave(reference_path)
    estimate, est_sr = _read_wave(estimate_path)
    if ref_sr != est_sr:
        raise ValueError(
            f"Sample rate mismatch: reference={ref_sr} Hz, estimate={est_sr} Hz "
            f"({reference_path.name}). Resample upstream before scoring."
        )
    ref_mono = _to_mono(reference)
    est_mono = _to_mono(estimate)
    return {
        "si_sdr_db": si_sdr_db(ref_mono, est_mono),
        "sdr_db": sdr_db(ref_mono, est_mono),
        "reference_rms": float(np.sqrt(np.mean(np.square(ref_mono)) + 1e-12)),
        "estimate_rms": float(np.sqrt(np.mean(np.square(est_mono)) + 1e-12)),
        "duration_sec": float(min(ref_mono.shape[0], est_mono.shape[0]) / ref_sr),
        "sample_rate": int(ref_sr),
    }


def discover_stems(estimates_dir: Path) -> list[str]:
    return sorted(p.stem for p in estimates_dir.glob("*.wav"))


def score_track(
    estimates_dir: Path,
    reference_dir: Path,
    stem_filter: Iterable[str] | None = None,
) -> dict[str, dict[str, float] | str]:
    estimates_dir = Path(estimates_dir)
    reference_dir = Path(reference_dir)
    estimate_stems = set(discover_stems(estimates_dir))
    if stem_filter is not None:
        wanted = {s.strip() for s in stem_filter if s.strip()}
        estimate_stems &= wanted

    per_stem: dict[str, dict[str, float] | str] = {}
    for stem in sorted(estimate_stems):
        est_path = estimates_dir / f"{stem}.wav"
        ref_path = reference_dir / f"{stem}.wav"
        if not ref_path.exists():
            per_stem[stem] = "missing_reference"
            continue
        try:
            metrics = score_pair(ref_path, est_path)
            per_stem[stem] = metrics
        except Exception as exc:  # noqa: BLE001
            per_stem[stem] = f"error: {exc!r}"
    return per_stem


def aggregate_track(per_stem: dict[str, dict[str, float] | str]) -> dict[str, float | int]:
    valid_si: list[float] = []
    valid_sdr: list[float] = []
    n_missing = 0
    n_error = 0
    for stem, metrics in per_stem.items():
        if isinstance(metrics, str):
            if metrics == "missing_reference":
                n_missing += 1
            else:
                n_error += 1
            continue
        si = metrics.get("si_sdr_db")
        sdr = metrics.get("sdr_db")
        if si is not None and not (math.isnan(si) or math.isinf(si)):
            valid_si.append(float(si))
        if sdr is not None and not (math.isnan(sdr) or math.isinf(sdr)):
            valid_sdr.append(float(sdr))
    return {
        "stems_scored": len(valid_si),
        "stems_missing_reference": n_missing,
        "stems_errored": n_error,
        "mean_si_sdr_db": float(np.mean(valid_si)) if valid_si else float("nan"),
        "median_si_sdr_db": float(np.median(valid_si)) if valid_si else float("nan"),
        "mean_sdr_db": float(np.mean(valid_sdr)) if valid_sdr else float("nan"),
    }


def run_single(args: argparse.Namespace) -> int:
    estimates = Path(args.estimates).resolve()
    reference = Path(args.reference).resolve()
    if not estimates.exists():
        print(f"[ERROR] estimates directory not found: {estimates}", file=sys.stderr)
        return 2
    if not reference.exists():
        print(f"[ERROR] reference directory not found: {reference}", file=sys.stderr)
        return 2

    stem_filter = args.stems.split(",") if args.stems else None
    per_stem = score_track(estimates, reference, stem_filter=stem_filter)
    aggregate = aggregate_track(per_stem)

    report = {
        "estimates_dir": str(estimates),
        "reference_dir": str(reference),
        "stem_filter": stem_filter,
        "per_stem": per_stem,
        "aggregate": aggregate,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] Report written to {args.out}")
    print(json.dumps(aggregate, indent=2))
    return 0


def run_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracks = manifest.get("tracks", [])
    if not tracks:
        print("[ERROR] manifest has no `tracks` array", file=sys.stderr)
        return 2

    stem_filter = args.stems.split(",") if args.stems else None
    track_reports: dict[str, dict[str, object]] = {}
    grand_si: list[float] = []
    grand_sdr: list[float] = []

    for track in tracks:
        track_id = str(track.get("track_id", track.get("id") or ""))
        estimates = Path(track["estimates"]).resolve()
        reference = Path(track["reference"]).resolve()
        if not estimates.exists() or not reference.exists():
            track_reports[track_id] = {"status": "missing", "estimates_exists": estimates.exists(), "reference_exists": reference.exists()}
            continue
        per_stem = score_track(estimates, reference, stem_filter=stem_filter)
        aggregate = aggregate_track(per_stem)
        track_reports[track_id] = {"per_stem": per_stem, "aggregate": aggregate}
        if not math.isnan(aggregate["mean_si_sdr_db"]):
            grand_si.append(aggregate["mean_si_sdr_db"])
        if not math.isnan(aggregate["mean_sdr_db"]):
            grand_sdr.append(aggregate["mean_sdr_db"])

    grand_total = {
        "tracks": len(tracks),
        "tracks_scored": len(grand_si),
        "grand_mean_si_sdr_db": float(np.mean(grand_si)) if grand_si else float("nan"),
        "grand_median_si_sdr_db": float(np.median(grand_si)) if grand_si else float("nan"),
        "grand_mean_sdr_db": float(np.mean(grand_sdr)) if grand_sdr else float("nan"),
    }

    report = {
        "manifest": str(manifest_path),
        "stem_filter": stem_filter,
        "tracks": track_reports,
        "grand_total": grand_total,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] Report written to {args.out}")
    print(json.dumps(grand_total, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SI-SDR scorer for separation outputs (R14).")
    parser.add_argument("--estimates", help="Directory containing estimated stems (e.g. preset output dir).")
    parser.add_argument("--reference", help="Directory containing ground-truth stems with matching file names.")
    parser.add_argument("--manifest", help="Optional JSON file: { tracks: [ {track_id, estimates, reference}, ... ] }")
    parser.add_argument("--stems", help="Comma-separated stems to score (default: all wavs in estimates dir).")
    parser.add_argument("--out", help="Optional path to write the JSON report.")
    args = parser.parse_args(argv)

    if args.manifest:
        return run_manifest(args)
    if args.estimates and args.reference:
        return run_single(args)
    parser.error("Either --manifest or both --estimates and --reference are required.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
