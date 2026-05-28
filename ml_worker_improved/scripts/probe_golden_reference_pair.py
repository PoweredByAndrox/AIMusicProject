#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = SCRIPT_ROOT.parent / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_audio(path: Path) -> dict:
    payload = run_json([
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ])
    audio_streams = [
        stream for stream in payload.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise RuntimeError(f"No audio stream found: {path}")

    audio_stream = audio_streams[0]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "format_name": payload.get("format", {}).get("format_name"),
        "duration_sec": round(float(payload.get("format", {}).get("duration") or 0.0), 6),
        "format_bit_rate": int(float(payload.get("format", {}).get("bit_rate") or 0)),
        "audio_codec": audio_stream.get("codec_name"),
        "sample_rate": int(audio_stream.get("sample_rate") or 0),
        "channels": int(audio_stream.get("channels") or 0),
        "channel_layout": audio_stream.get("channel_layout"),
        "audio_bit_rate": int(float(audio_stream.get("bit_rate") or 0)),
        "attached_picture_streams": sum(
            1
            for stream in payload.get("streams", [])
            if stream.get("disposition", {}).get("attached_pic") == 1
        ),
        "tags": payload.get("format", {}).get("tags", {}),
    }


def decode_mono(path: Path, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    decoded = np.frombuffer(completed.stdout, dtype=np.float32)
    if decoded.size == 0:
        raise RuntimeError(f"Decoded audio is empty: {path}")
    return decoded


def rms_dbfs(signal: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(signal))) + 1e-12)
    return 20.0 * math.log10(rms)


def peak_dbfs(signal: np.ndarray) -> float:
    peak = float(np.max(np.abs(signal)) + 1e-12)
    return 20.0 * math.log10(peak)


def make_energy_envelope(signal: np.ndarray, sample_rate: int, hop_seconds: float) -> np.ndarray:
    hop_samples = max(1, int(round(sample_rate * hop_seconds)))
    frame_count = len(signal) // hop_samples
    if frame_count <= 0:
        raise RuntimeError("Decoded audio is too short for energy-envelope alignment")

    framed = signal[:frame_count * hop_samples].reshape(frame_count, hop_samples)
    envelope = np.sqrt(np.mean(np.square(framed), axis=1)).astype(np.float32)
    return (envelope - envelope.mean()) / (envelope.std() + 1e-8)


def estimate_offset_seconds(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    max_offset_seconds: float,
    hop_seconds: float,
) -> tuple[float, float]:
    reference_envelope = make_energy_envelope(reference, sample_rate, hop_seconds)
    candidate_envelope = make_energy_envelope(candidate, sample_rate, hop_seconds)
    max_lag = int(round(max_offset_seconds / hop_seconds))

    best_score = -float("inf")
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ref_slice = reference_envelope[-lag:]
            cand_slice = candidate_envelope[:len(ref_slice)]
        elif lag > 0:
            ref_slice = reference_envelope[:-lag]
            cand_slice = candidate_envelope[lag:lag + len(ref_slice)]
        else:
            ref_slice = reference_envelope
            cand_slice = candidate_envelope[:len(ref_slice)]

        length = min(len(ref_slice), len(cand_slice))
        if length < 100:
            continue

        score = float(np.dot(ref_slice[:length], cand_slice[:length]) / length)
        if score > best_score:
            best_score = score
            best_lag = lag

    if not math.isfinite(best_score):
        raise RuntimeError("Could not estimate offset: audio overlap is too short")
    return round(best_lag * hop_seconds, 6), round(best_score, 6)


def align_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    candidate_offset_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    offset_samples = int(round(candidate_offset_seconds * sample_rate))
    if offset_samples > 0:
        reference_aligned = reference[:max(0, len(reference) - offset_samples)]
        candidate_aligned = candidate[offset_samples:]
    elif offset_samples < 0:
        reference_aligned = reference[-offset_samples:]
        candidate_aligned = candidate[:max(0, len(candidate) + offset_samples)]
    else:
        reference_aligned = reference
        candidate_aligned = candidate

    length = min(len(reference_aligned), len(candidate_aligned))
    if length <= 0:
        raise RuntimeError("Audio pair has no overlapping samples after alignment")
    return reference_aligned[:length], candidate_aligned[:length]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))


def best_gain(source: np.ndarray, target: np.ndarray) -> float:
    return float(np.dot(target, source) / (np.dot(source, source) + 1e-12))


def analyze_pair(
    reference_a_path: Path,
    reference_b_path: Path,
    sample_rate: int,
    max_offset_seconds: float,
) -> dict:
    reference_a = decode_mono(reference_a_path, sample_rate)
    reference_b = decode_mono(reference_b_path, sample_rate)

    offset_seconds, alignment_score = estimate_offset_seconds(
        reference_a,
        reference_b,
        sample_rate,
        max_offset_seconds=max_offset_seconds,
        hop_seconds=0.05,
    )
    aligned_a, aligned_b = align_pair(reference_a, reference_b, sample_rate, offset_seconds)

    gain_b_to_a = best_gain(aligned_b, aligned_a)
    residual_a_minus_b = aligned_a - gain_b_to_a * aligned_b

    gain_a_to_b = best_gain(aligned_a, aligned_b)
    residual_b_minus_a = aligned_b - gain_a_to_b * aligned_a

    return {
        "analysis_sample_rate": sample_rate,
        "reference_a": {
            "decoded_duration_sec": round(len(reference_a) / float(sample_rate), 6),
            "rms_dbfs": round(rms_dbfs(reference_a), 6),
            "peak_dbfs": round(peak_dbfs(reference_a), 6),
        },
        "reference_b": {
            "decoded_duration_sec": round(len(reference_b) / float(sample_rate), 6),
            "rms_dbfs": round(rms_dbfs(reference_b), 6),
            "peak_dbfs": round(peak_dbfs(reference_b), 6),
        },
        "alignment": {
            "candidate_b_offset_vs_a_sec": offset_seconds,
            "energy_envelope_score": alignment_score,
            "aligned_duration_sec": round(len(aligned_a) / float(sample_rate), 6),
            "aligned_cosine_similarity": round(cosine_similarity(aligned_a, aligned_b), 6),
        },
        "residual_if_a_is_full_mix_and_b_is_minus": {
            "gain_b_to_a": round(gain_b_to_a, 6),
            "residual_rms_dbfs": round(rms_dbfs(residual_a_minus_b), 6),
            "residual_vs_a_rms_db": round(rms_dbfs(residual_a_minus_b) - rms_dbfs(aligned_a), 6),
        },
        "residual_if_b_is_full_mix_and_a_is_minus": {
            "gain_a_to_b": round(gain_a_to_b, 6),
            "residual_rms_dbfs": round(rms_dbfs(residual_b_minus_a), 6),
            "residual_vs_b_rms_db": round(rms_dbfs(residual_b_minus_a) - rms_dbfs(aligned_b), 6),
        },
    }


def main() -> int:
    require_docker_runtime(
        tool_name="probe_golden_reference_pair.py",
        entrypoint_path="/app/scripts/probe_golden_reference_pair.py",
    )
    parser = argparse.ArgumentParser(
        description="Probe a full-mix/minus reference pair inside Docker and produce repeatable metadata."
    )
    parser.add_argument("--reference-a", required=True, type=Path)
    parser.add_argument("--reference-b", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-offset-seconds", type=float, default=8.0)
    args = parser.parse_args()

    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.max_offset_seconds < 0:
        raise ValueError("--max-offset-seconds must not be negative")
    if not args.reference_a.exists():
        raise FileNotFoundError(args.reference_a)
    if not args.reference_b.exists():
        raise FileNotFoundError(args.reference_b)

    report = {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/scripts/probe_golden_reference_pair.py",
            "requires": ["ffprobe", "ffmpeg", "numpy"],
            "intended_runtime": "docker ml-worker",
        },
        "inputs": {
            "reference_a": probe_audio(args.reference_a),
            "reference_b": probe_audio(args.reference_b),
        },
        "pair_analysis": analyze_pair(
            args.reference_a,
            args.reference_b,
            sample_rate=args.sample_rate,
            max_offset_seconds=args.max_offset_seconds,
        ),
        "interpretation_limits": [
            "MP3 inputs are lossy references, so this probe is suitable for smoke/regression baselines, not final objective quality claims.",
            "The script does not infer which file is the full mix; use the residual sections after listening/metadata confirmation.",
            "SDR/SI-SDR require a verified ground-truth stem and are intentionally not reported here.",
        ],
    }

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
