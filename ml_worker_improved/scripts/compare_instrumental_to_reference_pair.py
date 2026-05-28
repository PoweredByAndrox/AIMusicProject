#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = SCRIPT_ROOT.parent / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            reference_slice = reference_envelope[-lag:]
            candidate_slice = candidate_envelope[:len(reference_slice)]
        elif lag > 0:
            reference_slice = reference_envelope[:-lag]
            candidate_slice = candidate_envelope[lag:lag + len(reference_slice)]
        else:
            reference_slice = reference_envelope
            candidate_slice = candidate_envelope[:len(reference_slice)]

        length = min(len(reference_slice), len(candidate_slice))
        if length < 100:
            continue

        score = float(np.dot(reference_slice[:length], candidate_slice[:length]) / length)
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


def spectral_proxy_metrics(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power) + 1e-12)

    spectral_centroid = float(np.sum(frequencies * power) / total_power)
    high_band_ratio = float(np.sum(power[frequencies >= 4000.0]) / total_power)
    air_band_ratio = float(np.sum(power[frequencies >= 7000.0]) / total_power)

    return {
        "spectral_centroid_hz": round(spectral_centroid, 6),
        "high_band_ratio_4k": round(high_band_ratio, 9),
        "air_band_ratio_7k": round(air_band_ratio, 9),
    }


def basic_metrics(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    peak = peak_dbfs(signal)
    rms = rms_dbfs(signal)
    return {
        "sample_rate": sample_rate,
        "duration_sec": round(len(signal) / float(sample_rate), 6),
        "rms_dbfs": round(rms, 6),
        "peak_dbfs": round(peak, 6),
        "crest_db": round(peak - rms, 6),
        **spectral_proxy_metrics(signal, sample_rate),
    }


def resolve_reference_path(
    probe_report: dict[str, Any],
    reference_key: str,
    override_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    reference_info = (probe_report.get("inputs") or {}).get(reference_key)
    if not isinstance(reference_info, dict):
        raise RuntimeError(f"Probe report has no inputs.{reference_key}")

    reference_path = override_path or Path(str(reference_info.get("path") or ""))
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference path is not available inside this container: {reference_path}. "
            "Mount the host project or pass the matching --reference-a-path/--reference-b-path."
        )
    return reference_path, reference_info


def check_input_hash(path: Path, expected_hash: str | None) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    return {
        "path": str(path),
        "sha256": actual_hash,
        "matches_probe_sha256": bool(expected_hash and actual_hash.lower() == expected_hash.lower()),
        "expected_sha256": expected_hash,
    }


def threshold_result(name: str, value: float, threshold: float | None, mode: str) -> dict[str, Any]:
    if threshold is None:
        return {"name": name, "value": value, "threshold": None, "passed": None}
    if mode == "min":
        passed = value >= threshold
    elif mode == "max":
        passed = value <= threshold
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")
    return {"name": name, "value": value, "threshold": threshold, "passed": passed}


def build_verdict(checks: list[dict[str, Any]]) -> dict[str, Any]:
    active_checks = [check for check in checks if check["passed"] is not None]
    failed_checks = [check for check in active_checks if not check["passed"]]
    return {
        "status": "not_evaluated" if not active_checks else "failed" if failed_checks else "passed",
        "failed_checks": failed_checks,
        "active_check_count": len(active_checks),
    }


def compare_candidate_to_reference(
    candidate_path: Path,
    reference_path: Path,
    sample_rate: int,
    max_offset_seconds: float,
) -> dict[str, Any]:
    reference = decode_mono(reference_path, sample_rate)
    candidate = decode_mono(candidate_path, sample_rate)

    offset_seconds, envelope_score = estimate_offset_seconds(
        reference,
        candidate,
        sample_rate,
        max_offset_seconds=max_offset_seconds,
        hop_seconds=0.05,
    )
    reference_aligned, candidate_aligned = align_pair(reference, candidate, sample_rate, offset_seconds)

    cosine = cosine_similarity(reference_aligned, candidate_aligned)
    gain_candidate_to_reference = best_gain(candidate_aligned, reference_aligned)
    gain_matched_candidate = gain_candidate_to_reference * candidate_aligned
    residual = reference_aligned - gain_matched_candidate

    reference_metrics = basic_metrics(reference_aligned, sample_rate)
    candidate_metrics = basic_metrics(candidate_aligned, sample_rate)
    residual_metrics = basic_metrics(residual, sample_rate)

    return {
        "alignment": {
            "candidate_offset_vs_reference_sec": offset_seconds,
            "energy_envelope_score": envelope_score,
            "aligned_duration_sec": round(len(reference_aligned) / float(sample_rate), 6),
        },
        "metrics": {
            "reference": reference_metrics,
            "candidate": candidate_metrics,
            "residual_after_gain_match": residual_metrics,
        },
        "comparisons": {
            "cosine_similarity": round(cosine, 6),
            "absolute_cosine_similarity": round(abs(cosine), 6),
            "gain_candidate_to_reference": round(gain_candidate_to_reference, 6),
            "rms_drift_db": round(candidate_metrics["rms_dbfs"] - reference_metrics["rms_dbfs"], 6),
            "peak_drift_db": round(candidate_metrics["peak_dbfs"] - reference_metrics["peak_dbfs"], 6),
            "duration_drift_sec": round(
                candidate_metrics["duration_sec"] - reference_metrics["duration_sec"],
                6,
            ),
            "high_band_ratio_drift_4k": round(
                candidate_metrics["high_band_ratio_4k"] - reference_metrics["high_band_ratio_4k"],
                9,
            ),
            "air_band_ratio_drift_7k": round(
                candidate_metrics["air_band_ratio_7k"] - reference_metrics["air_band_ratio_7k"],
                9,
            ),
            "residual_vs_reference_rms_db": round(
                residual_metrics["rms_dbfs"] - reference_metrics["rms_dbfs"],
                6,
            ),
        },
    }


def main() -> int:
    require_docker_runtime(
        tool_name="compare_instrumental_to_reference_pair.py",
        entrypoint_path="/app/scripts/compare_instrumental_to_reference_pair.py",
    )
    parser = argparse.ArgumentParser(
        description="Compare a future separated instrumental against a known reference-minus pair."
    )
    parser.add_argument("--probe-json", required=True, type=Path)
    parser.add_argument("--candidate-instrumental", required=True, type=Path)
    parser.add_argument("--reference-key", choices=["reference_a", "reference_b"], default="reference_b")
    parser.add_argument("--reference-a-path", type=Path)
    parser.add_argument("--reference-b-path", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--max-offset-seconds", type=float, default=8.0)
    parser.add_argument("--min-envelope-score", type=float)
    parser.add_argument("--min-abs-cosine", type=float)
    parser.add_argument("--max-abs-rms-drift-db", type=float)
    parser.add_argument("--max-abs-duration-drift-sec", type=float)
    parser.add_argument("--max-abs-high-band-ratio-drift", type=float)
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()

    if not args.probe_json.exists():
        raise FileNotFoundError(args.probe_json)
    if not args.candidate_instrumental.exists():
        raise FileNotFoundError(args.candidate_instrumental)
    if args.sample_rate is not None and args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.max_offset_seconds < 0:
        raise ValueError("--max-offset-seconds must not be negative")

    probe_report = load_json(args.probe_json)
    if probe_report.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported probe schema_version: {probe_report.get('schema_version')}")

    reference_override = args.reference_a_path if args.reference_key == "reference_a" else args.reference_b_path
    reference_path, reference_info = resolve_reference_path(probe_report, args.reference_key, reference_override)

    sample_rate = args.sample_rate or int(
        (probe_report.get("pair_analysis") or {}).get("analysis_sample_rate") or 16000
    )
    comparison = compare_candidate_to_reference(
        candidate_path=args.candidate_instrumental,
        reference_path=reference_path,
        sample_rate=sample_rate,
        max_offset_seconds=args.max_offset_seconds,
    )

    comparisons = comparison["comparisons"]
    checks = [
        threshold_result(
            "min_envelope_score",
            comparison["alignment"]["energy_envelope_score"],
            args.min_envelope_score,
            "min",
        ),
        threshold_result(
            "min_abs_cosine",
            comparisons["absolute_cosine_similarity"],
            args.min_abs_cosine,
            "min",
        ),
        threshold_result(
            "max_abs_rms_drift_db",
            abs(comparisons["rms_drift_db"]),
            args.max_abs_rms_drift_db,
            "max",
        ),
        threshold_result(
            "max_abs_duration_drift_sec",
            abs(comparisons["duration_drift_sec"]),
            args.max_abs_duration_drift_sec,
            "max",
        ),
        threshold_result(
            "max_abs_high_band_ratio_drift",
            abs(comparisons["high_band_ratio_drift_4k"]),
            args.max_abs_high_band_ratio_drift,
            "max",
        ),
    ]
    verdict = build_verdict(checks)

    report = {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/scripts/compare_instrumental_to_reference_pair.py",
            "requires": ["ffmpeg", "numpy"],
            "intended_runtime": "docker ml-worker",
        },
        "inputs": {
            "probe_json": str(args.probe_json),
            "reference_key": args.reference_key,
            "reference": check_input_hash(reference_path, reference_info.get("sha256")),
            "candidate_instrumental": {
                "path": str(args.candidate_instrumental),
                "sha256": sha256_file(args.candidate_instrumental),
            },
        },
        "comparison": comparison,
        "thresholds": checks,
        "verdict": verdict,
        "interpretation_limits": [
            "This report compares against a lossy reference-minus proxy and is intended for smoke/regression tracking.",
            "The report does not compute SDR/SI-SDR because no verified isolated ground-truth stem is provided.",
            "Cosine/residual metrics are sensitive to mastering, polarity, phase, codec and alignment differences.",
            "Use explicit thresholds only after confirming which reference file is the actual instrumental minus.",
        ],
    }

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)

    if args.fail_on_threshold and verdict["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
