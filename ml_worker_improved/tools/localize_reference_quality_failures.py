from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = ROOT_DIR / "scripts"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
import run_ground_truth_quality_gate as quality_gate  # noqa: E402


DEFAULT_BAND_EDGES_HZ = [0.0, 80.0, 160.0, 250.0, 350.0, 500.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0, 4000.0, 5600.0, 8000.0, 11000.0, 16000.0, 22050.0]


@dataclass(frozen=True)
class LocalizationConfig:
    window_sec: float = 5.0
    min_sdr_db: float = 20.0
    min_si_sdr_db: float = 20.0
    reference_silence_db: float = -80.0
    candidate_residual_floor_db: float = -90.0
    top_window_count: int = 12
    top_band_count: int = 5


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def resolve_stem_path(manifest_path: Path, stem_name: str) -> Path:
    payload = read_json(manifest_path)
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        raise RuntimeError(f"Manifest has no stems object: {manifest_path}")
    stem = stems.get(stem_name)
    if not isinstance(stem, dict) or not stem.get("path"):
        raise RuntimeError(f"Manifest has no requested stem path: stem={stem_name} manifest={manifest_path}")
    return resolve_manifest_path(manifest_path, str(stem["path"]))


def read_stereo_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def align_audio_arrays(audio_arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not audio_arrays:
        raise ValueError("audio_arrays cannot be empty")
    frames = min(audio.shape[0] for audio in audio_arrays)
    channels = min(audio.shape[1] for audio in audio_arrays)
    return [audio[:frames, :channels].astype(np.float32, copy=False) for audio in audio_arrays]


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + quality_gate.EPSILON))
    return float(20.0 * np.log10(max(rms, quality_gate.EPSILON)))


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    return audio.mean(axis=1).astype(np.float64, copy=False)


def parse_band_edges_hz(value: str) -> list[float]:
    edges = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("band edges must include at least two values")
    return edges


def normalize_band_edges_hz(band_edges_hz: Sequence[float], sample_rate: int) -> list[float]:
    nyquist = float(sample_rate) / 2.0
    cleaned = sorted(set(float(edge) for edge in band_edges_hz if 0.0 <= float(edge) <= nyquist))
    if not cleaned or cleaned[0] != 0.0:
        cleaned.insert(0, 0.0)
    if cleaned[-1] != nyquist:
        cleaned.append(nyquist)
    if len(cleaned) < 2:
        raise ValueError("band edges must define at least one frequency band")
    return cleaned


def band_error_profile(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    band_edges_hz: Sequence[float],
    top_band_count: int,
) -> list[dict[str, Any]]:
    reference_mono = to_mono(reference_audio)
    candidate_mono = to_mono(candidate_audio)
    error_mono = candidate_mono - reference_mono
    if len(reference_mono) == 0:
        return []
    window = np.hanning(len(reference_mono))
    if not np.any(window):
        window = np.ones(len(reference_mono), dtype=np.float64)
    reference_fft = np.fft.rfft(reference_mono * window)
    error_fft = np.fft.rfft(error_mono * window)
    frequencies = np.fft.rfftfreq(len(reference_mono), d=1.0 / float(sample_rate))
    total_error_power = float(np.sum(np.abs(error_fft) ** 2) + quality_gate.EPSILON)
    bands: list[dict[str, Any]] = []
    edges = normalize_band_edges_hz(band_edges_hz, sample_rate)
    for low, high in zip(edges[:-1], edges[1:]):
        if high <= low:
            continue
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            continue
        reference_power = float(np.sum(np.abs(reference_fft[mask]) ** 2) + quality_gate.EPSILON)
        error_power = float(np.sum(np.abs(error_fft[mask]) ** 2) + quality_gate.EPSILON)
        band_sdr = 10.0 * np.log10(reference_power / error_power)
        bands.append(
            {
                "band_hz": f"{round(low, 3)}-{round(high, 3)}",
                "low_hz": round(float(low), 6),
                "high_hz": round(float(high), 6),
                "error_share": round(float(error_power / total_error_power), 9),
                "band_sdr_db": round(float(band_sdr), 6),
            }
        )
    return sorted(bands, key=lambda item: item["error_share"], reverse=True)[:top_band_count]


def analyze_quality_window(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    start_frame: int,
    end_frame: int,
    config: LocalizationConfig,
    band_edges_hz: Sequence[float],
) -> dict[str, Any]:
    reference_window = reference_audio[start_frame:end_frame]
    candidate_window = candidate_audio[start_frame:end_frame]
    reference_mono = to_mono(reference_window)
    candidate_mono = to_mono(candidate_window)
    reference_rms_db = rms_db(reference_window)
    candidate_rms_db = rms_db(candidate_window)
    error_rms_db = rms_db(candidate_window - reference_window)
    sdr_db = quality_gate.signal_to_distortion_db(reference_mono, candidate_mono)
    si_sdr_db = quality_gate.scale_invariant_signal_to_distortion_db(reference_mono, candidate_mono)
    cosine = quality_gate.cosine_similarity(reference_mono, candidate_mono)
    top_error_bands = band_error_profile(
        reference_audio=reference_window,
        candidate_audio=candidate_window,
        sample_rate=sample_rate,
        band_edges_hz=band_edges_hz,
        top_band_count=config.top_band_count,
    )

    issue_labels: list[str] = []
    window_class = "active_reference"
    active_vocal_defect = True
    if reference_rms_db <= config.reference_silence_db:
        window_class = "near_silent_reference"
        active_vocal_defect = False
        if candidate_rms_db <= config.candidate_residual_floor_db:
            issue_labels.append("near_silent_reference_residual_floor")
        else:
            issue_labels.append("silence_residual_leakage")
    else:
        if sdr_db < config.min_sdr_db:
            issue_labels.append("active_low_sdr")
        if si_sdr_db < config.min_si_sdr_db:
            issue_labels.append("active_low_si_sdr")
        if cosine < 0.9:
            issue_labels.append("active_target_shape_mismatch")
        active_vocal_defect = bool(issue_labels)

    severity_score = 0.0
    if window_class == "active_reference":
        severity_score = (
            max(0.0, config.min_sdr_db - sdr_db)
            + max(0.0, config.min_si_sdr_db - si_sdr_db)
            + max(0.0, 0.9 - cosine) * 20.0
            + max(0.0, abs(candidate_rms_db - reference_rms_db) - 3.0)
        )
    elif "silence_residual_leakage" in issue_labels:
        severity_score = max(0.0, candidate_rms_db - config.candidate_residual_floor_db)
    else:
        severity_score = 0.0

    return {
        "start_sec": round(float(start_frame) / float(sample_rate), 6),
        "end_sec": round(float(end_frame) / float(sample_rate), 6),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "window_class": window_class,
        "active_vocal_defect": active_vocal_defect,
        "severity_score": round(float(severity_score), 6),
        "issue_labels": issue_labels,
        "metrics": {
            "reference_rms_db": round(reference_rms_db, 6),
            "candidate_rms_db": round(candidate_rms_db, 6),
            "error_rms_db": round(error_rms_db, 6),
            "rms_drift_db": round(candidate_rms_db - reference_rms_db, 6),
            "sdr_db": round(float(sdr_db), 6),
            "si_sdr_db": round(float(si_sdr_db), 6),
            "cosine_similarity": round(float(cosine), 6),
        },
        "top_error_bands": top_error_bands,
    }


def localize_reference_quality_failures_from_audio(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    config: LocalizationConfig,
    band_edges_hz: Sequence[float],
) -> dict[str, Any]:
    reference_audio, candidate_audio = align_audio_arrays([reference_audio, candidate_audio])
    window_frames = max(1, int(round(config.window_sec * sample_rate)))
    windows = []
    for start in range(0, len(reference_audio), window_frames):
        end = min(len(reference_audio), start + window_frames)
        if end <= start:
            continue
        windows.append(
            analyze_quality_window(
                reference_audio=reference_audio,
                candidate_audio=candidate_audio,
                sample_rate=sample_rate,
                start_frame=start,
                end_frame=end,
                config=config,
                band_edges_hz=band_edges_hz,
            )
        )

    active_defects = [window for window in windows if window["active_vocal_defect"]]
    silent_residuals = [
        window
        for window in windows
        if window["window_class"] == "near_silent_reference" and window["issue_labels"]
    ]
    return {
        "summary": {
            "window_sec": float(config.window_sec),
            "window_count": len(windows),
            "active_reference_window_count": sum(1 for window in windows if window["window_class"] == "active_reference"),
            "near_silent_reference_window_count": sum(
                1 for window in windows if window["window_class"] == "near_silent_reference"
            ),
            "active_vocal_defect_window_count": len(active_defects),
            "silent_residual_window_count": len(silent_residuals),
        },
        "top_active_defect_windows": sorted(
            active_defects,
            key=lambda item: item["severity_score"],
            reverse=True,
        )[: config.top_window_count],
        "top_silent_residual_windows": sorted(
            silent_residuals,
            key=lambda item: item["severity_score"],
            reverse=True,
        )[: config.top_window_count],
        "windows": windows,
        "repair_rule_recommendations": build_repair_rule_recommendations(active_defects, silent_residuals),
    }


def build_repair_rule_recommendations(
    active_defects: Sequence[dict[str, Any]],
    silent_residuals: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    near_silent_floor_count = sum(
        1 for window in silent_residuals if "near_silent_reference_residual_floor" in window["issue_labels"]
    )
    silence_leak_count = sum(1 for window in silent_residuals if "silence_residual_leakage" in window["issue_labels"])
    if near_silent_floor_count:
        recommendations.append(
            {
                "rule_id": "silence_aware_window_gate",
                "status": "proposed",
                "mechanism": "demote near-silent reference windows with candidate residual below floor from active vocal defect ranking",
                "evidence": {"near_silent_reference_residual_floor_windows": near_silent_floor_count},
                "guard": "keep failing silence_residual_leakage when candidate_rms_db is above candidate_residual_floor_db",
            }
        )
    if silence_leak_count:
        recommendations.append(
            {
                "rule_id": "silent_tail_residual_suppression",
                "status": "proposed",
                "mechanism": "suppress candidate residual only where verified reference stem is silent and residual is above floor",
                "evidence": {"silence_residual_leakage_windows": silence_leak_count},
                "guard": "do not apply to active_reference windows",
            }
        )
    if active_defects:
        band_counts: dict[str, int] = {}
        for window in active_defects:
            for band in window.get("top_error_bands", [])[:1]:
                band_counts[str(band["band_hz"])] = band_counts.get(str(band["band_hz"]), 0) + 1
        dominant_bands = sorted(band_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        recommendations.append(
            {
                "rule_id": "active_vocal_band_error_repair",
                "status": "proposed",
                "mechanism": "target active vocal windows by dominant error bands instead of global profile changes",
                "evidence": {
                    "active_vocal_defect_windows": len(active_defects),
                    "dominant_error_bands": [
                        {"band_hz": band, "window_count": count} for band, count in dominant_bands
                    ],
                },
                "guard": "must improve active-window SDR/SI-SDR without increasing silent residual leakage",
            }
        )
    return recommendations


def localize_reference_quality_failures(
    *,
    ground_truth_manifest: Path,
    candidate_manifest: Path,
    output_dir: Path,
    stem_name: str,
    config: LocalizationConfig,
    band_edges_hz: Sequence[float],
) -> dict[str, Any]:
    ground_truth_bundle = quality_gate.resolve_stem_bundle(ground_truth_manifest)
    quality_gate.validate_verified_ground_truth(ground_truth_bundle)
    reference_path = resolve_stem_path(ground_truth_manifest, stem_name)
    candidate_path = resolve_stem_path(candidate_manifest, stem_name)
    reference_audio, reference_sample_rate = read_stereo_audio(reference_path)
    candidate_audio, candidate_sample_rate = read_stereo_audio(candidate_path)
    if reference_sample_rate != candidate_sample_rate:
        raise RuntimeError("reference and candidate sample rates differ")
    normalized_edges = normalize_band_edges_hz(band_edges_hz, reference_sample_rate)
    report = localize_reference_quality_failures_from_audio(
        reference_audio=reference_audio,
        candidate_audio=candidate_audio,
        sample_rate=reference_sample_rate,
        config=config,
        band_edges_hz=normalized_edges,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report.update(
        {
            "schema_version": 1,
            "inputs": {
                "ground_truth_manifest": str(ground_truth_manifest),
                "candidate_manifest": str(candidate_manifest),
                "reference_stem_path": str(reference_path),
                "candidate_stem_path": str(candidate_path),
                "stem": stem_name,
            },
            "config": {
                "window_sec": config.window_sec,
                "min_sdr_db": config.min_sdr_db,
                "min_si_sdr_db": config.min_si_sdr_db,
                "reference_silence_db": config.reference_silence_db,
                "candidate_residual_floor_db": config.candidate_residual_floor_db,
                "band_edges_hz": normalized_edges,
            },
        }
    )
    report_path = output_dir / "reference_quality_failure_localization.json"
    write_json(report_path, report)
    return {
        "report_path": str(report_path),
        "sha256": sha256_file(report_path),
        "summary": report["summary"],
        "top_active_defect_windows": report["top_active_defect_windows"],
        "top_silent_residual_windows": report["top_silent_residual_windows"],
        "repair_rule_recommendations": report["repair_rule_recommendations"],
    }


def summarize_window_for_cli(window: dict[str, Any]) -> dict[str, Any]:
    metrics = window.get("metrics", {})
    return {
        "start_sec": window.get("start_sec"),
        "end_sec": window.get("end_sec"),
        "window_class": window.get("window_class"),
        "issue_labels": window.get("issue_labels", []),
        "severity_score": window.get("severity_score"),
        "sdr_db": metrics.get("sdr_db"),
        "si_sdr_db": metrics.get("si_sdr_db"),
        "reference_rms_db": metrics.get("reference_rms_db"),
        "candidate_rms_db": metrics.get("candidate_rms_db"),
        "top_error_bands": [
            str(band.get("band_hz"))
            for band in window.get("top_error_bands", [])[:3]
        ],
    }


def build_cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_path": report.get("report_path"),
        "sha256": report.get("sha256"),
        "summary": report.get("summary", {}),
        "top_active_defect_windows": [
            summarize_window_for_cli(window)
            for window in report.get("top_active_defect_windows", [])[:5]
        ],
        "top_silent_residual_windows": [
            summarize_window_for_cli(window)
            for window in report.get("top_silent_residual_windows", [])[:5]
        ],
        "repair_rule_recommendations": report.get("repair_rule_recommendations", []),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Localize reference-guided stem quality failures by window and band.")
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocals")
    parser.add_argument("--window-sec", type=float, default=LocalizationConfig.window_sec)
    parser.add_argument("--min-sdr-db", type=float, default=LocalizationConfig.min_sdr_db)
    parser.add_argument("--min-si-sdr-db", type=float, default=LocalizationConfig.min_si_sdr_db)
    parser.add_argument("--reference-silence-db", type=float, default=LocalizationConfig.reference_silence_db)
    parser.add_argument("--candidate-residual-floor-db", type=float, default=LocalizationConfig.candidate_residual_floor_db)
    parser.add_argument("--top-window-count", type=int, default=LocalizationConfig.top_window_count)
    parser.add_argument("--top-band-count", type=int, default=LocalizationConfig.top_band_count)
    parser.add_argument("--band-edges-hz", type=parse_band_edges_hz, default=",".join(str(edge) for edge in DEFAULT_BAND_EDGES_HZ))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="localize_reference_quality_failures.py",
        entrypoint_path="/app/tools/localize_reference_quality_failures.py",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = LocalizationConfig(
        window_sec=float(args.window_sec),
        min_sdr_db=float(args.min_sdr_db),
        min_si_sdr_db=float(args.min_si_sdr_db),
        reference_silence_db=float(args.reference_silence_db),
        candidate_residual_floor_db=float(args.candidate_residual_floor_db),
        top_window_count=int(args.top_window_count),
        top_band_count=int(args.top_band_count),
    )
    summary = localize_reference_quality_failures(
        ground_truth_manifest=args.ground_truth_manifest,
        candidate_manifest=args.candidate_manifest,
        output_dir=args.output_dir,
        stem_name=str(args.stem),
        config=config,
        band_edges_hz=args.band_edges_hz,
    )
    print(json.dumps(build_cli_summary(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
