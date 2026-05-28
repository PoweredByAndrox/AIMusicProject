from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
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


MODEL_ID = "reference_guided_band_repair"
MODEL_DISPLAY_NAME = "Reference-guided active vocal band repair"
DEFAULT_STEM = "vocals"


@dataclass(frozen=True)
class BandRepairConfig:
    repair_strength: float = 0.65
    max_repaired_bands_per_window: int = 3
    min_error_share: float = 0.05
    edge_fade_sec: float = 0.025


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_file_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    try:
        info = sf.info(str(path))
    except Exception:
        return summary
    summary.update(
        {
            "sample_rate": int(info.samplerate),
            "frames": int(info.frames),
            "channels": int(info.channels),
            "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
        }
    )
    return summary


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


def resolve_ground_truth_mix_path(ground_truth_manifest: Path) -> Path:
    payload = read_json(ground_truth_manifest)
    require_manifest_object(ground_truth_manifest, payload)
    mix = payload.get("mix")
    if not isinstance(mix, dict) or not mix.get("path"):
        raise RuntimeError(f"Ground-truth manifest has no mix.path: {ground_truth_manifest}")
    return resolve_manifest_path(ground_truth_manifest, str(mix["path"]))


def require_manifest_object(manifest_path: Path, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest must be an object: {manifest_path}")
    return payload


def resolve_manifest_source_audio_path(manifest_path: Path, payload: Any) -> Path:
    payload = require_manifest_object(manifest_path, payload)
    source_audio_path = payload.get("source_audio_path")
    if isinstance(source_audio_path, str) and source_audio_path:
        return resolve_manifest_path(manifest_path, source_audio_path)
    mix = payload.get("mix")
    if isinstance(mix, dict) and isinstance(mix.get("path"), str) and mix["path"]:
        return resolve_manifest_path(manifest_path, str(mix["path"]))
    raise RuntimeError(f"Manifest has no source audio path or mix.path: {manifest_path}")


def validate_candidate_source_matches_ground_truth(*, ground_truth_manifest: Path, candidate_manifest: Path) -> None:
    ground_truth_payload = read_json(ground_truth_manifest)
    candidate_payload = read_json(candidate_manifest)
    ground_truth_source = resolve_manifest_source_audio_path(ground_truth_manifest, ground_truth_payload)
    candidate_source = resolve_manifest_source_audio_path(candidate_manifest, candidate_payload)
    if ground_truth_source.as_posix() != candidate_source.as_posix():
        raise RuntimeError(
            "Candidate source audio path does not match ground-truth mix: "
            f"candidate={candidate_source} ground_truth={ground_truth_source}"
        )


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


def edge_fade_envelope(frame_count: int, fade_frames: int) -> np.ndarray:
    envelope = np.ones((frame_count, 1), dtype=np.float32)
    if fade_frames <= 0 or frame_count <= 2:
        return envelope
    fade_frames = min(fade_frames, frame_count // 2)
    if fade_frames <= 0:
        return envelope
    fade_in = np.linspace(0.0, 1.0, fade_frames + 2, dtype=np.float32)[1:-1, None]
    fade_out = fade_in[::-1]
    envelope[:fade_frames] = fade_in
    envelope[-fade_frames:] = fade_out
    return envelope


def clipping_summary(audio: np.ndarray) -> dict[str, Any]:
    peak_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped_ratio = float(np.mean(np.abs(audio) > 1.0)) if audio.size else 0.0
    return {
        "peak_abs": round(peak_abs, 6),
        "clipped_sample_ratio": round(clipped_ratio, 9),
    }


def parse_top_error_band(band: Any) -> dict[str, float]:
    if not isinstance(band, dict):
        raise RuntimeError("Invalid top_error_bands entry: expected object")
    try:
        low = float(band["low_hz"])
        high = float(band["high_hz"])
        error_share = float(band.get("error_share", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid top_error_bands entry: {band}") from exc
    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or not math.isfinite(error_share)
        or low < 0.0
        or high <= low
        or error_share < 0.0
    ):
        raise RuntimeError(f"Invalid top_error_bands entry: {band}")
    return {"low_hz": low, "high_hz": high, "error_share": error_share}


def parse_window_frame_bounds(window: dict[str, Any], audio_frame_count: int) -> tuple[int, int]:
    try:
        start = int(window["start_frame"])
        end = int(window["end_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid localization window frame bounds: {window}") from exc
    if start < 0 or end < start:
        raise RuntimeError(f"Invalid localization window frame bounds: {window}")
    if audio_frame_count <= 0 or start >= audio_frame_count or end > audio_frame_count:
        raise RuntimeError(
            "Invalid localization window frame bounds outside audio bounds: "
            f"window={window} audio_frame_count={audio_frame_count}"
        )
    return start, end


def require_localization_report_object(localization_report: Any) -> dict[str, Any]:
    if not isinstance(localization_report, dict):
        raise RuntimeError(
            f"Localization report must be an object, got {type(localization_report).__name__}"
        )
    return localization_report


def require_localization_window_object(window: Any) -> dict[str, Any]:
    if not isinstance(window, dict):
        raise RuntimeError(f"Invalid localization window: expected object, got {type(window).__name__}")
    return window


def repair_window_frequency_bands(
    *,
    reference_window: np.ndarray,
    candidate_window: np.ndarray,
    sample_rate: int,
    top_error_bands: Sequence[dict[str, Any]],
    config: BandRepairConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(reference_window) != len(candidate_window):
        raise ValueError("reference_window and candidate_window must have equal frame counts")
    if len(reference_window) == 0:
        return candidate_window.copy(), []
    reference_spectrum = np.fft.rfft(reference_window, axis=0)
    candidate_spectrum = np.fft.rfft(candidate_window, axis=0)
    repaired_spectrum = candidate_spectrum.copy()
    frequencies = np.fft.rfftfreq(len(reference_window), d=1.0 / float(sample_rate))
    repaired_bands: list[dict[str, Any]] = []
    parsed_bands = [parse_top_error_band(band) for band in top_error_bands]
    sorted_bands = sorted(
        parsed_bands,
        key=lambda band: band["error_share"],
        reverse=True,
    )
    for band in sorted_bands[: config.max_repaired_bands_per_window]:
        error_share = band["error_share"]
        if error_share < config.min_error_share:
            continue
        low = band["low_hz"]
        high = band["high_hz"]
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            continue
        strength = min(1.0, max(0.0, float(config.repair_strength) * error_share))
        repaired_spectrum[mask, :] = (1.0 - strength) * repaired_spectrum[mask, :] + strength * reference_spectrum[mask, :]
        repaired_bands.append(
            {
                "band_hz": f"{low:.1f}-{high:.1f}",
                "low_hz": round(low, 6),
                "high_hz": round(high, 6),
                "error_share": round(error_share, 9),
                "strength": round(strength, 6),
            }
        )
    repaired = np.fft.irfft(repaired_spectrum, n=len(reference_window), axis=0).astype(np.float32)
    return repaired, repaired_bands


def repair_active_vocal_bands(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    localization_report: Any,
    config: BandRepairConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference_audio, candidate_audio = align_audio_arrays([reference_audio, candidate_audio])
    repaired_audio = candidate_audio.copy()
    report = require_localization_report_object(localization_report)
    windows = report.get("windows")
    if not isinstance(windows, list):
        raise RuntimeError("Localization report must include windows list")
    repaired_windows: list[dict[str, Any]] = []
    fade_frames = int(round(float(config.edge_fade_sec) * float(sample_rate)))
    for raw_window in windows:
        window = require_localization_window_object(raw_window)
        if not window.get("active_vocal_defect"):
            continue
        if window.get("window_class") != "active_reference":
            continue
        top_error_bands = window.get("top_error_bands")
        if not isinstance(top_error_bands, list) or not top_error_bands:
            continue
        start, end = parse_window_frame_bounds(window, len(repaired_audio))
        if end <= start:
            continue
        repaired_window, repaired_bands = repair_window_frequency_bands(
            reference_window=reference_audio[start:end],
            candidate_window=repaired_audio[start:end],
            sample_rate=sample_rate,
            top_error_bands=top_error_bands,
            config=config,
        )
        if not repaired_bands:
            continue
        envelope = edge_fade_envelope(end - start, fade_frames)
        repaired_audio[start:end] = repaired_audio[start:end] * (1.0 - envelope) + repaired_window * envelope
        repaired_windows.append(
            {
                "start_sec": window.get("start_sec"),
                "end_sec": window.get("end_sec"),
                "start_frame": start,
                "end_frame": end,
                "source_issue_labels": window.get("issue_labels", []),
                "source_sdr_db": window.get("metrics", {}).get("sdr_db"),
                "source_si_sdr_db": window.get("metrics", {}).get("si_sdr_db"),
                "repaired_bands": repaired_bands,
            }
        )
    pre_clip_summary = clipping_summary(repaired_audio)
    clipped_audio = np.clip(repaired_audio, -1.0, 1.0).astype(np.float32, copy=False)
    diagnostics = {
        "schema_version": 1,
        "summary": {
            "input_window_count": len(windows),
            "active_repaired_window_count": len(repaired_windows),
            "repair_strength": float(config.repair_strength),
            "max_repaired_bands_per_window": int(config.max_repaired_bands_per_window),
            "min_error_share": float(config.min_error_share),
            "edge_fade_sec": float(config.edge_fade_sec),
        },
        "vocal_repair_clipping": {
            "pre_clip": pre_clip_summary,
            "post_clip": clipping_summary(clipped_audio),
        },
        "repaired_windows": repaired_windows,
    }
    return clipped_audio, diagnostics


def build_band_repair_manifest(
    *,
    source_audio_path: Path,
    ground_truth_manifest: Path,
    candidate_manifest: Path,
    localization_report: Path,
    vocals_path: Path,
    instrumental_path: Path,
    diagnostics_path: Path,
    sample_rate: int,
    frames: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "model": {
            "model_id": MODEL_ID,
            "name": MODEL_DISPLAY_NAME,
            "family": "reference_assisted_repair",
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": True,
            "delivery_allowed_without_reference": False,
            "per_track_hardcode_allowed": False,
        },
        "reference_assistance": {
            "ground_truth_manifest": str(ground_truth_manifest),
            "source_candidate_manifest": str(candidate_manifest),
            "localization_report": str(localization_report),
            "stem": DEFAULT_STEM,
            "use": "offline_reference_guided_active_band_repair",
            "not_a_no_reference_production_route": True,
        },
        "repair_diagnostics": {
            "path": str(diagnostics_path),
            "sha256": sha256_file(diagnostics_path),
            "bytes": diagnostics_path.stat().st_size,
        },
        "stems": {
            "vocals": {
                "path": str(vocals_path),
                "role": "target",
                **audio_file_summary(vocals_path),
            },
            "instrumental": {
                "path": str(instrumental_path),
                "role": "companion_source_minus_target",
                **audio_file_summary(instrumental_path),
            },
        },
        "audio": {
            "sample_rate": int(sample_rate),
            "frames": int(frames),
            "duration_sec": round(float(frames) / float(sample_rate), 6) if sample_rate else 0.0,
        },
        "elapsed_sec": round(float(elapsed_sec), 3),
    }


def build_reference_guided_band_repair(
    *,
    ground_truth_manifest: Path,
    candidate_manifest: Path,
    localization_report: Path,
    output_dir: Path,
    stem_name: str,
    config: BandRepairConfig,
) -> dict[str, Any]:
    start_wall = time.time()
    if stem_name != DEFAULT_STEM:
        raise ValueError("reference-guided band repair currently supports only stem=vocals")
    ground_truth_bundle = quality_gate.resolve_stem_bundle(ground_truth_manifest)
    quality_gate.validate_verified_ground_truth(ground_truth_bundle)
    source_audio_path = resolve_ground_truth_mix_path(ground_truth_manifest)
    validate_candidate_source_matches_ground_truth(
        ground_truth_manifest=ground_truth_manifest,
        candidate_manifest=candidate_manifest,
    )
    reference_path = resolve_stem_path(ground_truth_manifest, stem_name)
    candidate_path = resolve_stem_path(candidate_manifest, stem_name)
    source_audio, source_sample_rate = read_stereo_audio(source_audio_path)
    reference_audio, reference_sample_rate = read_stereo_audio(reference_path)
    candidate_audio, candidate_sample_rate = read_stereo_audio(candidate_path)
    if len({source_sample_rate, reference_sample_rate, candidate_sample_rate}) != 1:
        raise RuntimeError("source, reference, and candidate sample rates must match")
    source_audio, reference_audio, candidate_audio = align_audio_arrays([source_audio, reference_audio, candidate_audio])
    report = read_json(localization_report)
    repaired_audio, diagnostics = repair_active_vocal_bands(
        reference_audio=reference_audio,
        candidate_audio=candidate_audio,
        sample_rate=reference_sample_rate,
        localization_report=report,
        config=config,
    )
    instrumental_audio = np.clip(source_audio - repaired_audio, -1.0, 1.0)
    diagnostics["instrumental_reconstruction_clipping"] = {
        "pre_clip": clipping_summary(source_audio - repaired_audio),
        "post_clip": clipping_summary(instrumental_audio),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / "vocals.flac"
    instrumental_path = output_dir / "instrumental.flac"
    diagnostics_path = output_dir / "band_repair_diagnostics.json"
    sf.write(str(vocals_path), repaired_audio, reference_sample_rate, subtype="PCM_24")
    sf.write(str(instrumental_path), instrumental_audio, reference_sample_rate, subtype="PCM_24")
    write_json(diagnostics_path, diagnostics)
    manifest = build_band_repair_manifest(
        source_audio_path=source_audio_path,
        ground_truth_manifest=ground_truth_manifest,
        candidate_manifest=candidate_manifest,
        localization_report=localization_report,
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        diagnostics_path=diagnostics_path,
        sample_rate=reference_sample_rate,
        frames=int(repaired_audio.shape[0]),
        elapsed_sec=time.time() - start_wall,
    )
    write_json(output_dir / "stem_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reference-guided active vocal band repair candidate.")
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--localization-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--repair-strength", type=float, default=BandRepairConfig.repair_strength)
    parser.add_argument("--max-repaired-bands-per-window", type=int, default=BandRepairConfig.max_repaired_bands_per_window)
    parser.add_argument("--min-error-share", type=float, default=BandRepairConfig.min_error_share)
    parser.add_argument("--edge-fade-sec", type=float, default=BandRepairConfig.edge_fade_sec)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_reference_guided_band_repair.py",
        entrypoint_path="/app/tools/build_reference_guided_band_repair.py",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = build_reference_guided_band_repair(
        ground_truth_manifest=args.ground_truth_manifest,
        candidate_manifest=args.candidate_manifest,
        localization_report=args.localization_report,
        output_dir=args.output_dir,
        stem_name=str(args.stem),
        config=BandRepairConfig(
            repair_strength=float(args.repair_strength),
            max_repaired_bands_per_window=int(args.max_repaired_bands_per_window),
            min_error_share=float(args.min_error_share),
            edge_fade_sec=float(args.edge_fade_sec),
        ),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / "stem_manifest.json"),
                "model_id": manifest["model_id"],
                "repair_diagnostics": manifest["repair_diagnostics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
