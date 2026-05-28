#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import soundfile as sf


EPSILON = 1.0e-12
REPAIR_LABELS = frozenset(
    {
        "bar_spanning_drop_melody_leakage",
        "micro_reference_stem_leakage",
        "micro_residual_reference_stem_leakage",
    }
)
UNAVAILABLE_REPAIR_SOURCES = frozenset({"unattributed_harmonic_residual"})


@dataclass(frozen=True)
class AuditGuidedDebleedConfig:
    repair_strength: float = 0.85
    max_projection_gain: float = 0.90
    min_residual_leakage_strength: float = 0.015
    min_candidate_leakage_strength: float = 0.04
    min_phrase_leakage_strength: float = 0.05
    phrase_projection_gain_multiplier: float = 1.0
    phrase_multi_source_projection: bool = False
    max_multi_source_correction_ratio: float = 0.90
    allowed_repair_reasons: tuple[str, ...] = ()
    allowed_repair_sources: tuple[str, ...] = ()
    context_ms: float = 35.0
    max_peak: float = 0.999


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError(f"Invalid audio shape for {path}: {audio.shape}")
    return audio.astype(np.float32, copy=False), int(sample_rate)


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate, format="FLAC", subtype="PCM_24")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: AuditGuidedDebleedConfig) -> None:
    if not 0.0 < config.repair_strength <= 1.0:
        raise ValueError("repair_strength must be in (0, 1]")
    if not 0.0 < config.max_projection_gain <= 1.0:
        raise ValueError("max_projection_gain must be in (0, 1]")
    if config.min_residual_leakage_strength < 0.0:
        raise ValueError("min_residual_leakage_strength must be non-negative")
    if config.min_candidate_leakage_strength < 0.0:
        raise ValueError("min_candidate_leakage_strength must be non-negative")
    if config.min_phrase_leakage_strength < 0.0:
        raise ValueError("min_phrase_leakage_strength must be non-negative")
    if config.phrase_projection_gain_multiplier < 0.0:
        raise ValueError("phrase_projection_gain_multiplier must be non-negative")
    if not 0.0 < config.max_multi_source_correction_ratio <= 1.0:
        raise ValueError("max_multi_source_correction_ratio must be in (0, 1]")
    if any(not str(reason).strip() for reason in config.allowed_repair_reasons):
        raise ValueError("allowed_repair_reasons must not contain empty values")
    if any(not str(source).strip() for source in config.allowed_repair_sources):
        raise ValueError("allowed_repair_sources must not contain empty values")
    if config.context_ms < 0.0:
        raise ValueError("context_ms must be non-negative")
    if not 0.0 < config.max_peak <= 1.0:
        raise ValueError("max_peak must be in (0, 1]")


def fit_length(audio: np.ndarray, frame_count: int) -> np.ndarray:
    if audio.shape[0] == frame_count:
        return audio
    if audio.shape[0] > frame_count:
        return audio[:frame_count]
    padded = np.zeros((frame_count, audio.shape[1]), dtype=np.float32)
    padded[: audio.shape[0]] = audio
    return padded


def ensure_same_layout(named_audio: dict[str, tuple[np.ndarray, int]]) -> tuple[int, int]:
    sample_rates = {sample_rate for _, sample_rate in named_audio.values()}
    if len(sample_rates) != 1:
        raise ValueError(f"sample-rate mismatch: {sorted(sample_rates)}")
    channel_counts = {audio.shape[1] for audio, _ in named_audio.values()}
    if len(channel_counts) != 1:
        raise ValueError(f"channel-count mismatch: {sorted(channel_counts)}")
    sample_rate = next(iter(sample_rates))
    channel_count = next(iter(channel_counts))
    return int(sample_rate), int(channel_count)


def parse_named_path(raw_value: str) -> tuple[str, Path]:
    if "=" not in raw_value:
        raise ValueError(f"expected NAME=PATH, got {raw_value!r}")
    raw_name, raw_path = raw_value.split("=", 1)
    name = raw_name.strip()
    if not name:
        raise ValueError(f"empty leakage source name in {raw_value!r}")
    return name, Path(raw_path.strip())


def canonical_stem_name(stem_name: str) -> str:
    canonical = "".join(character for character in stem_name.lower() if character.isalnum())
    if canonical.endswith("s"):
        canonical = canonical[:-1]
    return canonical


def validate_leakage_source_names(leakage_source_paths: dict[str, Path], *, stem_name: str) -> None:
    target_name = canonical_stem_name(stem_name)
    for source_name in leakage_source_paths:
        if canonical_stem_name(source_name) == target_name:
            raise ValueError(f"leakage sources must not include target stem: {source_name}")


def window_has_repair_label(window: dict[str, Any]) -> bool:
    return bool(REPAIR_LABELS.intersection({str(label) for label in window.get("issue_labels", [])}))


def candidate_source_for_window(window: dict[str, Any], config: AuditGuidedDebleedConfig) -> tuple[str | None, str]:
    metrics = window.get("metrics", {})
    phrase_source = metrics.get("phrase_leakage_source")
    phrase_strength = float(metrics.get("phrase_leakage_strength") or 0.0)
    if (
        phrase_source
        and str(phrase_source) not in UNAVAILABLE_REPAIR_SOURCES
        and phrase_strength >= config.min_phrase_leakage_strength
    ):
        return str(phrase_source), "phrase_reference_projection"
    residual_source = metrics.get("max_residual_leakage_source")
    residual_strength = float(metrics.get("max_residual_leakage_strength") or 0.0)
    if residual_source and residual_strength >= config.min_residual_leakage_strength:
        return str(residual_source), "residual_reference_projection"
    candidate_source = metrics.get("max_leakage_source")
    candidate_strength = float(metrics.get("max_leakage_strength") or 0.0)
    if candidate_source and candidate_strength >= config.min_candidate_leakage_strength:
        return str(candidate_source), "candidate_reference_projection"
    return None, "below_strength_floor"


def repair_window_allowed(source_name: str, source_reason: str, config: AuditGuidedDebleedConfig) -> bool:
    if config.allowed_repair_reasons and source_reason not in set(config.allowed_repair_reasons):
        return False
    if config.allowed_repair_sources and source_name not in set(config.allowed_repair_sources):
        return False
    return True


def iter_audit_windows(report: dict[str, Any], *, stem_name: str, config: AuditGuidedDebleedConfig) -> list[dict[str, Any]]:
    stem_report = next((stem for stem in report.get("stems", []) if stem.get("stem") == stem_name), None)
    if stem_report is None:
        raise ValueError(f"Audit report has no stem: {stem_name}")
    micro = stem_report.get("micro_quality_windows", {})
    seen: set[tuple[float, float, str]] = set()
    selected: list[dict[str, Any]] = []
    groups = [
        micro.get("worst_windows", []),
        micro.get("phrase_level_windows", []),
        *micro.get("worst_windows_by_time_bucket", {}).values(),
        *micro.get("phrase_level_windows_by_time_bucket", {}).values(),
        *micro.get("worst_windows_by_residual_leakage_source", {}).values(),
        *micro.get("worst_windows_by_leakage_source", {}).values(),
        *micro.get("worst_phrase_windows_by_source", {}).values(),
    ]
    for group in groups:
        for window in group:
            if not window_has_repair_label(window):
                continue
            source_name, source_reason = candidate_source_for_window(window, config)
            if source_name is None:
                continue
            if not repair_window_allowed(source_name, source_reason, config):
                continue
            key = (float(window["start_sec"]), float(window["end_sec"]), source_name)
            if key in seen:
                continue
            seen.add(key)
            selected.append({**window, "repair_source": source_name, "repair_source_reason": source_reason})
    return sorted(selected, key=lambda item: (float(item["start_sec"]), float(item["end_sec"]), str(item["repair_source"])))


def repair_window_weight(frame_count: int, context_frames: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float64)
    if context_frames <= 0:
        return np.ones(frame_count, dtype=np.float64)
    fade_frames = min(context_frames, frame_count // 2)
    if fade_frames <= 0:
        return np.ones(frame_count, dtype=np.float64)
    weight = np.ones(frame_count, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_frames, endpoint=True))
    weight[:fade_frames] = ramp
    weight[-fade_frames:] = ramp[::-1]
    return weight


def projection_gain(reference: np.ndarray, estimate: np.ndarray, max_gain: float) -> float:
    denominator = float(np.dot(reference, reference) + EPSILON)
    gain = float(np.dot(estimate, reference) / denominator)
    return float(np.clip(gain, -max_gain, max_gain))


def phrase_projection_source_names(leakage_sources: dict[str, np.ndarray], config: AuditGuidedDebleedConfig) -> list[str]:
    allowed_sources = set(config.allowed_repair_sources)
    source_names = []
    for source_name in sorted(leakage_sources):
        if source_name in UNAVAILABLE_REPAIR_SOURCES:
            continue
        if allowed_sources and source_name not in allowed_sources:
            continue
        source_names.append(source_name)
    return source_names


def multi_source_projection_correction(
    *,
    residual_segment: np.ndarray,
    source_segments: dict[str, np.ndarray],
    config: AuditGuidedDebleedConfig,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    if not source_segments:
        return np.zeros_like(residual_segment), {}
    correction = np.zeros_like(residual_segment)
    source_gains: dict[str, list[float]] = {source_name: [] for source_name in source_segments}
    ordered_names = list(source_segments)
    for channel_index in range(residual_segment.shape[1]):
        design_matrix = np.stack(
            [source_segments[source_name][:, channel_index] for source_name in ordered_names],
            axis=1,
        )
        gains, *_ = np.linalg.lstsq(design_matrix, residual_segment[:, channel_index], rcond=None)
        gains = np.clip(gains, -config.max_projection_gain, config.max_projection_gain)
        for source_index, source_name in enumerate(ordered_names):
            gain = float(gains[source_index]) * config.phrase_projection_gain_multiplier
            gain = float(np.clip(gain, -config.max_projection_gain, config.max_projection_gain))
            source_gains[source_name].append(gain)
            correction[:, channel_index] += gain * source_segments[source_name][:, channel_index]
    correction *= config.repair_strength
    residual_norm = float(np.linalg.norm(residual_segment))
    correction_norm = float(np.linalg.norm(correction))
    max_correction_norm = residual_norm * config.max_multi_source_correction_ratio
    if correction_norm > max_correction_norm > EPSILON:
        correction *= max_correction_norm / correction_norm
    return correction, source_gains


def apply_audit_guided_residual_debleed(
    *,
    candidate_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    leakage_sources: dict[str, np.ndarray],
    sample_rate: int,
    audit_windows: list[dict[str, Any]],
    config: AuditGuidedDebleedConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_count = min(candidate_audio.shape[0], target_reference_audio.shape[0])
    for source_name, source_audio in leakage_sources.items():
        frame_count = min(frame_count, source_audio.shape[0])
        if source_audio.shape[1] != candidate_audio.shape[1]:
            raise ValueError(f"channel-count mismatch for leakage source {source_name}: {source_audio.shape[1]}")
    candidate = candidate_audio[:frame_count].astype(np.float64, copy=True)
    target_reference = target_reference_audio[:frame_count].astype(np.float64, copy=False)
    aligned_sources = {
        source_name: source_audio[:frame_count].astype(np.float64, copy=False)
        for source_name, source_audio in leakage_sources.items()
    }
    correction_sum = np.zeros_like(candidate, dtype=np.float64)
    correction_weight = np.zeros((frame_count, 1), dtype=np.float64)
    applied_windows: list[dict[str, Any]] = []
    context_frames = int(round((config.context_ms / 1000.0) * sample_rate))
    for window in audit_windows:
        source_name = str(window["repair_source"])
        source_audio = aligned_sources.get(source_name)
        if source_audio is None:
            continue
        start_frame = max(0, int(math.floor(float(window["start_sec"]) * sample_rate)) - context_frames)
        end_frame = min(frame_count, int(math.ceil(float(window["end_sec"]) * sample_rate)) + context_frames)
        if end_frame <= start_frame:
            continue
        candidate_segment = candidate[start_frame:end_frame]
        target_segment = target_reference[start_frame:end_frame]
        residual_segment = candidate_segment - target_segment
        source_segment = source_audio[start_frame:end_frame]
        window_gain = repair_window_weight(end_frame - start_frame, context_frames)[:, np.newaxis]
        channel_gains: list[float] = []
        multi_source_gains: dict[str, list[float]] = {}
        if (
            config.phrase_multi_source_projection
            and str(window["repair_source_reason"]) == "phrase_reference_projection"
        ):
            source_names = phrase_projection_source_names(aligned_sources, config)
            source_segments = {
                name: aligned_sources[name][start_frame:end_frame]
                for name in source_names
            }
            correction, multi_source_gains = multi_source_projection_correction(
                residual_segment=residual_segment,
                source_segments=source_segments,
                config=config,
            )
            channel_gains = [
                float(max((abs(gain) for gains in multi_source_gains.values() for gain in gains), default=0.0))
            ]
        else:
            correction = np.zeros_like(candidate_segment)
            for channel_index in range(candidate_segment.shape[1]):
                gain = projection_gain(
                    source_segment[:, channel_index],
                    residual_segment[:, channel_index],
                    config.max_projection_gain,
                )
                if str(window["repair_source_reason"]) == "phrase_reference_projection":
                    gain *= config.phrase_projection_gain_multiplier
                    gain = float(np.clip(gain, -config.max_projection_gain, config.max_projection_gain))
                channel_gains.append(gain)
                correction[:, channel_index] = config.repair_strength * gain * source_segment[:, channel_index]
        correction_sum[start_frame:end_frame] += correction * window_gain
        correction_weight[start_frame:end_frame] += window_gain
        applied_window = {
            "start_sec": float(window["start_sec"]),
            "end_sec": float(window["end_sec"]),
            "source": source_name,
            "source_reason": str(window["repair_source_reason"]),
            "issue_labels": list(window.get("issue_labels", [])),
            "channel_projection_gains": [round(float(gain), 9) for gain in channel_gains],
        }
        if multi_source_gains:
            applied_window["multi_source_projection_gains"] = {
                name: [round(float(gain), 9) for gain in gains]
                for name, gains in multi_source_gains.items()
            }
        applied_windows.append(applied_window)
    normalized_weight = np.maximum(correction_weight, 1.0)
    output = candidate - (correction_sum / normalized_weight)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    clipped = False
    if peak > config.max_peak:
        output *= config.max_peak / peak
        clipped = True
    diagnostics = {
        "selected_window_count": len(audit_windows),
        "applied_window_count": len(applied_windows),
        "applied_windows": applied_windows,
        "peak_before_limit": round(peak, 9),
        "peak_limited": clipped,
    }
    return output.astype(np.float32), diagnostics


def build_audit_guided_residual_debleed(
    *,
    candidate_drum_path: Path,
    target_reference_path: Path,
    audit_report_path: Path,
    leakage_source_paths: dict[str, Path],
    output_dir: Path,
    listen_dir: Path | None,
    output_label: str,
    input_label: str,
    stem_name: str,
    config: AuditGuidedDebleedConfig,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    validate_leakage_source_names(leakage_source_paths, stem_name=stem_name)
    loaded: dict[str, tuple[np.ndarray, int]] = {
        "candidate": read_audio(candidate_drum_path),
        "target_reference": read_audio(target_reference_path),
    }
    for source_name, source_path in leakage_source_paths.items():
        loaded[f"source:{source_name}"] = read_audio(source_path)
    sample_rate, channel_count = ensure_same_layout(loaded)
    frame_count = min(audio.shape[0] for audio, _ in loaded.values())
    candidate_audio = fit_length(loaded["candidate"][0], frame_count)
    target_reference_audio = fit_length(loaded["target_reference"][0], frame_count)
    leakage_sources = {
        source_name.removeprefix("source:"): fit_length(audio, frame_count)
        for source_name, (audio, _) in loaded.items()
        if source_name.startswith("source:")
    }
    audit_report = load_json(audit_report_path)
    audit_windows = iter_audit_windows(audit_report, stem_name=stem_name, config=config)
    output_audio, diagnostics = apply_audit_guided_residual_debleed(
        candidate_audio=candidate_audio,
        target_reference_audio=target_reference_audio,
        leakage_sources=leakage_sources,
        sample_rate=sample_rate,
        audit_windows=audit_windows,
        config=config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_label
    input_copy_path = output_dir / input_label
    write_flac(input_copy_path, candidate_audio, sample_rate)
    write_flac(output_path, output_audio, sample_rate)
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        write_flac(listen_dir / "01_input_drums.flac", candidate_audio, sample_rate)
        write_flac(listen_dir / "02_audit_guided_residual_debleed.flac", output_audio, sample_rate)
        write_json(
            listen_dir / "candidate_map.json",
            {
                "01_input_drums.flac": str(input_copy_path),
                "02_audit_guided_residual_debleed.flac": str(output_path),
            },
        )
    manifest = {
        "schema_version": 1,
        "model_id": "audit_guided_residual_debleed",
        "model_name": "Audit-guided residual reference debleed",
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": True,
            "target_reference_used_at_application": True,
            "application_reference_free": False,
            "delivery_allowed_without_reference": False,
            "no_reference_production_claim": False,
            "delivery_allowed_without_quality_gate": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "route": {
            "type": "reference_assisted_audit_guided_residual_debleed",
            "audit_report_path": str(audit_report_path),
            "repair_labels": sorted(REPAIR_LABELS),
        },
        "inputs": {
            "candidate_drum_path": str(candidate_drum_path),
            "target_reference_path": str(target_reference_path),
            "leakage_source_paths": {name: str(path) for name, path in leakage_source_paths.items()},
        },
        "config": config.__dict__,
        "diagnostics": {
            **diagnostics,
            "sample_rate": sample_rate,
            "channels": channel_count,
            "aligned_frames": frame_count,
        },
        "stems": {
            stem_name: {
                "path": str(output_path),
                "sha256": sha256(output_path),
                "source_candidate_path": str(candidate_drum_path),
            }
        },
    }
    write_json(output_dir / "stem_manifest.json", manifest)
    write_json(output_dir / "drum_audit_guided_residual_debleed_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply reference-assisted audit-guided residual debleed to DRUM.")
    parser.add_argument("candidate_drum_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-reference-path", required=True, type=Path)
    parser.add_argument("--audit-report-path", required=True, type=Path)
    parser.add_argument("--leakage-source", action="append", default=[], help="Named source in NAME=PATH form.")
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--output-label", default="02_audit_guided_residual_debleed.flac")
    parser.add_argument("--input-label", default="01_input_drums.flac")
    parser.add_argument("--stem-name", default="drums")
    parser.add_argument("--repair-strength", type=float, default=AuditGuidedDebleedConfig.repair_strength)
    parser.add_argument("--max-projection-gain", type=float, default=AuditGuidedDebleedConfig.max_projection_gain)
    parser.add_argument(
        "--min-residual-leakage-strength",
        type=float,
        default=AuditGuidedDebleedConfig.min_residual_leakage_strength,
    )
    parser.add_argument(
        "--min-candidate-leakage-strength",
        type=float,
        default=AuditGuidedDebleedConfig.min_candidate_leakage_strength,
    )
    parser.add_argument(
        "--min-phrase-leakage-strength",
        type=float,
        default=AuditGuidedDebleedConfig.min_phrase_leakage_strength,
    )
    parser.add_argument(
        "--phrase-projection-gain-multiplier",
        type=float,
        default=AuditGuidedDebleedConfig.phrase_projection_gain_multiplier,
    )
    parser.add_argument(
        "--phrase-multi-source-projection",
        action="store_true",
        help="Use all allowed leakage sources in a phrase-window least-squares projection.",
    )
    parser.add_argument(
        "--max-multi-source-correction-ratio",
        type=float,
        default=AuditGuidedDebleedConfig.max_multi_source_correction_ratio,
        help="Limit phrase multi-source correction norm relative to residual norm.",
    )
    parser.add_argument(
        "--allowed-repair-reason",
        action="append",
        default=[],
        help="Restrict repair to a repair reason. Repeatable. Empty means all reasons.",
    )
    parser.add_argument(
        "--allowed-repair-source",
        action="append",
        default=[],
        help="Restrict repair to a leakage source name. Repeatable. Empty means all sources.",
    )
    parser.add_argument("--context-ms", type=float, default=AuditGuidedDebleedConfig.context_ms)
    parser.add_argument("--max-peak", type=float, default=AuditGuidedDebleedConfig.max_peak)
    return parser.parse_args()


def main() -> None:
    require_docker_runtime(
        tool_name="drum_audit_guided_residual_debleed.py",
        entrypoint_path="/app/tools/drum_audit_guided_residual_debleed.py",
    )
    args = parse_args()
    leakage_sources = dict(parse_named_path(raw_value) for raw_value in args.leakage_source)
    if not leakage_sources:
        raise ValueError("At least one --leakage-source NAME=PATH is required")
    manifest = build_audit_guided_residual_debleed(
        candidate_drum_path=args.candidate_drum_path,
        target_reference_path=args.target_reference_path,
        audit_report_path=args.audit_report_path,
        leakage_source_paths=leakage_sources,
        output_dir=args.output_dir,
        listen_dir=args.listen_dir,
        output_label=str(args.output_label),
        input_label=str(args.input_label),
        stem_name=str(args.stem_name),
        config=AuditGuidedDebleedConfig(
            repair_strength=float(args.repair_strength),
            max_projection_gain=float(args.max_projection_gain),
            min_residual_leakage_strength=float(args.min_residual_leakage_strength),
            min_candidate_leakage_strength=float(args.min_candidate_leakage_strength),
            min_phrase_leakage_strength=float(args.min_phrase_leakage_strength),
            phrase_projection_gain_multiplier=float(args.phrase_projection_gain_multiplier),
            phrase_multi_source_projection=bool(args.phrase_multi_source_projection),
            max_multi_source_correction_ratio=float(args.max_multi_source_correction_ratio),
            allowed_repair_reasons=tuple(str(value) for value in args.allowed_repair_reason),
            allowed_repair_sources=tuple(str(value) for value in args.allowed_repair_source),
            context_ms=float(args.context_ms),
            max_peak=float(args.max_peak),
        ),
        command_args=sys.argv[1:],
    )
    print(json.dumps({"output": manifest["stems"][str(args.stem_name)], "diagnostics": manifest["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
