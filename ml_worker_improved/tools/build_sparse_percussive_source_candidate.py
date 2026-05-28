#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from build_percussive_event_dictionary_candidate import (  # noqa: E402
    evaluate_percussive_aux_sanity,
)
from build_percussive_fx_aux_candidate import (  # noqa: E402
    EPSILON,
    TARGET_STEM,
    _fit_audio,
    _load_json,
    _peak_dbfs,
    _read_audio,
    _rms_dbfs,
    _safe_listen_path,
    _sha256_file,
    _stem_path_from_manifest,
    _validate_no_reference_manifest,
    _write_flac,
    _write_json,
)


DEFAULT_ROUTE_NAME = "sparse_percussive_source_v1"
AUX_STEM = "sparse_percussive_source_aux"


@dataclass(frozen=True)
class SparsePercussiveSourceConfig:
    frame_size: int = 2048
    hop_size: int = 512
    source_flux_percentile: float = 86.0
    min_event_flatness: float = 0.12
    min_high_band_ratio: float = 0.18
    activity_floor_ratio: float = 0.015
    max_raw_event_frames: int = 8
    pre_roll_frames: int = 1
    tail_frames: int = 2
    low_cut_hz: float = 120.0
    high_band_floor_hz: float = 1800.0
    aux_gain: float = 0.75
    max_aux_rms_ratio: float = 0.3
    limiter_peak: float = 0.999
    max_aux_active_fraction: float = 0.22
    max_aux_low_mid_to_high_ratio: float = 1.8
    harmonic_active_fraction_floor: float = 0.12


def read_json(path: Path) -> dict[str, Any]:
    return _load_json(path)


def _resolve_base_candidate_audio(base_manifest_path: Path, base_manifest: dict[str, Any]) -> Path:
    stem_path = _stem_path_from_manifest(base_manifest, base_manifest_path, TARGET_STEM)
    if stem_path is None:
        raise ValueError(f"base manifest does not contain required stem: {TARGET_STEM}")
    return stem_path


def _spectral_flatness(magnitude: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(magnitude + EPSILON))) / (np.mean(magnitude) + EPSILON))


def _frame_source_features(
    source_audio: np.ndarray,
    *,
    sample_rate: int,
    config: SparsePercussiveSourceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    mono = np.mean(source_audio, axis=1).astype(np.float32)
    frame_size = int(config.frame_size)
    hop_size = int(config.hop_size)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if mono.shape[0] < frame_size:
        mono = np.pad(mono, (0, frame_size - mono.shape[0]))
    frame_count = 1 + max(0, (mono.shape[0] - frame_size) // hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate)).astype(np.float32)
    high_band = frequencies >= float(config.high_band_floor_hz)
    low_mid_band = (frequencies >= 160.0) & (frequencies <= min(4200.0, float(sample_rate) * 0.48))

    rms_values: list[float] = []
    flatness_values: list[float] = []
    flux_values: list[float] = []
    high_band_ratio_values: list[float] = []
    previous_magnitude: np.ndarray | None = None
    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = mono[start : start + frame_size]
        if frame.shape[0] < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.shape[0]))
        magnitude = np.abs(np.fft.rfft(frame * window)).astype(np.float32) + EPSILON
        rms_values.append(float(np.sqrt(np.mean(frame * frame) + EPSILON)))
        flatness_values.append(_spectral_flatness(magnitude))
        if previous_magnitude is None:
            flux_values.append(0.0)
        else:
            positive_delta = np.maximum(magnitude - previous_magnitude, 0.0)
            flux_values.append(float(np.sum(positive_delta) / (np.sum(magnitude) + EPSILON)))
        high_energy = float(np.mean(magnitude[high_band])) if np.any(high_band) else 0.0
        low_mid_energy = float(np.mean(magnitude[low_mid_band])) if np.any(low_mid_band) else 0.0
        high_band_ratio_values.append(float(high_energy / (high_energy + low_mid_energy + EPSILON)))
        previous_magnitude = magnitude
    return (
        np.asarray(rms_values, dtype=np.float32),
        {
            "flatness": np.asarray(flatness_values, dtype=np.float32),
            "flux": np.asarray(flux_values, dtype=np.float32),
            "high_band_ratio": np.asarray(high_band_ratio_values, dtype=np.float32),
        },
    )


def _short_event_gate(raw_gate: np.ndarray, *, max_raw_event_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    kept = np.zeros_like(raw_gate, dtype=bool)
    dropped_long_event_count = 0
    kept_event_count = 0
    index = 0
    while index < raw_gate.shape[0]:
        if not raw_gate[index]:
            index += 1
            continue
        start = index
        while index < raw_gate.shape[0] and raw_gate[index]:
            index += 1
        end = index
        length = end - start
        if length <= int(max_raw_event_frames):
            kept[start:end] = True
            kept_event_count += 1
        else:
            dropped_long_event_count += 1
    return kept, {
        "raw_event_count": int(_count_true_runs(raw_gate)),
        "kept_event_count": kept_event_count,
        "dropped_long_event_count": dropped_long_event_count,
    }


def _count_true_runs(gate: np.ndarray) -> int:
    count = 0
    in_run = False
    for value in gate:
        if bool(value) and not in_run:
            count += 1
            in_run = True
        elif not bool(value):
            in_run = False
    return count


def _expand_frame_gate(frame_gate: np.ndarray, *, pre_roll_frames: int, tail_frames: int) -> np.ndarray:
    expanded = np.zeros_like(frame_gate, dtype=bool)
    for frame_index in np.flatnonzero(frame_gate):
        start = max(0, int(frame_index) - int(pre_roll_frames))
        end = min(frame_gate.shape[0], int(frame_index) + int(tail_frames) + 1)
        expanded[start:end] = True
    return expanded


def build_source_event_gate(
    source_audio: np.ndarray,
    *,
    sample_rate: int,
    config: SparsePercussiveSourceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rms_values, features = _frame_source_features(source_audio, sample_rate=sample_rate, config=config)
    flux_values = features["flux"]
    flatness_values = features["flatness"]
    high_band_ratio_values = features["high_band_ratio"]
    active_floor = max(float(np.max(rms_values)) * float(config.activity_floor_ratio), 1.0e-5)
    flux_threshold = float(np.percentile(flux_values, float(config.source_flux_percentile)))
    activity_gate = rms_values >= active_floor
    transient_gate = flux_values >= flux_threshold
    texture_gate = flatness_values >= float(config.min_event_flatness)
    high_band_gate = high_band_ratio_values >= float(config.min_high_band_ratio)
    raw_gate = activity_gate & transient_gate & texture_gate & high_band_gate
    short_gate, run_metrics = _short_event_gate(raw_gate, max_raw_event_frames=int(config.max_raw_event_frames))
    expanded_gate = _expand_frame_gate(
        short_gate,
        pre_roll_frames=int(config.pre_roll_frames),
        tail_frames=int(config.tail_frames),
    )
    return expanded_gate, {
        "frame_count": int(raw_gate.shape[0]),
        "raw_frame_count": int(np.count_nonzero(raw_gate)),
        "kept_frame_count": int(np.count_nonzero(short_gate)),
        "expanded_frame_count": int(np.count_nonzero(expanded_gate)),
        "active_floor": round(active_floor, 9),
        "flux_threshold": round(flux_threshold, 9),
        "mean_flatness": round(float(np.mean(flatness_values)), 9),
        "mean_high_band_ratio": round(float(np.mean(high_band_ratio_values)), 9),
        **run_metrics,
    }


def _fit_frame_gate(frame_gate: np.ndarray, frame_count: int) -> np.ndarray:
    if frame_gate.shape[0] == frame_count:
        return frame_gate
    if frame_gate.shape[0] > frame_count:
        return frame_gate[:frame_count]
    return np.pad(frame_gate, (0, frame_count - frame_gate.shape[0]), mode="constant", constant_values=False)


def _fit_vector(audio: np.ndarray, frame_count: int) -> np.ndarray:
    if audio.shape[0] == frame_count:
        return audio
    if audio.shape[0] > frame_count:
        return audio[:frame_count]
    return np.pad(audio, (0, frame_count - audio.shape[0]), mode="constant")


def render_sparse_percussive_aux(
    source_audio: np.ndarray,
    base_audio: np.ndarray,
    *,
    sample_rate: int,
    config: SparsePercussiveSourceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    event_gate, source_event_metrics = build_source_event_gate(source_audio, sample_rate=sample_rate, config=config)
    residual_audio = source_audio - base_audio
    aux_channels: list[np.ndarray] = []
    for channel_index in range(residual_audio.shape[1]):
        frequencies, _, spectrum = signal.stft(
            residual_audio[:, channel_index],
            fs=float(sample_rate),
            window="hann",
            nperseg=int(config.frame_size),
            noverlap=int(config.frame_size) - int(config.hop_size),
            boundary="zeros",
            padded=True,
        )
        frame_gate = _fit_frame_gate(event_gate, spectrum.shape[1])
        band_mask = np.ones((spectrum.shape[0], 1), dtype=np.float32)
        band_mask[frequencies < float(config.low_cut_hz), :] = 0.05
        gated_spectrum = spectrum * band_mask * frame_gate[None, :]
        _, channel_audio = signal.istft(
            gated_spectrum,
            fs=float(sample_rate),
            window="hann",
            nperseg=int(config.frame_size),
            noverlap=int(config.frame_size) - int(config.hop_size),
            input_onesided=True,
            boundary=True,
        )
        aux_channels.append(channel_audio.astype(np.float32))
    aux_audio = np.column_stack([_fit_vector(channel, residual_audio.shape[0]) for channel in aux_channels]).astype(np.float32)
    aux_audio *= float(config.aux_gain)
    return aux_audio, {
        "source_event_metrics": source_event_metrics,
        "render_method": "source_transient_islands_minus_base",
    }


def _limit_aux_against_base(
    base_audio: np.ndarray,
    aux_audio: np.ndarray,
    config: SparsePercussiveSourceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_rms = float(np.sqrt(np.mean(base_audio * base_audio) + EPSILON))
    aux_rms = float(np.sqrt(np.mean(aux_audio * aux_audio) + EPSILON))
    max_aux_rms = base_rms * float(config.max_aux_rms_ratio)
    rms_gain = 1.0
    if aux_rms > max_aux_rms > 0.0:
        rms_gain = float(max_aux_rms / aux_rms)
        aux_audio = aux_audio * rms_gain
    rendered = base_audio + aux_audio
    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    peak_gain = 1.0
    if peak > float(config.limiter_peak) > 0.0:
        peak_gain = float(config.limiter_peak) / peak
        aux_audio = aux_audio * peak_gain
        rendered = base_audio + aux_audio
    return aux_audio.astype(np.float32), {
        "base_audio_modified": False,
        "base_rms_dbfs": _rms_dbfs(base_audio),
        "raw_aux_rms_dbfs": _rms_dbfs(aux_audio / max(peak_gain * rms_gain, EPSILON)),
        "rms_gain": round(rms_gain, 9),
        "peak_gain": round(peak_gain, 9),
        "rendered_peak_dbfs": _peak_dbfs(rendered),
    }


def _delivery_report(*, aux_sanity: dict[str, Any], source_event_metrics: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    reject_reasons = list(aux_sanity.get("reject_reasons", []))
    if int(source_event_metrics.get("kept_event_count", 0)) <= 0:
        reject_reasons.append("no_sparse_percussive_source_events")
    delivery_allowed = aux_sanity.get("status") == "accepted" and not reject_reasons
    if delivery_allowed:
        return True, {
            "status": "requires_objective_quality_gate",
            "reason": "objective_quality_gate_required",
            "reject_reasons": [],
            "required_gate": "run_ground_truth_quality_gate.py",
        }
    return delivery_allowed, {
        "status": "rejected",
        "reason": "sparse_percussive_source_sanity_failed",
        "reject_reasons": reject_reasons,
    }


def _ralph_ai_delivery_contract(*, delivery_allowed: bool, delivery: dict[str, Any]) -> dict[str, Any]:
    if delivery_allowed:
        return {
            "method": "RALPH-AI",
            "status": "requires_objective_gate",
            "presentation_status": "diagnostic_requires_objective_gate",
            "result_delivery_allowed": False,
            "diagnostic_artifacts_allowed": True,
            "rejects_best_of_bad_candidate": True,
            "reason": "objective_quality_gate_required",
            "required_next_action": "run_objective_and_exact_reference_gate",
        }
    return {
        "method": "RALPH-AI",
        "status": "blocked",
        "presentation_status": "blocked_from_delivery",
        "result_delivery_allowed": False,
        "diagnostic_artifacts_allowed": True,
        "rejects_best_of_bad_candidate": True,
        "reason": str(delivery.get("reason") or "delivery_gate_failed"),
        "required_next_action": "iterate_until_exact_reference_delivery_passes",
    }


def _copy_listen_file(*, target_path: Path, aux_path: Path, listen_dir: Path, listen_label: str) -> dict[str, Any]:
    listen_dir = listen_dir.resolve()
    listen_path = _safe_listen_path(listen_dir, listen_label)
    aux_listen_path = _safe_listen_path(listen_dir, "02_sparse_percussive_source_aux.flac")
    listen_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_path, listen_path)
    shutil.copy2(aux_path, aux_listen_path)
    return {
        "status": "diagnostic_published",
        "reason": "objective_quality_gate_required",
        "listen_dir": str(listen_dir),
        "files": {
            listen_label: str(listen_path),
            "02_sparse_percussive_source_aux.flac": str(aux_listen_path),
        },
    }


def _listen_blocked_report(*, listen_dir: Path, delivery: dict[str, Any], aux_sanity: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": str(delivery.get("reason") or "sparse_percussive_source_sanity_failed"),
        "listen_dir": str(listen_dir.resolve()),
        "files": {},
        "reject_reasons": list(delivery.get("reject_reasons", [])),
        "aux_sanity": aux_sanity,
    }


def build_sparse_percussive_source_candidate(
    *,
    base_manifest_path: Path,
    source_audio_path: Path,
    output_dir: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
    config: SparsePercussiveSourceConfig = SparsePercussiveSourceConfig(),
) -> dict[str, Any]:
    started_at = time.perf_counter()
    base_manifest_path = base_manifest_path.resolve()
    source_audio_path = source_audio_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_audio_path.exists():
        raise FileNotFoundError(f"source audio is missing: {source_audio_path}")

    base_manifest = _load_json(base_manifest_path)
    base_lineage_report = _validate_no_reference_manifest(
        base_manifest,
        manifest_path=base_manifest_path,
        role="base",
    )
    base_candidate_path = _resolve_base_candidate_audio(base_manifest_path, base_manifest)
    base_audio, sample_rate = _read_audio(base_candidate_path)
    source_audio, source_sample_rate = _read_audio(source_audio_path)
    if source_sample_rate != sample_rate:
        raise ValueError(f"sample-rate mismatch: source={source_sample_rate}, base={sample_rate}")
    source_audio = _fit_audio(source_audio, base_audio.shape[0], base_audio.shape[1])
    raw_aux, render_metrics = render_sparse_percussive_aux(
        source_audio,
        base_audio,
        sample_rate=sample_rate,
        config=config,
    )
    aux_audio, limiter_metrics = _limit_aux_against_base(base_audio, raw_aux, config)
    sanity_config = config
    aux_sanity = evaluate_percussive_aux_sanity(aux_audio, sample_rate=sample_rate, config=sanity_config)
    source_event_metrics = render_metrics["source_event_metrics"]
    delivery_allowed, delivery = _delivery_report(aux_sanity=aux_sanity, source_event_metrics=source_event_metrics)
    ralph_ai_delivery_contract = _ralph_ai_delivery_contract(
        delivery_allowed=delivery_allowed,
        delivery=delivery,
    )
    rendered = np.clip(base_audio + aux_audio, -1.0, 1.0).astype(np.float32)

    target_path = output_dir / f"{TARGET_STEM}.flac"
    aux_path = output_dir / "estimated_sparse_percussive_source_aux.flac"
    _write_flac(target_path, rendered, sample_rate)
    _write_flac(aux_path, aux_audio, sample_rate)
    manifest_path = output_dir / "stem_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": str(route_name),
        "model_name": str(route_name),
        "contract": {
            "benchmark_compatible": True,
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": delivery_allowed,
            "reference_teacher_pack": False,
            "registered_model_candidate": False,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "no_reference_production_claim": False,
            "quality_gate_required": True,
            "delivery_blocked_reason": None if delivery_allowed else str(delivery["reason"]),
        },
        "delivery": delivery,
        "presentation_status": ralph_ai_delivery_contract["presentation_status"],
        "ralph_ai_delivery_contract": ralph_ai_delivery_contract,
        "quality_gate": {
            "status": "required" if delivery_allowed else "blocked_by_precheck",
            "required_tool": "ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
            "candidate_can_be_promoted_without_objective_gate": False,
        },
        "pipeline": {
            "route": str(route_name),
            "family": "sparse_percussive_source",
            "base_manifest": str(base_manifest_path),
            "base_candidate_path": str(base_candidate_path),
            "source_audio_path": str(source_audio_path),
            "source_audio_role": "full_mix_sparse_percussive_source",
            "aux_source_role": "source_transient_islands_minus_base",
            "reference_audio_used": False,
            "track_specific_state_used": False,
            "base_preserving_addition": True,
            "base_audio_unmodified": True,
            "deterministic": True,
            "config": asdict(config),
            "base_provenance_contract": base_lineage_report,
            "source_event_metrics": source_event_metrics,
            "render_metrics": render_metrics,
            "limiter_metrics": limiter_metrics,
            "aux_sanity": aux_sanity,
        },
        "stems": {
            TARGET_STEM: {
                "path": str(target_path),
                "stem_type": TARGET_STEM,
                "source_model_stem": TARGET_STEM,
                "source_audio_role": "base_plus_sparse_percussive_source_aux",
                "derived": False,
                "sha256": _sha256_file(target_path),
                "rms_dbfs": _rms_dbfs(rendered),
                "peak_dbfs": _peak_dbfs(rendered),
            },
            AUX_STEM: {
                "path": str(aux_path),
                "stem_type": AUX_STEM,
                "source_audio_role": "source_transient_islands_minus_base",
                "derived": False,
                "sha256": _sha256_file(aux_path),
                "rms_dbfs": _rms_dbfs(aux_audio),
                "peak_dbfs": _peak_dbfs(aux_audio),
            },
        },
        "manifest_path": str(manifest_path),
        "execution": {
            "tool": "ml_worker_improved/tools/build_sparse_percussive_source_candidate.py",
            "mode": "deterministic_sparse_percussive_source",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a no-reference sparse percussive-source candidate from full-mix transient islands.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--frame-size", type=int, default=SparsePercussiveSourceConfig.frame_size)
    parser.add_argument("--hop-size", type=int, default=SparsePercussiveSourceConfig.hop_size)
    parser.add_argument("--source-flux-percentile", type=float, default=SparsePercussiveSourceConfig.source_flux_percentile)
    parser.add_argument("--min-event-flatness", type=float, default=SparsePercussiveSourceConfig.min_event_flatness)
    parser.add_argument("--min-high-band-ratio", type=float, default=SparsePercussiveSourceConfig.min_high_band_ratio)
    parser.add_argument("--activity-floor-ratio", type=float, default=SparsePercussiveSourceConfig.activity_floor_ratio)
    parser.add_argument("--max-raw-event-frames", type=int, default=SparsePercussiveSourceConfig.max_raw_event_frames)
    parser.add_argument("--pre-roll-frames", type=int, default=SparsePercussiveSourceConfig.pre_roll_frames)
    parser.add_argument("--tail-frames", type=int, default=SparsePercussiveSourceConfig.tail_frames)
    parser.add_argument("--low-cut-hz", type=float, default=SparsePercussiveSourceConfig.low_cut_hz)
    parser.add_argument("--high-band-floor-hz", type=float, default=SparsePercussiveSourceConfig.high_band_floor_hz)
    parser.add_argument("--aux-gain", type=float, default=SparsePercussiveSourceConfig.aux_gain)
    parser.add_argument("--max-aux-rms-ratio", type=float, default=SparsePercussiveSourceConfig.max_aux_rms_ratio)
    parser.add_argument("--limiter-peak", type=float, default=SparsePercussiveSourceConfig.limiter_peak)
    parser.add_argument("--max-aux-active-fraction", type=float, default=SparsePercussiveSourceConfig.max_aux_active_fraction)
    parser.add_argument("--max-aux-low-mid-to-high-ratio", type=float, default=SparsePercussiveSourceConfig.max_aux_low_mid_to_high_ratio)
    parser.add_argument("--harmonic-active-fraction-floor", type=float, default=SparsePercussiveSourceConfig.harmonic_active_fraction_floor)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-label", default="01_sparse_percussive_source.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_sparse_percussive_source_candidate.py",
        entrypoint_path="/app/tools/build_sparse_percussive_source_candidate.py",
    )
    args = parse_args()
    config = SparsePercussiveSourceConfig(
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        source_flux_percentile=float(args.source_flux_percentile),
        min_event_flatness=float(args.min_event_flatness),
        min_high_band_ratio=float(args.min_high_band_ratio),
        activity_floor_ratio=float(args.activity_floor_ratio),
        max_raw_event_frames=int(args.max_raw_event_frames),
        pre_roll_frames=int(args.pre_roll_frames),
        tail_frames=int(args.tail_frames),
        low_cut_hz=float(args.low_cut_hz),
        high_band_floor_hz=float(args.high_band_floor_hz),
        aux_gain=float(args.aux_gain),
        max_aux_rms_ratio=float(args.max_aux_rms_ratio),
        limiter_peak=float(args.limiter_peak),
        max_aux_active_fraction=float(args.max_aux_active_fraction),
        max_aux_low_mid_to_high_ratio=float(args.max_aux_low_mid_to_high_ratio),
        harmonic_active_fraction_floor=float(args.harmonic_active_fraction_floor),
    )
    manifest = build_sparse_percussive_source_candidate(
        base_manifest_path=args.base_manifest,
        source_audio_path=args.source_audio,
        output_dir=args.output_dir,
        route_name=str(args.route_name),
        config=config,
    )
    if args.listen_dir is not None:
        aux_sanity = manifest.get("pipeline", {}).get("aux_sanity", {})
        if (
            manifest.get("delivery", {}).get("status") == "requires_objective_quality_gate"
            and aux_sanity.get("status") == "accepted"
        ):
            manifest["listen"] = _copy_listen_file(
                target_path=Path(manifest["stems"][TARGET_STEM]["path"]),
                aux_path=Path(manifest["stems"][AUX_STEM]["path"]),
                listen_dir=args.listen_dir,
                listen_label=str(args.listen_label),
            )
        else:
            manifest["listen"] = _listen_blocked_report(
                listen_dir=args.listen_dir,
                delivery=manifest.get("delivery", {}),
                aux_sanity=aux_sanity,
            )
        _write_json(Path(manifest["manifest_path"]), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
