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
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from build_percussive_fx_aux_candidate import (  # noqa: E402
    EPSILON,
    REQUIRED_NO_REFERENCE_CONTRACT,
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


DEFAULT_ROUTE_NAME = "percussive_event_dictionary_v1"
SOURCE_ROLE = "full_mix_minus_base_residual"


@dataclass(frozen=True)
class PercussiveEventDictionaryConfig:
    frame_size: int = 2048
    hop_size: int = 512
    harmonic_time_bins: int = 17
    percussive_frequency_bins: int = 17
    mask_power: float = 2.0
    event_flux_percentile: float = 68.0
    min_event_flatness: float = 0.13
    activity_floor_ratio: float = 0.015
    pre_roll_frames: int = 1
    tail_frames: int = 4
    low_cut_hz: float = 80.0
    aux_gain: float = 0.9
    max_aux_rms_ratio: float = 0.75
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


def _odd_kernel_size(value: int) -> int:
    size = max(3, int(value))
    return size if size % 2 else size + 1


def _expanded_gate(frame_gate: np.ndarray, *, pre_roll_frames: int, tail_frames: int) -> np.ndarray:
    expanded = np.zeros_like(frame_gate, dtype=bool)
    for frame_index in np.flatnonzero(frame_gate):
        start = max(0, int(frame_index) - int(pre_roll_frames))
        end = min(frame_gate.size, int(frame_index) + int(tail_frames) + 1)
        expanded[start:end] = True
    return expanded


def _event_gate_from_residual(residual_audio: np.ndarray, sample_rate: int, config: PercussiveEventDictionaryConfig) -> tuple[np.ndarray, dict[str, Any]]:
    mono = np.mean(residual_audio, axis=1).astype(np.float32)
    _, _, spectrum = signal.stft(
        mono,
        fs=float(sample_rate),
        window="hann",
        nperseg=int(config.frame_size),
        noverlap=int(config.frame_size) - int(config.hop_size),
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum).astype(np.float32)
    frame_energy = np.mean(magnitude * magnitude, axis=0)
    frame_rms = np.sqrt(frame_energy + EPSILON)
    flatness = np.exp(np.mean(np.log(magnitude + EPSILON), axis=0)) / (np.mean(magnitude + EPSILON, axis=0) + EPSILON)
    log_magnitude = np.log1p(magnitude)
    flux = np.zeros(magnitude.shape[1], dtype=np.float32)
    if magnitude.shape[1] > 1:
        flux[1:] = np.mean(np.maximum(0.0, np.diff(log_magnitude, axis=1)), axis=0)
    activity_floor = max(float(np.max(frame_rms)) * float(config.activity_floor_ratio), 1.0e-8)
    flux_threshold = float(np.percentile(flux, float(config.event_flux_percentile))) if flux.size else 0.0
    frame_gate = (
        (frame_rms >= activity_floor)
        & ((flux >= flux_threshold) | (flatness >= float(config.min_event_flatness)))
    )
    expanded = _expanded_gate(
        frame_gate,
        pre_roll_frames=int(config.pre_roll_frames),
        tail_frames=int(config.tail_frames),
    )
    return expanded.astype(np.float32), {
        "frame_count": int(frame_gate.size),
        "detected_frame_count": int(np.sum(frame_gate)),
        "expanded_frame_count": int(np.sum(expanded)),
        "activity_floor": round(activity_floor, 9),
        "flux_threshold": round(flux_threshold, 9),
        "mean_flatness": round(float(np.mean(flatness)) if flatness.size else 0.0, 9),
        "max_flatness": round(float(np.max(flatness)) if flatness.size else 0.0, 9),
    }


def _percussive_mask(magnitude: np.ndarray, sample_rate: int, config: PercussiveEventDictionaryConfig) -> np.ndarray:
    harmonic_bins = _odd_kernel_size(config.harmonic_time_bins)
    percussive_bins = _odd_kernel_size(config.percussive_frequency_bins)
    harmonic = median_filter(magnitude, size=(1, harmonic_bins), mode="nearest")
    percussive = median_filter(magnitude, size=(percussive_bins, 1), mode="nearest")
    power = max(0.5, float(config.mask_power))
    powered_percussive = np.power(percussive + EPSILON, power)
    powered_harmonic = np.power(harmonic + EPSILON, power)
    mask = powered_percussive / (powered_percussive + powered_harmonic + EPSILON)
    if float(config.low_cut_hz) > 0.0 and magnitude.shape[0] > 1:
        frequencies = np.linspace(0.0, sample_rate / 2.0, magnitude.shape[0], dtype=np.float32)
        mask[frequencies < float(config.low_cut_hz), :] *= 0.2
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def extract_percussive_event_aux(
    residual_audio: np.ndarray,
    *,
    sample_rate: int,
    config: PercussiveEventDictionaryConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    event_gate, gate_metrics = _event_gate_from_residual(residual_audio, sample_rate, config)
    aux_channels: list[np.ndarray] = []
    mask_means: list[float] = []
    for channel_index in range(residual_audio.shape[1]):
        _, _, spectrum = signal.stft(
            residual_audio[:, channel_index],
            fs=float(sample_rate),
            window="hann",
            nperseg=int(config.frame_size),
            noverlap=int(config.frame_size) - int(config.hop_size),
            boundary="zeros",
            padded=True,
        )
        magnitude = np.abs(spectrum).astype(np.float32)
        mask = _percussive_mask(magnitude, sample_rate, config)
        if event_gate.shape[0] != mask.shape[1]:
            event_gate = _fit_frame_gate(event_gate, mask.shape[1])
        gated_spectrum = spectrum * mask * event_gate[None, :]
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
        mask_means.append(float(np.mean(mask)))
    aux_audio = np.column_stack([_fit_vector(channel, residual_audio.shape[0]) for channel in aux_channels]).astype(np.float32)
    aux_audio *= float(config.aux_gain)
    return aux_audio, {
        "gate": gate_metrics,
        "mean_percussive_mask": round(float(np.mean(mask_means)) if mask_means else 0.0, 9),
        "render_method": "stft_hpss_event_dictionary",
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


def _limit_aux_against_base(
    base_audio: np.ndarray,
    aux_audio: np.ndarray,
    config: PercussiveEventDictionaryConfig,
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


def _band_mean(magnitude: np.ndarray, frequencies: np.ndarray, low_hz: float, high_hz: float) -> float:
    band = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    if not np.any(band):
        return 0.0
    return float(np.mean(magnitude[band, :]))


def evaluate_percussive_aux_sanity(
    aux_audio: np.ndarray,
    *,
    sample_rate: int,
    config: PercussiveEventDictionaryConfig,
) -> dict[str, Any]:
    mono = aux_audio.mean(axis=1).astype(np.float32) if aux_audio.ndim == 2 else aux_audio.astype(np.float32)
    if mono.size == 0:
        return {
            "status": "accepted",
            "reject_reasons": [],
            "active_fraction": 0.0,
            "low_mid_to_high_ratio": 0.0,
            "rms_dbfs": -120.0,
        }
    rms = float(np.sqrt(np.mean(mono * mono) + EPSILON))
    envelope_window = max(1, int(sample_rate) // 20)
    envelope_kernel = np.ones(envelope_window, dtype=np.float32) / float(envelope_window)
    envelope = np.sqrt(np.convolve(mono * mono, envelope_kernel, mode="same") + EPSILON)
    active_threshold = max(rms * 0.25, 1.0e-5)
    active_fraction = float(np.mean(envelope > active_threshold))

    nperseg = min(2048, max(64, int(mono.size)))
    noverlap = min(nperseg - 1, int(nperseg * 0.75))
    frequencies, _, spectrum = signal.stft(
        mono,
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spectrum).astype(np.float32) + EPSILON
    nyquist = max(float(sample_rate) / 2.0, 1.0)
    low_mid_high = min(4200.0, nyquist * 0.48)
    high_low = min(nyquist * 0.55, max(low_mid_high + 1.0, 5000.0))
    high_high = nyquist * 0.95
    low_mid_mean = _band_mean(magnitude, frequencies, 160.0, low_mid_high)
    high_mean = _band_mean(magnitude, frequencies, high_low, high_high)
    low_mid_to_high_ratio = float(low_mid_mean / max(high_mean, EPSILON))

    reject_reasons: list[str] = []
    if active_fraction > float(config.max_aux_active_fraction):
        reject_reasons.append("aux_not_sparse")
    if (
        active_fraction > float(config.harmonic_active_fraction_floor)
        and low_mid_to_high_ratio > float(config.max_aux_low_mid_to_high_ratio)
    ):
        reject_reasons.append("harmonic_melody_dominant")
    return {
        "status": "rejected" if reject_reasons else "accepted",
        "reject_reasons": reject_reasons,
        "active_fraction": round(active_fraction, 9),
        "active_threshold": round(active_threshold, 9),
        "low_mid_to_high_ratio": round(low_mid_to_high_ratio, 9),
        "rms_dbfs": _rms_dbfs(aux_audio),
        "max_aux_active_fraction": float(config.max_aux_active_fraction),
        "max_aux_low_mid_to_high_ratio": float(config.max_aux_low_mid_to_high_ratio),
        "harmonic_active_fraction_floor": float(config.harmonic_active_fraction_floor),
    }


def _listen_blocked_report(*, listen_dir: Path, aux_sanity: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": "percussive_event_aux_sanity_failed",
        "listen_dir": str(listen_dir.resolve()),
        "files": {},
        "aux_sanity": aux_sanity,
    }


def _copy_listen_file(*, target_path: Path, aux_path: Path, listen_dir: Path, listen_label: str) -> dict[str, Any]:
    listen_dir = listen_dir.resolve()
    listen_path = _safe_listen_path(listen_dir, listen_label)
    aux_listen_path = _safe_listen_path(listen_dir, "02_percussive_event_aux.flac")
    listen_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_path, listen_path)
    shutil.copy2(aux_path, aux_listen_path)
    return {
        "listen_dir": str(listen_dir),
        "files": {
            listen_label: str(listen_path),
            "02_percussive_event_aux.flac": str(aux_listen_path),
        },
    }


def build_percussive_event_dictionary_candidate(
    *,
    base_manifest_path: Path,
    source_audio_path: Path,
    output_dir: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
    config: PercussiveEventDictionaryConfig = PercussiveEventDictionaryConfig(),
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
    residual_audio = source_audio - base_audio
    raw_aux, extraction_metrics = extract_percussive_event_aux(
        residual_audio,
        sample_rate=sample_rate,
        config=config,
    )
    aux_audio, limiter_metrics = _limit_aux_against_base(base_audio, raw_aux, config)
    aux_sanity = evaluate_percussive_aux_sanity(aux_audio, sample_rate=sample_rate, config=config)
    delivery_allowed = aux_sanity["status"] == "accepted"
    rendered = np.clip(base_audio + aux_audio, -1.0, 1.0).astype(np.float32)

    target_path = output_dir / f"{TARGET_STEM}.flac"
    aux_path = output_dir / "estimated_percussive_event_aux.flac"
    residual_path = output_dir / "fullmix_minus_base_residual.flac"
    _write_flac(target_path, rendered, sample_rate)
    _write_flac(aux_path, aux_audio, sample_rate)
    _write_flac(residual_path, residual_audio, sample_rate)
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
            "no_reference_production_claim": delivery_allowed,
            "quality_gate_required": True,
            "delivery_blocked_reason": None if delivery_allowed else "percussive_event_aux_sanity_failed",
        },
        "delivery": {
            "status": "candidate" if delivery_allowed else "rejected",
            "reason": None if delivery_allowed else "percussive_event_aux_sanity_failed",
            "reject_reasons": [] if delivery_allowed else list(aux_sanity["reject_reasons"]),
        },
        "pipeline": {
            "route": str(route_name),
            "family": "percussive_event_dictionary",
            "base_manifest": str(base_manifest_path),
            "base_candidate_path": str(base_candidate_path),
            "source_audio_path": str(source_audio_path),
            "source_audio_role": "full_mix",
            "aux_source_role": SOURCE_ROLE,
            "reference_audio_used": False,
            "track_specific_state_used": False,
            "base_preserving_addition": True,
            "base_audio_unmodified": True,
            "deterministic": True,
            "config": asdict(config),
            "base_provenance_contract": base_lineage_report,
            "extraction_metrics": extraction_metrics,
            "limiter_metrics": limiter_metrics,
            "aux_sanity": aux_sanity,
            "residual_path": str(residual_path),
        },
        "stems": {
            TARGET_STEM: {
                "path": str(target_path),
                "stem_type": TARGET_STEM,
                "source_model_stem": TARGET_STEM,
                "source_audio_role": "base_plus_percussive_event_aux",
                "derived": False,
                "sha256": _sha256_file(target_path),
                "rms_dbfs": _rms_dbfs(rendered),
                "peak_dbfs": _peak_dbfs(rendered),
            },
            "percussive_fx_aux": {
                "path": str(aux_path),
                "stem_type": "percussive_fx_aux",
                "source_audio_role": SOURCE_ROLE,
                "derived": False,
                "sha256": _sha256_file(aux_path),
                "rms_dbfs": _rms_dbfs(aux_audio),
                "peak_dbfs": _peak_dbfs(aux_audio),
            },
        },
        "manifest_path": str(manifest_path),
        "execution": {
            "tool": "ml_worker_improved/tools/build_percussive_event_dictionary_candidate.py",
            "mode": "deterministic_fullmix_percussive_event_dictionary",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a no-reference percussive event dictionary candidate from full-mix residual.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--frame-size", type=int, default=PercussiveEventDictionaryConfig.frame_size)
    parser.add_argument("--hop-size", type=int, default=PercussiveEventDictionaryConfig.hop_size)
    parser.add_argument("--harmonic-time-bins", type=int, default=PercussiveEventDictionaryConfig.harmonic_time_bins)
    parser.add_argument("--percussive-frequency-bins", type=int, default=PercussiveEventDictionaryConfig.percussive_frequency_bins)
    parser.add_argument("--mask-power", type=float, default=PercussiveEventDictionaryConfig.mask_power)
    parser.add_argument("--event-flux-percentile", type=float, default=PercussiveEventDictionaryConfig.event_flux_percentile)
    parser.add_argument("--min-event-flatness", type=float, default=PercussiveEventDictionaryConfig.min_event_flatness)
    parser.add_argument("--activity-floor-ratio", type=float, default=PercussiveEventDictionaryConfig.activity_floor_ratio)
    parser.add_argument("--pre-roll-frames", type=int, default=PercussiveEventDictionaryConfig.pre_roll_frames)
    parser.add_argument("--tail-frames", type=int, default=PercussiveEventDictionaryConfig.tail_frames)
    parser.add_argument("--low-cut-hz", type=float, default=PercussiveEventDictionaryConfig.low_cut_hz)
    parser.add_argument("--aux-gain", type=float, default=PercussiveEventDictionaryConfig.aux_gain)
    parser.add_argument("--max-aux-rms-ratio", type=float, default=PercussiveEventDictionaryConfig.max_aux_rms_ratio)
    parser.add_argument("--limiter-peak", type=float, default=PercussiveEventDictionaryConfig.limiter_peak)
    parser.add_argument("--max-aux-active-fraction", type=float, default=PercussiveEventDictionaryConfig.max_aux_active_fraction)
    parser.add_argument("--max-aux-low-mid-to-high-ratio", type=float, default=PercussiveEventDictionaryConfig.max_aux_low_mid_to_high_ratio)
    parser.add_argument("--harmonic-active-fraction-floor", type=float, default=PercussiveEventDictionaryConfig.harmonic_active_fraction_floor)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-label", default="01_percussive_event_dictionary.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_percussive_event_dictionary_candidate.py",
        entrypoint_path="/app/tools/build_percussive_event_dictionary_candidate.py",
    )
    args = parse_args()
    config = PercussiveEventDictionaryConfig(
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        harmonic_time_bins=int(args.harmonic_time_bins),
        percussive_frequency_bins=int(args.percussive_frequency_bins),
        mask_power=float(args.mask_power),
        event_flux_percentile=float(args.event_flux_percentile),
        min_event_flatness=float(args.min_event_flatness),
        activity_floor_ratio=float(args.activity_floor_ratio),
        pre_roll_frames=int(args.pre_roll_frames),
        tail_frames=int(args.tail_frames),
        low_cut_hz=float(args.low_cut_hz),
        aux_gain=float(args.aux_gain),
        max_aux_rms_ratio=float(args.max_aux_rms_ratio),
        limiter_peak=float(args.limiter_peak),
        max_aux_active_fraction=float(args.max_aux_active_fraction),
        max_aux_low_mid_to_high_ratio=float(args.max_aux_low_mid_to_high_ratio),
        harmonic_active_fraction_floor=float(args.harmonic_active_fraction_floor),
    )
    manifest = build_percussive_event_dictionary_candidate(
        base_manifest_path=args.base_manifest,
        source_audio_path=args.source_audio,
        output_dir=args.output_dir,
        route_name=str(args.route_name),
        config=config,
    )
    if args.listen_dir is not None:
        aux_sanity = manifest.get("pipeline", {}).get("aux_sanity", {})
        if aux_sanity.get("status") == "accepted":
            manifest["listen"] = _copy_listen_file(
                target_path=Path(manifest["stems"][TARGET_STEM]["path"]),
                aux_path=Path(manifest["stems"]["percussive_fx_aux"]["path"]),
                listen_dir=args.listen_dir,
                listen_label=str(args.listen_label),
            )
        else:
            manifest["listen"] = _listen_blocked_report(listen_dir=args.listen_dir, aux_sanity=aux_sanity)
        _write_json(Path(manifest["manifest_path"]), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
