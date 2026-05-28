from __future__ import annotations

import argparse
import hashlib
import json
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


MODEL_ID = "reference_guided_vocal_ensemble"
MODEL_DISPLAY_NAME = "Reference-guided vocal ensemble"
DEFAULT_STEM = "vocals"
WINDOW_DIAGNOSTICS_FILENAME = "ensemble_window_diagnostics.json"


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    manifest_path: Path


@dataclass(frozen=True)
class EnsembleResult:
    audio: np.ndarray
    window_count: int
    candidate_labels: list[str]
    aggregate_weights: list[float]
    window_summaries: list[dict[str, Any]]


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
    summary: dict[str, Any] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
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


def write_ensemble_window_diagnostics(
    *,
    output_dir: Path,
    algorithm: str,
    mode: str,
    ensemble: EnsembleResult,
    extra_summary: dict[str, Any],
) -> dict[str, Any]:
    diagnostics_path = output_dir / WINDOW_DIAGNOSTICS_FILENAME
    diagnostics = {
        "schema_version": 1,
        "algorithm": algorithm,
        "mode": mode,
        "window_count": int(ensemble.window_count),
        "candidate_labels": ensemble.candidate_labels,
        "aggregate_weights": {
            label: weight for label, weight in zip(ensemble.candidate_labels, ensemble.aggregate_weights)
        },
        "windows": ensemble.window_summaries,
    }
    diagnostics.update(extra_summary)
    write_json(diagnostics_path, diagnostics)
    return {
        "schema_version": 1,
        "path": str(diagnostics_path),
        "sha256": sha256_file(diagnostics_path),
        "bytes": diagnostics_path.stat().st_size,
        "window_count": int(ensemble.window_count),
    }


def parse_candidate_manifest(value: str) -> CandidateSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate manifest must use LABEL=PATH format")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    manifest_path = Path(raw_path.strip())
    if not manifest_path.exists():
        raise argparse.ArgumentTypeError(f"candidate manifest does not exist: {manifest_path}")
    return CandidateSpec(label=label, manifest_path=manifest_path)


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def resolve_manifest_stem_path(manifest_path: Path, stem_name: str) -> Path:
    payload = read_json(manifest_path)
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        raise RuntimeError(f"Manifest has no stems object: {manifest_path}")
    stem_info = stems.get(stem_name)
    if not isinstance(stem_info, dict) or not stem_info.get("path"):
        raise RuntimeError(f"Manifest has no requested stem path: stem={stem_name} manifest={manifest_path}")
    return resolve_manifest_path(manifest_path, str(stem_info["path"]))


def resolve_ground_truth_mix_path(ground_truth_manifest: Path) -> Path:
    payload = read_json(ground_truth_manifest)
    mix_info = payload.get("mix")
    if not isinstance(mix_info, dict) or not mix_info.get("path"):
        raise RuntimeError(f"Ground-truth manifest has no mix.path: {ground_truth_manifest}")
    return resolve_manifest_path(ground_truth_manifest, str(mix_info["path"]))


def validate_candidate_manifest(manifest_path: Path) -> None:
    payload = read_json(manifest_path)
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        execution_manifest_path = manifest_path.parent / "registry_candidate_execution_manifest.json"
        if not execution_manifest_path.exists():
            raise RuntimeError(f"Candidate manifest must include contract: {manifest_path}")
        execution_manifest = read_json(execution_manifest_path)
        execution_contract = execution_manifest.get("contract")
        if not isinstance(execution_contract, dict):
            raise RuntimeError(f"Registry execution manifest must include contract: {execution_manifest_path}")
        if execution_contract.get("fallback_used") is not False:
            raise RuntimeError(f"Registry execution manifest must set contract.fallback_used=false: {execution_manifest_path}")
        if execution_contract.get("hidden_inference") is not False:
            raise RuntimeError(f"Registry execution manifest must set contract.hidden_inference=false: {execution_manifest_path}")
        if execution_contract.get("registered_model_required") is not True:
            raise RuntimeError(
                f"Registry execution manifest must set contract.registered_model_required=true: {execution_manifest_path}"
            )
        if execution_contract.get("expected_manifest_required") is not True:
            raise RuntimeError(
                f"Registry execution manifest must set contract.expected_manifest_required=true: {execution_manifest_path}"
            )
        if execution_manifest.get("status") != "completed":
            raise RuntimeError(f"Registry execution manifest must be completed: {execution_manifest_path}")
        candidate_manifest = execution_manifest.get("candidate_manifest")
        if not isinstance(candidate_manifest, dict) or candidate_manifest.get("exists") is not True:
            raise RuntimeError(f"Registry execution manifest must confirm candidate manifest exists: {execution_manifest_path}")
        recorded_candidate_manifest_path = candidate_manifest.get("path")
        if not recorded_candidate_manifest_path:
            raise RuntimeError(f"Registry execution manifest must record candidate manifest path: {execution_manifest_path}")
        resolved_candidate_manifest_path = resolve_manifest_path(
            execution_manifest_path,
            str(recorded_candidate_manifest_path),
        )
        if resolved_candidate_manifest_path.resolve() != manifest_path.resolve():
            raise RuntimeError(
                "Registry execution manifest candidate path does not match requested manifest: "
                f"execution={recorded_candidate_manifest_path} requested={manifest_path}"
            )
        return
    if contract.get("fallback_used") is not False:
        raise RuntimeError(f"Candidate manifest must set contract.fallback_used=false: {manifest_path}")
    if contract.get("benchmark_compatible") is not True:
        raise RuntimeError(f"Candidate manifest must set contract.benchmark_compatible=true: {manifest_path}")


def read_stereo_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def align_audio_arrays(audio_arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not audio_arrays:
        raise ValueError("audio_arrays cannot be empty")
    min_frames = min(audio.shape[0] for audio in audio_arrays)
    min_channels = min(audio.shape[1] for audio in audio_arrays)
    return [audio[:min_frames, :min_channels].astype(np.float32, copy=False) for audio in audio_arrays]


def solve_nonnegative_ridge_matrix(
    *,
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> np.ndarray:
    system = design.T @ design
    if ridge > 0.0:
        system = system + (np.eye(system.shape[0], dtype=np.float64) * float(ridge))
    rhs = design.T @ target
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return np.maximum(weights, 0.0).astype(np.float64)


def solve_nonnegative_ridge_weights(
    *,
    candidate_audio: Sequence[np.ndarray],
    reference_audio: np.ndarray,
    ridge: float,
) -> np.ndarray:
    if not candidate_audio:
        raise ValueError("at least one candidate is required")
    aligned = align_audio_arrays([*candidate_audio, reference_audio])
    candidates = aligned[:-1]
    reference = aligned[-1]
    design = np.stack([candidate.reshape(-1) for candidate in candidates], axis=1).astype(np.float64)
    target = reference.reshape(-1).astype(np.float64)
    weights = solve_nonnegative_ridge_matrix(design=design, target=target, ridge=ridge)
    if not np.any(weights > 0.0):
        errors = [float(np.mean((candidate - reference) ** 2)) for candidate in candidates]
        weights[int(np.argmin(errors))] = 1.0
    return weights.astype(np.float64)


def window_starts(frame_count: int, window_frames: int, hop_frames: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if hop_frames <= 0:
        raise ValueError("hop_frames must be positive")
    starts = list(range(0, max(frame_count - window_frames, 0) + 1, hop_frames))
    last_start = max(frame_count - window_frames, 0)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def overlap_window(length: int, overlap_frames: int, *, is_first: bool, is_last: bool) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    fade = min(int(overlap_frames), max(length // 2, 0))
    if fade > 0 and not is_first:
        weights[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    if fade > 0 and not is_last:
        weights[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
    return weights[:, None]


def build_segmented_ensemble_audio(
    *,
    candidate_audio: Sequence[np.ndarray],
    reference_audio: np.ndarray,
    sample_rate: int,
    window_sec: float,
    overlap_sec: float,
    ridge: float,
    candidate_labels: Sequence[str] | None = None,
) -> EnsembleResult:
    aligned = align_audio_arrays([*candidate_audio, reference_audio])
    candidates = aligned[:-1]
    reference = aligned[-1]
    frame_count = reference.shape[0]
    window_frames = max(1, int(round(float(window_sec) * float(sample_rate))))
    overlap_frames = max(0, int(round(float(overlap_sec) * float(sample_rate))))
    if overlap_frames >= window_frames:
        raise ValueError("overlap_sec must be smaller than window_sec")
    hop_frames = window_frames - overlap_frames
    starts = window_starts(frame_count, window_frames, hop_frames)
    output = np.zeros_like(reference, dtype=np.float32)
    normalizer = np.zeros((frame_count, 1), dtype=np.float32)
    aggregate_weights = np.zeros(len(candidates), dtype=np.float64)
    labels = list(candidate_labels or [f"candidate_{index + 1}" for index in range(len(candidates))])
    window_summaries: list[dict[str, Any]] = []

    for index, start in enumerate(starts):
        end = min(start + window_frames, frame_count)
        candidate_window = [candidate[start:end] for candidate in candidates]
        reference_window = reference[start:end]
        weights = solve_nonnegative_ridge_weights(
            candidate_audio=candidate_window,
            reference_audio=reference_window,
            ridge=ridge,
        )
        aggregate_weights += weights * float(end - start)
        blended = np.zeros_like(reference_window, dtype=np.float32)
        for candidate, weight in zip(candidate_window, weights):
            blended += candidate * float(weight)
        taper = overlap_window(
            end - start,
            overlap_frames,
            is_first=index == 0,
            is_last=index == len(starts) - 1,
        )
        output[start:end] += blended * taper
        normalizer[start:end] += taper
        window_summaries.append(
            {
                "index": index,
                "start_frame": int(start),
                "end_frame": int(end),
                "weights": {label: round(float(weight), 8) for label, weight in zip(labels, weights)},
            }
        )

    output = output / np.maximum(normalizer, 1.0e-8)
    aggregate_weights = aggregate_weights / max(float(frame_count), 1.0)
    return EnsembleResult(
        audio=np.clip(output, -1.0, 1.0).astype(np.float32, copy=False),
        window_count=len(starts),
        candidate_labels=labels,
        aggregate_weights=[round(float(weight), 8) for weight in aggregate_weights],
        window_summaries=window_summaries,
    )


def normalize_band_edges_hz(edges: Sequence[float], *, sample_rate: int) -> list[float]:
    edges = list(float(edge) for edge in edges)
    nyquist = float(sample_rate) / 2.0
    if not edges:
        edges = [0.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, nyquist]
    if edges[0] > 0.0:
        edges.insert(0, 0.0)
    if edges[-1] < nyquist:
        edges.append(nyquist)
    normalized = sorted(set(max(0.0, min(float(edge), nyquist)) for edge in edges))
    if len(normalized) < 2:
        raise ValueError("at least two band edges are required")
    return normalized


def parse_band_edges_hz(raw_value: str) -> list[float]:
    return [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def solve_spectral_band_weights(
    *,
    candidate_spectra: Sequence[np.ndarray],
    reference_spectrum: np.ndarray,
    band_mask: np.ndarray,
    ridge: float,
) -> np.ndarray:
    design_columns: list[np.ndarray] = []
    for spectrum in candidate_spectra:
        selected = spectrum[band_mask, :]
        design_columns.append(np.concatenate([selected.real.reshape(-1), selected.imag.reshape(-1)]))
    design = np.stack(design_columns, axis=1).astype(np.float64)
    selected_reference = reference_spectrum[band_mask, :]
    target = np.concatenate([selected_reference.real.reshape(-1), selected_reference.imag.reshape(-1)]).astype(np.float64)
    weights = solve_nonnegative_ridge_matrix(design=design, target=target, ridge=ridge)
    if not np.any(weights > 0.0):
        errors = []
        for spectrum in candidate_spectra:
            delta = spectrum[band_mask, :] - reference_spectrum[band_mask, :]
            errors.append(float(np.mean(np.abs(delta) ** 2)))
        weights[int(np.argmin(errors))] = 1.0
    return weights


def solve_spectral_band_complex_weights(
    *,
    candidate_spectra: Sequence[np.ndarray],
    reference_spectrum: np.ndarray,
    band_mask: np.ndarray,
    ridge: float,
) -> np.ndarray:
    design = np.stack(
        [spectrum[band_mask, :].reshape(-1) for spectrum in candidate_spectra],
        axis=1,
    ).astype(np.complex128)
    target = reference_spectrum[band_mask, :].reshape(-1).astype(np.complex128)
    system = design.conj().T @ design
    if ridge > 0.0:
        system = system + (np.eye(system.shape[0], dtype=np.complex128) * float(ridge))
    rhs = design.conj().T @ target
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if not np.any(np.abs(weights) > 0.0):
        errors = []
        for spectrum in candidate_spectra:
            delta = spectrum[band_mask, :] - reference_spectrum[band_mask, :]
            errors.append(float(np.mean(np.abs(delta) ** 2)))
        weights[int(np.argmin(errors))] = 1.0 + 0.0j
    return weights.astype(np.complex128)


def build_segmented_spectral_band_ensemble_audio(
    *,
    candidate_audio: Sequence[np.ndarray],
    reference_audio: np.ndarray,
    sample_rate: int,
    window_sec: float,
    overlap_sec: float,
    ridge: float,
    band_edges_hz: Sequence[float],
    candidate_labels: Sequence[str] | None = None,
) -> EnsembleResult:
    aligned = align_audio_arrays([*candidate_audio, reference_audio])
    candidates = aligned[:-1]
    reference = aligned[-1]
    frame_count = reference.shape[0]
    window_frames = max(1, int(round(float(window_sec) * float(sample_rate))))
    overlap_frames = max(0, int(round(float(overlap_sec) * float(sample_rate))))
    if overlap_frames >= window_frames:
        raise ValueError("overlap_sec must be smaller than window_sec")
    hop_frames = window_frames - overlap_frames
    starts = window_starts(frame_count, window_frames, hop_frames)
    output = np.zeros_like(reference, dtype=np.float32)
    normalizer = np.zeros((frame_count, 1), dtype=np.float32)
    aggregate_weights = np.zeros(len(candidates), dtype=np.float64)
    labels = list(candidate_labels or [f"candidate_{index + 1}" for index in range(len(candidates))])
    band_edges = list(float(edge) for edge in band_edges_hz)
    window_summaries: list[dict[str, Any]] = []

    for index, start in enumerate(starts):
        end = min(start + window_frames, frame_count)
        reference_window = reference[start:end]
        candidate_windows = [candidate[start:end] for candidate in candidates]
        reference_spectrum = np.fft.rfft(reference_window, axis=0)
        candidate_spectra = [np.fft.rfft(candidate_window, axis=0) for candidate_window in candidate_windows]
        frequencies = np.fft.rfftfreq(end - start, d=1.0 / float(sample_rate))
        blended_spectrum = np.zeros_like(reference_spectrum)
        band_summary: dict[str, dict[str, float]] = {}
        window_weight_sum = np.zeros(len(candidates), dtype=np.float64)

        for band_start, band_end in zip(band_edges[:-1], band_edges[1:]):
            if band_end <= band_start:
                continue
            band_mask = (frequencies >= band_start) & (frequencies < band_end)
            if band_end == band_edges[-1]:
                band_mask = (frequencies >= band_start) & (frequencies <= band_end)
            if not np.any(band_mask):
                continue
            weights = solve_spectral_band_weights(
                candidate_spectra=candidate_spectra,
                reference_spectrum=reference_spectrum,
                band_mask=band_mask,
                ridge=ridge,
            )
            for spectrum, weight in zip(candidate_spectra, weights):
                blended_spectrum[band_mask, :] += spectrum[band_mask, :] * float(weight)
            window_weight_sum += weights * float(np.count_nonzero(band_mask))
            band_summary[f"{band_start:.1f}-{band_end:.1f}"] = {
                label: round(float(weight), 8) for label, weight in zip(labels, weights)
            }

        blended = np.fft.irfft(blended_spectrum, n=end - start, axis=0).astype(np.float32)
        taper = overlap_window(
            end - start,
            overlap_frames,
            is_first=index == 0,
            is_last=index == len(starts) - 1,
        )
        output[start:end] += blended * taper
        normalizer[start:end] += taper
        aggregate_weights += window_weight_sum * float(end - start)
        window_summaries.append(
            {
                "index": index,
                "start_frame": int(start),
                "end_frame": int(end),
                "band_weights": band_summary,
            }
        )

    output = output / np.maximum(normalizer, 1.0e-8)
    aggregate_weights = aggregate_weights / max(float(frame_count * max(len(band_edges) - 1, 1)), 1.0)
    return EnsembleResult(
        audio=np.clip(output, -1.0, 1.0).astype(np.float32, copy=False),
        window_count=len(starts),
        candidate_labels=labels,
        aggregate_weights=[round(float(weight), 8) for weight in aggregate_weights],
        window_summaries=window_summaries,
    )


def build_segmented_spectral_band_complex_ensemble_audio(
    *,
    candidate_audio: Sequence[np.ndarray],
    reference_audio: np.ndarray,
    sample_rate: int,
    window_sec: float,
    overlap_sec: float,
    ridge: float,
    band_edges_hz: Sequence[float],
    candidate_labels: Sequence[str] | None = None,
) -> EnsembleResult:
    aligned = align_audio_arrays([*candidate_audio, reference_audio])
    candidates = aligned[:-1]
    reference = aligned[-1]
    frame_count = reference.shape[0]
    window_frames = max(1, int(round(float(window_sec) * float(sample_rate))))
    overlap_frames = max(0, int(round(float(overlap_sec) * float(sample_rate))))
    if overlap_frames >= window_frames:
        raise ValueError("overlap_sec must be smaller than window_sec")
    hop_frames = window_frames - overlap_frames
    starts = window_starts(frame_count, window_frames, hop_frames)
    output = np.zeros_like(reference, dtype=np.float32)
    normalizer = np.zeros((frame_count, 1), dtype=np.float32)
    aggregate_weights = np.zeros(len(candidates), dtype=np.float64)
    labels = list(candidate_labels or [f"candidate_{index + 1}" for index in range(len(candidates))])
    band_edges = list(float(edge) for edge in band_edges_hz)
    window_summaries: list[dict[str, Any]] = []

    for index, start in enumerate(starts):
        end = min(start + window_frames, frame_count)
        reference_window = reference[start:end]
        candidate_windows = [candidate[start:end] for candidate in candidates]
        reference_spectrum = np.fft.rfft(reference_window, axis=0)
        candidate_spectra = [np.fft.rfft(candidate_window, axis=0) for candidate_window in candidate_windows]
        frequencies = np.fft.rfftfreq(end - start, d=1.0 / float(sample_rate))
        blended_spectrum = np.zeros_like(reference_spectrum)
        band_summary: dict[str, dict[str, dict[str, float]]] = {}
        window_weight_sum = np.zeros(len(candidates), dtype=np.float64)

        for band_start, band_end in zip(band_edges[:-1], band_edges[1:]):
            if band_end <= band_start:
                continue
            band_mask = (frequencies >= band_start) & (frequencies < band_end)
            if band_end == band_edges[-1]:
                band_mask = (frequencies >= band_start) & (frequencies <= band_end)
            if not np.any(band_mask):
                continue
            weights = solve_spectral_band_complex_weights(
                candidate_spectra=candidate_spectra,
                reference_spectrum=reference_spectrum,
                band_mask=band_mask,
                ridge=ridge,
            )
            for spectrum, weight in zip(candidate_spectra, weights):
                blended_spectrum[band_mask, :] += spectrum[band_mask, :] * weight
            magnitudes = np.abs(weights)
            window_weight_sum += magnitudes * float(np.count_nonzero(band_mask))
            band_summary[f"{band_start:.1f}-{band_end:.1f}"] = {
                label: {
                    "real": round(float(weight.real), 8),
                    "imag": round(float(weight.imag), 8),
                    "magnitude": round(float(abs(weight)), 8),
                }
                for label, weight in zip(labels, weights)
            }

        blended = np.fft.irfft(blended_spectrum, n=end - start, axis=0).astype(np.float32)
        taper = overlap_window(
            end - start,
            overlap_frames,
            is_first=index == 0,
            is_last=index == len(starts) - 1,
        )
        output[start:end] += blended * taper
        normalizer[start:end] += taper
        aggregate_weights += window_weight_sum * float(end - start)
        window_summaries.append(
            {
                "index": index,
                "start_frame": int(start),
                "end_frame": int(end),
                "complex_band_weights": band_summary,
            }
        )

    output = output / np.maximum(normalizer, 1.0e-8)
    aggregate_weights = aggregate_weights / max(float(frame_count * max(len(band_edges) - 1, 1)), 1.0)
    return EnsembleResult(
        audio=np.clip(output, -1.0, 1.0).astype(np.float32, copy=False),
        window_count=len(starts),
        candidate_labels=labels,
        aggregate_weights=[round(float(weight), 8) for weight in aggregate_weights],
        window_summaries=window_summaries,
    )


def build_reference_guided_manifest(
    *,
    source_audio_path: Path,
    ground_truth_manifest: Path,
    candidate_specs: Sequence[CandidateSpec],
    vocals_path: Path,
    instrumental_path: Path,
    sample_rate: int,
    frames: int,
    elapsed_sec: float,
    ensemble_summary: dict[str, Any],
) -> dict[str, Any]:
    if "windows" in ensemble_summary:
        raise ValueError("stem_manifest must not embed per-window diagnostics; write them to the diagnostics sidecar")
    return {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "model": {
            "model_id": MODEL_ID,
            "name": MODEL_DISPLAY_NAME,
            "family": "reference_assisted_ensemble",
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
            "stem": DEFAULT_STEM,
            "use": "offline_reference_guided_candidate_selection_and_blending",
            "not_a_no_reference_production_route": True,
        },
        "candidate_manifests": [
            {"label": spec.label, "path": str(spec.manifest_path)}
            for spec in candidate_specs
        ],
        "ensemble": ensemble_summary,
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


def build_reference_guided_vocal_ensemble(
    *,
    ground_truth_manifest: Path,
    candidate_specs: Sequence[CandidateSpec],
    output_dir: Path,
    stem_name: str,
    window_sec: float,
    overlap_sec: float,
    ridge: float,
    mode: str = "spectral_band",
    band_edges_hz: Sequence[float] | None = None,
) -> dict[str, Any]:
    start_wall = time.time()
    if stem_name != DEFAULT_STEM:
        raise ValueError("reference-guided vocal ensemble currently supports only stem=vocals")
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_bundle = quality_gate.resolve_stem_bundle(ground_truth_manifest)
    quality_gate.validate_verified_ground_truth(ground_truth_bundle)
    reference_path = resolve_manifest_stem_path(ground_truth_manifest, stem_name)
    source_audio_path = resolve_ground_truth_mix_path(ground_truth_manifest)
    reference_audio, reference_sample_rate = read_stereo_audio(reference_path)
    source_audio, source_sample_rate = read_stereo_audio(source_audio_path)
    if source_sample_rate != reference_sample_rate:
        raise RuntimeError("source mix and reference stem sample rates differ")

    candidate_audio: list[np.ndarray] = []
    for spec in candidate_specs:
        validate_candidate_manifest(spec.manifest_path)
        candidate_path = resolve_manifest_stem_path(spec.manifest_path, stem_name)
        audio, sample_rate = read_stereo_audio(candidate_path)
        if sample_rate != reference_sample_rate:
            raise RuntimeError(f"candidate sample rate differs: label={spec.label} sample_rate={sample_rate}")
        candidate_audio.append(audio)

    aligned = align_audio_arrays([source_audio, reference_audio, *candidate_audio])
    source_audio = aligned[0]
    reference_audio = aligned[1]
    candidate_audio = aligned[2:]
    candidate_labels = [spec.label for spec in candidate_specs]
    if mode == "time_domain":
        ensemble = build_segmented_ensemble_audio(
            candidate_audio=candidate_audio,
            reference_audio=reference_audio,
            sample_rate=reference_sample_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            ridge=ridge,
            candidate_labels=candidate_labels,
        )
        algorithm = "segment_level_nonnegative_ridge_v1"
        extra_summary: dict[str, Any] = {}
    elif mode == "spectral_band":
        resolved_band_edges = normalize_band_edges_hz(band_edges_hz or [], sample_rate=reference_sample_rate)
        ensemble = build_segmented_spectral_band_ensemble_audio(
            candidate_audio=candidate_audio,
            reference_audio=reference_audio,
            sample_rate=reference_sample_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            ridge=ridge,
            band_edges_hz=resolved_band_edges,
            candidate_labels=candidate_labels,
        )
        algorithm = "segment_level_spectral_band_nonnegative_ridge_v1"
        extra_summary = {"band_edges_hz": resolved_band_edges}
    elif mode == "spectral_band_complex":
        resolved_band_edges = normalize_band_edges_hz(band_edges_hz or [], sample_rate=reference_sample_rate)
        ensemble = build_segmented_spectral_band_complex_ensemble_audio(
            candidate_audio=candidate_audio,
            reference_audio=reference_audio,
            sample_rate=reference_sample_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            ridge=ridge,
            band_edges_hz=resolved_band_edges,
            candidate_labels=candidate_labels,
        )
        algorithm = "segment_level_spectral_band_complex_ridge_teacher_v1"
        extra_summary = {
            "band_edges_hz": resolved_band_edges,
            "complex_weights": True,
            "teacher_mode": True,
        }
    else:
        raise ValueError(f"Unsupported ensemble mode: {mode}")
    instrumental_audio = np.clip(source_audio - ensemble.audio, -1.0, 1.0)
    vocals_path = output_dir / "vocals.flac"
    instrumental_path = output_dir / "instrumental.flac"
    sf.write(str(vocals_path), ensemble.audio, reference_sample_rate, subtype="PCM_24")
    sf.write(str(instrumental_path), instrumental_audio, reference_sample_rate, subtype="PCM_24")
    window_diagnostics = write_ensemble_window_diagnostics(
        output_dir=output_dir,
        algorithm=algorithm,
        mode=mode,
        ensemble=ensemble,
        extra_summary=extra_summary,
    )
    ensemble_summary = {
        "algorithm": algorithm,
        "mode": mode,
        "window_sec": float(window_sec),
        "overlap_sec": float(overlap_sec),
        "ridge": float(ridge),
        "window_count": ensemble.window_count,
        "candidate_labels": ensemble.candidate_labels,
        "aggregate_weights": {
            label: weight for label, weight in zip(ensemble.candidate_labels, ensemble.aggregate_weights)
        },
        "window_diagnostics": window_diagnostics,
    }
    ensemble_summary.update(extra_summary)
    manifest = build_reference_guided_manifest(
        source_audio_path=source_audio_path,
        ground_truth_manifest=ground_truth_manifest,
        candidate_specs=candidate_specs,
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        sample_rate=reference_sample_rate,
        frames=int(ensemble.audio.shape[0]),
        elapsed_sec=time.time() - start_wall,
        ensemble_summary=ensemble_summary,
    )
    write_json(output_dir / "stem_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reference-assisted vocal ensemble candidate.")
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=parse_candidate_manifest, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--overlap-sec", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--mode",
        choices=("time_domain", "spectral_band", "spectral_band_complex"),
        default="spectral_band_complex",
    )
    parser.add_argument(
        "--band-edges-hz",
        default="0,120,250,500,1000,2000,4000,8000,12000,22050",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_reference_guided_vocal_ensemble.py",
        entrypoint_path="/app/tools/build_reference_guided_vocal_ensemble.py",
    )
    args = build_arg_parser().parse_args(argv)
    manifest = build_reference_guided_vocal_ensemble(
        ground_truth_manifest=args.ground_truth_manifest,
        candidate_specs=args.candidate_manifest,
        output_dir=args.output_dir,
        stem_name=str(args.stem),
        window_sec=float(args.window_sec),
        overlap_sec=float(args.overlap_sec),
        ridge=float(args.ridge),
        mode=str(args.mode),
        band_edges_hz=parse_band_edges_hz(str(args.band_edges_hz)),
    )
    print(json.dumps({"manifest_path": str(args.output_dir / "stem_manifest.json"), "model_id": manifest["model_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
