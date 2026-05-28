from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    stem_path: Path
    leakage_reference_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RouterConfig:
    segment_seconds: float = 5.0
    crossfade_seconds: float = 0.15
    min_rms_dbfs: float = -34.0
    percussive_weight: float = 1.0
    harmonic_weight: float = 1.0
    leakage_corr_weight: float = 5.0
    vocal_leakage_multiplier: float = 2.5
    high_band_noise_weight: float = 50.0
    target_corr_weight: float = 20.0
    target_rms_delta_weight: float = 0.5
    quiet_penalty_weight: float = 0.35
    body_weight: float = 0.04
    switch_hysteresis: float = 0.2
    baseline_candidate_label: str | None = None
    baseline_switch_margin: float = 0.0
    baseline_max_switch_fraction: float | None = None
    baseline_calibration_profile_path: str | None = None


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio
    if audio.shape[0] > frames:
        return audio[:frames]
    pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=0)


def _rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    value = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)
    return 20.0 * math.log10(value)


def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    frames = min(left.shape[0], right.shape[0])
    if frames < 128:
        return 0.0
    left_mono = left[:frames].mean(axis=1)
    right_mono = right[:frames].mean(axis=1)
    left_mono = left_mono - float(np.mean(left_mono))
    right_mono = right_mono - float(np.mean(right_mono))
    denominator = float(np.sqrt(np.sum(left_mono * left_mono) * np.sum(right_mono * right_mono)) + 1e-12)
    return float(np.sum(left_mono * right_mono) / denominator)


def _percussive_minus_harmonic_db(audio: np.ndarray, sample_rate: int) -> float:
    if audio.shape[0] < 1024:
        return 0.0
    mono = audio.mean(axis=1)
    harmonic, percussive = librosa.effects.hpss(mono)
    return _rms_dbfs(percussive.reshape(-1, 1)) - _rms_dbfs(harmonic.reshape(-1, 1))


def _spectral_energy_ratio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float | None = None) -> float:
    if audio.shape[0] < 128:
        return 0.0
    mono = audio.mean(axis=1).astype(np.float64, copy=False)
    mono = mono - float(np.mean(mono))
    if not np.any(mono):
        return 0.0
    window = np.hanning(mono.shape[0])
    spectrum = np.abs(np.fft.rfft(mono * window)) ** 2
    frequencies = np.fft.rfftfreq(mono.shape[0], d=1.0 / float(sample_rate))
    total_energy = float(np.sum(spectrum) + 1e-12)
    band_mask = frequencies >= low_hz
    if high_hz is not None:
        band_mask &= frequencies < high_hz
    return float(np.sum(spectrum[band_mask]) / total_energy)


def _stem_score_profile(stem_name: str) -> str:
    normalized = stem_name.strip().lower().replace("-", "_")
    drum_like_targets = {
        "drum",
        "drums",
        "drums_plus_percussive_fx_aux",
        "percussive_fx_aux_drums",
    }
    if normalized in drum_like_targets:
        return "drums"
    if normalized in {"vocal", "vocals", "lead_vocal", "lead_vocals", "backing_vocal", "backing_vocals"}:
        return "vocals"
    return "generic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_selection_guard(
    *,
    scores: dict[str, dict[str, Any]],
    selected_label: str,
    baseline_candidate_label: str | None,
    baseline_switch_margin: float,
    target_reference_used: bool,
) -> tuple[str, dict[str, Any]]:
    confidence_metric = "score_delta_vs_baseline"
    if baseline_candidate_label is None:
        return selected_label, {
            "enabled": False,
            "applied": False,
            "reason": "baseline_not_configured",
            "baseline_label": None,
            "unguarded_selected": selected_label,
            "confidence_metric": confidence_metric,
            "no_reference_confidence_score": None,
            "score_delta": None,
            "required_margin": baseline_switch_margin,
        }
    if target_reference_used:
        return selected_label, {
            "enabled": False,
            "applied": False,
            "reason": "target_reference_mode",
            "baseline_label": baseline_candidate_label,
            "unguarded_selected": selected_label,
            "confidence_metric": confidence_metric,
            "no_reference_confidence_score": None,
            "score_delta": None,
            "required_margin": baseline_switch_margin,
        }
    baseline_score = float(scores[baseline_candidate_label]["score"])
    selected_score = float(scores[selected_label]["score"])
    score_delta = selected_score - baseline_score
    if selected_label == baseline_candidate_label:
        return selected_label, {
            "enabled": True,
            "applied": False,
            "reason": "baseline_already_selected",
            "baseline_label": baseline_candidate_label,
            "unguarded_selected": selected_label,
            "confidence_metric": confidence_metric,
            "no_reference_confidence_score": round(score_delta, 6),
            "score_delta": round(score_delta, 6),
            "required_margin": baseline_switch_margin,
        }
    if score_delta < baseline_switch_margin:
        return baseline_candidate_label, {
            "enabled": True,
            "applied": True,
            "reason": "baseline_score_margin_not_met",
            "baseline_label": baseline_candidate_label,
            "unguarded_selected": selected_label,
            "confidence_metric": confidence_metric,
            "no_reference_confidence_score": round(score_delta, 6),
            "score_delta": round(score_delta, 6),
            "required_margin": baseline_switch_margin,
        }
    return selected_label, {
        "enabled": True,
        "applied": False,
        "reason": "score_margin_met",
        "baseline_label": baseline_candidate_label,
        "unguarded_selected": selected_label,
        "confidence_metric": confidence_metric,
        "no_reference_confidence_score": round(score_delta, 6),
        "score_delta": round(score_delta, 6),
        "required_margin": baseline_switch_margin,
    }


def _apply_baseline_switch_budget(
    decisions: list[dict[str, Any]],
    *,
    baseline_candidate_label: str | None,
    baseline_max_switch_fraction: float | None,
    target_reference_used: bool,
) -> None:
    if baseline_candidate_label is None or baseline_max_switch_fraction is None or target_reference_used:
        return
    allowed_switch_count = int(math.floor(len(decisions) * baseline_max_switch_fraction))
    switch_candidates = [
        (index, float(decision["selection_guard"].get("no_reference_confidence_score") or 0.0))
        for index, decision in enumerate(decisions)
        if decision["selected"] != baseline_candidate_label and decision["selection_guard"].get("enabled") is True
    ]
    switch_candidates.sort(key=lambda item: item[1], reverse=True)
    allowed_indices = {index for index, _score in switch_candidates[:allowed_switch_count]}
    for budget_rank, (index, _score) in enumerate(switch_candidates, start=1):
        guard = decisions[index]["selection_guard"]
        guard["switch_budget_rank"] = budget_rank
        guard["allowed_switch_count"] = allowed_switch_count
        guard["baseline_max_switch_fraction"] = baseline_max_switch_fraction
        if index in allowed_indices:
            guard["switch_budget_allowed"] = True
            continue
        guard["switch_budget_allowed"] = False
        guard["applied"] = True
        guard["reason"] = "baseline_switch_budget_exceeded"
        decisions[index]["selected"] = baseline_candidate_label


def _score_segment(
    candidate_audio: np.ndarray,
    leakage_references: tuple[tuple[str, np.ndarray], ...],
    sample_rate: int,
    config: RouterConfig,
    target_reference: np.ndarray | None = None,
    stem_profile: str = "drums",
) -> dict[str, Any]:
    rms_dbfs = _rms_dbfs(candidate_audio)
    percussive_minus_harmonic_db = _percussive_minus_harmonic_db(candidate_audio, sample_rate)
    harmonic_minus_percussive_db = -percussive_minus_harmonic_db
    vocal_body_ratio = _spectral_energy_ratio(candidate_audio, sample_rate, 120.0, 4200.0)
    high_band_ratio = _spectral_energy_ratio(candidate_audio, sample_rate, 5600.0, None)
    unsupported_high_band_ratio = max(0.0, high_band_ratio - 0.35 * vocal_body_ratio)
    high_band_noise_penalty = config.high_band_noise_weight * unsupported_high_band_ratio
    leakage_corr_by_reference = {
        name: round(abs(_normalized_correlation(candidate_audio, reference_audio)), 6)
        for name, reference_audio in leakage_references
    }
    max_leakage_reference, leakage_corr_max = max(
        leakage_corr_by_reference.items(),
        key=lambda item: item[1],
        default=(None, 0.0),
    )
    target_corr = _normalized_correlation(candidate_audio, target_reference) if target_reference is not None else 0.0
    target_rms_delta = abs(rms_dbfs - _rms_dbfs(target_reference)) if target_reference is not None else 0.0
    quiet_penalty = max(0.0, config.min_rms_dbfs - rms_dbfs)
    if stem_profile == "vocals":
        leakage_penalty = config.leakage_corr_weight * config.vocal_leakage_multiplier * leakage_corr_max
        effective_quiet_penalty = quiet_penalty * min(1.0, vocal_body_ratio * 2.0)
        rms_body_score = config.body_weight * max(rms_dbfs, config.min_rms_dbfs)
        score = (
            config.harmonic_weight * harmonic_minus_percussive_db
            - leakage_penalty
            - high_band_noise_penalty
            + config.target_corr_weight * target_corr
            - config.target_rms_delta_weight * target_rms_delta
            - config.quiet_penalty_weight * effective_quiet_penalty
            + rms_body_score
        )
    else:
        leakage_penalty = config.leakage_corr_weight * leakage_corr_max
        effective_quiet_penalty = quiet_penalty
        rms_body_score = config.body_weight * rms_dbfs
        score = (
            config.percussive_weight * percussive_minus_harmonic_db
            - leakage_penalty
            + config.target_corr_weight * target_corr
            - config.target_rms_delta_weight * target_rms_delta
            - config.quiet_penalty_weight * effective_quiet_penalty
            + rms_body_score
        )
    return {
        "stem_profile": stem_profile,
        "score": round(score, 6),
        "rms_dbfs": round(rms_dbfs, 6),
        "percussive_minus_harmonic_db": round(percussive_minus_harmonic_db, 6),
        "harmonic_minus_percussive_db": round(harmonic_minus_percussive_db, 6),
        "vocal_body_ratio": round(vocal_body_ratio, 6),
        "high_band_ratio": round(high_band_ratio, 6),
        "unsupported_high_band_ratio": round(unsupported_high_band_ratio, 6),
        "high_band_noise_penalty": round(high_band_noise_penalty, 6),
        "leakage_penalty": round(leakage_penalty, 6),
        "leakage_corr_max": round(leakage_corr_max, 6),
        "max_leakage_reference": max_leakage_reference,
        "leakage_corr_by_reference": leakage_corr_by_reference,
        "target_corr": round(target_corr, 6),
        "target_rms_delta_db": round(target_rms_delta, 6),
        "quiet_penalty": round(quiet_penalty, 6),
        "effective_quiet_penalty": round(effective_quiet_penalty, 6),
        "rms_body_score": round(rms_body_score, 6),
    }


def route_drum_segments(
    candidates: list[CandidateSpec],
    output_dir: Path,
    *,
    listen_dir: Path | None = None,
    output_label: str | None = None,
    stem_name: str = "drums",
    target_reference_path: Path | None = None,
    config: RouterConfig | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one candidate is required.")
    config = config or RouterConfig()
    if config.segment_seconds <= 0.0:
        raise ValueError("segment_seconds must be positive.")
    if config.crossfade_seconds < 0.0:
        raise ValueError("crossfade_seconds must be non-negative.")
    if config.baseline_switch_margin < 0.0:
        raise ValueError("baseline_switch_margin must be non-negative.")
    if config.baseline_max_switch_fraction is not None and not 0.0 <= config.baseline_max_switch_fraction <= 1.0:
        raise ValueError("baseline_max_switch_fraction must be between 0.0 and 1.0.")
    stem_name = stem_name.strip()
    if not stem_name:
        raise ValueError("stem_name must not be empty.")
    stem_profile = _stem_score_profile(stem_name)
    route_name = "drum_model_zoo_segment_router_v1" if stem_profile == "drums" else "stem_model_zoo_segment_router_v1"
    output_label = output_label or f"06_segment_model_router_{stem_name}.flac"

    loaded_candidates: dict[str, np.ndarray] = {}
    loaded_references: dict[str, tuple[tuple[str, np.ndarray], ...]] = {}
    loaded_target_reference: np.ndarray | None = None
    sample_rate: int | None = None
    frames = 0
    for candidate in candidates:
        if candidate.label in loaded_candidates:
            raise ValueError(f"Duplicate candidate label: {candidate.label}")
        audio, candidate_sample_rate = _read_audio(candidate.stem_path)
        if sample_rate is None:
            sample_rate = candidate_sample_rate
            frames = audio.shape[0]
        elif candidate_sample_rate != sample_rate:
            raise ValueError(f"Sample-rate mismatch for {candidate.label}: {candidate_sample_rate} != {sample_rate}")
        loaded_candidates[candidate.label] = audio
        refs: list[tuple[str, np.ndarray]] = []
        for reference_path in candidate.leakage_reference_paths:
            reference_audio, reference_sample_rate = _read_audio(reference_path)
            if reference_sample_rate != sample_rate:
                raise ValueError(f"Sample-rate mismatch for leakage reference: {reference_path}")
            refs.append((reference_path.stem, reference_audio))
        loaded_references[candidate.label] = tuple(refs)

    assert sample_rate is not None
    if target_reference_path is not None:
        loaded_target_reference, target_sample_rate = _read_audio(target_reference_path)
        if target_sample_rate != sample_rate:
            raise ValueError(f"Sample-rate mismatch for target reference: {target_reference_path}")

    frames = min(audio.shape[0] for audio in loaded_candidates.values())
    if loaded_target_reference is not None:
        frames = min(frames, loaded_target_reference.shape[0])
    loaded_candidates = {label: _fit_length(audio, frames) for label, audio in loaded_candidates.items()}
    loaded_references = {
        label: tuple((reference_name, _fit_length(reference, frames)) for reference_name, reference in references)
        for label, references in loaded_references.items()
    }
    if loaded_target_reference is not None:
        loaded_target_reference = _fit_length(loaded_target_reference, frames)
    baseline_candidate_label = config.baseline_candidate_label.strip() if config.baseline_candidate_label else None
    if baseline_candidate_label is not None and baseline_candidate_label not in loaded_candidates:
        raise ValueError(f"baseline_candidate_label is not a candidate: {baseline_candidate_label}")

    segment_frames = max(1, int(round(config.segment_seconds * sample_rate)))
    crossfade_frames = min(int(round(config.crossfade_seconds * sample_rate)), max(0, segment_frames // 2))
    decisions: list[dict[str, Any]] = []
    previous_label: str | None = None

    for start in range(0, frames, segment_frames):
        end = min(frames, start + segment_frames)
        scores: dict[str, dict[str, Any]] = {}
        for label, audio in loaded_candidates.items():
            references = tuple((reference_name, reference[start:end]) for reference_name, reference in loaded_references[label])
            target_reference = loaded_target_reference[start:end] if loaded_target_reference is not None else None
            scores[label] = _score_segment(
                audio[start:end],
                references,
                sample_rate,
                config,
                target_reference,
                stem_profile,
            )
        ranked = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)
        selected_label = ranked[0][0]
        if previous_label and previous_label in scores:
            previous_score = scores[previous_label]["score"]
            best_score = scores[selected_label]["score"]
            if best_score - previous_score < config.switch_hysteresis:
                selected_label = previous_label
        selected_label, selection_guard = _build_selection_guard(
            scores=scores,
            selected_label=selected_label,
            baseline_candidate_label=baseline_candidate_label,
            baseline_switch_margin=config.baseline_switch_margin,
            target_reference_used=loaded_target_reference is not None,
        )
        decisions.append(
            {
                "start_sec": round(start / sample_rate, 6),
                "end_sec": round(end / sample_rate, 6),
                "selected": selected_label,
                "selection_guard": selection_guard,
                "ranked": [{"label": label, **metrics} for label, metrics in ranked],
            }
        )
        previous_label = selected_label
    _apply_baseline_switch_budget(
        decisions,
        baseline_candidate_label=baseline_candidate_label,
        baseline_max_switch_fraction=config.baseline_max_switch_fraction,
        target_reference_used=loaded_target_reference is not None,
    )

    output_audio = np.zeros_like(next(iter(loaded_candidates.values())))
    for decision in decisions:
        start = int(round(float(decision["start_sec"]) * sample_rate))
        end = int(round(float(decision["end_sec"]) * sample_rate))
        output_audio[start:end] = loaded_candidates[str(decision["selected"])][start:end]

    for index in range(1, len(decisions)):
        previous_label = str(decisions[index - 1]["selected"])
        current_label = str(decisions[index]["selected"])
        if previous_label == current_label or crossfade_frames <= 0:
            continue
        boundary = int(round(float(decisions[index]["start_sec"]) * sample_rate))
        start = max(0, boundary - crossfade_frames)
        end = min(frames, boundary + crossfade_frames)
        if end <= start:
            continue
        alpha = np.linspace(0.0, 1.0, end - start, endpoint=False, dtype=np.float32).reshape(-1, 1)
        previous_audio = loaded_candidates[previous_label][start:end]
        current_audio = loaded_candidates[current_label][start:end]
        output_audio[start:end] = previous_audio * (1.0 - alpha) + current_audio * alpha

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_label
    sf.write(str(output_path), output_audio, sample_rate, subtype="PCM_24")
    candidate_manifest_path = output_dir / "candidate_manifest.json"
    candidate_manifest = {
        "schema_version": 1,
        "candidate_contract": {
            "route": route_name,
            "stem_name": stem_name,
            "stem_score_profile": stem_profile,
            "fallback_used": False,
            "not_timestamp_hardcoded": True,
            "per_track_hardcode_allowed": False,
            "reference_free_baseline_guard_enabled": baseline_candidate_label is not None and target_reference_path is None,
            "baseline_candidate_label": baseline_candidate_label,
            "baseline_switch_margin": config.baseline_switch_margin,
            "baseline_max_switch_fraction": config.baseline_max_switch_fraction,
            "baseline_calibration_profile_path": config.baseline_calibration_profile_path,
            "baseline_guard_confidence_metric": "score_delta_vs_baseline",
            "source_candidates": [candidate.label for candidate in candidates],
            "target_reference_used": target_reference_path is not None,
            "target_reference_assisted": target_reference_path is not None,
            "target_reference_used_at_application": target_reference_path is not None,
            "target_reference_path": str(target_reference_path) if target_reference_path is not None else None,
            "application_reference_free": target_reference_path is None,
            "delivery_allowed_without_reference": False,
            "no_reference_production_claim": False,
            "segment_seconds": config.segment_seconds,
            "crossfade_seconds": config.crossfade_seconds,
            "quality_gate_required": True,
            "delivery_status": "requires_objective_quality_gate",
        },
        "sample_rate": sample_rate,
        "stems": {
            stem_name: {
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "frames": int(output_audio.shape[0]),
                "rms_dbfs": round(_rms_dbfs(output_audio), 6),
            }
        },
    }
    candidate_manifest_path.write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")

    listen_pack: dict[str, str] = {}
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        listen_output = listen_dir / output_label
        shutil.copy2(output_path, listen_output)
        listen_pack[output_label] = str(listen_output)

    manifest = {
        "contract": {
            "route": route_name,
            "stem_name": stem_name,
            "stem_score_profile": stem_profile,
            "accepted_by_listening": False,
            "fallback_used": False,
            "not_timestamp_hardcoded": True,
            "per_track_hardcode_allowed": False,
            "reference_free_baseline_guard_enabled": baseline_candidate_label is not None and target_reference_path is None,
            "baseline_candidate_label": baseline_candidate_label,
            "baseline_switch_margin": config.baseline_switch_margin,
            "baseline_max_switch_fraction": config.baseline_max_switch_fraction,
            "baseline_calibration_profile_path": config.baseline_calibration_profile_path,
            "baseline_guard_confidence_metric": "score_delta_vs_baseline",
            "selection_scope": "segment_level_model_candidate",
            "target_reference_used": target_reference_path is not None,
            "target_reference_assisted": target_reference_path is not None,
            "target_reference_used_at_application": target_reference_path is not None,
            "application_reference_free": target_reference_path is None,
        },
        "config": {
            "segment_seconds": config.segment_seconds,
            "crossfade_seconds": config.crossfade_seconds,
            "min_rms_dbfs": config.min_rms_dbfs,
            "percussive_weight": config.percussive_weight,
            "harmonic_weight": config.harmonic_weight,
            "leakage_corr_weight": config.leakage_corr_weight,
            "vocal_leakage_multiplier": config.vocal_leakage_multiplier,
            "high_band_noise_weight": config.high_band_noise_weight,
            "target_corr_weight": config.target_corr_weight,
            "target_rms_delta_weight": config.target_rms_delta_weight,
            "quiet_penalty_weight": config.quiet_penalty_weight,
            "body_weight": config.body_weight,
            "switch_hysteresis": config.switch_hysteresis,
            "baseline_candidate_label": baseline_candidate_label,
            "baseline_switch_margin": config.baseline_switch_margin,
            "baseline_max_switch_fraction": config.baseline_max_switch_fraction,
            "baseline_calibration_profile_path": config.baseline_calibration_profile_path,
        },
        "target_reference_path": str(target_reference_path) if target_reference_path is not None else None,
        "candidates": [
            {
                "label": candidate.label,
                "stem_path": str(candidate.stem_path),
                "leakage_reference_paths": [str(path) for path in candidate.leakage_reference_paths],
            }
            for candidate in candidates
        ],
        "decisions": decisions,
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "sample_rate": sample_rate,
            "frames": int(output_audio.shape[0]),
            "rms_dbfs": round(_rms_dbfs(output_audio), 6),
        },
        "candidate_manifest": {
            "path": str(candidate_manifest_path),
            "sha256": _sha256(candidate_manifest_path),
        },
        "listen_pack": listen_pack,
    }
    manifest_path = output_dir / "drum_model_zoo_segment_router_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_candidate(value: str) -> CandidateSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use LABEL=PATH format")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return CandidateSpec(label=label, stem_path=Path(raw_path))


def _parse_reference(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("reference must use LABEL=PATH format")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("reference label cannot be empty")
    return label, Path(raw_path)


def _load_baseline_calibration_profile(profile_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    contract = profile.get("contract", {})
    if contract.get("target_reference_assisted") is not True:
        raise ValueError("baseline calibration profile must be target-reference-assisted.")
    if contract.get("calibration_only") is not True:
        raise ValueError("baseline calibration profile must be marked calibration_only.")
    if contract.get("same_track_delivery_allowed") is not False:
        raise ValueError("baseline calibration profile must forbid same-track delivery.")
    recommended_config = profile.get("recommended_router_config")
    if not isinstance(recommended_config, dict):
        raise ValueError("baseline calibration profile missing recommended_router_config.")
    return recommended_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route stem model-zoo candidates per segment with crossfades.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    parser.add_argument("--leakage-ref", action="append", type=_parse_reference, default=[])
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--output-label")
    parser.add_argument("--stem-name", default="drums")
    parser.add_argument("--target-ref", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=RouterConfig.segment_seconds)
    parser.add_argument("--crossfade-seconds", type=float, default=RouterConfig.crossfade_seconds)
    parser.add_argument("--min-rms-dbfs", type=float, default=RouterConfig.min_rms_dbfs)
    parser.add_argument("--switch-hysteresis", type=float, default=RouterConfig.switch_hysteresis)
    parser.add_argument("--percussive-weight", type=float, default=RouterConfig.percussive_weight)
    parser.add_argument("--harmonic-weight", type=float, default=RouterConfig.harmonic_weight)
    parser.add_argument("--leakage-corr-weight", type=float, default=RouterConfig.leakage_corr_weight)
    parser.add_argument("--vocal-leakage-multiplier", type=float, default=RouterConfig.vocal_leakage_multiplier)
    parser.add_argument("--high-band-noise-weight", type=float, default=RouterConfig.high_band_noise_weight)
    parser.add_argument("--quiet-penalty-weight", type=float, default=RouterConfig.quiet_penalty_weight)
    parser.add_argument("--body-weight", type=float, default=RouterConfig.body_weight)
    parser.add_argument("--target-corr-weight", type=float, default=RouterConfig.target_corr_weight)
    parser.add_argument("--target-rms-delta-weight", type=float, default=RouterConfig.target_rms_delta_weight)
    parser.add_argument("--baseline-candidate-label")
    parser.add_argument("--baseline-switch-margin", type=float)
    parser.add_argument("--baseline-max-switch-fraction", type=float)
    parser.add_argument("--baseline-calibration-profile", type=Path)
    return parser


def main() -> None:
    require_docker_runtime(
        tool_name="route_drum_model_zoo_segments.py",
        entrypoint_path="/app/tools/route_drum_model_zoo_segments.py",
    )
    args = _build_arg_parser().parse_args()
    calibration_config: dict[str, Any] = {}
    if args.baseline_calibration_profile is not None:
        calibration_config = _load_baseline_calibration_profile(args.baseline_calibration_profile)
    baseline_candidate_label = args.baseline_candidate_label or calibration_config.get("baseline_candidate_label")
    baseline_switch_margin = (
        args.baseline_switch_margin
        if args.baseline_switch_margin is not None
        else calibration_config.get("baseline_switch_margin", RouterConfig.baseline_switch_margin)
    )
    baseline_max_switch_fraction = (
        args.baseline_max_switch_fraction
        if args.baseline_max_switch_fraction is not None
        else calibration_config.get("baseline_max_switch_fraction")
    )
    references_by_label: dict[str, list[Path]] = {}
    for label, reference_path in args.leakage_ref:
        references_by_label.setdefault(label, []).append(reference_path)
    candidates = [
        CandidateSpec(
            label=candidate.label,
            stem_path=candidate.stem_path,
            leakage_reference_paths=tuple(references_by_label.get(candidate.label, [])),
        )
        for candidate in args.candidate
    ]
    manifest = route_drum_segments(
        candidates,
        args.output_dir,
        listen_dir=args.listen_dir,
        output_label=args.output_label,
        stem_name=str(args.stem_name),
        target_reference_path=args.target_ref,
        config=RouterConfig(
            segment_seconds=float(args.segment_seconds),
            crossfade_seconds=float(args.crossfade_seconds),
            min_rms_dbfs=float(args.min_rms_dbfs),
            percussive_weight=float(args.percussive_weight),
            harmonic_weight=float(args.harmonic_weight),
            leakage_corr_weight=float(args.leakage_corr_weight),
            vocal_leakage_multiplier=float(args.vocal_leakage_multiplier),
            high_band_noise_weight=float(args.high_band_noise_weight),
            quiet_penalty_weight=float(args.quiet_penalty_weight),
            body_weight=float(args.body_weight),
            switch_hysteresis=float(args.switch_hysteresis),
            target_corr_weight=float(args.target_corr_weight),
            target_rms_delta_weight=float(args.target_rms_delta_weight),
            baseline_candidate_label=baseline_candidate_label,
            baseline_switch_margin=float(baseline_switch_margin),
            baseline_max_switch_fraction=baseline_max_switch_fraction,
            baseline_calibration_profile_path=(
                str(args.baseline_calibration_profile) if args.baseline_calibration_profile is not None else None
            ),
        ),
    )
    print(json.dumps({"output": manifest["output"], "listen_pack": manifest["listen_pack"]}, indent=2))


if __name__ == "__main__":
    main()
