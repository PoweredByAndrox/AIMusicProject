#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_ROOT.parent
TOOLS_ROOT = ROOT_DIR / "tools"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from run_nine_thou_quality_gate import (  # noqa: E402
    DEFAULT_MAX_OFFSET_SECONDS,
    EPSILON,
    align_pair_with_indices,
    basic_metrics,
    best_gain,
    bss_eval_like_metrics,
    cosine_similarity,
    decode_mono,
    estimate_offset_seconds,
    scale_invariant_signal_to_distortion_db,
    sha256_file,
    signal_to_distortion_db,
    utc_timestamp,
    validate_equal_length,
    write_json,
)
from waveform_forensic_compare import compare_waveforms_forensically, write_waveform_forensic_evidence_pack  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("/app/lab_artifacts/ground_truth_quality/synthetic_smoke")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION_SECONDS = 6.0
DEFAULT_STEMS = ("vocals", "drums", "bass", "other")
DEFAULT_HOP_SECONDS = 0.05
QUALITY_WINDOW_SECONDS = 5.0
QUALITY_WINDOW_MAX_ITEMS = 5
MICRO_QUALITY_WINDOW_SECONDS = (0.1, 0.25, 0.5)
MICRO_QUALITY_HOP_RATIO = 0.5
MICRO_QUALITY_MAX_ITEMS = 12
MICRO_QUALITY_MAX_ITEMS_PER_SOURCE = 6
MICRO_QUALITY_TIME_BUCKET_SECONDS = 10.0
MICRO_QUALITY_MAX_ITEMS_PER_TIME_BUCKET = 4
MICRO_RESIDUAL_SPIKE_THRESHOLD_DBFS = -45.0
MICRO_TRANSIENT_MISMATCH_THRESHOLD_DBFS = -50.0
PHRASE_QUALITY_WINDOW_SECONDS = (1.0, 2.0, 4.0)
PHRASE_QUALITY_HOP_RATIO = 0.25
PHRASE_QUALITY_MAX_ITEMS = 8
PHRASE_RESIDUAL_RMS_THRESHOLD_DBFS = -60.0
PHRASE_RESIDUAL_LEAKAGE_STRENGTH_THRESHOLD = 0.05
PHRASE_HARMONIC_FLATNESS_THRESHOLD = 0.30
PHRASE_HARMONIC_PEAK_TO_MEAN_THRESHOLD_DB = 10.0
DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM = "drums_plus_percussive_fx_aux"
DERIVED_REFERENCE_STEMS = frozenset({"instrumental", DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM})
LEAKAGE_ACTIVE_REFERENCE_FLOOR_DBFS = -80.0


@dataclass(frozen=True)
class ObjectiveThresholds:
    min_sdr_db: float = 20.0
    min_si_sdr_db: float = 20.0
    min_sir_db: float = 20.0
    min_sar_db: float = 20.0
    min_reference_likeness_percent: float = 95.0


@dataclass(frozen=True)
class StemBundle:
    manifest_path: Path
    manifest: dict[str, Any]
    stems: dict[str, Path]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def resolve_stem_bundle(manifest_path: Path) -> StemBundle:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    stems = manifest.get("stems")
    if not isinstance(stems, dict) or not stems:
        raise RuntimeError(f"Manifest has no stems object: {manifest_path}")

    resolved: dict[str, Path] = {}
    for stem_name, stem_info in stems.items():
        if not isinstance(stem_info, dict) or not stem_info.get("path"):
            raise RuntimeError(f"Stem entry has no path: {stem_name}")
        stem_path = resolve_manifest_path(manifest_path, str(stem_info["path"]))
        if not stem_path.exists():
            raise FileNotFoundError(stem_path)
        resolved[str(stem_name)] = stem_path
    return StemBundle(manifest_path=manifest_path, manifest=manifest, stems=resolved)


def validate_verified_ground_truth(bundle: StemBundle) -> dict[str, Any]:
    contract = bundle.manifest.get("ground_truth_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Ground-truth manifest must include ground_truth_contract")
    if contract.get("verified") is not True:
        raise RuntimeError("Ground-truth manifest must set ground_truth_contract.verified=true")
    if not contract.get("verification_method"):
        raise RuntimeError("Ground-truth manifest must include ground_truth_contract.verification_method")
    if not contract.get("source"):
        raise RuntimeError("Ground-truth manifest must include ground_truth_contract.source")
    return contract


def verify_bundle_hashes(bundle: StemBundle, *, require_hashes: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    stems = bundle.manifest.get("stems")
    if not isinstance(stems, dict):
        raise RuntimeError(f"Manifest has no stems object: {bundle.manifest_path}")

    for stem_name, stem_path in bundle.stems.items():
        stem_info = stems.get(stem_name)
        if not isinstance(stem_info, dict):
            raise RuntimeError(f"Stem entry is missing in manifest: {stem_name}")
        expected_hash = stem_info.get("sha256")
        if not expected_hash:
            if require_hashes:
                raise RuntimeError(f"Stem entry must include sha256: {stem_name}")
            continue
        actual_hash = sha256_file(stem_path)
        if str(expected_hash).lower() != actual_hash.lower():
            raise RuntimeError(
                f"Stem sha256 mismatch: stem={stem_name} expected={expected_hash} actual={actual_hash}"
            )
        checks.append(
            {
                "kind": "stem",
                "name": stem_name,
                "path": str(stem_path),
                "sha256": actual_hash,
            }
        )

    mix_info = bundle.manifest.get("mix")
    if isinstance(mix_info, dict) and mix_info.get("path"):
        mix_path = resolve_manifest_path(bundle.manifest_path, str(mix_info["path"]))
        expected_hash = mix_info.get("sha256")
        if not expected_hash:
            if require_hashes:
                raise RuntimeError("Mix entry must include sha256")
        elif mix_path.exists():
            actual_hash = sha256_file(mix_path)
            if str(expected_hash).lower() != actual_hash.lower():
                raise RuntimeError(f"Mix sha256 mismatch: expected={expected_hash} actual={actual_hash}")
            checks.append(
                {
                    "kind": "mix",
                    "name": "mix",
                    "path": str(mix_path),
                    "sha256": actual_hash,
                }
            )
        elif require_hashes:
            raise FileNotFoundError(mix_path)

    return checks


def threshold_check(name: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": round(float(value), 6),
        "threshold": threshold,
        "passed": float(value) >= threshold,
    }


def build_verdict(checks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failed_checks = [check for check in checks if not check["passed"]]
    return {
        "status": "failed" if failed_checks else "passed",
        "failed_checks": failed_checks,
        "failed_check_names": [str(check["name"]) for check in failed_checks],
        "active_check_count": len(checks),
    }


def spectral_convergence(reference: np.ndarray, estimate: np.ndarray) -> float:
    validate_equal_length(reference, estimate)
    reference_spectrum = np.abs(np.fft.rfft(reference.astype(np.float64, copy=False)))
    estimate_spectrum = np.abs(np.fft.rfft(estimate.astype(np.float64, copy=False)))
    return float(np.linalg.norm(reference_spectrum - estimate_spectrum) / (np.linalg.norm(reference_spectrum) + EPSILON))


def clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def cosine_to_percent(value: float) -> float:
    return clamp_percent(100.0 * max(0.0, float(value)))


def convergence_to_percent(value: float) -> float:
    return clamp_percent(100.0 * (1.0 - min(1.0, max(0.0, float(value)))))


def frame_rms_envelope(signal: np.ndarray, frame_size: int) -> np.ndarray:
    if len(signal) == 0:
        return np.zeros(1, dtype=np.float64)
    frame_size = max(1, frame_size)
    values: list[float] = []
    for start in range(0, len(signal), frame_size):
        frame = signal[start : start + frame_size]
        values.append(float(np.sqrt(np.mean(np.square(frame), dtype=np.float64) + EPSILON)))
    return np.asarray(values, dtype=np.float64)


def envelope_similarity_percent(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    frame_size = max(1, int(round(0.05 * sample_rate)))
    reference_envelope = frame_rms_envelope(reference, frame_size)
    estimate_envelope = frame_rms_envelope(estimate, frame_size)
    length = min(len(reference_envelope), len(estimate_envelope))
    if length <= 0:
        return 0.0
    reference_envelope = reference_envelope[:length]
    estimate_envelope = estimate_envelope[:length]
    gain = best_gain(estimate_envelope, reference_envelope)
    matched = gain * estimate_envelope
    relative_error = float(
        np.linalg.norm(reference_envelope - matched) / (np.linalg.norm(reference_envelope) + EPSILON)
    )
    return convergence_to_percent(relative_error)


def transient_similarity_percent(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    frame_size = max(1, int(round(0.02 * sample_rate)))
    reference_envelope = frame_rms_envelope(reference, frame_size)
    estimate_envelope = frame_rms_envelope(estimate, frame_size)
    if len(reference_envelope) < 2 or len(estimate_envelope) < 2:
        return envelope_similarity_percent(reference, estimate, sample_rate)
    reference_transients = np.maximum(0.0, np.diff(reference_envelope))
    estimate_transients = np.maximum(0.0, np.diff(estimate_envelope))
    length = min(len(reference_transients), len(estimate_transients))
    if length <= 0:
        return 0.0
    reference_transients = reference_transients[:length]
    estimate_transients = estimate_transients[:length]
    gain = best_gain(estimate_transients, reference_transients)
    matched = gain * estimate_transients
    relative_error = float(
        np.linalg.norm(reference_transients - matched) / (np.linalg.norm(reference_transients) + EPSILON)
    )
    return convergence_to_percent(relative_error)


def read_bundle_audio(bundle: StemBundle, stem_names: Sequence[str], sample_rate: int) -> dict[str, np.ndarray]:
    decoded: dict[str, np.ndarray] = {}
    for stem_name in stem_names:
        if stem_name not in bundle.stems:
            raise RuntimeError(f"Manifest {bundle.manifest_path} has no requested stem: {stem_name}")
        decoded[stem_name] = decode_mono(bundle.stems[stem_name], sample_rate)
    return decoded


def reference_stem_names_for_quality(bundle: StemBundle, requested_stems: Sequence[str]) -> list[str]:
    requested = set(requested_stems)
    derived_stems = derived_reference_stems_for_bundle(bundle)
    names: list[str] = []
    for stem_name in bundle.stems:
        if stem_name in derived_stems and stem_name not in requested:
            continue
        names.append(stem_name)
    return names


def derived_reference_stems_for_bundle(bundle: StemBundle) -> set[str]:
    derived_stems = set(DERIVED_REFERENCE_STEMS)
    stems = bundle.manifest.get("stems")
    if isinstance(stems, dict):
        for stem_name, stem_info in stems.items():
            if isinstance(stem_info, dict) and bool(stem_info.get("derived")):
                derived_stems.add(str(stem_name))
    target_definitions = bundle.manifest.get("target_definitions")
    if isinstance(target_definitions, dict):
        for stem_name, target_definition in target_definitions.items():
            if isinstance(target_definition, dict) and bool(target_definition.get("derived")):
                derived_stems.add(str(stem_name))
    return derived_stems


def child_target_stems_for_derived(bundle: StemBundle, derived_stem: str) -> set[str]:
    target_definitions = bundle.manifest.get("target_definitions")
    if isinstance(target_definitions, dict):
        target_definition = target_definitions.get(derived_stem)
        if isinstance(target_definition, dict) and isinstance(target_definition.get("child_stems"), list):
            return {
                str(stem_name)
                for stem_name in target_definition["child_stems"]
                if str(stem_name) in bundle.stems
            }
    if derived_stem == "instrumental":
        return {
            stem_name
            for stem_name in bundle.stems
            if stem_name not in DERIVED_REFERENCE_STEMS and stem_name != "vocals"
        }
    if derived_stem == DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM:
        return {stem_name for stem_name in ("drums", "fx") if stem_name in bundle.stems}
    return set()


def validate_objective_stem_selection(bundle: StemBundle, requested_stems: Sequence[str]) -> None:
    requested = set(requested_stems)
    for derived_stem in sorted(requested & derived_reference_stems_for_bundle(bundle)):
        child_targets = sorted(requested & child_target_stems_for_derived(bundle, derived_stem))
        if child_targets:
            raise RuntimeError(
                "derived reference stem cannot be evaluated with child target stems in the same objective report: "
                f"derived={derived_stem} child_target_stems={child_targets}"
            )


def validate_reference_lengths(reference_stems: dict[str, np.ndarray]) -> int:
    lengths = {stem_name: len(audio) for stem_name, audio in reference_stems.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise RuntimeError(f"Ground-truth stems must have equal decoded lengths: {lengths}")
    length = next(iter(unique_lengths))
    if length <= 0:
        raise RuntimeError("Ground-truth stems must not be empty")
    return length


def sum_interference(reference_stems: dict[str, np.ndarray], target_stem: str, start: int, sample_count: int) -> np.ndarray:
    interference: np.ndarray | None = None
    for stem_name, audio in reference_stems.items():
        if stem_name == target_stem:
            continue
        stem_slice = audio[start : start + sample_count]
        interference = stem_slice.astype(np.float32, copy=True) if interference is None else interference + stem_slice
    if interference is None:
        return np.zeros(sample_count, dtype=np.float32)
    return interference.astype(np.float32, copy=False)


def delivery_decision(
    overall_status: str,
    *,
    objective_failed: bool = False,
    forensic_contract_status: str | None = None,
    candidate_manifest_contract_status: str | None = None,
    candidate_manifest_contract_reason: str | None = None,
) -> dict[str, str]:
    if candidate_manifest_contract_status == "rejected":
        return {
            "status": "rejected",
            "reason": candidate_manifest_contract_reason or "candidate_manifest_delivery_blocked",
            "forensic_contract_status": forensic_contract_status or "not_evaluated",
            "candidate_manifest_contract_status": candidate_manifest_contract_status,
        }
    if overall_status == "passed":
        return {
            "status": "accepted",
            "reason": "objective_threshold_passed",
            "forensic_contract_status": forensic_contract_status or "not_evaluated",
            "candidate_manifest_contract_status": candidate_manifest_contract_status or "not_evaluated",
        }
    reason = "objective_threshold_failed" if objective_failed else "forensic_contract_failed"
    return {
        "status": "rejected",
        "reason": reason,
        "forensic_contract_status": forensic_contract_status or "not_evaluated",
        "candidate_manifest_contract_status": candidate_manifest_contract_status or "not_evaluated",
    }


def candidate_manifest_delivery_contract(candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    contract = candidate_manifest.get("contract")
    if not isinstance(contract, dict):
        return {
            "schema_version": 1,
            "status": "accepted",
            "delivery_allowed": True,
            "reason": "no_blocking_candidate_contract",
            "blocking_flags": [],
        }

    blocking_flags: list[str] = []
    delivery_allowed_without_reference = contract.get("delivery_allowed_without_reference")
    if delivery_allowed_without_reference is False:
        if contract.get("target_reference_reconstructive_prior") is True:
            blocking_flags.append("target_reference_reconstructive_prior")
        if contract.get("stores_timestamp_rules") is True:
            blocking_flags.append("stores_timestamp_rules")
        if contract.get("target_reference_used_at_application") is True:
            blocking_flags.append("target_reference_used_at_application")
        if contract.get("target_reference_assisted") is True:
            blocking_flags.append("target_reference_assisted")
        if contract.get("not_a_no_reference_production_route_until_holdout_passes") is True:
            blocking_flags.append("not_a_no_reference_production_route_until_holdout_passes")
        if contract.get("requires_holdout_validation") is True and contract.get("holdout_validation_status") != "passed":
            blocking_flags.append("holdout_validation_missing")

    status = "rejected" if blocking_flags else "accepted"
    return {
        "schema_version": 1,
        "status": status,
        "delivery_allowed": status == "accepted",
        "reason": "candidate_manifest_delivery_blocked" if status == "rejected" else "candidate_contract_allows_delivery",
        "blocking_flags": sorted(set(blocking_flags)),
        "contract": contract,
    }


def projection_gain(reference: np.ndarray, estimate: np.ndarray) -> float:
    validate_equal_length(reference, estimate)
    denominator = float(np.dot(reference, reference) + EPSILON)
    return float(np.dot(estimate, reference) / denominator)


def rms_dbfs(signal: np.ndarray) -> float:
    if signal.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)) + EPSILON)
    return 20.0 * math.log10(rms)


def leakage_projection_matrix(
    *,
    stem_name: str,
    reference_stems: dict[str, np.ndarray],
    candidate_aligned: np.ndarray,
    reference_start: int,
    sample_count: int,
) -> dict[str, Any]:
    sources: dict[str, dict[str, float]] = {}
    candidate_rms_dbfs = rms_dbfs(candidate_aligned)
    for reference_name, reference_audio in reference_stems.items():
        reference_slice = reference_audio[reference_start : reference_start + sample_count]
        reference_rms_dbfs = rms_dbfs(reference_slice)
        reference_active = reference_rms_dbfs >= LEAKAGE_ACTIVE_REFERENCE_FLOOR_DBFS
        gain = (
            projection_gain(reference_slice.astype(np.float64, copy=False), candidate_aligned.astype(np.float64, copy=False))
            if reference_active
            else 0.0
        )
        source_cosine = cosine_similarity(reference_slice, candidate_aligned) if reference_active else 0.0
        leakage_strength = (
            abs(gain) * (10.0 ** ((reference_rms_dbfs - candidate_rms_dbfs) / 20.0))
            if reference_active and candidate_rms_dbfs > LEAKAGE_ACTIVE_REFERENCE_FLOOR_DBFS
            else 0.0
        )
        sources[reference_name] = {
            "reference_active": reference_active,
            "reference_rms_dbfs": round(reference_rms_dbfs, 6),
            "projection_gain": round(gain, 9),
            "abs_projection_gain": round(abs(gain), 9),
            "leakage_strength": round(leakage_strength, 9),
            "cosine_similarity": round(source_cosine, 6),
        }
    target_projection = sources.get(stem_name, {}).get("projection_gain", 0.0)
    leakage_sources = {
        name: source for name, source in sources.items() if name != stem_name
    }
    max_leakage_source = max(
        leakage_sources.items(),
        key=lambda item: item[1]["leakage_strength"],
        default=(None, {"abs_projection_gain": 0.0, "leakage_strength": 0.0}),
    )
    return {
        "target_stem": stem_name,
        "target_projection_gain": target_projection,
        "max_leakage_source": max_leakage_source[0],
        "max_leakage_projection_gain": max_leakage_source[1]["abs_projection_gain"],
        "max_leakage_strength": max_leakage_source[1]["leakage_strength"],
        "sources": sources,
    }


def window_issue_labels(metrics: dict[str, float], thresholds: ObjectiveThresholds) -> list[str]:
    labels: list[str] = []
    if metrics["reference_likeness_percent"] < thresholds.min_reference_likeness_percent:
        labels.append("reference_likeness_below_threshold")
    if metrics["sdr_db"] < thresholds.min_sdr_db:
        labels.append("low_sdr")
    if metrics["si_sdr_db"] < thresholds.min_si_sdr_db:
        labels.append("low_si_sdr")
    if metrics["sar_db"] < thresholds.min_sar_db:
        labels.append("artifact_or_timbre_damage")
    if metrics["cosine_similarity"] < 0.9:
        labels.append("target_shape_mismatch")
    if abs(metrics["rms_drift_db"]) > 3.0:
        labels.append("level_or_body_drift")
    if metrics["max_leakage_strength"] > 0.05:
        labels.append("reference_stem_leakage")
    return labels


def reference_likeness_issue_labels(
    components: dict[str, float],
    score_percent: float,
    thresholds: ObjectiveThresholds,
) -> list[str]:
    labels: list[str] = []
    if score_percent < thresholds.min_reference_likeness_percent:
        labels.append("reference_likeness_below_threshold")
    if components["waveform_shape"] < 95.0:
        labels.append("waveform_shape_mismatch")
    if components["spectrum"] < 95.0:
        labels.append("spectral_timbre_mismatch")
    if components["energy_envelope"] < 95.0:
        labels.append("energy_envelope_mismatch")
    if components["transient_envelope"] < 95.0:
        labels.append("transient_or_attack_mismatch")
    if components["leakage_cleanliness"] < 95.0:
        labels.append("non_target_leakage")
    return labels


def build_problem_summary(issue_labels: Sequence[str]) -> list[str]:
    descriptions = {
        "reference_likeness_below_threshold": "candidate is below the 95 percent reference-likeness gate",
        "waveform_shape_mismatch": "waveform shape differs from the verified reference stem",
        "spectral_timbre_mismatch": "spectral/timbre content differs from the verified reference stem",
        "energy_envelope_mismatch": "loudness/body envelope does not track the reference stem",
        "transient_or_attack_mismatch": "attacks or transient envelope differ from the reference stem",
        "non_target_leakage": "candidate contains measurable energy explained by non-target reference stems",
        "low_sdr": "overall signal reconstruction error is too high",
        "low_si_sdr": "scale-invariant reconstruction error is too high",
        "artifact_or_timbre_damage": "artifact/timbre projection score is below threshold",
        "target_shape_mismatch": "candidate shape has weak correlation with target reference",
        "level_or_body_drift": "candidate level/body differs from the reference",
        "reference_stem_leakage": "named reference-stem leakage is above threshold",
        "micro_residual_spike": "short residual spike differs from the verified reference waveform",
        "micro_transient_mismatch": "short attack or transient event differs from the verified reference waveform",
        "micro_target_transient_dropout": "short target transient is weaker than the verified reference waveform",
        "micro_reference_stem_leakage": "short window contains named reference-stem leakage",
        "micro_residual_reference_stem_leakage": (
            "short residual is explained by a non-target reference stem after subtracting the verified target"
        ),
        "bar_spanning_drop_melody_leakage": (
            "long residual contains sustained non-drum harmonic/drop melody over beat/bar/phrase windows"
        ),
        "micro_level_or_body_drift": "short window level/body differs from the reference",
    }
    return [descriptions[label] for label in issue_labels if label in descriptions]


FORENSIC_DELIVERY_ACTIONS: dict[str, tuple[str, str]] = {
    "min_sdr_db": ("reroute_required", "overall reconstruction quality is below the objective SDR gate"),
    "min_si_sdr_db": ("reroute_required", "scale-invariant reconstruction quality is below the objective gate"),
    "min_sir_db": ("debleed_or_reroute_required", "interference rejection is below the objective SIR gate"),
    "min_sar_db": ("repair_candidate_required", "artifact rejection is below the objective SAR gate"),
    "reference_likeness_below_threshold": (
        "reroute_required",
        "candidate failed the verified reference-likeness gate",
    ),
    "waveform_shape_mismatch": ("repair_candidate_required", "waveform shape differs from the verified stem"),
    "spectral_timbre_mismatch": ("repair_candidate_required", "spectral/timbre content differs from the verified stem"),
    "energy_envelope_mismatch": ("repair_candidate_required", "energy/body envelope differs from the verified stem"),
    "transient_or_attack_mismatch": ("repair_candidate_required", "attack/transient envelope differs from the verified stem"),
    "non_target_leakage": ("debleed_or_reroute_required", "candidate contains measurable non-target reference leakage"),
    "low_sdr": ("reroute_required", "window reconstruction error is too high"),
    "low_si_sdr": ("reroute_required", "window scale-invariant reconstruction error is too high"),
    "artifact_or_timbre_damage": ("repair_candidate_required", "window artifact/timbre projection is below threshold"),
    "target_shape_mismatch": ("repair_candidate_required", "window waveform shape has weak target correlation"),
    "level_or_body_drift": ("repair_candidate_required", "window level/body differs from reference"),
    "reference_stem_leakage": ("debleed_or_reroute_required", "window contains named reference-stem leakage"),
    "sample_residual_above_threshold": ("repair_candidate_required", "sample-level residual exceeds forensic threshold"),
    "micro_residual_spike": ("repair_candidate_required", "short residual spike exceeds forensic threshold"),
    "micro_transient_mismatch": ("repair_candidate_required", "short attack/transient event differs from reference"),
    "micro_target_transient_dropout": ("reroute_required", "short target transient is weaker than reference"),
    "micro_reference_stem_leakage": ("debleed_or_reroute_required", "short window contains reference-stem leakage"),
    "micro_residual_reference_stem_leakage": (
        "debleed_or_reroute_required",
        "short residual contains non-target reference-stem leakage after target subtraction",
    ),
    "bar_spanning_drop_melody_leakage": (
        "debleed_or_reroute_required",
        "long phrase-level non-drum harmonic content remains in target stem",
    ),
    "micro_level_or_body_drift": ("repair_candidate_required", "short-window level/body differs from reference"),
}


def issue_delivery_action(label: str) -> dict[str, Any]:
    action, reason = FORENSIC_DELIVERY_ACTIONS.get(
        label,
        ("explicit_rejection_reason", "detected issue has no specialized repair route yet"),
    )
    return {
        "label": label,
        "action": action,
        "reason": reason,
    }


def build_stem_forensic_delivery_contract(
    *,
    stem_name: str,
    verdict: dict[str, Any],
    reference_likeness: dict[str, Any],
    waveform_forensics: dict[str, Any],
    quality_windows: dict[str, Any],
    micro_quality_windows: dict[str, Any],
) -> dict[str, Any]:
    label_sources: dict[str, set[str]] = {}

    def add_label(label: str, source: str) -> None:
        label_sources.setdefault(label, set()).add(source)

    if verdict.get("status") == "failed":
        for label in verdict.get("failed_check_names", []):
            add_label(str(label), "thresholds")
        for label in reference_likeness.get("issue_labels", []):
            add_label(str(label), "reference_likeness")
        for window in quality_windows.get("worst_windows", []):
            if float(window.get("severity_score", 0.0)) <= 0.0:
                continue
            for label in window.get("issue_labels", []):
                add_label(str(label), "quality_windows")

    if waveform_forensics.get("status") == "failed":
        for defect_range in waveform_forensics.get("defect_ranges", []):
            for label in defect_range.get("labels", []):
                add_label(str(label), "waveform_forensics")

    if int(micro_quality_windows.get("defect_window_count", 0)) > 0:
        for label in micro_quality_windows.get("issue_label_counts", {}):
            add_label(str(label), "micro_quality_windows")
        for window in micro_quality_windows.get("worst_windows", []):
            for label in window.get("issue_labels", []):
                add_label(str(label), "micro_quality_windows")

    issue_actions = []
    for label in sorted(label_sources):
        issue_action = issue_delivery_action(label)
        issue_action["evidence_sources"] = sorted(label_sources[label])
        issue_actions.append(issue_action)

    status = "rejected" if issue_actions else "accepted"
    return {
        "schema_version": 1,
        "stem": stem_name,
        "status": status,
        "delivery_allowed": status == "accepted",
        "delivery_action": "accept_for_delivery" if status == "accepted" else "reject_delivery_candidate",
        "blocking_issue_labels": [item["label"] for item in issue_actions],
        "uncovered_labels": [],
        "issue_actions": issue_actions,
    }


def build_report_forensic_delivery_contract(stem_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rejected_stems = [
        str(report["stem"])
        for report in stem_reports
        if report.get("forensic_delivery_contract", {}).get("status") != "accepted"
    ]
    issue_actions_by_label: dict[str, dict[str, Any]] = {}
    for report in stem_reports:
        contract = report.get("forensic_delivery_contract", {})
        for issue_action in contract.get("issue_actions", []):
            label = str(issue_action["label"])
            if label not in issue_actions_by_label:
                issue_actions_by_label[label] = {
                    "label": label,
                    "action": issue_action["action"],
                    "reason": issue_action["reason"],
                    "stems": [],
                    "evidence_sources": set(),
                }
            issue_actions_by_label[label]["stems"].append(str(report["stem"]))
            issue_actions_by_label[label]["evidence_sources"].update(issue_action.get("evidence_sources", []))

    issue_actions = []
    for label in sorted(issue_actions_by_label):
        action = issue_actions_by_label[label]
        issue_actions.append(
            {
                "label": action["label"],
                "action": action["action"],
                "reason": action["reason"],
                "stems": sorted(set(action["stems"])),
                "evidence_sources": sorted(action["evidence_sources"]),
            }
        )
    available_evidence_artifacts = sorted(
        {
            "waveform_forensic_evidence"
            for report in stem_reports
            if report.get("waveform_forensic_evidence", {}).get("status") == "available"
        }
    )
    missing_evidence_stems = sorted(
        str(report["stem"])
        for report in stem_reports
        if str(report["stem"]) in rejected_stems
        and report.get("waveform_forensic_evidence", {}).get("status") != "available"
    )

    return {
        "schema_version": 1,
        "status": "rejected" if rejected_stems else "accepted",
        "delivery_allowed": not rejected_stems,
        "accepted_stems": [str(report["stem"]) for report in stem_reports if str(report["stem"]) not in rejected_stems],
        "rejected_stems": rejected_stems,
        "blocking_issue_labels": [item["label"] for item in issue_actions],
        "uncovered_labels": [],
        "issue_actions": issue_actions,
        "evidence_requirements": {
            "required_for_rejected_delivery": ["waveform_forensic_evidence"] if rejected_stems else [],
            "available_artifacts": available_evidence_artifacts,
            "missing_artifact_stems": missing_evidence_stems,
        },
    }


def build_reference_likeness(
    *,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    leakage_matrix: dict[str, Any],
    thresholds: ObjectiveThresholds,
) -> dict[str, Any]:
    validate_equal_length(reference_audio, candidate_audio)
    spectrum_score = convergence_to_percent(spectral_convergence(reference_audio, candidate_audio))
    shape_score = cosine_to_percent(cosine_similarity(reference_audio, candidate_audio))
    envelope_score = envelope_similarity_percent(reference_audio, candidate_audio, sample_rate)
    transient_score = transient_similarity_percent(reference_audio, candidate_audio, sample_rate)
    structural_floor = min(shape_score, spectrum_score, envelope_score, transient_score)
    effective_leakage_strength = float(leakage_matrix["max_leakage_strength"]) * max(0.0, 1.0 - structural_floor / 100.0)
    components = {
        "waveform_shape": round(shape_score, 6),
        "spectrum": round(spectrum_score, 6),
        "energy_envelope": round(envelope_score, 6),
        "transient_envelope": round(transient_score, 6),
        "leakage_cleanliness": round(
            clamp_percent(100.0 * (1.0 - min(1.0, effective_leakage_strength / 0.05))),
            6,
        ),
    }
    score_percent = round(
        0.25 * components["waveform_shape"]
        + 0.25 * components["spectrum"]
        + 0.2 * components["energy_envelope"]
        + 0.15 * components["transient_envelope"]
        + 0.15 * components["leakage_cleanliness"],
        6,
    )
    issue_labels = reference_likeness_issue_labels(components, score_percent, thresholds)
    return {
        "score_percent": score_percent,
        "threshold_percent": thresholds.min_reference_likeness_percent,
        "status": "passed" if score_percent >= thresholds.min_reference_likeness_percent else "failed",
        "components": components,
        "effective_leakage_strength": round(effective_leakage_strength, 9),
        "raw_max_leakage_strength": leakage_matrix["max_leakage_strength"],
        "issue_labels": issue_labels,
        "problem_summary": build_problem_summary(issue_labels),
    }


def build_quality_window_ledger(
    *,
    stem_name: str,
    reference_stems: dict[str, np.ndarray],
    reference_aligned: np.ndarray,
    candidate_aligned: np.ndarray,
    reference_start: int,
    sample_rate: int,
    thresholds: ObjectiveThresholds,
) -> dict[str, Any]:
    window_frames = max(1, int(round(QUALITY_WINDOW_SECONDS * sample_rate)))
    windows: list[dict[str, Any]] = []
    aligned_frames = len(reference_aligned)
    for index, start in enumerate(range(0, aligned_frames, window_frames)):
        end = min(aligned_frames, start + window_frames)
        if end <= start:
            continue
        reference_window = reference_aligned[start:end]
        candidate_window = candidate_aligned[start:end]
        leakage_matrix = leakage_projection_matrix(
            stem_name=stem_name,
            reference_stems=reference_stems,
            candidate_aligned=candidate_window,
            reference_start=reference_start + start,
            sample_count=end - start,
        )
        reference_likeness = build_reference_likeness(
            reference_audio=reference_window,
            candidate_audio=candidate_window,
            sample_rate=sample_rate,
            leakage_matrix=leakage_matrix,
            thresholds=thresholds,
        )
        bss_like = bss_eval_like_metrics(
            target_reference=reference_window,
            interference_reference=sum_interference(
                reference_stems,
                target_stem=stem_name,
                start=reference_start + start,
                sample_count=end - start,
            ),
            estimate=candidate_window,
        )
        gain = best_gain(candidate_window, reference_window)
        metrics = {
            "sdr_db": round(signal_to_distortion_db(reference_window, candidate_window), 6),
            "si_sdr_db": round(scale_invariant_signal_to_distortion_db(reference_window, candidate_window), 6),
            "sir_db": bss_like["bss_eval_like_sir_db"],
            "sar_db": bss_like["bss_eval_like_sar_db"],
            "cosine_similarity": round(cosine_similarity(reference_window, candidate_window), 6),
            "rms_drift_db": round(
                basic_metrics(candidate_window, sample_rate)["rms_dbfs"]
                - basic_metrics(reference_window, sample_rate)["rms_dbfs"],
                6,
            ),
            "candidate_gain_to_reference": round(gain, 9),
            "max_leakage_projection_gain": leakage_matrix["max_leakage_projection_gain"],
            "max_leakage_strength": leakage_matrix["max_leakage_strength"],
            "reference_likeness_percent": reference_likeness["score_percent"],
        }
        labels = window_issue_labels(metrics, thresholds)
        severity_score = (
            max(0.0, thresholds.min_reference_likeness_percent - metrics["reference_likeness_percent"])
            + max(0.0, thresholds.min_sdr_db - metrics["sdr_db"])
            + max(0.0, thresholds.min_si_sdr_db - metrics["si_sdr_db"])
            + max(0.0, thresholds.min_sar_db - metrics["sar_db"])
            + 20.0 * metrics["max_leakage_strength"]
            + max(0.0, abs(metrics["rms_drift_db"]) - 3.0)
        )
        windows.append(
            {
                "window_index": index,
                "start_sec": round(start / float(sample_rate), 6),
                "end_sec": round(end / float(sample_rate), 6),
                "severity_score": round(severity_score, 6),
                "issue_labels": labels,
                "problem_summary": build_problem_summary(labels),
                "reference_likeness": reference_likeness,
                "metrics": metrics,
            }
        )
    worst_windows = sorted(windows, key=lambda item: item["severity_score"], reverse=True)[:QUALITY_WINDOW_MAX_ITEMS]
    return {
        "window_seconds": QUALITY_WINDOW_SECONDS,
        "window_count": len(windows),
        "worst_windows": worst_windows,
    }


def amplitude_dbfs(amplitude: float) -> float:
    if amplitude <= EPSILON:
        return -180.0
    return float(20.0 * math.log10(max(float(amplitude), EPSILON)))


def transient_peak(audio: np.ndarray) -> float:
    if len(audio) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(audio.astype(np.float64, copy=False)))))


def micro_window_issue_labels(metrics: dict[str, float]) -> list[str]:
    labels: list[str] = []
    residual_is_active = (
        metrics["residual_peak_dbfs"] > MICRO_RESIDUAL_SPIKE_THRESHOLD_DBFS
        or metrics["residual_transient_peak_dbfs"] > MICRO_TRANSIENT_MISMATCH_THRESHOLD_DBFS
        or abs(metrics["rms_drift_db"]) > 3.0
    )
    if metrics["residual_peak_dbfs"] > MICRO_RESIDUAL_SPIKE_THRESHOLD_DBFS:
        labels.append("micro_residual_spike")
    if metrics["residual_transient_peak_dbfs"] > MICRO_TRANSIENT_MISMATCH_THRESHOLD_DBFS:
        labels.append("micro_transient_mismatch")
    if (
        metrics["reference_transient_peak"] > EPSILON
        and metrics["candidate_transient_peak"] < 0.65 * metrics["reference_transient_peak"]
    ):
        labels.append("micro_target_transient_dropout")
    if residual_is_active and metrics["max_leakage_strength"] > 0.05:
        labels.append("micro_reference_stem_leakage")
    if metrics["max_residual_leakage_strength"] > 0.05:
        labels.append("micro_residual_reference_stem_leakage")
    if abs(metrics["rms_drift_db"]) > 3.0:
        labels.append("micro_level_or_body_drift")
    return labels


def sort_micro_quality_windows(windows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        windows,
        key=lambda item: (-float(item["severity_score"]), float(item["scale_window_seconds"]), int(item["start_sample"])),
    )


def group_micro_quality_windows_by_source(
    windows: Sequence[dict[str, Any]],
    *,
    source_metric_name: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        source_name = window.get("metrics", {}).get(source_metric_name)
        if not source_name:
            continue
        grouped.setdefault(str(source_name), []).append(window)
    return {
        source_name: sort_micro_quality_windows(source_windows)[:MICRO_QUALITY_MAX_ITEMS_PER_SOURCE]
        for source_name, source_windows in sorted(grouped.items())
    }


def group_micro_quality_windows_by_time_bucket(windows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    bucket_seconds = float(MICRO_QUALITY_TIME_BUCKET_SECONDS)
    for window in windows:
        start_sec = float(window["start_sec"])
        bucket_start = math.floor(start_sec / bucket_seconds) * bucket_seconds
        bucket_end = bucket_start + bucket_seconds
        bucket_label = f"{bucket_start:.3f}-{bucket_end:.3f}"
        grouped.setdefault(bucket_label, []).append(window)
    return {
        bucket_label: sort_micro_quality_windows(bucket_windows)[:MICRO_QUALITY_MAX_ITEMS_PER_TIME_BUCKET]
        for bucket_label, bucket_windows in sorted(
            grouped.items(),
            key=lambda item: float(item[0].split("-", 1)[0]),
        )
    }


def count_micro_quality_issue_labels(windows: Sequence[dict[str, Any]]) -> dict[str, int]:
    label_counts: dict[str, int] = {}
    for window in windows:
        for label in window.get("issue_labels", []):
            label_text = str(label)
            label_counts[label_text] = label_counts.get(label_text, 0) + 1
    return dict(sorted(label_counts.items()))


def harmonic_residual_profile(audio: np.ndarray) -> dict[str, float]:
    if audio.size < 8:
        return {
            "harmonic_flatness": 1.0,
            "harmonic_peak_to_mean_db": 0.0,
        }
    windowed = audio.astype(np.float64, copy=False) * np.hanning(audio.size)
    magnitude = np.abs(np.fft.rfft(windowed))[1:]
    if magnitude.size == 0:
        return {
            "harmonic_flatness": 1.0,
            "harmonic_peak_to_mean_db": 0.0,
        }
    magnitude = magnitude + EPSILON
    geometric_mean = float(np.exp(np.mean(np.log(magnitude))))
    arithmetic_mean = float(np.mean(magnitude))
    peak_to_mean_db = 20.0 * math.log10(float(np.max(magnitude)) / (arithmetic_mean + EPSILON))
    return {
        "harmonic_flatness": round(geometric_mean / (arithmetic_mean + EPSILON), 6),
        "harmonic_peak_to_mean_db": round(peak_to_mean_db, 6),
    }


def build_phrase_level_quality_windows(
    *,
    stem_name: str,
    reference_stems: dict[str, np.ndarray],
    reference_aligned: np.ndarray,
    candidate_aligned: np.ndarray,
    reference_start: int,
    sample_rate: int,
) -> tuple[int, list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    analyzed_window_count = 0
    aligned_frames = len(reference_aligned)
    for window_seconds in PHRASE_QUALITY_WINDOW_SECONDS:
        window_frames = max(1, int(round(window_seconds * sample_rate)))
        hop_frames = max(1, int(round(window_frames * PHRASE_QUALITY_HOP_RATIO)))
        if aligned_frames <= window_frames:
            starts = [0]
        else:
            starts = list(range(0, aligned_frames - window_frames + 1, hop_frames))
            final_start = aligned_frames - window_frames
            if starts[-1] != final_start:
                starts.append(final_start)
        for index, start in enumerate(starts):
            end = min(aligned_frames, start + window_frames)
            if end <= start:
                continue
            analyzed_window_count += 1
            reference_window = reference_aligned[start:end]
            candidate_window = candidate_aligned[start:end]
            residual = candidate_window - reference_window
            residual_rms_dbfs = rms_dbfs(residual)
            if residual_rms_dbfs < PHRASE_RESIDUAL_RMS_THRESHOLD_DBFS:
                continue
            residual_leakage_matrix = leakage_projection_matrix(
                stem_name=stem_name,
                reference_stems=reference_stems,
                candidate_aligned=residual,
                reference_start=reference_start + start,
                sample_count=end - start,
            )
            harmonic_profile = harmonic_residual_profile(residual)
            leakage_source = residual_leakage_matrix["max_leakage_source"]
            leakage_strength = residual_leakage_matrix["max_leakage_strength"]
            is_named_source_leakage = bool(leakage_source) and leakage_strength > PHRASE_RESIDUAL_LEAKAGE_STRENGTH_THRESHOLD
            is_unattributed_harmonic_residual = (
                not is_named_source_leakage
                and harmonic_profile["harmonic_flatness"] <= PHRASE_HARMONIC_FLATNESS_THRESHOLD
                and harmonic_profile["harmonic_peak_to_mean_db"] >= PHRASE_HARMONIC_PEAK_TO_MEAN_THRESHOLD_DB
            )
            if not is_named_source_leakage and not is_unattributed_harmonic_residual:
                continue
            phrase_source = str(leakage_source or "unattributed_harmonic_residual")
            labels = ["bar_spanning_drop_melody_leakage"]
            severity_score = (
                max(0.0, residual_rms_dbfs - PHRASE_RESIDUAL_RMS_THRESHOLD_DBFS)
                + 35.0 * float(leakage_strength)
                + max(0.0, PHRASE_HARMONIC_FLATNESS_THRESHOLD - harmonic_profile["harmonic_flatness"]) * 20.0
                + max(0.0, harmonic_profile["harmonic_peak_to_mean_db"] - PHRASE_HARMONIC_PEAK_TO_MEAN_THRESHOLD_DB)
            )
            windows.append(
                {
                    "scale_window_seconds": round(window_seconds, 6),
                    "scale_window_frames": int(window_frames),
                    "window_index": index,
                    "start_sample": int(reference_start + start),
                    "end_sample": int(reference_start + end),
                    "start_sec": round((reference_start + start) / float(sample_rate), 6),
                    "end_sec": round((reference_start + end) / float(sample_rate), 6),
                    "severity_score": round(severity_score, 6),
                    "issue_labels": labels,
                    "problem_summary": build_problem_summary(labels),
                    "metrics": {
                        "phrase_residual_rms_dbfs": round(residual_rms_dbfs, 6),
                        "phrase_harmonic_flatness": harmonic_profile["harmonic_flatness"],
                        "phrase_harmonic_peak_to_mean_db": harmonic_profile["harmonic_peak_to_mean_db"],
                        "phrase_leakage_source": phrase_source,
                        "phrase_leakage_strength": leakage_strength,
                        "phrase_leakage_projection_gain": residual_leakage_matrix["max_leakage_projection_gain"],
                    },
                }
            )
    return analyzed_window_count, windows


def build_micro_quality_window_ledger(
    *,
    stem_name: str,
    reference_stems: dict[str, np.ndarray],
    reference_aligned: np.ndarray,
    candidate_aligned: np.ndarray,
    reference_start: int,
    sample_rate: int,
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    analyzed_window_count = 0
    aligned_frames = len(reference_aligned)
    for window_seconds in MICRO_QUALITY_WINDOW_SECONDS:
        window_frames = max(1, int(round(window_seconds * sample_rate)))
        hop_frames = max(1, int(round(window_frames * MICRO_QUALITY_HOP_RATIO)))
        if aligned_frames <= window_frames:
            starts = [0]
        else:
            starts = list(range(0, aligned_frames - window_frames + 1, hop_frames))
            final_start = aligned_frames - window_frames
            if starts[-1] != final_start:
                starts.append(final_start)
        for index, start in enumerate(starts):
            end = min(aligned_frames, start + window_frames)
            if end <= start:
                continue
            analyzed_window_count += 1
            reference_window = reference_aligned[start:end]
            candidate_window = candidate_aligned[start:end]
            residual = candidate_window - reference_window
            leakage_matrix = leakage_projection_matrix(
                stem_name=stem_name,
                reference_stems=reference_stems,
                candidate_aligned=candidate_window,
                reference_start=reference_start + start,
                sample_count=end - start,
            )
            residual_leakage_matrix = leakage_projection_matrix(
                stem_name=stem_name,
                reference_stems=reference_stems,
                candidate_aligned=residual,
                reference_start=reference_start + start,
                sample_count=end - start,
            )
            reference_metrics = basic_metrics(reference_window, sample_rate)
            candidate_metrics = basic_metrics(candidate_window, sample_rate)
            residual_metrics = basic_metrics(residual, sample_rate)
            reference_transient_peak = transient_peak(reference_window)
            candidate_transient_peak = transient_peak(candidate_window)
            residual_transient_peak = transient_peak(residual)
            metrics = {
                "reference_rms_dbfs": reference_metrics["rms_dbfs"],
                "candidate_rms_dbfs": candidate_metrics["rms_dbfs"],
                "residual_rms_dbfs": residual_metrics["rms_dbfs"],
                "residual_peak_dbfs": round(amplitude_dbfs(float(np.max(np.abs(residual))) if residual.size else 0.0), 6),
                "rms_drift_db": round(candidate_metrics["rms_dbfs"] - reference_metrics["rms_dbfs"], 6),
                "cosine_similarity": round(cosine_similarity(reference_window, candidate_window), 6),
                "reference_transient_peak": round(reference_transient_peak, 9),
                "candidate_transient_peak": round(candidate_transient_peak, 9),
                "residual_transient_peak": round(residual_transient_peak, 9),
                "reference_transient_peak_dbfs": round(amplitude_dbfs(reference_transient_peak), 6),
                "candidate_transient_peak_dbfs": round(amplitude_dbfs(candidate_transient_peak), 6),
                "residual_transient_peak_dbfs": round(amplitude_dbfs(residual_transient_peak), 6),
                "max_leakage_projection_gain": leakage_matrix["max_leakage_projection_gain"],
                "max_leakage_strength": leakage_matrix["max_leakage_strength"],
                "max_leakage_source": leakage_matrix["max_leakage_source"],
                "max_residual_leakage_projection_gain": residual_leakage_matrix["max_leakage_projection_gain"],
                "max_residual_leakage_strength": residual_leakage_matrix["max_leakage_strength"],
                "max_residual_leakage_source": residual_leakage_matrix["max_leakage_source"],
            }
            labels = micro_window_issue_labels(metrics)
            residual_is_active = (
                metrics["residual_peak_dbfs"] > MICRO_RESIDUAL_SPIKE_THRESHOLD_DBFS
                or metrics["residual_transient_peak_dbfs"] > MICRO_TRANSIENT_MISMATCH_THRESHOLD_DBFS
                or abs(metrics["rms_drift_db"]) > 3.0
            )
            severity_score = (
                max(0.0, metrics["residual_peak_dbfs"] - MICRO_RESIDUAL_SPIKE_THRESHOLD_DBFS)
                + max(0.0, metrics["residual_transient_peak_dbfs"] - MICRO_TRANSIENT_MISMATCH_THRESHOLD_DBFS)
                + max(0.0, 0.65 * metrics["reference_transient_peak"] - metrics["candidate_transient_peak"]) * 50.0
                + (20.0 * metrics["max_leakage_strength"] if residual_is_active else 0.0)
                + (25.0 * metrics["max_residual_leakage_strength"])
                + max(0.0, abs(metrics["rms_drift_db"]) - 3.0)
            )
            if not labels and severity_score <= 0.0:
                continue
            windows.append(
                {
                    "scale_window_seconds": round(window_seconds, 6),
                    "scale_window_frames": int(window_frames),
                    "window_index": index,
                    "start_sample": int(reference_start + start),
                    "end_sample": int(reference_start + end),
                    "start_sec": round((reference_start + start) / float(sample_rate), 6),
                    "end_sec": round((reference_start + end) / float(sample_rate), 6),
                    "severity_score": round(severity_score, 6),
                    "issue_labels": labels,
                    "problem_summary": build_problem_summary(labels),
                    "metrics": metrics,
                }
            )
    phrase_analyzed_window_count, phrase_windows = build_phrase_level_quality_windows(
        stem_name=stem_name,
        reference_stems=reference_stems,
        reference_aligned=reference_aligned,
        candidate_aligned=candidate_aligned,
        reference_start=reference_start,
        sample_rate=sample_rate,
    )
    windows.extend(phrase_windows)
    sorted_windows = sort_micro_quality_windows(windows)
    sorted_phrase_windows = sort_micro_quality_windows(phrase_windows)
    return {
        "schema_version": 1,
        "window_seconds": [round(value, 6) for value in MICRO_QUALITY_WINDOW_SECONDS],
        "hop_ratio": MICRO_QUALITY_HOP_RATIO,
        "phrase_window_seconds": [round(value, 6) for value in PHRASE_QUALITY_WINDOW_SECONDS],
        "phrase_hop_ratio": PHRASE_QUALITY_HOP_RATIO,
        "analyzed_window_count": analyzed_window_count,
        "phrase_analyzed_window_count": phrase_analyzed_window_count,
        "defect_window_count": len(windows),
        "issue_label_counts": count_micro_quality_issue_labels(windows),
        "worst_windows": sorted_windows[:MICRO_QUALITY_MAX_ITEMS],
        "phrase_level_windows": sorted_phrase_windows[:PHRASE_QUALITY_MAX_ITEMS],
        "phrase_level_windows_by_time_bucket": group_micro_quality_windows_by_time_bucket(phrase_windows),
        "worst_windows_by_leakage_source": group_micro_quality_windows_by_source(
            windows,
            source_metric_name="max_leakage_source",
        ),
        "worst_windows_by_residual_leakage_source": group_micro_quality_windows_by_source(
            windows,
            source_metric_name="max_residual_leakage_source",
        ),
        "worst_phrase_windows_by_source": group_micro_quality_windows_by_source(
            phrase_windows,
            source_metric_name="phrase_leakage_source",
        ),
        "worst_windows_by_time_bucket": group_micro_quality_windows_by_time_bucket(windows),
    }


def evaluate_stem(
    *,
    stem_name: str,
    reference_stems: dict[str, np.ndarray],
    candidate_audio: np.ndarray,
    sample_rate: int,
    max_offset_seconds: float,
    thresholds: ObjectiveThresholds,
) -> dict[str, Any]:
    reference_audio = reference_stems[stem_name]
    if max_offset_seconds > 0.0:
        offset_seconds, envelope_score = estimate_offset_seconds(
            reference_audio,
            candidate_audio,
            sample_rate,
            max_offset_seconds=max_offset_seconds,
            hop_seconds=DEFAULT_HOP_SECONDS,
        )
    else:
        offset_seconds = 0.0
        envelope_score = 1.0

    reference_aligned, candidate_aligned, window = align_pair_with_indices(
        reference_audio,
        candidate_audio,
        sample_rate,
        candidate_offset_seconds=offset_seconds,
    )
    interference = sum_interference(
        reference_stems,
        target_stem=stem_name,
        start=window.reference_start,
        sample_count=window.sample_count,
    )
    candidate_gain_to_reference = best_gain(candidate_aligned, reference_aligned)
    gain_matched_candidate = candidate_gain_to_reference * candidate_aligned
    bss_like = bss_eval_like_metrics(
        target_reference=reference_aligned,
        interference_reference=interference,
        estimate=candidate_aligned,
    )

    metrics = {
        "sdr_db": round(signal_to_distortion_db(reference_aligned, candidate_aligned), 6),
        "gain_matched_sdr_db": round(signal_to_distortion_db(reference_aligned, gain_matched_candidate), 6),
        "si_sdr_db": round(scale_invariant_signal_to_distortion_db(reference_aligned, candidate_aligned), 6),
        "sir_db": bss_like["bss_eval_like_sir_db"],
        "sar_db": bss_like["bss_eval_like_sar_db"],
        "bss_eval_like_sdr_db": bss_like["bss_eval_like_sdr_db"],
        "spectral_convergence": round(spectral_convergence(reference_aligned, candidate_aligned), 9),
        "cosine_similarity": round(cosine_similarity(reference_aligned, candidate_aligned), 6),
        "candidate_gain_to_reference": round(candidate_gain_to_reference, 9),
        "rms_drift_db": round(
            basic_metrics(candidate_aligned, sample_rate)["rms_dbfs"] - basic_metrics(reference_aligned, sample_rate)["rms_dbfs"],
            6,
        ),
        "target_projection_gain": bss_like["target_projection_gain"],
        "interference_projection_gain": bss_like["interference_projection_gain"],
    }
    leakage_matrix = leakage_projection_matrix(
        stem_name=stem_name,
        reference_stems=reference_stems,
        candidate_aligned=candidate_aligned,
        reference_start=window.reference_start,
        sample_count=window.sample_count,
    )
    reference_likeness = build_reference_likeness(
        reference_audio=reference_aligned,
        candidate_audio=candidate_aligned,
        sample_rate=sample_rate,
        leakage_matrix=leakage_matrix,
        thresholds=thresholds,
    )
    waveform_forensics = compare_waveforms_forensically(
        reference_audio=reference_aligned,
        candidate_audio=candidate_aligned,
        sample_rate=sample_rate,
    )
    quality_windows = build_quality_window_ledger(
        stem_name=stem_name,
        reference_stems=reference_stems,
        reference_aligned=reference_aligned,
        candidate_aligned=candidate_aligned,
        reference_start=window.reference_start,
        sample_rate=sample_rate,
        thresholds=thresholds,
    )
    micro_quality_windows = build_micro_quality_window_ledger(
        stem_name=stem_name,
        reference_stems=reference_stems,
        reference_aligned=reference_aligned,
        candidate_aligned=candidate_aligned,
        reference_start=window.reference_start,
        sample_rate=sample_rate,
    )
    checks = [
        threshold_check("min_sdr_db", metrics["sdr_db"], thresholds.min_sdr_db),
        threshold_check("min_si_sdr_db", metrics["si_sdr_db"], thresholds.min_si_sdr_db),
        threshold_check("min_sir_db", metrics["sir_db"], thresholds.min_sir_db),
        threshold_check("min_sar_db", metrics["sar_db"], thresholds.min_sar_db),
        threshold_check(
            "reference_likeness_below_threshold",
            reference_likeness["score_percent"],
            thresholds.min_reference_likeness_percent,
        ),
    ]
    verdict = build_verdict(checks)
    forensic_delivery_contract = build_stem_forensic_delivery_contract(
        stem_name=stem_name,
        verdict=verdict,
        reference_likeness=reference_likeness,
        waveform_forensics=waveform_forensics,
        quality_windows=quality_windows,
        micro_quality_windows=micro_quality_windows,
    )
    return {
        "stem": stem_name,
        "alignment": {
            "candidate_offset_vs_reference_sec": offset_seconds,
            "energy_envelope_score": envelope_score,
            "aligned_duration_sec": round(window.sample_count / float(sample_rate), 6),
            "reference_start_sample": window.reference_start,
            "candidate_start_sample": window.candidate_start,
        },
        "metrics": metrics,
        "leakage_matrix": leakage_matrix,
        "reference_likeness": reference_likeness,
        "waveform_forensics": waveform_forensics,
        "quality_windows": quality_windows,
        "micro_quality_windows": micro_quality_windows,
        "forensic_delivery_contract": forensic_delivery_contract,
        "thresholds": checks,
        "verdict": verdict,
    }


def evaluate_candidate_manifest(
    *,
    ground_truth_manifest: Path,
    candidate_manifest: Path,
    output_json: Path,
    stem_names: Sequence[str],
    sample_rate: int,
    max_offset_seconds: float,
    thresholds: ObjectiveThresholds,
    forensic_evidence_dir: Path | None = None,
) -> dict[str, Any]:
    ground_truth_bundle = resolve_stem_bundle(ground_truth_manifest)
    candidate_bundle = resolve_stem_bundle(candidate_manifest)
    ground_truth_contract = validate_verified_ground_truth(ground_truth_bundle)
    ground_truth_hash_checks = verify_bundle_hashes(ground_truth_bundle, require_hashes=True)
    candidate_hash_checks = verify_bundle_hashes(candidate_bundle, require_hashes=False)
    validate_objective_stem_selection(ground_truth_bundle, stem_names)
    reference_stem_names = reference_stem_names_for_quality(ground_truth_bundle, stem_names)
    reference_stems = read_bundle_audio(ground_truth_bundle, reference_stem_names, sample_rate)
    validate_reference_lengths(reference_stems)

    candidate_audio = read_bundle_audio(candidate_bundle, stem_names, sample_rate)
    stem_reports = []
    for stem_name in stem_names:
        stem_report = evaluate_stem(
            stem_name=stem_name,
            reference_stems=reference_stems,
            candidate_audio=candidate_audio[stem_name],
            sample_rate=sample_rate,
            max_offset_seconds=max_offset_seconds,
            thresholds=thresholds,
        )
        if forensic_evidence_dir is not None:
            stem_evidence_dir = forensic_evidence_dir / stem_name
            evidence_report = write_waveform_forensic_evidence_pack(
                reference_audio=reference_stems[stem_name],
                candidate_audio=candidate_audio[stem_name],
                sample_rate=sample_rate,
                output_dir=stem_evidence_dir,
                stem_name=stem_name,
                max_offset_samples=int(round(max_offset_seconds * sample_rate)),
                reference_stems={
                    reference_name: reference_audio
                    for reference_name, reference_audio in reference_stems.items()
                    if reference_name != stem_name
                },
            )
            stem_report["waveform_forensic_evidence"] = {
                "status": "available",
                "stem_name": evidence_report["stem_name"],
                "output_dir": str(stem_evidence_dir),
                "evidence_artifacts": evidence_report["evidence_artifacts"],
            }
        stem_reports.append(stem_report)
    objective_failed_stems = [report["stem"] for report in stem_reports if report["verdict"]["status"] == "failed"]
    forensic_delivery_contract = build_report_forensic_delivery_contract(stem_reports)
    forensic_failed_stems = forensic_delivery_contract["rejected_stems"]
    candidate_delivery_contract = candidate_manifest_delivery_contract(candidate_bundle.manifest)
    contract_failed_stems = ["candidate_manifest"] if candidate_delivery_contract["status"] == "rejected" else []
    failed_stems = sorted(set(objective_failed_stems) | set(forensic_failed_stems) | set(contract_failed_stems))
    overall_status = "failed" if failed_stems else "passed"
    report = {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
            "generated_at_utc": utc_timestamp(),
            "intended_runtime": "docker ml-worker",
            "threshold_exit_gate": "manifest_delivery_policy",
        },
        "ground_truth_contract": {
            "mode": "verified_isolated_stems",
            "manifest_path": str(ground_truth_manifest),
            "verification_method": ground_truth_contract.get("verification_method"),
            "source": ground_truth_contract.get("source"),
            "verified": True,
            "integrity_checks": ground_truth_hash_checks,
        },
        "inputs": {
            "candidate_manifest": str(candidate_manifest),
            "candidate_integrity_checks": candidate_hash_checks,
            "stem_names": list(stem_names),
            "reference_stem_names": reference_stem_names,
            "sample_rate": sample_rate,
            "max_offset_seconds": max_offset_seconds,
        },
        "forensic_evidence": {
            "status": "available" if forensic_evidence_dir is not None else "not_requested",
            "output_dir": str(forensic_evidence_dir) if forensic_evidence_dir is not None else None,
            "stems": {
                str(stem_report["stem"]): stem_report.get("waveform_forensic_evidence", {})
                for stem_report in stem_reports
                if "waveform_forensic_evidence" in stem_report
            },
        },
        "stems": stem_reports,
        "overall": {
            "status": overall_status,
            "failed_stems": failed_stems,
            "objective_failed_stems": objective_failed_stems,
            "forensic_failed_stems": forensic_failed_stems,
            "contract_failed_stems": contract_failed_stems,
            "threshold_failures_do_not_change_exit_code": True,
        },
        "forensic_delivery_contract": forensic_delivery_contract,
        "candidate_manifest_delivery_contract": candidate_delivery_contract,
        "delivery": delivery_decision(
            overall_status,
            objective_failed=bool(objective_failed_stems),
            forensic_contract_status=forensic_delivery_contract["status"],
            candidate_manifest_contract_status=candidate_delivery_contract["status"],
            candidate_manifest_contract_reason=candidate_delivery_contract["reason"],
        ),
        "interpretation_limits": [
            "These metrics are true objective metrics only for manifests whose ground_truth_contract.verified=true.",
            "SIR/SAR are BSS Eval-like projections against the sum of the other verified reference stems.",
            "The script records threshold status in JSON and uses delivery.status, not process exit, as the listen-delivery gate.",
            "The forensic delivery contract maps detected waveform/window defects to repair, reroute, debleed, or explicit rejection actions.",
        ],
    }
    write_json(output_json, report)
    return report


def stereo_from_mono(mono: np.ndarray, left_gain: float, right_gain: float) -> np.ndarray:
    return np.stack([left_gain * mono, right_gain * mono], axis=1).astype(np.float32)


def normalize_stem_map(stems: dict[str, np.ndarray], peak: float = 0.86) -> dict[str, np.ndarray]:
    mix = sum(stems.values())
    max_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if max_peak <= 0.0 or max_peak <= peak:
        return stems
    gain = peak / max_peak
    return {name: (audio * gain).astype(np.float32) for name, audio in stems.items()}


def build_synthetic_stems(sample_rate: int, duration_seconds: float) -> dict[str, np.ndarray]:
    frame_count = int(round(sample_rate * duration_seconds))
    if frame_count <= sample_rate:
        raise ValueError("duration_seconds must be longer than 1 second")
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    phrase_envelope = (0.55 + 0.45 * np.sin(2.0 * np.pi * 0.5 * time) ** 2).astype(np.float32)

    vocals = phrase_envelope * (
        0.22 * np.sin(2.0 * np.pi * (220.0 + 3.0 * np.sin(2.0 * np.pi * 4.5 * time)) * time)
        + 0.08 * np.sin(2.0 * np.pi * 440.0 * time)
    )
    bass = (0.28 * np.sin(2.0 * np.pi * 55.0 * time) * (0.65 + 0.35 * np.sign(np.sin(2.0 * np.pi * 2.0 * time)))).astype(
        np.float32
    )
    other = (
        0.12 * np.sin(2.0 * np.pi * 330.0 * time)
        + 0.09 * np.sin(2.0 * np.pi * 495.0 * time)
        + 0.06 * np.sin(2.0 * np.pi * 660.0 * time)
    ).astype(np.float32)

    drums = np.zeros(frame_count, dtype=np.float32)
    impulse_positions = np.arange(0, frame_count, max(1, sample_rate // 4))
    transient = np.hanning(96).astype(np.float32)
    for index, position in enumerate(impulse_positions):
        end = min(frame_count, position + len(transient))
        gain = 0.45 if index % 2 == 0 else 0.26
        drums[position:end] += gain * transient[: end - position]

    return normalize_stem_map(
        {
            "vocals": stereo_from_mono(vocals.astype(np.float32), 1.0, 0.94),
            "drums": stereo_from_mono(drums.astype(np.float32), 0.96, 0.96),
            "bass": stereo_from_mono(bass.astype(np.float32), 0.98, 1.0),
            "other": stereo_from_mono(other.astype(np.float32), 0.85, 1.0),
        }
    )


def build_degraded_candidate(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    noise = np.random.default_rng(1337).normal(0.0, 0.0015, size=stems["vocals"].shape).astype(np.float32)
    return {
        "vocals": (0.9 * stems["vocals"] + 0.08 * stems["other"] + noise).astype(np.float32),
        "drums": (0.82 * stems["drums"] + 0.08 * stems["bass"]).astype(np.float32),
        "bass": (0.88 * stems["bass"] + 0.04 * stems["drums"]).astype(np.float32),
        "other": (0.84 * stems["other"] + 0.1 * stems["vocals"]).astype(np.float32),
    }


def write_stem_bundle(
    *,
    bundle_dir: Path,
    stems: dict[str, np.ndarray],
    sample_rate: int,
    manifest_name: str,
    contract: dict[str, Any],
) -> Path:
    stem_dir = bundle_dir / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    stem_entries: dict[str, dict[str, Any]] = {}
    mix: np.ndarray | None = None

    for stem_name, audio in stems.items():
        stem_path = stem_dir / f"{stem_name}.flac"
        sf.write(str(stem_path), audio, sample_rate, subtype="PCM_24")
        mix = audio.astype(np.float32, copy=True) if mix is None else mix + audio
        stem_entries[stem_name] = {
            "path": str(stem_path),
            "sha256": sha256_file(stem_path),
            "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
            "frames": int(audio.shape[0]),
            "duration_sec": round(audio.shape[0] / float(sample_rate), 6),
        }

    if mix is None:
        raise RuntimeError("No stems were written")
    mix_path = bundle_dir / "mix.flac"
    sf.write(str(mix_path), mix, sample_rate, subtype="PCM_24")

    manifest = {
        "schema_version": 1,
        **contract,
        "sample_rate": sample_rate,
        "mix": {
            "path": str(mix_path),
            "sha256": sha256_file(mix_path),
        },
        "stems": stem_entries,
    }
    manifest_path = bundle_dir / manifest_name
    write_json(manifest_path, manifest)
    return manifest_path


def create_synthetic_fixture(output_dir: Path, sample_rate: int, duration_seconds: float) -> dict[str, Path]:
    stems = build_synthetic_stems(sample_rate=sample_rate, duration_seconds=duration_seconds)
    ground_truth_manifest = write_stem_bundle(
        bundle_dir=output_dir / "ground_truth",
        stems=stems,
        sample_rate=sample_rate,
        manifest_name="ground_truth_manifest.json",
        contract={
            "ground_truth_contract": {
                "verified": True,
                "verification_method": "synthetic_deterministic_signal_generation",
                "source": "ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
                "notes": [
                    "Each stem is generated independently before summing the mix.",
                    "This fixture verifies metric plumbing; it does not represent real commercial music quality.",
                ],
            }
        },
    )
    perfect_manifest = write_stem_bundle(
        bundle_dir=output_dir / "candidates" / "perfect",
        stems=stems,
        sample_rate=sample_rate,
        manifest_name="candidate_manifest.json",
        contract={
            "candidate_contract": {
                "name": "synthetic_perfect",
                "source": "exact_copy_of_verified_synthetic_ground_truth",
            }
        },
    )
    degraded_manifest = write_stem_bundle(
        bundle_dir=output_dir / "candidates" / "degraded",
        stems=build_degraded_candidate(stems),
        sample_rate=sample_rate,
        manifest_name="candidate_manifest.json",
        contract={
            "candidate_contract": {
                "name": "synthetic_degraded",
                "source": "known_gain_drift_bleed_and_noise_injected",
            }
        },
    )
    return {
        "ground_truth": ground_truth_manifest,
        "synthetic_perfect": perfect_manifest,
        "synthetic_degraded": degraded_manifest,
    }


def run_synthetic_smoke(
    *,
    output_dir: Path,
    sample_rate: int,
    duration_seconds: float,
    max_offset_seconds: float,
    thresholds: ObjectiveThresholds,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = create_synthetic_fixture(output_dir, sample_rate, duration_seconds)
    candidate_reports: list[dict[str, Any]] = []
    for candidate_name in ("synthetic_perfect", "synthetic_degraded"):
        candidate_report_path = output_dir / f"{candidate_name}_objective_report.json"
        candidate_report = evaluate_candidate_manifest(
            ground_truth_manifest=manifests["ground_truth"],
            candidate_manifest=manifests[candidate_name],
            output_json=candidate_report_path,
            stem_names=DEFAULT_STEMS,
            sample_rate=sample_rate,
            max_offset_seconds=max_offset_seconds,
            thresholds=thresholds,
        )
        candidate_reports.append(
            {
                "name": candidate_name,
                "manifest_path": str(manifests[candidate_name]),
                "report_path": str(candidate_report_path),
                "status": candidate_report["overall"]["status"],
                "delivery_status": candidate_report["delivery"]["status"],
                "failed_stems": candidate_report["overall"]["failed_stems"],
            }
        )

    summary = {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
            "mode": "synthetic_smoke",
            "generated_at_utc": utc_timestamp(),
            "intended_runtime": "docker ml-worker",
        },
        "ground_truth": {
            "manifest_path": str(manifests["ground_truth"]),
            "mode": "verified_isolated_stems",
            "verification_method": "synthetic_deterministic_signal_generation",
        },
        "candidates": candidate_reports,
        "overall": {
            "status": "passed"
            if any(candidate["status"] == "passed" for candidate in candidate_reports)
            and any(candidate["status"] == "failed" for candidate in candidate_reports)
            else "failed",
            "control_gate": "perfect_passes_degraded_fails",
            "expected_control_behavior": "perfect_passes_degraded_fails",
        },
        "interpretation_limits": [
            "This is a verified synthetic fixture for objective-metric plumbing, not a real-song model benchmark.",
            "Use --mode evaluate with a real verified isolated-stem manifest to benchmark model output.",
        ],
    }
    write_json(output_dir / "ground_truth_quality_summary.json", summary)
    return summary


def parse_candidate_manifest_arg(value: str) -> Path:
    if "=" in value:
        _, raw_path = value.split("=", 1)
        return Path(raw_path)
    return Path(value)


def normalize_requested_stems(requested_stems: Sequence[str] | None) -> list[str]:
    raw_stems = list(DEFAULT_STEMS) if not requested_stems else list(requested_stems)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_stem in raw_stems:
        stem_name = str(raw_stem).strip()
        if not stem_name or stem_name in seen:
            continue
        normalized.append(stem_name)
        seen.add(stem_name)
    if not normalized:
        raise ValueError("At least one --stem must be provided")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Docker-only objective quality gate against verified isolated ground-truth stems."
    )
    parser.add_argument("--mode", choices=["synthetic-smoke", "evaluate"], default="synthetic-smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ground-truth-manifest", type=Path)
    parser.add_argument("--candidate-manifest", action="append", default=[])
    parser.add_argument("--stem", action="append")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--max-offset-seconds", type=float, default=0.0)
    parser.add_argument("--min-sdr-db", type=float, default=ObjectiveThresholds.min_sdr_db)
    parser.add_argument("--min-si-sdr-db", type=float, default=ObjectiveThresholds.min_si_sdr_db)
    parser.add_argument("--min-sir-db", type=float, default=ObjectiveThresholds.min_sir_db)
    parser.add_argument("--min-sar-db", type=float, default=ObjectiveThresholds.min_sar_db)
    parser.add_argument(
        "--min-reference-likeness-percent",
        type=float,
        default=ObjectiveThresholds.min_reference_likeness_percent,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_ground_truth_quality_gate.py",
        entrypoint_path="/app/scripts/run_ground_truth_quality_gate.py",
    )
    args = parse_args(argv)
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.max_offset_seconds < 0.0:
        raise ValueError("--max-offset-seconds must not be negative")

    thresholds = ObjectiveThresholds(
        min_sdr_db=args.min_sdr_db,
        min_si_sdr_db=args.min_si_sdr_db,
        min_sir_db=args.min_sir_db,
        min_sar_db=args.min_sar_db,
        min_reference_likeness_percent=args.min_reference_likeness_percent,
    )
    stem_names = normalize_requested_stems(args.stem)
    if args.mode == "synthetic-smoke":
        summary = run_synthetic_smoke(
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration_sec,
            max_offset_seconds=args.max_offset_seconds,
            thresholds=thresholds,
        )
    else:
        if args.ground_truth_manifest is None:
            raise ValueError("--ground-truth-manifest is required in evaluate mode")
        if not args.candidate_manifest:
            raise ValueError("--candidate-manifest is required in evaluate mode")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_summaries = []
        for index, candidate_manifest_value in enumerate(args.candidate_manifest, start=1):
            candidate_manifest = parse_candidate_manifest_arg(candidate_manifest_value)
            output_json = args.output_dir / f"candidate_{index}_objective_report.json"
            report = evaluate_candidate_manifest(
                ground_truth_manifest=args.ground_truth_manifest,
                candidate_manifest=candidate_manifest,
                output_json=output_json,
                stem_names=stem_names,
                sample_rate=args.sample_rate,
                max_offset_seconds=args.max_offset_seconds,
                thresholds=thresholds,
            )
            candidate_summaries.append(
                {
                    "candidate_manifest": str(candidate_manifest),
                    "report_path": str(output_json),
                    "status": report["overall"]["status"],
                    "delivery_status": report["delivery"]["status"],
                    "delivery_reason": report["delivery"]["reason"],
                    "failed_stems": report["overall"]["failed_stems"],
                }
            )
        summary = {
            "schema_version": 1,
            "execution": {
                "tool": "ml_worker_improved/scripts/run_ground_truth_quality_gate.py",
                "mode": "evaluate",
                "generated_at_utc": utc_timestamp(),
                "intended_runtime": "docker ml-worker",
            },
            "ground_truth_manifest": str(args.ground_truth_manifest),
            "candidates": candidate_summaries,
            "overall": {
                "status": "failed" if any(candidate["status"] == "failed" for candidate in candidate_summaries) else "passed"
            },
        }
        write_json(args.output_dir / "ground_truth_quality_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
