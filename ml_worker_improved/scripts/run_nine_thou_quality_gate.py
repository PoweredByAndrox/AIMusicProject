#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
TOOLS_ROOT = SCRIPT_ROOT.parent / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import cli_runtime_contract, require_docker_runtime  # noqa: E402
from build_instrumental_from_stem_manifest import (  # noqa: E402
    DEFAULT_EXCLUDED_STEMS,
    build_instrumental,
    load_json as load_stem_manifest_json,
    resolve_manifest_stems,
)
from compare_instrumental_to_reference_pair import (  # noqa: E402
    basic_metrics,
    best_gain,
    build_verdict,
    check_input_hash,
    compare_candidate_to_reference,
    cosine_similarity,
    decode_mono,
    estimate_offset_seconds,
    sha256_file,
    threshold_result,
)
from probe_golden_reference_pair import analyze_pair, probe_audio  # noqa: E402


DEFAULT_FULL_MIX_PATH = Path("/host_project/testing_stuff_music/Nine Thou (Grant Mohrman Superstars Remix) (320).mp3")
DEFAULT_REFERENCE_MINUS_PATH = Path(
    "/host_project/testing_stuff_music/ytmp3free.cc_styles-of-beyond-nine-thou-need-for-speed-most-wanted-2005-youtubemp3free.org.mp3"
)
DEFAULT_GOLDEN_ROOT = Path("/app/lab_artifacts/golden_audio")
DEFAULT_OUTPUT_DIR = DEFAULT_GOLDEN_ROOT / "quality_gate_repeatable"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_OFFSET_SECONDS = 8.0
DEFAULT_HOP_SECONDS = 0.05
EPSILON = 1e-12


@dataclass(frozen=True)
class GateThresholds:
    min_envelope_score: float = 0.95
    min_abs_cosine: float = 0.95
    max_abs_rms_drift_db: float = 1.0
    max_abs_duration_drift_sec: float = 0.05
    max_abs_high_band_ratio_drift: float = 0.003


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    audio_path: Path
    report_name: str
    stem_manifest_path: Path | None = None
    derived_summary_path: Path | None = None


@dataclass(frozen=True)
class AlignmentWindow:
    reference_start: int
    candidate_start: int
    sample_count: int
    candidate_offset_seconds: float


@dataclass(frozen=True)
class PseudoReferenceSet:
    full_mix: np.ndarray
    reference_minus: np.ndarray
    instrumental_reference: np.ndarray
    vocal_residual_reference: np.ndarray
    sample_rate: int
    minus_offset_vs_full_sec: float
    envelope_score: float
    minus_gain_to_full: float
    minus_cosine_to_full: float


DEFAULT_CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="reference_b_self_control",
        audio_path=DEFAULT_REFERENCE_MINUS_PATH,
        report_name="reference_b_self_gate.json",
    ),
    CandidateSpec(
        name="htdemucs_ft_from_reference_a",
        audio_path=DEFAULT_GOLDEN_ROOT / "nine_thou_htdemucs_ft_candidate" / "instrumental_from_non_vocals.flac",
        report_name="htdemucs_ft_from_reference_a_gate.json",
        stem_manifest_path=DEFAULT_GOLDEN_ROOT / "nine_thou_htdemucs_ft_candidate" / "stem_manifest.json",
        derived_summary_path=DEFAULT_GOLDEN_ROOT / "nine_thou_htdemucs_ft_candidate" / "instrumental_from_non_vocals.json",
    ),
    CandidateSpec(
        name="htdemucs_ft_from_reference_b",
        audio_path=DEFAULT_GOLDEN_ROOT / "nine_thou_ytmp3_htdemucs_ft_candidate" / "instrumental_from_non_vocals.flac",
        report_name="htdemucs_ft_from_reference_b_gate.json",
        stem_manifest_path=DEFAULT_GOLDEN_ROOT / "nine_thou_ytmp3_htdemucs_ft_candidate" / "stem_manifest.json",
        derived_summary_path=DEFAULT_GOLDEN_ROOT
        / "nine_thou_ytmp3_htdemucs_ft_candidate"
        / "instrumental_from_non_vocals.json",
    ),
    CandidateSpec(
        name="instrumental_resurrection_from_full_song",
        audio_path=DEFAULT_GOLDEN_ROOT
        / "nine_thou_instrumental_resurrection_from_full_song"
        / "instrumental_resurrection_minus.flac",
        report_name="instrumental_resurrection_from_full_song_gate.json",
    ),
    CandidateSpec(
        name="instrumental_resurrection_from_demucs_seed_b",
        audio_path=DEFAULT_GOLDEN_ROOT
        / "nine_thou_instrumental_resurrection_from_demucs_seed_b"
        / "instrumental_resurrection_minus.flac",
        report_name="instrumental_resurrection_from_demucs_seed_b_gate.json",
    ),
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def runtime_execution_contract(*, step: str = "", requires: Sequence[str] = ()) -> dict[str, Any]:
    runtime = cli_runtime_contract(allow_native_runtime=False)
    contract = {
        "tool": "ml_worker_improved/scripts/run_nine_thou_quality_gate.py",
        "runtime": str(runtime.get("container_runtime") or ""),
        "runtime_environment": str(runtime.get("runtime_environment") or ""),
        "container_runtime": str(runtime.get("container_runtime") or ""),
        "native_runtime_allowed": bool(runtime.get("native_runtime_allowed")),
        "intended_runtime": "project-approved container route",
    }
    if step:
        contract["step"] = step
    if requires:
        contract["requires"] = list(requires)
    return contract


def finite_db(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10((float(numerator) + EPSILON) / (float(denominator) + EPSILON))


def signal_energy(signal: np.ndarray) -> float:
    return float(np.sum(np.square(signal, dtype=np.float64)))


def signal_to_distortion_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    validate_equal_length(reference, estimate)
    error = reference - estimate
    return finite_db(signal_energy(reference), signal_energy(error))


def scale_invariant_signal_to_distortion_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    validate_equal_length(reference, estimate)
    reference_zero_mean = reference.astype(np.float64, copy=False) - float(np.mean(reference))
    estimate_zero_mean = estimate.astype(np.float64, copy=False) - float(np.mean(estimate))
    projection_gain = float(np.dot(estimate_zero_mean, reference_zero_mean) / (np.dot(reference_zero_mean, reference_zero_mean) + EPSILON))
    target_projection = projection_gain * reference_zero_mean
    noise = estimate_zero_mean - target_projection
    return finite_db(signal_energy(target_projection), signal_energy(noise))


def validate_equal_length(*signals: np.ndarray) -> None:
    lengths = {len(signal) for signal in signals}
    if len(lengths) != 1:
        raise ValueError(f"Signals must have equal length, got lengths={sorted(lengths)}")
    if not lengths or next(iter(lengths)) <= 0:
        raise ValueError("Signals must not be empty")


def bss_eval_like_metrics(
    target_reference: np.ndarray,
    interference_reference: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:
    validate_equal_length(target_reference, interference_reference, estimate)
    reference_matrix = np.stack(
        [
            target_reference.astype(np.float64, copy=False),
            interference_reference.astype(np.float64, copy=False),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(reference_matrix, estimate.astype(np.float64, copy=False), rcond=None)
    target_projection = coefficients[0] * reference_matrix[:, 0]
    interference_projection = coefficients[1] * reference_matrix[:, 1]
    artifact_projection = estimate.astype(np.float64, copy=False) - target_projection - interference_projection

    target_energy = signal_energy(target_projection)
    interference_energy = signal_energy(interference_projection)
    artifact_energy = signal_energy(artifact_projection)

    return {
        "bss_eval_like_sdr_db": round(finite_db(target_energy, interference_energy + artifact_energy), 6),
        "bss_eval_like_sir_db": round(finite_db(target_energy, interference_energy), 6),
        "bss_eval_like_sar_db": round(finite_db(target_energy + interference_energy, artifact_energy), 6),
        "target_projection_gain": round(float(coefficients[0]), 9),
        "interference_projection_gain": round(float(coefficients[1]), 9),
    }


def align_pair_with_indices(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    candidate_offset_seconds: float,
) -> tuple[np.ndarray, np.ndarray, AlignmentWindow]:
    offset_samples = int(round(candidate_offset_seconds * sample_rate))
    if offset_samples > 0:
        reference_start = 0
        candidate_start = offset_samples
    elif offset_samples < 0:
        reference_start = -offset_samples
        candidate_start = 0
    else:
        reference_start = 0
        candidate_start = 0

    sample_count = min(len(reference) - reference_start, len(candidate) - candidate_start)
    if sample_count <= 0:
        raise RuntimeError("Audio pair has no overlapping samples after alignment")

    window = AlignmentWindow(
        reference_start=reference_start,
        candidate_start=candidate_start,
        sample_count=sample_count,
        candidate_offset_seconds=round(candidate_offset_seconds, 6),
    )
    return (
        reference[reference_start : reference_start + sample_count],
        candidate[candidate_start : candidate_start + sample_count],
        window,
    )


def build_pseudo_references(
    full_mix_path: Path,
    reference_minus_path: Path,
    sample_rate: int,
    max_offset_seconds: float,
) -> PseudoReferenceSet:
    full_mix = decode_mono(full_mix_path, sample_rate)
    reference_minus = decode_mono(reference_minus_path, sample_rate)
    offset_seconds, envelope_score = estimate_offset_seconds(
        full_mix,
        reference_minus,
        sample_rate,
        max_offset_seconds=max_offset_seconds,
        hop_seconds=DEFAULT_HOP_SECONDS,
    )
    full_aligned, minus_aligned, _ = align_pair_with_indices(
        full_mix,
        reference_minus,
        sample_rate,
        candidate_offset_seconds=offset_seconds,
    )
    minus_gain_to_full = best_gain(minus_aligned, full_aligned)
    minus_cosine_to_full = cosine_similarity(full_aligned, minus_aligned)
    instrumental_reference = minus_gain_to_full * minus_aligned
    vocal_residual_reference = full_aligned - instrumental_reference

    return PseudoReferenceSet(
        full_mix=full_aligned.astype(np.float32, copy=False),
        reference_minus=minus_aligned.astype(np.float32, copy=False),
        instrumental_reference=instrumental_reference.astype(np.float32, copy=False),
        vocal_residual_reference=vocal_residual_reference.astype(np.float32, copy=False),
        sample_rate=sample_rate,
        minus_offset_vs_full_sec=offset_seconds,
        envelope_score=envelope_score,
        minus_gain_to_full=minus_gain_to_full,
        minus_cosine_to_full=minus_cosine_to_full,
    )


def pseudo_reference_warnings(pseudo_references: PseudoReferenceSet) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if pseudo_references.envelope_score < 0.95:
        warnings.append(
            {
                "code": "weak_full_minus_envelope_alignment",
                "severity": "medium",
                "value": round(pseudo_references.envelope_score, 6),
                "message": "Full mix and reference minus do not meet the default envelope-alignment threshold.",
            }
        )
    if pseudo_references.minus_gain_to_full <= 0.0:
        warnings.append(
            {
                "code": "negative_minus_gain_to_full_mix",
                "severity": "high",
                "value": round(pseudo_references.minus_gain_to_full, 9),
                "message": "The reference minus needs negative gain to project onto the full mix; polarity/source-match must be treated as suspect.",
            }
        )
    if abs(pseudo_references.minus_cosine_to_full) < 0.5:
        warnings.append(
            {
                "code": "low_full_minus_cosine",
                "severity": "high",
                "value": round(pseudo_references.minus_cosine_to_full, 6),
                "message": "Full mix and reference minus have low aligned cosine, so pseudo objective metrics are weak evidence.",
            }
        )
    return warnings


def calculate_pseudo_objective_quality(
    candidate_path: Path,
    pseudo_references: PseudoReferenceSet,
    max_offset_seconds: float,
) -> dict[str, Any]:
    candidate = decode_mono(candidate_path, pseudo_references.sample_rate)
    offset_seconds, envelope_score = estimate_offset_seconds(
        pseudo_references.reference_minus,
        candidate,
        pseudo_references.sample_rate,
        max_offset_seconds=max_offset_seconds,
        hop_seconds=DEFAULT_HOP_SECONDS,
    )
    _, candidate_aligned, window = align_pair_with_indices(
        pseudo_references.reference_minus,
        candidate,
        pseudo_references.sample_rate,
        candidate_offset_seconds=offset_seconds,
    )
    reference_slice = slice(window.reference_start, window.reference_start + window.sample_count)
    target_reference = pseudo_references.instrumental_reference[reference_slice]
    interference_reference = pseudo_references.vocal_residual_reference[reference_slice]
    full_mix_slice = pseudo_references.full_mix[reference_slice]

    candidate_gain_to_target = best_gain(candidate_aligned, target_reference)
    gain_matched_candidate = candidate_gain_to_target * candidate_aligned
    residual_after_gain_match = target_reference - gain_matched_candidate

    bss_like = bss_eval_like_metrics(
        target_reference=target_reference,
        interference_reference=interference_reference,
        estimate=candidate_aligned,
    )

    return {
        "schema_version": 1,
        "ground_truth_mode": "pseudo_from_full_mix_and_reference_minus",
        "target": "instrumental_reference_from_gain_matched_minus",
        "interference": "vocal_residual_reference_from_full_mix_minus_gain_matched_minus",
        "alignment": {
            "candidate_offset_vs_pseudo_reference_minus_sec": offset_seconds,
            "energy_envelope_score": envelope_score,
            "aligned_duration_sec": round(window.sample_count / float(pseudo_references.sample_rate), 6),
            "reference_start_sample": window.reference_start,
            "candidate_start_sample": window.candidate_start,
        },
        "reference_build": {
            "sample_rate": pseudo_references.sample_rate,
            "minus_offset_vs_full_mix_sec": pseudo_references.minus_offset_vs_full_sec,
            "minus_vs_full_mix_energy_envelope_score": pseudo_references.envelope_score,
            "minus_gain_to_full_mix": round(pseudo_references.minus_gain_to_full, 9),
            "minus_cosine_to_full_mix": round(pseudo_references.minus_cosine_to_full, 6),
            "warnings": pseudo_reference_warnings(pseudo_references),
            "pseudo_instrumental_rms_dbfs": round(basic_metrics(target_reference, pseudo_references.sample_rate)["rms_dbfs"], 6),
            "pseudo_vocal_residual_rms_dbfs": round(
                basic_metrics(interference_reference, pseudo_references.sample_rate)["rms_dbfs"],
                6,
            ),
        },
        "metrics": {
            "raw_sdr_db": round(signal_to_distortion_db(target_reference, candidate_aligned), 6),
            "gain_matched_sdr_db": round(signal_to_distortion_db(target_reference, gain_matched_candidate), 6),
            "si_sdr_db": round(scale_invariant_signal_to_distortion_db(target_reference, candidate_aligned), 6),
            "full_mix_leakage_cosine": round(cosine_similarity(candidate_aligned, full_mix_slice), 6),
            "vocal_residual_leakage_cosine": round(cosine_similarity(candidate_aligned, interference_reference), 6),
            "gain_candidate_to_pseudo_instrumental": round(candidate_gain_to_target, 9),
            "residual_after_gain_match_rms_dbfs": round(
                basic_metrics(residual_after_gain_match, pseudo_references.sample_rate)["rms_dbfs"],
                6,
            ),
            **bss_like,
        },
        "interpretation_limits": [
            "These are pseudo-objective metrics because the reference-minus file is lossy and not a verified isolated stem.",
            "The instrumental target is derived from full mix/minus alignment and gain matching, then the vocal reference is the residual.",
            "Use these values for repeatable local regression ranking, not for final source-separation benchmark claims.",
        ],
    }


def build_probe_report(
    full_mix_path: Path,
    reference_minus_path: Path,
    sample_rate: int,
    max_offset_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution": {
            **runtime_execution_contract(
                step="probe_golden_reference_pair",
                requires=("ffprobe", "ffmpeg", "numpy"),
            ),
        },
        "inputs": {
            "reference_a": probe_audio(full_mix_path),
            "reference_b": probe_audio(reference_minus_path),
        },
        "pair_analysis": analyze_pair(
            full_mix_path,
            reference_minus_path,
            sample_rate=sample_rate,
            max_offset_seconds=max_offset_seconds,
        ),
        "interpretation_limits": [
            "MP3 inputs are lossy references, so this probe is suitable for smoke/regression baselines, not final objective quality claims.",
            "reference_a is treated as full mix and reference_b as minus for this repeatable Nine Thou gate.",
            "True SDR/SI-SDR still require verified isolated stem ground truth; this gate separately reports pseudo-objective metrics.",
        ],
    }


def maybe_build_derived_instrumental(candidate: CandidateSpec) -> dict[str, Any] | None:
    if candidate.stem_manifest_path is None or candidate.derived_summary_path is None:
        return None
    if not candidate.stem_manifest_path.exists():
        return {
            "status": "skipped",
            "reason": "stem_manifest_missing",
            "stem_manifest": str(candidate.stem_manifest_path),
        }

    manifest = load_stem_manifest_json(candidate.stem_manifest_path)
    selected_stems = resolve_manifest_stems(manifest, set(DEFAULT_EXCLUDED_STEMS))
    instrumental, sample_rate, summary = build_instrumental(selected_stems)

    candidate.audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(candidate.audio_path), instrumental, sample_rate, subtype="PCM_24")

    report = {
        "schema_version": 1,
        "execution": {
            **runtime_execution_contract(step="build_instrumental_from_stem_manifest"),
        },
        "inputs": {
            "stem_manifest": str(candidate.stem_manifest_path),
            "excluded_stems": sorted(DEFAULT_EXCLUDED_STEMS),
        },
        "output": {
            "audio_path": str(candidate.audio_path),
            "subtype": "PCM_24",
        },
        "instrumental": summary,
    }
    write_json(candidate.derived_summary_path, report)
    return {
        "status": "rebuilt",
        "report_path": str(candidate.derived_summary_path),
        "audio_path": str(candidate.audio_path),
    }


def build_threshold_checks(comparison: dict[str, Any], thresholds: GateThresholds) -> list[dict[str, Any]]:
    comparisons = comparison["comparisons"]
    return [
        threshold_result(
            "min_envelope_score",
            comparison["alignment"]["energy_envelope_score"],
            thresholds.min_envelope_score,
            "min",
        ),
        threshold_result(
            "min_abs_cosine",
            comparisons["absolute_cosine_similarity"],
            thresholds.min_abs_cosine,
            "min",
        ),
        threshold_result(
            "max_abs_rms_drift_db",
            abs(comparisons["rms_drift_db"]),
            thresholds.max_abs_rms_drift_db,
            "max",
        ),
        threshold_result(
            "max_abs_duration_drift_sec",
            abs(comparisons["duration_drift_sec"]),
            thresholds.max_abs_duration_drift_sec,
            "max",
        ),
        threshold_result(
            "max_abs_high_band_ratio_drift",
            abs(comparisons["high_band_ratio_drift_4k"]),
            thresholds.max_abs_high_band_ratio_drift,
            "max",
        ),
    ]


def build_candidate_report(
    candidate: CandidateSpec,
    probe_json_path: Path,
    reference_minus_path: Path,
    reference_hash: str | None,
    thresholds: GateThresholds,
    pseudo_references: PseudoReferenceSet,
    sample_rate: int,
    max_offset_seconds: float,
    build_result: dict[str, Any] | None,
) -> dict[str, Any]:
    comparison = compare_candidate_to_reference(
        candidate_path=candidate.audio_path,
        reference_path=reference_minus_path,
        sample_rate=sample_rate,
        max_offset_seconds=max_offset_seconds,
    )
    checks = build_threshold_checks(comparison, thresholds)
    verdict = build_verdict(checks)
    pseudo_quality = calculate_pseudo_objective_quality(
        candidate_path=candidate.audio_path,
        pseudo_references=pseudo_references,
        max_offset_seconds=max_offset_seconds,
    )

    return {
        "schema_version": 1,
        "execution": {
            **runtime_execution_contract(),
            "threshold_exit_gate": "disabled",
            "does_not_call_fail_on_threshold": True,
        },
        "inputs": {
            "probe_json": str(probe_json_path),
            "reference_key": "reference_b",
            "reference": check_input_hash(reference_minus_path, reference_hash),
            "candidate_instrumental": {
                "name": candidate.name,
                "path": str(candidate.audio_path),
                "sha256": sha256_file(candidate.audio_path),
            },
            "derived_instrumental_build": build_result,
        },
        "comparison": comparison,
        "thresholds": checks,
        "verdict": verdict,
        "pseudo_objective_quality": pseudo_quality,
        "interpretation_limits": [
            "Threshold status is recorded in JSON only; this script does not use --fail-on-threshold.",
            "Proxy comparison uses a lossy reference-minus fixture.",
            "pseudo_objective_quality reports SDR/SI-SDR/SIR/SAR-like values against pseudo references derived from the provided full mix and minus.",
            "True objective separation metrics still require verified isolated ground-truth stems.",
        ],
    }


def summarize_candidate(name: str, report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    comparison = report["comparison"]
    comparisons = comparison["comparisons"]
    objective_metrics = report["pseudo_objective_quality"]["metrics"]
    return {
        "name": name,
        "report_path": str(report_path),
        "status": report["verdict"]["status"],
        "failed_checks": [check["name"] for check in report["verdict"]["failed_checks"]],
        "energy_envelope_score": comparison["alignment"]["energy_envelope_score"],
        "absolute_cosine_similarity": comparisons["absolute_cosine_similarity"],
        "rms_drift_db": comparisons["rms_drift_db"],
        "duration_drift_sec": comparisons["duration_drift_sec"],
        "high_band_ratio_drift_4k": comparisons["high_band_ratio_drift_4k"],
        "residual_vs_reference_rms_db": comparisons["residual_vs_reference_rms_db"],
        "pseudo_si_sdr_db": objective_metrics["si_sdr_db"],
        "pseudo_gain_matched_sdr_db": objective_metrics["gain_matched_sdr_db"],
        "pseudo_bss_eval_like_sir_db": objective_metrics["bss_eval_like_sir_db"],
        "pseudo_bss_eval_like_sar_db": objective_metrics["bss_eval_like_sar_db"],
    }


def missing_candidate_summary(candidate: CandidateSpec, report_path: Path, build_result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "report_path": str(report_path),
        "status": "missing",
        "candidate_path": str(candidate.audio_path),
        "derived_instrumental_build": build_result,
    }


def run_quality_gate(
    *,
    full_mix_path: Path,
    reference_minus_path: Path,
    output_dir: Path,
    candidates: Sequence[CandidateSpec],
    thresholds: GateThresholds,
    sample_rate: int,
    max_offset_seconds: float,
) -> dict[str, Any]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_offset_seconds < 0:
        raise ValueError("max_offset_seconds must not be negative")
    if not full_mix_path.exists():
        raise FileNotFoundError(full_mix_path)
    if not reference_minus_path.exists():
        raise FileNotFoundError(reference_minus_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    probe_json_path = output_dir / "reference_pair_probe.json"
    probe_report = build_probe_report(
        full_mix_path=full_mix_path,
        reference_minus_path=reference_minus_path,
        sample_rate=sample_rate,
        max_offset_seconds=max_offset_seconds,
    )
    write_json(probe_json_path, probe_report)
    reference_hash = (probe_report.get("inputs") or {}).get("reference_b", {}).get("sha256")

    pseudo_references = build_pseudo_references(
        full_mix_path=full_mix_path,
        reference_minus_path=reference_minus_path,
        sample_rate=sample_rate,
        max_offset_seconds=max_offset_seconds,
    )
    reference_warnings = pseudo_reference_warnings(pseudo_references)

    candidate_summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        report_path = output_dir / candidate.report_name
        build_result = maybe_build_derived_instrumental(candidate)
        if not candidate.audio_path.exists():
            candidate_summaries.append(missing_candidate_summary(candidate, report_path, build_result))
            continue

        candidate_report = build_candidate_report(
            candidate=candidate,
            probe_json_path=probe_json_path,
            reference_minus_path=reference_minus_path,
            reference_hash=reference_hash,
            thresholds=thresholds,
            pseudo_references=pseudo_references,
            sample_rate=sample_rate,
            max_offset_seconds=max_offset_seconds,
            build_result=build_result,
        )
        write_json(report_path, candidate_report)
        candidate_summaries.append(summarize_candidate(candidate.name, report_path, candidate_report))

    passed_candidates = [candidate["name"] for candidate in candidate_summaries if candidate["status"] == "passed"]
    failed_candidates = [candidate["name"] for candidate in candidate_summaries if candidate["status"] == "failed"]
    missing_candidates = [candidate["name"] for candidate in candidate_summaries if candidate["status"] == "missing"]

    summary = {
        "schema_version": 1,
        "execution": {
            **runtime_execution_contract(),
            "generated_at_utc": utc_timestamp(),
            "gate_type": "lossy_reference_minus_proxy_regression_with_pseudo_objective_metrics",
            "threshold_exit_gate": "disabled",
            "does_not_call_fail_on_threshold": True,
        },
        "reference_contract": {
            "probe_json": str(probe_json_path),
            "reference_key": "reference_b",
            "full_mix_path": str(full_mix_path),
            "reference_minus_path": str(reference_minus_path),
            "thresholds": {
                "min_envelope_score": thresholds.min_envelope_score,
                "min_abs_cosine": thresholds.min_abs_cosine,
                "max_abs_rms_drift_db": thresholds.max_abs_rms_drift_db,
                "max_abs_duration_drift_sec": thresholds.max_abs_duration_drift_sec,
                "max_abs_high_band_ratio_drift": thresholds.max_abs_high_band_ratio_drift,
            },
        },
        "pseudo_ground_truth_contract": {
            "mode": "pseudo_from_full_mix_and_reference_minus",
            "sample_rate": sample_rate,
            "minus_offset_vs_full_mix_sec": pseudo_references.minus_offset_vs_full_sec,
            "minus_gain_to_full_mix": round(pseudo_references.minus_gain_to_full, 9),
            "minus_cosine_to_full_mix": round(pseudo_references.minus_cosine_to_full, 6),
            "warnings": reference_warnings,
            "limits": [
                "The minus file is treated as a lossy instrumental reference, not a verified isolated studio stem.",
                "The vocal/interference reference is computed as full_mix - gain_matched_minus.",
                "SDR/SI-SDR/SIR/SAR-like metrics in this gate are repeatable local regression metrics, not final benchmark scores.",
            ],
        },
        "candidates": candidate_summaries,
        "overall": {
            "status": "failed" if failed_candidates else "incomplete" if missing_candidates else "passed",
            "reference_contract_status": "warning" if reference_warnings else "clean",
            "passed_candidates": passed_candidates,
            "failed_candidates": failed_candidates,
            "missing_candidates": missing_candidates,
            "threshold_failures_do_not_change_exit_code": True,
        },
        "interpretation_limits": [
            "This script is the repeatable project-container gate for the local Nine Thou fixture.",
            "It records threshold verdicts in JSON and intentionally does not use --fail-on-threshold.",
            "It adds pseudo-objective SDR/SI-SDR/SIR/SAR-like metrics from the available full mix/minus pair.",
            "Verified isolated stem ground truth is still required before claiming true source-separation objective quality.",
        ],
    }
    write_json(output_dir / "quality_gate_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the repeatable project-container Nine Thou instrumental quality gate without --fail-on-threshold."
    )
    parser.add_argument("--full-mix", type=Path, default=DEFAULT_FULL_MIX_PATH)
    parser.add_argument("--reference-minus", type=Path, default=DEFAULT_REFERENCE_MINUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max-offset-seconds", type=float, default=DEFAULT_MAX_OFFSET_SECONDS)
    parser.add_argument("--min-envelope-score", type=float, default=GateThresholds.min_envelope_score)
    parser.add_argument("--min-abs-cosine", type=float, default=GateThresholds.min_abs_cosine)
    parser.add_argument("--max-abs-rms-drift-db", type=float, default=GateThresholds.max_abs_rms_drift_db)
    parser.add_argument("--max-abs-duration-drift-sec", type=float, default=GateThresholds.max_abs_duration_drift_sec)
    parser.add_argument(
        "--max-abs-high-band-ratio-drift",
        type=float,
        default=GateThresholds.max_abs_high_band_ratio_drift,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_nine_thou_quality_gate.py",
        entrypoint_path="/app/scripts/run_nine_thou_quality_gate.py",
    )
    args = parse_args(argv)
    thresholds = GateThresholds(
        min_envelope_score=args.min_envelope_score,
        min_abs_cosine=args.min_abs_cosine,
        max_abs_rms_drift_db=args.max_abs_rms_drift_db,
        max_abs_duration_drift_sec=args.max_abs_duration_drift_sec,
        max_abs_high_band_ratio_drift=args.max_abs_high_band_ratio_drift,
    )
    summary = run_quality_gate(
        full_mix_path=args.full_mix,
        reference_minus_path=args.reference_minus,
        output_dir=args.output_dir,
        candidates=DEFAULT_CANDIDATES,
        thresholds=thresholds,
        sample_rate=args.sample_rate,
        max_offset_seconds=args.max_offset_seconds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
