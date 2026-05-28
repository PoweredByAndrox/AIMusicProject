from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from build_stem_delta_profile_repair import (
    StemDeltaProfileConfig,
    apply_stem_delta_profile,
    audio_file_summary,
    build_stem_delta_profile,
    fit_length,
    read_json,
    read_stereo_audio,
    resolve_stem_path,
    sha256_file,
    write_json,
)


MODEL_ID = "teacher_profile_bank_v1"
APPLICATION_MODEL_ID = "teacher_profile_bank_application_v1"
SELECTOR_MODEL_ID = "teacher_profile_bank_selector_v1"


@dataclass(frozen=True)
class TrainingExample:
    stem_name: str
    ground_truth_manifest: Path
    base_candidate_manifest: Path
    teacher_manifest: Path


def resolve_ground_truth_stem_path(ground_truth_manifest: Path, stem_name: str) -> Path:
    return resolve_stem_path(ground_truth_manifest, stem_name)


def parse_training_example(raw_value: str) -> TrainingExample:
    if "=" not in raw_value:
        raise argparse.ArgumentTypeError(
            "training example must use STEM=GROUND_TRUTH_MANIFEST,BASE_CANDIDATE_MANIFEST,TEACHER_MANIFEST"
        )
    stem_name, raw_paths = raw_value.split("=", 1)
    stem_name = stem_name.strip()
    parts = [Path(value.strip()) for value in raw_paths.split(",") if value.strip()]
    if not stem_name or len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "training example must use STEM=GROUND_TRUTH_MANIFEST,BASE_CANDIDATE_MANIFEST,TEACHER_MANIFEST"
        )
    return TrainingExample(
        stem_name=stem_name,
        ground_truth_manifest=parts[0],
        base_candidate_manifest=parts[1],
        teacher_manifest=parts[2],
    )


def aggregate_band_profiles(stem_name: str, profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        raise ValueError(f"no training profiles supplied for stem: {stem_name}")
    bands_by_name: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        for band in profile.get("bands", []):
            bands_by_name.setdefault(str(band["name"]), []).append(dict(band))

    aggregated_bands = []
    for band_name in sorted(bands_by_name):
        band_examples = bands_by_name[band_name]
        gain_values = np.array([float(item["gain_db"]) for item in band_examples], dtype=np.float32)
        raw_teacher_gain_values = np.array(
            [float(item.get("raw_teacher_gain_db", item["gain_db"])) for item in band_examples],
            dtype=np.float32,
        )
        low_hz = float(band_examples[0]["low_hz"])
        high_hz = float(band_examples[0]["high_hz"])
        aggregated_bands.append(
            {
                "name": band_name,
                "low_hz": low_hz,
                "high_hz": high_hz,
                "gain_db": round(float(np.median(gain_values)), 6),
                "gain_db_std": round(float(np.std(gain_values)), 6),
                "raw_teacher_gain_db_median": round(float(np.median(raw_teacher_gain_values)), 6),
                "training_example_count": len(band_examples),
                "positive_gain_ratio": round(float(np.mean(gain_values > 0.0)), 6),
            }
        )
    return {
        "stem": stem_name,
        "training_example_count": len(profiles),
        "bands": aggregated_bands,
    }


def build_profile_for_example(
    example: TrainingExample,
    config: StemDeltaProfileConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_stem_path = resolve_ground_truth_stem_path(example.ground_truth_manifest, example.stem_name)
    candidate_stem_path = resolve_stem_path(example.base_candidate_manifest, example.stem_name)
    teacher_stem_path = resolve_stem_path(example.teacher_manifest, example.stem_name)

    reference_audio, reference_sample_rate = read_stereo_audio(reference_stem_path)
    candidate_audio, candidate_sample_rate = read_stereo_audio(candidate_stem_path)
    teacher_audio, teacher_sample_rate = read_stereo_audio(teacher_stem_path)
    if len({reference_sample_rate, candidate_sample_rate, teacher_sample_rate}) != 1:
        raise ValueError("reference, candidate, and teacher sample rates must match")
    frames = min(reference_audio.shape[0], candidate_audio.shape[0], teacher_audio.shape[0])
    profile, diagnostics = build_stem_delta_profile(
        candidate_audio=fit_length(candidate_audio, frames),
        teacher_repair_audio=fit_length(teacher_audio, frames),
        reference_audio=fit_length(reference_audio, frames),
        sample_rate=reference_sample_rate,
        stem_name=example.stem_name,
        config=config,
    )
    example_summary = {
        "stem": example.stem_name,
        "ground_truth_manifest": str(example.ground_truth_manifest),
        "base_candidate_manifest": str(example.base_candidate_manifest),
        "teacher_manifest": str(example.teacher_manifest),
        "reference_stem": audio_file_summary(reference_stem_path),
        "base_candidate_stem": audio_file_summary(candidate_stem_path),
        "teacher_stem": audio_file_summary(teacher_stem_path),
        "diagnostics": diagnostics,
    }
    return profile, example_summary


def build_teacher_profile_bank(
    *,
    training_examples: Sequence[TrainingExample],
    output_path: Path,
    config: StemDeltaProfileConfig,
) -> dict[str, Any]:
    if not training_examples:
        raise ValueError("at least one training example is required")

    profiles_by_stem: dict[str, list[dict[str, Any]]] = {}
    example_summaries = []
    sample_rates: set[int] = set()
    for example in training_examples:
        profile, example_summary = build_profile_for_example(example, config)
        profiles_by_stem.setdefault(example.stem_name, []).append(profile)
        example_summaries.append(example_summary)
        sample_rates.add(int(profile["sample_rate"]))
    if len(sample_rates) != 1:
        raise ValueError("all training examples must share a sample rate")

    stem_profiles = {
        stem_name: aggregate_band_profiles(stem_name, profiles)
        for stem_name, profiles in sorted(profiles_by_stem.items())
    }
    profile_bank = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "sample_rate": sample_rates.pop(),
        "contract": {
            "profile_source_reference_assisted": True,
            "application_reference_free": True,
            "target_reference_used_at_application": False,
            "requires_holdout_validation": True,
            "holdout_validation_status": "missing",
            "delivery_allowed_without_reference": False,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "stores_timestamp_rules": False,
            "quality_gate_required": True,
        },
        "config": {
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "max_abs_gain_db": config.max_abs_gain_db,
            "profile_mix": config.profile_mix,
            "min_error_improvement_db": config.min_error_improvement_db,
            "peak_ceiling": config.peak_ceiling,
            "use_percussive_gate": config.use_percussive_gate,
            "allow_negative_gains": config.allow_negative_gains,
        },
        "stems": stem_profiles,
        "training_examples": example_summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, profile_bank)
    return profile_bank


def profile_for_application(profile_bank: dict[str, Any], stem_name: str) -> dict[str, Any]:
    stems = profile_bank.get("stems", {})
    if stem_name not in stems:
        raise ValueError(f"profile bank has no profile for stem: {stem_name}")
    stem_profile = stems[stem_name]
    return {
        "schema_version": 1,
        "profile_id": f"{MODEL_ID}:{stem_name}",
        "stem": stem_name,
        "sample_rate": profile_bank.get("sample_rate"),
        "contract": profile_bank.get("contract", {}),
        "bands": list(stem_profile.get("bands", [])),
    }


def build_application_manifest(
    *,
    profile_bank_path: Path,
    candidate_manifest: Path,
    output_stem_path: Path,
    profile_application_path: Path,
    stem_name: str,
    profile_bank: dict[str, Any],
    config: StemDeltaProfileConfig,
) -> dict[str, Any]:
    profile_contract = profile_bank.get("contract", {})
    delivery_allowed = bool(
        profile_contract.get("delivery_allowed_without_reference")
        and profile_contract.get("holdout_validation_status") == "passed"
    )
    return {
        "schema_version": 1,
        "model_id": APPLICATION_MODEL_ID,
        "model_name": APPLICATION_MODEL_ID,
        "model": {"model_id": APPLICATION_MODEL_ID, "family": "teacher_profile_bank"},
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": False,
            "profile_source_reference_assisted": bool(profile_contract.get("profile_source_reference_assisted")),
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "requires_holdout_validation": True,
            "holdout_validation_status": profile_contract.get("holdout_validation_status", "missing"),
            "delivery_allowed_without_reference": delivery_allowed,
            "not_a_no_reference_production_route_until_holdout_passes": not delivery_allowed,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "stores_timestamp_rules": False,
            "quality_gate_required": True,
        },
        "profile_application": {
            "stem": stem_name,
            "profile_bank_path": str(profile_bank_path),
            "applied_profile_path": str(profile_application_path),
            "uses_reference_during_application": False,
            "rule_scope": "teacher_profile_bank_frequency_band_delta",
        },
        "inputs": {
            "profile_bank_path": str(profile_bank_path),
            "candidate_manifest": str(candidate_manifest),
        },
        "config": {
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "max_abs_gain_db": config.max_abs_gain_db,
            "profile_mix": config.profile_mix,
            "peak_ceiling": config.peak_ceiling,
            "use_percussive_gate": config.use_percussive_gate,
            "allow_negative_gains": config.allow_negative_gains,
        },
        "stems": {
            stem_name: {
                "path": str(output_stem_path),
                "sha256": sha256_file(output_stem_path),
                "role": "teacher_profile_bank_reference_free_application_stem",
                "profile_source_reference_assisted": bool(profile_contract.get("profile_source_reference_assisted")),
                "target_reference_used_at_application": False,
            }
        },
    }


def apply_teacher_profile_bank(
    *,
    profile_bank_path: Path,
    candidate_manifest: Path,
    output_dir: Path,
    stem_name: str,
    config: StemDeltaProfileConfig,
    listen_dir: Path | None = None,
) -> dict[str, Any]:
    require_docker_runtime(
        tool_name="build_teacher_profile_bank.py",
        entrypoint_path="/app/tools/build_teacher_profile_bank.py",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_bank = read_json(profile_bank_path)
    profile = profile_for_application(profile_bank, stem_name)
    candidate_stem_path = resolve_stem_path(candidate_manifest, stem_name)
    candidate_audio, sample_rate = read_stereo_audio(candidate_stem_path)
    if int(profile_bank.get("sample_rate", sample_rate)) != sample_rate:
        raise ValueError("profile bank sample rate does not match candidate sample rate")

    repaired = apply_stem_delta_profile(
        candidate_audio,
        sample_rate,
        profile,
        config,
    )
    output_stem_path = output_dir / f"{stem_name}.flac"
    sf.write(str(output_stem_path), repaired, sample_rate, subtype="PCM_24")
    applied_profile_path = output_dir / "applied_teacher_profile.json"
    write_json(applied_profile_path, profile)
    manifest = build_application_manifest(
        profile_bank_path=profile_bank_path,
        candidate_manifest=candidate_manifest,
        output_stem_path=output_stem_path,
        profile_application_path=applied_profile_path,
        stem_name=stem_name,
        profile_bank=profile_bank,
        config=config,
    )
    write_json(output_dir / "stem_manifest.json", manifest)
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_stem_path, listen_dir / f"00_base_candidate_{stem_name}.flac")
        shutil.copyfile(output_stem_path, listen_dir / f"01_teacher_profile_bank_application_{stem_name}.flac")
        (listen_dir / "README.txt").write_text(
            "00 = base no-reference candidate.\n"
            "01 = teacher-derived profile bank applied without reading target reference during application.\n",
            encoding="utf-8",
        )
    return manifest


def _candidate_profile_name(candidate: dict[str, Any]) -> str:
    name = str(candidate.get("profile_name") or candidate.get("name") or "").strip()
    if not name:
        raise ValueError("candidate entry is missing profile_name")
    return name


def _candidate_score(candidate: dict[str, Any]) -> float:
    if "score" not in candidate:
        raise ValueError(f"candidate entry is missing score: {_candidate_profile_name(candidate)}")
    return float(candidate["score"])


def build_teacher_profile_selector_manifest(
    *,
    adaptive_manifest_path: Path,
    output_manifest_path: Path,
    min_score_delta: float,
) -> dict[str, Any]:
    require_docker_runtime(
        tool_name="build_teacher_profile_bank.py",
        entrypoint_path="/app/tools/build_teacher_profile_bank.py",
    )
    adaptive_manifest = read_json(adaptive_manifest_path)
    candidates = list(adaptive_manifest.get("candidates", []))
    candidates_by_name = {_candidate_profile_name(candidate): candidate for candidate in candidates}

    pair_decisions = []
    for calibrated_name, calibrated_candidate in sorted(candidates_by_name.items()):
        if not calibrated_name.startswith("cal_"):
            continue
        suffix = calibrated_name.removeprefix("cal_")
        original_name = f"orig_{suffix}"
        original_candidate = candidates_by_name.get(original_name)
        if original_candidate is None:
            pair_decisions.append(
                {
                    "calibrated_profile": calibrated_name,
                    "original_profile": original_name,
                    "status": "rejected_missing_original_pair",
                    "reason": "teacher_calibrated_candidate_has_no_orig_pair",
                }
            )
            continue
        original_score = _candidate_score(original_candidate)
        calibrated_score = _candidate_score(calibrated_candidate)
        score_delta = calibrated_score - original_score
        decision = (
            "keep_calibrated_for_next_stage"
            if score_delta >= min_score_delta
            else "reject_calibrated_regression"
        )
        pair_decisions.append(
            {
                "calibrated_profile": calibrated_name,
                "original_profile": original_name,
                "original_score": round(original_score, 6),
                "calibrated_score": round(calibrated_score, 6),
                "score_delta": round(score_delta, 6),
                "min_score_delta": round(float(min_score_delta), 6),
                "status": decision,
                "reason": (
                    "calibrated_candidate_did_not_regress_proxy_score"
                    if decision == "keep_calibrated_for_next_stage"
                    else "calibrated_candidate_regressed_proxy_score"
                ),
            }
        )

    selected_profile = str(adaptive_manifest.get("selected_profile") or "")
    selected_profile_valid = bool(selected_profile) and selected_profile in candidates_by_name
    selected_pair = next(
        (
            pair
            for pair in pair_decisions
            if pair.get("calibrated_profile") == selected_profile
            or pair.get("original_profile") == selected_profile
        ),
        None,
    )
    kept_pairs = [pair for pair in pair_decisions if pair.get("status") == "keep_calibrated_for_next_stage"]
    selected_regressed = bool(
        selected_profile.startswith("cal_")
        and selected_pair is not None
        and selected_pair.get("status") == "reject_calibrated_regression"
    )
    selected_missing_pair = bool(
        selected_profile.startswith("cal_")
        and selected_pair is not None
        and selected_pair.get("status") == "rejected_missing_original_pair"
    )
    if not selected_profile_valid:
        selector_status = "blocked_invalid_selected_profile"
    elif selected_missing_pair:
        selector_status = "blocked_selected_calibrated_missing_original_pair"
    elif selected_regressed:
        selector_status = "blocked_selected_calibrated_regression"
    elif kept_pairs:
        selector_status = "has_non_regressed_calibrated_candidates"
    else:
        selector_status = "no_calibrated_improvement"
    if selector_status == "blocked_invalid_selected_profile":
        next_action = "reject_invalid_selected_profile_and_rerun_adaptive_audit"
    elif selector_status in {
        "blocked_selected_calibrated_missing_original_pair",
        "blocked_selected_calibrated_regression",
    }:
        next_action = "reject_selected_calibrated_candidate_and_register_new_vocal_route"
    else:
        next_action = "audit_kept_calibrated_candidates_before_any_delivery"

    manifest = {
        "schema_version": 1,
        "model_id": SELECTOR_MODEL_ID,
        "model_name": SELECTOR_MODEL_ID,
        "mode": "teacher_profile_bank_selector",
        "status": selector_status,
        "adaptive_manifest_path": str(adaptive_manifest_path),
        "selected_profile": selected_profile,
        "selected_profile_valid": selected_profile_valid,
        "selected_score": adaptive_manifest.get("selected_score"),
        "pair_decisions": pair_decisions,
        "kept_calibrated_profiles": [pair["calibrated_profile"] for pair in kept_pairs],
        "rejected_calibrated_profiles": [
            pair["calibrated_profile"]
            for pair in pair_decisions
            if str(pair.get("status", "")).startswith("reject")
        ],
        "contract": {
            "profile_source_reference_assisted": True,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "requires_holdout_validation": True,
            "delivery_allowed_without_reference": False,
            "delivery_allowed": False,
            "presentation_status": "diagnostic_only",
            "quality_gate_required": True,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "stores_timestamp_rules": False,
            "does_not_modify_audio": True,
        },
        "next_action": next_action,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or apply a teacher-derived stem profile bank without reading target reference during application."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("build-bank")
    train.add_argument("--training-example", action="append", type=parse_training_example, required=True)
    train.add_argument("--output-profile-bank", type=Path, required=True)

    apply = subparsers.add_parser("apply-bank")
    apply.add_argument("--profile-bank", type=Path, required=True)
    apply.add_argument("--candidate-manifest", type=Path, required=True)
    apply.add_argument("--output-dir", type=Path, required=True)
    apply.add_argument("--stem", required=True)
    apply.add_argument("--listen-dir", type=Path)

    select = subparsers.add_parser("select-from-audit")
    select.add_argument("--adaptive-manifest", type=Path, required=True)
    select.add_argument("--output-manifest", type=Path, required=True)
    select.add_argument("--min-score-delta", type=float, default=0.0)

    for subparser in (train, apply):
        subparser.add_argument("--n-fft", type=int, default=StemDeltaProfileConfig.n_fft)
        subparser.add_argument("--hop-length", type=int, default=StemDeltaProfileConfig.hop_length)
        subparser.add_argument("--max-abs-gain-db", type=float, default=StemDeltaProfileConfig.max_abs_gain_db)
        subparser.add_argument("--profile-mix", type=float, default=StemDeltaProfileConfig.profile_mix)
        subparser.add_argument("--min-error-improvement-db", type=float, default=StemDeltaProfileConfig.min_error_improvement_db)
        subparser.add_argument("--peak-ceiling", type=float, default=StemDeltaProfileConfig.peak_ceiling)
        subparser.add_argument("--disable-percussive-gate", action="store_true")
        subparser.add_argument("--allow-negative-gains", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> StemDeltaProfileConfig:
    return StemDeltaProfileConfig(
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        max_abs_gain_db=float(args.max_abs_gain_db),
        profile_mix=float(args.profile_mix),
        min_error_improvement_db=float(args.min_error_improvement_db),
        peak_ceiling=float(args.peak_ceiling),
        use_percussive_gate=not bool(args.disable_percussive_gate),
        allow_negative_gains=bool(args.allow_negative_gains),
    )


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_teacher_profile_bank.py",
        entrypoint_path="/app/tools/build_teacher_profile_bank.py",
    )
    args = build_arg_parser().parse_args(argv)
    if args.command == "build-bank":
        config = config_from_args(args)
        profile_bank = build_teacher_profile_bank(
            training_examples=args.training_example,
            output_path=args.output_profile_bank,
            config=config,
        )
        print(json.dumps({"profile_bank_path": str(args.output_profile_bank), "contract": profile_bank["contract"]}, indent=2))
        return 0
    if args.command == "select-from-audit":
        manifest = build_teacher_profile_selector_manifest(
            adaptive_manifest_path=args.adaptive_manifest,
            output_manifest_path=args.output_manifest,
            min_score_delta=float(args.min_score_delta),
        )
        print(json.dumps({"selector_manifest_path": str(args.output_manifest), "status": manifest["status"]}, indent=2))
        return 0
    config = config_from_args(args)
    manifest = apply_teacher_profile_bank(
        profile_bank_path=args.profile_bank,
        candidate_manifest=args.candidate_manifest,
        output_dir=args.output_dir,
        stem_name=args.stem,
        listen_dir=args.listen_dir,
        config=config,
    )
    print(json.dumps({"manifest_path": str(args.output_dir / "stem_manifest.json"), "contract": manifest["contract"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
