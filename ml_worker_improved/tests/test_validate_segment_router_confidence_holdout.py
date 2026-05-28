from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools import route_drum_model_zoo_segments as segment_router
from tools import validate_segment_router_confidence_holdout as holdout


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _write_verified_ground_truth_manifest(base_dir: Path, stem_name: str, stem_path: Path, sample_rate: int) -> Path:
    manifest_path = base_dir / "ground_truth_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_rate": sample_rate,
                "ground_truth_contract": {
                    "verified": True,
                    "verification_method": "synthetic_holdout_unit_test",
                    "source": "test_validate_segment_router_confidence_holdout.py",
                },
                "stems": {
                    stem_name: {
                        "path": str(stem_path.relative_to(base_dir)),
                        "sha256": hashlib.sha256(stem_path.read_bytes()).hexdigest(),
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_calibration_profile(path: Path, *, baseline_label: str, switch_margin: float, switch_fraction: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_type": "segment_router_confidence_calibration",
                "stem_name": "drums",
                "recommended_router_config": {
                    "baseline_candidate_label": baseline_label,
                    "baseline_switch_margin": switch_margin,
                    "baseline_max_switch_fraction": switch_fraction,
                    "baseline_guard_confidence_metric": "score_delta_vs_baseline",
                },
                "contract": {
                    "target_reference_assisted": True,
                    "calibration_only": True,
                    "same_track_delivery_allowed": False,
                    "application_uses_target_reference": False,
                    "holdout_validation_required": True,
                    "no_reference_production_claim": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_holdout_validation_marks_profile_validated_when_applied_candidate_passes_gate(tmp_path: Path) -> None:
    sample_rate = 8000
    t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    reference = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    reference_audio = np.stack([reference, reference], axis=1)
    stem_path = tmp_path / "ground_truth" / "stems" / "drums.flac"
    baseline_path = tmp_path / "candidates" / "baseline.flac"
    challenger_path = tmp_path / "candidates" / "challenger.flac"
    _write_audio(stem_path, reference_audio, sample_rate)
    _write_audio(baseline_path, reference_audio, sample_rate)
    _write_audio(challenger_path, reference_audio * 0.5, sample_rate)
    ground_truth_manifest = _write_verified_ground_truth_manifest(tmp_path / "ground_truth", "drums", stem_path, sample_rate)
    profile_path = _write_calibration_profile(
        tmp_path / "router_confidence_profile.json",
        baseline_label="baseline",
        switch_margin=10.0,
        switch_fraction=0.0,
    )

    manifest = holdout.validate_holdout_profile(
        profile_path=profile_path,
        ground_truth_manifest_path=ground_truth_manifest,
        candidates=[
            segment_router.CandidateSpec(label="baseline", stem_path=baseline_path),
            segment_router.CandidateSpec(label="challenger", stem_path=challenger_path),
        ],
        output_dir=tmp_path / "holdout_validation",
        stem_name="drums",
        router_config=segment_router.RouterConfig(segment_seconds=1.0, crossfade_seconds=0.0, min_rms_dbfs=-80.0),
        benchmark_thresholds=holdout.quality_gate.ObjectiveThresholds(),
        sample_rate=sample_rate,
    )

    assert manifest["validation"]["status"] == "validated"
    assert manifest["contract"]["same_track_delivery_allowed"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["applied_route"]["contract"]["baseline_calibration_profile_path"] == str(profile_path)
    assert manifest["benchmark"]["overall"]["status"] == "accepted_candidate_found"


def test_holdout_validation_rejects_profile_when_applied_candidate_fails_gate(tmp_path: Path) -> None:
    sample_rate = 8000
    t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    reference = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    noisy = reference + 0.25 * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
    stem_path = tmp_path / "ground_truth" / "stems" / "drums.flac"
    baseline_path = tmp_path / "candidates" / "baseline.flac"
    challenger_path = tmp_path / "candidates" / "challenger.flac"
    _write_audio(stem_path, np.stack([reference, reference], axis=1), sample_rate)
    _write_audio(baseline_path, np.stack([noisy, noisy], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([noisy * 0.8, noisy * 0.8], axis=1), sample_rate)
    ground_truth_manifest = _write_verified_ground_truth_manifest(tmp_path / "ground_truth", "drums", stem_path, sample_rate)
    profile_path = _write_calibration_profile(
        tmp_path / "router_confidence_profile.json",
        baseline_label="baseline",
        switch_margin=10.0,
        switch_fraction=0.0,
    )

    manifest = holdout.validate_holdout_profile(
        profile_path=profile_path,
        ground_truth_manifest_path=ground_truth_manifest,
        candidates=[
            segment_router.CandidateSpec(label="baseline", stem_path=baseline_path),
            segment_router.CandidateSpec(label="challenger", stem_path=challenger_path),
        ],
        output_dir=tmp_path / "holdout_validation",
        stem_name="drums",
        router_config=segment_router.RouterConfig(segment_seconds=1.0, crossfade_seconds=0.0, min_rms_dbfs=-80.0),
        benchmark_thresholds=holdout.quality_gate.ObjectiveThresholds(),
        sample_rate=sample_rate,
    )

    assert manifest["validation"]["status"] == "rejected_for_generalization"
    assert manifest["benchmark"]["overall"]["status"] != "accepted_candidate_found"
    assert manifest["validation"]["failed_check_names"]


def test_holdout_validation_rejects_profile_for_different_stem(tmp_path: Path) -> None:
    sample_rate = 8000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    reference = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    stem_path = tmp_path / "ground_truth" / "stems" / "drums.flac"
    baseline_path = tmp_path / "candidates" / "baseline.flac"
    _write_audio(stem_path, np.stack([reference, reference], axis=1), sample_rate)
    _write_audio(baseline_path, np.stack([reference, reference], axis=1), sample_rate)
    ground_truth_manifest = _write_verified_ground_truth_manifest(tmp_path / "ground_truth", "drums", stem_path, sample_rate)
    profile_path = _write_calibration_profile(
        tmp_path / "router_confidence_profile.json",
        baseline_label="baseline",
        switch_margin=10.0,
        switch_fraction=0.0,
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["stem_name"] = "vocals"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="profile stem_name does not match"):
        holdout.validate_holdout_profile(
            profile_path=profile_path,
            ground_truth_manifest_path=ground_truth_manifest,
            candidates=[segment_router.CandidateSpec(label="baseline", stem_path=baseline_path)],
            output_dir=tmp_path / "holdout_validation",
            stem_name="drums",
            router_config=segment_router.RouterConfig(segment_seconds=1.0, crossfade_seconds=0.0, min_rms_dbfs=-80.0),
            benchmark_thresholds=holdout.quality_gate.ObjectiveThresholds(),
            sample_rate=sample_rate,
        )
