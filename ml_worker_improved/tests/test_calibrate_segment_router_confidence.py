from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import calibrate_segment_router_confidence as calibration
from tools import route_drum_model_zoo_segments as drum_router


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _write_ground_truth_manifest(base_dir: Path, stem_name: str, stem_path: Path, sample_rate: int) -> Path:
    manifest_path = base_dir / "ground_truth_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_rate": sample_rate,
                "ground_truth_contract": {
                    "verified_isolated_stems": True,
                },
                "stems": {
                    stem_name: {
                        "path": str(stem_path.relative_to(base_dir)),
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_route_manifest(
    path: Path,
    *,
    stem_name: str,
    baseline_path: Path,
    challenger_path: Path,
    confidence_scores: list[float],
) -> Path:
    decisions = []
    for index, confidence_score in enumerate(confidence_scores):
        baseline_score = 10.0
        challenger_score = baseline_score + confidence_score
        decisions.append(
            {
                "start_sec": float(index),
                "end_sec": float(index + 1),
                "selected": "challenger",
                "selection_guard": {
                    "enabled": True,
                    "applied": False,
                    "reason": "score_margin_met",
                    "baseline_label": "baseline",
                    "unguarded_selected": "challenger",
                    "confidence_metric": "score_delta_vs_baseline",
                    "no_reference_confidence_score": confidence_score,
                    "score_delta": confidence_score,
                    "required_margin": 0.0,
                },
                "ranked": [
                    {"label": "challenger", "score": challenger_score},
                    {"label": "baseline", "score": baseline_score},
                ],
            }
        )
    path.write_text(
        json.dumps(
            {
                "contract": {
                    "route": "drum_model_zoo_segment_router_v1",
                    "stem_name": stem_name,
                    "baseline_candidate_label": "baseline",
                    "target_reference_used": False,
                    "application_reference_free": True,
                },
                "candidates": [
                    {"label": "baseline", "stem_path": str(baseline_path)},
                    {"label": "challenger", "stem_path": str(challenger_path)},
                ],
                "decisions": decisions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_calibration_profile_learns_safe_switch_margin_from_verified_stem(tmp_path: Path) -> None:
    sample_rate = 8000
    t = np.linspace(0.0, 4.0, sample_rate * 4, endpoint=False, dtype=np.float32)
    reference = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    first_half_noise = np.zeros_like(reference)
    first_half_noise[: sample_rate * 2] = 0.18 * np.sin(2.0 * np.pi * 660.0 * t[: sample_rate * 2])
    second_half_noise = np.zeros_like(reference)
    second_half_noise[sample_rate * 2 :] = 0.18 * np.sin(2.0 * np.pi * 660.0 * t[sample_rate * 2 :])

    gt_stem_path = tmp_path / "stems" / "drums.flac"
    baseline_path = tmp_path / "baseline.flac"
    challenger_path = tmp_path / "challenger.flac"
    route_manifest_path = tmp_path / "route_manifest.json"
    output_dir = tmp_path / "calibration"
    _write_audio(gt_stem_path, np.stack([reference, reference], axis=1), sample_rate)
    _write_audio(baseline_path, np.stack([reference + second_half_noise, reference + second_half_noise], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([reference + first_half_noise, reference + first_half_noise], axis=1), sample_rate)
    ground_truth_manifest_path = _write_ground_truth_manifest(tmp_path, "drums", gt_stem_path, sample_rate)
    _write_route_manifest(
        route_manifest_path,
        stem_name="drums",
        baseline_path=baseline_path,
        challenger_path=challenger_path,
        confidence_scores=[0.5, 1.0, 3.0, 5.0],
    )

    manifest = calibration.calibrate_segment_router_confidence(
        route_manifest_path=route_manifest_path,
        ground_truth_manifest_path=ground_truth_manifest_path,
        output_dir=output_dir,
        stem_name="drums",
        baseline_candidate_label="baseline",
        min_objective_improvement_db=0.1,
        min_precision=1.0,
    )

    profile = json.loads(Path(manifest["profile"]["path"]).read_text(encoding="utf-8"))
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["calibration_only"] is True
    assert manifest["contract"]["same_track_delivery_allowed"] is False
    assert manifest["recommended_router_config"]["baseline_switch_margin"] == 3.0
    assert manifest["recommended_router_config"]["baseline_max_switch_fraction"] == 0.5
    assert profile["recommended_router_config"] == manifest["recommended_router_config"]
    assert [record["objective_switch_is_helpful"] for record in manifest["segment_records"]] == [False, False, True, True]


def test_calibration_fraction_preserves_single_safe_switch_budget(tmp_path: Path) -> None:
    sample_rate = 200
    segment_count = 260
    t = np.linspace(0.0, float(segment_count), sample_rate * segment_count, endpoint=False, dtype=np.float32)
    reference = 0.25 * np.sin(2.0 * np.pi * 20.0 * t).astype(np.float32)
    last_segment_start = sample_rate * (segment_count - 1)
    last_segment_noise = np.zeros_like(reference)
    last_segment_noise[last_segment_start:] = 0.18 * np.sin(2.0 * np.pi * 60.0 * t[last_segment_start:])
    baseline = reference + last_segment_noise
    challenger = reference.copy()

    gt_stem_path = tmp_path / "stems" / "drums.flac"
    baseline_path = tmp_path / "baseline.flac"
    challenger_path = tmp_path / "challenger.flac"
    route_manifest_path = tmp_path / "route_manifest.json"
    output_dir = tmp_path / "calibration"
    _write_audio(gt_stem_path, np.stack([reference, reference], axis=1), sample_rate)
    _write_audio(baseline_path, np.stack([baseline, baseline], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([challenger, challenger], axis=1), sample_rate)
    ground_truth_manifest_path = _write_ground_truth_manifest(tmp_path, "drums", gt_stem_path, sample_rate)
    confidence_scores = [0.0] * segment_count
    confidence_scores[-1] = 9.0
    _write_route_manifest(
        route_manifest_path,
        stem_name="drums",
        baseline_path=baseline_path,
        challenger_path=challenger_path,
        confidence_scores=confidence_scores,
    )

    manifest = calibration.calibrate_segment_router_confidence(
        route_manifest_path=route_manifest_path,
        ground_truth_manifest_path=ground_truth_manifest_path,
        output_dir=output_dir,
        stem_name="drums",
        baseline_candidate_label="baseline",
        min_objective_improvement_db=0.1,
        min_precision=1.0,
    )

    switch_fraction = manifest["recommended_router_config"]["baseline_max_switch_fraction"]
    assert manifest["calibration_summary"]["predicted_switch_count"] == 1
    assert math.floor(segment_count * switch_fraction) == 1


def test_router_cli_loads_baseline_guard_from_calibration_profile(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8000
    t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    baseline = 0.2 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    challenger = 1.03 * baseline
    baseline_path = tmp_path / "baseline.flac"
    challenger_path = tmp_path / "challenger.flac"
    output_dir = tmp_path / "router_out"
    profile_path = tmp_path / "router_confidence_profile.json"
    _write_audio(baseline_path, np.stack([baseline, baseline], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([challenger, challenger], axis=1), sample_rate)
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_type": "segment_router_confidence_calibration",
                "recommended_router_config": {
                    "baseline_candidate_label": "baseline",
                    "baseline_switch_margin": 2.0,
                    "baseline_max_switch_fraction": 0.25,
                },
                "contract": {
                    "target_reference_assisted": True,
                    "calibration_only": True,
                    "same_track_delivery_allowed": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "route_drum_model_zoo_segments.py",
            str(output_dir),
            "--candidate",
            f"baseline={baseline_path}",
            "--candidate",
            f"challenger={challenger_path}",
            "--baseline-calibration-profile",
            str(profile_path),
            "--segment-seconds",
            "1.0",
            "--crossfade-seconds",
            "0.0",
            "--min-rms-dbfs",
            "-80.0",
        ],
    )

    drum_router.main()

    manifest = json.loads((output_dir / "drum_model_zoo_segment_router_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"]["baseline_candidate_label"] == "baseline"
    assert manifest["contract"]["baseline_switch_margin"] == 2.0
    assert manifest["contract"]["baseline_max_switch_fraction"] == 0.25
    assert manifest["contract"]["baseline_calibration_profile_path"] == str(profile_path)
