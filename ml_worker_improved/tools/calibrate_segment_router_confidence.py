from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


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


def _resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _segment_sdr_db(candidate_audio: np.ndarray, reference_audio: np.ndarray) -> float:
    frames = min(candidate_audio.shape[0], reference_audio.shape[0])
    if frames == 0:
        return -120.0
    candidate = candidate_audio[:frames].astype(np.float64, copy=False)
    reference = reference_audio[:frames].astype(np.float64, copy=False)
    reference_energy = float(np.sum(reference * reference) + 1e-12)
    residual = candidate - reference
    residual_energy = float(np.sum(residual * residual) + 1e-12)
    return 10.0 * math.log10(reference_energy / residual_energy)


def _ground_truth_stem_path(ground_truth_manifest_path: Path, stem_name: str) -> Path:
    manifest = json.loads(ground_truth_manifest_path.read_text(encoding="utf-8"))
    stems = manifest.get("stems", {})
    if stem_name not in stems:
        raise ValueError(f"ground-truth stem not found: {stem_name}")
    return _resolve_manifest_path(ground_truth_manifest_path, str(stems[stem_name]["path"]))


def _candidate_paths(route_manifest_path: Path, route_manifest: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for candidate in route_manifest.get("candidates", []):
        label = str(candidate["label"])
        paths[label] = _resolve_manifest_path(route_manifest_path, str(candidate["stem_path"]))
    return paths


def _ranked_score(decision: dict[str, Any], label: str) -> float:
    for ranked in decision.get("ranked", []):
        if str(ranked.get("label")) == label:
            return float(ranked.get("score", 0.0))
    raise ValueError(f"ranked score not found for label: {label}")


def _decision_confidence_score(decision: dict[str, Any], baseline_label: str, challenger_label: str) -> float:
    guard = decision.get("selection_guard", {})
    confidence = guard.get("no_reference_confidence_score")
    if confidence is not None:
        return float(confidence)
    return _ranked_score(decision, challenger_label) - _ranked_score(decision, baseline_label)


def _build_segment_records(
    *,
    route_manifest_path: Path,
    ground_truth_manifest_path: Path,
    stem_name: str,
    baseline_candidate_label: str,
    min_objective_improvement_db: float,
) -> list[dict[str, Any]]:
    route_manifest = json.loads(route_manifest_path.read_text(encoding="utf-8"))
    candidate_paths = _candidate_paths(route_manifest_path, route_manifest)
    if baseline_candidate_label not in candidate_paths:
        raise ValueError(f"baseline candidate is not present in route manifest: {baseline_candidate_label}")

    reference_audio, reference_sample_rate = _read_audio(_ground_truth_stem_path(ground_truth_manifest_path, stem_name))
    loaded_candidates: dict[str, np.ndarray] = {}
    for label, candidate_path in candidate_paths.items():
        candidate_audio, candidate_sample_rate = _read_audio(candidate_path)
        if candidate_sample_rate != reference_sample_rate:
            raise ValueError(f"sample-rate mismatch for candidate {label}: {candidate_sample_rate} != {reference_sample_rate}")
        loaded_candidates[label] = candidate_audio
    frames = min([reference_audio.shape[0], *[audio.shape[0] for audio in loaded_candidates.values()]])
    reference_audio = _fit_length(reference_audio, frames)
    loaded_candidates = {label: _fit_length(audio, frames) for label, audio in loaded_candidates.items()}

    records: list[dict[str, Any]] = []
    for decision in route_manifest.get("decisions", []):
        guard = decision.get("selection_guard", {})
        unguarded_label = str(guard.get("unguarded_selected") or decision.get("ranked", [{}])[0].get("label"))
        if unguarded_label not in loaded_candidates:
            continue
        start_frame = int(round(float(decision["start_sec"]) * reference_sample_rate))
        end_frame = min(frames, int(round(float(decision["end_sec"]) * reference_sample_rate)))
        baseline_audio = loaded_candidates[baseline_candidate_label][start_frame:end_frame]
        challenger_audio = loaded_candidates[unguarded_label][start_frame:end_frame]
        reference_segment = reference_audio[start_frame:end_frame]
        baseline_sdr_db = _segment_sdr_db(baseline_audio, reference_segment)
        challenger_sdr_db = _segment_sdr_db(challenger_audio, reference_segment)
        objective_improvement_db = challenger_sdr_db - baseline_sdr_db
        confidence_score = _decision_confidence_score(decision, baseline_candidate_label, unguarded_label)
        records.append(
            {
                "start_sec": float(decision["start_sec"]),
                "end_sec": float(decision["end_sec"]),
                "baseline_label": baseline_candidate_label,
                "unguarded_label": unguarded_label,
                "selected_label": str(decision.get("selected")),
                "confidence_metric": "score_delta_vs_baseline",
                "no_reference_confidence_score": round(confidence_score, 6),
                "baseline_sdr_db": round(baseline_sdr_db, 6),
                "challenger_sdr_db": round(challenger_sdr_db, 6),
                "objective_improvement_db": round(objective_improvement_db, 6),
                "objective_switch_is_helpful": objective_improvement_db >= min_objective_improvement_db,
            }
        )
    return records


def _calibrate_threshold(
    records: Sequence[dict[str, Any]],
    *,
    min_precision: float,
) -> dict[str, Any]:
    switch_records = [record for record in records if record["unguarded_label"] != record["baseline_label"]]
    helpful_total = sum(1 for record in switch_records if record["objective_switch_is_helpful"])
    thresholds = sorted({float(record["no_reference_confidence_score"]) for record in switch_records})
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        predicted = [record for record in switch_records if float(record["no_reference_confidence_score"]) >= threshold]
        if not predicted:
            continue
        true_positive = sum(1 for record in predicted if record["objective_switch_is_helpful"])
        false_positive = len(predicted) - true_positive
        precision = true_positive / len(predicted)
        recall = true_positive / helpful_total if helpful_total else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = {
            "status": "calibrated",
            "threshold": threshold,
            "predicted_switch_count": len(predicted),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if precision < min_precision:
            continue
        if best is None or (candidate["f1"], candidate["recall"], candidate["precision"], -candidate["threshold"]) > (
            best["f1"],
            best["recall"],
            best["precision"],
            -best["threshold"],
        ):
            best = candidate
    if best is not None:
        return best
    max_confidence = max((float(record["no_reference_confidence_score"]) for record in switch_records), default=0.0)
    return {
        "status": "no_safe_switch_threshold",
        "threshold": round(max_confidence + 1e-6, 6),
        "predicted_switch_count": 0,
        "true_positive": 0,
        "false_positive": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def _switch_fraction_for_count(predicted_switch_count: int, total_segments: int) -> float:
    if predicted_switch_count <= 0 or total_segments <= 0:
        return 0.0
    fraction = predicted_switch_count / total_segments
    if math.floor(total_segments * fraction) < predicted_switch_count:
        return math.nextafter(fraction, math.inf)
    return fraction


def calibrate_segment_router_confidence(
    *,
    route_manifest_path: Path,
    ground_truth_manifest_path: Path,
    output_dir: Path,
    stem_name: str,
    baseline_candidate_label: str,
    min_objective_improvement_db: float = 0.1,
    min_precision: float = 0.8,
) -> dict[str, Any]:
    records = _build_segment_records(
        route_manifest_path=route_manifest_path,
        ground_truth_manifest_path=ground_truth_manifest_path,
        stem_name=stem_name,
        baseline_candidate_label=baseline_candidate_label,
        min_objective_improvement_db=min_objective_improvement_db,
    )
    threshold_report = _calibrate_threshold(records, min_precision=min_precision)
    total_segments = len(records)
    predicted_switch_count = int(threshold_report["predicted_switch_count"])
    baseline_max_switch_fraction = _switch_fraction_for_count(predicted_switch_count, total_segments)
    recommended_router_config = {
        "baseline_candidate_label": baseline_candidate_label,
        "baseline_switch_margin": round(float(threshold_report["threshold"]), 6),
        "baseline_max_switch_fraction": baseline_max_switch_fraction,
        "baseline_guard_confidence_metric": "score_delta_vs_baseline",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "router_confidence_profile.json"
    manifest_path = output_dir / "segment_router_confidence_calibration_manifest.json"
    profile = {
        "schema_version": 1,
        "profile_type": "segment_router_confidence_calibration",
        "stem_name": stem_name,
        "recommended_router_config": recommended_router_config,
        "contract": {
            "target_reference_assisted": True,
            "calibration_only": True,
            "same_track_delivery_allowed": False,
            "application_uses_target_reference": False,
            "holdout_validation_required": True,
            "no_reference_production_claim": False,
        },
    }
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "contract": profile["contract"],
        "inputs": {
            "route_manifest_path": str(route_manifest_path),
            "ground_truth_manifest_path": str(ground_truth_manifest_path),
            "stem_name": stem_name,
            "baseline_candidate_label": baseline_candidate_label,
            "min_objective_improvement_db": min_objective_improvement_db,
            "min_precision": min_precision,
        },
        "recommended_router_config": recommended_router_config,
        "calibration_summary": {
            **threshold_report,
            "segment_count": total_segments,
            "helpful_switch_count": sum(1 for record in records if record["objective_switch_is_helpful"]),
        },
        "segment_records": records,
        "profile": {
            "path": str(profile_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate reference-free segment-router switch confidence.")
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--baseline-candidate-label", required=True)
    parser.add_argument("--min-objective-improvement-db", type=float, default=0.1)
    parser.add_argument("--min-precision", type=float, default=0.8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="calibrate_segment_router_confidence.py",
        entrypoint_path="/app/tools/calibrate_segment_router_confidence.py",
    )
    args = _build_arg_parser().parse_args(argv)
    manifest = calibrate_segment_router_confidence(
        route_manifest_path=args.route_manifest,
        ground_truth_manifest_path=args.ground_truth_manifest,
        output_dir=args.output_dir,
        stem_name=str(args.stem),
        baseline_candidate_label=str(args.baseline_candidate_label),
        min_objective_improvement_db=float(args.min_objective_improvement_db),
        min_precision=float(args.min_precision),
    )
    print(json.dumps({"manifest_path": manifest["manifest_path"], "profile": manifest["profile"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
