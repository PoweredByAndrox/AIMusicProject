from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy import signal


SCHEMA_VERSION = 1
MANIFEST_NAME = "instrumental_self_judge_manifest.json"
SEGMENTS_NAME = "instrumental_self_judge_segments.jsonl"
PROGRESS_NAME = "instrumental_self_judge_progress.jsonl"
SHORT_MANIFEST_NAME = "manifest.json"
SHORT_SEGMENTS_NAME = "rows.jsonl"
SHORT_PROGRESS_NAME = "progress.jsonl"


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def mark(self, stage: str, *, status: str = "progress", **metadata: Any) -> None:
        event = {
            "type": "instrumental_self_judge_stage",
            "stage": stage,
            "status": status,
            "elapsed_sec": 0.0,
            "since_start_sec": round(time.monotonic() - self.started_at, 6),
            "metadata": metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(loaded)
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_path(value: Any) -> Path | None:
    if isinstance(value, str) and value:
        return Path(value)
    return None


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    return np.mean(audio.astype(np.float64, copy=False), axis=1)


def _read_window(path: Path, start_sec: float, end_sec: float) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    sample_rate = int(info.samplerate)
    start_frame = max(0, int(round(float(start_sec) * sample_rate)))
    end_frame = max(start_frame + 1, int(round(float(end_sec) * sample_rate)))
    frames = max(1, end_frame - start_frame)
    audio, sr = sf.read(str(path), start=start_frame, frames=frames, always_2d=True, dtype="float32")
    return _mono(audio), int(sr)


def _resample_to(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    gcd = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // gcd
    down = int(source_rate) // gcd
    return signal.resample_poly(audio, up, down).astype(np.float64, copy=False)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if audio.size == 0:
        return 0.0
    windowed = audio * np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(np.square(spectrum[mask]))) + 1e-12)


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(20.0 * math.log10(max(numerator, 1e-12) / max(denominator, 1e-12)))


def _correlation_abs(left: np.ndarray, right: np.ndarray) -> float:
    frame_count = min(left.size, right.size)
    if frame_count == 0:
        return 0.0
    left_centered = left[:frame_count] - float(np.mean(left[:frame_count]))
    right_centered = right[:frame_count] - float(np.mean(right[:frame_count]))
    denominator = math.sqrt(
        max(float(np.dot(left_centered, left_centered)) * float(np.dot(right_centered, right_centered)), 1e-24)
    )
    if denominator <= 0.0:
        return 0.0
    return abs(float(np.dot(left_centered, right_centered)) / denominator)


def _projection_db_vs_reference(audio: np.ndarray, reference: np.ndarray) -> float:
    frame_count = min(audio.size, reference.size)
    if frame_count == 0:
        return -240.0
    audio_centered = audio[:frame_count] - float(np.mean(audio[:frame_count]))
    reference_centered = reference[:frame_count] - float(np.mean(reference[:frame_count]))
    reference_energy = float(np.dot(reference_centered, reference_centered))
    if reference_energy <= 1e-18:
        return -240.0
    projection_gain = float(np.dot(audio_centered, reference_centered)) / reference_energy
    projected = reference_centered * projection_gain
    return _db_ratio(_rms(projected), _rms(audio_centered))


def _load_segments(dataset_manifest_path: Path | None, source_audio_path: Path, segment_sec: float) -> list[dict[str, Any]]:
    if dataset_manifest_path is not None:
        dataset = load_json(dataset_manifest_path)
        segments_path = _as_path(dataset.get("segments_jsonl"))
        if segments_path is None:
            raise ValueError("Dataset manifest has no segments_jsonl")
        segments = load_jsonl(segments_path)
        if not segments:
            raise ValueError(f"No segment records in {segments_path}")
        return segments

    if segment_sec <= 0.0:
        raise ValueError("segment_sec must be positive")
    duration = float(sf.info(str(source_audio_path)).duration)
    segments: list[dict[str, Any]] = []
    start_sec = 0.0
    index = 0
    while start_sec < duration - 0.001:
        end_sec = min(duration, start_sec + segment_sec)
        segments.append(
            {
                "segment_id": f"segment_{index:04d}",
                "window_sec": [round(start_sec, 6), round(end_sec, 6)],
                "defect_labels": [],
                "source": "uniform_self_judge",
            }
        )
        start_sec = end_sec
        index += 1
    return segments


def _selector_features(selector_manifest_path: Path | None) -> dict[str, dict[str, float]]:
    if selector_manifest_path is None:
        return {}
    manifest = load_json(selector_manifest_path)
    features_by_segment: dict[str, dict[str, float]] = {}
    for selection in manifest.get("segment_selection", []):
        if not isinstance(selection, dict):
            continue
        segment_id = selection.get("segment_id")
        selected_profile = selection.get("selected_profile")
        if not isinstance(segment_id, str) or not isinstance(selected_profile, str):
            continue
        for candidate in selection.get("candidate_scores", []):
            if not isinstance(candidate, dict) or candidate.get("profile") != selected_profile:
                continue
            features = candidate.get("features")
            if isinstance(features, dict):
                features_by_segment[segment_id] = {
                    key: float(value)
                    for key, value in features.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            break
    return features_by_segment


def segment_metrics(
    *,
    source_audio_path: Path,
    candidate_instrumental_path: Path,
    window_sec: Sequence[float],
    candidate_vocal_path: Path | None,
    diagnostic_non_vocal_support_path: Path | None,
    selector_segment_features: dict[str, float] | None,
) -> dict[str, float]:
    start_sec, end_sec = float(window_sec[0]), float(window_sec[1])
    source_audio, source_sr = _read_window(source_audio_path, start_sec, end_sec)
    candidate_audio, candidate_sr = _read_window(candidate_instrumental_path, start_sec, end_sec)
    candidate_audio = _resample_to(candidate_audio, candidate_sr, source_sr)
    candidate_sr = source_sr

    source_rms = _rms(source_audio)
    candidate_rms = _rms(candidate_audio)
    aligned_frame_count = min(source_audio.size, candidate_audio.size)
    source_candidate_delta_rms = _rms(source_audio[:aligned_frame_count] - candidate_audio[:aligned_frame_count])
    metrics = {
        "rms": round(candidate_rms, 9),
        "source_rms": round(source_rms, 9),
        "rms_ratio": round(candidate_rms / max(source_rms, 1e-12), 9),
        "rms_ratio_db_vs_source": round(_db_ratio(candidate_rms, source_rms), 6),
        "source_candidate_corr_abs": round(_correlation_abs(source_audio, candidate_audio), 9),
        "source_candidate_delta_rms_ratio": round(source_candidate_delta_rms / max(source_rms, 1e-12), 9),
        "presence_ratio_db_vs_source": round(
            _db_ratio(
                _band_energy(candidate_audio, candidate_sr, 1800.0, 6000.0),
                _band_energy(source_audio, source_sr, 1800.0, 6000.0),
            ),
            6,
        ),
        "air_ratio_db_vs_source": round(
            _db_ratio(
                _band_energy(candidate_audio, candidate_sr, 6000.0, 12000.0),
                _band_energy(source_audio, source_sr, 6000.0, 12000.0),
            ),
            6,
        ),
    }

    if candidate_vocal_path is not None:
        vocal_audio, vocal_sr = _read_window(candidate_vocal_path, start_sec, end_sec)
        vocal_audio = _resample_to(vocal_audio, vocal_sr, candidate_sr)
        metrics.update(
            {
                "vocal_stem_corr_abs": round(_correlation_abs(candidate_audio, vocal_audio), 9),
                "vocal_projection_db_vs_instrumental": round(
                    _projection_db_vs_reference(candidate_audio, vocal_audio),
                    6,
                ),
            }
        )
    elif selector_segment_features is not None:
        for key in ("vocal_stem_corr_abs", "vocal_projection_db_vs_instrumental"):
            if key in selector_segment_features:
                metrics[key] = round(float(selector_segment_features[key]), 9)

    if diagnostic_non_vocal_support_path is not None:
        support_audio, support_sr = _read_window(diagnostic_non_vocal_support_path, start_sec, end_sec)
        candidate_for_support = _resample_to(candidate_audio, candidate_sr, support_sr)
        aligned_support_frames = min(support_audio.size, candidate_for_support.size)
        support_rms = _rms(support_audio)
        candidate_support_rms = _rms(candidate_for_support)
        support_delta_rms = _rms(
            support_audio[:aligned_support_frames] - candidate_for_support[:aligned_support_frames]
        )
        metrics.update(
            {
                "non_vocal_support_corr_abs": round(
                    _correlation_abs(support_audio, candidate_for_support),
                    9,
                ),
                "non_vocal_support_delta_rms_ratio": round(
                    support_delta_rms / max(support_rms, 1e-12),
                    9,
                ),
                "rms_ratio_db_vs_non_vocal_support": round(
                    _db_ratio(candidate_support_rms, support_rms),
                    6,
                ),
                "presence_ratio_db_vs_non_vocal_support": round(
                    _db_ratio(
                        _band_energy(candidate_for_support, support_sr, 1800.0, 6000.0),
                        _band_energy(support_audio, support_sr, 1800.0, 6000.0),
                    ),
                    6,
                ),
            }
        )
    return metrics


def _presence_loss_explained_by_non_vocal_support(metrics: dict[str, float]) -> bool:
    required_metrics = {
        "vocal_stem_corr_abs",
        "vocal_projection_db_vs_instrumental",
        "non_vocal_support_corr_abs",
        "non_vocal_support_delta_rms_ratio",
        "rms_ratio_db_vs_non_vocal_support",
        "presence_ratio_db_vs_non_vocal_support",
    }
    if not required_metrics.issubset(metrics):
        return False
    return (
        float(metrics.get("presence_ratio_db_vs_source", 0.0)) < -9.0
        and float(metrics["vocal_stem_corr_abs"]) <= 0.05
        and float(metrics["vocal_projection_db_vs_instrumental"]) <= -24.0
        and float(metrics["non_vocal_support_corr_abs"]) >= 0.90
        and float(metrics["non_vocal_support_delta_rms_ratio"]) <= 0.45
        and abs(float(metrics["rms_ratio_db_vs_non_vocal_support"])) <= 3.0
        and float(metrics["presence_ratio_db_vs_non_vocal_support"]) >= -3.0
    )


def classify_issues(metrics: dict[str, float]) -> tuple[list[str], dict[str, float]]:
    ratio = float(metrics.get("rms_ratio", 0.0))
    presence_db = float(metrics.get("presence_ratio_db_vs_source", 0.0))
    air_db = float(metrics.get("air_ratio_db_vs_source", 0.0))
    vocal_corr = float(metrics.get("vocal_stem_corr_abs", 0.0))
    vocal_projection_db = float(metrics.get("vocal_projection_db_vs_instrumental", -240.0))
    source_candidate_corr = float(metrics.get("source_candidate_corr_abs", 0.0))
    source_candidate_delta_ratio = float(metrics.get("source_candidate_delta_rms_ratio", 1.0))
    full_mix_rms_risk = max(0.0, (ratio - 1.02) / 0.18)
    passthrough_similarity_risk = min(
        max(0.0, (source_candidate_corr - 0.96) / 0.04),
        max(0.0, (0.10 - source_candidate_delta_ratio) / 0.10),
    )
    vocal_corr_risk = max(0.0, (vocal_corr - 0.12) / 0.12)
    vocal_projection_risk = max(0.0, (vocal_projection_db + 22.0) / 12.0)
    has_vocal_evidence_metrics = (
        "vocal_stem_corr_abs" in metrics or "vocal_projection_db_vs_instrumental" in metrics
    )
    vocal_evidence_risk = max(vocal_corr_risk, vocal_projection_risk)
    gated_passthrough_similarity_risk = (
        passthrough_similarity_risk
        if (not has_vocal_evidence_metrics or vocal_evidence_risk > 0.0)
        else 0.0
    )
    full_mix_like_risk = max(full_mix_rms_risk, gated_passthrough_similarity_risk)

    severities = {
        "low_rms_ratio": max(0.0, (0.52 - ratio) / 0.52),
        "presence_loss": max(0.0, (-9.0 - presence_db) / 9.0),
        "air_loss": max(0.0, (-12.0 - air_db) / 12.0),
        "full_mix_like_risk": full_mix_like_risk,
        "vocal_stem_corr": vocal_corr_risk,
        "vocal_projection": vocal_projection_risk,
    }
    presence_loss_explained = _presence_loss_explained_by_non_vocal_support(metrics)
    metrics["presence_loss_explained_by_non_vocal_support"] = 1.0 if presence_loss_explained else 0.0
    if presence_loss_explained:
        severities["presence_loss"] = 0.0
    issues: list[str] = []
    if severities["low_rms_ratio"] > 0.0:
        issues.append("low_rms_muffling_proxy")
    if severities["presence_loss"] > 0.0:
        issues.append("presence_loss_proxy")
    if severities["air_loss"] > 0.0:
        issues.append("air_loss_proxy")
    if severities["full_mix_like_risk"] > 0.0:
        issues.append("full_mix_like_risk_proxy")
    if severities["vocal_stem_corr"] > 0.0:
        issues.append("vocal_stem_corr_proxy")
    if severities["vocal_projection"] > 0.0:
        issues.append("vocal_projection_proxy")
    return issues, {key: round(value, 6) for key, value in severities.items()}


def judge_instrumental(
    *,
    source_audio_path: Path,
    candidate_instrumental_path: Path,
    output_dir: Path,
    candidate_vocal_path: Path | None = None,
    diagnostic_non_vocal_support_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    selector_manifest_path: Path | None = None,
    candidate_label: str = "candidate",
    segment_sec: float = 5.0,
    short_output_names: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = SHORT_MANIFEST_NAME if short_output_names else MANIFEST_NAME
    segments_name = SHORT_SEGMENTS_NAME if short_output_names else SEGMENTS_NAME
    progress_name = SHORT_PROGRESS_NAME if short_output_names else PROGRESS_NAME
    manifest_path = output_dir / manifest_name
    segments_path = output_dir / segments_name
    progress_path = output_dir / progress_name
    progress = ProgressWriter(progress_path)
    progress.mark(
        "start",
        source_audio_path=str(source_audio_path),
        candidate_instrumental_path=str(candidate_instrumental_path),
        candidate_vocal_path=str(candidate_vocal_path) if candidate_vocal_path is not None else None,
        diagnostic_non_vocal_support_path=(
            str(diagnostic_non_vocal_support_path)
            if diagnostic_non_vocal_support_path is not None
            else None
        ),
        dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
        selector_manifest_path=str(selector_manifest_path) if selector_manifest_path is not None else None,
        expected_acceleration_mode="cpu_dsp_self_judge",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="self_judge_consumes_precomputed_audio",
    )

    segments = _load_segments(dataset_manifest_path, source_audio_path, segment_sec)
    selector_features = _selector_features(selector_manifest_path)
    records: list[dict[str, Any]] = []
    progress.mark("judge_segments", segment_count=len(segments))
    for segment in segments:
        window_sec = segment.get("window_sec")
        if (
            not isinstance(window_sec, list)
            or len(window_sec) != 2
            or not all(isinstance(value, (int, float)) for value in window_sec)
        ):
            raise ValueError(f"Invalid segment window: {window_sec!r}")
        segment_id = str(segment.get("segment_id") or f"segment_{len(records):04d}")
        metrics = segment_metrics(
            source_audio_path=source_audio_path,
            candidate_instrumental_path=candidate_instrumental_path,
            window_sec=window_sec,
            candidate_vocal_path=candidate_vocal_path,
            diagnostic_non_vocal_support_path=diagnostic_non_vocal_support_path,
            selector_segment_features=selector_features.get(segment_id),
        )
        issues, severities = classify_issues(metrics)
        blocker = bool(issues)
        severity_score = round(max(severities.values()) if severities else 0.0, 6)
        record = {
            "segment_id": segment_id,
            "window_sec": [round(float(window_sec[0]), 6), round(float(window_sec[1]), 6)],
            "candidate_label": candidate_label,
            "metrics": metrics,
            "issues": issues,
            "issue_severities": severities,
            "blocker": blocker,
            "severity_score": severity_score,
            "self_judge_contract": {
                "proxy_only": True,
                "not_delivery_result": True,
                "requires_reference_or_accepted_policy_gate": True,
            },
        }
        records.append(record)
        progress.mark("judge_segment", segment_id=segment_id, blocker=blocker, severity_score=severity_score)

    blockers = [record for record in records if record["blocker"]]
    issue_counts: dict[str, int] = {}
    for record in blockers:
        for issue in record["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    status = "proxy_pass" if not blockers else "proxy_blocked"
    ranked_blockers = sorted(
        blockers,
        key=lambda record: (-float(record["severity_score"]), str(record["segment_id"])),
    )
    overall = {
        "status": status,
        "blocker_count": len(blockers),
        "segment_count": len(records),
        "blocker_ratio": round(len(blockers) / max(len(records), 1), 6),
        "issue_counts": issue_counts,
        "top_blockers": ranked_blockers[:10],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "instrumental_self_judge_gate",
        "source_audio_path": str(source_audio_path),
        "source_audio_sha256": sha256_file(source_audio_path),
        "candidate": {
            "label": candidate_label,
            "instrumental_path": str(candidate_instrumental_path),
            "instrumental_sha256": sha256_file(candidate_instrumental_path),
            "vocal_path": str(candidate_vocal_path) if candidate_vocal_path is not None else None,
            "vocal_sha256": sha256_file(candidate_vocal_path) if candidate_vocal_path is not None else None,
            "diagnostic_non_vocal_support_path": (
                str(diagnostic_non_vocal_support_path)
                if diagnostic_non_vocal_support_path is not None
                else None
            ),
            "diagnostic_non_vocal_support_sha256": (
                sha256_file(diagnostic_non_vocal_support_path)
                if diagnostic_non_vocal_support_path is not None
                else None
            ),
            "diagnostic_non_vocal_support_is_production_minus": False,
        },
        "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path is not None else None,
        "selector_manifest_path": str(selector_manifest_path) if selector_manifest_path is not None else None,
        "overall": overall,
        "segments": records,
        "manifest_path": str(manifest_path),
        "segments_jsonl": str(segments_path),
        "progress_log_path": str(progress_path),
        "short_output_names": bool(short_output_names),
        "artifact_name_policy": {
            "path_segment_limit": 32,
            "raw_ids_in_paths": False,
            "provenance_in_manifest": True,
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "delivery_status": "not_a_delivery_artifact",
            "presentation_status": "diagnostic_self_judge_not_for_listen_pack",
            "release_status": "diagnostic_self_judge_only",
            "listen_pack_allowed": False,
            "requires_reference_or_accepted_policy_gate": True,
        },
        "execution_contract": {
            "route_name": "instrumental_self_judge_gate",
            "docker_runtime_required": True,
            "expected_acceleration_mode": "cpu_dsp_self_judge",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "self_judge_consumes_precomputed_audio",
            "per_track_hardcode_allowed": False,
            "fallback_used": False,
            "proxy_only_scoring": True,
        },
    }
    write_jsonl(segments_path, records)
    write_json(manifest_path, manifest)
    progress.mark(
        "finish",
        status=status,
        manifest_path=str(manifest_path),
        blocker_count=len(blockers),
        segment_count=len(records),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_audio")
    parser.add_argument("candidate_instrumental")
    parser.add_argument("output_dir")
    parser.add_argument("--candidate-vocal-stem")
    parser.add_argument("--diagnostic-non-vocal-support")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--selector-manifest")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--segment-sec", type=float, default=5.0)
    parser.add_argument("--short-output-names", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = judge_instrumental(
        source_audio_path=Path(args.source_audio),
        candidate_instrumental_path=Path(args.candidate_instrumental),
        output_dir=Path(args.output_dir),
        candidate_vocal_path=Path(args.candidate_vocal_stem) if args.candidate_vocal_stem else None,
        diagnostic_non_vocal_support_path=(
            Path(args.diagnostic_non_vocal_support)
            if args.diagnostic_non_vocal_support
            else None
        ),
        dataset_manifest_path=Path(args.dataset_manifest) if args.dataset_manifest else None,
        selector_manifest_path=Path(args.selector_manifest) if args.selector_manifest else None,
        candidate_label=str(args.candidate_label),
        segment_sec=float(args.segment_sec),
        short_output_names=bool(args.short_output_names),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest["manifest_path"]),
                "status": manifest["overall"]["status"],
                "blocker_count": manifest["overall"]["blocker_count"],
                "segment_count": manifest["overall"]["segment_count"],
                "delivery_allowed": manifest["delivery_contract"]["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
