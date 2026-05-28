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


SCHEMA_VERSION = 1
MANIFEST_NAME = "instrumental_diagnostic_comparison_manifest.json"
SEGMENTS_NAME = "segment_comparison.jsonl"
PROGRESS_NAME = "instrumental_diagnostic_comparison_progress.jsonl"

RESIDUAL_VOCAL_LABELS = {
    "residual_vocal_bleed",
    "vocal_bleed",
    "leaky_full_mix_risk",
    "kick_punch_bleed",
    "drop_melody_leakage",
    "low_band_bleed",
    "wide_stereo_non_vocal_leakage",
    "unsupported_anchor_leakage",
}
INSTRUMENTAL_PRESERVATION_LABELS = {
    "instrumental_muffling",
    "thin_or_overcleaned",
    "presence_loss",
    "air_detail_loss",
    "local_vocal_body_dip",
    "vocal_dropout",
    "vocal_tail_loss",
}
NOISE_ARTIFACT_LABELS = {
    "high_noise_or_robotic_artifact",
    "robotic_artifact",
    "watery_artifacts",
}


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def mark(self, stage: str, *, status: str = "progress", **metadata: Any) -> None:
        event = {
            "type": "instrumental_diagnostic_comparison_stage",
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


def segment_metrics(source_audio_path: Path, instrumental_path: Path, window_sec: Sequence[float]) -> dict[str, float]:
    start_sec, end_sec = float(window_sec[0]), float(window_sec[1])
    source_audio, source_sr = _read_window(source_audio_path, start_sec, end_sec)
    candidate_audio, candidate_sr = _read_window(instrumental_path, start_sec, end_sec)
    if source_sr != candidate_sr:
        raise ValueError(f"Sample rate mismatch: source={source_sr}, instrumental={candidate_sr}")

    source_rms = _rms(source_audio)
    candidate_rms = _rms(candidate_audio)
    source_presence = _band_energy(source_audio, source_sr, 1800.0, 6000.0)
    candidate_presence = _band_energy(candidate_audio, candidate_sr, 1800.0, 6000.0)
    source_air = _band_energy(source_audio, source_sr, 6000.0, 12000.0)
    candidate_air = _band_energy(candidate_audio, candidate_sr, 6000.0, 12000.0)
    residual = np.asarray(source_audio[: candidate_audio.size] - candidate_audio[: source_audio.size], dtype=np.float64)
    residual_rms = _rms(residual)
    return {
        "rms": round(candidate_rms, 9),
        "source_rms": round(source_rms, 9),
        "rms_ratio": round(candidate_rms / max(source_rms, 1e-12), 9),
        "rms_ratio_db_vs_source": round(_db_ratio(candidate_rms, source_rms), 6),
        "presence_ratio_db_vs_source": round(_db_ratio(candidate_presence, source_presence), 6),
        "air_ratio_db_vs_source": round(_db_ratio(candidate_air, source_air), 6),
        "source_minus_candidate_rms_ratio": round(residual_rms / max(source_rms, 1e-12), 9),
    }


def _target_ratio_for_labels(labels: set[str]) -> tuple[float, str]:
    if labels & RESIDUAL_VOCAL_LABELS and not labels & INSTRUMENTAL_PRESERVATION_LABELS:
        return 0.34, "residual_vocal_suppression"
    if labels & INSTRUMENTAL_PRESERVATION_LABELS and not labels & RESIDUAL_VOCAL_LABELS:
        return 0.86, "instrumental_preservation"
    if labels & RESIDUAL_VOCAL_LABELS and labels & INSTRUMENTAL_PRESERVATION_LABELS:
        return 0.86, "balanced_no_reference_preservation_first"
    return 0.64, "neutral_preference"


def score_metrics(metrics: dict[str, float], labels: Sequence[str]) -> tuple[float, list[str]]:
    label_set = set(labels)
    target_ratio, mode = _target_ratio_for_labels(label_set)
    ratio = float(metrics.get("rms_ratio", 0.0))
    score = 0.0
    reasons = [f"target_mode={mode}", f"target_rms_ratio={target_ratio:.3f}"]

    ratio_error = abs(ratio - target_ratio)
    score -= ratio_error * 3.0
    reasons.append(f"rms_ratio_error_penalty=-{ratio_error * 3.0:.3f}")

    if label_set & INSTRUMENTAL_PRESERVATION_LABELS:
        if ratio < 0.52:
            penalty = (0.52 - ratio) * 2.4
            score -= penalty
            reasons.append(f"muffling_floor_penalty=-{penalty:.3f}")
        if ratio > 1.02:
            penalty = (ratio - 1.02) * 5.0
            score -= penalty
            reasons.append(f"full_mix_like_penalty=-{penalty:.3f}")
        presence_db = float(metrics.get("presence_ratio_db_vs_source", 0.0))
        if presence_db < -9.0:
            penalty = min(2.5, (-9.0 - presence_db) * 0.08)
            score -= penalty
            reasons.append(f"presence_loss_penalty=-{penalty:.3f}")
        air_db = float(metrics.get("air_ratio_db_vs_source", 0.0))
        if air_db < -12.0:
            penalty = min(2.0, (-12.0 - air_db) * 0.05)
            score -= penalty
            reasons.append(f"air_loss_penalty=-{penalty:.3f}")

    if label_set & RESIDUAL_VOCAL_LABELS:
        ratio_bleed_threshold = 1.02 if label_set & INSTRUMENTAL_PRESERVATION_LABELS else 0.72
        if ratio > ratio_bleed_threshold:
            penalty = (ratio - ratio_bleed_threshold) * 4.2
            score -= penalty
            reasons.append(f"residual_vocal_bleed_proxy_penalty=-{penalty:.3f}")
        if ratio < 0.18:
            penalty = (0.18 - ratio) * 1.8
            score -= penalty
            reasons.append(f"over_suppression_penalty=-{penalty:.3f}")

    if label_set & NOISE_ARTIFACT_LABELS:
        presence_db = float(metrics.get("presence_ratio_db_vs_source", 0.0))
        if presence_db > 1.0:
            penalty = min(2.5, (presence_db - 1.0) * 0.25)
            score -= penalty
            reasons.append(f"artifact_presence_penalty=-{penalty:.3f}")

    return round(score, 6), reasons


def _load_segments(dataset_manifest_path: Path | None) -> list[dict[str, Any]]:
    if dataset_manifest_path is None:
        return [
            {
                "segment_id": "full_track",
                "window_sec": [0.0, 0.0],
                "defect_labels": [],
                "training_targets": [],
            }
        ]
    dataset = load_json(dataset_manifest_path)
    segments_path = _as_path(dataset.get("segments_jsonl"))
    if segments_path is None:
        raise ValueError("Dataset manifest has no segments_jsonl")
    segments = load_jsonl(segments_path)
    if not segments:
        raise ValueError(f"No segment records in {segments_path}")
    return segments


def compare_segments(
    *,
    source_audio_path: Path,
    baseline_instrumental_path: Path,
    candidate_instrumental_path: Path,
    segments: Sequence[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
    progress: ProgressWriter,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for segment in segments:
        window_sec = segment.get("window_sec")
        if (
            not isinstance(window_sec, list)
            or len(window_sec) != 2
            or not all(isinstance(value, (int, float)) for value in window_sec)
        ):
            raise ValueError(f"Invalid segment window: {window_sec!r}")
        labels = [str(label) for label in segment.get("defect_labels", []) if isinstance(label, str)]
        baseline_metrics = segment_metrics(source_audio_path, baseline_instrumental_path, window_sec)
        candidate_metrics = segment_metrics(source_audio_path, candidate_instrumental_path, window_sec)
        baseline_score, baseline_reasons = score_metrics(baseline_metrics, labels)
        candidate_score, candidate_reasons = score_metrics(candidate_metrics, labels)
        delta = round(candidate_score - baseline_score, 6)
        winner = "candidate" if delta > 0.02 else "baseline" if delta < -0.02 else "tie"
        record = {
            "segment_id": segment.get("segment_id"),
            "window_sec": [round(float(window_sec[0]), 6), round(float(window_sec[1]), 6)],
            "defect_labels": labels,
            "training_targets": segment.get("training_targets", []),
            "baseline": {
                "label": baseline_label,
                "path": str(baseline_instrumental_path),
                "score": baseline_score,
                "metrics": baseline_metrics,
                "score_reasons": baseline_reasons,
            },
            "candidate": {
                "label": candidate_label,
                "path": str(candidate_instrumental_path),
                "score": candidate_score,
                "metrics": candidate_metrics,
                "score_reasons": candidate_reasons,
            },
            "score_delta": delta,
            "winner": winner,
            "comparison_contract": {
                "proxy_only": True,
                "requires_reference_or_listening_gate": True,
                "not_delivery_result": True,
            },
        }
        records.append(record)
        progress.mark("compare_segment", segment_id=str(record["segment_id"]), winner=winner, score_delta=delta)
    return records


def _overall_status(records: Sequence[dict[str, Any]], *, improve_threshold: float, regression_threshold: float) -> dict[str, Any]:
    deltas = [float(record["score_delta"]) for record in records]
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    improved = [record for record in records if float(record["score_delta"]) > improve_threshold]
    regressed = [record for record in records if float(record["score_delta"]) < -regression_threshold]
    tied = len(records) - len(improved) - len(regressed)
    if records and mean_delta > improve_threshold and not regressed:
        status = "candidate_proxy_improved"
    elif mean_delta < -regression_threshold or len(regressed) > len(improved):
        status = "candidate_proxy_regressed"
    else:
        status = "mixed_or_inconclusive"
    return {
        "status": status,
        "mean_score_delta": round(mean_delta, 6),
        "improved_segment_count": len(improved),
        "regression_segment_count": len(regressed),
        "tie_segment_count": tied,
        "segment_count": len(records),
    }


def compare_instrumental_candidates(
    *,
    source_audio_path: Path,
    baseline_instrumental_path: Path,
    candidate_instrumental_path: Path,
    output_dir: Path,
    dataset_manifest_path: Path | None,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    selector_manifest_path: Path | None = None,
    improve_threshold: float = 0.02,
    regression_threshold: float = 0.02,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressWriter(output_dir / PROGRESS_NAME)
    progress.mark(
        "start",
        source_audio_path=str(source_audio_path),
        baseline_instrumental_path=str(baseline_instrumental_path),
        candidate_instrumental_path=str(candidate_instrumental_path),
        dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path else None,
        selector_manifest_path=str(selector_manifest_path) if selector_manifest_path else None,
        expected_acceleration_mode="cpu_audio_proxy_comparison",
    )

    segments = _load_segments(dataset_manifest_path)
    if dataset_manifest_path is None:
        source_info = sf.info(str(source_audio_path))
        segments[0]["window_sec"] = [0.0, round(float(source_info.duration), 6)]

    progress.mark("compare_segments", segment_count=len(segments))
    records = compare_segments(
        source_audio_path=source_audio_path,
        baseline_instrumental_path=baseline_instrumental_path,
        candidate_instrumental_path=candidate_instrumental_path,
        segments=segments,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        progress=progress,
    )
    overall = _overall_status(
        records,
        improve_threshold=float(improve_threshold),
        regression_threshold=float(regression_threshold),
    )

    selector_summary: dict[str, Any] | None = None
    if selector_manifest_path is not None and selector_manifest_path.exists():
        selector_manifest = load_json(selector_manifest_path)
        counts: dict[str, int] = {}
        for selection in selector_manifest.get("segment_selection", []):
            if isinstance(selection, dict):
                selected_profile = selection.get("selected_profile")
                if isinstance(selected_profile, str):
                    counts[selected_profile] = counts.get(selected_profile, 0) + 1
        selector_summary = {
            "manifest_path": str(selector_manifest_path),
            "segment_count": selector_manifest.get("segment_count"),
            "selected_profile_counts": counts,
            "presentation_status": (
                selector_manifest.get("delivery_contract", {}).get("presentation_status")
                if isinstance(selector_manifest.get("delivery_contract"), dict)
                else None
            ),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "instrumental_diagnostic_comparison_gate",
        "source_audio_path": str(source_audio_path),
        "source_audio_sha256": sha256_file(source_audio_path),
        "baseline": {
            "label": baseline_label,
            "instrumental_path": str(baseline_instrumental_path),
            "sha256": sha256_file(baseline_instrumental_path),
        },
        "candidate": {
            "label": candidate_label,
            "instrumental_path": str(candidate_instrumental_path),
            "sha256": sha256_file(candidate_instrumental_path),
        },
        "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
        "selector_summary": selector_summary,
        "overall": overall,
        "segments": records,
        "segment_comparison_jsonl": str(output_dir / SEGMENTS_NAME),
        "progress_log_path": str(output_dir / PROGRESS_NAME),
        "delivery_contract": {
            "delivery_allowed": False,
            "delivery_status": "not_a_delivery_artifact",
            "presentation_status": "diagnostic_comparison_not_for_listen_pack",
            "release_status": "diagnostic_proxy_comparison_only",
            "listen_pack_allowed": False,
            "requires_reference_or_human_gate": True,
        },
        "execution_contract": {
            "route_name": "instrumental_diagnostic_comparison_gate",
            "docker_runtime_required": True,
            "expected_acceleration_mode": "cpu_audio_proxy_comparison",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "comparison_consumes_precomputed_audio",
            "per_track_hardcode_allowed": False,
            "fallback_used": False,
            "proxy_only_scoring": True,
        },
    }
    write_jsonl(output_dir / SEGMENTS_NAME, records)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress.mark("finish", status="finish", manifest_path=str(output_dir / MANIFEST_NAME), status_overall=overall["status"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_audio")
    parser.add_argument("baseline_instrumental")
    parser.add_argument("candidate_instrumental")
    parser.add_argument("output_dir")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--selector-manifest")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--improve-threshold", type=float, default=0.02)
    parser.add_argument("--regression-threshold", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = compare_instrumental_candidates(
        source_audio_path=Path(args.source_audio),
        baseline_instrumental_path=Path(args.baseline_instrumental),
        candidate_instrumental_path=Path(args.candidate_instrumental),
        output_dir=Path(args.output_dir),
        dataset_manifest_path=Path(args.dataset_manifest) if args.dataset_manifest else None,
        selector_manifest_path=Path(args.selector_manifest) if args.selector_manifest else None,
        baseline_label=str(args.baseline_label),
        candidate_label=str(args.candidate_label),
        improve_threshold=float(args.improve_threshold),
        regression_threshold=float(args.regression_threshold),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(Path(args.output_dir) / MANIFEST_NAME),
                "status": manifest["overall"]["status"],
                "mean_score_delta": manifest["overall"]["mean_score_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
