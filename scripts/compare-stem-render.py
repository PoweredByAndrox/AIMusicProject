#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


APP_ROOT = Path("/app")


def normalize_stem_key(raw: str) -> str:
    value = (raw or "").strip()
    if value.startswith("stem_"):
        value = value[5:]
    if value.startswith("render_"):
        value = value[7:]
    return value


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    return data.mean(axis=1).astype(np.float32, copy=False)


def load_audio_mono(path: Path):
    data, sample_rate = sf.read(str(path), always_2d=False)
    return to_float_mono(np.asarray(data)), int(sample_rate)


def resample_if_needed(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return signal
    return librosa.resample(signal, orig_sr=source_sr, target_sr=target_sr)


def align_pair(a_signal: np.ndarray, a_sr: int, b_signal: np.ndarray, b_sr: int):
    target_sr = a_sr
    b_resampled = resample_if_needed(b_signal, b_sr, target_sr)
    length = min(len(a_signal), len(b_resampled))
    if length <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), target_sr
    return a_signal[:length], b_resampled[:length], target_sr


def slice_signal_to_windows(signal: np.ndarray, sample_rate: int, windows: list[dict]) -> np.ndarray:
    if not windows:
        return signal
    parts = []
    for window in windows:
        start_sec = max(0.0, float(window["startSec"]))
        end_sec = max(start_sec, float(window["endSec"]))
        start_sample = int(round(start_sec * sample_rate))
        end_sample = int(round(end_sec * sample_rate))
        start_sample = min(max(0, start_sample), len(signal))
        end_sample = min(max(start_sample, end_sample), len(signal))
        if end_sample > start_sample:
            parts.append(signal[start_sample:end_sample])
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def resolve_segment_windows(segments_json_path: Path, segment_tag: str, segment_limit: int | None, use_segment_windows: bool):
    payload = load_json(segments_json_path)
    selected = payload.get("selected") or {}
    windows = []

    if segment_tag in selected and isinstance(selected[segment_tag], list):
        for item in selected[segment_tag]:
            windows.append({
                "segmentId": item.get("segmentId"),
                "startSec": float(item["startSec"]),
                "endSec": float(item["endSec"]),
                "score": float(item.get("score") or 0.0),
            })
    else:
        for item in payload.get("segments") or []:
            kinds = item.get("kinds") or []
            if segment_tag not in kinds:
                continue
            source_window = item.get("window") if use_segment_windows else item
            windows.append({
                "segmentId": item.get("segmentId"),
                "startSec": float(source_window["startSec"]),
                "endSec": float(source_window["endSec"]),
                "score": float(item.get("scores", {}).get("segmentPriority") or source_window.get("score") or 0.0),
            })

    windows.sort(key=lambda entry: (-entry["score"], entry["startSec"]))
    if segment_limit and segment_limit > 0:
        windows = windows[:segment_limit]
    return payload, windows


def rms_dbfs(signal: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(signal))) + 1e-12)
    return 20.0 * math.log10(rms)


def peak_dbfs(signal: np.ndarray) -> float:
    peak = float(np.max(np.abs(signal)) + 1e-12)
    return 20.0 * math.log10(peak)


def cosine_similarity(a_signal: np.ndarray, b_signal: np.ndarray) -> float:
    numerator = float(np.dot(a_signal, b_signal))
    denominator = float(np.linalg.norm(a_signal) * np.linalg.norm(b_signal) + 1e-12)
    return numerator / denominator


def spectral_stats(signal: np.ndarray, sample_rate: int):
    if len(signal) < 2048:
        padded = np.pad(signal, (0, max(0, 2048 - len(signal))))
    else:
        padded = signal

    centroid = librosa.feature.spectral_centroid(y=padded, sr=sample_rate, n_fft=2048, hop_length=512)
    rolloff = librosa.feature.spectral_rolloff(y=padded, sr=sample_rate, n_fft=2048, hop_length=512, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=padded, n_fft=2048, hop_length=512)
    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(len(padded), d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power) + 1e-12)
    high_band_ratio = float(np.sum(power[freqs >= 4000.0]) / total_power)
    air_band_ratio = float(np.sum(power[freqs >= 8000.0]) / total_power)

    return {
        "spectral_centroid_hz_mean": float(np.mean(centroid)),
        "spectral_rolloff_hz_mean": float(np.mean(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "high_band_ratio_4k": high_band_ratio,
        "air_band_ratio_8k": air_band_ratio,
    }


def basic_metrics(signal: np.ndarray, sample_rate: int):
    rms_value = rms_dbfs(signal)
    peak_value = peak_dbfs(signal)
    return {
        "sample_rate": sample_rate,
        "duration_sec": round(len(signal) / float(sample_rate), 6),
        "rms_dbfs": round(rms_value, 6),
        "peak_dbfs": round(peak_value, 6),
        "crest_db": round(peak_value - rms_value, 6),
        **{
            key: round(value, 6)
            for key, value in spectral_stats(signal, sample_rate).items()
        },
    }


def sum_stems(manifest: dict, excluded_stem_key: str):
    sum_signal = None
    sum_sr = None
    included = []
    for stem_key, info in (manifest.get("stems") or {}).items():
        if stem_key == excluded_stem_key:
            continue
        stem_path = Path(info["path"])
        signal, sample_rate = load_audio_mono(stem_path)
        if sum_signal is None:
            sum_signal = signal
            sum_sr = sample_rate
        else:
            if sample_rate != sum_sr:
                signal = resample_if_needed(signal, sample_rate, sum_sr)
            length = min(len(sum_signal), len(signal))
            sum_signal = sum_signal[:length] + signal[:length]
        included.append(stem_key)

    if sum_signal is None:
        return np.zeros(1, dtype=np.float32), 44100, included
    return sum_signal.astype(np.float32, copy=False), int(sum_sr), included


def load_render_manifest(track_id: int, render_id: str):
    manifest_path = APP_ROOT / "lab_artifacts" / "tuning_lab" / f"track_{track_id}" / f"render_{render_id}" / "stem_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Render manifest not found: {manifest_path}")
    return manifest_path, load_json(manifest_path)


def score_bundle(candidate_metrics: dict, base_similarity: float, accompaniment_similarity: float, base_metrics: dict):
    rms_penalty = max(0.0, abs(candidate_metrics["rms_dbfs"] - base_metrics["rms_dbfs"]) - 1.5) * 2.0
    high_band_penalty = abs(candidate_metrics["high_band_ratio_4k"] - base_metrics["high_band_ratio_4k"]) * 100.0
    cleanliness_score = (1.0 - accompaniment_similarity) * 100.0
    preservation_score = (base_similarity * 100.0) - rms_penalty - high_band_penalty
    balanced_score = (preservation_score * 0.6) + (cleanliness_score * 0.4)
    return {
        "cleanliness_score": round(cleanliness_score, 6),
        "preservation_score": round(preservation_score, 6),
        "balanced_score": round(balanced_score, 6),
        "rms_penalty": round(rms_penalty, 6),
        "high_band_penalty": round(high_band_penalty, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare a rendered stem against base stems and an optional baseline render.")
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--render-id", required=True)
    parser.add_argument("--target-stem", required=True)
    parser.add_argument("--baseline-render-id")
    parser.add_argument("--segments-json")
    parser.add_argument("--segment-tag")
    parser.add_argument("--segment-limit", type=int)
    parser.add_argument("--use-segment-windows", action="store_true")
    args = parser.parse_args()

    target_stem = normalize_stem_key(args.target_stem)
    candidate_manifest_path, candidate_manifest = load_render_manifest(args.track_id, args.render_id)
    candidate_render_dir = candidate_manifest_path.parent
    candidate_stem_path = candidate_render_dir / f"{target_stem}.flac"
    if not candidate_stem_path.exists():
        raise FileNotFoundError(f"Candidate stem file not found: {candidate_stem_path}")

    base_manifest_path_raw = (
        candidate_manifest.get("pipeline", {})
        .get("source_pipeline", {})
        .get("base_stem_manifest")
        or candidate_manifest.get("source_stem_manifest")
    )
    if not base_manifest_path_raw:
        raise RuntimeError("Could not resolve base stem manifest from render manifest")

    base_manifest_path = Path(base_manifest_path_raw)
    base_manifest = load_json(base_manifest_path)
    base_stem_info = (base_manifest.get("stems") or {}).get(target_stem)
    if not base_stem_info:
        raise RuntimeError(f"Base manifest has no target stem '{target_stem}'")
    base_stem_path = Path(base_stem_info["path"])

    candidate_signal, candidate_sr = load_audio_mono(candidate_stem_path)
    base_signal, base_sr = load_audio_mono(base_stem_path)
    accompaniment_signal, accompaniment_sr, accompaniment_stems = sum_stems(base_manifest, target_stem)

    segment_context = None
    if args.segments_json and args.segment_tag:
        segments_json_path = Path(args.segments_json)
        if not segments_json_path.exists():
            raise FileNotFoundError(f"Segments JSON not found: {segments_json_path}")
        segment_payload, selected_windows = resolve_segment_windows(
            segments_json_path,
            args.segment_tag,
            args.segment_limit,
            args.use_segment_windows,
        )
        if not selected_windows:
            raise RuntimeError(f"No windows resolved for segment tag '{args.segment_tag}' from {segments_json_path}")
        candidate_signal = slice_signal_to_windows(candidate_signal, candidate_sr, selected_windows)
        base_signal = slice_signal_to_windows(base_signal, base_sr, selected_windows)
        accompaniment_signal = slice_signal_to_windows(accompaniment_signal, accompaniment_sr, selected_windows)
        segment_context = {
            "segmentsJsonPath": str(segments_json_path),
            "segmentTag": args.segment_tag,
            "segmentLimit": args.segment_limit,
            "useSegmentWindows": bool(args.use_segment_windows),
            "windows": selected_windows,
            "selectedCount": len(selected_windows),
            "trackSegmentsGeneratedAt": segment_payload.get("generatedAt"),
        }

    candidate_aligned, base_aligned, common_sr = align_pair(candidate_signal, candidate_sr, base_signal, base_sr)
    candidate_for_bleed, accompaniment_aligned, _ = align_pair(candidate_signal, candidate_sr, accompaniment_signal, accompaniment_sr)

    candidate_metrics = basic_metrics(candidate_signal, candidate_sr)
    base_metrics = basic_metrics(base_signal, base_sr)
    accompaniment_metrics = basic_metrics(accompaniment_signal, accompaniment_sr)

    candidate_vs_base = round(cosine_similarity(candidate_aligned, base_aligned), 6)
    candidate_vs_accompaniment = round(cosine_similarity(candidate_for_bleed, accompaniment_aligned), 6)
    scores = score_bundle(candidate_metrics, candidate_vs_base, candidate_vs_accompaniment, base_metrics)

    report = {
        "trackId": args.track_id,
        "renderId": args.render_id,
        "targetStem": target_stem,
        "candidate": {
            "path": str(candidate_stem_path),
            "metrics": candidate_metrics,
        },
        "baseStem": {
            "path": str(base_stem_path),
            "metrics": base_metrics,
        },
        "baseAccompaniment": {
            "stemKeys": accompaniment_stems,
            "metrics": accompaniment_metrics,
        },
        "comparisons": {
            "candidate_vs_base_cosine": candidate_vs_base,
            "candidate_vs_base_accompaniment_cosine": candidate_vs_accompaniment,
            "scores": scores,
            "score_notes": [
                "Cleanliness score rewards lower similarity to accompaniment.",
                "Preservation score rewards higher similarity to the base target stem, with penalties for loudness drift and high-band drift.",
                "Balanced score = 60% preservation + 40% cleanliness.",
                "Higher candidate_vs_base_cosine is better.",
                "Lower candidate_vs_base_accompaniment_cosine is better.",
                "These scores are only proxies, not ground-truth quality.",
            ],
        },
    }

    if segment_context is not None:
        report["segmentScope"] = segment_context

    if args.baseline_render_id:
        baseline_manifest_path, baseline_manifest = load_render_manifest(args.track_id, args.baseline_render_id)
        baseline_stem_path = baseline_manifest_path.parent / f"{target_stem}.flac"
        if not baseline_stem_path.exists():
            raise FileNotFoundError(f"Baseline stem file not found: {baseline_stem_path}")

        baseline_signal, baseline_sr = load_audio_mono(baseline_stem_path)
        if segment_context is not None:
            baseline_signal = slice_signal_to_windows(baseline_signal, baseline_sr, segment_context["windows"])
        baseline_aligned, base_for_baseline, _ = align_pair(baseline_signal, baseline_sr, base_signal, base_sr)
        baseline_for_bleed, accompaniment_for_baseline, _ = align_pair(baseline_signal, baseline_sr, accompaniment_signal, accompaniment_sr)
        baseline_metrics = basic_metrics(baseline_signal, baseline_sr)
        baseline_vs_base = round(cosine_similarity(baseline_aligned, base_for_baseline), 6)
        baseline_vs_accompaniment = round(cosine_similarity(baseline_for_bleed, accompaniment_for_baseline), 6)
        baseline_scores = score_bundle(baseline_metrics, baseline_vs_base, baseline_vs_accompaniment, base_metrics)

        report["baseline"] = {
            "renderId": args.baseline_render_id,
            "path": str(baseline_stem_path),
            "metrics": baseline_metrics,
            "candidate_vs_base_cosine": baseline_vs_base,
            "candidate_vs_base_accompaniment_cosine": baseline_vs_accompaniment,
            "scores": baseline_scores,
        }
        report["deltasVsBaseline"] = {
            "balanced_score_delta": round(scores["balanced_score"] - baseline_scores["balanced_score"], 6),
            "cleanliness_score_delta": round(scores["cleanliness_score"] - baseline_scores["cleanliness_score"], 6),
            "preservation_score_delta": round(scores["preservation_score"] - baseline_scores["preservation_score"], 6),
            "base_cosine_delta": round(candidate_vs_base - baseline_vs_base, 6),
            "accompaniment_cosine_delta": round(candidate_vs_accompaniment - baseline_vs_accompaniment, 6),
            "rms_dbfs_delta": round(candidate_metrics["rms_dbfs"] - baseline_metrics["rms_dbfs"], 6),
            "high_band_ratio_delta": round(candidate_metrics["high_band_ratio_4k"] - baseline_metrics["high_band_ratio_4k"], 6),
            "verdict": (
                "better"
                if scores["balanced_score"] > baseline_scores["balanced_score"]
                else "worse"
                if scores["balanced_score"] < baseline_scores["balanced_score"]
                else "same"
            ),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
