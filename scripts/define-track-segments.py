#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


APP_ROOT = Path("/app")
STORAGE_ROOT = APP_ROOT / "storage"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    return data.mean(axis=1).astype(np.float32, copy=False)


def load_audio(path: Path):
    data, sample_rate = sf.read(str(path), always_2d=False)
    return to_mono(np.asarray(data)), int(sample_rate)


def resample_if_needed(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return signal.astype(np.float32, copy=False)
    return librosa.resample(signal, orig_sr=source_sr, target_sr=target_sr).astype(np.float32, copy=False)


def ensure_common_rate(signals: list[tuple[np.ndarray, int]], target_sr: int = 22050):
    aligned = []
    min_length = None
    for signal, sample_rate in signals:
        resampled = resample_if_needed(signal, sample_rate, target_sr)
        aligned.append(resampled)
        min_length = len(resampled) if min_length is None else min(min_length, len(resampled))
    if min_length is None or min_length <= 0:
        return [], target_sr
    trimmed = [signal[:min_length].astype(np.float32, copy=False) for signal in aligned]
    return trimmed, target_sr


def normalize_series(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32, copy=False)


def percentile_rank(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values <= value))


def rolling_mean(values: np.ndarray, window_frames: int) -> np.ndarray:
    if values.size == 0:
        return values
    window_frames = max(1, int(window_frames))
    kernel = np.ones(window_frames, dtype=np.float32) / float(window_frames)
    return np.convolve(values, kernel, mode="same").astype(np.float32, copy=False)


def compute_frame_features(signal: np.ndarray, sample_rate: int, frame_sec: float):
    hop_length = max(256, int(round(sample_rate * frame_sec / 2.0)))
    frame_length = max(hop_length * 2, int(round(sample_rate * frame_sec)))

    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length, center=True)[0].astype(np.float32)
    onset = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=hop_length).astype(np.float32)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=2048, hop_length=hop_length)[0].astype(np.float32)

    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)
    return {
        "rms": rms,
        "onset": onset,
        "centroid": centroid,
        "times": frame_times.astype(np.float32),
        "hop_length": hop_length,
        "frame_length": frame_length,
    }


def resolve_structure_cache(track_id: int, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return STORAGE_ROOT / "analysis_cache" / "structure" / f"track_{track_id}.json"


def resolve_playback_manifest(structure_payload: dict, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    audio_path = Path(structure_payload["audio_path"])
    if str(audio_path).startswith("/app/storage/"):
        relative = Path(str(audio_path).replace("/app/storage/", "", 1))
        return STORAGE_ROOT / f"{relative}.playback.json"
    raise FileNotFoundError("Could not infer playback manifest path from structure cache")


def resolve_stem_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    return STORAGE_ROOT / relative


def build_selected_windows(segments: list[dict], tag: str, top_k: int):
    matching = [segment for segment in segments if tag in segment["kinds"]]
    matching.sort(key=lambda item: (-item["scores"]["segmentPriority"], item["startSec"]))
    selected = []
    for segment in matching[:top_k]:
        selected.append({
            "segmentId": segment["segmentId"],
            "startSec": segment["window"]["startSec"],
            "endSec": segment["window"]["endSec"],
            "score": segment["window"]["score"],
            "segmentStartSec": segment["startSec"],
            "segmentEndSec": segment["endSec"],
        })
    return selected


def main():
    parser = argparse.ArgumentParser(description="Define reusable segment windows for drop+vocal evaluation.")
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--structure-cache")
    parser.add_argument("--playback-manifest")
    parser.add_argument("--frame-sec", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=16.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    structure_cache_path = resolve_structure_cache(args.track_id, args.structure_cache)
    if not structure_cache_path.exists():
        raise FileNotFoundError(f"Structure cache not found: {structure_cache_path}")
    structure_payload = load_json(structure_cache_path)

    playback_manifest_path = resolve_playback_manifest(structure_payload, args.playback_manifest)
    if not playback_manifest_path.exists():
        raise FileNotFoundError(f"Playback manifest not found: {playback_manifest_path}")
    playback_manifest = load_json(playback_manifest_path)

    stems = playback_manifest.get("stems") or {}
    vocals_info = stems.get("vocals")
    accompaniment_keys = [key for key in ("drums", "bass", "other", "guitar", "piano") if stems.get(key)]
    if not vocals_info or not accompaniment_keys:
        raise RuntimeError("Playback manifest must contain vocals and accompaniment stems")

    vocals_signal, vocals_sr = load_audio(resolve_stem_path(vocals_info["path"]))
    accompaniment_signals = []
    for key in accompaniment_keys:
        accompaniment_signals.append(load_audio(resolve_stem_path(stems[key]["path"])))

    aligned, sample_rate = ensure_common_rate([(vocals_signal, vocals_sr), *accompaniment_signals])
    if len(aligned) < 2:
        raise RuntimeError("Could not align vocals/accompaniment signals")

    vocals_signal = aligned[0]
    accompaniment_signal = np.sum(np.stack(aligned[1:], axis=0), axis=0).astype(np.float32, copy=False)
    mix_signal = (vocals_signal + accompaniment_signal).astype(np.float32, copy=False)

    vocals_features = compute_frame_features(vocals_signal, sample_rate, args.frame_sec)
    accompaniment_features = compute_frame_features(accompaniment_signal, sample_rate, args.frame_sec)
    mix_features = compute_frame_features(mix_signal, sample_rate, args.frame_sec)

    frame_times = vocals_features["times"]
    if frame_times.size == 0:
        raise RuntimeError("Frame analysis produced no frames")

    vocal_rms_norm = normalize_series(vocals_features["rms"])
    accompaniment_rms_norm = normalize_series(accompaniment_features["rms"])
    accompaniment_onset_norm = normalize_series(accompaniment_features["onset"])
    mix_centroid_norm = normalize_series(mix_features["centroid"])

    vocal_activity_threshold = float(np.percentile(vocals_features["rms"], 62.0))
    accompaniment_activity_threshold = float(np.percentile(accompaniment_features["rms"], 68.0))
    vocal_active_mask = vocals_features["rms"] >= vocal_activity_threshold

    segments = []
    prev_accompaniment_mean = None
    window_frames = max(4, int(round(args.window_sec / max(args.frame_sec, 1e-6))))
    smooth_focus = rolling_mean(
        0.55 * accompaniment_rms_norm + 0.25 * accompaniment_onset_norm + 0.20 * vocal_rms_norm,
        max(3, int(round(2.0 / max(args.frame_sec, 1e-6)))),
    )

    for index, raw_segment in enumerate(structure_payload.get("segments") or []):
        start_sec = float(raw_segment["start"])
        end_sec = float(raw_segment["end"])
        label = str(raw_segment.get("label") or "unknown")
        if end_sec <= start_sec:
            continue

        frame_mask = (frame_times >= start_sec) & (frame_times < end_sec if index < len(structure_payload["segments"]) - 1 else frame_times <= end_sec)
        indices = np.flatnonzero(frame_mask)
        if indices.size == 0:
            continue

        vocal_mean = float(np.mean(vocals_features["rms"][indices]))
        vocal_peak = float(np.max(vocals_features["rms"][indices]))
        vocal_active_fraction = float(np.mean(vocal_active_mask[indices]))
        accompaniment_mean = float(np.mean(accompaniment_features["rms"][indices]))
        accompaniment_peak = float(np.max(accompaniment_features["rms"][indices]))
        accompaniment_onset_mean = float(np.mean(accompaniment_features["onset"][indices]))
        brightness_mean = float(np.mean(mix_features["centroid"][indices]))

        vocal_mean_pct = percentile_rank(vocals_features["rms"], vocal_mean)
        accompaniment_mean_pct = percentile_rank(accompaniment_features["rms"], accompaniment_mean)
        accompaniment_peak_pct = percentile_rank(accompaniment_features["rms"], accompaniment_peak)
        accompaniment_onset_pct = percentile_rank(accompaniment_features["onset"], accompaniment_onset_mean)

        transition_lift = 0.0 if prev_accompaniment_mean is None else accompaniment_mean - prev_accompaniment_mean
        prev_accompaniment_mean = accompaniment_mean
        transition_pct = percentile_rank(accompaniment_features["rms"], accompaniment_mean + max(0.0, transition_lift))

        chorus_like = label in {"chorus", "hook"}
        drop_like = chorus_like or (
            accompaniment_mean_pct >= 0.68
            and accompaniment_onset_pct >= 0.60
            and transition_lift > 0.0
        )
        has_vocals = vocal_active_fraction >= 0.38 and vocal_mean_pct >= 0.42
        vocal_drop = drop_like and has_vocals

        focus_values = smooth_focus[indices]
        local_window_frames = min(window_frames, max(2, len(indices)))
        kernel = np.ones(local_window_frames, dtype=np.float32) / float(local_window_frames)
        window_scores = np.convolve(focus_values, kernel, mode="valid")
        best_window_idx = int(np.argmax(window_scores))
        best_start_frame = int(indices[best_window_idx])
        best_end_frame = int(indices[min(best_window_idx + local_window_frames - 1, len(indices) - 1)])

        segment_priority = (
            (0.34 if chorus_like else 0.14 if drop_like else 0.0)
            + accompaniment_mean_pct * 0.22
            + accompaniment_onset_pct * 0.18
            + vocal_active_fraction * 0.14
            + vocal_mean_pct * 0.12
            + (min(1.0, max(0.0, transition_lift / max(accompaniment_activity_threshold, 1e-6))) * 0.10)
        )

        kinds = [label]
        if chorus_like:
            kinds.append("chorus_like")
        if drop_like:
            kinds.append("drop_like")
        if has_vocals:
            kinds.append("has_vocals")
        if vocal_drop:
            kinds.append("vocal_drop")

        segments.append({
            "segmentId": f"track_{args.track_id}_seg_{index:02d}",
            "startSec": round(start_sec, 3),
            "endSec": round(end_sec, 3),
            "durationSec": round(end_sec - start_sec, 3),
            "structureLabel": label,
            "kinds": kinds,
            "scores": {
                "segmentPriority": round(segment_priority, 6),
                "vocalActivityFraction": round(vocal_active_fraction, 6),
                "vocalMeanPercentile": round(vocal_mean_pct, 6),
                "vocalPeak": round(vocal_peak, 6),
                "accompanimentMeanPercentile": round(accompaniment_mean_pct, 6),
                "accompanimentPeakPercentile": round(accompaniment_peak_pct, 6),
                "accompanimentOnsetPercentile": round(accompaniment_onset_pct, 6),
                "transitionLift": round(transition_lift, 6),
                "transitionPercentile": round(transition_pct, 6),
                "brightnessNormMean": round(float(np.mean(mix_centroid_norm[indices])), 6),
            },
            "metrics": {
                "vocalMeanRms": round(vocal_mean, 8),
                "accompanimentMeanRms": round(accompaniment_mean, 8),
                "accompanimentOnsetMean": round(accompaniment_onset_mean, 8),
                "brightnessMeanHz": round(brightness_mean, 6),
            },
            "window": {
                "startSec": round(float(frame_times[best_start_frame]), 3),
                "endSec": round(float(min(end_sec, frame_times[best_end_frame] + args.frame_sec)), 3),
                "score": round(float(np.max(window_scores)), 6),
            },
        })

    selected = {
        "vocal_drop": build_selected_windows(segments, "vocal_drop", args.top_k),
        "drop_like": build_selected_windows(segments, "drop_like", args.top_k),
    }

    payload = {
        "trackId": args.track_id,
        "generatedAt": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "source": {
            "structureCachePath": str(structure_cache_path),
            "playbackManifestPath": str(playback_manifest_path),
            "audioPath": structure_payload.get("audio_path"),
            "modelName": structure_payload.get("model_name"),
        },
        "analysis": {
            "durationSec": round(float(structure_payload.get("duration") or len(mix_signal) / float(sample_rate)), 3),
            "frameSec": float(args.frame_sec),
            "windowSec": float(args.window_sec),
            "sampleRate": int(sample_rate),
            "vocalActivityThresholdRms": round(vocal_activity_threshold, 8),
            "accompanimentActivityThresholdRms": round(accompaniment_activity_threshold, 8),
        },
        "segments": segments,
        "selected": selected,
    }

    output_path = Path(args.output) if args.output else (
        STORAGE_ROOT / "analysis_cache" / "segment-definitions" / f"track_{args.track_id}_vocal_drops.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
