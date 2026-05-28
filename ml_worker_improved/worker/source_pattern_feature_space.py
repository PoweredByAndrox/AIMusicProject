from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import librosa
import numpy as np

from .instrument_melody_profile import (
    FEATURE_SAMPLE_RATE,
    NOTE_NAMES,
    _pitch_summary,
    _slice_audio,
    separate_track_stems,
)
from .melody_alignment import (
    _extract_melodic_event_chain,
    _filter_events_to_window,
    _segment_melodic_phrases,
    load_or_create_note_events,
)
from .stem_separator_defaults import DEFAULT_PATTERN_SEARCH_MODEL

IMPORTANT_STEMS = ("other", "bass", "vocals", "piano", "guitar", "drums")
MELODIC_STEMS = ("other", "vocals", "piano", "guitar")
DROP_STEMS = ("other", "bass", "drums")
DEFAULT_SEPARATOR_MODEL = DEFAULT_PATTERN_SEARCH_MODEL
DEFAULT_WINDOW_SEC = 20.0
DEFAULT_HOP_SEC = 10.0

_STEM_CACHE: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
_NOTE_EVENT_CACHE: dict[tuple[str, str, str | None], dict[str, Any]] = {}


def _normalize(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-8:
        return np.zeros_like(array)
    return array / norm


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = _normalize(left)
    right_vector = _normalize(right)
    if left_vector.size == 0 or right_vector.size == 0 or left_vector.shape != right_vector.shape:
        return 0.0
    return float(np.clip(np.dot(left_vector, right_vector), 0.0, 1.0))


def _resolved_audio_key(audio_path: str | Path) -> str:
    return str(Path(audio_path).resolve())


def _get_cached_stems(
    audio_path: str | Path,
    *,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    cache_key = (_resolved_audio_key(audio_path), separator_model, device)
    cached = _STEM_CACHE.get(cache_key)
    if cached is None:
        cached = separate_track_stems(audio_path, model_name=separator_model, device=device)
        _STEM_CACHE[cache_key] = cached
    return cached


def _get_cached_note_payload(
    audio_path: str | Path,
    *,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str = "cuda",
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    cache_key = (_resolved_audio_key(audio_path), separator_model, str(cache_root) if cache_root is not None else None)
    cached = _NOTE_EVENT_CACHE.get(cache_key)
    if cached is None:
        cached = load_or_create_note_events(
            audio_path,
            stems=("other", "bass"),
            separator_model=separator_model,
            device=device,
            cache_root=cache_root,
        )
        _NOTE_EVENT_CACHE[cache_key] = cached
    return cached


def _track_duration(audio_path: str | Path) -> float:
    return float(librosa.get_duration(path=str(audio_path)))


def _top_note_names_from_histogram(histogram: Sequence[float], *, limit: int = 4) -> list[str]:
    vector = np.asarray(histogram, dtype=np.float32).reshape(-1)
    if vector.size != 12 or not np.any(vector > 0):
        return []
    indexes = np.argsort(vector)[::-1][:limit]
    return [NOTE_NAMES[int(index)] for index in indexes if float(vector[int(index)]) > 0.0]


def _chroma_vector(audio: np.ndarray) -> np.ndarray:
    if audio.size < 2048:
        return np.zeros(12, dtype=np.float32)
    chroma = librosa.feature.chroma_cqt(
        y=np.asarray(audio, dtype=np.float32),
        sr=FEATURE_SAMPLE_RATE,
        hop_length=512,
        n_chroma=12,
    )
    if chroma.size == 0:
        return np.zeros(12, dtype=np.float32)
    return _normalize(np.mean(chroma, axis=1).astype(np.float32))


def _audio_window_summary(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {
            "rms": 0.0,
            "onset_mean": 0.0,
            "onset_std": 0.0,
            "centroid": 0.0,
            "bandwidth": 0.0,
            "flatness": 0.0,
            "zcr": 0.0,
        }

    onset_env = librosa.onset.onset_strength(y=audio, sr=FEATURE_SAMPLE_RATE, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=FEATURE_SAMPLE_RATE, hop_length=512)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=FEATURE_SAMPLE_RATE, hop_length=512)
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=512)
    return {
        "rms": round(float(np.sqrt(np.mean(np.square(audio, dtype=np.float32)))), 6),
        "onset_mean": round(float(np.mean(onset_env)) if onset_env.size else 0.0, 6),
        "onset_std": round(float(np.std(onset_env)) if onset_env.size else 0.0, 6),
        "centroid": round(float(np.mean(centroid) / (FEATURE_SAMPLE_RATE / 2.0)) if centroid.size else 0.0, 6),
        "bandwidth": round(float(np.mean(bandwidth) / (FEATURE_SAMPLE_RATE / 2.0)) if bandwidth.size else 0.0, 6),
        "flatness": round(float(np.mean(flatness)) if flatness.size else 0.0, 6),
        "zcr": round(float(np.mean(zcr)) if zcr.size else 0.0, 6),
    }


def _note_pattern_summary(
    note_events: list[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
    stem_name: str,
) -> dict[str, Any]:
    clipped = _filter_events_to_window(note_events, window_start, window_end)
    filtered = [
        event
        for event in clipped
        if float(event.get("confidence", 0.0)) >= (0.32 if stem_name == "other" else 0.28)
    ]
    pitch_hist = np.zeros(12, dtype=np.float32)
    interval_hist = np.zeros(13, dtype=np.float32)
    contour_hist = np.zeros(5, dtype=np.float32)

    ordered = sorted(filtered, key=lambda item: (float(item["start"]), int(item["midi"])))
    for event in ordered:
        pitch_hist[int(event["midi"]) % 12] += 1.0

    previous_midi: int | None = None
    for event in ordered:
        midi = int(event["midi"])
        if previous_midi is not None:
            delta = midi - previous_midi
            interval_hist[min(abs(delta), 12)] += 1.0
            if delta <= -5:
                contour_hist[0] += 1.0
            elif delta < 0:
                contour_hist[1] += 1.0
            elif delta == 0:
                contour_hist[2] += 1.0
            elif delta < 5:
                contour_hist[3] += 1.0
            else:
                contour_hist[4] += 1.0
        previous_midi = midi

    melodic_chain = _extract_melodic_event_chain(
        clipped,
        stem_name=stem_name,
        min_confidence=0.22 if stem_name == "other" else 0.18,
        min_duration_sec=0.12,
    )
    phrases = _segment_melodic_phrases(melodic_chain)
    phrase_signatures: list[list[str]] = []
    for phrase in phrases[:4]:
        note_classes = [str(item["note_class"]) for item in phrase.get("events", [])[:6]]
        if note_classes:
            phrase_signatures.append(note_classes)

    unique_pitch_classes = int(np.count_nonzero(pitch_hist))
    return {
        "event_count": len(ordered),
        "unique_pitch_classes": unique_pitch_classes,
        "phrase_count": len(phrases),
        "top_note_classes": _top_note_names_from_histogram(pitch_hist),
        "pitch_histogram": [round(float(value), 6) for value in _normalize(pitch_hist)],
        "interval_histogram": [round(float(value), 6) for value in _normalize(interval_hist)],
        "contour_histogram": [round(float(value), 6) for value in _normalize(contour_hist)],
        "phrase_signatures": phrase_signatures,
    }


def extract_window_source_profile(
    audio_path: str | Path,
    *,
    window_start: float,
    window_end: float,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str = "cuda",
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    stems = _get_cached_stems(audio_path, separator_model=separator_model, device=device)
    note_payload = _get_cached_note_payload(
        audio_path,
        separator_model=separator_model,
        device=device,
        cache_root=cache_root,
    )

    total_energy = 0.0
    stem_profiles: dict[str, dict[str, Any]] = {}
    for stem_name in IMPORTANT_STEMS:
        stem_audio = stems.get(stem_name)
        if stem_audio is None:
            continue
        clip = _slice_audio(stem_audio, window_start, window_end, FEATURE_SAMPLE_RATE)
        energy = float(np.sqrt(np.mean(clip ** 2))) if clip.size else 0.0
        total_energy += energy
        pitch_summary = _pitch_summary(clip, FEATURE_SAMPLE_RATE)
        chroma_vector = _chroma_vector(clip)
        spectral = _audio_window_summary(clip)
        stem_profiles[stem_name] = {
            "energy": energy,
            "energy_ratio": 0.0,
            "melodic_ratio": float(pitch_summary["melodic_ratio"]),
            "median_midi": pitch_summary["median_midi"],
            "mean_confidence": float(pitch_summary["mean_confidence"]),
            "pitch_top_notes": list(pitch_summary["pitch_top_notes"]),
            "top_chroma_notes": _top_note_names_from_histogram(chroma_vector),
            "chroma_vector": [round(float(value), 6) for value in chroma_vector],
            **spectral,
        }

    normalized_energy = total_energy or 1.0
    for features in stem_profiles.values():
        features["energy_ratio"] = round(float(features["energy"] / normalized_energy), 6)
        del features["energy"]

    other_payload = note_payload["stems"].get("other", {})
    other_events = other_payload.get("orchestral_note_events", other_payload.get("note_events", []))
    bass_events = note_payload["stems"].get("bass", {}).get("note_events", [])
    note_patterns = {
        "other": _note_pattern_summary(
            list(other_events),
            window_start=window_start,
            window_end=window_end,
            stem_name="other",
        ),
        "bass": _note_pattern_summary(
            list(bass_events),
            window_start=window_start,
            window_end=window_end,
            stem_name="bass",
        ),
    }

    energy_vector = [stem_profiles.get(stem_name, {}).get("energy_ratio", 0.0) for stem_name in IMPORTANT_STEMS]
    melodic_ratio_vector = [stem_profiles.get(stem_name, {}).get("melodic_ratio", 0.0) for stem_name in IMPORTANT_STEMS]
    aggregate_chroma = np.zeros(12, dtype=np.float32)
    total_weight = 0.0
    for stem_name in MELODIC_STEMS:
        features = stem_profiles.get(stem_name)
        if not features:
            continue
        weight = float(features["energy_ratio"]) * max(0.1, float(features["melodic_ratio"]))
        aggregate_chroma += np.asarray(features["chroma_vector"], dtype=np.float32) * weight
        total_weight += weight
    if total_weight > 0.0:
        aggregate_chroma = _normalize(aggregate_chroma / total_weight)

    return {
        "audio_path": str(audio_path),
        "window": [round(float(window_start), 3), round(float(window_end), 3)],
        "duration_sec": round(float(window_end - window_start), 3),
        "stems": stem_profiles,
        "note_patterns": note_patterns,
        "feature_vectors": {
            "energy": [round(float(value), 6) for value in _normalize(energy_vector)],
            "melodic_ratio": [round(float(value), 6) for value in _normalize(melodic_ratio_vector)],
            "aggregate_chroma": [round(float(value), 6) for value in aggregate_chroma],
        },
    }


def compare_window_source_profiles(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    instrumentation_similarity = _cosine_similarity(
        left["feature_vectors"]["energy"],
        right["feature_vectors"]["energy"],
    )
    stem_role_similarity = _cosine_similarity(
        left["feature_vectors"]["melodic_ratio"],
        right["feature_vectors"]["melodic_ratio"],
    )
    aggregate_chroma_similarity = _cosine_similarity(
        left["feature_vectors"]["aggregate_chroma"],
        right["feature_vectors"]["aggregate_chroma"],
    )
    other_pitch_similarity = _cosine_similarity(
        left["note_patterns"]["other"]["pitch_histogram"],
        right["note_patterns"]["other"]["pitch_histogram"],
    )
    other_interval_similarity = _cosine_similarity(
        left["note_patterns"]["other"]["interval_histogram"],
        right["note_patterns"]["other"]["interval_histogram"],
    )
    other_contour_similarity = _cosine_similarity(
        left["note_patterns"]["other"]["contour_histogram"],
        right["note_patterns"]["other"]["contour_histogram"],
    )
    bass_pitch_similarity = _cosine_similarity(
        left["note_patterns"]["bass"]["pitch_histogram"],
        right["note_patterns"]["bass"]["pitch_histogram"],
    )
    bass_interval_similarity = _cosine_similarity(
        left["note_patterns"]["bass"]["interval_histogram"],
        right["note_patterns"]["bass"]["interval_histogram"],
    )
    phrase_count_left = int(left["note_patterns"]["other"]["phrase_count"])
    phrase_count_right = int(right["note_patterns"]["other"]["phrase_count"])
    phrase_density_similarity = 1.0 - (
        abs(phrase_count_left - phrase_count_right) / max(1.0, float(max(phrase_count_left, phrase_count_right, 1)))
    )
    phrase_density_similarity = float(np.clip(phrase_density_similarity, 0.0, 1.0))

    overall_similarity = (
        0.18 * instrumentation_similarity
        + 0.12 * stem_role_similarity
        + 0.2 * aggregate_chroma_similarity
        + 0.2 * ((other_pitch_similarity + other_interval_similarity + other_contour_similarity) / 3.0)
        + 0.15 * ((bass_pitch_similarity + bass_interval_similarity) / 2.0)
        + 0.15 * phrase_density_similarity
    )
    shared_other_notes = sorted(
        set(left["note_patterns"]["other"]["top_note_classes"]) & set(right["note_patterns"]["other"]["top_note_classes"])
    )
    shared_chroma_notes = sorted(
        set(left["stems"].get("other", {}).get("top_chroma_notes", []))
        & set(right["stems"].get("other", {}).get("top_chroma_notes", []))
    )
    return {
        "overall_similarity": round(float(overall_similarity), 4),
        "component_similarities": {
            "instrumentation_similarity": round(float(instrumentation_similarity), 4),
            "stem_role_similarity": round(float(stem_role_similarity), 4),
            "aggregate_chroma_similarity": round(float(aggregate_chroma_similarity), 4),
            "other_pitch_similarity": round(float(other_pitch_similarity), 4),
            "other_interval_similarity": round(float(other_interval_similarity), 4),
            "other_contour_similarity": round(float(other_contour_similarity), 4),
            "bass_pitch_similarity": round(float(bass_pitch_similarity), 4),
            "bass_interval_similarity": round(float(bass_interval_similarity), 4),
            "phrase_density_similarity": round(float(phrase_density_similarity), 4),
        },
        "shared_other_notes": shared_other_notes,
        "shared_other_chroma_notes": shared_chroma_notes,
    }


def pick_top_source_windows(
    audio_path: str | Path,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    top_k: int = 3,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str = "cuda",
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    duration_sec = _track_duration(audio_path)
    ranked: list[dict[str, Any]] = []
    stop = max(0.0, duration_sec - window_sec)
    start = 0.0
    while start <= stop + 1e-6:
        end = min(duration_sec, start + window_sec)
        profile = extract_window_source_profile(
            audio_path,
            window_start=start,
            window_end=end,
            separator_model=separator_model,
            device=device,
            cache_root=cache_root,
        )
        other = profile["stems"].get("other", {})
        vocals = profile["stems"].get("vocals", {})
        piano = profile["stems"].get("piano", {})
        guitar = profile["stems"].get("guitar", {})
        bass = profile["stems"].get("bass", {})
        other_pattern = profile["note_patterns"]["other"]
        score = (
            0.32 * float(other.get("energy_ratio", 0.0))
            + 0.22 * float(other.get("melodic_ratio", 0.0))
            + 0.1 * float(vocals.get("melodic_ratio", 0.0))
            + 0.08 * float(piano.get("melodic_ratio", 0.0))
            + 0.08 * float(guitar.get("melodic_ratio", 0.0))
            + 0.08 * float(bass.get("melodic_ratio", 0.0))
            + 0.07 * min(1.0, float(other_pattern["phrase_count"]) / 8.0)
            + 0.05 * min(1.0, float(other_pattern["unique_pitch_classes"]) / 6.0)
        )
        ranked.append(
            {
                "window": [round(float(start), 3), round(float(end), 3)],
                "score": round(float(score), 4),
                "profile": profile,
            }
        )
        start += hop_sec

    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    picked: list[dict[str, Any]] = []
    for candidate in ranked:
        candidate_start = float(candidate["window"][0])
        if any(abs(candidate_start - float(item["window"][0])) <= (window_sec * 0.5) for item in picked):
            continue
        picked.append(candidate)
        if len(picked) >= top_k:
            break

    return {
        "audio_path": str(audio_path),
        "duration_sec": round(float(duration_sec), 3),
        "ranked": ranked[: max(top_k * 4, top_k)],
        "picked": picked,
    }


def _profile_change_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return round(1.0 - float(compare_window_source_profiles(left, right)["overall_similarity"]), 4)


def diagnose_segment_boundaries(
    audio_path: str | Path,
    *,
    segment_start: float,
    segment_end: float,
    window_sec: float = 6.0,
    hop_sec: float = 3.0,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str = "cuda",
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    current = float(segment_start)
    while current + 1.0 < float(segment_end):
        end = min(float(segment_end), current + window_sec)
        profiles.append(
            extract_window_source_profile(
                audio_path,
                window_start=current,
                window_end=end,
                separator_model=separator_model,
                device=device,
                cache_root=cache_root,
            )
        )
        if end >= float(segment_end):
            break
        current += hop_sec

    adjacent_changes: list[dict[str, Any]] = []
    for left, right in zip(profiles, profiles[1:]):
        change_score = _profile_change_score(left, right)
        adjacent_changes.append(
            {
                "left_window": left["window"],
                "right_window": right["window"],
                "change_score": change_score,
                "midpoint_sec": round((float(left["window"][1]) + float(right["window"][0])) / 2.0, 3),
            }
        )

    recommended_splits = [
        item for item in adjacent_changes
        if float(item["change_score"]) >= 0.28
    ]
    recommended_splits.sort(key=lambda item: float(item["change_score"]), reverse=True)
    return {
        "audio_path": str(audio_path),
        "segment_window": [round(float(segment_start), 3), round(float(segment_end), 3)],
        "subwindows": [profile["window"] for profile in profiles],
        "adjacent_changes": adjacent_changes,
        "recommended_splits": recommended_splits[:3],
    }
