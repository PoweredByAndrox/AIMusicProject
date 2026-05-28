from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import librosa
import numpy as np

from worker.instrument_melody_profile import FEATURE_SAMPLE_RATE, separate_track_stems
from worker.main import handle_embedding_recommendations
from worker.melody_alignment import compare_tracks_by_melody_alignment
from worker.queue_worker import QueueWorker
from worker.recommendation_lab_gate import evaluate_recommendation_lab_gate
from worker.recommendation_melodic_rerank import classify_section_family, score_section_aware_candidate

MANUAL_POSITIVE_SEEDS = [
    {"track_id": 9152, "start": 56.935, "end": 77.439, "type": "verse"},
    {"track_id": 5159, "start": 154.389, "end": 176.727, "type": "verse"},
    {"track_id": 4230, "start": 171.294, "end": 191.843, "type": "drop_bass"},
]

MANUAL_NEGATIVE_SEEDS = [
    {"track_id": 9152, "start": 85.775, "end": 99.59, "type": "verse"},
    {"track_id": 5159, "start": 49.714, "end": 77.508, "type": "melodic"},
]

HOOK_SEED_MIN_DURATION = 10.0
HOOK_SEED_MAX_DURATION = 40.0
DROP_SEED_MIN_DURATION = 8.0
DROP_SEED_MAX_DURATION = 60.0
HOOK_CANDIDATE_MIN_DURATION = 12.0
HOOK_CANDIDATE_MAX_DURATION = 36.0
DROP_CANDIDATE_MIN_DURATION = 3.0
DROP_CANDIDATE_MAX_DURATION = 60.0
MAX_HOOK_CANDIDATES = 3
MAX_DROP_CANDIDATES = 3
MAX_TOTAL_CANDIDATES = 5
DISCRIMINATIVE_POOL_LIMIT = 18
DISCRIMINATIVE_SAME_TOP_K = 2
DISCRIMINATIVE_OPPOSITE_TOP_K = 3
DROP_RAW_FEATURE_HOP = 512

_RAW_STEM_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def _parse_segment_type(notes: Any) -> str:
    for part in str(notes or "").split(";"):
        if part.startswith("type="):
            return part.split("=", 1)[1].strip()
    return "unknown"


def _load_embedding(path_str: str | None) -> np.ndarray | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    emb = np.load(path).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm <= 1e-10:
        return None
    return emb / norm


def _segment_family_sql_clause(family: str, *, alias: str = "pe") -> str:
    if family == "hook":
        return f"({alias}.notes ILIKE '%%type=verse%%' OR {alias}.notes ILIKE '%%type=melodic%%')"
    if family == "drop":
        return f"{alias}.notes ILIKE '%%type=drop_bass%%'"
    raise ValueError(f"Unsupported family: {family}")


def _segment_duration_bounds(family: str, *, candidate: bool) -> tuple[float, float]:
    if family == "hook":
        return (
            (HOOK_CANDIDATE_MIN_DURATION, HOOK_CANDIDATE_MAX_DURATION)
            if candidate
            else (HOOK_SEED_MIN_DURATION, HOOK_SEED_MAX_DURATION)
        )
    if family == "drop":
        return (
            (DROP_CANDIDATE_MIN_DURATION, DROP_CANDIDATE_MAX_DURATION)
            if candidate
            else (DROP_SEED_MIN_DURATION, DROP_SEED_MAX_DURATION)
        )
    raise ValueError(f"Unsupported family: {family}")


def _query_seed_rows(
    worker: QueueWorker,
    *,
    user_id: int,
    min_rating: int | None,
    max_rating: int | None,
    family: str,
    limit: int,
    distinct_track_limit: int | None = None,
    exclude_track_ids: Iterable[int] = (),
) -> list[dict[str, Any]]:
    clauses = ["upr.user_id = %s"]
    params: list[Any] = [int(user_id)]
    if min_rating is not None:
        clauses.append("upr.rating >= %s")
        params.append(int(min_rating))
    if max_rating is not None:
        clauses.append("upr.rating <= %s")
        params.append(int(max_rating))

    family_clause = _segment_family_sql_clause(family)
    min_duration, max_duration = _segment_duration_bounds(family, candidate=False)
    clauses.append(family_clause)
    clauses.append("(pe.end_time - pe.start_time) BETWEEN %s AND %s")
    params.extend([float(min_duration), float(max_duration)])

    conflict_family_clause = _segment_family_sql_clause(family, alias="pe2")
    if min_rating is not None and max_rating is None:
        clauses.append(
            f"""
            NOT EXISTS (
                SELECT 1
                FROM user_pattern_example_ratings upr2
                JOIN pattern_examples pe2 ON pe2.id = upr2.pattern_example_id
                WHERE upr2.user_id = upr.user_id
                  AND pe2.track_id = pe.track_id
                  AND {conflict_family_clause}
                  AND upr2.rating <= 2
            )
            """
        )
    if max_rating is not None and min_rating is None:
        clauses.append(
            f"""
            NOT EXISTS (
                SELECT 1
                FROM user_pattern_example_ratings upr2
                JOIN pattern_examples pe2 ON pe2.id = upr2.pattern_example_id
                WHERE upr2.user_id = upr.user_id
                  AND pe2.track_id = pe.track_id
                  AND {conflict_family_clause}
                  AND upr2.rating >= 4
            )
            """
        )

    order_by = "upr.rating DESC, (pe.end_time - pe.start_time) DESC, t.id, pe.start_time"
    if max_rating is not None and min_rating is None:
        order_by = "upr.rating ASC, (pe.end_time - pe.start_time) DESC, t.id, pe.start_time"

    query = f"""
        SELECT t.id,
               t.title,
               t.artist,
               t.file_path,
               ROUND(pe.start_time::numeric, 3) AS start_time,
               ROUND(pe.end_time::numeric, 3) AS end_time,
               pe.notes,
               upr.rating,
               pe.embedding_path
        FROM user_pattern_example_ratings upr
        JOIN pattern_examples pe ON pe.id = upr.pattern_example_id
        JOIN tracks t ON t.id = pe.track_id
        WHERE {' AND '.join(clauses)}
        ORDER BY {order_by}
    """

    excluded_track_ids = {int(track_id) for track_id in exclude_track_ids}
    selected: list[dict[str, Any]] = []
    seen_tracks: set[int] = set()
    with worker.db_conn.cursor() as cursor:
        cursor.execute(query, tuple(params))
        for track_id, title, artist, file_path, start_time, end_time, notes, rating, segment_embedding_path in cursor.fetchall():
            track_id = int(track_id)
            if track_id in excluded_track_ids:
                continue
            if distinct_track_limit is not None:
                if track_id in seen_tracks:
                    continue
                if len(seen_tracks) >= int(distinct_track_limit):
                    break
            segment_type = _parse_segment_type(notes)
            selected.append(
                {
                    "track_id": track_id,
                    "title": title,
                    "artist": artist,
                    "audio_path": str(Path("/app/storage") / str(file_path)),
                    "start": float(start_time),
                    "end": float(end_time),
                    "type": segment_type,
                    "rating": int(rating),
                    "family": classify_section_family(segment_type),
                    "segment_embedding_path": segment_embedding_path,
                }
            )
            seen_tracks.add(track_id)
            if len(selected) >= int(limit):
                break
    return selected


def _candidate_seed_key(seed: Mapping[str, Any]) -> tuple[int, float, float, str]:
    return (
        int(seed["track_id"]),
        round(float(seed["start"]), 3),
        round(float(seed["end"]), 3),
        str(seed["type"]),
    )


def _segment_embedding_for_seed(seed: Mapping[str, Any]) -> np.ndarray | None:
    direct_embedding = seed.get("_segment_embedding")
    if isinstance(direct_embedding, np.ndarray):
        return direct_embedding.astype(np.float32)
    return _load_embedding(seed.get("segment_embedding_path"))


def _safe_balance(left: float, right: float) -> float:
    left_value = abs(float(left))
    right_value = abs(float(right))
    if left_value <= 1e-9 and right_value <= 1e-9:
        return 1.0
    high = max(left_value, right_value)
    low = min(left_value, right_value)
    return low / high if high > 1e-9 else 0.0


def _get_cached_track_stems(audio_path: str, *, device: str = "cuda") -> dict[str, np.ndarray]:
    cache_key = (str(audio_path), str(device))
    cached = _RAW_STEM_CACHE.get(cache_key)
    if cached is None:
        cached = separate_track_stems(audio_path, model_name="htdemucs_6s", device=device)
        _RAW_STEM_CACHE[cache_key] = cached
    return cached


def _slice_stem_window(audio: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    start_sample = max(0, int(round(float(start_time) * FEATURE_SAMPLE_RATE)))
    end_sample = max(start_sample, int(round(float(end_time) * FEATURE_SAMPLE_RATE)))
    return np.asarray(audio[start_sample:end_sample], dtype=np.float32)


def _summarize_audio_window(audio: np.ndarray) -> dict[str, float]:
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

    rms = float(np.sqrt(np.mean(np.square(audio))))
    onset_env = librosa.onset.onset_strength(y=audio, sr=FEATURE_SAMPLE_RATE, hop_length=DROP_RAW_FEATURE_HOP)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=FEATURE_SAMPLE_RATE, hop_length=DROP_RAW_FEATURE_HOP)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=FEATURE_SAMPLE_RATE, hop_length=DROP_RAW_FEATURE_HOP)
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=DROP_RAW_FEATURE_HOP)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=DROP_RAW_FEATURE_HOP)

    return {
        "rms": round(float(np.log1p(max(rms, 0.0) * 1000.0)), 6),
        "onset_mean": round(float(np.log1p(float(np.mean(onset_env)) if onset_env.size else 0.0)), 6),
        "onset_std": round(float(np.log1p(float(np.std(onset_env)) if onset_env.size else 0.0)), 6),
        "centroid": round(float(np.mean(centroid) / (FEATURE_SAMPLE_RATE / 2.0)) if centroid.size else 0.0, 6),
        "bandwidth": round(float(np.mean(bandwidth) / (FEATURE_SAMPLE_RATE / 2.0)) if bandwidth.size else 0.0, 6),
        "flatness": round(float(np.mean(flatness)) if flatness.size else 0.0, 6),
        "zcr": round(float(np.mean(zcr)) if zcr.size else 0.0, 6),
    }


def _extract_drop_stem_window_features(
    stems: Mapping[str, np.ndarray],
    *,
    start_time: float,
    end_time: float,
) -> dict[str, float]:
    bass_audio = _slice_stem_window(np.asarray(stems.get("bass", np.zeros(0, dtype=np.float32))), start_time, end_time)
    drums_audio = _slice_stem_window(np.asarray(stems.get("drums", np.zeros(0, dtype=np.float32))), start_time, end_time)
    other_audio = _slice_stem_window(np.asarray(stems.get("other", np.zeros(0, dtype=np.float32))), start_time, end_time)

    bass = _summarize_audio_window(bass_audio)
    drums = _summarize_audio_window(drums_audio)
    other = _summarize_audio_window(other_audio)

    bass_energy = np.expm1(bass["rms"])
    drums_energy = np.expm1(drums["rms"])
    other_energy = np.expm1(other["rms"])
    total_energy = max(bass_energy + drums_energy + other_energy, 1e-9)

    bass_onset = np.expm1(bass["onset_mean"])
    drums_onset = np.expm1(drums["onset_mean"])
    other_onset = np.expm1(other["onset_mean"])
    total_onset = max(bass_onset + drums_onset + other_onset, 1e-9)

    return {
        "bass_rms": bass["rms"],
        "bass_onset_mean": bass["onset_mean"],
        "bass_onset_std": bass["onset_std"],
        "bass_centroid": bass["centroid"],
        "bass_bandwidth": bass["bandwidth"],
        "bass_zcr": bass["zcr"],
        "drums_rms": drums["rms"],
        "drums_onset_mean": drums["onset_mean"],
        "drums_onset_std": drums["onset_std"],
        "drums_centroid": drums["centroid"],
        "drums_bandwidth": drums["bandwidth"],
        "other_rms": other["rms"],
        "other_onset_mean": other["onset_mean"],
        "other_flatness": other["flatness"],
        "other_centroid": other["centroid"],
        "bass_energy_share": round(float(bass_energy / total_energy), 6),
        "drums_energy_share": round(float(drums_energy / total_energy), 6),
        "bass_onset_share": round(float(bass_onset / total_onset), 6),
        "drums_onset_share": round(float(drums_onset / total_onset), 6),
        "duration_sec": round(float(max(0.0, float(end_time) - float(start_time))), 6),
    }


def _drop_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    keys = (
        "bass_rms",
        "bass_onset_mean",
        "bass_onset_std",
        "bass_centroid",
        "bass_bandwidth",
        "bass_zcr",
        "drums_rms",
        "drums_onset_mean",
        "drums_onset_std",
        "drums_centroid",
        "drums_bandwidth",
        "other_rms",
        "other_onset_mean",
        "other_flatness",
        "other_centroid",
        "bass_energy_share",
        "drums_energy_share",
        "bass_onset_share",
        "drums_onset_share",
    )
    return np.asarray([float(features.get(key, 0.0) or 0.0) for key in keys], dtype=np.float32)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _extract_raw_drop_match_features(
    *,
    seed_audio_path: str,
    seed_start: float,
    seed_end: float,
    candidate_audio_path: str,
    candidate_start: float,
    candidate_end: float,
    device: str = "cuda",
) -> dict[str, float]:
    seed_stems = _get_cached_track_stems(seed_audio_path, device=device)
    candidate_stems = _get_cached_track_stems(candidate_audio_path, device=device)
    seed_features = _extract_drop_stem_window_features(seed_stems, start_time=seed_start, end_time=seed_end)
    candidate_features = _extract_drop_stem_window_features(candidate_stems, start_time=candidate_start, end_time=candidate_end)
    raw_similarity = _cosine_similarity(_drop_feature_vector(seed_features), _drop_feature_vector(candidate_features))

    return {
        "raw_drop_similarity": round(float(raw_similarity), 4),
        **{f"seed_{key}": value for key, value in seed_features.items()},
        **{f"candidate_{key}": value for key, value in candidate_features.items()},
    }


def _extract_drop_specific_features(comparison: Mapping[str, Any]) -> dict[str, float]:
    stems = comparison.get("stems") if isinstance(comparison, Mapping) else None
    other = stems.get("other") if isinstance(stems, Mapping) else {}
    bass = stems.get("bass") if isinstance(stems, Mapping) else {}

    other_struct = float(other.get("structural_similarity", 0.0) or 0.0)
    bass_struct = float(bass.get("structural_similarity", 0.0) or 0.0)
    bass_beat = float(bass.get("beat_sync_combined_similarity", 0.0) or 0.0)
    bass_contour = float(bass.get("contour_similarity", 0.0) or 0.0)

    other_event_balance = _safe_balance(other.get("event_count_a", 0), other.get("event_count_b", 0))
    bass_event_balance = _safe_balance(bass.get("event_count_a", 0), bass.get("event_count_b", 0))
    event_balance = 0.5 * other_event_balance + 0.5 * bass_event_balance

    drop_feature_similarity = (
        0.4 * bass_struct
        + 0.2 * bass_beat
        + 0.15 * bass_contour
        + 0.15 * other_struct
        + 0.1 * event_balance
    )

    return {
        "drop_feature_similarity": round(float(drop_feature_similarity), 4),
        "drop_bass_structural_similarity": round(float(bass_struct), 4),
        "drop_bass_beat_sync_similarity": round(float(bass_beat), 4),
        "drop_bass_contour_similarity": round(float(bass_contour), 4),
        "drop_other_structural_similarity": round(float(other_struct), 4),
        "drop_other_event_balance": round(float(other_event_balance), 4),
        "drop_bass_event_balance": round(float(bass_event_balance), 4),
        "drop_event_balance": round(float(event_balance), 4),
    }


def _rank_discriminative_seed_rows(
    seeds: Sequence[dict[str, Any]],
    *,
    opposite_seeds: Sequence[dict[str, Any]],
    same_top_k: int = DISCRIMINATIVE_SAME_TOP_K,
    opposite_top_k: int = DISCRIMINATIVE_OPPOSITE_TOP_K,
) -> list[dict[str, Any]]:
    prepared: list[tuple[dict[str, Any], np.ndarray]] = []
    for seed in seeds:
        embedding = _segment_embedding_for_seed(seed)
        if embedding is None:
            continue
        prepared.append((seed, embedding))

    prepared_opposites: list[tuple[dict[str, Any], np.ndarray]] = []
    for seed in opposite_seeds:
        embedding = _segment_embedding_for_seed(seed)
        if embedding is None:
            continue
        prepared_opposites.append((seed, embedding))

    ranked: list[dict[str, Any]] = []
    for seed, embedding in prepared:
        same_sims = [
            float(np.dot(embedding, other_embedding))
            for other_seed, other_embedding in prepared
            if _candidate_seed_key(other_seed) != _candidate_seed_key(seed)
        ]
        opposite_sims = [float(np.dot(embedding, other_embedding)) for _, other_embedding in prepared_opposites]

        same_sims.sort(reverse=True)
        opposite_sims.sort(reverse=True)

        same_support = float(np.mean(same_sims[: int(same_top_k)])) if same_sims else 0.0
        opposite_pressure = float(np.mean(opposite_sims[: int(opposite_top_k)])) if opposite_sims else 0.0
        discriminative_score = same_support - opposite_pressure
        ranked.append(
            {
                **seed,
                "same_support": round(same_support, 4),
                "opposite_pressure": round(opposite_pressure, 4),
                "discriminative_score": round(discriminative_score, 4),
            }
        )

    ranked.sort(
        key=lambda item: (
            float(item["discriminative_score"]),
            -float(item["opposite_pressure"]),
            float(item["same_support"]),
            float(item["end"]) - float(item["start"]),
        ),
        reverse=True,
    )
    return ranked


def _select_discriminative_seed_rows(
    seeds: Sequence[dict[str, Any]],
    *,
    opposite_seeds: Sequence[dict[str, Any]],
    limit: int,
    distinct_track_limit: int | None,
) -> list[dict[str, Any]]:
    ranked = _rank_discriminative_seed_rows(seeds, opposite_seeds=opposite_seeds)
    selected: list[dict[str, Any]] = []
    seen_tracks: set[int] = set()
    for seed in ranked:
        track_id = int(seed["track_id"])
        if distinct_track_limit is not None:
            if track_id in seen_tracks:
                continue
            if len(seen_tracks) >= int(distinct_track_limit):
                break
        selected.append(seed)
        seen_tracks.add(track_id)
        if len(selected) >= int(limit):
            break
    return selected


def _build_seed_windows(
    worker: QueueWorker,
    *,
    user_id: int,
    policy: str,
    exclude_track_ids: Iterable[int] = (),
) -> dict[str, list[dict[str, Any]]]:
    excluded_track_ids = {int(track_id) for track_id in exclude_track_ids}
    if policy == "manual_with_explicit_negative_drop":
        positive = [seed for seed in MANUAL_POSITIVE_SEEDS if int(seed["track_id"]) not in excluded_track_ids]
        negative = [seed for seed in MANUAL_NEGATIVE_SEEDS if int(seed["track_id"]) not in excluded_track_ids]
        negative.extend(
            _query_seed_rows(
                worker,
                user_id=user_id,
                min_rating=None,
                max_rating=2,
                family="drop",
                limit=3,
                distinct_track_limit=3,
                exclude_track_ids=excluded_track_ids,
            )
        )
        return {"positive": positive, "negative": negative}

    if policy == "family_balanced_from_ratings":
        positive = (
            _query_seed_rows(
                worker,
                user_id=user_id,
                min_rating=4,
                max_rating=None,
                family="hook",
                limit=3,
                distinct_track_limit=3,
                exclude_track_ids=excluded_track_ids,
            )
            + _query_seed_rows(
                worker,
                user_id=user_id,
                min_rating=4,
                max_rating=None,
                family="drop",
                limit=3,
                distinct_track_limit=3,
                exclude_track_ids=excluded_track_ids,
            )
        )
        negative = (
            _query_seed_rows(
                worker,
                user_id=user_id,
                min_rating=None,
                max_rating=2,
                family="hook",
                limit=3,
                distinct_track_limit=3,
                exclude_track_ids=excluded_track_ids,
            )
            + _query_seed_rows(
                worker,
                user_id=user_id,
                min_rating=None,
                max_rating=2,
                family="drop",
                limit=3,
                distinct_track_limit=3,
                exclude_track_ids=excluded_track_ids,
            )
        )
        return {"positive": positive, "negative": negative}

    if policy == "family_balanced_discriminative_drop":
        positive_hook = _query_seed_rows(
            worker,
            user_id=user_id,
            min_rating=4,
            max_rating=None,
            family="hook",
            limit=3,
            distinct_track_limit=3,
            exclude_track_ids=excluded_track_ids,
        )
        negative_hook = _query_seed_rows(
            worker,
            user_id=user_id,
            min_rating=None,
            max_rating=2,
            family="hook",
            limit=3,
            distinct_track_limit=3,
            exclude_track_ids=excluded_track_ids,
        )
        positive_drop_pool = _query_seed_rows(
            worker,
            user_id=user_id,
            min_rating=4,
            max_rating=None,
            family="drop",
            limit=DISCRIMINATIVE_POOL_LIMIT,
            distinct_track_limit=None,
            exclude_track_ids=excluded_track_ids,
        )
        negative_drop_pool = _query_seed_rows(
            worker,
            user_id=user_id,
            min_rating=None,
            max_rating=2,
            family="drop",
            limit=DISCRIMINATIVE_POOL_LIMIT,
            distinct_track_limit=None,
            exclude_track_ids=excluded_track_ids,
        )
        positive_drop = _select_discriminative_seed_rows(
            positive_drop_pool,
            opposite_seeds=negative_drop_pool,
            limit=3,
            distinct_track_limit=3,
        )
        negative_drop = _select_discriminative_seed_rows(
            negative_drop_pool,
            opposite_seeds=positive_drop_pool,
            limit=3,
            distinct_track_limit=3,
        )
        return {
            "positive": positive_hook + positive_drop,
            "negative": negative_hook + negative_drop,
        }

    raise ValueError(f"Unsupported seed policy: {policy}")


def _seed_family_summary(seed_windows: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for polarity, seeds in seed_windows.items():
        family_counts = {"hook": 0, "drop": 0, "other": 0}
        for seed in seeds:
            family = classify_section_family(seed.get("type"))
            family_counts[family] = family_counts.get(family, 0) + 1
        summary[str(polarity)] = family_counts
    return summary


def _load_shortlist(
    worker: QueueWorker,
    *,
    user_id: int,
    shortlist_limit: int,
    special_track_ids: Iterable[int],
) -> list[int]:
    result = handle_embedding_recommendations(
        {"limit": 200, "user_id": int(user_id), "task_id": f"codex_sectioned_{user_id}_{shortlist_limit}"},
        worker,
    )
    shortlist = [int(item["track"]["id"]) for item in (result.get("recommendations") or [])[: int(shortlist_limit)]]
    for track_id in special_track_ids:
        if int(track_id) not in shortlist:
            shortlist.append(int(track_id))
    return shortlist


def _candidate_segments_for_track(worker: QueueWorker, track_id: int) -> list[dict[str, Any]]:
    hook_segments: list[dict[str, Any]] = []
    drop_segments: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT pe.start_time, pe.end_time, pe.notes
            FROM pattern_examples pe
            JOIN patterns p ON p.id = pe.pattern_id
            WHERE pe.track_id = %s
              AND p.name LIKE '%%[auto%%'
            ORDER BY pe.start_time
            """,
            (int(track_id),),
        )
        for start_time, end_time, notes in cursor.fetchall():
            duration = float(end_time) - float(start_time)
            segment = {
                "start": float(start_time),
                "end": float(end_time),
                "duration": duration,
                "type": _parse_segment_type(notes),
            }
            if segment["type"] in {"verse", "melodic"}:
                if HOOK_CANDIDATE_MIN_DURATION <= duration <= HOOK_CANDIDATE_MAX_DURATION:
                    hook_segments.append(segment)
                continue
            if segment["type"] == "drop_bass":
                if DROP_CANDIDATE_MIN_DURATION <= duration <= DROP_CANDIDATE_MAX_DURATION:
                    drop_segments.append(segment)
                continue
            if HOOK_CANDIDATE_MIN_DURATION <= duration <= HOOK_CANDIDATE_MAX_DURATION:
                fallback.append(segment)
    hook_segments.sort(key=lambda item: (-float(item["duration"]), float(item["start"])))
    drop_segments.sort(key=lambda item: (-float(item["duration"]), float(item["start"])))
    fallback.sort(key=lambda item: (-float(item["duration"]), float(item["start"])))

    selected = list(hook_segments[:MAX_HOOK_CANDIDATES])
    remaining_slots = MAX_TOTAL_CANDIDATES - len(selected)
    drop_limit = min(MAX_DROP_CANDIDATES, max(remaining_slots, 0))
    selected.extend(drop_segments[:drop_limit])
    remaining_slots = MAX_TOTAL_CANDIDATES - len(selected)
    if remaining_slots > 0:
        selected.extend(fallback[:remaining_slots])
    selected.sort(key=lambda item: float(item["start"]))
    return selected[:MAX_TOTAL_CANDIDATES]


def run_section_aware_lab(
    *,
    user_id: int,
    seed_policy: str,
    shortlist_limit: int,
    holdout_track_ids: Sequence[int],
    special_track_ids: Sequence[int],
    cache_root: Path,
    conflict_top_rank_limit: int,
) -> dict[str, Any]:
    worker = QueueWorker()
    worker.connect_database()

    seed_windows = _build_seed_windows(
        worker,
        user_id=user_id,
        policy=seed_policy,
        exclude_track_ids=set(int(track_id) for track_id in holdout_track_ids) | set(int(track_id) for track_id in special_track_ids),
    )
    shortlist_ids = _load_shortlist(
        worker,
        user_id=user_id,
        shortlist_limit=shortlist_limit,
        special_track_ids=special_track_ids,
    )

    liked_embeddings: list[np.ndarray] = []
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT t.embedding_path
            FROM user_track_likes utl
            JOIN tracks t ON t.id = utl.track_id
            WHERE utl.user_id = %s
              AND t.embedding_path IS NOT NULL
            """,
            (int(user_id),),
        )
        for (emb_path,) in cursor.fetchall():
            emb = _load_embedding(emb_path)
            if emb is not None:
                liked_embeddings.append(emb)

        cursor.execute(
            """
            SELECT t.embedding_path
            FROM track_ratings tr
            JOIN tracks t ON t.id = tr.track_id
            WHERE tr.user_id = %s
              AND tr.rating >= 4
              AND t.embedding_path IS NOT NULL
            """,
            (int(user_id),),
        )
        for (emb_path,) in cursor.fetchall():
            emb = _load_embedding(emb_path)
            if emb is not None:
                liked_embeddings.append(emb)

        cursor.execute(
            """
            SELECT pe.embedding_path
            FROM user_pattern_example_ratings upr
            JOIN pattern_examples pe ON pe.id = upr.pattern_example_id
            WHERE upr.user_id = %s
              AND upr.rating >= 4
              AND pe.embedding_path IS NOT NULL
            ORDER BY COALESCE(upr.updated_at, upr.created_at) DESC
            LIMIT 256
            """,
            (int(user_id),),
        )
        for (emb_path,) in cursor.fetchall():
            emb = _load_embedding(emb_path)
            if emb is not None:
                liked_embeddings.append(emb)

    liked_matrix = np.array(liked_embeddings, dtype=np.float32) if liked_embeddings else np.empty((0,), dtype=np.float32)

    all_needed_ids = set(shortlist_ids)
    for seeds in seed_windows.values():
        for seed in seeds:
            all_needed_ids.add(int(seed["track_id"]))

    track_meta: dict[int, dict[str, Any]] = {}
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, title, artist, file_path, embedding_path
            FROM tracks
            WHERE id = ANY(%s)
            """,
            (list(all_needed_ids),),
        )
        for track_id, title, artist, file_path, embedding_path in cursor.fetchall():
            track_meta[int(track_id)] = {
                "track_id": int(track_id),
                "title": title,
                "artist": artist,
                "audio_path": str(Path("/app/storage") / str(file_path)),
                "embedding_path": embedding_path,
            }

    enriched_seed_windows: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    for polarity, seeds in seed_windows.items():
        for seed in seeds:
            meta = track_meta[int(seed["track_id"])]
            enriched_seed_windows[polarity].append({**seed, **meta})

    results: list[dict[str, Any]] = []
    for candidate_id in shortlist_ids:
        meta = track_meta[int(candidate_id)]
        candidate_emb = _load_embedding(meta.get("embedding_path"))
        embedding_score = 0.0
        if candidate_emb is not None and liked_matrix.size:
            sims = liked_matrix @ candidate_emb
            topk = np.sort(sims)[-3:]
            embedding_score = float(np.mean(topk)) if topk.size else 0.0

        candidate_segments = _candidate_segments_for_track(worker, int(candidate_id))
        if not candidate_segments:
            continue

        positive_matches: list[dict[str, Any]] = []
        negative_matches: list[dict[str, Any]] = []
        for segment in candidate_segments:
            for seed in enriched_seed_windows["positive"]:
                comparison = compare_tracks_by_melody_alignment(
                    seed["audio_path"],
                    meta["audio_path"],
                    separator_model="htdemucs_6s",
                    device="cuda",
                    cache_root=cache_root,
                    window_a=(float(seed["start"]), float(seed["end"])),
                    window_b=(float(segment["start"]), float(segment["end"])),
                )
                drop_features = {}
                if str(seed.get("type")) == "drop_bass" and str(segment.get("type")) == "drop_bass":
                    drop_features = _extract_raw_drop_match_features(
                        seed_audio_path=seed["audio_path"],
                        seed_start=float(seed["start"]),
                        seed_end=float(seed["end"]),
                        candidate_audio_path=meta["audio_path"],
                        candidate_start=float(segment["start"]),
                        candidate_end=float(segment["end"]),
                        device="cuda",
                    )
                positive_matches.append(
                    {
                        "similarity": float(comparison["overall_melodic_similarity"]),
                        "effective_similarity": float(drop_features.get("raw_drop_similarity", comparison["overall_melodic_similarity"])),
                        "seed": seed,
                        "candidate_segment": segment,
                        **drop_features,
                    }
                )

            for seed in enriched_seed_windows["negative"]:
                comparison = compare_tracks_by_melody_alignment(
                    seed["audio_path"],
                    meta["audio_path"],
                    separator_model="htdemucs_6s",
                    device="cuda",
                    cache_root=cache_root,
                    window_a=(float(seed["start"]), float(seed["end"])),
                    window_b=(float(segment["start"]), float(segment["end"])),
                )
                drop_features = {}
                if str(seed.get("type")) == "drop_bass" and str(segment.get("type")) == "drop_bass":
                    drop_features = _extract_raw_drop_match_features(
                        seed_audio_path=seed["audio_path"],
                        seed_start=float(seed["start"]),
                        seed_end=float(seed["end"]),
                        candidate_audio_path=meta["audio_path"],
                        candidate_start=float(segment["start"]),
                        candidate_end=float(segment["end"]),
                        device="cuda",
                    )
                negative_matches.append(
                    {
                        "similarity": float(comparison["overall_melodic_similarity"]),
                        "effective_similarity": float(drop_features.get("raw_drop_similarity", comparison["overall_melodic_similarity"])),
                        "seed": seed,
                        "candidate_segment": segment,
                        **drop_features,
                    }
                )

        scoring = score_section_aware_candidate(
            embedding_score=embedding_score,
            positive_matches=positive_matches,
            negative_matches=negative_matches,
        )
        results.append(
            {
                "track_id": int(candidate_id),
                "title": meta["title"],
                "artist": meta["artist"],
                "embedding_score": scoring["embedding_score"],
                "melodic_margin": scoring["melodic_margin"],
                "combined_score": scoring["combined_score"],
                "estimated_like_pct": scoring["estimated_like_pct"],
                "section_scores": scoring["section_scores"],
                "seed_coverage": scoring["seed_coverage"],
                "top_family_matches": scoring["top_family_matches"],
                "best_positive_match": scoring["best_positive_match"],
                "best_negative_match": scoring["best_negative_match"],
                "best_positive_similarity": round(float(scoring["best_positive_match"]["similarity"]), 4) if scoring["best_positive_match"] else 0.0,
                "best_negative_similarity": round(float(scoring["best_negative_match"]["similarity"]), 4) if scoring["best_negative_match"] else 0.0,
                "best_positive_seed": scoring["best_positive_match"]["seed"] if scoring["best_positive_match"] else None,
                "best_negative_seed": scoring["best_negative_match"]["seed"] if scoring["best_negative_match"] else None,
                "best_positive_segment": scoring["best_positive_match"]["candidate_segment"] if scoring["best_positive_match"] else None,
                "best_negative_segment": scoring["best_negative_match"]["candidate_segment"] if scoring["best_negative_match"] else None,
                "same_segment_conflict": scoring["same_segment_conflict"],
                "same_segment_conflict_penalty": scoring["same_segment_conflict_penalty"],
                "is_holdout": int(candidate_id) in {int(track_id) for track_id in holdout_track_ids},
                "candidate_segments_considered": candidate_segments,
            }
        )

    results.sort(key=lambda item: item["combined_score"], reverse=True)
    gate = evaluate_recommendation_lab_gate(
        results,
        holdout_track_ids=[int(track_id) for track_id in holdout_track_ids],
        min_like_pct=55.0,
        conflict_top_rank_limit=int(conflict_top_rank_limit),
    )
    return {
        "user_id": int(user_id),
        "seed_policy": seed_policy,
        "shortlist_limit": int(shortlist_limit),
        "seed_windows": enriched_seed_windows,
        "seed_family_summary": _seed_family_summary(enriched_seed_windows),
        "shortlist_track_ids": shortlist_ids,
        "results": results,
        "gate": gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run section-aware melodic recommendation lab rerank.")
    parser.add_argument("--user-id", dest="user_id", type=int, required=True)
    parser.add_argument(
        "--seed-policy",
        dest="seed_policy",
        required=True,
        choices=["manual_with_explicit_negative_drop", "family_balanced_from_ratings", "family_balanced_discriminative_drop"],
    )
    parser.add_argument("--shortlist-limit", dest="shortlist_limit", type=int, default=12)
    parser.add_argument("--holdout", dest="holdouts", action="append", type=int, required=True)
    parser.add_argument("--special-track", dest="special_tracks", action="append", type=int, default=[])
    parser.add_argument("--conflict-top-rank-limit", dest="conflict_top_rank_limit", type=int, default=3)
    parser.add_argument("--cache-root", dest="cache_root", default="/tmp/codex_recommendation_lab")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_section_aware_lab(
        user_id=int(args.user_id),
        seed_policy=str(args.seed_policy),
        shortlist_limit=int(args.shortlist_limit),
        holdout_track_ids=[int(track_id) for track_id in args.holdouts],
        special_track_ids=[int(track_id) for track_id in args.special_tracks],
        cache_root=Path(args.cache_root),
        conflict_top_rank_limit=int(args.conflict_top_rank_limit),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
