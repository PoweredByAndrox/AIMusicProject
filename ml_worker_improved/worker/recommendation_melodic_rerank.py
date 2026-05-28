from __future__ import annotations

from typing import Any, Mapping, Sequence


HOOK_SECTION_TYPES = {"verse", "melodic"}
DROP_SECTION_TYPES = {"drop_bass"}
SECTION_WEIGHTS = {"hook": 0.6, "drop": 0.4}
NEGATIVE_PENALTIES = {"hook": 0.72, "drop": 0.72}
DEFAULT_EMBEDDING_WEIGHT = 0.38
DEFAULT_CONFLICT_PENALTY = 0.06
DEFAULT_LIKE_PCT_SLOPE = 194.44
DEFAULT_LIKE_PCT_OFFSET = -47.22


def classify_section_family(section_type: Any) -> str:
    value = str(section_type or "").strip().lower()
    if value in HOOK_SECTION_TYPES:
        return "hook"
    if value in DROP_SECTION_TYPES:
        return "drop"
    return "other"


def _normalize_segment(segment: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(segment, Mapping):
        return None
    try:
        start = round(float(segment["start"]), 3)
        end = round(float(segment["end"]), 3)
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "start": start,
        "end": end,
        "type": str(segment.get("type") or "unknown"),
    }


def _segment_key(segment: Mapping[str, Any] | None) -> tuple[float, float, str] | None:
    normalized = _normalize_segment(segment)
    if normalized is None:
        return None
    return (
        float(normalized["start"]),
        float(normalized["end"]),
        str(normalized["type"]).strip().lower(),
    )


def _seed_key(seed: Mapping[str, Any] | None) -> tuple[int | None, float | None, float | None, str | None] | None:
    if not isinstance(seed, Mapping):
        return None
    try:
        track_id = int(seed.get("track_id")) if seed.get("track_id") is not None else None
        start = round(float(seed.get("start")), 3)
        end = round(float(seed.get("end")), 3)
    except (TypeError, ValueError):
        return None
    return (
        track_id,
        start,
        end,
        str(seed.get("type") or "unknown").strip().lower(),
    )


def _normalize_match(match: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(match, Mapping):
        return None
    seed = match.get("seed")
    candidate_segment = match.get("candidate_segment")
    normalized_segment = _normalize_segment(candidate_segment)
    if normalized_segment is None:
        return None

    try:
        similarity = float(match.get("similarity", 0.0) or 0.0)
    except (TypeError, ValueError):
        similarity = 0.0
    try:
        effective_similarity = float(match.get("effective_similarity", similarity) or similarity)
    except (TypeError, ValueError):
        effective_similarity = similarity

    seed_type = None
    if isinstance(seed, Mapping):
        seed_type = seed.get("type")

    return {
        "similarity": similarity,
        "effective_similarity": effective_similarity,
        "seed": dict(seed) if isinstance(seed, Mapping) else None,
        "seed_type": str(seed_type or "unknown"),
        "seed_family": classify_section_family(seed_type),
        "candidate_segment": normalized_segment,
        "candidate_family": classify_section_family(normalized_segment["type"]),
    }


def _best_match_by_family(matches: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for raw_match in matches:
        match = _normalize_match(raw_match)
        if match is None:
            continue
        family = match["candidate_family"]
        if family == "other" or family != match["seed_family"]:
            continue
        if family not in best or float(match["effective_similarity"]) > float(best[family]["effective_similarity"]):
            best[family] = match
    return best


def summarize_family_seed_coverage(
    matches: Sequence[Mapping[str, Any]],
    *,
    similarity_threshold: float = 0.55,
) -> dict[str, dict[str, Any]]:
    families = ("hook", "drop")
    coverage = {
        family: {
            "seed_count": 0,
            "matched_seed_count": 0,
            "coverage_ratio": 0.0,
            "best_similarity": 0.0,
        }
        for family in families
    }
    seen: dict[str, set[tuple[int | None, float | None, float | None, str | None]]] = {family: set() for family in families}
    matched: dict[str, set[tuple[int | None, float | None, float | None, str | None]]] = {family: set() for family in families}

    for raw_match in matches:
        match = _normalize_match(raw_match)
        if match is None:
            continue
        family = match["seed_family"]
        if family not in coverage or family != match["candidate_family"]:
            continue
        seed_key = _seed_key(match.get("seed"))
        if seed_key is None:
            continue
        seen[family].add(seed_key)
        coverage[family]["best_similarity"] = round(
            max(float(coverage[family]["best_similarity"]), float(match["effective_similarity"])),
            4,
        )
        if float(match["effective_similarity"]) >= float(similarity_threshold):
            matched[family].add(seed_key)

    for family in families:
        seed_count = len(seen[family])
        matched_count = len(matched[family])
        coverage[family]["seed_count"] = seed_count
        coverage[family]["matched_seed_count"] = matched_count
        coverage[family]["coverage_ratio"] = round((matched_count / seed_count), 4) if seed_count else 0.0

    return coverage


def summarize_top_family_matches(
    matches: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    families = ("hook", "drop")
    ranked: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    normalized_matches: list[dict[str, Any]] = []

    for raw_match in matches:
        match = _normalize_match(raw_match)
        if match is None:
            continue
        family = match["seed_family"]
        if family not in ranked or family != match["candidate_family"]:
            continue
        normalized_matches.append(match)

    for family in families:
        family_matches = [match for match in normalized_matches if match["seed_family"] == family]
        family_matches.sort(key=lambda item: float(item["effective_similarity"]), reverse=True)
        ranked[family] = [
            {
                "similarity": round(float(match["similarity"]), 4),
                "effective_similarity": round(float(match["effective_similarity"]), 4),
                "seed": match["seed"],
                "candidate_segment": match["candidate_segment"],
            }
            for match in family_matches[: int(top_k)]
        ]

    return ranked


def has_same_candidate_segment_conflict(
    positive_match: Mapping[str, Any] | None,
    negative_match: Mapping[str, Any] | None,
) -> bool:
    left = _segment_key(positive_match.get("candidate_segment") if isinstance(positive_match, Mapping) else None)
    right = _segment_key(negative_match.get("candidate_segment") if isinstance(negative_match, Mapping) else None)
    return left is not None and right is not None and left == right


def score_section_aware_candidate(
    *,
    embedding_score: float,
    positive_matches: Sequence[Mapping[str, Any]],
    negative_matches: Sequence[Mapping[str, Any]],
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    conflict_penalty: float = DEFAULT_CONFLICT_PENALTY,
    like_pct_slope: float = DEFAULT_LIKE_PCT_SLOPE,
    like_pct_offset: float = DEFAULT_LIKE_PCT_OFFSET,
) -> dict[str, Any]:
    positive_by_family = _best_match_by_family(positive_matches)
    negative_by_family = _best_match_by_family(negative_matches)
    seed_coverage = {
        "positive": summarize_family_seed_coverage(positive_matches),
        "negative": summarize_family_seed_coverage(negative_matches),
    }
    top_family_matches = {
        "positive": summarize_top_family_matches(positive_matches),
        "negative": summarize_top_family_matches(negative_matches),
    }

    family_scores: dict[str, dict[str, Any]] = {}
    melodic_margin = 0.0
    for family, weight in SECTION_WEIGHTS.items():
        positive_match = positive_by_family.get(family)
        negative_match = negative_by_family.get(family)
        positive_similarity = float(positive_match["effective_similarity"]) if positive_match else 0.0
        negative_similarity = float(negative_match["effective_similarity"]) if negative_match else 0.0
        margin = positive_similarity - NEGATIVE_PENALTIES[family] * negative_similarity
        melodic_margin += weight * margin
        family_scores[family] = {
            "weight": round(float(weight), 4),
            "positive_similarity": round(positive_similarity, 4),
            "negative_similarity": round(negative_similarity, 4),
            "margin": round(float(margin), 4),
            "best_positive_match": positive_match,
            "best_negative_match": negative_match,
        }

    best_positive_match = max(
        positive_by_family.values(),
        key=lambda item: float(item["similarity"]),
        default=None,
    )
    best_negative_match = max(
        negative_by_family.values(),
        key=lambda item: float(item["similarity"]),
        default=None,
    )

    same_segment_conflict = has_same_candidate_segment_conflict(best_positive_match, best_negative_match)
    applied_conflict_penalty = float(conflict_penalty) if same_segment_conflict else 0.0

    combined_score = (
        float(embedding_weight) * float(embedding_score)
        + (1.0 - float(embedding_weight)) * float(melodic_margin)
        - applied_conflict_penalty
    )
    estimated_like_pct = max(
        0.0,
        min(100.0, float(like_pct_slope) * float(combined_score) + float(like_pct_offset)),
    )

    return {
        "embedding_score": round(float(embedding_score), 4),
        "melodic_margin": round(float(melodic_margin), 4),
        "combined_score": round(float(combined_score), 4),
        "estimated_like_pct": round(float(estimated_like_pct), 1),
        "section_scores": family_scores,
        "best_positive_match": best_positive_match,
        "best_negative_match": best_negative_match,
        "seed_coverage": seed_coverage,
        "top_family_matches": top_family_matches,
        "same_segment_conflict": bool(same_segment_conflict),
        "same_segment_conflict_penalty": round(float(applied_conflict_penalty), 4),
    }
