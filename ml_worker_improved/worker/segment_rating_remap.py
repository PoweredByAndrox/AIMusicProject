from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


def _segment_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(float(left_end), float(right_end)) - max(float(left_start), float(right_start)))


def _normalize_rating(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_sentiment(rating: float | None, is_positive: Any) -> bool | None:
    if isinstance(is_positive, bool):
        return is_positive
    if rating is None:
        return None
    return bool(rating >= 3.0)


def _collect_overlap_targets(
    old_example: dict[str, Any],
    new_examples: Sequence[dict[str, Any]],
    *,
    min_overlap_seconds: float,
    min_old_fragment_coverage: float,
    min_old_total_coverage: float,
) -> list[dict[str, Any]]:
    old_start = float(old_example["start_time"])
    old_end = float(old_example["end_time"])
    old_duration = max(0.0, old_end - old_start)
    if old_duration <= 1e-6:
        return []

    overlaps: list[dict[str, Any]] = []
    for new_example in new_examples:
        overlap_seconds = _segment_overlap(
            old_start,
            old_end,
            float(new_example["start_time"]),
            float(new_example["end_time"]),
        )
        if overlap_seconds < min_overlap_seconds:
            continue

        old_coverage = overlap_seconds / old_duration
        if old_coverage < min_old_fragment_coverage:
            continue

        new_duration = max(0.0, float(new_example["end_time"]) - float(new_example["start_time"]))
        new_coverage = overlap_seconds / new_duration if new_duration > 1e-6 else 0.0
        overlaps.append(
            {
                "new_example_id": int(new_example["example_id"]),
                "overlap_seconds": float(overlap_seconds),
                "old_coverage": float(old_coverage),
                "new_coverage": float(new_coverage),
            }
        )

    if not overlaps:
        return []

    total_old_coverage = sum(item["overlap_seconds"] for item in overlaps) / old_duration
    if total_old_coverage < min_old_total_coverage:
        return []

    overlaps.sort(key=lambda item: (item["new_example_id"]))
    return overlaps


def _aggregate_group(
    contributions: Sequence[dict[str, Any]],
    *,
    group_label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    ratings = [float(item["rating"]) for item in contributions if item["rating"] is not None]
    if not ratings:
        return None, "missing_rating"

    sentiments = {
        bool(item["is_positive"])
        for item in contributions
        if item["is_positive"] is not None
    }
    if len(sentiments) > 1:
        return None, f"{group_label}_mixed_sentiment"

    if max(ratings) - min(ratings) > 1.0:
        return None, f"{group_label}_rating_conflict"

    total_weight = sum(float(item["weight"]) for item in contributions)
    if total_weight <= 1e-6:
        return None, f"{group_label}_zero_weight"

    weighted_rating = sum(float(item["rating"]) * float(item["weight"]) for item in contributions) / total_weight
    return {
        "rating": round(float(weighted_rating), 2),
        "is_positive": weighted_rating >= 3.0,
        "source_example_ids": sorted({int(item["source_example_id"]) for item in contributions}),
        "weight_total": round(float(total_weight), 4),
    }, None


def build_rating_safe_segment_remap(
    old_examples: Sequence[dict[str, Any]],
    new_examples: Sequence[dict[str, Any]],
    *,
    min_overlap_seconds: float = 2.0,
    min_old_fragment_coverage: float = 0.25,
    min_old_total_coverage: float = 0.85,
) -> dict[str, Any]:
    user_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    legacy_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)

    unmatched_old_examples: list[int] = []
    unmatched_old_examples_with_user_ratings: list[int] = []
    unmatched_old_examples_with_legacy_rating: list[int] = []
    overlap_debug: list[dict[str, Any]] = []
    remapped_old_examples = 0
    user_contributions_total = 0
    legacy_contributions_total = 0
    old_examples_with_user_ratings_total = 0
    old_examples_with_legacy_rating_total = 0
    old_examples_with_user_ratings_remapped = 0
    old_examples_with_legacy_rating_remapped = 0

    for old_example in old_examples:
        old_example_id = int(old_example["example_id"])
        has_user_ratings = bool(old_example.get("user_ratings"))
        has_legacy_rating = _normalize_rating(old_example.get("rating")) is not None
        if has_user_ratings:
            old_examples_with_user_ratings_total += 1
        if has_legacy_rating:
            old_examples_with_legacy_rating_total += 1
        overlap_targets = _collect_overlap_targets(
            old_example,
            new_examples,
            min_overlap_seconds=min_overlap_seconds,
            min_old_fragment_coverage=min_old_fragment_coverage,
            min_old_total_coverage=min_old_total_coverage,
        )
        if not overlap_targets:
            unmatched_old_examples.append(old_example_id)
            if has_user_ratings:
                unmatched_old_examples_with_user_ratings.append(old_example_id)
            if has_legacy_rating:
                unmatched_old_examples_with_legacy_rating.append(old_example_id)
            continue

        remapped_old_examples += 1
        if has_user_ratings:
            old_examples_with_user_ratings_remapped += 1
        if has_legacy_rating:
            old_examples_with_legacy_rating_remapped += 1
        overlap_debug.append(
            {
                "old_example_id": old_example_id,
                "target_ids": [int(item["new_example_id"]) for item in overlap_targets],
                "coverage_sum": round(float(sum(item["old_coverage"] for item in overlap_targets)), 4),
            }
        )

        legacy_rating = _normalize_rating(old_example.get("rating"))
        legacy_sentiment = _derive_sentiment(legacy_rating, old_example.get("is_positive"))
        if legacy_rating is not None:
            for target in overlap_targets:
                legacy_groups[int(target["new_example_id"])].append(
                    {
                        "source_example_id": old_example_id,
                        "rating": legacy_rating,
                        "is_positive": legacy_sentiment,
                        "weight": float(target["overlap_seconds"]),
                    }
                )
                legacy_contributions_total += 1

        for user_rating in old_example.get("user_ratings") or []:
            normalized_rating = _normalize_rating(user_rating.get("rating"))
            normalized_sentiment = _derive_sentiment(normalized_rating, user_rating.get("is_positive"))
            if normalized_rating is None:
                continue
            user_id = int(user_rating["user_id"])
            for target in overlap_targets:
                user_groups[(user_id, int(target["new_example_id"]))].append(
                    {
                        "source_example_id": old_example_id,
                        "rating": normalized_rating,
                        "is_positive": normalized_sentiment,
                        "weight": float(target["overlap_seconds"]),
                    }
                )
                user_contributions_total += 1

    user_updates: list[dict[str, Any]] = []
    user_skipped: list[dict[str, Any]] = []
    for (user_id, new_example_id), contributions in sorted(user_groups.items()):
        aggregated, reason = _aggregate_group(contributions, group_label="user")
        if aggregated is None:
            user_skipped.append(
                {
                    "user_id": int(user_id),
                    "pattern_example_id": int(new_example_id),
                    "reason": reason,
                    "source_example_ids": sorted({int(item["source_example_id"]) for item in contributions}),
                }
            )
            continue
        user_updates.append(
            {
                "user_id": int(user_id),
                "pattern_example_id": int(new_example_id),
                **aggregated,
            }
        )

    legacy_updates: list[dict[str, Any]] = []
    legacy_skipped: list[dict[str, Any]] = []
    for new_example_id, contributions in sorted(legacy_groups.items()):
        aggregated, reason = _aggregate_group(contributions, group_label="legacy")
        if aggregated is None:
            legacy_skipped.append(
                {
                    "pattern_example_id": int(new_example_id),
                    "reason": reason,
                    "source_example_ids": sorted({int(item["source_example_id"]) for item in contributions}),
                }
            )
            continue
        legacy_updates.append(
            {
                "pattern_example_id": int(new_example_id),
                **aggregated,
            }
        )

    return {
        "user_updates": user_updates,
        "legacy_updates": legacy_updates,
        "summary": {
            "old_examples_considered": len(old_examples),
            "old_examples_remapped": remapped_old_examples,
            "old_examples_unmatched": len(unmatched_old_examples),
            "unmatched_old_example_ids": unmatched_old_examples[:24],
            "old_examples_with_user_ratings_total": old_examples_with_user_ratings_total,
            "old_examples_with_user_ratings_remapped": old_examples_with_user_ratings_remapped,
            "old_examples_with_legacy_rating_total": old_examples_with_legacy_rating_total,
            "old_examples_with_legacy_rating_remapped": old_examples_with_legacy_rating_remapped,
            "unmatched_old_examples_with_user_ratings": unmatched_old_examples_with_user_ratings[:24],
            "unmatched_old_examples_with_legacy_rating": unmatched_old_examples_with_legacy_rating[:24],
            "user_contributions_total": user_contributions_total,
            "user_updates_applied": len(user_updates),
            "user_updates_skipped": len(user_skipped),
            "legacy_contributions_total": legacy_contributions_total,
            "legacy_updates_applied": len(legacy_updates),
            "legacy_updates_skipped": len(legacy_skipped),
            "overlap_debug": overlap_debug[:24],
            "user_skipped_debug": user_skipped[:24],
            "legacy_skipped_debug": legacy_skipped[:24],
        },
    }


def evaluate_rating_safe_remap_acceptance(remap_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(remap_payload.get("summary") or {})

    user_total = int(summary.get("old_examples_with_user_ratings_total", 0) or 0)
    user_remapped = int(summary.get("old_examples_with_user_ratings_remapped", 0) or 0)
    user_skipped = int(summary.get("user_updates_skipped", 0) or 0)
    legacy_total = int(summary.get("old_examples_with_legacy_rating_total", 0) or 0)
    legacy_remapped = int(summary.get("old_examples_with_legacy_rating_remapped", 0) or 0)
    legacy_skipped = int(summary.get("legacy_updates_skipped", 0) or 0)

    reasons: list[str] = []
    if user_remapped < user_total:
        reasons.append("user_rated_examples_unmatched")
    if user_skipped > 0:
        reasons.append("user_rating_conflicts")
    if legacy_remapped < legacy_total:
        reasons.append("legacy_rated_examples_unmatched")
    if legacy_skipped > 0:
        reasons.append("legacy_rating_conflicts")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "user_rated_examples_total": user_total,
        "user_rated_examples_remapped": user_remapped,
        "legacy_rated_examples_total": legacy_total,
        "legacy_rated_examples_remapped": legacy_remapped,
        "user_updates_skipped": user_skipped,
        "legacy_updates_skipped": legacy_skipped,
    }
