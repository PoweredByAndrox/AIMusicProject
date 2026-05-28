from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def load_recommendation_results(artifact_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(artifact_path).read_text(encoding='utf-8-sig'))
    if isinstance(payload, dict):
        results = payload.get('results') or []
        return [dict(item) for item in results if isinstance(item, dict)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    return []


def evaluate_holdout_gate(
    results: Sequence[dict[str, Any]],
    *,
    holdout_track_ids: Sequence[int],
    min_like_pct: float,
) -> dict[str, Any]:
    by_track_id: dict[int, dict[str, Any]] = {}
    for item in results:
        try:
            by_track_id[int(item['track_id'])] = dict(item)
        except Exception:
            continue

    passing: list[dict[str, Any]] = []
    failing: list[dict[str, Any]] = []
    missing: list[int] = []

    for holdout_track_id in holdout_track_ids:
        candidate = by_track_id.get(int(holdout_track_id))
        if candidate is None:
            missing.append(int(holdout_track_id))
            continue

        like_pct = float(candidate.get('estimated_like_pct', 0.0) or 0.0)
        entry = {
            'track_id': int(holdout_track_id),
            'title': candidate.get('title'),
            'estimated_like_pct': round(like_pct, 4),
            'combined_score': float(candidate.get('combined_score', 0.0) or 0.0),
        }
        if like_pct >= float(min_like_pct):
            passing.append(entry)
        else:
            failing.append(entry)

    return {
        'gate_passed': not missing and not failing,
        'min_like_pct': float(min_like_pct),
        'holdout_track_ids': [int(track_id) for track_id in holdout_track_ids],
        'passing_holdouts': passing,
        'failing_holdouts': failing,
        'missing_holdouts': missing,
    }


def evaluate_same_segment_conflict_gate(
    results: Sequence[dict[str, Any]],
    *,
    top_rank_limit: int,
    min_like_pct: float,
) -> dict[str, Any]:
    failing: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        if rank > int(top_rank_limit):
            break
        if not bool(item.get('same_segment_conflict')):
            continue

        like_pct = float(item.get('estimated_like_pct', 0.0) or 0.0)
        if like_pct < float(min_like_pct):
            continue

        failing.append(
            {
                'rank': int(rank),
                'track_id': int(item['track_id']),
                'title': item.get('title'),
                'estimated_like_pct': round(like_pct, 4),
                'combined_score': float(item.get('combined_score', 0.0) or 0.0),
                'best_positive_segment': item.get('best_positive_segment'),
                'best_negative_segment': item.get('best_negative_segment'),
            }
        )

    return {
        'gate_passed': not failing,
        'top_rank_limit': int(top_rank_limit),
        'min_like_pct': float(min_like_pct),
        'failing_candidates': failing,
    }


def evaluate_recommendation_lab_gate(
    results: Sequence[dict[str, Any]],
    *,
    holdout_track_ids: Sequence[int],
    min_like_pct: float,
    conflict_top_rank_limit: int,
) -> dict[str, Any]:
    holdout_gate = evaluate_holdout_gate(
        results,
        holdout_track_ids=holdout_track_ids,
        min_like_pct=min_like_pct,
    )
    conflict_gate = evaluate_same_segment_conflict_gate(
        results,
        top_rank_limit=conflict_top_rank_limit,
        min_like_pct=min_like_pct,
    )
    return {
        'gate_passed': bool(holdout_gate['gate_passed'] and conflict_gate['gate_passed']),
        'holdout_gate': holdout_gate,
        'same_segment_conflict_gate': conflict_gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Reject recommendation lab run when liked holdouts miss threshold.')
    parser.add_argument('artifact_path', help='Path to recommendation_rerank JSON artifact')
    parser.add_argument('--holdout', dest='holdouts', action='append', required=True, type=int, help='Track id that must clear the threshold')
    parser.add_argument('--min-like-pct', dest='min_like_pct', required=True, type=float, help='Minimum estimated_like_pct required for every holdout')
    parser.add_argument(
        '--conflict-top-rank-limit',
        dest='conflict_top_rank_limit',
        type=int,
        default=0,
        help='Reject run when a high-ranked candidate has same positive/negative candidate segment latch',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = load_recommendation_results(args.artifact_path)
    if int(args.conflict_top_rank_limit or 0) > 0:
        summary = evaluate_recommendation_lab_gate(
            results,
            holdout_track_ids=args.holdouts,
            min_like_pct=args.min_like_pct,
            conflict_top_rank_limit=int(args.conflict_top_rank_limit),
        )
    else:
        summary = evaluate_holdout_gate(
            results,
            holdout_track_ids=args.holdouts,
            min_like_pct=args.min_like_pct,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary['gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
