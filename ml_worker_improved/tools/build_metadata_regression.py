import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import psycopg2


def _db_connect():
    dbname = os.getenv("DB_NAME") or os.getenv("DB_DATABASE") or "music_recommendations"
    user = os.getenv("DB_USER") or os.getenv("DB_USERNAME") or "postgres"
    password = os.getenv("DB_PASSWORD") or "secret"
    host = os.getenv("DB_HOST") or "postgres"
    port = int(os.getenv("DB_PORT") or 5432)
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def _fetch_cases(conn, limit: int) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH rating_stats AS (
                SELECT
                    track_id,
                    MAX(rating) AS max_rating,
                    MIN(rating) AS min_rating
                FROM track_ratings
                GROUP BY track_id
            )
            SELECT
                t.id,
                t.title,
                COALESCE(t.artist, ''),
                t.file_path,
                COALESCE(t.genre, 'Unknown'),
                COALESCE(t.mood, 'Unknown'),
                COALESCE(t.has_vocals, false),
                COALESCE(t.is_liked, false),
                rs.max_rating,
                rs.min_rating
            FROM tracks t
            LEFT JOIN rating_stats rs ON rs.track_id = t.id
            WHERE t.file_path IS NOT NULL
              AND t.analysis_status IN ('completed', 'pending', 'processing')
            ORDER BY
                CASE
                    WHEN COALESCE(rs.max_rating, 0) >= 4 THEN 0
                    WHEN t.is_liked = true THEN 1
                    WHEN COALESCE(rs.min_rating, 6) <= 2 THEN 2
                    ELSE 3
                END,
                COALESCE(t.preference_score, 0) DESC,
                t.id
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cur.fetchall()

    cases: List[Dict[str, Any]] = []
    for row in rows:
        (
            track_id,
            title,
            artist,
            file_path,
            genre,
            mood,
            has_vocals,
            is_liked,
            max_rating,
            min_rating,
        ) = row
        cases.append(
            {
                "name": f"track_{int(track_id)}",
                "track_id": int(track_id),
                "title": str(title),
                "artist": str(artist),
                "audio_path": f"/app/storage/{file_path}",
                "feedback_context": {
                    "is_liked": bool(is_liked),
                    "max_rating": int(max_rating) if max_rating is not None else None,
                    "min_rating": int(min_rating) if min_rating is not None else None,
                },
                "predicted_snapshot": {
                    "genre": str(genre),
                    "mood": str(mood),
                    "has_vocals": bool(has_vocals),
                },
                "expected": {
                    "genre": None,
                    "mood": None,
                    "has_vocals": None,
                },
                "notes": "",
            }
        )
    return cases


def main() -> int:
    require_docker_runtime(
        tool_name="build_metadata_regression.py",
        entrypoint_path="/app/tools/build_metadata_regression.py",
    )
    ap = argparse.ArgumentParser(description="Build a small manual-annotation set for track metadata regression.")
    ap.add_argument("--out", required=True, help="Output path for metadata_regression.json")
    ap.add_argument("--limit", type=int, default=24, help="How many candidate tracks to include")
    args = ap.parse_args()

    conn = _db_connect()
    try:
        cases = _fetch_cases(conn, limit=int(args.limit))
    finally:
        conn.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "annotated_fields": ["genre", "mood", "has_vocals"],
        "cases": cases,
    }
    out_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
