import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import psycopg2

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"faiss is required for this script: {e}")


def _db_connect():
    dbname = os.getenv("DB_NAME") or os.getenv("DB_DATABASE") or "music_recommendations"
    user = os.getenv("DB_USER") or os.getenv("DB_USERNAME") or "postgres"
    password = os.getenv("DB_PASSWORD") or "secret"
    host = os.getenv("DB_HOST") or "postgres"
    port = int(os.getenv("DB_PORT") or 5432)
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def _load_embedding(path: str) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        emb = np.load(str(p)).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb) + 1e-10)
        return (emb / norm).astype(np.float32)
    except Exception:
        return None


def _build_track_index(conn) -> tuple["faiss.Index", List[int], np.ndarray]:
    track_ids: List[int] = []
    vectors: List[np.ndarray] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, embedding_path
            FROM tracks
            WHERE analysis_status = 'completed' AND embedding_path IS NOT NULL
            ORDER BY id
            """
        )
        rows = cur.fetchall()

    for tid, emb_path in rows:
        emb = _load_embedding(emb_path)
        if emb is None:
            continue
        track_ids.append(int(tid))
        vectors.append(emb)

    if not vectors:
        raise RuntimeError("No track embeddings found (tracks.embedding_path missing or files not present)")

    mat = np.vstack(vectors).astype(np.float32)
    dim = int(mat.shape[1])
    idx = faiss.IndexFlatIP(dim)  # cosine for normalized vectors
    idx.add(mat)

    return idx, track_ids, mat


def _pick_seed_ids(conn, n: int) -> List[int]:
    seeds: List[int] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM tracks
            WHERE is_liked = true AND analysis_status = 'completed' AND embedding_path IS NOT NULL
            ORDER BY id
            LIMIT %s
            """,
            (int(n),),
        )
        seeds = [int(r[0]) for r in cur.fetchall()]

    if len(seeds) >= n:
        return seeds[:n]

    # Fill remaining with analyzed tracks.
    remaining = int(n - len(seeds))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM tracks
            WHERE analysis_status = 'completed' AND embedding_path IS NOT NULL AND id <> ALL(%s)
            ORDER BY id
            LIMIT %s
            """,
            (seeds or [0], remaining),
        )
        seeds.extend([int(r[0]) for r in cur.fetchall()])

    return seeds[:n]


def _top_k_neighbors(index: "faiss.Index", track_ids: List[int], vectors: np.ndarray, seed_track_id: int, k: int) -> List[int]:
    try:
        pos = track_ids.index(int(seed_track_id))
    except ValueError:
        raise RuntimeError(f"Seed track {seed_track_id} has no embedding indexed")

    seed_vec = vectors[pos : pos + 1]
    _dist, idxs = index.search(seed_vec, k + 1)

    neighbors: List[int] = []
    for faiss_i in idxs[0].tolist():
        if faiss_i < 0:
            continue
        tid = int(track_ids[int(faiss_i)])
        if tid == int(seed_track_id):
            continue
        neighbors.append(tid)
        if len(neighbors) >= k:
            break
    return neighbors


def main() -> int:
    require_docker_runtime(
        tool_name="build_reco_regression.py",
        entrypoint_path="/app/tools/build_reco_regression.py",
    )
    ap = argparse.ArgumentParser(description="Build a baseline offline regression set for audio similarity recommendations.")
    ap.add_argument("--out", required=True, help="Output path for reco_regression.json")
    ap.add_argument("--num-seeds", type=int, default=8, help="How many seed tracks to include")
    ap.add_argument("--k", type=int, default=25, help="Neighbor list size checked in regression")
    ap.add_argument("--expected", type=int, default=5, help="How many expected similar track IDs to store per seed")
    args = ap.parse_args()

    out_path = Path(args.out)

    conn = _db_connect()
    try:
        index, track_ids, vectors = _build_track_index(conn)
        seeds = _pick_seed_ids(conn, n=int(args.num_seeds))
    finally:
        conn.close()

    cases = []
    for seed_id in seeds:
        neighbors = _top_k_neighbors(index, track_ids, vectors, seed_id, k=int(args.k))
        cases.append(
            {
                "name": f"seed_{seed_id}",
                "seed_track_id": int(seed_id),
                "expected_similar_track_ids": [int(x) for x in neighbors[: int(args.expected)]],
            }
        )

    spec = {
        "k": int(args.k),
        "expected_per_seed": int(args.expected),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": cases,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {len(cases)} cases to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
