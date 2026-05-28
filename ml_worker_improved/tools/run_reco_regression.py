import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


@dataclass(frozen=True)
class IndexData:
    index: "faiss.Index"
    track_ids: List[int]
    pos_by_track_id: Dict[int, int]
    vectors: np.ndarray


def _build_track_index(conn) -> IndexData:
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

    pos_by_track_id = {tid: i for i, tid in enumerate(track_ids)}
    return IndexData(index=idx, track_ids=track_ids, pos_by_track_id=pos_by_track_id, vectors=mat)


def _top_k_neighbors(index_data: IndexData, seed_track_id: int, k: int) -> List[int]:
    pos = index_data.pos_by_track_id.get(int(seed_track_id))
    if pos is None:
        raise RuntimeError(f"Seed track {seed_track_id} has no embedding indexed")

    seed_vec = index_data.vectors[pos : pos + 1]
    # Search k+1 to safely drop the seed itself.
    _dist, idxs = index_data.index.search(seed_vec, k + 1)
    neighbors: List[int] = []
    for faiss_i in idxs[0].tolist():
        if faiss_i < 0:
            continue
        tid = int(index_data.track_ids[int(faiss_i)])
        if tid == int(seed_track_id):
            continue
        neighbors.append(tid)
        if len(neighbors) >= k:
            break
    return neighbors


def main() -> int:
    require_docker_runtime(
        tool_name="run_reco_regression.py",
        entrypoint_path="/app/tools/run_reco_regression.py",
    )
    ap = argparse.ArgumentParser(description="Offline recommendation regression based on audio embeddings (track similarity).")
    ap.add_argument("spec", help="Path to reco_regression.json")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise SystemExit(f"Spec file not found: {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    k = int(spec.get("k") or 25)
    cases = spec.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Spec has no cases. Build or fill reco_regression.json first.")

    conn = _db_connect()
    try:
        index_data = _build_track_index(conn)
    finally:
        conn.close()

    failures: List[str] = []
    for c in cases:
        name = c.get("name") or f"seed_{c.get('seed_track_id')}"
        seed_id = int(c["seed_track_id"])
        expected = [int(x) for x in (c.get("expected_similar_track_ids") or [])]
        if not expected:
            failures.append(f"{name}: empty expected_similar_track_ids")
            continue

        try:
            neighbors = _top_k_neighbors(index_data, seed_id, k=k)
        except Exception as e:
            failures.append(f"{name}: failed to compute neighbors: {e}")
            continue

        missing = [tid for tid in expected if tid not in neighbors]
        if missing:
            failures.append(
                f"{name}: missing {missing} in top-{k} (got top-5={neighbors[:5]})"
            )

    if failures:
        print("FAIL")
        for f in failures:
            print(f"- {f}")
        return 2

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
