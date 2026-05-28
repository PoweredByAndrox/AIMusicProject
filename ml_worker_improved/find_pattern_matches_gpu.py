#!/usr/bin/env python3
"""
GPU-accelerated pattern matching - WITH PROPER LOGGING!
"""

import os
import sys
import time
import psycopg2
from pathlib import Path
from datetime import datetime
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("ПОИСК ПАТТЕРНОВ (GPU)")
print("=" * 60)

# GPU check
try:
    import cupy as cp
    GPU_AVAILABLE = True
    gpu_info = cp.cuda.runtime.getDeviceProperties(0)
    print(f"✓ CuPy: {gpu_info['name'].decode('utf-8')}")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("✗ CuPy недоступен")

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✓ PyTorch CUDA")
except ImportError:
    TORCH_AVAILABLE = False

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
PATTERNS_PATH = STORAGE_PATH / 'patterns'
MIN_MATCH_SCORE = 0.6


def progress_bar(current, total, prefix='', suffix='', width=30):
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} {suffix}", end='')
    if current == total:
        print()


def create_fingerprint(chroma: np.ndarray) -> np.ndarray:
    avg = np.mean(chroma, axis=1)
    var = np.var(chroma, axis=1)
    if chroma.shape[1] > 1:
        diff_var = np.var(np.diff(chroma, axis=1), axis=1)
    else:
        diff_var = np.zeros(12)
    return np.concatenate([avg, var, diff_var]).astype(np.float32)


def gpu_batch_cosine(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if GPU_AVAILABLE:
        q = cp.asarray(queries, dtype=cp.float32)
        t = cp.asarray(targets, dtype=cp.float32)
        q_norm = cp.maximum(cp.linalg.norm(q, axis=1, keepdims=True), 1e-10)
        t_norm = cp.maximum(cp.linalg.norm(t, axis=1, keepdims=True), 1e-10)
        sims = cp.dot(q / q_norm, (t / t_norm).T)
        result = cp.asnumpy(sims)
        del q, t, sims
        cp.get_default_memory_pool().free_all_blocks()
        return result
    else:
        q_norm = np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-10)
        t_norm = np.maximum(np.linalg.norm(targets, axis=1, keepdims=True), 1e-10)
        return np.dot(queries / q_norm, (targets / t_norm).T)


def fast_chroma_sim(c1: np.ndarray, c2: np.ndarray) -> float:
    best = 0.0
    s1 = c1[:, ::8][:, :50]
    s2 = c2[:, ::8][:, :50]
    
    max_len = max(s1.shape[1], s2.shape[1])
    if s1.shape[1] < max_len:
        s1 = np.pad(s1, ((0, 0), (0, max_len - s1.shape[1])))
    if s2.shape[1] < max_len:
        s2 = np.pad(s2, ((0, 0), (0, max_len - s2.shape[1])))
    
    for shift in range(12):
        shifted = np.roll(s1, shift, axis=0)
        dot = np.sum(shifted * s2, axis=0)
        n1 = np.linalg.norm(shifted, axis=0)
        n2 = np.linalg.norm(s2, axis=0)
        valid = (n1 > 1e-10) & (n2 > 1e-10)
        if np.any(valid):
            sim = np.mean(dot[valid] / (n1[valid] * n2[valid]))
            if sim > best:
                best = sim
    return float(best)


# =============================================================================
# STEP 1: Load Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: ЗАГРУЗКА")
print("=" * 60)

load_start = time.time()

# Patterns
cursor.execute("SELECT id, name, importance, match_threshold FROM patterns")
patterns_db = cursor.fetchall()
print(f"Паттернов в БД: {len(patterns_db)}")

pattern_data = {}
pattern_fps = []
pattern_ids = []

print("Загрузка pattern chroma...")
for i, (pid, name, importance, threshold) in enumerate(patterns_db):
    chroma_path = PATTERNS_PATH / f"pattern_{pid}_chroma.npy"
    if chroma_path.exists():
        chroma = np.load(chroma_path).astype(np.float32)
        pattern_data[pid] = {
            'name': name, 'importance': importance,
            'threshold': threshold or MIN_MATCH_SCORE, 'chroma': chroma
        }
        pattern_fps.append(create_fingerprint(chroma))
        pattern_ids.append(pid)
    
    if (i + 1) % 50 == 0:
        progress_bar(i + 1, len(patterns_db), "  Patterns")

print(f"\n  Загружено: {len(pattern_data)}")

if not pattern_data:
    print("ERROR: Нет паттернов!")
    sys.exit(0)

pattern_fp_matrix = np.stack(pattern_fps).astype(np.float32)

# Tracks
cursor.execute("""
    SELECT id, title, artist, chroma_path FROM tracks
    WHERE chroma_path IS NOT NULL AND analysis_status = 'completed'
""")
tracks_db = cursor.fetchall()
print(f"Треков в БД: {len(tracks_db)}")

track_data = {}
track_fps = []
track_ids = []

print("Загрузка track chroma...")
for i, (tid, title, artist, chroma_path) in enumerate(tracks_db):
    if chroma_path and Path(chroma_path).exists():
        chroma = np.load(chroma_path).astype(np.float32)
        track_data[tid] = {'title': title, 'artist': artist, 'chroma': chroma}
        track_fps.append(create_fingerprint(chroma))
        track_ids.append(tid)
    
    if (i + 1) % 200 == 0:
        progress_bar(i + 1, len(tracks_db), "  Tracks")

print(f"\n  Загружено: {len(track_data)}")

track_fp_matrix = np.stack(track_fps).astype(np.float32)

load_time = time.time() - load_start
print(f"Время: {load_time:.1f}s")

# =============================================================================
# STEP 2: GPU Fingerprint Matching
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: GPU FINGERPRINT MATCHING")
print("=" * 60)

fp_start = time.time()

print(f"Вычисление матрицы ({len(pattern_ids)} x {len(track_ids)})...")
all_fp_sims = gpu_batch_cosine(pattern_fp_matrix, track_fp_matrix)
print(f"  Размер: {all_fp_sims.shape}")

fp_time = time.time() - fp_start
print(f"Время: {fp_time:.1f}s")


# =============================================================================
# STEP 3: Chroma Matching
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: CHROMA MATCHING")
print("=" * 60)

chroma_start = time.time()

# Clear old
cursor.execute("DELETE FROM pattern_matches")
conn.commit()
print("Старые matches удалены")

FP_THRESHOLD = 0.65
all_matches = []
chroma_checks = 0

print(f"Порог fingerprint: {FP_THRESHOLD}")
print(f"Порог chroma: {MIN_MATCH_SCORE}")
print()

for p_idx, pid in enumerate(pattern_ids):
    pdata = pattern_data[pid]
    pattern_chroma = pdata['chroma']
    threshold = pdata['threshold']
    
    # Candidates by fingerprint
    fp_sims = all_fp_sims[p_idx]
    candidates = np.where(fp_sims > FP_THRESHOLD)[0]
    
    pattern_matches = 0
    
    for t_idx in candidates:
        tid = track_ids[t_idx]
        track_chroma = track_data[tid]['chroma']
        
        sim = fast_chroma_sim(pattern_chroma, track_chroma)
        chroma_checks += 1
        
        if sim >= threshold:
            all_matches.append((pid, tid, sim))
            pattern_matches += 1
    
    # Progress every pattern
    progress_bar(
        p_idx + 1, len(pattern_ids),
        f"  Pattern {pid:3}",
        f"cand:{len(candidates):4} match:{pattern_matches:3} total:{len(all_matches):6}"
    )

print()
chroma_time = time.time() - chroma_start
print(f"Chroma checks: {chroma_checks}")
print(f"Matches found: {len(all_matches)}")
print(f"Время: {chroma_time:.1f}s")

# =============================================================================
# STEP 4: Save to Database
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: СОХРАНЕНИЕ В БД")
print("=" * 60)

db_start = time.time()

print(f"Вставка {len(all_matches)} записей...")
insert_data = [(pid, tid, score, datetime.now(), datetime.now()) for pid, tid, score in all_matches]

# Batch insert
BATCH_SIZE = 10000
for i in range(0, len(insert_data), BATCH_SIZE):
    batch = insert_data[i:i + BATCH_SIZE]
    cursor.executemany("""
        INSERT INTO pattern_matches (pattern_id, track_id, start_time, end_time, score, confirmed, created_at, updated_at)
        VALUES (%s, %s, 0, 0, %s, NULL, %s, %s)
    """, batch)
    conn.commit()
    progress_bar(min(i + BATCH_SIZE, len(insert_data)), len(insert_data), "  Insert")

print()
db_time = time.time() - db_start
print(f"Время: {db_time:.1f}s")


# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)

# Top patterns
cursor.execute("""
    SELECT p.id, p.name, COUNT(pm.id), AVG(pm.score)
    FROM patterns p
    LEFT JOIN pattern_matches pm ON pm.pattern_id = p.id
    GROUP BY p.id, p.name
    ORDER BY COUNT(pm.id) DESC
    LIMIT 10
""")
print("\n--- ТОП-10 ПАТТЕРНОВ ---")
for pid, name, cnt, avg in cursor.fetchall():
    avg_str = f"{avg:.2f}" if avg else "N/A"
    print(f"  #{pid:3}: {name[:40]} - {cnt} ({avg_str})")

# Top matches
cursor.execute("""
    SELECT p.name, t.artist, t.title, pm.score
    FROM pattern_matches pm
    JOIN patterns p ON p.id = pm.pattern_id
    JOIN tracks t ON t.id = pm.track_id
    ORDER BY pm.score DESC
    LIMIT 10
""")
print("\n--- ТОП-10 MATCHES ---")
for pname, artist, title, score in cursor.fetchall():
    print(f"  [{score:.2f}] {pname[:20]} -> {artist or '?'} - {title[:25]}")

# Score distribution
cursor.execute("""
    SELECT 
        COUNT(*) FILTER (WHERE score >= 0.9),
        COUNT(*) FILTER (WHERE score >= 0.8 AND score < 0.9),
        COUNT(*) FILTER (WHERE score >= 0.7 AND score < 0.8),
        COUNT(*) FILTER (WHERE score >= 0.6 AND score < 0.7)
    FROM pattern_matches
""")
dist = cursor.fetchone()
print("\n--- РАСПРЕДЕЛЕНИЕ ---")
print(f"  0.9+:    {dist[0]}")
print(f"  0.8-0.9: {dist[1]}")
print(f"  0.7-0.8: {dist[2]}")
print(f"  0.6-0.7: {dist[3]}")

# Timing
total_time = time.time() - load_start
print("\n--- ВРЕМЯ ---")
print(f"Загрузка:    {load_time:.1f}s")
print(f"Fingerprint: {fp_time:.1f}s")
print(f"Chroma:      {chroma_time:.1f}s")
print(f"БД:          {db_time:.1f}s")
print(f"ИТОГО:       {total_time:.1f}s")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)

cursor.close()
conn.close()
