#!/usr/bin/env python3
"""
GPU-accelerated preference score calculation.
WITH PROPER LOGGING AND PROGRESS!
"""

import os
import sys
import time
import psycopg2
from pathlib import Path
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("РАСЧЁТ ВЕСОВ V3 GPU")
print("=" * 60)

# Check GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
    gpu_info = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = gpu_info['name'].decode('utf-8')
    gpu_mem = gpu_info['totalGlobalMem'] / (1024**3)
    print(f"✓ CuPy: {gpu_name} ({gpu_mem:.1f} GB)")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy недоступен, CPU mode")
    import numpy as cp

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch недоступен")

# Database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

EMBEDDING_WEIGHT = 0.5
CHROMA_WEIGHT = 0.35
PATTERN_WEIGHT = 0.15

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
PATTERNS_PATH = STORAGE_PATH / 'patterns'


def progress_bar(current, total, prefix='', width=40):
    """Print progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({pct*100:.1f}%)", end='')
    if current == total:
        print()


def create_fingerprint(chroma: np.ndarray) -> np.ndarray:
    """Create fingerprint from chroma."""
    avg = np.mean(chroma, axis=1)
    var = np.var(chroma, axis=1)
    if chroma.shape[1] > 1:
        diff_var = np.var(np.diff(chroma, axis=1), axis=1)
    else:
        diff_var = np.zeros(12)
    return np.concatenate([avg, var, diff_var]).astype(np.float32)


def gpu_batch_cosine(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Batch cosine similarity on GPU."""
    if GPU_AVAILABLE:
        q = cp.asarray(queries, dtype=cp.float32)
        t = cp.asarray(targets, dtype=cp.float32)
        
        q_norm = cp.maximum(cp.linalg.norm(q, axis=1, keepdims=True), 1e-10)
        t_norm = cp.maximum(cp.linalg.norm(t, axis=1, keepdims=True), 1e-10)
        
        sims = cp.dot(q / q_norm, (t / t_norm).T)
        return cp.asnumpy(sims)
    else:
        q_norm = np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-10)
        t_norm = np.maximum(np.linalg.norm(targets, axis=1, keepdims=True), 1e-10)
        return np.dot(queries / q_norm, (targets / t_norm).T)


def fast_chroma_similarity(c1: np.ndarray, c2: np.ndarray) -> float:
    """Fast chroma similarity with transpose invariance."""
    best_sim = 0.0
    
    # Heavy subsample for speed
    s1 = c1[:, ::8][:, :50]
    s2 = c2[:, ::8][:, :50]
    
    # Pad to same length
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
            if sim > best_sim:
                best_sim = sim
    
    return float(best_sim)


# Constants for time calculation
HOP_LENGTH = 512
SAMPLE_RATE = 22050
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH  # ~43 frames per second


def find_pattern_in_track(pattern_chroma: np.ndarray, track_chroma: np.ndarray, top_n: int = 3) -> list:
    """
    Find where pattern matches best in track using sliding window.
    Returns list of (score, start_frame, end_frame) for top_n matches.
    Pattern is ~8 sec (~344 frames), we slide it over track.
    """
    pattern_len = pattern_chroma.shape[1]
    track_len = track_chroma.shape[1]
    
    if track_len < pattern_len:
        # Track shorter than pattern - compare directly
        sim = fast_chroma_similarity(pattern_chroma, track_chroma)
        return [(sim, 0, track_len)]
    
    matches = []
    
    # Slide pattern over track with step = pattern_len // 4 (~2 sec)
    step = max(pattern_len // 4, 1)
    
    for start in range(0, track_len - pattern_len + 1, step):
        end = start + pattern_len
        track_segment = track_chroma[:, start:end]
        
        # Quick similarity check
        sim = fast_chroma_similarity(pattern_chroma, track_segment)
        
        if sim >= 0.6:  # Only keep good matches
            matches.append((sim, start, end))
    
    # Sort by score and return top_n non-overlapping
    matches.sort(key=lambda x: x[0], reverse=True)
    
    result = []
    for m in matches:
        # Check no overlap with already selected
        overlap = any(abs(m[1] - r[1]) < pattern_len // 2 for r in result)
        if not overlap:
            result.append(m)
            if len(result) >= top_n:
                break
    
    if not result:
        # Fallback - return best overall
        best_score = 0.0
        best_start = 0
        for start in range(0, track_len - pattern_len + 1, step):
            end = start + pattern_len
            track_segment = track_chroma[:, start:end]
            sim = fast_chroma_similarity(pattern_chroma, track_segment)
            if sim > best_score:
                best_score = sim
                best_start = start
        result = [(best_score, best_start, best_start + pattern_len)]
    
    return result


def frames_to_time(frames: int) -> str:
    """Convert frame index to MM:SS format."""
    seconds = frames / FRAMES_PER_SEC
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


# =============================================================================
# STEP 1: Load Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

load_start = time.time()

# Liked tracks
cursor.execute("""
    SELECT id, title, artist, embedding_path, chroma_path 
    FROM tracks WHERE is_liked = true AND embedding_path IS NOT NULL
""")
liked_tracks = cursor.fetchall()
print(f"Liked треков: {len(liked_tracks)}")

liked_embeddings = {}
liked_chromas = {}
liked_fingerprints = {}
liked_info = {}  # Store title/artist for display

print("Загрузка liked embeddings и chroma...")
for i, (tid, title, artist, emb_path, chroma_path) in enumerate(liked_tracks):
    liked_info[tid] = {'title': title, 'artist': artist}
    
    if emb_path and Path(emb_path).exists():
        liked_embeddings[tid] = np.load(emb_path).astype(np.float32)
    if chroma_path and Path(chroma_path).exists():
        chroma = np.load(chroma_path).astype(np.float32)
        liked_chromas[tid] = chroma
        liked_fingerprints[tid] = create_fingerprint(chroma)
    
    if (i + 1) % 20 == 0:
        progress_bar(i + 1, len(liked_tracks), "  Liked")

print(f"\n  Embeddings: {len(liked_embeddings)}, Chromas: {len(liked_chromas)}")

if not liked_embeddings:
    print("ERROR: Нет liked embeddings!")
    sys.exit(1)

# Prepare liked matrices
liked_ids = list(liked_embeddings.keys())
liked_emb_matrix = np.stack([liked_embeddings[tid] for tid in liked_ids]).astype(np.float32)

if liked_fingerprints:
    liked_fp_ids = list(liked_fingerprints.keys())
    liked_fp_matrix = np.stack([liked_fingerprints[tid] for tid in liked_fp_ids]).astype(np.float32)
else:
    liked_fp_matrix = None

# Patterns
cursor.execute("SELECT id, name, importance FROM patterns")
patterns_db = cursor.fetchall()
print(f"Паттернов: {len(patterns_db)}")

pattern_chromas = {}
pattern_fingerprints = {}

print("Загрузка pattern chroma...")
for pid, name, importance in patterns_db:
    chroma_path = PATTERNS_PATH / f"pattern_{pid}_chroma.npy"
    if chroma_path.exists():
        chroma = np.load(chroma_path).astype(np.float32)
        pattern_chromas[pid] = {'chroma': chroma, 'importance': importance, 'name': name}
        pattern_fingerprints[pid] = create_fingerprint(chroma)

print(f"  Загружено: {len(pattern_chromas)}")

if pattern_fingerprints:
    pattern_fp_ids = list(pattern_fingerprints.keys())
    pattern_fp_matrix = np.stack([pattern_fingerprints[pid] for pid in pattern_fp_ids]).astype(np.float32)
else:
    pattern_fp_matrix = None

# All tracks
cursor.execute("""
    SELECT id, title, artist, genre, bpm, mood, embedding_path, chroma_path, is_liked
    FROM tracks WHERE embedding_path IS NOT NULL AND analysis_status = 'completed'
""")
all_tracks = cursor.fetchall()
print(f"Всего треков: {len(all_tracks)}")

print("Загрузка всех embeddings и chroma...")
track_data = {}
track_emb_list = []
track_ids = []
track_fp_list = []

for i, row in enumerate(all_tracks):
    tid, title, artist, genre, bpm, mood, emb_path, chroma_path, is_liked = row
    
    if not emb_path or not Path(emb_path).exists():
        continue
    
    emb = np.load(emb_path).astype(np.float32)
    track_emb_list.append(emb)
    track_ids.append(tid)
    
    chroma = None
    fp = None
    if chroma_path and Path(chroma_path).exists():
        chroma = np.load(chroma_path).astype(np.float32)
        fp = create_fingerprint(chroma)
        track_fp_list.append(fp)
    else:
        track_fp_list.append(np.zeros(36, dtype=np.float32))
    
    track_data[tid] = {
        'title': title, 'artist': artist, 'genre': genre,
        'bpm': bpm, 'mood': mood, 'is_liked': is_liked, 'chroma': chroma
    }
    
    if (i + 1) % 200 == 0:
        progress_bar(i + 1, len(all_tracks), "  Tracks")

print(f"\n  Загружено: {len(track_ids)} треков")

track_emb_matrix = np.stack(track_emb_list).astype(np.float32)
track_fp_matrix = np.stack(track_fp_list).astype(np.float32)

load_time = time.time() - load_start
print(f"\nВремя загрузки: {load_time:.1f}s")


# =============================================================================
# STEP 2: Embedding Similarity (GPU Batch)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: EMBEDDING SIMILARITY (GPU)")
print("=" * 60)

emb_start = time.time()

# Normalize liked embeddings
print("Нормализация liked embeddings на GPU...")
if GPU_AVAILABLE:
    liked_gpu = cp.asarray(liked_emb_matrix)
    liked_norms = cp.maximum(cp.linalg.norm(liked_gpu, axis=1, keepdims=True), 1e-10)
    liked_emb_norm = cp.asnumpy(liked_gpu / liked_norms)
    del liked_gpu
else:
    liked_norms = np.maximum(np.linalg.norm(liked_emb_matrix, axis=1, keepdims=True), 1e-10)
    liked_emb_norm = liked_emb_matrix / liked_norms

# Normalize all track embeddings
print("Нормализация track embeddings на GPU...")
if GPU_AVAILABLE:
    track_gpu = cp.asarray(track_emb_matrix)
    track_norms = cp.maximum(cp.linalg.norm(track_gpu, axis=1, keepdims=True), 1e-10)
    track_emb_norm = cp.asnumpy(track_gpu / track_norms)
    del track_gpu
else:
    track_norms = np.maximum(np.linalg.norm(track_emb_matrix, axis=1, keepdims=True), 1e-10)
    track_emb_norm = track_emb_matrix / track_norms

# Compute ALL similarities at once
print(f"Вычисление матрицы сходства ({len(track_ids)} x {len(liked_ids)})...")
if GPU_AVAILABLE:
    track_gpu = cp.asarray(track_emb_norm)
    liked_gpu = cp.asarray(liked_emb_norm)
    all_sims = cp.asnumpy(cp.dot(track_gpu, liked_gpu.T))
    del track_gpu, liked_gpu
    cp.get_default_memory_pool().free_all_blocks()
else:
    all_sims = np.dot(track_emb_norm, liked_emb_norm.T)

print(f"  Матрица: {all_sims.shape}")

# Compute embedding scores
print("Расчёт embedding scores...")
emb_scores = {}
for i, tid in enumerate(track_ids):
    sims = all_sims[i]
    sorted_sims = np.sort(sims)[::-1]
    
    max_sim = float(sorted_sims[0])
    top5 = float(np.mean(sorted_sims[:min(5, len(sorted_sims))]))
    top10 = float(np.mean(sorted_sims[:min(10, len(sorted_sims))]))
    
    emb_scores[tid] = 0.5 * max_sim + 0.3 * top5 + 0.2 * top10
    
    if (i + 1) % 500 == 0:
        progress_bar(i + 1, len(track_ids), "  Scores")

print()
emb_time = time.time() - emb_start
print(f"Время: {emb_time:.1f}s")


# =============================================================================
# STEP 3: Chroma Similarity (GPU Fingerprint + Fast Comparison)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: CHROMA SIMILARITY")
print("=" * 60)

chroma_start = time.time()
chroma_scores = {}
chroma_computed = 0

if liked_fp_matrix is not None and len(liked_chromas) > 0:
    # Batch fingerprint similarity
    print(f"GPU fingerprint similarity ({len(track_ids)} x {len(liked_fp_ids)})...")
    all_fp_sims = gpu_batch_cosine(track_fp_matrix, liked_fp_matrix)
    print(f"  Матрица: {all_fp_sims.shape}")
    
    print("Расчёт chroma scores (top-5 candidates per track)...")
    for i, tid in enumerate(track_ids):
        track_chroma = track_data[tid]['chroma']
        
        if track_chroma is None:
            chroma_scores[tid] = 0.0
            continue
        
        # Get top-5 candidates by fingerprint
        fp_sims = all_fp_sims[i]
        top_indices = np.argsort(fp_sims)[::-1][:5]
        
        # Fast chroma comparison for top candidates
        chroma_sims = []
        for idx in top_indices:
            if fp_sims[idx] > 0.5:
                liked_id = liked_fp_ids[idx]
                liked_chroma = liked_chromas[liked_id]
                sim = fast_chroma_similarity(liked_chroma, track_chroma)
                chroma_sims.append(sim)
                chroma_computed += 1
        
        if chroma_sims:
            sorted_sims = sorted(chroma_sims, reverse=True)
            chroma_scores[tid] = 0.6 * sorted_sims[0] + 0.4 * np.mean(sorted_sims[:3])
        else:
            # Fallback to fingerprint
            sorted_fp = np.sort(fp_sims)[::-1]
            chroma_scores[tid] = float(0.6 * sorted_fp[0] + 0.4 * np.mean(sorted_fp[:3]))
        
        if (i + 1) % 200 == 0:
            progress_bar(i + 1, len(track_ids), f"  Chroma (computed: {chroma_computed})")
    
    print()
else:
    print("Нет chroma данных, пропускаем...")
    for tid in track_ids:
        chroma_scores[tid] = 0.0

chroma_time = time.time() - chroma_start
print(f"Chroma comparisons: {chroma_computed}")
print(f"Время: {chroma_time:.1f}s")


# =============================================================================
# STEP 4: Pattern Matching (GPU Fingerprint + Fast Comparison)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: PATTERN MATCHING")
print("=" * 60)

pattern_start = time.time()
pattern_scores = {}
pattern_matches_data = {}
patterns_matched = 0

if pattern_fp_matrix is not None and len(pattern_chromas) > 0:
    # Batch fingerprint similarity with patterns
    print(f"GPU fingerprint similarity ({len(track_ids)} x {len(pattern_fp_ids)})...")
    all_pattern_fp_sims = gpu_batch_cosine(track_fp_matrix, pattern_fp_matrix)
    print(f"  Матрица: {all_pattern_fp_sims.shape}")
    
    FP_THRESHOLD = 0.65
    MATCH_THRESHOLD = 0.70  # Higher threshold for better quality
    
    print("Поиск pattern matches...")
    for i, tid in enumerate(track_ids):
        track_chroma = track_data[tid]['chroma']
        
        if track_chroma is None:
            pattern_scores[tid] = 0.0
            pattern_matches_data[tid] = []
            continue
        
        # Get candidates by fingerprint
        fp_sims = all_pattern_fp_sims[i]
        candidates = np.where(fp_sims > FP_THRESHOLD)[0]
        
        matched = []
        for idx in candidates:
            pid = pattern_fp_ids[idx]
            pdata = pattern_chromas[pid]
            
            # Quick similarity check (no position yet)
            sim = fast_chroma_similarity(pdata['chroma'], track_chroma)
            
            if sim >= MATCH_THRESHOLD:
                matched.append({
                    'pattern_id': pid,
                    'name': pdata['name'],
                    'score': sim,
                    'importance': pdata['importance'],
                    'pattern_chroma': pdata['chroma']  # Save for position lookup later
                })
                patterns_matched += 1
        
        pattern_matches_data[tid] = matched
        
        if matched:
            weighted = [m['score'] * (1 + m['importance'] * 0.2) for m in matched]
            pattern_scores[tid] = min(1.0, max(weighted))
        else:
            pattern_scores[tid] = 0.0
        
        if (i + 1) % 50 == 0 or (i + 1) == len(track_ids):
            progress_bar(i + 1, len(track_ids), f"  Patterns (matches: {patterns_matched})")
    
    print()
else:
    print("Нет pattern данных, пропускаем...")
    for tid in track_ids:
        pattern_scores[tid] = 0.0
        pattern_matches_data[tid] = []

pattern_time = time.time() - pattern_start
print(f"Pattern matches: {patterns_matched}")
print(f"Время: {pattern_time:.1f}s")


# =============================================================================
# STEP 5: Combine Scores & Update Database
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: ОБЪЕДИНЕНИЕ И СОХРАНЕНИЕ")
print("=" * 60)

update_start = time.time()

scores = []
print("Расчёт финальных scores...")

for i, tid in enumerate(track_ids):
    data = track_data[tid]
    
    emb = emb_scores.get(tid, 0.0)
    chroma = chroma_scores.get(tid, 0.0)
    pattern = pattern_scores.get(tid, 0.0)
    matched = pattern_matches_data.get(tid, [])
    
    # Combined score
    if chroma > 0 and pattern > 0:
        final = EMBEDDING_WEIGHT * emb + CHROMA_WEIGHT * chroma + PATTERN_WEIGHT * pattern
    elif chroma > 0:
        final = (EMBEDDING_WEIGHT + PATTERN_WEIGHT/2) * emb + (CHROMA_WEIGHT + PATTERN_WEIGHT/2) * chroma
    else:
        final = emb
    
    scores.append({
        'id': tid,
        'title': data['title'],
        'artist': data['artist'],
        'genre': data['genre'],
        'bpm': data['bpm'],
        'emb_score': emb,
        'chroma_score': chroma,
        'pattern_score': pattern,
        'matched_patterns': matched,
        'score': final,
        'is_liked': data['is_liked'],
    })
    
    if (i + 1) % 500 == 0:
        progress_bar(i + 1, len(track_ids), "  Combine")

print()

# Batch update
print(f"Обновление БД ({len(scores)} записей)...")
batch = [(s['score'], s['id']) for s in scores]

cursor.executemany("""
    UPDATE tracks SET preference_score = %s, score_calculated_at = NOW() WHERE id = %s
""", batch)
conn.commit()

update_time = time.time() - update_start
print(f"Время: {update_time:.1f}s")


# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)

scores.sort(key=lambda x: x['score'], reverse=True)

# Distribution
print("\n--- РАСПРЕДЕЛЕНИЕ SCORES ---")
ranges = {'0.9+': 0, '0.8-0.9': 0, '0.7-0.8': 0, '0.6-0.7': 0, '0.5-0.6': 0,
          '0.4-0.5': 0, '0.3-0.4': 0, '0.2-0.3': 0, '0.1-0.2': 0, '0-0.1': 0}

for s in scores:
    v = s['score']
    if v >= 0.9: ranges['0.9+'] += 1
    elif v >= 0.8: ranges['0.8-0.9'] += 1
    elif v >= 0.7: ranges['0.7-0.8'] += 1
    elif v >= 0.6: ranges['0.6-0.7'] += 1
    elif v >= 0.5: ranges['0.5-0.6'] += 1
    elif v >= 0.4: ranges['0.4-0.5'] += 1
    elif v >= 0.3: ranges['0.3-0.4'] += 1
    elif v >= 0.2: ranges['0.2-0.3'] += 1
    elif v >= 0.1: ranges['0.1-0.2'] += 1
    else: ranges['0-0.1'] += 1

for r, c in ranges.items():
    bar = '█' * min(50, c // 10)
    print(f"  {r:10} {c:5} {bar}")

# Top 20 not liked
print("\n--- ТОП-20 РЕКОМЕНДАЦИЙ ---")
not_liked = [s for s in scores if not s['is_liked']][:20]

import re

def parse_pattern_time(pattern_name: str) -> tuple:
    """Parse time from pattern name like '[auto:drop] Artist - Title (3:51)' -> (231, '3:51')"""
    match = re.search(r'\((\d+):(\d+)\)\s*$', pattern_name)
    if match:
        mins, secs = int(match.group(1)), int(match.group(2))
        total_secs = mins * 60 + secs
        return (total_secs, f"{mins}:{secs:02d}")
    return (0, "?")

def parse_pattern_type(pattern_name: str) -> str:
    """Parse type from pattern name like '[auto:drop_bass]' -> 'drop_bass'"""
    match = re.search(r'\[auto:(\w+)\]', pattern_name)
    if match:
        return match.group(1)
    return "unknown"

def format_time_range(start_secs: float, duration_secs: float) -> str:
    """Format time range like '2:33-2:41'. Duration is REQUIRED, no hardcoded defaults!"""
    end_secs = start_secs + duration_secs
    start_m, start_s = int(start_secs // 60), int(start_secs % 60)
    end_m, end_s = int(end_secs // 60), int(end_secs % 60)
    return f"{start_m}:{start_s:02d}-{end_m}:{end_s:02d}"


def get_pattern_duration_secs(pattern_chroma: np.ndarray) -> float:
    """Calculate actual pattern duration in seconds from chroma shape."""
    return pattern_chroma.shape[1] / FRAMES_PER_SEC

print("Поиск позиций совпадений...")
for i, s in enumerate(not_liked, 1):
    title_short = s['title'][:45] if len(s['title']) > 45 else s['title']
    print(f"\n{i:2}. [{s['score']:.3f}] {s['artist'] or '?'} - {title_short}")
    print(f"    Scores: E={s['emb_score']:.2f} C={s['chroma_score']:.2f} P={s['pattern_score']:.2f}")
    
    if s['matched_patterns']:
        auto_patterns = [p for p in s['matched_patterns'] if '[auto' in p['name']]
        if auto_patterns:
            best = max(auto_patterns, key=lambda x: x['score'])
            
            # Parse liked track info from pattern name
            # Format: "[auto:type] Artist - Title (time)"
            ptype = parse_pattern_type(best['name'])
            pattern_info = re.sub(r'\[auto:\w+\]\s*', '', best['name'])
            liked_time_secs, liked_time_str = parse_pattern_time(best['name'])
            
            # Get ACTUAL pattern duration from chroma shape (NOT hardcoded!)
            pattern_duration_secs = get_pattern_duration_secs(best['pattern_chroma'])
            liked_time_range = format_time_range(liked_time_secs, pattern_duration_secs)
            
            # Extract just artist - title (without time)
            liked_track = re.sub(r'\s*\(\d+:\d+\)\s*$', '', pattern_info)
            
            print(f"    Похож на [{ptype}]: {liked_track}")
            
            # Find positions in recommended track
            track_chroma = track_data[s['id']]['chroma']
            if track_chroma is not None and 'pattern_chroma' in best:
                positions = find_pattern_in_track(best['pattern_chroma'], track_chroma, top_n=3)
                
                # Sort by time and format
                positions_sorted = sorted(positions, key=lambda x: x[1])
                
                for j, (score, start_f, end_f) in enumerate(positions_sorted):
                    if score >= 0.6:
                        start_secs = start_f / FRAMES_PER_SEC
                        # Use ACTUAL matched segment duration (end_f - start_f), NOT hardcoded!
                        matched_duration_secs = (end_f - start_f) / FRAMES_PER_SEC
                        rec_time_range = format_time_range(start_secs, matched_duration_secs)
                        
                        prefix = "└─" if j == len(positions_sorted) - 1 else "├─"
                        print(f"    {prefix} {rec_time_range} ← похоже на {liked_time_range} ({score:.2f})")
        else:
            p = s['matched_patterns'][0]
            print(f"    Pattern: {p['name'][:40]} ({p['score']:.2f})")

# Stats
print("\n--- СТАТИСТИКА ---")
all_scores = [s['score'] for s in scores]
print(f"Средний score: {np.mean(all_scores):.3f} (std: {np.std(all_scores):.3f})")
print(f"Liked avg: {np.mean([s['score'] for s in scores if s['is_liked']]):.3f}")
print(f"Not liked avg: {np.mean([s['score'] for s in scores if not s['is_liked']]):.3f}")
print(f"С chroma: {sum(1 for s in scores if s['chroma_score'] > 0)}")
print(f"С pattern: {sum(1 for s in scores if s['pattern_score'] > 0)}")

# Timing
total_time = time.time() - load_start
print("\n--- ВРЕМЯ ---")
print(f"Загрузка:   {load_time:.1f}s")
print(f"Embedding:  {emb_time:.1f}s")
print(f"Chroma:     {chroma_time:.1f}s")
print(f"Pattern:    {pattern_time:.1f}s")
print(f"Сохранение: {update_time:.1f}s")
print(f"ИТОГО:      {total_time:.1f}s")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)

cursor.close()
conn.close()
