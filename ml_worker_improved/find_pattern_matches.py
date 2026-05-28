#!/usr/bin/env python3
"""
Find pattern matches across all tracks.
Uses DTW to compare pattern chromas with track chromas.
Stores matches in pattern_matches table.
"""

import os
import sys
import psycopg2
from pathlib import Path
from datetime import datetime
import numpy as np

print("=== ПОИСК СОВПАДЕНИЙ ПАТТЕРНОВ ===\n")

# Try to import librosa for DTW
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✓ librosa доступна (DTW enabled)")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("✗ librosa недоступна")
    sys.exit(1)

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
PATTERNS_PATH = STORAGE_PATH / 'patterns'

# DTW parameters
MIN_MATCH_SCORE = 0.6  # Minimum score to consider a match
SUBSAMPLE_RATE = 4     # Subsample chroma for speed
MAX_FRAMES = 200       # Max frames for DTW


def load_chroma(path: str) -> np.ndarray:
    """Load chroma from file."""
    if Path(path).exists():
        return np.load(path)
    return None


def compute_dtw_similarity(query: np.ndarray, target: np.ndarray, transpose_invariant: bool = True) -> float:
    """Compute DTW-based similarity between two chroma sequences."""
    try:
        best_dist = float('inf')
        
        # Subsample for speed
        query_sub = query[:, ::SUBSAMPLE_RATE]
        target_sub = target[:, ::SUBSAMPLE_RATE]
        
        # Limit length
        if query_sub.shape[1] > MAX_FRAMES:
            query_sub = query_sub[:, :MAX_FRAMES]
        if target_sub.shape[1] > MAX_FRAMES:
            target_sub = target_sub[:, :MAX_FRAMES]
        
        shifts = range(12) if transpose_invariant else [0]
        
        for shift in shifts:
            if shift > 0:
                shifted_query = np.roll(query_sub, shift, axis=0)
            else:
                shifted_query = query_sub
            
            D, _ = librosa.sequence.dtw(shifted_query, target_sub, metric='cosine')
            dist = D[-1, -1] / (query_sub.shape[1] + target_sub.shape[1])
            
            if dist < best_dist:
                best_dist = dist
        
        similarity = max(0, 1 - best_dist * 2)
        return float(similarity)
        
    except Exception as e:
        return 0.0


def create_fingerprint(chroma: np.ndarray) -> np.ndarray:
    """Create compact fingerprint from chroma."""
    avg = np.mean(chroma, axis=1)
    var = np.var(chroma, axis=1)
    dominant = np.argmax(avg) / 12.0
    return np.concatenate([avg, var, [dominant]]).astype(np.float32)


def fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Quick fingerprint similarity for candidate filtering."""
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(fp1, fp2) / (norm1 * norm2))


# Load patterns
cursor.execute("""
    SELECT p.id, p.name, p.importance, p.match_threshold, p.transpose_invariant
    FROM patterns p
    ORDER BY p.id
""")
patterns = cursor.fetchall()
print(f"Паттернов: {len(patterns)}")

# Load pattern chromas
pattern_data = {}
for pattern_id, name, importance, threshold, transpose_inv in patterns:
    chroma_path = PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy"
    if chroma_path.exists():
        chroma = np.load(chroma_path)
        pattern_data[pattern_id] = {
            'name': name,
            'importance': importance,
            'threshold': threshold or MIN_MATCH_SCORE,
            'transpose_invariant': transpose_inv if transpose_inv is not None else True,
            'chroma': chroma,
            'fingerprint': create_fingerprint(chroma)
        }
        print(f"  #{pattern_id}: {name} (chroma shape: {chroma.shape})")

if not pattern_data:
    print("Нет паттернов с chroma!")
    sys.exit(0)

# Get all tracks with chroma
cursor.execute("""
    SELECT id, title, artist, chroma_path
    FROM tracks
    WHERE chroma_path IS NOT NULL
      AND analysis_status = 'completed'
    ORDER BY id
""")
tracks = cursor.fetchall()
print(f"\nТреков с chroma: {len(tracks)}")

# Clear old matches
cursor.execute("DELETE FROM pattern_matches")
conn.commit()
print("Очищены старые matches\n")

total_matches = 0
processed = 0

for track_id, title, artist, chroma_path in tracks:
    track_chroma = load_chroma(chroma_path)
    if track_chroma is None:
        continue
    
    track_fp = create_fingerprint(track_chroma)
    
    for pattern_id, pdata in pattern_data.items():
        # Quick fingerprint check
        fp_sim = fingerprint_similarity(pdata['fingerprint'], track_fp)
        
        if fp_sim < 0.5:  # Skip if fingerprint too different
            continue
        
        # Full DTW comparison
        dtw_sim = compute_dtw_similarity(
            pdata['chroma'], 
            track_chroma,
            transpose_invariant=pdata['transpose_invariant']
        )
        
        if dtw_sim >= pdata['threshold']:
            # Found a match!
            cursor.execute("""
                INSERT INTO pattern_matches (pattern_id, track_id, start_time, end_time, score, confirmed, created_at, updated_at)
                VALUES (%s, %s, 0, 0, %s, NULL, %s, %s)
            """, (pattern_id, track_id, dtw_sim, datetime.now(), datetime.now()))
            total_matches += 1
    
    processed += 1
    if processed % 100 == 0:
        conn.commit()
        print(f"  Обработано: {processed}/{len(tracks)}, matches: {total_matches}")

conn.commit()

print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Обработано треков: {processed}")
print(f"Найдено совпадений: {total_matches}")

# Show matches per pattern
cursor.execute("""
    SELECT p.id, p.name, COUNT(pm.id) as match_count, AVG(pm.score) as avg_score
    FROM patterns p
    LEFT JOIN pattern_matches pm ON pm.pattern_id = p.id
    GROUP BY p.id, p.name
    ORDER BY match_count DESC
""")
pattern_stats = cursor.fetchall()

print("\n=== СОВПАДЕНИЯ ПО ПАТТЕРНАМ ===")
for pid, name, count, avg_score in pattern_stats:
    avg_str = f"{avg_score:.2f}" if avg_score else "N/A"
    print(f"  #{pid}: {name[:40]} - {count} matches (avg: {avg_str})")

# Show top matches
cursor.execute("""
    SELECT pm.pattern_id, p.name, pm.track_id, t.title, t.artist, pm.score
    FROM pattern_matches pm
    JOIN patterns p ON p.id = pm.pattern_id
    JOIN tracks t ON t.id = pm.track_id
    ORDER BY pm.score DESC
    LIMIT 20
""")
top_matches = cursor.fetchall()

print("\n=== ТОП-20 СОВПАДЕНИЙ ===")
for pid, pname, tid, title, artist, score in top_matches:
    print(f"  [{score:.2f}] Pattern '{pname[:25]}' -> {artist or '?'} - {title[:30]}")

cursor.close()
conn.close()
