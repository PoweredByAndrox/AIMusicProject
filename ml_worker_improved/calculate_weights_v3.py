#!/usr/bin/env python3
"""
Calculate preference scores using:
1. Audio embeddings (general sound similarity)
2. Chroma features with DTW (melodic/harmonic similarity)
3. Pattern matching (liked fragment patterns)

V3: Uses DTW for more accurate melodic comparison instead of simple fingerprint.
"""

import numpy as np
import psycopg2
from pathlib import Path
import os
from typing import Optional, Tuple, List, Dict

print("=== РАСЧЁТ ВЕСОВ V3 (Embeddings + Chroma DTW + Patterns) ===\n")

# Try to import librosa for DTW
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✓ librosa доступна (DTW enabled)")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("✗ librosa недоступна (fallback to fingerprint)")

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

# Weights for combining scores
EMBEDDING_WEIGHT = 0.5   # General sound similarity
CHROMA_WEIGHT = 0.35     # Melodic/harmonic similarity (DTW)
PATTERN_WEIGHT = 0.15    # Pattern matching bonus

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
PATTERNS_PATH = STORAGE_PATH / 'patterns'


def load_chroma(chroma_path: str) -> Optional[np.ndarray]:
    """Load full chroma matrix."""
    if Path(chroma_path).exists():
        return np.load(chroma_path)
    return None


def compute_dtw_similarity(query: np.ndarray, target: np.ndarray, transpose_invariant: bool = True) -> float:
    """
    Compute DTW-based similarity between two chroma sequences.
    Returns similarity score 0-1 (higher is better).
    """
    if not LIBROSA_AVAILABLE:
        return 0.0
    
    try:
        best_dist = float('inf')
        
        # Subsample for speed (every 4th frame)
        query_sub = query[:, ::4]
        target_sub = target[:, ::4]
        
        # Limit length for speed
        max_frames = 200
        if query_sub.shape[1] > max_frames:
            query_sub = query_sub[:, :max_frames]
        if target_sub.shape[1] > max_frames:
            target_sub = target_sub[:, :max_frames]
        
        shifts = range(12) if transpose_invariant else [0]
        
        for shift in shifts:
            if shift > 0:
                shifted_query = np.roll(query_sub, shift, axis=0)
            else:
                shifted_query = query_sub
            
            # Compute DTW
            D, _ = librosa.sequence.dtw(shifted_query, target_sub, metric='cosine')
            
            # Normalized distance
            dist = D[-1, -1] / (query_sub.shape[1] + target_sub.shape[1])
            
            if dist < best_dist:
                best_dist = dist
        
        # Convert distance to similarity (0-1)
        similarity = max(0, 1 - best_dist * 2)  # Scale factor
        return float(similarity)
        
    except Exception as e:
        return 0.0


def compute_chroma_fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Fallback: simple cosine similarity between fingerprints."""
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(fp1, fp2) / (norm1 * norm2))


def create_fingerprint(chroma: np.ndarray) -> np.ndarray:
    """Create compact fingerprint from chroma."""
    avg = np.mean(chroma, axis=1)
    var = np.var(chroma, axis=1)
    
    # Temporal dynamics (how chroma changes over time)
    diff = np.diff(chroma, axis=1)
    temporal_var = np.var(diff, axis=1) if diff.shape[1] > 0 else np.zeros(12)
    
    # Dominant pitch class
    dominant = np.argmax(avg) / 12.0
    
    return np.concatenate([avg, var, temporal_var, [dominant]]).astype(np.float32)


# Get liked tracks with embeddings AND chroma
cursor.execute("""
    SELECT id, title, artist, embedding_path, chroma_path 
    FROM tracks 
    WHERE is_liked = true 
      AND embedding_path IS NOT NULL
      AND chroma_path IS NOT NULL
""")
liked_tracks = cursor.fetchall()
print(f"Эталонных треков (liked): {len(liked_tracks)}")

# Load liked data
liked_embeddings = {}
liked_chromas = {}
liked_fingerprints = {}

for track_id, title, artist, emb_path, chroma_path in liked_tracks:
    # Load embedding
    if emb_path and Path(emb_path).exists():
        try:
            liked_embeddings[track_id] = np.load(emb_path)
        except Exception as e:
            print(f"  Error loading embedding {track_id}: {e}")
    
    # Load chroma
    if chroma_path:
        try:
            chroma = load_chroma(chroma_path)
            if chroma is not None:
                liked_chromas[track_id] = chroma
                liked_fingerprints[track_id] = create_fingerprint(chroma)
        except Exception as e:
            print(f"  Error loading chroma {track_id}: {e}")

print(f"Загружено embeddings: {len(liked_embeddings)}")
print(f"Загружено chroma: {len(liked_chromas)}")

if not liked_embeddings:
    print("Нет эталонных embeddings!")
    exit(1)

# Load patterns
cursor.execute("""
    SELECT p.id, p.name, p.importance, pe.track_id, pe.start_time, pe.end_time
    FROM patterns p
    JOIN pattern_examples pe ON pe.pattern_id = p.id
    WHERE pe.is_positive = true
""")
patterns = cursor.fetchall()
print(f"Паттернов: {len(patterns)}")

# Load pattern chromas
pattern_chromas = {}
for pattern_id, name, importance, track_id, start_time, end_time in patterns:
    pattern_chroma_path = PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy"
    if pattern_chroma_path.exists():
        try:
            pattern_chromas[pattern_id] = {
                'chroma': np.load(pattern_chroma_path),
                'importance': importance,
                'name': name
            }
        except:
            pass

print(f"Загружено pattern chromas: {len(pattern_chromas)}")

# Prepare matrices for vectorized embedding computation
liked_ids = list(liked_embeddings.keys())
liked_emb_matrix = np.stack([liked_embeddings[tid] for tid in liked_ids])
liked_emb_norms = np.linalg.norm(liked_emb_matrix, axis=1, keepdims=True)
liked_emb_matrix_norm = liked_emb_matrix / (liked_emb_norms + 1e-10)

# Get all tracks
cursor.execute("""
    SELECT id, title, artist, genre, bpm, mood, embedding_path, chroma_path, is_liked, youtube_url
    FROM tracks 
    WHERE embedding_path IS NOT NULL AND analysis_status = 'completed'
""")
all_tracks = cursor.fetchall()
print(f"\nВсего треков для оценки: {len(all_tracks)}\n")

print("Расчёт similarity scores...")

scores = []
processed = 0
dtw_computed = 0

for row in all_tracks:
    track_id, title, artist, genre, bpm, mood, emb_path, chroma_path, is_liked, youtube_url = row
    
    if not emb_path or not Path(emb_path).exists():
        continue
    
    try:
        track_emb = np.load(emb_path)
    except:
        continue
    
    # Normalize embedding
    track_norm = np.linalg.norm(track_emb)
    if track_norm == 0:
        continue
    track_emb_norm = track_emb / track_norm
    
    # === EMBEDDING SIMILARITY ===
    emb_similarities = np.dot(liked_emb_matrix_norm, track_emb_norm)
    sorted_emb_sims = np.sort(emb_similarities)[::-1]
    
    max_emb = float(sorted_emb_sims[0])
    top5_emb = float(np.mean(sorted_emb_sims[:min(5, len(sorted_emb_sims))]))
    top10_emb = float(np.mean(sorted_emb_sims[:min(10, len(sorted_emb_sims))]))
    
    emb_score = 0.5 * max_emb + 0.3 * top5_emb + 0.2 * top10_emb
    
    # === CHROMA SIMILARITY (DTW or fingerprint) ===
    chroma_score = 0.0
    track_chroma = None
    if chroma_path and liked_chromas:
        try:
            track_chroma = load_chroma(chroma_path)
            if track_chroma is not None:
                # First pass: fingerprint similarity to find top candidates
                track_fp = create_fingerprint(track_chroma)
                fp_sims_with_id = []
                for liked_id, liked_fp in liked_fingerprints.items():
                    sim = compute_chroma_fingerprint_similarity(liked_fp, track_fp)
                    fp_sims_with_id.append((liked_id, sim))
                
                fp_sims_with_id.sort(key=lambda x: x[1], reverse=True)
                
                # Second pass: DTW for top-5 candidates (if librosa available)
                if LIBROSA_AVAILABLE and fp_sims_with_id:
                    top_candidates = fp_sims_with_id[:5]
                    dtw_sims = []
                    for liked_id, fp_sim in top_candidates:
                        if fp_sim > 0.5:  # Only compute DTW for good candidates
                            liked_chroma = liked_chromas[liked_id]
                            dtw_sim = compute_dtw_similarity(liked_chroma, track_chroma, transpose_invariant=True)
                            dtw_sims.append(dtw_sim)
                            dtw_computed += 1
                    
                    if dtw_sims:
                        sorted_dtw = sorted(dtw_sims, reverse=True)
                        chroma_score = 0.6 * sorted_dtw[0] + 0.4 * np.mean(sorted_dtw[:min(3, len(sorted_dtw))])
                    else:
                        # Fallback to fingerprint
                        fp_sims = [s for _, s in fp_sims_with_id]
                        chroma_score = 0.6 * fp_sims[0] + 0.4 * np.mean(fp_sims[:min(3, len(fp_sims))])
                else:
                    # Fallback to fingerprint for speed
                    fp_sims = [s for _, s in fp_sims_with_id]
                    if fp_sims:
                        chroma_score = 0.6 * fp_sims[0] + 0.4 * np.mean(fp_sims[:min(3, len(fp_sims))])
        except:
            pass
    
    # === PATTERN MATCHING ===
    pattern_score = 0.0
    matched_patterns = []
    if pattern_chromas and chroma_path:
        try:
            track_chroma = load_chroma(chroma_path) if track_chroma is None else track_chroma
            if track_chroma is not None:
                for pattern_id, pdata in pattern_chromas.items():
                    pattern_chroma = pdata['chroma']
                    importance = pdata['importance']
                    
                    # Quick fingerprint check first
                    pattern_fp = create_fingerprint(pattern_chroma)
                    track_fp = create_fingerprint(track_chroma)
                    fp_sim = compute_chroma_fingerprint_similarity(pattern_fp, track_fp)
                    
                    if fp_sim > 0.7:  # Candidate for DTW
                        if LIBROSA_AVAILABLE:
                            dtw_sim = compute_dtw_similarity(pattern_chroma, track_chroma, transpose_invariant=True)
                            if dtw_sim > 0.6:
                                matched_patterns.append({
                                    'pattern_id': pattern_id,
                                    'name': pdata['name'],
                                    'score': dtw_sim,
                                    'importance': importance
                                })
                        else:
                            if fp_sim > 0.8:
                                matched_patterns.append({
                                    'pattern_id': pattern_id,
                                    'name': pdata['name'],
                                    'score': fp_sim,
                                    'importance': importance
                                })
                
                if matched_patterns:
                    # Weight by importance
                    weighted_scores = [p['score'] * (1 + p['importance'] * 0.2) for p in matched_patterns]
                    pattern_score = min(1.0, max(weighted_scores))
        except:
            pass
    
    # === COMBINED SCORE ===
    if chroma_score > 0 and pattern_score > 0:
        final_score = EMBEDDING_WEIGHT * emb_score + CHROMA_WEIGHT * chroma_score + PATTERN_WEIGHT * pattern_score
    elif chroma_score > 0:
        final_score = (EMBEDDING_WEIGHT + PATTERN_WEIGHT/2) * emb_score + (CHROMA_WEIGHT + PATTERN_WEIGHT/2) * chroma_score
    else:
        final_score = emb_score
    
    # Update database
    cursor.execute("""
        UPDATE tracks 
        SET preference_score = %s, score_calculated_at = NOW() 
        WHERE id = %s
    """, (final_score, track_id))
    
    scores.append({
        'id': track_id,
        'title': title,
        'artist': artist,
        'genre': genre,
        'bpm': bpm,
        'mood': mood,
        'emb_score': emb_score,
        'chroma_score': chroma_score,
        'pattern_score': pattern_score,
        'matched_patterns': matched_patterns,
        'score': final_score,
        'is_liked': is_liked,
        'youtube_url': youtube_url,
    })
    
    processed += 1
    if processed % 100 == 0:
        print(f"  Обработано: {processed} (DTW: {dtw_computed})")
        conn.commit()

conn.commit()
print(f"\nОбработано треков: {processed}")
print(f"DTW вычислений: {dtw_computed}\n")

# Sort by score
scores.sort(key=lambda x: x['score'], reverse=True)

# Show distribution
print("=== РАСПРЕДЕЛЕНИЕ ВЕСОВ ===\n")

ranges = {
    '0.9+': 0, '0.8-0.9': 0, '0.7-0.8': 0, '0.6-0.7': 0, '0.5-0.6': 0,
    '0.4-0.5': 0, '0.3-0.4': 0, '0.2-0.3': 0, '0.1-0.2': 0, '0-0.1': 0,
}

for data in scores:
    s = data['score']
    if s >= 0.9: ranges['0.9+'] += 1
    elif s >= 0.8: ranges['0.8-0.9'] += 1
    elif s >= 0.7: ranges['0.7-0.8'] += 1
    elif s >= 0.6: ranges['0.6-0.7'] += 1
    elif s >= 0.5: ranges['0.5-0.6'] += 1
    elif s >= 0.4: ranges['0.4-0.5'] += 1
    elif s >= 0.3: ranges['0.3-0.4'] += 1
    elif s >= 0.2: ranges['0.2-0.3'] += 1
    elif s >= 0.1: ranges['0.1-0.2'] += 1
    else: ranges['0-0.1'] += 1

for range_name, count in ranges.items():
    bar = '█' * min(50, count // 10)
    print(f"{range_name:10} {count:5} {bar}")

print("\n=== ТОП-20 ТРЕКОВ (не liked) ===\n")

not_liked = [s for s in scores if not s['is_liked']]
top20 = not_liked[:20]

for i, data in enumerate(top20, 1):
    print(f"{i:2}. [{data['score']:.3f}] {data['artist'] or '?'} - {data['title']}")
    print(f"    Emb: {data['emb_score']:.3f} | Chroma: {data['chroma_score']:.3f} | Pattern: {data['pattern_score']:.3f}")
    if data['matched_patterns']:
        patterns_str = ', '.join([f"{p['name'][:20]}({p['score']:.2f})" for p in data['matched_patterns'][:2]])
        print(f"    Паттерны: {patterns_str}")
    print(f"    Жанр: {data['genre']} | BPM: {data['bpm']}")
    print()

# Stats
print("=== СТАТИСТИКА ===")
avg_score = np.mean([s['score'] for s in scores])
liked_avg = np.mean([s['score'] for s in scores if s['is_liked']])
not_liked_avg = np.mean([s['score'] for s in not_liked]) if not_liked else 0

with_chroma = [s for s in scores if s['chroma_score'] > 0]
with_pattern = [s for s in scores if s['pattern_score'] > 0]

print(f"Средний score всех: {avg_score:.3f}")
print(f"Средний score liked: {liked_avg:.3f}")
print(f"Средний score остальных: {not_liked_avg:.3f}")
print(f"Треков с chroma: {len(with_chroma)}")
print(f"Треков с pattern match: {len(with_pattern)}")

cursor.close()
conn.close()
