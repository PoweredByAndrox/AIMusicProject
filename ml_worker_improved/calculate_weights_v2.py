#!/usr/bin/env python3
"""
Calculate preference scores using BOTH:
1. Audio embeddings (general sound similarity)
2. Chroma features (melodic/harmonic similarity)

This gives better recommendations based on actual musical content.
"""

import numpy as np
import psycopg2
from pathlib import Path
import os

print("=== РАСЧЁТ ВЕСОВ V2 (Embeddings + Chroma) ===\n")

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

# Weights for combining scores
EMBEDDING_WEIGHT = 0.6  # General sound similarity
CHROMA_WEIGHT = 0.4     # Melodic/harmonic similarity


def load_chroma_fingerprint(chroma_path: str) -> np.ndarray:
    """Load chroma fingerprint (25-dim vector)."""
    # Fingerprint is stored alongside chroma
    fp_path = chroma_path.replace('chroma_', 'chroma_fp_')
    if Path(fp_path).exists():
        return np.load(fp_path)
    
    # Fallback: compute from full chroma
    if Path(chroma_path).exists():
        chroma = np.load(chroma_path)
        avg = np.mean(chroma, axis=1)
        var = np.var(chroma, axis=1)
        dominant = np.argmax(avg) / 12.0
        return np.concatenate([avg, var, [dominant]]).astype(np.float32)
    
    return None


# Get liked tracks with embeddings AND chroma
cursor.execute("""
    SELECT id, title, artist, embedding_path, chroma_path 
    FROM tracks 
    WHERE is_liked = true 
      AND embedding_path IS NOT NULL
""")
liked_tracks = cursor.fetchall()
print(f"Эталонных треков (liked): {len(liked_tracks)}")

# Load liked embeddings
liked_embeddings = {}
liked_chroma_fps = {}

for track_id, title, artist, emb_path, chroma_path in liked_tracks:
    # Load embedding
    if emb_path and Path(emb_path).exists():
        try:
            liked_embeddings[track_id] = np.load(emb_path)
        except Exception as e:
            print(f"  Error loading embedding {track_id}: {e}")
    
    # Load chroma fingerprint
    if chroma_path:
        try:
            fp = load_chroma_fingerprint(chroma_path)
            if fp is not None:
                liked_chroma_fps[track_id] = fp
        except Exception as e:
            print(f"  Error loading chroma {track_id}: {e}")

print(f"Загружено embeddings: {len(liked_embeddings)}")
print(f"Загружено chroma fingerprints: {len(liked_chroma_fps)}")

if not liked_embeddings:
    print("Нет эталонных embeddings!")
    exit(1)

# Prepare matrices for vectorized computation
liked_ids = list(liked_embeddings.keys())
liked_emb_matrix = np.stack([liked_embeddings[tid] for tid in liked_ids])
liked_emb_norms = np.linalg.norm(liked_emb_matrix, axis=1, keepdims=True)
liked_emb_matrix_norm = liked_emb_matrix / (liked_emb_norms + 1e-10)

# Chroma matrix (only for tracks that have it)
chroma_ids = [tid for tid in liked_ids if tid in liked_chroma_fps]
if chroma_ids:
    liked_chroma_matrix = np.stack([liked_chroma_fps[tid] for tid in chroma_ids])
    liked_chroma_norms = np.linalg.norm(liked_chroma_matrix, axis=1, keepdims=True)
    liked_chroma_matrix_norm = liked_chroma_matrix / (liked_chroma_norms + 1e-10)
    has_chroma = True
    print(f"Chroma матрица: {liked_chroma_matrix.shape}")
else:
    has_chroma = False
    print("Нет chroma данных для liked треков")

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
    median_emb = float(np.median(emb_similarities))
    
    emb_score = 0.4 * max_emb + 0.3 * top5_emb + 0.2 * top10_emb + 0.1 * median_emb
    
    # === CHROMA SIMILARITY ===
    chroma_score = 0.0
    if has_chroma and chroma_path:
        try:
            track_chroma_fp = load_chroma_fingerprint(chroma_path)
            if track_chroma_fp is not None:
                track_chroma_norm = track_chroma_fp / (np.linalg.norm(track_chroma_fp) + 1e-10)
                
                chroma_similarities = np.dot(liked_chroma_matrix_norm, track_chroma_norm)
                sorted_chroma_sims = np.sort(chroma_similarities)[::-1]
                
                max_chroma = float(sorted_chroma_sims[0])
                top5_chroma = float(np.mean(sorted_chroma_sims[:min(5, len(sorted_chroma_sims))]))
                
                chroma_score = 0.6 * max_chroma + 0.4 * top5_chroma
        except:
            pass
    
    # === COMBINED SCORE ===
    if chroma_score > 0:
        final_score = EMBEDDING_WEIGHT * emb_score + CHROMA_WEIGHT * chroma_score
    else:
        final_score = emb_score  # Fallback to embedding only
    
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
        'score': final_score,
        'is_liked': is_liked,
        'youtube_url': youtube_url,
    })
    
    processed += 1
    if processed % 200 == 0:
        print(f"  Обработано: {processed}")
        conn.commit()

conn.commit()
print(f"\nОбработано треков: {processed}\n")

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
    print(f"    Emb: {data['emb_score']:.3f} | Chroma: {data['chroma_score']:.3f}")
    print(f"    Жанр: {data['genre']} | BPM: {data['bpm']}")
    print()

# Stats
print("=== СТАТИСТИКА ===")
avg_score = np.mean([s['score'] for s in scores])
liked_avg = np.mean([s['score'] for s in scores if s['is_liked']])
not_liked_avg = np.mean([s['score'] for s in not_liked]) if not_liked else 0

with_chroma = [s for s in scores if s['chroma_score'] > 0]
print(f"Средний score всех: {avg_score:.3f}")
print(f"Средний score liked: {liked_avg:.3f}")
print(f"Средний score остальных: {not_liked_avg:.3f}")
print(f"Треков с chroma: {len(with_chroma)}")

cursor.close()
conn.close()
