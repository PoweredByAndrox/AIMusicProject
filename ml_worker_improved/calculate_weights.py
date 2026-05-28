#!/usr/bin/env python3
"""
Calculate preference scores for ALL tracks based on similarity to liked tracks.
Distributes weights across the entire library.
"""

import numpy as np
import psycopg2
from pathlib import Path
import os

print("=== РАСЧЁТ ВЕСОВ ДЛЯ ВСЕХ ТРЕКОВ ===\n")

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)

cursor = conn.cursor()

# Get liked tracks with embeddings
cursor.execute("""
    SELECT id, title, artist, embedding_path 
    FROM tracks 
    WHERE is_liked = true AND embedding_path IS NOT NULL
""")
liked_tracks = cursor.fetchall()
print(f"Эталонных треков (liked): {len(liked_tracks)}")

# Load liked embeddings
liked_embeddings = {}
for track_id, title, artist, emb_path in liked_tracks:
    if emb_path and Path(emb_path).exists():
        try:
            emb = np.load(emb_path)
            liked_embeddings[track_id] = emb
        except Exception as e:
            print(f"  Error loading {track_id}: {e}")

print(f"Загружено эталонных embeddings: {len(liked_embeddings)}")

if not liked_embeddings:
    print("Нет эталонных embeddings!")
    exit(1)

# Stack liked embeddings for vectorized computation
liked_ids = list(liked_embeddings.keys())
liked_matrix = np.stack([liked_embeddings[tid] for tid in liked_ids])
# Normalize
liked_norms = np.linalg.norm(liked_matrix, axis=1, keepdims=True)
liked_matrix_norm = liked_matrix / (liked_norms + 1e-10)

# Get all tracks
cursor.execute("""
    SELECT id, title, artist, genre, bpm, mood, embedding_path, is_liked, youtube_url
    FROM tracks 
    WHERE embedding_path IS NOT NULL AND analysis_status = 'completed'
""")
all_tracks = cursor.fetchall()
print(f"Всего треков для оценки: {len(all_tracks)}\n")

print("Расчёт similarity scores...")

scores = []
processed = 0

for row in all_tracks:
    track_id, title, artist, genre, bpm, mood, emb_path, is_liked, youtube_url = row
    
    if not emb_path or not Path(emb_path).exists():
        continue
    
    try:
        track_emb = np.load(emb_path)
    except:
        continue
    
    # Normalize
    track_norm = np.linalg.norm(track_emb)
    if track_norm == 0:
        continue
    track_emb_norm = track_emb / track_norm
    
    # Calculate cosine similarity to all liked tracks (vectorized)
    similarities = np.dot(liked_matrix_norm, track_emb_norm)
    
    # Improved scoring: weighted combination of metrics
    sorted_sims = np.sort(similarities)[::-1]  # descending
    
    # 1. Max similarity (best match)
    max_sim = float(sorted_sims[0])
    
    # 2. Top-5 average (consistency across multiple liked tracks)
    top_k = min(5, len(sorted_sims))
    top5_avg = float(np.mean(sorted_sims[:top_k]))
    
    # 3. Top-10 average (broader consistency)
    top10_k = min(10, len(sorted_sims))
    top10_avg = float(np.mean(sorted_sims[:top10_k]))
    
    # 4. Median of all similarities (overall fit)
    median_sim = float(np.median(similarities))
    
    # Combined score: weighted formula
    # - 40% max (reward best match)
    # - 30% top5 (consistency with favorites)
    # - 20% top10 (broader appeal)
    # - 10% median (penalize if most liked tracks don't match)
    score = 0.4 * max_sim + 0.3 * top5_avg + 0.2 * top10_avg + 0.1 * median_sim
    
    # Update database
    cursor.execute("""
        UPDATE tracks 
        SET preference_score = %s, score_calculated_at = NOW() 
        WHERE id = %s
    """, (score, track_id))
    
    scores.append({
        'id': track_id,
        'title': title,
        'artist': artist,
        'genre': genre,
        'bpm': bpm,
        'mood': mood,
        'score': score,
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
    '0.9+': 0,
    '0.8-0.9': 0,
    '0.7-0.8': 0,
    '0.6-0.7': 0,
    '0.5-0.6': 0,
    '0.4-0.5': 0,
    '0.3-0.4': 0,
    '0.2-0.3': 0,
    '0.1-0.2': 0,
    '0-0.1': 0,
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

print("\n=== ТОП-30 ТРЕКОВ ПО ВЕСУ (не liked) ===\n")

not_liked = [s for s in scores if not s['is_liked']]
top30 = not_liked[:30]

for i, data in enumerate(top30, 1):
    print(f"{i:2}. [{data['score']:.3f}] {data['artist'] or '?'} - {data['title']}")
    print(f"    Жанр: {data['genre']} | BPM: {data['bpm']} | Mood: {data['mood']}")
    if data['youtube_url']:
        print(f"    🔗 {data['youtube_url']}")
    print()

# Show stats
avg_score = np.mean([s['score'] for s in scores])
liked_avg = np.mean([s['score'] for s in scores if s['is_liked']])
not_liked_avg = np.mean([s['score'] for s in not_liked]) if not_liked else 0

print("=== СТАТИСТИКА ===")
print(f"Средний score всех треков: {avg_score:.3f}")
print(f"Средний score liked треков: {liked_avg:.3f}")
print(f"Средний score остальных: {not_liked_avg:.3f}")

cursor.close()
conn.close()
