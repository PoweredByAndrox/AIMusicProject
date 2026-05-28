#!/usr/bin/env python3
"""
Re-analyze all tracks with Essentia backend to get proper embeddings.
This replaces the garbage random embeddings with real Discogs-EffNet embeddings.
"""

import os
import sys
import numpy as np
import psycopg2
from pathlib import Path

# Add worker to path
sys.path.insert(0, '/app')

print("=== ПЕРЕАНАЛИЗ ВСЕХ ТРЕКОВ С ESSENTIA ===\n")

# Initialize analyzer
from worker.audio_analyzer import AudioAnalyzer
analyzer = AudioAnalyzer()

print(f"Backend: {'Essentia' if analyzer._using_essentia else 'PyTorch'}")
print(f"Embedder ready: {analyzer._ess_embedder is not None}")

if not analyzer._using_essentia:
    print("ERROR: Essentia not active! Cannot proceed.")
    sys.exit(1)

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

# Get all tracks with audio files
cursor.execute("""
    SELECT id, title, artist, file_path, embedding_path
    FROM tracks 
    WHERE file_path IS NOT NULL AND analysis_status = 'completed'
    ORDER BY id
""")
tracks = cursor.fetchall()
print(f"Треков для переанализа: {len(tracks)}\n")

storage_path = Path('/app/storage')
embedding_dir = storage_path / 'embeddings'
embedding_dir.mkdir(parents=True, exist_ok=True)

processed = 0
errors = 0

for track_id, title, artist, file_path, old_emb_path in tracks:
    audio_path = storage_path / file_path
    
    if not audio_path.exists():
        print(f"  SKIP {track_id}: audio not found")
        errors += 1
        continue
    
    try:
        # Extract new embedding with Essentia
        embedding = analyzer.extract_embeddings(str(audio_path))
        
        # Save embedding
        new_emb_path = embedding_dir / f"track_{track_id}.npy"
        np.save(str(new_emb_path), embedding)
        
        # Update database
        cursor.execute("""
            UPDATE tracks 
            SET embedding_path = %s
            WHERE id = %s
        """, (str(new_emb_path), track_id))
        
        processed += 1
        if processed % 50 == 0:
            print(f"  Обработано: {processed}/{len(tracks)}")
            conn.commit()
            
    except Exception as e:
        print(f"  ERROR {track_id} ({artist} - {title}): {e}")
        errors += 1

conn.commit()
print(f"\n=== ГОТОВО ===")
print(f"Успешно: {processed}")
print(f"Ошибок: {errors}")

cursor.close()
conn.close()
