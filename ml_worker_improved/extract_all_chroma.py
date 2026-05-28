#!/usr/bin/env python3
"""
Extract chroma features for all tracks that don't have them yet.
Chroma = 12-tone harmonic representation for melodic similarity.
"""

import os
import sys
import psycopg2
from pathlib import Path

# Add worker to path
sys.path.insert(0, '/app')

import numpy as np

print("=== ИЗВЛЕЧЕНИЕ CHROMA ДЛЯ ВСЕХ ТРЕКОВ ===\n")

# Check librosa
try:
    import librosa
    print("✓ librosa доступна")
except ImportError:
    print("✗ librosa не установлена!")
    sys.exit(1)

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

# Storage paths
STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
PATTERNS_PATH = STORAGE_PATH / 'patterns'
PATTERNS_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
CHROMA_HOP = 512


def extract_chroma(audio_path: str) -> np.ndarray:
    """Extract chroma features from audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # Extract chroma using CQT (more robust)
    chroma = librosa.feature.chroma_cqt(
        y=y, 
        sr=sr, 
        hop_length=CHROMA_HOP,
        n_chroma=12
    )
    
    # L2 normalize each frame
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms < 1e-10] = 1.0
    chroma = chroma / norms
    
    return chroma.astype(np.float32)


def create_chroma_fingerprint(chroma: np.ndarray) -> np.ndarray:
    """Create compact fingerprint for fast comparison."""
    # Average chroma profile (12 dims)
    avg_chroma = np.mean(chroma, axis=1)
    
    # Variance (12 dims)
    var_chroma = np.var(chroma, axis=1)
    
    # Dominant pitch class (1 dim)
    dominant = np.argmax(avg_chroma) / 12.0
    
    return np.concatenate([avg_chroma, var_chroma, [dominant]]).astype(np.float32)


# Get tracks without chroma
cursor.execute("""
    SELECT id, file_path, title, artist 
    FROM tracks 
    WHERE analysis_status = 'completed' 
      AND file_path IS NOT NULL
      AND (chroma_path IS NULL OR chroma_path = '')
    ORDER BY id
""")
tracks = cursor.fetchall()

print(f"Треков без chroma: {len(tracks)}\n")

if not tracks:
    print("Все треки уже имеют chroma!")
    sys.exit(0)

processed = 0
errors = 0

for track_id, file_path, title, artist in tracks:
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        print(f"  ✗ {track_id}: файл не найден")
        errors += 1
        continue
    
    try:
        # Extract chroma
        chroma = extract_chroma(audio_path)
        
        # Save chroma
        chroma_filename = f"chroma_{track_id}.npy"
        chroma_path = PATTERNS_PATH / chroma_filename
        np.save(chroma_path, chroma)
        
        # Create and save fingerprint
        fingerprint = create_chroma_fingerprint(chroma)
        fp_filename = f"chroma_fp_{track_id}.npy"
        fp_path = PATTERNS_PATH / fp_filename
        np.save(fp_path, fingerprint)
        
        # Update database
        cursor.execute("""
            UPDATE tracks 
            SET chroma_path = %s 
            WHERE id = %s
        """, (str(chroma_path), track_id))
        
        processed += 1
        
        if processed % 50 == 0:
            conn.commit()
            print(f"  Обработано: {processed}/{len(tracks)}")
            
    except Exception as e:
        print(f"  ✗ {track_id} ({artist} - {title}): {e}")
        errors += 1

conn.commit()

print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Обработано: {processed}")
print(f"Ошибок: {errors}")

cursor.close()
conn.close()
