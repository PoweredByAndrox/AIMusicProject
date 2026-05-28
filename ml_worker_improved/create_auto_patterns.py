#!/usr/bin/env python3
"""
Automatically create patterns from liked fragments.
Each liked fragment becomes a pattern that can be searched across the library.
"""

import os
import sys
import psycopg2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/app')

import numpy as np

print("=== АВТОМАТИЧЕСКОЕ СОЗДАНИЕ ПАТТЕРНОВ ИЗ LIKED ФРАГМЕНТОВ ===\n")

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

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
PATTERNS_PATH = STORAGE_PATH / 'patterns'
PATTERNS_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
CHROMA_HOP = 512


def extract_segment_chroma(audio_path: str, start_time: float, end_time: float) -> np.ndarray:
    """Extract chroma for a specific segment."""
    duration = end_time - start_time
    y, sr = librosa.load(
        audio_path, 
        sr=SAMPLE_RATE, 
        mono=True,
        offset=start_time,
        duration=duration
    )
    
    chroma = librosa.feature.chroma_cqt(
        y=y, 
        sr=sr, 
        hop_length=CHROMA_HOP,
        n_chroma=12
    )
    
    # L2 normalize
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms < 1e-10] = 1.0
    chroma = chroma / norms
    
    return chroma.astype(np.float32)


# Get liked fragments that don't have patterns yet
cursor.execute("""
    SELECT f.id, f.track_id, f.start_time, f.end_time, f.comment,
           t.title, t.artist, t.file_path
    FROM fragments f
    JOIN tracks t ON t.id = f.track_id
    WHERE f.is_liked = true
      AND NOT EXISTS (
          SELECT 1 FROM pattern_examples pe 
          JOIN patterns p ON p.id = pe.pattern_id
          WHERE pe.track_id = f.track_id 
            AND pe.start_time = f.start_time 
            AND pe.end_time = f.end_time
      )
    ORDER BY f.id
""")
liked_fragments = cursor.fetchall()

print(f"Liked фрагментов без паттернов: {len(liked_fragments)}\n")

if not liked_fragments:
    print("Все liked фрагменты уже имеют паттерны!")
    
    # Show existing patterns
    cursor.execute("SELECT id, name, importance FROM patterns ORDER BY id")
    patterns = cursor.fetchall()
    if patterns:
        print("\nСуществующие паттерны:")
        for p in patterns:
            print(f"  #{p[0]}: {p[1]} (importance={p[2]})")
    
    cursor.close()
    conn.close()
    sys.exit(0)

created = 0
errors = 0

for frag_id, track_id, start_time, end_time, comment, title, artist, file_path in liked_fragments:
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        print(f"  ✗ Fragment {frag_id}: файл не найден")
        errors += 1
        continue
    
    try:
        # Extract chroma for the fragment
        chroma = extract_segment_chroma(audio_path, start_time, end_time)
        
        # Generate pattern name
        time_str = f"{int(start_time//60)}:{int(start_time%60):02d}"
        pattern_name = f"{artist or 'Unknown'} - {title[:30]} ({time_str})"
        if comment:
            pattern_name = comment[:50]
        
        # Create pattern
        cursor.execute("""
            INSERT INTO patterns (name, representation_type, transpose_invariant, tempo_invariant, 
                                  match_threshold, importance, created_at, updated_at)
            VALUES (%s, 'chroma', true, false, 0.6, 2, %s, %s)
            RETURNING id
        """, (pattern_name, datetime.now(), datetime.now()))
        
        pattern_id = cursor.fetchone()[0]
        
        # Save pattern chroma
        pattern_chroma_path = PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy"
        np.save(pattern_chroma_path, chroma)
        
        # Create pattern example (the original liked fragment)
        cursor.execute("""
            INSERT INTO pattern_examples (pattern_id, track_id, start_time, end_time, 
                                          is_positive, notes, created_at, updated_at)
            VALUES (%s, %s, %s, %s, true, %s, %s, %s)
        """, (pattern_id, track_id, start_time, end_time, 
              comment or 'Auto-created from liked fragment', datetime.now(), datetime.now()))
        
        conn.commit()
        
        print(f"  ✓ Pattern #{pattern_id}: {pattern_name}")
        created += 1
        
    except Exception as e:
        print(f"  ✗ Fragment {frag_id}: {e}")
        errors += 1
        conn.rollback()

print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Создано паттернов: {created}")
print(f"Ошибок: {errors}")

# Show all patterns
cursor.execute("""
    SELECT p.id, p.name, p.importance, COUNT(pe.id) as examples
    FROM patterns p
    LEFT JOIN pattern_examples pe ON pe.pattern_id = p.id
    GROUP BY p.id, p.name, p.importance
    ORDER BY p.id
""")
patterns = cursor.fetchall()

if patterns:
    print(f"\n=== ВСЕ ПАТТЕРНЫ ({len(patterns)}) ===")
    for p in patterns:
        print(f"  #{p[0]}: {p[1]} (importance={p[2]}, examples={p[3]})")

cursor.close()
conn.close()
