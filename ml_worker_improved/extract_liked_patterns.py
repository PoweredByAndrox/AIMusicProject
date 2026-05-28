#!/usr/bin/env python3
"""
Extract characteristic patterns from liked tracks.
Finds high-energy segments (drops, hooks) and creates searchable patterns.

The idea:
1. Analyze each liked track for energy peaks (drops, buildups)
2. Extract chroma for these high-energy segments
3. Store as patterns for searching across the library
"""

import os
import sys
import psycopg2
from pathlib import Path
from datetime import datetime
import numpy as np

print("=== ИЗВЛЕЧЕНИЕ ПАТТЕРНОВ ИЗ LIKED ТРЕКОВ ===\n")

try:
    import librosa
    print("✓ librosa доступна")
except ImportError:
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
PATTERNS_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 512
SEGMENT_DURATION = 8.0  # seconds per pattern segment


def find_energy_peaks(audio_path: str, num_peaks: int = 3) -> list:
    """
    Find high-energy segments in a track (drops, hooks).
    Returns list of (start_time, end_time, energy) tuples.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(y) / sr
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        
        # Compute spectral flux (onset strength)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        
        # Combine RMS and onset for "energy" metric
        # Normalize both
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
        onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-10)
        
        # Combined energy (weight onset more for drops)
        energy = 0.4 * rms_norm + 0.6 * onset_norm[:len(rms_norm)]
        
        # Smooth energy
        window = int(sr / HOP_LENGTH * 2)  # 2 second window
        if window > 1:
            energy = np.convolve(energy, np.ones(window)/window, mode='same')
        
        # Find peaks
        segment_frames = int(SEGMENT_DURATION * sr / HOP_LENGTH)
        peaks = []
        
        # Divide track into segments and find highest energy ones
        for i in range(0, len(energy) - segment_frames, segment_frames // 2):
            segment_energy = np.mean(energy[i:i + segment_frames])
            start_time = i * HOP_LENGTH / sr
            end_time = min((i + segment_frames) * HOP_LENGTH / sr, duration)
            peaks.append((start_time, end_time, segment_energy))
        
        # Sort by energy and take top N
        peaks.sort(key=lambda x: x[2], reverse=True)
        
        # Filter out overlapping segments
        selected = []
        for peak in peaks:
            overlap = False
            for sel in selected:
                if abs(peak[0] - sel[0]) < SEGMENT_DURATION / 2:
                    overlap = True
                    break
            if not overlap:
                selected.append(peak)
                if len(selected) >= num_peaks:
                    break
        
        return selected
        
    except Exception as e:
        print(f"  Error finding peaks: {e}")
        return []


def extract_segment_chroma(audio_path: str, start_time: float, end_time: float) -> np.ndarray:
    """Extract chroma for a specific segment."""
    try:
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
            hop_length=HOP_LENGTH,
            n_chroma=12
        )
        
        # L2 normalize
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        chroma = chroma / norms
        
        return chroma.astype(np.float32)
        
    except Exception as e:
        print(f"  Error extracting chroma: {e}")
        return None


# Clear old auto-generated patterns (keep manual ones from fragments)
cursor.execute("""
    DELETE FROM pattern_matches WHERE pattern_id IN (
        SELECT id FROM patterns WHERE name LIKE '%[auto]%'
    )
""")
cursor.execute("DELETE FROM pattern_examples WHERE pattern_id IN (SELECT id FROM patterns WHERE name LIKE '%[auto]%')")
cursor.execute("DELETE FROM patterns WHERE name LIKE '%[auto]%'")
conn.commit()
print("Очищены старые auto-паттерны\n")

# Get liked tracks
cursor.execute("""
    SELECT id, title, artist, file_path, chroma_path
    FROM tracks
    WHERE is_liked = true
      AND file_path IS NOT NULL
    ORDER BY id
""")
liked_tracks = cursor.fetchall()
print(f"Liked треков: {len(liked_tracks)}\n")

patterns_created = 0
total_segments = 0

for track_id, title, artist, file_path, chroma_path in liked_tracks:
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        print(f"  ✗ Track {track_id}: файл не найден")
        continue
    
    print(f"Анализ: {artist or '?'} - {title[:40]}...")
    
    # Find high-energy segments
    peaks = find_energy_peaks(audio_path, num_peaks=2)
    
    if not peaks:
        print(f"  ✗ Не найдены энергичные сегменты")
        continue
    
    for i, (start_time, end_time, energy) in enumerate(peaks):
        # Extract chroma for this segment
        chroma = extract_segment_chroma(audio_path, start_time, end_time)
        
        if chroma is None or chroma.shape[1] < 10:
            continue
        
        # Create pattern
        time_str = f"{int(start_time//60)}:{int(start_time%60):02d}"
        pattern_name = f"[auto] {artist or 'Unknown'} - {title[:25]} ({time_str})"
        
        cursor.execute("""
            INSERT INTO patterns (name, representation_type, transpose_invariant, tempo_invariant,
                                  match_threshold, importance, created_at, updated_at)
            VALUES (%s, 'chroma', true, false, 0.65, 2, %s, %s)
            RETURNING id
        """, (pattern_name, datetime.now(), datetime.now()))
        
        pattern_id = cursor.fetchone()[0]
        
        # Save pattern chroma
        pattern_chroma_path = PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy"
        np.save(pattern_chroma_path, chroma)
        
        # Create pattern example
        cursor.execute("""
            INSERT INTO pattern_examples (pattern_id, track_id, start_time, end_time,
                                          is_positive, notes, created_at, updated_at)
            VALUES (%s, %s, %s, %s, true, %s, %s, %s)
        """, (pattern_id, track_id, start_time, end_time,
              f"Auto-extracted high-energy segment (energy={energy:.2f})",
              datetime.now(), datetime.now()))
        
        patterns_created += 1
        total_segments += 1
        
        print(f"  ✓ Pattern #{pattern_id}: {time_str} (energy={energy:.2f})")
    
    conn.commit()

print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Создано паттернов: {patterns_created}")
print(f"Всего сегментов: {total_segments}")

# Show all patterns
cursor.execute("""
    SELECT p.id, p.name, p.importance
    FROM patterns p
    ORDER BY p.id DESC
    LIMIT 20
""")
patterns = cursor.fetchall()

print(f"\n=== ПОСЛЕДНИЕ 20 ПАТТЕРНОВ ===")
for pid, name, importance in patterns:
    print(f"  #{pid}: {name}")

cursor.close()
conn.close()
