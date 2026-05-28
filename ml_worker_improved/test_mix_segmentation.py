#!/usr/bin/env python3
"""
Test script for mix/megamix segmentation.
Tests the improved algorithm on a specific track.
"""

import os
import sys
import psycopg2
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("ТЕСТ СЕГМЕНТАЦИИ МИКСОВ")
print("=" * 60)

# Import the segmenter
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("✗ CUDA недоступен, CPU mode")

# Connect to DB
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

# Find UKF Bass Culture or any long track
cursor.execute("""
    SELECT id, title, artist, file_path
    FROM tracks 
    WHERE (title ILIKE '%ukf%' OR title ILIKE '%megamix%' OR title ILIKE '%mix%')
    AND file_path IS NOT NULL
    ORDER BY id
    LIMIT 10
""")
mix_tracks = cursor.fetchall()

if not mix_tracks:
    # Fallback: find any track longer than 10 minutes
    cursor.execute("""
        SELECT id, title, artist, file_path
        FROM tracks 
        WHERE file_path IS NOT NULL
        ORDER BY id
        LIMIT 50
    """)
    all_tracks = cursor.fetchall()
    
    # Check duration
    for track in all_tracks:
        track_id, title, artist, file_path = track
        audio_path = f"/app/storage/{file_path}"
        if Path(audio_path).exists():
            try:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=1)
                info = librosa.get_duration(path=audio_path)
                if info > 600:  # >10 minutes
                    mix_tracks.append(track)
                    print(f"Found long track: {title} ({info/60:.1f} min)")
            except:
                pass

if not mix_tracks:
    print("Не найдено миксов или длинных треков для тестирования")
    sys.exit(1)

print(f"\nНайдено {len(mix_tracks)} потенциальных миксов:")
for t in mix_tracks:
    print(f"  #{t[0]}: {t[1][:50]}")

# Import the segmenter class from smart_segment_extractor_gpu
exec(open('/app/smart_segment_extractor_gpu.py').read().split('# =============================================================================')[0])

# Test on first mix
track_id, title, artist, file_path = mix_tracks[0]
audio_path = f"/app/storage/{file_path}"

print(f"\n" + "=" * 60)
print(f"ТЕСТИРУЕМ: {title}")
print("=" * 60)

if not Path(audio_path).exists():
    print(f"Файл не найден: {audio_path}")
    sys.exit(1)

# Get duration
y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
duration = len(y) / sr
print(f"Длительность: {duration/60:.1f} минут")
print(f"Это микс: {duration > 600}")

# Run segmentation
segmenter = SmartSegmenterGPU()
analysis = segmenter.analyze_track(audio_path)

if analysis is None:
    print("Ошибка анализа!")
    sys.exit(1)

print(f"\nОбнаружено сегментов: {len(analysis['segments'])}")
print(f"Режим микса: {analysis.get('is_mix', False)}")

print("\n--- СЕГМЕНТЫ ---")
for i, seg in enumerate(analysis['segments']):
    start_min = int(seg['start'] // 60)
    start_sec = int(seg['start'] % 60)
    end_min = int(seg['end'] // 60)
    end_sec = int(seg['end'] % 60)
    
    print(f"  {i+1:2}. [{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] "
          f"({seg['duration']:.0f}s) {seg['type']:12} imp={seg['importance']:.1f}")

# Statistics
durations = [seg['duration'] for seg in analysis['segments']]
print(f"\n--- СТАТИСТИКА ---")
print(f"  Мин. длина сегмента: {min(durations):.0f}s")
print(f"  Макс. длина сегмента: {max(durations):.0f}s")
print(f"  Средняя длина: {np.mean(durations):.0f}s")
print(f"  Медиана: {np.median(durations):.0f}s")

# Check for problematic long segments
long_segments = [s for s in analysis['segments'] if s['duration'] > 300]
if long_segments:
    print(f"\n⚠️  ВНИМАНИЕ: {len(long_segments)} сегментов длиннее 5 минут!")
    for seg in long_segments:
        print(f"    {seg['start']/60:.1f}min - {seg['end']/60:.1f}min ({seg['duration']/60:.1f}min)")
else:
    print(f"\n✓ Все сегменты короче 5 минут")

cursor.close()
conn.close()

print("\n" + "=" * 60)
print("ТЕСТ ЗАВЕРШЕН")
print("=" * 60)
