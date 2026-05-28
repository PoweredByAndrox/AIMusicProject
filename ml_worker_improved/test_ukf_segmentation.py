#!/usr/bin/env python3
"""
Test script for UKF Bass Culture segmentation.
"""

import os
import sys
import psycopg2
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("ТЕСТ СЕГМЕНТАЦИИ UKF BASS CULTURE")
print("=" * 60)

import torch
import numpy as np
import librosa

SAMPLE_RATE = 22050

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("✗ CUDA недоступен")

# Test track - regular track (not a mix)
audio_path = "/app/storage/audio/yt_697a9f137cfb0.mp3"  # UKF Bass Culture 10 min

if not Path(audio_path).exists():
    print(f"Файл не найден: {audio_path}")
    sys.exit(1)

# Get duration first
print("\nЗагрузка аудио...")
duration = librosa.get_duration(path=audio_path)
print(f"Длительность: {duration/60:.1f} минут ({duration:.0f} секунд)")
print(f"Это микс (>10 мин): {duration > 600}")

# Import and run segmenter
print("\nЗапуск сегментации...")

# Read the segmenter code
with open('/app/smart_segment_extractor_gpu.py', 'r') as f:
    code = f.read()
    
# Execute only the class definitions (before MAIN)
exec(code.split('# =============================================================================\n# MAIN')[0])

segmenter = SmartSegmenterGPU()
analysis = segmenter.analyze_track(audio_path)

if analysis is None:
    print("Ошибка анализа!")
    sys.exit(1)

print(f"\n" + "=" * 60)
print(f"РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"Обнаружено сегментов: {len(analysis['segments'])}")
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
print(f"  Всего сегментов: {len(durations)}")
print(f"  Мин. длина: {min(durations):.0f}s")
print(f"  Макс. длина: {max(durations):.0f}s")
print(f"  Средняя длина: {np.mean(durations):.0f}s")
print(f"  Медиана: {np.median(durations):.0f}s")

# Expected: ~20-30 tracks in a megamix, so ~20-30 segments of 2-4 minutes each
expected_segments = int(duration / 180)  # ~3 min per track
print(f"\n  Ожидаемое кол-во сегментов (по 3 мин): ~{expected_segments}")

# Check for problematic long segments
long_segments = [s for s in analysis['segments'] if s['duration'] > 300]
if long_segments:
    print(f"\n⚠️  ВНИМАНИЕ: {len(long_segments)} сегментов длиннее 5 минут!")
    for seg in long_segments:
        print(f"    {seg['start']/60:.1f}min - {seg['end']/60:.1f}min ({seg['duration']/60:.1f}min)")
else:
    print(f"\n✓ Все сегменты короче 5 минут")

print("\n" + "=" * 60)
print("ТЕСТ ЗАВЕРШЕН")
print("=" * 60)
