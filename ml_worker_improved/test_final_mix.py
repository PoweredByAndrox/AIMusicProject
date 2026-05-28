#!/usr/bin/env python3
"""
Final test of mix detection using the main segmenter.
"""

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

print("=" * 60)
print("ФИНАЛЬНЫЙ ТЕСТ СЕГМЕНТАЦИИ МИКСА")
print("=" * 60)

# Read and execute the segmenter code
with open('/app/smart_segment_extractor_gpu.py', 'r') as f:
    code = f.read()

# Execute only the class definitions (before MAIN)
exec(code.split('# =============================================================================\n# MAIN')[0])

# Test on UKF Bass Culture
audio_path = "/app/storage/audio/yt_697a9f137cfb0.mp3"

if not Path(audio_path).exists():
    print(f"Файл не найден: {audio_path}")
    sys.exit(1)

print(f"\nТестируем: {audio_path}")

segmenter = SmartSegmenterGPU()
analysis = segmenter.analyze_track(audio_path)

if analysis is None:
    print("Ошибка анализа!")
    sys.exit(1)

print(f"\nРежим микса: {analysis.get('is_mix', False)}")
print(f"Найдено сегментов: {len(analysis['segments'])}")

print("\n--- СЕГМЕНТЫ ---")
for i, seg in enumerate(analysis['segments']):
    start = seg['start']
    end = seg['end']
    dur = seg['duration']
    print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d} ({dur:.0f}s) {seg['type']}")

# Expected boundaries
expected = [
    0, 40, 82, 150, 179, 205, 233, 260, 288, 315, 341, 370, 397, 424, 452, 494, 548, 599
]

print(f"\n--- СРАВНЕНИЕ ---")
print(f"Ожидалось: {len(expected)-1} треков")
print(f"Найдено:   {len(analysis['segments'])} сегментов")

# Get boundaries from segments
boundaries = [0.0] + [seg['end'] for seg in analysis['segments']]

def find_closest(val, arr):
    return min(arr, key=lambda x: abs(x - val))

matched = 0
for exp in expected[1:-1]:
    closest = find_closest(exp, boundaries)
    diff = abs(exp - closest)
    if diff < 10:
        matched += 1
        status = "✓"
    else:
        status = "✗"
    print(f"  Ожидалось {exp//60}:{exp%60:02d}, ближайшее {int(closest)//60}:{int(closest)%60:02d} (diff={diff:.0f}s) {status}")

accuracy = matched / (len(expected) - 2) * 100
print(f"\nТочность: {accuracy:.0f}% ({matched}/{len(expected)-2} границ)")

print("\n" + "=" * 60)
print("ТЕСТ ЗАВЕРШЕН")
print("=" * 60)
