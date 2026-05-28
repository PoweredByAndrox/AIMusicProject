#!/usr/bin/env python3
"""
Test mix track detection on UKF Bass Culture.
Expected: ~17 tracks
"""

import os
import sys
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter
from sklearn.cluster import AgglomerativeClustering

sys.stdout.reconfigure(line_buffering=True)

print("=" * 60)
print("ТЕСТ ДЕТЕКЦИИ ТРЕКОВ В МИКСЕ")
print("=" * 60)

audio_path = "/app/storage/audio/yt_697a9f137cfb0.mp3"

if not Path(audio_path).exists():
    print(f"Файл не найден: {audio_path}")
    sys.exit(1)

print(f"\nЗагрузка: {audio_path}")
y, sr = librosa.load(audio_path, sr=22050, mono=True)
duration = len(y) / sr
print(f"Длительность: {duration/60:.1f} мин ({duration:.0f} сек)")

HOP_LENGTH = 512

# Expected boundaries from user
expected = [
    0, 40, 82, 150, 179, 205, 233, 260, 288, 315, 341, 370, 397, 424, 452, 494, 548, 599
]
print(f"\nОжидаемые границы ({len(expected)-1} треков):")
for i in range(len(expected)-1):
    print(f"  {i+1:2}. {expected[i]//60}:{expected[i]%60:02d} - {expected[i+1]//60}:{expected[i+1]%60:02d} ({expected[i+1]-expected[i]}s)")

print("\n" + "=" * 60)
print("МЕТОД: Комбинированный novelty detection")
print("=" * 60)

# Extract multiple features
print("\nИзвлечение признаков...")

# 1. Chroma (harmonic content)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
chroma_norm = librosa.util.normalize(chroma, axis=0)

# 2. MFCC (timbre)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
mfcc_norm = librosa.util.normalize(mfcc, axis=0)

# 3. Spectral contrast
contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
contrast_norm = librosa.util.normalize(contrast, axis=0)

# 4. Tonnetz (tonal centroid features)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
# Resample to match hop_length
if tonnetz.shape[1] != chroma.shape[1]:
    tonnetz = np.array([np.interp(
        np.linspace(0, 1, chroma.shape[1]),
        np.linspace(0, 1, tonnetz.shape[1]),
        row
    ) for row in tonnetz])
tonnetz_norm = librosa.util.normalize(tonnetz, axis=0)

n_frames = chroma.shape[1]
print(f"  Frames: {n_frames}")

# Compute novelty curves from each feature
print("\nВычисление novelty curves...")

def compute_novelty(features, window=20):
    """Compute novelty as cosine distance between adjacent windows."""
    n = features.shape[1]
    novelty = np.zeros(n)
    
    for i in range(window, n - window):
        # Get windows before and after
        before = features[:, i-window:i].mean(axis=1)
        after = features[:, i:i+window].mean(axis=1)
        
        # Cosine distance
        norm_before = np.linalg.norm(before) + 1e-10
        norm_after = np.linalg.norm(after) + 1e-10
        
        similarity = np.dot(before, after) / (norm_before * norm_after)
        novelty[i] = 1 - similarity
    
    return novelty

# Window size in frames (~2 seconds)
window_frames = int(2 * sr / HOP_LENGTH)

chroma_novelty = compute_novelty(chroma_norm, window=window_frames)
mfcc_novelty = compute_novelty(mfcc_norm, window=window_frames)
contrast_novelty = compute_novelty(contrast_norm, window=window_frames)
tonnetz_novelty = compute_novelty(tonnetz_norm, window=window_frames)

# Combine novelty curves with weights
# Chroma and tonnetz detect harmonic changes (new melody)
# MFCC detects timbre changes (new voice/instrument)
# Contrast detects texture changes
combined_novelty = (
    0.35 * chroma_novelty +
    0.25 * mfcc_novelty +
    0.20 * contrast_novelty +
    0.20 * tonnetz_novelty
)

# Normalize
if combined_novelty.max() > 0:
    combined_novelty = combined_novelty / combined_novelty.max()

# Smooth
combined_novelty = median_filter(combined_novelty, size=7)
combined_novelty = uniform_filter1d(combined_novelty, size=15)

# Find peaks
print("\nПоиск пиков...")
min_frames = int(22 * sr / HOP_LENGTH)  # 22 seconds minimum between tracks
threshold = np.percentile(combined_novelty, 65)

peaks, properties = signal.find_peaks(
    combined_novelty,
    height=threshold,
    distance=min_frames,
    prominence=0.04
)

# Convert to time
boundaries = [0.0]
for peak in peaks:
    time_sec = peak * HOP_LENGTH / sr
    if time_sec > 5 and time_sec < duration - 5:
        boundaries.append(time_sec)
boundaries.append(duration)

# Merge very short segments
def merge_short(bounds, min_dur=15):
    if len(bounds) <= 2:
        return bounds
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] >= min_dur:
            merged.append(b)
        elif b == bounds[-1]:
            merged[-1] = b
    return merged

boundaries = merge_short(boundaries, min_dur=15)

print(f"\nНайдено границ: {len(boundaries)} ({len(boundaries)-1} треков)")
for i in range(len(boundaries)-1):
    start = boundaries[i]
    end = boundaries[i+1]
    dur = end - start
    print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d} ({dur:.0f}s)")

# Compare with expected
print("\n--- СРАВНЕНИЕ С ОЖИДАЕМЫМ ---")
print(f"Ожидалось: {len(expected)-1} треков")
print(f"Найдено:   {len(boundaries)-1} треков")

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
