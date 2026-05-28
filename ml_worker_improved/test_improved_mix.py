#!/usr/bin/env python3
"""
Improved mix track detection with multi-scale novelty.
"""

import os
import sys
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter

sys.stdout.reconfigure(line_buffering=True)

print("=" * 60)
print("УЛУЧШЕННАЯ ДЕТЕКЦИЯ ТРЕКОВ В МИКСЕ")
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
print("ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ")
print("=" * 60)

# Extract features
print("\n1. Chroma...")
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
chroma_norm = librosa.util.normalize(chroma, axis=0)

print("2. MFCC...")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
mfcc_norm = librosa.util.normalize(mfcc, axis=0)

print("3. Spectral contrast...")
contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
contrast_norm = librosa.util.normalize(contrast, axis=0)

print("4. Tonnetz...")
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
if tonnetz.shape[1] != chroma.shape[1]:
    tonnetz = np.array([np.interp(
        np.linspace(0, 1, chroma.shape[1]),
        np.linspace(0, 1, tonnetz.shape[1]),
        row
    ) for row in tonnetz])
tonnetz_norm = librosa.util.normalize(tonnetz, axis=0)

print("5. Onset strength...")
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
onset_norm = onset_env / (onset_env.max() + 1e-10)

print("6. RMS energy...")
rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
rms_norm = rms / (rms.max() + 1e-10)

n_frames = chroma.shape[1]
print(f"\nFrames: {n_frames}")

print("\n" + "=" * 60)
print("MULTI-SCALE NOVELTY DETECTION")
print("=" * 60)

def compute_novelty(features, window):
    """Compute novelty as cosine distance between adjacent windows."""
    n = features.shape[1]
    novelty = np.zeros(n)
    
    for i in range(window, n - window):
        before = features[:, i-window:i].mean(axis=1)
        after = features[:, i:i+window].mean(axis=1)
        
        norm_before = np.linalg.norm(before) + 1e-10
        norm_after = np.linalg.norm(after) + 1e-10
        
        similarity = np.dot(before, after) / (norm_before * norm_after)
        novelty[i] = 1 - similarity
    
    return novelty

def compute_1d_novelty(signal, window):
    """Compute novelty for 1D signal."""
    n = len(signal)
    novelty = np.zeros(n)
    
    for i in range(window, n - window):
        before = np.mean(signal[i-window:i])
        after = np.mean(signal[i:i+window])
        novelty[i] = abs(after - before)
    
    return novelty

# Multi-scale windows (in seconds)
scales = [1.0, 2.0, 4.0]
scale_weights = [0.3, 0.4, 0.3]  # Weight for each scale

print("\nВычисление novelty для каждого масштаба...")

all_novelties = []

for scale, weight in zip(scales, scale_weights):
    window_frames = int(scale * sr / HOP_LENGTH)
    print(f"  Scale {scale}s (window={window_frames} frames, weight={weight})...")
    
    # Compute novelty for each feature
    chroma_nov = compute_novelty(chroma_norm, window=window_frames)
    mfcc_nov = compute_novelty(mfcc_norm, window=window_frames)
    contrast_nov = compute_novelty(contrast_norm, window=window_frames)
    tonnetz_nov = compute_novelty(tonnetz_norm, window=window_frames)
    onset_nov = compute_1d_novelty(onset_norm, window=window_frames)
    rms_nov = compute_1d_novelty(rms_norm, window=window_frames)
    
    # Combine features for this scale
    scale_novelty = (
        0.30 * chroma_nov +
        0.20 * mfcc_nov +
        0.15 * contrast_nov +
        0.15 * tonnetz_nov +
        0.10 * onset_nov +
        0.10 * rms_nov
    )
    
    # Normalize
    if scale_novelty.max() > 0:
        scale_novelty = scale_novelty / scale_novelty.max()
    
    all_novelties.append(scale_novelty * weight)

# Combine all scales
combined_novelty = np.sum(all_novelties, axis=0)

# Normalize
if combined_novelty.max() > 0:
    combined_novelty = combined_novelty / combined_novelty.max()

# Smooth
combined_novelty = median_filter(combined_novelty, size=5)
combined_novelty = uniform_filter1d(combined_novelty, size=10)

print("\n" + "=" * 60)
print("ПОИСК ГРАНИЦ")
print("=" * 60)

# Simple but effective approach: find peaks, then smart merge
min_frames = int(15 * sr / HOP_LENGTH)  # 15 seconds minimum (tracks can be 25-30s)
threshold = np.percentile(combined_novelty, 50)  # Lower threshold

print(f"\nПараметры:")
print(f"  min_distance: {15}s ({min_frames} frames)")
print(f"  threshold: {threshold:.4f} (50th percentile)")

peaks, properties = signal.find_peaks(
    combined_novelty,
    height=threshold,
    distance=min_frames,
    prominence=0.02
)

print(f"  Найдено пиков: {len(peaks)}")

# Also find peaks with lower threshold but higher prominence (strong transitions)
threshold_low = np.percentile(combined_novelty, 45)
peaks_strong, _ = signal.find_peaks(
    combined_novelty,
    height=threshold_low,
    distance=int(22 * sr / HOP_LENGTH),
    prominence=0.035  # Higher prominence = more distinct peaks
)
print(f"  Сильных пиков (высокая prominence): {len(peaks_strong)}")

# Combine: add strong peaks that aren't already in peaks
all_peaks = set(peaks)
for sp in peaks_strong:
    # Check if there's already a peak within 15 seconds
    sp_time = sp * HOP_LENGTH / sr
    has_nearby = False
    for p in all_peaks:
        p_time = p * HOP_LENGTH / sr
        if abs(sp_time - p_time) < 15:
            has_nearby = True
            break
    if not has_nearby:
        all_peaks.add(sp)

all_peaks = sorted(all_peaks)
print(f"  После объединения: {len(all_peaks)} пиков")

# Convert to time
boundaries = [0.0]
for peak in all_peaks:
    time_sec = peak * HOP_LENGTH / sr
    if time_sec > 5 and time_sec < duration - 5:
        boundaries.append(time_sec)
boundaries.append(duration)

# Simple merge: remove boundaries that are too close (keep first one)
def simple_merge(bounds, min_gap=12):
    if len(bounds) <= 2:
        return bounds
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] >= min_gap:
            merged.append(b)
    if bounds[-1] not in merged:
        merged[-1] = bounds[-1]  # Ensure we have the end
    return merged

boundaries = simple_merge(boundaries, min_gap=12)
print(f"  После merge: {len(boundaries)} границ")

# Post-process: find missing peaks in large gaps (>35 seconds)
def fill_large_gaps(bounds, novelty, min_gap=35, sr=sr, hop=HOP_LENGTH):
    """Find peaks in gaps larger than min_gap seconds."""
    filled = list(bounds)
    
    for i in range(len(bounds) - 1):
        gap_start = bounds[i]
        gap_end = bounds[i + 1]
        gap_size = gap_end - gap_start
        
        if gap_size > min_gap:
            # Look for peaks in this gap
            start_frame = int(gap_start * sr / hop)
            end_frame = int(gap_end * sr / hop)
            
            gap_novelty = novelty[start_frame:end_frame]
            if len(gap_novelty) > 0:
                # Find the highest peak in this gap
                local_peaks, _ = signal.find_peaks(gap_novelty, prominence=0.02)
                if len(local_peaks) > 0:
                    # Get the peak with highest novelty
                    best_peak = local_peaks[np.argmax(gap_novelty[local_peaks])]
                    peak_time = gap_start + best_peak * hop / sr
                    
                    # Only add if it's not too close to boundaries
                    if peak_time - gap_start > 8 and gap_end - peak_time > 8:  # Reduced from 12 to 8
                        filled.append(peak_time)
                        print(f"    Добавлен пик в промежутке {int(gap_start)//60}:{int(gap_start)%60:02d}-{int(gap_end)//60}:{int(gap_end)%60:02d}: {int(peak_time)//60}:{int(peak_time)%60:02d}")
    
    return sorted(filled)

print("\n  Заполнение больших промежутков...")
boundaries = fill_large_gaps(boundaries, combined_novelty, min_gap=20)  # Reduced from 32 to 20

# DEBUG: Analyze the 9:08 problem area
print("\n--- DEBUG: Область 8:30-9:30 ---")
for t in range(510, 570, 5):  # 8:30 to 9:30
    frame = int(t * sr / HOP_LENGTH)
    if frame < len(combined_novelty):
        nov = combined_novelty[frame]
        marker = ""
        if t == 548:  # 9:08
            marker = " <-- ОЖИДАЕМАЯ ГРАНИЦА"
        # Check if this time is in boundaries
        for b in boundaries:
            if abs(b - t) < 3:
                marker = f" <-- НАЙДЕНА ({b:.0f}s)"
                break
        print(f"  {t//60}:{t%60:02d} ({t}s): nov={nov:.4f}{marker}")

# Show all boundaries in 8:00-10:00 range
print("\n--- Границы в области 8:00-10:00 ---")
for b in boundaries:
    if 480 <= b <= 600:
        print(f"  {int(b)//60}:{int(b)%60:02d} ({b:.1f}s)")

# Debug: show novelty values at expected boundaries
print("\n--- DEBUG: Novelty в ожидаемых точках ---")
for exp in expected[1:-1]:
    frame = int(exp * sr / HOP_LENGTH)
    if frame < len(combined_novelty):
        nov_val = combined_novelty[frame]
        # Find local max within ±5 seconds
        start_f = max(0, frame - int(5 * sr / HOP_LENGTH))
        end_f = min(len(combined_novelty), frame + int(5 * sr / HOP_LENGTH))
        local_max = np.max(combined_novelty[start_f:end_f])
        local_max_idx = start_f + np.argmax(combined_novelty[start_f:end_f])
        local_max_time = local_max_idx * HOP_LENGTH / sr
        print(f"  {exp//60}:{exp%60:02d} -> nov={nov_val:.4f}, local_max={local_max:.4f} at {int(local_max_time)//60}:{int(local_max_time)%60:02d}")

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
