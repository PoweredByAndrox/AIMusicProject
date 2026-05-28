#!/usr/bin/env python3
"""Verify precise segmentation on This Is Dubstep 2011."""

import sys
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 256

def detect_boundaries_precise(y, sr, hop_length=HOP_LENGTH):
    duration = len(y) / sr
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    chroma_norm = librosa.util.normalize(chroma, axis=0)
    mfcc_norm = librosa.util.normalize(mfcc, axis=0)
    contrast_norm = librosa.util.normalize(contrast, axis=0)
    onset_norm = onset_env / (onset_env.max() + 1e-10)
    rms_norm = rms / (rms.max() + 1e-10)
    
    def compute_novelty(features, window):
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
    
    def compute_1d_novelty(sig, window):
        n = len(sig)
        novelty = np.zeros(n)
        for i in range(window, n - window):
            before = np.mean(sig[i-window:i])
            after = np.mean(sig[i:i+window])
            novelty[i] = abs(after - before)
        return novelty
    
    windows_sec = [0.3, 0.5, 1.0, 2.0]
    weights = [0.2, 0.3, 0.3, 0.2]
    
    all_novelties = []
    for win_sec, weight in zip(windows_sec, weights):
        window_frames = max(3, int(win_sec * sr / hop_length))
        chroma_nov = compute_novelty(chroma_norm, window=window_frames)
        mfcc_nov = compute_novelty(mfcc_norm, window=window_frames)
        contrast_nov = compute_novelty(contrast_norm, window=window_frames)
        onset_nov = compute_1d_novelty(onset_norm, window=window_frames)
        rms_nov = compute_1d_novelty(rms_norm, window=window_frames)
        
        scale_novelty = 0.30*chroma_nov + 0.25*mfcc_nov + 0.20*contrast_nov + 0.15*onset_nov + 0.10*rms_nov
        if scale_novelty.max() > 0:
            scale_novelty = scale_novelty / scale_novelty.max()
        all_novelties.append(scale_novelty * weight)
    
    combined = np.sum(all_novelties, axis=0)
    if combined.max() > 0:
        combined = combined / combined.max()
    combined = median_filter(combined, size=3)
    
    min_frames = int(5 * sr / hop_length)
    threshold = np.percentile(combined, 35)
    peaks, _ = signal.find_peaks(combined, height=threshold, distance=min_frames, prominence=0.01)
    
    boundaries = [0.0]
    for peak in peaks:
        time_sec = peak * hop_length / sr
        if time_sec > 2 and time_sec < duration - 2:
            boundaries.append(time_sec)
    boundaries.append(duration)
    return boundaries

def snap_to_beat(y, sr, boundaries):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times) == 0:
        return boundaries
    snapped = [boundaries[0]]
    for b in boundaries[1:-1]:
        nearest_beat = beat_times[np.argmin(np.abs(beat_times - b))]
        if abs(nearest_beat - b) < 0.5:
            snapped.append(nearest_beat)
        else:
            snapped.append(b)
    snapped.append(boundaries[-1])
    return snapped

def merge_close(boundaries, min_gap=5):
    if len(boundaries) <= 2:
        return boundaries
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= min_gap:
            merged.append(b)
    if boundaries[-1] not in merged:
        merged[-1] = boundaries[-1]
    return merged

print("=" * 60)
print("PRECISE MIX SEGMENTATION - THIS IS DUBSTEP 2011")
print("=" * 60)

audio_path = "/app/storage/audio/yt_697a9eea73f1d.mp3"
if not Path(audio_path).exists():
    print(f"File not found: {audio_path}")
    sys.exit(1)

print(f"\nLoading: {audio_path}")
y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
duration = len(y) / sr
print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")

# Expected boundaries for This Is Dubstep 2011 (40 tracks)
expected = [0, 28, 42, 69, 90, 117, 130, 145, 158, 172, 178, 186, 192, 206, 220, 244, 274, 302, 315, 323, 336, 350, 364, 378, 390, 398, 411, 452, 480, 507, 514, 527, 541, 555, 569, 582, 596, 610, 624, 637, 646]
print(f"\nExpected: {len(expected)-1} segments")

print("\nStep 1: Detecting boundaries...")
boundaries = detect_boundaries_precise(y, sr)
print(f"  Found {len(boundaries)-1} candidates")

print("\nStep 2: Snapping to beats...")
boundaries = snap_to_beat(y, sr, boundaries)

print("\nStep 3: Merging close boundaries...")
boundaries = merge_close(boundaries, min_gap=5)
print(f"  Final: {len(boundaries)-1} segments")

print(f"\n{'='*60}")
print(f"RESULTS: {len(boundaries)-1} SEGMENTS")
print(f"{'='*60}")

for i in range(min(20, len(boundaries)-1)):
    start = boundaries[i]
    end = boundaries[i+1]
    dur = end - start
    print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d}  ({dur:.0f}s)")
if len(boundaries) > 21:
    print(f"  ... and {len(boundaries)-21} more")

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")

def find_closest(val, arr):
    return min(arr, key=lambda x: abs(x - val))

matched = 0
for exp in expected[1:-1]:
    closest = find_closest(exp, boundaries)
    diff = abs(exp - closest)
    status = "OK" if diff < 8 else "MISS"
    if diff < 8:
        matched += 1
    print(f"  {exp//60}:{exp%60:02d} -> {int(closest)//60}:{int(closest)%60:02d} (diff={diff:.0f}s) [{status}]")

accuracy = matched / (len(expected) - 2) * 100
print(f"\nAccuracy: {accuracy:.0f}% ({matched}/{len(expected)-2})")
