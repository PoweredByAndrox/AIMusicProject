#!/usr/bin/env python3
"""Test segmentation on UKF Dubstep 2010 (Album Megamix)."""

import sys
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 256


def find_boundaries_v2(y, sr, duration, min_segment_sec=8.0):
    hop = HOP_LENGTH
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    
    chroma_norm = librosa.util.normalize(chroma, axis=0)
    mfcc_norm = librosa.util.normalize(mfcc, axis=0)
    contrast_norm = librosa.util.normalize(contrast, axis=0)
    onset_norm = onset_env / (onset_env.max() + 1e-10)
    rms_norm = rms / (rms.max() + 1e-10)
    
    def compute_novelty_2d(features, window):
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
    
    def compute_novelty_1d(sig, window):
        n = len(sig)
        novelty = np.zeros(n)
        for i in range(window, n - window):
            before = np.mean(sig[i-window:i])
            after = np.mean(sig[i:i+window])
            novelty[i] = abs(after - before)
        return novelty
    
    windows_sec = [1.0, 2.0, 4.0]
    weights = [0.25, 0.45, 0.30]
    
    all_novelties = []
    for win_sec, weight in zip(windows_sec, weights):
        window_frames = max(3, int(win_sec * sr / hop))
        
        chroma_nov = compute_novelty_2d(chroma_norm, window=window_frames)
        mfcc_nov = compute_novelty_2d(mfcc_norm, window=window_frames)
        contrast_nov = compute_novelty_2d(contrast_norm, window=window_frames)
        onset_nov = compute_novelty_1d(onset_norm, window=window_frames)
        rms_nov = compute_novelty_1d(rms_norm, window=window_frames)
        
        scale_novelty = (
            0.30 * chroma_nov +
            0.25 * mfcc_nov +
            0.15 * contrast_nov +
            0.15 * onset_nov +
            0.15 * rms_nov
        )
        
        if scale_novelty.max() > 0:
            scale_novelty = scale_novelty / scale_novelty.max()
        
        all_novelties.append(scale_novelty * weight)
    
    combined = np.sum(all_novelties, axis=0)
    if combined.max() > 0:
        combined = combined / combined.max()
    
    combined = median_filter(combined, size=5)
    
    min_frames = int(min_segment_sec * sr / hop)
    threshold = np.percentile(combined, 75)
    
    peaks, props = signal.find_peaks(
        combined,
        height=threshold,
        distance=min_frames,
        prominence=0.05
    )
    
    boundaries = [0.0]
    for peak in peaks:
        time_sec = peak * hop / sr
        if time_sec > 3 and time_sec < duration - 3:
            boundaries.append(time_sec)
    boundaries.append(duration)
    
    # Beat snapping
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        if len(beat_times) > 0:
            snapped = [boundaries[0]]
            for b in boundaries[1:-1]:
                nearest_beat = beat_times[np.argmin(np.abs(beat_times - b))]
                if abs(nearest_beat - b) < 0.5:
                    snapped.append(nearest_beat)
                else:
                    snapped.append(b)
            snapped.append(boundaries[-1])
            boundaries = snapped
    except:
        pass
    
    return sorted(set(boundaries)), rms_norm, onset_norm


def classify_segment(rms_norm, onset_norm, centroid_norm, start_frame, end_frame):
    if start_frame >= end_frame:
        return 'unknown', 1.0
    
    seg_rms = np.mean(rms_norm[start_frame:end_frame])
    seg_onset = np.mean(onset_norm[start_frame:end_frame])
    seg_centroid = np.mean(centroid_norm[start_frame:end_frame]) if centroid_norm is not None else 0.5
    
    rms_pct = np.percentile(rms_norm, [20, 40, 60, 80])
    onset_pct = np.percentile(onset_norm, [20, 40, 60, 80])
    centroid_pct = np.percentile(centroid_norm, [20, 40, 60, 80])
    
    is_high_energy = seg_rms > rms_pct[2]
    is_very_high_energy = seg_rms > rms_pct[3]
    is_low_energy = seg_rms < rms_pct[1]
    is_very_low_energy = seg_rms < rms_pct[0]
    
    is_rhythmic = seg_onset > onset_pct[2]
    is_very_rhythmic = seg_onset > onset_pct[3]
    
    is_bright = seg_centroid > centroid_pct[2]
    
    if is_very_high_energy and is_very_rhythmic:
        return 'drop', 3.0
    elif is_high_energy and is_rhythmic:
        if is_bright:
            return 'drop_high', 2.8
        else:
            return 'drop_bass', 2.8
    elif is_high_energy:
        return 'buildup', 2.5
    elif is_very_low_energy:
        if is_bright:
            return 'break', 1.5
        else:
            return 'ambient', 1.0
    elif is_low_energy:
        return 'intro', 1.2
    elif is_rhythmic:
        return 'verse', 2.0
    else:
        return 'melodic', 1.5


print("=" * 60)
print("UKF DUBSTEP 2010 SEGMENTATION TEST")
print("=" * 60)

audio_path = "/app/storage/audio/yt_697a9f8af05df.mp3"

if not Path(audio_path).exists():
    print(f"File not found: {audio_path}")
    sys.exit(1)

print(f"\nLoading: {audio_path}")
y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
duration = len(y) / sr
print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")

print("\nAnalyzing structure...")
boundaries, rms_norm, onset_norm = find_boundaries_v2(y, sr, duration, min_segment_sec=10.0)
print(f"Found {len(boundaries)-1} segments")

hop = HOP_LENGTH
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
centroid_norm = centroid / (centroid.max() + 1e-10)

print(f"\n{'='*60}")
print("SEGMENTS")
print(f"{'='*60}")

for i in range(len(boundaries)-1):
    start = boundaries[i]
    end = boundaries[i+1]
    dur = end - start
    
    sf = int(start * sr / hop)
    ef = min(int(end * sr / hop), len(rms_norm))
    
    seg_type, importance = classify_segment(rms_norm, onset_norm, centroid_norm, sf, ef)
    
    print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d}  ({dur:5.1f}s)  {seg_type:12}")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
