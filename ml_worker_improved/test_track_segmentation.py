#!/usr/bin/env python3
"""
Test improved track segmentation on a regular track (not a mix).
Tests precise boundary detection for drops, breaks, verses.
"""

import sys
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 22050
HOP_LENGTH_PRECISE = 256


def find_boundaries_precise(y, sr, duration):
    """Precise boundary detection using multi-scale novelty + onset + beat alignment."""
    hop = HOP_LENGTH_PRECISE
    
    print("  Extracting features...")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    
    # Normalize
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
    
    print("  Computing multi-scale novelty...")
    windows_sec = [0.2, 0.5, 1.0, 2.0]
    weights = [0.15, 0.30, 0.35, 0.20]
    
    all_novelties = []
    for win_sec, weight in zip(windows_sec, weights):
        window_frames = max(3, int(win_sec * sr / hop))
        
        chroma_nov = compute_novelty_2d(chroma_norm, window=window_frames)
        mfcc_nov = compute_novelty_2d(mfcc_norm, window=window_frames)
        contrast_nov = compute_novelty_2d(contrast_norm, window=window_frames)
        onset_nov = compute_novelty_1d(onset_norm, window=window_frames)
        rms_nov = compute_novelty_1d(rms_norm, window=window_frames)
        
        scale_novelty = (
            0.25 * chroma_nov +
            0.25 * mfcc_nov +
            0.20 * contrast_nov +
            0.20 * onset_nov +
            0.10 * rms_nov
        )
        
        if scale_novelty.max() > 0:
            scale_novelty = scale_novelty / scale_novelty.max()
        
        all_novelties.append(scale_novelty * weight)
    
    combined = np.sum(all_novelties, axis=0)
    if combined.max() > 0:
        combined = combined / combined.max()
    
    combined = median_filter(combined, size=3)
    
    print("  Detecting onsets...")
    onset_peaks = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop,
        backtrack=True,
        units='time'
    )
    print(f"    Found {len(onset_peaks)} onsets")
    
    print("  Finding novelty peaks...")
    min_frames = int(4 * sr / hop)
    threshold = np.percentile(combined, 60)
    
    peaks, props = signal.find_peaks(
        combined,
        height=threshold,
        distance=min_frames,
        prominence=0.02
    )
    
    boundaries = [0.0]
    for peak in peaks:
        time_sec = peak * hop / sr
        if time_sec > 2 and time_sec < duration - 2:
            # Refine using nearest onset
            nearest_onset = None
            min_dist = 1.0
            for onset_time in onset_peaks:
                dist = abs(onset_time - time_sec)
                if dist < min_dist:
                    min_dist = dist
                    nearest_onset = onset_time
            
            if nearest_onset is not None and min_dist < 0.3:
                boundaries.append(nearest_onset)
            else:
                boundaries.append(time_sec)
    
    boundaries.append(duration)
    
    # Beat snapping
    print("  Snapping to beats...")
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        print(f"    Tempo: {tempo:.0f} BPM, {len(beat_times)} beats")
        
        if len(beat_times) > 0:
            snapped = [boundaries[0]]
            for b in boundaries[1:-1]:
                nearest_beat = beat_times[np.argmin(np.abs(beat_times - b))]
                if abs(nearest_beat - b) < 0.3:
                    snapped.append(nearest_beat)
                else:
                    snapped.append(b)
            snapped.append(boundaries[-1])
            boundaries = snapped
    except:
        pass
    
    # Merge close
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= 3.0:
            merged.append(b)
    if boundaries[-1] not in merged:
        merged[-1] = boundaries[-1]
    
    return sorted(set(merged)), combined, rms_norm, onset_norm


def classify_segment(rms_norm, onset_norm, centroid_norm, start_frame, end_frame, rms_pct, centroid_pct):
    """Classify segment type based on audio features."""
    if start_frame >= end_frame:
        return 'unknown', 1.0
    
    seg_rms = np.mean(rms_norm[start_frame:end_frame])
    seg_onset = np.mean(onset_norm[start_frame:end_frame])
    seg_centroid = np.mean(centroid_norm[start_frame:end_frame]) if centroid_norm is not None else 0.5
    
    # Determine energy level
    if seg_rms > rms_pct[2]:
        energy = 'high'
    elif seg_rms > rms_pct[1]:
        energy = 'mid'
    else:
        energy = 'low'
    
    # Determine brightness
    if seg_centroid > centroid_pct[2]:
        brightness = 'high'
    elif seg_centroid > centroid_pct[1]:
        brightness = 'mid'
    else:
        brightness = 'low'
    
    # Determine rhythmic activity
    onset_pct = np.percentile(onset_norm, [33, 66])
    if seg_onset > onset_pct[1]:
        rhythm = 'active'
    elif seg_onset > onset_pct[0]:
        rhythm = 'moderate'
    else:
        rhythm = 'calm'
    
    # Classification logic
    if energy == 'high' and rhythm == 'active':
        if brightness == 'high':
            return 'drop_high', 3.0  # High energy drop with bright sounds
        else:
            return 'drop_bass', 3.0  # Bass-heavy drop
    elif energy == 'high' and rhythm != 'active':
        return 'buildup', 2.5  # High energy but less rhythmic = buildup
    elif energy == 'mid':
        if rhythm == 'active':
            return 'verse', 2.0  # Medium energy with rhythm
        else:
            return 'melodic', 1.5  # Medium energy, calmer
    else:  # low energy
        if brightness == 'high':
            return 'break', 1.5  # Low energy but bright = break
        else:
            return 'ambient', 1.0  # Low energy, dark = ambient/intro


print("=" * 60)
print("IMPROVED TRACK SEGMENTATION TEST")
print("=" * 60)

# Find a regular track (not a mix) to test
import psycopg2
import os

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

# Get a liked track that's NOT a mix (duration < 8 min)
cursor.execute("""
    SELECT id, title, artist, file_path, duration
    FROM tracks 
    WHERE is_liked = true 
      AND file_path IS NOT NULL
      AND duration < 480
      AND duration > 120
    ORDER BY id
    LIMIT 5
""")
tracks = cursor.fetchall()

print(f"\nFound {len(tracks)} regular tracks to test:")
for t in tracks:
    print(f"  #{t[0]}: {t[1][:40]} ({t[4]/60:.1f} min)")

if not tracks:
    print("No suitable tracks found!")
    sys.exit(1)

# Test on first track
track_id, title, artist, file_path, db_duration = tracks[0]
audio_path = f"/app/storage/{file_path}"

print(f"\n{'='*60}")
print(f"Testing: {title}")
print(f"Artist: {artist}")
print(f"{'='*60}")

if not Path(audio_path).exists():
    print(f"File not found: {audio_path}")
    sys.exit(1)

print(f"\nLoading audio...")
y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
duration = len(y) / sr
print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")

print("\nFinding boundaries...")
boundaries, novelty, rms_norm, onset_norm = find_boundaries_precise(y, sr, duration)
print(f"Found {len(boundaries)-1} segments")

# Get centroid for classification
hop = HOP_LENGTH_PRECISE
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
centroid_norm = centroid / (centroid.max() + 1e-10)

rms_pct = np.percentile(rms_norm, [25, 50, 75])
centroid_pct = np.percentile(centroid_norm, [25, 50, 75])

print(f"\n{'='*60}")
print("SEGMENTS")
print(f"{'='*60}")

for i in range(len(boundaries)-1):
    start = boundaries[i]
    end = boundaries[i+1]
    dur = end - start
    
    sf = int(start * sr / hop)
    ef = min(int(end * sr / hop), len(rms_norm))
    
    seg_type, importance = classify_segment(
        rms_norm, onset_norm, centroid_norm, sf, ef, rms_pct, centroid_pct
    )
    
    print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d}  ({dur:5.1f}s)  [{seg_type:12}] imp={importance:.1f}")

cursor.close()
conn.close()

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
