#!/usr/bin/env python3
"""
Improved track segmentation v2 - better segment merging.
"""

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
    """
    Improved boundary detection with smarter merging.
    Focuses on finding MAJOR structural changes, not every small transition.
    """
    hop = HOP_LENGTH
    
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
    # Use LARGER windows for structural changes (not small transitions)
    windows_sec = [1.0, 2.0, 4.0]  # 1s to 4s windows
    weights = [0.25, 0.45, 0.30]   # Emphasize medium scale
    
    all_novelties = []
    for win_sec, weight in zip(windows_sec, weights):
        window_frames = max(3, int(win_sec * sr / hop))
        
        chroma_nov = compute_novelty_2d(chroma_norm, window=window_frames)
        mfcc_nov = compute_novelty_2d(mfcc_norm, window=window_frames)
        contrast_nov = compute_novelty_2d(contrast_norm, window=window_frames)
        onset_nov = compute_novelty_1d(onset_norm, window=window_frames)
        rms_nov = compute_novelty_1d(rms_norm, window=window_frames)
        
        # Weight features for structural detection
        scale_novelty = (
            0.30 * chroma_nov +    # Harmonic changes (key changes)
            0.25 * mfcc_nov +      # Timbral changes
            0.15 * contrast_nov +  # Spectral texture
            0.15 * onset_nov +     # Rhythmic changes
            0.15 * rms_nov         # Energy changes (drops!)
        )
        
        if scale_novelty.max() > 0:
            scale_novelty = scale_novelty / scale_novelty.max()
        
        all_novelties.append(scale_novelty * weight)
    
    combined = np.sum(all_novelties, axis=0)
    if combined.max() > 0:
        combined = combined / combined.max()
    
    # Stronger smoothing to reduce noise
    combined = median_filter(combined, size=5)
    
    # Higher threshold for MAJOR changes only
    min_frames = int(min_segment_sec * sr / hop)
    threshold = np.percentile(combined, 75)  # Higher threshold = fewer boundaries
    
    peaks, props = signal.find_peaks(
        combined,
        height=threshold,
        distance=min_frames,
        prominence=0.05  # Higher prominence = more significant peaks
    )
    
    print(f"    Found {len(peaks)} major transitions")
    
    # Convert to time
    boundaries = [0.0]
    for peak in peaks:
        time_sec = peak * hop / sr
        if time_sec > 3 and time_sec < duration - 3:
            boundaries.append(time_sec)
    boundaries.append(duration)
    
    # Beat snapping
    print("  Snapping to beats...")
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        print(f"    Tempo: {tempo:.0f} BPM")
        
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
    
    return sorted(set(boundaries)), rms_norm, onset_norm, combined


def classify_segment_v2(rms_norm, onset_norm, centroid_norm, start_frame, end_frame):
    """Improved segment classification."""
    if start_frame >= end_frame:
        return 'unknown', 1.0
    
    seg_rms = np.mean(rms_norm[start_frame:end_frame])
    seg_onset = np.mean(onset_norm[start_frame:end_frame])
    seg_centroid = np.mean(centroid_norm[start_frame:end_frame]) if centroid_norm is not None else 0.5
    
    # Also look at variance (drops have high variance in onset)
    seg_onset_var = np.var(onset_norm[start_frame:end_frame])
    seg_rms_var = np.var(rms_norm[start_frame:end_frame])
    
    # Global percentiles
    rms_pct = np.percentile(rms_norm, [20, 40, 60, 80])
    onset_pct = np.percentile(onset_norm, [20, 40, 60, 80])
    centroid_pct = np.percentile(centroid_norm, [20, 40, 60, 80])
    
    # Classification based on multiple features
    is_high_energy = seg_rms > rms_pct[2]
    is_very_high_energy = seg_rms > rms_pct[3]
    is_low_energy = seg_rms < rms_pct[1]
    is_very_low_energy = seg_rms < rms_pct[0]
    
    is_rhythmic = seg_onset > onset_pct[2]
    is_very_rhythmic = seg_onset > onset_pct[3]
    is_calm = seg_onset < onset_pct[1]
    
    is_bright = seg_centroid > centroid_pct[2]
    is_dark = seg_centroid < centroid_pct[1]
    
    # Decision tree
    if is_very_high_energy and is_very_rhythmic:
        return 'drop', 3.0
    elif is_high_energy and is_rhythmic:
        if is_bright:
            return 'drop_high', 2.8
        else:
            return 'drop_bass', 2.8
    elif is_high_energy and not is_rhythmic:
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
print("TRACK SEGMENTATION V2 - STRUCTURAL ANALYSIS")
print("=" * 60)

import psycopg2
import os

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

# Get tracks
cursor.execute("""
    SELECT id, title, artist, file_path, duration
    FROM tracks 
    WHERE is_liked = true 
      AND file_path IS NOT NULL
      AND duration < 480
      AND duration > 120
    ORDER BY id
    LIMIT 3
""")
tracks = cursor.fetchall()

for track_id, title, artist, file_path, db_duration in tracks:
    audio_path = f"/app/storage/{file_path}"
    
    print(f"\n{'='*60}")
    print(f"Track: {title}")
    print(f"Artist: {artist}")
    print(f"{'='*60}")
    
    if not Path(audio_path).exists():
        print(f"  File not found!")
        continue
    
    print(f"\nLoading...")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")
    
    print("\nAnalyzing structure...")
    boundaries, rms_norm, onset_norm, novelty = find_boundaries_v2(y, sr, duration, min_segment_sec=10.0)
    print(f"Found {len(boundaries)-1} segments")
    
    # Get centroid
    hop = HOP_LENGTH
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    centroid_norm = centroid / (centroid.max() + 1e-10)
    
    print(f"\n--- SEGMENTS ---")
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        dur = end - start
        
        sf = int(start * sr / hop)
        ef = min(int(end * sr / hop), len(rms_norm))
        
        seg_type, importance = classify_segment_v2(rms_norm, onset_norm, centroid_norm, sf, ef)
        
        print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d}  ({dur:5.1f}s)  {seg_type:12}")

cursor.close()
conn.close()

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
