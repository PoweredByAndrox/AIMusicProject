#!/usr/bin/env python3
"""Test extended segment classification with more types."""

import sys
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 256


def extract_features(y, sr):
    """Extract all features needed for classification."""
    hop = HOP_LENGTH
    
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    
    return {
        'rms': rms,
        'centroid': centroid,
        'onset_env': onset_env,
        'flatness': flatness
    }


def find_boundaries(y, sr, duration, min_segment_sec=8.0):
    """Find segment boundaries."""
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
    
    peaks, _ = signal.find_peaks(
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
    
    return sorted(set(boundaries))


def classify_segments_extended(features, boundaries, duration):
    """
    Extended segment classification with more types.
    
    Types: pre_intro, intro, buildup, pre_drop, drop, drop_bass, drop_high,
           verse, vocal, melodic, break, ambient, outro
    """
    hop = HOP_LENGTH
    
    # First pass: basic features
    raw_segments = []
    
    rms_pct = np.percentile(features['rms'], [15, 30, 50, 70, 85])
    centroid_pct = np.percentile(features['centroid'], [20, 40, 60, 80])
    onset_pct = np.percentile(features['onset_env'], [20, 40, 60, 80])
    flatness_pct = np.percentile(features['flatness'], [30, 50, 70])
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_dur = end - start
        
        if seg_dur < 3:
            continue
        
        sf = int(start * SAMPLE_RATE / hop)
        ef = min(int(end * SAMPLE_RATE / hop), len(features['rms']))
        
        if sf >= ef:
            continue
        
        seg_rms = np.mean(features['rms'][sf:ef])
        seg_centroid = np.mean(features['centroid'][sf:ef])
        seg_onset = np.mean(features['onset_env'][sf:ef])
        seg_flatness = np.mean(features['flatness'][sf:ef])
        
        # Energy trend
        if ef - sf > 10:
            first_half_rms = np.mean(features['rms'][sf:sf+(ef-sf)//2])
            second_half_rms = np.mean(features['rms'][sf+(ef-sf)//2:ef])
            energy_trend = (second_half_rms - first_half_rms) / (first_half_rms + 1e-10)
        else:
            energy_trend = 0
        
        # Energy level (5 tiers)
        if seg_rms > rms_pct[4]:
            energy_level = 5
        elif seg_rms > rms_pct[3]:
            energy_level = 4
        elif seg_rms > rms_pct[2]:
            energy_level = 3
        elif seg_rms > rms_pct[1]:
            energy_level = 2
        else:
            energy_level = 1
        
        # Rhythm level
        if seg_onset > onset_pct[3]:
            rhythm_level = 4
        elif seg_onset > onset_pct[2]:
            rhythm_level = 3
        elif seg_onset > onset_pct[1]:
            rhythm_level = 2
        else:
            rhythm_level = 1
        
        is_bright = seg_centroid > centroid_pct[2]
        is_very_bright = seg_centroid > centroid_pct[3]
        
        # Vocal heuristic
        has_vocal = (
            centroid_pct[1] < seg_centroid < centroid_pct[3] and
            seg_flatness < flatness_pct[1] and
            2 <= energy_level <= 4
        )
        
        raw_segments.append({
            'start': start,
            'end': end,
            'duration': seg_dur,
            'energy_level': energy_level,
            'rhythm_level': rhythm_level,
            'energy_trend': energy_trend,
            'is_bright': is_bright,
            'is_very_bright': is_very_bright,
            'has_vocal': has_vocal,
            'position': start / duration
        })
    
    # Second pass: contextual classification
    segments = []
    n_segs = len(raw_segments)
    
    for i, seg in enumerate(raw_segments):
        prev_energy = raw_segments[i-1]['energy_level'] if i > 0 else 0
        next_energy = raw_segments[i+1]['energy_level'] if i < n_segs-1 else 0
        
        energy = seg['energy_level']
        rhythm = seg['rhythm_level']
        trend = seg['energy_trend']
        pos = seg['position']
        
        # Classification with context
        if pos < 0.05 and energy <= 2:
            seg_type = 'pre_intro'
            importance = 1.0
        elif pos > 0.92 and energy <= 3:
            seg_type = 'outro'
            importance = 1.0
        elif energy >= 4 and rhythm >= 3:
            if seg['is_very_bright']:
                seg_type = 'drop_high'
            elif not seg['is_bright']:
                seg_type = 'drop_bass'
            else:
                seg_type = 'drop'
            importance = 3.0
        elif energy >= 3 and next_energy >= 4 and trend > 0.1:
            seg_type = 'pre_drop'
            importance = 2.5
        elif trend > 0.15 and next_energy > energy:
            seg_type = 'buildup'
            importance = 2.3
        elif seg['has_vocal'] and rhythm >= 2:
            seg_type = 'vocal'
            importance = 2.2
        elif rhythm >= 3 and energy >= 3:
            seg_type = 'verse'
            importance = 2.0
        elif energy <= 2 and prev_energy >= 4:
            seg_type = 'break'
            importance = 1.5
        elif energy <= 1:
            seg_type = 'ambient'
            importance = 1.0
        elif energy <= 2 and pos < 0.3:
            seg_type = 'intro'
            importance = 1.2
        elif seg['is_bright'] and rhythm <= 2:
            seg_type = 'melodic'
            importance = 1.8
        else:
            seg_type = 'verse'
            importance = 1.5
        
        segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'duration': seg['duration'],
            'type': seg_type,
            'importance': importance
        })
    
    return segments


# Type icons for display
TYPE_ICONS = {
    'pre_intro': '🌅',
    'intro': '🎬',
    'buildup': '📈',
    'pre_drop': '⚡',
    'drop': '🔥',
    'drop_bass': '🎸',
    'drop_high': '✨',
    'verse': '🎤',
    'vocal': '🗣️',
    'melodic': '🎹',
    'break': '💫',
    'ambient': '🌊',
    'outro': '🌙'
}


print("=" * 60)
print("EXTENDED SEGMENT CLASSIFICATION TEST")
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
    
    print("\nExtracting features...")
    features = extract_features(y, sr)
    
    print("Finding boundaries...")
    boundaries = find_boundaries(y, sr, duration, min_segment_sec=8.0)
    print(f"Found {len(boundaries)-1} segments")
    
    print("\nClassifying segments...")
    segments = classify_segments_extended(features, boundaries, duration)
    
    print(f"\n--- SEGMENTS ---")
    for i, seg in enumerate(segments):
        icon = TYPE_ICONS.get(seg['type'], '❓')
        print(f"  {i+1:2}. {int(seg['start'])//60}:{int(seg['start'])%60:02d} - {int(seg['end'])//60}:{int(seg['end'])%60:02d}  ({seg['duration']:5.1f}s)  {icon} {seg['type']:12} imp={seg['importance']:.1f}")

cursor.close()
conn.close()

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
