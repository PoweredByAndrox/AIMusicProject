#!/usr/bin/env python3
"""
Advanced segment classification with:
1. ML-like vocal detection using harmonic-percussive separation
2. Harmonic analysis for buildup detection
3. Breakdown type for mid-track pauses
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


class AdvancedFeatureExtractor:
    """Extract advanced features for segment classification."""
    
    def __init__(self, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
        self.sr = sr
        self.hop = hop_length
    
    def extract_all(self, y):
        """Extract all features including harmonic/percussive separation."""
        # Basic features
        rms = librosa.feature.rms(y=y, hop_length=self.hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop)[0]
        
        # Harmonic-Percussive separation for vocal detection
        print("  Separating harmonic/percussive components...")
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Harmonic features (vocals are mostly harmonic)
        harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=self.hop)[0]
        harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=self.sr, hop_length=self.hop)[0]
        
        # Percussive features (drums/bass)
        percussive_rms = librosa.feature.rms(y=y_percussive, hop_length=self.hop)[0]
        
        # Harmonic-to-percussive ratio (high = more melodic/vocal)
        hp_ratio = harmonic_rms / (percussive_rms + 1e-10)
        
        # Chroma for harmonic analysis
        print("  Computing chroma features...")
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop)
        
        # Chroma variance (high = complex harmony, low = simple/repetitive)
        chroma_var = np.var(chroma, axis=0)
        
        # Spectral bandwidth (vocals have specific bandwidth)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, hop_length=self.hop)[0]
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, hop_length=self.hop)[0]
        
        # Zero crossing rate (high for noisy/percussive, low for tonal)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop)[0]
        
        return {
            'rms': rms,
            'centroid': centroid,
            'onset_env': onset_env,
            'flatness': flatness,
            'harmonic_rms': harmonic_rms,
            'harmonic_centroid': harmonic_centroid,
            'percussive_rms': percussive_rms,
            'hp_ratio': hp_ratio,
            'chroma': chroma,
            'chroma_var': chroma_var,
            'bandwidth': bandwidth,
            'rolloff': rolloff,
            'zcr': zcr
        }


class AdvancedSegmentClassifier:
    """
    Advanced segment classifier with:
    - ML-like vocal detection
    - Harmonic buildup analysis
    - Breakdown detection
    
    Types:
    - pre_intro: Very beginning, quiet
    - intro: Main introduction
    - buildup: Energy/harmony rising
    - pre_drop: Final buildup before drop
    - drop: Main high-energy section
    - drop_bass: Bass-heavy drop
    - drop_high: Bright/melodic drop
    - verse: Rhythmic section
    - vocal: Section with detected vocals
    - melodic: Melodic without strong rhythm
    - breakdown: Mid-track low-energy pause
    - break: Post-drop pause
    - ambient: Very quiet atmospheric
    - outro: Ending section
    """
    
    def __init__(self, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
        self.sr = sr
        self.hop = hop_length
    
    def detect_vocals(self, features, start_frame, end_frame):
        """
        Detect if segment contains vocals using multiple heuristics.
        
        Vocal characteristics:
        1. High harmonic-to-percussive ratio
        2. Centroid in vocal range (300-3000 Hz normalized)
        3. Low spectral flatness (tonal, not noisy)
        4. Moderate bandwidth
        """
        if start_frame >= end_frame:
            return False, 0.0
        
        # Get segment features
        hp_ratio = np.mean(features['hp_ratio'][start_frame:end_frame])
        harmonic_rms = np.mean(features['harmonic_rms'][start_frame:end_frame])
        total_rms = np.mean(features['rms'][start_frame:end_frame])
        flatness = np.mean(features['flatness'][start_frame:end_frame])
        centroid = np.mean(features['centroid'][start_frame:end_frame])
        bandwidth = np.mean(features['bandwidth'][start_frame:end_frame])
        zcr = np.mean(features['zcr'][start_frame:end_frame])
        
        # Compute percentiles for normalization
        hp_pct = np.percentile(features['hp_ratio'], [30, 70])
        flatness_pct = np.percentile(features['flatness'], [30, 70])
        centroid_pct = np.percentile(features['centroid'], [20, 40, 60, 80])
        bandwidth_pct = np.percentile(features['bandwidth'], [30, 70])
        
        # Vocal score (0-1)
        score = 0.0
        
        # High harmonic ratio = more vocal-like
        if hp_ratio > hp_pct[1]:
            score += 0.3
        elif hp_ratio > hp_pct[0]:
            score += 0.15
        
        # Low flatness = tonal (vocal-like)
        if flatness < flatness_pct[0]:
            score += 0.25
        elif flatness < flatness_pct[1]:
            score += 0.1
        
        # Centroid in mid-range (vocal range)
        if centroid_pct[1] < centroid < centroid_pct[2]:
            score += 0.25
        elif centroid_pct[0] < centroid < centroid_pct[3]:
            score += 0.1
        
        # Moderate bandwidth
        if bandwidth_pct[0] < bandwidth < bandwidth_pct[1]:
            score += 0.2
        
        # Harmonic content should be significant portion of total
        harmonic_ratio = harmonic_rms / (total_rms + 1e-10)
        if harmonic_ratio > 0.5:
            score += 0.1
        
        # Low ZCR = more tonal
        zcr_pct = np.percentile(features['zcr'], [40])
        if zcr < zcr_pct[0]:
            score += 0.1
        
        return score > 0.5, score
    
    def detect_buildup(self, features, start_frame, end_frame, next_energy):
        """
        Detect if segment is a buildup using harmonic analysis.
        
        Buildup characteristics:
        1. Rising energy trend
        2. Increasing harmonic complexity (chroma variance)
        3. Next segment has higher energy
        """
        if start_frame >= end_frame or end_frame - start_frame < 10:
            return False, 0.0
        
        # Energy trend
        rms = features['rms'][start_frame:end_frame]
        first_half = np.mean(rms[:len(rms)//2])
        second_half = np.mean(rms[len(rms)//2:])
        energy_trend = (second_half - first_half) / (first_half + 1e-10)
        
        # Chroma variance trend (harmonic complexity)
        chroma_var = features['chroma_var'][start_frame:end_frame]
        cv_first = np.mean(chroma_var[:len(chroma_var)//2])
        cv_second = np.mean(chroma_var[len(chroma_var)//2:])
        harmony_trend = (cv_second - cv_first) / (cv_first + 1e-10)
        
        # Current energy
        current_energy = np.mean(rms)
        rms_pct = np.percentile(features['rms'], [50, 70])
        
        score = 0.0
        
        # Rising energy
        if energy_trend > 0.1:
            score += 0.3
        elif energy_trend > 0.05:
            score += 0.15
        
        # Rising harmonic complexity
        if harmony_trend > 0.1:
            score += 0.25
        elif harmony_trend > 0.05:
            score += 0.1
        
        # Next segment is higher energy
        if next_energy > current_energy * 1.2:
            score += 0.3
        elif next_energy > current_energy:
            score += 0.15
        
        # Not already at peak energy
        if current_energy < rms_pct[1]:
            score += 0.15
        
        return score > 0.5, score
    
    def classify_segments(self, features, boundaries, duration):
        """Classify all segments with advanced analysis."""
        
        # Compute global percentiles
        rms_pct = np.percentile(features['rms'], [10, 25, 45, 65, 85])
        onset_pct = np.percentile(features['onset_env'], [20, 40, 60, 80])
        centroid_pct = np.percentile(features['centroid'], [20, 40, 60, 80])
        
        # First pass: extract raw segment data
        raw_segments = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            seg_dur = end - start
            
            if seg_dur < 3:
                continue
            
            sf = int(start * self.sr / self.hop)
            ef = min(int(end * self.sr / self.hop), len(features['rms']))
            
            if sf >= ef:
                continue
            
            seg_rms = np.mean(features['rms'][sf:ef])
            seg_onset = np.mean(features['onset_env'][sf:ef])
            seg_centroid = np.mean(features['centroid'][sf:ef])
            
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
            
            # Brightness
            is_bright = seg_centroid > centroid_pct[2]
            is_very_bright = seg_centroid > centroid_pct[3]
            
            # Vocal detection
            has_vocal, vocal_score = self.detect_vocals(features, sf, ef)
            
            raw_segments.append({
                'start': start,
                'end': end,
                'duration': seg_dur,
                'sf': sf,
                'ef': ef,
                'energy_level': energy_level,
                'rhythm_level': rhythm_level,
                'is_bright': is_bright,
                'is_very_bright': is_very_bright,
                'has_vocal': has_vocal,
                'vocal_score': vocal_score,
                'position': start / duration,
                'rms': seg_rms
            })
        
        # Second pass: contextual classification
        segments = []
        n_segs = len(raw_segments)
        
        for i, seg in enumerate(raw_segments):
            prev_energy = raw_segments[i-1]['energy_level'] if i > 0 else 0
            next_energy = raw_segments[i+1]['energy_level'] if i < n_segs-1 else 0
            next_rms = raw_segments[i+1]['rms'] if i < n_segs-1 else 0
            
            energy = seg['energy_level']
            rhythm = seg['rhythm_level']
            pos = seg['position']
            
            # Buildup detection with harmonic analysis
            is_buildup, buildup_score = self.detect_buildup(
                features, seg['sf'], seg['ef'], next_rms
            )
            
            # Classification logic
            if pos < 0.04 and energy <= 2:
                seg_type = 'pre_intro'
                importance = 1.0
            elif pos > 0.93 and energy <= 3:
                seg_type = 'outro'
                importance = 1.0
            elif energy >= 4 and rhythm >= 3:
                # High energy drop
                if seg['is_very_bright']:
                    seg_type = 'drop_high'
                elif not seg['is_bright']:
                    seg_type = 'drop_bass'
                else:
                    seg_type = 'drop'
                importance = 3.0
            elif is_buildup and next_energy >= 4:
                # Buildup right before drop = pre_drop
                seg_type = 'pre_drop'
                importance = 2.5
            elif is_buildup:
                # General buildup
                seg_type = 'buildup'
                importance = 2.3
            elif seg['has_vocal'] and rhythm >= 2:
                # Vocal section
                seg_type = 'vocal'
                importance = 2.2
            elif energy <= 2 and 0.3 < pos < 0.7 and prev_energy >= 3:
                # Low energy in middle after higher section = breakdown
                seg_type = 'breakdown'
                importance = 1.8
            elif rhythm >= 3 and energy >= 3:
                seg_type = 'verse'
                importance = 2.0
            elif energy <= 2 and prev_energy >= 4:
                # Low energy right after drop = break
                seg_type = 'break'
                importance = 1.5
            elif energy <= 1:
                seg_type = 'ambient'
                importance = 1.0
            elif energy <= 2 and pos < 0.25:
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
                'importance': importance,
                'vocal_score': seg['vocal_score'],
                'buildup_score': buildup_score if is_buildup else 0
            })
        
        return segments


def find_boundaries(y, sr, duration, min_segment_sec=8.0):
    """Find segment boundaries using multi-scale novelty."""
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


# Type icons
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
    'breakdown': '🔻',
    'break': '💫',
    'ambient': '🌊',
    'outro': '🌙'
}


print("=" * 70)
print("ADVANCED SEGMENT CLASSIFICATION")
print("With ML-like vocal detection + harmonic buildup analysis")
print("=" * 70)

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

extractor = AdvancedFeatureExtractor()
classifier = AdvancedSegmentClassifier()

for track_id, title, artist, file_path, db_duration in tracks:
    audio_path = f"/app/storage/{file_path}"
    
    print(f"\n{'='*70}")
    print(f"Track: {title}")
    print(f"Artist: {artist}")
    print(f"{'='*70}")
    
    if not Path(audio_path).exists():
        print(f"  File not found!")
        continue
    
    print(f"\nLoading audio...")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")
    
    print("\nExtracting advanced features...")
    features = extractor.extract_all(y)
    
    print("\nFinding boundaries...")
    boundaries = find_boundaries(y, sr, duration, min_segment_sec=8.0)
    print(f"Found {len(boundaries)-1} segments")
    
    print("\nClassifying with advanced analysis...")
    segments = classifier.classify_segments(features, boundaries, duration)
    
    print(f"\n{'─'*70}")
    print(f"{'#':>3} {'Time':^15} {'Dur':>6} {'Type':^14} {'Imp':>4} {'Vocal':>6} {'Build':>6}")
    print(f"{'─'*70}")
    
    for i, seg in enumerate(segments):
        icon = TYPE_ICONS.get(seg['type'], '❓')
        time_str = f"{int(seg['start'])//60}:{int(seg['start'])%60:02d} - {int(seg['end'])//60}:{int(seg['end'])%60:02d}"
        vocal_str = f"{seg['vocal_score']:.2f}" if seg['vocal_score'] > 0 else "-"
        build_str = f"{seg['buildup_score']:.2f}" if seg['buildup_score'] > 0 else "-"
        
        print(f"{i+1:3} {time_str:^15} {seg['duration']:5.1f}s {icon} {seg['type']:<10} {seg['importance']:4.1f} {vocal_str:>6} {build_str:>6}")

cursor.close()
conn.close()

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
