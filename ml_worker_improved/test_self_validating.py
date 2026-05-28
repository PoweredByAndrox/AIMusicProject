#!/usr/bin/env python3
"""
Self-Validating Segment Analysis.

Uses MULTIPLE independent methods to create ground truth:
1. Energy-based segmentation (RMS envelope)
2. Spectral-based segmentation (spectral flux)
3. Structural segmentation (self-similarity matrix)
4. Beat-aligned segmentation (downbeats)
5. Harmonic analysis (chroma changes)

If 3+ methods agree on a boundary/type -> it's ground truth.
Then compare our main algorithm against this consensus.
"""

import sys
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 256


class Method1_EnergyBased:
    """Segment by energy (RMS) changes."""
    
    def analyze(self, y, sr):
        print("  Method 1: Energy-based analysis...")
        hop = HOP_LENGTH
        
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_smooth = uniform_filter1d(rms, size=20)
        
        # Find significant energy changes
        rms_diff = np.abs(np.diff(rms_smooth))
        rms_diff = uniform_filter1d(rms_diff, size=10)
        
        threshold = np.percentile(rms_diff, 85)
        peaks, _ = signal.find_peaks(rms_diff, height=threshold, distance=int(5*sr/hop))
        
        boundaries = [0.0]
        for p in peaks:
            t = p * hop / sr
            if t > 3:
                boundaries.append(t)
        boundaries.append(len(y)/sr)
        
        # Classify by energy level
        segments = []
        rms_pct = np.percentile(rms, [25, 50, 75])
        
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            sf = int(start * sr / hop)
            ef = min(int(end * sr / hop), len(rms))
            
            if sf >= ef:
                continue
                
            seg_rms = np.mean(rms[sf:ef])
            
            if seg_rms > rms_pct[2]:
                seg_type = 'high_energy'  # drop
            elif seg_rms > rms_pct[1]:
                seg_type = 'mid_energy'   # verse
            else:
                seg_type = 'low_energy'   # intro/break
            
            segments.append({
                'start': start, 'end': end, 'type': seg_type,
                'method': 'energy'
            })
        
        return segments


class Method2_SpectralFlux:
    """Segment by spectral changes."""
    
    def analyze(self, y, sr):
        print("  Method 2: Spectral flux analysis...")
        hop = HOP_LENGTH
        
        # Spectral flux
        spec = np.abs(librosa.stft(y, hop_length=hop))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        flux = np.concatenate([[0], flux])
        flux_smooth = uniform_filter1d(flux, size=20)
        
        # Spectral centroid for brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        
        # Find flux peaks
        threshold = np.percentile(flux_smooth, 80)
        peaks, _ = signal.find_peaks(flux_smooth, height=threshold, distance=int(5*sr/hop))
        
        boundaries = [0.0]
        for p in peaks:
            t = p * hop / sr
            if t > 3 and t < len(y)/sr - 3:
                boundaries.append(t)
        boundaries.append(len(y)/sr)
        
        # Classify by spectral characteristics
        segments = []
        centroid_pct = np.percentile(centroid, [33, 66])
        flux_pct = np.percentile(flux, [33, 66])
        
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            sf = int(start * sr / hop)
            ef = min(int(end * sr / hop), len(centroid))
            
            if sf >= ef:
                continue
            
            seg_centroid = np.mean(centroid[sf:ef])
            seg_flux = np.mean(flux[sf:ef])
            
            if seg_flux > flux_pct[1] and seg_centroid > centroid_pct[1]:
                seg_type = 'bright_active'  # drop_high
            elif seg_flux > flux_pct[1]:
                seg_type = 'dark_active'    # drop_bass
            elif seg_centroid > centroid_pct[1]:
                seg_type = 'bright_calm'    # melodic
            else:
                seg_type = 'dark_calm'      # ambient
            
            segments.append({
                'start': start, 'end': end, 'type': seg_type,
                'method': 'spectral'
            })
        
        return segments


class Method3_SelfSimilarity:
    """Segment by structural repetition (self-similarity matrix)."""
    
    def analyze(self, y, sr):
        print("  Method 3: Self-similarity matrix analysis...")
        hop = HOP_LENGTH * 4  # Coarser for structure
        
        # Compute features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        
        # Combine features
        features = np.vstack([
            librosa.util.normalize(chroma, axis=0),
            librosa.util.normalize(mfcc, axis=0)
        ])
        
        # Compute novelty directly from feature changes
        n = features.shape[1]
        novelty = np.zeros(n)
        window = 10
        
        for i in range(window, n - window):
            before = features[:, i-window:i].mean(axis=1)
            after = features[:, i:i+window].mean(axis=1)
            norm_before = np.linalg.norm(before) + 1e-10
            norm_after = np.linalg.norm(after) + 1e-10
            similarity = np.dot(before, after) / (norm_before * norm_after)
            novelty[i] = 1 - similarity
        
        # Find structural boundaries
        threshold = np.percentile(novelty, 75)
        peaks, _ = signal.find_peaks(novelty, height=threshold, distance=int(8*sr/hop))
        
        boundaries = [0.0]
        for p in peaks:
            t = p * hop / sr
            if t > 5 and t < len(y)/sr - 5:
                boundaries.append(t)
        boundaries.append(len(y)/sr)
        
        # Classify by position (structural role)
        segments = []
        duration = len(y) / sr
        
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            pos = start / duration
            
            if pos < 0.15:
                seg_type = 'intro_section'
            elif pos > 0.85:
                seg_type = 'outro_section'
            elif 0.4 < pos < 0.6:
                seg_type = 'middle_section'  # often breakdown
            else:
                seg_type = 'main_section'
            
            segments.append({
                'start': start, 'end': end, 'type': seg_type,
                'method': 'structure'
            })
        
        return segments


class Method4_BeatAligned:
    """Segment aligned to musical bars/phrases."""
    
    def analyze(self, y, sr):
        print("  Method 4: Beat-aligned analysis...")
        
        # Get tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        print(f"    Tempo: {float(tempo):.0f} BPM, {len(beat_times)} beats")
        
        if len(beat_times) < 8:
            return []
        
        # Group into 8-bar phrases (typical EDM structure)
        beats_per_bar = 4
        bars_per_phrase = 8
        beats_per_phrase = beats_per_bar * bars_per_phrase
        
        boundaries = [0.0]
        for i in range(0, len(beat_times), beats_per_phrase):
            if i < len(beat_times):
                boundaries.append(beat_times[i])
        boundaries.append(len(y)/sr)
        boundaries = sorted(set(boundaries))
        
        # Get RMS for energy classification
        hop = HOP_LENGTH
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_pct = np.percentile(rms, [30, 60])
        
        segments = []
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            sf = int(start * sr / hop)
            ef = min(int(end * sr / hop), len(rms))
            
            if sf >= ef:
                continue
            
            seg_rms = np.mean(rms[sf:ef])
            
            # Check energy trend
            if ef - sf > 10:
                first = np.mean(rms[sf:sf+(ef-sf)//2])
                second = np.mean(rms[sf+(ef-sf)//2:ef])
                trend = (second - first) / (first + 1e-10)
            else:
                trend = 0
            
            if seg_rms > rms_pct[1]:
                seg_type = 'phrase_high'
            elif trend > 0.2:
                seg_type = 'phrase_rising'  # buildup
            elif seg_rms > rms_pct[0]:
                seg_type = 'phrase_mid'
            else:
                seg_type = 'phrase_low'
            
            segments.append({
                'start': start, 'end': end, 'type': seg_type,
                'method': 'beats'
            })
        
        return segments


class Method5_HarmonicPercussive:
    """Segment by harmonic vs percussive content."""
    
    def analyze(self, y, sr):
        print("  Method 5: Harmonic-Percussive analysis...")
        hop = HOP_LENGTH
        
        # Separate harmonic and percussive
        y_harm, y_perc = librosa.effects.hpss(y)
        
        harm_rms = librosa.feature.rms(y=y_harm, hop_length=hop)[0]
        perc_rms = librosa.feature.rms(y=y_perc, hop_length=hop)[0]
        
        # HP ratio
        hp_ratio = harm_rms / (perc_rms + 1e-10)
        hp_smooth = uniform_filter1d(hp_ratio, size=20)
        
        # Find significant changes in HP ratio
        hp_diff = np.abs(np.diff(hp_smooth))
        hp_diff = uniform_filter1d(hp_diff, size=10)
        
        threshold = np.percentile(hp_diff, 80)
        peaks, _ = signal.find_peaks(hp_diff, height=threshold, distance=int(5*sr/hop))
        
        boundaries = [0.0]
        for p in peaks:
            t = p * hop / sr
            if t > 3 and t < len(y)/sr - 3:
                boundaries.append(t)
        boundaries.append(len(y)/sr)
        
        # Classify by HP ratio
        segments = []
        hp_pct = np.percentile(hp_ratio, [33, 66])
        
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            sf = int(start * sr / hop)
            ef = min(int(end * sr / hop), len(hp_ratio))
            
            if sf >= ef:
                continue
            
            seg_hp = np.mean(hp_ratio[sf:ef])
            
            if seg_hp > hp_pct[1]:
                seg_type = 'harmonic'  # vocal/melodic
            elif seg_hp < hp_pct[0]:
                seg_type = 'percussive'  # drums/bass
            else:
                seg_type = 'balanced'
            
            segments.append({
                'start': start, 'end': end, 'type': seg_type,
                'method': 'hp_ratio'
            })
        
        return segments


def find_consensus_boundaries(all_segments, tolerance=3.0):
    """Find boundaries where multiple methods agree."""
    # Collect all boundaries
    all_boundaries = []
    for segments in all_segments:
        for seg in segments:
            all_boundaries.append(seg['start'])
            all_boundaries.append(seg['end'])
    
    # Cluster nearby boundaries
    all_boundaries = sorted(set(all_boundaries))
    
    clusters = []
    current_cluster = [all_boundaries[0]]
    
    for b in all_boundaries[1:]:
        if b - current_cluster[-1] < tolerance:
            current_cluster.append(b)
        else:
            clusters.append(current_cluster)
            current_cluster = [b]
    clusters.append(current_cluster)
    
    # Keep clusters with 3+ boundaries (consensus)
    consensus_boundaries = [0.0]
    for cluster in clusters:
        if len(cluster) >= 3:  # At least 3 methods agree
            consensus_boundaries.append(np.mean(cluster))
    
    return sorted(set(consensus_boundaries))


def classify_consensus_segment(start, end, all_segments, y, sr):
    """Classify a segment based on consensus of methods."""
    hop = HOP_LENGTH
    mid = (start + end) / 2
    
    # Collect votes from each method
    votes = defaultdict(int)
    
    for segments in all_segments:
        for seg in segments:
            if seg['start'] <= mid <= seg['end']:
                # Map method-specific types to universal types
                seg_type = seg['type']
                
                # Energy-based mapping
                if seg_type == 'high_energy':
                    votes['drop'] += 1
                elif seg_type == 'mid_energy':
                    votes['verse'] += 1
                elif seg_type == 'low_energy':
                    votes['ambient'] += 1
                
                # Spectral mapping
                elif seg_type == 'bright_active':
                    votes['drop_high'] += 1
                elif seg_type == 'dark_active':
                    votes['drop_bass'] += 1
                elif seg_type == 'bright_calm':
                    votes['melodic'] += 1
                elif seg_type == 'dark_calm':
                    votes['ambient'] += 1
                
                # Structure mapping
                elif seg_type == 'intro_section':
                    votes['intro'] += 1
                elif seg_type == 'outro_section':
                    votes['outro'] += 1
                elif seg_type == 'middle_section':
                    votes['breakdown'] += 1
                elif seg_type == 'main_section':
                    votes['verse'] += 1
                
                # Beat mapping
                elif seg_type == 'phrase_high':
                    votes['drop'] += 1
                elif seg_type == 'phrase_rising':
                    votes['buildup'] += 1
                elif seg_type == 'phrase_mid':
                    votes['verse'] += 1
                elif seg_type == 'phrase_low':
                    votes['ambient'] += 1
                
                # HP mapping
                elif seg_type == 'harmonic':
                    votes['vocal'] += 1
                elif seg_type == 'percussive':
                    votes['drop_bass'] += 1
                elif seg_type == 'balanced':
                    votes['verse'] += 1
                
                break
    
    # Get winning type
    if votes:
        winner = max(votes.items(), key=lambda x: x[1])
        return winner[0], winner[1]
    
    return 'unknown', 0


def run_self_validation(audio_path):
    """Run all methods and find consensus."""
    print(f"\nLoading: {audio_path}")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")
    
    print("\n" + "="*60)
    print("RUNNING 5 INDEPENDENT ANALYSIS METHODS")
    print("="*60)
    
    # Run all methods
    methods = [
        Method1_EnergyBased(),
        Method2_SpectralFlux(),
        Method3_SelfSimilarity(),
        Method4_BeatAligned(),
        Method5_HarmonicPercussive(),
    ]
    
    all_segments = []
    for method in methods:
        segments = method.analyze(y, sr)
        all_segments.append(segments)
        print(f"    Found {len(segments)} segments")
    
    print("\n" + "="*60)
    print("FINDING CONSENSUS (3+ methods must agree)")
    print("="*60)
    
    # Find consensus boundaries
    consensus_bounds = find_consensus_boundaries(all_segments, tolerance=3.0)
    consensus_bounds.append(duration)
    consensus_bounds = sorted(set(consensus_bounds))
    
    print(f"Consensus boundaries: {len(consensus_bounds)-1} segments")
    
    # Classify each consensus segment
    consensus_segments = []
    for i in range(len(consensus_bounds)-1):
        start, end = consensus_bounds[i], consensus_bounds[i+1]
        seg_type, confidence = classify_consensus_segment(
            start, end, all_segments, y, sr
        )
        consensus_segments.append({
            'start': start,
            'end': end,
            'type': seg_type,
            'confidence': confidence
        })
    
    print("\n" + "="*60)
    print("CONSENSUS GROUND TRUTH")
    print("="*60)
    
    for i, seg in enumerate(consensus_segments):
        conf_str = f"({seg['confidence']} votes)"
        print(f"  {i+1:2}. {int(seg['start'])//60}:{int(seg['start'])%60:02d} - "
              f"{int(seg['end'])//60}:{int(seg['end'])%60:02d}  "
              f"{seg['type']:12} {conf_str}")
    
    return consensus_segments, all_segments


if __name__ == "__main__":
    print("="*70)
    print("SELF-VALIDATING SEGMENT ANALYSIS")
    print("No human input needed - uses consensus of 5 methods")
    print("="*70)
    
    import psycopg2
    import os
    
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        database=os.getenv('DB_NAME', 'music_recommendations'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'secret')
    )
    cursor = conn.cursor()
    
    # Get a test track
    cursor.execute("""
        SELECT id, title, artist, file_path 
        FROM tracks 
        WHERE is_liked = true AND file_path IS NOT NULL AND duration < 480
        ORDER BY id LIMIT 1
    """)
    track = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if track:
        track_id, title, artist, file_path = track
        audio_path = f"/app/storage/{file_path}"
        
        print(f"\nTest track: {title} by {artist}")
        
        if Path(audio_path).exists():
            consensus, all_methods = run_self_validation(audio_path)
            
            print("\n" + "="*70)
            print("VALIDATION COMPLETE")
            print("="*70)
            print(f"Generated {len(consensus)} ground truth segments")
            print("This can now be used to validate the main algorithm!")
        else:
            print(f"File not found: {audio_path}")
    else:
        print("No tracks found!")
