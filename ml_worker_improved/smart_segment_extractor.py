#!/usr/bin/env python3
"""
Smart Music Segmentation - Adaptive pattern extraction.
Analyzes track structure and extracts multiple pattern types.
"""

import os
import sys
import time
import psycopg2
from pathlib import Path
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("УМНАЯ СЕГМЕНТАЦИЯ ТРЕКОВ")
print("=" * 60)

# GPU check
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    device = None

try:
    import librosa
    print("✓ librosa")
except ImportError:
    print("✗ librosa")
    sys.exit(1)

try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    print("✓ scipy")
except ImportError:
    print("✗ scipy")
    sys.exit(1)

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
PATTERNS_PATH = STORAGE_PATH / 'patterns'
PATTERNS_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 512


def progress_bar(current, total, prefix='', suffix='', width=30):
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} {suffix}", end='')
    if current == total:
        print()


def format_time(seconds: float) -> str:
    """Format seconds as M:SS"""
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m}:{s:02d}"


class SmartSegmenter:
    """Intelligent music segmentation based on structure analysis."""
    
    def __init__(self, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
        self.sr = sr
        self.hop_length = hop_length
    
    def analyze_track(self, audio_path: str) -> dict:
        """
        Analyze track and return segments with types.
        Returns dict with segments, features, and metadata.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            duration = len(y) / sr
            
            if duration < 30:
                return None
            
            # Extract features
            features = self._extract_features(y, sr)
            
            # Find segment boundaries
            boundaries = self._find_boundaries(y, sr, features)
            
            # Classify segments
            segments = self._classify_segments(y, sr, features, boundaries, duration)
            
            return {
                'duration': duration,
                'segments': segments,
                'features': features
            }
        except Exception as e:
            print(f"\n  Error analyzing: {e}")
            return None
    
    def _extract_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract audio features for segmentation."""
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Spectral contrast (texture)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        contrast_mean = np.mean(contrast, axis=0)
        
        # Onset strength (rhythmic activity)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Chroma (harmonic content)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        
        # Spectral flatness (noise vs tonal)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        
        return {
            'rms': rms,
            'centroid': centroid,
            'contrast': contrast_mean,
            'onset_env': onset_env,
            'chroma': chroma,
            'tempo': tempo,
            'beats': beats,
            'flatness': flatness
        }
    
    def _find_boundaries(self, y: np.ndarray, sr: int, features: dict) -> list:
        """Find segment boundaries using novelty detection."""
        # Compute novelty curve from multiple features
        n_frames = len(features['rms'])
        
        # Normalize features
        rms_norm = (features['rms'] - features['rms'].min()) / (features['rms'].max() - features['rms'].min() + 1e-10)
        centroid_norm = (features['centroid'] - features['centroid'].min()) / (features['centroid'].max() - features['centroid'].min() + 1e-10)
        contrast_norm = (features['contrast'] - features['contrast'].min()) / (features['contrast'].max() - features['contrast'].min() + 1e-10)
        
        # Compute self-similarity matrix from chroma
        chroma = features['chroma']
        chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
        
        # Novelty from feature changes
        novelty = np.zeros(n_frames)
        
        # RMS change
        rms_diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
        novelty += uniform_filter1d(rms_diff, size=10) * 0.3
        
        # Centroid change
        centroid_diff = np.abs(np.diff(centroid_norm, prepend=centroid_norm[0]))
        novelty += uniform_filter1d(centroid_diff, size=10) * 0.2
        
        # Contrast change
        contrast_diff = np.abs(np.diff(contrast_norm, prepend=contrast_norm[0]))
        novelty += uniform_filter1d(contrast_diff, size=10) * 0.2
        
        # Chroma change (harmonic novelty)
        chroma_diff = np.sum(np.abs(np.diff(chroma_norm, axis=1)), axis=0)
        chroma_diff = np.concatenate([[0], chroma_diff])
        novelty += uniform_filter1d(chroma_diff, size=10) * 0.3
        
        # Smooth novelty
        novelty = uniform_filter1d(novelty, size=20)
        
        # Find peaks in novelty curve
        # Minimum distance between boundaries: ~4 seconds
        min_distance = int(4 * sr / self.hop_length)
        
        # Adaptive threshold based on novelty distribution
        threshold = np.percentile(novelty, 75)
        
        peaks, _ = signal.find_peaks(novelty, height=threshold, distance=min_distance)
        
        # Convert to time
        boundaries = [0]  # Start
        for peak in peaks:
            time_sec = peak * self.hop_length / sr
            boundaries.append(time_sec)
        
        duration = len(y) / sr
        boundaries.append(duration)  # End
        
        return sorted(set(boundaries))
    
    def _classify_segments(self, y: np.ndarray, sr: int, features: dict, 
                          boundaries: list, duration: float) -> list:
        """Classify each segment by type."""
        segments = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            seg_duration = end - start
            
            # Skip very short segments
            if seg_duration < 3:
                continue
            
            # Get feature values for this segment
            start_frame = int(start * sr / self.hop_length)
            end_frame = int(end * sr / self.hop_length)
            end_frame = min(end_frame, len(features['rms']))
            
            if start_frame >= end_frame:
                continue
            
            seg_rms = np.mean(features['rms'][start_frame:end_frame])
            seg_centroid = np.mean(features['centroid'][start_frame:end_frame])
            seg_contrast = np.mean(features['contrast'][start_frame:end_frame])
            seg_onset = np.mean(features['onset_env'][start_frame:end_frame])
            seg_flatness = np.mean(features['flatness'][start_frame:end_frame])
            
            # Normalize relative to track
            rms_percentile = np.percentile(features['rms'], [25, 50, 75])
            centroid_percentile = np.percentile(features['centroid'], [25, 50, 75])
            
            # Classify segment type
            seg_type = self._determine_segment_type(
                seg_rms, seg_centroid, seg_contrast, seg_onset, seg_flatness,
                rms_percentile, centroid_percentile, seg_duration
            )
            
            # Calculate segment importance (how distinctive it is)
            importance = self._calculate_importance(
                seg_rms, seg_centroid, seg_onset, 
                rms_percentile, seg_type
            )
            
            segments.append({
                'start': start,
                'end': end,
                'duration': seg_duration,
                'type': seg_type,
                'importance': importance,
                'features': {
                    'energy': float(seg_rms),
                    'brightness': float(seg_centroid),
                    'contrast': float(seg_contrast),
                    'rhythm': float(seg_onset),
                    'noise': float(seg_flatness)
                }
            })
        
        return segments
    
    def _determine_segment_type(self, rms, centroid, contrast, onset, flatness,
                                rms_pct, centroid_pct, duration) -> str:
        """Determine segment type based on features."""
        # Normalize values relative to track percentiles
        rms_level = 'high' if rms > rms_pct[2] else ('mid' if rms > rms_pct[1] else 'low')
        centroid_level = 'high' if centroid > centroid_pct[2] else ('mid' if centroid > centroid_pct[1] else 'low')
        
        # High energy = DROP
        if rms_level == 'high':
            if centroid_level == 'high':
                return 'drop_high'  # Aggressive/bright drop
            else:
                return 'drop_bass'  # Bass-heavy drop
        
        # Medium energy
        if rms_level == 'mid':
            if centroid_level == 'high':
                return 'melodic'    # Melodic section
            elif contrast > np.median([contrast]):
                return 'buildup'    # Building tension
            else:
                return 'verse'      # Main verse
        
        # Low energy
        if rms_level == 'low':
            if centroid_level == 'high':
                return 'break'      # Melodic break
            elif duration < 15:
                return 'intro'      # Intro/outro
            else:
                return 'ambient'    # Atmospheric
        
        return 'verse'
    
    def _calculate_importance(self, rms, centroid, onset, rms_pct, seg_type) -> float:
        """Calculate how important/distinctive this segment is."""
        importance = 1.0
        
        # Drops are most important
        if seg_type in ['drop_high', 'drop_low']:
            importance = 3.0
        elif seg_type == 'buildup':
            importance = 2.0
        elif seg_type == 'melodic':
            importance = 2.0
        elif seg_type in ['intro', 'outro', 'ambient']:
            importance = 0.5
        
        # Boost for high energy
        if rms > rms_pct[2]:
            importance *= 1.2
        
        return min(importance, 3.0)
    
    def extract_chroma_segment(self, audio_path: str, start: float, end: float) -> np.ndarray:
        """Extract chroma features for a segment."""
        try:
            duration = end - start
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True,
                                offset=start, duration=duration)
            
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            
            # Normalize
            norms = np.linalg.norm(chroma, axis=0, keepdims=True)
            norms[norms < 1e-10] = 1.0
            chroma = chroma / norms
            
            return chroma.astype(np.float32)
        except Exception as e:
            return None


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: ОЧИСТКА СТАРЫХ AUTO-ПАТТЕРНОВ")
print("=" * 60)

start_time = time.time()

cursor.execute("""
    DELETE FROM pattern_matches WHERE pattern_id IN (
        SELECT id FROM patterns WHERE name LIKE '%[auto]%'
    )
""")
cursor.execute("DELETE FROM pattern_examples WHERE pattern_id IN (SELECT id FROM patterns WHERE name LIKE '%[auto]%')")
cursor.execute("DELETE FROM patterns WHERE name LIKE '%[auto]%'")
conn.commit()
print("✓ Старые auto-паттерны удалены")

print("\n" + "=" * 60)
print("STEP 2: ЗАГРУЗКА LIKED ТРЕКОВ")
print("=" * 60)

cursor.execute("""
    SELECT id, title, artist, file_path
    FROM tracks WHERE is_liked = true AND file_path IS NOT NULL
    ORDER BY id
""")
liked_tracks = cursor.fetchall()
print(f"Liked треков: {len(liked_tracks)}")

print("\n" + "=" * 60)
print("STEP 3: УМНАЯ СЕГМЕНТАЦИЯ")
print("=" * 60)

segmenter = SmartSegmenter()
patterns_created = 0
tracks_processed = 0
tracks_failed = 0

segment_type_counts = {}

for i, (track_id, title, artist, file_path) in enumerate(liked_tracks):
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        tracks_failed += 1
        progress_bar(i + 1, len(liked_tracks), "  Progress", f"ok:{tracks_processed} fail:{tracks_failed} patterns:{patterns_created}")
        continue
    
    # Analyze track structure
    analysis = segmenter.analyze_track(audio_path)
    
    if analysis is None or not analysis['segments']:
        tracks_failed += 1
        progress_bar(i + 1, len(liked_tracks), "  Progress", f"ok:{tracks_processed} fail:{tracks_failed} patterns:{patterns_created}")
        continue
    
    tracks_processed += 1
    
    # Create patterns for important segments
    for seg in analysis['segments']:
        # Skip only very low-importance segments
        if seg['importance'] < 1.0:
            continue
        
        # Extract chroma for segment
        chroma = segmenter.extract_chroma_segment(audio_path, seg['start'], seg['end'])
        
        if chroma is None or chroma.shape[1] < 20:
            continue
        
        # Create pattern name with type and time
        time_str = format_time(seg['start'])
        seg_type = seg['type']
        pattern_name = f"[auto:{seg_type}] {artist or 'Unknown'} - {title[:25]} ({time_str})"
        
        # Track segment types
        segment_type_counts[seg_type] = segment_type_counts.get(seg_type, 0) + 1
        
        cursor.execute("""
            INSERT INTO patterns (name, representation_type, transpose_invariant, tempo_invariant,
                                  match_threshold, importance, created_at, updated_at)
            VALUES (%s, 'chroma', true, false, 0.70, %s, %s, %s)
            RETURNING id
        """, (pattern_name, seg['importance'], datetime.now(), datetime.now()))
        
        pattern_id = cursor.fetchone()[0]
        
        np.save(PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy", chroma)
        
        cursor.execute("""
            INSERT INTO pattern_examples (pattern_id, track_id, start_time, end_time,
                                          is_positive, notes, created_at, updated_at)
            VALUES (%s, %s, %s, %s, true, %s, %s, %s)
        """, (pattern_id, track_id, seg['start'], seg['end'],
              f"type={seg_type}, importance={seg['importance']:.1f}", 
              datetime.now(), datetime.now()))
        
        patterns_created += 1
    
    conn.commit()
    
    progress_bar(i + 1, len(liked_tracks), "  Progress", f"ok:{tracks_processed} fail:{tracks_failed} patterns:{patterns_created}")
    
    if TORCH_AVAILABLE and (i + 1) % 10 == 0:
        torch.cuda.empty_cache()

print()

# =============================================================================
# RESULTS
# =============================================================================

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)

print(f"\n--- СТАТИСТИКА ---")
print(f"Треков обработано: {tracks_processed}")
print(f"Треков пропущено:  {tracks_failed}")
print(f"Паттернов создано: {patterns_created}")
print(f"Время: {elapsed:.1f}s")

print(f"\n--- ТИПЫ СЕГМЕНТОВ ---")
for seg_type, count in sorted(segment_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {seg_type:12} {count:4}")

cursor.execute("SELECT COUNT(*) FROM patterns")
total = cursor.fetchone()[0]
print(f"\nВсего паттернов в БД: {total}")

# Show sample patterns
cursor.execute("SELECT id, name FROM patterns WHERE name LIKE '%[auto:%' ORDER BY id DESC LIMIT 15")
print("\n--- ПРИМЕРЫ ПАТТЕРНОВ ---")
for pid, name in cursor.fetchall():
    print(f"  #{pid}: {name[:60]}")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)

cursor.close()
conn.close()
