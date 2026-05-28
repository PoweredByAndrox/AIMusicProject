#!/usr/bin/env python3
"""
ML-based Music Structure Analyzer using All-In-One model.

This replaces the heuristic-based segment classification with a neural network
that achieves state-of-the-art accuracy on music structure analysis.

Model: All-In-One (Taejun Kim et al.)
Paper: "All-In-One Metrical And Functional Structure Analysis With 
        Neighborhood Attentions on Demixed Audio"
GitHub: https://github.com/mir-aidj/all-in-one

Predicts:
- Tempo (BPM)
- Beats and Downbeats
- Functional segment boundaries
- Functional segment labels (intro, verse, chorus, bridge, outro, etc.)
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
print("ML MUSIC STRUCTURE ANALYZER")
print("Using All-In-One Neural Network Model")
print("=" * 60)

# Check if allin1 is installed
try:
    import allin1
    print("✓ allin1 package loaded")
except ImportError:
    print("✗ allin1 not installed!")
    print("\nTo install:")
    print("  1. pip install torch (with CUDA support)")
    print("  2. pip install natten -f https://shi-labs.com/natten/wheels/cu121/torch2.1/index.html")
    print("  3. pip install allin1")
    sys.exit(1)

import torch
if torch.cuda.is_available():
    device = 'cuda'
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("⚠ CUDA not available, using CPU (slower)")

try:
    import librosa
    print("✓ librosa")
except ImportError:
    print("✗ librosa not installed")
    sys.exit(1)

# Database connection
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
N_MELS = 128


def progress_bar(current, total, prefix='', suffix='', width=30):
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} {suffix}", end='')
    if current == total:
        print()


def format_time(seconds: float) -> str:
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m}:{s:02d}"


def map_label_to_type(label: str) -> tuple:
    """
    Map allin1 labels to our segment types with importance scores.
    
    allin1 labels: start, end, intro, outro, verse, chorus, bridge, inst, solo, break
    Our types: pre_intro, intro, buildup, pre_drop, drop, drop_bass, drop_high, 
               verse, vocal, melodic, breakdown, break, ambient, outro
    """
    label_map = {
        'start': ('pre_intro', 1.0),
        'end': ('outro', 1.0),
        'intro': ('intro', 1.5),
        'outro': ('outro', 1.2),
        'verse': ('verse', 2.0),
        'chorus': ('drop', 3.0),  # In EDM, chorus = drop
        'bridge': ('breakdown', 1.8),
        'inst': ('melodic', 2.0),  # Instrumental section
        'solo': ('melodic', 2.2),
        'break': ('break', 1.5),
    }
    return label_map.get(label, ('verse', 1.5))


class MLSegmentAnalyzer:
    """
    ML-based music structure analyzer using All-In-One model.
    """
    
    def __init__(self, model_name='harmonix-all'):
        """
        Initialize the analyzer.
        
        Args:
            model_name: Model to use. Options:
                - 'harmonix-all': Ensemble of all 8 folds (best accuracy, slower)
                - 'harmonix-fold0' to 'harmonix-fold7': Individual fold models
        """
        self.model_name = model_name
        print(f"  Model: {model_name}")
    
    def analyze(self, audio_path: str) -> dict:
        """
        Analyze a track using the ML model.
        
        Returns:
            dict with keys:
                - duration: float
                - bpm: float
                - beats: list of beat times
                - downbeats: list of downbeat times
                - segments: list of segment dicts
                - is_mix: bool (detected based on duration)
        """
        try:
            # Run allin1 analysis
            result = allin1.analyze(
                paths=audio_path,
                model=self.model_name,
                device=device,
            )
            
            # Handle single file result
            if isinstance(result, list):
                result = result[0]
            
            # Get duration
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration = len(y) / sr
            
            # Convert segments to our format
            segments = []
            for seg in result.segments:
                seg_type, importance = map_label_to_type(seg.label)
                seg_dur = seg.end - seg.start
                
                if seg_dur < 2:  # Skip very short segments
                    continue
                
                # Extract chroma for pattern matching
                start_sample = int(seg.start * SAMPLE_RATE)
                end_sample = min(int(seg.end * SAMPLE_RATE), len(y))
                seg_audio = y[start_sample:end_sample]
                
                chroma = None
                if len(seg_audio) > SAMPLE_RATE:
                    chroma = librosa.feature.chroma_cqt(
                        y=seg_audio, sr=SAMPLE_RATE, hop_length=512
                    )
                    # Normalize
                    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
                    norms[norms < 1e-10] = 1.0
                    chroma = (chroma / norms).astype(np.float32)
                
                segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'duration': seg_dur,
                    'type': seg_type,
                    'label': seg.label,  # Original label
                    'importance': importance,
                    'chroma': chroma
                })
            
            return {
                'duration': duration,
                'bpm': result.bpm,
                'beats': result.beats,
                'downbeats': result.downbeats,
                'segments': segments,
                'is_mix': duration > 480  # >8 minutes = likely a mix
            }
            
        except Exception as e:
            print(f"\n  Error analyzing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return None


def analyze_single_track(audio_path: str, verbose: bool = True):
    """
    Analyze a single track and display results.
    """
    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}")
        return None
    
    analyzer = MLSegmentAnalyzer()
    
    if verbose:
        print(f"\nAnalyzing: {audio_path}")
    
    start_time = time.time()
    result = analyzer.analyze(audio_path)
    elapsed = time.time() - start_time
    
    if result is None:
        print("Analysis failed!")
        return None
    
    if verbose:
        print(f"\nDuration: {result['duration']/60:.1f} min")
        print(f"BPM: {result['bpm']:.1f}")
        print(f"Beats: {len(result['beats'])}")
        print(f"Downbeats: {len(result['downbeats'])}")
        print(f"Segments: {len(result['segments'])}")
        print(f"Time: {elapsed:.1f}s")
        
        print("\n" + "─" * 70)
        print(f"{'#':>3}  {'Time':^15}  {'Dur':>6}  {'Type':^12}  {'Label':^10}  {'Imp':>4}")
        print("─" * 70)
        
        for i, seg in enumerate(result['segments']):
            time_str = f"{format_time(seg['start'])} - {format_time(seg['end'])}"
            print(f"{i+1:3}  {time_str:^15}  {seg['duration']:5.1f}s  "
                  f"{seg['type']:^12}  {seg['label']:^10}  {seg['importance']:4.1f}")
        
        print("─" * 70)
    
    return result


def process_all_liked_tracks():
    """
    Process all liked tracks and create patterns.
    """
    print("\n" + "=" * 60)
    print("STEP 1: CLEANUP")
    print("=" * 60)
    
    cursor.execute("DELETE FROM pattern_matches WHERE pattern_id IN (SELECT id FROM patterns WHERE name LIKE '%[ml:%')")
    cursor.execute("DELETE FROM pattern_examples WHERE pattern_id IN (SELECT id FROM patterns WHERE name LIKE '%[ml:%')")
    cursor.execute("DELETE FROM patterns WHERE name LIKE '%[ml:%'")
    conn.commit()
    print("✓ Old ML patterns deleted")
    
    print("\n" + "=" * 60)
    print("STEP 2: LOADING TRACKS")
    print("=" * 60)
    
    cursor.execute("""
        SELECT id, title, artist, file_path
        FROM tracks WHERE is_liked = true AND file_path IS NOT NULL
        ORDER BY id
    """)
    liked_tracks = cursor.fetchall()
    print(f"Liked tracks: {len(liked_tracks)}")
    
    print("\n" + "=" * 60)
    print("STEP 3: ML SEGMENTATION")
    print("=" * 60)
    
    analyzer = MLSegmentAnalyzer()
    patterns_created = 0
    tracks_ok = 0
    tracks_fail = 0
    type_counts = {}
    
    start_time = time.time()
    
    for i, (track_id, title, artist, file_path) in enumerate(liked_tracks):
        audio_path = f"/app/storage/{file_path}"
        
        if not Path(audio_path).exists():
            tracks_fail += 1
            progress_bar(i + 1, len(liked_tracks), "  ML", 
                        f"ok:{tracks_ok} fail:{tracks_fail} pat:{patterns_created}")
            continue
        
        result = analyzer.analyze(audio_path)
        
        if result is None or not result['segments']:
            tracks_fail += 1
            progress_bar(i + 1, len(liked_tracks), "  ML", 
                        f"ok:{tracks_ok} fail:{tracks_fail} pat:{patterns_created}")
            continue
        
        tracks_ok += 1
        
        # Create patterns for each segment
        for seg in result['segments']:
            if seg['importance'] < 1.0 or seg['chroma'] is None:
                continue
            
            if seg['chroma'].shape[1] < 20:
                continue
            
            time_str = format_time(seg['start'])
            seg_type = seg['type']
            pattern_name = f"[ml:{seg_type}] {artist or '?'} - {title[:25]} ({time_str})"
            
            type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
            
            cursor.execute("""
                INSERT INTO patterns (name, representation_type, transpose_invariant, tempo_invariant,
                                      match_threshold, importance, created_at, updated_at)
                VALUES (%s, 'chroma', true, false, 0.70, %s, %s, %s)
                RETURNING id
            """, (pattern_name, seg['importance'], datetime.now(), datetime.now()))
            
            pattern_id = cursor.fetchone()[0]
            np.save(PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy", seg['chroma'])
            
            cursor.execute("""
                INSERT INTO pattern_examples (pattern_id, track_id, start_time, end_time,
                                              is_positive, notes, created_at, updated_at)
                VALUES (%s, %s, %s, %s, true, %s, %s, %s)
            """, (pattern_id, track_id, seg['start'], seg['end'],
                  f"type={seg_type},label={seg['label']}", datetime.now(), datetime.now()))
            
            patterns_created += 1
        
        conn.commit()
        progress_bar(i + 1, len(liked_tracks), "  ML", 
                    f"ok:{tracks_ok} fail:{tracks_fail} pat:{patterns_created}")
        
        # Clear GPU memory periodically
        if (i + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Tracks: {tracks_ok} ok, {tracks_fail} fail")
    print(f"Patterns: {patterns_created}")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(tracks_ok,1):.1f}s/track)")
    
    print("\n--- SEGMENT TYPES ---")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:12} {c}")
    
    cursor.execute("SELECT id, name FROM patterns WHERE name LIKE '%[ml:%' ORDER BY id DESC LIMIT 10")
    print("\n--- EXAMPLES ---")
    for pid, name in cursor.fetchall():
        print(f"  #{pid}: {name[:55]}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Music Structure Analyzer')
    parser.add_argument('--file', '-f', type=str, help='Analyze single file')
    parser.add_argument('--all', '-a', action='store_true', help='Process all liked tracks')
    parser.add_argument('--test', '-t', action='store_true', help='Test on first liked track')
    
    args = parser.parse_args()
    
    if args.file:
        analyze_single_track(args.file)
    elif args.all:
        process_all_liked_tracks()
    elif args.test:
        # Get first liked track
        cursor.execute("""
            SELECT file_path, title, artist 
            FROM tracks 
            WHERE is_liked = true AND file_path IS NOT NULL AND duration < 480
            ORDER BY id LIMIT 1
        """)
        track = cursor.fetchone()
        if track:
            file_path, title, artist = track
            print(f"\nTest track: {title} by {artist}")
            analyze_single_track(f"/app/storage/{file_path}")
        else:
            print("No liked tracks found!")
    else:
        parser.print_help()
    
    cursor.close()
    conn.close()
