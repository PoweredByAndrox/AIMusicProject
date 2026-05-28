#!/usr/bin/env python3
"""
Extended test suite for All-In-One Music Structure Analyzer.

Tests:
1. Basic analysis (BPM, beats, segments)
2. Activations extraction (beat/downbeat/segment probabilities)
3. Embeddings extraction (neural network features)
4. Multiple tracks of different genres/lengths
5. Performance benchmarks
6. Edge cases (short tracks, long mixes)
"""

import os
import sys
import time
import json
import psycopg2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Flush prints immediately
sys.stdout.reconfigure(line_buffering=True)

print("=" * 70)
print("EXTENDED ALL-IN-ONE TEST SUITE")
print("=" * 70)

# Import dependencies
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠ CUDA not available")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import natten
    print(f"✓ natten {natten.__version__}")
except ImportError as e:
    print(f"✗ natten: {e}")
    sys.exit(1)

try:
    import allin1
    print(f"✓ allin1")
except ImportError as e:
    print(f"✗ allin1: {e}")
    sys.exit(1)

try:
    import librosa
    print(f"✓ librosa {librosa.__version__}")
except ImportError as e:
    print(f"✗ librosa: {e}")

# Database connection
try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        database=os.getenv('DB_NAME', 'music_recommendations'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'secret')
    )
    cursor = conn.cursor()
    print("✓ Database connected")
except Exception as e:
    print(f"✗ Database: {e}")
    conn = None
    cursor = None

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m}:{s:02d}"


def format_duration(seconds: float) -> str:
    """Format duration with milliseconds."""
    m, s = int(seconds // 60), seconds % 60
    return f"{m}:{s:05.2f}"


class TestResult:
    """Container for test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.data = {}
    
    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} {self.name} ({self.duration:.2f}s)"


def test_basic_analysis(audio_path: str, track_info: dict) -> TestResult:
    """
    Test 1: Basic analysis - BPM, beats, segments.
    """
    result = TestResult(f"Basic Analysis: {track_info.get('title', 'Unknown')[:30]}")
    
    try:
        start = time.time()
        
        analysis = allin1.analyze(
            paths=audio_path,
            model='harmonix-all',
            device=DEVICE,
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        result.duration = time.time() - start
        
        # Validate results
        assert analysis.bpm is not None, "BPM is None"
        assert 30 < analysis.bpm < 300, f"BPM {analysis.bpm} out of range"
        assert len(analysis.beats) > 0, "No beats detected"
        assert len(analysis.downbeats) > 0, "No downbeats detected"
        assert len(analysis.segments) > 0, "No segments detected"
        
        # Store data
        result.data = {
            'bpm': analysis.bpm,
            'num_beats': len(analysis.beats),
            'num_downbeats': len(analysis.downbeats),
            'num_segments': len(analysis.segments),
            'beat_positions': analysis.beat_positions[:20] if hasattr(analysis, 'beat_positions') else [],
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'label': seg.label,
                    'duration': seg.end - seg.start
                }
                for seg in analysis.segments
            ],
            'segment_labels': list(set(seg.label for seg in analysis.segments)),
            'total_duration': analysis.segments[-1].end if analysis.segments else 0,
        }
        
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_activations(audio_path: str, track_info: dict) -> TestResult:
    """
    Test 2: Extract activations (frame-level probabilities).
    """
    result = TestResult(f"Activations: {track_info.get('title', 'Unknown')[:30]}")
    
    try:
        start = time.time()
        
        analysis = allin1.analyze(
            paths=audio_path,
            model='harmonix-fold0',  # Single model for speed
            device=DEVICE,
            include_activations=True,
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        result.duration = time.time() - start
        
        # Validate activations
        assert analysis.activations is not None, "Activations is None"
        assert 'beat' in analysis.activations, "No beat activations"
        assert 'downbeat' in analysis.activations, "No downbeat activations"
        assert 'segment' in analysis.activations, "No segment activations"
        assert 'label' in analysis.activations, "No label activations"
        
        beat_act = analysis.activations['beat']
        downbeat_act = analysis.activations['downbeat']
        segment_act = analysis.activations['segment']
        label_act = analysis.activations['label']
        
        result.data = {
            'beat_shape': beat_act.shape,
            'downbeat_shape': downbeat_act.shape,
            'segment_shape': segment_act.shape,
            'label_shape': label_act.shape,
            'beat_mean': float(np.mean(beat_act)),
            'beat_max': float(np.max(beat_act)),
            'downbeat_mean': float(np.mean(downbeat_act)),
            'segment_mean': float(np.mean(segment_act)),
            'fps': 100,  # allin1 uses 100 FPS
            'duration_frames': len(beat_act),
            'duration_seconds': len(beat_act) / 100,
        }
        
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_embeddings(audio_path: str, track_info: dict) -> TestResult:
    """
    Test 3: Extract embeddings (neural network features).
    """
    result = TestResult(f"Embeddings: {track_info.get('title', 'Unknown')[:30]}")
    
    try:
        start = time.time()
        
        analysis = allin1.analyze(
            paths=audio_path,
            model='harmonix-fold0',
            device=DEVICE,
            include_embeddings=True,
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        result.duration = time.time() - start
        
        # Validate embeddings
        assert analysis.embeddings is not None, "Embeddings is None"
        
        emb = analysis.embeddings
        
        result.data = {
            'shape': emb.shape,
            'dtype': str(emb.dtype),
            'mean': float(np.mean(emb)),
            'std': float(np.std(emb)),
            'min': float(np.min(emb)),
            'max': float(np.max(emb)),
            # Embeddings shape: [stems=4, time_steps, embedding_size=24]
            'num_stems': emb.shape[0] if len(emb.shape) > 2 else 1,
            'time_steps': emb.shape[1] if len(emb.shape) > 2 else emb.shape[0],
            'embedding_dim': emb.shape[-1],
        }
        
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_beat_accuracy(audio_path: str, track_info: dict) -> TestResult:
    """
    Test 4: Beat detection accuracy analysis.
    """
    result = TestResult(f"Beat Accuracy: {track_info.get('title', 'Unknown')[:30]}")
    
    try:
        start = time.time()
        
        analysis = allin1.analyze(
            paths=audio_path,
            model='harmonix-all',
            device=DEVICE,
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        result.duration = time.time() - start
        
        beats = np.array(analysis.beats)
        downbeats = np.array(analysis.downbeats)
        
        # Calculate inter-beat intervals
        if len(beats) > 1:
            ibis = np.diff(beats)
            expected_ibi = 60.0 / analysis.bpm
            ibi_deviation = np.std(ibis - expected_ibi)
            
            # Check beat regularity
            regularity = 1.0 - min(1.0, ibi_deviation / expected_ibi)
        else:
            ibis = []
            ibi_deviation = 0
            regularity = 0
        
        # Calculate downbeat intervals (should be ~4 beats apart for 4/4)
        if len(downbeats) > 1:
            dbi = np.diff(downbeats)
            expected_dbi = 4 * 60.0 / analysis.bpm
            dbi_deviation = np.std(dbi - expected_dbi)
        else:
            dbi = []
            dbi_deviation = 0
        
        # Beat positions analysis
        beat_positions = analysis.beat_positions if hasattr(analysis, 'beat_positions') else []
        position_counts = {}
        for pos in beat_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        result.data = {
            'bpm': analysis.bpm,
            'num_beats': len(beats),
            'num_downbeats': len(downbeats),
            'expected_ibi': 60.0 / analysis.bpm,
            'actual_ibi_mean': float(np.mean(ibis)) if len(ibis) > 0 else 0,
            'actual_ibi_std': float(np.std(ibis)) if len(ibis) > 0 else 0,
            'ibi_deviation': float(ibi_deviation),
            'beat_regularity': float(regularity),
            'downbeat_interval_mean': float(np.mean(dbi)) if len(dbi) > 0 else 0,
            'downbeat_interval_std': float(np.std(dbi)) if len(dbi) > 0 else 0,
            'beat_position_distribution': position_counts,
            'beats_per_downbeat': len(beats) / max(1, len(downbeats)),
        }
        
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_segment_analysis(audio_path: str, track_info: dict) -> TestResult:
    """
    Test 5: Detailed segment analysis.
    """
    result = TestResult(f"Segments: {track_info.get('title', 'Unknown')[:30]}")
    
    try:
        start = time.time()
        
        analysis = allin1.analyze(
            paths=audio_path,
            model='harmonix-all',
            device=DEVICE,
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        result.duration = time.time() - start
        
        segments = analysis.segments
        
        # Segment statistics
        label_stats = {}
        for seg in segments:
            label = seg.label
            duration = seg.end - seg.start
            if label not in label_stats:
                label_stats[label] = {'count': 0, 'total_duration': 0, 'durations': []}
            label_stats[label]['count'] += 1
            label_stats[label]['total_duration'] += duration
            label_stats[label]['durations'].append(duration)
        
        # Calculate averages
        for label in label_stats:
            stats = label_stats[label]
            stats['avg_duration'] = stats['total_duration'] / stats['count']
            stats['min_duration'] = min(stats['durations'])
            stats['max_duration'] = max(stats['durations'])
            del stats['durations']  # Remove raw data
        
        # Segment transitions
        transitions = []
        for i in range(len(segments) - 1):
            transitions.append(f"{segments[i].label} -> {segments[i+1].label}")
        
        transition_counts = {}
        for t in transitions:
            transition_counts[t] = transition_counts.get(t, 0) + 1
        
        # Structure pattern detection
        label_sequence = [seg.label for seg in segments]
        
        result.data = {
            'num_segments': len(segments),
            'unique_labels': list(label_stats.keys()),
            'label_statistics': label_stats,
            'total_duration': segments[-1].end if segments else 0,
            'avg_segment_duration': sum(s.end - s.start for s in segments) / max(1, len(segments)),
            'transition_counts': transition_counts,
            'structure_sequence': label_sequence,
            'has_intro': 'intro' in label_stats,
            'has_outro': 'outro' in label_stats,
            'has_chorus': 'chorus' in label_stats,
            'has_verse': 'verse' in label_stats,
            'has_bridge': 'bridge' in label_stats,
        }
        
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_performance(audio_path: str, track_info: dict) -> TestResult:
    """
    Test 6: Performance benchmark.
    """
    result = TestResult(f"Performance: {track_info.get('title', 'Unknown')[:30]}")
    
    try:
        # Get audio duration
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=10)
        full_duration = librosa.get_duration(path=audio_path)
        
        # Warm-up run (models already loaded)
        _ = allin1.analyze(
            paths=audio_path,
            model='harmonix-fold0',
            device=DEVICE,
        )
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Timed run - single model
        start = time.time()
        _ = allin1.analyze(
            paths=audio_path,
            model='harmonix-fold0',
            device=DEVICE,
        )
        single_model_time = time.time() - start
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Timed run - ensemble (all 8 models)
        start = time.time()
        _ = allin1.analyze(
            paths=audio_path,
            model='harmonix-all',
            device=DEVICE,
        )
        ensemble_time = time.time() - start
        
        result.duration = ensemble_time
        
        # GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_memory = 0
        
        result.data = {
            'audio_duration': full_duration,
            'single_model_time': single_model_time,
            'ensemble_time': ensemble_time,
            'realtime_factor_single': full_duration / single_model_time,
            'realtime_factor_ensemble': full_duration / ensemble_time,
            'gpu_memory_gb': gpu_memory,
            'device': DEVICE,
            'speedup_vs_realtime': full_duration / ensemble_time,
        }
        
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def get_test_tracks(limit: int = 5) -> List[Dict]:
    """Get diverse test tracks from database."""
    if cursor is None:
        return []
    
    # Get tracks of different durations and genres
    cursor.execute("""
        SELECT id, title, artist, file_path, duration, bpm
        FROM tracks 
        WHERE file_path IS NOT NULL 
          AND duration IS NOT NULL
          AND duration > 60
          AND duration < 600
        ORDER BY RANDOM()
        LIMIT %s
    """, (limit,))
    
    tracks = []
    for row in cursor.fetchall():
        tracks.append({
            'id': row[0],
            'title': row[1],
            'artist': row[2],
            'file_path': row[3],
            'duration': row[4],
            'expected_bpm': row[5],
        })
    
    return tracks


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "=" * 70)
    print("RUNNING TEST SUITE")
    print("=" * 70)
    
    # Get test tracks
    tracks = get_test_tracks(limit=5)
    
    if not tracks:
        print("No tracks found in database!")
        # Use a fallback track if available
        cursor.execute("""
            SELECT id, title, artist, file_path, duration, bpm
            FROM tracks 
            WHERE file_path IS NOT NULL
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            tracks = [{
                'id': row[0],
                'title': row[1],
                'artist': row[2],
                'file_path': row[3],
                'duration': row[4],
                'expected_bpm': row[5],
            }]
    
    print(f"\nTest tracks: {len(tracks)}")
    for i, t in enumerate(tracks):
        print(f"  {i+1}. {t['title'][:40]} by {t['artist'][:20]} ({t['duration']/60:.1f} min)")
    
    all_results = []
    
    for track in tracks:
        audio_path = f"/app/storage/{track['file_path']}"
        
        if not Path(audio_path).exists():
            print(f"\n⚠ File not found: {audio_path}")
            continue
        
        print(f"\n{'─' * 70}")
        print(f"TESTING: {track['title']} by {track['artist']}")
        print(f"Duration: {track['duration']/60:.1f} min | Expected BPM: {track.get('expected_bpm', 'N/A')}")
        print('─' * 70)
        
        # Run tests
        tests = [
            test_basic_analysis,
            test_activations,
            test_embeddings,
            test_beat_accuracy,
            test_segment_analysis,
            test_performance,
        ]
        
        for test_func in tests:
            result = test_func(audio_path, track)
            all_results.append(result)
            
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.name}: {result.duration:.2f}s")
            
            if result.error:
                print(f"    Error: {result.error[:100]}")
            elif result.data:
                # Print key metrics
                if 'bpm' in result.data:
                    print(f"    BPM: {result.data['bpm']:.1f}")
                if 'num_segments' in result.data:
                    print(f"    Segments: {result.data['num_segments']}")
                if 'beat_regularity' in result.data:
                    print(f"    Beat regularity: {result.data['beat_regularity']:.2%}")
                if 'realtime_factor_ensemble' in result.data:
                    print(f"    Speed: {result.data['realtime_factor_ensemble']:.1f}x realtime")
                if 'shape' in result.data:
                    print(f"    Shape: {result.data['shape']}")
        
        # Clear GPU memory between tracks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    total_time = sum(r.duration for r in all_results)
    
    print(f"Passed: {passed}/{len(all_results)}")
    print(f"Failed: {failed}/{len(all_results)}")
    print(f"Total time: {total_time:.1f}s")
    
    if failed > 0:
        print("\nFailed tests:")
        for r in all_results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.error[:80] if r.error else 'Unknown error'}")
    
    # Detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    for r in all_results:
        if r.passed and r.data:
            print(f"\n{r.name}:")
            for key, value in r.data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, dict) and len(str(value)) < 200:
                    print(f"  {key}: {value}")
                elif isinstance(value, list) and len(value) < 20:
                    print(f"  {key}: {value}")
                elif isinstance(value, tuple):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")
    
    return all_results


if __name__ == "__main__":
    results = run_all_tests()
    
    if cursor:
        cursor.close()
        conn.close()
    
    # Exit with error code if any tests failed
    failed = sum(1 for r in results if not r.passed)
    sys.exit(1 if failed > 0 else 0)
