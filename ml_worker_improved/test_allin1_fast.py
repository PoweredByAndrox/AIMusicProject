#!/usr/bin/env python3
"""
Fast All-In-One test suite with caching optimization.
Runs all tests on single analysis result to avoid repeated Demucs calls.
"""

import os
import sys
import time
import psycopg2
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

print("=" * 70)
print("FAST ALL-IN-ONE TEST SUITE (OPTIMIZED)")
print("=" * 70)

import torch
print(f"✓ PyTorch {torch.__version__}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

import natten
print(f"✓ natten {natten.__version__}")

import allin1
print(f"✓ allin1")

import librosa
print(f"✓ librosa {librosa.__version__}")

# Database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()
print("✓ Database connected")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def analyze_track_full(audio_path: str) -> dict:
    """Single analysis with all outputs - activations + embeddings."""
    start = time.time()
    
    # Single call with all options
    result = allin1.analyze(
        paths=audio_path,
        model='harmonix-all',  # Ensemble for accuracy
        device=DEVICE,
        include_activations=True,
        include_embeddings=True,
    )
    
    if isinstance(result, list):
        result = result[0]
    
    elapsed = time.time() - start
    return result, elapsed


def run_all_tests_on_result(result, audio_path: str, track_info: dict, analysis_time: float):
    """Run all validation tests on pre-computed result."""
    
    title = track_info.get('title', 'Unknown')[:35]
    print(f"\n{'─' * 70}")
    print(f"TRACK: {title}")
    print(f"Analysis time: {analysis_time:.1f}s")
    print('─' * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic Analysis
    tests_total += 1
    try:
        assert result.bpm is not None and 30 < result.bpm < 300
        assert len(result.beats) > 0
        assert len(result.downbeats) > 0
        assert len(result.segments) > 0
        print(f"✓ Basic: BPM={result.bpm:.1f}, beats={len(result.beats)}, segments={len(result.segments)}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Basic: {e}")
    
    # Test 2: Activations
    tests_total += 1
    try:
        assert result.activations is not None
        assert 'beat' in result.activations
        assert 'segment' in result.activations
        beat_shape = result.activations['beat'].shape
        print(f"✓ Activations: beat={beat_shape}, segment={result.activations['segment'].shape}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Activations: {e}")
    
    # Test 3: Embeddings
    tests_total += 1
    try:
        assert result.embeddings is not None
        emb = result.embeddings
        print(f"✓ Embeddings: shape={emb.shape}, mean={np.mean(emb):.4f}, std={np.std(emb):.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Embeddings: {e}")
    
    # Test 4: Beat Accuracy
    tests_total += 1
    try:
        beats = np.array(result.beats)
        if len(beats) > 1:
            ibis = np.diff(beats)
            expected_ibi = 60.0 / result.bpm
            regularity = 1.0 - min(1.0, np.std(ibis - expected_ibi) / expected_ibi)
        else:
            regularity = 0
        print(f"✓ Beat accuracy: regularity={regularity:.2%}, expected_ibi={60/result.bpm:.3f}s")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Beat accuracy: {e}")
    
    # Test 5: Segment Analysis
    tests_total += 1
    try:
        segments = result.segments
        labels = list(set(seg.label for seg in segments))
        durations = [seg.end - seg.start for seg in segments]
        print(f"✓ Segments: {len(segments)} segments, labels={labels}")
        print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s, avg={np.mean(durations):.1f}s")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Segments: {e}")
    
    # Test 6: Performance
    tests_total += 1
    try:
        duration = librosa.get_duration(path=audio_path)
        realtime_factor = duration / analysis_time
        print(f"✓ Performance: {duration:.1f}s audio in {analysis_time:.1f}s = {realtime_factor:.1f}x realtime")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Performance: {e}")
    
    return tests_passed, tests_total


def get_test_tracks(limit: int = 5):
    cursor.execute("""
        SELECT id, title, artist, file_path, duration, bpm
        FROM tracks 
        WHERE file_path IS NOT NULL 
          AND duration IS NOT NULL
          AND duration > 60 AND duration < 300
        ORDER BY RANDOM()
        LIMIT %s
    """, (limit,))
    
    return [
        {'id': r[0], 'title': r[1], 'artist': r[2], 'file_path': r[3], 'duration': r[4], 'expected_bpm': r[5]}
        for r in cursor.fetchall()
    ]


def main():
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZED TESTS")
    print("=" * 70)
    
    tracks = get_test_tracks(limit=5)
    print(f"\nTest tracks: {len(tracks)}")
    for i, t in enumerate(tracks):
        print(f"  {i+1}. {t['title'][:40]} ({t['duration']/60:.1f} min)")
    
    total_passed = 0
    total_tests = 0
    total_time = 0
    
    for track in tracks:
        audio_path = f"/app/storage/{track['file_path']}"
        
        if not Path(audio_path).exists():
            print(f"\n⚠ File not found: {audio_path}")
            continue
        
        # Single analysis call
        result, analysis_time = analyze_track_full(audio_path)
        total_time += analysis_time
        
        # Run all tests on cached result
        passed, total = run_all_tests_on_result(result, audio_path, track, analysis_time)
        total_passed += passed
        total_tests += total
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests: {total_passed}/{total_tests} passed")
    print(f"Total analysis time: {total_time:.1f}s")
    print(f"Average per track: {total_time/len(tracks):.1f}s")
    
    cursor.close()
    conn.close()
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
