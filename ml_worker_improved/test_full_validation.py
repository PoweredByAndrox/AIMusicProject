#!/usr/bin/env python3
"""
Full validation: Compare main algorithm vs consensus ground truth.
Target: 99% accuracy
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

# Import from test_self_validating
from test_self_validating import (
    Method1_EnergyBased, Method2_SpectralFlux, Method3_SelfSimilarity,
    Method4_BeatAligned, Method5_HarmonicPercussive,
    find_consensus_boundaries, classify_consensus_segment
)

# Import main algorithm
from test_advanced_segments import (
    AdvancedFeatureExtractor, AdvancedSegmentClassifier, find_boundaries
)


def normalize_type(seg_type):
    """Normalize segment types for comparison."""
    type_map = {
        'pre_intro': 'intro',
        'intro': 'intro',
        'buildup': 'buildup',
        'pre_drop': 'buildup',
        'drop': 'drop',
        'drop_bass': 'drop',
        'drop_high': 'drop',
        'verse': 'verse',
        'vocal': 'verse',
        'melodic': 'verse',
        'breakdown': 'break',
        'break': 'break',
        'ambient': 'ambient',
        'outro': 'ambient',
    }
    return type_map.get(seg_type, seg_type)


def compare_segments(ground_truth, detected, boundary_tolerance=3.0):
    """Compare detected segments with ground truth."""
    
    # 1. Boundary accuracy
    gt_bounds = set()
    for seg in ground_truth:
        gt_bounds.add(seg['start'])
        gt_bounds.add(seg['end'])
    
    det_bounds = set()
    for seg in detected:
        det_bounds.add(seg['start'])
        det_bounds.add(seg['end'])
    
    boundary_matches = 0
    boundary_errors = []
    
    for gt_b in sorted(gt_bounds):
        if gt_b == 0:
            continue
        closest = min(det_bounds, key=lambda x: abs(x - gt_b))
        diff = abs(closest - gt_b)
        if diff <= boundary_tolerance:
            boundary_matches += 1
        else:
            boundary_errors.append((gt_b, closest, diff))
    
    boundary_acc = boundary_matches / max(len(gt_bounds)-1, 1) * 100
    
    # 2. Type accuracy
    type_matches = 0
    type_errors = []
    
    for gt_seg in ground_truth:
        gt_mid = (gt_seg['start'] + gt_seg['end']) / 2
        gt_type = normalize_type(gt_seg['type'])
        
        # Find matching detected segment
        best_match = None
        for det_seg in detected:
            if det_seg['start'] <= gt_mid <= det_seg['end']:
                best_match = det_seg
                break
        
        if best_match is None:
            best_match = min(detected, key=lambda x: abs((x['start']+x['end'])/2 - gt_mid))
        
        det_type = normalize_type(best_match['type'])
        
        if gt_type == det_type:
            type_matches += 1
        else:
            type_errors.append({
                'time': f"{int(gt_seg['start'])//60}:{int(gt_seg['start'])%60:02d}",
                'expected': gt_type,
                'got': det_type
            })
    
    type_acc = type_matches / max(len(ground_truth), 1) * 100
    
    # Overall
    overall = boundary_acc * 0.3 + type_acc * 0.7
    
    return {
        'boundary_accuracy': boundary_acc,
        'type_accuracy': type_acc,
        'overall_accuracy': overall,
        'boundary_errors': boundary_errors,
        'type_errors': type_errors
    }


def run_full_validation(audio_path):
    """Run full validation pipeline."""
    
    print(f"\nLoading: {audio_path}")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")
    
    # =========================================================================
    # STEP 1: Generate ground truth using consensus
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: GENERATING GROUND TRUTH (5 methods consensus)")
    print("="*70)
    
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
        print(f"    {method.__class__.__name__}: {len(segments)} segments")
    
    # Find consensus
    consensus_bounds = find_consensus_boundaries(all_segments, tolerance=3.0)
    consensus_bounds.append(duration)
    consensus_bounds = sorted(set(consensus_bounds))
    
    ground_truth = []
    for i in range(len(consensus_bounds)-1):
        start, end = consensus_bounds[i], consensus_bounds[i+1]
        seg_type, confidence = classify_consensus_segment(
            start, end, all_segments, y, sr
        )
        ground_truth.append({
            'start': start,
            'end': end,
            'type': seg_type,
            'confidence': confidence
        })
    
    print(f"\nGround truth: {len(ground_truth)} segments")
    
    # =========================================================================
    # STEP 2: Run main algorithm
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: RUNNING MAIN ALGORITHM")
    print("="*70)
    
    extractor = AdvancedFeatureExtractor()
    features = extractor.extract_all(y)
    
    boundaries = find_boundaries(y, sr, duration, min_segment_sec=8.0)
    
    classifier = AdvancedSegmentClassifier()
    detected = classifier.classify_segments(features, boundaries, duration)
    
    print(f"Main algorithm: {len(detected)} segments")
    
    # =========================================================================
    # STEP 3: Compare
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: COMPARING RESULTS")
    print("="*70)
    
    results = compare_segments(ground_truth, detected, boundary_tolerance=3.0)
    
    print(f"\nBoundary accuracy: {results['boundary_accuracy']:.1f}%")
    if results['boundary_errors']:
        print("  Errors:")
        for gt, det, diff in results['boundary_errors'][:5]:
            print(f"    Expected {gt:.1f}s, got {det:.1f}s (diff={diff:.1f}s)")
        if len(results['boundary_errors']) > 5:
            print(f"    ... and {len(results['boundary_errors'])-5} more")
    
    print(f"\nType accuracy: {results['type_accuracy']:.1f}%")
    if results['type_errors']:
        print("  Errors:")
        for err in results['type_errors'][:5]:
            print(f"    {err['time']}: expected '{err['expected']}', got '{err['got']}'")
        if len(results['type_errors']) > 5:
            print(f"    ... and {len(results['type_errors'])-5} more")
    
    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    
    print(f"\n  Boundary accuracy: {results['boundary_accuracy']:5.1f}% (weight 30%)")
    print(f"  Type accuracy:     {results['type_accuracy']:5.1f}% (weight 70%)")
    print(f"  ─────────────────────────────")
    print(f"  OVERALL ACCURACY:  {results['overall_accuracy']:5.1f}%")
    
    if results['overall_accuracy'] >= 99:
        print("\n  ✅ TARGET ACHIEVED: 99%+ accuracy!")
    elif results['overall_accuracy'] >= 95:
        print(f"\n  ⚠️  CLOSE: {results['overall_accuracy']:.1f}% (need 99%)")
    else:
        print(f"\n  ❌ NEEDS WORK: {results['overall_accuracy']:.1f}% (need 99%)")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("FULL VALIDATION: Main Algorithm vs Consensus Ground Truth")
    print("Target: 99% accuracy")
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
            results = run_full_validation(audio_path)
        else:
            print(f"File not found: {audio_path}")
    else:
        print("No tracks found!")
