#!/usr/bin/env python3
"""
LLM-assisted mix segmentation.
Uses audio features + LLM to determine track boundaries.
"""

import sys
import os
import json
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter
import requests

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 512

# OpenRouter API for LLM
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')


def extract_segment_features(y, sr, start_sec, end_sec):
    """Extract detailed features for a segment."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = y[start_sample:end_sample]
    
    if len(segment) < sr:  # Less than 1 second
        return None
    
    # Basic features
    rms = np.sqrt(np.mean(segment ** 2))
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))
    
    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    
    # Chroma (key/harmony)
    chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    dominant_chroma = int(np.argmax(chroma_mean))
    chroma_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # MFCC for timbre
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    return {
        'duration': end_sec - start_sec,
        'rms_energy': float(rms),
        'tempo_bpm': float(tempo) if not np.isnan(tempo) else 0,
        'spectral_centroid': float(spectral_centroid),
        'spectral_rolloff': float(spectral_rolloff),
        'dominant_note': chroma_notes[dominant_chroma],
        'mfcc_signature': mfcc_mean[:5].tolist()  # First 5 MFCCs as signature
    }


def compute_segment_similarity(feat1, feat2):
    """Compute similarity between two segments."""
    if feat1 is None or feat2 is None:
        return 0.0
    
    # Tempo similarity
    tempo_diff = abs(feat1['tempo_bpm'] - feat2['tempo_bpm'])
    tempo_sim = max(0, 1 - tempo_diff / 50)  # 50 BPM difference = 0 similarity
    
    # Energy similarity
    energy_ratio = min(feat1['rms_energy'], feat2['rms_energy']) / (max(feat1['rms_energy'], feat2['rms_energy']) + 1e-10)
    
    # Spectral similarity
    centroid_diff = abs(feat1['spectral_centroid'] - feat2['spectral_centroid'])
    centroid_sim = max(0, 1 - centroid_diff / 2000)
    
    # Key similarity (same note = 1, different = 0.5)
    key_sim = 1.0 if feat1['dominant_note'] == feat2['dominant_note'] else 0.5
    
    # MFCC similarity (cosine)
    mfcc1 = np.array(feat1['mfcc_signature'])
    mfcc2 = np.array(feat2['mfcc_signature'])
    mfcc_sim = np.dot(mfcc1, mfcc2) / (np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2) + 1e-10)
    mfcc_sim = (mfcc_sim + 1) / 2  # Normalize to 0-1
    
    # Weighted combination
    similarity = (
        0.25 * tempo_sim +
        0.15 * energy_ratio +
        0.20 * centroid_sim +
        0.15 * key_sim +
        0.25 * mfcc_sim
    )
    
    return similarity


def find_boundaries_adaptive(y, sr):
    """Find boundaries using adaptive multi-feature analysis."""
    duration = len(y) / sr
    hop = HOP_LENGTH
    
    # Extract features
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
    
    # Multi-scale novelty with smaller windows for short segments
    scales = [0.5, 1.0, 2.0]  # Smaller scales for finer detection
    scale_weights = [0.35, 0.40, 0.25]
    
    def compute_novelty(features, window):
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
    
    def compute_1d_novelty(sig, window):
        n = len(sig)
        novelty = np.zeros(n)
        for i in range(window, n - window):
            before = np.mean(sig[i-window:i])
            after = np.mean(sig[i:i+window])
            novelty[i] = abs(after - before)
        return novelty
    
    all_novelties = []
    for scale, weight in zip(scales, scale_weights):
        window_frames = max(5, int(scale * sr / hop))
        
        chroma_nov = compute_novelty(chroma_norm, window=window_frames)
        mfcc_nov = compute_novelty(mfcc_norm, window=window_frames)
        contrast_nov = compute_novelty(contrast_norm, window=window_frames)
        onset_nov = compute_1d_novelty(onset_norm, window=window_frames)
        rms_nov = compute_1d_novelty(rms_norm, window=window_frames)
        
        scale_novelty = (
            0.30 * chroma_nov +
            0.25 * mfcc_nov +
            0.20 * contrast_nov +
            0.15 * onset_nov +
            0.10 * rms_nov
        )
        
        if scale_novelty.max() > 0:
            scale_novelty = scale_novelty / scale_novelty.max()
        
        all_novelties.append(scale_novelty * weight)
    
    combined_novelty = np.sum(all_novelties, axis=0)
    if combined_novelty.max() > 0:
        combined_novelty = combined_novelty / combined_novelty.max()
    
    # Light smoothing to preserve sharp transitions
    combined_novelty = median_filter(combined_novelty, size=3)
    combined_novelty = uniform_filter1d(combined_novelty, size=5)
    
    # Find peaks with lower threshold for more candidates
    min_frames = int(6 * sr / hop)  # 6 seconds minimum
    threshold = np.percentile(combined_novelty, 40)
    
    peaks, props = signal.find_peaks(
        combined_novelty,
        height=threshold,
        distance=min_frames,
        prominence=0.015
    )
    
    # Convert to time
    candidates = [0.0]
    for peak in peaks:
        time_sec = peak * hop / sr
        if time_sec > 3 and time_sec < duration - 3:
            candidates.append(time_sec)
    candidates.append(duration)
    
    return sorted(set(candidates)), combined_novelty


def refine_boundaries_with_similarity(y, sr, candidates):
    """Refine boundaries by analyzing segment similarity."""
    refined = [candidates[0]]
    
    for i in range(1, len(candidates) - 1):
        prev_boundary = refined[-1]
        curr_boundary = candidates[i]
        next_boundary = candidates[i + 1] if i + 1 < len(candidates) else candidates[-1]
        
        # Extract features for segments before and after this boundary
        feat_before = extract_segment_features(y, sr, max(0, curr_boundary - 5), curr_boundary)
        feat_after = extract_segment_features(y, sr, curr_boundary, min(len(y)/sr, curr_boundary + 5))
        
        if feat_before is None or feat_after is None:
            continue
        
        # Compute similarity
        similarity = compute_segment_similarity(feat_before, feat_after)
        
        # If segments are very different (low similarity), this is likely a real boundary
        if similarity < 0.7:  # Threshold for "different enough"
            refined.append(curr_boundary)
    
    refined.append(candidates[-1])
    return refined


def merge_similar_segments(y, sr, boundaries, min_duration=6):
    """Merge segments that are too similar or too short."""
    if len(boundaries) <= 2:
        return boundaries
    
    merged = [boundaries[0]]
    
    i = 1
    while i < len(boundaries) - 1:
        curr = boundaries[i]
        
        # Check segment duration
        seg_duration = curr - merged[-1]
        
        if seg_duration < min_duration:
            # Too short, skip this boundary
            i += 1
            continue
        
        # Check similarity with previous segment
        feat_prev = extract_segment_features(y, sr, merged[-1], curr)
        
        if i + 1 < len(boundaries):
            feat_next = extract_segment_features(y, sr, curr, boundaries[i + 1])
            
            if feat_prev and feat_next:
                sim = compute_segment_similarity(feat_prev, feat_next)
                
                # If very similar, might be same track - skip boundary
                if sim > 0.85:
                    i += 1
                    continue
        
        merged.append(curr)
        i += 1
    
    merged.append(boundaries[-1])
    return merged


print("=" * 60)
print("LLM-ASSISTED MIX SEGMENTATION")
print("=" * 60)

# Test on "This Is Dubstep 2011"
audio_path = "/app/storage/audio/yt_697a9eea73f1d.mp3"

if not Path(audio_path).exists():
    print(f"File not found: {audio_path}")
    sys.exit(1)

print(f"\nLoading: {audio_path}")
y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
duration = len(y) / sr
print(f"Duration: {duration/60:.1f} min ({duration:.0f} sec)")

# Expected boundaries
expected = [0, 28, 42, 69, 90, 117, 130, 145, 158, 172, 178, 186, 192, 206, 220, 244, 274, 302, 315, 323, 336, 350, 364, 378, 390, 398, 411, 452, 480, 507, 514, 527, 541, 555, 569, 582, 596, 610, 624, 637, 646]

print(f"\nExpected: {len(expected)-1} segments")

print("\nStep 1: Finding candidate boundaries...")
candidates, novelty = find_boundaries_adaptive(y, sr)
print(f"  Found {len(candidates)-1} candidates")

print("\nStep 2: Refining with similarity analysis...")
refined = refine_boundaries_with_similarity(y, sr, candidates)
print(f"  Refined to {len(refined)-1} boundaries")

print("\nStep 3: Merging similar segments...")
final = merge_similar_segments(y, sr, refined, min_duration=5)
print(f"  Final: {len(final)-1} segments")

print(f"\n{'='*60}")
print(f"RESULTS: {len(final)-1} SEGMENTS")
print(f"{'='*60}")

for i in range(len(final)-1):
    start = final[i]
    end = final[i+1]
    dur = end - start
    print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d}  ({dur:.0f}s)")

# Compare with expected
print(f"\n{'='*60}")
print("COMPARISON WITH EXPECTED")
print(f"{'='*60}")

def find_closest(val, arr):
    return min(arr, key=lambda x: abs(x - val))

matched = 0
for exp in expected[1:-1]:
    closest = find_closest(exp, final)
    diff = abs(exp - closest)
    status = "OK" if diff < 8 else "MISS"
    if diff < 8:
        matched += 1
    print(f"  Expected {exp//60}:{exp%60:02d} -> Found {int(closest)//60}:{int(closest)%60:02d} (diff={diff:.0f}s) [{status}]")

accuracy = matched / (len(expected) - 2) * 100
print(f"\nAccuracy: {accuracy:.0f}% ({matched}/{len(expected)-2})")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
