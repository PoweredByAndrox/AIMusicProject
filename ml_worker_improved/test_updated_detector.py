#!/usr/bin/env python3
"""Quick test of the updated MixTrackDetector."""

import sys
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 512

class MixTrackDetector:
    """Copy of the updated detector for testing."""
    
    def __init__(self, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
        self.sr = sr
        self.hop_length = hop_length
    
    def detect_track_boundaries(self, y, expected_track_duration=30.0):
        duration = len(y) / self.sr
        
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20, hop_length=self.hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, hop_length=self.hop_length)
        tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        if tonnetz.shape[1] != chroma.shape[1]:
            tonnetz = np.array([np.interp(
                np.linspace(0, 1, chroma.shape[1]),
                np.linspace(0, 1, tonnetz.shape[1]),
                row
            ) for row in tonnetz])
        
        chroma_norm = librosa.util.normalize(chroma, axis=0)
        mfcc_norm = librosa.util.normalize(mfcc, axis=0)
        contrast_norm = librosa.util.normalize(contrast, axis=0)
        tonnetz_norm = librosa.util.normalize(tonnetz, axis=0)
        onset_norm = onset_env / (onset_env.max() + 1e-10)
        rms_norm = rms / (rms.max() + 1e-10)
        
        scales = [1.0, 2.0, 4.0]
        scale_weights = [0.3, 0.4, 0.3]
        
        all_novelties = []
        for scale, weight in zip(scales, scale_weights):
            window_frames = int(scale * self.sr / self.hop_length)
            
            chroma_nov = self._compute_novelty(chroma_norm, window=window_frames)
            mfcc_nov = self._compute_novelty(mfcc_norm, window=window_frames)
            contrast_nov = self._compute_novelty(contrast_norm, window=window_frames)
            tonnetz_nov = self._compute_novelty(tonnetz_norm, window=window_frames)
            onset_nov = self._compute_1d_novelty(onset_norm, window=window_frames)
            rms_nov = self._compute_1d_novelty(rms_norm, window=window_frames)
            
            scale_novelty = (
                0.30 * chroma_nov + 0.20 * mfcc_nov + 0.15 * contrast_nov +
                0.15 * tonnetz_nov + 0.10 * onset_nov + 0.10 * rms_nov
            )
            
            if scale_novelty.max() > 0:
                scale_novelty = scale_novelty / scale_novelty.max()
            
            all_novelties.append(scale_novelty * weight)
        
        combined_novelty = np.sum(all_novelties, axis=0)
        if combined_novelty.max() > 0:
            combined_novelty = combined_novelty / combined_novelty.max()
        
        combined_novelty = median_filter(combined_novelty, size=5)
        combined_novelty = uniform_filter1d(combined_novelty, size=10)
        
        min_frames = int(15 * self.sr / self.hop_length)
        threshold = np.percentile(combined_novelty, 50)
        
        peaks, _ = signal.find_peaks(combined_novelty, height=threshold, distance=min_frames, prominence=0.02)
        
        boundaries = [0.0]
        for peak in peaks:
            time_sec = peak * self.hop_length / self.sr
            if time_sec > 5 and time_sec < duration - 5:
                boundaries.append(time_sec)
        boundaries.append(duration)
        
        boundaries = self._merge_close_boundaries(boundaries, min_gap=12)
        boundaries = self._fill_large_gaps(boundaries, combined_novelty, min_gap=32)
        
        return sorted(set(boundaries))
    
    def _compute_novelty(self, features, window=20):
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
    
    def _compute_1d_novelty(self, sig, window=20):
        n = len(sig)
        novelty = np.zeros(n)
        for i in range(window, n - window):
            before = np.mean(sig[i-window:i])
            after = np.mean(sig[i:i+window])
            novelty[i] = abs(after - before)
        return novelty
    
    def _merge_close_boundaries(self, boundaries, min_gap=12.0):
        if len(boundaries) <= 2:
            return boundaries
        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] >= min_gap:
                merged.append(b)
        if boundaries[-1] not in merged:
            merged[-1] = boundaries[-1]
        return merged
    
    def _fill_large_gaps(self, boundaries, novelty, min_gap=20.0):  # Changed from 32 to 20
        filled = list(boundaries)
        for i in range(len(boundaries) - 1):
            gap_start = boundaries[i]
            gap_end = boundaries[i + 1]
            gap_size = gap_end - gap_start
            if gap_size > min_gap:
                start_frame = int(gap_start * self.sr / self.hop_length)
                end_frame = int(gap_end * self.sr / self.hop_length)
                gap_novelty = novelty[start_frame:end_frame]
                if len(gap_novelty) > 0:
                    local_peaks, _ = signal.find_peaks(gap_novelty, prominence=0.02)
                    if len(local_peaks) > 0:
                        best_peak = local_peaks[np.argmax(gap_novelty[local_peaks])]
                        peak_time = gap_start + best_peak * self.hop_length / self.sr
                        if peak_time - gap_start > 8 and gap_end - peak_time > 8:  # Changed from 12 to 8
                            filled.append(peak_time)
        return sorted(filled)

print("=" * 60)
print("ТЕСТ ОБНОВЛЁННОГО MixTrackDetector")
print("=" * 60)

audio_path = "/app/storage/audio/yt_697a9f137cfb0.mp3"

print(f"\nЗагрузка: {audio_path}")
y, sr = librosa.load(audio_path, sr=22050, mono=True)
duration = len(y) / sr
print(f"Длительность: {duration/60:.1f} мин ({duration:.0f} сек)")

# Expected boundaries
expected = [0, 40, 82, 150, 179, 205, 233, 260, 288, 315, 341, 370, 397, 424, 452, 494, 548, 599]

print("\nЗапуск детектора...")
detector = MixTrackDetector(sr=22050, hop_length=512)
boundaries = detector.detect_track_boundaries(y)

print(f"\nНайдено границ: {len(boundaries)} ({len(boundaries)-1} треков)")
for i in range(len(boundaries)-1):
    start = boundaries[i]
    end = boundaries[i+1]
    dur = end - start
    print(f"  {i+1:2}. {int(start)//60}:{int(start)%60:02d} - {int(end)//60}:{int(end)%60:02d} ({dur:.0f}s)")

# Compare
print("\n--- СРАВНЕНИЕ ---")
def find_closest(val, arr):
    return min(arr, key=lambda x: abs(x - val))

matched = 0
for exp in expected[1:-1]:
    closest = find_closest(exp, boundaries)
    diff = abs(exp - closest)
    status = "✓" if diff < 10 else "✗"
    if diff < 10:
        matched += 1
    print(f"  {exp//60}:{exp%60:02d} -> {int(closest)//60}:{int(closest)%60:02d} (diff={diff:.0f}s) {status}")

accuracy = matched / (len(expected) - 2) * 100
print(f"\nТочность: {accuracy:.0f}% ({matched}/{len(expected)-2})")
print("=" * 60)
