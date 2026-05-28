#!/usr/bin/env python3
"""
Score tracks using trained Segment Preference Model.

Analyzes each track, finds best segments, scores them with the model.
Updates preference_score in database.
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
print("SCORING TRACKS WITH SEGMENT MODEL (GPU)")
print("=" * 60)

# =============================================================================
# GPU Setup
# =============================================================================

import torch
import torchaudio
import torchaudio.transforms as T

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("⚠ CUDA недоступен")

import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, '/app/worker')
from segment_preference_model import SegmentPreferenceModel

# =============================================================================
# Config
# =============================================================================

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
MODEL_PATH = STORAGE_PATH / 'models' / 'segment_preference_model.pt'

# =============================================================================
# Database
# =============================================================================

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()


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


# =============================================================================
# Load Model
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"✗ Модель не найдена: {MODEL_PATH}")
    print("  Сначала запустите train_segment_model_gpu.py")
    sys.exit(1)

model = SegmentPreferenceModel(
    input_channels=N_MELS,
    embedding_dim=256,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
model.load(str(MODEL_PATH))
print(f"✓ Модель загружена: {MODEL_PATH}")


# =============================================================================
# Feature Extraction
# =============================================================================

class GPUAnalyzer:
    """GPU-accelerated track analysis and scoring."""
    
    def __init__(self, model: SegmentPreferenceModel):
        self.model = model
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            normalized=True
        ).to(device)
    
    def analyze_track(self, audio_path: str) -> dict:
        """
        Analyze track and return segment scores.
        
        Returns:
            {
                'best_score': float,
                'avg_score': float,
                'best_segments': [(start, end, score), ...]
            }
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration = len(y) / sr
            
            if duration < 30:
                return None
            
            # Find segment boundaries using novelty
            boundaries = self._find_boundaries(y, duration)
            
            # Score each segment
            segment_scores = []
            
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                seg_dur = end - start
                
                if seg_dur < 3 or seg_dur > 60:
                    continue
                
                # Extract segment
                start_sample = int(start * SAMPLE_RATE)
                end_sample = int(end * SAMPLE_RATE)
                seg_audio = y[start_sample:end_sample]
                
                if len(seg_audio) < SAMPLE_RATE:
                    continue
                
                # Extract mel
                waveform = torch.from_numpy(seg_audio).float().to(device)
                with torch.no_grad():
                    mel = self.mel_transform(waveform)
                    mel = torch.log(mel + 1e-9)
                mel_np = mel.cpu().numpy()
                
                # Score with model
                score = self.model.predict_preference(mel_np)
                segment_scores.append((start, end, score))
            
            if not segment_scores:
                return None
            
            # Sort by score
            segment_scores.sort(key=lambda x: -x[2])
            
            scores = [s[2] for s in segment_scores]
            
            return {
                'best_score': max(scores),
                'avg_score': np.mean(scores),
                'top3_avg': np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores),
                'best_segments': segment_scores[:5]
            }
            
        except Exception as e:
            print(f"\n  Error: {e}")
            return None
    
    def _find_boundaries(self, y: np.ndarray, duration: float) -> list:
        """Find segment boundaries using onset detection."""
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
        
        # Smooth
        onset_env = uniform_filter1d(onset_env, 10)
        
        # Find peaks
        min_distance = int(4 * SAMPLE_RATE / HOP_LENGTH)  # 4 seconds
        threshold = np.percentile(onset_env, 60)
        peaks, _ = signal.find_peaks(onset_env, height=threshold, distance=min_distance)
        
        # Convert to time
        boundaries = [0]
        for peak in peaks:
            boundaries.append(peak * HOP_LENGTH / SAMPLE_RATE)
        boundaries.append(duration)
        
        return sorted(set(boundaries))


# =============================================================================
# Score Tracks
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: ЗАГРУЗКА ТРЕКОВ")
print("=" * 60)

# Get all tracks that are not liked (to score for recommendations)
cursor.execute("""
    SELECT id, title, artist, file_path, duration
    FROM tracks 
    WHERE file_path IS NOT NULL 
    AND is_liked = false
    AND analysis_status = 'completed'
    ORDER BY id
""")

tracks = cursor.fetchall()
print(f"Треков для скоринга: {len(tracks)}")

print("\n" + "=" * 60)
print("STEP 3: СКОРИНГ (GPU)")
print("=" * 60)

analyzer = GPUAnalyzer(model)

scored = 0
failed = 0
results = []

start_time = time.time()

for i, (track_id, title, artist, file_path, duration) in enumerate(tracks):
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        failed += 1
        progress_bar(i + 1, len(tracks), "  Score", f"ok:{scored} fail:{failed}")
        continue
    
    analysis = analyzer.analyze_track(audio_path)
    
    if analysis is None:
        failed += 1
        progress_bar(i + 1, len(tracks), "  Score", f"ok:{scored} fail:{failed}")
        continue
    
    # Compute final score (weighted combination)
    # Best segment matters most, but average also counts
    final_score = 0.6 * analysis['best_score'] + 0.4 * analysis['top3_avg']
    
    # Update database
    cursor.execute("""
        UPDATE tracks 
        SET preference_score = %s, updated_at = %s
        WHERE id = %s
    """, (final_score, datetime.now(), track_id))
    
    results.append({
        'id': track_id,
        'title': title,
        'artist': artist,
        'score': final_score,
        'best_segments': analysis['best_segments'][:3]
    })
    
    scored += 1
    progress_bar(i + 1, len(tracks), "  Score", f"ok:{scored} fail:{failed}")
    
    # Commit periodically
    if (i + 1) % 20 == 0:
        conn.commit()
        torch.cuda.empty_cache()

conn.commit()
print()

elapsed = time.time() - start_time

# =============================================================================
# Results
# =============================================================================

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"Треков обработано: {scored}")
print(f"Failed: {failed}")
print(f"Время: {elapsed:.1f}s ({elapsed/max(scored,1):.2f}s/трек)")

# Top recommendations
print("\n--- TOP 15 РЕКОМЕНДАЦИЙ ---")
results.sort(key=lambda x: -x['score'])

for i, r in enumerate(results[:15]):
    print(f"\n{i+1}. [{r['score']:.3f}] {r['artist'] or '?'} - {r['title'][:40]}")
    for start, end, seg_score in r['best_segments'][:2]:
        print(f"   └─ {format_time(start)}-{format_time(end)}: {seg_score:.3f}")

# Score distribution
if results:
    scores = [r['score'] for r in results]
    print("\n--- РАСПРЕДЕЛЕНИЕ СКОРОВ ---")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    
    # Histogram
    print("\n  Гистограмма:")
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(scores, bins=bins)
    for i in range(len(bins) - 1):
        bar = '█' * (hist[i] // 5 + 1) if hist[i] > 0 else ''
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:4d} {bar}")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)

cursor.close()
conn.close()
