#!/usr/bin/env python3
"""
GPU-accelerated pattern extraction - WITH PROPER LOGGING!
"""

import os
import sys
import time
import psycopg2
from pathlib import Path
from datetime import datetime
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ ПАТТЕРНОВ (GPU)")
print("=" * 60)

# GPU check
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("✗ PyTorch CUDA недоступен")
        device = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    device = None
    print("✗ PyTorch недоступен")

try:
    import librosa
    print("✓ librosa")
except ImportError:
    print("✗ librosa недоступна")
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
SEGMENT_DURATION = 8.0


def progress_bar(current, total, prefix='', suffix='', width=30):
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} {suffix}", end='')
    if current == total:
        print()


def compute_energy_gpu(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute energy on GPU."""
    if TORCH_AVAILABLE:
        try:
            y_t = torch.tensor(y, dtype=torch.float32, device=device)
            
            frame_length = 2048
            hop = HOP_LENGTH
            pad_len = frame_length // 2
            y_padded = torch.nn.functional.pad(y_t, (pad_len, pad_len))
            
            frames = y_padded.unfold(0, frame_length, hop)
            rms = torch.sqrt(torch.mean(frames ** 2, dim=1))
            
            window = torch.hann_window(frame_length, device=device)
            windowed = frames * window
            spec = torch.abs(torch.fft.rfft(windowed, dim=1))
            
            flux = torch.zeros(spec.shape[0], device=device)
            flux[1:] = torch.sum(torch.clamp(spec[1:] - spec[:-1], min=0), dim=1)
            
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
            flux_norm = (flux - flux.min()) / (flux.max() - flux.min() + 1e-10)
            
            energy = 0.4 * rms_norm + 0.6 * flux_norm
            
            window_size = int(sr / HOP_LENGTH * 2)
            if window_size > 1:
                kernel = torch.ones(1, 1, window_size, device=device) / window_size
                energy = torch.nn.functional.conv1d(
                    energy.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2
                ).squeeze()
            
            return energy.cpu().numpy()
        except Exception as e:
            print(f"\n  GPU error: {e}, fallback to CPU")
    
    # CPU fallback
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
    onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-10)
    
    energy = 0.4 * rms_norm + 0.6 * onset_norm[:len(rms_norm)]
    
    window = int(sr / HOP_LENGTH * 2)
    if window > 1:
        energy = np.convolve(energy, np.ones(window)/window, mode='same')
    
    return energy


def find_energy_peaks(audio_path: str, num_peaks: int = 3) -> list:
    """Find high-energy segments."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(y) / sr
        
        energy = compute_energy_gpu(y, sr)
        
        segment_frames = int(SEGMENT_DURATION * sr / HOP_LENGTH)
        peaks = []
        
        for i in range(0, len(energy) - segment_frames, segment_frames // 2):
            seg_energy = np.mean(energy[i:i + segment_frames])
            start_time = i * HOP_LENGTH / sr
            end_time = min((i + segment_frames) * HOP_LENGTH / sr, duration)
            peaks.append((start_time, end_time, seg_energy))
        
        peaks.sort(key=lambda x: x[2], reverse=True)
        
        selected = []
        for peak in peaks:
            overlap = any(abs(peak[0] - s[0]) < SEGMENT_DURATION / 2 for s in selected)
            if not overlap:
                selected.append(peak)
                if len(selected) >= num_peaks:
                    break
        
        return selected
    except Exception as e:
        return []


def extract_chroma_segment(audio_path: str, start_time: float, end_time: float) -> np.ndarray:
    """Extract chroma for segment."""
    try:
        duration = end_time - start_time
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, 
                            offset=start_time, duration=duration)
        
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH, n_chroma=12)
        
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        chroma = chroma / norms
        
        return chroma.astype(np.float32)
    except Exception as e:
        return None


# =============================================================================
# STEP 1: Cleanup
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: ОЧИСТКА СТАРЫХ ПАТТЕРНОВ")
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

# =============================================================================
# STEP 2: Load Liked Tracks
# =============================================================================

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

# =============================================================================
# STEP 3: Extract Patterns
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: ИЗВЛЕЧЕНИЕ ПАТТЕРНОВ")
print("=" * 60)

patterns_created = 0
tracks_processed = 0
tracks_failed = 0

for i, (track_id, title, artist, file_path) in enumerate(liked_tracks):
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        tracks_failed += 1
        progress_bar(i + 1, len(liked_tracks), "  Progress", f"created:{patterns_created} failed:{tracks_failed}")
        continue
    
    # Find peaks
    peaks = find_energy_peaks(audio_path, num_peaks=2)
    
    if not peaks:
        tracks_failed += 1
        progress_bar(i + 1, len(liked_tracks), "  Progress", f"created:{patterns_created} failed:{tracks_failed}")
        continue
    
    tracks_processed += 1
    
    for seg_start, seg_end, energy in peaks:
        chroma = extract_chroma_segment(audio_path, seg_start, seg_end)
        
        if chroma is None or chroma.shape[1] < 10:
            continue
        
        time_str = f"{int(seg_start//60)}:{int(seg_start%60):02d}"
        pattern_name = f"[auto] {artist or 'Unknown'} - {title[:25]} ({time_str})"
        
        cursor.execute("""
            INSERT INTO patterns (name, representation_type, transpose_invariant, tempo_invariant,
                                  match_threshold, importance, created_at, updated_at)
            VALUES (%s, 'chroma', true, false, 0.65, 2, %s, %s)
            RETURNING id
        """, (pattern_name, datetime.now(), datetime.now()))
        
        pattern_id = cursor.fetchone()[0]
        
        np.save(PATTERNS_PATH / f"pattern_{pattern_id}_chroma.npy", chroma)
        
        cursor.execute("""
            INSERT INTO pattern_examples (pattern_id, track_id, start_time, end_time,
                                          is_positive, notes, created_at, updated_at)
            VALUES (%s, %s, %s, %s, true, %s, %s, %s)
        """, (pattern_id, track_id, seg_start, seg_end,
              f"energy={energy:.2f}", datetime.now(), datetime.now()))
        
        patterns_created += 1
    
    conn.commit()
    
    progress_bar(i + 1, len(liked_tracks), "  Progress", f"created:{patterns_created} failed:{tracks_failed}")
    
    if TORCH_AVAILABLE and (i + 1) % 20 == 0:
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

cursor.execute("SELECT COUNT(*) FROM patterns")
total = cursor.fetchone()[0]
print(f"\nВсего паттернов в БД: {total}")

# Last 10 patterns
cursor.execute("SELECT id, name FROM patterns ORDER BY id DESC LIMIT 10")
print("\n--- ПОСЛЕДНИЕ 10 ПАТТЕРНОВ ---")
for pid, name in cursor.fetchall():
    print(f"  #{pid}: {name[:50]}")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)

cursor.close()
conn.close()
