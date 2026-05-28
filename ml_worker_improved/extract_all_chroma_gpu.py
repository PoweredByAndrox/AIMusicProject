#!/usr/bin/env python3
"""
GPU-accelerated chroma extraction - WITH PROPER LOGGING!
"""

import os
import sys
import time
import psycopg2
from pathlib import Path
import numpy as np
import subprocess
import tempfile

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ CHROMA (GPU)")
print("=" * 60)

# GPU check
try:
    import cupy as cp
    GPU_AVAILABLE = True
    gpu_info = cp.cuda.runtime.getDeviceProperties(0)
    print(f"✓ CuPy: {gpu_info['name'].decode('utf-8')}")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("✗ CuPy недоступен")

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✓ PyTorch CUDA")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch недоступен")

try:
    import soundfile as sf
    print("✓ soundfile")
except ImportError:
    print("✗ soundfile недоступен")
    sys.exit(1)

try:
    import librosa
    print("✓ librosa")
except ImportError:
    print("✗ librosa недоступна")

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'music_recommendations'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'secret')
)
cursor = conn.cursor()

STORAGE_PATH = Path(os.getenv('STORAGE_PATH', '/app/storage'))
CHROMA_PATH = STORAGE_PATH / 'chroma'
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_CHROMA = 12


def progress_bar(current, total, prefix='', suffix='', width=30):
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} {suffix}", end='')
    if current == total:
        print()


def load_audio(audio_path: str) -> np.ndarray:
    """Load audio file."""
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy import signal
            y = signal.resample(y, int(len(y) * SAMPLE_RATE / sr))
        return y.astype(np.float32)
    except:
        pass
    
    # Convert MP3 with ffmpeg
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', str(SAMPLE_RATE), '-ac', '1',
            '-f', 'wav', tmp_path
        ], capture_output=True, check=True)
        
        y, sr = sf.read(tmp_path)
        os.unlink(tmp_path)
        return y.astype(np.float32)
    except:
        return None


def extract_chroma(y: np.ndarray) -> np.ndarray:
    """Extract chroma features."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_chroma=N_CHROMA)
        
        # L2 normalize
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        chroma = chroma / norms
        
        return chroma.astype(np.float32)
    except:
        return None


# =============================================================================
# STEP 1: Get Tracks
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: ЗАГРУЗКА СПИСКА ТРЕКОВ")
print("=" * 60)

start_time = time.time()

# Check mode
REEXTRACT_ALL = os.getenv('REEXTRACT_ALL', '0') == '1'

if REEXTRACT_ALL:
    cursor.execute("""
        SELECT id, title, artist, file_path FROM tracks
        WHERE file_path IS NOT NULL AND analysis_status = 'completed'
        ORDER BY id
    """)
    print("Режим: ПЕРЕИЗВЛЕЧЕНИЕ ВСЕХ")
else:
    cursor.execute("""
        SELECT id, title, artist, file_path FROM tracks
        WHERE file_path IS NOT NULL
          AND (chroma_path IS NULL OR chroma_path = '')
          AND analysis_status = 'completed'
        ORDER BY id
    """)
    print("Режим: ТОЛЬКО БЕЗ CHROMA")

tracks = cursor.fetchall()
print(f"Треков для обработки: {len(tracks)}")

if not tracks:
    print("\nВсе треки уже имеют chroma!")
    cursor.close()
    conn.close()
    sys.exit(0)

# =============================================================================
# STEP 2: Extract Chroma
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: ИЗВЛЕЧЕНИЕ CHROMA")
print("=" * 60)

processed = 0
errors = 0
batch_start = time.time()

for i, (track_id, title, artist, file_path) in enumerate(tracks):
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        errors += 1
        progress_bar(i + 1, len(tracks), "  Progress", f"done:{processed} err:{errors}")
        continue
    
    # Load audio
    y = load_audio(audio_path)
    if y is None:
        errors += 1
        progress_bar(i + 1, len(tracks), "  Progress", f"done:{processed} err:{errors}")
        continue
    
    # Extract chroma
    chroma = extract_chroma(y)
    if chroma is None:
        errors += 1
        progress_bar(i + 1, len(tracks), "  Progress", f"done:{processed} err:{errors}")
        continue
    
    # Save
    chroma_file = CHROMA_PATH / f"track_{track_id}_chroma.npy"
    np.save(chroma_file, chroma)
    
    # Update DB
    cursor.execute("UPDATE tracks SET chroma_path = %s WHERE id = %s", (str(chroma_file), track_id))
    
    processed += 1
    
    # Commit and show progress
    if (i + 1) % 50 == 0:
        conn.commit()
        elapsed = time.time() - batch_start
        rate = processed / elapsed if elapsed > 0 else 0
        progress_bar(i + 1, len(tracks), "  Progress", f"done:{processed} err:{errors} ({rate:.1f}/s)")
        
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
    else:
        progress_bar(i + 1, len(tracks), "  Progress", f"done:{processed} err:{errors}")

conn.commit()
print()

# =============================================================================
# RESULTS
# =============================================================================

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)

print(f"\n--- СТАТИСТИКА ---")
print(f"Обработано: {processed}")
print(f"Ошибок:     {errors}")
print(f"Время:      {total_time:.1f}s")
print(f"Скорость:   {processed / total_time:.1f} треков/сек")

cursor.execute("SELECT COUNT(*) FROM tracks WHERE chroma_path IS NOT NULL")
total_with_chroma = cursor.fetchone()[0]
print(f"\nВсего треков с chroma: {total_with_chroma}")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)

cursor.close()
conn.close()
