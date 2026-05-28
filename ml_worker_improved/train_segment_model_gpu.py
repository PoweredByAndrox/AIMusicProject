#!/usr/bin/env python3
"""
GPU-accelerated training script for Segment Preference Model.

Loads labeled segments from database, extracts features, trains model.
"""

import os
import sys
import time
import psycopg2
from pathlib import Path
from datetime import datetime
import numpy as np

# Force flush output
sys.stdout.reconfigure(line_buffering=True)
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

print("=" * 60)
print("SEGMENT PREFERENCE MODEL TRAINING (GPU)")
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
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("⚠ CUDA недоступен, CPU mode")

try:
    import librosa
    print("✓ librosa")
except ImportError:
    print("✗ librosa - required!")
    sys.exit(1)

# Import our model
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
MODELS_PATH = STORAGE_PATH / 'models'
MODELS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_PATH / 'segment_preference_model.pt'

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
# Feature Extraction
# =============================================================================

class GPUMelExtractor:
    """GPU-accelerated mel spectrogram extraction."""
    
    def __init__(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            normalized=True
        ).to(device)
    
    def extract(self, audio_path: str, start: float, end: float) -> np.ndarray:
        """Extract mel spectrogram for segment."""
        try:
            # Load segment
            duration = end - start
            y, sr = librosa.load(
                audio_path, 
                sr=SAMPLE_RATE, 
                mono=True,
                offset=start,
                duration=duration
            )
            
            if len(y) < SAMPLE_RATE:  # Less than 1 second
                return None
            
            # Convert to tensor and extract mel
            waveform = torch.from_numpy(y).float().to(device)
            
            with torch.no_grad():
                mel = self.mel_transform(waveform)
                # Log scale
                mel = torch.log(mel + 1e-9)
            
            return mel.cpu().numpy()
            
        except Exception as e:
            print(f"\n  Error extracting {audio_path}: {e}")
            return None


# =============================================================================
# Load Training Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: ЗАГРУЗКА РАЗМЕЧЕННЫХ СЕГМЕНТОВ")
print("=" * 60)

# Get labeled segments from pattern_examples
# is_positive = true means liked, false means disliked
cursor.execute("""
    SELECT 
        pe.id,
        pe.track_id,
        pe.start_time,
        pe.end_time,
        pe.is_positive,
        t.file_path,
        t.title,
        t.artist,
        p.name as pattern_name
    FROM pattern_examples pe
    JOIN tracks t ON pe.track_id = t.id
    JOIN patterns p ON pe.pattern_id = p.id
    WHERE t.file_path IS NOT NULL
    ORDER BY pe.is_positive DESC, pe.id
""")

segments = cursor.fetchall()
print(f"Всего сегментов в базе: {len(segments)}")

# Count by label
liked_count = sum(1 for s in segments if s[4])
disliked_count = sum(1 for s in segments if not s[4])
print(f"  Liked: {liked_count}")
print(f"  Disliked: {disliked_count}")

if liked_count < 5:
    print("\n⚠ Недостаточно liked сегментов для обучения!")
    print("  Нужно минимум 5 liked сегментов.")
    print("  Используйте UI для разметки: /track/{id}/segments")
    sys.exit(0)

# =============================================================================
# Extract Features
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ (GPU)")
print("=" * 60)

extractor = GPUMelExtractor()

liked_features = []
disliked_features = []
failed = 0

start_time = time.time()

for i, (seg_id, track_id, start, end, is_positive, file_path, title, artist, pattern_name) in enumerate(segments):
    audio_path = f"/app/storage/{file_path}"
    
    if not Path(audio_path).exists():
        failed += 1
        progress_bar(i + 1, len(segments), "  Extract", f"liked:{len(liked_features)} disliked:{len(disliked_features)} fail:{failed}")
        continue
    
    mel = extractor.extract(audio_path, start, end)
    
    if mel is None:
        failed += 1
    elif is_positive:
        liked_features.append(mel)
    else:
        disliked_features.append(mel)
    
    progress_bar(i + 1, len(segments), "  Extract", f"liked:{len(liked_features)} disliked:{len(disliked_features)} fail:{failed}")
    
    # Clear GPU cache periodically
    if (i + 1) % 50 == 0:
        torch.cuda.empty_cache()

print()

extract_time = time.time() - start_time
print(f"\nИзвлечено за {extract_time:.1f}s:")
print(f"  Liked: {len(liked_features)}")
print(f"  Disliked: {len(disliked_features)}")
print(f"  Failed: {failed}")

if len(liked_features) < 5:
    print("\n⚠ Недостаточно liked сегментов после извлечения!")
    sys.exit(0)

# =============================================================================
# Train Model
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: ОБУЧЕНИЕ МОДЕЛИ")
print("=" * 60)

# Create model
model = SegmentPreferenceModel(
    input_channels=N_MELS,
    embedding_dim=256,
    learning_rate=0.001,
    temperature=0.07,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Training parameters
epochs = 100
batch_size = 32

# If no disliked segments, use random segments from non-liked tracks as negatives
if len(disliked_features) < 5:
    print("\n⚠ Мало disliked сегментов, добавляю случайные из не-liked треков...")
    
    cursor.execute("""
        SELECT id, file_path, duration
        FROM tracks 
        WHERE is_liked = false 
        AND file_path IS NOT NULL 
        AND duration > 30
        ORDER BY RANDOM()
        LIMIT 50
    """)
    
    random_tracks = cursor.fetchall()
    
    for track_id, file_path, duration in random_tracks:
        if len(disliked_features) >= len(liked_features):
            break
        
        audio_path = f"/app/storage/{file_path}"
        if not Path(audio_path).exists():
            continue
        
        # Random segment
        start = np.random.uniform(10, max(11, duration - 20))
        end = start + np.random.uniform(5, 15)
        
        mel = extractor.extract(audio_path, start, end)
        if mel is not None:
            disliked_features.append(mel)
    
    print(f"  Добавлено случайных: {len(disliked_features) - disliked_count}")

print(f"\nОбучение: {len(liked_features)} liked vs {len(disliked_features)} disliked")
print(f"Epochs: {epochs}, Batch: {batch_size}")

train_start = time.time()

def progress_callback(epoch, loss):
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{epochs}: loss={loss:.4f}")

history = model.train(
    liked_features=liked_features,
    disliked_features=disliked_features,
    epochs=epochs,
    batch_size=batch_size,
    progress_callback=progress_callback
)

train_time = time.time() - train_start
print(f"\nОбучение завершено за {train_time:.1f}s")

# Compute liked centroid for inference
model.compute_liked_centroid(liked_features)

# =============================================================================
# Save Model
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: СОХРАНЕНИЕ МОДЕЛИ")
print("=" * 60)

model.save(str(MODEL_PATH))
print(f"✓ Модель сохранена: {MODEL_PATH}")

# =============================================================================
# Test Model
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: ТЕСТИРОВАНИЕ")
print("=" * 60)

# Test on training data
print("\nПредсказания на обучающих данных:")

# Sample liked
print("\n  LIKED сегменты (должны быть высокие скоры):")
for i, mel in enumerate(liked_features[:5]):
    score = model.predict_preference(mel)
    print(f"    #{i+1}: {score:.3f}")

# Sample disliked
if disliked_features:
    print("\n  DISLIKED сегменты (должны быть низкие скоры):")
    for i, mel in enumerate(disliked_features[:5]):
        score = model.predict_preference(mel)
        print(f"    #{i+1}: {score:.3f}")

# =============================================================================
# Summary
# =============================================================================

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"Сегментов обработано: {len(liked_features) + len(disliked_features)}")
print(f"  Liked: {len(liked_features)}")
print(f"  Disliked: {len(disliked_features)}")
print(f"Final loss: {history['loss'][-1]:.4f}")
print(f"Final accuracy: {history['accuracy'][-1]:.2%}")
print(f"Время: {total_time:.1f}s")
print(f"Модель: {MODEL_PATH}")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)

cursor.close()
conn.close()
