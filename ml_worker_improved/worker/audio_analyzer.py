"""
AudioAnalyzer - Audio analysis using PyTorch with full GPU acceleration.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7

Uses PyTorch for:
- Audio loading and preprocessing on GPU
- Mel spectrogram extraction on GPU
- Neural network inference on GPU
- All heavy computations on GPU

Models:
- CLAP (Contrastive Language-Audio Pretraining) for embeddings
- Or custom EfficientNet-based model for audio embeddings
"""

import json
import math
import os
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from collections import OrderedDict

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from .device_policy import resolve_torch_device, visible_cuda_summary

# Optional Essentia backend (recommended for real, pre-trained audio embeddings/tagging).
# This worker can run without Essentia, but the PyTorch fallback is not useful
# unless you provide real pretrained weights.
try:  # pragma: no cover
    import essentia.standard as essentia_std  # type: ignore
except Exception:  # pragma: no cover
    essentia_std = None

# TensorFlow is used to run Essentia's frozen .pb graphs directly (Discogs-EffNet).
# This avoids relying on optional Essentia TF wrapper classes that may be missing
# in some builds.
try:  # pragma: no cover
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None

logger = logging.getLogger(__name__)

# Check CUDA availability
DEVICE = resolve_torch_device(task_name='AudioAnalyzer', allow_cpu=True)
logger.info(f"PyTorch device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
logger.info("Visible CUDA devices: %s", visible_cuda_summary())


class _FrozenGraphPredictor:
    """Minimal TF1-style runner for a frozen GraphDef (.pb)."""

    def __init__(self, pb_path: Path, input_tensor: str, output_tensor: str):
        if tf is None:
            raise RuntimeError("tensorflow is not available (required for Discogs-EffNet .pb models)")

        # The Discogs-EffNet models are frozen graphs with a fixed batch size.
        tf.compat.v1.disable_eager_execution()

        self._graph = tf.compat.v1.Graph()
        with self._graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(Path(pb_path).read_bytes())
            tf.import_graph_def(graph_def, name="")

            self._in = self._graph.get_tensor_by_name(input_tensor)
            self._out = self._graph.get_tensor_by_name(output_tensor)

        # Allow coexistence with PyTorch by avoiding "grab all VRAM" behavior.
        cfg = tf.compat.v1.ConfigProto()
        try:  # pragma: no cover
            cfg.gpu_options.allow_growth = True  # type: ignore[attr-defined]
        except Exception:
            pass

        self._sess = tf.compat.v1.Session(graph=self._graph, config=cfg)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._sess.run(self._out, feed_dict={self._in: x})


class AudioEmbeddingModel(nn.Module):
    """
    Lightweight CNN for audio embeddings.
    Processes mel spectrograms on GPU.
    Output: 1280-dim embedding vector.
    """
    
    def __init__(self, embedding_dim: int = 1280):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, embedding_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, n_mels, time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x


class GenreClassifier(nn.Module):
    """Genre classifier head on top of embeddings."""
    
    def __init__(self, embedding_dim: int = 1280, num_genres: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, num_genres)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class MoodClassifier(nn.Module):
    """Mood classifier head."""
    
    def __init__(self, embedding_dim: int = 1280, num_moods: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, num_moods)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class AudioAnalyzer:
    """
    Analyzes audio files using PyTorch with full GPU acceleration.
    
    All heavy computations run on GPU:
    - Audio loading via torchaudio
    - Mel spectrogram extraction
    - Neural network inference
    - BPM/key detection
    """
    
    EMBEDDING_DIM = 1280
    BPM_MIN = 60
    BPM_MAX = 200
    SAMPLE_RATE = 22050
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    LOUDNESS_TARGET_LUFS = -14.0
    LOUDNESS_TARGET_TRUE_PEAK_DB = -1.5
    LOUDNESS_TARGET_LRA = 11.0
    
    KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    GENRES = [
        'Dubstep', 'Drum & Bass', 'House', 'Techno', 'Trance',
        'Electro', 'Breakbeat', 'Ambient', 'Downtempo', 'IDM',
        'Garage', 'Jungle', 'Hardcore', 'Hardstyle', 'Progressive',
        'Minimal', 'Deep House', 'Tech House', 'Bass', 'Halftime'
    ]
    
    MOODS = ['Energetic', 'Aggressive', 'Happy', 'Melancholic', 'Calm', 'Dark']
    
    def __init__(self, models_path: Optional[str] = None):
        self.models_path = Path(models_path or os.getenv('MODELS_PATH', '/app/models'))
        self.device = DEVICE
        self._full_audio_cache_max_items = max(0, int(os.getenv('FULL_AUDIO_CACHE_MAX_ITEMS', '4')))
        self._full_audio_cache: "OrderedDict[Tuple[str, int, int, int], np.ndarray]" = OrderedDict()

        # Backend selection:
        # - "essentia": use Essentia pretrained models (.pb) for real embeddings/tags
        # - "torch"    : use PyTorch models (requires your own real weights)
        # - "auto"     : prefer Essentia if available, else torch
        self.backend = os.getenv('AUDIO_BACKEND', 'auto').lower()
        self._using_essentia = False

        # Essentia model handles (initialized lazily)
        self._ess_embedder = None
        self._ess_genre = None
        self._ess_mood_heads = {}
        self._ess_voice = None
        self._ess_danceability = None
        self._ess_labels = {}
        
        # Initialize transforms on GPU
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            power=2.0
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(self.device)
        
        # Initialize models
        self.embedding_model = AudioEmbeddingModel(self.EMBEDDING_DIM).to(self.device)
        self.genre_classifier = GenreClassifier(self.EMBEDDING_DIM, len(self.GENRES)).to(self.device)
        self.mood_classifier = MoodClassifier(self.EMBEDDING_DIM, len(self.MOODS)).to(self.device)
        
        # Set to eval mode
        self.embedding_model.eval()
        self.genre_classifier.eval()
        self.mood_classifier.eval()
        
        # Load backend models
        self._load_models()
        
        self._models_loaded = True
        logger.info(f"AudioAnalyzer initialized on {self.device}")

    @staticmethod
    def _audio_file_signature(audio_path: str) -> Tuple[int, int]:
        stat = Path(audio_path).stat()
        return int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1_000_000_000))), int(stat.st_size)

    def _load_full_audio_cached(
        self,
        audio_path: str,
        sample_rate: int,
        loader,
    ) -> np.ndarray:
        """Load a whole-file mono waveform with a tiny LRU cache for repeated backfill calls."""
        if self._full_audio_cache_max_items <= 0:
            return np.asarray(loader(), dtype=np.float32).reshape(-1)

        audio_mtime_ns, audio_size = self._audio_file_signature(audio_path)
        cache_key = (str(audio_path), int(sample_rate), audio_mtime_ns, audio_size)

        cached = self._full_audio_cache.get(cache_key)
        if cached is not None:
            self._full_audio_cache.move_to_end(cache_key)
            return cached

        audio = np.asarray(loader(), dtype=np.float32).reshape(-1)
        stale_keys = [
            key for key in self._full_audio_cache.keys()
            if key[0] == str(audio_path) and key[1] == int(sample_rate)
        ]
        for stale_key in stale_keys:
            self._full_audio_cache.pop(stale_key, None)

        self._full_audio_cache[cache_key] = audio
        self._full_audio_cache.move_to_end(cache_key)
        while len(self._full_audio_cache) > self._full_audio_cache_max_items:
            self._full_audio_cache.popitem(last=False)

        return audio
    
    def _load_models(self) -> None:
        """Initialize the analysis backend.

        If Essentia is installed and the required .pb models are present, we use it by default
        because it provides *real* pretrained embeddings/taggers.

        The PyTorch path is kept only as a fallback for teams that have their own trained weights.
        """

        # 1) Try Essentia first (if requested)
        if self.backend in ('auto', 'essentia'):
            if self._load_essentia_models():
                self._using_essentia = True
                logger.info("Using Essentia backend (pretrained models)")
                return
            if self.backend == 'essentia':
                raise RuntimeError(
                    "AUDIO_BACKEND=essentia requested but Essentia models are not available. "
                    "Ensure Essentia is installed and required .pb models exist in MODELS_PATH. "
                    "You can download them via: python -m worker.download_models"
                )

        # 2) Fall back to PyTorch (if requested)
        if self.backend in ('auto', 'torch'):
            self._using_essentia = False
            self._load_torch_models()
            logger.info("Using PyTorch backend")
            return

        raise RuntimeError(f"Unknown AUDIO_BACKEND={self.backend}. Expected: auto|essentia|torch")

    def _load_torch_models(self) -> None:
        """Load PyTorch model weights.

        IMPORTANT: Randomly initialized networks produce *garbage* embeddings and labels.
        To prevent silent bad recommendations, we hard-fail if weights are missing.
        """
        embedding_path = self.models_path / 'pytorch_embedding_model.pt'
        genre_path = self.models_path / 'pytorch_genre_classifier.pt'
        mood_path = self.models_path / 'pytorch_mood_classifier.pt'

        missing = []
        if embedding_path.exists():
            self.embedding_model.load_state_dict(torch.load(embedding_path, map_location=self.device))
            logger.info("Loaded pretrained embedding model")
        else:
            missing.append(str(embedding_path))

        if genre_path.exists():
            self.genre_classifier.load_state_dict(torch.load(genre_path, map_location=self.device))
            logger.info("Loaded pretrained genre classifier")
        else:
            missing.append(str(genre_path))

        if mood_path.exists():
            self.mood_classifier.load_state_dict(torch.load(mood_path, map_location=self.device))
            logger.info("Loaded pretrained mood classifier")
        else:
            missing.append(str(mood_path))

        if missing:
            raise RuntimeError(
                "PyTorch weights are missing; random-weight fallback is disabled. "
                "Either install Essentia + download .pb models (preferred), or provide real PyTorch weights. "
                f"Missing: {missing}"
            )

    def _load_essentia_models(self) -> bool:
        """Load Essentia pretrained models if available.

        Expects the .pb models to be present in MODELS_PATH (download_models.py does this).
        Returns True if Essentia backend is ready (at minimum, embeddings model must work).
        """
        if essentia_std is None:
            return False

        # Required: embedding model (Discogs-EffNet)
        embed_pb = self.models_path / 'discogs-effnet-bs64-1.pb'
        if not embed_pb.exists():
            return False

        # Optional heads
        genre_pb = self.models_path / 'genre_discogs400-discogs-effnet-1.pb'
        voice_pb = self.models_path / 'voice_instrumental-discogs-effnet-1.pb'
        dance_pb = self.models_path / 'danceability-discogs-effnet-1.pb'
        mood_head_paths = {
            'Aggressive': self.models_path / 'mood_aggressive-discogs-effnet-1.pb',
            'Happy': self.models_path / 'mood_happy-discogs-effnet-1.pb',
            'Calm': self.models_path / 'mood_relaxed-discogs-effnet-1.pb',
            'Melancholic': self.models_path / 'mood_sad-discogs-effnet-1.pb',
        }

        try:
            # Run the frozen .pb graph directly via TensorFlow.
            # Input:  serving_default_melspectrogram:0  (float32, shape [64, 128, 96])
            # Output: PartitionedCall:1                (float32, shape [64, 1280])
            self._ess_embedder = _FrozenGraphPredictor(
                embed_pb,
                input_tensor='serving_default_melspectrogram:0',
                output_tensor='PartitionedCall:1'
            )
            logger.info("Loaded Discogs-EffNet embedding graph (TensorFlow)")
        except Exception as e:
            logger.warning(f"Failed to load Discogs-EffNet embedding graph: {e}")
            return False

        # Load optional classifier heads directly from frozen TF graphs.
        # These models operate on 1280-dim Discogs-EffNet embeddings.
        if genre_pb.exists():
            try:
                self._ess_genre = _FrozenGraphPredictor(
                    genre_pb,
                    input_tensor='serving_default_model_Placeholder:0',
                    output_tensor='PartitionedCall:0',
                )
                self._ess_labels['genre_discogs400'] = self._load_model_labels(
                    self.models_path / 'genre_discogs400-discogs-effnet-1.json'
                )
                logger.info("Loaded Essentia genre model (TensorFlow)")
            except Exception as e:
                logger.warning(f"Failed to load Essentia genre model (will use fallback): {e}")

        for mood_name, mood_pb in mood_head_paths.items():
            if not mood_pb.exists():
                continue
            try:
                self._ess_mood_heads[mood_name] = _FrozenGraphPredictor(
                    mood_pb,
                    input_tensor='model/Placeholder:0',
                    output_tensor='model/Softmax:0',
                )
                logger.info("Loaded Essentia mood model (TensorFlow): %s", mood_name)
            except Exception as e:
                logger.warning(f"Failed to load Essentia mood model {mood_name} (will use fallback): {e}")

        if voice_pb.exists():
            try:
                self._ess_voice = _FrozenGraphPredictor(
                    voice_pb,
                    input_tensor='model/Placeholder:0',
                    output_tensor='model/Softmax:0',
                )
                self._ess_labels['voice_instrumental'] = self._load_model_labels(
                    self.models_path / 'voice_instrumental-discogs-effnet-1.json'
                )
                logger.info("Loaded Essentia voice model (TensorFlow)")
            except Exception as e:
                logger.warning(f"Failed to load Essentia voice model (will use fallback): {e}")

        if dance_pb.exists():
            try:
                self._ess_danceability = _FrozenGraphPredictor(
                    dance_pb,
                    input_tensor='model/Placeholder:0',
                    output_tensor='model/Softmax:0',
                )
                logger.info("Loaded Essentia danceability model (TensorFlow)")
            except Exception as e:
                logger.warning(f"Failed to load Essentia danceability model (will use fallback): {e}")

        # Return True if at least the embedding model loaded successfully
        return self._ess_embedder is not None

    def _load_model_labels(self, json_path: Path) -> Optional[list]:
        """Load class labels from an Essentia model JSON metadata file.

        The exact schema can differ slightly across model packs, so we keep it tolerant.
        """
        try:
            if not json_path.exists():
                return None
            import json
            data = json.loads(json_path.read_text(encoding='utf-8'))
            # Common keys observed in Essentia model metadata
            for k in ('classes', 'class_labels', 'labels', 'classes_labels'):
                if k in data and isinstance(data[k], list):
                    return data[k]
            # Some packs nest labels
            if 'model' in data and isinstance(data['model'], dict):
                for k in ('classes', 'class_labels', 'labels'):
                    v = data['model'].get(k)
                    if isinstance(v, list):
                        return v
        except Exception as e:
            logger.debug(f"Failed to load labels from {json_path}: {e}")
        return None

    def _essentia_load_audio(self, audio_path: str, sample_rate: int, start_time: Optional[float], end_time: Optional[float]) -> np.ndarray:
        """Load mono audio with Essentia."""
        if essentia_std is None:
            raise RuntimeError("Essentia not available")

        # EasyLoader supports startTime/endTime in seconds.
        kwargs = {
            'filename': audio_path,
            'sampleRate': sample_rate,
        }
        if start_time is not None:
            kwargs['startTime'] = float(start_time)
        if end_time is not None:
            kwargs['endTime'] = float(end_time)
        return essentia_std.EasyLoader(**kwargs)()

    def _essentia_predict_avg(self, model, audio: np.ndarray) -> np.ndarray:
        """Run an Essentia TF model and average across time patches."""
        audio = np.asarray(audio, dtype=np.float32)
        if hasattr(model, 'predict'):
            out = model.predict(audio)
        else:
            out = model(audio)
        out = np.asarray(out)
        if out.ndim == 1:
            return out.astype(np.float32)
        if out.ndim == 2:
            return out.mean(axis=0).astype(np.float32)
        # Unexpected shape
        return out.reshape(-1).astype(np.float32)

    def _measure_loudness(self, audio_path: str) -> Optional[Dict[str, float]]:
        """Measure integrated loudness and derive a safe playback gain."""
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-nostats',
            '-i',
            audio_path,
            '-af',
            (
                f'loudnorm=I={self.LOUDNESS_TARGET_LUFS}:'
                f'TP={self.LOUDNESS_TARGET_TRUE_PEAK_DB}:'
                f'LRA={self.LOUDNESS_TARGET_LRA}:print_format=json'
            ),
            '-f',
            'null',
            '-',
        ]

        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
        except Exception as e:
            logger.warning(f"Loudness analysis failed to start for {audio_path}: {e}")
            return None

        if completed.returncode != 0:
            logger.warning(
                "Loudness analysis failed for %s: returncode=%s stderr_tail=%s",
                audio_path,
                completed.returncode,
                (completed.stderr or '')[-500:],
            )
            return None

        stderr_text = completed.stderr or ''
        json_start = stderr_text.rfind('{')
        json_end = stderr_text.rfind('}')
        if json_start == -1 or json_end <= json_start:
            logger.warning(f"Loudness analysis produced no JSON payload for {audio_path}")
            return None

        try:
            payload = json.loads(stderr_text[json_start:json_end + 1])
            integrated_lufs = float(payload['input_i'])
            true_peak_db = float(payload['input_tp'])
        except Exception as e:
            logger.warning(f"Failed to parse loudness payload for {audio_path}: {e}")
            return None

        if not math.isfinite(integrated_lufs):
            logger.warning(f"Integrated loudness is not finite for {audio_path}: {integrated_lufs}")
            return None

        gain_db = self.LOUDNESS_TARGET_LUFS - integrated_lufs
        if math.isfinite(true_peak_db) and gain_db > 0:
            gain_db = min(gain_db, self.LOUDNESS_TARGET_TRUE_PEAK_DB - true_peak_db)

        if not math.isfinite(gain_db):
            logger.warning(f"Derived loudness gain is not finite for {audio_path}: {gain_db}")
            return None

        logger.info(
            "Loudness measured: path=%s input_i=%.2fLUFS true_peak=%.2fdB gain=%.2fdB",
            audio_path,
            integrated_lufs,
            true_peak_db,
            gain_db,
        )

        return {
            'loudness_integrated_lufs': float(integrated_lufs),
            'loudness_true_peak_db': float(true_peak_db) if math.isfinite(true_peak_db) else None,
            'loudness_gain_db': float(gain_db),
            'loudness_target_lufs': float(self.LOUDNESS_TARGET_LUFS),
        }

    @staticmethod
    def _normalize_discogs_label(label: str) -> str:
        """Convert Discogs labels into cleaner UI-facing tags."""
        text = str(label).strip()
        if '---' in text:
            _, text = text.split('---', 1)
        text = ' '.join(text.replace('_', ' ').split())
        replacements = {
            'Drum n Bass': 'Drum & Bass',
            "Drum'n'Bass": 'Drum & Bass',
            'Hip Hop': 'Hip-Hop',
            'RnB/Swing': 'R&B/Swing',
            'Folk, World, & Country': 'Folk/World/Country',
        }
        return replacements.get(text, text)

    def _essentia_top_labels(self, probs: np.ndarray, labels: Optional[list], top_k: int = 5, min_p: float = 0.05) -> list:
        """Return top labels as list of (label, prob) sorted desc."""
        probs = np.asarray(probs).astype(np.float32)
        idxs = np.argsort(-probs)[:top_k]
        out = []
        for idx in idxs:
            p = float(probs[idx])
            if p < min_p:
                continue
            name = str(labels[idx]) if labels and idx < len(labels) else str(idx)
            out.append((name, p))
        return out

    @torch.no_grad()
    def _load_audio_gpu(self, audio_path: str, start_time: float = None, end_time: float = None) -> torch.Tensor:
        """
        Load audio file and transfer to GPU immediately.
        All subsequent processing happens on GPU.
        """
        import soundfile as sf
        
        # Load audio using soundfile (supports MP3 via ffmpeg)
        try:
            # Try soundfile first
            data, sr = sf.read(audio_path, dtype='float32')
        except Exception:
            # Fallback to torchaudio with sox backend
            import subprocess
            import tempfile
            import os
            
            # Convert to WAV using ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            subprocess.run(['ffmpeg', '-i', audio_path, '-ar', str(self.SAMPLE_RATE), '-ac', '1', '-y', tmp_path], 
                          capture_output=True, check=True)
            data, sr = sf.read(tmp_path, dtype='float32')
            os.unlink(tmp_path)
        
        # Convert to tensor
        if len(data.shape) == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            # Stereo to mono
            waveform = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        
        # Resample if needed
        if sr != self.SAMPLE_RATE:
            resampler = T.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Extract segment if times provided
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * self.SAMPLE_RATE)
            end_sample = int(end_time * self.SAMPLE_RATE)
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[1], end_sample)
            if end_sample > start_sample:
                waveform = waveform[:, start_sample:end_sample]
        
        # Move to GPU
        waveform = waveform.to(self.device)
        
        return waveform
    
    @torch.no_grad()
    def _extract_mel_spectrogram_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram entirely on GPU.
        """
        # Compute mel spectrogram on GPU
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize to [-1, 1]
        mel_spec_db = (mel_spec_db + 40) / 40  # Assuming range [-80, 0] dB
        mel_spec_db = torch.clamp(mel_spec_db, -1, 1)
        
        return mel_spec_db
    
    @torch.no_grad()
    def _create_patches_gpu(self, mel_spec: torch.Tensor, patch_size: int = 128) -> torch.Tensor:
        """
        Create overlapping patches from mel spectrogram on GPU.
        """
        # mel_spec shape: (1, n_mels, time)
        time_dim = mel_spec.shape[2]
        
        if time_dim < patch_size:
            # Pad if too short
            padding = patch_size - time_dim
            mel_spec = F.pad(mel_spec, (0, padding))
            time_dim = patch_size
        
        # Create patches with 50% overlap
        hop = patch_size // 2
        patches = []
        
        for i in range(0, time_dim - patch_size + 1, hop):
            patch = mel_spec[:, :, i:i+patch_size]
            patches.append(patch)
        
        if len(patches) == 0:
            patches.append(mel_spec[:, :, :patch_size])
        
        # Stack patches: (num_patches, 1, n_mels, patch_size)
        patches = torch.stack(patches, dim=0)
        
        return patches
    
    # =========================================================================
    # OPTIMIZED METHODS - Work with pre-loaded data (no file I/O)
    # =========================================================================
    
    @torch.no_grad()
    def _extract_embeddings_from_mel(self, mel_spec: torch.Tensor) -> np.ndarray:
        """Extract embeddings from pre-computed mel spectrogram."""
        try:
            patches = self._create_patches_gpu(mel_spec)
            embeddings = self.embedding_model(patches)
            embedding = torch.mean(embeddings, dim=0)
            embedding = F.normalize(embedding, p=2, dim=0)
            return embedding.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            raise
    
    @torch.no_grad()
    def _detect_bpm_from_waveform(self, waveform: torch.Tensor) -> int:
        """Detect BPM from pre-loaded waveform on GPU."""
        try:
            spec = torch.stft(
                waveform.squeeze(0),
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                return_complex=True
            )
            mag = torch.abs(spec)
            flux = torch.sum(F.relu(mag[:, 1:] - mag[:, :-1]), dim=0)
            
            flux_np = flux.cpu().numpy()
            threshold = np.mean(flux_np) + np.std(flux_np)
            peaks = np.where(flux_np > threshold)[0]
            
            if len(peaks) < 2:
                return 128
            
            intervals = np.diff(peaks) * self.HOP_LENGTH / self.SAMPLE_RATE
            intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
            
            if len(intervals) == 0:
                return 128
            
            bpm = 60.0 / np.median(intervals)
            
            if bpm < self.BPM_MIN:
                bpm *= 2
            elif bpm > self.BPM_MAX:
                bpm /= 2
            
            return int(round(max(self.BPM_MIN, min(self.BPM_MAX, bpm))))
        except Exception:
            return 128
    
    @torch.no_grad()
    def _detect_genre_from_features(self, waveform: torch.Tensor, mel_spec: torch.Tensor, bpm: float) -> str:
        """Detect genre from pre-loaded audio features."""
        try:
            n_mels = mel_spec.shape[0]
            
            # Bass/mid/high energy
            bass_end = max(1, n_mels // 8)
            mid_end = n_mels // 2
            
            bass_energy = torch.mean(mel_spec[:bass_end, :]).item()
            mid_energy = torch.mean(mel_spec[bass_end:mid_end, :]).item()
            high_energy = torch.mean(mel_spec[mid_end:, :]).item()
            
            if np.isnan(bass_energy): bass_energy = 0.0
            if np.isnan(mid_energy): mid_energy = 0.0
            if np.isnan(high_energy): high_energy = 0.0
            
            total = bass_energy + mid_energy + high_energy
            if total < 1e-10:
                bass_ratio, mid_ratio, high_ratio = 0.3, 0.5, 0.2
            else:
                bass_ratio = bass_energy / total
                mid_ratio = mid_energy / total
                high_ratio = high_energy / total
            
            # Spectral flux
            spec_diff = torch.diff(mel_spec, dim=1)
            spectral_flux = torch.mean(torch.abs(spec_diff)).item()
            
            # Spectral centroid
            freqs = torch.arange(n_mels, device=self.device, dtype=torch.float32)
            spectral_centroid = (torch.sum(freqs.unsqueeze(1) * mel_spec) / (torch.sum(mel_spec) + 1e-10)).item()
            
            # Dynamic range
            db_spec = self.amplitude_to_db(mel_spec.unsqueeze(0)).squeeze(0)
            dynamic_range = (torch.max(db_spec) - torch.min(db_spec)).item()
            
            return self._classify_genre_by_features(
                bpm, bass_ratio, mid_ratio, high_ratio,
                spectral_flux, spectral_centroid, dynamic_range
            )
        except Exception as e:
            logger.error(f"Error detecting genre: {e}")
            return 'Electronic'
    
    @torch.no_grad()
    def _detect_key_from_waveform(self, waveform: torch.Tensor) -> Tuple[str, str]:
        """Detect key from pre-loaded waveform."""
        try:
            spec = torch.stft(
                waveform.squeeze(0),
                n_fft=4096,
                hop_length=512,
                return_complex=True
            )
            mag = torch.abs(spec)
            
            freqs = torch.fft.rfftfreq(4096, 1/self.SAMPLE_RATE).to(self.device)
            chroma = torch.zeros(12, mag.shape[1], device=self.device)
            
            for i, freq in enumerate(freqs):
                if freq > 0:
                    midi = 12 * torch.log2(freq / 440.0) + 69
                    pitch_class = int(midi.item()) % 12
                    if 0 <= pitch_class < 12 and i < mag.shape[0]:
                        chroma[pitch_class] += mag[i]
            
            chroma_sum = torch.sum(chroma, dim=1)
            key_idx = torch.argmax(chroma_sum).item()
            
            major_profile = torch.tensor([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], device=self.device)
            minor_profile = torch.tensor([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], device=self.device)
            
            major_corr = F.cosine_similarity(chroma_sum.unsqueeze(0), major_profile.unsqueeze(0))
            minor_corr = F.cosine_similarity(chroma_sum.unsqueeze(0), minor_profile.unsqueeze(0))
            
            mode = 'major' if major_corr > minor_corr else 'minor'
            return self.KEYS[key_idx], mode
        except Exception:
            return 'C', 'major'
    
    @torch.no_grad()
    def _detect_mood_from_mel(self, mel_spec: torch.Tensor) -> str:
        """Detect mood from pre-computed mel spectrogram."""
        try:
            patches = self._create_patches_gpu(mel_spec)
            embeddings = self.embedding_model(patches)
            avg_embedding = torch.mean(embeddings, dim=0, keepdim=True)
            mood_probs = self.mood_classifier(avg_embedding)
            mood_idx = torch.argmax(mood_probs[0]).item()
            return self.MOODS[mood_idx]
        except Exception:
            return 'Energetic'
    
    @torch.no_grad()
    def _detect_vocals_from_mel(self, mel_spec: torch.Tensor) -> bool:
        """Detect vocals from pre-computed mel spectrogram."""
        try:
            if mel_spec.dim() == 3:
                mel_2d = mel_spec.squeeze(0)
            else:
                mel_2d = mel_spec
            
            n_mels = mel_2d.shape[0]
            vocal_start = n_mels // 4
            vocal_end = 3 * n_mels // 4
            
            vocal_energy = torch.mean(mel_2d[vocal_start:vocal_end, :]).item()
            total_energy = torch.mean(mel_2d).item()
            
            if total_energy < 1e-10:
                return False
            
            vocal_ratio = vocal_energy / total_energy
            vocal_std = torch.std(mel_2d[vocal_start:vocal_end, :]).item()
            
            return vocal_ratio > 0.4 and vocal_std > 0.1
        except Exception:
            return False
    
    @torch.no_grad()
    def _detect_acoustic_from_mel(self, mel_spec: torch.Tensor) -> bool:
        """Detect if track is acoustic from mel spectrogram."""
        try:
            if mel_spec.dim() == 3:
                mel_2d = mel_spec.squeeze(0)
            else:
                mel_2d = mel_spec
            
            n_mels = mel_2d.shape[0]
            high_start = 3 * n_mels // 4
            
            high_energy = torch.mean(mel_2d[high_start:, :]).item()
            low_energy = torch.mean(mel_2d[:n_mels//4, :]).item()
            
            if low_energy < 1e-10:
                return False
            
            brightness = high_energy / low_energy
            spectral_var = torch.var(mel_2d).item()
            
            return brightness < 0.5 and spectral_var < 0.1
        except Exception:
            return False
    
    # =========================================================================
    # ORIGINAL METHODS - For standalone use (load from file)
    # =========================================================================
    
    def _build_essentia_effnet_patches(self, audio: np.ndarray) -> np.ndarray:
        """Build Discogs-EffNet input patches for a single audio clip."""
        if essentia_std is None:
            raise RuntimeError("Essentia not available")

        tf_input = essentia_std.TensorflowInputMusiCNN()

        bands = [
            np.asarray(tf_input(np.asarray(frame, dtype=np.float32)), dtype=np.float32)
            for frame in essentia_std.FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=True)
        ]
        if not bands:
            bands = [np.zeros((96,), dtype=np.float32)]

        bands = np.asarray(bands, dtype=np.float32)
        patch_size = 128
        patch_hop = 62

        if bands.shape[0] < patch_size:
            pad_count = patch_size - bands.shape[0]
            pad_frame = bands[-1:] if bands.size else np.zeros((1, 96), dtype=np.float32)
            bands = np.concatenate([bands, np.repeat(pad_frame, pad_count, axis=0)], axis=0)

        max_start = max(0, int(bands.shape[0] - patch_size))
        starts = list(range(0, max_start + 1, patch_hop)) or [0]
        return np.asarray([bands[s:s + patch_size] for s in starts], dtype=np.float32)

    def _predict_essentia_patch_embeddings(self, patches: np.ndarray) -> np.ndarray:
        """Predict Discogs-EffNet embeddings for a patch tensor."""
        if self._ess_embedder is None:
            raise RuntimeError("Essentia backend is selected but embedder isn't initialized")

        batch_size = 64
        original_count = patches.shape[0]
        remainder = original_count % batch_size
        if remainder:
            pad = np.repeat(patches[-1:], batch_size - remainder, axis=0)
            patches = np.concatenate([patches, pad], axis=0)

        outputs = []
        for idx in range(0, patches.shape[0], batch_size):
            outputs.append(np.asarray(self._ess_embedder.predict(patches[idx:idx + batch_size]), dtype=np.float32))

        return np.concatenate(outputs, axis=0)[:original_count]

    def _normalize_embedding_vector(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize an embedding vector and validate the output dimension."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb) + 1e-12)
        emb = (emb / norm).astype(np.float32)
        if emb.shape[0] != self.EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {self.EMBEDDING_DIM}, got {emb.shape}")
        return emb

    def _slice_audio_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_time: Optional[float],
        end_time: Optional[float],
    ) -> np.ndarray:
        """Slice a mono audio array by time while preserving contiguous float32 output."""
        mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if mono_audio.size == 0:
            return mono_audio

        start_sample = 0 if start_time is None else max(0, int(float(start_time) * sample_rate))
        end_sample = mono_audio.shape[0] if end_time is None else min(
            mono_audio.shape[0],
            int(float(end_time) * sample_rate),
        )

        if end_sample <= start_sample:
            return np.zeros((0,), dtype=np.float32)

        return np.ascontiguousarray(mono_audio[start_sample:end_sample], dtype=np.float32)

    def _extract_essentia_embeddings_batch_from_audio(
        self,
        audio: np.ndarray,
        segments: List[Tuple[float, float]],
        sample_rate: int = 16000,
    ) -> List[Optional[np.ndarray]]:
        """Extract multiple segment embeddings from one preloaded Essentia waveform."""
        results: List[Optional[np.ndarray]] = [None] * len(segments)
        patches_by_segment: List[np.ndarray] = []
        patch_counts: List[int] = []
        segment_indexes: List[int] = []

        for idx, (start_time, end_time) in enumerate(segments):
            clip = self._slice_audio_array(audio, sample_rate, start_time, end_time)
            if clip.size == 0:
                continue

            patches = self._build_essentia_effnet_patches(clip)
            if patches.size == 0:
                continue

            patches_by_segment.append(patches)
            patch_counts.append(int(patches.shape[0]))
            segment_indexes.append(idx)

        if not patches_by_segment:
            return results

        all_patches = np.concatenate(patches_by_segment, axis=0)
        all_patch_embeddings = self._predict_essentia_patch_embeddings(all_patches)

        cursor = 0
        for idx, patch_count in zip(segment_indexes, patch_counts):
            patch_embeddings = all_patch_embeddings[cursor:cursor + patch_count]
            cursor += patch_count
            results[idx] = self._normalize_embedding_vector(np.mean(patch_embeddings, axis=0))

        return results

    @torch.no_grad()
    def _extract_essentia_patch_embeddings(
        self,
        audio_path: str,
        start_time: float = None,
        end_time: float = None,
    ) -> np.ndarray:
        """Extract patch-level Discogs-EffNet embeddings using Essentia's official input pipeline."""
        if self._ess_embedder is None:
            raise RuntimeError("Essentia backend is selected but embedder isn't initialized")
        if essentia_std is None:
            raise RuntimeError("Essentia not available")

        audio = self._essentia_load_audio(
            audio_path,
            sample_rate=16000,
            start_time=start_time,
            end_time=end_time,
        )
        if audio is None or len(audio) == 0:
            raise ValueError("Failed to load audio for Discogs-EffNet embedding")

        patches = self._build_essentia_effnet_patches(audio)
        return self._predict_essentia_patch_embeddings(patches)

    @torch.no_grad()
    def extract_embeddings(self, audio_path: str, start_time: float = None, end_time: float = None) -> np.ndarray:
        """
        Extract a track/segment embedding.

        - Essentia backend: Discogs-EffNet pretrained embedding (recommended)
        - PyTorch backend : your own embedding model (requires real weights)

        Returns: 1280-dim L2-normalized float32 vector.
        """
        try:
            if self._using_essentia:
                embeddings = self.extract_embeddings_batch(
                    audio_path,
                    [(0.0 if start_time is None else float(start_time), float(end_time) if end_time is not None else math.inf)],
                )
                emb = embeddings[0] if embeddings else None
                if emb is None:
                    raise ValueError(f"Failed to extract embedding for clip: {audio_path}")
                return emb

            # Load audio to GPU
            waveform = self._load_audio_gpu(audio_path, start_time, end_time)
            
            # Extract mel spectrogram on GPU
            mel_spec = self._extract_mel_spectrogram_gpu(waveform)
            
            # Create patches on GPU
            patches = self._create_patches_gpu(mel_spec)
            
            # Run embedding model on GPU
            embeddings = self.embedding_model(patches)
            
            # Average over patches
            embedding = torch.mean(embeddings, dim=0)
            
            # Already normalized in model, but ensure
            embedding = F.normalize(embedding, p=2, dim=0)
            
            # Move to CPU and convert to numpy
            return embedding.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            raise

    @torch.no_grad()
    def extract_embeddings_batch(
        self,
        audio_path: str,
        segments: List[Tuple[float, float]],
        preloaded_audio: Optional[np.ndarray] = None,
        preloaded_sample_rate: Optional[int] = None,
    ) -> List[Optional[np.ndarray]]:
        """
        Extract multiple segment embeddings with a single audio load when possible.

        Returns a list aligned with `segments`; failed/empty clips yield None.
        """
        if not segments:
            return []

        try:
            if self._using_essentia:
                audio = None
                sample_rate = 16000
                if preloaded_audio is not None:
                    audio_np = np.asarray(preloaded_audio, dtype=np.float32)
                    if audio_np.ndim > 1:
                        audio_np = audio_np.mean(axis=0)

                    source_rate = int(preloaded_sample_rate or sample_rate)
                    if source_rate != sample_rate:
                        waveform = torch.from_numpy(audio_np.reshape(1, -1))
                        resampler = T.Resample(source_rate, sample_rate)
                        audio = resampler(waveform).squeeze(0).cpu().numpy().astype(np.float32)
                    else:
                        audio = audio_np.reshape(-1)
                else:
                    audio = self._load_full_audio_cached(
                        audio_path,
                        sample_rate=sample_rate,
                        loader=lambda: self._essentia_load_audio(
                            audio_path,
                            sample_rate=sample_rate,
                            start_time=None,
                            end_time=None,
                        ),
                    )

                if audio is None or len(audio) == 0:
                    raise ValueError(f"Failed to load audio for Discogs-EffNet embedding: {audio_path}")

                normalized_segments = [
                    (
                        0.0 if start_time is None else float(start_time),
                        None if end_time is None or not math.isfinite(float(end_time)) else float(end_time),
                    )
                    for start_time, end_time in segments
                ]
                return self._extract_essentia_embeddings_batch_from_audio(audio, normalized_segments, sample_rate=sample_rate)

            waveform = None
            if preloaded_audio is not None:
                audio_np = np.asarray(preloaded_audio, dtype=np.float32)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=0)
                waveform = torch.from_numpy(audio_np.reshape(1, -1))
                source_rate = int(preloaded_sample_rate or self.SAMPLE_RATE)
                if source_rate != self.SAMPLE_RATE:
                    resampler = T.Resample(source_rate, self.SAMPLE_RATE)
                    waveform = resampler(waveform)
                waveform = waveform.to(self.device)
            else:
                waveform = self._load_audio_gpu(audio_path)

            results: List[Optional[np.ndarray]] = [None] * len(segments)
            total_samples = waveform.shape[1]

            for idx, (start_time, end_time) in enumerate(segments):
                start_sample = max(0, int(float(start_time) * self.SAMPLE_RATE))
                if end_time is None or not math.isfinite(float(end_time)):
                    end_sample = total_samples
                else:
                    end_sample = min(total_samples, int(float(end_time) * self.SAMPLE_RATE))

                if end_sample <= start_sample:
                    continue

                clip = waveform[:, start_sample:end_sample]
                if clip.numel() == 0:
                    continue

                mel_spec = self._extract_mel_spectrogram_gpu(clip)
                results[idx] = self._extract_embeddings_from_mel(mel_spec)

            return results
        except Exception as e:
            logger.error(f"Error extracting embedding batch: {e}")
            raise
    
    @torch.no_grad()
    def detect_genre(self, audio_path: str, bpm: float = None) -> str:
        """
        Detect genre using audio feature analysis.
        
        Uses BPM, spectral features, bass content, and rhythm patterns
        to classify into electronic music subgenres.
        
        Args:
            audio_path: Path to audio file
            bpm: Pre-calculated BPM (optional, will detect if not provided)
        """
        try:
            waveform = self._load_audio_gpu(audio_path)
            
            # Use provided BPM or detect
            if bpm is None:
                bpm = self._detect_bpm_internal(waveform)
            
            # Analyze spectral features
            mel_spec = self._extract_mel_spectrogram_gpu(waveform)
            
            # mel_spec shape is (1, n_mels, time) - squeeze to (n_mels, time)
            if mel_spec.dim() == 3:
                mel_spec = mel_spec.squeeze(0)
            
            # Handle edge cases
            if mel_spec.numel() == 0 or torch.isnan(mel_spec).any():
                logger.warning("Invalid mel spectrogram, using defaults")
                return self._classify_genre_by_features(
                    bpm=bpm, bass_ratio=0.3, mid_ratio=0.5, high_ratio=0.2,
                    spectral_flux=0.03, spectral_centroid=40.0, dynamic_range=45.0
                )
            
            # Ensure mel_spec has enough bins (should be n_mels x time)
            n_mels = mel_spec.shape[0]
            
            # Bass content (low frequencies ~0-15% of mel bins)
            bass_end = max(1, n_mels // 8)
            bass_energy = torch.mean(mel_spec[:bass_end, :]).item()
            
            # Mid content (~15-50% of mel bins)  
            mid_start = bass_end
            mid_end = n_mels // 2
            mid_energy = torch.mean(mel_spec[mid_start:mid_end, :]).item()
            
            # High content (50%+ of mel bins)
            high_energy = torch.mean(mel_spec[mid_end:, :]).item()
            
            # Handle NaN values
            if np.isnan(bass_energy): bass_energy = 0.0
            if np.isnan(mid_energy): mid_energy = 0.0
            if np.isnan(high_energy): high_energy = 0.0
            
            # Total energy
            total_energy = bass_energy + mid_energy + high_energy
            
            # Normalize (with fallback)
            if total_energy < 1e-10:
                bass_ratio, mid_ratio, high_ratio = 0.3, 0.5, 0.2
            else:
                bass_ratio = bass_energy / total_energy
                mid_ratio = mid_energy / total_energy
                high_ratio = high_energy / total_energy
            
            # Spectral flux (rhythm intensity)
            spec_diff = torch.diff(mel_spec, dim=1)
            spectral_flux = torch.mean(torch.abs(spec_diff)).item()
            
            # Spectral centroid (brightness)
            freqs = torch.arange(mel_spec.shape[0], device=self.device, dtype=torch.float32)
            spectral_centroid = torch.sum(freqs.unsqueeze(1) * mel_spec) / (torch.sum(mel_spec) + 1e-10)
            spectral_centroid = spectral_centroid.item()
            
            # Dynamic range
            db_spec = self.amplitude_to_db(mel_spec)
            dynamic_range = (torch.max(db_spec) - torch.min(db_spec)).item()
            
            # Log features for debugging
            logger.info(f"Genre features: bpm={bpm:.0f}, bass={bass_ratio:.3f}, mid={mid_ratio:.3f}, high={high_ratio:.3f}, flux={spectral_flux:.4f}, centroid={spectral_centroid:.1f}, dyn={dynamic_range:.1f}")
            
            # Genre classification based on features
            genre = self._classify_genre_by_features(
                bpm=bpm,
                bass_ratio=bass_ratio,
                mid_ratio=mid_ratio,
                high_ratio=high_ratio,
                spectral_flux=spectral_flux,
                spectral_centroid=spectral_centroid,
                dynamic_range=dynamic_range
            )
            
            return genre
            
        except Exception as e:
            logger.error(f"Error detecting genre: {e}")
            return 'Electronic'
    
    def _detect_bpm_internal(self, waveform: torch.Tensor) -> float:
        """Internal BPM detection from already loaded waveform."""
        try:
            # Onset detection on GPU
            spec = torch.abs(torch.stft(
                waveform,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=torch.hann_window(2048, device=self.device),
                return_complex=True
            ))
            
            onset_env = torch.mean(spec, dim=0)
            onset_env = onset_env / (torch.max(onset_env) + 1e-10)
            
            # Find peaks
            onset_np = onset_env.cpu().numpy()
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(onset_np, height=0.3, distance=10)
            
            if len(peaks) < 4:
                return 128.0
            
            intervals = np.diff(peaks) * 512 / self.SAMPLE_RATE
            intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
            
            if len(intervals) == 0:
                return 128.0
            
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval
            
            if bpm < self.BPM_MIN:
                bpm *= 2
            elif bpm > self.BPM_MAX:
                bpm /= 2
            
            return max(self.BPM_MIN, min(self.BPM_MAX, bpm))
            
        except Exception:
            return 128.0
    
    def _classify_genre_by_features(
        self,
        bpm: float,
        bass_ratio: float,
        mid_ratio: float,
        high_ratio: float,
        spectral_flux: float,
        spectral_centroid: float,
        dynamic_range: float
    ) -> str:
        """
        Classify genre based on audio features.
        
        Optimized for electronic/bass music (UKF, Monstercat style).
        
        Feature ranges (approximate):
        - bass_ratio: 0.25-0.45 (higher = more bass-heavy)
        - spectral_flux: 0.005-0.05 (higher = more rhythmic variation)
        - spectral_centroid: 40-80 (higher = brighter sound)
        - dynamic_range: 3-8 dB (normalized scale)
        """
        
        # Very high BPM genres (170+) - DnB territory
        if bpm >= 170:
            if bass_ratio > 0.32:
                return 'Drum & Bass'
            elif spectral_flux > 0.03:
                return 'Jungle'
            else:
                return 'Drum & Bass'
        
        # High BPM genres (155-170)
        if bpm >= 155:
            if bass_ratio > 0.35 and spectral_flux > 0.02:
                return 'Hardcore'
            elif bass_ratio > 0.32:
                return 'Drum & Bass'
            elif high_ratio > 0.35:
                return 'Hardstyle'
            else:
                return 'Trance'
        
        # Mid-high BPM (145-155) - Dubstep/DnB crossover
        if bpm >= 145:
            if bass_ratio > 0.33:
                return 'Dubstep'
            elif spectral_flux > 0.015:
                return 'Drum & Bass'
            else:
                return 'Trance'
        
        # Dubstep tempo (138-145)
        if bpm >= 138:
            if bass_ratio > 0.32:
                return 'Dubstep'
            elif bass_ratio > 0.28:
                return 'Bass'
            elif spectral_centroid > 65:
                return 'Electro'
            else:
                return 'Techno'
        
        # Standard EDM tempo (128-138)
        if bpm >= 128:
            if bass_ratio > 0.33:
                return 'Dubstep'
            elif bass_ratio > 0.30:
                return 'Bass'
            elif spectral_centroid > 60:
                return 'Electro'
            elif spectral_centroid < 50:
                return 'Deep House'
            else:
                return 'House'
        
        # House tempo (120-128)
        if bpm >= 120:
            if bass_ratio > 0.32:
                return 'Garage'
            elif spectral_centroid < 50:
                return 'Deep House'
            elif spectral_flux < 0.008:
                return 'Minimal'
            else:
                return 'House'
        
        # Slower tempos (100-120)
        if bpm >= 100:
            if bass_ratio > 0.35:
                return 'Halftime'
            elif spectral_flux < 0.006:
                return 'Ambient'
            elif spectral_centroid < 50:
                return 'Downtempo'
            else:
                return 'Breakbeat'
        
        # Very slow (< 100 BPM)
        if spectral_flux < 0.005:
            return 'Ambient'
        elif bass_ratio > 0.32:
            return 'Downtempo'
        else:
            return 'IDM'
    
    def _get_genre_scores(
        self,
        bpm: float,
        bass_ratio: float,
        mid_ratio: float,
        high_ratio: float,
        spectral_flux: float,
        spectral_centroid: float,
        dynamic_range: float
    ) -> list:
        """
        Get top-N genre tags with probability scores.
        Returns list of {"tag": "Genre", "prob": 0.85} sorted by probability.
        """
        scores = {}
        
        # BPM-based scoring
        if bpm >= 170:
            scores['Drum & Bass'] = scores.get('Drum & Bass', 0) + 0.4
            scores['Jungle'] = scores.get('Jungle', 0) + 0.2
        elif bpm >= 155:
            scores['Drum & Bass'] = scores.get('Drum & Bass', 0) + 0.3
            scores['Hardcore'] = scores.get('Hardcore', 0) + 0.2
            scores['Hardstyle'] = scores.get('Hardstyle', 0) + 0.15
            scores['Trance'] = scores.get('Trance', 0) + 0.15
        elif bpm >= 145:
            scores['Dubstep'] = scores.get('Dubstep', 0) + 0.3
            scores['Drum & Bass'] = scores.get('Drum & Bass', 0) + 0.2
            scores['Trance'] = scores.get('Trance', 0) + 0.15
        elif bpm >= 138:
            scores['Dubstep'] = scores.get('Dubstep', 0) + 0.25
            scores['Bass'] = scores.get('Bass', 0) + 0.2
            scores['Techno'] = scores.get('Techno', 0) + 0.15
            scores['Electro'] = scores.get('Electro', 0) + 0.1
        elif bpm >= 128:
            scores['House'] = scores.get('House', 0) + 0.25
            scores['Electro'] = scores.get('Electro', 0) + 0.2
            scores['Dubstep'] = scores.get('Dubstep', 0) + 0.15
            scores['Bass'] = scores.get('Bass', 0) + 0.1
        elif bpm >= 120:
            scores['House'] = scores.get('House', 0) + 0.3
            scores['Deep House'] = scores.get('Deep House', 0) + 0.2
            scores['Garage'] = scores.get('Garage', 0) + 0.15
        elif bpm >= 100:
            scores['Halftime'] = scores.get('Halftime', 0) + 0.2
            scores['Breakbeat'] = scores.get('Breakbeat', 0) + 0.2
            scores['Downtempo'] = scores.get('Downtempo', 0) + 0.15
        else:
            scores['Ambient'] = scores.get('Ambient', 0) + 0.25
            scores['Downtempo'] = scores.get('Downtempo', 0) + 0.2
            scores['IDM'] = scores.get('IDM', 0) + 0.15
        
        # Bass content scoring
        if bass_ratio > 0.35:
            scores['Dubstep'] = scores.get('Dubstep', 0) + 0.25
            scores['Bass'] = scores.get('Bass', 0) + 0.2
            scores['Drum & Bass'] = scores.get('Drum & Bass', 0) + 0.15
        elif bass_ratio > 0.30:
            scores['Bass'] = scores.get('Bass', 0) + 0.15
            scores['Dubstep'] = scores.get('Dubstep', 0) + 0.1
        elif bass_ratio < 0.25:
            scores['Trance'] = scores.get('Trance', 0) + 0.1
            scores['Ambient'] = scores.get('Ambient', 0) + 0.1
        
        # Spectral flux scoring (rhythm intensity)
        if spectral_flux > 0.03:
            scores['Jungle'] = scores.get('Jungle', 0) + 0.15
            scores['Hardcore'] = scores.get('Hardcore', 0) + 0.1
        elif spectral_flux < 0.008:
            scores['Ambient'] = scores.get('Ambient', 0) + 0.15
            scores['Minimal'] = scores.get('Minimal', 0) + 0.1
        
        # Spectral centroid scoring (brightness)
        if spectral_centroid > 65:
            scores['Electro'] = scores.get('Electro', 0) + 0.1
            scores['Hardstyle'] = scores.get('Hardstyle', 0) + 0.05
        elif spectral_centroid < 50:
            scores['Deep House'] = scores.get('Deep House', 0) + 0.1
            scores['Downtempo'] = scores.get('Downtempo', 0) + 0.05
        
        # Normalize scores to probabilities
        total = sum(scores.values()) or 1.0
        result = [
            {"tag": genre, "prob": round(score / total, 3)}
            for genre, score in sorted(scores.items(), key=lambda x: -x[1])
        ]
        
        return result[:5]  # Top 5 genres
    
    @torch.no_grad()
    def _get_mood_scores(self, mel_spec: torch.Tensor) -> list:
        """
        Get top-N mood tags with probability scores.
        Returns list of {"tag": "Mood", "prob": 0.72} sorted by probability.
        """
        try:
            patches = self._create_patches_gpu(mel_spec)
            embeddings = self.embedding_model(patches)
            avg_embedding = torch.mean(embeddings, dim=0, keepdim=True)
            mood_probs = self.mood_classifier(avg_embedding)
            
            probs = mood_probs[0].cpu().numpy()
            result = [
                {"tag": self.MOODS[i], "prob": round(float(probs[i]), 3)}
                for i in range(len(self.MOODS))
            ]
            result.sort(key=lambda x: -x['prob'])
            return result[:3]  # Top 3 moods
        except Exception:
            return [{"tag": "Energetic", "prob": 0.5}]
    
    @torch.no_grad()
    def detect_bpm(self, audio_path: str) -> float:
        """
        Detect BPM using onset detection on GPU.
        """
        try:
            # Load audio to GPU
            waveform = self._load_audio_gpu(audio_path)
            
            # Compute spectral flux for onset detection
            spec = torch.stft(
                waveform.squeeze(0),
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Spectral flux (difference between consecutive frames)
            flux = torch.sum(F.relu(mag[:, 1:] - mag[:, :-1]), dim=0)
            
            # Find peaks (onsets)
            flux_np = flux.cpu().numpy()
            threshold = np.mean(flux_np) + np.std(flux_np)
            peaks = np.where(flux_np > threshold)[0]
            
            if len(peaks) < 2:
                return 120.0
            
            # Calculate intervals between peaks
            intervals = np.diff(peaks) * self.HOP_LENGTH / self.SAMPLE_RATE
            intervals = intervals[intervals > 0.2]  # Filter out very short intervals
            intervals = intervals[intervals < 2.0]  # Filter out very long intervals
            
            if len(intervals) == 0:
                return 120.0
            
            # Estimate BPM from median interval
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval
            
            # Handle half-time / double-time
            if bpm < self.BPM_MIN:
                bpm *= 2
            elif bpm > self.BPM_MAX:
                bpm /= 2
            
            bpm = max(self.BPM_MIN, min(self.BPM_MAX, bpm))
            return int(round(bpm))  # Round to whole number
            
        except Exception as e:
            logger.error(f"Error detecting BPM: {e}")
            return 120
    
    @torch.no_grad()
    def detect_key(self, audio_path: str) -> Tuple[str, str]:
        """
        Detect musical key using chroma features on GPU.
        """
        try:
            # Load audio to GPU
            waveform = self._load_audio_gpu(audio_path)
            
            # Compute STFT
            spec = torch.stft(
                waveform.squeeze(0),
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Simple chroma computation
            # Map FFT bins to chroma bins
            freqs = torch.fft.rfftfreq(self.N_FFT, 1.0 / self.SAMPLE_RATE).to(self.device)
            
            # Compute chroma by summing energy in each pitch class
            chroma = torch.zeros(12, mag.shape[1], device=self.device)
            
            for i, freq in enumerate(freqs):
                if freq > 0:
                    # Convert frequency to MIDI note, then to chroma
                    midi = 12 * torch.log2(freq / 440.0) + 69
                    chroma_bin = int(midi.item()) % 12
                    if 0 <= chroma_bin < 12:
                        chroma[chroma_bin] += mag[i]
            
            # Average chroma over time
            chroma_avg = torch.mean(chroma, dim=1).cpu().numpy()
            
            # Krumhansl-Schmuckler key profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            best_corr = -1
            best_key = 'C'
            best_mode = 'major'
            
            for i in range(12):
                rotated = np.roll(chroma_avg, -i)
                
                major_corr = np.corrcoef(rotated, major_profile)[0, 1]
                minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]
                
                if not np.isnan(major_corr) and major_corr > best_corr:
                    best_corr = major_corr
                    best_key = self.KEYS[i]
                    best_mode = 'major'
                
                if not np.isnan(minor_corr) and minor_corr > best_corr:
                    best_corr = minor_corr
                    best_key = self.KEYS[i]
                    best_mode = 'minor'
            
            return best_key, best_mode
            
        except Exception as e:
            logger.error(f"Error detecting key: {e}")
            return 'C', 'major'

    @torch.no_grad()
    def detect_mood(self, audio_path: str) -> str:
        """Detect mood using GPU-accelerated classifier."""
        try:
            # Load and process on GPU
            waveform = self._load_audio_gpu(audio_path)
            mel_spec = self._extract_mel_spectrogram_gpu(waveform)
            patches = self._create_patches_gpu(mel_spec)
            
            # Get embeddings
            embeddings = self.embedding_model(patches)
            avg_embedding = torch.mean(embeddings, dim=0, keepdim=True)
            
            # Classify mood
            mood_probs = self.mood_classifier(avg_embedding)
            
            # Get top mood
            top_idx = torch.argmax(mood_probs[0]).item()
            return self.MOODS[top_idx]
            
        except Exception as e:
            logger.error(f"Error detecting mood: {e}")
            return 'Energetic'
    
    @torch.no_grad()
    def detect_vocals(self, audio_path: str) -> bool:
        """
        Detect presence of vocals using spectral analysis on GPU.
        """
        try:
            # Load audio to GPU
            waveform = self._load_audio_gpu(audio_path)
            
            # Compute spectrogram
            spec = torch.stft(
                waveform.squeeze(0),
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Frequency bins
            freqs = torch.fft.rfftfreq(self.N_FFT, 1.0 / self.SAMPLE_RATE).to(self.device)
            
            # Vocal frequency range (300 Hz - 3400 Hz)
            vocal_mask = (freqs >= 300) & (freqs <= 3400)
            
            # Calculate energy in vocal range vs total
            vocal_energy = torch.mean(mag[vocal_mask]).item()
            total_energy = torch.mean(mag).item()
            
            # If vocal range has significant energy relative to total
            return vocal_energy > total_energy * 0.4
            
        except Exception as e:
            logger.error(f"Error detecting vocals: {e}")
            return False
    
    @torch.no_grad()
    def detect_acoustic(self, audio_path: str) -> bool:
        """
        Detect if track is acoustic vs electronic using spectral flatness on GPU.
        """
        try:
            # Load audio to GPU
            waveform = self._load_audio_gpu(audio_path)
            
            # Compute spectrogram
            spec = torch.stft(
                waveform.squeeze(0),
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                return_complex=True
            )
            mag = torch.abs(spec) + 1e-10  # Avoid log(0)
            
            # Spectral flatness = geometric mean / arithmetic mean
            # High flatness = noise-like (electronic), low = tonal (acoustic)
            log_mag = torch.log(mag)
            geometric_mean = torch.exp(torch.mean(log_mag, dim=0))
            arithmetic_mean = torch.mean(mag, dim=0)
            
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            avg_flatness = torch.mean(flatness).item()
            
            # Low flatness = acoustic
            return avg_flatness < 0.1
            
        except Exception as e:
            logger.error(f"Error detecting acoustic: {e}")
            return False
    
    def analyze_full(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform full analysis of a track using PyTorch on GPU.
        
        OPTIMIZED: Loads audio once and reuses for all analysis steps.
        Includes timeout protection and GPU memory cleanup.
        
        Returns:
            Dict with all analysis results including:
            - embedding: 1280-dim vector
            - genre: Detected genre(s)
            - bpm: Tempo in BPM
            - key: Musical key
            - mode: Major/minor
            - mood: Detected mood
            - has_vocals: Boolean
            - is_acoustic: Boolean
        """
        # If Essentia backend is active, route to the pretrained pipeline.
        if self._using_essentia:
            return self._analyze_full_essentia(audio_path)
        import signal
        from pathlib import Path
        
        logger.info(f"Analyzing: {audio_path}")
        
        # Validate file exists and is readable
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        file_size = audio_file.stat().st_size
        if file_size < 1000:  # Less than 1KB is likely corrupt
            raise ValueError(f"Audio file too small ({file_size} bytes), likely corrupt: {audio_path}")
        
        if file_size > 500 * 1024 * 1024:  # More than 500MB
            raise ValueError(f"Audio file too large ({file_size / 1024 / 1024:.1f}MB): {audio_path}")
        
        try:
            # Load audio ONCE and keep on GPU
            waveform = self._load_audio_gpu(audio_path)
            
            # Validate waveform
            if waveform is None or waveform.numel() == 0:
                raise ValueError(f"Failed to load audio waveform: {audio_path}")
            
            if waveform.shape[1] < self.SAMPLE_RATE:  # Less than 1 second
                raise ValueError(f"Audio too short ({waveform.shape[1] / self.SAMPLE_RATE:.1f}s): {audio_path}")
            
            mel_spec = self._extract_mel_spectrogram_gpu(waveform)
            
            # Squeeze mel_spec for analysis (remove batch dim if present)
            if mel_spec.dim() == 3:
                mel_spec_2d = mel_spec.squeeze(0)
            else:
                mel_spec_2d = mel_spec
            
            # All analysis uses pre-loaded data
            embedding = self._extract_embeddings_from_mel(mel_spec)
            bpm = self._detect_bpm_from_waveform(waveform)
            
            # Get features for genre classification
            n_mels = mel_spec_2d.shape[0]
            bass_end = max(1, n_mels // 8)
            mid_end = n_mels // 2
            bass_energy = torch.mean(mel_spec_2d[:bass_end, :]).item()
            mid_energy = torch.mean(mel_spec_2d[bass_end:mid_end, :]).item()
            high_energy = torch.mean(mel_spec_2d[mid_end:, :]).item()
            total = bass_energy + mid_energy + high_energy
            if total < 1e-10:
                bass_ratio, mid_ratio, high_ratio = 0.3, 0.5, 0.2
            else:
                bass_ratio = bass_energy / total
                mid_ratio = mid_energy / total
                high_ratio = high_energy / total
            spec_diff = torch.diff(mel_spec_2d, dim=1)
            spectral_flux = torch.mean(torch.abs(spec_diff)).item()
            freqs = torch.arange(n_mels, device=self.device, dtype=torch.float32)
            spectral_centroid = (torch.sum(freqs.unsqueeze(1) * mel_spec_2d) / (torch.sum(mel_spec_2d) + 1e-10)).item()
            db_spec = self.amplitude_to_db(mel_spec_2d.unsqueeze(0)).squeeze(0)
            dynamic_range = (torch.max(db_spec) - torch.min(db_spec)).item()
            
            genre = self._classify_genre_by_features(
                bpm, bass_ratio, mid_ratio, high_ratio,
                spectral_flux, spectral_centroid, dynamic_range
            )
            genre_tags = self._get_genre_scores(
                bpm, bass_ratio, mid_ratio, high_ratio,
                spectral_flux, spectral_centroid, dynamic_range
            )
            
            key, mode = self._detect_key_from_waveform(waveform)
            mood = self._detect_mood_from_mel(mel_spec)
            mood_tags = self._get_mood_scores(mel_spec)
            has_vocals = self._detect_vocals_from_mel(mel_spec)
            is_acoustic = self._detect_acoustic_from_mel(mel_spec)
            
            logger.info(f"Analysis complete: {genre}, {bpm} BPM, {key} {mode}, {mood}")
            
            payload = {
                'embedding': embedding,
                'genre': genre,
                'genre_tags': genre_tags,
                'bpm': bpm,
                'key': key,
                'mode': mode,
                'mood': mood,
                'mood_tags': mood_tags,
                'has_vocals': has_vocals,
                'is_acoustic': is_acoustic,
            }
            loudness_payload = self._measure_loudness(audio_path)
            if loudness_payload:
                payload.update(loudness_payload)
            return payload
            
        except Exception as e:
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    @torch.no_grad()
    def analyze_full_segmented(self, audio_path: str, segment_duration: float = 20.0) -> Dict[str, Any]:
        """
        Perform full analysis with segment-based embedding aggregation.
        
        Instead of analyzing a single clip, splits track into segments and
        aggregates their embeddings for better representation.
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds (default 20s)
            
        Returns:
            Dict with analysis results including aggregated embedding
        """
        from pathlib import Path
        
        logger.info(f"Segmented analysis: {audio_path}")
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load full audio
            waveform = self._load_audio_gpu(audio_path)
            total_samples = waveform.shape[1]
            total_duration = total_samples / self.SAMPLE_RATE
            
            # Calculate segment parameters
            segment_samples = int(segment_duration * self.SAMPLE_RATE)
            hop_samples = segment_samples // 2  # 50% overlap
            
            # Extract embeddings for each segment
            segment_embeddings = []
            segment_start = 0
            
            while segment_start + segment_samples <= total_samples:
                segment_waveform = waveform[:, segment_start:segment_start + segment_samples]
                segment_mel = self._extract_mel_spectrogram_gpu(segment_waveform)
                segment_emb = self._extract_embeddings_from_mel(segment_mel)
                segment_embeddings.append(segment_emb)
                segment_start += hop_samples
            
            # Handle short tracks (less than one segment)
            if not segment_embeddings:
                mel_spec = self._extract_mel_spectrogram_gpu(waveform)
                segment_embeddings.append(self._extract_embeddings_from_mel(mel_spec))
            
            # Aggregate embeddings (mean pooling)
            segment_embeddings = np.array(segment_embeddings)
            aggregated_embedding = np.mean(segment_embeddings, axis=0)
            
            # L2 normalize the aggregated embedding
            norm = np.linalg.norm(aggregated_embedding)
            if norm > 0:
                aggregated_embedding = aggregated_embedding / norm
            
            logger.info(f"Aggregated {len(segment_embeddings)} segment embeddings")
            
            # Get other features from full track
            mel_spec = self._extract_mel_spectrogram_gpu(waveform)
            if mel_spec.dim() == 3:
                mel_spec_2d = mel_spec.squeeze(0)
            else:
                mel_spec_2d = mel_spec
            
            bpm = self._detect_bpm_from_waveform(waveform)
            
            # Get features for genre classification
            n_mels = mel_spec_2d.shape[0]
            bass_end = max(1, n_mels // 8)
            mid_end = n_mels // 2
            bass_energy = torch.mean(mel_spec_2d[:bass_end, :]).item()
            mid_energy = torch.mean(mel_spec_2d[bass_end:mid_end, :]).item()
            high_energy = torch.mean(mel_spec_2d[mid_end:, :]).item()
            total = bass_energy + mid_energy + high_energy
            if total < 1e-10:
                bass_ratio, mid_ratio, high_ratio = 0.3, 0.5, 0.2
            else:
                bass_ratio = bass_energy / total
                mid_ratio = mid_energy / total
                high_ratio = high_energy / total
            spec_diff = torch.diff(mel_spec_2d, dim=1)
            spectral_flux = torch.mean(torch.abs(spec_diff)).item()
            freqs = torch.arange(n_mels, device=self.device, dtype=torch.float32)
            spectral_centroid = (torch.sum(freqs.unsqueeze(1) * mel_spec_2d) / (torch.sum(mel_spec_2d) + 1e-10)).item()
            db_spec = self.amplitude_to_db(mel_spec_2d.unsqueeze(0)).squeeze(0)
            dynamic_range = (torch.max(db_spec) - torch.min(db_spec)).item()
            
            genre = self._classify_genre_by_features(
                bpm, bass_ratio, mid_ratio, high_ratio,
                spectral_flux, spectral_centroid, dynamic_range
            )
            genre_tags = self._get_genre_scores(
                bpm, bass_ratio, mid_ratio, high_ratio,
                spectral_flux, spectral_centroid, dynamic_range
            )
            
            key, mode = self._detect_key_from_waveform(waveform)
            mood = self._detect_mood_from_mel(mel_spec)
            mood_tags = self._get_mood_scores(mel_spec)
            has_vocals = self._detect_vocals_from_mel(mel_spec)
            is_acoustic = self._detect_acoustic_from_mel(mel_spec)
            
            logger.info(f"Segmented analysis complete: {genre}, {bpm} BPM, {key} {mode}")
            
            payload = {
                'embedding': aggregated_embedding.astype(np.float32),
                'segment_count': len(segment_embeddings),
                'genre': genre,
                'genre_tags': genre_tags,
                'bpm': bpm,
                'key': key,
                'mode': mode,
                'mood': mood,
                'mood_tags': mood_tags,
                'has_vocals': has_vocals,
                'is_acoustic': is_acoustic,
            }
            loudness_payload = self._measure_loudness(audio_path)
            if loudness_payload:
                payload.update(loudness_payload)
            return payload
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _build_essentia_analysis_payload(
        self,
        audio_path: str,
        patch_embeddings: np.ndarray,
        clip_start: float,
    ) -> Dict[str, Any]:
        """Assemble a full analysis payload from Discogs-EffNet patch embeddings."""
        embedding = np.mean(patch_embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(embedding) + 1e-12)
        embedding = (embedding / norm).astype(np.float32)
        embedding_for_classifiers = patch_embeddings.astype(np.float32)

        genre = None
        mood = None
        genre_tags = []
        mood_tags = []
        has_vocals = False
        danceability = None

        if self._ess_genre is not None:
            probs = self._essentia_predict_avg(self._ess_genre, embedding_for_classifiers)
            labels = self._ess_labels.get('genre_discogs400')
            top = self._essentia_top_labels(probs, labels, top_k=8, min_p=0.01)
            folded_scores = {}
            for label, prob in top:
                clean = self._normalize_discogs_label(label)
                folded_scores[clean] = folded_scores.get(clean, 0.0) + float(prob)
            genre_tags = [
                {"tag": tag, "prob": round(score, 3)}
                for tag, score in sorted(folded_scores.items(), key=lambda x: -x[1])[:5]
            ]
            if genre_tags:
                genre = genre_tags[0]["tag"]

        if self._ess_mood_heads:
            mood_scores = {}
            for mood_name, mood_head in self._ess_mood_heads.items():
                probs = self._essentia_predict_avg(mood_head, embedding_for_classifiers)
                if probs.size >= 2:
                    score = float(probs[1])
                elif probs.size == 1:
                    score = float(probs[0])
                else:
                    continue
                mood_scores[mood_name] = max(0.0, min(1.0, score))
            mood_tags = [
                {"tag": tag, "prob": round(score, 3)}
                for tag, score in sorted(mood_scores.items(), key=lambda x: -x[1])[:3]
            ]
            if mood_tags:
                mood = mood_tags[0]["tag"]

        if self._ess_voice is not None:
            probs = self._essentia_predict_avg(self._ess_voice, embedding_for_classifiers)
            labels = self._ess_labels.get('voice_instrumental')
            if labels and 'voice' in labels:
                voice_idx = labels.index('voice')
                voice_prob = float(probs[voice_idx]) if voice_idx < probs.shape[0] else 0.0
            elif probs.shape[0] == 2:
                voice_prob = float(probs[1])
            else:
                voice_prob = float(np.max(probs))
            has_vocals = voice_prob >= 0.5

        if self._ess_danceability is not None:
            probs = self._essentia_predict_avg(self._ess_danceability, embedding_for_classifiers)
            if probs.size == 1:
                danceability = float(probs[0])
            elif probs.size >= 2:
                danceability = float(probs[1])

        bpm_clip_dur = float(os.getenv('ANALYSIS_BPM_CLIP_DUR', '120'))
        bpm_clip_end = clip_start + bpm_clip_dur
        audio_44k = self._essentia_load_audio(audio_path, sample_rate=44100, start_time=clip_start, end_time=bpm_clip_end)

        bpm = 128
        key = 'C'
        mode = 'Major'
        try:
            bpm, _, _, _ = essentia_std.RhythmExtractor2013(method='multifeature')(audio_44k)
            bpm = int(round(float(bpm)))
        except Exception as e:
            logger.debug(f"Essentia BPM failed: {e}")

        try:
            key, scale, strength = essentia_std.KeyExtractor()(audio_44k)
            key = str(key)
            mode = 'Major' if str(scale).lower().startswith('major') else 'Minor'
        except Exception as e:
            logger.debug(f"Essentia key detection failed: {e}")

        payload = {
            'embedding': embedding,
            'genre': genre or 'Unknown',
            'genre_tags': genre_tags or None,
            'bpm': int(max(self.BPM_MIN, min(self.BPM_MAX, bpm))),
            'key': key,
            'mode': mode,
            'mood': mood or 'Unknown',
            'mood_tags': mood_tags or None,
            'has_vocals': bool(has_vocals),
            'is_acoustic': False,
            'danceability': danceability,
        }
        loudness_payload = self._measure_loudness(audio_path)
        if loudness_payload:
            payload.update(loudness_payload)
        return payload

    def analyze_full_batch(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple tracks in one AudioAnalyzer call.

        Optimized for throughput:
        - Phase 1: Parallel audio loading (ThreadPool, I/O-bound)
        - Phase 2: Batched EffNet embedding inference (GPU)
        - Phase 3: Parallel BPM/key/loudness extraction (ThreadPool, CPU-bound)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not audio_paths:
            return []

        if not self._using_essentia:
            return [self.analyze_full(audio_path) for audio_path in audio_paths]

        clip_start = float(os.getenv('ANALYSIS_CLIP_START', '30'))
        clip_dur = float(os.getenv('ANALYSIS_CLIP_DUR', '90'))
        clip_end = clip_start + clip_dur

        max_workers = min(len(audio_paths), int(os.getenv('ANALYSIS_IO_WORKERS', '6')))

        # ── Phase 1: Parallel audio loading + patch building ─────────
        def _load_and_patch(idx_path):
            idx, audio_path = idx_path
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            if audio_file.stat().st_size < 1000:
                raise ValueError(f"Audio file too small, likely corrupt: {audio_path}")

            audio = self._essentia_load_audio(
                audio_path,
                sample_rate=16000,
                start_time=clip_start,
                end_time=clip_end,
            )
            if audio is None or len(audio) == 0:
                raise ValueError(f"Failed to load audio for Discogs-EffNet embedding: {audio_path}")

            patches = self._build_essentia_effnet_patches(audio)
            return idx, patches

        patches_by_track = [None] * len(audio_paths)
        patch_counts = [0] * len(audio_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_load_and_patch, (i, p)) for i, p in enumerate(audio_paths)]
            for future in as_completed(futures):
                idx, patches = future.result()
                patches_by_track[idx] = patches
                patch_counts[idx] = int(patches.shape[0])

        logger.info(
            "Batch audio loaded: tracks=%d patches=%s",
            len(audio_paths),
            sum(patch_counts),
        )

        # ── Phase 2: Batched GPU embedding inference ─────────────────
        all_patches = np.concatenate(patches_by_track, axis=0)
        all_patch_embeddings = self._predict_essentia_patch_embeddings(all_patches)

        # Split embeddings back per track
        track_patch_embeddings = []
        cursor = 0
        for patch_count in patch_counts:
            track_patch_embeddings.append(all_patch_embeddings[cursor:cursor + patch_count])
            cursor += patch_count

        # ── Phase 3: Parallel CPU-bound analysis (BPM, key, loudness) ──
        def _build_payload(idx_args):
            idx, audio_path, patch_embeddings = idx_args
            return idx, self._build_essentia_analysis_payload(audio_path, patch_embeddings, clip_start)

        results = [None] * len(audio_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_build_payload, (i, audio_path, track_patch_embeddings[i]))
                for i, audio_path in enumerate(audio_paths)
            ]
            for future in as_completed(futures):
                idx, payload = future.result()
                results[idx] = payload

        logger.info("Batch analysis complete: tracks=%d", len(audio_paths))
        return results

    def _analyze_full_essentia(self, audio_path: str) -> Dict[str, Any]:
        """Full track analysis using Essentia pretrained models."""
        clip_start = float(os.getenv('ANALYSIS_CLIP_START', '30'))
        clip_dur = float(os.getenv('ANALYSIS_CLIP_DUR', '90'))
        clip_end = clip_start + clip_dur

        patch_embeddings = self._extract_essentia_patch_embeddings(
            audio_path,
            start_time=clip_start,
            end_time=clip_end,
        )
        return self._build_essentia_analysis_payload(audio_path, patch_embeddings, clip_start)

    def analyze_fragment(self, audio_path: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze a short fragment (used for segment search & UI stats)."""
        if end_time <= start_time:
            raise ValueError(f"Invalid fragment times: start={start_time}, end={end_time}")

        # Embedding (backend-specific)
        embedding = self.extract_embeddings(audio_path, start_time=start_time, end_time=end_time)

        # Lightweight numeric features on CPU (stable + fast)
        try:
            audio = self._essentia_load_audio(audio_path, sample_rate=44100, start_time=start_time, end_time=end_time)
        except Exception:
            # Fall back to torchaudio loader
            wav = self._load_audio_gpu(audio_path, start_time=start_time, end_time=end_time)
            audio = wav.detach().cpu().numpy().reshape(-1)

        if audio.size == 0:
            raise ValueError("Empty audio fragment")

        energy = float(np.sqrt(np.mean(audio ** 2)))
        spectral_centroid = self._spectral_centroid(audio, sample_rate=44100)

        events = self._simple_events(audio, sample_rate=44100)

        return {
            'embedding': embedding,
            'energy': energy,
            'spectral_centroid': spectral_centroid,
            'events': events,
        }

    def _spectral_centroid(self, audio: np.ndarray, sample_rate: int) -> float:
        """Compute spectral centroid (Hz) using a simple FFT-based estimator."""
        try:
            n_fft = 2048
            if audio.size < n_fft:
                pad = n_fft - audio.size
                audio = np.pad(audio, (0, pad))
            window = np.hanning(n_fft).astype(np.float32)
            frame = audio[:n_fft].astype(np.float32) * window
            mag = np.abs(np.fft.rfft(frame)) + 1e-12
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
            return float(np.sum(freqs * mag) / np.sum(mag))
        except Exception:
            return 0.0

    def _simple_events(self, audio: np.ndarray, sample_rate: int) -> list:
        """Very simple event heuristic: detect a big energy rise as a "drop"."""
        try:
            hop = int(0.1 * sample_rate)  # 100ms
            if hop <= 0:
                return []
            # Frame-wise RMS
            rms = []
            for i in range(0, max(0, audio.size - hop), hop):
                frame = audio[i:i + hop]
                if frame.size == 0:
                    continue
                rms.append(float(np.sqrt(np.mean(frame ** 2))))
            if len(rms) < 5:
                return []
            rms = np.asarray(rms, dtype=np.float32)
            # Detect largest positive jump
            diff = np.diff(rms)
            j = int(np.argmax(diff))
            if float(diff[j]) > float(np.std(diff) * 3.0):
                t = (j * hop) / float(sample_rate)
                return [{'type': 'drop', 'time': float(t), 'strength': float(diff[j])}]
            return []
        except Exception:
            return []
    
    def is_models_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded
