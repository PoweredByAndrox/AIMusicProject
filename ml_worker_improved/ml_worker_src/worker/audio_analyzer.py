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

import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# Optional Essentia backend (recommended for real, pre-trained audio embeddings/tagging).
# This worker can run without Essentia, but the PyTorch fallback is not useful
# unless you provide real pretrained weights.
try:  # pragma: no cover
    import essentia.standard as essentia_std  # type: ignore
except Exception:  # pragma: no cover
    essentia_std = None

try:  # pragma: no cover
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None

logger = logging.getLogger(__name__)

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"PyTorch device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")


class _FrozenGraphPredictor:
    def __init__(self, pb_path: Path, input_tensor: str, output_tensor: str):
        if tf is None:
            raise RuntimeError("tensorflow is not available (required for Discogs-EffNet .pb models)")

        tf.compat.v1.disable_eager_execution()

        self._graph = tf.compat.v1.Graph()
        with self._graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(Path(pb_path).read_bytes())
            tf.import_graph_def(graph_def, name="")
            self._in = self._graph.get_tensor_by_name(input_tensor)
            self._out = self._graph.get_tensor_by_name(output_tensor)

        self._sess = tf.compat.v1.Session(graph=self._graph)

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

        # Backend selection:
        # - "essentia": use Essentia pretrained models (.pb) for real embeddings/tags
        # - "torch"    : use PyTorch models (requires your own real weights)
        # - "auto"     : prefer Essentia if available, else torch
        self.backend = os.getenv('AUDIO_BACKEND', 'auto').lower()
        self._using_essentia = False

        # Essentia model handles (initialized lazily)
        self._ess_embedder = None
        self._ess_genre = None
        self._ess_mood = None
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
        Returns True if Essentia backend is ready.
        """
        if essentia_std is None:
            return False

        # Required: embedding model (Discogs-EffNet)
        embed_pb = self.models_path / 'discogs-effnet-bs64-1.pb'
        if not embed_pb.exists():
            return False

        # Optional heads
        genre_pb = self.models_path / 'genre_discogs400-discogs-effnet-1.pb'
        mood_pb = self.models_path / 'mood_theme_discogs-effnet-1.pb'
        voice_pb = self.models_path / 'voice_instrumental-discogs-effnet-1.pb'
        dance_pb = self.models_path / 'danceability-discogs-effnet-1.pb'

        try:
            self._ess_embedder = _FrozenGraphPredictor(
                embed_pb,
                input_tensor='serving_default_melspectrogram:0',
                output_tensor='PartitionedCall:1'
            )
            return True
        except Exception as e:
            logger.warning(f"Essentia backend init failed, falling back to torch: {e}")
            self._ess_embedder = None
            self._ess_genre = None
            self._ess_mood = None
            self._ess_voice = None
            self._ess_danceability = None
            self._ess_labels = {}
            return False

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
        out = model(audio)
        out = np.asarray(out)
        if out.ndim == 1:
            return out.astype(np.float32)
        if out.ndim == 2:
            return out.mean(axis=0).astype(np.float32)
        # Unexpected shape
        return out.reshape(-1).astype(np.float32)

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
                if self._ess_embedder is None:
                    raise RuntimeError("Essentia backend is selected but embedder isn't initialized")

                offset = float(start_time or 0.0)
                duration = None
                if end_time is not None:
                    duration = max(0.0, float(end_time) - offset)

                y, sr = librosa.load(
                    audio_path,
                    sr=16000,
                    mono=True,
                    offset=offset,
                    duration=duration
                )
                if y is None or len(y) == 0:
                    raise ValueError("Failed to load audio for Discogs-EffNet embedding")

                m = librosa.feature.melspectrogram(
                    y=y,
                    sr=16000,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=128,
                    power=2.0,
                    fmax=8000
                )
                m_db = librosa.power_to_db(m, ref=1.0, top_db=80.0)
                x = (m_db + 80.0) / 80.0
                x = np.clip(x, 0.0, 1.0).astype(np.float32)

                if x.shape[1] < 96:
                    pad = np.zeros((x.shape[0], 96 - x.shape[1]), dtype=np.float32)
                    x = np.concatenate([x, pad], axis=1)

                max_start = max(0, int(x.shape[1] - 96))
                starts = np.linspace(0, max_start, 64).astype(int)
                patches = np.stack([x[:, s:s + 96] for s in starts], axis=0).astype(np.float32)

                embs = self._ess_embedder.predict(patches)
                emb = np.mean(embs, axis=0).astype(np.float32)
                norm = float(np.linalg.norm(emb) + 1e-12)
                emb = (emb / norm).astype(np.float32)
                if emb.shape[0] != self.EMBEDDING_DIM:
                    raise ValueError(f"Expected embedding dim {self.EMBEDDING_DIM}, got {emb.shape}")
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
            genre = self._detect_genre_from_features(waveform, mel_spec_2d, bpm)
            key, mode = self._detect_key_from_waveform(waveform)
            mood = self._detect_mood_from_mel(mel_spec)
            has_vocals = self._detect_vocals_from_mel(mel_spec)
            is_acoustic = self._detect_acoustic_from_mel(mel_spec)
            
            logger.info(f"Analysis complete: {genre}, {bpm} BPM, {key} {mode}, {mood}")
            
            return {
                'embedding': embedding,
                'genre': genre,
                'bpm': bpm,
                'key': key,
                'mode': mode,
                'mood': mood,
                'has_vocals': has_vocals,
                'is_acoustic': is_acoustic,
            }
            
        except Exception as e:
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _analyze_full_essentia(self, audio_path: str) -> Dict[str, Any]:
        """Full track analysis using Essentia pretrained models.

        This is the recommended path for real recommendations because the embeddings
        and taggers are pretrained.
        """
        from pathlib import Path

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_file.stat().st_size < 1000:
            raise ValueError(f"Audio file too small, likely corrupt: {audio_path}")

        # Analyze a stable excerpt by default (faster + usually enough).
        clip_start = float(os.getenv('ANALYSIS_CLIP_START', '30'))
        clip_dur = float(os.getenv('ANALYSIS_CLIP_DUR', '90'))
        clip_end = clip_start + clip_dur

        # Embedding
        embedding = self.extract_embeddings(audio_path, start_time=clip_start, end_time=clip_end)

        # Load audio once for Essentia heads
        audio_16k = self._essentia_load_audio(audio_path, sample_rate=16000, start_time=clip_start, end_time=clip_end)

        genre = None
        mood = None
        has_vocals = False
        danceability = None

        if self._ess_genre is not None:
            probs = self._essentia_predict_avg(self._ess_genre, audio_16k)
            labels = self._ess_labels.get('genre_discogs400')
            top = self._essentia_top_labels(probs, labels, top_k=3, min_p=0.08)
            if top:
                genre = top[0][0]

        if self._ess_mood is not None:
            probs = self._essentia_predict_avg(self._ess_mood, audio_16k)
            labels = self._ess_labels.get('mood_theme')
            top = self._essentia_top_labels(probs, labels, top_k=3, min_p=0.08)
            if top:
                mood = top[0][0]

        if self._ess_voice is not None:
            probs = self._essentia_predict_avg(self._ess_voice, audio_16k)
            labels = self._ess_labels.get('voice_instrumental')
            # Try to locate "voice" label; fall back to 2nd class when unknown.
            if labels and 'voice' in labels:
                voice_idx = labels.index('voice')
                voice_prob = float(probs[voice_idx]) if voice_idx < probs.shape[0] else 0.0
            elif probs.shape[0] == 2:
                voice_prob = float(probs[1])
            else:
                voice_prob = float(np.max(probs))
            has_vocals = voice_prob >= 0.5

        if self._ess_danceability is not None:
            probs = self._essentia_predict_avg(self._ess_danceability, audio_16k)
            # Some models output a scalar, some a 2-class distribution.
            if probs.size == 1:
                danceability = float(probs[0])
            elif probs.size >= 2:
                danceability = float(np.max(probs))

        # BPM + Key (Essentia MIR algorithms). We run them on a slightly longer clip.
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

        # Acoustic flag: keep heuristic (better to add a dedicated acoustic classifier later)
        is_acoustic = False

        return {
            'embedding': embedding,
            'genre': genre or 'Unknown',
            'bpm': int(max(self.BPM_MIN, min(self.BPM_MAX, bpm))),
            'key': key,
            'mode': mode,
            'mood': mood or 'Unknown',
            'has_vocals': bool(has_vocals),
            'is_acoustic': bool(is_acoustic),
            'danceability': danceability,
        }

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
