"""
PatternAnalyzer - Extract melodic patterns (pitch contour, chroma) for pattern matching.

This module provides:
- Pitch contour extraction (predominant melody)
- Chroma/HPCP extraction (12-tone harmonic representation)
- DTW-based pattern matching
- Window-based fingerprinting for fast search
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try to import librosa for chroma extraction
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - chroma extraction disabled")

# Try to import essentia for pitch extraction
try:
    import essentia.standard as essentia_std
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logger.warning("essentia not available - pitch extraction disabled")


class PatternAnalyzer:
    """
    Analyzes audio for melodic patterns using pitch contour and chroma features.
    """
    
    SAMPLE_RATE = 22050
    HOP_LENGTH = 512
    CHROMA_HOP = 512
    WINDOW_SIZE = 8.0  # seconds for pattern windows
    WINDOW_HOP = 1.0   # seconds between windows
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or os.getenv('STORAGE_PATH', '/app/storage'))
        self.patterns_path = self.storage_path / 'patterns'
        self.patterns_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PatternAnalyzer initialized (librosa={LIBROSA_AVAILABLE}, essentia={ESSENTIA_AVAILABLE})")
    
    def extract_chroma(self, audio_path: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Extract chroma features from audio file.
        
        Returns:
            np.ndarray of shape (12, time_frames) or None if extraction fails
        """
        if not LIBROSA_AVAILABLE:
            logger.error("librosa not available for chroma extraction")
            return None
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
            
            # Extract chroma using CQT (more robust than STFT-based)
            chroma = librosa.feature.chroma_cqt(
                y=y, 
                sr=sr, 
                hop_length=self.CHROMA_HOP,
                n_chroma=12
            )
            
            if normalize:
                # L2 normalize each frame
                norms = np.linalg.norm(chroma, axis=0, keepdims=True)
                norms[norms < 1e-10] = 1.0
                chroma = chroma / norms
            
            return chroma.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting chroma: {e}")
            return None
    
    def extract_pitch_contour(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract predominant pitch contour from audio.
        
        Returns:
            Dict with 'times', 'frequencies', 'confidence' arrays or None
        """
        if not ESSENTIA_AVAILABLE:
            logger.error("essentia not available for pitch extraction")
            return None
        
        try:
            # Load audio
            audio = essentia_std.MonoLoader(filename=audio_path, sampleRate=self.SAMPLE_RATE)()
            
            # Extract predominant melody
            pitch_extractor = essentia_std.PredominantPitchMelodia(
                frameSize=2048,
                hopSize=128,
                sampleRate=self.SAMPLE_RATE,
                minFrequency=80,
                maxFrequency=1760
            )
            
            pitch, confidence = pitch_extractor(audio)
            
            # Create time array
            times = np.arange(len(pitch)) * 128 / self.SAMPLE_RATE
            
            # Convert to MIDI for easier comparison
            midi = np.zeros_like(pitch)
            valid = pitch > 0
            midi[valid] = 12 * np.log2(pitch[valid] / 440.0) + 69
            
            return {
                'times': times.astype(np.float32),
                'frequencies': pitch.astype(np.float32),
                'midi': midi.astype(np.float32),
                'confidence': confidence.astype(np.float32)
            }
            
        except Exception as e:
            logger.error(f"Error extracting pitch contour: {e}")
            return None
    
    def extract_segment_chroma(
        self, 
        audio_path: str, 
        start_time: float, 
        end_time: float
    ) -> Optional[np.ndarray]:
        """Extract chroma for a specific segment."""
        if not LIBROSA_AVAILABLE:
            return None
        
        try:
            duration = end_time - start_time
            y, sr = librosa.load(
                audio_path, 
                sr=self.SAMPLE_RATE, 
                mono=True,
                offset=start_time,
                duration=duration
            )
            
            chroma = librosa.feature.chroma_cqt(
                y=y, 
                sr=sr, 
                hop_length=self.CHROMA_HOP,
                n_chroma=12
            )
            
            # L2 normalize
            norms = np.linalg.norm(chroma, axis=0, keepdims=True)
            norms[norms < 1e-10] = 1.0
            chroma = chroma / norms
            
            return chroma.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting segment chroma: {e}")
            return None
    
    def compute_dtw_distance(
        self, 
        query: np.ndarray, 
        target: np.ndarray,
        transpose_invariant: bool = True
    ) -> Tuple[float, int]:
        """
        Compute DTW distance between two chroma sequences.
        
        Args:
            query: Query chroma (12, T1)
            target: Target chroma (12, T2)
            transpose_invariant: If True, try all 12 transpositions
            
        Returns:
            (best_distance, best_shift) tuple
        """
        if not LIBROSA_AVAILABLE:
            return float('inf'), 0
        
        try:
            best_dist = float('inf')
            best_shift = 0
            
            shifts = range(12) if transpose_invariant else [0]
            
            for shift in shifts:
                # Circular shift for transposition
                if shift > 0:
                    shifted_query = np.roll(query, shift, axis=0)
                else:
                    shifted_query = query
                
                # Compute DTW
                D, wp = librosa.sequence.dtw(
                    shifted_query, 
                    target, 
                    metric='cosine'
                )
                
                # Normalized distance
                dist = D[-1, -1] / (query.shape[1] + target.shape[1])
                
                if dist < best_dist:
                    best_dist = dist
                    best_shift = shift
            
            return float(best_dist), best_shift
            
        except Exception as e:
            logger.error(f"Error computing DTW: {e}")
            return float('inf'), 0
    
    def create_chroma_fingerprint(self, chroma: np.ndarray) -> np.ndarray:
        """
        Create a compact fingerprint from chroma for fast candidate filtering.
        
        Returns:
            1D fingerprint vector
        """
        try:
            # Average chroma profile
            avg_chroma = np.mean(chroma, axis=1)
            
            # Chroma variance
            var_chroma = np.var(chroma, axis=1)
            
            # Dominant pitch class
            dominant = np.argmax(avg_chroma)
            
            # Combine into fingerprint
            fingerprint = np.concatenate([
                avg_chroma,           # 12 dims
                var_chroma,           # 12 dims
                [dominant / 12.0],    # 1 dim (normalized)
            ])
            
            return fingerprint.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error creating fingerprint: {e}")
            return np.zeros(25, dtype=np.float32)
    
    def build_window_index(
        self, 
        audio_path: str, 
        track_id: int
    ) -> List[Dict[str, Any]]:
        """
        Build window-based index for a track.
        
        Returns:
            List of window dicts with fingerprints
        """
        if not LIBROSA_AVAILABLE:
            return []
        
        try:
            # Load full audio
            y, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
            duration = len(y) / sr
            
            # Extract full chroma
            chroma = librosa.feature.chroma_cqt(
                y=y, 
                sr=sr, 
                hop_length=self.CHROMA_HOP,
                n_chroma=12
            )
            
            # Normalize
            norms = np.linalg.norm(chroma, axis=0, keepdims=True)
            norms[norms < 1e-10] = 1.0
            chroma = chroma / norms
            
            # Create windows
            windows = []
            window_samples = int(self.WINDOW_SIZE * sr / self.CHROMA_HOP)
            hop_samples = int(self.WINDOW_HOP * sr / self.CHROMA_HOP)
            
            for i in range(0, chroma.shape[1] - window_samples, hop_samples):
                window_chroma = chroma[:, i:i + window_samples]
                fingerprint = self.create_chroma_fingerprint(window_chroma)
                
                start_time = i * self.CHROMA_HOP / sr
                end_time = (i + window_samples) * self.CHROMA_HOP / sr
                
                windows.append({
                    'track_id': track_id,
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'fingerprint': fingerprint,
                })
            
            logger.info(f"Built {len(windows)} windows for track {track_id}")
            return windows
            
        except Exception as e:
            logger.error(f"Error building window index: {e}")
            return []
    
    def search_pattern(
        self,
        pattern_chroma: np.ndarray,
        candidate_windows: List[Dict[str, Any]],
        audio_paths: Dict[int, str],
        top_k: int = 20,
        transpose_invariant: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for pattern matches across candidate windows.
        
        Args:
            pattern_chroma: Query pattern chroma (12, T)
            candidate_windows: Pre-filtered candidate windows
            audio_paths: Dict mapping track_id to audio path
            top_k: Number of top matches to return
            transpose_invariant: Allow transposition matching
            
        Returns:
            List of match dicts sorted by score (best first)
        """
        if not LIBROSA_AVAILABLE:
            return []
        
        try:
            matches = []
            
            for window in candidate_windows:
                track_id = window['track_id']
                audio_path = audio_paths.get(track_id)
                
                if not audio_path:
                    continue
                
                # Extract window chroma
                window_chroma = self.extract_segment_chroma(
                    audio_path,
                    window['start_time'],
                    window['end_time']
                )
                
                if window_chroma is None:
                    continue
                
                # Compute DTW distance
                dist, shift = self.compute_dtw_distance(
                    pattern_chroma,
                    window_chroma,
                    transpose_invariant=transpose_invariant
                )
                
                # Convert distance to similarity score (0-1, higher is better)
                score = max(0, 1 - dist)
                
                matches.append({
                    'track_id': track_id,
                    'start_time': window['start_time'],
                    'end_time': window['end_time'],
                    'score': score,
                    'transpose_shift': shift,
                })
            
            # Sort by score (descending)
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching pattern: {e}")
            return []
    
    def save_chroma(self, chroma: np.ndarray, track_id: int) -> str:
        """Save chroma to file and return path."""
        filename = f"chroma_{track_id}.npy"
        filepath = self.patterns_path / filename
        np.save(filepath, chroma)
        return str(filepath)
    
    def load_chroma(self, path: str) -> Optional[np.ndarray]:
        """Load chroma from file."""
        try:
            return np.load(path)
        except Exception as e:
            logger.error(f"Error loading chroma: {e}")
            return None
    
    def save_pitch_contour(self, pitch_data: Dict[str, np.ndarray], track_id: int) -> str:
        """Save pitch contour to file and return path."""
        filename = f"pitch_{track_id}.npz"
        filepath = self.patterns_path / filename
        np.savez(filepath, **pitch_data)
        return str(filepath)
    
    def load_pitch_contour(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        """Load pitch contour from file."""
        try:
            data = np.load(path)
            return {k: data[k] for k in data.files}
        except Exception as e:
            logger.error(f"Error loading pitch contour: {e}")
            return None
