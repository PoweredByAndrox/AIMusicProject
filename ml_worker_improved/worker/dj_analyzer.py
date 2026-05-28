"""
AI DJ Analyzer - Audio analysis for DJ features using PyTorch GPU.

Requirements: 27.1, 27.2, 27.3, 27.4, 27.5, 27.6

Provides:
- Camelot key detection (1A-12A, 1B-12B)
- Precise BPM detection (±0.5 BPM accuracy) via All-In-One model
- Energy level analysis (1-10 scale)
- Structural point detection (intro, outro, drops, breakdowns) via All-In-One

Uses All-In-One neural network for structure analysis when available.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from .device_policy import resolve_torch_device

logger = logging.getLogger(__name__)

# Check CUDA availability
DEVICE = resolve_torch_device(task_name='AIDJAnalyzer', allow_cpu=False)

# Import structure analyzer (uses allin1) - REQUIRED
from .structure_analyzer import StructureAnalyzer

# Camelot Wheel mapping: (key, scale) -> Camelot notation
CAMELOT_MAP = {
    # Minor keys (A - inner wheel)
    ('A', 'minor'): '8A', ('E', 'minor'): '9A', ('B', 'minor'): '10A',
    ('F#', 'minor'): '11A', ('C#', 'minor'): '12A', ('G#', 'minor'): '1A',
    ('D#', 'minor'): '2A', ('A#', 'minor'): '3A', ('F', 'minor'): '4A',
    ('C', 'minor'): '5A', ('G', 'minor'): '6A', ('D', 'minor'): '7A',
    # Major keys (B - outer wheel)
    ('C', 'major'): '8B', ('G', 'major'): '9B', ('D', 'major'): '10B',
    ('A', 'major'): '11B', ('E', 'major'): '12B', ('B', 'major'): '1B',
    ('F#', 'major'): '2B', ('C#', 'major'): '3B', ('G#', 'major'): '4B',
    ('D#', 'major'): '5B', ('A#', 'major'): '6B', ('F', 'major'): '7B',
}

KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class DJAnalysisResult:
    """Result of AI DJ analysis."""
    track_id: int
    camelot_key: str
    bpm: float
    energy_level: int
    intro_point: float
    outro_point: float
    drop_points: List[float]
    breakdown_points: List[float]


class AIDJAnalyzer:
    """
    AI DJ Analyzer using PyTorch GPU acceleration.
    Uses All-In-One model for BPM and structure when available.
    """
    
    SAMPLE_RATE = 44100
    N_FFT = 2048
    HOP_LENGTH = 512
    BPM_MIN = 60
    BPM_MAX = 200
    
    def __init__(self):
        self.device = DEVICE
        self.structure_analyzer = StructureAnalyzer(model_name='harmonix-all')
        logger.info(f"AIDJAnalyzer initialized with All-In-One model on {self.device}")
    
    @torch.no_grad()
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, float]:
        """Load audio and return waveform on GPU with duration."""
        import soundfile as sf
        
        try:
            # Try soundfile first (supports MP3 via libsndfile)
            data, sr = sf.read(audio_path, dtype='float32')
        except Exception:
            # Fallback: convert to WAV using ffmpeg
            import subprocess
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            subprocess.run(
                ['ffmpeg', '-i', audio_path, '-ar', str(self.SAMPLE_RATE), '-ac', '1', '-y', tmp_path],
                capture_output=True, check=True
            )
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
        
        duration = waveform.shape[1] / self.SAMPLE_RATE
        waveform = waveform.to(self.device)
        
        return waveform, duration
    
    @torch.no_grad()
    def _detect_key_gpu(self, waveform: torch.Tensor) -> Tuple[str, str]:
        """Detect key using chroma features on GPU."""
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            return_complex=True
        )
        mag = torch.abs(spec)
        
        freqs = torch.fft.rfftfreq(self.N_FFT, 1.0 / self.SAMPLE_RATE).to(self.device)
        chroma = torch.zeros(12, mag.shape[1], device=self.device)
        
        for i, freq in enumerate(freqs):
            if freq > 20:
                midi = 12 * torch.log2(freq / 440.0) + 69
                chroma_bin = int(midi.item()) % 12
                if 0 <= chroma_bin < 12:
                    chroma[chroma_bin] += mag[i]
        
        chroma_avg = torch.mean(chroma, dim=1).cpu().numpy()
        
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        best_corr, best_key, best_mode = -1, 'C', 'minor'
        
        for i in range(12):
            rotated = np.roll(chroma_avg, -i)
            major_corr = np.corrcoef(rotated, major_profile)[0, 1] if not np.isnan(np.corrcoef(rotated, major_profile)[0, 1]) else 0
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1] if not np.isnan(np.corrcoef(rotated, minor_profile)[0, 1]) else 0
            
            if major_corr > best_corr:
                best_corr, best_key, best_mode = major_corr, KEYS[i], 'major'
            if minor_corr > best_corr:
                best_corr, best_key, best_mode = minor_corr, KEYS[i], 'minor'
        
        return best_key, best_mode
    
    @torch.no_grad()
    def _detect_bpm_gpu(self, waveform: torch.Tensor) -> float:
        """Detect BPM using onset detection on GPU."""
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            return_complex=True
        )
        mag = torch.abs(spec)
        
        flux = torch.sum(F.relu(mag[:, 1:] - mag[:, :-1]), dim=0)
        flux_np = flux.cpu().numpy()
        
        threshold = np.mean(flux_np) + 0.5 * np.std(flux_np)
        peaks = np.where(flux_np > threshold)[0]
        
        if len(peaks) < 4:
            return 128.0
        
        intervals = np.diff(peaks) * self.HOP_LENGTH / self.SAMPLE_RATE
        intervals = intervals[(intervals > 0.25) & (intervals < 1.5)]
        
        if len(intervals) < 2:
            return 128
        
        median_interval = np.median(intervals)
        bpm = 60.0 / median_interval
        
        if bpm < self.BPM_MIN:
            bpm *= 2
        elif bpm > self.BPM_MAX:
            bpm /= 2
        
        return int(round(max(self.BPM_MIN, min(self.BPM_MAX, bpm))))  # Round to whole number
    
    @torch.no_grad()
    def _calculate_energy_gpu(self, waveform: torch.Tensor) -> int:
        """Calculate energy level (1-10) on GPU."""
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()
        
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            return_complex=True
        )
        mag = torch.abs(spec)
        
        freqs = torch.fft.rfftfreq(self.N_FFT, 1.0 / self.SAMPLE_RATE).to(self.device)
        high_freq_mask = freqs > 4000
        high_energy = torch.mean(mag[high_freq_mask]).item() if high_freq_mask.any() else 0
        total_energy = torch.mean(mag).item()
        
        brightness = high_energy / (total_energy + 1e-10)
        
        energy_score = (rms * 20) * 0.6 + brightness * 0.4
        energy_level = int(round(energy_score * 10))
        
        return max(1, min(10, energy_level))
    
    @torch.no_grad()
    def _detect_structure_gpu(self, waveform: torch.Tensor, duration: float) -> Tuple[float, float, List[float], List[float]]:
        """Detect structural points on GPU."""
        frame_size = int(self.SAMPLE_RATE * 0.5)
        hop = int(self.SAMPLE_RATE * 0.25)
        
        energies = []
        for i in range(0, waveform.shape[1] - frame_size, hop):
            frame = waveform[:, i:i+frame_size]
            energy = torch.sqrt(torch.mean(frame ** 2)).item()
            energies.append((i / self.SAMPLE_RATE, energy))
        
        if not energies:
            return 0.0, duration * 0.9, [], []
        
        energy_values = [e[1] for e in energies]
        threshold_high = np.percentile(energy_values, 85)
        threshold_low = np.percentile(energy_values, 25)
        
        # Find intro (first significant energy)
        intro_point = 0.0
        for t, e in energies[:20]:
            if e > threshold_low:
                intro_point = max(0, t - 2)
                break
        
        # Find outro (last significant energy)
        outro_point = duration - 10
        for t, e in reversed(energies[-20:]):
            if e > threshold_low:
                outro_point = min(duration, t + 2)
                break
        
        # Find drops (sudden energy increases)
        drops = []
        for i in range(1, len(energies)):
            prev_e = energies[i-1][1]
            curr_e = energies[i][1]
            if curr_e > threshold_high and curr_e > prev_e * 1.5:
                drops.append(energies[i][0])
        
        # Find breakdowns (low energy sections)
        breakdowns = []
        for i in range(1, len(energies) - 1):
            prev_e = energies[i-1][1]
            curr_e = energies[i][1]
            next_e = energies[i+1][1]
            if curr_e < threshold_low and curr_e < prev_e and curr_e < next_e:
                breakdowns.append(energies[i][0])
        
        return intro_point, outro_point, drops[:5], breakdowns[:5]
    
    def analyze(self, audio_path: str, track_id: int = 0) -> DJAnalysisResult:
        """Perform full DJ analysis using All-In-One model (no fallback)."""
        logger.info(f"DJ analyzing: {audio_path}")
        
        # Use All-In-One model for BPM and structure
        structure = self.structure_analyzer.analyze(audio_path)
        
        # Get BPM from allin1 (more accurate)
        bpm = structure.bpm
        
        # Get structure points from allin1
        intro, outro = self.structure_analyzer.get_intro_outro(structure)
        drops = self.structure_analyzer.get_drop_points(structure)
        breakdowns = self.structure_analyzer.get_breakdown_points(structure)
        
        logger.info(f"All-In-One analysis: BPM={bpm:.1f}, segments={len(structure.segments)}")
        
        # Use our key detection (allin1 doesn't do key)
        waveform, _ = self._load_audio(audio_path)
        key, mode = self._detect_key_gpu(waveform)
        camelot_key = CAMELOT_MAP.get((key, mode), '8A')
        energy_level = self._calculate_energy_gpu(waveform)
        
        result = DJAnalysisResult(
            track_id=track_id,
            camelot_key=camelot_key,
            bpm=bpm,
            energy_level=energy_level,
            intro_point=round(intro, 2),
            outro_point=round(outro, 2),
            drop_points=[round(d, 2) for d in drops[:5]],
            breakdown_points=[round(b, 2) for b in breakdowns[:5]],
        )
        
        logger.info(f"DJ analysis complete: {camelot_key}, {bpm} BPM, energy {energy_level}")
        return result


def to_camelot(key: str, scale: str) -> str:
    """Convert musical key to Camelot notation."""
    key_normalize = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    key = key_normalize.get(key, key)
    scale = scale.lower()
    if scale not in ('major', 'minor'):
        scale = 'minor' if 'min' in scale.lower() else 'major'
    return CAMELOT_MAP.get((key, scale), '8A')


def is_valid_camelot_key(camelot_key: str) -> bool:
    """Validate Camelot key format."""
    import re
    if not isinstance(camelot_key, str):
        return False
    pattern = r'^(1[0-2]|[1-9])[AB]$'
    return bool(re.match(pattern, camelot_key))


def get_compatible_keys(camelot_key: str) -> List[str]:
    """Get harmonically compatible keys."""
    if not is_valid_camelot_key(camelot_key):
        return []
    
    number = int(camelot_key[:-1])
    letter = camelot_key[-1]
    
    compatible = [camelot_key]
    prev_num = 12 if number == 1 else number - 1
    next_num = 1 if number == 12 else number + 1
    
    compatible.append(f"{prev_num}{letter}")
    compatible.append(f"{next_num}{letter}")
    
    other_letter = 'B' if letter == 'A' else 'A'
    compatible.append(f"{number}{other_letter}")
    
    return compatible
