"""
Property Test: ML Characteristics Valid Range

@tag Feature: ai-music-recommendations, Property 4: ML Characteristics Valid Range
**Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6**

For any analyzed track:
- BPM should be in range [20, 300]
- Genre should be from predefined list
- Key should be one of 12 notes × 2 modes (major/minor)
- Mood should be from predefined list
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.audio_analyzer import AudioAnalyzer


class TestMLCharacteristicsValidRange:
    """Property tests for ML characteristics valid ranges."""
    
    @pytest.fixture
    def analyzer(self):
        return AudioAnalyzer()
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
        amplitude=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_bpm_in_valid_range(self, duration, amplitude):
        """
        Property: BPM is always in range [20, 300].
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        # Generate audio with varying characteristics
        audio = amplitude * np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            bpm = analyzer.detect_bpm('/fake/path.mp3')
            
            assert AudioAnalyzer.BPM_MIN <= bpm <= AudioAnalyzer.BPM_MAX, \
                f"BPM {bpm} is outside valid range [{AudioAnalyzer.BPM_MIN}, {AudioAnalyzer.BPM_MAX}]"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=100)
    def test_genre_from_predefined_list(self, duration):
        """
        Property: Genre is always from the predefined list.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            genre = analyzer.detect_genre('/fake/path.mp3')
            
            # Genre should be from predefined list or 'Unknown'
            valid_genres = AudioAnalyzer.GENRES + ['Unknown']
            assert genre in valid_genres, \
                f"Genre '{genre}' is not in valid genres list"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=100)
    def test_key_is_valid_note(self, duration):
        """
        Property: Key is one of 12 musical notes.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            key, mode = analyzer.detect_key('/fake/path.mp3')
            
            assert key in AudioAnalyzer.KEYS, \
                f"Key '{key}' is not a valid musical note"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=100)
    def test_mode_is_major_or_minor(self, duration):
        """
        Property: Mode is either 'major' or 'minor'.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            key, mode = analyzer.detect_key('/fake/path.mp3')
            
            assert mode in AudioAnalyzer.MODES, \
                f"Mode '{mode}' is not valid (expected 'major' or 'minor')"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=100)
    def test_mood_from_predefined_list(self, duration):
        """
        Property: Mood is always from the predefined list.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            mood = analyzer.detect_mood('/fake/path.mp3')
            
            assert mood in AudioAnalyzer.MOODS, \
                f"Mood '{mood}' is not in valid moods list"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=100)
    def test_vocals_is_boolean(self, duration):
        """
        Property: has_vocals is always a boolean.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            has_vocals = analyzer.detect_vocals('/fake/path.mp3')
            
            assert isinstance(has_vocals, bool), \
                f"has_vocals should be boolean, got {type(has_vocals)}"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=100)
    def test_acoustic_is_boolean(self, duration):
        """
        Property: is_acoustic is always a boolean.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            is_acoustic = analyzer.detect_acoustic('/fake/path.mp3')
            
            assert isinstance(is_acoustic, bool), \
                f"is_acoustic should be boolean, got {type(is_acoustic)}"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=50)
    def test_full_analysis_all_fields_valid(self, duration):
        """
        Property: Full analysis returns all fields with valid values.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            result = analyzer.analyze_full('/fake/path.mp3')
            
            # Verify all required fields exist
            assert 'embedding' in result
            assert 'genre' in result
            assert 'bpm' in result
            assert 'key' in result
            assert 'mode' in result
            assert 'mood' in result
            assert 'has_vocals' in result
            assert 'is_acoustic' in result
            
            # Verify all values are valid
            assert AudioAnalyzer.BPM_MIN <= result['bpm'] <= AudioAnalyzer.BPM_MAX
            assert result['genre'] in AudioAnalyzer.GENRES + ['Unknown']
            assert result['key'] in AudioAnalyzer.KEYS
            assert result['mode'] in AudioAnalyzer.MODES
            assert result['mood'] in AudioAnalyzer.MOODS
            assert isinstance(result['has_vocals'], bool)
            assert isinstance(result['is_acoustic'], bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
