"""
Property Test: Camelot Key Validity

@tag Feature: advanced-music-player, Property 19: Валидность Camelot ключа
**Validates: Requirements 27.1, 27.3, 27.4**

For any AI_DJ_Analyzer result:
- camelot_key matches format: number (1-12) + letter (A or B)
- energy_level is in range 1-10
- intro_point < outro_point
- bpm is in valid range (60-200)
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from unittest.mock import Mock, patch
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.dj_analyzer import (
    AIDJAnalyzer,
    DJAnalysisResult,
    to_camelot,
    is_valid_camelot_key,
    get_compatible_keys,
    CAMELOT_MAP,
    VALID_CAMELOT_KEYS,
)


class TestCamelotKeyValidity:
    """Property tests for Camelot key validity."""
    
    # All valid musical keys
    MUSICAL_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    SCALES = ['major', 'minor']
    
    @given(
        key=st.sampled_from(MUSICAL_KEYS),
        scale=st.sampled_from(SCALES),
    )
    @settings(max_examples=100)
    def test_camelot_key_format_valid(self, key, scale):
        """
        Property: to_camelot always returns a valid Camelot key format.
        
        Valid format: number (1-12) + letter (A or B)
        Examples: '8A', '11B', '1A', '12B'
        
        **Validates: Requirement 27.1**
        """
        camelot = to_camelot(key, scale)
        
        # Verify format with regex
        pattern = r'^(1[0-2]|[1-9])[AB]$'
        assert re.match(pattern, camelot), f"Invalid Camelot key format: {camelot}"
        
        # Verify using validation function
        assert is_valid_camelot_key(camelot), f"is_valid_camelot_key returned False for: {camelot}"
    
    @given(
        camelot_key=st.sampled_from(list(VALID_CAMELOT_KEYS)),
    )
    @settings(max_examples=50)
    def test_valid_camelot_keys_pass_validation(self, camelot_key):
        """
        Property: All keys in VALID_CAMELOT_KEYS pass validation.
        
        **Validates: Requirement 27.1**
        """
        assert is_valid_camelot_key(camelot_key)
    
    @given(
        invalid_key=st.text(min_size=0, max_size=10).filter(
            lambda x: not re.match(r'^(1[0-2]|[1-9])[AB]$', x)
        ),
    )
    @settings(max_examples=100)
    def test_invalid_camelot_keys_fail_validation(self, invalid_key):
        """
        Property: Invalid Camelot key formats fail validation.
        
        **Validates: Requirement 27.1**
        """
        assert not is_valid_camelot_key(invalid_key), f"Should be invalid: {invalid_key}"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=300.0),
        sample_rate=st.sampled_from([22050, 44100]),
    )
    @settings(max_examples=50, deadline=None)
    def test_dj_analysis_energy_level_in_range(self, duration, sample_rate):
        """
        Property: energy_level is always in range 1-10.
        
        **Validates: Requirement 27.3**
        """
        analyzer = AIDJAnalyzer()
        
        # Generate synthetic audio
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.5
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.get_duration') as mock_duration, \
             patch('librosa.beat.beat_track') as mock_beat, \
             patch('librosa.feature.chroma_cqt') as mock_chroma, \
             patch('librosa.feature.rms') as mock_rms, \
             patch('librosa.feature.spectral_centroid') as mock_centroid:
            
            mock_load.return_value = (audio, sample_rate)
            mock_duration.return_value = duration
            mock_beat.return_value = (120.0, np.array([0, 100, 200]))
            mock_chroma.return_value = np.random.rand(12, 100)
            mock_rms.return_value = np.random.rand(1, 100) * 0.3
            mock_centroid.return_value = np.random.rand(1, 100) * 3000
            
            result = analyzer.analyze('/fake/path.mp3', track_id=1)
            
            assert 1 <= result.energy_level <= 10, \
                f"Energy level {result.energy_level} out of range 1-10"
    
    @given(
        duration=st.floats(min_value=30.0, max_value=600.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_dj_analysis_intro_before_outro(self, duration):
        """
        Property: intro_point < outro_point for any track.
        
        **Validates: Requirement 27.4**
        """
        analyzer = AIDJAnalyzer()
        sample_rate = 22050
        
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.5
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.get_duration') as mock_duration, \
             patch('librosa.beat.beat_track') as mock_beat, \
             patch('librosa.feature.chroma_cqt') as mock_chroma, \
             patch('librosa.feature.rms') as mock_rms, \
             patch('librosa.feature.spectral_centroid') as mock_centroid, \
             patch('librosa.frames_to_time') as mock_frames:
            
            mock_load.return_value = (audio, sample_rate)
            mock_duration.return_value = duration
            
            # Generate beat times
            num_beats = int(duration * 2)  # ~2 beats per second
            beat_frames = np.arange(0, num_beats)
            beat_times = np.linspace(0, duration, num_beats)
            
            mock_beat.return_value = (120.0, beat_frames)
            mock_frames.return_value = beat_times
            mock_chroma.return_value = np.random.rand(12, 100)
            mock_rms.return_value = np.random.rand(1, 100) * 0.3
            mock_centroid.return_value = np.random.rand(1, 100) * 3000
            
            result = analyzer.analyze('/fake/path.mp3', track_id=1)
            
            assert result.intro_point < result.outro_point, \
                f"intro_point ({result.intro_point}) should be < outro_point ({result.outro_point})"
    
    @given(
        raw_bpm=st.floats(min_value=30.0, max_value=300.0),
    )
    @settings(max_examples=100)
    def test_dj_analysis_bpm_in_valid_range(self, raw_bpm):
        """
        Property: BPM is always in valid DJ range (60-200) after refinement.
        
        **Validates: Requirement 27.2**
        """
        analyzer = AIDJAnalyzer()
        
        # Test the BPM refinement directly
        refined_bpm = analyzer._refine_bpm(raw_bpm)
        
        assert 60 <= refined_bpm <= 200, \
            f"Refined BPM {refined_bpm} out of range 60-200 (raw: {raw_bpm})"
    
    @given(
        camelot_key=st.sampled_from(list(VALID_CAMELOT_KEYS)),
    )
    @settings(max_examples=50)
    def test_compatible_keys_are_valid(self, camelot_key):
        """
        Property: All compatible keys returned are valid Camelot keys.
        
        **Validates: Requirement 28.1**
        """
        compatible = get_compatible_keys(camelot_key)
        
        assert len(compatible) == 4, f"Should return 4 compatible keys, got {len(compatible)}"
        
        for key in compatible:
            assert is_valid_camelot_key(key), f"Compatible key {key} is not valid"
    
    @given(
        camelot_key=st.sampled_from(list(VALID_CAMELOT_KEYS)),
    )
    @settings(max_examples=50)
    def test_same_key_is_always_compatible(self, camelot_key):
        """
        Property: A key is always compatible with itself.
        
        **Validates: Requirement 28.1**
        """
        compatible = get_compatible_keys(camelot_key)
        
        assert camelot_key in compatible, \
            f"Key {camelot_key} should be compatible with itself"
    
    @given(
        camelot_key=st.sampled_from(list(VALID_CAMELOT_KEYS)),
    )
    @settings(max_examples=50)
    def test_ab_transition_is_compatible(self, camelot_key):
        """
        Property: A↔B transition (same number) is always compatible.
        
        **Validates: Requirement 28.1**
        """
        compatible = get_compatible_keys(camelot_key)
        
        number = camelot_key[:-1]
        letter = camelot_key[-1]
        other_letter = 'B' if letter == 'A' else 'A'
        ab_transition = f"{number}{other_letter}"
        
        assert ab_transition in compatible, \
            f"A↔B transition {ab_transition} should be compatible with {camelot_key}"
    
    @given(
        key=st.sampled_from(MUSICAL_KEYS),
        scale=st.sampled_from(SCALES),
    )
    @settings(max_examples=100)
    def test_camelot_mapping_is_deterministic(self, key, scale):
        """
        Property: Same key/scale always maps to same Camelot key.
        
        **Validates: Requirement 27.1**
        """
        camelot1 = to_camelot(key, scale)
        camelot2 = to_camelot(key, scale)
        
        assert camelot1 == camelot2, \
            f"Mapping should be deterministic: {key} {scale} -> {camelot1} vs {camelot2}"
    
    def test_all_musical_keys_have_camelot_mapping(self):
        """
        Property: Every musical key (major and minor) has a Camelot mapping.
        
        **Validates: Requirement 27.1**
        """
        for key in self.MUSICAL_KEYS:
            for scale in self.SCALES:
                camelot = to_camelot(key, scale)
                assert is_valid_camelot_key(camelot), \
                    f"No valid mapping for {key} {scale}"
    
    def test_camelot_wheel_completeness(self):
        """
        Property: All 24 Camelot keys (1A-12A, 1B-12B) are covered.
        
        **Validates: Requirement 27.1**
        """
        expected_keys = set()
        for num in range(1, 13):
            expected_keys.add(f"{num}A")
            expected_keys.add(f"{num}B")
        
        assert VALID_CAMELOT_KEYS == expected_keys, \
            f"Missing keys: {expected_keys - VALID_CAMELOT_KEYS}"


class TestDJAnalysisResultDataclass:
    """Tests for DJAnalysisResult dataclass invariants."""
    
    @given(
        track_id=st.integers(min_value=1, max_value=1000000),
        camelot_key=st.sampled_from(list(VALID_CAMELOT_KEYS)),
        bpm=st.floats(min_value=60.0, max_value=200.0),
        energy_level=st.integers(min_value=1, max_value=10),
        intro_point=st.floats(min_value=0.0, max_value=30.0),
        outro_offset=st.floats(min_value=10.0, max_value=300.0),
    )
    @settings(max_examples=100)
    def test_dj_analysis_result_creation(
        self, track_id, camelot_key, bpm, energy_level, intro_point, outro_offset
    ):
        """
        Property: DJAnalysisResult can be created with valid values.
        
        **Validates: Requirements 27.1, 27.2, 27.3, 27.4**
        """
        outro_point = intro_point + outro_offset
        
        result = DJAnalysisResult(
            track_id=track_id,
            camelot_key=camelot_key,
            bpm=bpm,
            energy_level=energy_level,
            intro_point=intro_point,
            outro_point=outro_point,
            drop_points=[],
            breakdown_points=[],
        )
        
        assert result.track_id == track_id
        assert is_valid_camelot_key(result.camelot_key)
        assert 60 <= result.bpm <= 200
        assert 1 <= result.energy_level <= 10
        assert result.intro_point < result.outro_point


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
