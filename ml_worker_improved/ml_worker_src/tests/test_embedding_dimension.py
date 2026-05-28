"""
Property Test: Embedding Dimension Consistency

@tag Feature: ai-music-recommendations, Property 3: Embedding Dimension Consistency
**Validates: Requirements 3.1, 3.8**

For any audio file, the extracted embedding should have a fixed dimension
(e.g., 128 or 512 depending on model) and contain only finite numeric values.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.audio_analyzer import AudioAnalyzer


class TestEmbeddingDimensionConsistency:
    """Property tests for embedding dimension consistency."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an AudioAnalyzer instance."""
        return AudioAnalyzer()
    
    @given(
        duration=st.floats(min_value=0.5, max_value=10.0),
        sample_rate=st.sampled_from([22050, 44100]),
    )
    @settings(max_examples=100)
    def test_embedding_has_fixed_dimension(self, duration, sample_rate):
        """
        Property: All embeddings have the same fixed dimension.
        """
        analyzer = AudioAnalyzer()
        
        # Generate synthetic audio
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        # Mock librosa.load to return our synthetic audio
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            embedding = analyzer.extract_embeddings('/fake/path.mp3')
            
            # Verify dimension
            assert embedding.shape == (AudioAnalyzer.EMBEDDING_DIM,)
            assert len(embedding) == AudioAnalyzer.EMBEDDING_DIM
    
    @given(
        duration=st.floats(min_value=0.5, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_embedding_contains_only_finite_values(self, duration):
        """
        Property: Embeddings contain only finite numeric values (no NaN, no Inf).
        """
        analyzer = AudioAnalyzer()
        
        # Generate synthetic audio
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            embedding = analyzer.extract_embeddings('/fake/path.mp3')
            
            # Verify all values are finite
            assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
            assert not np.any(np.isnan(embedding)), "Embedding contains NaN values"
            assert not np.any(np.isinf(embedding)), "Embedding contains Inf values"
    
    @given(
        duration=st.floats(min_value=0.5, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_embedding_is_normalized(self, duration):
        """
        Property: Embeddings are normalized (L2 norm close to 1).
        """
        analyzer = AudioAnalyzer()
        
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            embedding = analyzer.extract_embeddings('/fake/path.mp3')
            
            # Verify normalization (L2 norm should be close to 1)
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01, f"Embedding norm {norm} is not close to 1"
    
    @given(
        duration1=st.floats(min_value=1.0, max_value=5.0),
        duration2=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_different_audio_same_dimension(self, duration1, duration2):
        """
        Property: Different audio files produce embeddings of the same dimension.
        """
        analyzer = AudioAnalyzer()
        sample_rate = 22050
        
        # Generate two different audio samples
        audio1 = np.random.randn(int(duration1 * sample_rate)).astype(np.float32)
        audio2 = np.random.randn(int(duration2 * sample_rate)).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            # First audio
            mock_load.return_value = (audio1, sample_rate)
            embedding1 = analyzer.extract_embeddings('/fake/path1.mp3')
            
            # Second audio
            mock_load.return_value = (audio2, sample_rate)
            embedding2 = analyzer.extract_embeddings('/fake/path2.mp3')
            
            # Both should have same dimension
            assert embedding1.shape == embedding2.shape
            assert len(embedding1) == len(embedding2) == AudioAnalyzer.EMBEDDING_DIM
    
    @given(
        duration=st.floats(min_value=0.5, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_embedding_dtype_is_float32(self, duration):
        """
        Property: Embeddings are stored as float32 for consistency.
        """
        analyzer = AudioAnalyzer()
        
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (audio, sample_rate)
            
            embedding = analyzer.extract_embeddings('/fake/path.mp3')
            
            assert embedding.dtype == np.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
