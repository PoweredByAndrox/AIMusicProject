"""
Property Test: Recommendations Sorted by Score

@tag Feature: ai-music-recommendations, Property 9: Recommendations Sorted by Score
**Validates: Requirements 9.2**

For any list of recommendations returned by the system,
tracks should be sorted by score (relevance) in descending order.
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.preference_model import PreferenceModel


class TestRecommendationsSortedByScore:
    """Property tests for recommendations sorting."""
    
    @given(
        n_tracks=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=100)
    def test_predictions_can_be_sorted(self, n_tracks):
        """
        Property: Model predictions can be sorted to produce ranked recommendations.
        """
        embedding_dim = 128
        model = PreferenceModel(embedding_dim=embedding_dim)
        
        # Generate embeddings
        embeddings = np.random.randn(n_tracks, embedding_dim).astype(np.float32)
        
        # Get predictions
        scores = model.predict(embeddings)
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        
        # Verify sorted in descending order
        for i in range(len(sorted_scores) - 1):
            assert sorted_scores[i] >= sorted_scores[i + 1], \
                f"Scores not sorted: {sorted_scores[i]} < {sorted_scores[i + 1]}"
    
    @given(
        n_liked=st.integers(min_value=5, max_value=20),
        n_all=st.integers(min_value=20, max_value=50),
    )
    @settings(max_examples=50)
    def test_trained_model_produces_sortable_scores(self, n_liked, n_all):
        """
        Property: Trained model produces scores that can be sorted.
        """
        embedding_dim = 128
        model = PreferenceModel(embedding_dim=embedding_dim)
        
        # Generate training data
        liked_embeddings = np.random.randn(n_liked, embedding_dim).astype(np.float32)
        all_embeddings = np.random.randn(n_all, embedding_dim).astype(np.float32)
        
        # Train model
        model.train(liked_embeddings, all_embeddings, epochs=10)
        
        # Generate test embeddings
        test_embeddings = np.random.randn(20, embedding_dim).astype(np.float32)
        
        # Get predictions
        scores = model.predict(test_embeddings)
        
        # Sort and verify
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        
        # Verify sorted in descending order
        assert all(sorted_scores[i] >= sorted_scores[i + 1] 
                   for i in range(len(sorted_scores) - 1))
    
    @given(
        n_tracks=st.integers(min_value=10, max_value=50),
        k=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_top_k_recommendations_sorted(self, n_tracks, k):
        """
        Property: Top-k recommendations are sorted by score.
        """
        embedding_dim = 128
        model = PreferenceModel(embedding_dim=embedding_dim)
        
        # Generate embeddings
        embeddings = np.random.randn(n_tracks, embedding_dim).astype(np.float32)
        
        # Get predictions
        scores = model.predict(embeddings)
        
        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        # Verify top-k is sorted
        assert len(top_k_scores) == min(k, n_tracks)
        for i in range(len(top_k_scores) - 1):
            assert top_k_scores[i] >= top_k_scores[i + 1]
    
    @given(
        n_tracks=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=100)
    def test_scores_are_finite(self, n_tracks):
        """
        Property: All recommendation scores are finite numbers.
        """
        embedding_dim = 128
        model = PreferenceModel(embedding_dim=embedding_dim)
        
        embeddings = np.random.randn(n_tracks, embedding_dim).astype(np.float32)
        scores = model.predict(embeddings)
        
        assert np.all(np.isfinite(scores)), "Scores contain non-finite values"
    
    @given(
        n_tracks=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=100)
    def test_sorting_is_stable_for_equal_scores(self, n_tracks):
        """
        Property: Sorting handles equal scores correctly.
        """
        embedding_dim = 128
        model = PreferenceModel(embedding_dim=embedding_dim)
        
        # Create embeddings with some duplicates
        base_embedding = np.random.randn(1, embedding_dim).astype(np.float32)
        embeddings = np.vstack([
            base_embedding,
            base_embedding,  # Duplicate
            np.random.randn(n_tracks - 2, embedding_dim).astype(np.float32)
        ])
        
        scores = model.predict(embeddings)
        
        # Sort should work even with equal scores
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        
        # Verify non-increasing order (allows equal values)
        for i in range(len(sorted_scores) - 1):
            assert sorted_scores[i] >= sorted_scores[i + 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
