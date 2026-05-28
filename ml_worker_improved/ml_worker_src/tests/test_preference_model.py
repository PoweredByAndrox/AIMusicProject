"""
Property Test: Preference Model Round-Trip

@tag Feature: ai-music-recommendations, Property 8: Preference Model Round-Trip
**Validates: Requirements 8.3**

For any trained preference model, saving to disk and subsequent loading
should produce a model that returns identical predictions for the same input embeddings.
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.preference_model import PreferenceModel


class TestPreferenceModelRoundTrip:
    """Property tests for preference model round-trip consistency."""
    
    @given(
        n_samples=st.integers(min_value=5, max_value=20),
        embedding_dim=st.sampled_from([128, 256, 512]),
    )
    @settings(max_examples=100)
    def test_save_load_produces_identical_predictions(self, n_samples, embedding_dim):
        """
        Property: Save/load produces identical predictions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.npz')
            
            # Create and train model
            model = PreferenceModel(embedding_dim=embedding_dim)
            
            # Generate test embeddings
            test_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
            
            # Get predictions before save
            predictions_before = model.predict(test_embeddings)
            
            # Save model
            model.save(model_path)
            
            # Create new model and load
            loaded_model = PreferenceModel(embedding_dim=embedding_dim)
            loaded_model.load(model_path)
            
            # Get predictions after load
            predictions_after = loaded_model.predict(test_embeddings)
            
            # Verify predictions are identical
            np.testing.assert_array_almost_equal(
                predictions_before,
                predictions_after,
                decimal=6,
                err_msg="Predictions differ after save/load"
            )
    
    @given(
        n_liked=st.integers(min_value=5, max_value=20),
        n_all=st.integers(min_value=20, max_value=50),
    )
    @settings(max_examples=50)
    def test_trained_model_round_trip(self, n_liked, n_all):
        """
        Property: Trained model maintains predictions after save/load.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.npz')
            embedding_dim = 128
            
            # Create model
            model = PreferenceModel(embedding_dim=embedding_dim)
            
            # Generate training data
            liked_embeddings = np.random.randn(n_liked, embedding_dim).astype(np.float32)
            all_embeddings = np.random.randn(n_all, embedding_dim).astype(np.float32)
            
            # Train model
            model.train(liked_embeddings, all_embeddings, epochs=10)
            
            # Generate test embeddings
            test_embeddings = np.random.randn(10, embedding_dim).astype(np.float32)
            
            # Get predictions before save
            predictions_before = model.predict(test_embeddings)
            
            # Save and load
            model.save(model_path)
            loaded_model = PreferenceModel(embedding_dim=embedding_dim)
            loaded_model.load(model_path)
            
            # Get predictions after load
            predictions_after = loaded_model.predict(test_embeddings)
            
            # Verify predictions are identical
            np.testing.assert_array_almost_equal(
                predictions_before,
                predictions_after,
                decimal=6
            )
    
    @given(
        embedding_dim=st.sampled_from([128, 256, 512]),
    )
    @settings(max_examples=50)
    def test_model_weights_preserved(self, embedding_dim):
        """
        Property: Model weights are exactly preserved after save/load.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.npz')
            
            model = PreferenceModel(embedding_dim=embedding_dim)
            
            # Store original weights
            W1_before = model.W1.copy()
            b1_before = model.b1.copy()
            W2_before = model.W2.copy()
            b2_before = model.b2.copy()
            
            # Save and load
            model.save(model_path)
            loaded_model = PreferenceModel(embedding_dim=embedding_dim)
            loaded_model.load(model_path)
            
            # Verify weights are identical
            np.testing.assert_array_equal(W1_before, loaded_model.W1)
            np.testing.assert_array_equal(b1_before, loaded_model.b1)
            np.testing.assert_array_equal(W2_before, loaded_model.W2)
            np.testing.assert_array_equal(b2_before, loaded_model.b2)
    
    @given(
        n_samples=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_predict_returns_correct_shape(self, n_samples):
        """
        Property: Predict returns array of correct shape.
        """
        embedding_dim = 128
        model = PreferenceModel(embedding_dim=embedding_dim)
        
        embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        predictions = model.predict(embeddings)
        
        assert predictions.shape == (n_samples,)
    
    @given(
        embedding_dim=st.sampled_from([128, 256, 512]),
    )
    @settings(max_examples=50)
    def test_single_embedding_prediction(self, embedding_dim):
        """
        Property: Single embedding prediction works correctly.
        """
        model = PreferenceModel(embedding_dim=embedding_dim)
        
        # Single embedding (1D array)
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        prediction = model.predict(embedding)
        
        assert prediction.shape == (1,)
        assert np.isfinite(prediction[0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
