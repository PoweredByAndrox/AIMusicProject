"""
Property Tests for FAISS Index

@tag Feature: ai-music-recommendations, Property 5: FAISS Index Synchronization
@tag Feature: ai-music-recommendations, Property 10: Similar Search Returns K Results
**Validates: Requirements 4.5, 3.8, 10.1, 10.2, 10.3, 10.4**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.faiss_manager import FAISSManager


class TestFAISSIndexSynchronization:
    """
    Property 5: FAISS Index Synchronization
    
    For any system state, the number of records in FAISS index should equal
    the number of tracks with analysis_status='completed' in the database.
    """
    
    @given(
        track_ids=st.lists(st.integers(min_value=1, max_value=10000), min_size=1, max_size=50, unique=True),
    )
    @settings(max_examples=100)
    def test_index_count_matches_added_tracks(self, track_ids):
        """
        Property: Index count equals number of added tracks.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            # Add embeddings for each track
            for track_id in track_ids:
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                manager.add_embedding(track_id, embedding)
            
            # Verify count matches
            assert manager.count == len(track_ids)
    
    @given(
        track_ids=st.lists(st.integers(min_value=1, max_value=10000), min_size=5, max_size=50, unique=True),
        remove_count=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100)
    def test_index_count_after_removal(self, track_ids, remove_count):
        """
        Property: Index count decreases correctly after removal.
        """
        assume(remove_count < len(track_ids))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            # Add all embeddings
            for track_id in track_ids:
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                manager.add_embedding(track_id, embedding)
            
            initial_count = manager.count
            
            # Remove some tracks
            for track_id in track_ids[:remove_count]:
                manager.remove_embedding(track_id)
            
            # Verify count decreased
            assert manager.count == initial_count - remove_count
    
    @given(
        track_id=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_contains_after_add(self, track_id):
        """
        Property: Track is in index after adding.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            embedding = np.random.randn(128).astype(np.float32)
            manager.add_embedding(track_id, embedding)
            
            assert manager.contains(track_id)
    
    @given(
        track_id=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_not_contains_after_remove(self, track_id):
        """
        Property: Track is not in index after removal.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            embedding = np.random.randn(128).astype(np.float32)
            manager.add_embedding(track_id, embedding)
            manager.remove_embedding(track_id)
            
            assert not manager.contains(track_id)


class TestSimilarSearchReturnsKResults:
    """
    Property 10: Similar Search Returns K Results
    
    For any search query with parameter k, if the index has >= k records,
    the result should contain exactly k elements, sorted by similarity descending.
    """
    
    @given(
        num_tracks=st.integers(min_value=10, max_value=50),
        k=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_search_returns_k_results(self, num_tracks, k):
        """
        Property: Search returns exactly k results when index has >= k items.
        """
        assume(k <= num_tracks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            # Add embeddings
            for i in range(num_tracks):
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                manager.add_embedding(i + 1, embedding)
            
            # Search
            query = np.random.randn(128).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            results = manager.search_similar(query, k=k)
            
            assert len(results) == k
    
    @given(
        num_tracks=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=100)
    def test_search_results_sorted_by_similarity(self, num_tracks):
        """
        Property: Search results are sorted by similarity (descending).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            # Add embeddings
            for i in range(num_tracks):
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                manager.add_embedding(i + 1, embedding)
            
            # Search
            query = np.random.randn(128).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            results = manager.search_similar(query, k=min(5, num_tracks))
            
            # Verify sorted by similarity (descending)
            similarities = [sim for _, sim in results]
            assert similarities == sorted(similarities, reverse=True)
    
    @given(
        num_tracks=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=100)
    def test_search_results_have_valid_track_ids(self, num_tracks):
        """
        Property: All returned track IDs are valid (exist in index).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            track_ids = list(range(1, num_tracks + 1))
            
            # Add embeddings
            for track_id in track_ids:
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                manager.add_embedding(track_id, embedding)
            
            # Search
            query = np.random.randn(128).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            results = manager.search_similar(query, k=min(5, num_tracks))
            
            # Verify all returned IDs are valid
            for track_id, _ in results:
                assert track_id in track_ids
    
    @given(
        num_tracks=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=100)
    def test_search_similarity_in_valid_range(self, num_tracks):
        """
        Property: Similarity scores are in range [0, 1].
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            # Add embeddings
            for i in range(num_tracks):
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                manager.add_embedding(i + 1, embedding)
            
            # Search
            query = np.random.randn(128).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            results = manager.search_similar(query, k=min(5, num_tracks))
            
            # Verify similarity scores are in valid range
            for _, similarity in results:
                assert 0 <= similarity <= 1
    
    @given(
        num_tracks=st.integers(min_value=5, max_value=20),
        exclude_count=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_search_excludes_specified_ids(self, num_tracks, exclude_count):
        """
        Property: Excluded IDs are not in search results.
        """
        assume(exclude_count < num_tracks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FAISSManager(dimension=128, index_path=tmpdir)
            
            track_ids = list(range(1, num_tracks + 1))
            
            # Add embeddings
            for track_id in track_ids:
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                manager.add_embedding(track_id, embedding)
            
            # Search with exclusions
            exclude_ids = track_ids[:exclude_count]
            query = np.random.randn(128).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            results = manager.search_similar(query, k=num_tracks, exclude_ids=exclude_ids)
            
            # Verify excluded IDs are not in results
            result_ids = [track_id for track_id, _ in results]
            for excluded_id in exclude_ids:
                assert excluded_id not in result_ids


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
