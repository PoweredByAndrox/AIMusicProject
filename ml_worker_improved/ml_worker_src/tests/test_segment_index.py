"""
Property-based tests for SegmentIndexManager.

Tests Properties 4, 5, 6 from design document.
Requirements: 2.1, 2.2, 2.3, 2.5
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from worker.segment_index import SegmentIndexManager, SegmentMatch


# Strategies for generating test data
embedding_strategy = arrays(
    dtype=np.float32,
    shape=(1280,),
    elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)
)

fragment_id_strategy = st.integers(min_value=1, max_value=100000)
track_id_strategy = st.integers(min_value=1, max_value=10000)
time_strategy = st.floats(min_value=0.0, max_value=600.0, allow_nan=False, allow_infinity=False)


class TestSegmentIndexInvariants:
    """
    Property 4: Инвариант добавления в индекс сегментов
    Property 5: Инвариант удаления из индекса сегментов
    
    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    
    @given(
        fragment_id=fragment_id_strategy,
        track_id=track_id_strategy,
        embedding=embedding_strategy,
        start_time=time_strategy,
        end_time=time_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_add_segment_invariant(
        self,
        fragment_id: int,
        track_id: int,
        embedding: np.ndarray,
        start_time: float,
        end_time: float
    ):
        """
        Property 4: For any fragment with embedding, after add_segment(),
        the index must contain that fragment (contains() returns true).
        
        **Validates: Requirements 2.1, 2.2**
        """
        # Ensure valid time range
        assume(end_time > start_time)
        
        # Normalize embedding to avoid numerical issues
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SegmentIndexManager(index_path=tmpdir)
            
            # Add segment
            index.add_segment(fragment_id, track_id, embedding, start_time, end_time)
            
            # Verify invariant: index must contain the fragment
            assert index.contains(fragment_id), \
                f"Index should contain fragment {fragment_id} after add_segment()"
            
            # Verify count increased
            assert index.count >= 1, \
                "Index count should be at least 1 after adding a segment"
    
    @given(
        fragment_id=fragment_id_strategy,
        track_id=track_id_strategy,
        embedding=embedding_strategy,
        start_time=time_strategy,
        end_time=time_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_remove_segment_invariant(
        self,
        fragment_id: int,
        track_id: int,
        embedding: np.ndarray,
        start_time: float,
        end_time: float
    ):
        """
        Property 5: For any fragment present in the index, after remove_segment(),
        the index must not contain that fragment (contains() returns false).
        
        **Validates: Requirements 2.3**
        """
        assume(end_time > start_time)
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SegmentIndexManager(index_path=tmpdir)
            
            # Add segment first
            index.add_segment(fragment_id, track_id, embedding, start_time, end_time)
            assert index.contains(fragment_id)
            
            # Remove segment
            index.remove_segment(fragment_id)
            
            # Verify invariant: index must NOT contain the fragment
            assert not index.contains(fragment_id), \
                f"Index should not contain fragment {fragment_id} after remove_segment()"
    
    @given(
        fragment_id=fragment_id_strategy,
        track_id=track_id_strategy,
        embedding=embedding_strategy
    )
    @settings(max_examples=50, deadline=None)
    def test_remove_nonexistent_segment_is_safe(
        self,
        fragment_id: int,
        track_id: int,
        embedding: np.ndarray
    ):
        """
        Removing a non-existent segment should not raise an error.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SegmentIndexManager(index_path=tmpdir)
            
            # Should not raise
            index.remove_segment(fragment_id)
            
            # Index should still be empty
            assert index.count == 0


class TestSegmentIndexPersistence:
    """
    Property 6: Round-trip персистентности индекса
    
    **Validates: Requirements 2.5**
    """
    
    @given(
        st.lists(
            st.tuples(
                fragment_id_strategy,
                track_id_strategy,
                embedding_strategy,
                time_strategy,
                time_strategy
            ),
            min_size=1,
            max_size=10,
            unique_by=lambda x: x[0]  # Unique fragment_ids
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_persistence_roundtrip(self, segments):
        """
        Property 6: For any segment index state, saving to disk and loading
        should produce an index with the same fragments and embeddings.
        
        **Validates: Requirements 2.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate index
            index = SegmentIndexManager(index_path=tmpdir)
            
            valid_fragments = []
            for frag_id, track_id, embedding, start, end in segments:
                # Skip invalid time ranges
                if end <= start:
                    continue
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                index.add_segment(frag_id, track_id, embedding, start, end)
                valid_fragments.append(frag_id)
            
            if not valid_fragments:
                return  # Skip if no valid segments
            
            original_count = index.count
            
            # Save index
            index.save()
            
            # Create new index and load
            loaded_index = SegmentIndexManager(index_path=tmpdir)
            
            # Verify all fragments are present
            for frag_id in valid_fragments:
                assert loaded_index.contains(frag_id), \
                    f"Loaded index should contain fragment {frag_id}"
            
            # Verify count matches
            assert loaded_index.count == original_count, \
                f"Loaded index count ({loaded_index.count}) should match original ({original_count})"


class TestSegmentSearch:
    """
    Tests for search_similar() functionality.
    
    **Validates: Requirements 3.2, 3.4**
    """
    
    def test_search_excludes_source_track(self):
        """
        Property 8: For any search with track_id, no result should have
        track_id equal to the source track_id.
        
        **Validates: Requirements 3.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SegmentIndexManager(index_path=tmpdir)
            
            # Add segments from different tracks
            embedding1 = np.random.randn(1280).astype(np.float32)
            embedding1 /= np.linalg.norm(embedding1)
            
            embedding2 = embedding1 + np.random.randn(1280).astype(np.float32) * 0.1
            embedding2 /= np.linalg.norm(embedding2)
            
            index.add_segment(1, track_id=100, embedding=embedding1, start_time=0, end_time=10)
            index.add_segment(2, track_id=200, embedding=embedding2, start_time=0, end_time=10)
            
            # Search excluding track 100
            results = index.search_similar(embedding1, k=10, exclude_track_id=100)
            
            # No result should be from track 100
            for match in results:
                assert match.track_id != 100, \
                    "Search results should not include segments from excluded track"
    
    def test_search_respects_limit(self):
        """
        Property 7: For any search with limit=N, results count should be <= N.
        
        **Validates: Requirements 3.2**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SegmentIndexManager(index_path=tmpdir)
            
            # Add many segments
            for i in range(20):
                embedding = np.random.randn(1280).astype(np.float32)
                embedding /= np.linalg.norm(embedding)
                index.add_segment(i, track_id=i, embedding=embedding, start_time=0, end_time=10)
            
            query = np.random.randn(1280).astype(np.float32)
            query /= np.linalg.norm(query)
            
            # Search with limit
            for limit in [1, 5, 10]:
                results = index.search_similar(query, k=limit)
                assert len(results) <= limit, \
                    f"Results count ({len(results)}) should be <= limit ({limit})"
    
    def test_search_respects_min_similarity(self):
        """
        Property 10: For any search with min_similarity=T, all results
        should have similarity >= T.
        
        **Validates: Requirements 6.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SegmentIndexManager(index_path=tmpdir)
            
            # Add segments
            for i in range(10):
                embedding = np.random.randn(1280).astype(np.float32)
                embedding /= np.linalg.norm(embedding)
                index.add_segment(i, track_id=i, embedding=embedding, start_time=0, end_time=10)
            
            query = np.random.randn(1280).astype(np.float32)
            query /= np.linalg.norm(query)
            
            min_sim = 0.3
            results = index.search_similar(query, k=10, min_similarity=min_sim)
            
            for match in results:
                assert match.similarity >= min_sim, \
                    f"Result similarity ({match.similarity}) should be >= min_similarity ({min_sim})"
