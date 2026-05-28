"""
SegmentIndexManager - FAISS index for audio segment embeddings.

Manages a separate index from track-level embeddings for searching
similar audio moments across the library.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import os
import json
import logging
from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass, asdict
import threading

import numpy as np
import faiss

logger = logging.getLogger(__name__)


@dataclass
class SegmentMatch:
    """Result of a segment similarity search."""
    fragment_id: int
    track_id: int
    start_time: float
    end_time: float
    similarity: float  # 0.0 - 1.0


@dataclass
class SegmentMetadata:
    """Metadata for an indexed segment."""
    fragment_id: int
    track_id: int
    start_time: float
    end_time: float
    faiss_idx: int


class SegmentIndexManager:
    """
    Manages FAISS index for segment/fragment embeddings.
    
    Separate from track-level index to enable searching for
    similar audio moments across the entire library.
    
    Features:
    - Add/remove segment embeddings
    - Search for similar segments
    - Persist index to disk
    - Thread-safe operations
    """
    
    def __init__(
        self,
        dimension: int = 1280,  # Discogs-EffNet embedding size
        index_path: Optional[str] = None
    ):
        self.dimension = dimension
        self.index_path = Path(index_path or os.getenv('FAISS_INDEX_PATH', '/app/faiss_index')) / 'segments'
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._index: Optional[faiss.Index] = None
        self._metadata: Dict[int, SegmentMetadata] = {}  # fragment_id -> metadata
        self._faiss_to_fragment: Dict[int, int] = {}  # faiss_idx -> fragment_id
        self._next_idx = 0
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        index_file = self.index_path / 'segments.index'
        metadata_file = self.index_path / 'segments_metadata.json'
        
        if index_file.exists() and metadata_file.exists():
            try:
                self.load()
                logger.info(f"Loaded segment index with {len(self._metadata)} segments")
                return
            except Exception as e:
                logger.warning(f"Failed to load segment index: {e}, creating new one")
        
        # Create new index (inner product).
        # With L2-normalized embeddings this is cosine similarity.
        self._index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Created new segment FAISS index (IP/cosine) with dimension {self.dimension}")
    
    def add_segment(
        self,
        fragment_id: int,
        track_id: int,
        embedding: np.ndarray,
        start_time: float,
        end_time: float
    ) -> None:
        """
        Add a segment embedding to the index.
        
        Args:
            fragment_id: Unique identifier for the fragment
            track_id: ID of the parent track
            embedding: Vector of shape (dimension,)
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Requirements: 2.1, 2.2
        """
        with self._lock:
            # Validate embedding
            if embedding.shape != (self.dimension,):
                raise ValueError(f"Expected embedding of shape ({self.dimension},), got {embedding.shape}")
            
            # Remove existing if present
            if fragment_id in self._metadata:
                self._remove_segment_unsafe(fragment_id)
            
            # Add to FAISS index (ensure float32 + L2-normalized)
            embedding = embedding.astype(np.float32)
            norm = float(np.linalg.norm(embedding) + 1e-12)
            embedding = (embedding / norm).reshape(1, -1)
            self._index.add(embedding)
            
            # Store metadata
            faiss_idx = self._next_idx
            metadata = SegmentMetadata(
                fragment_id=fragment_id,
                track_id=track_id,
                start_time=start_time,
                end_time=end_time,
                faiss_idx=faiss_idx
            )
            self._metadata[fragment_id] = metadata
            self._faiss_to_fragment[faiss_idx] = fragment_id
            self._next_idx += 1
            
            logger.debug(f"Added segment {fragment_id} (track {track_id}, {start_time:.1f}s-{end_time:.1f}s)")
    
    def remove_segment(self, fragment_id: int) -> None:
        """
        Remove a segment from the index.
        
        Note: FAISS IndexFlat* doesn't support direct removal.
        We mark it as removed; rebuild_index() cleans up.
        
        Requirement: 2.3
        """
        with self._lock:
            self._remove_segment_unsafe(fragment_id)
    
    def _remove_segment_unsafe(self, fragment_id: int) -> None:
        """Remove segment without lock (internal use)."""
        if fragment_id in self._metadata:
            metadata = self._metadata.pop(fragment_id)
            self._faiss_to_fragment.pop(metadata.faiss_idx, None)
            logger.debug(f"Marked segment {fragment_id} as removed")
    
    def contains(self, fragment_id: int) -> bool:
        """Check if fragment is in index."""
        with self._lock:
            return fragment_id in self._metadata
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_track_id: Optional[int] = None,
        min_similarity: float = 0.0
    ) -> List[SegmentMatch]:
        """
        Search for k most similar segments.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Maximum number of results to return
            exclude_track_id: Track ID to exclude from results (source track)
            min_similarity: Minimum similarity threshold (0.0-1.0)
        
        Returns:
            List of SegmentMatch sorted by similarity (descending)
        
        Requirements: 2.4, 3.2, 3.4
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []
            
            # Validate embedding
            if query_embedding.shape != (self.dimension,):
                raise ValueError(f"Expected embedding of shape ({self.dimension},), got {query_embedding.shape}")
            
            # Search for more results to account for exclusions
            search_k = min(k * 3, self._index.ntotal)
            query = query_embedding.astype(np.float32)
            norm = float(np.linalg.norm(query) + 1e-12)
            query = (query / norm).reshape(1, -1)
            
            scores, indices = self._index.search(query, search_k)
            
            # Filter and convert results
            results = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:  # Invalid index
                    continue
                
                fragment_id = self._faiss_to_fragment.get(int(idx))
                if fragment_id is None:  # Removed
                    continue
                
                metadata = self._metadata.get(fragment_id)
                if metadata is None:
                    continue
                
                # Exclude source track
                if exclude_track_id is not None and metadata.track_id == exclude_track_id:
                    continue
                
                # Inner product for normalized vectors is cosine similarity in [-1, 1].
                # Map to [0, 1] to keep the rest of the API stable.
                similarity = (float(score) + 1.0) / 2.0
                similarity = max(0.0, min(1.0, similarity))
                
                # Apply minimum similarity threshold
                if similarity < min_similarity:
                    continue
                
                results.append(SegmentMatch(
                    fragment_id=metadata.fragment_id,
                    track_id=metadata.track_id,
                    start_time=metadata.start_time,
                    end_time=metadata.end_time,
                    similarity=similarity
                ))
                
                if len(results) >= k:
                    break
            
            return results
    
    def rebuild_index(self, db_conn) -> int:
        """
        Rebuild index from all fragments in database.
        
        Args:
            db_conn: PostgreSQL connection
        
        Returns:
            Number of segments indexed
        """
        with self._lock:
            # Clear existing index
            self._index = faiss.IndexFlatIP(self.dimension)
            self._metadata.clear()
            self._faiss_to_fragment.clear()
            self._next_idx = 0
        
        # Load all fragments with embeddings
        with db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, track_id, start_time, end_time, embedding_path
                FROM fragments
                WHERE embedding_path IS NOT NULL
            """)
            rows = cursor.fetchall()
        
        count = 0
        for row in rows:
            frag_id, track_id, start_time, end_time, emb_path = row
            
            if emb_path and Path(emb_path).exists():
                try:
                    embedding = np.load(emb_path)
                    self.add_segment(frag_id, track_id, embedding, start_time, end_time)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to load embedding for fragment {frag_id}: {e}")
        
        logger.info(f"Rebuilt segment index with {count} segments")
        return count
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save index and metadata to disk.
        
        Requirement: 2.5
        """
        with self._lock:
            save_path = Path(path) if path else self.index_path
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_file = save_path / 'segments.index'
            faiss.write_index(self._index, str(index_file))
            
            # Save metadata as JSON
            metadata_file = save_path / 'segments_metadata.json'
            metadata_dict = {
                'metadata': {str(k): asdict(v) for k, v in self._metadata.items()},
                'faiss_to_fragment': {str(k): v for k, v in self._faiss_to_fragment.items()},
                'next_idx': self._next_idx
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f)
            
            logger.info(f"Saved segment index to {save_path} ({len(self._metadata)} segments)")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load index and metadata from disk.
        
        Requirement: 2.5
        """
        with self._lock:
            load_path = Path(path) if path else self.index_path
            
            # Load FAISS index
            index_file = load_path / 'segments.index'
            self._index = faiss.read_index(str(index_file))
            
            # Load metadata from JSON
            metadata_file = load_path / 'segments_metadata.json'
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            self._metadata = {
                int(k): SegmentMetadata(**v)
                for k, v in data['metadata'].items()
            }
            self._faiss_to_fragment = {
                int(k): v
                for k, v in data['faiss_to_fragment'].items()
            }
            self._next_idx = data['next_idx']
            
            logger.info(f"Loaded segment index from {load_path}")
    
    @property
    def count(self) -> int:
        """Return number of active segments in index."""
        with self._lock:
            return len(self._metadata)
