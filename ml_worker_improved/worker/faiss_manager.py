"""
FAISSManager - Vector similarity search using FAISS.

Requirements: 3.8, 4.5, 10.1, 10.2
"""

import os
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import threading

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSManager:
    """
    Manages FAISS index for vector similarity search.
    
    Features:
    - Add/remove embeddings
    - Search for similar vectors
    - Persist index to disk
    - Thread-safe operations
    """
    
    def __init__(
        self, 
        dimension: int = 1280,  # Discogs-EffNet embedding size
        index_path: Optional[str] = None
    ):
        self.dimension = dimension
        self.index_path = Path(index_path or os.getenv('FAISS_INDEX_PATH', '/app/faiss_index'))
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._index: Optional[faiss.Index] = None
        self._id_map: Dict[int, int] = {}  # track_id -> faiss_id
        self._reverse_map: Dict[int, int] = {}  # faiss_id -> track_id
        self._next_id = 0
        
        self._initialize_index()
        
    def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        index_file = self.index_path / 'tracks.index'
        map_file = self.index_path / 'id_map.npy'
        
        if index_file.exists() and map_file.exists():
            try:
                self.load_index()
                logger.info(f"Loaded existing index with {self._index.ntotal} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, creating new one")
        
        # Create new index (inner product).
        # With L2-normalized embeddings this is cosine similarity.
        self._index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Created new FAISS index (IP/cosine) with dimension {self.dimension}")
        
    def add_embedding(self, track_id: int, embedding: np.ndarray) -> None:
        """
        Add an embedding to the index.
        
        Args:
            track_id: Unique identifier for the track
            embedding: Vector of shape (dimension,)
        """
        with self._lock:
            # Validate embedding
            if embedding.shape != (self.dimension,):
                raise ValueError(f"Expected embedding of shape ({self.dimension},), got {embedding.shape}")
            
            # Remove existing if present
            if track_id in self._id_map:
                self._remove_embedding_unsafe(track_id)
            
            # Add to index (ensure float32 + L2-normalized)
            embedding = embedding.astype(np.float32)
            norm = float(np.linalg.norm(embedding) + 1e-12)
            embedding = (embedding / norm).reshape(1, -1)
            self._index.add(embedding)
            
            # Update mappings
            faiss_id = self._next_id
            self._id_map[track_id] = faiss_id
            self._reverse_map[faiss_id] = track_id
            self._next_id += 1
            
            logger.debug(f"Added embedding for track {track_id} (faiss_id: {faiss_id})")
            
    def remove_embedding(self, track_id: int) -> None:
        """
        Remove an embedding from the index.
        
        Note: FAISS IndexFlat* doesn't support direct removal.
        We mark it as removed and rebuild periodically.
        """
        with self._lock:
            self._remove_embedding_unsafe(track_id)
            
    def _remove_embedding_unsafe(self, track_id: int) -> None:
        """Remove embedding without lock (internal use)."""
        if track_id in self._id_map:
            faiss_id = self._id_map.pop(track_id)
            self._reverse_map.pop(faiss_id, None)
            logger.debug(f"Marked embedding for track {track_id} as removed")
            
    def search_similar(
        self, 
        embedding: np.ndarray, 
        k: int = 10,
        exclude_ids: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for k most similar embeddings.
        
        Args:
            embedding: Query vector of shape (dimension,)
            k: Number of results to return
            exclude_ids: Track IDs to exclude from results
            
        Returns:
            List of (track_id, similarity) tuples, sorted by similarity descending
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []
            
            # Validate embedding
            if embedding.shape != (self.dimension,):
                raise ValueError(f"Expected embedding of shape ({self.dimension},), got {embedding.shape}")
            
            # Search for more results to account for exclusions and removed items
            search_k = min(k * 3, self._index.ntotal)
            embedding = embedding.astype(np.float32)
            norm = float(np.linalg.norm(embedding) + 1e-12)
            embedding = (embedding / norm).reshape(1, -1)

            scores, indices = self._index.search(embedding, search_k)
            
            # Filter and map results
            exclude_ids = set(exclude_ids or [])
            results = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:  # Invalid index
                    continue
                    
                track_id = self._reverse_map.get(idx)
                if track_id is None:  # Removed
                    continue
                    
                if track_id in exclude_ids:
                    continue
                    
                # Inner product for normalized vectors is cosine similarity.
                similarity = float(score)
                # Keep a sane range for downstream UI.
                if similarity < 0.0:
                    similarity = 0.0
                elif similarity > 1.0:
                    similarity = 1.0
                results.append((track_id, similarity))
                
                if len(results) >= k:
                    break
            
            # Keep most similar first
            return results
            
    def save_index(self, path: Optional[str] = None) -> None:
        """Save index and mappings to disk."""
        with self._lock:
            save_path = Path(path) if path else self.index_path
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_file = save_path / 'tracks.index'
            faiss.write_index(self._index, str(index_file))
            
            # Save ID mappings
            map_file = save_path / 'id_map.npy'
            np.save(str(map_file), {
                'id_map': self._id_map,
                'reverse_map': self._reverse_map,
                'next_id': self._next_id,
            })
            
            logger.info(f"Saved index to {save_path}")
            
    def load_index(self, path: Optional[str] = None) -> None:
        """Load index and mappings from disk."""
        with self._lock:
            load_path = Path(path) if path else self.index_path
            
            # Load FAISS index
            index_file = load_path / 'tracks.index'
            self._index = faiss.read_index(str(index_file))
            
            # Load ID mappings
            map_file = load_path / 'id_map.npy'
            data = np.load(str(map_file), allow_pickle=True).item()
            self._id_map = data['id_map']
            self._reverse_map = data['reverse_map']
            self._next_id = data['next_id']
            
            logger.info(f"Loaded index from {load_path}")
            
    @property
    def count(self) -> int:
        """Return number of active embeddings in index."""
        with self._lock:
            return len(self._id_map)
            
    def contains(self, track_id: int) -> bool:
        """Check if track is in index."""
        with self._lock:
            return track_id in self._id_map
