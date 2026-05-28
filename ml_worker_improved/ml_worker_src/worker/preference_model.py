"""
PreferenceModel - Personal preference model using BPR algorithm.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PreferenceModel:
    """
    Lightweight preference model using MLP over embeddings.
    
    Uses Bayesian Personalized Ranking (BPR) for training.
    
    Architecture:
    - Input: embedding vector (1280 dim from Discogs-EffNet)
    - Hidden: 256 units with ReLU
    - Output: 1 unit (relevance score)
    """
    
    def __init__(
        self,
        embedding_dim: int = 1280,  # Discogs-EffNet embedding size
        hidden_dim: int = 256,      # Larger hidden layer for richer embeddings
        learning_rate: float = 0.001
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize model weights with Xavier initialization."""
        # Hidden layer
        self.W1 = np.random.randn(self.embedding_dim, self.hidden_dim) * np.sqrt(2.0 / self.embedding_dim)
        self.b1 = np.zeros(self.hidden_dim)
        
        # Output layer
        self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(1)
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through the network."""
        # Hidden layer
        h = self._relu(np.dot(x, self.W1) + self.b1)
        # Output layer
        out = np.dot(h, self.W2) + self.b2
        return out, h
        
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores for embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            Array of shape (n_samples,) with relevance scores
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        scores, _ = self._forward(embeddings)
        return scores.flatten()
        
    def train(
        self,
        liked_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        sample_weights: np.ndarray = None
    ) -> dict:
        """
        Train the model using BPR algorithm with optional sample weights.
        
        Args:
            liked_embeddings: Embeddings of liked items (n_liked, embedding_dim)
            all_embeddings: All available embeddings (n_all, embedding_dim)
            epochs: Number of training epochs
            batch_size: Batch size for training
            sample_weights: Optional weights for liked samples (n_liked,)
            
        Returns:
            Training history dict
        """
        n_liked = len(liked_embeddings)
        n_all = len(all_embeddings)
        
        if n_liked == 0:
            raise ValueError("No liked embeddings provided")
        
        # Default weights if not provided
        if sample_weights is None:
            sample_weights = np.ones(n_liked)
        else:
            sample_weights = np.array(sample_weights)
            
        logger.info(f"Training with {n_liked} liked items, {n_all} total items")
        
        history = {'loss': []}
        
        # NOTE: This is intentionally lightweight (numpy-only). If you outgrow it,
        # move training to PyTorch and use a proper optimizer + batching.

        # Treat "same sample picked as negative" as a bug: it collapses gradients.
        # We'll use a small rejection-sampling loop to avoid near-identical negatives.
        max_neg_sim = 0.999  # assuming L2-normalized embeddings

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Sample positive and negative pairs
            for _ in range(max(1, n_liked // batch_size)):
                # Sample positive items (liked) with weights as probabilities
                weights_normalized = sample_weights / sample_weights.sum()
                pos_idx = np.random.choice(n_liked, size=min(batch_size, n_liked), replace=True, p=weights_normalized)
                pos_embeddings = liked_embeddings[pos_idx]
                pos_weights = sample_weights[pos_idx]
                
                # Sample negative items (random from all, but avoid selecting near-identical vectors)
                neg_embeddings = np.empty_like(pos_embeddings)
                for i, pos in enumerate(pos_embeddings):
                    for _attempt in range(10):
                        cand = all_embeddings[np.random.randint(0, n_all)]
                        # dot(pos,cand) is cosine similarity if embeddings are normalized
                        if float(np.dot(pos, cand)) < max_neg_sim:
                            neg_embeddings[i] = cand
                            break
                    else:
                        # Fallback (should be rare)
                        neg_embeddings[i] = all_embeddings[np.random.randint(0, n_all)]
                
                # BPR loss and gradient with weights
                loss = self._bpr_step(pos_embeddings, neg_embeddings, pos_weights)
                epoch_loss += loss
                n_batches += 1
                
            avg_loss = epoch_loss / max(1, n_batches)
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                
        return history
        
    def _bpr_step(
        self,
        pos_embeddings: np.ndarray,
        neg_embeddings: np.ndarray,
        sample_weights: np.ndarray = None
    ) -> float:
        """
        Perform one BPR training step with optional sample weights.
        
        BPR Loss: -log(sigmoid(score_pos - score_neg))
        """
        # Forward pass
        pos_scores, pos_h = self._forward(pos_embeddings)
        neg_scores, neg_h = self._forward(neg_embeddings)
        
        # BPR loss with weights
        diff = pos_scores - neg_scores
        
        if sample_weights is not None:
            weights = sample_weights.reshape(-1, 1)
            loss = -np.mean(weights * np.log(self._sigmoid(diff) + 1e-10))
        else:
            loss = -np.mean(np.log(self._sigmoid(diff) + 1e-10))
        
        # Backward pass (gradient descent)
        # Gradient of BPR loss
        grad = -self._sigmoid(-diff)  # Shape: (batch, 1)
        
        if sample_weights is not None:
            grad = grad * weights
        
        # Output layer gradients
        dW2_pos = np.dot(pos_h.T, grad) / len(grad)
        dW2_neg = np.dot(neg_h.T, -grad) / len(grad)
        # Important: b2 cancels out in (score_pos - score_neg), so its gradient is ~0.
        # Updating b2 here injects noise and can dominate training.
        db2 = np.zeros_like(self.b2)
        
        # Hidden layer gradients (simplified)
        dh_pos = np.dot(grad, self.W2.T) * (pos_h > 0)
        dh_neg = np.dot(-grad, self.W2.T) * (neg_h > 0)
        
        dW1_pos = np.dot(pos_embeddings.T, dh_pos) / len(grad)
        dW1_neg = np.dot(neg_embeddings.T, dh_neg) / len(grad)
        db1 = np.mean(dh_pos + dh_neg, axis=0)
        
        # Update weights
        self.W2 -= self.learning_rate * (dW2_pos + dW2_neg)
        # b2 update is intentionally a no-op (see comment above)
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * (dW1_pos + dW1_neg)
        self.b1 -= self.learning_rate * db1
        
        return float(loss)
        
    def save(self, path: str) -> None:
        """Save model weights to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            str(save_path),
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model weights from disk."""
        data = np.load(path)
        
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.embedding_dim = int(data['embedding_dim'])
        self.hidden_dim = int(data['hidden_dim'])
        
        logger.info(f"Model loaded from {path}")
