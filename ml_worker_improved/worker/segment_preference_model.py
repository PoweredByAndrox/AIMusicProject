#!/usr/bin/env python3
"""
SegmentPreferenceModel - ML model for learning user preferences from labeled segments.

Uses contrastive learning to distinguish liked vs disliked audio segments.
Trained on mel spectrograms and chroma features extracted from labeled segments.

Architecture:
- Input: Mel spectrogram (128 x T) or Chroma (12 x T)
- CNN encoder: extracts fixed-size embedding
- Contrastive loss: pulls liked segments together, pushes disliked apart
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch for GPU training
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using numpy fallback")
else:
    from .device_policy import resolve_torch_device


class SegmentEncoder(nn.Module):
    """CNN encoder for audio segments."""
    
    def __init__(self, input_channels: int = 128, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # CNN layers for mel spectrogram (128 x T)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Global average pooling + projection
        self.fc = nn.Linear(256, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = x.mean(dim=2)
        
        # Projection
        x = self.fc(x)
        
        # L2 normalize for contrastive learning
        x = F.normalize(x, p=2, dim=1)
        
        return x


class SegmentPreferenceModel:
    """
    Segment preference model using contrastive learning.
    
    Learns to distinguish liked vs disliked audio segments.
    """
    
    def __init__(
        self,
        input_channels: int = 128,  # Mel spectrogram bins
        embedding_dim: int = 256,
        learning_rate: float = 0.001,
        temperature: float = 0.07,  # Contrastive loss temperature
        device: str = 'auto'
    ):
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Set device
        if device == 'auto':
            self.device = resolve_torch_device(task_name='SegmentPreferenceModel', allow_cpu=False)
        else:
            self.device = resolve_torch_device(task_name='SegmentPreferenceModel', requested=device, allow_cpu=False)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.encoder = SegmentEncoder(input_channels, embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        # Training state
        self.trained = False
        self.train_history = []
        
    def _prepare_input(self, features: np.ndarray) -> torch.Tensor:
        """Prepare input tensor from numpy array."""
        # Ensure correct shape (channels, time)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Pad or truncate to fixed length
        target_length = 256  # ~6 seconds at 22050/512 hop
        
        if features.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        elif features.shape[1] > target_length:
            # Random crop during training, center crop during inference
            if self.encoder.training:
                start = np.random.randint(0, features.shape[1] - target_length)
            else:
                start = (features.shape[1] - target_length) // 2
            features = features[:, start:start + target_length]
        
        return torch.from_numpy(features).float()
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode audio features to embedding.
        
        Args:
            features: Mel spectrogram or chroma of shape (channels, time)
            
        Returns:
            Embedding of shape (embedding_dim,)
        """
        self.encoder.eval()
        
        with torch.no_grad():
            x = self._prepare_input(features).unsqueeze(0).to(self.device)
            embedding = self.encoder(x)
            
        return embedding.cpu().numpy().flatten()
    
    def encode_batch(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        Encode batch of audio features.
        
        Args:
            features_list: List of mel spectrograms
            
        Returns:
            Embeddings of shape (n_samples, embedding_dim)
        """
        self.encoder.eval()
        
        with torch.no_grad():
            batch = torch.stack([self._prepare_input(f) for f in features_list])
            batch = batch.to(self.device)
            embeddings = self.encoder(batch)
            
        return embeddings.cpu().numpy()
    
    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised contrastive loss.
        
        Pulls together samples with same label, pushes apart different labels.
        
        Args:
            embeddings: (batch, embedding_dim) normalized embeddings
            labels: (batch,) binary labels (1=liked, 0=disliked)
            
        Returns:
            Scalar loss
        """
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask_pos = mask_pos - torch.eye(batch_size, device=self.device)
        
        # For numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        
        # Compute log softmax
        exp_sim = torch.exp(sim_matrix)
        
        # Mask out self
        mask_self = 1 - torch.eye(batch_size, device=self.device)
        exp_sim = exp_sim * mask_self
        
        # Log probability of positive pairs
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-10)
        
        # Mean log probability over positive pairs
        n_pos = mask_pos.sum(dim=1)
        n_pos = torch.clamp(n_pos, min=1)  # Avoid division by zero
        
        loss = -(mask_pos * log_prob).sum(dim=1) / n_pos
        
        return loss.mean()
    
    def train(
        self,
        liked_features: List[np.ndarray],
        disliked_features: List[np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        progress_callback=None
    ) -> Dict:
        """
        Train the model on labeled segments.
        
        Args:
            liked_features: List of mel spectrograms for liked segments
            disliked_features: List of mel spectrograms for disliked segments
            epochs: Number of training epochs
            batch_size: Batch size
            progress_callback: Optional callback(epoch, loss) for progress
            
        Returns:
            Training history dict
        """
        n_liked = len(liked_features)
        n_disliked = len(disliked_features)
        
        if n_liked < 2:
            raise ValueError(f"Need at least 2 liked segments, got {n_liked}")
        
        logger.info(f"Training with {n_liked} liked, {n_disliked} disliked segments")
        
        # Prepare data
        all_features = liked_features + disliked_features
        all_labels = [1] * n_liked + [0] * n_disliked
        
        # Convert to tensors
        all_tensors = [self._prepare_input(f) for f in all_features]
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
        
        history = {'loss': [], 'accuracy': []}
        
        self.encoder.train()
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(all_tensors))
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]
                
                if len(batch_idx) < 4:  # Need at least 4 samples for contrastive
                    continue
                
                # Get batch
                batch_features = torch.stack([all_tensors[j] for j in batch_idx])
                batch_labels = labels_tensor[batch_idx]
                
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward
                self.optimizer.zero_grad()
                embeddings = self.encoder(batch_features)
                
                # Loss
                loss = self._contrastive_loss(embeddings, batch_labels)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(1, n_batches)
            history['loss'].append(avg_loss)
            
            # Compute accuracy (liked vs disliked separation)
            self.encoder.eval()
            with torch.no_grad():
                all_batch = torch.stack(all_tensors).to(self.device)
                all_emb = self.encoder(all_batch)
                
                # Compute centroid of liked
                liked_emb = all_emb[:n_liked]
                centroid = liked_emb.mean(dim=0, keepdim=True)
                
                # Distance to centroid
                distances = torch.norm(all_emb - centroid, dim=1)
                
                # Accuracy: liked should be closer
                threshold = distances[:n_liked].max()
                correct = (distances[:n_liked] <= threshold).sum() + \
                          (distances[n_liked:] > threshold).sum()
                accuracy = correct.item() / len(all_tensors)
                
            history['accuracy'].append(accuracy)
            self.encoder.train()
            
            if progress_callback:
                progress_callback(epoch + 1, avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")
        
        self.trained = True
        self.train_history = history
        
        return history
    
    def predict_preference(self, features: np.ndarray) -> float:
        """
        Predict preference score for a segment.
        
        Args:
            features: Mel spectrogram of shape (channels, time)
            
        Returns:
            Preference score (higher = more likely to be liked)
        """
        if not self.trained:
            logger.warning("Model not trained, returning 0.5")
            return 0.5
        
        embedding = self.encode(features)
        
        # Use distance to liked centroid as score
        # (requires storing centroid during training)
        if hasattr(self, 'liked_centroid'):
            distance = np.linalg.norm(embedding - self.liked_centroid)
            # Convert distance to score (closer = higher)
            score = 1.0 / (1.0 + distance)
        else:
            score = 0.5
        
        return float(score)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'encoder_state': self.encoder.state_dict(),
            'input_channels': self.input_channels,
            'embedding_dim': self.embedding_dim,
            'temperature': self.temperature,
            'trained': self.trained,
            'train_history': self.train_history,
        }
        
        if hasattr(self, 'liked_centroid'):
            state['liked_centroid'] = self.liked_centroid
        
        torch.save(state, str(save_path))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        # Recreate encoder if dimensions changed
        if state['input_channels'] != self.input_channels or \
           state['embedding_dim'] != self.embedding_dim:
            self.input_channels = state['input_channels']
            self.embedding_dim = state['embedding_dim']
            self.encoder = SegmentEncoder(
                self.input_channels, 
                self.embedding_dim
            ).to(self.device)
        
        self.encoder.load_state_dict(state['encoder_state'])
        self.temperature = state['temperature']
        self.trained = state['trained']
        self.train_history = state.get('train_history', [])
        
        if 'liked_centroid' in state:
            self.liked_centroid = state['liked_centroid']
        
        logger.info(f"Model loaded from {path}")
    
    def compute_liked_centroid(self, liked_features: List[np.ndarray]) -> None:
        """Compute and store centroid of liked segments."""
        if len(liked_features) == 0:
            return
        
        embeddings = self.encode_batch(liked_features)
        self.liked_centroid = embeddings.mean(axis=0)
        logger.info(f"Computed liked centroid from {len(liked_features)} segments")


# =============================================================================
# Numpy fallback for environments without PyTorch
# =============================================================================

class SegmentPreferenceModelNumpy:
    """
    Lightweight numpy-based segment preference model.
    Uses simple feature aggregation + cosine similarity.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.liked_centroid = None
        self.trained = False
    
    def _extract_features(self, mel: np.ndarray) -> np.ndarray:
        """Extract simple statistical features from mel spectrogram."""
        features = []
        
        # Mean per frequency band
        features.extend(mel.mean(axis=1)[:self.embedding_dim // 4])
        
        # Std per frequency band
        features.extend(mel.std(axis=1)[:self.embedding_dim // 4])
        
        # Temporal statistics
        temporal_mean = mel.mean(axis=0)
        features.extend([
            temporal_mean.mean(),
            temporal_mean.std(),
            temporal_mean.max(),
            temporal_mean.min(),
        ])
        
        # Pad to embedding_dim
        features = np.array(features[:self.embedding_dim])
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 1e-10:
            features = features / norm
        
        return features
    
    def encode(self, mel: np.ndarray) -> np.ndarray:
        return self._extract_features(mel)
    
    def train(
        self,
        liked_features: List[np.ndarray],
        disliked_features: List[np.ndarray],
        **kwargs
    ) -> Dict:
        """Train by computing liked centroid."""
        if len(liked_features) == 0:
            raise ValueError("No liked features")
        
        liked_emb = np.array([self._extract_features(f) for f in liked_features])
        self.liked_centroid = liked_emb.mean(axis=0)
        self.trained = True
        
        return {'loss': [0.0]}
    
    def predict_preference(self, features: np.ndarray) -> float:
        if self.liked_centroid is None:
            return 0.5
        
        emb = self._extract_features(features)
        similarity = np.dot(emb, self.liked_centroid)
        return float((similarity + 1) / 2)  # Map [-1, 1] to [0, 1]
    
    def save(self, path: str) -> None:
        np.savez(path, liked_centroid=self.liked_centroid, trained=self.trained)
    
    def load(self, path: str) -> None:
        data = np.load(path)
        self.liked_centroid = data['liked_centroid']
        self.trained = bool(data['trained'])


# Factory function
def create_segment_model(use_gpu: bool = True, **kwargs):
    """Create segment preference model with appropriate backend."""
    if use_gpu and TORCH_AVAILABLE:
        return SegmentPreferenceModel(**kwargs)
    else:
        return SegmentPreferenceModelNumpy(**kwargs)
