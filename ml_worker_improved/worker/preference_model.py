"""
PreferenceModel - Personal preference model using BPR algorithm.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS for hard negative mining
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, hard negative mining disabled")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, preference model training is unavailable")
else:
    from .device_policy import resolve_torch_device


class _PreferenceMLP(nn.Module):
    """Small MLP scorer over audio embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class PreferenceModel:
    """
    Lightweight preference model using an MLP over embeddings.

    Uses Bayesian Personalized Ranking (BPR) for training.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        hidden_dim: int = 256,
        learning_rate: float = 0.001,
        device: str = 'auto',
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PreferenceModel")

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)

        self.device = self._resolve_device(device)
        self.model = _PreferenceMLP(self.embedding_dim, self.hidden_dim, self.dropout).to(self.device)

        logger.info(
            "PreferenceModel initialized: device=%s, embedding_dim=%d, hidden_dim=%d",
            self.device,
            self.embedding_dim,
            self.hidden_dim,
        )

    @staticmethod
    def _resolve_device(device: str):
        if device == 'auto':
            return resolve_torch_device(task_name='PreferenceModel', allow_cpu=False)
        return resolve_torch_device(task_name='PreferenceModel', requested=device, allow_cpu=False)

    @staticmethod
    def _ensure_2d_float32(embeddings: np.ndarray) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
        arr = PreferenceModel._ensure_2d_float32(embeddings)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (arr / norms).astype(np.float32)

    @staticmethod
    def calibrate_scores(
        raw_scores: np.ndarray,
        positive_reference_scores: Optional[np.ndarray] = None,
        negative_reference_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert raw ranking scores into a stable probability-like score.

        Uses positive/negative score references when available instead of
        rank-based normalization, so real score gaps survive calibration.
        """
        raw = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        if raw.size == 0:
            return raw

        raw_std = float(raw.std())
        if raw_std < 1e-10:
            return np.full(raw.shape[0], 0.5, dtype=np.float32)

        pos = None
        if positive_reference_scores is not None:
            pos = np.asarray(positive_reference_scores, dtype=np.float32).reshape(-1)
            if pos.size == 0:
                pos = None

        neg = None
        if negative_reference_scores is not None:
            neg = np.asarray(negative_reference_scores, dtype=np.float32).reshape(-1)
            if neg.size == 0:
                neg = None

        if pos is not None:
            pos_median = float(np.median(pos))
        else:
            pos_median = float(np.percentile(raw, 75))

        if neg is not None:
            neg_median = float(np.median(neg))
        else:
            neg_median = float(np.percentile(raw, 50))

        if pos_median > neg_median:
            center = (pos_median + neg_median) / 2.0
            pos_std = float(pos.std()) if pos is not None and pos.size > 1 else raw_std
            neg_std = float(neg.std()) if neg is not None and neg.size > 1 else raw_std
            separation = max(pos_median - neg_median, 1e-6)
            scale = max(separation / 4.0, (pos_std + neg_std) / 2.0, raw_std * 0.35, 1e-3)
        else:
            q25, q75 = np.percentile(raw, [25, 75])
            iqr = float(q75 - q25)
            robust_scale = iqr / 1.349 if iqr > 1e-6 else raw_std
            center = float(np.median(raw))
            scale = max(robust_scale, raw_std * 0.5, 1e-3)

        logits = (raw - center) / scale
        normalized = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(normalized, 0.001, 0.999).astype(np.float32)

    def _tensor_from_numpy(self, embeddings: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(embeddings).to(self.device)

    def _topk_anchor_affinity(
        self,
        candidate_embeddings: np.ndarray,
        anchor_embeddings: Optional[np.ndarray],
        top_k: int,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        candidates = self._ensure_2d_float32(candidate_embeddings)
        if anchor_embeddings is None or len(anchor_embeddings) == 0:
            return np.zeros(candidates.shape[0], dtype=np.float32)

        anchors = self._ensure_2d_float32(anchor_embeddings)
        k = max(1, min(int(top_k), anchors.shape[0]))

        if self.device.type != 'cuda':
            sims = candidates @ anchors.T
            if k >= sims.shape[1]:
                return sims.mean(axis=1).astype(np.float32)

            top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            top_vals = np.take_along_axis(sims, top_idx, axis=1)
            return top_vals.mean(axis=1).astype(np.float32)

        anchor_tensor = self._tensor_from_numpy(anchors)
        scores = np.empty(candidates.shape[0], dtype=np.float32)

        with torch.no_grad():
            for start in range(0, candidates.shape[0], max(1, int(chunk_size))):
                stop = min(start + max(1, int(chunk_size)), candidates.shape[0])
                chunk_tensor = self._tensor_from_numpy(candidates[start:stop])
                sims = torch.matmul(chunk_tensor, anchor_tensor.T)
                if k >= sims.shape[1]:
                    vals = sims.mean(dim=1)
                else:
                    vals = torch.topk(sims, k=k, dim=1, largest=True, sorted=False).values.mean(dim=1)
                scores[start:stop] = vals.detach().cpu().numpy().astype(np.float32)

        return scores

    def score_anchor_margin(
        self,
        candidate_embeddings: np.ndarray,
        positive_anchor_embeddings: Optional[np.ndarray],
        negative_anchor_embeddings: Optional[np.ndarray] = None,
        positive_top_k: int = 12,
        negative_top_k: int = 12,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        pos_affinity = self._topk_anchor_affinity(
            candidate_embeddings,
            positive_anchor_embeddings,
            top_k=positive_top_k,
            chunk_size=chunk_size,
        )
        neg_affinity = self._topk_anchor_affinity(
            candidate_embeddings,
            negative_anchor_embeddings,
            top_k=negative_top_k,
            chunk_size=chunk_size,
        )
        return (pos_affinity - neg_affinity).astype(np.float32)

    def _predict_tensor(self, embeddings: np.ndarray) -> np.ndarray:
        arr = self._l2_normalize_rows(embeddings)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(self._tensor_from_numpy(arr))
        return scores.detach().cpu().numpy().astype(np.float32).reshape(-1)

    @property
    def W1(self) -> np.ndarray:
        return self.model.fc1.weight.detach().cpu().numpy().T.copy()

    @W1.setter
    def W1(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.embedding_dim, self.hidden_dim):
            raise ValueError(f"W1 shape mismatch: expected {(self.embedding_dim, self.hidden_dim)}, got {arr.shape}")
        with torch.no_grad():
            self.model.fc1.weight.copy_(torch.from_numpy(arr.T).to(self.device))

    @property
    def b1(self) -> np.ndarray:
        return self.model.fc1.bias.detach().cpu().numpy().copy()

    @b1.setter
    def b1(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.hidden_dim,):
            raise ValueError(f"b1 shape mismatch: expected {(self.hidden_dim,)}, got {arr.shape}")
        with torch.no_grad():
            self.model.fc1.bias.copy_(torch.from_numpy(arr).to(self.device))

    @property
    def W2(self) -> np.ndarray:
        return self.model.fc2.weight.detach().cpu().numpy().T.copy()

    @W2.setter
    def W2(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.hidden_dim, 1):
            raise ValueError(f"W2 shape mismatch: expected {(self.hidden_dim, 1)}, got {arr.shape}")
        with torch.no_grad():
            self.model.fc2.weight.copy_(torch.from_numpy(arr.T).to(self.device))

    @property
    def b2(self) -> np.ndarray:
        return self.model.fc2.bias.detach().cpu().numpy().copy()

    @b2.setter
    def b2(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (1,):
            raise ValueError(f"b2 shape mismatch: expected {(1,)}, got {arr.shape}")
        with torch.no_grad():
            self.model.fc2.bias.copy_(torch.from_numpy(arr).to(self.device))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores for embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Array of shape (n_samples,) with relevance scores
        """
        return self._predict_tensor(embeddings)

    def train(
        self,
        liked_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        sample_weights: np.ndarray = None,
        use_hard_negatives: bool = True,
        explicit_negative_embeddings: np.ndarray = None,
        explicit_negative_weights: np.ndarray = None,
        explicit_negative_prob: float = 0.5,
    ) -> dict:
        """
        Train the model using BPR algorithm with optional sample weights.
        """
        liked_embeddings = self._l2_normalize_rows(liked_embeddings)
        all_embeddings = self._l2_normalize_rows(all_embeddings)

        n_liked = len(liked_embeddings)
        n_all = len(all_embeddings)

        if n_liked == 0:
            raise ValueError("No liked embeddings provided")

        if sample_weights is None:
            sample_weights = np.ones(n_liked, dtype=np.float32)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
            if len(sample_weights) != n_liked:
                raise ValueError("sample_weights must match liked_embeddings length")
        sample_weights = np.clip(sample_weights, 1e-6, None)

        explicit_negative_prob = float(np.clip(explicit_negative_prob, 0.0, 1.0))
        explicit_neg = None
        explicit_neg_w = None
        if explicit_negative_embeddings is not None and len(explicit_negative_embeddings) > 0:
            explicit_neg = self._l2_normalize_rows(explicit_negative_embeddings)
            if explicit_neg.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Explicit negative embeddings dim mismatch: expected {self.embedding_dim}, got {explicit_neg.shape}"
                )
            if explicit_negative_weights is None:
                explicit_neg_w = np.ones(len(explicit_neg), dtype=np.float32)
            else:
                explicit_neg_w = np.asarray(explicit_negative_weights, dtype=np.float32).reshape(-1)
                if len(explicit_neg_w) != len(explicit_neg):
                    raise ValueError("explicit_negative_weights must match explicit_negative_embeddings length")
            explicit_neg_w = np.clip(explicit_neg_w, 1e-6, None)
        else:
            explicit_negative_prob = 0.0

        logger.info(
            "Training PreferenceModel with device=%s, liked=%d, total=%d, explicit_negatives=%d",
            self.device,
            n_liked,
            n_all,
            0 if explicit_neg is None else len(explicit_neg),
        )

        faiss_index = None
        torch_negative_pool = None
        liked_candidate_mask = None
        negative_fallback_indices = None
        if use_hard_negatives and self.device.type == 'cuda' and n_all > 100:
            torch_negative_pool = self._tensor_from_numpy(all_embeddings.astype(np.float32))
            liked_hashes = {hash(emb.tobytes()) for emb in liked_embeddings}
            liked_candidate_mask = np.fromiter(
                (hash(emb.tobytes()) in liked_hashes for emb in all_embeddings),
                dtype=bool,
                count=n_all,
            )
            negative_fallback_indices = np.flatnonzero(~liked_candidate_mask)
            logger.info("Using torch/CUDA hard negative mining (%d vectors)", n_all)
        elif use_hard_negatives and FAISS_AVAILABLE and n_all > 100:
            try:
                faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                faiss_index.add(all_embeddings.astype(np.float32))
                logger.info("Built FAISS index for hard negative mining (%d vectors)", n_all)
            except Exception as e:
                logger.warning("Failed to build FAISS index: %s", e)
                faiss_index = None

        history = {'loss': [], 'device': str(self.device)}
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_neg_sim = 0.999
        batches_per_epoch = max(1, int(np.ceil(n_liked / max(1, batch_size))))
        weight_probs = sample_weights / sample_weights.sum()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for _ in range(batches_per_epoch):
                cur_batch = min(batch_size, n_liked)
                pos_idx = np.random.choice(n_liked, size=cur_batch, replace=True, p=weight_probs)
                pos_embeddings = liked_embeddings[pos_idx]
                pos_weights = sample_weights[pos_idx]
                neg_embeddings = np.empty_like(pos_embeddings)

                use_explicit = None
                if explicit_neg is not None and explicit_negative_prob > 0.0:
                    use_explicit = np.random.rand(cur_batch) < explicit_negative_prob
                    if np.any(use_explicit):
                        probs = explicit_neg_w / explicit_neg_w.sum()
                        neg_idx = np.random.choice(len(explicit_neg), size=int(use_explicit.sum()), replace=True, p=probs)
                        neg_embeddings[use_explicit] = explicit_neg[neg_idx]

                remaining_mask = np.ones(cur_batch, dtype=bool) if use_explicit is None else ~use_explicit
                if np.any(remaining_mask):
                    rem_idx = np.flatnonzero(remaining_mask)
                    pos_rem = pos_embeddings[rem_idx]
                    if torch_negative_pool is not None and liked_candidate_mask is not None:
                        neg_embeddings[rem_idx] = self._sample_hard_negatives_torch(
                            pos_rem,
                            all_embeddings,
                            torch_negative_pool,
                            liked_candidate_mask,
                            negative_fallback_indices,
                        )
                    elif faiss_index is not None:
                        neg_embeddings[rem_idx] = self._sample_hard_negatives(
                            pos_rem,
                            all_embeddings,
                            faiss_index,
                            liked_embeddings,
                        )
                    else:
                        for out_i, pos in enumerate(pos_rem):
                            for _attempt in range(10):
                                cand = all_embeddings[np.random.randint(0, n_all)]
                                if float(np.dot(pos, cand)) < max_neg_sim:
                                    neg_embeddings[rem_idx[out_i]] = cand
                                    break
                            else:
                                neg_embeddings[rem_idx[out_i]] = all_embeddings[np.random.randint(0, n_all)]

                loss = self._bpr_step(pos_embeddings, neg_embeddings, pos_weights, optimizer)
                epoch_loss += loss

            avg_loss = epoch_loss / batches_per_epoch
            history['loss'].append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(
                    "PreferenceModel epoch %d/%d, loss=%.4f, device=%s",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    self.device,
                )

        self.model.eval()
        return history

    def _sample_hard_negatives_torch(
        self,
        pos_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_embeddings_tensor: torch.Tensor,
        liked_candidate_mask: np.ndarray,
        fallback_indices: Optional[np.ndarray],
        k_neighbors: int = 50,
    ) -> np.ndarray:
        """
        Sample hard negatives using top-k cosine neighbors on the configured torch device.
        """
        batch_size = len(pos_embeddings)
        neg_embeddings = np.empty_like(pos_embeddings)
        if batch_size == 0:
            return neg_embeddings

        pos_normalized = self._l2_normalize_rows(pos_embeddings).astype(np.float32)
        top_k = min(max(int(k_neighbors) * 4, 1), len(all_embeddings))

        with torch.no_grad():
            sims = torch.matmul(self._tensor_from_numpy(pos_normalized), all_embeddings_tensor.T)
            values, indices = torch.topk(sims, k=top_k, dim=1, largest=True, sorted=True)

        distances = values.detach().cpu().numpy().astype(np.float32)
        neighbor_indices = indices.detach().cpu().numpy()

        for i in range(batch_size):
            candidates = []
            candidate_sims = []

            for sim, idx in zip(distances[i], neighbor_indices[i]):
                idx = int(idx)
                if idx < 0 or idx >= len(all_embeddings):
                    continue
                if liked_candidate_mask[idx]:
                    continue
                if float(sim) > 0.999:
                    continue

                candidates.append(all_embeddings[idx])
                candidate_sims.append(max(0.1, float(sim)))

                if len(candidates) >= k_neighbors:
                    break

            if candidates:
                probs = np.asarray(candidate_sims, dtype=np.float32)
                probs = probs / probs.sum()
                chosen_idx = np.random.choice(len(candidates), p=probs)
                neg_embeddings[i] = candidates[chosen_idx]
            elif fallback_indices is not None and len(fallback_indices) > 0:
                neg_embeddings[i] = all_embeddings[np.random.choice(fallback_indices)]
            else:
                neg_embeddings[i] = all_embeddings[np.random.randint(0, len(all_embeddings))]

        return neg_embeddings

    def _sample_hard_negatives(
        self,
        pos_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        faiss_index,
        liked_embeddings: np.ndarray,
        k_neighbors: int = 50,
    ) -> np.ndarray:
        """
        Sample hard negatives using FAISS nearest neighbor search.
        """
        batch_size = len(pos_embeddings)
        neg_embeddings = np.empty_like(pos_embeddings)

        pos_normalized = self._l2_normalize_rows(pos_embeddings).astype(np.float32)
        distances, indices = faiss_index.search(pos_normalized, k_neighbors)

        liked_hashes = {hash(emb.tobytes()) for emb in liked_embeddings}

        for i in range(batch_size):
            candidates = []
            candidate_sims = []

            for j, idx in enumerate(indices[i]):
                if idx < 0 or idx >= len(all_embeddings):
                    continue

                cand_emb = all_embeddings[idx]
                cand_hash = hash(cand_emb.tobytes())

                if cand_hash in liked_hashes:
                    continue

                if distances[i][j] > 0.999:
                    continue

                candidates.append(cand_emb)
                candidate_sims.append(max(0.1, float(distances[i][j])))

            if candidates:
                sims = np.asarray(candidate_sims, dtype=np.float32)
                probs = sims / sims.sum()
                chosen_idx = np.random.choice(len(candidates), p=probs)
                neg_embeddings[i] = candidates[chosen_idx]
            else:
                neg_embeddings[i] = all_embeddings[np.random.randint(0, len(all_embeddings))]

        return neg_embeddings

    def _bpr_step(
        self,
        pos_embeddings: np.ndarray,
        neg_embeddings: np.ndarray,
        sample_weights: np.ndarray,
        optimizer,
    ) -> float:
        """Perform one weighted BPR training step on the configured device."""
        pos_tensor = self._tensor_from_numpy(self._ensure_2d_float32(pos_embeddings))
        neg_tensor = self._tensor_from_numpy(self._ensure_2d_float32(neg_embeddings))
        weight_tensor = self._tensor_from_numpy(sample_weights.astype(np.float32).reshape(-1)).reshape(-1)

        optimizer.zero_grad(set_to_none=True)

        pos_scores = self.model(pos_tensor)
        neg_scores = self.model(neg_tensor)
        diff = pos_scores - neg_scores

        normalized_weights = weight_tensor / torch.clamp(weight_tensor.mean(), min=1e-6)
        loss = (normalized_weights * F.softplus(-diff)).mean()
        score_reg = 1e-4 * (pos_scores.pow(2).mean() + neg_scores.pow(2).mean())
        total_loss = loss + score_reg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        optimizer.step()

        return float(total_loss.detach().cpu().item())

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
            dropout=self.dropout,
            weight_decay=self.weight_decay,
        )
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        data = np.load(path)

        embedding_dim = int(data['embedding_dim'])
        hidden_dim = int(data['hidden_dim'])
        dropout = float(data['dropout']) if 'dropout' in data else self.dropout
        weight_decay = float(data['weight_decay']) if 'weight_decay' in data else self.weight_decay

        if (
            embedding_dim != self.embedding_dim
            or hidden_dim != self.hidden_dim
            or abs(dropout - self.dropout) > 1e-9
        ):
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.weight_decay = weight_decay
            self.model = _PreferenceMLP(self.embedding_dim, self.hidden_dim, self.dropout).to(self.device)

        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.model.eval()

        logger.info("Model loaded from %s", path)
