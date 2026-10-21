"""
Local BAAI/bge-m3 Embedding Generator (1024 dimensions).

Generates dense vector representations with L2 normalization for legal retrieval.
Includes resilient fallback for offline execution environments.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import List, Optional

from scrag.config import Settings, settings as global_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1024


class LocalEmbedder:
    """
    BAAI/bge-m3 dense vector embedding generator.
    Produces 1024-dimensional float embeddings normalized to unit length.
    """

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.cfg = config or global_settings
        self.model_name = self.cfg.embedding_model_name
        self.device = self.cfg.embedding_device
        self.batch_size = self.cfg.embedding_batch_size
        self._model = None
        self._is_fallback = False

    def _load_model(self) -> None:
        """Lazy load SentenceTransformer or configure fallback."""
        if self._model is not None or self._is_fallback:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("BAAI/bge-m3 loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer '{self.model_name}' ({e}). "
                "Enabling deterministic 1024-d projection engine for local execution."
            )
            self._is_fallback = True

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate 1024-dimensional dense embeddings for a list of passages.
        """
        if not texts:
            return []

        self._load_model()

        if self._model is not None and not self._is_fallback:
            try:
                embeddings = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=len(texts) > 20,
                    normalize_embeddings=self.cfg.embedding_normalize,
                )
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error(f"Inference error with BGE-M3 model: {e}. Falling back.")

        # Fallback deterministic unit vector generator (1024 dimensions)
        return [self._generate_deterministic_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Generate dense embedding for single search query."""
        results = self.embed_texts([query])
        return results[0]

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """
        Generate deterministic pseudo-semantic 1024-d unit vector from text tokens.
        Ensures pipeline testing and offline execution without downloading ~2GB model.
        """
        vec = [0.0] * EMBEDDING_DIMENSION
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            h = hashlib.sha256(word.encode("utf-8")).digest()
            for i in range(0, min(len(h), 32), 2):
                idx = int.from_bytes(h[i : i + 2], "big") % EMBEDDING_DIMENSION
                val = (h[i] - 128) / 128.0
                vec[idx] += val

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec


# Global embedder instance
embedder = LocalEmbedder()
