"""
Embedding module with pluggable backends.
Provides a unified interface for generating text embeddings.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode a list of texts into embeddings. Returns (N, dim) array."""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Embedding backend using Sentence Transformers (HuggingFace).
    Uses all-MiniLM-L6-v2 by default — lightweight (80MB) with good retrieval quality.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None  # lazy load to save memory

    def _load_model(self):
        if self._model is None:
            print(f"  Loading embedding model: {self._model_name}...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            print(f"  Model loaded successfully.")

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode texts in batches to avoid memory issues."""
        self._load_model()

        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,  # for cosine similarity via dot product
            )
            all_embeddings.append(embeddings)

            if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= total:
                print(f"  Embedded {min(i + batch_size, total)}/{total} chunks")

        return np.vstack(all_embeddings).astype(np.float32)

    def dimension(self) -> int:
        return 384  # all-MiniLM-L6-v2


def get_embedding_backend(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingBackend:
    """Factory to create the embedding backend."""
    return SentenceTransformerBackend(model_name)
