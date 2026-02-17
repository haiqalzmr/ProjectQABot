"""
FAISS vector store module.
Handles building, saving, loading, and searching the FAISS index.
"""

import json
import gc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.chunking import Chunk
from core.embeddings import EmbeddingBackend


class VectorStore:
    """FAISS-based vector store for policy document chunks."""

    def __init__(self, embedding_backend: EmbeddingBackend, db_dir: Path):
        self._backend = embedding_backend
        self._db_dir = db_dir
        self._index = None
        self._chunks: List[Chunk] = []
        self._is_loaded = False

    @property
    def index_path(self) -> Path:
        return self._db_dir / "index.faiss"

    @property
    def meta_path(self) -> Path:
        return self._db_dir / "chunks.json"

    def build_index(self, chunks: List[Chunk], batch_size: int = 16) -> None:
        """Build a FAISS index from chunks."""
        import faiss

        if not chunks:
            print("[WARNING] No chunks to index.")
            return

        self._chunks = chunks
        texts = [c.text for c in chunks]

        print(f"  Embedding {len(texts)} chunks (batch_size={batch_size})...")
        embeddings = self._backend.encode(texts, batch_size=batch_size)

        # Build FAISS index (Inner Product = cosine similarity with normalized vectors)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        print(f"  FAISS index built: {self._index.ntotal} vectors, dim={dim}")
        self._is_loaded = True

        # Force garbage collection after heavy embedding work
        gc.collect()

    def save(self) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        import faiss

        if self._index is None:
            print("[WARNING] No index to save.")
            return

        self._db_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))

        # Save chunk metadata as JSON
        meta = []
        for c in self._chunks:
            meta.append({
                "text": c.text,
                "doc_name": c.doc_name,
                "page": c.page,
                "section": c.section,
                "clause_number": c.clause_number,
                "heading_path": c.heading_path,
                "cross_references": c.cross_references,
                "chunk_id": c.chunk_id,
            })

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  Index saved to {self._db_dir}")

    def load(self) -> bool:
        """Load a persisted index from disk. Returns True if successful."""
        import faiss

        if not self.index_path.exists() or not self.meta_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(self.index_path))

            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self._chunks = [
                Chunk(
                    text=m["text"],
                    doc_name=m["doc_name"],
                    page=m["page"],
                    section=m.get("section", ""),
                    clause_number=m.get("clause_number", ""),
                    heading_path=m.get("heading_path", ""),
                    cross_references=m.get("cross_references", []),
                    chunk_id=m.get("chunk_id", i),
                )
                for i, m in enumerate(meta)
            ]

            self._is_loaded = True
            print(f"  Loaded index: {self._index.ntotal} vectors, {len(self._chunks)} chunks")
            return True
        except Exception as e:
            print(f"[WARNING] Failed to load index: {e}")
            return False

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """Search the index and return top-k (chunk, score) pairs."""
        if self._index is None or not self._is_loaded:
            print("[ERROR] Index not loaded.")
            return []

        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self._index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def doc_names(self) -> List[str]:
        return list(set(c.doc_name for c in self._chunks))
