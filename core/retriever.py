"""
Retriever module.
Handles query embedding, FAISS search, filtering, and citation formatting.
"""

from typing import Dict, List, Tuple

import numpy as np

from core.chunking import Chunk
from core.embeddings import EmbeddingBackend
from core.vectorstore import VectorStore


class Retriever:
    """Retrieves relevant chunks for a given query and formats citations."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_backend: EmbeddingBackend,
        top_k: int = 5,
        similarity_threshold: float = 0.25,
    ):
        self._store = vector_store
        self._backend = embedding_backend
        self._top_k = top_k
        self._threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.
        Returns list of (chunk, similarity_score) filtered by threshold.
        """
        # Embed the query
        query_emb = self._backend.encode([query], batch_size=1)

        # Search FAISS
        results = self._store.search(query_emb, top_k=self._top_k)

        # Filter by similarity threshold
        filtered = [(chunk, score) for chunk, score in results if score >= self._threshold]

        # Deduplicate overlapping chunks (same doc, same page, high text overlap)
        deduped = self._deduplicate(filtered)

        return deduped

    def _deduplicate(
        self, results: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """Remove near-duplicate chunks (same doc + page + high overlap)."""
        if len(results) <= 1:
            return results

        kept = []
        seen_texts = []

        for chunk, score in results:
            is_dup = False
            for seen in seen_texts:
                # Check text overlap
                overlap = self._text_overlap(chunk.text, seen)
                if overlap > 0.7:
                    is_dup = True
                    break

            if not is_dup:
                kept.append((chunk, score))
                seen_texts.append(chunk.text)

        return kept

    @staticmethod
    def _text_overlap(text_a: str, text_b: str) -> float:
        """Compute word-level Jaccard overlap between two texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def format_citations(results: List[Tuple[Chunk, float]]) -> str:
        """
        Format citation strings from retrieved chunks.
        Output: Sources: Policy_A.pdf §3.2 (Exclusions), p.12; ...
        """
        if not results:
            return ""

        citations = []
        seen = set()

        for chunk, _ in results:
            cite = chunk.citation_string()
            if cite not in seen:
                citations.append(cite)
                seen.add(cite)

        return "Sources: " + "; ".join(citations)

    @staticmethod
    def build_context(results: List[Tuple[Chunk, float]]) -> str:
        """Build context string from retrieved chunks for the LLM."""
        if not results:
            return ""

        context_parts = []
        for i, (chunk, score) in enumerate(results, 1):
            header = f"[Source {i}: {chunk.doc_name}"
            if chunk.clause_number:
                header += f" §{chunk.clause_number}"
            if chunk.section:
                header += f" ({chunk.section})"
            header += f", p.{chunk.page}]"

            context_parts.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(context_parts)

    def get_best_score(self, results: List[Tuple[Chunk, float]]) -> float:
        """Return the highest similarity score from results."""
        if not results:
            return 0.0
        return max(score for _, score in results)
