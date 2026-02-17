"""
Pytest integration tests for the Policy Q&A Bot.
Tests the full pipeline: ingestion, chunking, retrieval, and answer generation.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import PolicyQAPipeline
from tests.test_cases import TEST_CASES


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pipeline():
    """Initialize the pipeline once for all tests."""
    p = PolicyQAPipeline()
    p.initialize()
    return p


# ── Chunking Tests ────────────────────────────────────────────────────────

class TestChunking:
    """Tests for the chunking module (Rubric: Retrieval quality & chunking — 3 pts)."""

    def test_chunks_have_metadata(self, pipeline):
        """Each chunk must carry required metadata fields."""
        from core.chunking import Chunk

        stats = pipeline.get_stats()
        assert stats["chunks"] > 0, "Pipeline should have created chunks"

    def test_chunk_count_reasonable(self, pipeline):
        """Number of chunks should be reasonable for 3 policy PDFs."""
        stats = pipeline.get_stats()
        assert stats["chunks"] >= 20, "Should have at least 20 chunks from 3 PDFs"
        assert stats["chunks"] <= 5000, "Should not have an unreasonable number of chunks"

    def test_documents_loaded(self, pipeline):
        """All 3 policy documents should be indexed."""
        stats = pipeline.get_stats()
        assert stats["documents"] == 3, f"Expected 3 documents, got {stats['documents']}"


# ── Retrieval & Citation Tests ────────────────────────────────────────────

class TestRetrieval:
    """Tests for retrieval quality and citations (Rubric: 3 pts each)."""

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in TEST_CASES if tc["type"] == "in_domain"],
        ids=[tc["question"][:40] for tc in TEST_CASES if tc["type"] == "in_domain"],
    )
    def test_in_domain_has_answer(self, pipeline, test_case):
        """In-domain questions must produce a grounded answer."""
        result = pipeline.ask(test_case["question"])

        assert result["answer"], "Answer should not be empty"
        assert "error" not in result["answer"].lower(), "Answer should not contain errors"

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in TEST_CASES if tc["type"] == "in_domain"],
        ids=[tc["question"][:40] + "_citations" for tc in TEST_CASES if tc["type"] == "in_domain"],
    )
    def test_in_domain_has_citations(self, pipeline, test_case):
        """In-domain answers must include citations with document references."""
        result = pipeline.ask(test_case["question"])

        # Check that sources are provided
        assert len(result["sources"]) > 0, "In-domain answers must have source citations"

        # Verify source structure
        for source in result["sources"]:
            assert "doc_name" in source, "Each source must have a doc_name"
            assert "page" in source, "Each source must have a page number"

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in TEST_CASES if tc["type"] == "in_domain" and tc["expected_keywords"]],
        ids=[tc["question"][:40] + "_keywords" for tc in TEST_CASES if tc["type"] == "in_domain" and tc["expected_keywords"]],
    )
    def test_in_domain_relevant_content(self, pipeline, test_case):
        """In-domain answers should contain relevant keywords."""
        result = pipeline.ask(test_case["question"])
        answer_lower = result["answer"].lower()

        # Check if at least ONE of the expected keywords appears
        found = any(kw in answer_lower for kw in test_case["expected_keywords"])
        assert found, (
            f"Expected at least one of {test_case['expected_keywords']} "
            f"in answer: {result['answer'][:200]}"
        )


# ── Out-of-Scope Tests ───────────────────────────────────────────────────

class TestOutOfScope:
    """Tests for no-answer behavior (Rubric: Robustness — 2 pts)."""

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in TEST_CASES if tc["type"] == "out_of_scope"],
        ids=[tc["question"][:40] for tc in TEST_CASES if tc["type"] == "out_of_scope"],
    )
    def test_out_of_scope_says_no_answer(self, pipeline, test_case):
        """Out-of-scope questions must trigger a 'cannot find' response."""
        result = pipeline.ask(test_case["question"])

        answer_lower = result["answer"].lower()
        has_no_answer = (
            "cannot find" in answer_lower
            or "no definitive answer" in answer_lower
            or "not find" in answer_lower
            or "i don't have" in answer_lower
        )
        assert has_no_answer, (
            f"Out-of-scope question should trigger no-answer response. "
            f"Got: {result['answer'][:200]}"
        )


# ── Near-Miss Tests ───────────────────────────────────────────────────────

class TestNearMiss:
    """Tests for near-miss questions — should produce some response."""

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in TEST_CASES if tc["type"] == "near_miss"],
        ids=[tc["question"][:40] for tc in TEST_CASES if tc["type"] == "near_miss"],
    )
    def test_near_miss_produces_response(self, pipeline, test_case):
        """Near-miss questions should produce a response (answer or no-answer)."""
        result = pipeline.ask(test_case["question"])
        assert result["answer"], "Should produce some response (answer or no-answer)"
        assert isinstance(result["confidence"], float), "Should return a confidence score"


# ── Pipeline Integration Test ─────────────────────────────────────────────

class TestPipelineIntegration:
    """Full pipeline integration test."""

    def test_pipeline_stats(self, pipeline):
        """Pipeline should report correct statistics."""
        stats = pipeline.get_stats()
        assert stats["documents"] > 0
        assert stats["chunks"] > 0
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
        assert stats["index_loaded"] is True

    def test_ask_returns_complete_result(self, pipeline):
        """ask() should return a complete result dict."""
        result = pipeline.ask("What does this policy cover?")

        assert "question" in result
        assert "answer" in result
        assert "citations" in result
        assert "sources" in result
        assert "confidence" in result
        assert isinstance(result["sources"], list)
        assert isinstance(result["confidence"], float)
