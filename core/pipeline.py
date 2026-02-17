"""
End-to-end Policy Q&A Pipeline.
Orchestrates: ingestion → chunking → indexing → retrieval → generation.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DOCS_DIR,
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LLM_BACKEND,
)
from core.ingestion import load_all_documents, DocumentPage
from core.chunking import chunk_pages, Chunk
from core.embeddings import get_embedding_backend
from core.vectorstore import VectorStore
from core.retriever import Retriever
from core.llm_backend import get_llm_backend, LLMBackend
from core.prompts import build_qa_prompt


class PolicyQAPipeline:
    """
    Main pipeline class that orchestrates the full RAG workflow.
    
    Usage:
        pipeline = PolicyQAPipeline()
        pipeline.initialize()
        result = pipeline.ask("Is wear and tear covered?")
    """

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        db_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        llm_backend_name: Optional[str] = None,
    ):
        self._docs_dir = docs_dir or DOCS_DIR
        self._db_dir = db_dir or VECTOR_DB_DIR
        self._embedding_model = embedding_model or EMBEDDING_MODEL
        self._llm_backend_name = llm_backend_name or LLM_BACKEND

        # Components (initialized lazily)
        self._embedding_backend = None
        self._vector_store = None
        self._retriever = None
        self._llm: Optional[LLMBackend] = None

        # State
        self._is_initialized = False
        self._doc_count = 0
        self._chunk_count = 0
        self._pages: List[DocumentPage] = []
        self._chunks: List[Chunk] = []

    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the pipeline: load docs, chunk, embed, index.
        Uses cached index if available unless force_rebuild is True.
        """
        print("\n=== Initializing Policy Q&A Pipeline ===")
        start = time.time()

        # Initialize embedding backend
        print("[1/4] Loading embedding model...")
        self._embedding_backend = get_embedding_backend(self._embedding_model)

        # Initialize vector store
        self._vector_store = VectorStore(self._embedding_backend, self._db_dir)

        # Try loading cached index
        if not force_rebuild and self._vector_store.load():
            print("[2/4] Loaded cached index.")
            self._chunk_count = self._vector_store.chunk_count
            self._doc_count = len(self._vector_store.doc_names)
        else:
            # Full pipeline: ingest → chunk → index
            print("[2/4] Ingesting documents...")
            self._pages = load_all_documents(self._docs_dir)

            if not self._pages:
                raise RuntimeError(f"No documents found in {self._docs_dir}")

            self._doc_count = len(set(p.doc_name for p in self._pages))

            print("[3/4] Chunking documents...")
            self._chunks = chunk_pages(
                self._pages,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                min_chunk_size=MIN_CHUNK_SIZE,
            )
            self._chunk_count = len(self._chunks)

            print("[3/4] Building FAISS index...")
            self._vector_store.build_index(
                self._chunks,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
            self._vector_store.save()

        # Initialize retriever
        print("[4/4] Setting up retriever and LLM...")
        self._retriever = Retriever(
            self._vector_store,
            self._embedding_backend,
            top_k=TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        # Initialize LLM backend
        self._llm = get_llm_backend(self._llm_backend_name)

        elapsed = time.time() - start
        self._is_initialized = True
        print(f"\n=== Pipeline ready ({elapsed:.1f}s) ===")
        print(f"  Documents: {self._doc_count}")
        print(f"  Chunks: {self._chunk_count}")
        print(f"  LLM Backend: {self._llm.name}")
        print()

    # ── Conversational patterns ──────────────────────────────────────────
    _GREETING_PATTERNS = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "howdy", "hola", "sup", "yo", "hii", "hiii", "hihi",
    }
    _IDENTITY_PATTERNS = {
        "who are you", "who are u", "what are you", "what are u",
        "what is this", "what do you do", "introduce yourself",
        "what can you do", "what can u do",
    }
    _THANKS_PATTERNS = {
        "thanks", "thank you", "thank u", "thx", "ty", "cheers",
    }
    _HELP_PATTERNS = {
        "help", "help me", "what can i ask", "how to use",
    }

    def _detect_conversational(self, question: str) -> Optional[Dict]:
        """
        Detect conversational intents (greetings, identity, thanks, help).
        Returns a response dict if matched, None otherwise.
        """
        q = question.lower().strip().rstrip("?!.")

        # Greetings
        if q in self._GREETING_PATTERNS or any(q.startswith(g) for g in ["hi ", "hey ", "hello "]):
            doc_names = self._vector_store.doc_names if self._vector_store else []
            doc_count = len(doc_names)
            return {
                "question": question,
                "answer": (
                    f"Hello! I'm your **Policy Q&A Assistant**. I have "
                    f"**{doc_count} policy document{'s' if doc_count != 1 else ''}** loaded "
                    f"and ready to search.\n\n"
                    f"Ask me anything about your insurance policies — for example:\n"
                    f"- \"Is wear-and-tear covered?\"\n"
                    f"- \"What is the excess for water damage?\"\n"
                    f"- \"What definitions apply to Insured Person?\"\n\n"
                    f"I'll find the relevant sections and cite the exact source."
                ),
                "citations": "",
                "sources": [],
                "confidence": 1.0,
                "follow_ups": [
                    "What does this policy cover?",
                    "What are the general exclusions?",
                    "How do I make a claim?",
                ],
            }

        # Identity questions
        if q in self._IDENTITY_PATTERNS:
            return {
                "question": question,
                "answer": (
                    "I'm a **Policy Q&A Assistant** — an AI-powered chatbot that answers "
                    "questions about insurance policy documents.\n\n"
                    "I search through the loaded policy PDFs and give you **grounded answers "
                    "with clause-level citations** so you know exactly where the information "
                    "comes from.\n\n"
                    "Just type your question about any policy topic!"
                ),
                "citations": "",
                "sources": [],
                "confidence": 1.0,
                "follow_ups": [
                    "What documents are loaded?",
                    "What items are excluded from coverage?",
                    "What is the claims process?",
                ],
            }

        # Thanks
        if q in self._THANKS_PATTERNS:
            return {
                "question": question,
                "answer": "You're welcome! Feel free to ask more questions about your policy anytime.",
                "citations": "",
                "sources": [],
                "confidence": 1.0,
                "follow_ups": [],
            }

        # Help
        if q in self._HELP_PATTERNS:
            return {
                "question": question,
                "answer": (
                    "Here's how to get the best results:\n\n"
                    "**Ask specific questions** about your policy, such as:\n"
                    "- Coverage: \"What does the contents insurance cover?\"\n"
                    "- Exclusions: \"Is flood damage excluded?\"\n"
                    "- Definitions: \"How is 'Insured Event' defined?\"\n"
                    "- Claims: \"How do I lodge a claim?\"\n"
                    "- Conditions: \"What are my obligations under the policy?\"\n\n"
                    "I'll search the policy documents and provide answers with exact citations."
                ),
                "citations": "",
                "sources": [],
                "confidence": 1.0,
                "follow_ups": [
                    "What does this policy cover?",
                    "What are the general exclusions?",
                    "What is the excess amount?",
                ],
            }

        return None

    def ask(self, question: str) -> Dict:
        """
        Ask a question and get a grounded answer with citations.
        Handles both conversational messages and policy questions.
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Check for conversational intent first (no retrieval needed)
        conv_response = self._detect_conversational(question)
        if conv_response:
            return conv_response

        # Retrieve relevant chunks
        results = self._retriever.retrieve(question)

        # Build context
        context = self._retriever.build_context(results)

        # Build prompt
        prompt = build_qa_prompt(question, context)

        # Generate answer (returns {answer, follow_ups})
        llm_result = self._llm.generate(prompt, results)
        answer_text = llm_result["answer"]
        follow_ups = llm_result.get("follow_ups", [])

        # Format citations
        citations = self._retriever.format_citations(results)

        # Build source details
        sources = []
        for chunk, score in results:
            sources.append({
                "doc_name": chunk.doc_name,
                "section": chunk.section,
                "clause": chunk.clause_number,
                "page": chunk.page,
                "heading_path": chunk.heading_path,
                "score": round(score, 4),
                "snippet": chunk.text[:200],
            })

        # Confidence based on best similarity score
        confidence = self._retriever.get_best_score(results)

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "sources": sources,
            "confidence": round(confidence, 4),
            "follow_ups": follow_ups,
        }

    def get_stats(self) -> Dict:
        """Return pipeline statistics."""
        return {
            "documents": self._doc_count,
            "chunks": self._chunk_count,
            "embedding_model": self._embedding_model,
            "llm_backend": self._llm.name if self._llm else "not initialized",
            "index_loaded": self._vector_store.is_loaded if self._vector_store else False,
            "doc_names": self._vector_store.doc_names if self._vector_store else [],
        }

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
