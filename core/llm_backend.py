"""
Pluggable LLM backend module.
Provides abstract interface and concrete implementations for answer generation.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from core.chunking import Chunk


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, context_chunks: List[Tuple[Chunk, float]]) -> dict:
        """
        Generate an answer given a prompt and context chunks.
        Returns: {"answer": str, "follow_ups": List[str]}
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass


class MockLLMBackend(LLMBackend):
    """
    Rule-based mock LLM backend.
    Extracts relevant content from context and constructs structured, grounded answers.
    No external API or heavy model required.
    """

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, context_chunks: List[Tuple[Chunk, float]]) -> dict:
        """Generate a grounded answer by extracting and organizing relevant content."""
        if not context_chunks:
            return {
                "answer": self._no_answer_response(
                    [],
                    "I couldn't find anything related to that in the loaded policy documents."
                ),
                "follow_ups": [
                    "What does this policy cover?",
                    "What are the general exclusions?",
                    "How do I make a claim?",
                ],
            }

        question = self._extract_question(prompt)
        question_lower = question.lower()
        best_score = max(score for _, score in context_chunks)

        if best_score < 0.25:
            return {
                "answer": self._no_answer_response(
                    context_chunks,
                    "I couldn't find a direct answer to that, but here are some related sections that might help."
                ),
                "follow_ups": self._generate_follow_ups(context_chunks, question_lower),
            }

        # Build structured answer from context
        answer = self._build_structured_answer(question, question_lower, context_chunks)
        follow_ups = self._generate_follow_ups(context_chunks, question_lower)

        return {"answer": answer, "follow_ups": follow_ups}

    def _extract_question(self, prompt: str) -> str:
        """Extract the question from the full prompt."""
        match = re.search(r"QUESTION:\s*(.+?)(?:\n|INSTRUCTIONS:)", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        return prompt.split("\n")[-1].strip()

    @staticmethod
    def _clean_pdf_text(text: str) -> str:
        """
        Clean up raw PDF-extracted text:
        - Join broken lines (PDF wraps mid-sentence)
        - Remove excessive whitespace
        - Fix common PDF artifacts
        """
        # Replace single newlines (mid-sentence wraps) with spaces
        # But preserve double newlines (paragraph breaks)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"  +", " ", text)
        # Fix bullet points
        text = re.sub(r"\s*•\s*", "\n• ", text)
        # Remove "(continued...)" artifacts
        text = re.sub(r"\(continued\.{0,3}\)", "", text)
        return text.strip()

    def _build_structured_answer(
        self,
        question: str,
        question_lower: str,
        context_chunks: List[Tuple[Chunk, float]],
    ) -> str:
        """Build a well-formatted answer organized by source sections."""

        # Group chunks by document and section
        grouped = {}
        for chunk, score in context_chunks:
            key = (chunk.doc_name, chunk.section or chunk.heading_path or "General")
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((chunk, score))

        # Build answer sections
        answer_parts = []
        answer_parts.append("Based on the policy documents:\n")

        for (doc_name, section), chunks in grouped.items():
            # Clean and combine the text from related chunks
            combined_text = ""
            for chunk, score in chunks:
                cleaned = self._clean_pdf_text(chunk.text)
                # Extract the most relevant passage from this chunk
                passage = self._extract_best_passage(cleaned, question_lower)
                if passage:
                    combined_text += passage + " "

            if not combined_text.strip():
                continue

            # Format the section
            section_label = section if section else "General"
            clause_info = ""
            if chunks[0][0].clause_number:
                clause_info = f" (§{chunks[0][0].clause_number})"

            answer_parts.append(f"**{section_label}{clause_info}** — *{doc_name}, p.{chunks[0][0].page}*")
            answer_parts.append(f"> {combined_text.strip()}\n")

        if len(answer_parts) <= 1:
            return self._no_answer_response(
                context_chunks,
                "While related sections were found, they do not directly answer this specific question."
            )

        # Add citations
        answer_parts.append("")
        answer_parts.append(self._format_citations(context_chunks))

        return "\n".join(answer_parts)

    def _extract_best_passage(self, text: str, question_lower: str, max_chars: int = 400) -> str:
        """
        Extract the most relevant passage from cleaned text.
        Returns a coherent passage rather than individual sentence fragments.
        """
        question_words = self._get_keywords(question_lower)

        # Split into sentences (handle abbreviations)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        if not sentences:
            return ""

        # Score each sentence
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            sent_lower = sent.lower()
            sent_words = set(re.findall(r'\b\w+\b', sent_lower))
            overlap = len(question_words & sent_words)
            scored.append((sent, overlap))

        if not scored:
            return ""

        # Sort by relevance score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take the top sentences, keeping total under max_chars
        result = []
        total_len = 0
        for sent, _ in scored:
            if total_len + len(sent) > max_chars:
                break
            result.append(sent)
            total_len += len(sent)
            if len(result) >= 3:
                break

        return " ".join(result)

    @staticmethod
    def _get_keywords(text: str) -> set:
        """Extract meaningful keywords from text."""
        stop_words = {
            "is", "the", "a", "an", "and", "or", "of", "to", "in", "for",
            "on", "with", "by", "at", "from", "as", "it", "that", "this",
            "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "shall", "can", "what", "how", "which", "who",
            "when", "where", "why", "under", "my", "your", "their", "its",
            "there", "here", "about", "than", "then", "also", "just",
        }
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return words - stop_words

    def _generate_follow_ups(
        self,
        context_chunks: List[Tuple[Chunk, float]],
        question_lower: str,
    ) -> List[str]:
        """Generate 3 follow-up question suggestions based on retrieved context."""
        follow_ups = []
        seen_topics = set()

        # Templates based on common policy topics
        topic_templates = {
            "exclusion": "What specific exclusions apply to {topic}?",
            "cover": "What are the limits for {topic} coverage?",
            "definition": "How does the policy define '{topic}'?",
            "claim": "How do I make a claim for {topic}?",
            "condition": "What conditions must be met for {topic}?",
            "excess": "What excess applies to {topic}?",
            "limit": "What are the coverage limits for {topic}?",
            "waiting": "Is there a waiting period for {topic}?",
        }

        for chunk, score in context_chunks:
            section = (chunk.section or chunk.heading_path or "").lower()
            
            # Generate follow-up based on section type
            for topic_key, template in topic_templates.items():
                if topic_key in section and topic_key not in seen_topics:
                    # Extract a topic noun from the chunk heading
                    topic = self._extract_topic(chunk, question_lower)
                    if topic and topic not in seen_topics:
                        follow_ups.append(template.format(topic=topic))
                        seen_topics.add(topic_key)
                        seen_topics.add(topic)
                    break

            if len(follow_ups) >= 3:
                break

        # Fill in with generic follow-ups if we don't have 3
        generic_follow_ups = [
            "What items are excluded from coverage?",
            "What is the claims process?",
            "What are the policy's general conditions?",
            "How is the excess calculated?",
            "What additional optional covers are available?",
            "What are my obligations under this policy?",
        ]

        for q in generic_follow_ups:
            if len(follow_ups) >= 3:
                break
            # Don't suggest something too similar to what was already asked
            q_words = self._get_keywords(q.lower())
            question_words = self._get_keywords(question_lower)
            if len(q_words & question_words) < 2:
                follow_ups.append(q)

        return follow_ups[:3]

    @staticmethod
    def _extract_topic(chunk: Chunk, question_lower: str) -> str:
        """Extract a topic noun from the chunk's heading or section."""
        heading = chunk.heading_path or chunk.section or ""
        # Get the last part of the heading path
        parts = heading.split(">")
        if parts:
            topic = parts[-1].strip().strip('"').strip("'")
            topic_lower = topic.lower()
            # Don't return the same topic as the question
            if topic_lower not in question_lower and len(topic) > 2:
                return topic
        return ""

    def _no_answer_response(
        self,
        context_chunks: List[Tuple[Chunk, float]],
        explanation: str,
    ) -> str:
        """Build a friendly no-answer response with closest related clauses."""
        response = f"{explanation}\n"

        if context_chunks:
            response += "\n**Related sections I found:**\n"
            seen = set()
            for chunk, score in context_chunks[:3]:
                cite = chunk.citation_string()
                if cite not in seen:
                    snippet = self._clean_pdf_text(chunk.text)[:150].strip()
                    if len(chunk.text) > 150:
                        snippet += "..."
                    response += f"- {cite}: \"{snippet}\"\n"
                    seen.add(cite)

            response += "\n" + self._format_citations(context_chunks)
        else:
            response += "\nTry rephrasing your question or asking about a specific policy topic like coverage, exclusions, or claims."

        return response

    def _format_citations(self, chunks: List[Tuple[Chunk, float]]) -> str:
        """Format citations line."""
        if not chunks:
            return ""
        seen = set()
        cites = []
        for chunk, _ in chunks:
            cite = chunk.citation_string()
            if cite not in seen:
                cites.append(cite)
                seen.add(cite)
        return "Sources: " + "; ".join(cites)


class TransformersBackend(LLMBackend):
    """
    HuggingFace Transformers backend stub.
    Uses a local model for generation (e.g., google/flan-t5-base).
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "transformers"

    def _load_model(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"  Loading LLM: {self._model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            print(f"  LLM loaded.")

    def generate(self, prompt: str, context_chunks: List[Tuple[Chunk, float]]) -> dict:
        self._load_model()
        inputs = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self._model.generate(**inputs, max_new_tokens=256)
        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        citations = MockLLMBackend()._format_citations(context_chunks)
        return {
            "answer": f"{answer}\n\n{citations}",
            "follow_ups": [],
        }


class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend stub. Demonstrates pluggable architecture.
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

    def generate(self, prompt: str, context_chunks: List[Tuple[Chunk, float]]) -> dict:
        raise NotImplementedError(
            "OpenAI backend requires an API key. Set OPENAI_API_KEY environment variable "
            "and install the 'openai' package. This stub demonstrates the pluggable architecture."
        )


def get_llm_backend(backend_name: str = "mock") -> LLMBackend:
    """Factory to create an LLM backend by name."""
    backends = {
        "mock": MockLLMBackend,
        "transformers": TransformersBackend,
        "openai": OpenAIBackend,
    }

    if backend_name not in backends:
        raise ValueError(
            f"Unknown LLM backend: {backend_name}. Available: {list(backends.keys())}"
        )

    return backends[backend_name]()
