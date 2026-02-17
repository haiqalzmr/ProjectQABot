"""
Section-aware chunking module.
Splits document pages into semantically meaningful chunks with rich metadata.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from core.ingestion import DocumentPage


@dataclass
class Chunk:
    """A text chunk with metadata for retrieval and citation."""
    text: str
    doc_name: str
    page: int
    section: str = ""
    clause_number: str = ""
    heading_path: str = ""
    cross_references: List[str] = field(default_factory=list)
    chunk_id: int = 0

    def to_metadata(self) -> dict:
        """Return metadata dict for storage."""
        return {
            "doc_name": self.doc_name,
            "page": self.page,
            "section": self.section,
            "clause_number": self.clause_number,
            "heading_path": self.heading_path,
            "cross_references": self.cross_references,
            "chunk_id": self.chunk_id,
        }

    def citation_string(self) -> str:
        """Format this chunk's source as a citation string."""
        parts = [self.doc_name]
        if self.clause_number:
            parts.append(f"§{self.clause_number}")
        if self.section:
            parts.append(f"({self.section})")
        elif self.heading_path:
            parts.append(f'"{self.heading_path}"')
        parts.append(f"p.{self.page}")
        return " ".join(parts)


# ── Heading / Section Detection ────────────────────────────────────────────

# Matches: "1.", "1.1", "1.1.1", "Section 3.2"
NUMBERED_SECTION_RE = re.compile(
    r"^(?:Section\s+)?(\d+(?:\.\d+)*)\.\s+(.*)", re.IGNORECASE
)

# Matches: ALL CAPS headings (at least 3 chars, ≤10 words)
ALLCAPS_HEADING_RE = re.compile(r"^([A-Z][A-Z\s\-&/]{2,})$")

# Matches cross-references: "see Section 3.2", "refer to clause 4.1"
CROSS_REF_RE = re.compile(
    r"(?:see|refer\s+to|as\s+defined\s+in|under)\s+(?:Section|Clause|Part)\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Common insurance policy section keywords
SECTION_KEYWORDS = {
    "definitions", "exclusions", "coverage", "conditions", "endorsements",
    "general conditions", "special conditions", "claims", "how to claim",
    "what is covered", "what is not covered", "limits", "excess",
    "policy schedule", "insuring clause", "preamble", "declarations",
    "optional covers", "additional benefits", "extensions",
}


def _detect_section_label(line: str) -> Optional[str]:
    """Detect if a line is a section heading; return the label or None."""
    stripped = line.strip()
    if not stripped:
        return None

    # Check numbered section
    m = NUMBERED_SECTION_RE.match(stripped)
    if m:
        return stripped[:120]

    # Check ALL CAPS heading
    m = ALLCAPS_HEADING_RE.match(stripped)
    if m and len(stripped) >= 3:
        return stripped[:120]

    # Check if line matches known section keywords
    lower = stripped.lower().rstrip(":")
    if lower in SECTION_KEYWORDS:
        return stripped[:120]

    return None


def _find_cross_references(text: str) -> List[str]:
    """Extract cross-reference targets from text."""
    return [m.group(1) for m in CROSS_REF_RE.finditer(text)]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters."""
    return len(text) // 4


# ── Chunking Logic ─────────────────────────────────────────────────────────

def _split_into_sections(text: str) -> List[dict]:
    """Split a page's text into sections based on detected headings."""
    lines = text.split("\n")
    sections = []
    current_section = {"heading": "", "clause": "", "lines": []}

    for line in lines:
        heading = _detect_section_label(line)
        if heading:
            # Save previous section if it has content
            if current_section["lines"]:
                sections.append(current_section)

            # Extract clause number if present
            clause = ""
            m = NUMBERED_SECTION_RE.match(line.strip())
            if m:
                clause = m.group(1)

            current_section = {
                "heading": heading,
                "clause": clause,
                "lines": [line],
            }
        else:
            current_section["lines"].append(line)

    # Don't forget the last section
    if current_section["lines"]:
        sections.append(current_section)

    return sections


def _sub_chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk_size: int = 80,
) -> List[str]:
    """
    Split text into sub-chunks of approximately chunk_size tokens
    with overlap. Splits at sentence boundaries when possible.
    """
    if _estimate_tokens(text) <= chunk_size:
        return [text] if _estimate_tokens(text) >= min_chunk_size else [text] if text.strip() else []

    # Split at sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)

        if current_tokens + sent_tokens > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            if _estimate_tokens(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)

            # Overlap: keep last few sentences
            overlap_tokens = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                overlap_tokens += _estimate_tokens(s)
                if overlap_tokens >= chunk_overlap:
                    break
                overlap_sentences.insert(0, s)

            current_chunk = overlap_sentences
            current_tokens = sum(_estimate_tokens(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    # Last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if _estimate_tokens(chunk_text) >= min_chunk_size or not chunks:
            chunks.append(chunk_text)
        elif chunks:
            # Append to previous chunk if too small
            chunks[-1] += " " + chunk_text

    return chunks


def chunk_pages(
    pages: List[DocumentPage],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk_size: int = 80,
) -> List[Chunk]:
    """
    Chunk document pages into retrieval-ready pieces.
    
    Strategy:
    1. Split each page's text into sections by detecting headings
    2. Sub-chunk large sections to 400-800 token pieces with overlap
    3. Attach rich metadata (doc_name, section, clause, page, heading_path)
    4. Detect and store cross-references
    """
    all_chunks = []
    chunk_id = 0

    # Track heading hierarchy across pages per document
    heading_stack = {}  # doc_name -> list of headings

    for page in pages:
        doc_name = page.doc_name

        if doc_name not in heading_stack:
            heading_stack[doc_name] = []

        sections = _split_into_sections(page.text)

        for sec in sections:
            heading = sec["heading"]
            clause = sec["clause"]
            section_text = "\n".join(sec["lines"]).strip()

            if not section_text:
                continue

            # Update heading stack
            if heading:
                # Determine depth by clause number or position
                depth = len(clause.split(".")) if clause else 1
                stack = heading_stack[doc_name]

                # Trim stack to current depth
                while len(stack) >= depth:
                    stack.pop() if stack else None
                stack.append(heading)
                heading_stack[doc_name] = stack

            # Build heading path
            stack = heading_stack[doc_name]
            heading_path = " > ".join(stack) if stack else ""

            # Determine section label
            section_label = ""
            for kw in SECTION_KEYWORDS:
                if kw in heading.lower() if heading else "":
                    section_label = kw.title()
                    break
            if not section_label and heading:
                section_label = heading[:60]

            # Find cross-references
            cross_refs = _find_cross_references(section_text)

            # Sub-chunk if needed
            sub_chunks = _sub_chunk_text(
                section_text, chunk_size, chunk_overlap, min_chunk_size
            )

            for sub_text in sub_chunks:
                chunk = Chunk(
                    text=sub_text,
                    doc_name=doc_name,
                    page=page.page_num,
                    section=section_label,
                    clause_number=clause,
                    heading_path=heading_path,
                    cross_references=cross_refs,
                    chunk_id=chunk_id,
                )
                all_chunks.append(chunk)
                chunk_id += 1

    print(f"  Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks
