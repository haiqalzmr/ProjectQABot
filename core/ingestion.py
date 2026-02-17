"""
Document ingestion module.
Loads PDF policy documents and extracts structured text page-by-page.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class DocumentPage:
    """Represents a single page extracted from a document."""
    doc_name: str
    page_num: int
    text: str
    headings: List[str] = field(default_factory=list)


def extract_headings_from_text(text: str) -> List[str]:
    """
    Detect likely headings from text using heuristics:
    - Lines in ALL CAPS (≥3 chars)
    - Lines matching numbered section patterns (e.g., '1.', '1.1', 'Section 3')
    - Short bold-style lines (often extracted as standalone lines)
    """
    headings = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        # Numbered sections: "1.", "1.1", "1.1.1", "Section 3.2"
        if re.match(r"^(?:Section\s+)?\d+(?:\.\d+)*\.?\s+\S", stripped, re.IGNORECASE):
            headings.append(stripped[:120])  # cap length

        # ALL CAPS lines (likely headings) – minimum 3 word chars
        elif stripped.isupper() and len(stripped.split()) <= 12 and re.search(r"[A-Z]{3,}", stripped):
            headings.append(stripped[:120])

    return headings


def load_pdf(pdf_path: Path) -> List[DocumentPage]:
    """Load a single PDF and return a list of DocumentPage objects."""
    pages = []
    doc_name = pdf_path.name

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"[WARNING] Could not open {doc_name}: {e}")
        return pages

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if not text or not text.strip():
            continue

        headings = extract_headings_from_text(text)

        pages.append(DocumentPage(
            doc_name=doc_name,
            page_num=page_num + 1,  # 1-indexed
            text=text.strip(),
            headings=headings,
        ))

    doc.close()
    return pages


def load_all_documents(docs_dir: Path) -> List[DocumentPage]:
    """
    Load all supported documents from a directory.
    Currently supports: .pdf
    """
    all_pages = []
    supported = [".pdf"]

    if not docs_dir.exists():
        print(f"[ERROR] Documents directory not found: {docs_dir}")
        return all_pages

    files = sorted(
        f for f in docs_dir.iterdir()
        if f.suffix.lower() in supported and not f.name.startswith(".")
    )

    if not files:
        print(f"[WARNING] No supported documents found in {docs_dir}")
        return all_pages

    for f in files:
        print(f"  Loading: {f.name}")
        if f.suffix.lower() == ".pdf":
            pages = load_pdf(f)
            all_pages.extend(pages)

    print(f"  Total pages loaded: {len(all_pages)} from {len(files)} document(s)")
    return all_pages
