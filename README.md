# Policy Q&A Bot with Grounded Citations

An AI-powered Q&A assistant that answers questions about insurance policy documents using a Retrieval-Augmented Generation (RAG) pipeline. Answers are **strictly grounded** in the source documents with **clause-level citations**.

## Features

- **Section-aware chunking** — preserves headings, clause numbers, and cross-references
- **Semantic search** — SentenceTransformers (`all-MiniLM-L6-v2`) + FAISS vector store
- **Grounded answers** — every claim is backed by specific document/section/page citations
- **No-answer behavior** — clearly states when information isn't found and lists related clauses
- **Pluggable LLM backends** — Mock (default), HuggingFace Transformers, OpenAI (stub)
- **Modern web interface** — collapsible sidebar, chat history, light/dark theme toggle, citation cards, and contextual follow-up suggestions
- **Conversational awareness** — handles greetings, identity questions, and help requests without triggering the RAG pipeline

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Policy Documents

Place PDF files in the `data/` directory (3 QBE policy PDFs are included).

### 3. Run the Web UI

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 4. CLI Mode

```bash
# Single question
python app.py --question "Is wear and tear covered?"

# Interactive mode
python app.py --interactive

# Force rebuild index
python app.py --rebuild
```

## Project Structure

```
ProjectQABot/
├── app.py                     # Entry point (CLI + web)
├── config.py                  # Configuration
├── requirements.txt
├── data/                      # Policy PDFs
├── core/                      # RAG pipeline
│   ├── ingestion.py           # PDF loading (PyMuPDF)
│   ├── chunking.py            # Section-aware chunking
│   ├── embeddings.py          # SentenceTransformers embeddings
│   ├── vectorstore.py         # FAISS vector store
│   ├── retriever.py           # Retrieval + citations
│   ├── llm_backend.py         # Pluggable LLM backends
│   ├── prompts.py             # Prompt templates
│   └── pipeline.py            # Orchestration
├── frontend/                  # Web UI
│   ├── server.py              # Flask API
│   ├── templates/index.html   # Main interface
│   └── static/                # CSS, JS, and assets
├── tests/                     # Test suite
│   ├── test_cases.py          # 10 Q&A examples
│   └── test_qa.py             # Pytest integration
└── vector_db/                 # Auto-created FAISS index
```

## Architecture

```
User Question → Embedding (SentenceTransformers)
                    ↓
             FAISS Search (top-k)
                    ↓
             Retriever (filter + deduplicate)
                    ↓
             Prompt Builder (context + instructions)
                    ↓
             LLM Backend (generate grounded answer)
                    ↓
             Answer + Citations
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- Chunking quality and metadata
- In-domain retrieval + citations (5 questions)
- Out-of-scope no-answer behavior (2 questions)
- Near-miss handling (3 questions)
- Full pipeline integration

## Configuration

Edit `config.py` to adjust:
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking parameters
- `TOP_K` — number of retrieved chunks
- `SIMILARITY_THRESHOLD` — minimum similarity to include
- `LLM_BACKEND` — switch between "mock", "transformers", "openai"
- `EMBEDDING_MODEL` — SentenceTransformers model name

## LLM Backend Options

| Backend | Command | Requirements |
|---------|---------|-------------|
| Mock (default) | `LLM_BACKEND = "mock"` | None |
| HuggingFace | `LLM_BACKEND = "transformers"` | `transformers` package |
| OpenAI | `LLM_BACKEND = "openai"` | `openai` package + API key |
