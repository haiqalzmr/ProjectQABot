"""
Central configuration for the Policy Q&A Bot.
All tuneable parameters in one place.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

# ── Embedding ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # SentenceTransformers model
EMBEDDING_BATCH_SIZE = 16                      # keep small to avoid OOM
EMBEDDING_DIMENSION = 384                      # dimension for all-MiniLM-L6-v2

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 500          # target tokens per chunk
CHUNK_OVERLAP = 100       # overlap tokens between chunks
MIN_CHUNK_SIZE = 80       # discard tiny fragments

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K = 5                          # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.25       # minimum cosine similarity to include

# ── LLM Backend ───────────────────────────────────────────────────────────
LLM_BACKEND = "mock"               # "mock" | "transformers" | "openai"

# ── Web Server ────────────────────────────────────────────────────────────
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
WEB_DEBUG = False
