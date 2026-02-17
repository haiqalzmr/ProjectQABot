# Design Notes — Policy Q&A Bot

## 1. Chunking Strategy

### Approach: Section-Aware Chunking with Metadata

Insurance policy PDFs have a well-defined hierarchical structure (sections, sub-sections, clauses). Our chunking strategy exploits this:

1. **Heading Detection** — We detect section boundaries using:
   - Numbered patterns: `1.`, `1.1`, `Section 3.2`
   - ALL-CAPS headings: `DEFINITIONS`, `EXCLUSIONS`
   - Known insurance keywords: "Coverage", "Conditions", "Endorsements"

2. **Hierarchical Heading Path** — Track a heading stack per-document so each chunk knows its full path, e.g., `"General Conditions > Claims > How to Claim"`. This powers richer citations.

3. **Token-Based Sub-Chunking** — After splitting at section boundaries, large sections are further chunked to 400–800 tokens with ~100-token overlap. Splits occur at sentence boundaries to avoid cutting mid-sentence.

4. **Rich Metadata** — Each chunk carries:
   ```json
   {
     "doc_name": "QM8698-1124 QBE Contents Insurance PDS.pdf",
     "section": "Exclusions",
     "clause_number": "3.2",
     "page": 12,
     "heading_path": "Part 2 > Contents Cover > Exclusions",
     "cross_references": ["4.1", "5.3"]
   }
   ```

### Why Not Fixed-Size Chunking?

Fixed-size chunking (e.g., every 500 tokens) would cut across section and clause boundaries, losing the structural context that makes insurance documents navigable. Section-aware chunking preserves the semantic units that users ask about.

---

## 2. Prompt Design

### System Prompt Philosophy

The system prompt enforces strict grounding:

- **Answer ONLY from context** — prevents hallucination
- **Cite specific sources** — doc name, section, clause, page
- **"I cannot find a definitive answer"** — explicit instruction for low-confidence responses
- **Use exact policy terminology** — avoids paraphrasing that could change meaning

### Prompt Structure

```
[System Prompt: strict grounding rules]
[Context: numbered source blocks with metadata headers]
[Question]
[Instructions: answer format, citation requirements, no-answer behavior]
```

Each context source is labeled:
```
[Source 1: QBE Contents Insurance PDS.pdf §3.2 (Exclusions), p.12]
<chunk text>
```

This labeling allows the LLM (or mock backend) to directly reference sources in its answer.

---

## 3. Negative-Question / No-Answer Handling

### Three-Tier Confidence System

1. **High confidence (score ≥ 0.5)** — Strong semantic match; produce grounded answer with citations
2. **Medium confidence (0.25–0.5)** — Partial match; answer with caveats, cite closest clauses
3. **Low confidence (< 0.25)** — Poor match; trigger explicit no-answer response

### No-Answer Response Format

When triggered, the response:
1. States: "I cannot find a definitive answer in the provided policy wording."
2. Explains the ambiguity briefly
3. Lists the closest related clauses found (with snippets)
4. Still provides citation references

### Why This Matters

In insurance, a wrong answer is worse than no answer. The system errs on the side of caution: if the retrieved context doesn't strongly match, it says so explicitly rather than guessing.

---

## 4. Embedding Model Choice

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

| Factor | Detail |
|--------|--------|
| Size | ~80 MB (vs 400MB+ for larger models) |
| Dimension | 384 |
| Speed | ~3000 sentences/sec on CPU |
| Quality | Competitive retrieval accuracy for domain-specific search |

Embeddings are normalized for cosine similarity via FAISS `IndexFlatIP` (inner product = cosine when vectors are unit-normalized).

**Batched encoding** (batch_size=16) prevents memory issues on machines with limited RAM.

---

## 5. Citation Format Design

Citations follow the rubric's required format:

```
Sources: Policy_A.pdf §3.2 (Exclusions), p.12; Policy_B.pdf "Definitions > Accident", p.4
```

Components:
- **Document name** — file basename
- **Clause number** — extracted from numbered section headings (if present)
- **Section label** — from detected heading or known keywords
- **Page number** — 1-indexed from PDF extraction

Citations are deduplicated to avoid repetition when multiple chunks from the same source are retrieved.

---

## 6. LLM Backend Architecture

The system uses a **pluggable backend pattern**:

```
LLMBackend (abstract)
├── MockLLMBackend      ← Default: rule-based extraction, no ML model
├── TransformersBackend ← Optional: local HuggingFace model
└── OpenAIBackend       ← Optional: API-based (stub)
```

The **MockLLMBackend** works by:
1. Extracting the question from the prompt
2. Scoring sentences by keyword overlap with the question
3. Selecting the most relevant sentences
4. Formatting them with the standard citation block

This approach requires zero additional model loading and demonstrates the full pipeline architecture without API dependencies.
