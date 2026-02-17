"""
Test cases for the Policy Q&A Bot.
10 Q&A examples: 5 in-domain, 3 near-miss, 2 out-of-scope.

Each test case defines:
  - question: The question to ask
  - type: "in_domain" | "near_miss" | "out_of_scope"
  - description: What this test validates
  - expected_keywords: Keywords expected in the answer (for in-domain)
  - expected_docs: Which documents should be cited
  - should_have_answer: Whether a definitive answer is expected
"""

TEST_CASES = [
    # ── In-Domain (5) ─ Questions with clear answers in the policies ──────

    {
        "question": "Is wear-and-tear covered under the standard policy?",
        "type": "in_domain",
        "description": "Tests retrieval of exclusion clauses — wear and tear is typically excluded",
        "expected_keywords": ["wear", "tear", "exclu", "not covered"],
        "expected_docs": [],  # any policy doc is acceptable
        "should_have_answer": True,
    },
    {
        "question": "What is the waiting period for accidental damage?",
        "type": "in_domain",
        "description": "Tests retrieval of conditions/waiting period clauses",
        "expected_keywords": ["wait", "period", "day", "damage"],
        "expected_docs": [],
        "should_have_answer": True,
    },
    {
        "question": "How does the excess or deductible apply to water damage claims?",
        "type": "in_domain",
        "description": "Tests retrieval of excess/deductible information for water damage",
        "expected_keywords": ["excess", "deductible", "water", "claim"],
        "expected_docs": [],
        "should_have_answer": True,
    },
    {
        "question": "What definitions apply to 'Insured Person' in the policy?",
        "type": "in_domain",
        "description": "Tests retrieval from Definitions section",
        "expected_keywords": ["insured", "person", "defin"],
        "expected_docs": [],
        "should_have_answer": True,
    },
    {
        "question": "What are the general exclusions listed in the policy?",
        "type": "in_domain",
        "description": "Tests retrieval of general exclusions section",
        "expected_keywords": ["exclusion", "not", "cover"],
        "expected_docs": [],
        "should_have_answer": True,
    },

    # ── Near-Miss (3) ─ Related but not directly answerable ──────────────

    {
        "question": "Does the policy cover damage caused by my pet?",
        "type": "near_miss",
        "description": "Pet damage may or may not be explicitly addressed — tests nuanced retrieval",
        "expected_keywords": ["pet", "animal", "damage"],
        "expected_docs": [],
        "should_have_answer": None,  # might or might not have a definitive answer
    },
    {
        "question": "Am I covered if I accidentally damage a neighbour's property?",
        "type": "near_miss",
        "description": "Liability coverage may exist but might not be the focus of the home policy",
        "expected_keywords": ["liab", "neighbour", "damage", "third"],
        "expected_docs": [],
        "should_have_answer": None,
    },
    {
        "question": "What happens if I miss a premium payment?",
        "type": "near_miss",
        "description": "Premium payment terms may be in conditions but might not be detailed",
        "expected_keywords": ["premium", "payment", "cancel"],
        "expected_docs": [],
        "should_have_answer": None,
    },

    # ── Out-of-Scope (2) ─ Completely unrelated questions ─────────────────

    {
        "question": "What is the capital of France?",
        "type": "out_of_scope",
        "description": "Completely unrelated — should trigger no-answer response",
        "expected_keywords": [],
        "expected_docs": [],
        "should_have_answer": False,
    },
    {
        "question": "Can you write me a Python function to sort a list?",
        "type": "out_of_scope",
        "description": "Programming question — should trigger no-answer response",
        "expected_keywords": [],
        "expected_docs": [],
        "should_have_answer": False,
    },
]
