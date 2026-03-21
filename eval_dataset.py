"""
PromptAutopsy — Retrieval Evaluation Dataset
=============================================
Tests Precision@K, Recall@K and F1@K for each failure mode.

Structure:
- query: what retrieve() will be called with
- relevant_sources: which sources SHOULD appear in results
- relevant_failure_modes: which failure modes SHOULD appear
- k_values: [3, 5] — we test both

Run with: python eval.py
"""

EVAL_DATA = [

    # ── Vague Instruction ─────────────────────────────────────────────────────
    {
        "id"                    : "vi_001",
        "failure_mode"          : "vague_instruction",
        "query"                 : "how to fix a vague instruction in a prompt",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["vague_instruction"],
        "notes"                 : "Core vague instruction query — Anthropic docs should dominate"
    },
    {
        "id"                    : "vi_002",
        "failure_mode"          : "vague_instruction",
        "query"                 : "how to be specific in LLM instructions",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["vague_instruction"],
        "notes"                 : "Specificity — golden rule from Anthropic docs"
    },
    {
        "id"                    : "vi_003",
        "failure_mode"          : "vague_instruction",
        "query"                 : "prompt has no clear success criterion",
        "relevant_sources"      : ["anthropic_prompting_docs"],
        "relevant_failure_modes": ["vague_instruction"],
        "notes"                 : "Success criterion — directly in Anthropic docs"
    },

    # ── Missing Context ───────────────────────────────────────────────────────
    {
        "id"                    : "mc_001",
        "failure_mode"          : "missing_context",
        "query"                 : "why does context placement matter in prompts",
        "relevant_sources"      : ["lost_in_the_middle", "anthropic_prompting_docs"],
        "relevant_failure_modes": ["missing_context"],
        "notes"                 : "Lost in the Middle paper should be top result"
    },
    {
        "id"                    : "mc_002",
        "failure_mode"          : "missing_context",
        "query"                 : "how to add audience and role to a prompt",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["missing_context"],
        "notes"                 : "Role + audience — Give Claude a role section"
    },
    {
        "id"                    : "mc_003",
        "failure_mode"          : "missing_context",
        "query"                 : "how to write a good system prompt role",
        "relevant_sources"      : ["anthropic_prompting_docs"],
        "relevant_failure_modes": ["missing_context"],
        "notes"                 : "System prompt role — directly in Anthropic docs"
    },

    # ── Wrong Format ──────────────────────────────────────────────────────────
    {
        "id"                    : "wf_001",
        "failure_mode"          : "wrong_format",
        "query"                 : "how to get structured JSON output from an LLM",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["wrong_format"],
        "notes"                 : "Known weak query — documented failure case"
    },
    {
        "id"                    : "wf_002",
        "failure_mode"          : "wrong_format",
        "query"                 : "how to use XML tags to structure a prompt",
        "relevant_sources"      : ["anthropic_prompting_docs"],
        "relevant_failure_modes": ["wrong_format"],
        "notes"                 : "XML tags — dedicated section in Anthropic docs"
    },
    {
        "id"                    : "wf_003",
        "failure_mode"          : "wrong_format",
        "query"                 : "prompt asks for table or list without defining structure",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["wrong_format"],
        "notes"                 : "Format without schema — control format section"
    },

    # ── Conflicting Instructions ───────────────────────────────────────────────
    {
        "id"                    : "ci_001",
        "failure_mode"          : "conflicting_instructions",
        "query"                 : "prompt says be comprehensive but also concise",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["conflicting_instructions"],
        "notes"                 : "Classic conflict — quantify constraints fix"
    },
    {
        "id"                    : "ci_002",
        "failure_mode"          : "conflicting_instructions",
        "query"                 : "how to resolve contradictory instructions in a prompt",
        "relevant_sources"      : ["anthropic_prompting_docs"],
        "relevant_failure_modes": ["conflicting_instructions"],
        "notes"                 : "Contradiction resolution — Anthropic docs"
    },
    {
        "id"                    : "ci_003",
        "failure_mode"          : "conflicting_instructions",
        "query"                 : "tell claude what to do instead of what not to do",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["conflicting_instructions"],
        "notes"                 : "Positive vs negative instructions — both sources"
    },

    # ── Missing Examples ──────────────────────────────────────────────────────
    {
        "id"                    : "me_001",
        "failure_mode"          : "missing_examples",
        "query"                 : "when should I use chain of thought prompting",
        "relevant_sources"      : ["chain_of_thought", "zero_shot_reasoners"],
        "relevant_failure_modes": ["missing_examples", "general"],
        "notes"                 : "CoT paper should dominate — high confidence"
    },
    {
        "id"                    : "me_002",
        "failure_mode"          : "missing_examples",
        "query"                 : "how to add few shot examples to a prompt",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["missing_examples"],
        "notes"                 : "Few-shot — use examples effectively section"
    },
    {
        "id"                    : "me_003",
        "failure_mode"          : "missing_examples",
        "query"                 : "how many examples should I include in a prompt",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["missing_examples"],
        "notes"                 : "3-5 examples rule — directly in Anthropic docs"
    },

    # ── Cross failure mode ────────────────────────────────────────────────────
    {
        "id"                    : "xf_001",
        "failure_mode"          : "general",
        "query"                 : "how to write a better prompt for any task",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["vague_instruction", "missing_context", "general"],
        "notes"                 : "General improvement — should hit multiple sources"
    },
    {
        "id"                    : "xf_002",
        "failure_mode"          : "general",
        "query"                 : "prompt engineering best practices",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["general"],
        "notes"                 : "Broad query — Anthropic docs should lead"
    },
    {
        "id"                    : "xf_003",
        "failure_mode"          : "general",
        "query"                 : "why does my LLM give inconsistent outputs",
        "relevant_sources"      : ["anthropic_prompting_docs", "dair_prompting_advanced"],
        "relevant_failure_modes": ["vague_instruction", "missing_context"],
        "notes"                 : "Inconsistency — maps to missing context and vague instruction"
    },
]

# ── Precision@K calculation ───────────────────────────────────────────────────

def precision_at_k(retrieved_sources: list, relevant_sources: list, k: int) -> float:
    """
    Of the top-k retrieved chunks, what fraction
    came from a relevant source?
    """
    top_k   = retrieved_sources[:k]
    hits    = sum(1 for s in top_k if s in relevant_sources)
    return hits / k


def recall_at_k(retrieved_sources: list, relevant_sources: list, k: int) -> float:
    """
    Of all relevant sources, what fraction appeared
    in the top-k retrieved chunks?
    """
    top_k           = retrieved_sources[:k]
    retrieved_set   = set(top_k)
    relevant_set    = set(relevant_sources)
    hits            = len(retrieved_set & relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0


def f1_at_k(precision: float, recall: float) -> float:
    """
    Harmonic mean of precision and recall.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ── Failure mode precision ────────────────────────────────────────────────────

def failure_mode_precision(
    retrieved_modes: list,
    relevant_modes: list,
    k: int
) -> float:
    """
    Of the top-k retrieved chunks, what fraction
    had the correct failure mode tag?
    """
    top_k   = retrieved_modes[:k]
    hits    = sum(1 for m in top_k if m in relevant_modes)
    return hits / k


if __name__ == "__main__":
    print(f"Eval dataset loaded: {len(EVAL_DATA)} queries")
    print(f"Failure modes covered:")
    modes = {}
    for item in EVAL_DATA:
        m = item["failure_mode"]
        modes[m] = modes.get(m, 0) + 1
    for mode, count in modes.items():
        print(f"  {mode}: {count} queries")