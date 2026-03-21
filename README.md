# 🔬 PromptAutopsy

> **A compiler-style RAG system that diagnoses broken LLM prompts and reconstructs them using grounded transformation rules.**

PromptAutopsy is not a prompt improver — it's a prompt compiler. It decomposes your prompt into failure modes, retrieves evidence-backed fixes from a curated knowledge base, and reconstructs a production-grade prompt with role, context, instructions, and examples — all cited from source.

---

## 🎯 The Problem

Most people write prompts like this:

```
Write something about climate change
```

And get wildly inconsistent outputs. The root cause isn't the LLM — it's the prompt. PromptAutopsy acts as a pre-flight checklist before you submit to any LLM.

---

## 🏗️ Architecture

```
User Input
    ↓
classify_input()          → 11 input types (valid, keyword_dump, harmful, etc.)
    ↓
diagnose_prompt()         → 5 failure modes scored with severity
    ↓
retrieve_best_practices() → RAG retrieval from 7 curated sources
    ↓
build_fix_plan()          → Executable transformation actions
    ↓
rewrite_prompt()          → Dual output: Ready-to-Use + Editable Template
    ↓
score_improvement()       → LLM-as-Judge: before/after scores across 3 dimensions
```

> *"I moved from unstructured prompt rewriting to a compiler-style architecture where prompts are decomposed into role, context, instructions, and examples, and reconstructed deterministically using RAG-grounded transformation rules."*

---

## 🔍 5 Failure Modes Detected

| Failure Mode | Severity | Description |
|---|---|---|
| `vague_instruction` | 🔴 Critical | No format, length, tone, or success criteria defined |
| `missing_context` | 🔴 Critical | No role, audience, or purpose specified |
| `wrong_format` | 🟠 High | Structured output requested without schema |
| `conflicting_instructions` | 🟡 Medium | Contradictory instructions the LLM cannot resolve |
| `missing_examples` | 🟢 Low | Pattern-sensitive task with no few-shot examples |

---

## 📚 Knowledge Base (RAG Index)

| Source | Type | Focus |
|---|---|---|
| Anthropic Prompting Best Practices | Official docs | Core prompting principles |
| Chain-of-Thought Prompting (Wei et al.) | arXiv paper | Reasoning and examples |
| Zero-Shot Reasoners (Kojima et al.) | arXiv paper | Zero-shot techniques |
| Lost in the Middle (Liu et al.) | arXiv paper | Context placement |
| DAIR Prompt Engineering Guide (Advanced) | Community guide | Advanced techniques |
| DAIR Prompt Engineering Guide (ChatGPT) | Community guide | Applied techniques |

**Index stats:** 955 chunks · chunk size 512 · overlap 64 · MMR reranking

---

## 📊 Retrieval Evaluation Results

Evaluated across 18 queries covering all 5 failure modes:

| Metric | K=3 | K=5 |
|---|---|---|
| Source Precision | **0.80** | 0.78 |
| Source Recall | 0.72 | 0.83 |
| Source F1 | **0.73** | 0.80 |
| Failure Mode Precision | 0.46 | 0.47 |

**Failure mode precision is lower (0.46)** because V1 uses keyword heuristics for chunk tagging. V2 will replace this with an LLM classifier (estimated improvement to ~0.85).

**Known weak queries:** JSON structured output (scores < 0.60), inconsistent LLM outputs (no relevant source in index).

---

## 🔀 11 Edge Cases Handled

| Input Type | Example | Behaviour |
|---|---|---|
| `valid_prompt` | "Write about climate change" | Full pipeline → dual output |
| `keyword_dump` | "marketing email SaaS B2B" | 2-3 intent variants generated |
| `too_short` | "summarize" | Stopped + partial intent hint |
| `meta_question` | "How do I write a good prompt?" | Stopped + explanation |
| `harmful` | "How to make a bomb" | Hard reject |
| `non_english` | "Bonjour, écris un article" | Stopped + English only message |
| `has_placeholders` | "Write about [TOPIC] for [AUDIENCE]" | Placeholders preserved + filled |
| `contains_code` | "def sort_list(lst): pass" | Code preserved + instructions rewritten |
| `too_long` | 500+ words | Summarised + single rewrite |
| `not_a_prompt` | "yo" | Stopped |
| `empty` | "" | Stopped |

---

## ✍️ Dual Output

Every rewrite produces two versions:

**✅ Ready to Use** — all fields filled with best-guess values based on context. Paste directly into ChatGPT or Claude.

**🧩 Editable Template** — structured template with `[REQUIRED: field]`, `Default: value`, and `Other options:` for full control.

---

## 📈 Improvement Scoring (LLM-as-Judge)

Each rewrite is scored before and after across 3 dimensions (1-5 scale):

| Dimension | Measures |
|---|---|
| Clarity | How unambiguous is the task? |
| Specificity | How precisely are constraints defined? |
| Completeness | How much of the necessary context is present? |

`show_rewrite = True` only when `overall_delta ≥ 2` — the system won't suggest a rewrite for prompts that are already well-structured.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector DB | ChromaDB (local persistent) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Indexing | LlamaIndex `VectorStoreIndex` |
| Reranking | Manual MMR implementation |
| LLM | Anthropic Claude Sonnet (via API) |
| Agent | LangChain tools + direct pipeline |
| UI | Streamlit |

---

## 🚀 Setup

```bash
# Clone
git clone https://github.com/agentbharath/promptAutopsy.git
cd promptAutopsy

# Install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY and OPENAI_API_KEY

# Scrape knowledge sources
python scraper.py

# Build the index
python ingest.py

# Run the app
streamlit run app.py
```

---

## 📁 File Structure

```
promptAutopsy/
├── scraper.py        → Scrapes 7 knowledge sources into raw_docs/
├── ingest.py         → Chunks + embeds + indexes into ChromaDB
├── retrieve.py       → MMR retrieval with caching
├── tools.py          → 6 LangChain tools (classify → diagnose → retrieve → fix → rewrite → score)
├── agent.py          → Sequential RAG pipeline
├── eval_dataset.py   → 18 eval queries with ground truth
├── eval.py           → Precision@K, Recall@K, F1@K evaluation
├── app.py            → Streamlit UI
└── raw_docs/         → Scraped knowledge base (gitignored)
```

---

## 🗺️ V2 Roadmap

Key upgrades planned:

- **Deterministic prompt assembly** — programmatic builder replaces LLM rewrite step
- **LLM classifier for chunk tagging** — replaces keyword heuristics (FM@K 0.46 → ~0.85)
- **Adaptive verbosity** — output complexity scales with score delta
- **One-click execution** — run the improved prompt and show output preview
- **infer_intent() tool** — confidence-based routing to single vs multi-variant output
- **Feedback loop** — thumbs up/down rating stored in SQLite for index improvement

---

## 💬 Interview Talking Points

**What's unique about your system?**
> "I moved from unstructured prompt rewriting to a compiler-style architecture where prompts are decomposed into role, context, instructions, and examples, and reconstructed deterministically using RAG-grounded transformation rules."

**Why RAG instead of just prompting?**
> "Every fix is cited from a retrieved source — Anthropic's prompting guidelines, peer-reviewed NLP papers, or community guides. The rewrite is grounded in evidence, not hallucination."

**What did you measure?**
> "I built an 18-query evaluation set across all 5 failure modes and measured Precision@K, Recall@K, and F1@K. Source precision at K=3 is 0.80. I also documented known failure cases honestly — which matters as much as the wins."

---

## 📄 License

MIT
