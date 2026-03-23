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
infer_intent()            → intent + confidence score (NEW in V2)
    ↓
diagnose_prompt()         → 5 failure modes scored with severity
    ↓
estimate_delta()          → pre-rewrite verbosity estimate (NEW in V2)
    ↓
retrieve_best_practices() → RAG retrieval from 7 curated sources
    ↓
build_fix_plan()          → Executable transformation actions
    ↓
rewrite_prompt()          → Dual output: Ready-to-Use + Editable Template
                            Adaptive verbosity based on delta (NEW in V2)
                            Confidence-based variant count (NEW in V2)
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

## 🧠 V2 Features

### Intent Inference
Every prompt is analysed for intent before diagnosis. The system infers:
- Primary intent and confidence score (0.0–1.0)
- 2 alternative interpretations
- Topic, format, and target audience

```python
{
  "intent"      : "cold outreach email",
  "confidence"  : 0.85,
  "alternatives": ["nurture sequence", "onboarding email"],
  "topic"       : "SaaS marketing",
  "format"      : "email",
  "audience"    : "B2B decision makers"
}
```

### Adaptive Verbosity
Rewrite depth scales to how broken the prompt actually is:

| Delta | Severity | Output |
|---|---|---|
| 2–4 | Light | Minimal targeted edits — no XML structure |
| 5–8 | Moderate | Role + context block + constraints |
| 9+ | Full | Complete reconstruction — role, context, instructions, examples |

### Confidence-Based Routing
Number of output variants scales to intent confidence:

| Confidence | Variants |
|---|---|
| ≥ 0.8 | Single rewrite — intent is clear |
| 0.5–0.8 | 2 intent variants |
| < 0.5 | 3 intent variants — maximum ambiguity |

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

**Failure mode precision is lower (0.46)** because V1 uses keyword heuristics for chunk tagging. V2.2 will replace this with an LLM classifier (estimated improvement to ~0.85).

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

**✅ Ready to Use** — all fields filled with best-guess values based on context. Paste directly into ChatGPT or Claude. Zero editing required.

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
├── tools.py          → 7 LangChain tools (classify → infer_intent → diagnose → retrieve → fix → rewrite → score)
├── agent.py          → Sequential RAG pipeline with adaptive verbosity
├── eval_dataset.py   → 18 eval queries with ground truth
├── eval.py           → Precision@K, Recall@K, F1@K evaluation
├── app.py            → Streamlit UI with dual output tabs
└── raw_docs/         → Scraped knowledge base (gitignored)
```

---

## 🗺️ V2 Roadmap (Parked)

Documented and ready to implement after next two projects:

- **Deterministic prompt assembly** — programmatic builder replaces LLM rewrite step
- **LLM classifier for chunk tagging** — replaces keyword heuristics (FM@K 0.46 → ~0.85)
- **3-5 diverse examples** — dynamically generated instead of 1 static example
- **One-click execution** — run the improved prompt and show output preview
- **Feedback loop** — thumbs up/down stored in SQLite for index improvement

---

## 💬 

**What's unique about this system?**
> "I moved from unstructured prompt rewriting to a compiler-style architecture where prompts are decomposed into role, context, instructions, and examples, and reconstructed deterministically using RAG-grounded transformation rules."

**Why RAG instead of just prompting?**
> "Every fix is cited from a retrieved source — Anthropic's prompting guidelines, peer-reviewed NLP papers, or community guides. The rewrite is grounded in evidence, not hallucination."

**What did you measure?**
> "I built an 18-query evaluation set across all 5 failure modes and measured Precision@K, Recall@K, and F1@K. Source precision at K=3 is 0.80. I also documented known failure cases honestly — which matters as much as the wins."

**What's new in V2?**
> "V2 added intent inference with confidence scoring, adaptive verbosity that scales rewrite depth to how broken the prompt actually is, and confidence-based routing that generates 1-3 output variants depending on how ambiguous the intent is."

---

## 📄 License

MIT