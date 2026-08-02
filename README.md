# Enterprise RAG

A production-style RAG system I'm building as a portfolio project — not a toy chatbot demo. The goal is to show how I'd actually design retrieval for messy enterprise documents: multiple formats, noisy sources, agentic routing, guardrails, and measurable quality.

Backend is FastAPI. Frontend is Next.js. Vectors live in Qdrant Cloud. Embeddings go through OpenAI; inference is primarily Groq (with OpenAI as a fallback option depending on the path). Observability is wired through Logfire (and Langfuse keys are in the config for tracing later).

---

## What problem this solves

Most RAG tutorials index a clean PDF and call it a day. Real orgs don't work like that. You get HTML docs, Word files, slide decks, scraped pages, duplicates, and half-broken encodings sitting next to the "good" material.

So this repo is built around that reality:

- **`DATA/true_data/`** — cleaner, curated-style sources (the stuff you'd trust)
- **`DATA/noisy_data/`** — messier corpus (PDFs, HTML, PPTX, odd encoding, etc.)

The point isn't just "can the model answer." It's whether the pipeline can ingest both worlds, retrieve well, refuse bad asks, and still be evaluable.

---

## Strategies / design choices

### 1. Multi-format ingestion first
Loaders are extension-routed (`load_document(path)`):

| Type | Tools |
|------|--------|
| Text | native read + encoding fallback |
| HTML | BeautifulSoup + lxml (strip scripts/styles) |
| PDF | pdfplumber, fall back to pypdf |
| Office | python-docx / python-pptx |

Everything normalizes into a shared `LoadedDocument` (text + metadata). Logfire spans wrap each load so I can see failures and empty extracts in the dashboard.

Chunking is paragraph-aware: empty blocks are dropped, paragraphs are packed into ~1000–1500 character windows, and a small overlap is kept between chunks so answers that sit on a boundary don't disappear from both sides.

### 2. Embeddings ≠ inference
- **OpenAI** `text-embedding-3-small` for vectors (stable, 1536-dim)
- Batched at **50** texts/request with **4** exponential retries (1s → 8s); local sentence-transformers fallback only if OpenAI still fails
- **Groq** for fast chat / agent responses (with a fallback Groq key in config)
- Same OpenAI chat model available when I want the "main" path on OpenAI instead

Keeping these separate makes it easier to swap cost/latency without rewriting the whole stack.

### 3. Agentic graph (LangGraph)
Planned query path isn't a single retrieve→stuff→generate hop. The graph is sketched as:

`planner → retriever → responder`

Planner decides how to approach the question; retriever hits Qdrant (+ FlashRank reranking); responder drafts the answer. State lives in `AgentState`. This is the structure under `app/agents/` — graph wiring is the next big chunk after ingestion.

### 4. Guardrails before "trust me bro"
NeMo Guardrails (Colang) sits in `app/guardrails/` so unsafe / off-policy prompts don't silently sail through. Evals for rails live under `evals/guardrails_eval.py`.

### 5. Gateway via Portkey
Portkey is the optional LLM gateway (`app/gateway/`) — useful for routing, fallbacks, and not hard-coding every provider call in business logic.

### 6. Evaluate, don't just vibe-check
`evals/` is set up for:

- a golden dataset
- RAGAS metrics
- a pipeline runner
- optional Streamlit UI for local evals only (production UI is Next.js)

If retrieval quality isn't measured, it's marketing.

### 7. Observability as a default
Logfire is configured (project credentials under `.logfire/`, ignored by git). Loader spans already ship. FastAPI instrumentation and Langfuse hooks are part of the longer plan.

---

## Repo layout

```
enterprise-rag/
├── app/                      # FastAPI backend
│   ├── main.py               # /health /ready /metrics /graph /query
│   ├── config.py             # pydantic-settings from .env
│   ├── agents/               # LangGraph state, graph, nodes
│   ├── ingestion/
│   │   ├── loaders/          # text, html, pdf, office  ✅ implemented
│   │   ├── chunking/         # paragraph-aware splitter (~1–1.5k + overlap) ✅
│   │   └── processor.py      # CLI + load→chunk→embed→upsert ✅
│   ├── services/retrieval/   # embeddings ✅, Qdrant upsert ✅, FlashRank
│   ├── guardrails/           # NeMo rails
│   └── gateway/              # Portkey client
├── evals/                    # RAGAS + golden set + guardrail checks
├── web/                      # Next.js App Router UI
├── DATA/true_data/
├── DATA/noisy_data/
├── processed_data/
├── docs/
├── tests/
├── .env.example
├── requirements.txt
├── pyproject.toml            # uv-managed deps
└── Dockerfile
```

---

## Status (honest)

| Area | Status |
|------|--------|
| Project layout + config | Done |
| Document loaders + Logfire spans | Done |
| Paragraph-aware chunking (+ overlap) | Done |
| OpenAI embeddings (batch 50, 4 retries) | Done |
| Processor + Qdrant upsert (CLI) | Done |
| FlashRank reranker | Next |
| Embeddings + retrieval + rerank | Embeddings + Qdrant upsert done; rerank next |
| LangGraph query path | Scaffolded |
| Guardrails + Portkey | Scaffolded |
| RAGAS evals | Scaffolded |
| Next.js chat UI | Bootstrap (Next 16) |

I'm building this layer by layer on purpose — ingestion that actually works on my datasets before pretending the chat endpoint is "enterprise ready."

---

## Stack

**Backend:** Python 3.11, FastAPI, uvicorn, uv  
**RAG / agents:** LangGraph, LangChain, Qdrant, FlashRank, sentence-transformers  
**Models:** OpenAI embeddings (+ optional chat), Groq inference  
**Safety / gateway:** NeMo Guardrails, Portkey  
**Evals:** RAGAS  
**Obs:** Logfire, Langfuse (keys ready)  
**Frontend:** Next.js 16, React 19, TypeScript, Tailwind  
**Packaging:** `pyproject.toml` + `uv.lock`, `requirements.txt` export, Dockerfile

---

## Quick start

### 1. Clone & env
```bash
git clone <your-repo-url>
cd Enterprise_RAG
cp .env.example .env
# fill OPENAI_API_KEY, GROQ_API_KEY, QDRANT_*, etc.
```

### 2. Backend
```bash
uv sync

# Windows: prefer module form (avoids a uv trampoline bug)
uv run python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

Activate the venv manually if you want:
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Frontend (separate terminal)
```bash
cd web
npm install
npm run dev
```

UI: http://localhost:3000

### 4. Try a loader
```bash
uv run python app/scripts/smoke_loaders.py
```

Or in code:
```python
from app.ingestion.loaders import load_document

doc = load_document("DATA/true_data/cronjobs.docx")
print(doc.doc_type, doc.char_count, doc.metadata)
```

### 5. Ingest into Qdrant
```bash
# single file
uv run python -m app.ingestion.processor --file DATA/true_data/parallel_work_queue.txt --corpus true

# whole folder
uv run python -m app.ingestion.processor --dir DATA/true_data --corpus true

# both true_data + noisy_data
uv run python -m app.ingestion.processor --universal

# dry run (no Qdrant write)
uv run python -m app.ingestion.processor --file DATA/true_data/cronjobs.docx --no-upsert
```

---

## Config

Settings live in `app/config.py` and read the root `.env`. Important knobs:

- `OPENAI_*` / `EMBEDDING_DIMENSIONS`
- `GROQ_*` / `GROQ_FALLBACK_API_KEY` / `JUDGE_GROQ_API_KEY`
- `QDRANT_CLUSTER_ENDPOINT` or `QDRANT_URL` + `QDRANT_API_KEY` + `QDRANT_COLLECTION`
- `PORTKEY_*`, `LANGFUSE_*`, `LOGFIRE_TOKEN` (optional if `.logfire/` credentials exist)

Never commit `.env` or `.logfire/`.

---

## Why this belongs in a portfolio

I'm not optimizing for "another LangChain hello world." I'm optimizing for decisions you have to defend in a review:

- separate clean vs noisy corpora
- provider split (embed vs generate)
- agent graph instead of a single prompt
- guardrails + evals as first-class folders
- observability from day one on ingestion

If you're reading this on GitHub and want to poke at something specific (retrieval quality, chunking strategy, UI), open an issue or just star it and stalk the commits — that's where the real progress shows up.

---

## Author

**Mirza Shahbaz Ali Baig**  
Backend / AI systems work — building this in public as the stack comes together.
