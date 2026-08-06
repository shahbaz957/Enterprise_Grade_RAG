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
- **Jina reranker** (`jina-reranker-v2-base-multilingual`) re-orders Qdrant candidates before they hit the LLM
- **Groq** for fast chat / agent responses (with a fallback Groq key in config)
- Same OpenAI chat model available when I want the "main" path on OpenAI instead

Keeping these separate makes it easier to swap cost/latency without rewriting the whole stack.

### 3. Agentic graph (LangGraph)
Planned query path isn't a single retrieve→stuff→generate hop. The graph is sketched as:

`planner → retriever → responder`

Planner decides how to approach the question; retriever hits Qdrant (+ FlashRank reranking); responder drafts the answer. State lives in `AgentState`. This is the structure under `app/agents/` — graph wiring is the next big chunk after ingestion.

### 4. Guardrails before "trust me bro"
NeMo Guardrails (Colang 1.x) sits in `app/guardrails/`. Input rails + dialog Colang run **before** the LangGraph invoke; output rails run on `final_answer`.

**Setup (no NVIDIA account):**
1. `uv sync` (includes `nemoguardrails[sdd,jailbreak]`)
2. Reuse existing `GROQ_API_KEY` or `OPENAI_API_KEY` — no new NeMo keys
3. Optional Presidio spaCy model (richer PII): `uv run python -m spacy download en_core_web_lg`
   Phone/email blocking also works via a built-in regex rail without spaCy.
4. Keep `GUARDRAILS_ENABLED=true` in `.env` (see `.env.example`)

Config lives at `app/guardrails/config/` (`config.yml` + `*.co`). FastEmbed downloads `all-MiniLM-L6-v2` on first run.

Acceptance:
```bash
uv run python -m app.scripts.test_guardrails
# or
uv run python -m evals.guardrails_eval --unit-only
```

Mentor note: NeMo can be bypassed with clever prompting — keep output rails, Portkey gateway policies, and this eval suite. For stricter enterprise deployments, prefer Bedrock native rails or a hybrid.

### 5. Gateway via Portkey
Portkey sits in [`app/gateway/`](app/gateway/) as the LLM gateway for agent chat (planner / responder).

**What you get:** virtual keys (no raw provider keys in call path), primary→fallback routing, optional load balance, retries, timeouts, simple/semantic cache, and per-request metadata (`user_id`, `route`, `environment`, `feature`) for logs/cost.

**Setup:**
1. Create a Portkey API key + Virtual Keys (Groq primary, OpenAI fallback) at https://app.portkey.ai  
2. Copy into `.env` (see `.env.example`):
   - `PORTKEY_API_KEY`
   - `PORTKEY_VIRTUAL_KEY_PRIMARY` (Groq slug)
   - `PORTKEY_VIRTUAL_KEY_FALLBACK` (OpenAI slug)
   - `PORTKEY_CACHE_MODE=simple` (or `semantic` / `off`)
   - `PORTKEY_STRATEGY=fallback` (or `loadbalance`)
3. **Saved Config ID (required if your org has `block_inline_config`):**  
   ```bash
   uv run python -m app.scripts.print_portkey_config
   ```  
   Paste the JSON into Portkey → Configs → Create, then set `PORTKEY_CONFIG_ID=pc-...`.  
   Keep `PORTKEY_ALLOW_INLINE_CONFIG=false` (default). Without a Config ID the app uses a single virtual key only (no automatic fallback/cache in the request).

```bash
# print status + config JSON for the dashboard
uv run python -m app.scripts.print_portkey_config
uv run python -m app.scripts.test_portkey

# live call (and optional cache timing)
uv run python -m app.scripts.test_portkey --live --cache
```

**Fault-tolerance check:** with `PORTKEY_CONFIG_ID` set, break/revoke the primary virtual key → next agent call should hit the fallback VK (visible in Portkey Logs).  
If Portkey is unset (or blocks inline config with no Config ID), the app falls back to direct Groq/OpenAI when those keys are present.

### 6. Evaluate, don't just vibe-check (RAGAS suite)

`evals/` is a runnable suite — not slides:

| Step | Module | What it does |
|------|--------|----------------|
| Golden set | [`evals/golden_dataset.json`](evals/golden_dataset.json) | 15 K8s/RAG Q&A + 6 guardrail cases (`should_block`, `expected_tools`, `reference`) |
| Live pipeline | [`evals/pipeline.py`](evals/pipeline.py) | `POST /query` → stores `actual_response`, `contexts`, `tools` |
| RAGAS metrics | [`evals/metrics.py`](evals/metrics.py) | Faithfulness, Answer Relevancy, Context Precision/Recall, Answer Correctness + tool Jaccard |
| Guardrails matrix | [`evals/guardrails_eval.py`](evals/guardrails_eval.py) | TP/TN/FP/FN for `should_block` vs blocked |

**Prereqs:** API up, Qdrant ingested (`true_data`), `OPENAI_API_KEY` (embeddings for some metrics), `JUDGE_GROQ_API_KEY` (or Groq) for the judge LLM.

```bash
# terminal A
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal B — full suite
uv run python -m evals

# or step by step
uv run python -m evals.pipeline --limit 5
uv run python -m evals.metrics
uv run python -m evals.guardrails_eval

# NeMo unit rails only (no API)
uv run python -m evals.guardrails_eval --unit-only
```

Reports land in `evals/results/` (`latest_run.json`, `metrics_report.json`, `guardrails_report.json`).

### Eval Monitor UI (Streamlit — separate from Next.js chat)

Local dashboard to watch scores over time (not the product UI):

```bash
# API must be running for live evals
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Eval monitor
uv run streamlit run eval_monitor.py
```

Open http://localhost:8501 → **Run full eval suite** or **Load saved results**. Tabs show RAGAS, tool correctness, guardrail TP/TN/FP/FN, and a case explorer. Details: [`evals_ui/README.md`](evals_ui/README.md).

**Metrics in plain English:**
- **Faithfulness** — answer claims supported by retrieved context (hallucination catch); score ≈ grounded_claims / total_claims
- **Answer Relevancy** — does the answer address the question?
- **Context Precision / Recall** — were the right chunks retrieved and ranked well?
- **Answer Correctness** — mix of factual overlap + semantic similarity vs reference
- **Tool correctness** — expected vs actual tools (e.g. `retriever` called or skipped); Jaccard / exact, **zero LLM cost**

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
│   ├── services/retrieval/   # embeddings ✅, Qdrant search+upsert ✅, Jina rerank ✅
│   ├── guardrails/           # NeMo rails
│   └── gateway/              # Portkey client
├── evals/                    # RAGAS suite
│   ├── golden_dataset.json   # 15 RAG + 6 guardrail cases
│   ├── pipeline.py           # live POST /query enrichment
│   ├── metrics.py            # RAGAS + tool correctness
│   ├── guardrails_eval.py    # confusion matrix
│   └── results/              # latest_run + reports (gitignored)
├── evals_ui/                 # Streamlit eval monitor (local only)
│   ├── app.py
│   └── services.py
├── web/                      # Next.js App Router UI (product chat)
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
| Qdrant search + Jina rerank | Done |
| AgentState + planner contract | Done |
| History-aware planner (LLM) | Done |
| Retriever node (search 15 → rerank 5) | Done |
| Responder (dual prompts) + LangGraph + thread_id | Done |
| Neon session memory (messages only) | Done |
| Langfuse agent tracing | Done |
| NeMo Guardrails (input/output + Colang) | Done |
| Portkey LLM gateway (VK / fallback / cache) | Done |
| Guardrails / Portkey polish | Done |
| FlashRank reranker | Skipped (using Jina API instead) |
| Embeddings + retrieval + rerank | Embeddings + Qdrant upsert done; rerank next |
| LangGraph query path | Done |
| Guardrails + Portkey | Done |
| RAGAS eval suite (golden + pipeline + metrics + confusion matrix) | Done |
| Streamlit Eval Monitor (`evals_ui/`) | Done |
| Next.js chat UI | Done (Hearth — cozy dark chat + proxy) |

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
cp .env.local.example .env.local   # set RAG_BACKEND_URL + optional RAG_API_KEY
npm install
npm run dev
```

UI: http://localhost:3000 (cozy chat — **Hearth**)

The browser talks only to Next.js `/api/query`; the route proxies to FastAPI and attaches `Authorization: Bearer $RAG_API_KEY` when set. Never put the API key in client code.

**Prod auth / rate limit (backend `.env`):**
- `RAG_API_KEY` (+ optional `RAG_JWT_SECRET`) — required for `/query` when `DEBUG=false`
- `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` — Redis rate limit; in-memory fallback locally
- `RATE_LIMIT_PER_MINUTE=60`
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

### 6. Ask the agent (Neon session memory)
Browser / UI should:
1. `POST /sessions` → store `session_id` in localStorage
2. `GET /sessions/{id}/messages` on load to paint history
3. `POST /query` with `{ "question": "...", "session_id": "<id>" }` each turn

```bash
uv run python -m uvicorn app.main:app --reload --port 8000

# new session
curl -X POST http://127.0.0.1:8000/sessions

# chat (reuse session_id)
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How do Jobs use a work queue?\",\"session_id\":\"YOUR-UUID\"}"

# reload transcript
curl http://127.0.0.1:8000/sessions/YOUR-UUID/messages
```

We persist **only** user/assistant messages in Neon — not full LangGraph checkpoints
(docs, plan objects, scores). The graph runs stateless each turn with history hydrated from the DB.

Or in Python:
```python
from app.agents import invoke_agent
r = invoke_agent("What is a DaemonSet?", session_id=None)  # creates session
r2 = invoke_agent("how do I monitor it?", session_id=r["session_id"])
```

---

## Config

Settings live in `app/config.py` and read the root `.env`. Important knobs:

- `OPENAI_*` / `EMBEDDING_DIMENSIONS`
- `GROQ_*` / `GROQ_FALLBACK_API_KEY` / `JUDGE_GROQ_API_KEY`
- `QDRANT_CLUSTER_ENDPOINT` or `QDRANT_URL` + `QDRANT_API_KEY` + `QDRANT_COLLECTION`
- `PORTKEY_*`, `JINA_API_KEY` / `JINA_RERANK_MODEL`, `LANGFUSE_*`, `LOGFIRE_TOKEN` (optional if `.logfire/` credentials exist)
- `DATABASE_URL` (Neon) for chat session message persistence

Never commit `.env` or `.logfire/`.

---

## Why this belongs in a portfolio

I'm not optimizing for "another LangChain hello world." I'm optimizing for decisions you have to defend in a review:

- separate clean vs noisy corpora
- provider split (embed vs generate)
- agent graph instead of a single prompt
- guardrails + evals as first-class folders
- observability from day one on ingestion

### Resume / LinkedIn bullets (eval + safety)

- Built a **RAGAS evaluation suite** with a 21-case golden set (15 grounded K8s/RAG Q&A + 6 jailbreak/off-topic/PII), live `/query` pipeline, and scored **Faithfulness, Answer Relevancy, Context Precision/Recall, Answer Correctness**.
- Added **tool-correctness** checks (expected vs actual retriever use via Jaccard/exact) for the agentic path with **zero judge LLM cost**.
- Measured guardrail quality with a **TP/TN/FP/FN confusion matrix** (`should_block` vs blocked) on top of NeMo input/output rails.
- Separated **judge LLM credentials** (`JUDGE_GROQ_API_KEY`) from the serving path so evals do not starve production inference.

If you're reading this on GitHub and want to poke at something specific (retrieval quality, chunking strategy, UI), open an issue or just star it and stalk the commits — that's where the real progress shows up.

---

## Author

**Mirza Shahbaz Ali Baig**  
Backend / AI systems work — building this in public as the stack comes together.
