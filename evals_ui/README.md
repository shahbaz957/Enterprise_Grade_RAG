# Eval Monitor (Streamlit)

Local dashboard for **RAGAS**, **tool correctness**, and **guardrail confusion matrices**.

This is **not** the product chat UI (that will be Next.js under `web/`).

## Run

From the repo root (API should already be up for live runs):

```bash
# terminal A — RAG API
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal B — eval dashboard (preferred launcher)
uv run streamlit run eval_monitor.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

> Prefer `eval_monitor.py` over `evals_ui/app.py` so Streamlit does not load the
> same widgets twice (package import + script run).

## What you can do

1. **Run full eval suite** — live `POST /query` over the golden set, then score RAGAS + tools + guardrails.
2. **Load saved results** — read `evals/results/latest_run.json`, `metrics_report.json`, `guardrails_report.json` (e.g. after `uv run python -m evals` in a terminal).
3. Browse tabs: RAGAS chart/table, tool correctness table, TP/TN/FP/FN matrix, case explorer.

## Tips

- Use **Skip RAGAS** for a fast rails/tool-only pass (no judge LLM).
- Set `JUDGE_GROQ_API_KEY` + `OPENAI_API_KEY` for full RAGAS scoring.
- Limit cases while iterating (sidebar **Limit cases**).
