"""Step 3 — RAGAS metrics + tool correctness on enriched pipeline results.

Uses USE_OPENAI_LLM for the judge chat model (OpenAI when true, Groq when false).
OpenAI embeddings are still required for some RAGAS metrics.

Metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
  - Answer Correctness
  - Tool correctness (Jaccard / exact — no LLM)

Usage:
  uv run python -m evals.metrics
  uv run python -m evals.metrics --from evals/results/latest_run.json --rag-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals._compat import ensure_ragas_imports
from evals.common import DEFAULT_RESULTS_PATH, RESULTS_DIR, load_results, save_results


def _judge_chat_model():
    """Judge LLM for RAGAS — follows USE_OPENAI_LLM (same as guardrails/direct)."""
    from app.config import settings

    if settings.use_openai_llm:
        if not settings.has_openai:
            raise RuntimeError("USE_OPENAI_LLM=true but OPENAI_API_KEY is unset")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    key = settings.judge_groq_api_key or settings.active_groq_api_key
    if not key:
        raise RuntimeError(
            "USE_OPENAI_LLM=false — set JUDGE_GROQ_API_KEY or GROQ_API_KEY."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_chat_model,
        api_key=key,
        temperature=0,
    )


def _judge_embeddings():
    from langchain_openai import OpenAIEmbeddings

    from app.config import settings

    if not settings.has_openai:
        raise RuntimeError(
            "OpenAI embeddings required for Answer Relevancy / Correctness. "
            "Set OPENAI_API_KEY."
        )
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def tool_jaccard(expected: list[str], actual: list[str]) -> float:
    e, a = set(expected or []), set(actual or [])
    if not e and not a:
        return 1.0
    if not e and a:
        return 0.0
    if e and not a:
        return 0.0
    return len(e & a) / len(e | a)


def tool_exact(expected: list[str], actual: list[str]) -> float:
    return 1.0 if set(expected or []) == set(actual or []) else 0.0


def compute_tool_correctness(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    jaccards: list[float] = []
    exacts: list[float] = []
    for c in cases:
        exp = list(c.get("expected_tools") or [])
        act = list(c.get("actual_tools") or c.get("tools") or [])
        j = tool_jaccard(exp, act)
        x = tool_exact(exp, act)
        jaccards.append(j)
        exacts.append(x)
        rows.append(
            {
                "id": c.get("id"),
                "kind": c.get("kind"),
                "expected_tools": exp,
                "actual_tools": act,
                "jaccard": round(j, 4),
                "exact": x,
            }
        )
    n = len(rows) or 1
    return {
        "mean_jaccard": round(sum(jaccards) / n, 4),
        "mean_exact": round(sum(exacts) / n, 4),
        "rows": rows,
    }


def _rag_rows_for_ragas(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only RAG cases with a reference answer and some response."""
    out = []
    for c in cases:
        if c.get("kind") != "rag":
            continue
        question = (c.get("question") or "").strip()
        answer = (c.get("actual_response") or "").strip()
        reference = (c.get("reference") or "").strip()
        contexts = list(c.get("contexts") or [])
        if not question or not answer or not reference:
            continue
        if not contexts:
            # Faithfulness/precision need contexts; keep a placeholder to avoid hard crash
            contexts = [""]
        out.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
                # legacy aliases some ragas versions still accept
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference,
                "_id": c.get("id"),
            }
        )
    return out


def run_ragas(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_ragas_imports()
    from datasets import Dataset

    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = _rag_rows_for_ragas(cases)
    if not rows:
        return {
            "n": 0,
            "means": {},
            "note": "No RAG rows with reference + actual_response + contexts",
        }

    ds = Dataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )

    llm = LangchainLLMWrapper(_judge_chat_model())
    embeddings = LangchainEmbeddingsWrapper(_judge_embeddings())

    # Bind judge models onto metric instances where needed
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]
    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    # result may be EvaluationResult with to_pandas()
    try:
        df = result.to_pandas()
        means = {
            col: round(float(df[col].mean()), 4)
            for col in df.columns
            if col
            not in {
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
                "question",
                "answer",
                "contexts",
                "ground_truth",
            }
            and df[col].dtype != object
        }
        per_row = df.to_dict(orient="records")
        # attach ids
        for i, rec in enumerate(per_row):
            if i < len(rows):
                rec["id"] = rows[i].get("_id")
    except Exception:
        means = {}
        per_row = []
        try:
            means = {k: round(float(v), 4) for k, v in dict(result).items()}
        except Exception:
            means = {"raw": str(result)}

    return {"n": len(rows), "means": means, "rows": per_row}


def run_metrics(results_path: Path, *, rag_only: bool = False) -> dict[str, Any]:
    payload = load_results(results_path)
    cases = list(payload.get("cases") or [])
    if rag_only:
        cases = [c for c in cases if c.get("kind") == "rag"]

    tool = compute_tool_correctness(cases)
    ragas = run_ragas([c for c in cases if c.get("kind") == "rag"])

    summary = {
        "source": str(results_path),
        "tool_correctness": {
            "mean_jaccard": tool["mean_jaccard"],
            "mean_exact": tool["mean_exact"],
        },
        "ragas_means": ragas.get("means") or {},
        "n_ragas": ragas.get("n", 0),
        "n_tool_rows": len(tool["rows"]),
    }
    full = {
        **summary,
        "tool_rows": tool["rows"],
        "ragas": ragas,
    }
    return full


def _print_tables(report: dict[str, Any]) -> None:
    print("\n=== Tool correctness ===")
    print(
        f"mean Jaccard={report['tool_correctness']['mean_jaccard']}  "
        f"mean exact={report['tool_correctness']['mean_exact']}"
    )
    for row in report.get("tool_rows") or []:
        print(
            f"  {row['id']}: expected={row['expected_tools']} actual={row['actual_tools']} "
            f"jaccard={row['jaccard']} exact={row['exact']}"
        )

    print("\n=== RAGAS means (RAG cases) ===")
    means = report.get("ragas_means") or {}
    if not means:
        print("  (no scores — check OPENAI_API_KEY + JUDGE_GROQ_API_KEY and pipeline results)")
    else:
        for k, v in means.items():
            print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS + tool correctness metrics")
    parser.add_argument(
        "--from",
        dest="results_path",
        default=str(DEFAULT_RESULTS_PATH),
        help="Enriched results JSON from evals.pipeline",
    )
    parser.add_argument("--rag-only", action="store_true", help="Only score kind=rag rows")
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "metrics_report.json"),
        help="Where to write the metrics report",
    )
    args = parser.parse_args()

    path = Path(args.results_path)
    if not path.is_file():
        raise SystemExit(
            f"Results not found: {path}. Run: uv run python -m evals.pipeline"
        )

    report = run_metrics(path, rag_only=args.rag_only)
    _print_tables(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote metrics report → {out}")


if __name__ == "__main__":
    main()
