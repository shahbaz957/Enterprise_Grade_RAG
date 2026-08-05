"""Retriever node — Qdrant search (15) → Jina rerank (5) → state.documents.

Skipped entirely when intent is not ``technical`` (conversational path already
has a final_answer from the planner).
"""

from __future__ import annotations

from typing import Any

import logfire

from app.agents.state import AgentState, RetrievedDocument
from app.ingestion.loaders.base import ensure_logfire
from app.services.retrieval.qdrant_service import search_enterprise_knowledge

# Vector candidates from Qdrant, then keep top-k after Jina rerank.
RETRIEVE_CANDIDATES = 15
RERANK_TOP_K = 5


def _is_technical(state: AgentState) -> bool:
    intent = state.get("intent") or "unknown"
    if intent == "technical":
        return True
    plan = state.get("plan") or {}
    if isinstance(plan, dict) and plan.get("intent") == "technical":
        return True
    if getattr(plan, "intent", None) == "technical":
        return True
    return False


def _search_query(state: AgentState) -> str:
    """Prefer planner rewrite; fall back to current_query."""
    plan = state.get("plan")
    if isinstance(plan, dict):
        rewritten = (plan.get("rewritten_query") or "").strip()
        if rewritten:
            return rewritten
    elif plan is not None:
        rewritten = (getattr(plan, "rewritten_query", None) or "").strip()
        if rewritten:
            return rewritten
    return (state.get("current_query") or "").strip()


def _hits_to_documents(hits: list[Any]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for hit in hits:
        meta = dict(getattr(hit, "metadata", None) or {})
        docs.append(
            RetrievedDocument(
                text=hit.text,
                score=hit.score,
                vector_score=hit.vector_score,
                rank=hit.rank,
                source=meta.get("source"),
                filename=meta.get("filename"),
                doc_type=meta.get("doc_type"),
                corpus=meta.get("corpus"),
                chunk_index=meta.get("chunk_index"),
                metadata=meta,
            ).model_dump()
        )
    return docs


def retriever_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: retrieve + rerank only on the technical path."""
    ensure_logfire()

    with logfire.span(
        "agent.retriever",
        intent=state.get("intent"),
        status=state.get("status"),
    ):
        if not _is_technical(state):
            logfire.info(
                "Retriever skipped (non-technical intent)",
                intent=state.get("intent"),
            )
            # Leave documents empty; do not disturb conversational final_answer.
            return {
                "documents": [],
            }

        query = _search_query(state)
        if not query:
            logfire.error("Retriever: empty search query on technical path")
            return {
                "documents": [],
                "status": "error",
                "error": "technical intent but no rewritten_query / current_query",
            }

        try:
            hits = search_enterprise_knowledge(
                query,
                limit=RERANK_TOP_K,
                candidate_limit=RETRIEVE_CANDIDATES,
                rerank_results=True,
            )
            documents = _hits_to_documents(hits)
            logfire.info(
                "Retriever complete",
                query=query[:200],
                candidates=RETRIEVE_CANDIDATES,
                kept=len(documents),
                top_score=documents[0].score if documents else None,
            )
            return {
                "documents": documents,
                "current_query": query,
                "status": "responding",
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — surface on graph state
            logfire.error("Retriever failed", error=str(exc), query=query[:200])
            return {
                "documents": [],
                "status": "error",
                "error": f"retriever failed: {exc}",
            }
