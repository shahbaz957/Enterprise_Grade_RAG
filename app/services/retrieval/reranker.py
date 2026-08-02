"""Jina AI reranker — re-order retrieved chunks by query relevance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx
import logfire

from app.config import settings
from app.ingestion.loaders.base import ensure_logfire

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


@dataclass(slots=True)
class RerankHit:
    """One document after Jina reranking."""

    index: int
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RerankError(RuntimeError):
    """Raised when the Jina rerank API call fails."""


def rerank(
    query: str,
    documents: Sequence[str],
    *,
    top_n: int | None = None,
    model: str | None = None,
    metadata: Sequence[dict[str, Any]] | None = None,
) -> list[RerankHit]:
    """Rerank `documents` for `query` via Jina.

    `metadata[i]` (optional) is attached to the hit whose original index is `i`.
    """
    ensure_logfire()

    if not query.strip():
        raise ValueError("query must be non-empty")
    if not documents:
        return []
    if not settings.jina_api_key:
        raise RerankError("JINA_API_KEY is not set")

    limit = top_n if top_n is not None else settings.jina_rerank_top_n
    limit = max(1, min(limit, len(documents)))
    model_name = model or settings.jina_rerank_model
    docs = [d if d and d.strip() else " " for d in documents]

    payload = {
        "model": model_name,
        "query": query,
        "documents": docs,
        "top_n": limit,
        "return_documents": True,
    }
    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    with logfire.span(
        "rerank.jina",
        model=model_name,
        candidates=len(docs),
        top_n=limit,
    ):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(JINA_RERANK_URL, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            logfire.error("Jina rerank request failed", error=str(exc))
            raise RerankError(f"Jina rerank failed: {exc}") from exc

        results = body.get("results") or []
        hits: list[RerankHit] = []
        for row in results:
            idx = int(row["index"])
            score = float(row.get("relevance_score", row.get("score", 0.0)))
            # Prefer returned document text; fall back to our input list.
            doc_obj = row.get("document")
            if isinstance(doc_obj, dict):
                text = str(doc_obj.get("text") or docs[idx])
            elif isinstance(doc_obj, str):
                text = doc_obj
            else:
                text = str(row.get("text") or docs[idx])

            meta: dict[str, Any] = {}
            if metadata is not None and 0 <= idx < len(metadata):
                meta = dict(metadata[idx])

            hits.append(
                RerankHit(index=idx, text=text, score=score, metadata=meta)
            )

        logfire.info(
            "Jina rerank complete",
            model=model_name,
            returned=len(hits),
            top_score=hits[0].score if hits else None,
        )
        return hits


def rerank_hits(
    query: str,
    candidates: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    top_n: int | None = None,
) -> list[RerankHit]:
    """Convenience: candidates are dicts with at least a text field + metadata."""
    documents = [str(c.get(text_key, "")) for c in candidates]
    metas = [dict(c) for c in candidates]
    return rerank(query, documents, top_n=top_n, metadata=metas)
