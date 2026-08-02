"""Qdrant client helpers — ensure collection, upsert, search + Jina rerank."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.ingestion.loaders.base import ensure_logfire
from app.services.retrieval.embedding import EmbeddedChunk, embed_query
from app.services.retrieval.reranker import rerank

_client: QdrantClient | None = None

# Pull extra vector hits before reranking so Jina has room to re-order.
DEFAULT_SEARCH_LIMIT = 15
DEFAULT_CANDIDATE_MULTIPLIER = 3
DEFAULT_MIN_CANDIDATES = 30


@dataclass(slots=True)
class SearchResult:
    """One enterprise-knowledge hit after vector search + Jina rerank."""

    text: str
    score: float
    vector_score: float | None
    rank: int
    id: str | int | None
    metadata: dict[str, Any] = field(default_factory=dict)


def get_qdrant_client() -> QdrantClient:
    """Lazy singleton Qdrant client (cloud or local)."""
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"url": settings.qdrant_endpoint, "timeout": 60}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def point_id_for(source: str, chunk_index: int) -> str:
    """Stable UUID so re-ingesting the same file overwrites the same points."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{chunk_index}"))
    # that's a smart move to avoid collisions author : Mirza Shahbaz Ali Baig


def ensure_collection(
    *,
    collection: str | None = None,
    vector_size: int | None = None,
) -> str:
    """Create the collection if missing (cosine, named size from settings)."""
    ensure_logfire()
    name = collection or settings.qdrant_collection
    size = vector_size or settings.embedding_dimensions
    client = get_qdrant_client()

    with logfire.span("qdrant.ensure_collection", collection=name, size=size):
        existing = {c.name for c in client.get_collections().collections}
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=size,
                    distance=qm.Distance.COSINE,
                ),
            )
            logfire.info("Created Qdrant collection", collection=name, size=size)
        else:
            logfire.info("Qdrant collection already exists", collection=name)
    return name


def upsert_embedded_chunks(
    chunks: Sequence[EmbeddedChunk],
    *,
    collection: str | None = None,
    corpus: str | None = None,
) -> int:
    """Upsert embedded chunks. Returns number of points written."""
    ensure_logfire()
    if not chunks:
        return 0

    name = collection or settings.qdrant_collection
    dims = chunks[0].dimensions
    ensure_collection(collection=name, vector_size=dims)

    # Refuse mixed / wrong dimensions (e.g. MiniLM fallback vs OpenAI 1536).
    bad = [c for c in chunks if c.dimensions != dims]
    if bad:
        raise ValueError(
            f"{len(bad)} chunks have dim != {dims}; "
            "rebuild with a single embedding provider before upsert"
        )

    client = get_qdrant_client()
    points: list[qm.PointStruct] = []
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        payload: dict[str, Any] = {
            "text": chunk.text,
            "source": source,
            "chunk_index": chunk.index,
            "doc_type": chunk.metadata.get("doc_type"),
            "filename": chunk.metadata.get("filename"),
            "embedding_provider": chunk.provider,
            "embedding_model": chunk.model,
            "start_char": chunk.metadata.get("start_char"),
            "end_char": chunk.metadata.get("end_char"),
        }
        if corpus:
            payload["corpus"] = corpus
        # Keep extra loader metadata without blowing payload size.
        for key in ("title", "page_count", "slide_count", "extractor"):
            if key in chunk.metadata:
                payload[key] = chunk.metadata[key]

        points.append(
            qm.PointStruct(
                id=point_id_for(source, chunk.index),
                vector=chunk.embedding,
                payload=payload,
            )
        )

    with logfire.span(
        "qdrant.upsert",
        collection=name,
        points=len(points),
        corpus=corpus,
    ):
        # Batch upserts to avoid huge payloads.
        batch_size = 64
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            client.upsert(collection_name=name, points=batch, wait=True)

        logfire.info(
            "Upserted points to Qdrant",
            collection=name,
            points=len(points),
            corpus=corpus,
        )
    return len(points)


def search_enterprise_knowledge(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    *,
    collection: str | None = None,
    corpus: str | None = None,
    candidate_limit: int | None = None,
    rerank_results: bool = True,
) -> list[SearchResult]:
    """Semantic search over the enterprise collection, then Jina-rerank.

    Flow
    ----
    1. Embed `query` with the configured OpenAI embedding model.
    2. Pull a wider candidate set from Qdrant (vector similarity).
    3. Rerank candidates with Jina (`JINA_API_KEY`) and return top `limit`
       hits including full payload metadata.
    """
    ensure_logfire()
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    name = collection or settings.qdrant_collection
    final_limit = max(1, limit)
    fetch_k = candidate_limit or max(
        final_limit * DEFAULT_CANDIDATE_MULTIPLIER,
        DEFAULT_MIN_CANDIDATES,
    )

    query_filter: qm.Filter | None = None
    if corpus:
        query_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="corpus",
                    match=qm.MatchValue(value=corpus),
                )
            ]
        )

    with logfire.span(
        "qdrant.search_enterprise_knowledge",
        collection=name,
        limit=final_limit,
        candidate_limit=fetch_k,
        corpus=corpus,
        rerank=rerank_results,
    ):
        vector = embed_query(query, allow_fallback=False)
        client = get_qdrant_client()

        response = client.query_points(
            collection_name=name,
            query=vector,
            query_filter=query_filter,
            limit=fetch_k,
            with_payload=True,
        )
        points = list(response.points or [])

        if not points:
            logfire.info("No Qdrant hits for query", collection=name)
            return []

        documents: list[str] = []
        metas: list[dict[str, Any]] = []
        vector_scores: list[float | None] = []
        point_ids: list[str | int | None] = []

        for point in points:
            payload = dict(point.payload or {})
            text = str(payload.get("text") or "")
            documents.append(text)
            vector_scores.append(float(point.score) if point.score is not None else None)
            point_ids.append(point.id)
            meta = {key: value for key, value in payload.items() if key != "text"}
            meta["point_id"] = point.id
            meta["vector_score"] = point.score
            metas.append(meta)

        if not rerank_results or not settings.jina_api_key:
            if rerank_results and not settings.jina_api_key:
                logfire.warn("JINA_API_KEY missing — returning vector order only")
            results = [
                SearchResult(
                    text=documents[i],
                    score=float(vector_scores[i] or 0.0),
                    vector_score=vector_scores[i],
                    rank=i + 1,
                    id=point_ids[i],
                    metadata=metas[i],
                )
                for i in range(min(final_limit, len(documents)))
            ]
            logfire.info(
                "Enterprise search complete (no rerank)",
                hits=len(results),
            )
            return results

        reranked = rerank(
            query,
            documents,
            top_n=final_limit,
            metadata=metas,
        )

        results = []
        for rank, hit in enumerate(reranked, start=1):
            idx = hit.index
            results.append(
                SearchResult(
                    text=hit.text,
                    score=hit.score,
                    vector_score=vector_scores[idx] if 0 <= idx < len(vector_scores) else None,
                    rank=rank,
                    id=point_ids[idx] if 0 <= idx < len(point_ids) else None,
                    metadata={
                        **hit.metadata,
                        "rerank_score": hit.score,
                        "rerank_model": settings.jina_rerank_model,
                    },
                )
            )

        logfire.info(
            "Enterprise search complete",
            hits=len(results),
            top_rerank_score=results[0].score if results else None,
        )
        return results
