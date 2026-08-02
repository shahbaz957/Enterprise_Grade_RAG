"""Retrieval services — embeddings, Qdrant, reranking."""

from app.services.retrieval.embedding import (
    DEFAULT_BATCH_SIZE,
    MAX_RETRIES,
    EmbeddedChunk,
    EmbeddingError,
    embed_chunks,
    embed_query,
    embed_texts,
    probe,
)
from app.services.retrieval.qdrant_service import (
    DEFAULT_SEARCH_LIMIT,
    SearchResult,
    ensure_collection,
    get_qdrant_client,
    point_id_for,
    search_enterprise_knowledge,
    upsert_embedded_chunks,
)
from app.services.retrieval.reranker import RerankError, RerankHit, rerank, rerank_hits

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SEARCH_LIMIT",
    "MAX_RETRIES",
    "EmbeddedChunk",
    "EmbeddingError",
    "SearchResult",
    "RerankError",
    "RerankHit",
    "embed_chunks",
    "embed_query",
    "embed_texts",
    "probe",
    "ensure_collection",
    "get_qdrant_client",
    "point_id_for",
    "search_enterprise_knowledge",
    "upsert_embedded_chunks",
    "rerank",
    "rerank_hits",
]
