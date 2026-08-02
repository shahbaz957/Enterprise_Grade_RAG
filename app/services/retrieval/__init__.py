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
    ensure_collection,
    get_qdrant_client,
    point_id_for,
    upsert_embedded_chunks,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "MAX_RETRIES",
    "EmbeddedChunk",
    "EmbeddingError",
    "embed_chunks",
    "embed_query",
    "embed_texts",
    "probe",
    "ensure_collection",
    "get_qdrant_client",
    "point_id_for",
    "upsert_embedded_chunks",
]
