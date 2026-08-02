"""Qdrant client helpers — ensure collection + upsert embedded chunks."""

from __future__ import annotations

import uuid
from typing import Any, Sequence

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.ingestion.loaders.base import ensure_logfire
from app.services.retrieval.embedding import EmbeddedChunk

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Lazy singleton Qdrant client (cloud or local)."""
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"url": settings.qdrant_endpoint}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def point_id_for(source: str, chunk_index: int) -> str:
    """Stable UUID so re-ingesting the same file overwrites the same points."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{chunk_index}"))


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
