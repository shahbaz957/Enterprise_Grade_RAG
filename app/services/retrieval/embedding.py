"""OpenAI embeddings for chunk / query vectors.

- Batch size: 50 (API-friendly, keeps payloads under rate-limit spikes)
- Retries: 4 attempts with exponential backoff (1s → 2s → 4s → 8s)
- Model / dims: from `settings` (`OPENAI_EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`)

`probe()` checks the key + expected dimensionality before a full ingest run.
If a batch still fails after retries, we fall back to a local
`sentence-transformers` model so ingestion can continue (vectors from the
fallback are flagged in metadata — swap/rebuild if you need a pure OpenAI
collection).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import logfire
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.config import settings
from app.ingestion.chunking.splitter import TextChunk
from app.ingestion.loaders.base import ensure_logfire

DEFAULT_BATCH_SIZE = 50
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.0

# Local fallback when OpenAI is unavailable after retries.
_FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_fallback_model: Any = None


@dataclass(slots=True)
class EmbeddedChunk:
    """Chunk text + embedding vector ready for Qdrant upsert."""

    text: str
    index: int
    embedding: list[float]
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        return len(self.embedding)


class EmbeddingError(RuntimeError):
    """Raised when embedding fails and no usable fallback vector is produced."""


def _get_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise EmbeddingError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=settings.openai_api_key)


def _get_fallback_model() -> Any:
    global _fallback_model
    if _fallback_model is None:
        from sentence_transformers import SentenceTransformer

        logfire.warn(
            "Loading local embedding fallback model",
            model=_FALLBACK_MODEL_NAME,
        )
        _fallback_model = SentenceTransformer(_FALLBACK_MODEL_NAME)
    return _fallback_model


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False


def _embed_batch_openai(client: OpenAI, texts: Sequence[str]) -> list[list[float]]:
    """Call OpenAI embeddings once for a batch (no retry here)."""
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=list(texts),
        dimensions=settings.embedding_dimensions,
    )
    # API may not guarantee order in all SDKs — sort by index to be safe.
    ordered = sorted(response.data, key=lambda row: row.index)
    return [row.embedding for row in ordered]


def _embed_batch_fallback(texts: Sequence[str]) -> list[list[float]]:
    model = _get_fallback_model()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return [vec.tolist() for vec in vectors]


def _embed_batch_with_retries(
    client: OpenAI | None,
    texts: Sequence[str],
    *,
    batch_index: int,
    allow_fallback: bool = True,
) -> tuple[list[list[float]], str, str]:
    """Embed one batch. Returns (vectors, provider, model_name)."""
    last_error: BaseException | None = None

    if client is not None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vectors = _embed_batch_openai(client, texts)
                if len(vectors) != len(texts):
                    raise EmbeddingError(
                        f"Expected {len(texts)} embeddings, got {len(vectors)}"
                    )
                return (
                    vectors,
                    "openai",
                    settings.openai_embedding_model,
                )
            except Exception as exc:  # noqa: BLE001 — classified below
                last_error = exc
                if not _is_retryable(exc) or attempt == MAX_RETRIES:
                    logfire.error(
                        "OpenAI embedding batch failed",
                        batch_index=batch_index,
                        attempt=attempt,
                        error=str(exc),
                        retryable=_is_retryable(exc),
                    )
                    break

                delay = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logfire.warn(
                    "OpenAI embedding batch retry",
                    batch_index=batch_index,
                    attempt=attempt,
                    max_retries=MAX_RETRIES,
                    sleep_seconds=delay,
                    error=str(exc),
                )
                time.sleep(delay)

    if allow_fallback:
        logfire.warn(
            "Using sentence-transformers fallback for batch",
            batch_index=batch_index,
            last_error=str(last_error) if last_error else None,
        )
        vectors = _embed_batch_fallback(texts)
        return vectors, "sentence-transformers", _FALLBACK_MODEL_NAME

    raise EmbeddingError(
        f"Embedding batch {batch_index} failed after {MAX_RETRIES} retries: {last_error}"
    )


def probe(*, use_fallback_on_failure: bool = False) -> dict[str, Any]:
    """Smoke-check the embedding path (one short string).

    Returns model/provider/dimension info. Raises if OpenAI is required and down.
    """
    ensure_logfire()
    sample = "enterprise rag embedding probe"

    with logfire.span("embedding.probe"):
        client: OpenAI | None = None
        try:
            client = _get_openai_client()
        except EmbeddingError:
            if not use_fallback_on_failure:
                raise
            client = None

        vectors, provider, model = _embed_batch_with_retries(
            client,
            [sample],
            batch_index=0,
            allow_fallback=use_fallback_on_failure,
        )
        dims = len(vectors[0])
        expected = settings.embedding_dimensions if provider == "openai" else dims

        info = {
            "ok": True,
            "provider": provider,
            "model": model,
            "dimensions": dims,
            "expected_dimensions": expected,
            "dimensions_match": dims == expected if provider == "openai" else True,
        }
        logfire.info("Embedding probe ok", **info)
        return info


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    allow_fallback: bool = True,
) -> list[list[float]]:
    """Embed an arbitrary list of strings in batches of `batch_size`."""
    ensure_logfire()
    if not texts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    cleaned = [t if t and t.strip() else " " for t in texts]

    with logfire.span(
        "embedding.embed_texts",
        total=len(cleaned),
        batch_size=batch_size,
    ):
        client: OpenAI | None
        try:
            client = _get_openai_client()
        except EmbeddingError:
            if not allow_fallback:
                raise
            client = None
            logfire.warn("OPENAI_API_KEY missing; embedding via fallback only")

        all_vectors: list[list[float]] = []
        for batch_index, start in enumerate(range(0, len(cleaned), batch_size)):
            batch = cleaned[start : start + batch_size]
            vectors, provider, model = _embed_batch_with_retries(
                client,
                batch,
                batch_index=batch_index,
                allow_fallback=allow_fallback,
            )
            all_vectors.extend(vectors)
            logfire.info(
                "Embedded batch",
                batch_index=batch_index,
                batch_len=len(batch),
                provider=provider,
                model=model,
                dimensions=len(vectors[0]) if vectors else 0,
            )

        logfire.info(
            "Embedded all texts",
            total=len(all_vectors),
            dimensions=len(all_vectors[0]) if all_vectors else 0,
        )
        return all_vectors


def embed_query(query: str, *, allow_fallback: bool = True) -> list[float]:
    """Embed a single user query."""
    vectors = embed_texts([query], batch_size=1, allow_fallback=allow_fallback)
    return vectors[0]


def embed_chunks(
    chunks: Sequence[TextChunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    allow_fallback: bool = True,
) -> list[EmbeddedChunk]:
    """Embed `TextChunk`s and return `EmbeddedChunk`s (same order)."""
    ensure_logfire()
    if not chunks:
        return []

    with logfire.span("embedding.embed_chunks", chunks=len(chunks), batch_size=batch_size):
        texts = [c.text for c in chunks]
        # Track provider per batch so mixed OpenAI/fallback runs stay honest.
        client: OpenAI | None
        try:
            client = _get_openai_client()
        except EmbeddingError:
            if not allow_fallback:
                raise
            client = None

        embedded: list[EmbeddedChunk] = []
        for batch_index, start in enumerate(range(0, len(chunks), batch_size)):
            batch_chunks = list(chunks[start : start + batch_size])
            batch_texts = [c.text for c in batch_chunks]
            vectors, provider, model = _embed_batch_with_retries(
                client,
                batch_texts,
                batch_index=batch_index,
                allow_fallback=allow_fallback,
            )
            for chunk, vector in zip(batch_chunks, vectors, strict=True):
                embedded.append(
                    EmbeddedChunk(
                        text=chunk.text,
                        index=chunk.index,
                        embedding=vector,
                        provider=provider,
                        model=model,
                        metadata={
                            **chunk.metadata,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "embedding_provider": provider,
                            "embedding_model": model,
                            "embedding_dimensions": len(vector),
                        },
                    )
                )

        logfire.info(
            "Embedded chunks",
            chunks=len(embedded),
            providers=sorted({e.provider for e in embedded}),
        )
        return embedded
