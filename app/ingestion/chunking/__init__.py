"""Chunking utilities for ingestion."""

from app.ingestion.chunking.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    TextChunk,
    split_document,
    split_text,
)

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_CHUNK_SIZE",
    "TextChunk",
    "split_document",
    "split_text",
]
