"""Paragraph-aware text splitter for RAG chunking.

Design
------
We split on paragraph boundaries (blank lines), drop empty paragraphs, then
pack paragraphs into chunks of roughly 1000–1500 characters.

Why overlap?
------------
Hard cuts at chunk boundaries lose cross-boundary context. A definition that
ends in chunk N and an example that starts in chunk N+1 often need each other
for retrieval *and* for the LLM. Overlap copies a short tail of the previous
chunk onto the next one so boundary phrases still appear in both embeddings.
Without it, recall drops on questions whose answer sits on the seam between
two chunks. We keep overlap modest (~10–15% of target size) so we don't blow
up storage with near-duplicates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import logfire

from app.ingestion.loaders.base import ensure_logfire

# Target window for packed chunks (chars). Soft max before we force a split.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MAX_CHUNK_SIZE = 1500

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class TextChunk:
    """One retrieval unit after splitting."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


def _normalize_paragraphs(text: str) -> list[str]:
    """Split on blank lines and strip empties / excess whitespace."""
    parts = _PARAGRAPH_SPLIT.split(text.replace("\r\n", "\n").replace("\r", "\n"))
    paragraphs: list[str] = []
    for part in parts:
        cleaned = "\n".join(line.rstrip() for line in part.split("\n")).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    """If a single paragraph exceeds max_size, break on sentence boundaries."""
    if len(paragraph) <= max_size:
        return [paragraph]

    sentences = _SENTENCE_SPLIT.split(paragraph)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        # Hard-wrap a single oversized sentence.
        if len(sentence) > max_size:
            for i in range(0, len(sentence), max_size):
                pieces.append(sentence[i : i + max_size])
            current = ""
        else:
            current = sentence

    if current:
        pieces.append(current)
    return pieces


def _overlap_prefix(text: str, overlap: int) -> str:
    """Take up to `overlap` chars from the end, preferring a clean cut."""
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text

    tail = text[-overlap:]
    # Prefer starting at a whitespace boundary inside the tail.
    space = tail.find(" ")
    if space != -1 and space < len(tail) - 1:
        return tail[space + 1 :].lstrip()
    return tail.lstrip()


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """Split `text` into paragraph-aware overlapping chunks.

    Parameters
    ----------
    chunk_size:
        Soft target size (~1000–1500). We pack paragraphs until the next one
        would push past this.
    chunk_overlap:
        Characters of previous-chunk tail prepended to the next chunk so
        boundary context isn't lost (see module docstring).
    max_chunk_size:
        Hard ceiling. Oversized paragraphs are sentence-split to stay under it.
    """
    ensure_logfire()

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if max_chunk_size < chunk_size:
        raise ValueError("max_chunk_size must be >= chunk_size")

    base_meta = dict(metadata or {})

    with logfire.span(
        "chunking.split_text",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunk_size=max_chunk_size,
    ):
        paragraphs = _normalize_paragraphs(text)
        if not paragraphs:
            logfire.info("Splitter found no paragraphs", chars=len(text or ""))
            return []

        units: list[str] = []
        for para in paragraphs:
            units.extend(_split_long_paragraph(para, max_chunk_size))

        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_len = 0
        # Approximate source offsets for debugging / citations.
        cursor = 0

        def flush(parts: list[str], start: int) -> None:
            body = "\n\n".join(parts).strip()
            if not body:
                return
            end = start + len(body)
            chunks.append(
                TextChunk(
                    text=body,
                    index=len(chunks),
                    start_char=start,
                    end_char=end,
                    metadata={
                        **base_meta,
                        "chunk_size_target": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                )
            )

        chunk_start = 0
        for unit in units:
            unit_len = len(unit)
            separator = 2 if current_parts else 0  # "\n\n"
            projected = current_len + separator + unit_len

            if current_parts and projected > chunk_size:
                flush(current_parts, chunk_start)
                prev_text = chunks[-1].text
                overlap_text = _overlap_prefix(prev_text, chunk_overlap)

                current_parts = []
                current_len = 0
                chunk_start = max(0, chunks[-1].end_char - len(overlap_text))

                if overlap_text:
                    current_parts.append(overlap_text)
                    current_len = len(overlap_text)

                # If unit alone still exceeds soft target but fits max, start fresh.
                if current_len and current_len + 2 + unit_len > max_chunk_size:
                    flush(current_parts, chunk_start)
                    current_parts = []
                    current_len = 0
                    chunk_start = cursor

            if not current_parts:
                chunk_start = cursor

            current_parts.append(unit)
            current_len = len("\n\n".join(current_parts))
            cursor += unit_len + 2  # rough advance including paragraph gap

        if current_parts:
            flush(current_parts, chunk_start)

        # Re-index after any empty skips (defensive).
        for i, chunk in enumerate(chunks):
            chunk.index = i

        logfire.info(
            "Split text into chunks",
            paragraphs=len(paragraphs),
            units=len(units),
            chunks=len(chunks),
            avg_chars=(
                round(sum(c.char_count for c in chunks) / len(chunks)) if chunks else 0
            ),
        )
        return chunks


def split_document(
    text: str,
    *,
    source: str | None = None,
    doc_type: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[TextChunk]:
    """Convenience wrapper that stamps source/doc_type onto each chunk."""
    meta: dict[str, Any] = dict(extra_metadata or {})
    if source is not None:
        meta["source"] = source
    if doc_type is not None:
        meta["doc_type"] = doc_type
    return split_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunk_size=max_chunk_size,
        metadata=meta,
    )
