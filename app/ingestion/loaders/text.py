"""Plain-text loader (.txt, .md, .csv, .log, …)."""

from __future__ import annotations

from pathlib import Path

import logfire

from app.ingestion.loaders.base import LoadedDocument, ensure_logfire, resolve_path

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".jsonl", ".yaml", ".yml"}


def load_text(path: str | Path, *, encoding: str = "utf-8") -> LoadedDocument:
    ensure_logfire()
    file_path = resolve_path(path)

    with logfire.span("loader.text", source=str(file_path)):
        try:
            text = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin-1")
            encoding = "latin-1"
            logfire.warn(
                "Text decode fell back to latin-1",
                source=str(file_path),
            )

        doc = LoadedDocument(
            text=text.strip(),
            source=str(file_path),
            doc_type="text",
            metadata={
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "encoding": encoding,
                "size_bytes": file_path.stat().st_size,
            },
        )
        logfire.info(
            "Loaded text document",
            source=str(file_path),
            chars=doc.char_count,
            encoding=encoding,
        )
        return doc
