"""Document loaders — route by file extension."""

from __future__ import annotations

from pathlib import Path

import logfire

from app.ingestion.loaders.base import LoadedDocument, ensure_logfire, resolve_path
from app.ingestion.loaders.html import HTML_EXTENSIONS, load_html
from app.ingestion.loaders.office import (
    DOCX_EXTENSIONS,
    OFFICE_EXTENSIONS,
    PPTX_EXTENSIONS,
    load_docx,
    load_office,
    load_pptx,
)
from app.ingestion.loaders.pdf import PDF_EXTENSIONS, load_pdf
from app.ingestion.loaders.text import TEXT_EXTENSIONS, load_text

SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | HTML_EXTENSIONS | PDF_EXTENSIONS | OFFICE_EXTENSIONS
)


def load_document(path: str | Path) -> LoadedDocument:
    """Load a file with the matching loader (text / html / pdf / office)."""
    ensure_logfire()
    file_path = resolve_path(path)
    ext = file_path.suffix.lower()

    with logfire.span("loader.dispatch", source=str(file_path), extension=ext):
        if ext in TEXT_EXTENSIONS:
            doc = load_text(file_path)
        elif ext in HTML_EXTENSIONS:
            doc = load_html(file_path)
        elif ext in PDF_EXTENSIONS:
            doc = load_pdf(file_path)
        elif ext in OFFICE_EXTENSIONS:
            doc = load_office(file_path)
        else:
            logfire.error("Unsupported document type", source=str(file_path), extension=ext)
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        logfire.info(
            "Document load complete",
            source=str(file_path),
            doc_type=doc.doc_type,
            chars=doc.char_count,
            empty=doc.is_empty,
        )
        return doc


__all__ = [
    "LoadedDocument",
    "SUPPORTED_EXTENSIONS",
    "load_document",
    "load_text",
    "load_html",
    "load_pdf",
    "load_docx",
    "load_pptx",
    "load_office",
    "TEXT_EXTENSIONS",
    "HTML_EXTENSIONS",
    "PDF_EXTENSIONS",
    "DOCX_EXTENSIONS",
    "PPTX_EXTENSIONS",
    "OFFICE_EXTENSIONS",
]
