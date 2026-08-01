"""PDF loader (.pdf) — pdfplumber primary, pypdf fallback."""

from __future__ import annotations

from pathlib import Path

import logfire
from pypdf import PdfReader

from app.ingestion.loaders.base import LoadedDocument, ensure_logfire, resolve_path

PDF_EXTENSIONS = {".pdf"}


def _extract_with_pdfplumber(file_path: Path) -> tuple[str, int]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
        return "\n\n".join(pages), len(pdf.pages)


def _extract_with_pypdf(file_path: Path) -> tuple[str, int]:
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())
    return "\n\n".join(pages), len(reader.pages)


def load_pdf(path: str | Path) -> LoadedDocument:
    ensure_logfire()
    file_path = resolve_path(path)

    with logfire.span("loader.pdf", source=str(file_path)):
        extractor = "pdfplumber"
        try:
            text, page_count = _extract_with_pdfplumber(file_path)
        except Exception as exc:
            logfire.warn(
                "pdfplumber failed; falling back to pypdf",
                source=str(file_path),
                error=str(exc),
            )
            extractor = "pypdf"
            text, page_count = _extract_with_pypdf(file_path)

        doc = LoadedDocument(
            text=text.strip(),
            source=str(file_path),
            doc_type="pdf",
            metadata={
                "filename": file_path.name,
                "extension": ".pdf",
                "page_count": page_count,
                "extractor": extractor,
                "size_bytes": file_path.stat().st_size,
            },
        )
        logfire.info(
            "Loaded PDF document",
            source=str(file_path),
            pages=page_count,
            chars=doc.char_count,
            extractor=extractor,
            empty=doc.is_empty,
        )
        return doc
