"""HTML loader (.html, .htm)."""

from __future__ import annotations

from pathlib import Path

import logfire
from bs4 import BeautifulSoup

from app.ingestion.loaders.base import LoadedDocument, ensure_logfire, resolve_path

HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}


def load_html(path: str | Path) -> LoadedDocument:
    ensure_logfire()
    file_path = resolve_path(path)

    with logfire.span("loader.html", source=str(file_path)):
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        for tag in soup(["script", "style", "noscript", "svg", "template"]):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)

        doc = LoadedDocument(
            text=text,
            source=str(file_path),
            doc_type="html",
            metadata={
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "title": title,
                "size_bytes": file_path.stat().st_size,
            },
        )
        logfire.info(
            "Loaded HTML document",
            source=str(file_path),
            title=title or None,
            chars=doc.char_count,
        )
        return doc
